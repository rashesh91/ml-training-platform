#!/bin/bash
# Bootstrap script for new server: k3s + ArgoCD + both ML projects
# Run on: ubuntu@217.18.55.59
# Usage: bash bootstrap-new-server.sh
set -euo pipefail

LOG=/tmp/bootstrap.log
exec > >(tee -a $LOG) 2>&1

echo "========================================"
echo " ML Platform Bootstrap — $(date)"
echo "========================================"

# ─── Phase 1: NVIDIA Container Toolkit ──────────────────────────────────────
echo ""
echo "=== Phase 1: NVIDIA Container Toolkit ==="
if ! command -v nvidia-ctk &>/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update -q
  sudo apt-get install -y nvidia-container-toolkit
  echo "NVIDIA Container Toolkit installed."
else
  echo "NVIDIA Container Toolkit already installed."
fi

# ─── Phase 2: Install k3s ───────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Install k3s ==="
if ! command -v k3s &>/dev/null; then
  curl -sfL https://get.k3s.io | sudo INSTALL_K3S_VERSION=v1.28.15+k3s1 sh -s - \
    --write-kubeconfig-mode 644 \
    --disable traefik
  echo "k3s installed."
else
  echo "k3s already installed."
fi

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
grep -q 'KUBECONFIG' ~/.bashrc || echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> ~/.bashrc

echo "Waiting for k3s node to be Ready..."
until sudo kubectl get node --no-headers 2>/dev/null | grep -q Ready; do
  sleep 5
  echo -n "."
done
echo ""
echo "k3s is Ready."

# ─── Phase 3: Configure NVIDIA runtime in k3s containerd ────────────────────
echo ""
echo "=== Phase 3: Configure NVIDIA runtime in k3s containerd ==="
sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd/

sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl > /dev/null << 'TOML'
version = 2

[plugins."io.containerd.internal.v1.opt"]
  path = "{{ .NodeConfig.Containerd.Opt }}"

[plugins."io.containerd.grpc.v1.cri"]
  stream_server_address = "127.0.0.1"
  stream_server_port = "10010"
  enable_selinux = false
  enable_unprivileged_ports = false
  enable_unprivileged_icmp = false
  sandbox_image = "{{ .PauseImage }}"

[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "runc"
  snapshotter = "{{ .NodeConfig.Containerd.Snapshotter }}"
  disable_snapshot_annotations = {{ .NodeConfig.Containerd.DiscardUnpackedLayers }}

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
TOML

echo "Restarting k3s to apply NVIDIA runtime config..."
sudo systemctl restart k3s
sleep 20

until sudo kubectl get node --no-headers 2>/dev/null | grep -q Ready; do
  sleep 5
  echo -n "."
done
echo ""
echo "k3s restarted with NVIDIA runtime support."

# ─── Phase 4: NVIDIA RuntimeClass + Device Plugin ───────────────────────────
echo ""
echo "=== Phase 4: NVIDIA RuntimeClass + Device Plugin ==="

sudo kubectl apply -f - << 'EOF'
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF

sudo kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/deployments/static/nvidia-device-plugin.yml

# Label node so Ollama scheduling works (nodeSelector: nvidia.com/gpu: "true")
NODE=$(sudo kubectl get node --no-headers -o custom-columns=NAME:.metadata.name | head -1)
sudo kubectl label node "$NODE" nvidia.com/gpu=true --overwrite
echo "Node $NODE labeled nvidia.com/gpu=true"

# Wait for device plugin to expose GPU
echo "Waiting for GPU to be allocatable..."
for i in $(seq 1 24); do
  GPUS=$(sudo kubectl get node "$NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
  if [ "$GPUS" != "" ] && [ "$GPUS" != "0" ]; then
    echo "GPU allocatable: $GPUS"
    break
  fi
  echo "  Attempt $i/24 — GPU not yet allocatable, waiting 10s..."
  sleep 10
done

# ─── Phase 5: Install ArgoCD ────────────────────────────────────────────────
echo ""
echo "=== Phase 5: Install ArgoCD ==="
sudo kubectl create namespace argocd --dry-run=client -o yaml | sudo kubectl apply -f -
sudo kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/v2.10.0/manifests/install.yaml

echo "Waiting for ArgoCD server to be available..."
sudo kubectl wait --for=condition=available deployment/argocd-server \
  -n argocd --timeout=300s
echo "ArgoCD is ready."

# ─── Phase 6: Apply ArgoCD Projects ─────────────────────────────────────────
echo ""
echo "=== Phase 6: Apply ArgoCD Projects ==="

# Create ml-training namespace first (needed for project destination)
sudo kubectl create namespace ml-training --dry-run=client -o yaml | sudo kubectl apply -f -
sudo kubectl create namespace voice-platform --dry-run=client -o yaml | sudo kubectl apply -f -
sudo kubectl create namespace infra --dry-run=client -o yaml | sudo kubectl apply -f -

# Clone repos to get project manifests
WORK=/tmp/argocd-bootstrap
rm -rf $WORK && mkdir -p $WORK

git clone --depth=1 https://github.com/rashesh91/ml-training-platform.git $WORK/ml-training-platform
git clone --depth=1 https://github.com/rashesh91/ml-inference-gitops.git $WORK/ml-inference-gitops

# Apply ArgoCD projects
sudo kubectl apply -f $WORK/ml-training-platform/argocd-config/projects.yaml -n argocd
sudo kubectl apply -f $WORK/ml-inference-gitops/argocd-config/projects.yaml -n argocd

echo "ArgoCD projects created."

# ─── Phase 7: Create required secrets ───────────────────────────────────────
echo ""
echo "=== Phase 7: Create secrets ==="

# minio-credentials (matches MinIO Helm values: rootUser/rootPassword = minioadmin)
sudo kubectl apply -f - << 'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: minio-credentials
  namespace: ml-training
type: Opaque
stringData:
  endpoint: "http://minio.ml-training.svc.cluster.local:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin"
EOF
echo "minio-credentials secret created in ml-training."

# minio-credentials in voice-platform namespace (for vLLM model downloader)
sudo kubectl apply -f - << 'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: minio-credentials
  namespace: voice-platform
type: Opaque
stringData:
  endpoint: "http://minio.ml-training.svc.cluster.local:9000"
  access_key: "minioadmin"
  secret_key: "minioadmin"
EOF
echo "minio-credentials secret created in voice-platform."

# HF token (needed for Llama 3.2 and other gated models)
echo ""
echo "Enter your HuggingFace token (hf_...) or press Enter to skip:"
read -r HF_TOKEN_VALUE
if [ -n "$HF_TOKEN_VALUE" ]; then
  sudo kubectl create secret generic hf-token \
    --from-literal=token="$HF_TOKEN_VALUE" \
    --namespace=ml-training \
    --dry-run=client -o yaml | sudo kubectl apply -f -
  sudo kubectl create secret generic hf-token \
    --from-literal=token="$HF_TOKEN_VALUE" \
    --namespace=voice-platform \
    --dry-run=client -o yaml | sudo kubectl apply -f -
  echo "hf-token secret created in ml-training and voice-platform."
else
  echo "HF token skipped. Create later:"
  echo "  sudo kubectl create secret generic hf-token --from-literal=token=hf_YOUR_TOKEN -n ml-training"
  echo "  sudo kubectl create secret generic hf-token --from-literal=token=hf_YOUR_TOKEN -n voice-platform"
fi

# ─── Phase 7b: PriorityClasses (GPU scheduling) ─────────────────────────────
echo ""
echo "=== Phase 7b: PriorityClasses ==="
sudo kubectl apply -f $WORK/ml-training-platform/infrastructure/rbac/priority-classes.yaml
echo "PriorityClasses ml-training-high and ml-inference-low created."

# ─── Phase 8: Apply ArgoCD Root Apps ────────────────────────────────────────
echo ""
echo "=== Phase 8: Apply ArgoCD Root Apps ==="

sudo kubectl apply -f $WORK/ml-training-platform/argocd-config/root-app-e2e.yaml -n argocd
sudo kubectl apply -f $WORK/ml-inference-gitops/argocd-config/root-app.yaml -n argocd

echo "Root apps applied — ArgoCD will sync both projects from GitHub."
echo "Monitor: sudo kubectl get applications -n argocd"

# ─── Phase 9: Build and import custom images ────────────────────────────────
echo ""
echo "=== Phase 9: Build custom images ==="
echo "Building training-api, fine-tuner (CUDA), evaluator (CUDA),"
echo "whisper-stt, tts-service, voice-gateway in parallel..."

BUILD_DIR=/tmp/image-builds
rm -rf $BUILD_DIR && mkdir -p $BUILD_DIR

# Copy service source dirs
cp -r $WORK/ml-training-platform/services/training-api $BUILD_DIR/
cp -r $WORK/ml-training-platform/services/fine-tuner $BUILD_DIR/
cp -r $WORK/ml-training-platform/services/evaluator $BUILD_DIR/
cp -r $WORK/ml-inference-gitops/services/whisper-stt $BUILD_DIR/
cp -r $WORK/ml-inference-gitops/services/tts-service $BUILD_DIR/
cp -r $WORK/ml-inference-gitops/services/voice-gateway $BUILD_DIR/

# Build non-CUDA images quickly (parallel)
echo "Building fast images (training-api, whisper-stt, tts-service, voice-gateway)..."
(cd $BUILD_DIR/training-api && sudo docker build -t training-api:latest . > /tmp/build-training-api.log 2>&1 && echo "training-api: DONE") &
(cd $BUILD_DIR/whisper-stt && sudo docker build -t voice-platform/whisper-stt:latest . > /tmp/build-whisper.log 2>&1 && echo "whisper-stt: DONE") &
(cd $BUILD_DIR/tts-service && sudo docker build -t voice-platform/tts-service:latest . > /tmp/build-tts.log 2>&1 && echo "tts-service: DONE") &
(cd $BUILD_DIR/voice-gateway && sudo docker build -t voice-platform/voice-gateway:latest . > /tmp/build-gateway.log 2>&1 && echo "voice-gateway: DONE") &

# Wait for fast builds
wait
echo "Fast image builds complete."

# CUDA images take ~10-15 min each (download PyTorch CUDA 2GB)
echo "Building CUDA images (fine-tuner, evaluator) — this takes 10-20 min..."
(cd $BUILD_DIR/fine-tuner && sudo docker build -t fine-tuner:latest . > /tmp/build-fine-tuner.log 2>&1 && echo "fine-tuner: DONE") &
FINE_PID=$!
(cd $BUILD_DIR/evaluator && sudo docker build -t evaluator:latest . > /tmp/build-evaluator.log 2>&1 && echo "evaluator: DONE") &
EVAL_PID=$!

wait $FINE_PID && echo "fine-tuner build complete."
wait $EVAL_PID && echo "evaluator build complete."

# ─── Phase 10: Import images into k3s containerd ────────────────────────────
echo ""
echo "=== Phase 10: Import images into k3s ==="

for IMG in training-api:latest voice-platform/whisper-stt:latest voice-platform/tts-service:latest voice-platform/voice-gateway:latest fine-tuner:latest evaluator:latest; do
  echo "Importing $IMG..."
  sudo docker save "$IMG" | sudo k3s ctr images import -
  echo "  $IMG imported."
done

# Clean up Docker images to free space
echo "Cleaning up Docker build cache..."
sudo docker system prune -f
echo "Docker cleanup done."

# ─── Phase 11: Apply WorkflowTemplate ───────────────────────────────────────
echo ""
echo "=== Phase 11: Apply WorkflowTemplate ==="
echo "Waiting for ml-training namespace and Argo Workflows CRDs..."

for i in $(seq 1 30); do
  if sudo kubectl get crd workflowtemplates.argoproj.io &>/dev/null; then
    echo "Argo Workflows CRDs ready."
    break
  fi
  echo "  Attempt $i/30 — waiting for CRDs..."
  sleep 15
done

sudo kubectl apply -f $WORK/ml-training-platform/workflows/finetune-pipeline-e2e.yaml -n ml-training
echo "WorkflowTemplate finetune-pipeline applied."

# ─── Phase 12: Expose ArgoCD UI ─────────────────────────────────────────────
echo ""
echo "=== Phase 12: Expose ArgoCD UI via NodePort ==="
sudo kubectl patch svc argocd-server -n argocd \
  -p '{"spec":{"type":"NodePort","ports":[{"port":443,"targetPort":8080,"nodePort":30443,"name":"https"},{"port":80,"targetPort":8080,"nodePort":30080,"name":"http"}]}}'

ARGOCD_PASS=$(sudo kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath='{.data.password}' | base64 -d)

echo ""
echo "========================================"
echo " Bootstrap Complete! — $(date)"
echo "========================================"
echo ""
SERVER_IP=$(hostname -I | awk '{print $1}')
echo " ArgoCD UI:      http://$SERVER_IP:30080"
echo " ArgoCD User:    admin"
echo " ArgoCD Pass:    $ARGOCD_PASS"
echo ""
echo " TRAINING API:  (wait for ArgoCD sync ~5 min)"
echo "   sudo kubectl get pods -n ml-training"
echo "   sudo kubectl get applications -n argocd"
echo ""
echo " GPU check:"
echo "   sudo kubectl get node -o jsonpath='{.items[0].status.allocatable}'"
echo ""
echo " Build logs: /tmp/build-*.log"
echo " Full log:   /tmp/bootstrap.log"
