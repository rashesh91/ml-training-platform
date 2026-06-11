# Quickstart Guide

## Prerequisites

- Docker v24+ and Docker Compose
- OR: `kubectl` + `helm` v3 for Kubernetes deployment

---

## Option A — Local Dev (Docker Compose)

```bash
git clone https://github.com/rashesh91/ml-training-platform.git
cd ml-training-platform

docker compose up --build
```

Services start in order (takes 5–10 min on first run):
1. MinIO + init buckets
2. PostgreSQL
3. MLflow (waits for DB + MinIO)
4. fine-tuner + evaluator (download TinyLlama ~1.1 GB on first start)
5. training-api (dashboard)

**URLs:**
| Service | URL |
|---------|-----|
| Training Dashboard | http://localhost:8004 |
| MLflow UI | http://localhost:5000 |
| MinIO Console | http://localhost:9001 (admin/minioadmin) |

**Submit a test job:**
```bash
# Download sample dataset
curl -o sample.jsonl http://localhost:8004/static/sample.jsonl  # or use Download button in UI

# Submit via API
curl -X POST http://localhost:8004/api/jobs \
  -F "dataset=@sample.jsonl" \
  -F "model_name=test-model" \
  -F "base_model=TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  -F "lora_rank=8" \
  -F "epochs=1"
```

**Check status:**
```bash
curl http://localhost:8004/api/jobs
```

---

## Option B — Kubernetes (GPU cluster, ArgoCD)

### Phase 1 — Namespaces + RBAC

```bash
kubectl apply -f infrastructure/namespaces/namespaces.yaml
kubectl apply -f infrastructure/rbac/rbac.yaml
```

### Phase 2 — Operators

```bash
# KubeRay
helm repo add kuberay https://kuberay.github.io/kuberay-helm-chart && helm repo update
helm install kuberay-operator kuberay/kuberay-operator \
  -n ml-training \
  -f infrastructure/kuberay/values.yaml

# Argo Workflows
helm repo add argo https://argoproj.github.io/argo-helm && helm repo update
helm install argo-workflows argo/argo-workflows \
  -n ml-training \
  -f infrastructure/argo-workflows/values.yaml
```

### Phase 3 — Storage + MLflow

```bash
# MinIO
helm repo add minio https://charts.min.io && helm repo update
helm install minio minio/minio -n ml-training -f infrastructure/minio/values.yaml

# PostgreSQL
kubectl apply -f infrastructure/postgres/deployment.yaml

# MLflow
kubectl apply -f infrastructure/mlflow/deployment.yaml

# Wait for MLflow
kubectl rollout status deployment/mlflow -n ml-training --timeout=120s
```

### Phase 4 — Training Platform

```bash
# Update image registry in values.yaml first
# applications/training-platform/values.yaml → global.imageRegistry

helm install training-platform ./applications/training-platform \
  -n ml-training \
  -f applications/training-platform/values.yaml

# Apply Argo WorkflowTemplate
kubectl apply -f workflows/finetune-pipeline.yaml

# Apply RayCluster
kubectl apply -f infrastructure/kuberay/raycluster.yaml
```

### Phase 5 — ArgoCD (GitOps)

```bash
# Install ArgoCD (if not already installed)
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=argocd-server -n argocd --timeout=300s

# Bootstrap App of Apps — ArgoCD will manage everything from here
kubectl apply -f argocd-config/projects.yaml
kubectl apply -f argocd-config/root-app.yaml -n argocd

# Watch sync progress
kubectl get applications -n argocd -w
```

---

## Validation Checklist

```bash
# KubeRay
kubectl get raycluster -n ml-training
# Expected: training-cluster   ready

# MLflow
curl http://mlflow.YOUR_DOMAIN/health
# Expected: {"status": "OK"}

# MinIO buckets
kubectl exec -n ml-training deploy/minio -- mc ls local/
# Expected: datasets/ checkpoints/ models/ mlflow/

# Submit test job via training-api
curl -X POST http://training.YOUR_DOMAIN/api/jobs \
  -F "dataset=@sample.jsonl" \
  -F "model_name=k8s-test" \
  -F "epochs=1"

# Watch Argo Workflow
kubectl get workflows -n ml-training -w

# Check MLflow for run
curl http://mlflow.YOUR_DOMAIN/api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["1"]}'
```

---

## GPU Fine-tuning on MX450 Laptop

MX450 has 2 GB VRAM — too small for the fine-tuner. But you can run everything else on CPU:

```bash
# In docker-compose.yml:
# fine-tuner: do NOT uncomment the GPU deploy block
# The service uses CPU automatically

# Use TinyLlama (1.1B params) — fine-tunes in ~16 GB RAM
# Set epochs=1 and lora_rank=8 to keep memory low

docker compose up --build
```

For real GPU demo: rent **Vast.ai RTX 3080** (~$0.20/hr):
```bash
# On GPU VPS:
# Uncomment GPU block in docker-compose.yml fine-tuner section
WHISPER_DEVICE=cuda docker compose up -d
```

---

## Troubleshooting

**"fine-tuner is unhealthy after 5 min"**
```bash
docker compose logs fine-tuner --tail=50
# Likely downloading TinyLlama model (~1.1 GB). Wait another 2-3 min.
```

**"MLflow can't connect to PostgreSQL"**
```bash
docker compose logs postgres --tail=20
docker compose restart mlflow
```

**"Training job stays 'running' forever"**
```bash
docker compose logs fine-tuner -f
# Check if model download is still in progress
# CTRL+C to stop watching
```

**"Out of memory during training"**
```bash
# Reduce batch size or max_seq_length
# Set epochs=1 and lora_rank=4 for minimal memory use
```
