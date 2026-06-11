# ml-training-platform — LLM Fine-tuning & Evaluation Platform

[![GitHub](https://img.shields.io/badge/GitHub-rashesh91%2Fml--training--platform-181717?logo=github)](https://github.com/rashesh91/ml-training-platform)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Ray](https://img.shields.io/badge/KubeRay-Distributed%20Training-028CF0)](https://ray.io)
[![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194e2)](https://mlflow.org)
[![Argo](https://img.shields.io/badge/Argo%20Workflows-Pipeline-EF7B4D)](https://argoproj.github.io/workflows)
[![ArgoCD](https://img.shields.io/badge/ArgoCD-GitOps-EF7B4D)](https://argoproj.github.io/cd)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-GPU--aware-326CE5)](https://kubernetes.io)

A production-grade **LLM fine-tuning platform** on Kubernetes.

**Upload dataset → LoRA fine-tune on KubeRay → auto-evaluate → MLflow registry → ArgoCD deploys to prod**

> Companion project to [ml-inference-gitops](https://github.com/rashesh91/ml-inference-gitops). That project **serves** models. This project **trains** them. Together they demonstrate the complete ML lifecycle.

---

## Architecture

```
Browser / API Client
  │
  ▼
training-api (FastAPI + Dashboard)
  │
  ├── MinIO ◄─── dataset upload (S3-compatible object storage)
  │
  ├── Argo Workflows ─── triggers pipeline DAG
  │       │
  │       ├── Step 1: Preprocess dataset (validate, count records)
  │       │
  │       ├── Step 2: KubeRay ─── distributed LoRA fine-tuning
  │       │           ├── Ray head node
  │       │           └── GPU worker nodes (1-4×)
  │       │                 └── HuggingFace PEFT (LoRA)
  │       │                       └── logs metrics → MLflow in real-time
  │       │
  │       ├── Step 3: Evaluator ─── ROUGE benchmark vs base model
  │       │           └── saves comparison report → MLflow
  │       │
  │       ├── Step 4: Auto-promote (if eval passes threshold)
  │       │           └── model → MLflow registry "Production" stage
  │       │
  │       └── Step 5: ArgoCD deploy
  │                   └── triggers ml-inference-gitops to load new model
  │
  └── MLflow ─── experiment tracking + model registry
        └── PostgreSQL (metadata) + MinIO (artifacts)
```

---

## Minimum Server Requirements

### Local Development (Docker Compose, no GPU)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8 cores |
| RAM | 12 GB | 16 GB |
| Disk | 30 GB free | 60 GB free |
| OS | Linux / macOS / Windows (WSL2) | Ubuntu 22.04 |
| Docker | v24+ | v24+ |

> Fine-tunes `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on CPU by default.  
> Training a 10-sample dataset takes ~5 min on CPU. Realistic dataset (1000 samples) = ~1 hour CPU.

---

### Production Kubernetes (GPU cluster)

| Node type | Count | CPU | RAM | GPU | Role |
|-----------|-------|-----|-----|-----|------|
| GPU worker | 1–4 | 8 vCPU | 32 GB | 1× NVIDIA T4 (16 GB) | Ray training workers |
| CPU node | 2 | 4 vCPU | 8 GB | — | training-api, MLflow, MinIO |
| **Total min** | **3 nodes** | **16 vCPU** | **48 GB** | **1× T4** | |

**GPU requirements by model:**

| Model | VRAM needed | Training time (1k samples) |
|-------|------------|----------------------------|
| TinyLlama 1.1B (LoRA) | 4 GB | ~15 min on T4 |
| Mistral 7B (LoRA Q4) | 10 GB | ~45 min on T4 |
| Llama 3.2 3B (LoRA) | 8 GB | ~25 min on T4 |

---

## Services

| Service | Port | Purpose |
|---------|------|---------|
| `training-api` | 8004 | Job submission, status API, dashboard UI |
| `fine-tuner` | 8003 | LoRA fine-tuning service (Ray Train + PEFT) |
| `evaluator` | 8005 | ROUGE evaluation vs base model |
| `mlflow` | 5000 | Experiment tracking + model registry |
| `minio` | 9000/9001 | S3-compatible artifact storage |
| `postgres` | 5432 | MLflow backend metadata store |
| Ray Dashboard | 8265 | KubeRay cluster monitoring |
| Argo UI | 2746 | Workflow pipeline visualization |

---

## Quick Start — Local (Docker Compose, no GPU)

```bash
git clone https://github.com/rashesh91/ml-training-platform.git
cd ml-training-platform

# Build and start all services (~10 min first time, downloads TinyLlama)
docker compose up --build

# Wait for all services to be healthy:
docker compose ps
```

Open **http://localhost:8004** for the training dashboard.

### Submit your first fine-tuning job

1. Click **Download sample.jsonl** in the sidebar — gets a 10-example ML Q&A dataset
2. Drag the file onto the drop zone
3. Set model name: `my-first-model`
4. Keep defaults (TinyLlama, LoRA rank 16, 3 epochs)
5. Click **Start Fine-tuning**

Watch progress in the **Training Jobs** tab. When complete:
- MLflow run appears at **http://localhost:5000** with loss curves
- Model registered in **Model Registry** tab
- Click **Deploy →** to promote to Production

---

## Kubernetes Deployment (GPU cluster, ArgoCD)

### Step 1 — Build and push images

```bash
REGISTRY=ghcr.io/rashesh91

docker build -t $REGISTRY/training-api:latest  services/training-api/
docker build -t $REGISTRY/fine-tuner:latest    services/fine-tuner/
docker build -t $REGISTRY/evaluator:latest     services/evaluator/

docker push $REGISTRY/training-api:latest
docker push $REGISTRY/fine-tuner:latest
docker push $REGISTRY/evaluator:latest
```

### Step 2 — Update registry in values.yaml

```yaml
# applications/training-platform/values.yaml
global:
  imageRegistry: "ghcr.io/rashesh91/"
```

### Step 3 — Bootstrap the cluster

```bash
# Install ArgoCD (if not already from ml-inference-gitops)
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Apply AppProject + root App of Apps
kubectl apply -f argocd-config/projects.yaml
kubectl apply -f argocd-config/root-app.yaml -n argocd
```

### Step 4 — ArgoCD deploys in sync waves

| Wave | Apps deployed |
|------|--------------|
| 0 | Namespaces + RBAC |
| 1 | KubeRay Operator + Argo Workflows |
| 2 | MinIO + PostgreSQL |
| 3 | MLflow (waits for DB + storage) |
| 4 | training-api + RayCluster + WorkflowTemplate |

Watch: `kubectl get applications -n argocd -w`

---

## Dataset Format

JSONL file, one record per line. Two formats supported:

**Instruction/response format (recommended):**
```jsonl
{"prompt": "Explain gradient descent.", "response": "Gradient descent is an optimization algorithm..."}
{"prompt": "What is overfitting?", "response": "Overfitting occurs when a model memorizes..."}
```

**Free text format:**
```jsonl
{"text": "### Instruction:\nWhat is LoRA?\n\n### Response:\nLoRA is a parameter-efficient..."}
```

---

## Pipeline (Argo Workflow DAG)

```
preprocess ──► ray-train ──► evaluate ──► promote (if passed) ──► notify
                  │               │
                  ▼               ▼
               MLflow          ROUGE-L
               (live)       comparison report
```

Auto-promotion threshold: **ROUGE-L improvement ≥ 5%** over base model.  
Adjust via `ROUGE_THRESHOLD` env var or Helm values.

---

## Project Structure

```
ml-training-platform/
├── services/
│   ├── training-api/        ← FastAPI job API + HTML dashboard
│   │   ├── app/main.py      ← REST endpoints, MinIO upload, Argo trigger
│   │   ├── static/          ← Dashboard UI (dark mode, drag-drop, live status)
│   │   │   ├── index.html
│   │   │   └── app.js
│   │   └── Dockerfile
│   ├── fine-tuner/          ← Ray Train + HuggingFace PEFT LoRA training
│   │   ├── train.py         ← Distributed training loop + MLflow logging
│   │   └── Dockerfile
│   └── evaluator/           ← ROUGE benchmark + MLflow promotion
│       ├── evaluate.py      ← Benchmark vs base model, auto-promote if passes
│       └── Dockerfile
├── workflows/
│   └── finetune-pipeline.yaml   ← Argo Workflow DAG (5-step pipeline)
├── infrastructure/
│   ├── kuberay/             ← KubeRay Operator Helm values + RayCluster CRD
│   ├── mlflow/              ← MLflow Deployment + Service + Ingress
│   ├── minio/               ← MinIO Helm values (4 buckets)
│   ├── argo-workflows/      ← Argo Workflows Helm values
│   ├── postgres/            ← PostgreSQL StatefulSet
│   ├── namespaces/          ← Namespace + ResourceQuota + LimitRange
│   └── rbac/                ← ServiceAccount + Role + RoleBinding + dev Secrets
├── applications/
│   └── training-platform/   ← Umbrella Helm chart
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/       ← training-api Deployment, RayCluster CRD
├── argocd-config/
│   ├── projects.yaml        ← AppProject with allowed repos
│   ├── root-app.yaml        ← App of Apps root
│   └── apps/
│       ├── infra.yaml       ← Wave 0-3: operators, storage, MLflow
│       └── training-platform.yaml ← Wave 4: training services
├── docker-compose.yml       ← Full stack local dev (no K8s)
└── docs/
    └── QUICKSTART.md
```

---

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| **Distributed training** | Ray Train + KubeRay multi-GPU workers |
| **Parameter-efficient fine-tuning** | HuggingFace PEFT LoRA (<1% params trained) |
| **Experiment tracking** | MLflow autolog — loss, LR, GPU metrics |
| **Model registry + versioning** | MLflow registry (None → Staging → Production) |
| **ML pipeline automation** | Argo Workflow DAG (5 steps, conditional promote) |
| **S3-compatible artifact storage** | MinIO on Kubernetes (4 buckets) |
| **GitOps** | ArgoCD App of Apps (5 sync waves) |
| **GPU resource management** | KubeRay GPU worker group, K8s device plugin |
| **End-to-end ML lifecycle** | Training + eval + registry + serving (both projects) |

---

## Integration with ml-inference-gitops

When a model is promoted to `Production` in MLflow:

```
MLflow: model → Production
    ↓
Webhook (or manual) → ArgoCD
    ↓
ArgoCD updates ml-inference-gitops/applications/voice-platform/values.yaml
    ↓
voice-gateway reloads with fine-tuned model
    ↓
voice.rashesh.dev now uses the custom fine-tuned model
```

See [ml-inference-gitops](https://github.com/rashesh91/ml-inference-gitops) for the serving side.
