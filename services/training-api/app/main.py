import json
import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional

import boto3
import mlflow
import httpx
from botocore.client import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Training Platform API")

# Config from env
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ARGO_SERVER = os.getenv("ARGO_SERVER", "http://argo-workflows-server:2746")
ARGO_NAMESPACE = os.getenv("ARGO_NAMESPACE", "ml-training")
FINE_TUNER_URL = os.getenv("FINE_TUNER_URL", "http://fine-tuner:8003")  # local dev

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# In-memory job store (use Redis/Postgres in production)
_jobs: dict[str, dict] = {}


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _ensure_bucket(name: str):
    s3 = _s3()
    try:
        s3.head_bucket(Bucket=name)
    except Exception:
        s3.create_bucket(Bucket=name)


# --- Models ---

class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | evaluating | succeeded | failed
    model_name: str
    base_model: str
    created_at: str
    updated_at: str
    mlflow_run_id: Optional[str] = None
    mlflow_run_url: Optional[str] = None
    eval_score: Optional[float] = None
    eval_passed: Optional[bool] = None
    error: Optional[str] = None


class ModelEntry(BaseModel):
    name: str
    version: str
    stage: str
    run_id: str
    eval_score: Optional[float] = None
    created_at: str


class DeployRequest(BaseModel):
    version: str


# --- Routes ---

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/jobs", response_model=JobStatus)
async def submit_job(
    dataset: UploadFile = File(...),
    model_name: str = Form(...),
    base_model: str = Form("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    lora_rank: int = Form(16),
    lora_alpha: int = Form(32),
    epochs: int = Form(3),
    learning_rate: float = Form(2e-4),
    max_seq_length: int = Form(512),
):
    job_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()

    # Upload dataset to MinIO
    _ensure_bucket("datasets")
    dataset_key = f"{job_id}/dataset.jsonl"
    content = await dataset.read()
    _s3().put_object(Bucket="datasets", Key=dataset_key, Body=content)
    logger.info(f"Uploaded dataset to s3://datasets/{dataset_key} ({len(content)} bytes)")

    job_config = {
        "job_id": job_id,
        "model_name": model_name,
        "base_model": base_model,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "dataset_bucket": "datasets",
        "dataset_key": dataset_key,
        "checkpoint_bucket": "checkpoints",
    }

    job_record = {
        "job_id": job_id,
        "status": "queued",
        "model_name": model_name,
        "base_model": base_model,
        "created_at": now,
        "updated_at": now,
        "mlflow_run_id": None,
        "mlflow_run_url": None,
        "eval_score": None,
        "eval_passed": None,
        "error": None,
        "config": job_config,
    }
    _jobs[job_id] = job_record

    # Trigger pipeline (Argo Workflow in K8s, direct HTTP call in local dev)
    asyncio.create_task(_trigger_pipeline(job_id, job_config))

    return JobStatus(**{k: v for k, v in job_record.items() if k != "config"})


async def _trigger_pipeline(job_id: str, config: dict):
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    argo_available = os.getenv("USE_ARGO", "false").lower() == "true"

    if argo_available:
        await _submit_argo_workflow(job_id, config)
    else:
        # Local dev: call fine-tuner service directly
        await _call_fine_tuner_direct(job_id, config)


async def _call_fine_tuner_direct(job_id: str, config: dict):
    try:
        async with httpx.AsyncClient(timeout=7200.0) as client:
            resp = await client.post(
                f"{FINE_TUNER_URL}/train",
                json=config,
            )
            resp.raise_for_status()
            result = resp.json()

        _jobs[job_id].update({
            "status": "evaluating",
            "mlflow_run_id": result.get("mlflow_run_id"),
            "mlflow_run_url": f"{MLFLOW_TRACKING_URI}/#/experiments/1/runs/{result.get('mlflow_run_id', '')}",
            "updated_at": datetime.utcnow().isoformat(),
        })

        # Evaluation is triggered automatically by fine-tuner in local dev
        if result.get("eval_passed") is not None:
            _jobs[job_id].update({
                "status": "succeeded" if result["eval_passed"] else "failed",
                "eval_score": result.get("eval_score"),
                "eval_passed": result.get("eval_passed"),
                "updated_at": datetime.utcnow().isoformat(),
            })
        else:
            _jobs[job_id]["status"] = "succeeded"
            _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        _jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.utcnow().isoformat(),
        })


async def _submit_argo_workflow(job_id: str, config: dict):
    workflow_manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"finetune-{job_id}-",
            "namespace": ARGO_NAMESPACE,
        },
        "spec": {
            "workflowTemplateRef": {"name": "finetune-pipeline"},
            "arguments": {
                "parameters": [{"name": "config", "value": json.dumps(config)}]
            },
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ARGO_SERVER}/api/v1/workflows/{ARGO_NAMESPACE}",
                json=workflow_manifest,
            )
            resp.raise_for_status()
            argo_name = resp.json()["metadata"]["name"]
            logger.info(f"Argo workflow submitted: {argo_name}")
    except Exception as e:
        logger.error(f"Failed to submit Argo workflow: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


@app.get("/api/jobs", response_model=list[JobStatus])
async def list_jobs():
    return [
        JobStatus(**{k: v for k, v in job.items() if k != "config"})
        for job in sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    ]


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    return JobStatus(**{k: v for k, v in job.items() if k != "config"})


@app.get("/api/models", response_model=list[ModelEntry])
async def list_models():
    client = mlflow.tracking.MlflowClient()
    results = []
    try:
        for rm in client.search_registered_models():
            for mv in client.search_model_versions(f"name='{rm.name}'"):
                run_data = {}
                try:
                    run = client.get_run(mv.run_id)
                    run_data = run.data.metrics
                except Exception:
                    pass
                results.append(ModelEntry(
                    name=rm.name,
                    version=mv.version,
                    stage=mv.current_stage,
                    run_id=mv.run_id,
                    eval_score=run_data.get("eval_rouge_l"),
                    created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000).isoformat(),
                ))
    except Exception as e:
        logger.warning(f"Could not fetch models from MLflow: {e}")
    return results


@app.post("/api/models/{model_name}/deploy")
async def deploy_model(model_name: str, req: DeployRequest):
    client = mlflow.tracking.MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=req.version,
            stage="Production",
        )
        return {"status": "promoted", "model": model_name, "version": req.version, "stage": "Production"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve UI
ui_path = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(ui_path):
    app.mount("/static", StaticFiles(directory=ui_path), name="static")

    @app.get("/")
    async def ui():
        return FileResponse(os.path.join(ui_path, "index.html"))
