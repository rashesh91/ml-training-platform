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
import redis.asyncio as aioredis
from botocore.client import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from deploy_to_inference import trigger_inference_deploy

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
REDIS_URL = os.getenv("REDIS_URL", "")  # empty = no Redis (local dev without queue)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


async def _redis_push(job_id: str):
    if not REDIS_URL:
        return
    try:
        r = await aioredis.from_url(REDIS_URL)
        await r.rpush("training:job_queue", job_id)
        await r.aclose()
    except Exception as e:
        logger.warning(f"Redis push failed for job {job_id}: {e}")


async def _redis_remove(job_id: str):
    if not REDIS_URL:
        return
    try:
        r = await aioredis.from_url(REDIS_URL)
        await r.lrem("training:job_queue", 1, job_id)
        await r.aclose()
    except Exception as e:
        logger.warning(f"Redis remove failed for job {job_id}: {e}")

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
    run_id: str = ""


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
    batch_size: int = Form(2),
    grad_accum: int = Form(4),
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
        "batch_size": batch_size,
        "grad_accum": grad_accum,
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

    # Push to Redis queue — KEDA ScaledObject watches this list to scale Ray workers
    asyncio.create_task(_redis_push(job_id))

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

        # Remove from Redis queue — signals KEDA the job slot is free
        await _redis_remove(job_id)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        _jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "updated_at": datetime.utcnow().isoformat(),
        })
        await _redis_remove(job_id)


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
                json={"workflow": workflow_manifest},
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Kick off inference deploy in the background so the HTTP response returns
    # immediately; any failure is logged but does not affect the promote result.
    run_id = req.run_id or ""
    asyncio.create_task(
        _deploy_to_inference_safe(model_name, req.version, run_id)
    )
    return {"status": "promoted", "model": model_name, "version": req.version, "stage": "Production"}


async def _deploy_to_inference_safe(model_name: str, version: str, run_id: str):
    try:
        result = await trigger_inference_deploy(model_name, version, run_id)
        logger.info("Inference deploy result: %s", result)
    except Exception as e:
        logger.error("Inference deploy failed for %s v%s: %s", model_name, version, e)


class ExternalJobRequest(BaseModel):
    job_id: str
    model_name: str
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    status: str = "running"
    log_line: Optional[str] = None
    progress: Optional[float] = None      # 0.0–1.0
    epoch: Optional[float] = None
    loss: Optional[float] = None
    mlflow_run_id: Optional[str] = None
    eval_score: Optional[float] = None
    eval_passed: Optional[bool] = None
    error: Optional[str] = None


@app.post("/api/external-jobs")
async def register_external_job(req: ExternalJobRequest):
    """Register or update a training job running outside k8s (e.g. systemd on host)."""
    now = datetime.utcnow().isoformat()
    if req.job_id not in _jobs:
        _jobs[req.job_id] = {
            "job_id": req.job_id,
            "model_name": req.model_name,
            "base_model": req.base_model,
            "status": req.status,
            "created_at": now,
            "updated_at": now,
            "mlflow_run_id": None,
            "mlflow_run_url": None,
            "eval_score": None,
            "eval_passed": None,
            "error": None,
            "log_lines": [],
        }
    job = _jobs[req.job_id]
    job["status"] = req.status
    job["updated_at"] = now
    if req.log_line:
        job.setdefault("log_lines", []).append(req.log_line)
        if len(job["log_lines"]) > 100:
            job["log_lines"] = job["log_lines"][-100:]
    if req.progress is not None:
        job["progress"] = req.progress
    if req.epoch is not None:
        job["epoch"] = req.epoch
    if req.loss is not None:
        job["loss"] = req.loss
    if req.mlflow_run_id:
        job["mlflow_run_id"] = req.mlflow_run_id
        job["mlflow_run_url"] = f"{MLFLOW_TRACKING_URI}/#/experiments/1/runs/{req.mlflow_run_id}"
    if req.eval_score is not None:
        job["eval_score"] = req.eval_score
    if req.eval_passed is not None:
        job["eval_passed"] = req.eval_passed
    if req.error:
        job["error"] = req.error
    return {"ok": True, "job_id": req.job_id}


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"logs": _jobs[job_id].get("log_lines", [])}


class InternalDeployRequest(BaseModel):
    model_name: str
    version: str
    run_id: str = ""


@app.post("/api/internal/deploy-to-inference")
async def internal_deploy_to_inference(req: InternalDeployRequest):
    """Called by the Argo Workflow deploy-to-inference step after auto-promote."""
    try:
        result = await trigger_inference_deploy(req.model_name, req.version, req.run_id)
        return result
    except Exception as e:
        logger.error("Internal inference deploy failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Serve UI
ui_path = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(ui_path):
    app.mount("/static", StaticFiles(directory=ui_path), name="static")

    @app.get("/")
    async def ui():
        return FileResponse(os.path.join(ui_path, "index.html"))
