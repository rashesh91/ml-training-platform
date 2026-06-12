"""
LoRA fine-tuning with Ray Train + HuggingFace PEFT.

In Kubernetes: launched as a RayJob, reads config from /configs/job.json
In local dev:  called via HTTP POST /train from training-api
"""

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import boto3
import mlflow
import torch
from botocore.client import Config
from datasets import Dataset
from fastapi import FastAPI
from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EVALUATOR_URL = os.getenv("EVALUATOR_URL", "http://evaluator:8005")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


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


def download_dataset(bucket: str, key: str) -> list[dict]:
    logger.info(f"Downloading dataset from s3://{bucket}/{key}")
    obj = _s3().get_object(Bucket=bucket, Key=key)
    lines = obj["Body"].read().decode("utf-8").strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def upload_checkpoint(local_dir: str, bucket: str, prefix: str):
    _ensure_bucket(bucket)
    s3 = _s3()
    local_path = Path(local_dir)
    for f in local_path.rglob("*"):
        if f.is_file():
            key = f"{prefix}/{f.relative_to(local_path)}"
            s3.upload_file(str(f), bucket, key)
            logger.info(f"Uploaded {f.name} → s3://{bucket}/{key}")


def build_dataset(records: list[dict], tokenizer, max_seq_length: int) -> Dataset:
    texts = []
    for r in records:
        if "text" in r:
            texts.append(r["text"])
        elif "prompt" in r and "response" in r:
            texts.append(f"### Instruction:\n{r['prompt']}\n\n### Response:\n{r['response']}")
        elif "instruction" in r and "output" in r:
            inp = r.get("input", "")
            body = f"\n\n### Input:\n{inp}" if inp else ""
            texts.append(f"### Instruction:\n{r['instruction']}{body}\n\n### Response:\n{r['output']}")
        else:
            logger.warning(f"Skipping record with unknown format: {list(r.keys())}")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return ds


def train(config: dict) -> dict:
    job_id = config["job_id"]
    model_name = config["model_name"]
    base_model = config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 32)
    epochs = config.get("epochs", 3)
    learning_rate = config.get("learning_rate", 2e-4)
    max_seq_length = config.get("max_seq_length", 512)
    batch_size = config.get("batch_size", 2)
    grad_accum = config.get("grad_accum", 4)
    dataset_bucket = config.get("dataset_bucket", "datasets")
    dataset_key = config.get("dataset_key", f"{job_id}/dataset.jsonl")
    checkpoint_bucket = config.get("checkpoint_bucket", "checkpoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device} | base_model: {base_model}")

    # Download dataset
    records = download_dataset(dataset_bucket, dataset_key)
    logger.info(f"Loaded {len(records)} training examples")

    # Load tokenizer + model
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Tokenize dataset
    dataset = build_dataset(records, tokenizer, max_seq_length)

    with tempfile.TemporaryDirectory() as output_dir:
        bf16_supported = device == "cuda" and torch.cuda.is_bf16_supported()
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            bf16=bf16_supported,
            fp16=(device == "cuda" and not bf16_supported),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to=[],  # MLflow autolog handles this
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            dataloader_num_workers=0,
        )

        with mlflow.start_run(run_name=f"finetune-{job_id}") as run:
            mlflow.log_params({
                "job_id": job_id,
                "base_model": base_model,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "max_seq_length": max_seq_length,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "dataset_size": len(records),
                "device": device,
            })

            class MLflowCallback:
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        step = state.global_step
                        for k, v in logs.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(k, v, step=step)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                tokenizer=tokenizer,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )

            logger.info("Starting training...")
            t0 = time.time()
            trainer.train()
            elapsed = time.time() - t0
            mlflow.log_metric("training_time_seconds", elapsed)
            logger.info(f"Training complete in {elapsed:.0f}s")

            # Save model + upload to MinIO
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            checkpoint_prefix = f"{job_id}/checkpoint"
            upload_checkpoint(output_dir, checkpoint_bucket, checkpoint_prefix)

            # Log model artifact to MLflow (save_model already ran above)
            mlflow.log_artifacts(output_dir, artifact_path="model")

            # Register in MLflow model registry
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model {model_name} version {mv.version}")

            run_id = run.info.run_id

    return {
        "job_id": job_id,
        "mlflow_run_id": run_id,
        "model_name": model_name,
        "checkpoint_path": f"s3://{checkpoint_bucket}/{checkpoint_prefix}",
    }


# --- FastAPI server for local dev mode ---

app = FastAPI(title="Fine-tuner Service")


class TrainRequest(BaseModel):
    job_id: str
    model_name: str
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    dataset_bucket: str = "datasets"
    dataset_key: str = ""
    checkpoint_bucket: str = "checkpoints"


@app.get("/health")
async def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.post("/train")
async def train_endpoint(req: TrainRequest):
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, train, req.dict())

    # Run evaluation after training (local dev)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=600.0) as client:
            eval_resp = await client.post(
                f"{EVALUATOR_URL}/evaluate",
                json={
                    "job_id": result["job_id"],
                    "model_name": result["model_name"],
                    "mlflow_run_id": result["mlflow_run_id"],
                    "base_model": req.base_model,
                    "checkpoint_bucket": req.checkpoint_bucket,
                    "checkpoint_prefix": f"{req.job_id}/checkpoint",
                },
            )
            eval_resp.raise_for_status()
            eval_result = eval_resp.json()
            result.update({
                "eval_score": eval_result.get("rouge_l"),
                "eval_passed": eval_result.get("passed"),
            })
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")

    return result


# --- Entry point for RayJob (K8s mode) ---

if __name__ == "__main__":
    config_path = os.getenv("JOB_CONFIG_PATH", "/configs/job.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        result = train(config)
        logger.info(f"Training complete: {result}")
    else:
        # Run as HTTP service (local dev)
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8003)
