"""
Evaluation service: compares fine-tuned model vs base model on a benchmark.
Promotes fine-tuned model to MLflow Staging if it passes the threshold.

Local dev: called via HTTP POST /evaluate from fine-tuner
K8s:       run as Argo Workflow step container
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import boto3
import mlflow
import torch
from botocore.client import Config
from evaluate import load as load_metric
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ROUGE_THRESHOLD = float(os.getenv("ROUGE_THRESHOLD", "0.1"))  # 10% improvement over base

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Benchmark prompts — generic instruction-following tests
BENCHMARK_PROMPTS = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("Name three primary colors.", "red, blue, yellow"),
    ("What does CPU stand for?", "Central Processing Unit"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Convert 0 degrees Celsius to Fahrenheit.", "32"),
    ("What is the square root of 16?", "4"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many days are in a leap year?", "366"),
]


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def download_checkpoint(bucket: str, prefix: str, local_dir: str):
    s3 = _s3()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            dest = Path(local_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            logger.info(f"Downloaded {key} → {dest}")


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(
        f"### Instruction:\n{prompt}\n\n### Response:\n",
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_rouge(predictions: list[str], references: list[str]) -> float:
    rouge = load_metric("rouge")
    scores = rouge.compute(predictions=predictions, references=references)
    return scores["rougeL"]


def evaluate(config: dict) -> dict:
    job_id = config["job_id"]
    model_name = config["model_name"]
    mlflow_run_id = config["mlflow_run_id"]
    base_model_name = config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    checkpoint_bucket = config.get("checkpoint_bucket", "checkpoints")
    checkpoint_prefix = config.get("checkpoint_prefix", f"{job_id}/checkpoint")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluating on device: {device}")

    prompts = [p for p, _ in BENCHMARK_PROMPTS]
    references = [r for _, r in BENCHMARK_PROMPTS]

    # --- Evaluate base model ---
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    base_model.eval()

    base_preds = [generate_response(base_model, tokenizer, p) for p in prompts]
    base_rouge = compute_rouge(base_preds, references)
    logger.info(f"Base model ROUGE-L: {base_rouge:.4f}")

    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Evaluate fine-tuned model ---
    with tempfile.TemporaryDirectory() as ckpt_dir:
        logger.info(f"Downloading checkpoint from s3://{checkpoint_bucket}/{checkpoint_prefix}")
        download_checkpoint(checkpoint_bucket, checkpoint_prefix, ckpt_dir)

        ft_base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)

        # Try loading as PEFT model, fall back to base if adapter not found
        try:
            ft_model = PeftModel.from_pretrained(ft_base, ckpt_dir).to(device)
            ft_model = ft_model.merge_and_unload()
        except Exception as e:
            logger.warning(f"PEFT load failed ({e}), evaluating base checkpoint")
            ft_model = ft_base

        ft_model.eval()
        ft_preds = [generate_response(ft_model, tokenizer, p) for p in prompts]
        ft_rouge = compute_rouge(ft_preds, references)
        logger.info(f"Fine-tuned model ROUGE-L: {ft_rouge:.4f}")

    improvement = ft_rouge - base_rouge
    passed = improvement >= ROUGE_THRESHOLD
    logger.info(f"Improvement: {improvement:+.4f} | Threshold: {ROUGE_THRESHOLD} | Passed: {passed}")

    # Log evaluation metrics to MLflow
    client = mlflow.tracking.MlflowClient()
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics({
            "eval_base_rouge_l": base_rouge,
            "eval_ft_rouge_l": ft_rouge,
            "eval_rouge_improvement": improvement,
            "eval_passed": float(passed),
        })

        # Save benchmark results as artifact
        results = {
            "prompts": prompts,
            "base_responses": base_preds,
            "ft_responses": ft_preds,
            "references": references,
            "base_rouge_l": base_rouge,
            "ft_rouge_l": ft_rouge,
            "improvement": improvement,
            "passed": passed,
        }
        import json, tempfile as tf
        with tf.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f, indent=2)
            mlflow.log_artifact(f.name, "evaluation")

    # Promote to Staging if passed
    if passed:
        try:
            for mv in client.search_model_versions(f"name='{model_name}'"):
                if mv.run_id == mlflow_run_id:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Staging",
                    )
                    logger.info(f"Promoted {model_name} v{mv.version} → Staging")
                    break
        except Exception as e:
            logger.warning(f"Could not promote model: {e}")

    return {
        "job_id": job_id,
        "base_rouge_l": base_rouge,
        "rouge_l": ft_rouge,
        "improvement": improvement,
        "passed": passed,
        "threshold": ROUGE_THRESHOLD,
    }


# --- FastAPI server ---

app = FastAPI(title="Evaluator Service")


class EvalRequest(BaseModel):
    job_id: str
    model_name: str
    mlflow_run_id: str
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    checkpoint_bucket: str = "checkpoints"
    checkpoint_prefix: str = ""


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate_endpoint(req: EvalRequest):
    import asyncio
    loop = asyncio.get_event_loop()
    config = req.dict()
    if not config["checkpoint_prefix"]:
        config["checkpoint_prefix"] = f"{req.job_id}/checkpoint"
    return await loop.run_in_executor(None, evaluate, config)


if __name__ == "__main__":
    config_path = os.getenv("JOB_CONFIG_PATH", "/configs/job.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        result = evaluate(config)
        logger.info(f"Evaluation result: {result}")
        # Write result to /tmp/eval-result.json for Argo Workflow to pick up
        with open("/tmp/eval-result.json", "w") as f:
            json.dump(result, f)
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8005)
