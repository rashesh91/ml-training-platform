"""
Bridges ml-training-platform → ml-inference-gitops.

When a model is promoted to Production in MLflow this module:
  1. Patches applications/voice-platform/values.yaml in the inference GitOps
     repo via the GitHub Contents API (no git binary needed in the container).
  2. Triggers an ArgoCD app sync so the new model rolls out immediately.

Required env vars (all optional — skipped gracefully if absent):
  GITHUB_TOKEN                 PAT with repo write access
  INFERENCE_GITOPS_REPO        e.g. "rashesh91/ml-inference-gitops"
  INFERENCE_GITOPS_BRANCH      branch to patch (default: main)
  INFERENCE_GITOPS_VALUES_PATH path in repo (default: applications/voice-platform/values.yaml)
  ARGOCD_SERVER                e.g. "https://argocd.example.com"
  ARGOCD_TOKEN                 ArgoCD API token (from ServiceAccount or UI)
  ARGOCD_APP_NAME              app to sync (default: voice-platform-prod)
  ARGOCD_INSECURE              set "true" to skip TLS verification (self-signed certs)
"""

import base64
import logging
import os

import httpx
import yaml

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
INFERENCE_GITOPS_REPO = os.getenv("INFERENCE_GITOPS_REPO", "rashesh91/ml-inference-gitops")
INFERENCE_GITOPS_BRANCH = os.getenv("INFERENCE_GITOPS_BRANCH", "main")
INFERENCE_GITOPS_VALUES_PATH = os.getenv(
    "INFERENCE_GITOPS_VALUES_PATH",
    "applications/voice-platform/values.yaml",
)
ARGOCD_SERVER = os.getenv("ARGOCD_SERVER", "")
ARGOCD_TOKEN = os.getenv("ARGOCD_TOKEN", "")
ARGOCD_APP_NAME = os.getenv("ARGOCD_APP_NAME", "voice-platform-prod")
ARGOCD_INSECURE = os.getenv("ARGOCD_INSECURE", "false").lower() == "true"


def _is_configured() -> bool:
    missing = [v for v, k in [
        (GITHUB_TOKEN, "GITHUB_TOKEN"),
        (ARGOCD_SERVER, "ARGOCD_SERVER"),
        (ARGOCD_TOKEN, "ARGOCD_TOKEN"),
    ] if not v]
    if missing:
        logger.warning("Inference deploy skipped — missing env vars: %s", ", ".join(missing))
        return False
    return True


async def trigger_inference_deploy(model_name: str, version: str, run_id: str) -> dict:
    """
    Patch the inference GitOps repo values.yaml with the promoted model,
    then trigger an ArgoCD sync. Safe to call without env vars — returns
    {"skipped": True} instead of raising.
    """
    if not _is_configured():
        return {"skipped": True, "reason": "env vars not configured"}

    mlflow_uri = f"models:/{model_name}/Production"
    git_result = await _patch_values_yaml(model_name, version, mlflow_uri)
    argocd_result = await _sync_argocd_app()

    logger.info(
        "Inference deploy complete: model=%s version=%s uri=%s",
        model_name, version, mlflow_uri,
    )
    return {
        "skipped": False,
        "model": model_name,
        "version": version,
        "mlflow_uri": mlflow_uri,
        "git": git_result,
        "argocd": argocd_result,
    }


async def _patch_values_yaml(model_name: str, version: str, mlflow_uri: str) -> dict:
    """
    Fetch values.yaml from GitHub, update llmAgent.model fields, commit back.
    Uses the GitHub Contents API so no git binary is required in the container.
    """
    api_url = (
        f"https://api.github.com/repos/{INFERENCE_GITOPS_REPO}"
        f"/contents/{INFERENCE_GITOPS_VALUES_PATH}"
    )
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(api_url, headers=headers, params={"ref": INFERENCE_GITOPS_BRANCH})
        resp.raise_for_status()
        file_meta = resp.json()
        sha = file_meta["sha"]
        current_yaml = base64.b64decode(file_meta["content"]).decode("utf-8")

        values = yaml.safe_load(current_yaml)
        model_block = values.setdefault("llmAgent", {}).setdefault("model", {})
        model_block["name"] = model_name
        model_block["mlflowUri"] = mlflow_uri
        # Keep huggingfaceId in sync — vLLM uses this field; a custom init
        # container can download from MLflow and expose at cacheDir instead.
        model_block["huggingfaceId"] = mlflow_uri
        updated_yaml = yaml.dump(values, default_flow_style=False, allow_unicode=True)

        commit_msg = (
            f"chore(model): promote {model_name} v{version} to inference\n\n"
            f"MLflow URI: {mlflow_uri}\n"
            f"Automated by ml-training-platform"
        )
        put_resp = await client.put(
            api_url,
            headers=headers,
            json={
                "message": commit_msg,
                "content": base64.b64encode(updated_yaml.encode()).decode(),
                "sha": sha,
                "branch": INFERENCE_GITOPS_BRANCH,
            },
        )
        put_resp.raise_for_status()
        commit_sha = put_resp.json()["commit"]["sha"]
        logger.info("Committed values.yaml update: %s", commit_sha)
        return {"status": "committed", "sha": commit_sha, "branch": INFERENCE_GITOPS_BRANCH}


async def _sync_argocd_app() -> dict:
    """POST to ArgoCD sync endpoint and return the initial operation phase."""
    url = f"{ARGOCD_SERVER}/api/v1/applications/{ARGOCD_APP_NAME}/sync"
    headers = {"Authorization": f"Bearer {ARGOCD_TOKEN}"}

    async with httpx.AsyncClient(timeout=30.0, verify=not ARGOCD_INSECURE) as client:
        resp = await client.post(url, headers=headers, json={})
        resp.raise_for_status()
        phase = (
            resp.json()
            .get("status", {})
            .get("operationState", {})
            .get("phase", "Running")
        )
        logger.info("ArgoCD sync triggered: app=%s phase=%s", ARGOCD_APP_NAME, phase)
        return {"status": "syncing", "phase": phase, "app": ARGOCD_APP_NAME}
