from __future__ import annotations

import os
from pathlib import Path


def validate_cloud_run_runtime(
    *,
    job_execution_backend: str,
    job_store_backend: str,
    job_store_postgres_dsn: str | None,
    model_registry_backend: str,
    model_registry_postgres_dsn: str | None,
    model_registry_db_path: Path,
    model_artifacts_root: Path,
    request_audit_log_path: Path,
    drift_baseline_path: Path,
    promotion_registry_path: Path,
) -> None:
    if not os.getenv("K_SERVICE"):
        return
    if job_execution_backend == "inprocess":
        raise RuntimeError(
            "Cloud Run deployment requires "
            "RULFM_JOB_EXECUTION_BACKEND=worker; "
            "inprocess jobs are not durable"
        )
    if job_store_backend != "postgres" or not job_store_postgres_dsn:
        raise RuntimeError(
            "Cloud Run deployment requires "
            "RULFM_JOB_STORE_BACKEND=postgres and "
            "RULFM_JOB_STORE_POSTGRES_DSN"
        )
    if model_registry_backend != "postgres" or not model_registry_postgres_dsn:
        raise RuntimeError(
            "Cloud Run deployment requires "
            "RULFM_MODEL_REGISTRY_BACKEND=postgres and "
            "RULFM_MODEL_REGISTRY_POSTGRES_DSN"
        )

    default_like_roots = [
        Path("/app/runtime"),
        Path("/app/src/forecasting_api/data"),
        Path("runtime"),
        Path("src/forecasting_api/data"),
    ]
    persisted_targets = {
        "RULFM_MODEL_REGISTRY_DB_PATH": model_registry_db_path,
        "RULFM_MODEL_ARTIFACTS_ROOT": model_artifacts_root,
        "RULFM_FORECASTING_API_AUDIT_LOG_PATH": request_audit_log_path,
        "RULFM_DRIFT_BASELINE_PATH": drift_baseline_path,
        "RULFM_MODEL_PROMOTION_REGISTRY_PATH": promotion_registry_path,
    }
    for env_name, target in persisted_targets.items():
        if env_name not in os.environ:
            raise RuntimeError(
                f"Cloud Run deployment requires explicit {env_name} for persistent runtime state"
            )
        normalized = target.resolve()
        if any(str(normalized).startswith(str(root)) for root in default_like_roots):
            raise RuntimeError(
                "Cloud Run deployment requires "
                f"{env_name} to point to persistent external storage, "
                "not packaged app data"
            )