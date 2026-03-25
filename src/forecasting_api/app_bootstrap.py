from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .config import env_first, env_path
from .job_store import JobStore
from .runtime_state import AppRuntimeState


@dataclass(frozen=True)
class RuntimePaths:
    data_dir: Path
    trained_models_store_path: Path
    model_registry_db_path: Path
    model_artifacts_root: Path
    request_audit_log_path: Path
    drift_baseline_path: Path
    model_promotion_registry_path: Path
    fd004_benchmark_summary_path: Path


class RuntimeBootstrap:
    def __init__(
        self,
        *,
        paths: RuntimePaths,
        build_job_store: Callable[[], JobStore],
        load_trained_models: Callable[[], dict[str, dict[str, Any]]],
    ) -> None:
        self.paths = paths
        self.app_factory_lock = Lock()
        self.rate_limit_lock = Lock()
        self.rate_limit_hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self.runtime_state = AppRuntimeState(
            build_job_store=build_job_store,
            load_trained_models=load_trained_models,
        )

    def set_job_store(self, value: JobStore | None) -> JobStore | None:
        return self.runtime_state.set_job_store(value)

    def set_trained_models(
        self,
        value: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]] | None:
        return self.runtime_state.set_trained_models(value)

    def require_job_store(self, current: JobStore | None = None) -> JobStore:
        return self.runtime_state.require_job_store(current)

    def require_trained_models(
        self,
        current: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        return self.runtime_state.require_trained_models(current)


def default_runtime_paths(app_file: str) -> RuntimePaths:
    repo_root = Path(app_file).resolve().parents[2]
    data_dir = repo_root / "runtime"
    return RuntimePaths(
        data_dir=data_dir,
        trained_models_store_path=env_path(
            data_dir / "trained_models.json",
            "RULFM_TRAINED_MODELS_STORE_PATH",
        ),
        model_registry_db_path=env_path(
            data_dir / "trained_models.db",
            "RULFM_MODEL_REGISTRY_DB_PATH",
        ),
        model_artifacts_root=env_path(
            data_dir / "model_artifacts",
            "RULFM_MODEL_ARTIFACTS_ROOT",
        ),
        request_audit_log_path=env_path(
            data_dir / "request_audit_log.jsonl",
            "RULFM_FORECASTING_API_AUDIT_LOG_PATH",
        ),
        drift_baseline_path=env_path(
            data_dir / "drift_baseline.json",
            "RULFM_DRIFT_BASELINE_PATH",
        ),
        model_promotion_registry_path=env_path(
            data_dir / "model_promotions.json",
            "RULFM_MODEL_PROMOTION_REGISTRY_PATH",
        ),
        fd004_benchmark_summary_path=env_path(
            data_dir / "fd004_benchmark_summary.json",
            "RULFM_FD004_BENCHMARK_SUMMARY_PATH",
        ),
    )


def build_runtime_bootstrap(
    *,
    paths: RuntimePaths,
    build_job_store: Callable[[], JobStore],
    load_trained_models: Callable[[], dict[str, dict[str, Any]]],
) -> RuntimeBootstrap:
    return RuntimeBootstrap(
        paths=paths,
        build_job_store=build_job_store,
        load_trained_models=load_trained_models,
    )


def audit_log_path(default_path: Path) -> Path:
    raw = env_first("RULFM_FORECASTING_API_AUDIT_LOG_PATH")
    return Path(raw) if raw else default_path