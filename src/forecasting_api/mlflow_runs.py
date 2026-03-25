from __future__ import annotations

import math
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _tracking_uri() -> str | None:
    value = os.getenv("RULFM_MLFLOW_TRACKING_URI", "").strip()
    return value or None


def _experiment_name() -> str:
    value = os.getenv("RULFM_MLFLOW_EXPERIMENT", "rulfm-forecasting").strip()
    return value or "rulfm-forecasting"


def mlflow_enabled() -> bool:
    return _tracking_uri() is not None


def _load_mlflow() -> Any | None:
    if not mlflow_enabled():
        return None
    try:
        # MLflow 2.20.x pulls gateway config models that still emit a
        # Pydantic v2 class-config deprecation warning. Keep stable runs
        # warning-free until the dependency is upgraded upstream.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Support for class-based .*",
                category=DeprecationWarning,
            )
            import mlflow

            mlflow.set_tracking_uri(str(_tracking_uri()))
            mlflow.set_experiment(_experiment_name())
        return mlflow
    except Exception:
        return None


@contextmanager
def start_run(
    run_name: str, *, tags: dict[str, str] | None = None, nested: bool = False
) -> Iterator[Any | None]:
    mlflow = _load_mlflow()
    if mlflow is None:
        yield None
        return
    try:
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            if tags:
                mlflow.set_tags({str(key): str(value) for key, value in tags.items()})
            yield run
    except Exception:
        yield None


def log_params(params: dict[str, Any]) -> None:
    mlflow = _load_mlflow()
    if mlflow is None:
        return
    safe: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        safe[str(key)] = str(value)
    if safe:
        try:
            mlflow.log_params(safe)
        except Exception:
            return


def log_metrics(metrics: dict[str, Any]) -> None:
    mlflow = _load_mlflow()
    if mlflow is None:
        return
    safe: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, int | float) and math.isfinite(float(value)):
            safe[str(key)] = float(value)
    if safe:
        try:
            mlflow.log_metrics(safe)
        except Exception:
            return


def log_dict_artifact(name: str, payload: dict[str, Any]) -> None:
    mlflow = _load_mlflow()
    if mlflow is None:
        return
    try:
        mlflow.log_dict(payload, name)
    except Exception:
        return


def log_artifact(path: str | Path) -> None:
    mlflow = _load_mlflow()
    if mlflow is None:
        return
    artifact_path = Path(path)
    if not artifact_path.exists():
        return
    try:
        mlflow.log_artifact(str(artifact_path))
    except Exception:
        return
