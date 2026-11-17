from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import app_support
from .config import env_first, env_path
from .schemas import (
    BacktestRequest,
    BacktestResponse,
    ErrorDetails,
    ErrorResponse,
    ForecastPoint,
    ForecastRequest,
    JobCreateRequest,
    JobCreateResponse,
    JobStatusResponse,
    TimeSeriesRecord,
    TrainRequest,
    TrainResponse,
)

bi = app_support.bi
as_dict = app_support.as_dict
as_list = app_support.as_list
as_float_list = app_support.as_float_list
sigmoid = app_support.sigmoid
extract_state_dict = app_support.extract_state_dict
try_torch_load_weights = app_support.try_torch_load_weights
load_joblib_artifact = app_support.load_joblib_artifact
env_first_alias = env_first
env_path_alias = env_path


@dataclass(frozen=True)
class RuntimeCompatBindings:
    model_artifact_dir: Callable[[str], Path]
    artifact_relpath: Callable[[str, str], str]
    artifact_abspath: Callable[[str], Path]
    write_json: Callable[[Path, dict[str, Any]], None]
    read_json: Callable[[Path], dict[str, Any]]
    load_models_from_store: Callable[[], dict[str, dict[str, Any]]]
    save_models_to_store: Callable[[dict[str, dict[str, Any]]], None]
    load_fd004_benchmark_summary: Callable[[], dict[str, Any]]


def bind_runtime_accessors(
    *,
    model_artifacts_root: Callable[[], Path],
    trained_models_store_path: Callable[[], Path],
    fd004_benchmark_summary_path: Callable[[], Path],
    logger: Callable[[], Any],
) -> RuntimeCompatBindings:
    return RuntimeCompatBindings(
        model_artifact_dir=lambda model_id: app_support.model_artifact_dir(
            model_artifacts_root(),
            model_id,
        ),
        artifact_relpath=lambda model_id, filename: app_support.artifact_relpath(
            model_id, filename
        ),
        artifact_abspath=lambda relpath: app_support.artifact_abspath(
            model_artifacts_root(),
            relpath,
        ),
        write_json=lambda path, payload: app_support.write_json(path, payload),
        read_json=lambda path: app_support.read_json(path),
        load_models_from_store=lambda: app_support.load_models_from_store(
            trained_models_store_path()
        ),
        save_models_to_store=lambda models: app_support.save_models_to_store(
            trained_models_store_path(),
            models,
        ),
        load_fd004_benchmark_summary=lambda: app_support.load_fd004_benchmark_summary(
            fd004_benchmark_summary_path(),
            logger=logger(),
        ),
    )


__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "ErrorDetails",
    "ErrorResponse",
    "ForecastRequest",
    "ForecastPoint",
    "JobCreateRequest",
    "JobCreateResponse",
    "JobStatusResponse",
    "TimeSeriesRecord",
    "TrainRequest",
    "TrainResponse",
    "bi",
    "as_dict",
    "as_list",
    "as_float_list",
    "sigmoid",
    "extract_state_dict",
    "try_torch_load_weights",
    "load_joblib_artifact",
    "env_first_alias",
    "env_path_alias",
    "RuntimeCompatBindings",
    "bind_runtime_accessors",
]
