from __future__ import annotations

from contextlib import AbstractContextManager
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from forecasting_api.errors import ApiError
from forecasting_api.job_store import JobStore
from forecasting_api.schemas import (
    BacktestRequest,
    BacktestResponse,
    ForecastRequest,
    ForecastResponse,
    TimeSeriesRecord,
    TrainRequest,
)


class ForecastWithTrainedModelFn(Protocol):
    def __call__(
        self,
        req: ForecastRequest,
        *,
        step: timedelta,
        trained: dict[str, Any],
    ) -> ForecastResponse: ...


class ForecastWithGbdtModelFn(Protocol):
    def __call__(
        self,
        req: ForecastRequest,
        *,
        step: timedelta,
        trained: dict[str, Any],
    ) -> ForecastResponse: ...


class HybridForecastModelFn(Protocol):
    def __call__(
        self,
        req: ForecastRequest,
        *,
        step: timedelta,
        trained: dict[str, Any],
    ) -> ForecastResponse: ...


class TorchForecastModelFn(Protocol):
    def __call__(
        self,
        req: ForecastRequest,
        *,
        step: timedelta,
        trained: dict[str, Any],
    ) -> ForecastResponse: ...


class RidgeBacktestFn(Protocol):
    def __call__(self, req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse: ...


class GbdtBacktestFn(Protocol):
    def __call__(self, req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse: ...


class HybridBacktestFn(Protocol):
    def __call__(self, req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse: ...


class TorchBacktestFn(Protocol):
    def __call__(self, req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse: ...


class StartRunFn(Protocol):
    def __call__(
        self,
        *,
        run_name: str,
        tags: dict[str, str],
    ) -> AbstractContextManager[Any]: ...


class TrainPublicGbdtEntryFn(Protocol):
    def __call__(self, req: TrainRequest, *, model_id: str) -> dict[str, Any]: ...


class TrainHybridEntryFn(Protocol):
    def __call__(self, req: TrainRequest, *, model_id: str) -> dict[str, Any]: ...


class BuildJobErrorPayloadFn(Protocol):
    def __call__(
        self,
        *,
        error_code: str,
        message: str,
        details: dict[str, Any] | Any | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ForecastServiceDeps:
    api_error_cls: type[ApiError]
    trained_models: Callable[[], MutableMapping[str, dict[str, Any]]]
    ensure_quantiles_level_exclusive: Callable[[ForecastRequest], None]
    require_monotonic_increasing: Callable[[list[TimeSeriesRecord]], None]
    require_trained_model: Callable[[str | None], None]
    sync_max_points: Callable[[], int]
    require_frequency_or_infer: Callable[[ForecastRequest], timedelta]
    require_no_gaps_if_missing_policy_error: Callable[[ForecastRequest, timedelta], None]
    assert_model_algo_available: Callable[[str | None], str]
    forecast_with_trained_model: ForecastWithTrainedModelFn
    forecast_with_gbdt_model: ForecastWithGbdtModelFn
    hybrid_forecast_model: HybridForecastModelFn
    torch_forecast_model: TorchForecastModelFn
    naive_forecast: Callable[[ForecastRequest, timedelta], ForecastResponse]


@dataclass(frozen=True)
class BacktestServiceDeps:
    trained_models: Callable[[], MutableMapping[str, dict[str, Any]]]
    require_monotonic_increasing: Callable[[list[TimeSeriesRecord]], None]
    require_trained_model: Callable[[str | None], None]
    assert_model_algo_available: Callable[[str | None], str]
    ridge_lags_backtest: RidgeBacktestFn
    gbdt_backtest: GbdtBacktestFn
    hybrid_backtest: HybridBacktestFn
    torch_backtest: TorchBacktestFn
    naive_backtest: Callable[[BacktestRequest], BacktestResponse]
    start_run: StartRunFn
    log_params: Callable[[dict[str, Any]], None]
    log_metrics: Callable[[dict[str, float]], None]
    log_dict_artifact: Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class TrainServiceDeps:
    api_error_cls: type[ApiError]
    trained_models: Callable[[], MutableMapping[str, dict[str, Any]]]
    require_monotonic_increasing: Callable[[list[TimeSeriesRecord]], None]
    normalize_base_model_name: Callable[[str | None], str]
    assert_model_algo_available: Callable[[str | None], str]
    fit_ridge_lags_model: Callable[[TrainRequest], dict[str, Any]]
    train_public_gbdt_entry: TrainPublicGbdtEntryFn
    train_hybrid_entry: TrainHybridEntryFn
    model_artifact_dir: Callable[[str], Path]
    write_json: Callable[[Path, dict[str, Any]], None]
    artifact_relpath: Callable[[str, str], str]
    artifact_abspath: Callable[[str], Path]
    save_trained_model: Callable[[dict[str, Any]], None]
    start_run: StartRunFn
    log_params: Callable[[dict[str, Any]], None]
    log_metrics: Callable[[dict[str, float]], None]
    log_dict_artifact: Callable[[str, dict[str, Any]], None]
    log_artifact: Callable[[Path], None]


@dataclass(frozen=True)
class JobsServiceDeps:
    api_error_cls: type[ApiError]
    job_store: Callable[[], JobStore]
    build_job_error_payload: BuildJobErrorPayloadFn
    run_forecast: Callable[[ForecastRequest], Any]
    run_train: Callable[[TrainRequest], Any]
    run_backtest_request: Callable[[BacktestRequest], Any]
