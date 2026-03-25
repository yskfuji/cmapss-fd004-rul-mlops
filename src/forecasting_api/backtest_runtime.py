from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from forecasting_api.domain import stable_models
from forecasting_api.errors import ApiError
from forecasting_api.schemas import BacktestRequest, BacktestResponse
from forecasting_api.services.runtime import BacktestServiceDeps, TorchBacktestFn

from . import hybrid_runtime
from . import training_helpers


def build_backtest_service_deps(
    *,
    api_error_cls: type[ApiError],
    trained_models: Callable[[], dict[str, dict[str, Any]]],
    require_monotonic_increasing: Callable[[list[Any]], None],
    require_trained_model: Callable[[str | None], None],
    read_json: Callable[[Path], dict[str, Any]],
    artifact_abspath: Callable[[str], Path],
    try_torch_load_weights: Callable[[Path], dict[str, Any]],
    extract_state_dict: Callable[[dict[str, Any]], dict[str, Any]],
    load_joblib_artifact: Callable[[Path], Any],
    torch_backtest: TorchBacktestFn,
    naive_backtest: Callable[[BacktestRequest], BacktestResponse],
    start_run: Callable[..., AbstractContextManager[Any]],
    log_params: Callable[[dict[str, Any]], None],
    log_metrics: Callable[[dict[str, float]], None],
    log_dict_artifact: Callable[[str, dict[str, Any]], None],
) -> BacktestServiceDeps:
    def _assert_model_algo_available(algo: str | None) -> str:
        return training_helpers.assert_model_algo_available(algo, api_error_cls=api_error_cls)

    def _ridge_lags_backtest(req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse:
        return stable_models.ridge_lags_backtest(
            req,
            trained=trained,
            ridge_lags_choose_k_fn=stable_models.ridge_lags_choose_k,
            ridge_lags_fit_series_fn=stable_models.ridge_lags_fit_series,
            ridge_lags_forecast_series_fn=stable_models.ridge_lags_forecast_series,
            metric_value_fn=stable_models.metric_value,
        )

    def _gbdt_backtest(req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse:
        return stable_models.gbdt_backtest(
            req,
            trained=trained,
            read_json=read_json,
            artifact_abspath=artifact_abspath,
            load_joblib_artifact=load_joblib_artifact,
            predict_hgb_next_fn=stable_models.predict_hgb_next,
            metric_value_fn=stable_models.metric_value,
        )

    def _hybrid_backtest(req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse:
        return hybrid_runtime.hybrid_backtest(
            req,
            trained=trained,
            read_json=read_json,
            artifact_abspath=artifact_abspath,
            try_torch_load_weights=try_torch_load_weights,
            extract_state_dict=extract_state_dict,
            load_joblib_artifact=load_joblib_artifact,
            predict_hgb_next_fn=stable_models.predict_hgb_next,
            gate_step_payload_fn=training_helpers.hybrid_gate_step_payload,
            hybrid_condition_cluster_key_fn=training_helpers.hybrid_condition_cluster_key,
            metric_value_fn=stable_models.metric_value,
        )

    def _torch_backtest(req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse:
        return torch_backtest(req, trained=trained)

    return BacktestServiceDeps(
        trained_models=trained_models,
        require_monotonic_increasing=require_monotonic_increasing,
        require_trained_model=require_trained_model,
        assert_model_algo_available=_assert_model_algo_available,
        ridge_lags_backtest=_ridge_lags_backtest,
        gbdt_backtest=_gbdt_backtest,
        hybrid_backtest=_hybrid_backtest,
        torch_backtest=_torch_backtest,
        naive_backtest=naive_backtest,
        start_run=start_run,
        log_params=log_params,
        log_metrics=log_metrics,
        log_dict_artifact=log_dict_artifact,
    )