from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from forecasting_api.domain import stable_models
from forecasting_api.errors import ApiError
from forecasting_api.schemas import ForecastRequest, ForecastResponse
from forecasting_api.services.runtime import ForecastServiceDeps

from . import hybrid_runtime
from . import torch_runtime
from . import training_helpers


def build_forecast_service_deps(
    *,
    api_error_cls: type[ApiError],
    bi: Callable[[str, str], str],
    trained_models: Callable[[], dict[str, dict[str, Any]]],
    ensure_quantiles_level_exclusive: Callable[[ForecastRequest], None],
    require_monotonic_increasing: Callable[[list[Any]], None],
    require_trained_model: Callable[[str | None], None],
    sync_max_points: Callable[[], int],
    require_frequency_or_infer: Callable[[ForecastRequest], timedelta],
    require_no_gaps_if_missing_policy_error: Callable[[ForecastRequest, timedelta], None],
    read_json: Callable[[Path], dict[str, Any]],
    artifact_abspath: Callable[[str], Path],
    try_torch_load_weights: Callable[[Path], dict[str, Any]],
    extract_state_dict: Callable[[dict[str, Any]], dict[str, Any]],
    load_joblib_artifact: Callable[[Path], Any],
) -> ForecastServiceDeps:
    def _assert_model_algo_available(algo: str | None) -> str:
        return training_helpers.assert_model_algo_available(algo, api_error_cls=api_error_cls)

    def _forecast_with_trained_model(
        req: ForecastRequest,
        *,
        step: timedelta,
        trained: dict[str, Any],
    ) -> ForecastResponse:
        return stable_models.forecast_with_trained_model(
            req,
            step=step,
            trained=trained,
            ridge_lags_choose_k_fn=stable_models.ridge_lags_choose_k,
            ridge_lags_fit_series_fn=stable_models.ridge_lags_fit_series,
            ridge_lags_forecast_series_fn=stable_models.ridge_lags_forecast_series,
            safe_std_fn=stable_models.safe_std,
            quantile_nearest_rank_fn=stable_models.quantile_nearest_rank,
            build_residuals_evidence_fn=stable_models.build_residuals_evidence,
            naive_forecast_fn=stable_models.naive_forecast,
        )

    def _forecast_with_gbdt_model(
        req: ForecastRequest,
        *,
        step: timedelta,
        trained: dict[str, Any],
    ) -> ForecastResponse:
        return stable_models.forecast_with_gbdt_model(
            req,
            step=step,
            trained=trained,
            read_json=read_json,
            artifact_abspath=artifact_abspath,
            load_joblib_artifact=load_joblib_artifact,
            predict_hgb_next_fn=stable_models.predict_hgb_next,
            quantile_nearest_rank_fn=stable_models.quantile_nearest_rank,
            build_residuals_evidence_fn=stable_models.build_residuals_evidence,
        )

    def _hybrid_forecast_model(
        req: ForecastRequest,
        *,
        step: timedelta,
        trained: dict[str, Any],
    ) -> ForecastResponse:
        return hybrid_runtime.forecast_with_hybrid_model(
            req,
            step=step,
            trained=trained,
            read_json=read_json,
            artifact_abspath=artifact_abspath,
            try_torch_load_weights=try_torch_load_weights,
            extract_state_dict=extract_state_dict,
            load_joblib_artifact=load_joblib_artifact,
            predict_hgb_next_fn=stable_models.predict_hgb_next,
            gate_step_payload_fn=training_helpers.hybrid_gate_step_payload,
            hybrid_condition_cluster_key_fn=training_helpers.hybrid_condition_cluster_key,
        )

    def _torch_forecast_model(
        req: ForecastRequest,
        *,
        step: timedelta,
        trained: dict[str, Any],
    ) -> ForecastResponse:
        return torch_runtime.forecast_with_torch_model(
            req,
            step=step,
            trained=trained,
            read_json=read_json,
            artifact_abspath=artifact_abspath,
            try_torch_load_weights=try_torch_load_weights,
            extract_state_dict=extract_state_dict,
            quantile_nearest_rank_fn=stable_models.quantile_nearest_rank,
            build_residuals_evidence_fn=stable_models.build_residuals_evidence,
            bi=bi,
            api_error_cls=api_error_cls,
        )

    return ForecastServiceDeps(
        api_error_cls=api_error_cls,
        trained_models=trained_models,
        ensure_quantiles_level_exclusive=ensure_quantiles_level_exclusive,
        require_monotonic_increasing=require_monotonic_increasing,
        require_trained_model=require_trained_model,
        sync_max_points=sync_max_points,
        require_frequency_or_infer=require_frequency_or_infer,
        require_no_gaps_if_missing_policy_error=require_no_gaps_if_missing_policy_error,
        assert_model_algo_available=_assert_model_algo_available,
        forecast_with_trained_model=_forecast_with_trained_model,
        forecast_with_gbdt_model=_forecast_with_gbdt_model,
        hybrid_forecast_model=_hybrid_forecast_model,
        torch_forecast_model=_torch_forecast_model,
        naive_forecast=stable_models.naive_forecast,
    )