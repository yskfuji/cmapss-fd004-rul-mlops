from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

from forecasting_api.domain import stable_models
from forecasting_api.errors import ApiError
from forecasting_api.services.runtime import TrainServiceDeps

from . import training_helpers


def build_train_service_deps(
    *,
    api_error_cls: type[ApiError],
    trained_models: Callable[[], dict[str, dict[str, Any]]],
    require_monotonic_increasing: Callable[[list[Any]], None],
    model_artifact_dir: Callable[[str], Path],
    write_json: Callable[[Path, dict[str, Any]], None],
    artifact_relpath: Callable[[str, str], str],
    artifact_abspath: Callable[[str], Path],
    save_trained_model: Callable[[dict[str, Any]], None],
    start_run: Callable[..., AbstractContextManager[Any]],
    log_params: Callable[[dict[str, Any]], None],
    log_metrics: Callable[[dict[str, float]], None],
    log_dict_artifact: Callable[[str, dict[str, Any]], None],
    log_artifact: Callable[[Path], None],
) -> TrainServiceDeps:
    def _assert_model_algo_available(algo: str | None) -> str:
        return training_helpers.assert_model_algo_available(algo, api_error_cls=api_error_cls)

    def _fit_ridge_lags_model(req: Any) -> dict[str, Any]:
        return training_helpers.fit_ridge_lags_model(
            req,
            ridge_lags_choose_k=stable_models.ridge_lags_choose_k,
            ridge_lags_fit_series=stable_models.ridge_lags_fit_series,
        )

    def _train_public_gbdt_entry(req: Any, *, model_id: str) -> dict[str, Any]:
        return training_helpers.train_public_gbdt_entry(
            req,
            model_id=model_id,
            model_artifact_dir=model_artifact_dir,
            write_json=write_json,
            artifact_relpath=artifact_relpath,
        )

    def _train_hybrid_entry(req: Any, *, model_id: str) -> dict[str, Any]:
        return training_helpers.train_hybrid_entry(
            req,
            model_id=model_id,
            model_artifact_dir=model_artifact_dir,
            write_json=write_json,
            artifact_relpath=artifact_relpath,
        )

    return TrainServiceDeps(
        api_error_cls=api_error_cls,
        trained_models=trained_models,
        require_monotonic_increasing=require_monotonic_increasing,
        normalize_base_model_name=training_helpers.normalize_base_model_name,
        assert_model_algo_available=_assert_model_algo_available,
        fit_ridge_lags_model=_fit_ridge_lags_model,
        train_public_gbdt_entry=_train_public_gbdt_entry,
        train_hybrid_entry=_train_hybrid_entry,
        model_artifact_dir=model_artifact_dir,
        write_json=write_json,
        artifact_relpath=artifact_relpath,
        artifact_abspath=artifact_abspath,
        save_trained_model=save_trained_model,
        start_run=start_run,
        log_params=log_params,
        log_metrics=log_metrics,
        log_dict_artifact=log_dict_artifact,
        log_artifact=log_artifact,
    )