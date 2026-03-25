# ruff: noqa: E501
from forecasting_api import app as app_module
from forecasting_api import hybrid_runtime, training_helpers
from forecasting_api.domain import stable_models
from forecasting_api.services.backtest_service import (
    configure_backtest_service,
    run_backtest_request,
)
from forecasting_api.services.runtime import BacktestServiceDeps


def _backtest_request(model_id: str) -> app_module.BacktestRequest:
    return app_module.BacktestRequest(
        horizon=2,
        folds=2,
        metric="rmse",
        model_id=model_id,
        data=[
            {
                "series_id": "s1",
                "timestamp": f"2026-03-{day:02d}T00:00:00Z",
                "y": float(day),
            }
            for day in range(1, 10)
        ],
    )


def _configure_backtest_service() -> None:
    configure_backtest_service(
        BacktestServiceDeps(
            trained_models=app_module._require_trained_models,
            require_monotonic_increasing=lambda records: app_module._require_monotonic_increasing(records),
            require_trained_model=lambda model_id: app_module._require_trained_model(model_id),
            assert_model_algo_available=lambda algo: training_helpers.assert_model_algo_available(
                algo,
                api_error_cls=app_module.ApiError,
            ),
            ridge_lags_backtest=lambda req, *, trained: stable_models.ridge_lags_backtest(
                req,
                trained=trained,
                ridge_lags_choose_k_fn=stable_models.ridge_lags_choose_k,
                ridge_lags_fit_series_fn=stable_models.ridge_lags_fit_series,
                ridge_lags_forecast_series_fn=stable_models.ridge_lags_forecast_series,
                metric_value_fn=stable_models.metric_value,
            ),
            gbdt_backtest=lambda req, *, trained: stable_models.gbdt_backtest(
                req,
                trained=trained,
                read_json=app_module._read_json,
                artifact_abspath=app_module._artifact_abspath,
                load_joblib_artifact=app_module._load_joblib_artifact,
                predict_hgb_next_fn=stable_models.predict_hgb_next,
                metric_value_fn=stable_models.metric_value,
            ),
            hybrid_backtest=lambda req, *, trained: hybrid_runtime.hybrid_backtest(
                req,
                trained=trained,
                read_json=app_module._read_json,
                artifact_abspath=app_module._artifact_abspath,
                try_torch_load_weights=app_module._try_torch_load_weights,
                extract_state_dict=app_module._extract_state_dict,
                load_joblib_artifact=app_module._load_joblib_artifact,
                predict_hgb_next_fn=stable_models.predict_hgb_next,
                gate_step_payload_fn=training_helpers.hybrid_gate_step_payload,
                hybrid_condition_cluster_key_fn=training_helpers.hybrid_condition_cluster_key,
                metric_value_fn=stable_models.metric_value,
            ),
            torch_backtest=lambda req, *, trained: app_module._torch_backtest(req, trained=trained),
            naive_backtest=lambda req: app_module._naive_backtest(req),
            start_run=lambda *args, **kwargs: app_module.start_run(*args, **kwargs),
            log_params=lambda payload: app_module.log_params(payload),
            log_metrics=lambda payload: app_module.log_metrics(payload),
            log_dict_artifact=lambda name, payload: app_module.log_dict_artifact(name, payload),
        )
    )


def test_run_backtest_request_uses_hybrid_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        training_helpers,
        "assert_model_algo_available",
        lambda algo, api_error_cls: str(algo),
    )
    app_module._set_runtime_trained_models(
        {"model_hybrid": {"model_id": "model_hybrid", "algo": "gbdt_afno_hybrid_v1"}}
    )
    monkeypatch.setattr(
        hybrid_runtime,
        "hybrid_backtest",
        lambda req, trained, **kwargs: app_module.BacktestResponse(metrics={"rmse": 0.33}),
    )
    _configure_backtest_service()

    result = run_backtest_request(_backtest_request("model_hybrid"))

    assert result.metrics["rmse"] == 0.33


def test_run_backtest_request_uses_torch_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        training_helpers,
        "assert_model_algo_available",
        lambda algo, api_error_cls: str(algo),
    )
    app_module._set_runtime_trained_models(
        {"model_torch": {"model_id": "model_torch", "algo": "afnocg3_v1"}}
    )
    monkeypatch.setattr(
        app_module,
        "_torch_backtest",
        lambda req, trained: app_module.BacktestResponse(metrics={"rmse": 0.21}),
    )
    _configure_backtest_service()

    result = run_backtest_request(_backtest_request("model_torch"))

    assert result.metrics["rmse"] == 0.21