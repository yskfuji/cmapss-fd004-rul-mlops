from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from forecasting_api import app as app_module
from forecasting_api import hybrid_runtime, training_helpers
from forecasting_api.app import create_app
from forecasting_api.domain import stable_models


def _forecast_payload() -> dict[str, object]:
    return {
        "horizon": 1,
        "data": [
            {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
            {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
        ],
    }


def _backtest_payload() -> dict[str, object]:
    data = []
    for day in range(20, 28):
        data.append(
            {
                "series_id": "s1",
                "timestamp": f"2026-03-{day:02d}T00:00:00Z",
                "y": float(day - 10),
            }
        )
    return {
        "horizon": 2,
        "folds": 2,
        "metric": "rmse",
        "data": data,
    }
def test_forecast_endpoint_rejects_quantiles_and_level_together(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    payload = _forecast_payload()
    payload["quantiles"] = [0.1, 0.9]
    payload["level"] = [90]
    response = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)
    assert response.status_code == 400
    assert response.json()["error_code"] == "V03"


def test_forecast_endpoint_rejects_non_monotonic_series(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    payload = {
        "horizon": 1,
        "data": [
            {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
            {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
        ],
    }
    response = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)
    assert response.status_code == 400
    assert response.json()["error_code"] == "V04"


def test_forecast_endpoint_requires_tenant_header_when_enabled(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_REQUIRE_TENANT", "1")
    client = TestClient(create_app())

    response = client.post(
        "/v1/forecast",
        headers={"x-api-key": "test-key"},
        json=_forecast_payload(),
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "A14"


def test_forecast_endpoint_rejects_invalid_tenant_header(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())

    response = client.post(
        "/v1/forecast",
        headers={"x-api-key": "test-key", "x-tenant-id": "tenant/../bad"},
        json=_forecast_payload(),
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "A14"


def test_forecast_endpoint_enforces_network_policy(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_PRIVATE_CONNECTIVITY_REQUIRED", "1")
    monkeypatch.setenv("RULFM_FORECASTING_API_IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("RULFM_FORECASTING_API_IP_ALLOWLIST", "10.0.0.0/8")
    client = TestClient(create_app())

    response = client.post(
        "/v1/forecast",
        headers={
            "x-api-key": "test-key",
            "x-tenant-id": "tenant-a",
            "x-connection-type": "public",
            "x-forwarded-for": "192.168.1.20",
        },
        json=_forecast_payload(),
    )

    assert response.status_code == 403
    assert response.json()["error_code"] == "A15"


def test_forecast_endpoint_requires_frequency_when_inference_fails(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    payload = {
        "horizon": 1,
        "data": [
            {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
        ],
    }
    response = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)
    assert response.status_code == 400
    assert response.json()["error_code"] == "V02"


def test_forecast_endpoint_rejects_unknown_options_fields(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    payload = _forecast_payload()
    payload["options"] = {"missing_policy": "ignore", "unexpected": True}

    response = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)

    assert response.status_code == 400


def test_forecast_endpoint_rejects_missing_timestamps_when_policy_is_error(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    payload = {
        "horizon": 1,
        "frequency": "1d",
        "options": {"missing_policy": "error"},
        "data": [
            {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
            {"series_id": "s1", "timestamp": "2026-03-22T00:00:00Z", "y": 12.0},
        ],
    }
    response = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "V05"
    assert body["details"]["missing_policy"] == "error"
    assert body["details"]["gaps"][0]["series_id"] == "s1"


def test_forecast_endpoint_rejects_unknown_model_id(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    payload = _forecast_payload()
    payload["model_id"] = "model_missing"
    response = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)
    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "M01"
    assert body["details"]["model_id"] == "model_missing"


def test_backtest_endpoint_rejects_unknown_model_id(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    response = client.post(
        "/v1/backtest",
        headers={"x-api-key": "test-key"},
        json={
            "horizon": 1,
            "folds": 2,
            "metric": "rmse",
            "model_id": "model_missing",
            "data": [
                {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
                {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
            ],
        },
    )
    assert response.status_code == 404
    assert response.json()["error_code"] == "M01"


def test_backtest_endpoint_returns_metrics_for_naive_model(monkeypatch, patched_backtest_runtime):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    response = client.post(
        "/v1/backtest",
        headers={"x-api-key": "test-key"},
        json=_backtest_payload(),
    )
    assert response.status_code == 200
    body = response.json()
    assert "rmse" in body["metrics"]
    assert len(body["by_series"]) == 1
    assert len(body["by_horizon"]) == 2
    assert len(body["by_fold"]) == 2


def _train_naive_get_model_id(client, headers: dict) -> str:
    # runtime_state is already isolated and persistence is patched to no-op,
    # so just POST to /v1/train and return the model_id.
    response = client.post(
        "/v1/train",
        headers=headers,
        json={
            "algo": "naive",
            "training_hours": 0.05,
            "data": [
                {"series_id": "s1", "timestamp": f"2026-03-{d:02d}T00:00:00Z", "y": float(d)}
                for d in range(1, 9)
            ],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["model_id"]


def test_forecast_with_trained_model_id_returns_different_predictions(
    monkeypatch,
    patched_app_runtime,
):
    """Non-naive path: train a model, forecast using its model_id, verify model_id roundtrip."""
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")

    client = TestClient(create_app())
    headers = {"x-api-key": "test-key"}

    model_id = _train_naive_get_model_id(client, headers)
    assert model_id.startswith("model_")

    # Verify trained model appears in /v1/models listing
    models_resp = client.get("/v1/models", headers=headers)
    assert models_resp.status_code == 200
    model_ids = [m["model_id"] for m in models_resp.json()["models"]]
    assert model_id in model_ids

    # Forecast using the trained model_id
    payload = _forecast_payload()
    payload["model_id"] = model_id
    payload["frequency"] = "1d"
    response = client.post("/v1/forecast", headers=headers, json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["forecasts"][0]["series_id"] == "s1"


@pytest.mark.experimental
def test_forecast_with_trained_torch_model_id(monkeypatch):
    from forecasting_api import torch_forecasters as torch_module

    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    app_module._set_runtime_trained_models({
        "model_torch": {
            "model_id": "model_torch",
            "algo": "afnocg3_v1",
            "context_len": 2,
            "input_dim": 1,
            "pooled_residuals": [0.2, 0.3, 0.4, 0.5, 0.6] * 4,
            "artifact": {
                "snapshot_json": "model_torch/snapshot.json",
                "weights_pt": "model_torch/weights.pt",
            },
        }
    })
    monkeypatch.setattr(app_module, "_read_json", lambda path: {"context_len": 2})
    monkeypatch.setattr(app_module, "_try_torch_load_weights", lambda path: {"state_dict": {}})
    monkeypatch.setattr(app_module, "_extract_state_dict", lambda ckpt: {})
    monkeypatch.setattr(
        torch_module,
        "forecast_univariate_torch",
        lambda **kwargs: [12.5],
    )

    client = TestClient(create_app())
    response = client.post(
        "/v1/forecast",
        headers={"x-api-key": "test-key"},
        json={
            "model_id": "model_torch",
            "horizon": 1,
            "frequency": "1d",
            "quantiles": [0.1, 0.5, 0.9],
            "data": [
                {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
                {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["forecasts"][0]["point"] == 12.5
    assert body["forecasts"][0]["quantiles"]["0.5"] == 12.5
    assert body["calibration"]["method"] == "split_conformal_abs_error"


@pytest.mark.experimental
def test_forecast_with_trained_torch_model_id_without_calibration_uses_intervals(monkeypatch):
    from forecasting_api import torch_forecasters as torch_module

    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    app_module._set_runtime_trained_models({
        "model_torch": {
            "model_id": "model_torch",
            "algo": "afnocg3_v1",
            "context_len": 3,
            "input_dim": 1,
            "pooled_residuals": [0.2, 0.3],
            "artifact": {
                "snapshot_json": "model_torch/snapshot.json",
                "weights_pt": "model_torch/weights.pt",
            },
        }
    })
    monkeypatch.setattr(app_module, "_read_json", lambda path: {"context_len": 3})
    monkeypatch.setattr(app_module, "_try_torch_load_weights", lambda path: {"state_dict": {}})
    monkeypatch.setattr(app_module, "_extract_state_dict", lambda ckpt: {})
    monkeypatch.setattr(torch_module, "forecast_univariate_torch", lambda **kwargs: [13.0, 14.0])

    client = TestClient(create_app())
    response = client.post(
        "/v1/forecast",
        headers={"x-api-key": "test-key"},
        json={
            "model_id": "model_torch",
            "horizon": 2,
            "frequency": "1d",
            "level": [90],
            "data": [
                {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
                {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["calibration"] is None
    assert body["warnings"][0].startswith("[EN] CALIB01")
    assert body["forecasts"][0]["intervals"][0]["level"] == 90.0
    assert body["forecasts"][1]["point"] == 14.0


@pytest.mark.experimental
def test_forecast_with_trained_hybrid_model_id_returns_uncertainty_and_explanations(monkeypatch):
    from forecasting_api import torch_forecasters as torch_module

    def _fake_read_json(path):
        path_str = str(path)
        if path_str.endswith("hybrid.json"):
            return {
                "feature_keys": ["sensor_1", "sensor_2"],
                "gate": {"interval_scale": 1.1},
                "model_explainability": {
                    "gbdt": {"top_features": [{"feature": "sensor_1", "score": 0.8}]},
                    "afno": {"method": "occlusion"},
                },
            }
        return {"context_len": 3}

    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    app_module._set_runtime_trained_models({
        "model_hybrid": {
            "model_id": "model_hybrid",
            "algo": "gbdt_afno_hybrid_v1",
            "context_len": 3,
            "pooled_residuals": [0.2 + (0.01 * idx) for idx in range(20)],
            "artifact": {
                "snapshot_json": "model_hybrid/snapshot.json",
                "weights_pt": "model_hybrid/weights.pt",
                "gbdt_joblib": "model_hybrid/gbdt.joblib",
                "hybrid_json": "model_hybrid/hybrid.json",
            },
        }
    })
    monkeypatch.setattr(app_module, "_read_json", _fake_read_json)
    monkeypatch.setattr(app_module, "_try_torch_load_weights", lambda path: {"state_dict": {}})
    monkeypatch.setattr(app_module, "_extract_state_dict", lambda ckpt: {})
    monkeypatch.setattr(app_module, "_load_joblib_artifact", lambda path: {"bundle": "ok"})
    monkeypatch.setattr(
        stable_models,
        "predict_hgb_next",
        lambda bundle, *, context_records, feature_keys: (25.0, 23.0, 27.0),
    )
    monkeypatch.setattr(
        training_helpers,
        "hybrid_gate_step_payload",
        lambda **kwargs: {
            "afno_weight": 0.3,
            "gbdt_weight": 0.7,
            "score": 0.4,
            "term_delta": 0.1,
            "term_overlap": 0.2,
            "term_width": -0.1,
            "term_tail": 0.0,
            "term_condition": 0.5,
        },
    )
    monkeypatch.setattr(
        torch_module,
        "forecast_univariate_torch_with_details",
        lambda **kwargs: {
            "point": [30.0, 31.0],
            "mc_dropout": {"per_step_var": [0.25, 0.36]},
            "occlusion": {
                "per_step": [
                    {"top_features": [{"feature": "sensor_1", "score": 0.8}]},
                    {"top_features": [{"feature": "sensor_2", "score": 0.6}]},
                ]
            },
        },
    )

    client = TestClient(create_app())
    response = client.post(
        "/v1/forecast",
        headers={"x-api-key": "test-key"},
        json={
            "model_id": "model_hybrid",
            "horizon": 2,
            "frequency": "1d",
            "level": [90],
            "data": [
                {
                    "series_id": "s1",
                    "timestamp": "2026-03-20T00:00:00Z",
                    "y": 10.0,
                    "x": {"sensor_1": 1.0, "sensor_2": 2.0},
                },
                {
                    "series_id": "s1",
                    "timestamp": "2026-03-21T00:00:00Z",
                    "y": 11.0,
                    "x": {"sensor_1": 2.0, "sensor_2": 3.0},
                },
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["calibration"]["method"] == "hybrid_validation_residual_evidence"
    assert body["uncertainty_summary"]["method"] == "hybrid_3way_v1"
    assert body["model_explainability"]["gbdt"]["top_features"][0]["feature"] == "sensor_1"
    assert body["forecasts"][0]["intervals"][0]["level"] == 90.0
    assert body["forecasts"][0]["uncertainty"]["components"]["total_var"] > 0.0
    assert body["forecasts"][0]["explanation"]["gate"]["afno_weight"] == 0.3


def test_backtest_with_trained_model_id(monkeypatch, patched_app_runtime, patched_backtest_runtime):
    """Non-naive path: train a model, run backtest using its model_id."""
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")

    client = TestClient(create_app())
    headers = {"x-api-key": "test-key"}

    model_id = _train_naive_get_model_id(client, headers)

    # Backtest using the trained model_id
    payload = _backtest_payload()
    payload["model_id"] = model_id
    response = client.post("/v1/backtest", headers=headers, json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "rmse" in body["metrics"]


def test_forecast_rate_limit_returns_429(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_RATE_LIMIT_PER_WINDOW", "1")
    monkeypatch.setenv("RULFM_FORECASTING_API_RATE_LIMIT_WINDOW_SECONDS", "60")
    client = TestClient(create_app())
    payload = _forecast_payload()

    first = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)
    assert first.status_code == 200

    second = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)
    assert second.status_code == 429
    body = second.json()
    assert body["error_code"] == "R01"
    assert "Retry-After" in second.headers


def test_forecast_endpoint_rejects_large_request_body(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_MAX_BODY_BYTES", "1024")
    client = TestClient(create_app())
    response = client.post(
        "/v1/forecast",
        headers={"x-api-key": "test-key"},
        json={
            "horizon": 1,
            "frequency": "1d",
            "data": [
                {
                    "series_id": "series-with-a-very-long-identifier-to-trigger-body-limit",
                    "timestamp": f"2026-03-{day:02d}T00:00:00Z",
                    "y": float(day),
                    "x": {"sensor_1": float(day), "sensor_2": float(day + 1)},
                }
                for day in range(1, 25)
            ],
        },
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "S01"


def test_forecast_endpoint_rejects_sync_payload_over_cost_limit(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_SYNC_MAX_POINTS", "100")
    client = TestClient(create_app())
    payload = {
        "horizon": 1,
        "frequency": "1d",
        "data": [
            {
                "series_id": "s1",
                "timestamp": (
                    datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=offset)
                ).isoformat().replace("+00:00", "Z"),
                "y": float(offset + 1),
            }
            for offset in range(101)
        ],
    }

    response = client.post("/v1/forecast", headers={"x-api-key": "test-key"}, json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "COST01"
    assert body["details"]["max_points"] == 100


def test_backtest_endpoint_uses_gbdt_branch(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    app_module._set_runtime_trained_models(
        {"model_gbdt": {"model_id": "model_gbdt", "algo": "gbdt_hgb_v1"}}
    )
    monkeypatch.setattr(
        stable_models,
        "gbdt_backtest",
        lambda req, trained, **kwargs: app_module.BacktestResponse(metrics={"rmse": 0.5}),
    )
    client = TestClient(create_app())

    response = client.post(
        "/v1/backtest",
        headers={"x-api-key": "test-key"},
        json={**_backtest_payload(), "model_id": "model_gbdt"},
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["rmse"] == 0.5


@pytest.mark.experimental
def test_backtest_endpoint_uses_hybrid_branch(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    app_module._set_runtime_trained_models(
        {"model_hybrid": {"model_id": "model_hybrid", "algo": "gbdt_afno_hybrid_v1"}}
    )
    monkeypatch.setattr(
        hybrid_runtime,
        "hybrid_backtest",
        lambda req, trained, **kwargs: app_module.BacktestResponse(metrics={"rmse": 0.75}),
    )
    client = TestClient(create_app())

    response = client.post(
        "/v1/backtest",
        headers={"x-api-key": "test-key"},
        json={**_backtest_payload(), "model_id": "model_hybrid"},
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["rmse"] == 0.75


@pytest.mark.experimental
def test_backtest_endpoint_uses_torch_branch(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    app_module._set_runtime_trained_models(
        {"model_torch": {"model_id": "model_torch", "algo": "afnocg3_v1"}}
    )
    monkeypatch.setattr(
        app_module,
        "_torch_backtest",
        lambda req, trained: app_module.BacktestResponse(metrics={"rmse": 0.9}),
    )
    client = TestClient(create_app())

    response = client.post(
        "/v1/backtest",
        headers={"x-api-key": "test-key"},
        json={**_backtest_payload(), "model_id": "model_torch"},
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["rmse"] == 0.9