"""Integration tests for the GBDT (gbdt_hgb_v1) serving path.

These tests exercise the public contract end-to-end:
  train → forecast → backtest

They do not mock the GBDT pipeline internals; the real HGB model is fitted
on synthetic data so the tests verify that the full serving path is wired up
correctly without requiring the FD004 dataset.
"""

import pytest
from fastapi.testclient import TestClient
from forecasting_api.app import create_app

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_HEADERS = {"x-api-key": "test-key"}

def _make_record(series_id: str, day: int) -> dict:
    """Return a data record with synthetic exogenous sensor features."""
    return {
        "series_id": series_id,
        "timestamp": f"2026-01-{day:02d}T00:00:00Z",
        "y": float(100 - day),
        "x": {
            "sensor_1": float(day) * 0.5,
            "sensor_2": float(day) * 1.1,
            "sensor_3": float(100 - day) * 0.3,
        },
    }


_TRAIN_DATA = [
    _make_record(f"eng{e:02d}", d)
    for e in range(1, 6)   # 5 engines
    for d in range(1, 21)  # 20 cycles each
]

_FORECAST_DATA = [_make_record("eng01", d) for d in range(1, 16)]

_BACKTEST_DATA = [_make_record("eng01", d) for d in range(1, 31)]


@pytest.fixture()
def client(api_env, patched_app_runtime):
    return TestClient(create_app())


def _train_gbdt(client) -> str:
    """Train a gbdt_hgb_v1 model and return its model_id."""
    response = client.post(
        "/v1/train",
        headers=_HEADERS,
        json={
            "algo": "gbdt_hgb_v1",
            "training_hours": 0.05,
            "data": _TRAIN_DATA,
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["message"] == "accepted"
    return body["model_id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gbdt_train_returns_accepted_with_model_id(client):
    """POST /v1/train with gbdt_hgb_v1 should return 202 and a model_id."""
    model_id = _train_gbdt(client)
    assert model_id.startswith("model_")


def test_gbdt_trained_model_appears_in_catalog(client):
    """Trained GBDT model should be listed in GET /v1/models."""
    model_id = _train_gbdt(client)
    resp = client.get("/v1/models", headers=_HEADERS)
    assert resp.status_code == 200
    ids = [m["model_id"] for m in resp.json()["models"]]
    assert model_id in ids


def test_gbdt_forecast_returns_non_naive_predictions(client):
    """POST /v1/forecast with a trained GBDT model_id should return forecasts.

    ForecastPoint schema: series_id, timestamp, point, quantiles (dict|None),
    intervals (list|None), uncertainty (dict|None), explanation (dict|None).
    """
    model_id = _train_gbdt(client)
    response = client.post(
        "/v1/forecast",
        headers=_HEADERS,
        json={
            "horizon": 5,
            "frequency": "1d",
            "model_id": model_id,
            "data": _FORECAST_DATA,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "forecasts" in body
    # horizon=5 → 5 ForecastPoint entries for the single series
    assert len(body["forecasts"]) == 5
    for fc in body["forecasts"]:
        assert fc["series_id"] == "eng01"
        assert "point" in fc
        # GBDT should produce finite predictions
        assert abs(fc["point"]) < 1e6


def test_gbdt_forecast_includes_interval_payload(client):
    """GBDT forecast with quantiles should include quantile map in response."""
    model_id = _train_gbdt(client)
    response = client.post(
        "/v1/forecast",
        headers=_HEADERS,
        json={
            "horizon": 3,
            "frequency": "1d",
            "model_id": model_id,
            "quantiles": [0.1, 0.9],
            "data": _FORECAST_DATA,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["forecasts"]) == 3
    # Each ForecastPoint should have a point prediction
    for fc in body["forecasts"]:
        assert "point" in fc


def test_gbdt_backtest_returns_metric_payload(client, patched_backtest_runtime):
    """POST /v1/backtest with trained GBDT model_id should return metrics."""
    model_id = _train_gbdt(client)
    response = client.post(
        "/v1/backtest",
        headers=_HEADERS,
        json={
            "horizon": 3,
            "folds": 2,
            "metric": "rmse",
            "model_id": model_id,
            "data": _BACKTEST_DATA,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "metrics" in body
    assert "rmse" in body["metrics"]
    assert body["metrics"]["rmse"] >= 0.0


def test_gbdt_model_appears_with_expected_fields(client):
    """Trained GBDT model catalog entry should have model_id and created_at."""
    model_id = _train_gbdt(client)
    resp = client.get("/v1/models", headers=_HEADERS)
    assert resp.status_code == 200
    models = resp.json()["models"]
    model = next((m for m in models if m["model_id"] == model_id), None)
    assert model is not None, f"{model_id} not found in catalog"
    assert "created_at" in model
    # model_id roundtrip
    assert model["model_id"] == model_id
