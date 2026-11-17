import sqlite3
from contextlib import nullcontext

from fastapi.testclient import TestClient

from forecasting_api import app as app_module
from forecasting_api.app import create_app
from forecasting_api.job_worker import process_next_job_once


def _use_isolated_job_db(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_module, "_MODEL_REGISTRY_DB_PATH", tmp_path / "jobs.db")
    app_module._set_runtime_job_store(None)


def test_job_status_returns_404_for_unknown_job(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    _use_isolated_job_db(monkeypatch, tmp_path)
    client = TestClient(create_app())
    response = client.get("/v1/jobs/missing-job", headers={"x-api-key": "test-key"})
    assert response.status_code == 404
    assert response.json()["error_code"] == "J01"


def test_job_result_returns_409_for_incomplete_job(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    _use_isolated_job_db(monkeypatch, tmp_path)
    client = TestClient(create_app())
    create_response = client.post(
        "/v1/jobs",
        headers={"x-api-key": "test-key"},
        json={
            "type": "forecast",
            "payload": {
                "horizon": 1,
                "data": [
                    {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
                    {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
                ],
            },
        },
    )
    job_id = create_response.json()["job_id"]
    response = client.get(f"/v1/jobs/{job_id}/result", headers={"x-api-key": "test-key"})
    assert response.status_code == 409
    body = response.json()
    assert body["error_code"] == "J02"
    assert body["details"]["status"] == "queued"


def test_job_status_returns_error_payload_for_failed_job(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    _use_isolated_job_db(monkeypatch, tmp_path)
    client = TestClient(create_app())
    create_response = client.post(
        "/v1/jobs",
        headers={"x-api-key": "test-key"},
        json={
            "type": "forecast",
            "payload": {"horizon": "bad", "data": []},
        },
    )
    job_id = create_response.json()["job_id"]
    assert process_next_job_once() is True
    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]["error_code"] == "V01"


def test_job_result_returns_payload_for_succeeded_forecast_job(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    _use_isolated_job_db(monkeypatch, tmp_path)
    client = TestClient(create_app())
    create_response = client.post(
        "/v1/jobs",
        headers={"x-api-key": "test-key"},
        json={
            "type": "forecast",
            "payload": {
                "horizon": 1,
                "frequency": "1d",
                "data": [
                    {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
                    {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
                ],
            },
        },
    )
    job_id = create_response.json()["job_id"]
    assert process_next_job_once() is True
    status_response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "succeeded"

    result_response = client.get(f"/v1/jobs/{job_id}/result", headers={"x-api-key": "test-key"})
    assert result_response.status_code == 200
    body = result_response.json()
    assert len(body["forecasts"]) == 1
    assert body["forecasts"][0]["series_id"] == "s1"


def test_job_result_returns_payload_for_succeeded_backtest_job(monkeypatch, tmp_path):
    """Async backtest job: create → poll status → get result shape."""
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    _use_isolated_job_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "start_run", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(app_module, "log_params", lambda payload: None)
    monkeypatch.setattr(app_module, "log_metrics", lambda payload: None)
    monkeypatch.setattr(app_module, "log_dict_artifact", lambda name, payload: None)
    client = TestClient(create_app())
    headers = {"x-api-key": "test-key"}

    data = [
        {"series_id": "s1", "timestamp": f"2026-03-{d:02d}T00:00:00Z", "y": float(d)}
        for d in range(1, 12)
    ]
    create_response = client.post(
        "/v1/jobs",
        headers=headers,
        json={
            "type": "backtest",
            "payload": {
                "horizon": 2,
                "folds": 2,
                "metric": "rmse",
                "data": data,
            },
        },
    )
    assert create_response.status_code in {200, 202}
    job_id = create_response.json()["job_id"]
    assert process_next_job_once() is True

    status_response = client.get(f"/v1/jobs/{job_id}", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "succeeded"

    result_response = client.get(f"/v1/jobs/{job_id}/result", headers=headers)
    assert result_response.status_code == 200
    body = result_response.json()
    assert "rmse" in body["metrics"]
    assert len(body["by_series"]) >= 1
    assert len(body["by_horizon"]) >= 1


def test_job_status_persists_across_app_instances(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    _use_isolated_job_db(monkeypatch, tmp_path)

    first_client = TestClient(create_app())
    create_response = first_client.post(
        "/v1/jobs",
        headers={"x-api-key": "test-key"},
        json={
            "type": "forecast",
            "payload": {
                "horizon": 1,
                "frequency": "1d",
                "data": [
                    {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
                    {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
                ],
            },
        },
    )
    assert create_response.status_code == 202
    job_id = create_response.json()["job_id"]

    second_client = TestClient(create_app())
    status_response = second_client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})

    assert status_response.status_code == 200
    body = status_response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"


def test_worker_recovers_stale_running_job_before_processing(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    _use_isolated_job_db(monkeypatch, tmp_path)

    client = TestClient(create_app())
    create_response = client.post(
        "/v1/jobs",
        headers={"x-api-key": "test-key"},
        json={
            "type": "forecast",
            "payload": {
                "horizon": 1,
                "frequency": "1d",
                "data": [
                    {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
                    {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
                ],
            },
        },
    )
    job_id = create_response.json()["job_id"]
    job_store = app_module._require_job_store()
    claimed = job_store.claim_next_queued()
    assert claimed is not None
    assert claimed.job_id == job_id

    with sqlite3.connect(str(tmp_path / "jobs.db")) as conn:
        conn.execute(
            "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
            ("2000-01-01 00:00:00", job_id),
        )
        conn.commit()

    assert process_next_job_once(stale_after_seconds=60) is True
    response = client.get(f"/v1/jobs/{job_id}", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
