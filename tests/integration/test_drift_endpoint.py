from fastapi.testclient import TestClient

from forecasting_api.app import create_app


def test_drift_baseline_endpoint_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_DRIFT_BASELINE_PATH", str(tmp_path / "baseline.json"))
    client = TestClient(create_app())
    response = client.post(
        "/v1/monitoring/drift/baseline",
        headers={"x-api-key": "test-key"},
        json={
            "baseline_records": [
                {"x": {"sensor_1": 1.0}},
                {"x": {"sensor_1": 1.2}},
                {"x": {"sensor_1": 1.1}},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["persisted"] is True
    assert body["feature_summaries"][0]["selected_bin_count"] >= 1


def test_drift_baseline_status_endpoint_reports_persisted_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_DRIFT_BASELINE_PATH", str(tmp_path / "baseline-status.json"))
    client = TestClient(create_app())
    client.post(
        "/v1/monitoring/drift/baseline",
        headers={"x-api-key": "test-key"},
        json={
            "baseline_records": [
                {"x": {"sensor_1": 1.0}},
                {"x": {"sensor_1": 1.1}},
                {"x": {"sensor_1": 1.2}},
            ]
        },
    )
    response = client.get(
        "/v1/monitoring/drift/baseline/status",
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["baseline_exists"] is True
    assert body["feature_count"] >= 1
    assert body["sample_size"] >= 3
    assert body["feature_summaries"][0]["selected_bin_count"] >= 1


def test_drift_baseline_status_endpoint_reports_missing_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_DRIFT_BASELINE_PATH", str(tmp_path / "missing-status.json"))
    client = TestClient(create_app())
    response = client.get(
        "/v1/monitoring/drift/baseline/status",
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["baseline_exists"] is False
    assert body["feature_count"] == 0
    assert body["sample_size"] == 0
    assert body["feature_summaries"] == []


def test_drift_report_endpoint_includes_baseline_bin_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv(
        "RULFM_DRIFT_BASELINE_PATH", str(tmp_path / "drift-report-metadata.json")
    )
    client = TestClient(create_app())
    response = client.post(
        "/v1/monitoring/drift/report",
        headers={"x-api-key": "test-key"},
        json={
            "baseline_records": [
                {"x": {"sensor_1": 1.0}},
                {"x": {"sensor_1": 1.1}},
                {"x": {"sensor_1": 1.2}},
                {"x": {"sensor_1": 1.3}},
            ],
            "candidate_records": [
                {"x": {"sensor_1": 4.0}},
                {"x": {"sensor_1": 4.1}},
            ],
        },
    )
    assert response.status_code == 200
    feature_report = response.json()["feature_reports"][0]
    assert feature_report["baseline_binning_strategy"] == "adaptive_equal_width_without_empty_bins"
    assert (
        feature_report["baseline_requested_bin_count"]
        >= feature_report["baseline_selected_bin_count"]
    )
    assert feature_report["baseline_selected_bin_count"] >= 1


def test_openapi_drift_examples_include_selected_bin_count(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    schema = create_app().openapi()["components"]["schemas"]
    baseline_example = schema["DriftBaselineResponse"]["examples"][0]
    report_example = schema["DriftReportResponse"]["examples"][0]
    assert baseline_example["feature_summaries"][0]["selected_bin_count"] >= 1
    assert report_example["feature_reports"][0]["baseline_selected_bin_count"] >= 1


def test_drift_report_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_DRIFT_BASELINE_PATH", str(tmp_path / "drift-report.json"))
    client = TestClient(create_app())
    response = client.post(
        "/v1/monitoring/drift/report",
        headers={"x-api-key": "test-key"},
        json={
            "baseline_records": [{"x": {"sensor_1": 1.0}}, {"x": {"sensor_1": 1.2}}],
            "candidate_records": [{"x": {"sensor_1": 4.0}}, {"x": {"sensor_1": 4.1}}],
        },
    )
    assert response.status_code == 200
    assert response.json()["severity"] in {"medium", "high"}


def test_drift_baseline_endpoint_rejects_non_numeric_records(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_DRIFT_BASELINE_PATH", str(tmp_path / "baseline.json"))
    client = TestClient(create_app())
    response = client.post(
        "/v1/monitoring/drift/baseline",
        headers={"x-api-key": "test-key"},
        json={"baseline_records": [{"x": {"sensor_1": "bad"}}]},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "V01"


def test_drift_report_endpoint_requires_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_DRIFT_BASELINE_PATH", str(tmp_path / "missing-baseline.json"))
    client = TestClient(create_app())
    response = client.post(
        "/v1/monitoring/drift/report",
        headers={"x-api-key": "test-key"},
        json={"candidate_records": [{"x": {"sensor_1": 1.0}}]},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "V01"


def test_drift_report_uses_persisted_baseline_when_no_baseline_records_in_body(
    monkeypatch, tmp_path
):
    """Drift report with no baseline_records field uses the persisted baseline only."""
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    baseline_path = str(tmp_path / "persisted-baseline.json")
    monkeypatch.setenv("RULFM_DRIFT_BASELINE_PATH", baseline_path)
    client = TestClient(create_app())

    # Persist baseline first
    persist_resp = client.post(
        "/v1/monitoring/drift/baseline",
        headers={"x-api-key": "test-key"},
        json={
            "baseline_records": [
                {"x": {"sensor_1": v}} for v in [1.0, 1.1, 1.2, 1.3, 1.4]
            ]
        },
    )
    assert persist_resp.status_code == 200
    assert persist_resp.json()["persisted"] is True

    # Report using only candidate_records — no baseline_records in body
    report_resp = client.post(
        "/v1/monitoring/drift/report",
        headers={"x-api-key": "test-key"},
        json={
            "candidate_records": [
                {"x": {"sensor_1": v}} for v in [5.0, 6.0, 7.0]
            ]
        },
    )
    assert report_resp.status_code == 200
    body = report_resp.json()
    assert body["severity"] in {"low", "medium", "high"}
    assert len(body["feature_reports"]) >= 1

