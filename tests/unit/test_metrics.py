from forecasting_api.metrics import (
    metrics_enabled,
    record_drift,
    record_model_promotion,
    render_metrics,
    track_request,
)


def test_metrics_enabled_default_true(monkeypatch):
    monkeypatch.delenv("RULFM_METRICS_ENABLED", raising=False)
    assert metrics_enabled() is True


def test_track_request_emits_metrics():
    with track_request("GET", "/health") as status_holder:
        status_holder[0] = 200
    payload, content_type = render_metrics()
    assert b"rulfm_http_requests_total" in payload
    assert "text/plain" in content_type


def test_metrics_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RULFM_METRICS_ENABLED", "false")
    assert metrics_enabled() is False
    with track_request("GET", "/disabled") as status_holder:
        status_holder[0] = 204
    record_drift({"sensor_1": 0.5}, severity="low")
    record_model_promotion("prod")


def test_record_helpers_emit_metrics_when_enabled(monkeypatch):
    monkeypatch.setenv("RULFM_METRICS_ENABLED", "1")
    record_drift({"sensor_1": 0.5}, severity="medium")
    record_model_promotion("prod")
    payload, _ = render_metrics()
    assert b"rulfm_feature_drift_score" in payload
    assert b"rulfm_model_promotions_total" in payload
