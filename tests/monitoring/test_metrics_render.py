from forecasting_api.metrics import record_drift, render_metrics


def test_record_drift_updates_metric_payload():
    record_drift({"sensor_1": 0.25}, "medium")
    payload, _ = render_metrics()
    text = payload.decode("utf-8")
    assert "rulfm_feature_drift_score" in text
    assert "sensor_1" in text
