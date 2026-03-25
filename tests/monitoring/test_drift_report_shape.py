from monitoring.drift_detector import DriftDetector, drift_report_to_dict


def test_drift_report_to_dict_contains_expected_keys():
    detector = DriftDetector()
    baseline = detector.summarize_baseline([{"x": {"sensor_1": 1.0}}])
    report = detector.detect(baseline, [{"x": {"sensor_1": 1.1}}])
    payload = drift_report_to_dict(report)
    assert set(payload) == {"severity", "drift_score", "sample_size", "feature_reports"}
