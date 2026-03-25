import json
import statistics
from pathlib import Path

from monitoring.drift_detector import (
    DriftDetector,
    default_baseline_path,
    has_sufficient_baseline_samples,
    has_valid_baseline,
    load_baseline,
    save_baseline,
)


def test_baseline_can_be_saved_and_loaded(tmp_path):
    detector = DriftDetector()
    baseline = detector.summarize_baseline([{"x": {"sensor_1": 1.0}}, {"x": {"sensor_1": 2.0}}])
    path = tmp_path / "baseline.json"
    save_baseline(baseline, path)
    loaded = load_baseline(path)
    assert loaded == baseline
    assert has_valid_baseline(loaded) is True


def test_repository_baseline_fixture_is_valid(monkeypatch):
    monkeypatch.delenv("RULFM_DRIFT_BASELINE_PATH", raising=False)
    loaded = load_baseline(default_baseline_path())
    assert has_valid_baseline(loaded) is True
    assert has_sufficient_baseline_samples(loaded, minimum_count=50) is True
    assert len(loaded) >= 3
    assert all(min(feature["bin_probabilities"]) > 0.01 for feature in loaded.values())
    assert all(
        feature["selected_bin_count"] == len(feature["bin_probabilities"])
        for feature in loaded.values()
    )
    assert all(
        feature["requested_bin_count"] >= feature["selected_bin_count"]
        for feature in loaded.values()
    )
    assert len({feature["selected_bin_count"] for feature in loaded.values()}) >= 2
    assert any(
        (max(feature["bin_probabilities"]) - min(feature["bin_probabilities"])) > 0.01
        for feature in loaded.values()
    )


def test_repository_candidate_fixture_is_non_empty_and_shifted() -> None:
    candidate_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "forecasting_api"
        / "data"
        / "drift_candidate_records.json"
    )
    reference_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "forecasting_api"
        / "data"
        / "drift_reference_records.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    assert isinstance(candidate, list)
    assert len(candidate) >= 50
    assert len(candidate) == len(reference)
    assert all(isinstance(item.get("x"), dict) and item["x"] for item in candidate)
    candidate_sensor3 = [float(item["x"]["sensor_3"]) for item in candidate]
    reference_sensor3 = [float(item["x"]["sensor_3"]) for item in reference]
    assert statistics.mean(candidate_sensor3) > statistics.mean(reference_sensor3) + 1.0
