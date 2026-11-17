import math

import pytest

from monitoring.drift_detector import DriftDetector


def test_summarize_baseline_extracts_feature_stats():
    detector = DriftDetector()
    summary = detector.summarize_baseline([
        {"x": {"sensor_1": 1.0, "sensor_2": 10.0}},
        {"x": {"sensor_1": 2.0, "sensor_2": 12.0}},
    ])
    assert summary["sensor_1"]["mean"] == 1.5
    assert len(summary["sensor_2"]["bin_probabilities"]) >= 2


def test_summarize_baseline_preserves_non_uniform_distribution():
    detector = DriftDetector(bins=6)
    summary = detector.summarize_baseline(
        [
            {"x": {"sensor_1": 0.95}},
            {"x": {"sensor_1": 0.96}},
            {"x": {"sensor_1": 0.97}},
            {"x": {"sensor_1": 0.98}},
            {"x": {"sensor_1": 0.99}},
            {"x": {"sensor_1": 1.00}},
            {"x": {"sensor_1": 1.01}},
            {"x": {"sensor_1": 1.15}},
            {"x": {"sensor_1": 1.20}},
            {"x": {"sensor_1": 1.25}},
            {"x": {"sensor_1": 1.30}},
            {"x": {"sensor_1": 1.45}},
        ]
    )
    probabilities = summary["sensor_1"]["bin_probabilities"]
    assert max(probabilities) - min(probabilities) > 0.05
    assert min(probabilities) > 0.01


def test_detect_flags_large_shift_as_medium_or_high():
    detector = DriftDetector(psi_threshold=0.1, severe_threshold=0.2)
    baseline_records = [
        {"x": {"sensor_1": 1.0 + (index % 20) * 0.01}}
        for index in range(100)
    ]
    baseline = detector.summarize_baseline(baseline_records)
    report = detector.detect(
        baseline,
        [
            {"x": {"sensor_1": 2.0 + (index % 20) * 0.01}}
            for index in range(100)
        ],
    )
    assert report.severity in {"medium", "high"}
    assert report.drift_score > 0.0


def test_sparse_baseline_uses_minimum_bins():
    detector = DriftDetector()
    assert detector._effective_bin_count(8) == 2
    assert detector._effective_bin_count(10) == 2
    assert detector._effective_bin_count(11) == 2
    assert detector._effective_bin_count(49) == 9
    assert detector._effective_bin_count(50) == 10
    assert detector._effective_bin_count(51) == 10
    assert detector._effective_bin_count(100) == 10
    summary = detector.summarize_baseline([
        {"x": {"sensor_1": 1.00}},
        {"x": {"sensor_1": 1.02}},
        {"x": {"sensor_1": 1.04}},
        {"x": {"sensor_1": 1.06}},
        {"x": {"sensor_1": 1.08}},
        {"x": {"sensor_1": 1.10}},
        {"x": {"sensor_1": 1.12}},
        {"x": {"sensor_1": 1.14}},
    ])
    probabilities = summary["sensor_1"]["bin_probabilities"]
    assert len(probabilities) == 2
    assert min(probabilities) > 0.1


def test_histogram_edges_reduce_bins_until_no_empty_bucket():
    detector = DriftDetector(bins=10)
    values = detector.summarize_baseline(
        [
            {"x": {"sensor_1": 0.95}},
            {"x": {"sensor_1": 0.96}},
            {"x": {"sensor_1": 0.97}},
            {"x": {"sensor_1": 0.98}},
            {"x": {"sensor_1": 1.20}},
            {"x": {"sensor_1": 1.21}},
            {"x": {"sensor_1": 1.22}},
            {"x": {"sensor_1": 1.23}},
            {"x": {"sensor_1": 1.45}},
            {"x": {"sensor_1": 1.46}},
            {"x": {"sensor_1": 1.47}},
            {"x": {"sensor_1": 1.48}},
        ]
    )
    probabilities = values["sensor_1"]["bin_probabilities"]
    assert all(probability > 0.01 for probability in probabilities)


def test_constant_feature_collapses_to_single_bucket_when_needed():
    detector = DriftDetector(bins=10)
    summary = detector.summarize_baseline(
        [
            {"x": {"sensor_1": 1.0}},
            {"x": {"sensor_1": 1.0}},
            {"x": {"sensor_1": 1.0}},
            {"x": {"sensor_1": 1.0}},
        ]
    )
    assert summary["sensor_1"]["selected_bin_count"] == 1
    assert summary["sensor_1"]["bin_edges"] == []
    assert summary["sensor_1"]["bin_probabilities"] == pytest.approx([1.0])


def test_detect_low_shift_as_low():
    detector = DriftDetector()
    baseline_records = [
        {"x": {"sensor_1": 1.0 + (index % 5) * 0.01}}
        for index in range(100)
    ]
    baseline = detector.summarize_baseline(baseline_records)
    report = detector.detect(
        baseline,
        baseline_records,
    )
    assert report.severity == "low"


def test_detect_preserves_explicit_falsey_baseline_metadata_values():
    detector = DriftDetector()
    report = detector.detect(
        {
            "sensor_1": {
                "mean": 1.0,
                "std": 0.0,
                "count": 4,
                "binning_strategy": "",
                "requested_bin_count": 0,
                "selected_bin_count": 0,
                "bin_edges": [],
                "bin_probabilities": [1.0],
            }
        },
        [{"x": {"sensor_1": 1.0}}],
    )
    feature = report.feature_reports[0]
    assert feature.baseline_binning_strategy == ""
    assert feature.baseline_requested_bin_count == 0
    assert feature.baseline_selected_bin_count == 0


def test_population_stability_index_rejects_mismatched_bin_lengths() -> None:
    detector = DriftDetector()

    with pytest.raises(ValueError, match="identical lengths"):
        detector._population_stability_index(
            baseline_probabilities=[0.5, 0.5],
            candidate_probabilities=[1.0],
        )


def test_population_stability_index_matches_standard_definition() -> None:
    detector = DriftDetector()

    psi = detector._population_stability_index(
        baseline_probabilities=[0.2, 0.8],
        candidate_probabilities=[0.5, 0.5],
    )

    expected = ((0.5 - 0.2) * math.log(0.5 / 0.2)) + ((0.5 - 0.8) * math.log(0.5 / 0.8))
    assert psi == pytest.approx(expected)
