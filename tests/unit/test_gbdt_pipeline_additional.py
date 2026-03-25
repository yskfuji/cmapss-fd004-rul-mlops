import math

import numpy as np
from models import gbdt_pipeline


def _records(count: int = 40) -> list[dict[str, object]]:
    rows = []
    for idx in range(count):
        rows.append(
            {
                "series_id": "engine-1",
                "y": float(max(0, 125 - idx)),
                "x": {
                    "op_setting_1": 1.0,
                    "op_setting_2": 0.25,
                    "op_setting_3": 3.0,
                    "sensor_1": float(idx),
                    "sensor_2": float(idx % 5),
                    "sensor_3": 10.0,
                },
            }
        )
    return rows


def _multi_series_records(series_count: int = 12, count: int = 40) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for series_idx in range(series_count):
        series_id = f"engine-{series_idx + 1:03d}"
        for idx in range(count):
            rows.append(
                {
                    "series_id": series_id,
                    "timestamp": (
                        f"2026-01-{(idx // 24) + 1:02d}T{idx % 24:02d}:00:{series_idx:02d}Z"
                    ),
                    "y": float(max(0, 125 - idx)),
                    "x": {
                        "op_setting_1": float((series_idx % 3) + 1),
                        "op_setting_2": 0.25,
                        "op_setting_3": 3.0,
                        "sensor_1": float(idx + series_idx),
                        "sensor_2": float((idx + series_idx) % 7),
                    },
                }
            )
    return rows


class _DummyModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray([self.value] * len(rows), dtype=float)


def test_op_cluster_feature_selection_norm_stats_and_dataset_building() -> None:
    rows = _records()
    assert gbdt_pipeline.op_cluster_key(rows[0]) == (1, 25, 3)

    feature_keys = gbdt_pipeline.select_feature_keys(rows)
    assert "sensor_1" in feature_keys
    assert "sensor_2" in feature_keys
    assert "sensor_3" not in feature_keys

    norm_stats = gbdt_pipeline.compute_norm_stats(rows, feature_keys=feature_keys)
    assert norm_stats[(1, 25, 3)]["sensor_1"][1] > 0.0

    x_all, y_all = gbdt_pipeline.build_gbdt_dataset(
        rows,
        feature_keys=feature_keys,
        norm_stats=norm_stats,
    )
    assert x_all.shape[0] == len(rows) - gbdt_pipeline.WINDOW
    assert y_all.shape[0] == len(rows) - gbdt_pipeline.WINDOW


def test_build_dataset_returns_empty_arrays_for_short_series() -> None:
    rows = _records(10)
    x_all, y_all = gbdt_pipeline.build_gbdt_dataset(rows, feature_keys=["sensor_1"])
    assert x_all.shape == (0, 0)
    assert y_all.shape == (0,)


def test_predict_rul_supports_fallback_and_ensemble_paths() -> None:
    rows = _records()
    feature_keys = ["sensor_1", "sensor_2"]
    norm_stats = gbdt_pipeline.compute_norm_stats(rows, feature_keys=feature_keys)

    fallback_bundle = {
        "feature_keys": feature_keys,
        "norm_stats": norm_stats,
        "interval_scale": 1.5,
        "point": _DummyModel(20.0),
        "q05": _DummyModel(18.0),
        "q95": _DummyModel(24.0),
        "lgbm": None,
        "catboost": None,
    }
    point, lower, upper = gbdt_pipeline.predict_rul(fallback_bundle, rows)
    assert point == 20.0
    assert lower == 17.0
    assert upper == 26.0

    ensemble_bundle = {
        **fallback_bundle,
        "lgbm": _DummyModel(30.0),
        "catboost": _DummyModel(10.0),
        "lgbm_weight": 0.75,
    }
    assert "catboost_weight" not in ensemble_bundle
    point, _, _ = gbdt_pipeline.predict_rul(ensemble_bundle, rows)
    assert point == 25.0


def test_predict_rul_clamps_outputs_to_max_rul() -> None:
    rows = _records()
    feature_keys = ["sensor_1", "sensor_2"]
    norm_stats = gbdt_pipeline.compute_norm_stats(rows, feature_keys=feature_keys)

    bundle = {
        "feature_keys": feature_keys,
        "norm_stats": norm_stats,
        "interval_scale": 2.0,
        "point": _DummyModel(200.0),
        "q05": _DummyModel(180.0),
        "q95": _DummyModel(260.0),
        "lgbm": None,
        "catboost": None,
    }
    point, lower, upper = gbdt_pipeline.predict_rul(bundle, rows)
    assert point == gbdt_pipeline.MAX_RUL
    assert lower == gbdt_pipeline.MAX_RUL
    assert upper == gbdt_pipeline.MAX_RUL


def test_group_records_supports_unit_id_and_sorts_timestamps() -> None:
    rows = [
        {"unit_id": "engine-7", "timestamp": "2026-01-02T00:00:00Z", "x": {}, "y": 1.0},
        {"unit_id": "engine-7", "timestamp": "2026-01-01T00:00:00Z", "x": {}, "y": 2.0},
    ]

    grouped = gbdt_pipeline.group_records(rows)

    assert list(grouped) == ["engine-7"]
    assert [row["timestamp"] for row in grouped["engine-7"]] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
    ]


def test_group_records_strips_series_ids() -> None:
    rows = [
        {"series_id": "  engine-8  ", "x": {}, "y": 1.0},
        {"series_id": "engine-8", "x": {}, "y": 2.0},
    ]

    grouped = gbdt_pipeline.group_records(rows)

    assert list(grouped) == ["engine-8"]


def test_record_series_id_uses_unit_id_fallback_and_strips() -> None:
    assert gbdt_pipeline.record_series_id({"series_id": "  engine-9  "}) == "engine-9"
    assert gbdt_pipeline.record_series_id({"unit_id": "  unit-3  "}) == "unit-3"
    assert gbdt_pipeline.record_series_id({}) == "unknown"


def test_group_records_preserves_insertion_order_without_timestamps() -> None:
    rows = [
        {"series_id": "engine-8", "x": {}, "y": 1.0, "marker": "first"},
        {"series_id": "engine-8", "x": {}, "y": 2.0, "marker": "second"},
    ]

    grouped = gbdt_pipeline.group_records(rows)

    # Records with no timestamp sort to the same empty-string key, so Python's
    # stable sort preserves the original insertion order within the series.
    assert [row["marker"] for row in grouped["engine-8"]] == ["first", "second"]


def test_as_float_accepts_numeric_strings() -> None:
    assert gbdt_pipeline.as_float("42.0") == 42.0


def test_as_dict_stringifies_keys() -> None:
    assert gbdt_pipeline.as_dict({1: "value", "sensor_2": 2.0}) == {
        "1": "value",
        "sensor_2": 2.0,
    }


def test_calibration_split_is_seeded_and_not_fixed_to_sorted_tail() -> None:
    rows = _multi_series_records(series_count=20)

    _, calib_records_a = gbdt_pipeline.calibration_split(
        rows,
        holdout_fraction=0.1,
        min_holdout_series=8,
        max_holdout_series=24,
        random_state=0,
    )
    _, calib_records_b = gbdt_pipeline.calibration_split(
        rows,
        holdout_fraction=0.1,
        min_holdout_series=8,
        max_holdout_series=24,
        random_state=0,
    )

    calib_ids_a = sorted(gbdt_pipeline.group_records(calib_records_a))
    calib_ids_b = sorted(gbdt_pipeline.group_records(calib_records_b))
    expected_count = min(24, max(8, int(math.ceil(20 * 0.1))))
    assert calib_ids_a == calib_ids_b
    assert len(calib_ids_a) == expected_count
    assert calib_ids_a != sorted(gbdt_pipeline.group_records(rows))[-8:]


def test_build_gbdt_calibration_proxy_dataset_returns_one_row_per_series() -> None:
    rows = _multi_series_records(series_count=6, count=140)
    feature_keys = ["sensor_1", "sensor_2"]
    norm_stats = gbdt_pipeline.compute_norm_stats(rows, feature_keys=feature_keys)
    target_ruls = [12.0, 24.0, 52.0, 76.0, 90.0, 95.0]

    x_proxy, y_proxy = gbdt_pipeline.build_gbdt_calibration_proxy_dataset(
        rows,
        feature_keys=feature_keys,
        norm_stats=norm_stats,
        target_ruls=target_ruls,
    )

    assert x_proxy.shape[0] == 6
    assert y_proxy.shape == (6,)
    assert np.allclose(y_proxy, np.asarray(target_ruls, dtype=float))


def test_fit_gbdt_pipeline_reports_proxy_interval_calibration_metadata() -> None:
    rows = _multi_series_records(series_count=12, count=140)
    target_ruls = [float(value) for value in (12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120, 125)]

    bundle = gbdt_pipeline.fit_gbdt_pipeline(
        rows,
        feature_keys=["sensor_1", "sensor_2"],
        preset="fast",
        enable_ensemble=False,
        interval_calibration_targets=target_ruls,
        interval_calibration_holdout_fraction=0.1,
        interval_calibration_min_holdout_series=8,
        interval_calibration_max_holdout_series=24,
    )

    fit_meta = bundle["fit_meta"]
    assert fit_meta["interval_calibration_protocol"] == "heldout_unit_proxy_final_cycle"
    assert fit_meta["interval_calibration_units"] == 8
    assert fit_meta["interval_calibration_rows"] == 8
    assert fit_meta["interval_calibration_target_mean"] is not None
    assert fit_meta["interval_calibration_target_p50"] is not None
    assert fit_meta["interval_calibration_requested_holdout_fraction"] == 0.1
    assert fit_meta["interval_calibration_holdout_fraction"] == 8 / 12