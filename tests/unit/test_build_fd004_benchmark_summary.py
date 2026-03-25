from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from models import gbdt_pipeline


@pytest.fixture(scope="module")
def benchmark_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "build_fd004_benchmark_summary.py"
    )
    spec = importlib.util.spec_from_file_location("build_fd004_benchmark_summary", module_path)
    if spec is None:
        pytest.fail(f"failed to create module spec for {module_path}")
    if spec.loader is None:
        pytest.fail(f"failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Load once per test module to cut import cost; per-test monkeypatch still
    # restores shared_gbdt_pipeline mutations after each test.
    spec.loader.exec_module(module)
    return module


def _records(count: int = 40) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(count):
        rows.append(
            {
                "series_id": "engine-1",
                "timestamp": f"2026-01-{(idx % 28) + 1:02d}T00:00:{idx:02d}Z",
                "y": float(max(0, 125 - idx)),
                "x": {
                    "op_setting_1": 1.0,
                    "op_setting_2": 0.25,
                    "op_setting_3": 3.0,
                    "sensor_1": float(idx),
                    "sensor_2": float(idx % 5),
                },
            }
        )
    return rows


def test_standard_eval_outputs_uses_shared_predict_rul(
    monkeypatch, benchmark_module: ModuleType
) -> None:
    benchmark = benchmark_module
    calls: list[int] = []

    def _fake_predict_rul(
        bundle: dict[str, object], rows: list[dict[str, object]]
    ) -> tuple[float, float, float]:
        calls.append(len(rows))
        assert bundle["feature_keys"] == ["sensor_1", "sensor_2"]
        assert len(rows) == benchmark.WINDOW
        return (gbdt_pipeline.MAX_RUL, gbdt_pipeline.MAX_RUL, gbdt_pipeline.MAX_RUL)

    monkeypatch.setattr(benchmark.shared_gbdt_pipeline, "predict_rul", _fake_predict_rul)

    outputs = benchmark._standard_eval_outputs(
        {"feature_keys": ["sensor_1", "sensor_2"]},
        records=_records(),
        feature_keys=["sensor_1", "sensor_2"],
        kind="gbdt",
    )

    assert len(calls) == 1
    assert all(length == benchmark.WINDOW for length in calls)
    assert outputs["y_pred"] == [gbdt_pipeline.MAX_RUL]
    assert outputs["lower"] == [gbdt_pipeline.MAX_RUL]
    assert outputs["upper"] == [gbdt_pipeline.MAX_RUL]


def test_regime_one_step_outputs_uses_shared_predict_rul(
    monkeypatch, benchmark_module: ModuleType
) -> None:
    benchmark = benchmark_module
    calls: list[int] = []

    def _fake_predict_rul(
        bundle: dict[str, object], rows: list[dict[str, object]]
    ) -> tuple[float, float, float]:
        calls.append(len(rows))
        assert bundle["feature_keys"] == ["sensor_1", "sensor_2"]
        assert len(rows) >= benchmark.WINDOW
        return (gbdt_pipeline.MAX_RUL, gbdt_pipeline.MAX_RUL, gbdt_pipeline.MAX_RUL)

    monkeypatch.setattr(benchmark.shared_gbdt_pipeline, "predict_rul", _fake_predict_rul)

    outputs = benchmark._regime_one_step_outputs(
        {"feature_keys": ["sensor_1", "sensor_2"]},
        records=_records(),
        feature_keys=["sensor_1", "sensor_2"],
        regime=(1.0, 0.25, 3.0),
        kind="gbdt",
    )

    assert len(calls) == 10
    assert all(length >= benchmark.WINDOW for length in calls)
    assert outputs["y_pred"] == [gbdt_pipeline.MAX_RUL] * 10
    assert outputs["lower"] == [gbdt_pipeline.MAX_RUL] * 10
    assert outputs["upper"] == [gbdt_pipeline.MAX_RUL] * 10


def test_metrics_from_outputs_reports_low_rul_diagnostics(
    benchmark_module: ModuleType,
) -> None:
    benchmark = benchmark_module

    outputs = {
        "y_true": [10.0, 50.0, 80.0],
        "y_pred": [20.0, 40.0, 70.0],
        "lower": [15.0, 35.0, 65.0],
        "upper": [25.0, 45.0, 75.0],
    }

    metrics = benchmark._metrics_from_outputs(outputs)

    assert metrics["bias"] == pytest.approx(-10.0 / 3.0)
    assert metrics["overprediction_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["overprediction_mean"] == pytest.approx(10.0)
    assert metrics["low_rul_count_30"] == pytest.approx(1.0)
    assert metrics["low_rul_rmse_30"] == pytest.approx(10.0)
    assert metrics["low_rul_nasa_score_30"] == pytest.approx(
        benchmark._nasa_score([10.0], [20.0])
    )
    assert metrics["low_rul_count_60"] == pytest.approx(2.0)
    assert metrics["low_rul_rmse_60"] == pytest.approx(10.0)
    assert metrics["low_rul_nasa_score_60"] == pytest.approx(
        benchmark._nasa_score([10.0, 50.0], [20.0, 40.0])
    )


def test_afno_benchmark_row_exposes_training_protocol_and_diagnostics(
    benchmark_module: ModuleType,
) -> None:
    benchmark = benchmark_module

    row = benchmark._afno_benchmark_row(
        label="afnocg3_v1_exog_w30_f24_log1p_autoexpand_asym",
        metrics={
            "rmse": 1.0,
            "mae": 0.5,
            "nasa_score": 2.0,
            "cov90": 0.9,
            "width90": 5.0,
            "bias": -3.0,
            "overprediction_rate": 0.25,
            "overprediction_mean": 7.5,
            "low_rul_count_30": 4.0,
            "low_rul_rmse_30": 6.0,
            "low_rul_nasa_score_30": 12.0,
            "low_rul_count_60": 8.0,
            "low_rul_rmse_60": 5.0,
            "low_rul_nasa_score_60": 10.0,
        },
        afno_meta={
            "variant_family": "afno_asymmetric_loss",
            "training_protocol": {"loss": "asymmetric_rul", "asym_over_penalty": 2.0},
        },
    )

    assert row["variant_family"] == "afno_asymmetric_loss"
    assert row["training_protocol"] == {
        "loss": "asymmetric_rul",
        "asym_over_penalty": 2.0,
    }
    assert row["loss"] == "asymmetric_rul"
    assert row["bias"] == pytest.approx(-3.0)
    assert row["low_rul_rmse_30"] == pytest.approx(6.0)


def test_build_benchmark_summary_documents_point_diagnostics(
    benchmark_module: ModuleType,
) -> None:
    benchmark = benchmark_module

    summary = benchmark._build_benchmark_summary(
        feature_keys=["sensor_1"],
        horizon=1,
        folds=1,
        rows=[],
        experiments={},
        benchmark_notes=[],
        phase="full",
    )

    evaluation = summary["diagnostics"]["evaluation"]
    assert evaluation["bias"] == "mean(y_hat - y)"
    assert evaluation["low_rul_rmse_30"] == "rmse restricted to y <= 30"
    assert any("asymmetric-RUL loss ablations" in note for note in summary["notes"])


def test_benchmark_should_run_afno_defaults_off(
    monkeypatch, benchmark_module: ModuleType
) -> None:
    benchmark = benchmark_module

    monkeypatch.delenv("RULFM_BENCHMARK_ENABLE_AFNO", raising=False)

    assert benchmark._benchmark_should_run_afno() is False
    assert any(
        "AFNO benchmark stages are skipped by default" in note
        for note in benchmark._benchmark_mode_notes()
    )


def test_benchmark_should_run_afno_can_be_enabled(
    monkeypatch, benchmark_module: ModuleType
) -> None:
    benchmark = benchmark_module

    monkeypatch.setenv("RULFM_BENCHMARK_ENABLE_AFNO", "1")

    assert benchmark._benchmark_should_run_afno() is True
    assert not any(
        "AFNO benchmark stages are skipped by default" in note
        for note in benchmark._benchmark_mode_notes()
    )


def test_predict_adapter_residuals_sanitizes_non_finite_values(
    benchmark_module: ModuleType,
) -> None:
    benchmark = benchmark_module

    residual = benchmark._predict_adapter_residuals(
        np.asarray([[float("inf"), -float("inf"), float("nan")]], dtype=float),
        {
            "type": "residual_trend_two_stage_summary_only_bounded",
            "mag_coef": [float("inf"), -float("inf"), float("nan")],
            "mag_intercept": float("inf"),
            "magnitude_scale": float("inf"),
            "fallback_sign": 1.0,
            "residual_lower": -50.0,
            "residual_upper": 50.0,
        },
    )

    assert np.all(np.isfinite(residual))
    assert residual.shape == (1,)
    assert residual[0] == pytest.approx(50.0)


def test_normalize_adapter_design_sanitizes_extreme_inputs(
    benchmark_module: ModuleType,
) -> None:
    benchmark = benchmark_module

    train_norm, valid_norm, _mean, _scale = benchmark._normalize_adapter_design(
        np.asarray([[float("inf"), 1.0], [2.0, float("nan")]], dtype=float),
        np.asarray([[float("-inf"), 3.0]], dtype=float),
        standardize=True,
    )

    assert np.all(np.isfinite(train_norm))
    assert np.all(np.isfinite(valid_norm))
    assert np.max(np.abs(train_norm)) <= benchmark._ADAPTER_MATRIX_ABS_CLIP
    assert np.max(np.abs(valid_norm)) <= benchmark._ADAPTER_MATRIX_ABS_CLIP