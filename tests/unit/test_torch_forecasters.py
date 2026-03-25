from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")
torch_forecasters = pytest.importorskip("forecasting_api.torch_forecasters")
pytestmark = pytest.mark.experimental


def test_torch_helper_functions_cover_basic_paths(monkeypatch) -> None:
    assert torch_forecasters._choose_device("cpu") == "cpu"

    monkeypatch.setenv("RULFM_FORECASTING_TORCH_DEVICE", "mps")
    assert torch_forecasters._choose_device(None) == "mps"
    monkeypatch.delenv("RULFM_FORECASTING_TORCH_DEVICE", raising=False)

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert torch_forecasters._choose_device(None, algo="afnocg3_v1") == "cpu"

    assert torch_forecasters._infer_context_len(2) == 1
    assert torch_forecasters._infer_context_len(6) == 3
    assert torch_forecasters._infer_context_len(30) == 14
    assert torch_forecasters._infer_context_len(5, requested=10) == 4

    assert torch_forecasters._record_y({"y": 4.5}) == 4.5
    assert torch_forecasters._record_y({"y": "bad"}) == 0.0
    assert torch_forecasters._record_x({"x": {"a": 1, "b": "x", "c": float("inf")}}) == {"a": 1.0}
    assert torch_forecasters._sorted_records([
        {"timestamp": "2026-01-02", "y": 2.0},
        {"timestamp": "2026-01-01", "y": 1.0},
    ])[0]["y"] == 1.0

    assert torch_forecasters._target_transform_name("log1p") == "log1p"
    assert torch_forecasters._target_transform_name("other") == "none"
    assert (
        torch_forecasters._missing_model_registry(
            ModuleNotFoundError(name="src.models.registry")
        )
        is True
    )
    assert torch_forecasters._missing_model_registry(ValueError("x")) is False


def test_sequence_helpers_and_flat_ridge_fit_cover_data_paths() -> None:
    records_by_series = {
        "s1": [
            {"timestamp": "2026-01-01", "y": 10.0, "x": {"sensor_1": 1.0, "cycle": 99.0}},
            {"timestamp": "2026-01-02", "y": 11.0, "x": {"sensor_1": 2.0}},
            {"timestamp": "2026-01-03", "y": 12.0, "x": {"sensor_1": 3.0}},
            {"timestamp": "2026-01-04", "y": 13.0, "x": {"sensor_1": 4.0}},
        ]
    }

    feature_keys = torch_forecasters._select_feature_keys(
        records_by_series,
        max_features=4,
    )
    assert feature_keys == ["sensor_1"]

    rows = records_by_series["s1"][:2]
    matrix_x = torch_forecasters._rows_to_feature_matrix(
        rows,
        feature_keys=feature_keys,
        feature_source="x",
    )
    matrix_y = torch_forecasters._rows_to_feature_matrix(
        rows,
        feature_keys=[],
        feature_source="y",
    )
    assert matrix_x.shape == (2, 1)
    assert matrix_y.shape == (2, 1)

    x_all, y_all = torch_forecasters._build_supervised_sequences(
        records_by_series,
        context_len=2,
        feature_keys=feature_keys,
        feature_source="x",
    )
    assert x_all.shape == (2, 2, 1)
    assert y_all.tolist() == [12.0, 13.0]

    train_idx, valid_idx = torch_forecasters._split_sequence_indices(30)
    assert train_idx.size > 0
    assert valid_idx.size > 0

    flat_state, score = torch_forecasters._fit_flat_ridge(
        x_train=x_all,
        y_train=y_all,
        x_valid=x_all,
        y_valid=y_all,
        target_transform="none",
    )
    assert "coef" in flat_state
    assert math.isfinite(score)

    assert torch_forecasters._rmse_from_predictions(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 2.0]),
    ) == 0.0
    assert torch_forecasters._rmse_from_absolute_residuals([1.0, -2.0]) > 0.0


def test_forecast_with_details_flat_ridge_covers_occlusion_and_wrapper() -> None:
    snapshot = {
        "context_len": 2,
        "feature_source": "x",
        "feature_keys": ["sensor_1"],
        "target_transform": "none",
        "structure_mode": "flat_ridge",
        "flat_ridge": {"coef": [0.5, 0.5], "intercept": 1.0},
    }
    context_records = [
        {"timestamp": "2026-01-01", "y": 10.0, "x": {"sensor_1": 2.0}},
        {"timestamp": "2026-01-02", "y": 11.0, "x": {"sensor_1": 4.0}},
    ]
    future_feature_rows = [
        {"timestamp": "2026-01-03", "x": {"sensor_1": 5.0}},
        {"timestamp": "2026-01-04", "x": {"sensor_1": 6.0}},
    ]

    details = torch_forecasters.forecast_univariate_torch_with_details(
        algo="afnocg3_v1",
        snapshot=snapshot,
        state_dict={},
        context_records=context_records,
        future_feature_rows=future_feature_rows,
        horizon=2,
        mc_dropout_samples=0,
        occlusion_feature_keys=["sensor_1"],
        occlusion_baseline={"sensor_1": 0.0},
    )

    assert len(details["point"]) == 2
    assert details["mc_dropout"] is None
    assert details["occlusion"] is None

    wrapper_points = torch_forecasters.forecast_univariate_torch(
        algo="afnocg3_v1",
        snapshot=snapshot,
        state_dict={},
        context_records=context_records,
        future_feature_rows=future_feature_rows,
        horizon=2,
    )
    assert wrapper_points == [float(value) for value in details["point"]]


def test_context_padding_and_finetune_helpers_cover_small_model() -> None:
    rows = torch_forecasters._ensure_context_rows(
        context_records=None,
        context=[3.0],
        context_len=3,
    )
    assert len(rows) == 3
    assert rows[-1]["y"] == 3.0

    model = torch.nn.Linear(2, 1)
    x_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    pred_tensor = torch_forecasters._predict_torch_tensor(model, x_tensor)
    assert tuple(pred_tensor.shape) == (1, 1)

    class DummyToggleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(2, 1)
            self.dropout_enabled = False

        def forward(self, x):
            return self.head(x).reshape(-1)

        def enable_final_layer_dropout(self):
            self.dropout_enabled = True

        def disable_final_layer_dropout(self):
            self.dropout_enabled = False

    dummy = DummyToggleModel()
    torch_forecasters._set_mc_dropout_enabled(dummy, True)
    assert dummy.dropout_enabled is True
    torch_forecasters._set_mc_dropout_enabled(dummy, False)
    assert dummy.dropout_enabled is False

    gate_model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    x_train = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    y_train = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)
    torch_forecasters.finetune_gate_stardast2v5(
        model=gate_model,
        X=x_train,
        y=y_train,
        n_total=60,
        device="cpu",
    )


def test_train_univariate_torch_forecaster_falls_back_without_model_registry() -> None:
    records_by_series = {
        "engine-1": [
            {
                "timestamp": f"2026-01-01T00:00:{idx:02d}Z",
                "y": float(125 - idx),
                "x": {"sensor_1": float(idx), "sensor_2": float(idx % 3)},
            }
            for idx in range(20)
        ]
    }

    artifact = torch_forecasters.train_univariate_torch_forecaster(
        algo="afnocg3_v1",
        records_by_series=records_by_series,
        training_hours=0.05,
        context_len=8,
        max_exogenous_features=2,
        prefer_exogenous=True,
        allow_structure_fallback=True,
    )

    assert artifact.snapshot["structure_mode"] == "flat_ridge"
    assert artifact.snapshot.get("missing_dependency") == "src.models.registry"
    assert artifact.state_dict == {}