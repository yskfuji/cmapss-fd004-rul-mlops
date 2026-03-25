from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.experimental

from forecasting_api import torch_forecasters


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