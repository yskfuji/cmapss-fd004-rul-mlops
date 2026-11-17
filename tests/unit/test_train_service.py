# ruff: noqa: I001, E501
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from forecasting_api import app as app_module
from forecasting_api.domain import stable_models
from forecasting_api import training_helpers
from forecasting_api.services import train_service
from forecasting_api.services.runtime import TrainServiceDeps


def _train_records() -> list[app_module.TimeSeriesRecord]:
    return [
        app_module.TimeSeriesRecord(
            series_id="s1",
            timestamp=f"2026-03-{day:02d}T00:00:00Z",
            y=float(day),
            x={"sensor_1": float(day), "sensor_2": float(day % 3)},
        )
        for day in range(1, 11)
    ]


@pytest.fixture
def configured_train_service(
    patched_app_runtime: dict[str, list[object]],
) -> dict[str, list[object]]:
    train_service.configure_train_service(
        TrainServiceDeps(
            api_error_cls=app_module.ApiError,
            trained_models=app_module._require_trained_models,
            require_monotonic_increasing=lambda records: app_module._require_monotonic_increasing(records),
            normalize_base_model_name=training_helpers.normalize_base_model_name,
            assert_model_algo_available=lambda algo: training_helpers.assert_model_algo_available(
                algo,
                api_error_cls=app_module.ApiError,
            ),
            fit_ridge_lags_model=lambda req: training_helpers.fit_ridge_lags_model(
                req,
                ridge_lags_choose_k=stable_models.ridge_lags_choose_k,
                ridge_lags_fit_series=stable_models.ridge_lags_fit_series,
            ),
            train_public_gbdt_entry=lambda req, *, model_id: training_helpers.train_public_gbdt_entry(
                req,
                model_id=model_id,
                model_artifact_dir=app_module._model_artifact_dir,
                write_json=app_module._write_json,
                artifact_relpath=app_module._artifact_relpath,
            ),
            train_hybrid_entry=lambda req, *, model_id: training_helpers.train_hybrid_entry(
                req,
                model_id=model_id,
                model_artifact_dir=app_module._model_artifact_dir,
                write_json=app_module._write_json,
                artifact_relpath=app_module._artifact_relpath,
            ),
            model_artifact_dir=lambda model_id: app_module._model_artifact_dir(model_id),
            write_json=lambda path, data: app_module._write_json(path, data),
            artifact_relpath=lambda model_id, filename: app_module._artifact_relpath(model_id, filename),
            artifact_abspath=lambda relpath: app_module._artifact_abspath(relpath),
            save_trained_model=lambda entry: app_module._save_trained_model(entry),
            start_run=lambda *args, **kwargs: app_module.start_run(*args, **kwargs),
            log_params=lambda payload: app_module.log_params(payload),
            log_metrics=lambda payload: app_module.log_metrics(payload),
            log_dict_artifact=lambda name, payload: app_module.log_dict_artifact(name, payload),
            log_artifact=lambda path: app_module.log_artifact(path),
        )
    )
    return patched_app_runtime


def test_run_train_uses_ridge_base_model_and_persists_entry(
    monkeypatch,
    configured_train_service: dict[str, list[object]],
) -> None:
    logged = configured_train_service
    monkeypatch.setattr(
        training_helpers,
        "fit_ridge_lags_model",
        lambda req, ridge_lags_choose_k, ridge_lags_fit_series: {
            "algo": "ridge_lags_v1",
            "pooled_residuals": [0.1, 0.2],
            "series": {},
        },
    )

    response = train_service.run_train(
        app_module.TrainRequest(
            base_model="ridge",
            model_name="stable ridge",
            training_hours=0.1,
            data=_train_records(),
        )
    )

    assert response.message == "accepted"
    stored = app_module._require_trained_models()[response.model_id]
    assert stored["algo"] == "ridge_lags_v1"
    assert stored["base_model"] == "ridge"
    assert stored["memo"] == "stable ridge"
    assert logged["params"][0]["algo"] == "ridge_lags_v1"
    assert logged["artifacts"] == []


def test_run_train_uses_gbdt_base_model_and_logs_artifacts(
    monkeypatch,
    configured_train_service: dict[str, list[object]],
) -> None:
    logged = configured_train_service
    monkeypatch.setattr(
        training_helpers,
        "train_public_gbdt_entry",
        lambda req, model_id, model_artifact_dir, write_json, artifact_relpath: {
            "context_len": 3,
            "input_dim": 6,
            "pooled_residuals": [0.2, 0.3, 0.4],
            "artifact": {
                "snapshot_json": f"{model_id}/snapshot.json",
                "gbdt_joblib": f"{model_id}/gbdt.joblib",
            },
        },
    )
    monkeypatch.setattr(app_module, "_artifact_abspath", lambda rel: Path("/tmp") / rel)

    response = train_service.run_train(
        app_module.TrainRequest(base_model="gbdt", training_hours=0.1, data=_train_records())
    )

    assert response.message == "accepted"
    stored = app_module._require_trained_models()[response.model_id]
    assert stored["algo"] == "gbdt_hgb_v1"
    assert stored["artifact"]["gbdt_joblib"].endswith("gbdt.joblib")
    assert logged["metrics"][0]["residuals.count"] == 3
    assert logged["artifacts"] == [
        Path("/tmp") / f"{response.model_id}/snapshot.json",
        Path("/tmp") / f"{response.model_id}/gbdt.joblib",
    ]


def test_run_train_maps_gbdt_value_error_to_api_error(
    monkeypatch,
    configured_train_service: dict[str, list[object]],
) -> None:
    monkeypatch.setattr(
        training_helpers,
        "train_public_gbdt_entry",
        lambda req, model_id, model_artifact_dir, write_json, artifact_relpath: (_ for _ in ()).throw(
            ValueError("missing numeric x")
        ),
    )

    with pytest.raises(app_module.ApiError) as exc_info:
        train_service.run_train(
            app_module.TrainRequest(base_model="gbdt", training_hours=0.1, data=_train_records())
        )

    assert exc_info.value.error_code == "V01"
    assert exc_info.value.details["error"] == "missing numeric x"


def test_run_train_hybrid_branch_logs_hybrid_artifact(
    monkeypatch,
    configured_train_service: dict[str, list[object]],
) -> None:
    logged = configured_train_service
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    monkeypatch.setattr(
        training_helpers,
        "train_hybrid_entry",
        lambda req, model_id, model_artifact_dir, write_json, artifact_relpath: {
            "context_len": 7,
            "input_dim": 12,
            "pooled_residuals": [0.5, 0.4],
            "artifact": {
                "hybrid_json": f"{model_id}/hybrid.json",
            },
        },
    )
    monkeypatch.setattr(app_module, "_artifact_abspath", lambda rel: Path("/tmp") / rel)

    response = train_service.run_train(
        app_module.TrainRequest(
            algo="gbdt_afno_hybrid_v1",
            training_hours=0.1,
            data=_train_records(),
        )
    )

    assert response.message == "accepted"
    stored = app_module._require_trained_models()[response.model_id]
    assert stored["algo"] == "gbdt_afno_hybrid_v1"
    assert logged["metrics"][0]["residuals.count"] == 2
    assert logged["artifacts"] == [Path("/tmp") / f"{response.model_id}/hybrid.json"]


def test_run_train_maps_hybrid_value_error_to_api_error(
    monkeypatch,
    configured_train_service: dict[str, list[object]],
) -> None:
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    monkeypatch.setattr(
        training_helpers,
        "train_hybrid_entry",
        lambda req, model_id, model_artifact_dir, write_json, artifact_relpath: (_ for _ in ()).throw(
            ValueError("hybrid needs more numeric features")
        ),
    )

    with pytest.raises(app_module.ApiError) as exc_info:
        train_service.run_train(
            app_module.TrainRequest(
                algo="gbdt_afno_hybrid_v1",
                training_hours=0.1,
                data=_train_records(),
            )
        )

    assert exc_info.value.error_code == "V01"
    assert exc_info.value.details["error"] == "hybrid needs more numeric features"


def test_run_train_torch_branch_writes_snapshot_and_weights(
    monkeypatch,
    tmp_path,
    configured_train_service: dict[str, list[object]],
) -> None:
    logged = configured_train_service
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    monkeypatch.setattr(app_module, "_MODEL_ARTIFACTS_ROOT", tmp_path)

    fake_torch_forecasters = ModuleType("forecasting_api.torch_forecasters")
    fake_torch_forecasters.train_univariate_torch_forecaster = lambda **kwargs: SimpleNamespace(
        snapshot={"algo": kwargs["algo"], "kind": "fake-torch"},
        state_dict={"weight": [1.0, 2.0]},
        context_len=30,
        input_dim=4,
        pooled_residuals=[0.2, 0.3, 0.4],
    )
    fake_torch = ModuleType("torch")
    fake_torch.save = lambda payload, path: Path(path).write_text("weights", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "forecasting_api.torch_forecasters", fake_torch_forecasters)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    response = train_service.run_train(
        app_module.TrainRequest(algo="afnocg3_v1", training_hours=0.1, data=_train_records())
    )

    snapshot_path = tmp_path / response.model_id / "snapshot.json"
    weights_path = tmp_path / response.model_id / "weights.pt"
    assert snapshot_path.exists()
    assert weights_path.exists()
    stored = app_module._require_trained_models()[response.model_id]
    assert stored["artifact"]["weights_pt"].endswith("weights.pt")
    assert logged["metrics"][0]["residuals.count"] == 3
    assert logged["artifacts"] == [snapshot_path.resolve(), weights_path.resolve()]


def test_run_train_maps_torch_value_error_to_api_error(
    monkeypatch,
    configured_train_service: dict[str, list[object]],
) -> None:
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")

    fake_torch_forecasters = ModuleType("forecasting_api.torch_forecasters")

    def _raise_value_error(**kwargs):
        raise ValueError("torch needs longer series")

    fake_torch_forecasters.train_univariate_torch_forecaster = _raise_value_error
    monkeypatch.setitem(sys.modules, "forecasting_api.torch_forecasters", fake_torch_forecasters)

    with pytest.raises(app_module.ApiError) as exc_info:
        train_service.run_train(
            app_module.TrainRequest(algo="afnocg2", training_hours=0.1, data=_train_records())
        )

    assert exc_info.value.error_code == "V01"
    assert exc_info.value.details["error"] == "torch needs longer series"