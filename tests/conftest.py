from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pytest

from forecasting_api import app as app_module


@pytest.fixture
def api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def isolated_experimental_models_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", raising=False)


@pytest.fixture(autouse=True)
def isolated_trained_models(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    app_module._set_runtime_trained_models(models)
    yield models
    app_module._set_runtime_trained_models(None)


@pytest.fixture
def patched_backtest_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "start_run", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(app_module, "log_params", lambda payload: None)
    monkeypatch.setattr(app_module, "log_metrics", lambda payload: None)
    monkeypatch.setattr(app_module, "log_dict_artifact", lambda name, payload: None)


@pytest.fixture
def patched_app_runtime(
    monkeypatch: pytest.MonkeyPatch,
    isolated_trained_models: dict[str, dict[str, Any]],
) -> dict[str, list[Any]]:
    logged: dict[str, list[Any]] = {
        "params": [],
        "metrics": [],
        "dict_artifacts": [],
        "artifacts": [],
    }
    monkeypatch.setattr(app_module, "_save_trained_models", lambda models: None)
    monkeypatch.setattr(app_module, "_save_trained_model", lambda entry: None)
    monkeypatch.setattr(app_module, "start_run", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(app_module, "log_params", lambda payload: logged["params"].append(payload))
    monkeypatch.setattr(
        app_module,
        "log_metrics",
        lambda payload: logged["metrics"].append(payload),
    )
    monkeypatch.setattr(
        app_module,
        "log_dict_artifact",
        lambda name, payload: logged["dict_artifacts"].append((name, payload)),
    )
    monkeypatch.setattr(app_module, "log_artifact", lambda path: logged["artifacts"].append(path))
    app_module._set_runtime_trained_models(isolated_trained_models)
    return logged