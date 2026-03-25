from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from types import SimpleNamespace

from forecasting_api import mlflow_runs

from tests.helpers import raising_callable


class _DummyRun:
    def __enter__(self):
        return {"run_id": "abc"}

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyMlflow:
    def __init__(self) -> None:
        self.tracking_uri = None
        self.experiment = None
        self.tags = None
        self.logged_params = None
        self.logged_metrics = None
        self.logged_dict = None
        self.logged_artifact = None

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, *, run_name: str, nested: bool = False):
        self.last_run_name = run_name
        self.last_nested = nested
        return _DummyRun()

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags = tags

    def log_params(self, params: dict[str, str]) -> None:
        self.logged_params = params

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.logged_metrics = metrics

    def log_dict(self, payload: dict[str, object], name: str) -> None:
        self.logged_dict = (name, payload)

    def log_artifact(self, path: str) -> None:
        self.logged_artifact = path


def test_tracking_uri_and_experiment_name(monkeypatch):
    monkeypatch.setenv("RULFM_MLFLOW_TRACKING_URI", " http://mlflow ")
    monkeypatch.setenv("RULFM_MLFLOW_EXPERIMENT", " custom-exp ")
    assert mlflow_runs._tracking_uri() == "http://mlflow"
    assert mlflow_runs._experiment_name() == "custom-exp"
    assert mlflow_runs.mlflow_enabled() is True


def test_load_mlflow_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("RULFM_MLFLOW_TRACKING_URI", raising=False)
    assert mlflow_runs._load_mlflow() is None


def test_start_run_and_logging_helpers(monkeypatch, tmp_path):
    dummy = _DummyMlflow()
    monkeypatch.setattr(mlflow_runs, "_load_mlflow", lambda: dummy)

    with mlflow_runs.start_run("test-run", tags={"stage": "dev"}, nested=True) as run:
        assert run == {"run_id": "abc"}

    mlflow_runs.log_params({"alpha": 1, "skip": None})
    mlflow_runs.log_metrics({"rmse": 1.2, "bad": float("inf"), "name": "x"})
    mlflow_runs.log_dict_artifact("payload.json", {"severity": "low"})

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")
    mlflow_runs.log_artifact(artifact)

    assert dummy.tracking_uri is None
    assert dummy.tags == {"stage": "dev"}
    assert dummy.logged_params == {"alpha": "1"}
    assert dummy.logged_metrics == {"rmse": 1.2}
    assert dummy.logged_dict == ("payload.json", {"severity": "low"})
    assert dummy.logged_artifact == str(artifact)


def test_start_run_yields_none_on_backend_failure(monkeypatch):
    class _BrokenMlflow(_DummyMlflow):
        def start_run(self, *, run_name: str, nested: bool = False):
            raise RuntimeError("boom")

    monkeypatch.setattr(mlflow_runs, "_load_mlflow", lambda: _BrokenMlflow())
    with mlflow_runs.start_run("broken") as run:
        assert run is None


def test_logging_helpers_swallow_backend_errors(monkeypatch, tmp_path):
    failing = SimpleNamespace(
        log_params=raising_callable(RuntimeError("params")),
        log_metrics=raising_callable(RuntimeError("metrics")),
        log_dict=raising_callable(RuntimeError("dict")),
        log_artifact=raising_callable(RuntimeError("artifact")),
    )
    monkeypatch.setattr(mlflow_runs, "_load_mlflow", lambda: failing)

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")

    mlflow_runs.log_params({"alpha": 1})
    mlflow_runs.log_metrics({"rmse": 1.2})
    mlflow_runs.log_dict_artifact("payload.json", {"severity": "low"})
    mlflow_runs.log_artifact(artifact)


def test_log_artifact_skips_missing_paths(monkeypatch, tmp_path):
    dummy = _DummyMlflow()
    monkeypatch.setattr(mlflow_runs, "_load_mlflow", lambda: dummy)
    mlflow_runs.log_artifact(Path(tmp_path / "missing.txt"))
    assert dummy.logged_artifact is None


def test_real_file_tracking_creates_mlflow_run(monkeypatch, tmp_path) -> None:
    tracking_db = tmp_path / "mlflow.db"
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RULFM_MLFLOW_TRACKING_URI", f"sqlite:///{tracking_db}")
    monkeypatch.setenv("RULFM_MLFLOW_EXPERIMENT", "unit-test-exp")

    with mlflow_runs.start_run("real-run", tags={"stage": "test"}) as run:
        assert run is not None
        mlflow_runs.log_params({"alpha": 1})
        mlflow_runs.log_metrics({"rmse": 1.25})
        mlflow_runs.log_dict_artifact("payload.json", {"severity": "low"})
        mlflow_runs.log_artifact(artifact)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Support for class-based .*",
            category=DeprecationWarning,
        )
        mlflow = importlib.import_module("mlflow")
        client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tracking_db}")
    experiment = client.get_experiment_by_name("unit-test-exp")

    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert runs
    latest = runs[0]
    assert latest.data.params["alpha"] == "1"
    assert latest.data.metrics["rmse"] == 1.25
    artifact_paths = [item.path for item in client.list_artifacts(latest.info.run_id)]
    assert "artifact.txt" in artifact_paths
    assert "payload.json" in artifact_paths
