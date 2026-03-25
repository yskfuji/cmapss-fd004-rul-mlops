# ruff: noqa: I001
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks

from forecasting_api import app as app_module
from forecasting_api import hybrid_runtime
from forecasting_api import request_audit as request_audit_module
from forecasting_api.domain import stable_models
from forecasting_api import training_helpers


def _fake_request(
    *,
    path: str = "/v1/forecast",
    method: str = "GET",
    scheme: str = "http",
    headers: dict[str, str] | None = None,
    client_host: str = "127.0.0.1",
    request_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        headers=headers or {},
        url=SimpleNamespace(path=path, scheme=scheme),
        client=SimpleNamespace(host=client_host),
        method=method,
        state=SimpleNamespace(request_id=request_id),
    )


def test_build_job_error_payload_preserves_error_details() -> None:
    payload = app_module.build_job_error_payload(
        error_code="V01",
        message="bad request",
        details=app_module.ErrorDetails(next_action="fix payload"),
    )

    assert payload["error_code"] == "V01"
    assert payload["message"] == "bad request"
    assert payload["details"]["next_action"] == "fix payload"
    assert "request_id" not in payload


def test_small_helpers_cover_env_json_and_artifact_paths(monkeypatch, tmp_path) -> None:
    assert app_module._bi("hello", "こんにちは") == "[EN] hello\n[JA] こんにちは"
    assert app_module._as_dict({"x": 1}) == {"x": 1}
    assert app_module._as_dict([1, 2]) == {}
    assert app_module._as_list([1, 2]) == [1, 2]
    assert app_module._as_list((1, 2)) == []
    assert app_module._as_float_list([1, 2.5, float("inf"), "x"]) == [1.0, 2.5]

    monkeypatch.setenv("RULFM_HELPER_PATH", str(tmp_path / "helper.json"))
    assert app_module._env_first("RULFM_HELPER_PATH") == str(tmp_path / "helper.json")
    assert app_module._env_path(tmp_path / "fallback.json", "RULFM_HELPER_PATH") == tmp_path / "helper.json"

    monkeypatch.setattr(app_module, "_MODEL_ARTIFACTS_ROOT", tmp_path)
    artifact_dir = app_module._model_artifact_dir("model-a")
    assert artifact_dir == tmp_path / "model-a"
    assert app_module._artifact_relpath("model-a", "snapshot.json") == "model-a/snapshot.json"
    assert app_module._artifact_abspath("model-a/snapshot.json") == (tmp_path / "model-a" / "snapshot.json").resolve()

    payload_path = tmp_path / "payload.json"
    app_module._write_json(payload_path, {"ok": True})
    assert app_module._read_json(payload_path) == {"ok": True}


def test_storage_and_torch_helpers_cover_fallback_paths(monkeypatch, tmp_path) -> None:
    non_dict_path = tmp_path / "list.json"
    non_dict_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert app_module._read_json(non_dict_path) == {}

    summary_path = tmp_path / "missing-summary.json"
    monkeypatch.setattr(app_module, "_FD004_BENCHMARK_SUMMARY_PATH", summary_path)
    payload = app_module._load_fd004_benchmark_summary()
    assert payload["rows"] == []
    assert payload["notes"][0] == "FD004 benchmark summary has not been generated yet."

    calls: list[str] = []
    monkeypatch.setattr(app_module, "save_models_to_store", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        app_module,
        "_LOGGER",
        SimpleNamespace(exception=lambda message, **kwargs: calls.append(str(message))),
    )
    app_module._save_trained_models({"m1": {"model_id": "m1"}})
    assert calls == ["Failed to persist trained models"]

    assert app_module._extract_state_dict({"state_dict": {"a": 1}}) == {"a": 1}
    assert app_module._extract_state_dict({"model_state": {"b": 2}}) == {"b": 2}
    assert app_module._extract_state_dict({"model": {"c": 3}}) == {"c": 3}
    assert app_module._extract_state_dict({"weights": {"d": 4}}) == {"weights": {"d": 4}}

    torch = pytest.importorskip("torch")
    load_calls: list[dict[str, object]] = []

    def _fake_torch_load(path, map_location=None, **kwargs):
        load_calls.append({"path": path, "map_location": map_location, **kwargs})
        if kwargs.get("weights_only") is True:
            raise TypeError("weights_only unsupported")
        return "tensor-payload"

    monkeypatch.setattr(torch, "load", _fake_torch_load)
    loaded = app_module._try_torch_load_weights(tmp_path / "dummy.pt")
    assert loaded == {"state_dict": "tensor-payload"}
    assert load_calls[0]["weights_only"] is True
    assert "weights_only" not in load_calls[1]


def test_model_algo_and_validation_helpers_cover_error_paths(monkeypatch) -> None:
    monkeypatch.delenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", raising=False)

    assert training_helpers.normalize_base_model_name(None) == "default"
    assert training_helpers.normalize_base_model_name(" Ridge ") == "ridge"
    assert training_helpers.is_experimental_model_algo(" afnocg2 ") is True
    assert training_helpers.assert_model_algo_available(
        "ridge_lags_v1",
        api_error_cls=app_module.ApiError,
    ) == "ridge_lags_v1"

    with pytest.raises(app_module.ApiError) as exc_info:
        training_helpers.assert_model_algo_available(
            "afnocg2",
            api_error_cls=app_module.ApiError,
        )
    assert exc_info.value.message == "experimental algorithm は公開APIでは無効です"

    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    assert training_helpers.assert_model_algo_available(
        "afnocg2",
        api_error_cls=app_module.ApiError,
    ) == "afnocg2"

    req = app_module.ForecastRequest(
        horizon=2,
        quantiles=[0.1],
        level=[90],
        data=_daily_series_records("s1", 3),
    )
    with pytest.raises(app_module.ApiError) as exc_info:
        app_module._ensure_quantiles_level_exclusive(req)
    assert exc_info.value.error_code == "V03"


def test_frequency_and_gap_helpers_cover_infer_and_validation_paths(monkeypatch) -> None:
    ordered = [
        app_module.TimeSeriesRecord(series_id="s1", timestamp="2026-06-01T00:00:00Z", y=1.0),
        app_module.TimeSeriesRecord(series_id="s1", timestamp="2026-06-02T00:00:00Z", y=2.0),
        app_module.TimeSeriesRecord(series_id="s1", timestamp="2026-06-03T00:00:00Z", y=3.0),
    ]
    irregular = [
        app_module.TimeSeriesRecord(series_id="s1", timestamp="2026-06-01T00:00:00Z", y=1.0),
        app_module.TimeSeriesRecord(series_id="s1", timestamp="2026-06-03T00:00:00Z", y=2.0),
        app_module.TimeSeriesRecord(series_id="s1", timestamp="2026-06-04T12:00:00Z", y=3.0),
    ]
    not_monotonic = [
        app_module.TimeSeriesRecord(series_id="s1", timestamp="2026-06-02T00:00:00Z", y=2.0),
        app_module.TimeSeriesRecord(series_id="s1", timestamp="2026-06-01T00:00:00Z", y=1.0),
    ]

    assert app_module._infer_step_delta(ordered) == timedelta(days=1)
    assert app_module._infer_step_delta(irregular) is None
    assert app_module._require_frequency_or_infer(
        app_module.ForecastRequest(horizon=1, frequency="15min", data=ordered)
    ) == timedelta(minutes=15)
    assert app_module._require_frequency_or_infer(
        app_module.ForecastRequest(horizon=1, data=ordered)
    ) == timedelta(days=1)

    with pytest.raises(app_module.ApiError) as exc_info:
        app_module._require_frequency_or_infer(app_module.ForecastRequest(horizon=1, data=irregular))
    assert exc_info.value.message == "frequency を推定できません"

    with pytest.raises(app_module.ApiError) as exc_info:
        app_module._require_monotonic_increasing(not_monotonic)
    assert exc_info.value.error_code == "V04"

    gap_req = SimpleNamespace(
        data=irregular,
        options=SimpleNamespace(missing_policy="error"),
    )
    with pytest.raises(app_module.ApiError) as exc_info:
        app_module._require_no_gaps_if_missing_policy_error(gap_req, timedelta(days=1))
    assert exc_info.value.error_code == "V05"

    no_gap_req = SimpleNamespace(
        data=ordered,
        options=SimpleNamespace(missing_policy="ignore"),
    )
    app_module._require_no_gaps_if_missing_policy_error(no_gap_req, timedelta(days=1))

    monkeypatch.setattr(app_module, "TRAINED_MODELS", {"model-ok": {"model_id": "model-ok"}})
    app_module._require_trained_model("model-ok")
    with pytest.raises(app_module.ApiError) as exc_info:
        app_module._require_trained_model("missing-model")
    assert exc_info.value.error_code == "M01"


def test_request_auth_rate_limit_and_audit_helpers_cover_runtime_paths(monkeypatch, tmp_path) -> None:
    request = _fake_request(request_id="req-123")
    response = app_module._error_json(
        request,
        app_module.ApiError(
            status_code=401,
            error_code="A12",
            message="APIキーが無効です",
            details={"next_action": "configure auth"},
        ),
    )
    body = json.loads(response.body)
    assert app_module._get_request_id(request) == "req-123"
    assert body["request_id"] == "req-123"
    assert body["details"]["next_action"] == "configure auth"

    assert app_module._is_https_request(_fake_request(headers={"x-forwarded-proto": "https"})) is True
    assert app_module._is_https_request(_fake_request(scheme="https")) is True
    assert app_module._is_https_request(_fake_request()) is False

    audit_path = tmp_path / "audit" / "requests.jsonl"
    monkeypatch.setenv("RULFM_FORECASTING_API_AUDIT_LOG_PATH", str(audit_path))
    assert app_module._audit_log_path() == audit_path
    assert app_module._sha256_hex("payload") == hashlib.sha256(b"payload").hexdigest()
    assert request_audit_module._last_entry_hash(audit_path) is None

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("not-json\n", encoding="utf-8")
    assert request_audit_module._last_entry_hash(audit_path) is None

    request_audit_module.append_request_audit_log({"event": "skip"}, path=audit_path, enabled=False)
    assert audit_path.read_text(encoding="utf-8") == "not-json\n"

    audit_path.unlink()
    request_audit_module.append_request_audit_log({"event": "first"}, path=audit_path)
    request_audit_module.append_request_audit_log({"event": "second"}, path=audit_path)

    lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8") .splitlines()]
    assert lines[0]["event_type"] == "API_REQUEST_COMPLETED"
    assert lines[0]["tenant_id"] == "public"
    assert lines[0]["details"]["prev_hash"] is None
    assert lines[1]["details"]["prev_hash"] == lines[0]["details"]["entry_hash"]

    request = _fake_request(method="POST", path="/v1/train", client_host="10.0.0.5")
    request.state.auth_method = "bearer"
    assert app_module._rate_limit_key(request) == "POST:/v1/train:10.0.0.5:bearer"

    app_module._RATE_LIMIT_HITS.clear()
    ticks = iter([100.0, 101.0, 102.0, 113.0])
    monkeypatch.setattr(app_module, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(app_module, "_rate_limit_window_seconds", lambda: 10)
    monkeypatch.setattr(app_module, "_rate_limit_per_window", lambda: 2)

    assert app_module._consume_rate_limit(request) == (True, 0)
    assert app_module._consume_rate_limit(request) == (True, 0)
    allowed, retry_after = app_module._consume_rate_limit(request)
    assert allowed is False
    assert retry_after == 8
    assert app_module._consume_rate_limit(request) == (True, 0)


def test_require_api_key_covers_misconfigured_success_and_oidc_fallback(monkeypatch) -> None:
    request = _fake_request()

    monkeypatch.setattr(app_module, "_expected_api_key", lambda: None)
    monkeypatch.setattr(app_module, "_expected_bearer_token", lambda: None)
    monkeypatch.setattr(app_module, "_oidc_enabled", lambda: False)
    with pytest.raises(app_module.ApiError) as exc_info:
        app_module._require_api_key(request, authorization=None)
    assert exc_info.value.error_code == "A12"

    request = _fake_request()
    monkeypatch.setattr(app_module, "_expected_api_key", lambda: "secret-key")
    monkeypatch.setattr(app_module, "_expected_bearer_token", lambda: None)
    monkeypatch.setattr(app_module, "_oidc_enabled", lambda: False)
    app_module._require_api_key(request, x_api_key="secret-key", authorization=None)
    assert request.state.auth_method == "x-api-key"

    request = _fake_request()
    monkeypatch.setattr(app_module, "_expected_api_key", lambda: None)
    monkeypatch.setattr(app_module, "_expected_bearer_token", lambda: "bearer-secret")
    app_module._require_api_key(request, authorization="Bearer bearer-secret")
    assert request.state.auth_method == "bearer"

    request = _fake_request()
    monkeypatch.setattr(app_module, "_expected_api_key", lambda: None)
    monkeypatch.setattr(app_module, "_expected_bearer_token", lambda: None)
    monkeypatch.setattr(app_module, "_oidc_enabled", lambda: True)
    monkeypatch.setattr(
        app_module,
        "_validate_oidc_bearer_token",
        lambda token: {"sub": "user-1", "groups": ["ml-approvers"]},
    )
    app_module._require_api_key(request, authorization="Bearer oidc-token")
    assert request.state.auth_method == "oidc-bearer"
    assert request.state.auth_subject == "user-1"
    assert request.state.auth_claims == {"sub": "user-1", "groups": ["ml-approvers"]}

    warnings: list[str] = []
    request = _fake_request()
    monkeypatch.setattr(app_module, "_validate_oidc_bearer_token", lambda token: (_ for _ in ()).throw(RuntimeError("bad token")))
    monkeypatch.setattr(
        app_module,
        "_LOGGER",
        SimpleNamespace(warning=lambda message, **kwargs: warnings.append(str(message))),
    )
    with pytest.raises(app_module.ApiError) as exc_info:
        app_module._require_api_key(request, authorization="Bearer oidc-token")
    assert exc_info.value.error_code == "A12"
    assert request.state.auth_method == "invalid"
    assert warnings == ["OIDC bearer token validation failed"]


def test_require_api_access_enforces_policy_after_auth(monkeypatch) -> None:
    request = _fake_request()
    calls: list[tuple[str, str | None, str | None]] = []

    monkeypatch.setattr(
        app_module,
        "_require_api_key",
        lambda req, x_api_key=None, authorization=None: setattr(req.state, "auth_method", "x-api-key"),
    )
    monkeypatch.setattr(
        app_module,
        "enforce_request_policy",
        lambda req, tenant_id, connection_type: calls.append((req.state.auth_method, tenant_id, connection_type)),
    )

    app_module._require_api_access(
        request,
        x_api_key="secret",
        x_tenant_id="tenant-a",
        x_connection_type="private",
    )

    assert calls == [("x-api-key", "tenant-a", "private")]


def test_build_inprocess_job_enqueuer_adds_background_task(monkeypatch) -> None:
    scheduled: list[tuple[object, tuple[object, ...]]] = []
    background = BackgroundTasks()

    monkeypatch.setattr(
        background,
        "add_task",
        lambda func, *args, **kwargs: scheduled.append((func, args)),
    )

    enqueuer = app_module.build_inprocess_job_enqueuer(
        background=background,
        run_job=lambda job_id, job_type, payload: None,
    )
    enqueuer.enqueue(job_id="job-1", job_type="forecast", payload={"horizon": 1})

    assert len(scheduled) == 1
    assert scheduled[0][1] == ("job-1", "forecast", {"horizon": 1})


def test_build_persistent_job_enqueuer_is_noop() -> None:
    enqueuer = app_module.build_persistent_job_enqueuer(background=None)
    assert enqueuer.enqueue(job_id="job-1", job_type="forecast", payload={"horizon": 1}) is None


def test_statistical_helpers_cover_quantiles_and_seasonality() -> None:
    assert stable_models.ridge_lags_choose_k(3) == 1
    assert stable_models.ridge_lags_choose_k(8) == 2
    assert stable_models.ridge_lags_choose_k(20) == 5
    assert stable_models.ridge_lags_choose_k(21) == 14

    assert stable_models.quantile_nearest_rank([], 0.5) == 0.0
    assert stable_models.quantile_nearest_rank([1.0, 3.0, 2.0], 0.0) == 1.0
    assert stable_models.quantile_nearest_rank([1.0, 3.0, 2.0], 1.0) == 3.0

    evidence = stable_models.build_residuals_evidence([float(idx) for idx in range(600)])
    assert evidence is not None
    assert evidence["n"] == 500

    assert stable_models.safe_std([5.0]) == 0.0
    assert stable_models.safe_std([1.0, 3.0]) > 0.0
    assert stable_models._infer_seasonal_period_steps(timedelta(hours=1)) == 24
    assert stable_models._infer_seasonal_period_steps(timedelta(days=1)) == 7
    assert stable_models._infer_seasonal_period_steps(timedelta(days=30)) is None


def test_run_backtest_request_uses_ridge_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        stable_models,
        "ridge_lags_backtest",
        lambda req, trained=None, **kwargs: app_module.BacktestResponse(metrics={"rmse": 0.42}),
    )
    app_module._set_runtime_trained_models(
        {"model_ridge": {"model_id": "model_ridge", "algo": "ridge_lags_v1"}}
    )

    result = app_module.run_backtest_request(
        app_module.BacktestRequest(
            horizon=2,
            folds=2,
            metric="rmse",
            model_id="model_ridge",
            data=[
                {
                    "series_id": "s1",
                    "timestamp": f"2026-03-{day:02d}T00:00:00Z",
                    "y": float(day),
                }
                for day in range(1, 10)
            ],
        )
    )

    assert result.metrics["rmse"] == 0.42


def test_run_backtest_request_falls_back_to_naive(
    monkeypatch,
    isolated_trained_models: dict[str, dict[str, object]],
) -> None:
    monkeypatch.setattr(
        app_module,
        "_naive_backtest",
        lambda req: app_module.BacktestResponse(metrics={"rmse": 1.23}),
    )

    result = app_module.run_backtest_request(
        app_module.BacktestRequest(
            horizon=2,
            folds=2,
            metric="rmse",
            data=[
                {
                    "series_id": "s1",
                    "timestamp": f"2026-03-{day:02d}T00:00:00Z",
                    "y": float(day),
                }
                for day in range(1, 10)
            ],
        )
    )

    assert result.metrics["rmse"] == 1.23


def _series_records() -> list[app_module.TimeSeriesRecord]:
    records: list[app_module.TimeSeriesRecord] = []
    for series_id, offset in (("s1", 0.0), ("s2", 5.0)):
        for day in range(1, 19):
            records.append(
                app_module.TimeSeriesRecord(
                    series_id=series_id,
                    timestamp=f"2026-03-{day:02d}T00:00:00Z",
                    y=offset + float(day),
                    x={
                        "sensor_1": float(day),
                        "sensor_2": float(day % 4),
                        "status": "ignore-me",
                        "cycle": float(day),
                    },
                )
            )
    return records


def _long_series_records() -> list[app_module.TimeSeriesRecord]:
    return [
        app_module.TimeSeriesRecord(
            series_id="long",
            timestamp=f"2026-04-{day:02d}T00:00:00Z",
            y=float(day),
            x={"sensor_1": float(day), "sensor_2": float(day % 5)},
        )
        for day in range(1, 25)
    ]


def _daily_series_records(series_id: str, total_days: int) -> list[app_module.TimeSeriesRecord]:
    return [
        app_module.TimeSeriesRecord(
            series_id=series_id,
            timestamp=f"2026-05-{day:02d}T00:00:00Z",
            y=float(day * 2),
            x={"sensor_1": float(day), "sensor_2": float(day % 7)},
        )
        for day in range(1, total_days + 1)
    ]


def test_training_payload_helper_functions_cover_stable_gbdt_prep() -> None:
    ys_by_series, records_by_series = training_helpers.build_request_training_payload(
        _series_records()
    )

    assert sorted(ys_by_series) == ["s1", "s2"]
    assert ys_by_series["s1"][0] == 1.0

    feature_keys = training_helpers.select_series_feature_keys(records_by_series, max_features=4)
    assert feature_keys == ["sensor_1", "sensor_2"]

    assert training_helpers.hybrid_context_len({"tiny": records_by_series["s1"][:3]}) == 1
    assert training_helpers.hybrid_context_len({"short": records_by_series["s1"][:8]}) == 3
    assert training_helpers.hybrid_context_len({"mid": records_by_series["s1"][:18]}) == 7
    _, long_records_by_series = training_helpers.build_request_training_payload(
        _long_series_records()
    )
    assert training_helpers.hybrid_context_len(long_records_by_series) == 30

    supervised = training_helpers.build_hgb_supervised_rows(
        records_by_series,
        context_len=3,
        feature_keys=feature_keys,
    )

    assert len(supervised) == 30
    assert supervised[0]["series_id"] == "s1"
    assert len(supervised[0]["features"]) == 6
    assert supervised[0]["future_row"]["y"] == 4.0


def test_numerical_helper_functions_cover_split_summary_and_sigmoid() -> None:
    train_small, valid_small = training_helpers.split_train_valid_indices(12)
    assert train_small.tolist() == list(range(12))
    assert valid_small.size == 0

    train_large, valid_large = training_helpers.split_train_valid_indices(60)
    assert train_large[0] == 0
    assert len(valid_large) == 12
    assert train_large[-1] == 47

    summary = training_helpers.top_feature_summary(
        {"sensor_1": 0.25, "sensor_2": float("nan"), "sensor_3": 0.5},
        method="permutation_importance_rmse_v1",
        sample_count=12,
    )
    assert summary["top_features"][0]["feature"] == "sensor_3"
    assert "sensor_2" not in summary["feature_importance"]

    assert math.isclose(app_module._sigmoid(0.0), 0.5)
    assert math.isclose(app_module._sigmoid(1_000.0), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert 0.0 <= app_module._sigmoid(-1_000.0) < 1e-20


def test_fit_hgb_forecaster_and_train_public_entry_persist_artifacts(monkeypatch, tmp_path) -> None:
    _, records_by_series = training_helpers.build_request_training_payload(_series_records())
    artifact = training_helpers.fit_hgb_forecaster(
        records_by_series=records_by_series,
        context_len=3,
        feature_keys=["sensor_1", "sensor_2"],
    )

    assert sorted(artifact["bundle"]) == ["point", "q05", "q95"]
    assert artifact["snapshot"]["context_len"] == 3
    assert artifact["snapshot"]["feature_keys"] == ["sensor_1", "sensor_2"]
    assert len(artifact["pooled_residuals"]) > 0
    assert len(artifact["valid_rows"]) > 0

    pred, lower, upper = stable_models.predict_hgb_next(
        artifact["bundle"],
        context_records=records_by_series["s1"][-3:],
        feature_keys=["sensor_1", "sensor_2"],
    )
    assert math.isfinite(pred)
    assert lower <= upper

    monkeypatch.setattr(app_module, "_MODEL_ARTIFACTS_ROOT", tmp_path)
    req = app_module.TrainRequest(algo="gbdt_hgb_v1", training_hours=0.1, data=_series_records())
    entry = training_helpers.train_public_gbdt_entry(
        req,
        model_id="model_gbdt_test",
        model_artifact_dir=app_module._model_artifact_dir,
        write_json=app_module._write_json,
        artifact_relpath=app_module._artifact_relpath,
    )

    snapshot_path = Path(tmp_path / "model_gbdt_test" / "snapshot.json")
    joblib_path = Path(tmp_path / "model_gbdt_test" / "gbdt.joblib")

    assert snapshot_path.exists()
    assert joblib_path.exists()
    assert entry["artifact"]["snapshot_json"] == "model_gbdt_test/snapshot.json"
    assert entry["artifact"]["gbdt_joblib"] == "model_gbdt_test/gbdt.joblib"
    loaded_bundle = app_module._load_joblib_artifact(joblib_path)
    assert sorted(loaded_bundle) == ["point", "q05", "q95"]


def test_build_job_store_reads_postgres_env(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_build_job_store(*, sqlite_db_path, backend=None, postgres_dsn=None):
        captured["sqlite_db_path"] = sqlite_db_path
        captured["backend"] = backend
        captured["postgres_dsn"] = postgres_dsn
        return object()

    monkeypatch.setattr(app_module, "_MODEL_REGISTRY_DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(app_module, "build_job_store", _fake_build_job_store)
    monkeypatch.setenv("RULFM_JOB_STORE_BACKEND", "postgres")
    monkeypatch.setenv(
        "RULFM_JOB_STORE_POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/rulfm_test",
    )

    app_module._build_job_store()

    assert captured["sqlite_db_path"] == tmp_path / "jobs.db"
    assert captured["backend"] == "postgres"
    assert captured["postgres_dsn"] == "postgresql://postgres:postgres@localhost:5432/rulfm_test"


def test_load_trained_models_returns_empty_dict_and_logs_on_failure(monkeypatch) -> None:
    calls: list[str] = []

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "load_models_from_store", _raise)
    monkeypatch.setattr(
        app_module,
        "_LOGGER",
        SimpleNamespace(exception=lambda message, **kwargs: calls.append(str(message))),
    )

    assert app_module._load_trained_models() == {}
    assert calls == ["Failed to load trained models"]


def test_require_trained_models_loads_once(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(app_module, "TRAINED_MODELS", None)
    monkeypatch.setattr(
        app_module,
        "_load_trained_models",
        lambda: calls.append("loaded") or {"m1": {"model_id": "m1"}},
    )

    first = app_module._require_trained_models()
    second = app_module._require_trained_models()

    assert first == {"m1": {"model_id": "m1"}}
    assert second is first
    assert calls == ["loaded"]


def test_require_job_store_builds_once(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(app_module, "JOB_STORE", None)
    monkeypatch.setattr(
        app_module,
        "_build_job_store",
        lambda: calls.append("built") or object(),
    )

    first = app_module._require_job_store()
    second = app_module._require_job_store()

    assert second is first
    assert calls == ["built"]


def test_expected_secret_helpers_log_and_fallback_on_secret_resolution_failure(monkeypatch) -> None:
    warnings: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "plain-api-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_BEARER_TOKEN", "plain-bearer")
    monkeypatch.setattr(
        app_module,
        "_LOGGER",
        SimpleNamespace(warning=lambda message, **kwargs: warnings.append((str(message), kwargs))),
    )

    class _BrokenSecrets:
        @staticmethod
        def cache_clear() -> None:
            return None

        def __call__(self, **kwargs):
            raise RuntimeError("kms decrypt failed")

    monkeypatch.setattr(app_module, "resolve_secret", _BrokenSecrets())

    assert app_module._expected_api_key() == "plain-api-key"
    assert app_module._expected_bearer_token() == "plain-bearer"
    assert len(warnings) == 2
    assert warnings[0][1]["extra"]["secret_kind"] == "api_key"
    assert warnings[1][1]["extra"]["secret_kind"] == "bearer_token"


def test_create_app_does_not_eagerly_build_job_store(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(app_module, "_build_job_store", lambda: calls.append("built") or object())

    app_module.create_app()

    assert calls == []


def test_lazy_app_proxy_defers_factory_until_first_access(monkeypatch) -> None:
    created: list[str] = []

    class _FakeApp:
        title = "fake-title"

    monkeypatch.setattr(
        app_module,
        "create_app",
        lambda: created.append("created") or _FakeApp(),
    )

    proxy = app_module._LazyAppProxy()

    assert created == []
    assert proxy.title == "fake-title"
    assert created == ["created"]


def test_load_fd004_benchmark_summary_logs_parse_failures(monkeypatch, tmp_path) -> None:
    summary_path = tmp_path / "fd004_benchmark_summary.json"
    summary_path.write_text("{not-json", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(app_module, "_FD004_BENCHMARK_SUMMARY_PATH", summary_path)
    monkeypatch.setattr(
        app_module,
        "_LOGGER",
        SimpleNamespace(exception=lambda message, **kwargs: calls.append(str(message))),
    )

    payload = app_module._load_fd004_benchmark_summary()

    assert payload["rows"] == []
    assert payload["notes"] == ["FD004 benchmark summary exists but could not be parsed."]
    assert calls == ["Failed to parse FD004 benchmark summary"]


def test_build_hgb_supervised_rows_sorts_missing_timestamps_last() -> None:
    rows = {
        "s1": [
            {"timestamp": "2026-03-03T00:00:00+09:00", "y": 3.0, "x": {"sensor_1": 3.0}},
            {"timestamp": None, "y": 99.0, "x": {"sensor_1": 99.0}},
            {"timestamp": "2026-03-01T00:00:00+00:00", "y": 1.0, "x": {"sensor_1": 1.0}},
            {"timestamp": "2026-03-02T00:00:00+00:00", "y": 2.0, "x": {"sensor_1": 2.0}},
        ]
    }

    supervised = training_helpers.build_hgb_supervised_rows(
        rows,
        context_len=2,
        feature_keys=["sensor_1"],
    )

    assert supervised[0]["future_row"]["y"] == 3.0


def test_ridge_lags_helpers_cover_fit_and_forecast_response_paths() -> None:
    ys = [float(day * 1.5) for day in range(1, 25)]
    state = stable_models.ridge_lags_fit_series(ys, lag_k=5)
    preds = stable_models.ridge_lags_forecast_series(state, last_values=ys[-5:], horizon=3)

    assert state["algo"] == "ridge_lags_v1"
    assert len(state["coef"]) == 5
    assert len(preds) == 3
    assert all(math.isfinite(value) for value in preds)

    evidence = stable_models.build_residuals_evidence([0.2, 0.4, 0.6, 0.8, 1.0])
    assert evidence is not None
    assert evidence["hist"]["bins"] == 20

    training_req = app_module.TrainRequest(
        training_hours=0.1,
        data=_daily_series_records("s1", 24) + _daily_series_records("s2", 24),
    )
    trained_state = training_helpers.fit_ridge_lags_model(
        training_req,
        ridge_lags_choose_k=stable_models.ridge_lags_choose_k,
        ridge_lags_fit_series=stable_models.ridge_lags_fit_series,
    )
    trained_state["pooled_residuals"] = [0.1 + 0.01 * idx for idx in range(20)]

    forecast_req = app_module.ForecastRequest(
        horizon=2,
        frequency="1d",
        quantiles=[0.1, 0.5, 0.9],
        data=_daily_series_records("s1", 24) + _daily_series_records("s2", 24),
    )
    response = stable_models.forecast_with_trained_model(
        forecast_req,
        step=timedelta(days=1),
        trained={"algo": "ridge_lags_v1", "state": trained_state},
        ridge_lags_choose_k_fn=stable_models.ridge_lags_choose_k,
        ridge_lags_fit_series_fn=stable_models.ridge_lags_fit_series,
        ridge_lags_forecast_series_fn=stable_models.ridge_lags_forecast_series,
        safe_std_fn=stable_models.safe_std,
        quantile_nearest_rank_fn=stable_models.quantile_nearest_rank,
        build_residuals_evidence_fn=stable_models.build_residuals_evidence,
        naive_forecast_fn=stable_models.naive_forecast,
    )

    assert response.calibration is not None
    assert response.calibration["method"] == "split_conformal_abs_error"
    assert len(response.forecasts) == 4
    assert response.forecasts[0].quantiles is not None
    assert response.residuals_evidence is not None


def test_ridge_lags_helpers_cover_fallback_padding_and_missing_residuals() -> None:
    state = stable_models.ridge_lags_fit_series([1.0, float("nan"), 3.0], lag_k=5)
    preds = stable_models.ridge_lags_forecast_series(
        {"lag_k": 3, "coef": [1.0], "intercept": 2.0},
        last_values=[10.0],
        horizon=2,
    )

    assert state["algo"] == "naive_last_value"
    assert state["lag_k"] == 2
    assert state["residuals"] == []
    assert preds == [12.0, 14.0]
    assert stable_models.build_residuals_evidence([]) is None


def test_ridge_lags_backtest_uses_trained_hint_and_skips_short_series(monkeypatch) -> None:
    original_fit = stable_models.ridge_lags_fit_series
    lag_ks: list[int] = []

    def _tracking_fit(ys: list[float], *, lag_k: int):
        lag_ks.append(lag_k)
        return original_fit(ys, lag_k=lag_k)

    monkeypatch.setattr(stable_models, "ridge_lags_fit_series", _tracking_fit)

    data = _daily_series_records("hinted", 10) + _daily_series_records("short", 4)
    response = stable_models.ridge_lags_backtest(
        app_module.BacktestRequest(horizon=2, folds=3, metric="rmse", data=data),
        trained={"state": {"series": {"hinted": {"lag_k": "4"}}}},
        ridge_lags_choose_k_fn=stable_models.ridge_lags_choose_k,
        ridge_lags_fit_series_fn=stable_models.ridge_lags_fit_series,
        ridge_lags_forecast_series_fn=stable_models.ridge_lags_forecast_series,
        metric_value_fn=stable_models.metric_value,
    )

    assert lag_ks == [4, 4, 4]
    assert response.metrics["rmse"] >= 0.0
    assert response.by_series is not None
    assert [entry["series_id"] for entry in response.by_series] == ["hinted"]
    assert len(response.by_horizon or []) == 2
    assert len(response.by_fold or []) == 3


def test_ridge_lags_backtest_handles_invalid_hint_and_stops_when_train_too_short(monkeypatch) -> None:
    original_fit = stable_models.ridge_lags_fit_series
    lag_ks: list[int] = []

    def _tracking_fit(ys: list[float], *, lag_k: int):
        lag_ks.append(lag_k)
        return original_fit(ys, lag_k=lag_k)

    monkeypatch.setattr(stable_models, "ridge_lags_fit_series", _tracking_fit)

    response = stable_models.ridge_lags_backtest(
        app_module.BacktestRequest(
            horizon=2,
            folds=3,
            metric="mae",
            data=_daily_series_records("default-hint", 6),
        ),
        trained={"series": {"default-hint": {"lag_k": object()}}},
        ridge_lags_choose_k_fn=stable_models.ridge_lags_choose_k,
        ridge_lags_fit_series_fn=stable_models.ridge_lags_fit_series,
        ridge_lags_forecast_series_fn=stable_models.ridge_lags_forecast_series,
        metric_value_fn=stable_models.metric_value,
    )

    assert lag_ks == [2]
    assert response.metrics["mae"] >= 0.0
    assert response.by_series is not None
    assert len(response.by_fold or []) == 1


def test_naive_helpers_cover_forecast_series_and_backtest_paths() -> None:
    seasonal_preds, seasonal_scale = stable_models.forecast_series_values(
        [float(day) for day in range(1, 22)],
        horizon=3,
        step=timedelta(days=1),
    )
    trend_preds, trend_scale = stable_models.forecast_series_values(
        [10.0, 12.0, 14.0, 16.0],
        horizon=2,
        step=timedelta(hours=1),
    )
    flat_preds, flat_scale = stable_models.forecast_series_values(
        [7.0],
        horizon=2,
        step=timedelta(hours=1),
    )

    assert len(seasonal_preds) == 3
    assert seasonal_scale >= 0.0
    assert trend_preds[0] > 16.0
    assert trend_scale >= 0.0
    assert flat_preds == [7.0, 7.0]
    assert flat_scale == 0.0

    data = _daily_series_records("s1", 20) + _daily_series_records("s2", 20)
    forecast_req = app_module.ForecastRequest(
        horizon=3,
        frequency="1d",
        level=[80, 90],
        data=data,
    )
    forecast_response = stable_models.naive_forecast(forecast_req, timedelta(days=1))

    assert forecast_response.calibration is not None
    assert forecast_response.calibration["qhat_by_level"]
    assert forecast_response.forecasts[0].intervals is not None
    assert forecast_response.residuals_evidence is not None

    backtest_response = app_module._naive_backtest(
        app_module.BacktestRequest(horizon=2, folds=3, metric="rmse", data=data)
    )

    assert backtest_response.metrics["rmse"] >= 0.0
    assert backtest_response.by_series is not None
    assert len(backtest_response.by_horizon or []) == 2
    assert len(backtest_response.by_fold or []) == 3


def test_naive_helpers_cover_low_history_quantiles_and_short_backtest_paths() -> None:
    empty_preds, empty_scale = stable_models.forecast_series_values(
        [],
        horizon=3,
        step=timedelta(hours=1),
    )
    assert empty_preds == [0.0, 0.0, 0.0]
    assert empty_scale == 0.0

    sparse_data = _daily_series_records("s1", 4) + _daily_series_records("s2", 4)
    quantile_req = app_module.ForecastRequest(
        horizon=2,
        frequency="1d",
        quantiles=[-1.0, 0.1, 0.5, 0.9, 1.5],
        data=sparse_data,
    )

    forecast_response = stable_models.naive_forecast(quantile_req, timedelta(days=1))

    assert forecast_response.calibration is None
    assert forecast_response.warnings is not None
    assert "CALIB01" in forecast_response.warnings[0]
    assert forecast_response.forecasts[0].quantiles is not None
    assert sorted(forecast_response.forecasts[0].quantiles or {}) == ["0.1", "0.5", "0.9"]
    assert forecast_response.forecasts[0].intervals is None

    short_backtest = app_module._naive_backtest(
        app_module.BacktestRequest(
            horizon=3,
            folds=4,
            metric="mae",
            data=_daily_series_records("short", 4),
        )
    )

    assert short_backtest.metrics["mae"] == 0.0
    assert short_backtest.by_series is None
    assert len(short_backtest.by_horizon or []) == 3
    assert short_backtest.by_fold is None


def test_metric_value_covers_zero_denominator_and_unknown_metric_paths() -> None:
    assert stable_models.metric_value(
        "mape",
        y_true=[0.0, 0.0],
        y_pred=[1.0, 2.0],
        train_y=[1.0, 2.0],
    ) == 0.0
    assert stable_models.metric_value(
        "smape",
        y_true=[0.0],
        y_pred=[0.0],
        train_y=[1.0, 2.0],
    ) == 0.0
    assert stable_models.metric_value(
        "wape",
        y_true=[0.0, 0.0],
        y_pred=[1.0, 2.0],
        train_y=[1.0, 2.0],
    ) == 0.0
    assert stable_models.metric_value(
        "mase",
        y_true=[1.0, 2.0],
        y_pred=[1.5, 2.5],
        train_y=[5.0],
    ) == 0.0
    assert stable_models.metric_value(  # type: ignore[arg-type]
        "unknown",
        y_true=[1.0],
        y_pred=[1.0],
        train_y=[1.0, 2.0],
    ) == 0.0


def test_config_helpers_cover_auth_oidc_and_limits(monkeypatch) -> None:
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "plain-api-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_BEARER_TOKEN", "plain-bearer")
    monkeypatch.setenv("RULFM_FORECASTING_API_REQUIRE_TLS", "1")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_ISSUER", "https://issuer.example")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_AUDIENCE", "rulfm-audience")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_JWKS_URL", "https://issuer.example/jwks")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_ALGORITHMS", "RS256,ES256")
    monkeypatch.setenv("RULFM_FORECASTING_API_MAX_BODY_BYTES", "2048")
    monkeypatch.setenv("RULFM_FORECASTING_API_SYNC_MAX_POINTS", "250")
    monkeypatch.setenv("RULFM_FORECASTING_API_AUDIT_LOG_ENABLED", "0")
    monkeypatch.setenv("RULFM_FORECASTING_API_RATE_LIMIT_ENABLED", "1")
    monkeypatch.setenv("RULFM_FORECASTING_API_RATE_LIMIT_PER_WINDOW", "17")
    monkeypatch.setenv("RULFM_FORECASTING_API_RATE_LIMIT_WINDOW_SECONDS", "30")
    monkeypatch.setattr(app_module, "resolve_secret", SimpleNamespace(cache_clear=lambda: None))

    assert app_module._expected_api_key() == "plain-api-key"
    assert app_module._expected_bearer_token() == "plain-bearer"
    assert app_module._require_tls() is True
    assert app_module._oidc_issuer() == "https://issuer.example"
    assert app_module._oidc_audience() == "rulfm-audience"
    assert app_module._oidc_jwks_url() == "https://issuer.example/jwks"
    assert app_module._oidc_algorithms() == ["RS256", "ES256"]
    assert app_module._oidc_enabled() is True
    assert app_module._max_body_bytes() == 2048
    assert app_module._sync_max_points() == 250
    assert app_module._audit_log_enabled() is False
    assert app_module._rate_limit_enabled() is True
    assert app_module._rate_limit_per_window() == 17
    assert app_module._rate_limit_window_seconds() == 30


def test_metric_value_covers_all_public_metrics() -> None:
    y_true = [10.0, 20.0, 40.0]
    y_pred = [12.0, 18.0, 38.0]
    train_y = [8.0, 9.0, 11.0, 15.0]

    assert stable_models.metric_value("mae", y_true=y_true, y_pred=y_pred, train_y=train_y) > 0.0
    assert stable_models.metric_value("rmse", y_true=y_true, y_pred=y_pred, train_y=train_y) > 0.0
    assert (
        stable_models.metric_value("nasa_score", y_true=y_true, y_pred=y_pred, train_y=train_y)
        > 0.0
    )
    assert stable_models.metric_value("mape", y_true=y_true, y_pred=y_pred, train_y=train_y) > 0.0
    assert stable_models.metric_value("smape", y_true=y_true, y_pred=y_pred, train_y=train_y) > 0.0
    assert stable_models.metric_value("wape", y_true=y_true, y_pred=y_pred, train_y=train_y) > 0.0
    assert stable_models.metric_value("mase", y_true=y_true, y_pred=y_pred, train_y=train_y) > 0.0
    assert stable_models.metric_value("mae", y_true=[], y_pred=[], train_y=train_y) == 0.0


def test_gbdt_helpers_cover_forecast_and_backtest_paths(monkeypatch) -> None:
    predict_calls: list[dict[str, object]] = []

    def _fake_predict(bundle, *, context_records, feature_keys):
        predict_calls.append(
            {
                "context_len": len(context_records),
                "feature_keys": list(feature_keys),
                "last_y": float(context_records[-1]["y"]),
            }
        )
        base = float(len(predict_calls) + 10)
        return base, base - 1.5, base + 1.5

    monkeypatch.setattr(
        app_module,
        "_read_json",
        lambda path: {"context_len": 3, "feature_keys": ["sensor_1", "sensor_2"]},
    )
    monkeypatch.setattr(app_module, "_load_joblib_artifact", lambda path: {"bundle": "ok"})
    monkeypatch.setattr(stable_models, "predict_hgb_next", _fake_predict)

    forecast_response = stable_models.forecast_with_gbdt_model(
        app_module.ForecastRequest(
            horizon=2,
            frequency="1d",
            quantiles=[0.1, 0.5, 0.9],
            level=[80],
            model_id="model_gbdt",
            data=_daily_series_records("gbdt-forecast", 2),
        ),
        step=timedelta(days=1),
        trained={
            "context_len": 3,
            "pooled_residuals": [0.2 + (0.01 * idx) for idx in range(20)],
            "artifact": {
                "snapshot_json": "model_gbdt/snapshot.json",
                "gbdt_joblib": "model_gbdt/gbdt.joblib",
            },
        },
        read_json=app_module._read_json,
        artifact_abspath=app_module._artifact_abspath,
        load_joblib_artifact=app_module._load_joblib_artifact,
        predict_hgb_next_fn=stable_models.predict_hgb_next,
        quantile_nearest_rank_fn=stable_models.quantile_nearest_rank,
        build_residuals_evidence_fn=stable_models.build_residuals_evidence,
    )

    assert forecast_response.calibration is not None
    assert forecast_response.calibration["method"] == "split_conformal_abs_error"
    assert len(forecast_response.forecasts) == 2
    assert forecast_response.forecasts[0].quantiles is not None
    assert forecast_response.forecasts[0].intervals is not None
    assert all(call["context_len"] == 3 for call in predict_calls[:2])
    assert predict_calls[0]["feature_keys"] == ["sensor_1", "sensor_2"]

    predict_calls.clear()
    backtest_data = _daily_series_records("gbdt-long", 7) + _daily_series_records("gbdt-short", 4)
    backtest_response = stable_models.gbdt_backtest(
        app_module.BacktestRequest(
            horizon=2,
            folds=2,
            metric="rmse",
            model_id="model_gbdt",
            data=backtest_data,
        ),
        trained={
            "context_len": 5,
            "artifact": {
                "snapshot_json": "model_gbdt/snapshot.json",
                "gbdt_joblib": "model_gbdt/gbdt.joblib",
            },
        },
        read_json=app_module._read_json,
        artifact_abspath=app_module._artifact_abspath,
        load_joblib_artifact=app_module._load_joblib_artifact,
        predict_hgb_next_fn=stable_models.predict_hgb_next,
        metric_value_fn=stable_models.metric_value,
    )

    assert backtest_response.metrics["rmse"] >= 0.0
    assert backtest_response.by_series is not None
    assert [entry["series_id"] for entry in (backtest_response.by_series or [])] == ["gbdt-long"]
    assert len(backtest_response.by_horizon or []) == 2
    assert len(backtest_response.by_fold or []) == 2
    assert all(call["context_len"] == 5 for call in predict_calls)


@pytest.mark.experimental
def test_torch_backtest_covers_padding_and_fold_paths(monkeypatch) -> None:
    from forecasting_api import torch_forecasters as torch_module

    forecast_calls: list[dict[str, object]] = []

    def _fake_forecast(**kwargs):
        forecast_calls.append(
            {
                "context": list(kwargs["context"]),
                "future_rows": list(kwargs["future_feature_rows"]),
                "algo": kwargs["algo"],
            }
        )
        horizon = int(kwargs["horizon"])
        context = [float(value) for value in kwargs["context"]]
        base = float(sum(context) / max(len(context), 1))
        return [base + float(idx + 1) for idx in range(horizon)]

    monkeypatch.setattr(app_module, "_read_json", lambda path: {"context_len": 4})
    monkeypatch.setattr(app_module, "_try_torch_load_weights", lambda path: {"state_dict": {}})
    monkeypatch.setattr(app_module, "_extract_state_dict", lambda ckpt: {"ok": True})
    monkeypatch.setattr(torch_module, "forecast_univariate_torch", _fake_forecast)

    response = app_module._torch_backtest(
        app_module.BacktestRequest(
            horizon=2,
            folds=2,
            metric="rmse",
            model_id="model_torch",
            data=_daily_series_records("torch-long", 7) + _daily_series_records("torch-short", 4),
        ),
        trained={
            "algo": "afnocg3_v1",
            "context_len": 5,
            "artifact": {
                "snapshot_json": "model_torch/snapshot.json",
                "weights_pt": "model_torch/weights.pt",
            },
        },
    )

    assert response.metrics["rmse"] >= 0.0
    assert response.by_series is not None
    assert [entry["series_id"] for entry in (response.by_series or [])] == ["torch-long"]
    assert len(response.by_horizon or []) == 2
    assert len(response.by_fold or []) == 2
    assert forecast_calls
    assert all(len(call["context"]) == 5 for call in forecast_calls)
    assert all(call["algo"] == "afnocg3_v1" for call in forecast_calls)


@pytest.mark.experimental
def test_hybrid_backtest_covers_gate_and_padding_paths(monkeypatch) -> None:
    from forecasting_api import torch_forecasters as torch_module

    detail_calls: list[dict[str, object]] = []
    predict_calls: list[int] = []

    def _fake_read_json(path):
        path_str = str(path)
        if path_str.endswith("hybrid.json"):
            return {
                "feature_keys": ["sensor_1", "sensor_2"],
                "gate": {"interval_scale": 1.2},
            }
        return {"context_len": 3}

    def _fake_predict(bundle, *, context_records, feature_keys):
        predict_calls.append(len(context_records))
        base = float(20 + len(predict_calls))
        return base, base - 2.0, base + 2.0

    def _fake_details(**kwargs):
        detail_calls.append({"context_len": len(kwargs["context_records"]), "horizon": kwargs["horizon"]})
        horizon = int(kwargs["horizon"])
        return {
            "point": [40.0 + float(idx) for idx in range(horizon)],
            "mc_dropout": {"per_step_var": [0.25 + (0.05 * idx) for idx in range(horizon)]},
            "occlusion": {"per_step": [{"top_features": [{"feature": "sensor_1", "score": 0.9}]} for _ in range(horizon)]},
        }

    monkeypatch.setattr(app_module, "_read_json", _fake_read_json)
    monkeypatch.setattr(app_module, "_try_torch_load_weights", lambda path: {"state_dict": {}})
    monkeypatch.setattr(app_module, "_extract_state_dict", lambda ckpt: {"ok": True})
    monkeypatch.setattr(app_module, "_load_joblib_artifact", lambda path: {"bundle": "ok"})
    monkeypatch.setattr(stable_models, "predict_hgb_next", _fake_predict)
    monkeypatch.setattr(
        training_helpers,
        "hybrid_gate_step_payload",
        lambda **kwargs: {
            "afno_weight": 0.4,
            "gbdt_weight": 0.6,
            "score": 0.2,
            "term_delta": 0.1,
            "term_overlap": 0.2,
            "term_width": -0.1,
            "term_tail": 0.0,
            "term_condition": 0.3,
        },
    )
    monkeypatch.setattr(torch_module, "forecast_univariate_torch_with_details", _fake_details)

    response = hybrid_runtime.hybrid_backtest(
        app_module.BacktestRequest(
            horizon=2,
            folds=2,
            metric="rmse",
            model_id="model_hybrid",
            data=_daily_series_records("hybrid-long", 7) + _daily_series_records("hybrid-short", 4),
        ),
        trained={
            "context_len": 5,
            "artifact": {
                "snapshot_json": "model_hybrid/snapshot.json",
                "weights_pt": "model_hybrid/weights.pt",
                "gbdt_joblib": "model_hybrid/gbdt.joblib",
                "hybrid_json": "model_hybrid/hybrid.json",
            },
        },
        read_json=app_module._read_json,
        artifact_abspath=app_module._artifact_abspath,
        try_torch_load_weights=app_module._try_torch_load_weights,
        extract_state_dict=app_module._extract_state_dict,
        load_joblib_artifact=app_module._load_joblib_artifact,
        predict_hgb_next_fn=stable_models.predict_hgb_next,
        gate_step_payload_fn=training_helpers.hybrid_gate_step_payload,
        hybrid_condition_cluster_key_fn=training_helpers.hybrid_condition_cluster_key,
        metric_value_fn=stable_models.metric_value,
    )

    assert response.metrics["rmse"] >= 0.0
    assert response.by_series is not None
    assert [entry["series_id"] for entry in (response.by_series or [])] == ["hybrid-long"]
    assert len(response.by_horizon or []) == 2
    assert len(response.by_fold or []) == 2
    assert detail_calls
    assert all(call["context_len"] == 5 for call in detail_calls)
    assert predict_calls
    assert all(length == 5 for length in predict_calls)