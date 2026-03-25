import json
import os
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from forecasting_api import app as app_module
from forecasting_api.app import create_app
from tests.helpers import raising_callable

_MISSING = object()


@pytest.fixture(scope="module")
def public_client() -> TestClient:
    previous_api_key: object = os.environ.pop("RULFM_FORECASTING_API_KEY", _MISSING)
    with TestClient(create_app()) as client:
        yield client
    if previous_api_key is _MISSING:
        os.environ.pop("RULFM_FORECASTING_API_KEY", None)
    else:
        os.environ["RULFM_FORECASTING_API_KEY"] = str(previous_api_key)


@pytest.fixture
def api_client(api_env: None, isolated_trained_models: dict[str, dict[str, object]]) -> TestClient:
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def client_factory(
    api_env: None,
) -> Iterator[Callable[..., AbstractContextManager[TestClient]]]:
    @contextmanager
    def build(*, base_url: str = "http://testserver") -> Iterator[TestClient]:
        # Apply app_module monkeypatches before calling this factory so create_app()
        # always sees the patched globals used by the handlers under test.
        with TestClient(create_app(), base_url=base_url) as client:
            yield client

    yield build


def test_root_redirects_to_gui(public_client: TestClient) -> None:
    response = public_client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/forecasting/"


def test_docs_root_renders_language_choices(public_client: TestClient) -> None:
    response = public_client.get("/docs?lang=ja")
    assert response.status_code == 200
    assert "日本語" in response.text
    assert "English" in response.text


def test_docs_root_prefers_accept_language_header(public_client: TestClient) -> None:
    response = public_client.get("/docs", headers={"accept-language": "ja-JP,ja;q=0.9"})
    assert response.status_code == 200
    assert "推奨言語: 日本語" in response.text


def test_docs_root_falls_back_to_english_for_unknown_language(public_client: TestClient) -> None:
    response = public_client.get("/docs?lang=fr", headers={"accept-language": "fr-FR,fr;q=0.9"})
    assert response.status_code == 200
    assert "推奨言語: English" in response.text


def test_docs_language_routes_reference_localized_openapi(public_client: TestClient) -> None:
    response_en = public_client.get("/docs/en")
    response_ja = public_client.get("/docs/ja")
    assert response_en.status_code == 200
    assert response_ja.status_code == 200
    assert "lang-banner" in response_en.text
    assert "Language selector" in response_en.text
    assert "lang-banner" in response_ja.text
    assert "/openapi.en.json" in response_en.text
    assert "/openapi.ja.json" in response_ja.text


def test_openapi_language_routes_return_schema(public_client: TestClient) -> None:
    response_default = public_client.get("/openapi.json")
    response_en = public_client.get("/openapi.en.json")
    response_ja = public_client.get("/openapi.ja.json")
    assert response_default.status_code == 200
    assert response_en.status_code == 200
    assert response_ja.status_code == 200
    default_schema = response_default.json()
    en_schema = response_en.json()
    ja_schema = response_ja.json()
    assert en_schema["info"]["title"]
    assert ja_schema["info"]["title"]
    default_desc = default_schema["components"]["schemas"]["ForecastRequest"]["properties"][
        "horizon"
    ]["description"]
    en_desc = en_schema["components"]["schemas"]["ForecastRequest"]["properties"]["horizon"][
        "description"
    ]
    ja_desc = ja_schema["components"]["schemas"]["ForecastRequest"]["properties"]["horizon"][
        "description"
    ]
    assert "Forecast horizon (steps)." in default_desc
    assert "予測期間（ステップ数）。" in default_desc
    assert en_desc == "Forecast horizon (steps)."
    assert ja_desc == "予測期間（ステップ数）。"


def test_openapi_default_is_cached_per_app_instance() -> None:
    app = create_app()

    assert app.openapi_schema is None
    first = app.openapi()
    second = app.openapi()

    assert app.openapi_schema is first
    assert second is first


def test_cmapss_sample_endpoint_returns_payload_from_profile_builder(
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Callable[..., AbstractContextManager[TestClient]],
) -> None:
    monkeypatch.setattr(
        app_module,
        "build_fd004_profile_payload",
        lambda profile: {
            "profile": profile,
            "split": "train",
            "task": "forecast",
            "horizon": 12,
            "frequency": "1d",
            "quantiles": [0.1, 0.5, 0.9],
            "level": None,
            "missing_policy": "ignore",
            "chartType": "trend",
            "value_unit": "cycles",
            "records": [{"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0}],
            "meta": {"source": "stubbed"},
            "description": {"en": "stub", "ja": "stub"},
        },
    )
    with client_factory() as client:
        response = client.get(
            "/v1/cmapss/fd004/sample",
            headers={"x-api-key": "test-key"},
            params={"profile": "fd004_rul_forecast_unit01"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "fd004_rul_forecast_unit01"
    assert body["meta"]["source"] == "stubbed"


def test_cmapss_sample_endpoint_returns_v01_for_unknown_profile(api_client: TestClient) -> None:
    response = api_client.get(
        "/v1/cmapss/fd004/sample",
        headers={"x-api-key": "test-key"},
        params={"profile": "missing_profile"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "V01"
    assert "available_profiles" in body["details"]


def test_cmapss_preprocess_endpoint_returns_payload_from_builder(
    monkeypatch: pytest.MonkeyPatch,
    client_factory,
) -> None:
    monkeypatch.setattr(
        app_module,
        "build_fd004_payload",
        lambda **kwargs: {
            "split": "train",
            "task": kwargs["task"],
            "horizon": kwargs["horizon"],
            "frequency": "1d",
            "quantiles": kwargs["quantiles"],
            "level": kwargs["level"],
            "missing_policy": "ignore",
            "chartType": "trend",
            "value_unit": "cycles",
            "records": [{"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0}],
            "meta": {"window_size": kwargs["window_size"]},
        },
    )
    with client_factory() as client:
        response = client.post(
            "/v1/cmapss/fd004/preprocess",
            headers={"x-api-key": "test-key"},
            json={"split": "train", "task": "forecast", "horizon": 12, "window_size": 40},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "forecast"
    assert body["meta"]["window_size"] == 40


def test_cmapss_preprocess_endpoint_maps_missing_dataset_to_v01(
    monkeypatch: pytest.MonkeyPatch,
    client_factory,
) -> None:
    monkeypatch.setattr(
        app_module,
        "build_fd004_payload",
        raising_callable(FileNotFoundError("dataset missing")),
    )
    with client_factory() as client:
        response = client.post(
            "/v1/cmapss/fd004/preprocess",
            headers={"x-api-key": "test-key"},
            json={"split": "train", "task": "forecast", "horizon": 12, "window_size": 40},
        )
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "V01"
    assert "available_profiles" in body["details"]


def test_cmapss_benchmark_endpoint_returns_missing_snapshot_note(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    api_client: TestClient,
) -> None:
    monkeypatch.setattr(
        app_module,
        "_FD004_BENCHMARK_SUMMARY_PATH",
        tmp_path / "missing-summary.json",
    )
    response = api_client.get("/v1/cmapss/fd004/benchmarks", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["generated_at"] is None
    assert body["rows"] == []
    assert "not been generated" in body["notes"][0]


def test_cmapss_benchmark_endpoint_returns_committed_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    api_client: TestClient,
) -> None:
    summary_path = tmp_path / "fd004_benchmark_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-23T00:00:00Z",
                "rows": [{"model": "gbdt_afno_hybrid_v1", "rmse": 12.34, "coverage_90": 0.91}],
                "notes": ["stubbed benchmark"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "_FD004_BENCHMARK_SUMMARY_PATH", summary_path)
    response = api_client.get("/v1/cmapss/fd004/benchmarks", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["rows"][0]["model"] == "gbdt_afno_hybrid_v1"
    assert body["rows"][0]["rmse"] == 12.34


def test_model_catalog_endpoint_sorts_by_created_at_desc(
    monkeypatch: pytest.MonkeyPatch,
    client_factory,
) -> None:
    app_module._set_runtime_trained_models(
        {
            "m1": {"model_id": "m1", "created_at": "2026-03-22T00:00:00Z", "memo": "older"},
            "m2": {"model_id": "m2", "created_at": "2026-03-23T00:00:00Z", "memo": "newer"},
        },
    )
    with client_factory() as client:
        response = client.get("/v1/models", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert [item["model_id"] for item in body["models"]] == ["m2", "m1"]


def test_v1_response_includes_request_id_and_security_headers(
    api_client: TestClient,
) -> None:
    response = api_client.get("/v1/models", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_request_audit_log_is_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv(
        "RULFM_FORECASTING_API_AUDIT_LOG_PATH",
        str(tmp_path / "request-audit.jsonl"),
    )

    with TestClient(create_app()) as client:
        response = client.get(
            "/v1/models",
            headers={"x-api-key": "test-key", "x-tenant-id": "tenant-a"},
        )

    assert response.status_code == 200
    audit_log = tmp_path / "request-audit.jsonl"
    assert audit_log.exists()
    payload = json.loads(audit_log.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["event_type"] == "API_REQUEST_COMPLETED"
    assert payload["tenant_id"] == "tenant-a"
    assert payload["details"]["path"] == "/v1/models"
    assert payload["details"]["status_code"] == 200


def test_forecast_validation_errors_map_to_v01(api_client: TestClient) -> None:
    response = api_client.post(
        "/v1/forecast",
        headers={"x-api-key": "test-key"},
        json={"horizon": "bad", "data": []},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "V01"


def test_v1_endpoint_requires_tls_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    client_factory,
) -> None:
    monkeypatch.setenv("RULFM_FORECASTING_API_REQUIRE_TLS", "1")
    app_module._set_runtime_trained_models({})
    with client_factory(base_url="http://testserver") as client:
        response = client.get("/v1/models", headers={"x-api-key": "test-key"})
    assert response.status_code == 400
    assert response.json()["error_code"] == "A13"


def test_v1_response_sets_hsts_for_https_requests(
    monkeypatch: pytest.MonkeyPatch,
    client_factory,
) -> None:
    app_module._set_runtime_trained_models({})
    # TestClient reflects base_url scheme into request.url.scheme, which is what
    # _is_https_request() inspects before adding the HSTS header.
    with client_factory(base_url="https://testserver") as client:
        response = client.get("/v1/models", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    assert "Strict-Transport-Security" in response.headers