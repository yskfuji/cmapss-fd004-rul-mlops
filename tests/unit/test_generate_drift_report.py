from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import httpx
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_drift_report.py"
_SPEC = importlib.util.spec_from_file_location("generate_drift_report", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
post_with_retry = _MODULE.post_with_retry


def test_post_with_retry_retries_before_success(monkeypatch):
    calls = {"count": 0}
    response_holder: dict[str, httpx.Response] = {}

    def _post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary failure")
        request = httpx.Request(
            "POST",
            "http://localhost:8000/v1/monitoring/drift/report",
            headers={"X-API-Key": "secret"},
            json={"candidate_records": []},
        )
        response = httpx.Response(200, request=request, json={"severity": "low"})
        response_holder["response"] = response
        return response

    monkeypatch.setattr(_MODULE.time, "sleep", lambda *_args: None)
    result = post_with_retry(
        httpx_module=types.SimpleNamespace(post=_post),
        api_url="http://localhost:8000",
        api_key="secret",
        payload={"candidate_records": []},
    )
    assert result == {"severity": "low"}
    assert calls["count"] == 3
    assert response_holder["response"].request.headers["X-API-Key"] == "secret"


def test_post_with_retry_fails_fast_on_non_retryable_4xx(monkeypatch):
    calls = {"count": 0}

    def _post(*args, **kwargs):
        calls["count"] += 1
        request = httpx.Request("POST", "http://localhost:8000/v1/monitoring/drift/report")
        return httpx.Response(422, request=request, json={"detail": "bad payload"})

    monkeypatch.setattr(_MODULE.time, "sleep", lambda *_args: None)
    with pytest.raises(RuntimeError, match="non-retryable status 422"):
        post_with_retry(
            httpx_module=types.SimpleNamespace(post=_post),
            api_url="http://localhost:8000",
            api_key="secret",
            payload={"candidate_records": []},
        )

    assert calls["count"] == 1
