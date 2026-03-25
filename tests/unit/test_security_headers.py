from __future__ import annotations

from fastapi import Request, Response
from forecasting_api.middleware.security_headers import apply_standard_security_headers


def _request(path: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_apply_standard_security_headers_for_v1_path():
    response = Response()
    request = _request("/v1/forecast")

    apply_standard_security_headers(response, request, is_https=True)

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"
    assert "strict-transport-security" in response.headers


def test_apply_standard_security_headers_for_non_v1_path():
    response = Response()
    request = _request("/health")

    apply_standard_security_headers(response, request, is_https=False)

    assert "cache-control" not in response.headers
    assert "strict-transport-security" not in response.headers
