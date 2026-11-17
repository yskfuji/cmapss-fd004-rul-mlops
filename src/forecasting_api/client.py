from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ApiCallError(RuntimeError):
    status_code: int
    error_code: str | None
    message: str
    request_id: str | None
    details: dict[str, Any] | None


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip()
    if not base_url:
        raise ValueError("base_url must be non-empty")
    if "://" not in base_url:
        return f"http://{base_url}"
    return base_url


class ForecastingApiClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = _normalize_base_url(base_url)
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        resp = self._client.request(method, url, headers={"X-API-Key": self._api_key}, json=json_body)

        request_id = resp.headers.get("x-request-id")
        content_type = resp.headers.get("content-type", "")

        if resp.status_code >= 400:
            body: dict[str, Any] | None = None
            if "application/json" in content_type:
                try:
                    body = resp.json()
                except Exception:
                    body = None

            raise ApiCallError(
                status_code=resp.status_code,
                error_code=(body or {}).get("error_code"),
                message=(body or {}).get("message") or resp.text,
                request_id=(body or {}).get("request_id") or request_id,
                details=(body or {}).get("details") if isinstance((body or {}).get("details"), dict) else None,
            )

        if "application/json" not in content_type:
            raise ApiCallError(
                status_code=resp.status_code,
                error_code=None,
                message="Unexpected content-type",
                request_id=request_id,
                details={"content_type": content_type},
            )

        return resp.json()


def load_json_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data
