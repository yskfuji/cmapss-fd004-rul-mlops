from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request


def register_request_context_middleware(
    app: FastAPI,
    *,
    api_error_cls: type[Exception],
    track_request: Any,
    error_json: Any,
    get_request_id: Any,
    append_request_audit_log: Any,
    apply_standard_security_headers: Any,
    logger: Any,
    require_tls: Any,
    is_https_request: Any,
    max_body_bytes: Any,
    rate_limit_enabled: Any,
    consume_rate_limit: Any,
    rate_limit_window_seconds: Any,
    rate_limit_per_window: Any,
) -> None:
    @app.middleware("http")
    async def request_id_and_body_limit(request: Request, call_next):
        with track_request(request.method, request.url.path) as metric_status:
            request.state.request_id = uuid.uuid4().hex
            request.state.auth_method = "none"
            request.state.auth_subject = None
            request.state.auth_claims = {}
            request.state.request_body_bytes = 0
            request.state.tenant_id = "public"

            if (
                require_tls()
                and (request.url.path.startswith("/v1/") or request.url.path == "/health")
                and not is_https_request(request)
            ):
                response = error_json(
                    request,
                    api_error_cls(
                        status_code=400,
                        error_code="A13",
                        message="TLS が必須です",
                        details={
                            "next_action": "HTTPS または x-forwarded-proto=https で接続してください"
                        },
                    ),
                )
                metric_status[0] = response.status_code
                return response

            max_bytes = max_body_bytes()
            if request.method in {"POST", "PUT", "PATCH"}:
                body = await request.body()
                request.state.request_body_bytes = len(body)
                if len(body) > max_bytes:
                    response = error_json(
                        request,
                        api_error_cls(
                            status_code=413,
                            error_code="S01",
                            message="payload が大きすぎます",
                            details={
                                "max_bytes": max_bytes,
                                "next_action": "payload を縮小してください",
                            },
                        ),
                    )
                    metric_status[0] = response.status_code
                    return response
                request._body = body  # type: ignore[attr-defined]

            if rate_limit_enabled() and (
                request.url.path.startswith("/v1/") or request.url.path == "/metrics"
            ):
                allowed, retry_after = consume_rate_limit(request)
                if not allowed:
                    response = error_json(
                        request,
                        api_error_cls(
                            status_code=429,
                            error_code="R01",
                            message="リクエスト数が上限を超えました",
                            details={
                                "window_seconds": rate_limit_window_seconds(),
                                "limit": rate_limit_per_window(),
                                "next_action": "しばらく待ってから再試行してください",
                            },
                        ),
                    )
                    response.headers["Retry-After"] = str(retry_after)
                    metric_status[0] = response.status_code
                    return response

            response = await call_next(request)
            metric_status[0] = int(response.status_code)
            rid = get_request_id(request)
            if rid is not None:
                response.headers["X-Request-Id"] = rid
            apply_standard_security_headers(
                response,
                request,
                is_https=is_https_request(request),
            )

            try:
                append_request_audit_log(
                    {
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "request_id": rid,
                        "method": request.method,
                        "path": request.url.path,
                        "query": str(request.url.query or ""),
                        "status_code": int(response.status_code),
                        "client_host": request.client.host if request.client else None,
                        "auth_method": getattr(request.state, "auth_method", "none"),
                        "actor": getattr(request.state, "auth_subject", None),
                        "tenant_id": getattr(request.state, "tenant_id", "public"),
                        "request_body_bytes": int(
                            getattr(request.state, "request_body_bytes", 0) or 0
                        ),
                    }
                )
            except Exception:
                logger.exception("request_audit_failed", extra={"event_type": "REQUEST_AUDIT"})
            return response