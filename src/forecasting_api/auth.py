from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from typing import Any

from fastapi import Request


def expected_api_key(
    *,
    resolve_secret: Any,
    logger: Any,
    env_first: Callable[..., str | None],
) -> str | None:
    token: str | None = None
    try:
        resolve_secret.cache_clear()
        token = resolve_secret(
            plain_env="RULFM_FORECASTING_API_KEY",
            encrypted_env="RULFM_FORECASTING_API_KEY_ENCRYPTED_B64",
        )
        if token:
            return token
    except Exception as exc:
        logger.warning(
            "Falling back to plain API key after encrypted secret resolution failed",
            extra={
                "event_type": "SECRET_RESOLUTION_FALLBACK",
                "secret_kind": "api_key",
                "error": str(exc),
            },
        )
    return env_first("RULFM_FORECASTING_API_KEY")


def expected_bearer_token(
    *,
    resolve_secret: Any,
    logger: Any,
    env_first: Callable[..., str | None],
    expected_api_key_fn: Callable[[], str | None],
) -> str | None:
    token: str | None = None
    try:
        resolve_secret.cache_clear()
        token = resolve_secret(
            plain_env="RULFM_FORECASTING_API_BEARER_TOKEN",
            encrypted_env="RULFM_FORECASTING_API_BEARER_TOKEN_ENCRYPTED_B64",
        )
    except Exception as exc:
        logger.warning(
            "Falling back to plain bearer token after encrypted secret resolution failed",
            extra={
                "event_type": "SECRET_RESOLUTION_FALLBACK",
                "secret_kind": "bearer_token",
                "error": str(exc),
            },
        )
        token = env_first("RULFM_FORECASTING_API_BEARER_TOKEN")
    if token:
        return token
    return expected_api_key_fn()


def oidc_issuer() -> str | None:
    value = os.getenv("RULFM_FORECASTING_API_OIDC_ISSUER", "").strip()
    return value or None


def oidc_audience() -> str | None:
    value = os.getenv("RULFM_FORECASTING_API_OIDC_AUDIENCE", "").strip()
    return value or None


def oidc_jwks_url() -> str | None:
    value = os.getenv("RULFM_FORECASTING_API_OIDC_JWKS_URL", "").strip()
    return value or None


def oidc_algorithms() -> list[str]:
    raw = os.getenv("RULFM_FORECASTING_API_OIDC_ALGORITHMS", "RS256,ES256,PS256")
    algos = [token.strip() for token in raw.split(",") if token.strip()]
    return algos or ["RS256"]


def oidc_enabled() -> bool:
    return oidc_issuer() is not None and oidc_audience() is not None and oidc_jwks_url() is not None


def validate_oidc_bearer_token(token: str) -> dict[str, Any]:
    import jwt

    jwks_client = jwt.PyJWKClient(str(oidc_jwks_url()))
    signing_key = jwks_client.get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=oidc_algorithms(),
        audience=str(oidc_audience()),
        issuer=str(oidc_issuer()),
        options={"require": ["exp", "iat", "iss", "aud"]},
    )
    return claims if isinstance(claims, dict) else {}


def require_api_key(
    *,
    request: Request,
    x_api_key: str | None,
    authorization: str | None,
    api_error_cls: type[Exception],
    expected_api_key_fn: Callable[[], str | None],
    expected_bearer_token_fn: Callable[[], str | None],
    oidc_enabled_fn: Callable[[], bool],
    validate_oidc_bearer_token_fn: Callable[[str], dict[str, Any]],
    logger: Any,
) -> None:
    expected = expected_api_key_fn()
    bearer_expected = expected_bearer_token_fn()
    oidc_ready = oidc_enabled_fn()
    if expected is None and bearer_expected is None and not oidc_ready:
        raise api_error_cls(
            status_code=401,
            error_code="A12",
            message="APIキーが無効です",
            details={
                "next_action": (
                    "RULFM_FORECASTING_API_KEY または "
                    "RULFM_FORECASTING_API_BEARER_TOKEN または "
                    "OIDC 環境変数を設定してください"
                )
            },
        )

    bearer_token: str | None = None
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.strip().lower() == "bearer":
            bearer_token = token.strip() or None

    if (
        expected is not None
        and x_api_key is not None
        and secrets.compare_digest(str(x_api_key), str(expected))
    ):
        request.state.auth_method = "x-api-key"
        request.state.auth_claims = {}
        return

    if (
        bearer_expected is not None
        and bearer_token is not None
        and secrets.compare_digest(str(bearer_token), str(bearer_expected))
    ):
        request.state.auth_method = "bearer"
        request.state.auth_claims = {}
        return

    if oidc_ready and bearer_token is not None:
        try:
            claims = validate_oidc_bearer_token_fn(str(bearer_token))
            request.state.auth_method = "oidc-bearer"
            request.state.auth_subject = str(claims.get("sub") or "")
            request.state.auth_claims = dict(claims)
            return
        except Exception:
            logger.warning("OIDC bearer token validation failed", exc_info=True)

    request.state.auth_method = "invalid"
    request.state.auth_claims = {}
    raise api_error_cls(
        status_code=401,
        error_code="A12",
        message="APIキーが無効です",
        details={
            "next_action": (
                "X-API-Key または Authorization: Bearer または "
                "OIDC 設定を確認してください"
            )
        },
    )


def require_api_access(
    *,
    request: Request,
    x_api_key: str | None,
    authorization: str | None,
    x_tenant_id: str | None,
    x_connection_type: str | None,
    require_api_key_fn: Callable[..., None],
    enforce_request_policy_fn: Callable[..., Any],
) -> None:
    require_api_key_fn(
        request,
        x_api_key=x_api_key,
        authorization=authorization,
    )
    enforce_request_policy_fn(
        request,
        tenant_id=x_tenant_id,
        connection_type=x_connection_type,
    )


def require_train_access(
    *,
    request: Request,
    x_api_key: str | None,
    authorization: str | None,
    x_tenant_id: str | None,
    x_connection_type: str | None,
    x_approved_by: str | None,
    x_approval_reason: str | None,
    require_api_access_fn: Callable[..., None],
    enforce_train_request_approval_fn: Callable[..., Any],
) -> None:
    require_api_access_fn(
        request,
        x_api_key=x_api_key,
        authorization=authorization,
        x_tenant_id=x_tenant_id,
        x_connection_type=x_connection_type,
    )
    enforce_train_request_approval_fn(
        request,
        approved_by=x_approved_by,
        approval_reason=x_approval_reason,
    )