"""Reader-only auth/env facade for app.py.

This module centralizes env-driven auth and request-setting readers, while
app.py keeps the thin monkeypatch-compatible request entry wrappers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import app_bootstrap
from . import auth as auth_helpers


@dataclass(frozen=True)
class AppAuthHelpers:
    expected_api_key: Callable[[], str | None]
    expected_bearer_token: Callable[[], str | None]
    require_tls: Callable[[], bool]
    oidc_issuer: Callable[[], str | None]
    oidc_audience: Callable[[], str | None]
    oidc_jwks_url: Callable[[], str | None]
    oidc_algorithms: Callable[[], list[str]]
    oidc_enabled: Callable[[], bool]
    validate_oidc_bearer_token: Callable[[str], dict[str, Any]]
    max_body_bytes: Callable[[], int]
    sync_max_points: Callable[[], int]
    job_store_backend: Callable[[], str]
    job_store_postgres_dsn: Callable[[], str | None]
    model_registry_backend: Callable[[], str]
    model_registry_postgres_dsn: Callable[[], str | None]
    job_execution_backend: Callable[[], str]
    audit_log_path: Callable[[], Path]
    audit_log_enabled: Callable[[], bool]
    rate_limit_enabled: Callable[[], bool]
    rate_limit_per_window: Callable[[], int]
    rate_limit_window_seconds: Callable[[], int]


def build_app_auth_helpers(
    *,
    resolve_secret_getter: Callable[[], Any],
    logger_getter: Callable[[], Any],
    env_first_fn: Callable[..., str | None],
    env_bool_fn: Callable[..., bool],
    env_int_fn: Callable[..., int],
    request_audit_log_path_getter: Callable[[], Path],
) -> AppAuthHelpers:
    def _expected_api_key() -> str | None:
        return auth_helpers.expected_api_key(
            resolve_secret=resolve_secret_getter(),
            logger=logger_getter(),
            env_first=env_first_fn,
        )

    def _expected_bearer_token() -> str | None:
        return auth_helpers.expected_bearer_token(
            resolve_secret=resolve_secret_getter(),
            logger=logger_getter(),
            env_first=env_first_fn,
            expected_api_key_fn=_expected_api_key,
        )

    def _require_tls() -> bool:
        return env_bool_fn(
            "RULFM_FORECASTING_API_REQUIRE_TLS",
            default=False,
        )

    def _max_body_bytes() -> int:
        return env_int_fn(
            "RULFM_FORECASTING_API_MAX_BODY_BYTES",
            default=1_000_000,
            min_value=1_024,
        )

    def _sync_max_points() -> int:
        return env_int_fn(
            "RULFM_FORECASTING_API_SYNC_MAX_POINTS",
            default=10_000,
            min_value=100,
        )

    def _job_store_backend() -> str:
        return (env_first_fn("RULFM_JOB_STORE_BACKEND") or "sqlite").strip().lower()

    def _job_store_postgres_dsn() -> str | None:
        return env_first_fn("RULFM_JOB_STORE_POSTGRES_DSN")

    def _model_registry_backend() -> str:
        return (env_first_fn("RULFM_MODEL_REGISTRY_BACKEND") or "sqlite").strip().lower()

    def _model_registry_postgres_dsn() -> str | None:
        return env_first_fn("RULFM_MODEL_REGISTRY_POSTGRES_DSN")

    def _job_execution_backend() -> str:
        return (env_first_fn("RULFM_JOB_EXECUTION_BACKEND") or "worker").strip().lower()

    def _audit_log_path() -> Path:
        return app_bootstrap.audit_log_path(request_audit_log_path_getter())

    def _audit_log_enabled() -> bool:
        return env_bool_fn(
            "RULFM_FORECASTING_API_AUDIT_LOG_ENABLED",
            default=True,
        )

    def _rate_limit_enabled() -> bool:
        return env_bool_fn(
            "RULFM_FORECASTING_API_RATE_LIMIT_ENABLED",
            default=True,
        )

    def _rate_limit_per_window() -> int:
        raw = env_first_fn(
            "RULFM_FORECASTING_API_RATE_LIMIT_PER_WINDOW",
        ) or env_first_fn(
            "RULFM_FORECASTING_API_RATE_LIMIT_PER_MINUTE",
        ) or "120"
        try:
            value = int(raw)
        except ValueError:
            value = 120
        return max(1, min(50_000, value))

    def _rate_limit_window_seconds() -> int:
        return env_int_fn(
            "RULFM_FORECASTING_API_RATE_LIMIT_WINDOW_SECONDS",
            default=60,
            min_value=1,
            max_value=3600,
        )

    return AppAuthHelpers(
        expected_api_key=_expected_api_key,
        expected_bearer_token=_expected_bearer_token,
        require_tls=_require_tls,
        oidc_issuer=auth_helpers.oidc_issuer,
        oidc_audience=auth_helpers.oidc_audience,
        oidc_jwks_url=auth_helpers.oidc_jwks_url,
        oidc_algorithms=auth_helpers.oidc_algorithms,
        oidc_enabled=auth_helpers.oidc_enabled,
        validate_oidc_bearer_token=auth_helpers.validate_oidc_bearer_token,
        max_body_bytes=_max_body_bytes,
        sync_max_points=_sync_max_points,
        job_store_backend=_job_store_backend,
        job_store_postgres_dsn=_job_store_postgres_dsn,
        model_registry_backend=_model_registry_backend,
        model_registry_postgres_dsn=_model_registry_postgres_dsn,
        job_execution_backend=_job_execution_backend,
        audit_log_path=_audit_log_path,
        audit_log_enabled=_audit_log_enabled,
        rate_limit_enabled=_rate_limit_enabled,
        rate_limit_per_window=_rate_limit_per_window,
        rate_limit_window_seconds=_rate_limit_window_seconds,
    )