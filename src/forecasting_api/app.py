from __future__ import annotations

import hashlib
import math
import os
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from time import monotonic
from typing import Any, Literal, cast

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from . import app_auth_facade
from . import app_compat
from . import app_bootstrap
from . import app_docs
from . import auth as auth_helpers
from . import backtest_runtime
from . import forecast_runtime
from . import torch_runtime
from . import train_runtime
from . import training_helpers
from .cmapss_fd004 import available_profiles, build_fd004_payload, build_fd004_profile_payload
from .config import env_bool, env_int
from .deployment_guard import validate_cloud_run_runtime
from .domain import stable_models
from .errors import ApiError
from .job_dispatcher import build_inprocess_job_enqueuer, build_persistent_job_enqueuer
from .job_store import JobStore, build_job_store
from .logging_config import configure_logging, get_logger
from .metrics import render_metrics, track_request
from .middleware.security_headers import apply_standard_security_headers
from .mlflow_runs import log_artifact, log_dict_artifact, log_metrics, log_params, start_run
from .request_approval import enforce_train_request_approval
from .request_audit import append_request_audit_log as append_request_audit_log_entry
from .request_middleware import register_request_context_middleware
from .request_policy import enforce_request_policy
from .routers.backtest import build_backtest_router
from .routers.forecast import build_forecast_router
from .routers.jobs import build_jobs_router
from .routers.monitoring import build_monitoring_router
from .routers.train import build_train_router
from .schemas import (
    CmapssFd004PayloadResponse,
    CmapssFd004PreprocessRequest,
    ModelCatalogEntry,
    ModelCatalogResponse,
)
from .secrets_provider import resolve_secret


BacktestRequest = app_compat.BacktestRequest
BacktestResponse = app_compat.BacktestResponse
ErrorDetails = app_compat.ErrorDetails
ErrorResponse = app_compat.ErrorResponse
ForecastRequest = app_compat.ForecastRequest
ForecastPoint = app_compat.ForecastPoint
JobCreateRequest = app_compat.JobCreateRequest
JobCreateResponse = app_compat.JobCreateResponse
JobStatusResponse = app_compat.JobStatusResponse
TimeSeriesRecord = app_compat.TimeSeriesRecord
TrainRequest = app_compat.TrainRequest
TrainResponse = app_compat.TrainResponse


APP_COMPAT_EXPORTS = [
    "BacktestRequest",
    "BacktestResponse",
    "ErrorDetails",
    "ErrorResponse",
    "ForecastRequest",
    "ForecastPoint",
    "JobCreateRequest",
    "JobCreateResponse",
    "JobStatusResponse",
    "TimeSeriesRecord",
    "TrainRequest",
    "TrainResponse",
]

APP_RUNTIME_EXPORTS = [
    "_set_runtime_job_store",
    "_set_runtime_trained_models",
    "_sha256_hex",
    "_oidc_issuer",
    "_oidc_audience",
    "_oidc_jwks_url",
    "_oidc_algorithms",
    "create_app",
    "app",
]

__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "ErrorDetails",
    "ErrorResponse",
    "ForecastRequest",
    "ForecastPoint",
    "JobCreateRequest",
    "JobCreateResponse",
    "JobStatusResponse",
    "TimeSeriesRecord",
    "TrainRequest",
    "TrainResponse",
    "_set_runtime_job_store",
    "_set_runtime_trained_models",
    "_sha256_hex",
    "_oidc_issuer",
    "_oidc_audience",
    "_oidc_jwks_url",
    "_oidc_algorithms",
    "create_app",
    "app",
]


def _resolve_app_version() -> str:
    override = os.getenv("RULFM_APP_VERSION", "").strip()
    if override:
        return override
    try:
        return package_version("cmapss-fd004-rul-mlops")
    except PackageNotFoundError:
        return "0.0.0"


def _store_registered_route_handlers(app: FastAPI, *handlers: Any) -> None:
    app.state._registered_route_handlers = handlers


_LOGGER = get_logger("api")
_RUNTIME_PATHS = app_bootstrap.default_runtime_paths(__file__)
_DATA_DIR = _RUNTIME_PATHS.data_dir
_TRAINED_MODELS_STORE_PATH = _RUNTIME_PATHS.trained_models_store_path
_MODEL_REGISTRY_DB_PATH = _RUNTIME_PATHS.model_registry_db_path
_MODEL_ARTIFACTS_ROOT = _RUNTIME_PATHS.model_artifacts_root
_REQUEST_AUDIT_LOG_PATH = _RUNTIME_PATHS.request_audit_log_path
_DRIFT_BASELINE_PATH = _RUNTIME_PATHS.drift_baseline_path
_MODEL_PROMOTION_REGISTRY_PATH = _RUNTIME_PATHS.model_promotion_registry_path
_FD004_BENCHMARK_SUMMARY_PATH = _RUNTIME_PATHS.fd004_benchmark_summary_path
_env_first = app_compat.env_first_alias
_env_path = app_compat.env_path_alias
_bi = app_compat.bi
_as_dict = app_compat.as_dict
_as_list = app_compat.as_list
_as_float_list = app_compat.as_float_list
_sigmoid = app_compat.sigmoid
_extract_state_dict = app_compat.extract_state_dict
_try_torch_load_weights = app_compat.try_torch_load_weights
_load_joblib_artifact = app_compat.load_joblib_artifact
_COMPAT_BINDINGS = app_compat.bind_runtime_accessors(
    model_artifacts_root=lambda: _MODEL_ARTIFACTS_ROOT,
    trained_models_store_path=lambda: _TRAINED_MODELS_STORE_PATH,
    fd004_benchmark_summary_path=lambda: _FD004_BENCHMARK_SUMMARY_PATH,
    logger=lambda: _LOGGER,
)
_AUTH_HELPERS = app_auth_facade.build_app_auth_helpers(
    resolve_secret_getter=lambda: resolve_secret,
    logger_getter=lambda: _LOGGER,
    env_first_fn=_env_first,
    env_bool_fn=env_bool,
    env_int_fn=env_int,
    request_audit_log_path_getter=lambda: _REQUEST_AUDIT_LOG_PATH,
)
_model_artifact_dir = _COMPAT_BINDINGS.model_artifact_dir
_artifact_relpath = _COMPAT_BINDINGS.artifact_relpath
_artifact_abspath = _COMPAT_BINDINGS.artifact_abspath
_write_json = _COMPAT_BINDINGS.write_json
_read_json = _COMPAT_BINDINGS.read_json
load_models_from_store = _COMPAT_BINDINGS.load_models_from_store
save_models_to_store = _COMPAT_BINDINGS.save_models_to_store
_load_fd004_benchmark_summary = _COMPAT_BINDINGS.load_fd004_benchmark_summary
_require_tls = _AUTH_HELPERS.require_tls
_oidc_issuer = _AUTH_HELPERS.oidc_issuer
_oidc_audience = _AUTH_HELPERS.oidc_audience
_oidc_jwks_url = _AUTH_HELPERS.oidc_jwks_url
_oidc_algorithms = _AUTH_HELPERS.oidc_algorithms
_oidc_enabled = _AUTH_HELPERS.oidc_enabled
_validate_oidc_bearer_token = _AUTH_HELPERS.validate_oidc_bearer_token
_max_body_bytes = _AUTH_HELPERS.max_body_bytes
_sync_max_points = _AUTH_HELPERS.sync_max_points
_job_store_backend = _AUTH_HELPERS.job_store_backend
_job_store_postgres_dsn = _AUTH_HELPERS.job_store_postgres_dsn
_model_registry_backend = _AUTH_HELPERS.model_registry_backend
_model_registry_postgres_dsn = _AUTH_HELPERS.model_registry_postgres_dsn
_job_execution_backend = _AUTH_HELPERS.job_execution_backend
_audit_log_path = _AUTH_HELPERS.audit_log_path
_audit_log_enabled = _AUTH_HELPERS.audit_log_enabled
_rate_limit_enabled = _AUTH_HELPERS.rate_limit_enabled
_rate_limit_per_window = _AUTH_HELPERS.rate_limit_per_window
_rate_limit_window_seconds = _AUTH_HELPERS.rate_limit_window_seconds


def _expected_api_key() -> str | None:
    return _AUTH_HELPERS.expected_api_key()


def _expected_bearer_token() -> str | None:
    return _AUTH_HELPERS.expected_bearer_token()


def _require_api_key(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> None:
    auth_helpers.require_api_key(
        request=request,
        x_api_key=x_api_key,
        authorization=authorization,
        api_error_cls=ApiError,
        expected_api_key_fn=_expected_api_key,
        expected_bearer_token_fn=_expected_bearer_token,
        oidc_enabled_fn=_oidc_enabled,
        validate_oidc_bearer_token_fn=_validate_oidc_bearer_token,
        logger=_LOGGER,
    )


def _require_api_access(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    x_connection_type: str | None = Header(None, alias="X-Connection-Type"),
) -> None:
    auth_helpers.require_api_access(
        request=request,
        x_api_key=x_api_key,
        authorization=authorization,
        x_tenant_id=x_tenant_id,
        x_connection_type=x_connection_type,
        require_api_key_fn=_require_api_key,
        enforce_request_policy_fn=enforce_request_policy,
    )


def _require_train_access(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    x_connection_type: str | None = Header(None, alias="X-Connection-Type"),
    x_approved_by: str | None = Header(None, alias="X-Approved-By"),
    x_approval_reason: str | None = Header(None, alias="X-Approval-Reason"),
) -> None:
    auth_helpers.require_train_access(
        request=request,
        x_api_key=x_api_key,
        authorization=authorization,
        x_tenant_id=x_tenant_id,
        x_connection_type=x_connection_type,
        x_approved_by=x_approved_by,
        x_approval_reason=x_approval_reason,
        require_api_access_fn=_require_api_access,
        enforce_train_request_approval_fn=enforce_train_request_approval,
    )


def _load_trained_models() -> dict[str, dict[str, Any]]:
    try:
        models = load_models_from_store()
    except Exception:
        _LOGGER.exception("Failed to load trained models")
        return {}
    return dict(models)


def _save_trained_models(models: dict[str, dict[str, Any]]) -> None:
    try:
        save_models_to_store(models)
    except Exception:
        _LOGGER.exception("Failed to persist trained models")


def _save_trained_model(entry: dict[str, Any]) -> None:
    model_id = str(entry.get("model_id") or "").strip()
    if not model_id:
        return
    models = _require_trained_models()
    models[model_id] = dict(entry)
    _save_trained_models(models)


def _build_job_store() -> JobStore:
    return build_job_store(
        sqlite_db_path=_MODEL_REGISTRY_DB_PATH,
        backend=_job_store_backend(),
        postgres_dsn=_job_store_postgres_dsn(),
    )


def _set_runtime_job_store(value: JobStore | None) -> JobStore | None:
    globals()["JOB_STORE"] = value
    return _BOOTSTRAP.set_job_store(value)


def _set_runtime_trained_models(
    value: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    globals()["TRAINED_MODELS"] = value
    return _BOOTSTRAP.set_trained_models(value)


_BOOTSTRAP = app_bootstrap.build_runtime_bootstrap(
    paths=_RUNTIME_PATHS,
    build_job_store=lambda: _build_job_store(),
    load_trained_models=lambda: _load_trained_models(),
)
_APP_FACTORY_LOCK = _BOOTSTRAP.app_factory_lock
_RATE_LIMIT_LOCK = _BOOTSTRAP.rate_limit_lock
_RATE_LIMIT_HITS = _BOOTSTRAP.rate_limit_hits
_RUNTIME_STATE = _BOOTSTRAP.runtime_state


# =============================================================================
# Models (Pydantic)
# =============================================================================

# Schemas are defined in forecasting_api.schemas.


# =============================================================================
# Error handling (contract: error_code + request_id)
# =============================================================================


def _get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_json(request: Request, exc: ApiError) -> JSONResponse:
    body = ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        details=(ErrorDetails.model_validate(exc.details) if exc.details is not None else None),
        request_id=_get_request_id(request),
    ).model_dump(exclude_none=True)
    return JSONResponse(status_code=exc.status_code, content=body)


# =============================================================================
# Config
# =============================================================================


def _is_https_request(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    if forwarded == "https":
        return True
    return request.url.scheme == "https"


def _rate_limit_key(request: Request) -> str:
    client_ip = request.client.host if request.client else "unknown"
    auth_method = str(getattr(request.state, "auth_method", "none") or "none")
    return f"{request.method}:{request.url.path}:{client_ip}:{auth_method}"



def _consume_rate_limit(request: Request) -> tuple[bool, int]:
    window_seconds = _rate_limit_window_seconds()
    max_requests = _rate_limit_per_window()
    now = monotonic()
    key = _rate_limit_key(request)
    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_HITS[key]
        while bucket and (now - bucket[0]) >= float(window_seconds):
            bucket.popleft()
        if len(bucket) >= max_requests:
            retry_after = max(1, int(math.ceil(float(window_seconds) - (now - bucket[0]))))
            return False, retry_after
        bucket.append(now)
    return True, 0


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# Validation helpers (contract: V01/V02/V03/V04)
# =============================================================================


def _ensure_quantiles_level_exclusive(
    req: ForecastRequest | CmapssFd004PreprocessRequest,
) -> None:
    if req.quantiles and req.level:
        raise ApiError(
            status_code=400,
            error_code="V03",
            message="quantiles と level は同時指定できません",
            details={"next_action": "quantiles か level のどちらか一方のみ指定してください"},
        )


def _infer_step_delta(records: list[TimeSeriesRecord]) -> timedelta | None:
    # Infer a single global step from the first series that has >=2 points.
    by_series: dict[str, list[datetime]] = {}
    for r in records:
        by_series.setdefault(r.series_id, []).append(r.timestamp)

    for _, ts_list in by_series.items():
        if len(ts_list) < 2:
            continue
        # Use input order to respect V04 check elsewhere.
        diffs = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
        first = diffs[0]
        if first.total_seconds() <= 0:
            return None
        if any(d != first for d in diffs):
            return None
        return first

    return None


def _require_monotonic_increasing(records: list[TimeSeriesRecord]) -> None:
    last_seen: dict[str, datetime] = {}
    for r in records:
        prev = last_seen.get(r.series_id)
        if prev is not None and r.timestamp <= prev:
            raise ApiError(
                status_code=400,
                error_code="V04",
                message="系列内 timestamp が単調増加でありません（ソートが必要）",
                details={
                    "series_id": r.series_id,
                    "next_action": "series_id ごとに timestamp で昇順ソートしてください",
                },
            )
        last_seen[r.series_id] = r.timestamp


def _require_frequency_or_infer(req: ForecastRequest) -> timedelta:
    if req.frequency:
        f = req.frequency.strip().lower()
        if f in {"d", "day", "1d"}:
            return timedelta(days=1)
        if f in {"h", "hour", "1h"}:
            return timedelta(hours=1)
        if f.endswith("min"):
            try:
                n = int(f[:-3]) if f[:-3] else 1
                return timedelta(minutes=n)
            except ValueError:
                pass
        if f.endswith("s"):
            try:
                n = int(f[:-1])
                return timedelta(seconds=n)
            except ValueError:
                pass
        # Unknown format: accept as-is but require infer from timestamps.

    inferred = _infer_step_delta(req.data)
    if inferred is None:
        raise ApiError(
            status_code=400,
            error_code="V02",
            message="frequency を推定できません",
            details={"next_action": "frequency を明示してください（例: D, H, 15min）"},
        )
    return inferred


def _require_trained_model(model_id: str | None) -> None:
    if not model_id:
        return
    trained_models = _require_trained_models()
    if model_id not in trained_models:
        raise ApiError(
            status_code=404,
            error_code="M01",
            message="model_id が存在しません",
            details={
                "model_id": model_id,
                "next_action": "GET /v1/models で一覧を確認してください",
            },
        )


def _missing_policy(req: ForecastRequest) -> str | None:
    try:
        return req.options.missing_policy if req.options is not None else None
    except Exception:
        return None


def _require_no_gaps_if_missing_policy_error(req: ForecastRequest, step: timedelta) -> None:
    if _missing_policy(req) != "error":
        return

    # Collect gaps per series in request order (assumes V04 monotonic check already passed).
    gaps: list[dict[str, str]] = []
    last_seen: dict[str, datetime] = {}
    for r in req.data:
        prev = last_seen.get(r.series_id)
        if prev is not None:
            expected = prev + step
            if r.timestamp != expected:
                gaps.append(
                    {
                        "series_id": r.series_id,
                        "prev_timestamp": prev.isoformat(),
                        "expected_timestamp": expected.isoformat(),
                        "timestamp": r.timestamp.isoformat(),
                    }
                )
                if len(gaps) >= 50:
                    break
        last_seen[r.series_id] = r.timestamp

    if gaps:
        raise ApiError(
            status_code=400,
            error_code="V05",
            message="欠損timestampが存在します（missing_policy=error）",
            details={
                "missing_policy": "error",
                "step_seconds": step.total_seconds(),
                "gaps": gaps,
                "next_action": "欠損を補完するか options.missing_policy を変更してください",
            },
        )


# =============================================================================
# Backtest (lightweight baseline)
# =============================================================================


def _naive_backtest(req: BacktestRequest) -> BacktestResponse:
    return stable_models.naive_backtest(req)


def _torch_backtest(req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse:
    return torch_runtime.torch_backtest(
        req,
        trained=trained,
        read_json=_read_json,
        artifact_abspath=_artifact_abspath,
        try_torch_load_weights=_try_torch_load_weights,
        extract_state_dict=_extract_state_dict,
        metric_value_fn=stable_models.metric_value,
        api_error_cls=ApiError,
    )


def _torch_backtest_adapter(req: BacktestRequest, *, trained: dict[str, Any]) -> BacktestResponse:
    return _torch_backtest(req, trained=trained)


def _naive_backtest_adapter(req: BacktestRequest) -> BacktestResponse:
    return _naive_backtest(req)


def run_backtest_request(req: BacktestRequest) -> BacktestResponse:
    from .services.backtest_service import run_backtest_request as _run_backtest_request

    try:
        return _run_backtest_request(req)
    except RuntimeError as exc:
        if str(exc) != "backtest service is not configured":
            raise
        create_app()
        return _run_backtest_request(req)


def build_job_error_payload(
    *, error_code: str, message: str, details: dict[str, Any] | ErrorDetails | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_code": error_code,
        "message": message,
        "request_id": None,
    }
    if details is not None:
        payload["details"] = (
            details.model_dump(exclude_none=True) if isinstance(details, ErrorDetails) else details
        )
    return ErrorResponse.model_validate(payload).model_dump(exclude_none=True)

TRAINED_MODELS: dict[str, dict[str, Any]] | None = None
JOB_STORE: JobStore | None = None


# =============================================================================
# FastAPI app
# =============================================================================


def create_app() -> FastAPI:
    configure_logging()
    logger = get_logger("api")
    validate_cloud_run_runtime(
        job_execution_backend=_job_execution_backend(),
        job_store_backend=_job_store_backend(),
        job_store_postgres_dsn=_job_store_postgres_dsn(),
        model_registry_backend=_model_registry_backend(),
        model_registry_postgres_dsn=_model_registry_postgres_dsn(),
        model_registry_db_path=_MODEL_REGISTRY_DB_PATH,
        model_artifacts_root=_MODEL_ARTIFACTS_ROOT,
        request_audit_log_path=_audit_log_path(),
        drift_baseline_path=_DRIFT_BASELINE_PATH,
        promotion_registry_path=_MODEL_PROMOTION_REGISTRY_PATH,
    )
    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_HITS.clear()
    from .services import backtest_service, forecast_service, jobs_service, train_service
    from .services.runtime import JobsServiceDeps

    app = FastAPI(
        title="RULFM Forecasting API",
        version=_resolve_app_version(),
        description=(
            "[EN] Time series forecasting API with sync and async (job) execution.\n"
            "[EN] Language: Swagger UI has no built-in language toggle. Use /docs/ja or /docs/en.\n"
            "[EN] - All errors return error_code and request_id.\n"
            "[EN] - X-Request-Id is returned in response headers for support correlation.\n"
            "[EN] - Use /v1/jobs for large inputs or long-running tasks.\n"
            "[EN] - Timestamps must be ascending within each series.\n"
            "[EN] - quantiles and level cannot be specified together.\n"
            "[JA] 時系列予測API（同期/非同期）です。\n"
            "[JA] 言語: Swagger UIに言語切替はありません。/docs/ja または /docs/en を利用してください。\n"
            "[JA] 注意:\n"
            "[JA] - すべてのエラーで error_code と request_id を返します。\n"
            "[JA] - 応答ヘッダーに X-Request-Id が返ります。\n"
            "[JA] - 大規模入力や長時間タスクは /v1/jobs を使用します。\n"
            "[JA] - timestamp は系列内で昇順に並べてください。\n"
            "[JA] - quantiles と level は同時指定できません。\n"
        ),
        docs_url=None,
    )

    cors_origins_raw = os.getenv("RULFM_FORECASTING_API_CORS_ORIGINS", "").strip()
    if cors_origins_raw:
        origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_methods=["*"],
                allow_headers=["*"],
                allow_credentials=False,
            )

    docs_routes = app_docs.configure_openapi_and_docs(app)
    response_docs = app_docs.build_api_response_docs(bi=_bi, error_response_model=ErrorResponse)
    _request_id_header = response_docs.request_id_header
    _r400 = response_docs.responses_400
    _r401 = response_docs.responses_401
    _r403 = response_docs.responses_403
    _r404 = response_docs.responses_404
    _r409 = response_docs.responses_409
    _r413 = response_docs.responses_413
    _r429 = response_docs.responses_429

    static_gui_dir = Path(__file__).resolve().parent / "static" / "forecasting_gui"
    root_redirect_to_gui = app_docs.mount_static_gui(app, static_gui_dir=static_gui_dir)

    def _append_request_audit_entry(entry: dict[str, Any]) -> None:
        append_request_audit_log_entry(
            entry,
            path=_audit_log_path(),
            enabled=_audit_log_enabled(),
        )

    register_request_context_middleware(
        app,
        api_error_cls=ApiError,
        track_request=track_request,
        error_json=_error_json,
        get_request_id=_get_request_id,
        append_request_audit_log=_append_request_audit_entry,
        apply_standard_security_headers=apply_standard_security_headers,
        logger=logger,
        require_tls=_require_tls,
        is_https_request=_is_https_request,
        max_body_bytes=_max_body_bytes,
        rate_limit_enabled=_rate_limit_enabled,
        consume_rate_limit=_consume_rate_limit,
        rate_limit_window_seconds=_rate_limit_window_seconds,
        rate_limit_per_window=_rate_limit_per_window,
    )

    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError):
        return _error_json(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        # Map structural/typing errors to V01.
        return _error_json(
            request,
            ApiError(
                status_code=400,
                error_code="V01",
                message="入力が不正です",
                details={
                    "errors": exc.errors(),
                    "next_action": "必須フィールドと型を確認してください",
                },
            ),
        )

    @app.get("/health", summary=_bi("Health check", "ヘルスチェック"), tags=["health"])
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "public_profile": "gbdt-only",
            "experimental_models": "enabled"
            if training_helpers.experimental_models_enabled()
            else "disabled",
        }

    @app.get("/metrics", include_in_schema=False, dependencies=[Depends(_require_api_key)])
    async def metrics() -> Response:
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    @app.get(
        "/v1/cmapss/fd004/sample",
        summary=_bi(
            "Get a prepared CMAPSS FD004 sample payload", "CMAPSS FD004 サンプル payload を取得"
        ),
        tags=["cmapss"],
        response_model=CmapssFd004PayloadResponse,
        dependencies=[Depends(_require_api_access)],
        responses={
            200: {"description": _bi("OK", "OK"), "headers": _request_id_header},
            **_r401,
            **_r403,
            **_r400,
            **_r413,
        },
    )
    async def get_cmapss_fd004_sample(profile: str) -> CmapssFd004PayloadResponse:
        try:
            return CmapssFd004PayloadResponse(**build_fd004_profile_payload(profile))
        except ValueError as exc:
            raise ApiError(
                status_code=400,
                error_code="V01",
                message=str(exc),
                details={
                    "available_profiles": available_profiles(),
                    "next_action": "profile を見直してください",
                },
            ) from exc

    @app.post(
        "/v1/cmapss/fd004/preprocess",
        summary=_bi(
            "Preprocess CMAPSS FD004 into API records", "CMAPSS FD004 を API レコードへ前処理"
        ),
        tags=["cmapss"],
        response_model=CmapssFd004PayloadResponse,
        dependencies=[Depends(_require_api_access)],
        responses={
            200: {"description": _bi("OK", "OK"), "headers": _request_id_header},
            **_r401,
            **_r403,
            **_r400,
            **_r413,
        },
    )
    async def preprocess_cmapss_fd004(
        req: CmapssFd004PreprocessRequest,
    ) -> CmapssFd004PayloadResponse:
        _ensure_quantiles_level_exclusive(req)
        try:
            payload = build_fd004_payload(
                split=req.split,
                unit_ids=req.unit_ids,
                window_size=req.window_size,
                task=req.task,
                horizon=req.horizon,
                quantiles=req.quantiles,
                level=req.level,
                max_rul=req.max_rul,
            )
            return CmapssFd004PayloadResponse(**payload)
        except (ValueError, FileNotFoundError) as exc:
            raise ApiError(
                status_code=400,
                error_code="V01",
                message=str(exc),
                details={
                    "available_profiles": available_profiles(),
                    "next_action": "split / unit_ids / dataset 配置を確認してください",
                },
            ) from exc

    @app.get(
        "/v1/cmapss/fd004/benchmarks",
        summary=_bi("Get FD004 benchmark summary", "FD004 ベンチマーク要約を取得"),
        tags=["cmapss"],
        dependencies=[Depends(_require_api_access)],
        responses={200: {"description": _bi("OK", "OK"), "headers": _request_id_header}, **_r401, **_r403},
    )
    async def get_cmapss_fd004_benchmarks() -> dict[str, Any]:
        return _load_fd004_benchmark_summary()

    def _run_job(
        job_id: str, job_type: Literal["forecast", "train", "backtest"], payload: dict[str, Any]
    ) -> None:
        run_job_handler(job_id, job_type, payload)

    def _runtime_job_store() -> JobStore:
        return _require_job_store()

    def _runtime_trained_models() -> dict[str, dict[str, Any]]:
        return _require_trained_models()

    forecast_deps = forecast_runtime.build_forecast_service_deps(
        api_error_cls=ApiError,
        bi=_bi,
        trained_models=_runtime_trained_models,
        ensure_quantiles_level_exclusive=_ensure_quantiles_level_exclusive,
        require_monotonic_increasing=_require_monotonic_increasing,
        require_trained_model=_require_trained_model,
        sync_max_points=_sync_max_points,
        require_frequency_or_infer=_require_frequency_or_infer,
        require_no_gaps_if_missing_policy_error=_require_no_gaps_if_missing_policy_error,
        read_json=_read_json,
        artifact_abspath=_artifact_abspath,
        try_torch_load_weights=_try_torch_load_weights,
        extract_state_dict=_extract_state_dict,
        load_joblib_artifact=_load_joblib_artifact,
    )
    run_forecast_handler = forecast_service.build_run_forecast(forecast_deps)
    forecast_service.configure_forecast_service(forecast_deps)

    backtest_deps = backtest_runtime.build_backtest_service_deps(
        api_error_cls=ApiError,
        trained_models=_runtime_trained_models,
        require_monotonic_increasing=_require_monotonic_increasing,
        require_trained_model=_require_trained_model,
        read_json=_read_json,
        artifact_abspath=_artifact_abspath,
        try_torch_load_weights=_try_torch_load_weights,
        extract_state_dict=_extract_state_dict,
        load_joblib_artifact=_load_joblib_artifact,
        torch_backtest=_torch_backtest_adapter,
        naive_backtest=_naive_backtest_adapter,
        start_run=start_run,
        log_params=log_params,
        log_metrics=log_metrics,
        log_dict_artifact=log_dict_artifact,
    )
    run_backtest_request_handler = backtest_service.build_run_backtest_request(backtest_deps)
    run_backtest_handler = backtest_service.build_run_backtest(backtest_deps)
    backtest_service.configure_backtest_service(backtest_deps)

    train_deps = train_runtime.build_train_service_deps(
        api_error_cls=ApiError,
        trained_models=_runtime_trained_models,
        require_monotonic_increasing=_require_monotonic_increasing,
        model_artifact_dir=_model_artifact_dir,
        write_json=_write_json,
        artifact_relpath=_artifact_relpath,
        artifact_abspath=_artifact_abspath,
        save_trained_model=_save_trained_model,
        start_run=start_run,
        log_params=log_params,
        log_metrics=log_metrics,
        log_dict_artifact=log_dict_artifact,
        log_artifact=log_artifact,
    )
    run_train_handler = train_service.build_run_train(train_deps)
    train_service.configure_train_service(train_deps)

    jobs_deps = JobsServiceDeps(
        api_error_cls=ApiError,
        job_store=_runtime_job_store,
        build_job_error_payload=build_job_error_payload,
        run_forecast=run_forecast_handler,
        run_train=run_train_handler,
        run_backtest_request=run_backtest_request_handler,
    )
    create_job_handler = jobs_service.build_create_job(jobs_deps)
    get_job_status_handler = jobs_service.build_get_job_status(jobs_deps)
    get_job_result_handler = jobs_service.build_get_job_result(jobs_deps)
    run_job_handler = jobs_service.build_run_job(jobs_deps)
    jobs_service.configure_jobs_service(jobs_deps)

    app.include_router(
        build_forecast_router(
            require_api_access=_require_api_access,
            request_id_header=_request_id_header,
            responses_400=_r400,
            responses_401=_r401,
            responses_403=_r403,
            responses_413=_r413,
            responses_429=_r429,
            bi=_bi,
            run_forecast=run_forecast_handler,
        )
    )
    app.include_router(
        build_jobs_router(
            require_api_access=_require_api_access,
            request_id_header=_request_id_header,
            responses_400=_r400,
            responses_401=_r401,
            responses_403=_r403,
            responses_404=_r404,
            responses_409=_r409,
            responses_413=_r413,
            responses_429=_r429,
            bi=_bi,
            job_enqueuer_factory=lambda background: (
                build_inprocess_job_enqueuer(
                    background=background,
                    run_job=_run_job,
                )
                if _job_execution_backend() == "inprocess"
                else build_persistent_job_enqueuer(background=background)
            ),
            create_job_handler=create_job_handler,
            get_job_status_handler=get_job_status_handler,
            get_job_result_handler=get_job_result_handler,
        )
    )
    app.include_router(
        build_train_router(
            require_train_access=_require_train_access,
            request_id_header=_request_id_header,
            responses_400=_r400,
            responses_401=_r401,
            responses_403=_r403,
            responses_413=_r413,
            responses_429=_r429,
            bi=_bi,
            run_train=run_train_handler,
        )
    )
    app.include_router(
        build_monitoring_router(
            require_api_key=_require_api_key,
            request_id_header=_request_id_header,
            responses_400=_r400,
            responses_401=_r401,
            responses_413=_r413,
            bi=_bi,
            log_ephemeral_baseline=lambda: logger.info(
                "using_ephemeral_drift_baseline",
                extra={"event_type": "DRIFT_BASELINE_EPHEMERAL"},
            ),
        )
    )
    app.include_router(
        build_backtest_router(
            require_api_access=_require_api_access,
            request_id_header=_request_id_header,
            responses_400=_r400,
            responses_401=_r401,
            responses_403=_r403,
            responses_413=_r413,
            responses_429=_r429,
            bi=_bi,
            run_backtest=run_backtest_handler,
        )
    )

    @app.get(
        "/v1/models",
        summary=_bi("List trained models", "学習済みモデル一覧"),
        description=_bi(
            "Returns catalog of trained model IDs.", "学習済みモデルの一覧を返します。"
        ),
        tags=["models"],
        response_model=ModelCatalogResponse,
        dependencies=[Depends(_require_api_access)],
        responses={
            200: {"description": _bi("OK", "OK"), "headers": _request_id_header},
            **_r401,
            **_r403,
        },
    )
    async def list_models() -> ModelCatalogResponse:
        models = list(_runtime_trained_models().values())
        def _created_at_key(m: dict[str, Any]) -> datetime:
            raw = m.get("created_at")
            if not raw:
                return datetime.min.replace(tzinfo=UTC)
            try:
                dt = datetime.fromisoformat(str(raw))
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
            except ValueError:
                return datetime.min.replace(tzinfo=UTC)

        models.sort(key=_created_at_key, reverse=True)
        out: list[ModelCatalogEntry] = []
        for m in models:
            out.append(
                ModelCatalogEntry(
                    model_id=str(m.get("model_id") or "").strip(),
                    created_at=(
                        str(m.get("created_at")) if m.get("created_at") is not None else None
                    ),
                    memo=(str(m.get("memo")) if m.get("memo") is not None else None),
                )
            )
        return ModelCatalogResponse(models=out)

    _store_registered_route_handlers(
        app,
        docs_routes.docs_root,
        docs_routes.docs_en,
        docs_routes.docs_ja,
        docs_routes.openapi_en,
        docs_routes.openapi_ja,
        root_redirect_to_gui,
        _handle_api_error,
        _handle_validation_error,
        health,
        metrics,
        get_cmapss_fd004_sample,
        preprocess_cmapss_fd004,
        get_cmapss_fd004_benchmarks,
        list_models,
    )

    return app


def _require_job_store() -> JobStore:
    current = cast(JobStore | None, globals().get("JOB_STORE"))
    if current is None:
        _BOOTSTRAP.set_job_store(None)
    job_store = _BOOTSTRAP.require_job_store(current)
    globals()["JOB_STORE"] = job_store
    return job_store


def _require_trained_models() -> dict[str, dict[str, Any]]:
    current = cast(dict[str, dict[str, Any]] | None, globals().get("TRAINED_MODELS"))
    if current is None:
        _BOOTSTRAP.set_trained_models(None)
    trained_models = _BOOTSTRAP.require_trained_models(current)
    globals()["TRAINED_MODELS"] = trained_models
    return trained_models


class _LazyAppProxy:
    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def _get_app(self) -> FastAPI:
        if self._app is None:
            with _APP_FACTORY_LOCK:
                if self._app is None:
                    self._app = create_app()
        return self._app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self._get_app()(scope, receive, send)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_app(), name)


app = _LazyAppProxy()
