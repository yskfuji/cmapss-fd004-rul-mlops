from __future__ import annotations

from typing import Any, Literal

from fastapi.responses import JSONResponse
from forecasting_api.schemas import (
    BacktestRequest,
    ErrorDetails,
    ErrorResponse,
    ForecastRequest,
    JobCreateResponse,
    JobStatusResponse,
    TrainRequest,
)
from pydantic import ValidationError

from .backtest_service import run_backtest_request
from .forecast_service import run_forecast
from .runtime import JobsServiceDeps
from .train_service import run_train

_service_deps: JobsServiceDeps | None = None


def configure_jobs_service(deps: JobsServiceDeps) -> None:
    global _service_deps
    _service_deps = deps


def _require_deps() -> JobsServiceDeps:
    if _service_deps is None:
        raise RuntimeError("jobs service is not configured")
    return _service_deps


def build_run_job(deps: JobsServiceDeps):
    def run_job(
        job_id: str,
        job_type: Literal["forecast", "train", "backtest"],
        payload: dict[str, Any],
    ) -> None:
        api_error_cls = deps.api_error_cls
        job_store = deps.job_store()

        try:
            job_store.set_running(job_id)
            if job_type == "forecast":
                forecast_req = ForecastRequest.model_validate(payload)
                result = deps.run_forecast(forecast_req).model_dump(mode="json", exclude_none=True)
                job_store.set_succeeded(job_id, result)
                return

            if job_type == "train":
                train_req = TrainRequest.model_validate(payload)
                result = deps.run_train(train_req).model_dump(mode="json", exclude_none=True)
                job_store.set_succeeded(job_id, result)
                return

            if job_type == "backtest":
                backtest_req = BacktestRequest.model_validate(payload)
                result = deps.run_backtest_request(backtest_req).model_dump(mode="json", exclude_none=True)
                job_store.set_succeeded(job_id, result)
                return

            job_store.set_failed(
                job_id,
                deps.build_job_error_payload(
                    error_code="V01",
                    message="入力が不正です",
                    details=ErrorDetails(next_action="type を確認してください"),
                ),
            )
        except api_error_cls as exc:
            typed_exc = exc
            job_store.set_failed(
                job_id,
                deps.build_job_error_payload(
                    error_code=typed_exc.error_code,
                    message=typed_exc.message,
                    details=(
                        ErrorDetails.model_validate(typed_exc.details)
                        if typed_exc.details is not None
                        else None
                    ),
                ),
            )
        except ValidationError as exc:
            job_store.set_failed(
                job_id,
                deps.build_job_error_payload(
                    error_code="V01",
                    message="入力が不正です",
                    details=ErrorDetails.model_validate({"errors": exc.errors()}),
                ),
            )
        except Exception as exc:
            job_store.set_failed(
                job_id,
                deps.build_job_error_payload(
                    error_code="E00",
                    message="内部エラー",
                    details=ErrorDetails.model_validate({"error": str(exc)}),
                ),
            )

    return run_job


def build_create_job(deps: JobsServiceDeps):
    def create_job(req: Any) -> JobCreateResponse:
        job_store = deps.job_store()
        job = job_store.create(req.type, req.payload)
        return JobCreateResponse(job_id=job.job_id, status=job.status)

    return create_job


def build_get_job_status(deps: JobsServiceDeps):
    def get_job_status(job_id: str) -> JobStatusResponse:
        api_error_cls = deps.api_error_cls
        job_store = deps.job_store()

        job = job_store.get(job_id)
        if job is None:
            raise api_error_cls(
                status_code=404,
                error_code="J01",
                message="job_id が存在しません",
                details={"job_id": job_id},
            )
        error = ErrorResponse.model_validate(job.error) if job.error else None
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            error=error,
        )

    return get_job_status


def build_get_job_result(deps: JobsServiceDeps):
    def get_job_result(job_id: str) -> JSONResponse:
        api_error_cls = deps.api_error_cls
        job_store = deps.job_store()

        job = job_store.get(job_id)
        if job is None:
            raise api_error_cls(
                status_code=404,
                error_code="J01",
                message="job_id が存在しません",
                details={"job_id": job_id},
            )
        if job.status != "succeeded":
            raise api_error_cls(
                status_code=409,
                error_code="J02",
                message="job が未完了です",
                details={
                    "status": job.status,
                    "next_action": "GET /v1/jobs/{job_id} で状態を確認してください",
                },
            )
        return JSONResponse(content=job.result or {})

    return get_job_result


def run_job(
    job_id: str,
    job_type: Literal["forecast", "train", "backtest"],
    payload: dict[str, Any],
) -> None:
    build_run_job(_require_deps())(job_id, job_type, payload)


def create_job(req: Any) -> JobCreateResponse:
    return build_create_job(_require_deps())(req)


def get_job_status(job_id: str) -> JobStatusResponse:
    return build_get_job_status(_require_deps())(job_id)


def get_job_result(job_id: str) -> JSONResponse:
    return build_get_job_result(_require_deps())(job_id)
