from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends

from forecasting_api.job_dispatcher import JobEnqueuerFactory
from forecasting_api.schemas import (
    JobCreateRequest,
    JobCreateResponse,
    JobStatusResponse,
)


def build_jobs_router(
    *,
    require_api_access: Callable[..., Any],
    request_id_header: dict[str, Any],
    responses_400: dict[int | str, Any],
    responses_401: dict[int | str, Any],
    responses_403: dict[int | str, Any],
    responses_404: dict[int | str, Any],
    responses_409: dict[int | str, Any],
    responses_413: dict[int | str, Any],
    responses_429: dict[int | str, Any],
    bi: Callable[[str, str], str],
    job_enqueuer_factory: JobEnqueuerFactory,
    create_job_handler: Callable[[JobCreateRequest], JobCreateResponse],
    get_job_status_handler: Callable[[str], JobStatusResponse],
    get_job_result_handler: Callable[[str], Any],
) -> APIRouter:
    router = APIRouter(tags=["jobs"])

    @router.post(
        "/v1/jobs",
        status_code=202,
        summary=bi("Create job (async)", "ジョブ作成（非同期）"),
        description=(
            bi(
                "Creates an async job for forecast/train/backtest. "
                "Use GET /v1/jobs/{job_id} to poll.",
                "予測/学習/バックテストの非同期ジョブを作成します。"
                "GET /v1/jobs/{job_id} で状態確認。",
            )
            + "\n"
            + bi(
                "Use GET /v1/jobs/{job_id}/result after status is succeeded.",
                "status が succeeded になったら GET /v1/jobs/{job_id}/result を利用してください。",
            )
        ),
        response_model=JobCreateResponse,
        dependencies=[Depends(require_api_access)],
        responses={
            202: {"description": bi("Accepted", "受付済み"), "headers": request_id_header},
            **responses_401,
            **responses_403,
            **responses_400,
            **responses_429,
            **responses_413,
        },
    )
    async def create_job(req: JobCreateRequest, background: BackgroundTasks) -> JobCreateResponse:
        response = create_job_handler(req)
        job_enqueuer_factory(background).enqueue(
            job_id=response.job_id,
            job_type=req.type,
            payload=req.payload,
        )
        return response

    @router.get(
        "/v1/jobs/{job_id}",
        summary=bi("Get job status", "ジョブ状態を取得"),
        description=bi(
            "Returns job status and error details (if failed).",
            "ジョブ状態と失敗時のエラー詳細を返します。",
        ),
        response_model=JobStatusResponse,
        dependencies=[Depends(require_api_access)],
        responses={
            200: {"description": bi("OK", "OK"), "headers": request_id_header},
            **responses_401,
            **responses_403,
            **responses_404,
        },
    )
    async def get_job(job_id: str) -> JobStatusResponse:
        return get_job_status_handler(job_id)

    @router.get(
        "/v1/jobs/{job_id}/result",
        summary=bi("Get job result", "ジョブ結果を取得"),
        description=bi(
            "Returns job result when status is succeeded.",
            "status が succeeded のとき結果を返します。",
        ),
        dependencies=[Depends(require_api_access)],
        responses={
            200: {"description": bi("OK", "OK"), "headers": request_id_header},
            **responses_401,
            **responses_403,
            **responses_404,
            **responses_409,
        },
    )
    async def get_job_result(job_id: str):
        return get_job_result_handler(job_id)

    return router