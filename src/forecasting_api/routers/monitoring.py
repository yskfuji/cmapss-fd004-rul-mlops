from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends

from forecasting_api.schemas import (
    DriftBaselineRequest,
    DriftBaselineResponse,
    DriftBaselineStatusResponse,
    DriftReportRequest,
    DriftReportResponse,
)
from forecasting_api.services import monitoring_service


def build_monitoring_router(
    *,
    require_api_key: Callable[..., Any],
    request_id_header: dict[str, Any],
    responses_400: dict[int | str, Any],
    responses_401: dict[int | str, Any],
    responses_413: dict[int | str, Any],
    bi: Callable[[str, str], str],
    log_ephemeral_baseline: Callable[[], None],
) -> APIRouter:
    router = APIRouter(tags=["monitoring"])

    @router.post(
        "/v1/monitoring/drift/baseline",
        summary=bi("Persist drift baseline", "ドリフトベースラインを保存"),
        description=bi(
            "Stores a validated feature baseline for future drift reports.",
            "将来のドリフトレポートに使う検証済みベースラインを保存します。",
        ),
        response_model=DriftBaselineResponse,
        dependencies=[Depends(require_api_key)],
        responses={
            200: {"description": bi("OK", "OK"), "headers": request_id_header},
            **responses_401,
            **responses_400,
            **responses_413,
        },
    )
    async def persist_drift_baseline(req: DriftBaselineRequest) -> DriftBaselineResponse:
        return monitoring_service.persist_drift_baseline(req)

    @router.get(
        "/v1/monitoring/drift/baseline/status",
        summary=bi("Get drift baseline status", "ドリフトベースライン状態を取得"),
        description=bi(
            "Returns whether a persisted drift baseline exists and summarizes its feature bins.",
            "保存済みドリフトベースラインの有無と feature ごとの bins 概要を返します。",
        ),
        response_model=DriftBaselineStatusResponse,
        dependencies=[Depends(require_api_key)],
        responses={
            200: {"description": bi("OK", "OK"), "headers": request_id_header},
            **responses_401,
        },
    )
    async def get_drift_baseline_status() -> DriftBaselineStatusResponse:
        return monitoring_service.get_drift_baseline_status()

    @router.post(
        "/v1/monitoring/drift/report",
        summary=bi("Generate drift report", "ドリフトレポートを生成"),
        description=bi(
            "Compares candidate records against a stored or provided baseline.",
            "候補レコードを保存済みまたは同梱のベースラインと比較します。",
        ),
        response_model=DriftReportResponse,
        dependencies=[Depends(require_api_key)],
        responses={
            200: {"description": bi("OK", "OK"), "headers": request_id_header},
            **responses_401,
            **responses_400,
            **responses_413,
        },
    )
    async def generate_drift_report(req: DriftReportRequest) -> DriftReportResponse:
        return monitoring_service.generate_drift_report(
            req,
            log_ephemeral_baseline=log_ephemeral_baseline,
        )

    return router