from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends

from forecasting_api.schemas import ForecastRequest, ForecastResponse


def build_forecast_router(
    *,
    require_api_access: Callable[..., Any],
    request_id_header: dict[str, Any],
    responses_400: dict[int | str, Any],
    responses_401: dict[int | str, Any],
    responses_403: dict[int | str, Any],
    responses_413: dict[int | str, Any],
    responses_429: dict[int | str, Any],
    bi: Callable[[str, str], str],
    run_forecast: Callable[[ForecastRequest], ForecastResponse],
) -> APIRouter:
    router = APIRouter(tags=["forecast"])

    @router.post(
        "/v1/forecast",
        summary=bi("Forecast (sync)", "予測（同期）"),
        description=(
            bi(
                "Runs a forecast synchronously. Use /v1/jobs for large inputs.",
                "同期で予測を実行します。大規模入力は /v1/jobs を使用してください。",
            )
            + "\n"
            + bi(
                "Supports quantiles or level (not both).",
                "quantiles または level を指定できます（併用不可）。",
            )
        ),
        response_model=ForecastResponse,
        dependencies=[Depends(require_api_access)],
        responses={
            200: {"description": bi("OK", "OK"), "headers": request_id_header},
            **responses_401,
            **responses_403,
            **responses_400,
            **responses_429,
            **responses_413,
        },
    )
    async def forecast(req: ForecastRequest) -> ForecastResponse:
        return run_forecast(req)

    return router