from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends

from forecasting_api.schemas import BacktestRequest, BacktestResponse


def build_backtest_router(
    *,
    require_api_access: Callable[..., Any],
    request_id_header: dict[str, Any],
    responses_400: dict[int | str, Any],
    responses_401: dict[int | str, Any],
    responses_403: dict[int | str, Any],
    responses_413: dict[int | str, Any],
    responses_429: dict[int | str, Any],
    bi: Callable[[str, str], str],
    run_backtest: Callable[[BacktestRequest], BacktestResponse],
) -> APIRouter:
    router = APIRouter(tags=["backtest"])

    @router.post(
        "/v1/backtest",
        summary=bi("Backtest", "バックテスト"),
        description=bi(
            "Runs backtest evaluation and returns metrics.",
            "バックテストを実行し評価指標を返します。",
        ),
        response_model=BacktestResponse,
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
    async def backtest(req: BacktestRequest) -> BacktestResponse:
        return run_backtest(req)

    return router