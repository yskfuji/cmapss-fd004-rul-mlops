from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends
from forecasting_api.schemas import TrainRequest, TrainResponse


def build_train_router(
    *,
    require_train_access: Callable[..., Any],
    request_id_header: dict[str, Any],
    responses_400: dict[int | str, Any],
    responses_401: dict[int | str, Any],
    responses_403: dict[int | str, Any],
    responses_413: dict[int | str, Any],
    responses_429: dict[int | str, Any],
    bi: Callable[[str, str], str],
    run_train: Callable[[TrainRequest], TrainResponse],
) -> APIRouter:
    router = APIRouter(tags=["train"])

    @router.post(
        "/v1/train",
        status_code=202,
        summary=bi("Train model", "モデル学習"),
        description=bi(
            "Starts training and returns a model_id.",
            "学習を開始し model_id を返します。",
        ),
        response_model=TrainResponse,
        dependencies=[Depends(require_train_access)],
        responses={
            202: {"description": bi("Accepted", "受付済み"), "headers": request_id_header},
            **responses_401,
            **responses_403,
            **responses_400,
            **responses_429,
            **responses_413,
        },
    )
    async def train(req: TrainRequest) -> TrainResponse:
        return run_train(req)

    return router