from __future__ import annotations

from typing import TYPE_CHECKING

from .runtime import ForecastServiceDeps

if TYPE_CHECKING:
    from forecasting_api.schemas import ForecastRequest, ForecastResponse


_service_deps: ForecastServiceDeps | None = None


def configure_forecast_service(deps: ForecastServiceDeps) -> None:
    global _service_deps
    _service_deps = deps


def _require_deps() -> ForecastServiceDeps:
    if _service_deps is None:
        raise RuntimeError("forecast service is not configured")
    return _service_deps


def build_run_forecast(deps: ForecastServiceDeps):
    def run_forecast(req: ForecastRequest) -> ForecastResponse:
        deps.ensure_quantiles_level_exclusive(req)
        deps.require_monotonic_increasing(req.data)
        deps.require_trained_model(req.model_id)

        max_points = deps.sync_max_points()
        if len(req.data) > max_points:
            api_error_cls = deps.api_error_cls
            raise api_error_cls(
                status_code=400,
                error_code="COST01",
                message="同期には重すぎます（非同期へ誘導）",
                details={
                    "max_points": max_points,
                    "next_action": "POST /v1/jobs を使用してください",
                },
            )

        step = deps.require_frequency_or_infer(req)
        deps.require_no_gaps_if_missing_policy_error(req, step)

        trained_models = deps.trained_models()
        if req.model_id and req.model_id in trained_models:
            trained = trained_models.get(req.model_id) or {}
            algo = deps.assert_model_algo_available(trained.get("algo"))
            if algo.startswith("ridge"):
                return deps.forecast_with_trained_model(req, step=step, trained=trained)
            if algo == "gbdt_hgb_v1":
                return deps.forecast_with_gbdt_model(req, step=step, trained=trained)
            if algo == "gbdt_afno_hybrid_v1":
                return deps.hybrid_forecast_model(req, step=step, trained=trained)
            if algo in {"afnocg2", "cifnocg2", "afnocg3", "afnocg3_v1", "stardast2_v5"}:
                return deps.torch_forecast_model(req, step=step, trained=trained)
        return deps.naive_forecast(req, step)

    return run_forecast


def run_forecast(req: ForecastRequest) -> ForecastResponse:
    return build_run_forecast(_require_deps())(req)
