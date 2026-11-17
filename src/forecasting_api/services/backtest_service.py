from __future__ import annotations

from typing import TYPE_CHECKING

from .runtime import BacktestServiceDeps

if TYPE_CHECKING:
    from forecasting_api.schemas import BacktestRequest, BacktestResponse


_DEPS: BacktestServiceDeps | None = None


def configure_backtest_service(deps: BacktestServiceDeps) -> None:
    global _DEPS
    _DEPS = deps


def _require_deps() -> BacktestServiceDeps:
    if _DEPS is None:
        raise RuntimeError("backtest service is not configured")
    return _DEPS


def build_run_backtest_request(deps: BacktestServiceDeps):
    def run_backtest_request(req: BacktestRequest) -> BacktestResponse:
        deps.require_monotonic_increasing(req.data)
        deps.require_trained_model(req.model_id)
        trained_models = deps.trained_models()
        trained = trained_models.get(req.model_id) if req.model_id else None
        if (
            trained
            and isinstance(trained, dict)
            and deps.assert_model_algo_available(trained.get("algo")).startswith("ridge")
        ):
            return deps.ridge_lags_backtest(req, trained=trained)
        if (
            trained
            and isinstance(trained, dict)
            and deps.assert_model_algo_available(trained.get("algo")) == "gbdt_hgb_v1"
        ):
            return deps.gbdt_backtest(req, trained=trained)
        if (
            trained
            and isinstance(trained, dict)
            and deps.assert_model_algo_available(trained.get("algo")) == "gbdt_afno_hybrid_v1"
        ):
            return deps.hybrid_backtest(req, trained=trained)
        if (
            trained
            and isinstance(trained, dict)
            and deps.assert_model_algo_available(trained.get("algo"))
            in {"afnocg2", "cifnocg2", "afnocg3", "afnocg3_v1"}
        ):
            return deps.torch_backtest(req, trained=trained)
        return deps.naive_backtest(req)

    return run_backtest_request


def build_run_backtest(deps: BacktestServiceDeps):
    run_backtest_request = build_run_backtest_request(deps)

    def run_backtest(req: BacktestRequest) -> BacktestResponse:
        result = run_backtest_request(req)
        with deps.start_run(
            run_name=f"forecasting-backtest-{req.metric}",
            tags={
                "flow": "backtest",
                "metric": req.metric,
                "model_id": str(req.model_id or "naive"),
            },
        ):
            deps.log_params(
                {
                    "metric": req.metric,
                    "horizon": req.horizon,
                    "folds": req.folds,
                    "model_id": req.model_id,
                }
            )
            deps.log_metrics({f"backtest.{key}": value for key, value in result.metrics.items()})
            deps.log_dict_artifact("backtest_request.json", req.model_dump(mode="json"))
            deps.log_dict_artifact(
                "backtest_result.json",
                result.model_dump(mode="json", exclude_none=True),
            )
        return result

    return run_backtest


def run_backtest_request(req: BacktestRequest) -> BacktestResponse:
    return build_run_backtest_request(_require_deps())(req)


def run_backtest(req: BacktestRequest) -> BacktestResponse:
    return build_run_backtest(_require_deps())(req)