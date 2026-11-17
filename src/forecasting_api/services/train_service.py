from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from forecasting_api.schemas import TrainResponse

from .runtime import TrainServiceDeps

if TYPE_CHECKING:
    from forecasting_api.schemas import TrainRequest


_service_deps: TrainServiceDeps | None = None


def configure_train_service(deps: TrainServiceDeps) -> None:
    global _service_deps
    _service_deps = deps


def _require_deps() -> TrainServiceDeps:
    if _service_deps is None:
        raise RuntimeError("train service is not configured")
    return _service_deps


def build_run_train(deps: TrainServiceDeps):
    def run_train(req: TrainRequest) -> TrainResponse:
        deps.require_monotonic_increasing(req.data)
        model_id = f"model_{uuid4().hex}"
        created_at = datetime.now(UTC).isoformat()
        memo = (req.model_name or "").strip()
        base = deps.normalize_base_model_name(req.base_model)
        algo = str(req.algo or "").strip().lower() or (
            "ridge_lags_v1"
            if base in {"default", "ridge", "ridge_lags", "ridge_lags_v1"}
            else "gbdt_hgb_v1"
            if base in {"gbdt", "gbdt_hgb", "gbdt_hgb_v1", "hgb"}
            else "naive"
        )
        algo = deps.assert_model_algo_available(algo)

        entry: dict[str, Any] = {
            "model_id": model_id,
            "created_at": created_at,
            "memo": memo,
            "algo": algo,
            "base_model": base,
        }

        if algo.startswith("ridge"):
            entry["state"] = deps.fit_ridge_lags_model(req)
        elif algo == "gbdt_hgb_v1":
            try:
                entry.update(deps.train_public_gbdt_entry(req, model_id=model_id))
            except ValueError as exc:
                api_error_cls = deps.api_error_cls
                raise api_error_cls(
                    status_code=400,
                    error_code="V01",
                    message="入力が不正です",
                    details={
                        "error": str(exc),
                        "next_action": "x に数値特徴量を含む学習データを渡してください",
                    },
                ) from exc
        elif algo == "gbdt_afno_hybrid_v1":
            try:
                entry.update(deps.train_hybrid_entry(req, model_id=model_id))
            except ValueError as exc:
                api_error_cls = deps.api_error_cls
                raise api_error_cls(
                    status_code=400,
                    error_code="V01",
                    message="入力が不正です",
                    details={
                        "error": str(exc),
                        "next_action": "数値特徴量を含む学習データ量を増やしてください",
                    },
                ) from exc
        elif algo in {"afnocg2", "cifnocg2", "afnocg3", "afnocg3_v1"}:
            from forecasting_api.torch_forecasters import train_univariate_torch_forecaster

            ys_by_series: dict[str, list[float]] = {}
            records_by_series: dict[str, list[dict[str, Any]]] = {}
            for row in req.data:
                ys_by_series.setdefault(row.series_id, []).append(float(row.y))
                records_by_series.setdefault(row.series_id, []).append(
                    {
                        "timestamp": row.timestamp.isoformat(),
                        "y": float(row.y),
                        "x": dict(row.x or {}),
                    }
                )

            try:
                artifact = train_univariate_torch_forecaster(
                    algo=algo,
                    ys_by_series=ys_by_series,
                    records_by_series=records_by_series,
                    training_hours=req.training_hours,
                    context_len=30,
                    max_exogenous_features=24,
                    prefer_exogenous=True,
                    device=None,
                )
            except ValueError as exc:
                api_error_cls = deps.api_error_cls
                raise api_error_cls(
                    status_code=400,
                    error_code="V01",
                    message="入力が不正です",
                    details={
                        "error": str(exc),
                        "next_action": "学習データ量（系列長）を増やしてください",
                    },
                ) from exc

            snapshot_path = deps.model_artifact_dir(model_id) / "snapshot.json"
            deps.write_json(snapshot_path, artifact.snapshot)

            import torch

            weights_path = deps.model_artifact_dir(model_id) / "weights.pt"
            cast(Any, torch).save({"state_dict": artifact.state_dict}, weights_path)

            entry.update(
                {
                    "context_len": int(artifact.context_len),
                    "input_dim": int(artifact.input_dim),
                    "pooled_residuals": list(artifact.pooled_residuals),
                    "artifact": {
                        "snapshot_json": deps.artifact_relpath(model_id, "snapshot.json"),
                        "weights_pt": deps.artifact_relpath(model_id, "weights.pt"),
                    },
                }
            )

        trained_models = deps.trained_models()
        trained_models[model_id] = entry
        deps.save_trained_model(entry)

        with deps.start_run(
            run_name=f"forecasting-train-{algo}",
            tags={"flow": "train", "algo": algo, "model_id": model_id},
        ):
            deps.log_params(
                {
                    "algo": algo,
                    "model_id": model_id,
                    "base_model": base,
                    "training_hours": req.training_hours,
                    "context_len": entry.get("context_len"),
                    "input_dim": entry.get("input_dim"),
                    "records": len(req.data),
                }
            )
            pooled = cast(list[Any], entry.get("pooled_residuals") or [])
            deps.log_metrics({"residuals.count": len(pooled)})
            deps.log_dict_artifact("train_metadata.json", {"model_id": model_id, "entry": entry})
            artifact_meta = cast(dict[str, Any], entry.get("artifact") or {})
            snapshot_rel = (
                artifact_meta.get("snapshot_json")
                if isinstance(artifact_meta.get("snapshot_json"), str)
                else None
            )
            weights_rel = (
                artifact_meta.get("weights_pt")
                if isinstance(artifact_meta.get("weights_pt"), str)
                else None
            )
            if snapshot_rel:
                deps.log_artifact(deps.artifact_abspath(snapshot_rel))
            if weights_rel:
                deps.log_artifact(deps.artifact_abspath(weights_rel))
            gbdt_rel = (
                artifact_meta.get("gbdt_joblib")
                if isinstance(artifact_meta.get("gbdt_joblib"), str)
                else None
            )
            if gbdt_rel:
                deps.log_artifact(deps.artifact_abspath(gbdt_rel))
            hybrid_rel = (
                artifact_meta.get("hybrid_json")
                if isinstance(artifact_meta.get("hybrid_json"), str)
                else None
            )
            if hybrid_rel:
                deps.log_artifact(deps.artifact_abspath(hybrid_rel))

        return TrainResponse(model_id=model_id, message="accepted")

    return run_train


def run_train(req: TrainRequest) -> TrainResponse:
    return build_run_train(_require_deps())(req)
