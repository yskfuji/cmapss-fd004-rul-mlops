from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .file_store import load_json, update_json_file
from .metrics import record_model_promotion
from .mlflow_runs import log_dict_artifact, log_metrics, log_params, start_run


@dataclass(frozen=True)
class PromotionDecision:
    model_id: str
    target_stage: str
    approved: bool
    reasons: list[str]
    metrics: dict[str, float]
    promoted_at: str


def _registry_path() -> Path:
    raw = os.getenv("RULFM_MODEL_PROMOTION_REGISTRY_PATH", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent / "data" / "model_promotions.json"


def load_promotion_registry() -> list[dict[str, Any]]:
    path = _registry_path()
    payload: list[dict[str, Any]] = load_json(path, [])
    return payload if isinstance(payload, list) else []


def save_promotion_registry(entries: list[dict[str, Any]]) -> None:
    path = _registry_path()
    empty_entries: list[dict[str, Any]] = []
    update_json_file(path, default=empty_entries, updater=lambda _: entries)


def _validated_metrics(metrics: dict[str, float]) -> dict[str, float]:
    required = {"coverage", "rmse", "drift_score"}
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"missing metrics: {', '.join(missing)}")

    out: dict[str, float] = {}
    for key, raw_value in metrics.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"metric must be finite: {key}")
        if key == "coverage" and not (0.0 <= value <= 1.0):
            raise ValueError("coverage must be between 0 and 1")
        if key in {"rmse", "drift_score"} and value < 0.0:
            raise ValueError(f"{key} must be non-negative")
        out[key] = value
    return out


def evaluate_promotion_candidate(
    model_id: str,
    metrics: dict[str, float],
    *,
    target_stage: str,
    minimum_coverage: float = 0.9,
    maximum_rmse: float = 20.0,
    maximum_drift_score: float = 0.2,
) -> PromotionDecision:
    validated_metrics = _validated_metrics(metrics)
    reasons: list[str] = []
    coverage = validated_metrics["coverage"]
    rmse = validated_metrics["rmse"]
    drift_score = validated_metrics["drift_score"]

    if coverage < minimum_coverage:
        reasons.append(f"coverage_below_threshold:{coverage:.3f}<{minimum_coverage:.3f}")
    if rmse > maximum_rmse:
        reasons.append(f"rmse_above_threshold:{rmse:.3f}>{maximum_rmse:.3f}")
    if drift_score > maximum_drift_score:
        reasons.append(f"drift_above_threshold:{drift_score:.3f}>{maximum_drift_score:.3f}")

    approved = not reasons
    return PromotionDecision(
        model_id=model_id,
        target_stage=target_stage,
        approved=approved,
        reasons=reasons or ["approved"],
        metrics=validated_metrics,
        promoted_at=datetime.now(UTC).isoformat(),
    )


def promote_model(
    model_id: str,
    metrics: dict[str, float],
    *,
    target_stage: str = "staging",
) -> PromotionDecision:
    decision = evaluate_promotion_candidate(model_id, metrics, target_stage=target_stage)
    path = _registry_path()
    empty_entries: list[dict[str, Any]] = []
    update_json_file(
        path,
        default=empty_entries,
        updater=lambda current: [*current, asdict(decision)],
    )

    with start_run(
        run_name=f"model-promotion-{model_id}",
        tags={"flow": "promotion", "model_id": model_id, "target_stage": target_stage},
    ):
        log_params(
            {
                "model_id": model_id,
                "target_stage": target_stage,
                "approved": decision.approved,
            }
        )
        log_metrics({f"promotion.{key}": value for key, value in decision.metrics.items()})
        log_dict_artifact("promotion_decision.json", asdict(decision))

    if decision.approved:
        record_model_promotion(target_stage)
    return decision
