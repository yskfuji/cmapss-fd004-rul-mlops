from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from forecasting_api.config import env_bool, env_first
from forecasting_api.domain import stable_models
from forecasting_api.hybrid_xai_uncertainty import (
    apply_soft_gate_envelope_interval,
    condition_advantage_map,
    gate_step_payload,
    interval_overlap_ratio,
    normalize_advantage_lookup,
    soft_gate_feature_scales,
    soft_gate_outputs,
    soft_gate_weight_entropy,
)
from forecasting_api.schemas import TimeSeriesRecord, TrainRequest


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def experimental_models_enabled() -> bool:
    return env_bool(
        "RULFM_ENABLE_EXPERIMENTAL_MODELS",
        default=False,
    )


def is_experimental_model_algo(raw: str | None) -> bool:
    normalized = str(raw or "").strip().lower()
    return normalized in {
        "afnocg2",
        "afnocg3",
        "afnocg3_v1",
        "cifnocg2",
        "gbdt_afno_hybrid_v1",
        "stardast2_v5",
    }


def assert_model_algo_available(raw: str | None, *, api_error_cls: type[Exception]) -> str:
    normalized = str(raw or "naive").strip().lower() or "naive"
    if is_experimental_model_algo(normalized) and not experimental_models_enabled():
        raise api_error_cls(
            status_code=400,
            error_code="M01",
            message="experimental algorithm は公開APIでは無効です",
            details={
                "algo": normalized,
                "next_action": (
                    "RULFM_ENABLE_EXPERIMENTAL_MODELS=1 を設定するか、"
                    "gbdt_hgb_v1 / ridge_lags_v1 / naive を使用してください"
                ),
            },
        )
    return normalized


def normalize_base_model_name(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    return value or "default"


def build_request_training_payload(
    records: list[TimeSeriesRecord],
) -> tuple[dict[str, list[float]], dict[str, list[dict[str, Any]]]]:
    ys_by_series: dict[str, list[float]] = {}
    records_by_series: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        ys_by_series.setdefault(record.series_id, []).append(float(record.y))
        records_by_series.setdefault(record.series_id, []).append(
            {
                "timestamp": record.timestamp,
                "y": float(record.y),
                "x": _as_dict(record.x),
            }
        )
    return ys_by_series, records_by_series


def select_series_feature_keys(
    records_by_series: dict[str, list[dict[str, Any]]], *, max_features: int = 8
) -> list[str]:
    scores: dict[str, int] = {}
    for rows in records_by_series.values():
        for row in rows:
            for key, value in _as_dict(row.get("x")).items():
                if not isinstance(key, str):
                    continue
                if not isinstance(value, int | float) or not math.isfinite(float(value)):
                    continue
                if key in {"cycle", "rul", "y"}:
                    continue
                scores[key] = scores.get(key, 0) + 1
    preferred = [
        key
        for key, _count in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if key.startswith("sensor_") or key.startswith("op_setting_")
    ]
    if preferred:
        return preferred[: max(1, max_features)]
    return [
        key for key, _count in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ][: max(1, max_features)]


def hybrid_context_len(records_by_series: dict[str, list[dict[str, Any]]]) -> int:
    max_len = max((len(rows) for rows in records_by_series.values()), default=0)
    if max_len <= 3:
        return 1
    if max_len <= 8:
        return 3
    if max_len < 24:
        return 7
    return 30


def build_hgb_supervised_rows(
    records_by_series: dict[str, list[dict[str, Any]]],
    *,
    context_len: int,
    feature_keys: list[str],
) -> list[dict[str, Any]]:
    def _timestamp_sort_key(row: dict[str, Any]) -> tuple[int, datetime]:
        raw = row.get("timestamp")
        if isinstance(raw, datetime):
            return (0, raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC))
        if isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return (0, parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC))
            except ValueError:
                pass
        return (1, datetime.max.replace(tzinfo=UTC))

    rows_out: list[dict[str, Any]] = []
    for series_id, rows in records_by_series.items():
        ordered = sorted(rows, key=_timestamp_sort_key)
        if len(ordered) <= context_len:
            continue
        for idx in range(context_len, len(ordered)):
            context_rows = ordered[idx - context_len : idx]
            features: list[float] = []
            for ctx_row in context_rows:
                x_dict = _as_dict(ctx_row.get("x"))
                features.extend(
                    float(x_dict.get(key, 0.0))
                    if isinstance(x_dict.get(key, 0.0), int | float)
                    else 0.0
                    for key in feature_keys
                )
            rows_out.append(
                {
                    "series_id": str(series_id),
                    "features": features,
                    "y": float(ordered[idx].get("y") or 0.0),
                    "context_records": [dict(row) for row in context_rows],
                    "future_row": dict(ordered[idx]),
                }
            )
    return rows_out


def split_train_valid_indices(total: int, *, min_valid: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if total <= min_valid + 8:
        idx = np.arange(total, dtype=int)
        return idx, np.zeros((0,), dtype=int)
    valid_n = max(min_valid, min(32, total // 5))
    train_end = max(1, total - valid_n)
    return np.arange(train_end, dtype=int), np.arange(train_end, total, dtype=int)


def save_joblib_artifact(path: Path, payload: Any) -> None:
    import joblib

    joblib.dump(payload, path)


def load_joblib_artifact(path: Path) -> Any:
    import joblib

    return joblib.load(path)


def top_feature_summary(
    feature_scores: dict[str, float], *, method: str, sample_count: int, top_k: int = 5
) -> dict[str, Any]:
    ranked = sorted(
        (
            (str(key), float(value))
            for key, value in feature_scores.items()
            if math.isfinite(float(value))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "method": method,
        "sample_count": int(sample_count),
        "top_features": [
            {"feature": key, "importance": float(score), "rank": idx + 1}
            for idx, (key, score) in enumerate(ranked[: max(1, top_k)])
        ],
        "feature_importance": {key: float(score) for key, score in ranked},
    }


def sigmoid(value: float) -> float:
    clipped = max(-60.0, min(60.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def fit_hgb_forecaster(
    *,
    records_by_series: dict[str, list[dict[str, Any]]],
    context_len: int,
    feature_keys: list[str],
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    supervised = build_hgb_supervised_rows(
        records_by_series, context_len=context_len, feature_keys=feature_keys
    )
    if not supervised:
        raise ValueError("Insufficient training data for hybrid GBDT model.")
    x_all = np.asarray([row["features"] for row in supervised], dtype=float)
    y_all = np.asarray([float(row["y"]) for row in supervised], dtype=float)
    train_idx, valid_idx = split_train_valid_indices(int(x_all.shape[0]))
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_valid = x_all[valid_idx] if valid_idx.size > 0 else x_train
    y_valid = y_all[valid_idx] if valid_idx.size > 0 else y_train
    params = {
        "max_iter": 240,
        "learning_rate": 0.05,
        "max_depth": 5,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 5,
    }
    point_model = HistGradientBoostingRegressor(random_state=0, early_stopping=False, **params)
    lower_model = HistGradientBoostingRegressor(
        random_state=0, early_stopping=False, loss="quantile", quantile=0.05, **params
    )
    upper_model = HistGradientBoostingRegressor(
        random_state=0, early_stopping=False, loss="quantile", quantile=0.95, **params
    )
    point_model.fit(x_train, y_train)
    lower_model.fit(x_train, y_train)
    upper_model.fit(x_train, y_train)
    valid_pred = np.asarray(point_model.predict(x_valid), dtype=float).reshape(-1)
    valid_lower_raw = np.asarray(lower_model.predict(x_valid), dtype=float).reshape(-1)
    valid_upper_raw = np.asarray(upper_model.predict(x_valid), dtype=float).reshape(-1)
    valid_lower = np.minimum(valid_lower_raw, valid_upper_raw)
    valid_upper = np.maximum(valid_lower_raw, valid_upper_raw)
    residuals = [float(v) for v in np.abs(valid_pred - y_valid).tolist() if math.isfinite(float(v))]
    raw_importance = permutation_importance(
        point_model,
        x_valid,
        y_valid,
        n_repeats=5,
        random_state=0,
        scoring="neg_root_mean_squared_error",
    )
    grouped_importance: dict[str, list[float]] = {}
    importance_values = (
        np.asarray(raw_importance.importances_mean, dtype=float).reshape(-1).tolist()
    )
    for idx, score in enumerate(importance_values):
        key = feature_keys[idx % len(feature_keys)] if feature_keys else str(idx)
        grouped_importance.setdefault(key, []).append(abs(float(score)))
    feature_scores = {
        key: float(np.mean(values)) for key, values in grouped_importance.items() if values
    }
    baseline_map: dict[str, float] = {}
    for key in feature_keys:
        values = [
            float(_as_dict(row.get("x")).get(key, 0.0))
            for rows in records_by_series.values()
            for row in rows
            if isinstance(_as_dict(row.get("x")).get(key), int | float)
        ]
        baseline_map[key] = float(np.median(np.asarray(values, dtype=float))) if values else 0.0
    valid_rows = [
        supervised[int(idx)]
        for idx in (valid_idx.tolist() if valid_idx.size > 0 else train_idx.tolist())
    ]
    return {
        "bundle": {"point": point_model, "q05": lower_model, "q95": upper_model},
        "snapshot": {
            "context_len": int(context_len),
            "feature_keys": list(feature_keys),
            "feature_source": "x",
            "validation_rmse": float(np.sqrt(np.mean((valid_pred - y_valid) ** 2)))
            if y_valid.size > 0
            else None,
            "interval_quantiles": [0.05, 0.95],
            "feature_baseline": baseline_map,
            "global_importance": top_feature_summary(
                feature_scores,
                method="permutation_importance_rmse_v1",
                sample_count=int(x_valid.shape[0]),
            ),
        },
        "pooled_residuals": residuals[-500:],
        "valid_rows": valid_rows,
        "valid_pred": valid_pred.tolist(),
        "valid_lower": valid_lower.tolist(),
        "valid_upper": valid_upper.tolist(),
    }


def predict_hgb_next(
    bundle: dict[str, Any], *, context_records: list[dict[str, Any]], feature_keys: list[str]
) -> tuple[float, float, float]:
    return stable_models.predict_hgb_next(
        bundle,
        context_records=context_records,
        feature_keys=feature_keys,
    )


def hybrid_condition_cluster_key(record: dict[str, Any]) -> str:
    x_dict = _as_dict(record.get("x"))
    op_value = (
        float(x_dict.get("op_setting_1", 0.0))
        if isinstance(x_dict.get("op_setting_1"), int | float)
        else 0.0
    )
    sensor_1 = (
        float(x_dict.get("sensor_1", 0.0))
        if isinstance(x_dict.get("sensor_1"), int | float)
        else 0.0
    )
    sensor_2 = (
        float(x_dict.get("sensor_2", 0.0))
        if isinstance(x_dict.get("sensor_2"), int | float)
        else 0.0
    )
    op_bucket = int(round(op_value))
    s1_bucket = int(math.floor(sensor_1 / 10.0))
    s2_bucket = int(math.floor(sensor_2 / 5.0))
    return f"{op_bucket}|{s1_bucket}|{s2_bucket}"


def hybrid_interval_overlap_ratio(
    g_lower: float,
    g_upper: float,
    a_lower: float,
    a_upper: float,
) -> float:
    return interval_overlap_ratio(g_lower, g_upper, a_lower, a_upper)


def hybrid_soft_gate_feature_scales(
    gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]
) -> dict[str, float]:
    return soft_gate_feature_scales(gbdt_outputs, afno_outputs)


def hybrid_normalize_advantage_lookup(grouped: dict[str, list[float]]) -> dict[str, float]:
    return normalize_advantage_lookup(grouped)


def hybrid_condition_advantage_map(
    gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]
) -> dict[str, float]:
    return condition_advantage_map(gbdt_outputs, afno_outputs)


def hybrid_soft_gate_outputs(
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    temperature: float,
    tau: float,
    coef_delta: float,
    coef_overlap: float,
    coef_width: float,
    coef_tail: float,
    coef_condition: float,
    condition_advantage: dict[str, float] | None = None,
) -> dict[str, list[float]]:
    return soft_gate_outputs(
        gbdt_outputs,
        afno_outputs,
        temperature=temperature,
        tau=tau,
        coef_delta=coef_delta,
        coef_overlap=coef_overlap,
        coef_width=coef_width,
        coef_tail=coef_tail,
        coef_condition=coef_condition,
        condition_advantage=condition_advantage,
    )


def hybrid_apply_envelope_interval(
    soft_outputs: dict[str, list[float]],
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    interval_scale: float,
) -> dict[str, list[float]]:
    return apply_soft_gate_envelope_interval(
        soft_outputs,
        gbdt_outputs,
        afno_outputs,
        interval_scale=interval_scale,
    )


def hybrid_metrics_from_outputs(outputs: dict[str, list[float]]) -> dict[str, float]:
    y_true = [float(v) for v in outputs.get("y_true") or []]
    y_pred = [float(v) for v in outputs.get("y_pred") or []]
    lower = [float(v) for v in outputs.get("lower") or []]
    upper = [float(v) for v in outputs.get("upper") or []]
    if (
        not y_true
        or len(y_true) != len(y_pred)
        or len(lower) != len(y_pred)
        or len(upper) != len(y_pred)
    ):
        return {
            "rmse": float("nan"),
            "nasa_score": float("nan"),
            "cov90": float("nan"),
            "width90": float("nan"),
        }
    rmse = float(
        math.sqrt(
            sum((pred - truth) ** 2 for truth, pred in zip(y_true, y_pred, strict=True))
            / len(y_true)
        )
    )
    nasa_score = 0.0
    for truth, pred in zip(y_true, y_pred, strict=True):
        err = float(pred) - float(truth)
        nasa_score += math.exp((-err) / 13.0) - 1.0 if err < 0 else math.exp(err / 10.0) - 1.0
    cov = float(
        sum(
            1
            for truth, lo, hi in zip(y_true, lower, upper, strict=True)
            if float(lo) <= float(truth) <= float(hi)
        )
        / len(y_true)
    )
    width = float(
        sum(max(0.0, float(hi) - float(lo)) for lo, hi in zip(lower, upper, strict=True))
        / len(lower)
    )
    return {"rmse": rmse, "nasa_score": nasa_score, "cov90": cov, "width90": width}


def hybrid_soft_gate_candidate_rank(
    *,
    point_metrics: dict[str, float],
    interval_coverage: float,
    interval_width: float,
    avg_entropy: float,
    target_cov: float,
) -> tuple[float, ...]:
    coverage_shortfall = (
        max(0.0, float(target_cov) - float(interval_coverage))
        if math.isfinite(float(interval_coverage))
        else float("inf")
    )
    feasible = 0.0 if coverage_shortfall <= 0.0 else 1.0
    point_objective = 0.25 * max(float(point_metrics.get("rmse") or float("inf")), 0.0)
    point_objective += 1.0 * math.log1p(
        max(0.0, float(point_metrics.get("nasa_score") or float("inf")))
    )
    return (
        feasible,
        coverage_shortfall,
        point_objective,
        float(point_metrics.get("nasa_score") or float("inf")),
        float(point_metrics.get("rmse") or float("inf")),
        max(0.0, float(avg_entropy)),
        float(interval_width),
    )


def hybrid_soft_gate_weight_entropy(weights: list[float]) -> float:
    return soft_gate_weight_entropy(weights)


def hybrid_optimize_interval_scale(
    soft_outputs: dict[str, list[float]],
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    target_cov: float,
) -> dict[str, Any]:
    candidate_scales = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
    best: dict[str, Any] | None = None
    for scale in candidate_scales:
        calibrated = hybrid_apply_envelope_interval(
            soft_outputs,
            gbdt_outputs,
            afno_outputs,
            interval_scale=float(scale),
        )
        metrics = hybrid_metrics_from_outputs(calibrated)
        coverage = float(metrics["cov90"])
        width = float(metrics["width90"])
        shortfall = (
            max(0.0, float(target_cov) - coverage)
            if math.isfinite(coverage)
            else float("inf")
        )
        rank = (0.0 if shortfall <= 0.0 else 1.0, shortfall, width)
        candidate: dict[str, Any] = {
            "scale": float(scale),
            "coverage": coverage,
            "width90": width,
            "rank": rank,
        }
        if best is None or tuple(candidate["rank"]) < tuple(best["rank"]):
            best = candidate
    return best or {
        "scale": 1.0,
        "coverage": float("nan"),
        "width90": float("nan"),
        "rank": (1.0, float("inf"), float("inf")),
    }


def hybrid_optimize_soft_gate_strategy(
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    target_cov: float,
    use_condition_clusters: bool,
    interval_gbdt_outputs: dict[str, list[float]] | None = None,
    interval_afno_outputs: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    scales = hybrid_soft_gate_feature_scales(gbdt_outputs, afno_outputs)
    tau_candidates_raw = [0.0] + [
        abs(float(a) - float(g))
        for g, a in zip(
            gbdt_outputs.get("y_pred") or [],
            afno_outputs.get("y_pred") or [],
            strict=False,
        )
    ]
    tau_grid = sorted(
        {
            round(float(stable_models.quantile_nearest_rank(tau_candidates_raw, q)), 6)
            for q in (0.0, 0.1, 0.25, 0.5, 0.75)
        }
        | {0.0}
    )
    temperature_grid = [0.05, 0.1, 0.2, 0.35]
    coef_delta_grid = [0.5, 1.0, 2.0]
    coef_overlap_grid = [0.0, 0.5, 1.0]
    coef_width_grid = [-0.5, 0.0, 0.5]
    coef_tail_grid = [-0.5, 0.0, 0.5]
    coef_condition_grid = [0.0, 0.5, 1.0] if use_condition_clusters else [0.0]
    max_trials_raw = env_first("RULFM_HYBRID_GRID_MAX_TRIALS")
    try:
        max_trials = int(max_trials_raw) if max_trials_raw is not None else 384
    except ValueError:
        max_trials = 384
    max_trials = max(1, min(10_000, max_trials))
    condition_advantage = (
        hybrid_condition_advantage_map(gbdt_outputs, afno_outputs)
        if use_condition_clusters
        else {}
    )
    best: dict[str, Any] | None = None
    evaluated_trials = 0
    grid = itertools.product(
        temperature_grid,
        tau_grid,
        coef_delta_grid,
        coef_overlap_grid,
        coef_width_grid,
        coef_tail_grid,
        coef_condition_grid,
    )
    for temperature, tau, coef_delta, coef_overlap, coef_width, coef_tail, coef_condition in grid:
        evaluated_trials += 1
        if evaluated_trials > max_trials:
            break
        soft_outputs = hybrid_soft_gate_outputs(
            gbdt_outputs,
            afno_outputs,
            temperature=float(temperature),
            tau=float(tau),
            coef_delta=float(coef_delta),
            coef_overlap=float(coef_overlap),
            coef_width=float(coef_width),
            coef_tail=float(coef_tail),
            coef_condition=float(coef_condition),
            condition_advantage=condition_advantage,
        )
        point_metrics = hybrid_metrics_from_outputs(soft_outputs)
        avg_entropy = hybrid_soft_gate_weight_entropy(
            [float(v) for v in soft_outputs.get("afno_weight") or []]
        )
        interval_scale = 1.0
        interval_coverage = float("nan")
        interval_width = float("nan")
        if interval_gbdt_outputs is not None and interval_afno_outputs is not None:
            interval_soft_outputs = hybrid_soft_gate_outputs(
                interval_gbdt_outputs,
                interval_afno_outputs,
                temperature=float(temperature),
                tau=float(tau),
                coef_delta=float(coef_delta),
                coef_overlap=float(coef_overlap),
                coef_width=float(coef_width),
                coef_tail=float(coef_tail),
                coef_condition=float(coef_condition),
                condition_advantage=condition_advantage,
            )
            interval_meta = hybrid_optimize_interval_scale(
                interval_soft_outputs,
                interval_gbdt_outputs,
                interval_afno_outputs,
                target_cov=target_cov,
            )
            interval_scale = float(interval_meta.get("scale") or 1.0)
            interval_coverage = float(interval_meta.get("coverage") or float("nan"))
            interval_width = float(interval_meta.get("width90") or float("nan"))
        rank = hybrid_soft_gate_candidate_rank(
            point_metrics=point_metrics,
            interval_coverage=interval_coverage,
            interval_width=interval_width,
            avg_entropy=avg_entropy,
            target_cov=target_cov,
        )
        candidate: dict[str, Any] = {
            "mode": "risk_aware",
            "temperature": float(temperature),
            "tau": float(tau),
            "coef_delta": float(coef_delta),
            "coef_overlap": float(coef_overlap),
            "coef_width": float(coef_width),
            "coef_tail": float(coef_tail),
            "coef_condition": float(coef_condition),
            "delta_scale": float(scales.get("delta_scale") or 1.0),
            "width_scale": float(scales.get("width_scale") or 1.0),
            "condition_advantage": condition_advantage,
            "interval_scale": interval_scale,
            "interval_calib_cov90": interval_coverage,
            "interval_calib_width90": interval_width,
            "soft_gate_avg_entropy": avg_entropy,
            "soft_gate_avg_afno_weight": float(np.mean(soft_outputs.get("afno_weight") or [0.0])),
            "grid_trials_evaluated": min(evaluated_trials, max_trials),
            "grid_trials_max": max_trials,
            "rank": rank,
        }
        if best is None or tuple(candidate["rank"]) < tuple(best["rank"]):
            best = candidate
    return best or {
        "mode": "risk_aware",
        "temperature": 0.35,
        "tau": 0.0,
        "coef_delta": 1.0,
        "coef_overlap": 0.0,
        "coef_width": 0.0,
        "coef_tail": 0.0,
        "coef_condition": 0.0,
        "delta_scale": 1.0,
        "width_scale": 1.0,
        "condition_advantage": {},
        "interval_scale": 1.0,
        "interval_calib_cov90": float("nan"),
        "interval_calib_width90": float("nan"),
        "soft_gate_avg_entropy": float("nan"),
        "soft_gate_avg_afno_weight": float("nan"),
        "grid_trials_evaluated": 0,
        "grid_trials_max": max_trials,
        "rank": (1.0, float("inf"), float("inf"), float("inf")),
    }


def fit_hybrid_gate(
    *,
    gbdt_bundle: dict[str, Any],
    gbdt_snapshot: dict[str, Any],
    valid_rows: list[dict[str, Any]],
    afno_snapshot: dict[str, Any],
    afno_state_dict: dict[str, Any],
    afno_pooled_residuals: list[float],
    feature_keys: list[str],
) -> dict[str, Any]:
    from forecasting_api.torch_forecasters import (
        forecast_univariate_torch,
        forecast_univariate_torch_with_details,
    )

    sample_rows = valid_rows[: max(1, min(len(valid_rows), 48))]
    if not sample_rows:
        return {
            "mode": "risk_aware",
            "temperature": 0.35,
            "coef_delta": 1.0,
            "coef_overlap": 0.0,
            "coef_width": 0.0,
            "coef_tail": 0.0,
            "coef_condition": 0.0,
            "tau": 0.0,
            "condition_advantage": {},
            "interval_scale": 1.0,
            "interval_calib_cov90": float("nan"),
            "interval_calib_width90": float("nan"),
            "afno_occlusion_global": {
                "method": "occlusion_delta_v1",
                "sample_count": 0,
                "top_features": [],
                "feature_importance": {},
            },
        }
    gbdt_outputs: dict[str, list[Any]] = {
        "y_true": [],
        "y_pred": [],
        "lower": [],
        "upper": [],
        "condition_key": [],
        "tail_pos": [],
    }
    afno_outputs: dict[str, list[Any]] = {
        "y_true": [],
        "y_pred": [],
        "lower": [],
        "upper": [],
        "condition_key": [],
        "tail_pos": [],
    }
    occlusion_scores: dict[str, list[float]] = {}
    feature_baseline = _as_dict(gbdt_snapshot.get("feature_baseline"))
    afno_qhat = (
        stable_models.quantile_nearest_rank(
            [float(v) for v in afno_pooled_residuals],
            0.9,
        )
        if afno_pooled_residuals
        else 0.0
    )
    for row in sample_rows:
        context_records = [dict(item) for item in _as_list(row.get("context_records"))]
        future_row = _as_dict(row.get("future_row"))
        g_pred, g_low, g_high = predict_hgb_next(
            gbdt_bundle, context_records=context_records, feature_keys=feature_keys
        )
        afno_pred = float(
            forecast_univariate_torch(
                algo="afnocg3_v1",
                snapshot=afno_snapshot,
                state_dict=afno_state_dict,
                context_records=context_records,
                future_feature_rows=[future_row],
                horizon=1,
                device=None,
            )[0]
        )
        gbdt_outputs["y_true"].append(float(row.get("y") or 0.0))
        gbdt_outputs["y_pred"].append(float(g_pred))
        gbdt_outputs["lower"].append(float(g_low))
        gbdt_outputs["upper"].append(float(g_high))
        gbdt_outputs["condition_key"].append(hybrid_condition_cluster_key(future_row))
        gbdt_outputs["tail_pos"].append(
            float(len(gbdt_outputs["tail_pos"]) / max(len(sample_rows) - 1, 1))
        )
        afno_outputs["y_true"].append(float(row.get("y") or 0.0))
        afno_outputs["y_pred"].append(float(afno_pred))
        afno_outputs["lower"].append(float(afno_pred - afno_qhat))
        afno_outputs["upper"].append(float(afno_pred + afno_qhat))
        afno_outputs["condition_key"].append(hybrid_condition_cluster_key(future_row))
        afno_outputs["tail_pos"].append(
            float(len(afno_outputs["tail_pos"]) / max(len(sample_rows) - 1, 1))
        )
        details = forecast_univariate_torch_with_details(
            algo="afnocg3_v1",
            snapshot=afno_snapshot,
            state_dict=afno_state_dict,
            context_records=context_records,
            future_feature_rows=[future_row],
            horizon=1,
            device=None,
            occlusion_feature_keys=feature_keys,
            occlusion_baseline={str(k): float(v) for k, v in feature_baseline.items()},
            occlusion_top_k=5,
        )
        feature_importance = _as_dict(
            _as_dict(_as_dict(details.get("occlusion")).get("global")).get(
                "feature_importance"
            )
        )
        for key, value in feature_importance.items():
            occlusion_scores.setdefault(str(key), []).append(float(value))
    split_idx = max(4, int(round(len(sample_rows) * 0.6)))
    split_idx = (
        min(split_idx, max(1, len(sample_rows) - 2))
        if len(sample_rows) > 2
        else len(sample_rows)
    )

    def _subset(outputs: dict[str, list[float]], start: int, end: int) -> dict[str, list[float]]:
        return {
            key: [values[idx] for idx in range(start, min(end, len(values)))]
            for key, values in outputs.items()
        }

    calib_gbdt = _subset(gbdt_outputs, 0, split_idx)
    calib_afno = _subset(afno_outputs, 0, split_idx)
    interval_gbdt = (
        _subset(gbdt_outputs, split_idx, len(sample_rows))
        if split_idx < len(sample_rows)
        else calib_gbdt
    )
    interval_afno = (
        _subset(afno_outputs, split_idx, len(sample_rows))
        if split_idx < len(sample_rows)
        else calib_afno
    )
    gate_meta = hybrid_optimize_soft_gate_strategy(
        calib_gbdt,
        calib_afno,
        target_cov=0.93,
        use_condition_clusters=True,
        interval_gbdt_outputs=interval_gbdt,
        interval_afno_outputs=interval_afno,
    )
    ranked_occlusion = {
        key: float(np.mean(values))
        for key, values in occlusion_scores.items()
        if values
    }
    gate_meta["afno_occlusion_global"] = top_feature_summary(
        ranked_occlusion,
        method="occlusion_delta_v1",
        sample_count=len(sample_rows),
    )
    return gate_meta


def hybrid_gate_step_payload(
    *,
    g_pred: float,
    g_lower: float,
    g_upper: float,
    a_pred: float,
    a_lower: float,
    a_upper: float,
    gate_meta: dict[str, Any],
    condition_key: str,
    tail_pos: float,
) -> dict[str, float]:
    return gate_step_payload(
        g_pred=g_pred,
        g_lower=g_lower,
        g_upper=g_upper,
        a_pred=a_pred,
        a_lower=a_lower,
        a_upper=a_upper,
        gate_meta=gate_meta,
        condition_key=condition_key,
        tail_pos=tail_pos,
    )


def train_hybrid_entry(
    req: TrainRequest,
    *,
    model_id: str,
    model_artifact_dir: Callable[[str], Path],
    write_json: Callable[[Path, dict[str, Any]], None],
    artifact_relpath: Callable[[str, str], str],
) -> dict[str, Any]:
    from forecasting_api.torch_forecasters import (
        forecast_univariate_torch,
        train_univariate_torch_forecaster,
    )

    ys_by_series, records_by_series = build_request_training_payload(req.data)
    context_len = hybrid_context_len(records_by_series)
    feature_keys = select_series_feature_keys(records_by_series, max_features=24)
    if not feature_keys:
        raise ValueError("Hybrid training requires numeric exogenous features.")
    torch_artifact = train_univariate_torch_forecaster(
        algo="afnocg3_v1",
        ys_by_series=ys_by_series,
        records_by_series=records_by_series,
        training_hours=req.training_hours,
        context_len=context_len,
        max_exogenous_features=24,
        prefer_exogenous=True,
        device=None,
    )
    gbdt_artifact = fit_hgb_forecaster(
        records_by_series=records_by_series,
        context_len=int(torch_artifact.context_len),
        feature_keys=feature_keys,
    )
    gate_config = fit_hybrid_gate(
        gbdt_bundle=gbdt_artifact["bundle"],
        gbdt_snapshot=gbdt_artifact["snapshot"],
        valid_rows=gbdt_artifact["valid_rows"],
        afno_snapshot=torch_artifact.snapshot,
        afno_state_dict=torch_artifact.state_dict,
        afno_pooled_residuals=torch_artifact.pooled_residuals,
        feature_keys=feature_keys,
    )
    hybrid_residuals: list[float] = []
    for row in gbdt_artifact["valid_rows"][: max(1, min(len(gbdt_artifact["valid_rows"]), 64))]:
        context_records = [dict(item) for item in row.get("context_records") or []]
        future_row = _as_dict(row.get("future_row"))
        g_pred, g_low, g_high = predict_hgb_next(
            gbdt_artifact["bundle"],
            context_records=context_records,
            feature_keys=feature_keys,
        )
        a_pred = float(
            forecast_univariate_torch(
                algo="afnocg3_v1",
                snapshot=torch_artifact.snapshot,
                state_dict=torch_artifact.state_dict,
                context_records=context_records,
                future_feature_rows=[future_row],
                horizon=1,
                device=None,
            )[0]
        )
        delta = float(a_pred) - float(g_pred)
        term_delta = (
            -float(gate_config.get("coef_delta") or 0.8)
            * delta
            / max(float(gate_config.get("delta_scale") or 1.0), 1e-6)
        )
        term_width = (
            float(gate_config.get("coef_width") or 0.35)
            * (
                (float(g_high) - float(g_low))
                - (
                    2.0
                    * stable_models.quantile_nearest_rank(
                        torch_artifact.pooled_residuals,
                        0.9,
                    )
                    if torch_artifact.pooled_residuals
                    else 0.0
                )
            )
            / max(float(gate_config.get("width_scale") or 1.0), 1e-6)
        )
        weight = sigmoid(
            (term_delta + term_width)
            / max(float(gate_config.get("temperature") or 0.35), 1e-6)
        )
        hybrid_pred = (1.0 - weight) * float(g_pred) + weight * float(a_pred)
        hybrid_residuals.append(abs(float(row.get("y") or 0.0) - hybrid_pred))
    snap_path = model_artifact_dir(model_id) / "snapshot.json"
    write_json(snap_path, torch_artifact.snapshot)
    import torch

    weights_path = model_artifact_dir(model_id) / "weights.pt"
    torch.save({"state_dict": torch_artifact.state_dict}, weights_path)
    gbdt_path = model_artifact_dir(model_id) / "gbdt.joblib"
    save_joblib_artifact(gbdt_path, gbdt_artifact["bundle"])
    hybrid_meta = {
        "algo": "gbdt_afno_hybrid_v1",
        "context_len": int(torch_artifact.context_len),
        "feature_keys": list(feature_keys),
        "gbdt": gbdt_artifact["snapshot"],
        "gate": gate_config,
        "model_explainability": {
            "gate": {"method": "analytic_term_decomposition_v1"},
            "gbdt": gbdt_artifact["snapshot"].get("global_importance"),
            "afno": gate_config.get("afno_occlusion_global"),
        },
    }
    hybrid_meta_path = model_artifact_dir(model_id) / "hybrid.json"
    write_json(hybrid_meta_path, hybrid_meta)
    return {
        "context_len": int(torch_artifact.context_len),
        "input_dim": int(torch_artifact.input_dim),
        "pooled_residuals": hybrid_residuals[-500:],
        "artifact": {
            "snapshot_json": artifact_relpath(model_id, "snapshot.json"),
            "weights_pt": artifact_relpath(model_id, "weights.pt"),
            "gbdt_joblib": artifact_relpath(model_id, "gbdt.joblib"),
            "hybrid_json": artifact_relpath(model_id, "hybrid.json"),
        },
    }


def train_public_gbdt_entry(
    req: TrainRequest,
    *,
    model_id: str,
    model_artifact_dir: Callable[[str], Path],
    write_json: Callable[[Path, dict[str, Any]], None],
    artifact_relpath: Callable[[str, str], str],
) -> dict[str, Any]:
    _ys_by_series, records_by_series = build_request_training_payload(req.data)
    context_len = hybrid_context_len(records_by_series)
    feature_keys = select_series_feature_keys(records_by_series, max_features=24)
    if not feature_keys:
        raise ValueError("GBDT training requires numeric exogenous features in x.")
    gbdt_artifact = fit_hgb_forecaster(
        records_by_series=records_by_series,
        context_len=context_len,
        feature_keys=feature_keys,
    )
    snapshot = {**gbdt_artifact["snapshot"], "algo": "gbdt_hgb_v1"}
    snap_path = model_artifact_dir(model_id) / "snapshot.json"
    write_json(snap_path, snapshot)
    gbdt_path = model_artifact_dir(model_id) / "gbdt.joblib"
    save_joblib_artifact(gbdt_path, gbdt_artifact["bundle"])
    return {
        "context_len": int(snapshot.get("context_len") or context_len),
        "input_dim": int(len(feature_keys) * max(1, context_len)),
        "pooled_residuals": _as_list(gbdt_artifact.get("pooled_residuals")),
        "artifact": {
            "snapshot_json": artifact_relpath(model_id, "snapshot.json"),
            "gbdt_joblib": artifact_relpath(model_id, "gbdt.joblib"),
        },
    }


def fit_ridge_lags_model(
    req: TrainRequest,
    *,
    ridge_lags_choose_k: Callable[[int], int],
    ridge_lags_fit_series: Callable[[list[float]], dict[str, Any]] | Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    by_series: dict[str, list[TimeSeriesRecord]] = {}
    for record in req.data:
        by_series.setdefault(record.series_id, []).append(record)

    series_state: dict[str, Any] = {}
    pooled_residuals: list[float] = []
    for series_id, rows in by_series.items():
        rows_sorted = sorted(rows, key=lambda item: item.timestamp)
        ys = [float(row.y) for row in rows_sorted]
        lag_k = ridge_lags_choose_k(len(ys))
        state = ridge_lags_fit_series(ys, lag_k=lag_k)
        pooled_residuals.extend(
            [
                float(value)
                for value in (state.get("residuals") or [])
                if isinstance(value, int | float)
            ]
        )
        k = int(state.get("lag_k") or 1)
        last_vals = ys[-k:] if ys else []
        series_state[series_id] = {
            **state,
            "last_values": [float(value) for value in last_vals],
            "last_timestamp": rows_sorted[-1].timestamp.isoformat() if rows_sorted else None,
            "train_n": len(ys),
        }

    return {
        "algo": "ridge_lags_v1",
        "pooled_residuals": [
            float(value)
            for value in pooled_residuals
            if isinstance(value, int | float) and math.isfinite(float(value))
        ],
        "series": series_state,
    }