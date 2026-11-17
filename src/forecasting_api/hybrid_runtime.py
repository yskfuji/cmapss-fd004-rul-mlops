from __future__ import annotations

import math
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from . import training_helpers
from .domain import stable_models
from .errors import ApiError
from .schemas import (
    BacktestRequest,
    BacktestResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_float_list(value: Any) -> list[float]:
    return [
        float(item)
        for item in _as_list(value)
        if isinstance(item, int | float) and math.isfinite(float(item))
    ]


def hybrid_backtest(
    req: BacktestRequest,
    *,
    trained: dict[str, Any],
    read_json: Callable[[Path], dict[str, Any]],
    artifact_abspath: Callable[[str], Path],
    try_torch_load_weights: Callable[[Path], dict[str, Any]],
    extract_state_dict: Callable[[dict[str, Any]], dict[str, Any]],
    load_joblib_artifact: Callable[[Path], Any],
    predict_hgb_next_fn: Callable[..., tuple[float, float, float]] = stable_models.predict_hgb_next,
    gate_step_payload_fn: Callable[
        ..., dict[str, float]
    ] = training_helpers.hybrid_gate_step_payload,
    hybrid_condition_cluster_key_fn: Callable[
        [dict[str, Any]], str
    ] = training_helpers.hybrid_condition_cluster_key,
    metric_value_fn: Callable[..., float] = stable_models.metric_value,
) -> BacktestResponse:
    from .torch_forecasters import forecast_univariate_torch_with_details

    art = _as_dict(trained.get("artifact"))
    snapshot_rel = art.get("snapshot_json") if isinstance(art.get("snapshot_json"), str) else None
    weights_rel = art.get("weights_pt") if isinstance(art.get("weights_pt"), str) else None
    gbdt_rel = art.get("gbdt_joblib") if isinstance(art.get("gbdt_joblib"), str) else None
    hybrid_rel = art.get("hybrid_json") if isinstance(art.get("hybrid_json"), str) else None
    if not snapshot_rel or not weights_rel or not gbdt_rel or not hybrid_rel:
        raise ApiError(
            status_code=400,
            error_code="M02",
            message="hybrid model artifact が見つかりません",
            details={"model_id": req.model_id, "next_action": "再学習（/v1/train）してください"},
        )

    snapshot = read_json(artifact_abspath(snapshot_rel))
    ckpt = try_torch_load_weights(artifact_abspath(weights_rel))
    state_dict = extract_state_dict(ckpt)
    gbdt_bundle = load_joblib_artifact(artifact_abspath(gbdt_rel))
    hybrid_meta = read_json(artifact_abspath(hybrid_rel))
    gate_meta = _as_dict(hybrid_meta.get("gate"))
    feature_keys = [
        str(key) for key in _as_list(hybrid_meta.get("feature_keys")) if isinstance(key, str)
    ]
    context_len = int(trained.get("context_len") or snapshot.get("context_len") or 14)

    by_series_records: dict[str, list[Any]] = {}
    for record in req.data:
        by_series_records.setdefault(record.series_id, []).append(record)

    overall_true: list[float] = []
    overall_pred: list[float] = []
    overall_train_y: list[float] = []

    by_h_true: dict[int, list[float]] = {h: [] for h in range(1, req.horizon + 1)}
    by_h_pred: dict[int, list[float]] = {h: [] for h in range(1, req.horizon + 1)}
    by_h_train: dict[int, list[float]] = {h: [] for h in range(1, req.horizon + 1)}
    by_f_true: dict[int, list[float]] = {f: [] for f in range(1, req.folds + 1)}
    by_f_pred: dict[int, list[float]] = {f: [] for f in range(1, req.folds + 1)}
    by_f_train: dict[int, list[float]] = {f: [] for f in range(1, req.folds + 1)}

    per_series_entries: list[dict[str, Any]] = []

    for series_id, rows in by_series_records.items():
        rows_sorted = sorted(rows, key=lambda row: row.timestamp)
        ys = [float(row.y) for row in rows_sorted]
        n = len(ys)
        if n < req.horizon + 3:
            continue

        y_true: list[float] = []
        y_pred: list[float] = []
        train_y: list[float] = []

        for fold_index in range(req.folds):
            end = n - fold_index * req.horizon
            start = end - req.horizon
            train_end = start
            if train_end <= 0 or start < 0 or end > n:
                break

            train_rows = rows_sorted[:train_end]
            actual_rows = rows_sorted[start:end]
            train = [float(row.y) for row in train_rows]
            actual = [float(row.y) for row in actual_rows]
            if len(actual) != req.horizon or len(train_rows) < 2:
                break

            context_records = [
                {"timestamp": row.timestamp.isoformat(), "y": float(row.y), "x": dict(row.x or {})}
                for row in train_rows[-context_len:]
            ]
            need = max(1, context_len)
            if len(context_records) < need:
                pad = (
                    dict(context_records[0])
                    if context_records
                    else {
                        "timestamp": train_rows[-1].timestamp.isoformat(),
                        "y": float(train_rows[-1].y),
                        "x": dict(train_rows[-1].x or {}),
                    }
                )
                context_records = [
                    dict(pad) for _ in range(need - len(context_records))
                ] + context_records

            future_feature_rows = [
                {"timestamp": row.timestamp.isoformat(), "y": float(row.y), "x": dict(row.x or {})}
                for row in actual_rows
            ]
            torch_details = forecast_univariate_torch_with_details(
                algo="afnocg3_v1",
                snapshot=snapshot,
                state_dict=state_dict,
                context_records=context_records,
                future_feature_rows=future_feature_rows,
                horizon=req.horizon,
                device=None,
                mc_dropout_samples=0,
            )
            afno_points = [float(value) for value in torch_details.get("point") or []]
            afno_vars = [
                float(value)
                for value in (
                    ((torch_details.get("mc_dropout") or {}).get("per_step_var"))
                    or [0.0] * req.horizon
                )
            ]

            gbdt_rows = [dict(row) for row in context_records]
            blended_preds: list[float] = []
            for step_idx, actual_row in enumerate(actual_rows):
                future_x = dict(actual_row.x or {})
                g_pred, g_low, g_high = predict_hgb_next_fn(
                    gbdt_bundle,
                    context_records=gbdt_rows[-context_len:],
                    feature_keys=feature_keys,
                )
                a_pred = (
                    float(afno_points[step_idx]) if step_idx < len(afno_points) else float(g_pred)
                )
                afno_std = math.sqrt(
                    max(0.0, float(afno_vars[step_idx]) if step_idx < len(afno_vars) else 0.0)
                )
                gate_payload = gate_step_payload_fn(
                    g_pred=float(g_pred),
                    g_lower=float(g_low),
                    g_upper=float(g_high),
                    a_pred=a_pred,
                    a_lower=float(a_pred - afno_std),
                    a_upper=float(a_pred + afno_std),
                    gate_meta=_as_dict(gate_meta),
                    condition_key=hybrid_condition_cluster_key_fn({"x": future_x}),
                    tail_pos=(
                        float(step_idx / max(req.horizon - 1, 1)) if req.horizon > 1 else 0.5
                    ),
                )
                blended_pred = (
                    float(gate_payload["gbdt_weight"]) * float(g_pred)
                    + float(gate_payload["afno_weight"]) * a_pred
                )
                blended_preds.append(blended_pred)
                gbdt_rows.append(
                    {
                        "timestamp": actual_row.timestamp.isoformat(),
                        "y": blended_pred,
                        "x": future_x,
                    }
                )

            y_true.extend(actual)
            y_pred.extend(blended_preds)
            train_y.extend(train)

            for idx, (truth, pred) in enumerate(zip(actual, blended_preds, strict=False), start=1):
                by_h_true[idx].append(float(truth))
                by_h_pred[idx].append(float(pred))
                by_h_train[idx].extend(train)
                by_f_true[fold_index + 1].append(float(truth))
                by_f_pred[fold_index + 1].append(float(pred))
                by_f_train[fold_index + 1].extend(train)

        if not y_true:
            continue

        per_series_metric = metric_value_fn(
            req.metric, y_true=y_true, y_pred=y_pred, train_y=train_y
        )
        per_series_entries.append(
            {"series_id": series_id, "metric": req.metric, "value": float(per_series_metric)}
        )
        overall_true.extend(y_true)
        overall_pred.extend(y_pred)
        overall_train_y.extend(train_y)

    overall = metric_value_fn(
        req.metric, y_true=overall_true, y_pred=overall_pred, train_y=overall_train_y
    )

    by_h_entries: list[dict[str, Any]] = []
    for horizon in range(1, req.horizon + 1):
        value = metric_value_fn(
            req.metric,
            y_true=by_h_true[horizon],
            y_pred=by_h_pred[horizon],
            train_y=by_h_train[horizon],
        )
        by_h_entries.append({"horizon": horizon, "metric": req.metric, "value": float(value)})

    by_f_entries: list[dict[str, Any]] = []
    for fold in range(1, req.folds + 1):
        if not by_f_true[fold]:
            continue
        value = metric_value_fn(
            req.metric,
            y_true=by_f_true[fold],
            y_pred=by_f_pred[fold],
            train_y=by_f_train[fold],
        )
        by_f_entries.append({"fold": fold, "metric": req.metric, "value": float(value)})

    return BacktestResponse(
        metrics={req.metric: float(overall)},
        by_series=per_series_entries or None,
        by_horizon=by_h_entries or None,
        by_fold=by_f_entries or None,
    )


def forecast_with_hybrid_model(
    req: ForecastRequest,
    *,
    step: timedelta,
    trained: dict[str, Any],
    read_json: Callable[[Path], dict[str, Any]],
    artifact_abspath: Callable[[str], Path],
    try_torch_load_weights: Callable[[Path], dict[str, Any]],
    extract_state_dict: Callable[[dict[str, Any]], dict[str, Any]],
    load_joblib_artifact: Callable[[Path], Any],
    predict_hgb_next_fn: Callable[..., tuple[float, float, float]] = stable_models.predict_hgb_next,
    gate_step_payload_fn: Callable[
        ..., dict[str, float]
    ] = training_helpers.hybrid_gate_step_payload,
    hybrid_condition_cluster_key_fn: Callable[
        [dict[str, Any]], str
    ] = training_helpers.hybrid_condition_cluster_key,
) -> ForecastResponse:
    from .torch_forecasters import forecast_univariate_torch_with_details

    art = _as_dict(trained.get("artifact"))
    snapshot_rel = art.get("snapshot_json") if isinstance(art.get("snapshot_json"), str) else None
    weights_rel = art.get("weights_pt") if isinstance(art.get("weights_pt"), str) else None
    gbdt_rel = art.get("gbdt_joblib") if isinstance(art.get("gbdt_joblib"), str) else None
    hybrid_rel = art.get("hybrid_json") if isinstance(art.get("hybrid_json"), str) else None
    if not snapshot_rel or not weights_rel or not gbdt_rel or not hybrid_rel:
        raise ApiError(
            status_code=400,
            error_code="M02",
            message="hybrid model artifact が見つかりません",
            details={"model_id": req.model_id, "next_action": "再学習（/v1/train）してください"},
        )

    snapshot = read_json(artifact_abspath(snapshot_rel))
    ckpt = try_torch_load_weights(artifact_abspath(weights_rel))
    state_dict = extract_state_dict(ckpt)
    gbdt_bundle = load_joblib_artifact(artifact_abspath(gbdt_rel))
    hybrid_meta = read_json(artifact_abspath(hybrid_rel))
    gbdt_meta = _as_dict(hybrid_meta.get("gbdt"))
    gate_meta = _as_dict(hybrid_meta.get("gate"))
    feature_keys = [
        str(key) for key in _as_list(hybrid_meta.get("feature_keys")) if isinstance(key, str)
    ]
    context_len = int(trained.get("context_len") or snapshot.get("context_len") or 14)
    residuals = _as_float_list(trained.get("pooled_residuals"))
    baseline_map = {
        str(key): float(value)
        for key, value in _as_dict(gbdt_meta.get("feature_baseline")).items()
        if isinstance(value, int | float)
    }
    mc_samples = (
        int(getattr(req.options, "mc_dropout_samples", 16))
        if req.options and getattr(req.options, "mc_dropout_samples", None) is not None
        else 16
    )
    normal = NormalDist()

    by_series: dict[str, list[Any]] = {}
    for record in req.data:
        by_series.setdefault(record.series_id, []).append(record)

    forecasts: list[ForecastPoint] = []
    component_rollups: dict[str, list[float]] = {
        "gbdt_interval_var": [],
        "afno_mc_dropout_var": [],
        "expert_disagreement_var": [],
        "total_var": [],
    }
    for series_id, rows in by_series.items():
        rows_sorted = sorted(rows, key=lambda row: row.timestamp)
        last_ts = rows_sorted[-1].timestamp
        context_records = [
            {
                "timestamp": record.timestamp.isoformat(),
                "y": float(record.y),
                "x": dict(record.x or {}),
            }
            for record in rows_sorted[-context_len:]
        ]
        if len(context_records) < max(1, context_len):
            pad = (
                dict(context_records[0])
                if context_records
                else {"timestamp": last_ts.isoformat(), "y": 0.0, "x": {}}
            )
            context_records = [
                dict(pad) for _ in range(max(1, context_len) - len(context_records))
            ] + context_records
        torch_details = forecast_univariate_torch_with_details(
            algo="afnocg3_v1",
            snapshot=snapshot,
            state_dict=state_dict,
            context_records=context_records,
            horizon=req.horizon,
            device=None,
            mc_dropout_samples=mc_samples,
            occlusion_feature_keys=feature_keys,
            occlusion_baseline=baseline_map,
            occlusion_top_k=5,
        )
        afno_points = [float(value) for value in torch_details.get("point") or []]
        afno_vars = [
            float(value)
            for value in (
                ((torch_details.get("mc_dropout") or {}).get("per_step_var")) or [0.0] * req.horizon
            )
        ]
        afno_occlusion_steps = (torch_details.get("occlusion") or {}).get("per_step") or []

        gbdt_rows = [dict(row) for row in context_records]
        last_x = _as_dict(gbdt_rows[-1].get("x")) if gbdt_rows else {}
        gbdt_points: list[float] = []
        gbdt_lowers: list[float] = []
        gbdt_uppers: list[float] = []
        for _step_idx in range(req.horizon):
            pred, low, high = predict_hgb_next_fn(
                gbdt_bundle,
                context_records=gbdt_rows[-context_len:],
                feature_keys=feature_keys,
            )
            gbdt_points.append(pred)
            gbdt_lowers.append(low)
            gbdt_uppers.append(high)
            gbdt_rows.append({"y": pred, "x": dict(last_x)})

        for step_idx in range(req.horizon):
            ts = last_ts + step * (step_idx + 1)
            g_pred = float(gbdt_points[step_idx])
            a_pred = float(afno_points[step_idx]) if step_idx < len(afno_points) else g_pred
            g_half = max(0.0, (float(gbdt_uppers[step_idx]) - float(gbdt_lowers[step_idx])) / 2.0)
            afno_var = max(0.0, float(afno_vars[step_idx]) if step_idx < len(afno_vars) else 0.0)
            tail_pos = float(step_idx / max(req.horizon - 1, 1)) if req.horizon > 1 else 0.5
            afno_std = math.sqrt(afno_var)
            condition_key = hybrid_condition_cluster_key_fn({"x": last_x})
            gate_payload = gate_step_payload_fn(
                g_pred=g_pred,
                g_lower=float(gbdt_lowers[step_idx]),
                g_upper=float(gbdt_uppers[step_idx]),
                a_pred=a_pred,
                a_lower=float(a_pred - afno_std),
                a_upper=float(a_pred + afno_std),
                gate_meta=_as_dict(gate_meta),
                condition_key=condition_key,
                tail_pos=tail_pos,
            )
            afno_weight = float(gate_payload["afno_weight"])
            gbdt_weight = float(gate_payload["gbdt_weight"])
            point = gbdt_weight * g_pred + afno_weight * a_pred

            gbdt_interval_var = (g_half / 1.645) ** 2 if g_half > 0.0 else 0.0
            expert_disagreement_var = afno_weight * gbdt_weight * ((a_pred - g_pred) ** 2)
            total_var = max(
                0.0,
                (gbdt_weight * gbdt_interval_var)
                + (afno_weight * afno_var)
                + expert_disagreement_var,
            )
            total_std = math.sqrt(total_var)
            interval_scale = max(float(gate_meta.get("interval_scale") or 1.0), 1e-6)
            calibrated_total_std = total_std * interval_scale
            component_rollups["gbdt_interval_var"].append(gbdt_interval_var)
            component_rollups["afno_mc_dropout_var"].append(afno_var)
            component_rollups["expert_disagreement_var"].append(expert_disagreement_var)
            component_rollups["total_var"].append(total_var)

            q_map: dict[str, float] | None = None
            if req.quantiles:
                q_map = {}
                for quantile in req.quantiles:
                    z_value = normal.inv_cdf(float(quantile))
                    q_map[str(quantile)] = float(point + z_value * calibrated_total_std)

            intervals: list[dict[str, float]] | None = None
            if req.level:
                intervals = []
                for level in req.level:
                    z_value = normal.inv_cdf(0.5 + (float(level) / 200.0))
                    width = float(z_value * calibrated_total_std)
                    intervals.append(
                        {
                            "level": float(level),
                            "lower": float(point - width),
                            "upper": float(point + width),
                        }
                    )

            uncertainty_payload = {
                "method": "hybrid_3way_v1",
                "components": {
                    "gbdt_interval_var": float(gbdt_interval_var),
                    "afno_mc_dropout_var": float(afno_var),
                    "expert_disagreement_var": float(expert_disagreement_var),
                    "total_var": float(total_var),
                },
                "interval_scale": float(interval_scale),
                "mc_dropout_samples": int(mc_samples),
            }
            explanation_payload = {
                "gate": {
                    "afno_weight": float(afno_weight),
                    "gbdt_weight": float(gbdt_weight),
                    "score": float(gate_payload["score"]),
                    "condition_key": condition_key,
                    "terms": {
                        "delta": float(gate_payload["term_delta"]),
                        "overlap": float(gate_payload["term_overlap"]),
                        "width": float(gate_payload["term_width"]),
                        "tail": float(gate_payload["term_tail"]),
                        "condition": float(gate_payload["term_condition"]),
                    },
                },
                "gbdt": _as_dict(hybrid_meta.get("model_explainability")).get("gbdt"),
                "afno": afno_occlusion_steps[step_idx]
                if step_idx < len(afno_occlusion_steps)
                else _as_dict(hybrid_meta.get("model_explainability")).get("afno"),
            }
            forecasts.append(
                ForecastPoint(
                    series_id=series_id,
                    timestamp=ts,
                    point=float(point),
                    quantiles=q_map,
                    intervals=intervals,
                    uncertainty=uncertainty_payload,
                    explanation=explanation_payload,
                )
            )

    uncertainty_summary = {
        "method": "hybrid_3way_v1",
        "components_mean": {
            key: (float(np.mean(values)) if values else 0.0)
            for key, values in component_rollups.items()
        },
        "interval_scale": float(gate_meta.get("interval_scale") or 1.0),
        "mc_dropout_samples": int(mc_samples),
    }
    calibration = {
        "method": "hybrid_validation_residual_evidence",
        "residuals_n": len(residuals),
        "gbdt_interval_method": "quantile_boosting_90",
        "afno_uncertainty_method": "mc_dropout",
    }
    return ForecastResponse(
        forecasts=forecasts,
        warnings=None,
        calibration=calibration,
        residuals_evidence=stable_models.build_residuals_evidence(residuals),
        model_explainability=hybrid_meta.get("model_explainability")
        if isinstance(hybrid_meta.get("model_explainability"), dict)
        else None,
        uncertainty_summary=uncertainty_summary,
    )
