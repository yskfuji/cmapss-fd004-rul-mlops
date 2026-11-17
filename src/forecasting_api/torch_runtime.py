from __future__ import annotations

import importlib
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from forecasting_api.errors import ApiError
from forecasting_api.schemas import (
    BacktestRequest,
    BacktestResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    TimeSeriesRecord,
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, int | float) and math.isfinite(float(item)):
            out.append(float(item))
    return out


def _resolve_torch_device(*, algo: str, requested: str | None = None) -> str:
    if requested:
        return str(requested)

    env_device = os.getenv("RULFM_FORECASTING_TORCH_DEVICE", "").strip()
    if env_device:
        return env_device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if algo in {"afnocg3", "afnocg3_v1"}:
                return "cpu"
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_torch_artifacts(
    *,
    req_model_id: str | None,
    trained: dict[str, Any],
    read_json: Callable[[Path], dict[str, Any]],
    artifact_abspath: Callable[[str], Path],
    try_torch_load_weights: Callable[[Path], dict[str, Any]],
    extract_state_dict: Callable[[dict[str, Any]], dict[str, Any]],
    api_error_cls: type[ApiError],
) -> tuple[str, dict[str, Any], dict[str, Any], int]:
    algo = str(trained.get("algo") or "").strip().lower()
    artifact = _as_dict(trained.get("artifact"))
    snapshot_value = artifact.get("snapshot_json")
    weights_value = artifact.get("weights_pt")
    snapshot_rel = snapshot_value if isinstance(snapshot_value, str) else None
    weights_rel = weights_value if isinstance(weights_value, str) else None
    if not snapshot_rel or not weights_rel:
        raise api_error_cls(
            status_code=400,
            error_code="M02",
            message="model artifact が見つかりません",
            details={
                "model_id": req_model_id,
                "next_action": "再学習（/v1/train）してください",
            },
        )

    snapshot = read_json(artifact_abspath(snapshot_rel))
    ckpt = try_torch_load_weights(artifact_abspath(weights_rel))
    state_dict = extract_state_dict(ckpt)
    context_len = int(trained.get("context_len") or 14)
    return algo, snapshot, state_dict, context_len


def _build_torch_calibration(
    *,
    algo: str,
    req: ForecastRequest,
    residuals: list[float],
    quantile_nearest_rank_fn: Callable[[list[float], float], float],
    bi: Callable[[str, str], str],
) -> tuple[
    list[float] | None,
    list[float] | None,
    float,
    float | None,
    dict[str, float] | None,
    list[str],
    dict[str, Any] | None,
]:
    calib_min = 12
    has_calib = len(residuals) >= calib_min

    quantiles: list[float] | None = None
    max_dev = 0.0
    if req.quantiles:
        quantiles = [float(q) for q in req.quantiles if 0.0 < float(q) < 1.0]
        max_dev = max((abs(q - 0.5) for q in quantiles), default=0.0)

    levels: list[float] | None = None
    if req.level:
        levels = [float(level) for level in req.level if 0.0 < float(level) <= 100.0]

    qhat_for_quantiles: float | None = None
    qhat_by_level: dict[str, float] | None = None
    if has_calib:
        if quantiles and max_dev > 0:
            coverage = max(0.0, min(0.999, 2.0 * max_dev))
            qhat_for_quantiles = quantile_nearest_rank_fn(residuals, coverage)
        if levels:
            qhat_by_level = {}
            for level in levels:
                coverage = max(0.0, min(0.999, float(level) / 100.0))
                qhat_by_level[str(level)] = quantile_nearest_rank_fn(residuals, coverage)

    warnings: list[str] = []
    if not has_calib:
        warnings.append(
            bi(
                (
                    "CALIB01: insufficient residuals for split conformal "
                    f"(n={len(residuals)} < {calib_min})."
                ),
                (
                    "CALIB01: split conformal の残差が不足しています"
                    f"（n={len(residuals)} < {calib_min}）。"
                ),
            )
        )

    calibration: dict[str, Any] | None = None
    if has_calib and (
        qhat_for_quantiles is not None
        or (qhat_by_level is not None and len(qhat_by_level) > 0)
    ):
        calibration = {
            "method": "split_conformal_abs_error",
            "residuals_n": len(residuals),
            "base": f"{algo}_one_step_abs_error",
            "scaling": "sqrt(h)",
        }
        if qhat_for_quantiles is not None and quantiles and max_dev > 0:
            calibration.update(
                {
                    "quantiles": quantiles,
                    "quantiles_max_dev": max_dev,
                    "coverage": max(0.0, min(0.999, 2.0 * max_dev)),
                    "qhat": float(qhat_for_quantiles),
                }
            )
        if qhat_by_level is not None and levels:
            calibration.update(
                {
                    "levels": levels,
                    "qhat_by_level": {key: float(value) for key, value in qhat_by_level.items()},
                }
            )

    return (
        quantiles,
        levels,
        max_dev,
        qhat_for_quantiles,
        qhat_by_level,
        warnings,
        calibration,
    )


def _load_multivariate_sequence_model(
    *,
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
) -> tuple[Any, Any, str]:
    import torch

    device_str = _resolve_torch_device(algo="stardast2_v5")
    device = torch.device(device_str)
    cfg = SimpleNamespace(
        MODEL_NAME="stardast2_v5",
        TARGET_KIND="regression",
        MODEL_SEED=0,
        MODEL_PARAMS={},
    )

    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    module = importlib.import_module("src.models.stardast2_v5_adapter")
    loader = cast(Callable[..., Any], module.load_from_snapshot)
    model = loader(cfg, device, snapshot, model_params=None)
    model.load_state_dict(state_dict, strict=False)
    return model, device, device_str


def _prepare_multivariate_finetune(
    *,
    by_series: dict[str, list[TimeSeriesRecord]],
    context_len: int,
    feature_count: int,
) -> tuple[list[list[list[float]]], list[float]]:
    feature_keys = [f"f{i}" for i in range(feature_count)]
    x_rows: list[list[list[float]]] = []
    y_values: list[float] = []
    for series_rows in by_series.values():
        ordered_rows = sorted(series_rows, key=lambda row: row.timestamp)
        for index in range(context_len, len(ordered_rows)):
            window = ordered_rows[index - context_len : index]
            feature_matrix: list[list[float]] = []
            for record in window:
                x_dict = record.x or {}
                feature_matrix.append([float(x_dict.get(key, 0.0)) for key in feature_keys])
            x_rows.append(feature_matrix)
            y_values.append(float(ordered_rows[index].y))
    return x_rows, y_values


def forecast_with_torch_model(
    req: ForecastRequest,
    *,
    step: Any,
    trained: dict[str, Any],
    read_json: Callable[[Path], dict[str, Any]],
    artifact_abspath: Callable[[str], Path],
    try_torch_load_weights: Callable[[Path], dict[str, Any]],
    extract_state_dict: Callable[[dict[str, Any]], dict[str, Any]],
    quantile_nearest_rank_fn: Callable[[list[float], float], float],
    build_residuals_evidence_fn: Callable[[list[float]], dict[str, Any] | None],
    bi: Callable[[str, str], str],
    api_error_cls: type[ApiError],
) -> ForecastResponse:
    from .torch_forecasters import finetune_gate_stardast2v5, forecast_univariate_torch

    algo, snapshot, state_dict, context_len = _load_torch_artifacts(
        req_model_id=req.model_id,
        trained=trained,
        read_json=read_json,
        artifact_abspath=artifact_abspath,
        try_torch_load_weights=try_torch_load_weights,
        extract_state_dict=extract_state_dict,
        api_error_cls=api_error_cls,
    )

    by_series: dict[str, list[TimeSeriesRecord]] = {}
    for record in req.data:
        by_series.setdefault(record.series_id, []).append(record)

    residuals = _as_float_list(trained.get("pooled_residuals"))
    (
        quantiles,
        levels,
        max_dev,
        qhat_for_quantiles,
        qhat_by_level,
        warnings,
        calibration,
    ) = _build_torch_calibration(
        algo=algo,
        req=req,
        residuals=residuals,
        quantile_nearest_rank_fn=quantile_nearest_rank_fn,
        bi=bi,
    )

    sequence_model: Any = None
    sequence_device: Any = None
    if algo == "stardast2_v5":
        import torch

        sequence_model, sequence_device, device_str = _load_multivariate_sequence_model(
            snapshot=snapshot,
            state_dict=state_dict,
        )
        feature_count = int(trained.get("input_dim") or snapshot.get("in_feats") or 21)
        finetune_rows, finetune_targets = _prepare_multivariate_finetune(
            by_series=by_series,
            context_len=context_len,
            feature_count=feature_count,
        )
        if finetune_rows:
            x_tensor = torch.tensor(finetune_rows, dtype=torch.float32, device=sequence_device)
            y_tensor = torch.tensor(finetune_targets, dtype=torch.float32, device=sequence_device)
            finetune_gate_stardast2v5(
                model=sequence_model,
                X=x_tensor,
                y=y_tensor,
                n_total=len(finetune_targets),
                device=device_str,
            )
        else:
            sequence_model.eval()

    forecasts: list[ForecastPoint] = []
    for series_id, rows in by_series.items():
        rows_sorted = sorted(rows, key=lambda row: row.timestamp)
        last_ts = rows_sorted[-1].timestamp
        ys = [float(row.y) for row in rows_sorted]
        context = ys[-context_len:] if context_len > 0 else ys[-1:]
        if len(context) < max(1, context_len):
            pad = context[-1] if context else 0.0
            context = ([pad] * (max(1, context_len) - len(context))) + context

        if algo == "stardast2_v5" and sequence_model is not None and sequence_device is not None:
            import torch

            feature_count = int(trained.get("input_dim") or snapshot.get("in_feats") or 21)
            feature_keys = [f"f{i}" for i in range(feature_count)]
            context_rows = rows_sorted[-context_len:] if context_len > 0 else rows_sorted[-1:]
            pad_n = max(0, context_len - len(context_rows))
            context_features: list[list[float]] = [[0.0] * feature_count for _ in range(pad_n)]
            for record in context_rows:
                x_dict = record.x or {}
                context_features.append([float(x_dict.get(key, 0.0)) for key in feature_keys])

            preds: list[float] = []
            sliding = list(context_features)
            with torch.no_grad():
                for _ in range(req.horizon):
                    x_tensor = torch.tensor([sliding], dtype=torch.float32, device=sequence_device)
                    output = sequence_model(x_tensor)
                    yhat = (output[0] if isinstance(output, tuple | list) else output).reshape(-1)
                    value = float(yhat[-1].detach().cpu().item())
                    preds.append(value)
                    next_feature = [value] + [0.0] * (feature_count - 1)
                    sliding = sliding[1:] + [next_feature]
        else:
            preds = forecast_univariate_torch(
                algo=algo,
                snapshot=snapshot,
                state_dict=state_dict,
                context=context,
                context_records=[
                    {
                        "timestamp": record.timestamp.isoformat(),
                        "y": float(record.y),
                        "x": dict(record.x or {}),
                    }
                    for record in rows_sorted[-context_len:]
                ],
                horizon=req.horizon,
                device=None,
            )

        for index in range(1, req.horizon + 1):
            ts = last_ts + step * index
            point = float(preds[index - 1])

            uncertainty = (abs(point) * 0.05 + 1.0) * math.sqrt(index)
            if qhat_for_quantiles is not None and quantiles and max_dev > 0:
                uncertainty = float(qhat_for_quantiles) * math.sqrt(index)

            quantile_map: dict[str, float] | None = None
            if quantiles:
                quantile_map = {}
                denom = 2.0 * max_dev if max_dev > 0 else 1.0
                scale = (uncertainty / denom) if denom > 0 else 0.0
                for quantile in quantiles:
                    offset = (float(quantile) - 0.5) * 2.0 * scale
                    quantile_map[str(quantile)] = float(point + offset)

            intervals: list[dict[str, float]] | None = None
            if levels:
                intervals = []
                for level in levels:
                    if qhat_by_level and str(level) in qhat_by_level:
                        width = float(qhat_by_level[str(level)]) * math.sqrt(index)
                    else:
                        width = uncertainty * max(0.5, float(level) / 100.0)
                    intervals.append(
                        {
                            "level": float(level),
                            "lower": float(point - width),
                            "upper": float(point + width),
                        }
                    )

            forecasts.append(
                ForecastPoint(
                    series_id=series_id,
                    timestamp=ts,
                    point=point,
                    quantiles=quantile_map,
                    intervals=intervals,
                    uncertainty=None,
                    explanation=None,
                )
            )

    response_payload: dict[str, Any] = {
        "forecasts": forecasts,
        "residuals_evidence": build_residuals_evidence_fn(residuals),
        "model_explainability": None,
        "uncertainty_summary": None,
    }
    if calibration is not None:
        response_payload["calibration"] = calibration
    if warnings:
        response_payload["warnings"] = warnings
    return ForecastResponse(**response_payload)


def torch_backtest(
    req: BacktestRequest,
    *,
    trained: dict[str, Any],
    read_json: Callable[[Path], dict[str, Any]],
    artifact_abspath: Callable[[str], Path],
    try_torch_load_weights: Callable[[Path], dict[str, Any]],
    extract_state_dict: Callable[[dict[str, Any]], dict[str, Any]],
    metric_value_fn: Callable[..., float],
    api_error_cls: type[ApiError],
) -> BacktestResponse:
    from .torch_forecasters import forecast_univariate_torch

    algo, snapshot, state_dict, context_len = _load_torch_artifacts(
        req_model_id=req.model_id,
        trained=trained,
        read_json=read_json,
        artifact_abspath=artifact_abspath,
        try_torch_load_weights=try_torch_load_weights,
        extract_state_dict=extract_state_dict,
        api_error_cls=api_error_cls,
    )

    by_series_records: dict[str, list[TimeSeriesRecord]] = {}
    for record in req.data:
        by_series_records.setdefault(record.series_id, []).append(record)

    overall_true: list[float] = []
    overall_pred: list[float] = []
    overall_train: list[float] = []
    by_h_true: dict[int, list[float]] = {h: [] for h in range(1, req.horizon + 1)}
    by_h_pred: dict[int, list[float]] = {h: [] for h in range(1, req.horizon + 1)}
    by_h_train: dict[int, list[float]] = {h: [] for h in range(1, req.horizon + 1)}
    by_f_true: dict[int, list[float]] = {fold: [] for fold in range(1, req.folds + 1)}
    by_f_pred: dict[int, list[float]] = {fold: [] for fold in range(1, req.folds + 1)}
    by_f_train: dict[int, list[float]] = {fold: [] for fold in range(1, req.folds + 1)}
    per_series_entries: list[dict[str, Any]] = []

    for series_id, rows in by_series_records.items():
        ys = [float(row.y) for row in rows]
        series_len = len(ys)
        if series_len < req.horizon + 3:
            continue

        y_true: list[float] = []
        y_pred: list[float] = []
        train_y: list[float] = []

        for fold_index in range(req.folds):
            end = series_len - fold_index * req.horizon
            start = end - req.horizon
            train_end = start
            if train_end <= 0 or start < 0 or end > series_len:
                break

            train = ys[:train_end]
            actual = ys[start:end]
            if len(actual) != req.horizon or len(train) < 2:
                break

            context = train[-context_len:] if context_len > 0 else train[-1:]
            need = max(1, context_len)
            if len(context) < need:
                pad = context[-1] if context else 0.0
                context = ([pad] * (need - len(context))) + context

            preds = forecast_univariate_torch(
                algo=algo,
                snapshot=snapshot,
                state_dict=state_dict,
                context=context,
                context_records=[
                    {
                        "timestamp": row.timestamp.isoformat(),
                        "y": float(row.y),
                        "x": dict(row.x or {}),
                    }
                    for row in rows[max(0, train_end - context_len) : train_end]
                ],
                future_feature_rows=[
                    {
                        "timestamp": row.timestamp.isoformat(),
                        "y": float(row.y),
                        "x": dict(row.x or {}),
                    }
                    for row in rows[start:end]
                ],
                horizon=req.horizon,
                device=None,
            )

            y_true.extend(float(value) for value in actual)
            y_pred.extend(float(value) for value in preds)
            train_y.extend(float(value) for value in train)

            for horizon_index in range(req.horizon):
                horizon = horizon_index + 1
                by_h_true[horizon].append(float(actual[horizon_index]))
                by_h_pred[horizon].append(float(preds[horizon_index]))
                by_h_train[horizon].extend(train)

            fold = fold_index + 1
            by_f_true[fold].extend(float(value) for value in actual)
            by_f_pred[fold].extend(float(value) for value in preds)
            by_f_train[fold].extend(float(value) for value in train)

        if not y_true:
            continue

        metric_value = metric_value_fn(
            req.metric,
            y_true=y_true,
            y_pred=y_pred,
            train_y=train_y,
        )
        per_series_entries.append(
            {
                "series_id": series_id,
                "metric": req.metric,
                "value": float(metric_value),
            }
        )
        overall_true.extend(y_true)
        overall_pred.extend(y_pred)
        overall_train.extend(train_y)

    overall = metric_value_fn(
        req.metric,
        y_true=overall_true,
        y_pred=overall_pred,
        train_y=overall_train,
    )

    by_h_entries = [
        {
            "horizon": horizon,
            "metric": req.metric,
            "value": float(
                metric_value_fn(
                    req.metric,
                    y_true=by_h_true[horizon],
                    y_pred=by_h_pred[horizon],
                    train_y=by_h_train[horizon],
                )
            ),
        }
        for horizon in range(1, req.horizon + 1)
    ]

    by_f_entries = [
        {
            "fold": fold,
            "metric": req.metric,
            "value": float(
                metric_value_fn(
                    req.metric,
                    y_true=by_f_true[fold],
                    y_pred=by_f_pred[fold],
                    train_y=by_f_train[fold],
                )
            ),
        }
        for fold in range(1, req.folds + 1)
        if by_f_true[fold]
    ]

    return BacktestResponse(
        metrics={req.metric: float(overall)},
        by_series=per_series_entries or None,
        by_horizon=by_h_entries or None,
        by_fold=by_f_entries or None,
    )