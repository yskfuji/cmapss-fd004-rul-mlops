# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportUnnecessaryIsInstance=false
from __future__ import annotations

import math
from datetime import timedelta
from typing import Any, Literal, cast

import numpy as np
from forecasting_api.errors import ApiError
from forecasting_api.schemas import (
    BacktestRequest,
    BacktestResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    TimeSeriesRecord,
)


def _bi(en: str, ja: str) -> str:
    return f"[EN] {en}\n[JA] {ja}"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_float_list(value: Any) -> list[float]:
    return [
        float(item)
        for item in _as_list(value)
        if isinstance(item, (int, float)) and math.isfinite(float(item))
    ]


def predict_hgb_next(
    bundle: dict[str, Any], *, context_records: list[dict[str, Any]], feature_keys: list[str]
) -> tuple[float, float, float]:
    features: list[float] = []
    for row in context_records:
        x_dict = _as_dict(row.get("x"))
        features.extend(
            float(x_dict.get(key, 0.0)) if isinstance(x_dict.get(key, 0.0), (int, float)) else 0.0
            for key in feature_keys
        )
    x_row = np.asarray([features], dtype=float)
    pred = float(np.asarray(bundle["point"].predict(x_row), dtype=float).reshape(-1)[0])
    low_raw = float(np.asarray(bundle["q05"].predict(x_row), dtype=float).reshape(-1)[0])
    high_raw = float(np.asarray(bundle["q95"].predict(x_row), dtype=float).reshape(-1)[0])
    return pred, min(low_raw, high_raw), max(low_raw, high_raw)


def ridge_lags_choose_k(n: int) -> int:
    if n <= 3:
        return 1
    if n <= 8:
        return 2
    if n <= 20:
        return 5
    return 14


def ridge_lags_fit_series(ys: list[float], *, lag_k: int) -> dict[str, Any]:
    from sklearn.linear_model import Ridge

    n = len(ys)
    k = max(1, min(int(lag_k), max(1, n - 1)))
    rows: list[list[float]] = []
    targets: list[float] = []
    for t in range(k, n):
        feat = [float(ys[t - i]) for i in range(1, k + 1)]
        if any(not math.isfinite(v) for v in feat):
            continue
        y_t = float(ys[t])
        if not math.isfinite(y_t):
            continue
        rows.append(feat)
        targets.append(y_t)

    if len(rows) < 2:
        return {
            "algo": "naive_last_value",
            "lag_k": k,
            "coef": [0.0] * k,
            "intercept": float(ys[-1]) if ys else 0.0,
            "residuals": [],
        }

    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    model = Ridge(alpha=1.0, fit_intercept=True, random_state=0)
    model.fit(X, y)
    tail_n = min(50, max(8, int(len(rows) * 0.3)))
    tail_start = max(0, len(rows) - tail_n)
    preds = cast(
        list[Any],
        np.asarray(model.predict(X[tail_start:]), dtype=float).reshape(-1).tolist(),
    )
    tail_targets = cast(
        list[Any],
        np.asarray(y[tail_start:], dtype=float).reshape(-1).tolist(),
    )
    abs_err = [float(abs(float(p) - float(t))) for p, t in zip(preds, tail_targets)]
    coef_values = cast(
        list[Any],
        np.asarray(getattr(model, "coef_", []), dtype=float).reshape(-1).tolist(),
    )
    return {
        "algo": "ridge_lags_v1",
        "lag_k": k,
        "coef": [float(c) for c in coef_values],
        "intercept": float(getattr(model, "intercept_", 0.0)),
        "residuals": [
            float(e) for e in abs_err if isinstance(e, (int, float)) and math.isfinite(float(e))
        ],
    }


def ridge_lags_forecast_series(
    state: dict[str, Any], *, last_values: list[float], horizon: int
) -> list[float]:
    k = int(state.get("lag_k") or 1)
    coef = [float(value) for value in _as_list(state.get("coef")) if isinstance(value, (int, float))]
    intercept = float(state.get("intercept") or 0.0)
    hist = [
        float(v)
        for v in (last_values or [])
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if len(hist) < k:
        pad = hist[-1] if hist else intercept
        hist = ([pad] * (k - len(hist))) + hist

    preds: list[float] = []
    for _ in range(int(horizon)):
        feat = [float(hist[-i]) for i in range(1, k + 1)]
        y_hat = intercept
        for j, xj in enumerate(feat):
            try:
                cj = float(coef[j])
            except Exception:
                cj = 0.0
            y_hat += cj * xj
        preds.append(float(y_hat))
        hist.append(float(y_hat))
    return preds


def quantile_nearest_rank(values: list[float], q: float) -> float:
    xs = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not xs:
        return 0.0
    qq = 0.0 if q <= 0 else 1.0 if q >= 1 else float(q)
    idx = max(0, min(len(xs) - 1, int(math.ceil(qq * len(xs))) - 1))
    return float(xs[idx])


def build_residuals_evidence(values: list[float]) -> dict[str, Any] | None:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not xs:
        return None
    xs = xs[-500:]
    xs_sorted = sorted(xs)

    def _pct(q: float) -> float:
        qq = 0.0 if q <= 0 else 1.0 if q >= 1 else float(q)
        idx = max(0, min(len(xs_sorted) - 1, int(math.ceil(qq * len(xs_sorted))) - 1))
        return float(xs_sorted[idx])

    mn = float(xs_sorted[0])
    mx = float(xs_sorted[-1])
    bins = 20
    span = (mx - mn) if mx > mn else 1.0
    counts = [0] * bins
    for value in xs_sorted:
        idx = int((float(value) - mn) / span * bins)
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    return {
        "kind": "abs_error_residuals",
        "n": len(xs_sorted),
        "p50": _pct(0.5),
        "p90": _pct(0.9),
        "p95": _pct(0.95),
        "min": mn,
        "max": mx,
        "hist": {"bins": bins, "min": mn, "max": mx, "counts": counts},
    }


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return float(math.sqrt(var))


def _infer_seasonal_period_steps(step: timedelta) -> int | None:
    step_sec = step.total_seconds()
    if step_sec <= 0:
        return None
    day = 24 * 60 * 60
    week = 7 * day
    if step_sec < day:
        season = round(day / step_sec)
    elif step_sec <= week * 2:
        season = round(week / step_sec)
    else:
        return None
    return season if season >= 2 else None


def forecast_series_values(
    ys: list[float],
    horizon: int,
    step: timedelta,
) -> tuple[list[float], float]:
    n = len(ys)
    if n == 0:
        return [0.0] * horizon, 0.0
    season = _infer_seasonal_period_steps(step)
    if season and n >= season * 2:
        base = ys[-season:]
        residuals = [ys[-season + idx] - ys[-2 * season + idx] for idx in range(season)]
        scale = safe_std(residuals)
        preds = [float(base[idx % season]) for idx in range(horizon)]
        return preds, scale
    if n >= 2:
        diffs = [ys[idx] - ys[idx - 1] for idx in range(1, n)]
        window = min(len(diffs), max(3, int(len(diffs) * 0.3)))
        tail = diffs[-window:] if window > 0 else diffs
        avg_diff = sum(tail) / len(tail) if tail else 0.0
        scale = safe_std(tail)
        preds = [float(ys[-1] + avg_diff * (idx + 1)) for idx in range(horizon)]
        return preds, scale
    return [float(ys[-1])] * horizon, 0.0


def naive_forecast(req: ForecastRequest, step: timedelta) -> ForecastResponse:
    by_series: dict[str, list[TimeSeriesRecord]] = {}
    for record in req.data:
        by_series.setdefault(record.series_id, []).append(record)

    residuals: list[float] = []
    for rows in by_series.values():
        rows_sorted = sorted(rows, key=lambda row: row.timestamp)
        ys = [float(row.y) for row in rows_sorted]
        for idx in range(1, len(ys)):
            residuals.append(abs(ys[idx] - ys[idx - 1]))

    calib_min = 12
    has_calib = len(residuals) >= calib_min
    quantiles: list[float] | None = None
    max_dev = 0.0
    if req.quantiles:
        quantiles = [
            float(q) for q in req.quantiles if isinstance(q, (int, float)) and 0.0 < float(q) < 1.0
        ]
        max_dev = max((abs(q - 0.5) for q in quantiles), default=0.0)
    levels: list[float] | None = None
    if req.level:
        levels = [
            float(l) for l in req.level if isinstance(l, (int, float)) and 0.0 < float(l) <= 100.0
        ]

    qhat_for_quantiles = None
    qhat_by_level: dict[str, float] | None = None
    if has_calib:
        if quantiles and max_dev > 0:
            coverage = max(0.0, min(0.999, 2.0 * max_dev))
            qhat_for_quantiles = quantile_nearest_rank(residuals, coverage)
        if levels:
            qhat_by_level = {}
            for level in levels:
                coverage = max(0.0, min(0.999, float(level) / 100.0))
                qhat_by_level[str(level)] = quantile_nearest_rank(residuals, coverage)

    warnings: list[str] = []
    if not has_calib:
        warnings.append(
            _bi(
                f"CALIB01: insufficient history for split conformal (n_residuals={len(residuals)} < {calib_min}).",
                f"CALIB01: split conformal の履歴が不足しています（n_residuals={len(residuals)} < {calib_min}）。",
            )
        )

    calibration: dict[str, Any] | None = None
    if has_calib and (
        qhat_for_quantiles is not None or (qhat_by_level is not None and len(qhat_by_level) > 0)
    ):
        calibration = {
            "method": "split_conformal_abs_error",
            "residuals_n": len(residuals),
            "base": "one_step_naive_abs_error",
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

    forecasts: list[ForecastPoint] = []
    for series_id, rows in by_series.items():
        rows_sorted = sorted(rows, key=lambda row: row.timestamp)
        last = rows_sorted[-1]
        last_ts = last.timestamp
        ys = [float(row.y) for row in rows_sorted]
        preds, scale = forecast_series_values(ys, req.horizon, step)
        for horizon_idx in range(1, req.horizon + 1):
            ts = last_ts + step * horizon_idx
            q_map: dict[str, float] | None = None
            intervals: list[dict[str, float]] | None = None
            point = preds[horizon_idx - 1]
            uncertainty = scale * math.sqrt(horizon_idx)
            if has_calib:
                if quantiles and max_dev > 0 and qhat_for_quantiles is not None:
                    uncertainty = float(qhat_for_quantiles) * math.sqrt(horizon_idx)
                elif levels and qhat_by_level:
                    uncertainty = 0.0
            if quantiles:
                q_map = {}
                denom = 2.0 * max_dev if max_dev > 0 else 1.0
                scale_for_offsets = (uncertainty / denom) if denom > 0 else 0.0
                for quantile in quantiles:
                    offset = (float(quantile) - 0.5) * 2.0 * scale_for_offsets
                    q_map[str(quantile)] = float(point + offset)
            if levels:
                intervals = []
                for level in levels:
                    level_value = float(level)
                    if has_calib and qhat_by_level and str(level) in qhat_by_level:
                        width = float(qhat_by_level[str(level)]) * math.sqrt(horizon_idx)
                    else:
                        width = uncertainty * max(0.5, level_value / 100.0)
                    intervals.append(
                        {
                            "level": level_value,
                            "lower": float(point - width),
                            "upper": float(point + width),
                        }
                    )
            forecasts.append(
                ForecastPoint(
                    series_id=series_id,
                    timestamp=ts,
                    point=point,
                    quantiles=q_map,
                    intervals=intervals,
                    uncertainty=None,
                    explanation=None,
                )
            )
    return ForecastResponse(
        forecasts=forecasts,
        warnings=warnings or None,
        calibration=calibration,
        residuals_evidence=build_residuals_evidence(residuals),
        model_explainability=None,
        uncertainty_summary=None,
    )


def forecast_with_trained_model(
    req: ForecastRequest,
    *,
    step: timedelta,
    trained: dict[str, Any],
    ridge_lags_choose_k_fn: Any = None,
    ridge_lags_fit_series_fn: Any = None,
    ridge_lags_forecast_series_fn: Any = None,
    safe_std_fn: Any = None,
    quantile_nearest_rank_fn: Any = None,
    build_residuals_evidence_fn: Any = None,
    naive_forecast_fn: Any = None,
) -> ForecastResponse:
    ridge_lags_choose_k_fn = ridge_lags_choose_k_fn or ridge_lags_choose_k
    ridge_lags_fit_series_fn = ridge_lags_fit_series_fn or ridge_lags_fit_series
    ridge_lags_forecast_series_fn = ridge_lags_forecast_series_fn or ridge_lags_forecast_series
    safe_std_fn = safe_std_fn or safe_std
    quantile_nearest_rank_fn = quantile_nearest_rank_fn or quantile_nearest_rank
    build_residuals_evidence_fn = build_residuals_evidence_fn or build_residuals_evidence
    naive_forecast_fn = naive_forecast_fn or naive_forecast

    by_series: dict[str, list[TimeSeriesRecord]] = {}
    for record in req.data:
        by_series.setdefault(record.series_id, []).append(record)

    model_state = _as_dict(trained.get("state")) or trained
    algo = str((trained.get("algo") or model_state.get("algo") or "")).strip().lower()
    series_state = _as_dict(model_state.get("series"))
    residuals = _as_float_list(model_state.get("pooled_residuals"))
    calib_min = 12
    has_calib = len(residuals) >= calib_min
    quantiles: list[float] | None = None
    max_dev = 0.0
    if req.quantiles:
        quantiles = [
            float(q) for q in req.quantiles if isinstance(q, (int, float)) and 0.0 < float(q) < 1.0
        ]
        max_dev = max((abs(q - 0.5) for q in quantiles), default=0.0)
    levels: list[float] | None = None
    if req.level:
        levels = [
            float(l) for l in req.level if isinstance(l, (int, float)) and 0.0 < float(l) <= 100.0
        ]
    qhat_for_quantiles = None
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
    if algo not in {"ridge_lags_v1", "naive_last_value"}:
        warnings.append(
            _bi(
                f"MODEL01: unknown trained algo='{algo}', fallback to naive.",
                f"MODEL01: 未対応の学習アルゴリズム '{algo}' のため naive にフォールバックします。",
            )
        )
        response = naive_forecast_fn(req, step)
        response.warnings = warnings
        return response
    if not has_calib:
        warnings.append(
            _bi(
                f"CALIB01: insufficient residuals for split conformal (n={len(residuals)} < {calib_min}).",
                f"CALIB01: split conformal の残差が不足しています（n={len(residuals)} < {calib_min}）。",
            )
        )
    calibration: dict[str, Any] | None = None
    if has_calib and (
        qhat_for_quantiles is not None or (qhat_by_level is not None and len(qhat_by_level) > 0)
    ):
        calibration = {
            "method": "split_conformal_abs_error",
            "residuals_n": len(residuals),
            "base": "ridge_lags_one_step_abs_error",
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
    forecasts: list[ForecastPoint] = []
    for series_id, rows in by_series.items():
        rows_sorted = sorted(rows, key=lambda row: row.timestamp)
        last = rows_sorted[-1]
        last_ts = last.timestamp
        ys = [float(row.y) for row in rows_sorted]
        state = series_state.get(series_id)
        if not isinstance(state, dict):
            state = None
        if not state:
            lag_k = ridge_lags_choose_k_fn(len(ys))
            state = ridge_lags_fit_series_fn(ys, lag_k=lag_k)
            k = int(state.get("lag_k") or 1)
            state["last_values"] = ys[-k:] if ys else []
        k = int(state.get("lag_k") or 1)
        last_values = _as_float_list(state.get("last_values")) or (ys[-k:] if ys else [])
        preds = ridge_lags_forecast_series_fn(
            state,
            last_values=[float(value) for value in last_values],
            horizon=req.horizon,
        )
        for horizon_idx in range(1, req.horizon + 1):
            ts = last_ts + step * horizon_idx
            point = float(preds[horizon_idx - 1])
            q_map: dict[str, float] | None = None
            intervals: list[dict[str, float]] | None = None
            if has_calib and qhat_for_quantiles is not None:
                uncertainty = float(qhat_for_quantiles) * math.sqrt(horizon_idx)
            elif has_calib and qhat_by_level:
                uncertainty = 0.0
            else:
                uncertainty = safe_std_fn(_as_float_list(state.get("residuals"))) * math.sqrt(horizon_idx)
            if quantiles:
                q_map = {}
                denom = 2.0 * max_dev if max_dev > 0 else 1.0
                scale_for_offsets = (uncertainty / denom) if denom > 0 else 0.0
                for quantile in quantiles:
                    offset = (float(quantile) - 0.5) * 2.0 * scale_for_offsets
                    q_map[str(quantile)] = float(point + offset)
            if levels:
                intervals = []
                for level in levels:
                    level_value = float(level)
                    if has_calib and qhat_by_level and str(level) in qhat_by_level:
                        width = float(qhat_by_level[str(level)]) * math.sqrt(horizon_idx)
                    else:
                        width = (uncertainty or 0.0) * max(0.5, level_value / 100.0)
                    intervals.append(
                        {
                            "level": level_value,
                            "lower": float(point - width),
                            "upper": float(point + width),
                        }
                    )
            forecasts.append(
                ForecastPoint(
                    series_id=series_id,
                    timestamp=ts,
                    point=point,
                    quantiles=q_map,
                    intervals=intervals,
                    uncertainty=None,
                    explanation=None,
                )
            )
    return ForecastResponse(
        forecasts=forecasts,
        warnings=warnings or None,
        calibration=calibration,
        residuals_evidence=build_residuals_evidence_fn(residuals),
        model_explainability=None,
        uncertainty_summary=None,
    )


def forecast_with_gbdt_model(
    req: ForecastRequest,
    *,
    step: timedelta,
    trained: dict[str, Any],
    read_json: Any,
    artifact_abspath: Any,
    load_joblib_artifact: Any,
    predict_hgb_next_fn: Any = None,
    quantile_nearest_rank_fn: Any = None,
    build_residuals_evidence_fn: Any = None,
) -> ForecastResponse:
    predict_hgb_next_fn = predict_hgb_next_fn or predict_hgb_next
    quantile_nearest_rank_fn = quantile_nearest_rank_fn or quantile_nearest_rank
    build_residuals_evidence_fn = build_residuals_evidence_fn or build_residuals_evidence

    art = _as_dict(trained.get("artifact"))
    snapshot_rel = art.get("snapshot_json") if isinstance(art.get("snapshot_json"), str) else None
    gbdt_rel = art.get("gbdt_joblib") if isinstance(art.get("gbdt_joblib"), str) else None
    if not snapshot_rel or not gbdt_rel:
        raise ApiError(
            status_code=400,
            error_code="M02",
            message="GBDT model artifact が見つかりません",
            details={"model_id": req.model_id, "next_action": "再学習（/v1/train）してください"},
        )
    snapshot = read_json(artifact_abspath(snapshot_rel))
    gbdt_bundle = load_joblib_artifact(artifact_abspath(gbdt_rel))
    context_len = int(trained.get("context_len") or snapshot.get("context_len") or 14)
    feature_keys = [str(key) for key in snapshot.get("feature_keys") or [] if isinstance(key, str)]
    residuals = _as_float_list(trained.get("pooled_residuals"))
    calib_min = 12
    has_calib = len(residuals) >= calib_min
    quantiles: list[float] | None = None
    max_dev = 0.0
    if req.quantiles:
        quantiles = [
            float(q) for q in req.quantiles if isinstance(q, (int, float)) and 0.0 < float(q) < 1.0
        ]
        max_dev = max((abs(q - 0.5) for q in quantiles), default=0.0)
    levels: list[float] | None = None
    if req.level:
        levels = [
            float(l) for l in req.level if isinstance(l, (int, float)) and 0.0 < float(l) <= 100.0
        ]
    qhat_for_quantiles = None
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
    calibration: dict[str, Any] | None = None
    if has_calib and (
        qhat_for_quantiles is not None or (qhat_by_level is not None and len(qhat_by_level) > 0)
    ):
        calibration = {
            "method": "split_conformal_abs_error",
            "residuals_n": len(residuals),
            "base": "gbdt_hgb_one_step_abs_error",
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
    warnings: list[str] = []
    if not has_calib:
        warnings.append(
            _bi(
                f"CALIB01: insufficient residuals for split conformal (n={len(residuals)} < {calib_min}).",
                f"CALIB01: split conformal の残差が不足しています（n={len(residuals)} < {calib_min}）。",
            )
        )
    by_series: dict[str, list[TimeSeriesRecord]] = {}
    for record in req.data:
        by_series.setdefault(record.series_id, []).append(record)
    forecasts: list[ForecastPoint] = []
    for series_id, rows in by_series.items():
        rows_sorted = sorted(rows, key=lambda row: row.timestamp)
        last_ts = rows_sorted[-1].timestamp
        context_records = [
            {"timestamp": row.timestamp.isoformat(), "y": float(row.y), "x": _as_dict(row.x)}
            for row in rows_sorted[-context_len:]
        ]
        need = max(1, context_len)
        if len(context_records) < need:
            pad = (
                dict(context_records[0])
                if context_records
                else {"timestamp": last_ts.isoformat(), "y": 0.0, "x": {}}
            )
            context_records = [dict(pad) for _ in range(need - len(context_records))] + context_records
        last_x = _as_dict(context_records[-1].get("x")) if context_records else {}
        for horizon_idx in range(1, req.horizon + 1):
            point, _low, _high = predict_hgb_next_fn(
                gbdt_bundle,
                context_records=context_records[-context_len:],
                feature_keys=feature_keys,
            )
            ts = last_ts + step * horizon_idx
            uncertainty = (
                float(qhat_for_quantiles) * math.sqrt(horizon_idx)
                if has_calib and qhat_for_quantiles is not None
                else 0.0
            )
            q_map: dict[str, float] | None = None
            if quantiles:
                q_map = {}
                denom = 2.0 * max_dev if max_dev > 0 else 1.0
                scale_for_offsets = (uncertainty / denom) if denom > 0 else 0.0
                for quantile in quantiles:
                    offset = (float(quantile) - 0.5) * 2.0 * scale_for_offsets
                    q_map[str(quantile)] = float(point + offset)
            intervals: list[dict[str, float]] | None = None
            if levels:
                intervals = []
                for level in levels:
                    if has_calib and qhat_by_level and str(level) in qhat_by_level:
                        width = float(qhat_by_level[str(level)]) * math.sqrt(horizon_idx)
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
                    point=float(point),
                    quantiles=q_map,
                    intervals=intervals,
                    uncertainty=None,
                    explanation=None,
                )
            )
            context_records.append({"timestamp": ts.isoformat(), "y": float(point), "x": dict(last_x)})
    return ForecastResponse(
        forecasts=forecasts,
        warnings=warnings or None,
        calibration=calibration,
        residuals_evidence=build_residuals_evidence_fn(residuals),
        model_explainability=None,
        uncertainty_summary=None,
    )


def metric_value(
    metric: Literal["mae", "rmse", "mape", "smape", "mase", "wape", "nasa_score"],
    *,
    y_true: list[float],
    y_pred: list[float],
    train_y: list[float],
) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    errs = [pred - truth for truth, pred in zip(y_true, y_pred)]
    if metric == "mae":
        return float(sum(abs(err) for err in errs) / len(errs))
    if metric == "rmse":
        return float(math.sqrt(sum(err * err for err in errs) / len(errs)))
    if metric == "nasa_score":
        total = 0.0
        for err in errs:
            total += math.exp((-err) / 13.0) - 1.0 if err < 0 else math.exp(err / 10.0) - 1.0
        return float(total)
    if metric == "mape":
        fracs = [abs(err) / abs(truth) for truth, err in zip(y_true, errs) if truth != 0]
        return float(sum(fracs) / len(fracs)) if fracs else 0.0
    if metric == "smape":
        fracs = []
        for truth, pred in zip(y_true, y_pred):
            denom = abs(truth) + abs(pred)
            if denom == 0:
                continue
            fracs.append(2.0 * abs(pred - truth) / denom)
        return float(sum(fracs) / len(fracs)) if fracs else 0.0
    if metric == "wape":
        denom = sum(abs(truth) for truth in y_true)
        return float(sum(abs(err) for err in errs) / denom) if denom != 0 else 0.0
    if metric == "mase":
        diffs = [abs(train_y[idx] - train_y[idx - 1]) for idx in range(1, len(train_y))]
        scale = (sum(diffs) / len(diffs)) if diffs else 0.0
        mae = sum(abs(err) for err in errs) / len(errs)
        return float(mae / scale) if scale != 0 else 0.0
    return 0.0


def naive_backtest(req: BacktestRequest) -> BacktestResponse:
    by_series_records: dict[str, list[TimeSeriesRecord]] = {}
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
        ys = [float(row.y) for row in rows]
        n = len(ys)
        if n < req.horizon + 2:
            continue
        y_true: list[float] = []
        y_pred: list[float] = []
        train_y: list[float] = []
        for fold_idx in range(req.folds):
            end = n - fold_idx * req.horizon
            start = end - req.horizon
            train_end = start
            if train_end <= 0 or start < 0 or end > n:
                break
            train = ys[:train_end]
            actual = ys[start:end]
            if len(actual) != req.horizon or not train:
                break
            pred_val = float(train[-1])
            preds = [pred_val] * req.horizon
            y_true.extend(actual)
            y_pred.extend(preds)
            train_y.extend(train)
            for horizon_idx in range(req.horizon):
                step_idx = horizon_idx + 1
                by_h_true[step_idx].append(float(actual[horizon_idx]))
                by_h_pred[step_idx].append(float(preds[horizon_idx]))
                by_h_train[step_idx].extend(train)
            by_f_true[fold_idx + 1].extend([float(value) for value in actual])
            by_f_pred[fold_idx + 1].extend([float(value) for value in preds])
            by_f_train[fold_idx + 1].extend([float(value) for value in train])
        if not y_true:
            continue
        value = metric_value(req.metric, y_true=y_true, y_pred=y_pred, train_y=train_y)
        per_series_entries.append({"series_id": series_id, "metric": req.metric, "value": float(value)})
        overall_true.extend(y_true)
        overall_pred.extend(y_pred)
        overall_train_y.extend(train_y)
    overall = metric_value(req.metric, y_true=overall_true, y_pred=overall_pred, train_y=overall_train_y)
    by_h_entries = [
        {
            "horizon": horizon_idx,
            "metric": req.metric,
            "value": float(
                metric_value(
                    req.metric,
                    y_true=by_h_true[horizon_idx],
                    y_pred=by_h_pred[horizon_idx],
                    train_y=by_h_train[horizon_idx],
                )
            ),
        }
        for horizon_idx in range(1, req.horizon + 1)
    ]
    by_f_entries = [
        {
            "fold": fold_idx,
            "metric": req.metric,
            "value": float(
                metric_value(
                    req.metric,
                    y_true=by_f_true[fold_idx],
                    y_pred=by_f_pred[fold_idx],
                    train_y=by_f_train[fold_idx],
                )
            ),
        }
        for fold_idx in range(1, req.folds + 1)
        if by_f_true[fold_idx]
    ]
    return BacktestResponse(
        metrics={req.metric: float(overall)},
        by_series=per_series_entries or None,
        by_horizon=by_h_entries or None,
        by_fold=by_f_entries or None,
    )


def ridge_lags_backtest(
    req: BacktestRequest,
    *,
    trained: dict[str, Any] | None = None,
    ridge_lags_choose_k_fn: Any = None,
    ridge_lags_fit_series_fn: Any = None,
    ridge_lags_forecast_series_fn: Any = None,
    metric_value_fn: Any = None,
) -> BacktestResponse:
    ridge_lags_choose_k_fn = ridge_lags_choose_k_fn or ridge_lags_choose_k
    ridge_lags_fit_series_fn = ridge_lags_fit_series_fn or ridge_lags_fit_series
    ridge_lags_forecast_series_fn = ridge_lags_forecast_series_fn or ridge_lags_forecast_series
    metric_value_fn = metric_value_fn or metric_value

    by_series_records: dict[str, list[TimeSeriesRecord]] = {}
    for record in req.data:
        by_series_records.setdefault(record.series_id, []).append(record)
    if trained and isinstance(trained, dict) and isinstance(trained.get("state"), dict):
        model_state = trained.get("state")
    elif trained and isinstance(trained, dict):
        model_state = trained
    else:
        model_state = None
    series_state_hint = (
        model_state.get("series")
        if isinstance(model_state, dict) and isinstance(model_state.get("series"), dict)
        else {}
    )
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
        ys = [float(row.y) for row in rows]
        n = len(ys)
        if n < req.horizon + 3:
            continue
        hint = series_state_hint.get(series_id) if isinstance(series_state_hint, dict) else None
        hint_k = None
        if isinstance(hint, dict) and hint.get("lag_k") is not None:
            try:
                raw_hint_k = hint.get("lag_k")
                if raw_hint_k is not None:
                    hint_k = int(raw_hint_k)
            except Exception:
                hint_k = None
        y_true: list[float] = []
        y_pred: list[float] = []
        train_y: list[float] = []
        for fold_idx in range(req.folds):
            end = n - fold_idx * req.horizon
            start = end - req.horizon
            train_end = start
            if train_end <= 0 or start < 0 or end > n:
                break
            train = ys[:train_end]
            actual = ys[start:end]
            if len(actual) != req.horizon or len(train) < 3:
                break
            lag_k = hint_k if hint_k is not None else ridge_lags_choose_k_fn(len(train))
            state = ridge_lags_fit_series_fn(train, lag_k=lag_k)
            k = int(state.get("lag_k") or 1)
            last_vals = train[-k:] if train else []
            preds = ridge_lags_forecast_series_fn(state, last_values=last_vals, horizon=req.horizon)
            y_true.extend([float(value) for value in actual])
            y_pred.extend([float(value) for value in preds])
            train_y.extend([float(value) for value in train])
            for horizon_idx in range(req.horizon):
                step_idx = horizon_idx + 1
                by_h_true[step_idx].append(float(actual[horizon_idx]))
                by_h_pred[step_idx].append(float(preds[horizon_idx]))
                by_h_train[step_idx].extend(train)
            by_f_true[fold_idx + 1].extend([float(value) for value in actual])
            by_f_pred[fold_idx + 1].extend([float(value) for value in preds])
            by_f_train[fold_idx + 1].extend([float(value) for value in train])
        if not y_true:
            continue
        value = metric_value_fn(req.metric, y_true=y_true, y_pred=y_pred, train_y=train_y)
        per_series_entries.append({"series_id": series_id, "metric": req.metric, "value": float(value)})
        overall_true.extend(y_true)
        overall_pred.extend(y_pred)
        overall_train_y.extend(train_y)
    overall = metric_value_fn(req.metric, y_true=overall_true, y_pred=overall_pred, train_y=overall_train_y)
    by_h_entries = [
        {
            "horizon": horizon_idx,
            "metric": req.metric,
            "value": float(
                metric_value_fn(
                    req.metric,
                    y_true=by_h_true[horizon_idx],
                    y_pred=by_h_pred[horizon_idx],
                    train_y=by_h_train[horizon_idx],
                )
            ),
        }
        for horizon_idx in range(1, req.horizon + 1)
    ]
    by_f_entries = [
        {
            "fold": fold_idx,
            "metric": req.metric,
            "value": float(
                metric_value_fn(
                    req.metric,
                    y_true=by_f_true[fold_idx],
                    y_pred=by_f_pred[fold_idx],
                    train_y=by_f_train[fold_idx],
                )
            ),
        }
        for fold_idx in range(1, req.folds + 1)
        if by_f_true[fold_idx]
    ]
    return BacktestResponse(
        metrics={req.metric: float(overall)},
        by_series=per_series_entries or None,
        by_horizon=by_h_entries or None,
        by_fold=by_f_entries or None,
    )


def gbdt_backtest(
    req: BacktestRequest,
    *,
    trained: dict[str, Any],
    read_json: Any,
    artifact_abspath: Any,
    load_joblib_artifact: Any,
    predict_hgb_next_fn: Any = None,
    metric_value_fn: Any = None,
) -> BacktestResponse:
    predict_hgb_next_fn = predict_hgb_next_fn or predict_hgb_next
    metric_value_fn = metric_value_fn or metric_value

    art = _as_dict(trained.get("artifact"))
    snapshot_rel = art.get("snapshot_json") if isinstance(art.get("snapshot_json"), str) else None
    gbdt_rel = art.get("gbdt_joblib") if isinstance(art.get("gbdt_joblib"), str) else None
    if not snapshot_rel or not gbdt_rel:
        raise ApiError(
            status_code=400,
            error_code="M02",
            message="GBDT model artifact が見つかりません",
            details={"model_id": req.model_id, "next_action": "再学習（/v1/train）してください"},
        )
    snapshot = read_json(artifact_abspath(snapshot_rel))
    gbdt_bundle = load_joblib_artifact(artifact_abspath(gbdt_rel))
    context_len = int(trained.get("context_len") or snapshot.get("context_len") or 14)
    feature_keys = [str(key) for key in snapshot.get("feature_keys") or [] if isinstance(key, str)]
    by_series_records: dict[str, list[TimeSeriesRecord]] = {}
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
        if len(rows_sorted) < req.horizon + max(3, context_len):
            continue
        y_true: list[float] = []
        y_pred: list[float] = []
        train_y: list[float] = []
        for fold_idx in range(req.folds):
            end = len(rows_sorted) - fold_idx * req.horizon
            start = end - req.horizon
            train_end = start
            if train_end <= 0 or start < 0 or end > len(rows_sorted):
                break
            train_rows = rows_sorted[:train_end]
            actual_rows = rows_sorted[start:end]
            if len(actual_rows) != req.horizon or len(train_rows) < 2:
                break
            context_records = [
                {
                    "timestamp": row.timestamp.isoformat(),
                    "y": float(row.y),
                    "x": dict(row.x or {}),
                }
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
                context_records = [dict(pad) for _ in range(need - len(context_records))] + context_records
            fold_preds: list[float] = []
            for actual_row in actual_rows:
                pred, _low, _high = predict_hgb_next_fn(
                    gbdt_bundle,
                    context_records=context_records[-context_len:],
                    feature_keys=feature_keys,
                )
                fold_preds.append(float(pred))
                context_records.append(
                    {
                        "timestamp": actual_row.timestamp.isoformat(),
                        "y": float(pred),
                        "x": dict(actual_row.x or {}),
                    }
                )
            actual = [float(row.y) for row in actual_rows]
            train = [float(row.y) for row in train_rows]
            y_true.extend(actual)
            y_pred.extend(fold_preds)
            train_y.extend(train)
            for horizon_idx, (truth, pred) in enumerate(zip(actual, fold_preds), start=1):
                by_h_true[horizon_idx].append(float(truth))
                by_h_pred[horizon_idx].append(float(pred))
                by_h_train[horizon_idx].extend(train)
                by_f_true[fold_idx + 1].append(float(truth))
                by_f_pred[fold_idx + 1].append(float(pred))
                by_f_train[fold_idx + 1].extend(train)
        if not y_true:
            continue
        per_series_metric = metric_value_fn(req.metric, y_true=y_true, y_pred=y_pred, train_y=train_y)
        per_series_entries.append(
            {"series_id": series_id, "metric": req.metric, "value": float(per_series_metric)}
        )
        overall_true.extend(y_true)
        overall_pred.extend(y_pred)
        overall_train_y.extend(train_y)
    overall = metric_value_fn(req.metric, y_true=overall_true, y_pred=overall_pred, train_y=overall_train_y)
    by_h_entries = [
        {
            "horizon": horizon_idx,
            "metric": req.metric,
            "value": float(
                metric_value_fn(
                    req.metric,
                    y_true=by_h_true[horizon_idx],
                    y_pred=by_h_pred[horizon_idx],
                    train_y=by_h_train[horizon_idx],
                )
            ),
        }
        for horizon_idx in range(1, req.horizon + 1)
    ]
    by_f_entries = [
        {
            "fold": fold_idx,
            "metric": req.metric,
            "value": float(
                metric_value_fn(
                    req.metric,
                    y_true=by_f_true[fold_idx],
                    y_pred=by_f_pred[fold_idx],
                    train_y=by_f_train[fold_idx],
                )
            ),
        }
        for fold_idx in range(1, req.folds + 1)
        if by_f_true[fold_idx]
    ]
    return BacktestResponse(
        metrics={req.metric: float(overall)},
        by_series=per_series_entries or None,
        by_horizon=by_h_entries or None,
        by_fold=by_f_entries or None,
    )
