"""GBDT-based RUL prediction pipeline for CMAPSS FD004.

Key design decisions:
- Phase-aware Z-score normalisation per operating-condition cluster
  (6 conditions in FD004 → avoids cross-condition distribution leakage)
- Multi-scale rolling features: sub-windows [5, 10, 20, 30] × {mean, std, min, max, trend, last}
  plus diff stats and linear slope → rich temporal representation without deep learning
- Exponential sample weights to counteract RUL=125 label clipping imbalance
  (~50% of training samples sit at y=125 without this correction)
- LightGBM + CatBoost ensemble with inverse-RMSE weighting for the point predictor
- HGB quantile models for 90% prediction intervals, calibrated on a held-out split
"""

from __future__ import annotations

import logging
import math
from typing import Any, TypeAlias

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "WINDOW",
    "FEATURES",
    "MAX_RUL",
    "NormStats",
    "as_float",
    "as_dict",
    "record_series_id",
    "group_records",
    "op_cluster_key",
    "select_feature_keys",
    "compute_norm_stats",
    "normalise_value",
    "rolling_features",
    "window_matrix",
    "gbdt_feature_vector",
    "build_gbdt_dataset",
    "build_gbdt_calibration_proxy_dataset",
    "rul_sample_weights",
    "fit_lgbm_catboost_ensemble",
    "predict_interval_dataset",
    "interval_metrics_from_scaled_deltas",
    "calibration_split",
    "fit_gbdt_pipeline",
    "predict_rul",
]

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW = 30       # context window length (cycles)
FEATURES = 24     # max sensor features before constant-sensor pruning
MAX_RUL = 125.0   # piecewise-linear RUL cap (CMAPSS convention)

# Multi-scale sub-windows for rolling feature extraction
_SUB_WINDOWS = (5, 10, 20, 30)
_DEFAULT_CALIBRATION_HOLDOUT_FRACTION = 0.1
_MIN_CALIBRATION_HOLDOUT_SERIES = 8
_MAX_CALIBRATION_HOLDOUT_SERIES = 24
_CALIBRATION_SPLIT_RANDOM_STATE = 0

# Type alias: operating-condition cluster → {sensor_key: (mean, std)}
NormStats: TypeAlias = dict[tuple[int, int, int], dict[str, tuple[float, float]]]


def _resolve_fit_preset(preset: str) -> str:
    raw = str(preset or "full").strip().lower()
    return "fast" if raw == "fast" else "full"


def _fit_search_spaces(
    preset: str,
) -> tuple[list[dict[str, Any]], list[tuple[float, float]], list[float], int]:
    if _resolve_fit_preset(preset) == "fast":
        return (
            [
                {"max_iter": 60, "learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 31, "min_samples_leaf": 10},
                {"max_iter": 90, "learning_rate": 0.04, "max_depth": 4, "max_leaf_nodes": 31, "min_samples_leaf": 8},
            ],
            [(0.05, 0.95), (0.1, 0.9)],
            [1.0, 1.25, 1.5],
            1,
        )
    return (
        [
            {"max_iter": 200, "learning_rate": 0.03, "max_depth": 3, "max_leaf_nodes": 31, "min_samples_leaf": 8},
            {"max_iter": 300, "learning_rate": 0.05, "max_depth": 4, "max_leaf_nodes": 31, "min_samples_leaf": 8},
            {"max_iter": 400, "learning_rate": 0.03, "max_depth": 5, "max_leaf_nodes": 63, "min_samples_leaf": 5},
            {"max_iter": 350, "learning_rate": 0.04, "max_depth": 6, "max_leaf_nodes": 63, "min_samples_leaf": 5},
            {"max_iter": 500, "learning_rate": 0.025, "max_depth": 7, "max_leaf_nodes": 127, "min_samples_leaf": 4},
            {"max_iter": 600, "learning_rate": 0.02, "max_depth": 8, "max_leaf_nodes": 127, "min_samples_leaf": 3},
        ],
        [(0.001, 0.999), (0.005, 0.995), (0.01, 0.99), (0.02, 0.98), (0.03, 0.97), (0.05, 0.95), (0.08, 0.92), (0.1, 0.9)],
        [1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0],
        2,
    )


def _predict_interval_dataset(
    point_model: HistGradientBoostingRegressor,
    lower_model: HistGradientBoostingRegressor,
    upper_model: HistGradientBoostingRegressor,
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> dict[str, np.ndarray]:
    if x_data.size == 0 or y_data.size == 0:
        empty = np.zeros((0,), dtype=float)
        return {"y_true": empty, "y_pred": empty, "low_delta": empty, "high_delta": empty}
    pred = np.asarray(point_model.predict(x_data), dtype=float).reshape(-1)
    low = np.asarray(lower_model.predict(x_data), dtype=float).reshape(-1)
    high = np.asarray(upper_model.predict(x_data), dtype=float).reshape(-1)
    lower = np.minimum(low, high)
    upper = np.maximum(low, high)
    return {
        "y_true": np.asarray(y_data, dtype=float).reshape(-1),
        "y_pred": pred,
        "low_delta": np.maximum(0.0, pred - lower),
        "high_delta": np.maximum(0.0, upper - pred),
    }


def predict_interval_dataset(
    point_model: HistGradientBoostingRegressor,
    lower_model: HistGradientBoostingRegressor,
    upper_model: HistGradientBoostingRegressor,
    *,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> dict[str, np.ndarray]:
    return _predict_interval_dataset(
        point_model,
        lower_model,
        upper_model,
        x_data=x_data,
        y_data=y_data,
    )


def _interval_metrics_from_scaled_deltas(
    outputs: dict[str, np.ndarray],
    *,
    interval_scale: float,
) -> dict[str, float]:
    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    lower = y_pred - interval_scale * outputs["low_delta"]
    upper = y_pred + interval_scale * outputs["high_delta"]
    cov = float(np.mean((lower <= y_true) & (y_true <= upper))) if y_true.size else float("nan")
    width = float(np.mean(upper - lower)) if y_true.size else float("nan")
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else float("nan")
    return {"cov90": cov, "width90": width, "rmse": rmse}


def interval_metrics_from_scaled_deltas(
    outputs: dict[str, np.ndarray],
    *,
    interval_scale: float,
) -> dict[str, float]:
    return _interval_metrics_from_scaled_deltas(outputs, interval_scale=interval_scale)


def _calibration_split(
    records: list[dict[str, object]],
    *,
    holdout_fraction: float = _DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
    min_holdout_series: int = _MIN_CALIBRATION_HOLDOUT_SERIES,
    max_holdout_series: int = _MAX_CALIBRATION_HOLDOUT_SERIES,
    random_state: int = _CALIBRATION_SPLIT_RANDOM_STATE,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    series_ids = sorted(_group_records(records))
    if len(series_ids) <= 2:
        return records, records
    fraction = min(max(float(holdout_fraction), 0.1), 0.9)
    calib_n = max(int(min_holdout_series), int(math.ceil(len(series_ids) * fraction)))
    calib_n = min(calib_n, int(max_holdout_series))
    calib_n = max(1, min(calib_n, len(series_ids) - 1))
    shuffled = list(series_ids)
    rng = np.random.default_rng(int(random_state))
    rng.shuffle(shuffled)
    calib_ids = set(shuffled[:calib_n])
    fit_ids = set(series_ids) - calib_ids
    fit_records = [record for record in records if _record_series_id(record) in fit_ids]
    calib_records = [record for record in records if _record_series_id(record) in calib_ids]
    return (fit_records or records, calib_records or records)


def calibration_split(
    records: list[dict[str, object]],
    *,
    holdout_fraction: float = _DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
    min_holdout_series: int = _MIN_CALIBRATION_HOLDOUT_SERIES,
    max_holdout_series: int = _MAX_CALIBRATION_HOLDOUT_SERIES,
    random_state: int = _CALIBRATION_SPLIT_RANDOM_STATE,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return _calibration_split(
        records,
        holdout_fraction=holdout_fraction,
        min_holdout_series=min_holdout_series,
        max_holdout_series=max_holdout_series,
        random_state=random_state,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _as_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def as_float(value: object, default: float = 0.0) -> float:
    return _as_float(value, default)


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def as_dict(value: object) -> dict[str, object]:
    return _as_dict(value)


def _record_series_id(record: dict[str, object]) -> str:
    return str(record.get("series_id") or record.get("unit_id") or "unknown").strip()


def record_series_id(record: dict[str, object]) -> str:
    return _record_series_id(record)


def _group_records(
    records: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Group records by series_id (engine unit)."""
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        sid = _record_series_id(record)
        groups.setdefault(sid, []).append(record)
    for sid in list(groups):
        groups[sid] = sorted(groups[sid], key=lambda row: str(row.get("timestamp") or ""))
    return groups


def group_records(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return _group_records(records)


# ── Operating-condition cluster ───────────────────────────────────────────────

def op_cluster_key(record: dict[str, object]) -> tuple[int, int, int]:
    """Discretise (altitude, Mach, TRA) into a cluster key.

    FD004 has 6 distinct operating conditions encoded in op_setting_{1,2,3}.
    Rounding to integer units gives a stable discrete key for per-cluster stats.
    """
    x = _as_dict(record.get("x"))
    return (
        round(_as_float(x.get("op_setting_1"))),
        round(_as_float(x.get("op_setting_2")) * 100),
        round(_as_float(x.get("op_setting_3"))),
    )


# ── Feature selection: drop constant sensors ──────────────────────────────────

def select_feature_keys(records: list[dict[str, object]]) -> list[str]:
    """Return active sensor keys, dropping constants (std ≤ 1e-4) and op_settings.

    FD004 has sensors {1..21} but several are near-constant across all conditions
    (e.g. sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19).
    Including them adds noise without signal.
    """
    values: dict[str, list[float]] = {}
    for record in records:
        for key, value in _as_dict(record.get("x")).items():
            if key == "cycle" or key.startswith("op_setting"):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.setdefault(key, []).append(float(value))
    active = sorted(
        key for key, vals in values.items()
        if len(vals) > 1 and float(np.std(vals)) > 1e-4
    )
    return active[:FEATURES]


# ── Phase-aware normalisation ─────────────────────────────────────────────────

def compute_norm_stats(
    records: list[dict[str, object]],
    *,
    feature_keys: list[str],
) -> NormStats:
    """Compute per-cluster Z-score parameters from training records.

    Normalising within each operating condition prevents the model from
    learning sensor-level offsets that are purely due to altitude/Mach changes
    rather than degradation.
    """
    buckets: dict[tuple[int, int, int], dict[str, list[float]]] = {}
    for record in records:
        cluster = op_cluster_key(record)
        x = _as_dict(record.get("x"))
        buckets.setdefault(cluster, {})
        for key in feature_keys:
            v = _as_float(x.get(key), float("nan"))
            if math.isfinite(v):
                buckets[cluster].setdefault(key, []).append(v)
    stats: NormStats = {}
    for cluster, key_vals in buckets.items():
        stats[cluster] = {}
        for key, vals in key_vals.items():
            arr = np.asarray(vals, dtype=float)
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            stats[cluster][key] = (mu, max(sigma, 1e-6))
    return stats


def _normalise_value(
    v: float,
    key: str,
    cluster: tuple[int, int, int],
    norm_stats: NormStats,
) -> float:
    """Z-score normalise v using cluster stats; fall back to global mean if missing."""
    cluster_stats = norm_stats.get(cluster) or {}
    if key in cluster_stats:
        mu, sigma = cluster_stats[key]
    else:
        all_mu = [s[key][0] for s in norm_stats.values() if key in s]
        all_sigma = [s[key][1] for s in norm_stats.values() if key in s]
        mu = float(np.mean(all_mu)) if all_mu else 0.0
        sigma = float(np.mean(all_sigma)) if all_sigma else 1.0
    return (v - mu) / sigma


def normalise_value(v: float, key: str, cluster: tuple[int, int, int], norm_stats: NormStats) -> float:
    return _normalise_value(v, key, cluster, norm_stats)


# ── Multi-scale rolling feature extraction ────────────────────────────────────

def rolling_features(window: np.ndarray) -> np.ndarray:
    """Compute rich rolling statistics from a normalised window (WINDOW × n_sensors).

    For each sensor, across sub-windows [5, 10, 20, 30]:
        mean, std, min, max, trend (last − first), last value
    Plus for each sensor over the full window:
        diff mean, diff std (rate-of-change statistics)
        linear regression slope (long-range degradation trend)

    Total output dimension: n_sensors × (6 × 4 + 3)
    """
    n, n_feat = window.shape
    if n == 0:
        return np.zeros(n_feat * (6 * len(_SUB_WINDOWS) + 3), dtype=float)

    parts = []
    for sw in _SUB_WINDOWS:
        sub = window[-min(sw, n):]
        parts.append(np.mean(sub, axis=0))
        parts.append(np.std(sub, axis=0))
        parts.append(np.min(sub, axis=0))
        parts.append(np.max(sub, axis=0))
        parts.append(sub[-1] - sub[0])   # trend within sub-window
        parts.append(sub[-1])            # most recent value

    # 1st-order diff stats over full window
    if n > 1:
        diff = np.diff(window, axis=0)
        parts.append(np.mean(diff, axis=0))
        parts.append(np.std(diff, axis=0))
    else:
        parts.append(np.zeros(n_feat, dtype=float))
        parts.append(np.zeros(n_feat, dtype=float))

    # Linear slope via least-squares (long-range trend)
    t = np.arange(n, dtype=float)
    t_c = t - t.mean()
    denom = float(np.sum(t_c ** 2)) or 1.0
    slope = (t_c @ window) / denom
    parts.append(slope)

    return np.concatenate(parts)


def _window_matrix(
    rows: list[dict[str, object]],
    *,
    feature_keys: list[str],
    norm_stats: NormStats | None = None,
) -> np.ndarray:
    """Build (WINDOW × n_sensors) matrix, optionally phase-normalised."""
    context = rows[-WINDOW:]
    if len(context) < WINDOW:
        pad = context[0]
        context = [pad] * (WINDOW - len(context)) + context
    matrix = []
    for row in context:
        x = _as_dict(row.get("x"))
        cluster = op_cluster_key(row) if norm_stats is not None else (0, 0, 0)
        row_vals = []
        for key in feature_keys:
            v = _as_float(x.get(key))
            if norm_stats is not None:
                v = _normalise_value(v, key, cluster, norm_stats)
            row_vals.append(v)
        matrix.append(row_vals)
    return np.asarray(matrix, dtype=float)


def window_matrix(
    rows: list[dict[str, object]],
    *,
    feature_keys: list[str],
    norm_stats: NormStats | None = None,
) -> np.ndarray:
    return _window_matrix(rows, feature_keys=feature_keys, norm_stats=norm_stats)


def gbdt_feature_vector(
    rows: list[dict[str, object]],
    *,
    feature_keys: list[str],
    norm_stats: NormStats,
) -> np.ndarray:
    """Flat feature vector for one prediction: rolling stats + cluster ID.

    Appends (altitude_rounded, mach_rounded/100, TRA_rounded) so the model
    can learn condition-specific degradation curves without one-hot encoding.
    """
    win = _window_matrix(rows, feature_keys=feature_keys, norm_stats=norm_stats)
    rolling = rolling_features(win)
    cluster = op_cluster_key(rows[-1])
    op_feats = np.array(
        [float(cluster[0]), float(cluster[1]) / 100.0, float(cluster[2])],
        dtype=float,
    )
    return np.concatenate([rolling, op_feats])


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_gbdt_dataset(
    records: list[dict[str, object]],
    *,
    feature_keys: list[str],
    norm_stats: NormStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays by sliding a window over every engine time-series."""
    by_series = _group_records(records)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for rows in by_series.values():
        if len(rows) <= WINDOW:
            continue
        for idx in range(WINDOW, len(rows)):
            window_rows = rows[idx - WINDOW:idx]
            if norm_stats is not None:
                fv = gbdt_feature_vector(
                    window_rows, feature_keys=feature_keys, norm_stats=norm_stats
                )
            else:
                fv = _window_matrix(window_rows, feature_keys=feature_keys).reshape(-1)
            xs.append(fv)
            ys.append(_as_float(rows[idx].get("y")))
    if not xs:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)
    return np.vstack(xs), np.asarray(ys, dtype=float)


def _resolve_proxy_targets(
    target_ruls: list[float] | np.ndarray | None,
    *,
    target_count: int,
) -> np.ndarray:
    if target_count <= 0:
        return np.zeros((0,), dtype=float)
    raw = np.asarray([] if target_ruls is None else target_ruls, dtype=float).reshape(-1)
    finite = raw[np.isfinite(raw)]
    positive = finite[(finite > 0.0) & (finite <= MAX_RUL)]
    if positive.size == 0:
        return np.linspace(10.0, MAX_RUL, num=target_count, dtype=float)
    sorted_positive = np.sort(positive, axis=None).astype(float)
    if sorted_positive.size == target_count:
        return sorted_positive
    quantiles = np.linspace(0.0, 1.0, num=target_count, dtype=float)
    return np.quantile(sorted_positive, quantiles, method="nearest").astype(float)


def _proxy_target_index(rows: list[dict[str, object]], *, target_rul: float) -> int | None:
    if len(rows) <= WINDOW:
        return None
    candidates: list[int] = []
    for idx in range(WINDOW, len(rows)):
        target = _as_float(rows[idx].get("y"), default=float("nan"))
        if math.isfinite(target) and 0.0 < target <= MAX_RUL:
            candidates.append(idx)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda idx: (
            abs(_as_float(rows[idx].get("y"), default=0.0) - target_rul),
            -idx,
        ),
    )


def build_gbdt_calibration_proxy_dataset(
    records: list[dict[str, object]],
    *,
    feature_keys: list[str],
    norm_stats: NormStats,
    target_ruls: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one benchmark-like proxy calibration row per held-out unit.

    Calibration should match the benchmark's per-engine evaluation grain rather
    than scoring every sliding-window timestep from the held-out units. When
    target_ruls is supplied, proxy rows are chosen to match that empirical RUL
    distribution via deterministic quantile assignment.
    """
    by_series = _group_records(records)
    ordered_rows = [series_rows for _, series_rows in sorted(by_series.items())]
    proxy_targets = _resolve_proxy_targets(target_ruls, target_count=len(ordered_rows))
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for target_rul, rows in zip(proxy_targets, ordered_rows, strict=True):
        proxy_idx = _proxy_target_index(rows, target_rul=float(target_rul))
        if proxy_idx is None:
            continue
        window_rows = rows[proxy_idx - WINDOW:proxy_idx]
        xs.append(
            gbdt_feature_vector(
                window_rows,
                feature_keys=feature_keys,
                norm_stats=norm_stats,
            )
        )
        ys.append(_as_float(rows[proxy_idx].get("y")))
    if not xs:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)
    return np.vstack(xs), np.asarray(ys, dtype=float)


# ── Label-imbalance correction ────────────────────────────────────────────────

def rul_sample_weights(
    y: np.ndarray,
    *,
    max_rul: float = MAX_RUL,
    tau: float = 40.0,
) -> np.ndarray:
    """Exponential weights that upweight low-RUL (near-failure) samples.

    w_i = exp(-y_i / tau), normalised so mean weight = 1.

    Without this, the piecewise-linear RUL cap causes ~50% of training samples
    to pile up at y=125, biasing the model toward healthy-engine predictions.
    tau=40 gives ~3× more weight to y=0 vs y=125.
    """
    w = np.exp(-np.clip(y, 0.0, max_rul) / tau)
    mean_w = float(np.mean(w))
    if mean_w > 0:
        w = w / mean_w
    return w.astype(float)


# ── LightGBM + CatBoost ensemble ─────────────────────────────────────────────

def fit_lgbm_catboost_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    w_train: np.ndarray,
) -> tuple[Any, Any, float]:
    """Train LightGBM and CatBoost point models; return (lgbm, catboost, lgbm_weight).

    Each model uses early stopping against the validation set.
    Final blend weight is determined by inverse-RMSE on the validation set:
        lgbm_weight = (1/RMSE_lgbm) / (1/RMSE_lgbm + 1/RMSE_cb)

    Requires the optional dependencies lightgbm and catboost. Calling this
    function without them installed raises ImportError at runtime.
    """
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    lgbm_model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        lambda_l2=1.0,
        min_child_samples=10,
        n_estimators=2000,
        random_state=0,
        verbose=-1,
        n_jobs=-1,
    )
    lgbm_model.fit(
        x_train, y_train,
        sample_weight=w_train,
        eval_set=[(x_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    cb_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=3.0,
        subsample=0.8,
        random_seed=0,
        loss_function="RMSE",
        early_stopping_rounds=100,
        verbose=False,
    )
    cb_model.fit(x_train, y_train, sample_weight=w_train, eval_set=(x_valid, y_valid))

    lgbm_preds = np.asarray(lgbm_model.predict(x_valid), dtype=float)
    cb_preds = np.asarray(cb_model.predict(x_valid), dtype=float)
    lgbm_rmse = float(np.sqrt(np.mean((lgbm_preds - y_valid) ** 2))) or 1.0
    cb_rmse = float(np.sqrt(np.mean((cb_preds - y_valid) ** 2))) or 1.0

    w_lgbm = 1.0 / lgbm_rmse
    w_cb = 1.0 / cb_rmse
    lgbm_weight = w_lgbm / (w_lgbm + w_cb)
    return lgbm_model, cb_model, lgbm_weight


# ── Full pipeline: train + return model bundle ─────────────────────────────────

def fit_gbdt_pipeline(
    train_records: list[dict[str, object]],
    *,
    feature_keys: list[str] | None = None,
    preset: str = "full",
    enable_ensemble: bool | None = None,
    interval_calibration_targets: list[float] | np.ndarray | None = None,
    interval_calibration_holdout_fraction: float = _DEFAULT_CALIBRATION_HOLDOUT_FRACTION,
    interval_calibration_min_holdout_series: int = _MIN_CALIBRATION_HOLDOUT_SERIES,
    interval_calibration_max_holdout_series: int = _MAX_CALIBRATION_HOLDOUT_SERIES,
) -> dict[str, Any]:
    """Train the full GBDT pipeline and return a model bundle.

    Pipeline steps:
        1. Select active sensor features (drop constants)
        2. Compute phase-aware normalisation statistics
        3. Build training dataset with multi-scale rolling features
        4. Grid-search HGB hyperparameters; fit quantile models for 90% interval
        5. Train LightGBM + CatBoost ensemble as point predictor

    Returns a bundle dict with keys:
        point       HGB point model (fallback)
        q05 / q95   HGB quantile models for prediction interval
        lgbm        LightGBM point model
        catboost    CatBoost point model
        lgbm_weight Blend weight for lgbm (1 − weight for catboost)
        norm_stats  Phase-aware normalisation statistics
        feature_keys Active sensor keys used
        interval_scale Calibrated scale factor for the 90% interval
    """
    if feature_keys is None:
        feature_keys = select_feature_keys(train_records)

    norm_stats = compute_norm_stats(train_records, feature_keys=feature_keys)
    x_all, y_all = build_gbdt_dataset(
        train_records, feature_keys=feature_keys, norm_stats=norm_stats
    )
    if x_all.size == 0 or y_all.size == 0:
        raise RuntimeError("No training samples produced from the provided records")

    # Engine-aware train/val split: hold out the last ~20% of engines so that
    # no engine appears in both train and val (GroupKFold equivalent).
    # Row-level tail slicing would leak time-series autocorrelation across the
    # boundary; engine-level holdout prevents this.
    all_series = sorted(_group_records(train_records))
    # With fewer than 3 engines the engine-level split degenerates (1 val / 1 train).
    # Fall back to using all data for both train and validation in that case.
    if len(all_series) < 3:
        x_train, y_train = x_all, y_all
        x_valid, y_valid = x_all, y_all
    else:
        val_n = max(1, min(len(all_series) // 5, len(all_series) - 1))
        val_series = set(all_series[-val_n:])
        val_records = [r for r in train_records if _record_series_id(r) in val_series]
        fit_records_inner = [r for r in train_records if _record_series_id(r) not in val_series]
        x_train, y_train = build_gbdt_dataset(
            fit_records_inner, feature_keys=feature_keys, norm_stats=norm_stats
        )
        x_valid, y_valid = build_gbdt_dataset(
            val_records, feature_keys=feature_keys, norm_stats=norm_stats
        )
        if x_train.size == 0 or y_train.size == 0:
            x_train, y_train = x_all, y_all
            x_valid, y_valid = x_all, y_all
    w_train = rul_sample_weights(y_train)
    w_all = rul_sample_weights(y_all)

    resolved_preset = _resolve_fit_preset(preset)
    train_ensemble = resolved_preset == "full" if enable_ensemble is None else bool(enable_ensemble)
    candidates, interval_quantile_pairs, interval_scale_grid, top_interval_candidates = _fit_search_spaces(resolved_preset)
    best_params = candidates[0]
    best_score = float("inf")
    ranked_candidates: list[tuple[float, dict[str, Any]]] = []
    for params in candidates:
        model = HistGradientBoostingRegressor(random_state=0, early_stopping=False, **params)
        model.fit(x_train, y_train, sample_weight=w_train)
        preds = np.asarray(
            model.predict(x_valid if x_valid.size > 0 else x_train), dtype=float
        )
        truth = y_valid if y_valid.size > 0 else y_train
        score = float(np.sqrt(np.mean((preds - truth) ** 2)))
        ranked_candidates.append((score, dict(params)))
        if score < best_score:
            best_score = score
            best_params = dict(params)
    ranked_candidates.sort(key=lambda item: item[0])

    interval_candidates = [params for _, params in ranked_candidates[: max(1, min(top_interval_candidates, len(ranked_candidates)))]]
    best_interval_pair = (0.05, 0.95)
    best_interval_scale = 1.0
    best_interval_params = dict(best_params)
    best_interval_cov = 0.0
    best_interval_width = float("inf")
    best_interval_objective = float("inf")
    fit_records, calib_records = _calibration_split(
        train_records,
        holdout_fraction=interval_calibration_holdout_fraction,
        min_holdout_series=interval_calibration_min_holdout_series,
        max_holdout_series=interval_calibration_max_holdout_series,
    )
    fit_x, fit_y = build_gbdt_dataset(fit_records, feature_keys=feature_keys, norm_stats=norm_stats)
    calib_x, calib_y = build_gbdt_calibration_proxy_dataset(
        calib_records,
        feature_keys=feature_keys,
        norm_stats=norm_stats,
        target_ruls=interval_calibration_targets,
    )
    calibration_protocol = "heldout_unit_proxy_final_cycle"
    if calib_x.size == 0 or calib_y.size == 0:
        calib_x, calib_y = build_gbdt_dataset(
            calib_records,
            feature_keys=feature_keys,
            norm_stats=norm_stats,
        )
        calibration_protocol = "heldout_unit_sliding_window_fallback"
    calib_unit_count = int(len(_group_records(calib_records)))
    train_unit_count = int(len(_group_records(train_records)))
    actual_holdout_fraction = float(calib_unit_count) / max(train_unit_count, 1)
    if x_valid.size > 0 and y_valid.size > 0:
        y_scale = max(float(np.std(y_train)), 1.0)
        if fit_x.size > 0 and fit_y.size > 0 and calib_x.size > 0 and calib_y.size > 0:
            w_fit = rul_sample_weights(fit_y)
            for params in interval_candidates:
                for q_low, q_high in interval_quantile_pairs:
                    point_model = HistGradientBoostingRegressor(random_state=0, early_stopping=False, **params)
                    lower_candidate = HistGradientBoostingRegressor(random_state=0, early_stopping=False, loss="quantile", quantile=q_low, **params)
                    upper_candidate = HistGradientBoostingRegressor(random_state=0, early_stopping=False, loss="quantile", quantile=q_high, **params)
                    point_model.fit(fit_x, fit_y, sample_weight=w_fit)
                    lower_candidate.fit(fit_x, fit_y, sample_weight=w_fit)
                    upper_candidate.fit(fit_x, fit_y, sample_weight=w_fit)
                    calib_outputs = _predict_interval_dataset(
                        point_model,
                        lower_candidate,
                        upper_candidate,
                        x_data=calib_x,
                        y_data=calib_y,
                    )
                    if calib_outputs["y_true"].size == 0:
                        continue
                    for interval_scale in interval_scale_grid:
                        calib_metrics = _interval_metrics_from_scaled_deltas(calib_outputs, interval_scale=interval_scale)
                        cov = float(calib_metrics["cov90"])
                        width = float(calib_metrics["width90"])
                        rmse = float(calib_metrics["rmse"])
                        under = max(0.0, 0.9 - cov)
                        objective = abs(cov - 0.9) + (4.0 * under) + 0.003 * (width / y_scale) + 0.01 * (rmse / y_scale)
                        if objective < best_interval_objective:
                            best_interval_objective = objective
                            best_interval_params = dict(params)
                            best_interval_pair = (q_low, q_high)
                            best_interval_scale = float(interval_scale)
                            best_interval_cov = cov
                            best_interval_width = width

    lower_model = HistGradientBoostingRegressor(
        random_state=0, early_stopping=False,
        loss="quantile", quantile=best_interval_pair[0], **best_interval_params,
    )
    upper_model = HistGradientBoostingRegressor(
        random_state=0, early_stopping=False,
        loss="quantile", quantile=best_interval_pair[1], **best_interval_params,
    )
    final_model = HistGradientBoostingRegressor(
        random_state=0, early_stopping=False, **best_params
    )
    final_model.fit(x_all, y_all, sample_weight=w_all)
    lower_model.fit(x_all, y_all, sample_weight=w_all)
    upper_model.fit(x_all, y_all, sample_weight=w_all)

    # LightGBM + CatBoost ensemble
    lgbm_model = catboost_model = None
    lgbm_weight = 0.5
    ensemble_validation_rmse = best_score
    if train_ensemble and x_valid.size > 0 and y_valid.size > 0:
        try:
            lgbm_model, catboost_model, lgbm_weight = fit_lgbm_catboost_ensemble(
                x_train, y_train, x_valid, y_valid, w_train,
            )
            lgbm_preds = np.asarray(lgbm_model.predict(x_valid), dtype=float)
            cb_preds = np.asarray(catboost_model.predict(x_valid), dtype=float)
            ens_preds = lgbm_weight * lgbm_preds + (1 - lgbm_weight) * cb_preds
            ensemble_validation_rmse = float(np.sqrt(np.mean((ens_preds - y_valid) ** 2)))
        except Exception:
            _LOGGER.exception(
                "LightGBM+CatBoost ensemble fit failed; falling back to single HGB model",
                extra={"feature_dim": int(x_train.shape[1]), "train_samples": int(x_train.shape[0])},
            )
            lgbm_model = catboost_model = None

    fit_meta = {
        "preset": resolved_preset,
        "ensemble_enabled": train_ensemble,
        "feature_dim": int(x_all.shape[1]),
        "samples": int(x_all.shape[0]),
        "validation_rmse": best_score,
        "ensemble_validation_rmse": ensemble_validation_rmse,
        "best_params": best_params,
        "interval_method": "quantile_boosting_90",
        "interval_params": best_interval_params,
        "interval_quantiles": [best_interval_pair[0], best_interval_pair[1]],
        "interval_scale": best_interval_scale,
        "interval_validation_cov90": best_interval_cov,
        "interval_validation_width90": best_interval_width,
        "interval_calibration_protocol": calibration_protocol,
        "interval_calibration_units": calib_unit_count,
        "interval_calibration_rows": int(calib_y.shape[0]),
        "interval_calibration_target_mean": float(np.mean(calib_y)) if calib_y.size else None,
        "interval_calibration_target_p50": float(np.quantile(calib_y, 0.5)) if calib_y.size else None,
        "interval_calibration_holdout_fraction": actual_holdout_fraction,
        "interval_calibration_requested_holdout_fraction": float(interval_calibration_holdout_fraction),
        "interval_calibration_min_holdout_series": int(interval_calibration_min_holdout_series),
        "interval_calibration_max_holdout_series": int(interval_calibration_max_holdout_series),
    }

    return {
        "point": final_model,
        "q05": lower_model,
        "q95": upper_model,
        "lgbm": lgbm_model,
        "catboost": catboost_model,
        "lgbm_weight": lgbm_weight,
        "norm_stats": norm_stats,
        "feature_keys": feature_keys,
        "interval_scale": best_interval_scale,
        "fit_meta": fit_meta,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_rul(
    bundle: dict[str, Any],
    rows: list[dict[str, object]],
) -> tuple[float, float, float]:
    """Predict RUL point estimate and 90% interval for a single engine context.

    Args:
        bundle:  Model bundle returned by fit_gbdt_pipeline().
        rows:    Recent sensor records for one engine (≥ WINDOW cycles recommended).

    Returns:
        (point_pred, lower_bound, upper_bound)
    """
    feature_keys: list[str] = bundle["feature_keys"]
    norm_stats: NormStats = bundle["norm_stats"]
    interval_scale: float = float(bundle.get("interval_scale") or 1.0)

    fv = gbdt_feature_vector(rows[-WINDOW:], feature_keys=feature_keys, norm_stats=norm_stats).reshape(1, -1)

    # Point prediction: ensemble if available, else HGB fallback
    if bundle.get("lgbm") is not None and bundle.get("catboost") is not None:
        lgbm_w = float(bundle.get("lgbm_weight") or 0.5)
        point = (
            lgbm_w * float(np.asarray(bundle["lgbm"].predict(fv), dtype=float).reshape(-1)[0])
            + (1 - lgbm_w) * float(np.asarray(bundle["catboost"].predict(fv), dtype=float).reshape(-1)[0])
        )
    else:
        point = float(np.asarray(bundle["point"].predict(fv), dtype=float).reshape(-1)[0])

    lo = float(np.asarray(bundle["q05"].predict(fv), dtype=float).reshape(-1)[0])
    hi = float(np.asarray(bundle["q95"].predict(fv), dtype=float).reshape(-1)[0])
    point = min(MAX_RUL, max(0.0, point))
    low_delta = max(0.0, point - min(lo, hi))
    high_delta = max(0.0, max(lo, hi) - point)
    lower = min(MAX_RUL, max(0.0, point - interval_scale * low_delta))
    upper = min(MAX_RUL, max(0.0, point + interval_scale * high_delta))
    return point, lower, upper
