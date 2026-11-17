from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecasting_api.cmapss_fd004 import build_fd004_payload, build_fd004_profile_payload
from forecasting_api.file_store import atomic_write_text, exclusive_lock
from forecasting_api.hybrid_xai_uncertainty import (
    apply_soft_gate_envelope_interval as _shared_apply_soft_gate_envelope_interval,
    condition_advantage_map as _shared_condition_advantage_map,
    interval_overlap_ratio as _shared_interval_overlap_ratio,
    normalize_advantage_lookup as _shared_normalize_advantage_lookup,
    soft_gate_feature_scales as _shared_soft_gate_feature_scales,
    soft_gate_outputs as _shared_soft_gate_outputs,
    soft_gate_weight_entropy as _shared_soft_gate_weight_entropy,
)
from forecasting_api.mlflow_runs import log_artifact, log_dict_artifact, log_metrics, log_params, start_run
from forecasting_api.torch_forecasters import forecast_univariate_torch, train_univariate_torch_forecaster
from models import gbdt_pipeline as shared_gbdt_pipeline

JsonDict = dict[str, object]
NormStats = shared_gbdt_pipeline.NormStats
WINDOW = shared_gbdt_pipeline.WINDOW
FEATURES = shared_gbdt_pipeline.FEATURES
AFNO_ALGO = "afnocg3_v1"
BENCHMARK_AFNO_CACHE_VERSION = 7
_ADAPTER_MATRIX_ABS_CLIP = 1_000.0
_ADAPTER_RESPONSE_ABS_CLIP = 1_000.0
_ADAPTER_RESIDUAL_ABS_CLIP = 250.0


def _benchmark_afno_training_hours() -> float:
    raw = os.getenv("RULFM_BENCHMARK_AFNO_TRAINING_HOURS", "").strip()
    if raw:
        try:
            return max(0.05, float(raw))
        except ValueError:
            pass
    return 0.05


def _benchmark_sota_training_hours() -> float:
    """Training hours for public SOTA models (BiLSTM / TCN / Transformer).

    Default 0.1 h → up to 120 epochs before the time cap; early stopping
    (patience=10) terminates earlier in practice.  Override with
    RULFM_BENCHMARK_SOTA_TRAINING_HOURS to cap or expand the budget.
    """
    raw = os.getenv("RULFM_BENCHMARK_SOTA_TRAINING_HOURS", "").strip()
    if raw:
        try:
            return max(0.05, float(raw))
        except ValueError:
            pass
    return 0.1


def _benchmark_afno_max_epochs() -> int | None:
    raw = os.getenv("RULFM_FORECASTING_TORCH_MAX_EPOCHS", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _benchmark_stage() -> str:
    raw = os.getenv("RULFM_BENCHMARK_STAGE", "full").strip().lower().replace("_", "-")
    if raw in {"gbdt", "gbdt-only", "gbdt-baseline"}:
        return "gbdt-only"
    return "full"


def _benchmark_gbdt_preset() -> str:
    raw = os.getenv("RULFM_BENCHMARK_GBDT_PRESET", "full").strip().lower()
    if raw in {"fast"}:
        return "fast"
    return "full"


def _benchmark_should_train_gbdt_ensemble() -> bool:
    if _benchmark_stage() == "gbdt-only":
        return False
    raw = os.getenv("RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _benchmark_stage() == "full" and _benchmark_gbdt_preset() == "full"


def _benchmark_should_run_afno() -> bool:
    raw = os.getenv("RULFM_BENCHMARK_ENABLE_AFNO", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _benchmark_mode_notes() -> list[str]:
    stage = _benchmark_stage()
    preset = _benchmark_gbdt_preset()
    ensemble_enabled = _benchmark_should_train_gbdt_ensemble()
    afno_enabled = _benchmark_should_run_afno()
    notes = [
        f"Benchmark stage={stage}, gbdt_preset={preset}, gbdt_ensemble_enabled={ensemble_enabled}, afno_enabled={afno_enabled}.",
        "Stage matrix: gbdt-only writes the GBDT baseline snapshot and exits; full continues into SOTA stages after persisting the GBDT baseline, with AFNO stages gated behind RULFM_BENCHMARK_ENABLE_AFNO=1.",
    ]
    if preset == "fast":
        notes.append("GBDT results use fast preset (max_iter <= 90); run with RULFM_BENCHMARK_GBDT_PRESET=full for publication-quality results, especially because interval coverage can remain well below the target in fast runs.")
    if not afno_enabled:
        notes.append("AFNO benchmark stages are skipped by default in the public full run so the summary artifact can complete deterministically; set RULFM_BENCHMARK_ENABLE_AFNO=1 to opt back in.")
    return notes


def _benchmark_summary_paths() -> tuple[Path, Path]:
    data_path = ROOT / "src" / "forecasting_api" / "data" / "fd004_benchmark_summary.json"
    csv_path = ROOT / "docs" / "references" / "fd004_legacy_extract" / "fd004_benchmark_summary.csv"
    return data_path, csv_path


def _write_benchmark_outputs(summary: dict[str, Any], rows: list[JsonDict]) -> tuple[Path, Path]:
    data_path, csv_path = _benchmark_summary_paths()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(data_path):
        atomic_write_text(data_path, json.dumps(summary, ensure_ascii=False, indent=2))
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["model", "rmse", "mae", "nasa_score", "cov90", "width90", "interval_method", "source", "scope", "comparability"])
    for row in rows:
        writer.writerow([row.get("model"), row.get("rmse"), row.get("mae"), row.get("nasa_score"), row.get("cov90"), row.get("width90"), row.get("interval_method"), row.get("source"), row.get("scope"), row.get("comparability")])
    with exclusive_lock(csv_path):
        atomic_write_text(csv_path, csv_buffer.getvalue())
    return data_path, csv_path


def _build_benchmark_summary(
    *,
    feature_keys: list[str],
    horizon: int,
    folds: int,
    rows: list[JsonDict],
    experiments: dict[str, dict[str, Any]],
    benchmark_notes: list[str],
    diagnostics: dict[str, Any] | None = None,
    phase: str = "full",
) -> dict[str, Any]:
    benchmark_stage = _benchmark_stage()
    gbdt_preset = _benchmark_gbdt_preset()
    notes = [
        "Benchmark reset to NASA CMAPSS-style evaluation with RMSE and asymmetric NASA score.",
        "Input is fixed to window=30 and 24 exogenous features (op settings + sensors, cycle excluded).",
        "GBDT now logs 90% intervals via quantile boosting and reports coverage/width.",
    ]
    if phase != "full":
        notes.append(f"Summary snapshot written after the {phase} phase so benchmark progress is persisted incrementally.")
    else:
        notes.extend(
            [
                "AFNO rows include exogenous input, log1p target transform ablation, auto-expand flat fallback ablation, and an integrated log1p+autoexpand variant.",
                "Torch benchmark rows now include asymmetric-RUL loss ablations and low-RUL diagnostics for overprediction-sensitive comparison.",
                "AFNO component ablations now isolate a GraphProjector-held ladder: GraphProjector+FeatureMixer baseline, then +Fourier, +DilatedConv, and +CrossRouter under the same FD004 protocol.",
                "Soft gate now retains three tracked variants: risk-aware baseline, point-first baseline, and correctness meta-gate; final intervals use a separate envelope calibration rule.",
                "Harder OoD diagnostics include unseen-unit holdout and operating-condition holdout comparisons.",
            ]
        )
    notes.extend(_benchmark_mode_notes())
    notes.extend(benchmark_notes)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_stage": benchmark_stage,
        "phase": phase,
        "gbdt_preset": gbdt_preset,
        "train_profile": "fd004_full_cycles",
        "backtest_profile": "fd004_backtest_multiunit",
        "window": WINDOW,
        "features": FEATURES,
        "feature_keys": feature_keys,
        "horizon": horizon,
        "folds": folds,
        "rows": rows,
        "experiments": experiments,
        "diagnostics": diagnostics
        if diagnostics is not None
        else {
            "distribution_shift": None,
            "uncertainty": [
                {
                    "model": row.get("model"),
                    "cov90": row.get("cov90"),
                    "width90": row.get("width90"),
                    "interval_method": row.get("interval_method"),
                }
                for row in rows
                if row.get("comparability") == "same_protocol"
            ],
            "xai_stability": [],
            "architecture_ablations": [],
            "uncertainty_quality": [],
            "harder_ood": {},
            "soft_gate_pareto": {},
            "evaluation": {
                "rmse": "sqrt(mean((y_hat - y)^2))",
                "nasa_score": "error = y_hat - y; exp(-error/13)-1 for error<0 else exp(error/10)-1",
                "bias": "mean(y_hat - y)",
                "overprediction_rate": "mean(y_hat > y)",
                "overprediction_mean": "mean(y_hat - y | y_hat > y)",
                "low_rul_rmse_30": "rmse restricted to y <= 30",
                "low_rul_rmse_60": "rmse restricted to y <= 60",
                "low_rul_nasa_score_30": "nasa_score restricted to y <= 30",
                "low_rul_nasa_score_60": "nasa_score restricted to y <= 60",
                "cov90": "mean(lower_90 <= y <= upper_90)",
                "width90": "mean(upper_90 - lower_90)",
            },
        },
        "notes": notes,
    }


def _headers() -> dict[str, str]:
    return {"X-API-Key": os.environ.get("RULFM_FORECASTING_API_KEY", "dev-key")}


def _benchmark_afno_cache_dir(label: str) -> Path:
    return ROOT / "src" / "forecasting_api" / "data" / "model_artifacts" / f"benchmark_{label}"


def _load_cached_benchmark_afno(label: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    cache_dir = _benchmark_afno_cache_dir(label)
    snapshot_path = cache_dir / "snapshot.json"
    weights_path = cache_dir / "weights.pt"
    meta_path = cache_dir / "meta.json"
    if not snapshot_path.exists() or not weights_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or int(meta.get("cache_version") or 0) != BENCHMARK_AFNO_CACHE_VERSION:
            return None
        expected_hours = _benchmark_afno_training_hours()
        cached_hours = meta.get("training_hours")
        if not isinstance(cached_hours, (int, float)) or not math.isclose(float(cached_hours), float(expected_hours), rel_tol=0.0, abs_tol=1e-12):
            return None
        expected_max_epochs = _benchmark_afno_max_epochs()
        cached_max_epochs = meta.get("max_epochs_env")
        if expected_max_epochs != (int(cached_max_epochs) if isinstance(cached_max_epochs, (int, float)) else None):
            return None
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict):
            return None
        import torch

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
        state_dict = ckpt if isinstance(ckpt, dict) else {"state_dict": ckpt}
        return snapshot, state_dict, meta
    except TypeError:
        try:
            import torch

            ckpt = torch.load(weights_path, map_location="cpu")
            state_dict = ckpt if isinstance(ckpt, dict) else {"state_dict": ckpt}
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            return snapshot if isinstance(snapshot, dict) else {}, state_dict, meta if isinstance(meta, dict) else {}
        except Exception:
            return None
    except Exception:
        return None


def _save_cached_benchmark_afno(label: str, *, snapshot: dict[str, Any], state_dict: dict[str, Any], meta: dict[str, Any]) -> None:
    cache_dir = _benchmark_afno_cache_dir(label)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import torch

        torch.save(state_dict, cache_dir / "weights.pt")
    except Exception:
        return
    cache_meta = {
        **meta,
        "cache_version": BENCHMARK_AFNO_CACHE_VERSION,
        "label": label,
        "training_hours": _benchmark_afno_training_hours(),
        "max_epochs_env": _benchmark_afno_max_epochs(),
    }
    (cache_dir / "meta.json").write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable_model_params(params: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if isinstance(value, (bool, int, float, str)):
            result[str(key)] = value
        elif isinstance(value, (list, tuple)):
            result[str(key)] = list(value)
    return result


def _afno_graph_held_training_protocol() -> dict[str, Any]:
    return {
        "name": "graph_held_constrained_v1",
        "optimizer": "adamw",
        "learning_rate": 2e-4,
        "weight_decay": 1e-3,
        "loss": "huber",
        "huber_delta": 0.5,
        "grad_clip_norm": 0.5,
        "patience": 6,
        "min_delta": 1e-4,
        "aux_loss_weight": 0.1,
    }


def _afno_asymmetric_training_protocol() -> dict[str, Any]:
    return {
        "name": "graph_held_asymmetric_rul_v1",
        "optimizer": "adamw",
        "learning_rate": 2e-4,
        "weight_decay": 1e-3,
        "loss": "asymmetric_rul",
        "asym_over_penalty": 2.0,
        "asym_under_penalty": 1.0,
        "asym_max_rul": 125.0,
        "grad_clip_norm": 0.5,
        "patience": 6,
        "min_delta": 1e-4,
        "aux_loss_weight": 0.1,
    }


def _afno_component_ablation_specs() -> list[dict[str, Any]]:
    return [
        {
            "label": "afnocg3_v1_ablation_graph_feature_mixer_only_w30_f24_log1p",
            "target_transform": "log1p",
            "allow_structure_fallback": False,
            "base_label": None,
            "variant_family": "afno_component_ablation",
            "ablation_stage": "graph_projector_feature_mixer_only",
            "story_component": "graph_projector_regularized_baseline",
            "model_params": {
                "dropout": 0.05,
                "enable_consistency_loss": True,
                "manifold_consistency_weight": 0.05,
                "enable_fourier_mixer": False,
                "enable_temporal_conv": False,
                "enable_feature_mixer": True,
                "enable_router": False,
                "use_cross_router": False,
                "enable_mode_routing": False,
                "enable_multi_manifold": True,
                "projector_node_space": "latent",
                "latent_graph_mode": "running_cov",
                "num_manifolds": 1,
            },
            "training_protocol": _afno_graph_held_training_protocol(),
        },
        {
            "label": "afnocg3_v1_ablation_graph_feature_mixer_fourier_w30_f24_log1p",
            "target_transform": "log1p",
            "allow_structure_fallback": False,
            "base_label": None,
            "variant_family": "afno_component_ablation",
            "ablation_stage": "graph_projector_feature_mixer_fourier",
            "story_component": "long_range_low_frequency_trend",
            "model_params": {
                "dropout": 0.05,
                "enable_consistency_loss": True,
                "manifold_consistency_weight": 0.05,
                "enable_fourier_mixer": True,
                "enable_temporal_conv": False,
                "enable_feature_mixer": True,
                "enable_router": False,
                "use_cross_router": False,
                "enable_mode_routing": False,
                "enable_multi_manifold": True,
                "projector_node_space": "latent",
                "latent_graph_mode": "running_cov",
                "num_manifolds": 1,
            },
            "training_protocol": _afno_graph_held_training_protocol(),
        },
        {
            "label": "afnocg3_v1_ablation_graph_feature_mixer_fourier_dilated_w30_f24_log1p",
            "target_transform": "log1p",
            "allow_structure_fallback": False,
            "base_label": None,
            "variant_family": "afno_component_ablation",
            "ablation_stage": "graph_projector_feature_mixer_fourier_dilated_conv",
            "story_component": "local_transition_multiscale_patterns",
            "model_params": {
                "dropout": 0.05,
                "enable_consistency_loss": True,
                "manifold_consistency_weight": 0.05,
                "enable_fourier_mixer": True,
                "enable_temporal_conv": True,
                "enable_feature_mixer": True,
                "enable_router": False,
                "use_cross_router": False,
                "enable_mode_routing": False,
                "enable_multi_manifold": True,
                "projector_node_space": "latent",
                "latent_graph_mode": "running_cov",
                "num_manifolds": 1,
            },
            "training_protocol": _afno_graph_held_training_protocol(),
        },
        {
            "label": "afnocg3_v1_ablation_graph_feature_mixer_fourier_router_w30_f24_log1p",
            "target_transform": "log1p",
            "allow_structure_fallback": False,
            "base_label": None,
            "variant_family": "afno_component_ablation",
            "ablation_stage": "graph_projector_feature_mixer_fourier_cross_router",
            "story_component": "condition_dependent_path_routing",
            "model_params": {
                "dropout": 0.05,
                "enable_consistency_loss": True,
                "manifold_consistency_weight": 0.05,
                "enable_fourier_mixer": True,
                "enable_temporal_conv": True,
                "enable_feature_mixer": True,
                "enable_router": True,
                "use_cross_router": True,
                "enable_mode_routing": True,
                "enable_multi_manifold": True,
                "projector_node_space": "latent",
                "latent_graph_mode": "running_cov",
                "num_manifolds": 1,
            },
            "training_protocol": _afno_graph_held_training_protocol(),
        },
    ]


def _afno_benchmark_row(*, label: str, metrics: dict[str, float], afno_meta: dict[str, Any]) -> dict[str, Any]:
    row = {
        "model": label,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "nasa_score": metrics["nasa_score"],
        "cov90": metrics["cov90"],
        "width90": metrics["width90"],
        "interval_method": "validation_residual_conformal_90",
        "source": "fd004_benchmark",
        "scope": "standard CMAPSS evaluation: final-cycle RUL per engine",
        "comparability": "same_protocol",
        **_point_diagnostic_row_fields(metrics),
    }
    if afno_meta.get("variant_family") is not None:
        row["variant_family"] = afno_meta.get("variant_family")
    if afno_meta.get("ablation_stage") is not None:
        row["ablation_stage"] = afno_meta.get("ablation_stage")
    if afno_meta.get("story_component") is not None:
        row["story_component"] = afno_meta.get("story_component")
    if isinstance(afno_meta.get("ablation_config"), dict) and afno_meta.get("ablation_config"):
        row["ablation_config"] = afno_meta.get("ablation_config")
    if isinstance(afno_meta.get("training_protocol"), dict) and afno_meta.get("training_protocol"):
        row["training_protocol"] = afno_meta.get("training_protocol")
        row["loss"] = afno_meta["training_protocol"].get("loss")
    return row


def _as_float(value: object, default: float = 0.0) -> float:
    return shared_gbdt_pipeline.as_float(value, default)


def _as_str(value: object) -> str:
    return str(value) if value is not None else ""


def _as_dict(value: object) -> JsonDict:
    return shared_gbdt_pipeline.as_dict(value)


def _rmse(y_true: list[float], y_pred: list[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yp - yt) ** 2))) if yt.size else float("nan")


def _mae(y_true: list[float], y_pred: list[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yp - yt))) if yt.size else float("nan")


def _nasa_score(y_true: list[float], y_pred: list[float]) -> float:
    total = 0.0
    for truth, pred in zip(y_true, y_pred):
        err = float(pred) - float(truth)
        exponent = ((-err) / 13.0) if err < 0 else (err / 10.0)
        total += math.exp(min(exponent, 60.0)) - 1.0
    return float(total)


def _signed_bias(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return float("nan")
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(errors.mean()) if errors.size else float("nan")


def _overprediction_rate(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return float("nan")
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(errors > 0.0)) if errors.size else float("nan")


def _overprediction_mean(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return float("nan")
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    positive = errors[errors > 0.0]
    return float(positive.mean()) if positive.size else 0.0


def _slice_pair(
    y_true: list[float], y_pred: list[float], *, max_rul: float
) -> tuple[list[float], list[float]]:
    sliced_true: list[float] = []
    sliced_pred: list[float] = []
    for truth, pred in zip(y_true, y_pred):
        if float(truth) <= float(max_rul):
            sliced_true.append(float(truth))
            sliced_pred.append(float(pred))
    return sliced_true, sliced_pred


def _point_diagnostic_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    low_30_true, low_30_pred = _slice_pair(y_true, y_pred, max_rul=30.0)
    low_60_true, low_60_pred = _slice_pair(y_true, y_pred, max_rul=60.0)
    return {
        "bias": _signed_bias(y_true, y_pred),
        "overprediction_rate": _overprediction_rate(y_true, y_pred),
        "overprediction_mean": _overprediction_mean(y_true, y_pred),
        "low_rul_count_30": float(len(low_30_true)),
        "low_rul_rmse_30": _rmse(low_30_true, low_30_pred),
        "low_rul_nasa_score_30": (
            _nasa_score(low_30_true, low_30_pred) if low_30_true else float("nan")
        ),
        "low_rul_count_60": float(len(low_60_true)),
        "low_rul_rmse_60": _rmse(low_60_true, low_60_pred),
        "low_rul_nasa_score_60": (
            _nasa_score(low_60_true, low_60_pred) if low_60_true else float("nan")
        ),
    }


def _point_diagnostic_row_fields(metrics: dict[str, float]) -> dict[str, float]:
    keys = (
        "bias",
        "overprediction_rate",
        "overprediction_mean",
        "low_rul_count_30",
        "low_rul_rmse_30",
        "low_rul_nasa_score_30",
        "low_rul_count_60",
        "low_rul_rmse_60",
        "low_rul_nasa_score_60",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _weighted_rmse_from_cluster_csv(path: Path) -> tuple[float | None, int]:
    if not path.exists():
        return (None, 0)
    total = 0
    total_sq = 0.0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                count = int(float(row.get("Count") or 0))
                rmse = float(row.get("RMSE") or "nan")
            except Exception:
                continue
            if count <= 0 or not math.isfinite(rmse):
                continue
            total += count
            total_sq += count * (rmse ** 2)
    return ((math.sqrt(total_sq / total), total) if total > 0 else (None, 0))


def _nearest_rank(values: list[float], q: float) -> float:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return 0.0
    qq = min(max(float(q), 0.0), 1.0)
    idx = max(0, min(len(finite) - 1, math.ceil(qq * len(finite)) - 1))
    return float(finite[idx])


def _fit_affine_point_calibrator(y_true: list[float], y_pred: list[float]) -> tuple[float, float]:
    if not y_true or len(y_true) != len(y_pred):
        return (1.0, 0.0)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size < 2:
        return (1.0, float(np.median(yt - yp)))
    design = np.column_stack([yp, np.ones_like(yp)])
    coef, *_ = np.linalg.lstsq(design, yt, rcond=None)
    return (float(coef[0]), float(coef[1]))


def _apply_affine(values: list[float], *, scale: float, bias: float) -> list[float]:
    return [float(scale * float(value) + bias) for value in values]


def _interval_metrics(y_true: list[float], lower: list[float], upper: list[float]) -> dict[str, float | None]:
    if not y_true or len(y_true) != len(lower) or len(y_true) != len(upper):
        return {"cov90": None, "width90": None}
    covered = 0
    widths: list[float] = []
    for truth, lo, hi in zip(y_true, lower, upper):
        lo_b = min(float(lo), float(hi))
        hi_b = max(float(lo), float(hi))
        widths.append(hi_b - lo_b)
        if lo_b <= float(truth) <= hi_b:
            covered += 1
    return {
        "cov90": float(covered / len(y_true)),
        "width90": float(sum(widths) / len(widths)) if widths else None,
    }


def _series_id(record: dict[str, object]) -> str:
    return shared_gbdt_pipeline.record_series_id(record)


def _records_for_series(records: list[dict[str, object]], allowed: set[str]) -> list[dict[str, object]]:
    return [record for record in records if _series_id(record) in allowed]


def _regime_key(record: dict[str, object]) -> tuple[float, float, float]:
    x = _as_dict(record.get("x"))
    return (
        round(_as_float(x.get("op_setting_1")), 0),
        round(_as_float(x.get("op_setting_2")), 2),
        round(_as_float(x.get("op_setting_3")), 0),
    )


def _condition_cluster_key(record: dict[str, object]) -> str:
    regime = _regime_key(record)
    return f"{regime[0]:.0f}|{regime[1]:.2f}|{regime[2]:.0f}"


def _records_for_regime(records: list[dict[str, object]], regime: tuple[float, float, float], *, include: bool) -> list[dict[str, object]]:
    return [record for record in records if (_regime_key(record) == regime) is include]


def _series_subset_for_regime(
    records: list[dict[str, object]],
    regime: tuple[float, float, float],
    *,
    min_regime_points: int,
    max_series: int,
) -> list[dict[str, object]]:
    selected: list[str] = []
    for series_id, rows in _group_records(records).items():
        regime_points = sum(1 for row in rows if _regime_key(row) == regime)
        if regime_points >= min_regime_points:
            selected.append(series_id)
    return _records_for_series(records, set(selected[: max(1, max_series)]))


def _group_records(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return shared_gbdt_pipeline.group_records(records)


# ── Shared GBDT feature-engineering aliases ──────────────────────────────────
_op_cluster_key = shared_gbdt_pipeline.op_cluster_key
_select_feature_keys = shared_gbdt_pipeline.select_feature_keys
_compute_norm_stats = shared_gbdt_pipeline.compute_norm_stats
_normalise_value = shared_gbdt_pipeline.normalise_value
_rolling_features = shared_gbdt_pipeline.rolling_features
_window_matrix = shared_gbdt_pipeline.window_matrix
_gbdt_feature_vector = shared_gbdt_pipeline.gbdt_feature_vector
_build_gbdt_dataset = shared_gbdt_pipeline.build_gbdt_dataset
_predict_interval_dataset = shared_gbdt_pipeline.predict_interval_dataset
_gbdt_calibration_split = shared_gbdt_pipeline.calibration_split
_rul_sample_weights = shared_gbdt_pipeline.rul_sample_weights
_fit_lgbm_catboost_ensemble = shared_gbdt_pipeline.fit_lgbm_catboost_ensemble
_interval_metrics_from_scaled_deltas = shared_gbdt_pipeline.interval_metrics_from_scaled_deltas


def _window_matrix_selected(rows: list[dict[str, object]], *, feature_keys: list[str], selected_keys: list[str]) -> np.ndarray:
    # Benchmark-only helper: the shared pipeline does not need subset selection.
    if not selected_keys:
        return np.zeros((WINDOW, 0), dtype=float)
    selected = [key for key in feature_keys if key in set(selected_keys)]
    return _window_matrix(rows, feature_keys=selected)


def _gate_interval_holdout_split(
    records: list[dict[str, object]],
    *,
    holdout_fraction: float = 0.5,
    min_holdout_series: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    series_ids = sorted(_group_records(records))
    if len(series_ids) <= 3:
        return records, records
    fraction = min(max(float(holdout_fraction), 0.1), 0.9)
    holdout_n = max(int(min_holdout_series), int(math.ceil(len(series_ids) * fraction)))
    holdout_n = max(1, min(holdout_n, len(series_ids) - 1))
    holdout_ids = set(series_ids[-holdout_n:])
    gate_ids = set(series_ids) - holdout_ids
    gate_records = _records_for_series(records, gate_ids)
    holdout_records = _records_for_series(records, holdout_ids)
    return (gate_records or records, holdout_records or records)


def _standard_eval_target_ruls(records: list[dict[str, object]]) -> list[float]:
    targets: list[float] = []
    for rows in _group_records(records).values():
        if len(rows) < WINDOW:
            continue
        value = _as_float(rows[-1].get("y"), default=float("nan"))
        if math.isfinite(value) and 0.0 < value <= shared_gbdt_pipeline.MAX_RUL:
            targets.append(value)
    return targets


def _fit_hgb(
    train_records: list[dict[str, object]],
    *,
    feature_keys: list[str],
    interval_calibration_targets: list[float] | None = None,
) -> tuple[dict[str, HistGradientBoostingRegressor], dict[str, Any]]:
    bundle = shared_gbdt_pipeline.fit_gbdt_pipeline(
        train_records,
        feature_keys=feature_keys,
        preset=_benchmark_gbdt_preset(),
        enable_ensemble=_benchmark_should_train_gbdt_ensemble(),
        interval_calibration_targets=interval_calibration_targets,
    )
    meta = bundle.get("fit_meta") if isinstance(bundle.get("fit_meta"), dict) else {}
    return bundle, meta


def _predict_gbdt_horizon(
    model_bundle: dict[str, HistGradientBoostingRegressor],
    *,
    history: list[dict[str, object]],
    future_rows: list[dict[str, object]],
    feature_keys: list[str],
    horizon: int,
) -> tuple[list[float], list[float], list[float]]:
    rows = list(history)
    preds: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    last_x = dict(_as_dict(rows[-1].get("x"))) if rows else {}
    inference_bundle = dict(model_bundle)
    inference_bundle.setdefault("feature_keys", feature_keys)
    for step_idx in range(horizon):
        pred, lower, upper = shared_gbdt_pipeline.predict_rul(inference_bundle, rows)
        preds.append(pred)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        next_x = dict(_as_dict(future_rows[step_idx].get("x"))) if step_idx < len(future_rows) else dict(last_x)
        rows.append({"y": pred, "x": next_x})
    return preds, lower_bounds, upper_bounds


def _build_regime_adaptation_samples(
    *,
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
    records: list[dict[str, object]],
    regime: tuple[float, float, float],
    feature_keys: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _collect_regime_adaptation_rows(
        snapshot=snapshot,
        state_dict=state_dict,
        records=records,
        regime=regime,
        feature_keys=feature_keys,
    )
    if not rows:
        return np.zeros((0, WINDOW, len(feature_keys)), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    xs = [np.asarray(row["window"], dtype=float) for row in rows]
    ys = [float(row["y_true"]) for row in rows]
    base_preds = [float(row["base_pred"]) for row in rows]
    return np.stack(xs).astype(float), np.asarray(ys, dtype=float), np.asarray(base_preds, dtype=float)


def _collect_regime_adaptation_rows(
    *,
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
    records: list[dict[str, object]],
    regime: tuple[float, float, float],
    feature_keys: list[str],
) -> list[dict[str, Any]]:
    xs: list[np.ndarray] = []
    collected: list[dict[str, Any]] = []
    for series_id, rows in _group_records(records).items():
        if len(rows) <= WINDOW:
            continue
        series_rows: list[dict[str, Any]] = []
        for idx in range(WINDOW, len(rows)):
            target = rows[idx]
            if _regime_key(target) != regime:
                continue
            history = rows[idx - WINDOW : idx]
            base_pred = forecast_univariate_torch(
                algo=AFNO_ALGO,
                snapshot=snapshot,
                state_dict=state_dict,
                context_records=history,
                future_feature_rows=[target],
                horizon=1,
                device=None,
            )[0]
            series_rows.append(
                {
                    "series_id": str(series_id),
                    "window": _window_matrix(history, feature_keys=feature_keys),
                    "y_true": _as_float(target.get("y")),
                    "base_pred": float(base_pred),
                    "timestamp": str(target.get("timestamp") or ""),
                }
            )
        regime_count = len(series_rows)
        for regime_pos, row in enumerate(series_rows):
            collected.append(
                {
                    **row,
                    "regime_pos": regime_pos,
                    "regime_count": regime_count,
                    "is_recent": regime_pos >= max(0, regime_count // 2),
                }
            )
    return collected


def _rank_regime_features(x_windows: np.ndarray, y_true: np.ndarray, feature_keys: list[str], *, mode: str) -> list[tuple[float, str]]:
    if x_windows.size == 0 or not feature_keys:
        return []
    if mode == "last":
        feature_view = x_windows[:, -1, :]
    elif mode == "mean":
        feature_view = np.mean(x_windows, axis=1)
    elif mode == "delta":
        feature_view = x_windows[:, -1, :] - x_windows[:, 0, :]
    else:
        feature_view = np.std(x_windows, axis=1)
    scores: list[tuple[float, str]] = []
    for idx, key in enumerate(feature_keys):
        values = feature_view[:, idx]
        if np.allclose(values, values[0]):
            score = 0.0
        else:
            score = float(abs(np.corrcoef(values, y_true)[0, 1])) if y_true.size >= 2 else 0.0
            if not math.isfinite(score):
                score = 0.0
        scores.append((score, key))
    scores.sort(key=lambda item: item[0], reverse=True)
    return scores


def _select_regime_feature_subset(
    x_windows: np.ndarray,
    y_true: np.ndarray,
    feature_keys: list[str],
    *,
    max_features: int = 8,
    mode: str = "last",
) -> list[str]:
    scores = _rank_regime_features(x_windows, y_true, feature_keys, mode=mode)
    selected = [key for _, key in scores[:max_features]]
    for op_key in ["op_setting_1", "op_setting_2", "op_setting_3"]:
        if op_key in feature_keys and op_key not in selected:
            selected.append(op_key)
    return selected[: max_features + 3]


def _summarize_regime_windows(x_windows: np.ndarray, *, feature_keys: list[str], selected_keys: list[str]) -> np.ndarray:
    selected_idx = [feature_keys.index(key) for key in selected_keys if key in feature_keys]
    if not selected_idx:
        return np.zeros((x_windows.shape[0], 0), dtype=float)
    selected = x_windows[:, :, selected_idx]
    last = selected[:, -1, :]
    mean = np.mean(selected, axis=1)
    delta = selected[:, -1, :] - selected[:, 0, :]
    return np.concatenate([last, mean, delta], axis=1)


def _normalize_adapter_design(
    train_design: np.ndarray,
    valid_design: np.ndarray,
    *,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    train_design = _sanitize_adapter_matrix(train_design)
    valid_design = _sanitize_adapter_matrix(valid_design)
    if not standardize or train_design.shape[1] == 0:
        return train_design, valid_design, None, None
    mean = np.mean(train_design, axis=0)
    std = np.std(train_design, axis=0)
    safe_std = np.where(std > 1e-8, std, 1.0)
    train_norm = (train_design - mean) / safe_std
    valid_norm = (valid_design - mean) / safe_std
    return (
        _sanitize_adapter_matrix(train_norm),
        _sanitize_adapter_matrix(valid_norm),
        mean.astype(float),
        safe_std.astype(float),
    )


def _sanitize_adapter_matrix(
    values: np.ndarray | list[float], *, clip: float = _ADAPTER_MATRIX_ABS_CLIP
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array.astype(float)
    cleaned = np.nan_to_num(array, nan=0.0, posinf=float(clip), neginf=-float(clip))
    return np.clip(cleaned, -float(clip), float(clip)).astype(float)


def _safe_adapter_scalar(value: Any, *, default: float = 0.0, clip: float = _ADAPTER_MATRIX_ABS_CLIP) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return float(min(max(parsed, -float(clip)), float(clip)))


def _safe_adapter_linear_response(
    design: np.ndarray,
    coef: np.ndarray,
    *,
    intercept: float,
    clip: float = _ADAPTER_RESPONSE_ABS_CLIP,
) -> np.ndarray:
    design_2d = _sanitize_adapter_matrix(design)
    coef_1d = _sanitize_adapter_matrix(np.asarray(coef, dtype=float).reshape(-1), clip=clip)
    if coef_1d.size != design_2d.shape[1]:
        coef_1d = np.resize(coef_1d, design_2d.shape[1])
    safe_intercept = _safe_adapter_scalar(intercept, default=0.0, clip=clip)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        response = design_2d @ coef_1d + safe_intercept
    return _sanitize_adapter_matrix(response, clip=clip)


def _residual_bounds(
    residuals: np.ndarray,
    *,
    lower_q: float,
    upper_q: float,
    abs_q: float,
    sign_mode: str,
) -> tuple[float, float]:
    if residuals.size == 0:
        return (float("-inf"), float("inf"))
    lower = float(np.quantile(residuals, min(max(lower_q, 0.0), 1.0)))
    upper = float(np.quantile(residuals, min(max(upper_q, 0.0), 1.0)))
    abs_clip = _nearest_rank(np.abs(residuals).tolist(), abs_q)
    if math.isfinite(abs_clip) and abs_clip > 0.0:
        lower = max(lower, -float(abs_clip))
        upper = min(upper, float(abs_clip))
    pos_ratio = float(np.mean(residuals >= 0.0))
    neg_ratio = float(np.mean(residuals <= 0.0))
    if sign_mode == "majority" and pos_ratio >= 0.7:
        lower = max(lower, 0.0)
    elif sign_mode == "majority" and neg_ratio >= 0.7:
        upper = min(upper, 0.0)
    elif sign_mode == "mean":
        mean_residual = float(np.mean(residuals))
        if mean_residual >= 0.0:
            lower = max(lower, 0.0)
        else:
            upper = min(upper, 0.0)
    if lower > upper:
        midpoint = float(np.median(residuals))
        lower = midpoint
        upper = midpoint
    return (lower, upper)


def _predict_adapter_residuals(design: np.ndarray, adapter: dict[str, Any], *, trend_signs: np.ndarray | None = None) -> np.ndarray:
    design_2d = _sanitize_adapter_matrix(np.asarray(design, dtype=float))
    if design_2d.ndim == 1:
        design_2d = design_2d.reshape(1, -1)
    adapter_type = str(adapter.get("type") or "")
    if adapter_type.startswith("residual_trend_two_stage"):
        mag_coef = _sanitize_adapter_matrix(
            np.asarray(adapter.get("mag_coef") or [], dtype=float),
            clip=float(adapter.get("coef_clip") or _ADAPTER_MATRIX_ABS_CLIP),
        )
        if mag_coef.size != design_2d.shape[1]:
            mag_coef = np.resize(mag_coef, design_2d.shape[1])
        if trend_signs is None:
            sign_direction = np.full(
                design_2d.shape[0],
                _safe_adapter_scalar(adapter.get("fallback_sign") or 1.0, default=1.0, clip=1.0),
                dtype=float,
            )
        else:
            sign_direction = _sanitize_adapter_matrix(np.asarray(trend_signs, dtype=float).reshape(-1), clip=1.0)
            if sign_direction.size != design_2d.shape[0]:
                sign_direction = np.resize(sign_direction, design_2d.shape[0])
            sign_direction = np.where(sign_direction >= 0.0, 1.0, -1.0)
        magnitude = np.maximum(
            0.0,
            _safe_adapter_linear_response(
                design_2d,
                mag_coef,
                intercept=float(adapter.get("mag_intercept") or 0.0),
            ),
        )
        magnitude_scale = _safe_adapter_scalar(
            adapter.get("magnitude_scale") or 1.0,
            default=1.0,
            clip=4.0,
        )
        residual = sign_direction * magnitude * magnitude_scale
    else:
        coef = _sanitize_adapter_matrix(np.asarray(adapter.get("coef") or [], dtype=float))
        residual = _safe_adapter_linear_response(
            design_2d,
            coef,
            intercept=float(adapter.get("intercept") or 0.0),
        )
    residual_lower = float(adapter.get("residual_lower") or float("-inf"))
    residual_upper = float(adapter.get("residual_upper") or float("inf"))
    if math.isfinite(residual_lower) or math.isfinite(residual_upper):
        residual = np.clip(residual, residual_lower, residual_upper)
    else:
        residual_clip = float(adapter.get("residual_clip") or 0.0)
        if math.isfinite(residual_clip) and residual_clip > 0.0:
            residual = np.clip(residual, -residual_clip, residual_clip)
    return _sanitize_adapter_matrix(residual, clip=_ADAPTER_RESIDUAL_ABS_CLIP)


def _series_trend_signs_from_rows(
    adaptation_rows: list[dict[str, Any]],
    *,
    scale: float,
    bias: float,
    trend_window: int,
) -> np.ndarray:
    if not adaptation_rows:
        return np.zeros((0,), dtype=float)
    residuals = np.asarray(
        [float(row["y_true"]) - float(scale * float(row["base_pred"]) + bias) for row in adaptation_rows],
        dtype=float,
    )
    fallback_sign = 1.0 if float(np.mean(residuals)) >= 0.0 else -1.0
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(adaptation_rows):
        grouped[str(row.get("series_id") or "")].append(idx)
    signs = np.full((len(adaptation_rows),), fallback_sign, dtype=float)
    for indices in grouped.values():
        for pos, row_idx in enumerate(indices):
            start = max(0, pos - trend_window)
            lookback = indices[start:pos]
            if not lookback:
                continue
            lookback_errors = residuals[lookback]
            lookback_pos = np.asarray(
                [float(adaptation_rows[idx].get("regime_pos") or 0.0) / max(float((adaptation_rows[idx].get("regime_count") or 1) - 1), 1.0) for idx in lookback],
                dtype=float,
            )
            current_pos = float(adaptation_rows[row_idx].get("regime_pos") or 0.0) / max(float((adaptation_rows[row_idx].get("regime_count") or 1) - 1), 1.0)
            mean_error = float(np.mean(lookback_errors))
            if lookback_errors.size >= 2 and not np.allclose(lookback_pos, lookback_pos[0]):
                slope = float(np.polyfit(lookback_pos, lookback_errors, 1)[0])
            else:
                slope = 0.0
            score = mean_error + slope * current_pos
            signs[row_idx] = 1.0 if score >= 0.0 else -1.0
    return signs


def _history_affine_trend_sign(
    *,
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
    history: list[dict[str, object]],
    affine: tuple[float, float],
    trend_window: int,
) -> float:
    if len(history) <= WINDOW:
        return 1.0
    scale, bias = affine
    residuals: list[float] = []
    start_idx = max(WINDOW, len(history) - trend_window)
    for idx in range(start_idx, len(history)):
        target = history[idx]
        prefix = history[idx - WINDOW:idx]
        if len(prefix) < WINDOW:
            continue
        base_pred = float(
            forecast_univariate_torch(
                algo=AFNO_ALGO,
                snapshot=snapshot,
                state_dict=state_dict,
                context_records=prefix,
                future_feature_rows=[target],
                horizon=1,
                device=None,
            )[0]
        )
        residuals.append(float(target.get("y") or 0.0) - float(scale * base_pred + bias))
    if not residuals:
        return 1.0
    residual_arr = np.asarray(residuals, dtype=float)
    rel_pos = np.linspace(0.0, 1.0, num=residual_arr.size, dtype=float)
    mean_error = float(np.mean(residual_arr))
    slope = float(np.polyfit(rel_pos, residual_arr, 1)[0]) if residual_arr.size >= 2 else 0.0
    score = mean_error + slope * 1.0
    return 1.0 if score >= 0.0 else -1.0


def _recent_regime_affine_calibration(adaptation_rows: list[dict[str, Any]], *, scale: float, bias: float) -> dict[str, Any]:
    if not adaptation_rows:
        return {"qhat": 0.0, "count": 0, "tail_fraction": 1.0, "coverage": float("nan"), "rank_quantile": 0.9}
    residuals = [abs(float(row["y_true"]) - float(scale * float(row["base_pred"]) + bias)) for row in adaptation_rows]
    best: dict[str, Any] | None = None
    total = len(residuals)
    for tail_fraction in (0.25, 1.0 / 3.0, 0.4, 0.5, 2.0 / 3.0):
        tail_count = max(24, int(math.ceil(total * tail_fraction)))
        subset = residuals[-tail_count:]
        if len(subset) < 24:
            continue
        subset_arr = np.asarray(subset, dtype=float)
        for rank_quantile in (0.9, 0.92, 0.93, 0.95, 0.975):
            qhat = _nearest_rank(subset, rank_quantile)
            coverage = float(np.mean(subset_arr <= qhat)) if subset else float("nan")
            candidate = {
                "qhat": float(qhat),
                "count": len(subset),
                "tail_fraction": float(tail_fraction),
                "coverage": coverage,
                "rank_quantile": float(rank_quantile),
            }
            if best is None:
                best = candidate
                continue
            candidate_ok = coverage >= 0.93
            best_ok = float(best.get("coverage") or 0.0) >= 0.93
            if candidate_ok and not best_ok:
                best = candidate
            elif candidate_ok == best_ok and float(candidate["qhat"]) < float(best["qhat"]):
                best = candidate
            elif not candidate_ok and not best_ok and float(candidate["coverage"]) > float(best.get("coverage") or float("-inf")):
                best = candidate
    return best or {"qhat": _nearest_rank(residuals, 0.95), "count": total, "tail_fraction": 1.0, "coverage": float(np.mean(np.asarray(residuals, dtype=float) <= _nearest_rank(residuals, 0.95))), "rank_quantile": 0.95}


def _backtest_outputs(
    predictor: Any,
    *,
    records: list[dict[str, object]],
    horizon: int,
    folds: int,
    feature_keys: list[str],
    kind: str,
    snapshot: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    algo: str | None = None,
    interval_qhat: float | None = None,
) -> dict[str, list[float]]:
    by_series = _group_records(records)
    y_true: list[float] = []
    y_pred: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    condition_keys: list[str] = []
    tail_pos: list[float] = []
    for rows in by_series.values():
        n = len(rows)
        if n < WINDOW + horizon + 1:
            continue
        for fold_idx in range(folds):
            end = n - fold_idx * horizon
            start = end - horizon
            train_end = start
            if train_end <= WINDOW or start < 0 or end > n:
                break
            history = rows[:train_end]
            future = rows[start:end]
            if len(future) != horizon:
                break
            if kind == "gbdt":
                preds, lowers, uppers = _predict_gbdt_horizon(predictor, history=history, future_rows=future, feature_keys=feature_keys, horizon=horizon)
                lower_bounds.extend(lowers)
                upper_bounds.extend(uppers)
            else:
                preds = forecast_univariate_torch(
                    algo=str(algo or AFNO_ALGO),
                    snapshot=snapshot or {},
                    state_dict=state_dict or {},
                    context_records=history[-WINDOW:],
                    future_feature_rows=future,
                    horizon=horizon,
                    device=None,
                )
                if interval_qhat is not None and math.isfinite(float(interval_qhat)):
                    width = float(max(interval_qhat, 0.0))
                    lower_bounds.extend([float(value) - width for value in preds])
                    upper_bounds.extend([float(value) + width for value in preds])
            y_true.extend([_as_float(row.get("y")) for row in future])
            y_pred.extend([float(value) for value in preds])
            condition_keys.extend([_condition_cluster_key(row) for row in future])
            tail_pos.extend([float((start + offset) / max(n - 1, 1)) for offset in range(len(future))])
    return {"y_true": y_true, "y_pred": y_pred, "lower": lower_bounds, "upper": upper_bounds, "condition_key": condition_keys, "tail_pos": tail_pos}


def _standard_eval_outputs(
    predictor: Any,
    *,
    records: list[dict[str, object]],
    feature_keys: list[str],
    kind: str,
    snapshot: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    algo: str | None = None,
    interval_qhat: float | None = None,
) -> dict[str, list[float]]:
    """Standard CMAPSS evaluation: predict RUL at the final cycle of each test engine.

    For each engine, the last WINDOW cycles are used as context, and a single
    1-step-ahead prediction is made. The true RUL comes from the y value at the
    final record (which encodes the RUL_FD004.txt terminal RUL).
    Only engines with at least WINDOW records are eligible.
    """
    by_series = _group_records(records)
    y_true: list[float] = []
    y_pred: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    condition_keys: list[str] = []
    tail_pos: list[float] = []
    for rows in by_series.values():
        n = len(rows)
        if n < WINDOW:
            continue
        history = rows[-WINDOW:]
        last_row = rows[-1]
        true_rul = _as_float(last_row.get("y"))
        if not math.isfinite(true_rul):
            continue
        if kind == "gbdt":
            inference_bundle = dict(predictor)
            inference_bundle.setdefault("feature_keys", feature_keys)
            pred, lower, upper = shared_gbdt_pipeline.predict_rul(inference_bundle, history)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        else:
            future_rows = [last_row]
            preds_seq = forecast_univariate_torch(
                algo=str(algo or AFNO_ALGO),
                snapshot=snapshot or {},
                state_dict=state_dict or {},
                context_records=history,
                future_feature_rows=future_rows,
                horizon=1,
                device=None,
            )
            pred = float(preds_seq[0])
            if interval_qhat is not None and math.isfinite(float(interval_qhat)):
                width = float(max(interval_qhat, 0.0))
                lower_bounds.append(pred - width)
                upper_bounds.append(pred + width)
        y_true.append(true_rul)
        y_pred.append(pred)
        condition_keys.append(_condition_cluster_key(last_row))
        tail_pos.append(1.0)
    return {"y_true": y_true, "y_pred": y_pred, "lower": lower_bounds, "upper": upper_bounds, "condition_key": condition_keys, "tail_pos": tail_pos}


def _metrics_from_outputs(outputs: dict[str, list[float]]) -> dict[str, float]:
    interval_stats = _interval_metrics(outputs["y_true"], outputs["lower"], outputs["upper"])
    return {
        "rmse": _rmse(outputs["y_true"], outputs["y_pred"]),
        "mae": _mae(outputs["y_true"], outputs["y_pred"]),
        "nasa_score": _nasa_score(outputs["y_true"], outputs["y_pred"]),
        "count": float(len(outputs["y_true"])),
        "cov90": float(interval_stats["cov90"]) if interval_stats["cov90"] is not None else float("nan"),
        "width90": float(interval_stats["width90"]) if interval_stats["width90"] is not None else float("nan"),
        **_point_diagnostic_metrics(outputs["y_true"], outputs["y_pred"]),
    }


def _interval_overlap_ratio(g_lower: float, g_upper: float, a_lower: float, a_upper: float) -> float:
    return _shared_interval_overlap_ratio(g_lower, g_upper, a_lower, a_upper)


def _candidate_tau_grid(gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]) -> list[float]:
    diffs = sorted(abs(float(a) - float(g)) for g, a in zip(gbdt_outputs["y_pred"], afno_outputs["y_pred"]))
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    values = {0.0, 1.0, 2.0, 5.0, 10.0}
    for q in quantiles:
        values.add(round(_nearest_rank(diffs, q), 6))
    return sorted(value for value in values if value >= 0.0)


def _candidate_disagreement_grid(gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]) -> list[float]:
    diffs = [abs(float(a) - float(g)) for g, a in zip(gbdt_outputs["y_pred"], afno_outputs["y_pred"]) if math.isfinite(float(a)) and math.isfinite(float(g))]
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9]
    values = {0.0}
    for q in quantiles:
        values.add(round(_nearest_rank(diffs, q), 6))
    return sorted(value for value in values if value >= 0.0)


def _dense_threshold_grid(values: list[float], *, points: int = 25) -> list[float]:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)) and float(v) >= 0.0)
    if not finite:
        return [0.0]
    quantiles = np.linspace(0.0, 1.0, num=max(5, points), dtype=float)
    grid = {0.0}
    for q in quantiles:
        grid.add(round(_nearest_rank(finite, float(q)), 6))
    lo = float(finite[0])
    hi = float(finite[-1])
    if hi > lo:
        for value in np.linspace(lo, hi, num=max(7, points), dtype=float):
            grid.add(round(float(value), 6))
    return sorted(grid)


def _compact_numeric_grid(values: list[float], *, max_points: int) -> list[float]:
    ordered = sorted({round(float(value), 6) for value in values if math.isfinite(float(value))})
    if len(ordered) <= max_points:
        return ordered
    indices = sorted({int(round(value)) for value in np.linspace(0, len(ordered) - 1, num=max_points, dtype=float)})
    return [ordered[idx] for idx in indices]


def _refine_numeric_grid(base_grid: list[float], center: float, *, steps: int = 9, width_ratio: float = 0.2) -> list[float]:
    if not base_grid:
        return [max(0.0, center)]
    lo = float(min(base_grid))
    hi = float(max(base_grid))
    span = max(hi - lo, max(abs(center), 1.0) * width_ratio)
    left = max(0.0, center - span)
    right = max(left, center + span)
    refined = set(base_grid)
    for value in np.linspace(left, right, num=max(5, steps), dtype=float):
        refined.add(round(float(value), 6))
    return sorted(refined)


def _apply_asymmetric_interval(outputs: dict[str, list[float]], *, lower_offset: float, upper_offset: float) -> dict[str, list[float]]:
    return {
        **outputs,
        "lower": [float(pred) + float(lower_offset) for pred in outputs["y_pred"]],
        "upper": [float(pred) + float(upper_offset) for pred in outputs["y_pred"]],
    }


def _safe_metric_value(metrics: dict[str, float], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float("inf")


def _subset_outputs(outputs: dict[str, list[float]], indices: list[int]) -> dict[str, list[float]]:
    condition_keys = [str(key) for key in outputs.get("condition_key") or []]
    tail_pos = [float(value) for value in outputs.get("tail_pos") or []]
    return {
        "y_true": [outputs["y_true"][idx] for idx in indices],
        "y_pred": [outputs["y_pred"][idx] for idx in indices],
        "lower": [outputs["lower"][idx] for idx in indices],
        "upper": [outputs["upper"][idx] for idx in indices],
        "condition_key": [condition_keys[idx] for idx in indices] if condition_keys else [],
        "tail_pos": [tail_pos[idx] for idx in indices] if tail_pos else [],
    }


def _coverage_shortfall(coverage: float, target_cov: float) -> float:
    if not math.isfinite(coverage):
        return float("inf")
    return max(0.0, float(target_cov) - float(coverage))


def _gate_candidate_rank(
    metrics: dict[str, float],
    *,
    target_cov: float,
    primary_metric: str,
    secondary_metrics: tuple[str, ...],
) -> tuple[float, ...]:
    coverage = _safe_metric_value(metrics, "cov90")
    shortfall = _coverage_shortfall(coverage, target_cov)
    feasible = 0.0 if shortfall <= 0.0 else 1.0
    ranked_metrics = [_safe_metric_value(metrics, primary_metric)]
    ranked_metrics.extend(_safe_metric_value(metrics, name) for name in secondary_metrics)
    ranked_metrics.append(_safe_metric_value(metrics, "width90"))
    return (feasible, shortfall, *ranked_metrics)


def _soft_gate_weight_entropy(weights: list[float]) -> float:
    return _shared_soft_gate_weight_entropy(weights)


def _soft_gate_objective_config(scope_name: str, *, mode: str) -> dict[str, float | str]:
    if mode == "risk_aware":
        if scope_name == "unit_holdout":
            return {
                "primary_metric": "rmse",
                "secondary_metric": "nasa_score",
                "rmse_weight": 0.8,
                "nasa_weight": 0.35,
                "entropy_weight": 0.0,
            }
        return {
            "primary_metric": "nasa_score",
            "secondary_metric": "rmse",
            "rmse_weight": 0.25,
            "nasa_weight": 1.0,
            "entropy_weight": 0.0,
        }
    if scope_name == "main_rolling_backtest":
        return {
            "primary_metric": "rmse",
            "secondary_metric": "nasa_score",
            "rmse_weight": 1.0,
            "nasa_weight": 0.2,
            "entropy_weight": 0.35,
        }
    if scope_name == "unit_holdout":
        return {
            "primary_metric": "rmse",
            "secondary_metric": "nasa_score",
            "rmse_weight": 1.0,
            "nasa_weight": 0.15,
            "entropy_weight": 0.25,
        }
    return {
        "primary_metric": "nasa_score",
        "secondary_metric": "rmse",
        "rmse_weight": 0.35,
        "nasa_weight": 1.0,
        "entropy_weight": 0.1,
    }


def _soft_gate_candidate_rank(
    *,
    point_metrics: dict[str, float],
    interval_coverage: float,
    interval_width: float,
    avg_entropy: float,
    target_cov: float,
    scope_name: str,
    mode: str,
) -> tuple[float, ...]:
    config = _soft_gate_objective_config(scope_name, mode=mode)
    primary_metric = str(config["primary_metric"])
    secondary_metric = str(config["secondary_metric"])
    rmse = _safe_metric_value(point_metrics, "rmse")
    nasa_score = _safe_metric_value(point_metrics, "nasa_score")
    primary_value = _safe_metric_value(point_metrics, primary_metric)
    secondary_value = _safe_metric_value(point_metrics, secondary_metric)
    coverage_shortfall = _coverage_shortfall(interval_coverage, target_cov)
    feasible = 0.0 if coverage_shortfall <= 0.0 else 1.0
    point_objective = float(config["rmse_weight"]) * rmse
    point_objective += float(config["nasa_weight"]) * math.log1p(max(0.0, nasa_score))
    point_objective += float(config["entropy_weight"]) * max(0.0, float(avg_entropy))
    return (
        feasible,
        coverage_shortfall,
        point_objective,
        primary_value,
        secondary_value,
        max(0.0, float(avg_entropy)),
        _safe_metric_value({"width90": interval_width}, "width90"),
    )


def _interval_calibration_groups(outputs: dict[str, list[float]]) -> list[list[int]]:
    n = len(outputs.get("y_true") or [])
    if n <= 0:
        return []
    condition_keys = [str(key) for key in outputs.get("condition_key") or []]
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(condition_keys):
        grouped[key].append(idx)
    cluster_groups = [indices for indices in grouped.values() if len(indices) >= 4]
    if len(cluster_groups) >= 2:
        return cluster_groups
    fold_count = max(2, min(4, n // 10))
    contiguous = [list(chunk.astype(int)) for chunk in np.array_split(np.arange(n, dtype=int), fold_count) if len(chunk) > 0]
    return contiguous


def _fit_asymmetric_interval(
    outputs: dict[str, list[float]],
    *,
    target_cov: float,
) -> dict[str, float]:
    residuals = [float(y) - float(pred) for y, pred in zip(outputs["y_true"], outputs["y_pred"])]
    if not residuals:
        return {
            "lower_offset": 0.0,
            "upper_offset": 0.0,
            "coverage": float("nan"),
            "lower_q": 0.05,
            "upper_q": 0.95,
            "padding": 0.0,
            "min_group_coverage": float("nan"),
            "meets_constraint": 0.0,
        }
    lower_grid = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    upper_grid = [0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.975, 0.99, 0.995]
    padding_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
    groups = _interval_calibration_groups(outputs)
    best: dict[str, float] | None = None
    def evaluate(lower_q: float, upper_q: float, padding: float) -> dict[str, float]:
        lower_offset = _nearest_rank(residuals, lower_q) - float(padding)
        upper_offset = _nearest_rank(residuals, upper_q) + float(padding)
        calibrated = _apply_asymmetric_interval(outputs, lower_offset=lower_offset, upper_offset=upper_offset)
        metrics = _metrics_from_outputs(calibrated)
        coverage = float(metrics["cov90"])
        width = float(metrics["width90"])
        lower_width = abs(float(lower_offset))
        upper_width = abs(float(upper_offset))
        min_group_cov = coverage
        for indices in groups:
            group_metrics = _metrics_from_outputs(_subset_outputs(calibrated, indices))
            min_group_cov = min(min_group_cov, _safe_metric_value(group_metrics, "cov90"))
        fold_target = max(0.0, float(target_cov) - 0.02)
        overall_shortfall = _coverage_shortfall(coverage, target_cov)
        group_shortfall = _coverage_shortfall(min_group_cov, fold_target)
        feasible = 1.0 if overall_shortfall <= 0.0 and group_shortfall <= 0.0 else 0.0
        rank = (
            0.0 if feasible else 1.0,
            overall_shortfall,
            group_shortfall,
            width,
            upper_width,
            lower_width,
        )
        return {
            "lower_offset": float(lower_offset),
            "upper_offset": float(upper_offset),
            "coverage": coverage,
            "lower_q": float(lower_q),
            "upper_q": float(upper_q),
            "padding": float(padding),
            "min_group_coverage": float(min_group_cov),
            "meets_constraint": float(feasible),
            "lower_width": float(lower_width),
            "upper_width": float(upper_width),
            "rank": rank,
        }

    for lower_q in lower_grid:
        for upper_q in upper_grid:
            if upper_q <= lower_q:
                continue
            for padding in padding_grid:
                candidate = evaluate(lower_q, upper_q, padding)
                if best is None or tuple(candidate["rank"]) < tuple(best["rank"]):
                    best = candidate

    if best is not None:
        refined_lower = sorted({round(float(v), 6) for v in list(np.linspace(max(0.0005, float(best["lower_q"]) / 2.0), min(0.2, float(best["lower_q"]) * 2.0 + 0.01), num=11, dtype=float))})
        refined_upper = sorted({round(float(v), 6) for v in list(np.linspace(max(float(best["lower_q"]) + 0.02, float(best["upper_q"]) - 0.08), min(0.999, float(best["upper_q"]) + 0.08), num=13, dtype=float))})
        refined_padding = sorted({round(float(v), 6) for v in list(np.linspace(max(0.0, float(best["padding"]) - 4.0), float(best["padding"]) + 6.0, num=9, dtype=float))})
        for lower_q in refined_lower:
            for upper_q in refined_upper:
                if upper_q <= lower_q:
                    continue
                for padding in refined_padding:
                    candidate = evaluate(lower_q, upper_q, padding)
                    if tuple(candidate["rank"]) < tuple(best["rank"]):
                        best = candidate
    return best or {
        "lower_offset": 0.0,
        "upper_offset": 0.0,
        "coverage": float("nan"),
        "lower_q": 0.05,
        "upper_q": 0.95,
        "padding": 0.0,
        "min_group_coverage": float("nan"),
        "meets_constraint": 0.0,
        "lower_width": 0.0,
        "upper_width": 0.0,
        "rank": (1.0, float("inf"), float("inf"), float("inf"), float("inf"), float("inf")),
    }


def _hybrid_weight_from_validation(gbdt_meta: dict[str, Any], afno_meta: dict[str, Any]) -> float:
    gbdt_rmse = max(_as_float(gbdt_meta.get("validation_rmse"), 1.0), 1e-6)
    afno_rmse = max(_as_float(afno_meta.get("validation_rmse"), 1.0), 1e-6)
    gbdt_inv = 1.0 / (gbdt_rmse ** 2)
    afno_inv = 1.0 / (afno_rmse ** 2)
    total = gbdt_inv + afno_inv
    return float(afno_inv / total) if total > 0.0 else 0.5


def _blend_backtest_outputs(gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]], *, afno_weight: float) -> dict[str, list[float]]:
    weight = min(max(float(afno_weight), 0.0), 1.0)
    one_minus = 1.0 - weight
    y_true = gbdt_outputs["y_true"]
    if len(y_true) != len(afno_outputs["y_true"]):
        raise ValueError("Backtest outputs must have aligned lengths for blending")
    return {
        "y_true": y_true,
        "y_pred": [one_minus * g + weight * a for g, a in zip(gbdt_outputs["y_pred"], afno_outputs["y_pred"])],
        "lower": [one_minus * g + weight * a for g, a in zip(gbdt_outputs["lower"], afno_outputs["lower"])],
        "upper": [one_minus * g + weight * a for g, a in zip(gbdt_outputs["upper"], afno_outputs["upper"])],
        "condition_key": [str(key) for key in gbdt_outputs.get("condition_key") or []],
        "tail_pos": [float(value) for value in gbdt_outputs.get("tail_pos") or []],
    }


def _sigmoid(value: float) -> float:
    clipped = max(-60.0, min(60.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-clipped)))


def _soft_gate_variant_specs() -> list[dict[str, str]]:
    return [
        {"mode": "risk_aware", "label": "GBDT_AFNO_gated_riskaware_w30_f24", "display": "Risk-aware soft gate"},
        {"mode": "point_first", "label": "GBDT_AFNO_gated_pointfirst_w30_f24", "display": "Point-first soft gate"},
        {"mode": "correctness_meta", "label": "GBDT_AFNO_gated_w30_f24", "display": "Correctness meta-gate"},
    ]


def _soft_gate_bucket(value: float, edges: list[float]) -> str:
    if not edges:
        return "0"
    for idx, edge in enumerate(edges):
        if float(value) <= float(edge):
            return str(idx)
    return str(len(edges))


def _normalize_advantage_lookup(grouped: dict[str, list[float]]) -> dict[str, float]:
    return _shared_normalize_advantage_lookup(grouped)


def _condition_advantage_map(gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]) -> dict[str, float]:
    return _shared_condition_advantage_map(gbdt_outputs, afno_outputs)


def _soft_gate_correctness_priors(gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    condition_grouped: dict[str, list[float]] = defaultdict(list)
    tail_grouped: dict[str, list[float]] = defaultdict(list)
    delta_grouped: dict[str, list[float]] = defaultdict(list)
    overlap_grouped: dict[str, list[float]] = defaultdict(list)
    width_grouped: dict[str, list[float]] = defaultdict(list)

    diffs = [abs(float(a) - float(g)) for g, a in zip(gbdt_outputs["y_pred"], afno_outputs["y_pred"])]
    width_diffs = [
        float(g_upper - g_lower) - float(a_upper - a_lower)
        for g_lower, g_upper, a_lower, a_upper in zip(
            gbdt_outputs["lower"],
            gbdt_outputs["upper"],
            afno_outputs["lower"],
            afno_outputs["upper"],
        )
    ]
    finite_width = sorted(abs(float(value)) for value in width_diffs if math.isfinite(float(value)))
    delta_edges = sorted({round(_nearest_rank(diffs, q), 6) for q in (0.2, 0.4, 0.6, 0.8)}) if diffs else []
    overlap_edges = [0.2, 0.4, 0.6, 0.8]
    tail_edges = [0.2, 0.4, 0.6, 0.8]
    width_edges = sorted({round(_nearest_rank(finite_width, q), 6) for q in (0.3, 0.6, 0.85)}) if finite_width else []

    condition_keys = [str(key) for key in gbdt_outputs.get("condition_key") or []]
    tail_pos = [float(value) for value in gbdt_outputs.get("tail_pos") or []]
    for idx, truth in enumerate(gbdt_outputs["y_true"]):
        if idx >= len(afno_outputs["y_pred"]) or idx >= len(gbdt_outputs["y_pred"]):
            continue
        g_pred = float(gbdt_outputs["y_pred"][idx])
        a_pred = float(afno_outputs["y_pred"][idx])
        signed_advantage = abs(float(truth) - g_pred) - abs(float(truth) - a_pred)
        condition_key = condition_keys[idx] if idx < len(condition_keys) else "global"
        tail_value = tail_pos[idx] if idx < len(tail_pos) else 0.5
        delta_key = _soft_gate_bucket(abs(a_pred - g_pred), delta_edges)
        overlap_key = _soft_gate_bucket(
            _interval_overlap_ratio(
                float(gbdt_outputs["lower"][idx]),
                float(gbdt_outputs["upper"][idx]),
                float(afno_outputs["lower"][idx]),
                float(afno_outputs["upper"][idx]),
            ),
            overlap_edges,
        )
        width_key = _soft_gate_bucket(abs(width_diffs[idx]) if idx < len(width_diffs) else 0.0, width_edges)
        tail_key = _soft_gate_bucket(tail_value, tail_edges)
        condition_grouped[condition_key].append(signed_advantage)
        tail_grouped[tail_key].append(signed_advantage)
        delta_grouped[delta_key].append(signed_advantage)
        overlap_grouped[overlap_key].append(signed_advantage)
        width_grouped[width_key].append(signed_advantage)

    return {
        "condition": _normalize_advantage_lookup(condition_grouped),
        "tail": _normalize_advantage_lookup(tail_grouped),
        "delta": _normalize_advantage_lookup(delta_grouped),
        "overlap": _normalize_advantage_lookup(overlap_grouped),
        "width": _normalize_advantage_lookup(width_grouped),
        "delta_edges": {str(idx): float(value) for idx, value in enumerate(delta_edges)},
        "width_edges": {str(idx): float(value) for idx, value in enumerate(width_edges)},
    }


def _soft_gate_feature_scales(gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]) -> dict[str, float]:
    return _shared_soft_gate_feature_scales(gbdt_outputs, afno_outputs)


def _soft_gate_outputs(
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
    coef_correctness: float,
    condition_advantage: dict[str, float] | None = None,
    correctness_priors: dict[str, dict[str, float]] | None = None,
) -> dict[str, list[float]]:
    return _shared_soft_gate_outputs(
        gbdt_outputs,
        afno_outputs,
        temperature=temperature,
        tau=tau,
        coef_delta=coef_delta,
        coef_overlap=coef_overlap,
        coef_width=coef_width,
        coef_tail=coef_tail,
        coef_condition=coef_condition,
        coef_correctness=coef_correctness,
        condition_advantage=condition_advantage,
        correctness_priors=correctness_priors,
    )


def _distribution_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if finite.size == 0:
        return {"mean": None, "std": None, "p50": None, "p90": None, "mean_abs": None, "max_abs": None}
    abs_values = np.abs(finite)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p50": float(np.quantile(finite, 0.5)),
        "p90": float(np.quantile(finite, 0.9)),
        "mean_abs": float(np.mean(abs_values)),
        "max_abs": float(np.max(abs_values)),
    }


def _score_rank_consistency(score_a: dict[str, float], score_b: dict[str, float]) -> float | None:
    keys = sorted(set(score_a) | set(score_b))
    if len(keys) < 2:
        return None
    ranked_a = {key: idx + 1 for idx, (key, _) in enumerate(sorted(score_a.items(), key=lambda item: item[1], reverse=True))}
    ranked_b = {key: idx + 1 for idx, (key, _) in enumerate(sorted(score_b.items(), key=lambda item: item[1], reverse=True))}
    fallback_rank = len(keys) + 1
    arr_a = np.asarray([float(ranked_a.get(key, fallback_rank)) for key in keys], dtype=float)
    arr_b = np.asarray([float(ranked_b.get(key, fallback_rank)) for key in keys], dtype=float)
    if arr_a.size < 2 or float(np.std(arr_a)) <= 1e-8 or float(np.std(arr_b)) <= 1e-8:
        return None
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def _topk_jaccard(score_a: dict[str, float], score_b: dict[str, float], *, k: int) -> float | None:
    top_a = {key for key, _ in sorted(score_a.items(), key=lambda item: item[1], reverse=True)[: max(1, k)]}
    top_b = {key for key, _ in sorted(score_b.items(), key=lambda item: item[1], reverse=True)[: max(1, k)]}
    union = top_a | top_b
    if not union:
        return None
    return float(len(top_a & top_b) / len(union))


def _gate_term_rank_consistency(soft_outputs: dict[str, list[float]]) -> float | None:
    term_keys = {
        "delta": "gate_term_delta",
        "overlap": "gate_term_overlap",
        "width": "gate_term_width",
        "tail": "gate_term_tail",
        "condition": "gate_term_condition",
        "correctness": "gate_term_correctness",
    }
    n = len(soft_outputs.get("y_pred") or [])
    if n < 4:
        return None
    split = max(2, n // 2)
    score_a = {
        name: float(np.mean(np.abs(np.asarray((soft_outputs.get(key) or [])[:split], dtype=float))))
        for name, key in term_keys.items()
        if (soft_outputs.get(key) or [])[:split]
    }
    score_b = {
        name: float(np.mean(np.abs(np.asarray((soft_outputs.get(key) or [])[split:], dtype=float))))
        for name, key in term_keys.items()
        if (soft_outputs.get(key) or [])[split:]
    }
    return _score_rank_consistency(score_a, score_b)


def _soft_gate_uncertainty_quality(
    soft_outputs: dict[str, list[float]],
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    coverage_guard_triggered: bool,
) -> dict[str, float | None]:
    abs_errors: list[float] = []
    total_uncertainty: list[float] = []
    disagreement_values: list[float] = []
    for truth, pred, alpha, g_pred, a_pred, g_lower, g_upper, a_lower, a_upper in zip(
        soft_outputs.get("y_true") or [],
        soft_outputs.get("y_pred") or [],
        soft_outputs.get("afno_weight") or [],
        gbdt_outputs.get("y_pred") or [],
        afno_outputs.get("y_pred") or [],
        gbdt_outputs.get("lower") or [],
        gbdt_outputs.get("upper") or [],
        afno_outputs.get("lower") or [],
        afno_outputs.get("upper") or [],
    ):
        weight = min(max(float(alpha), 0.0), 1.0)
        g_half = max(0.0, (float(g_upper) - float(g_lower)) / 2.0)
        a_half = max(0.0, (float(a_upper) - float(a_lower)) / 2.0)
        disagreement = weight * (1.0 - weight) * ((float(a_pred) - float(g_pred)) ** 2)
        total_proxy = (1.0 - weight) * (g_half**2) + weight * (a_half**2) + disagreement
        abs_errors.append(abs(float(truth) - float(pred)))
        total_uncertainty.append(float(math.sqrt(max(total_proxy, 0.0))))
        disagreement_values.append(float(disagreement))

    def _corr(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 2 or len(xs) != len(ys):
            return None
        arr_x = np.asarray(xs, dtype=float)
        arr_y = np.asarray(ys, dtype=float)
        if arr_x.size < 2 or float(np.std(arr_x)) <= 1e-8 or float(np.std(arr_y)) <= 1e-8:
            return None
        return float(np.corrcoef(arr_x, arr_y)[0, 1])

    return {
        "corr_abs_error_vs_total_uncertainty": _corr(abs_errors, total_uncertainty),
        "corr_abs_error_vs_expert_disagreement": _corr(abs_errors, disagreement_values),
        "coverage_guard_trigger_rate": 1.0 if coverage_guard_triggered else 0.0,
    }


def _rank_global_feature_scores(
    feature_scores: dict[str, float],
    *,
    method: str,
    sample_count: int,
    top_k: int = 8,
) -> dict[str, Any]:
    ranked = sorted(
        ((str(key), float(value)) for key, value in feature_scores.items() if math.isfinite(float(value))),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "method": method,
        "sample_count": int(sample_count),
        "top_features_global": [
            {"feature": key, "importance_mean_abs": float(score), "rank": idx + 1}
            for idx, (key, score) in enumerate(ranked[: max(1, top_k)])
        ],
        "feature_importance_mean_abs": {key: float(score) for key, score in ranked},
    }


def _aggregate_flattened_feature_scores(scores: np.ndarray, *, feature_keys: list[str]) -> dict[str, float]:
    if scores.size == 0 or not feature_keys:
        return {}
    feature_count = len(feature_keys)
    grouped: dict[str, list[float]] = defaultdict(list)
    flat = np.asarray(scores, dtype=float).reshape(-1)
    for idx, score in enumerate(flat.tolist()):
        grouped[str(feature_keys[idx % feature_count])].append(abs(float(score)))
    return {key: float(np.mean(values)) for key, values in grouped.items() if values}


def _gbdt_global_importance_summary(
    model_bundle: dict[str, HistGradientBoostingRegressor],
    *,
    records: list[dict[str, object]],
    feature_keys: list[str],
) -> dict[str, Any]:
    _ns = model_bundle.get("norm_stats")
    _fk = model_bundle.get("feature_keys") or feature_keys
    x_eval, y_eval = _build_gbdt_dataset(records, feature_keys=_fk, norm_stats=_ns)
    if x_eval.size == 0 or y_eval.size == 0 or not feature_keys:
        return _rank_global_feature_scores({}, method="permutation_importance_rmse_v1", sample_count=0)
    if x_eval.shape[0] > 256:
        sample_idx = np.linspace(0, x_eval.shape[0] - 1, num=256, dtype=int)
        x_eval = x_eval[sample_idx]
        y_eval = y_eval[sample_idx]
    result = permutation_importance(
        model_bundle["point"],
        x_eval,
        y_eval,
        n_repeats=5,
        random_state=0,
        scoring="neg_root_mean_squared_error",
    )
    scores = _aggregate_flattened_feature_scores(np.asarray(result.importances_mean, dtype=float), feature_keys=_fk)
    return _rank_global_feature_scores(scores, method="permutation_importance_rmse_v1", sample_count=int(x_eval.shape[0]))


def _collect_afno_explanation_windows(
    *,
    records: list[dict[str, object]],
    max_windows: int,
    regime: tuple[float, float, float] | None = None,
) -> list[tuple[list[dict[str, object]], dict[str, object]]]:
    windows: list[tuple[list[dict[str, object]], dict[str, object]]] = []
    for rows in _group_records(records).values():
        if len(rows) <= WINDOW:
            continue
        for idx in range(WINDOW, len(rows)):
            target = rows[idx]
            if regime is not None and _regime_key(target) != regime:
                continue
            windows.append((rows[idx - WINDOW : idx], target))
    if len(windows) <= max_windows:
        return windows
    sample_idx = sorted({int(value) for value in np.linspace(0, len(windows) - 1, num=max_windows, dtype=float)})
    return [windows[idx] for idx in sample_idx]


def _feature_baseline_map(records: list[dict[str, object]], *, feature_keys: list[str]) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for key in feature_keys:
        values = [
            _as_float(_as_dict(row.get("x")).get(key))
            for row in records
            if math.isfinite(_as_float(_as_dict(row.get("x")).get(key)))
        ]
        baselines[key] = float(np.median(np.asarray(values, dtype=float))) if values else 0.0
    return baselines


def _afno_occlusion_summary(
    *,
    records: list[dict[str, object]],
    feature_keys: list[str],
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
    algo: str,
    point_scale: float = 1.0,
    point_bias: float = 0.0,
    point_adapter: dict[str, Any] | None = None,
    regime: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    if not feature_keys:
        return _rank_global_feature_scores({}, method="occlusion_delta_v1", sample_count=0)
    windows = _collect_afno_explanation_windows(records=records, max_windows=24, regime=regime)
    if not windows:
        return _rank_global_feature_scores({}, method="occlusion_delta_v1", sample_count=0)
    baselines = _feature_baseline_map(records, feature_keys=feature_keys)
    deltas: dict[str, list[float]] = defaultdict(list)
    per_window_scores: list[dict[str, float]] = []
    for history, target in windows:
        base_pred = float(
            forecast_univariate_torch(
                algo=algo,
                snapshot=snapshot,
                state_dict=state_dict,
                context_records=history,
                future_feature_rows=[target],
                horizon=1,
                device=None,
            )[0]
        )
        base_pred = _apply_point_adapter(
            base_pred,
            history=history,
            feature_keys=feature_keys,
            adapter=point_adapter,
            affine=(point_scale, point_bias),
            snapshot=snapshot,
            state_dict=state_dict,
        )
        local_scores: dict[str, float] = {}
        for key in feature_keys:
            occluded_history: list[dict[str, object]] = []
            for row in history:
                new_x = dict(_as_dict(row.get("x")))
                new_x[key] = float(baselines.get(key, 0.0))
                occluded_history.append({**row, "x": new_x})
            occluded_pred = float(
                forecast_univariate_torch(
                    algo=algo,
                    snapshot=snapshot,
                    state_dict=state_dict,
                    context_records=occluded_history,
                    future_feature_rows=[target],
                    horizon=1,
                    device=None,
                )[0]
            )
            occluded_pred = _apply_point_adapter(
                occluded_pred,
                history=occluded_history,
                feature_keys=feature_keys,
                adapter=point_adapter,
                affine=(point_scale, point_bias),
                snapshot=snapshot,
                state_dict=state_dict,
            )
            score = abs(float(base_pred) - float(occluded_pred))
            deltas[key].append(score)
            local_scores[str(key)] = float(score)
        per_window_scores.append(local_scores)
    scores = {key: float(np.mean(values)) for key, values in deltas.items() if values}
    summary = _rank_global_feature_scores(scores, method="occlusion_delta_v1", sample_count=len(windows))
    split = max(1, len(per_window_scores) // 2)
    score_a: dict[str, float] = defaultdict(float)
    score_b: dict[str, float] = defaultdict(float)
    count_a = max(1, len(per_window_scores[:split]))
    count_b = max(1, len(per_window_scores[split:]))
    for payload in per_window_scores[:split]:
        for key, value in payload.items():
            score_a[str(key)] += float(value)
    for payload in per_window_scores[split:]:
        for key, value in payload.items():
            score_b[str(key)] += float(value)
    score_a = {key: float(value / count_a) for key, value in score_a.items()}
    score_b = {key: float(value / count_b) for key, value in score_b.items()}
    summary["stability"] = {
        "topk_jaccard": _topk_jaccard(score_a, score_b, k=min(5, len(feature_keys))),
        "rank_consistency": _score_rank_consistency(score_a, score_b),
    }
    return summary


def _soft_gate_explanation_summary(soft_outputs: dict[str, list[float]]) -> dict[str, Any]:
    term_keys = {
        "delta": "gate_term_delta",
        "overlap": "gate_term_overlap",
        "width": "gate_term_width",
        "tail": "gate_term_tail",
        "condition": "gate_term_condition",
        "correctness": "gate_term_correctness",
    }
    loo_keys = {
        "delta": "gate_loo_delta",
        "overlap": "gate_loo_overlap",
        "width": "gate_loo_width",
        "tail": "gate_loo_tail",
        "condition": "gate_loo_condition",
        "correctness": "gate_loo_correctness",
    }
    gate_term_distribution = {name: _distribution_summary([float(v) for v in soft_outputs.get(key) or []]) for name, key in term_keys.items()}
    gate_leave_one_out_distribution = {name: _distribution_summary([float(v) for v in soft_outputs.get(key) or []]) for name, key in loo_keys.items()}
    top_gate_term = max(
        gate_term_distribution,
        key=lambda name: _as_float(gate_term_distribution[name].get("mean_abs"), -1.0),
        default="delta",
    )
    top_gate_loo = max(
        gate_leave_one_out_distribution,
        key=lambda name: _as_float(gate_leave_one_out_distribution[name].get("mean_abs"), -1.0),
        default="delta",
    )
    return {
        "gate_term_distribution": gate_term_distribution,
        "gate_leave_one_out_distribution": gate_leave_one_out_distribution,
        "top_gate_term_by_abs_contribution": top_gate_term,
        "top_gate_term_by_abs_leave_one_out": top_gate_loo,
        "gate_term_mean_abs": {name: stats.get("mean_abs") for name, stats in gate_term_distribution.items()},
        "gate_terms_rank_consistency": _gate_term_rank_consistency(soft_outputs),
    }


def _soft_gate_uncertainty_summary(
    soft_outputs: dict[str, list[float]],
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
) -> dict[str, Any]:
    gbdt_half_width: list[float] = []
    afno_half_width: list[float] = []
    expert_disagreement_var: list[float] = []
    total_variance_proxy: list[float] = []
    for alpha, g_pred, a_pred, g_lower, g_upper, a_lower, a_upper in zip(
        soft_outputs.get("afno_weight") or [],
        gbdt_outputs.get("y_pred") or [],
        afno_outputs.get("y_pred") or [],
        gbdt_outputs.get("lower") or [],
        gbdt_outputs.get("upper") or [],
        afno_outputs.get("lower") or [],
        afno_outputs.get("upper") or [],
    ):
        g_half = max(0.0, (float(g_upper) - float(g_lower)) / 2.0)
        a_half = max(0.0, (float(a_upper) - float(a_lower)) / 2.0)
        weight = min(max(float(alpha), 0.0), 1.0)
        disagreement = weight * (1.0 - weight) * ((float(a_pred) - float(g_pred)) ** 2)
        total_proxy = (1.0 - weight) * (g_half**2) + weight * (a_half**2) + disagreement
        gbdt_half_width.append(g_half)
        afno_half_width.append(a_half)
        expert_disagreement_var.append(disagreement)
        total_variance_proxy.append(total_proxy)
    components_distribution = {
        "gbdt_interval_half_width": _distribution_summary(gbdt_half_width),
        "afno_interval_half_width": _distribution_summary(afno_half_width),
        "expert_disagreement_var": _distribution_summary(expert_disagreement_var),
        "total_variance_proxy": _distribution_summary(total_variance_proxy),
    }
    return {
        "method": "hybrid_interval_proxy_v1",
        "components_summary": {
            "mean_gbdt_half_width": components_distribution["gbdt_interval_half_width"].get("mean"),
            "mean_afno_half_width": components_distribution["afno_interval_half_width"].get("mean"),
            "mean_expert_disagreement_var": components_distribution["expert_disagreement_var"].get("mean"),
            "mean_total_variance_proxy": components_distribution["total_variance_proxy"].get("mean"),
        },
        "components_distribution": components_distribution,
    }


def _apply_soft_gate_envelope_interval(
    soft_outputs: dict[str, list[float]],
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    interval_scale: float,
) -> dict[str, list[float]]:
    return _shared_apply_soft_gate_envelope_interval(soft_outputs, gbdt_outputs, afno_outputs, interval_scale=interval_scale)


def _optimize_soft_interval_scale(
    soft_outputs: dict[str, list[float]],
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    target_cov: float,
    scale_grid: list[float] | None = None,
) -> dict[str, float]:
    candidate_scales = scale_grid or [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
    best: dict[str, float] | None = None
    for scale in candidate_scales:
        calibrated = _apply_soft_gate_envelope_interval(soft_outputs, gbdt_outputs, afno_outputs, interval_scale=float(scale))
        metrics = _metrics_from_outputs(calibrated)
        coverage = float(metrics["cov90"])
        width = float(metrics["width90"])
        rank = (0.0 if coverage >= target_cov else 1.0, _coverage_shortfall(coverage, target_cov), width)
        candidate = {"scale": float(scale), "coverage": coverage, "width90": width, "rank": rank}
        if best is None or tuple(candidate["rank"]) < tuple(best["rank"]):
            best = candidate
    if best is not None:
        center = float(best["scale"])
        refined_grid = sorted({round(float(v), 6) for v in list(np.linspace(max(0.6, center - 0.35), center + 0.35, num=9, dtype=float))})
        for scale in refined_grid:
            calibrated = _apply_soft_gate_envelope_interval(soft_outputs, gbdt_outputs, afno_outputs, interval_scale=float(scale))
            metrics = _metrics_from_outputs(calibrated)
            coverage = float(metrics["cov90"])
            width = float(metrics["width90"])
            rank = (0.0 if coverage >= target_cov else 1.0, _coverage_shortfall(coverage, target_cov), width)
            candidate = {"scale": float(scale), "coverage": coverage, "width90": width, "rank": rank}
            if tuple(candidate["rank"]) < tuple(best["rank"]):
                best = candidate
    return best or {"scale": 1.0, "coverage": float("nan"), "width90": float("nan"), "rank": (1.0, float("inf"), float("inf"))}


def _optimize_soft_gate_strategy(
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    target_cov: float,
    use_condition_clusters: bool,
    scope_name: str = "default",
    interval_gbdt_outputs: dict[str, list[float]] | None = None,
    interval_afno_outputs: dict[str, list[float]] | None = None,
    mode: str = "point_first",
) -> dict[str, Any]:
    tau_grid = _compact_numeric_grid(_candidate_tau_grid(gbdt_outputs, afno_outputs), max_points=7 if mode == "correctness_meta" else 5)
    if mode == "risk_aware":
        temperature_grid = [0.05, 0.1, 0.2, 0.35]
        coef_delta_grid = [0.5, 1.0, 2.0]
        coef_overlap_grid = [0.0, 0.5, 1.0]
        coef_width_grid = [-0.5, 0.0, 0.5]
        coef_tail_grid = [-0.5, 0.0, 0.5]
        coef_condition_grid = [0.0, 0.5, 1.0] if use_condition_clusters else [0.0]
        coef_correctness_grid = [0.0]
    elif mode == "point_first":
        temperature_grid = [0.03, 0.05, 0.1, 0.2, 0.35]
        coef_delta_grid = [0.5, 1.0, 2.0]
        coef_overlap_grid = [0.0, 0.5, 1.0]
        coef_width_grid = [-0.5, 0.0, 0.5]
        coef_tail_grid = [-0.5, 0.0, 0.5]
        coef_condition_grid = [0.0, 0.5, 1.0] if use_condition_clusters else [0.0]
        coef_correctness_grid = [0.0]
    else:
        temperature_grid = [0.03, 0.05, 0.1, 0.2, 0.35]
        coef_delta_grid = [0.5, 1.0, 2.0]
        coef_overlap_grid = [0.0, 0.5, 1.0]
        coef_width_grid = [-0.5, 0.0, 0.5]
        coef_tail_grid = [-0.5, 0.0, 0.5]
        coef_condition_grid = [0.0, 0.5, 1.0] if use_condition_clusters else [0.0]
        coef_correctness_grid = [0.25, 0.5, 1.0]
    condition_advantage = _condition_advantage_map(gbdt_outputs, afno_outputs) if use_condition_clusters else {}
    correctness_priors = _soft_gate_correctness_priors(gbdt_outputs, afno_outputs) if mode == "correctness_meta" else {}
    best: dict[str, Any] | None = None
    for temperature in temperature_grid:
        for tau in tau_grid:
            for coef_delta in coef_delta_grid:
                for coef_overlap in coef_overlap_grid:
                    for coef_width in coef_width_grid:
                        for coef_tail in coef_tail_grid:
                            for coef_condition in coef_condition_grid:
                                for coef_correctness in coef_correctness_grid:
                                    soft_outputs = _soft_gate_outputs(
                                        gbdt_outputs,
                                        afno_outputs,
                                        temperature=float(temperature),
                                        tau=float(tau),
                                        coef_delta=float(coef_delta),
                                        coef_overlap=float(coef_overlap),
                                        coef_width=float(coef_width),
                                        coef_tail=float(coef_tail),
                                        coef_condition=float(coef_condition),
                                        coef_correctness=float(coef_correctness),
                                        condition_advantage=condition_advantage,
                                        correctness_priors=correctness_priors,
                                    )
                                    point_metrics = {
                                        "rmse": _rmse(soft_outputs["y_true"], soft_outputs["y_pred"]),
                                        "mae": _mae(soft_outputs["y_true"], soft_outputs["y_pred"]),
                                        "nasa_score": _nasa_score(soft_outputs["y_true"], soft_outputs["y_pred"]),
                                    }
                                    avg_entropy = _soft_gate_weight_entropy([float(value) for value in soft_outputs.get("afno_weight") or []])

                                    interval_scale = 1.0
                                    interval_coverage = float("nan")
                                    interval_width = float("nan")
                                    if interval_gbdt_outputs is not None and interval_afno_outputs is not None:
                                        interval_soft_outputs = _soft_gate_outputs(
                                            interval_gbdt_outputs,
                                            interval_afno_outputs,
                                            temperature=float(temperature),
                                            tau=float(tau),
                                            coef_delta=float(coef_delta),
                                            coef_overlap=float(coef_overlap),
                                            coef_width=float(coef_width),
                                            coef_tail=float(coef_tail),
                                            coef_condition=float(coef_condition),
                                            coef_correctness=float(coef_correctness),
                                            condition_advantage=condition_advantage,
                                            correctness_priors=correctness_priors,
                                        )
                                        interval_meta = _optimize_soft_interval_scale(
                                            interval_soft_outputs,
                                            interval_gbdt_outputs,
                                            interval_afno_outputs,
                                            target_cov=target_cov,
                                        )
                                        interval_scale = float(interval_meta.get("scale") or 1.0)
                                        interval_coverage = float(interval_meta.get("coverage") or float("nan"))
                                        interval_width = float(interval_meta.get("width90") or float("nan"))

                                    rank = _soft_gate_candidate_rank(
                                        point_metrics=point_metrics,
                                        interval_coverage=interval_coverage,
                                        interval_width=interval_width,
                                        avg_entropy=avg_entropy,
                                        target_cov=target_cov,
                                        scope_name=scope_name,
                                        mode=mode,
                                    )
                                    candidate = {
                                        "temperature": float(temperature),
                                        "tau": float(tau),
                                        "coef_delta": float(coef_delta),
                                        "coef_overlap": float(coef_overlap),
                                        "coef_width": float(coef_width),
                                        "coef_tail": float(coef_tail),
                                        "coef_condition": float(coef_condition),
                                        "coef_correctness": float(coef_correctness),
                                        "condition_advantage": condition_advantage,
                                        "correctness_priors": correctness_priors,
                                        "interval_scale": interval_scale,
                                        "interval_calib_cov90": interval_coverage,
                                        "interval_calib_width90": interval_width,
                                        "coverage": interval_coverage,
                                        "rmse": float(point_metrics["rmse"]),
                                        "nasa_score": float(point_metrics["nasa_score"]),
                                        "width90": interval_width,
                                        "soft_gate_avg_entropy": avg_entropy,
                                        "soft_gate_avg_afno_weight": float(np.mean(soft_outputs.get("afno_weight") or [0.0])),
                                        "rank": rank,
                                        "scope_name": scope_name,
                                        "mode": mode,
                                    }
                                    if best is None or tuple(candidate["rank"]) < tuple(best["rank"]):
                                        best = candidate
    return best or {
        "temperature": 0.5,
        "tau": 0.0,
        "coef_delta": 1.0,
        "coef_overlap": 0.0,
        "coef_width": 0.0,
        "coef_tail": 0.0,
        "coef_condition": 0.0,
        "coef_correctness": 0.0,
        "condition_advantage": {},
        "correctness_priors": {},
        "interval_scale": 1.0,
        "interval_calib_cov90": float("nan"),
        "interval_calib_width90": float("nan"),
        "coverage": float("nan"),
        "rmse": float("nan"),
        "nasa_score": float("nan"),
        "width90": float("nan"),
        "soft_gate_avg_entropy": float("nan"),
        "soft_gate_avg_afno_weight": float("nan"),
        "rank": (1.0, float("inf"), float("inf"), float("inf")),
        "scope_name": scope_name,
        "mode": mode,
    }


def _evaluate_soft_gate_variant(
    *,
    label: str,
    mode: str,
    gbdt_bundle: dict[str, HistGradientBoostingRegressor],
    feature_keys: list[str],
    explanation_records: list[dict[str, object]],
    afno_snapshot: dict[str, Any],
    afno_state_dict: dict[str, Any],
    afno_algo: str,
    afno_point_scale: float = 1.0,
    afno_point_bias: float = 0.0,
    afno_point_adapter: dict[str, Any] | None = None,
    explanation_regime: tuple[float, float, float] | None = None,
    eval_gbdt_outputs: dict[str, list[float]],
    eval_afno_outputs: dict[str, list[float]],
    calib_gbdt_outputs: dict[str, list[float]],
    calib_afno_outputs: dict[str, list[float]],
    interval_gbdt_outputs: dict[str, list[float]],
    interval_afno_outputs: dict[str, list[float]],
    target_cov: float,
    scope_name: str,
    use_condition_clusters: bool,
) -> dict[str, Any]:
    gate_meta = _optimize_soft_gate_strategy(
        calib_gbdt_outputs,
        calib_afno_outputs,
        target_cov=target_cov,
        use_condition_clusters=use_condition_clusters,
        scope_name=scope_name,
        interval_gbdt_outputs=interval_gbdt_outputs,
        interval_afno_outputs=interval_afno_outputs,
        mode=mode,
    )
    gated_outputs = _soft_gate_outputs(
        eval_gbdt_outputs,
        eval_afno_outputs,
        temperature=float(gate_meta.get("temperature") or 0.5),
        tau=float(gate_meta.get("tau") or 0.0),
        coef_delta=float(gate_meta.get("coef_delta") or 1.0),
        coef_overlap=float(gate_meta.get("coef_overlap") or 0.0),
        coef_width=float(gate_meta.get("coef_width") or 0.0),
        coef_tail=float(gate_meta.get("coef_tail") or 0.0),
        coef_condition=float(gate_meta.get("coef_condition") or 0.0),
        coef_correctness=float(gate_meta.get("coef_correctness") or 0.0),
        condition_advantage={str(k): float(v) for k, v in (gate_meta.get("condition_advantage") or {}).items()},
        correctness_priors=gate_meta.get("correctness_priors") if isinstance(gate_meta.get("correctness_priors"), dict) else {},
    )
    interval_meta = {
        "scale": gate_meta.get("interval_scale"),
        "coverage": gate_meta.get("interval_calib_cov90"),
        "width90": gate_meta.get("interval_calib_width90"),
    }
    calibrated_gated_outputs = _apply_soft_gate_envelope_interval(
        gated_outputs,
        eval_gbdt_outputs,
        eval_afno_outputs,
        interval_scale=float(interval_meta.get("scale") or 1.0),
    )
    gated_metrics = _metrics_from_outputs(calibrated_gated_outputs)
    coverage_guard_triggered = False
    coverage_guard_source = "none"

    if mode == "risk_aware" and float(gated_metrics.get("cov90") or float("nan")) < float(target_cov):
        interval_soft_outputs = _soft_gate_outputs(
            interval_gbdt_outputs,
            interval_afno_outputs,
            temperature=float(gate_meta.get("temperature") or 0.5),
            tau=float(gate_meta.get("tau") or 0.0),
            coef_delta=float(gate_meta.get("coef_delta") or 1.0),
            coef_overlap=float(gate_meta.get("coef_overlap") or 0.0),
            coef_width=float(gate_meta.get("coef_width") or 0.0),
            coef_tail=float(gate_meta.get("coef_tail") or 0.0),
            coef_condition=float(gate_meta.get("coef_condition") or 0.0),
            coef_correctness=float(gate_meta.get("coef_correctness") or 0.0),
            condition_advantage={str(k): float(v) for k, v in (gate_meta.get("condition_advantage") or {}).items()},
            correctness_priors=gate_meta.get("correctness_priors") if isinstance(gate_meta.get("correctness_priors"), dict) else {},
        )
        stronger_interval_meta = _optimize_soft_interval_scale(
            interval_soft_outputs,
            interval_gbdt_outputs,
            interval_afno_outputs,
            target_cov=min(0.98, float(target_cov) + 0.02),
            scale_grid=[1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
        )
        strengthened_outputs = _apply_soft_gate_envelope_interval(
            gated_outputs,
            eval_gbdt_outputs,
            eval_afno_outputs,
            interval_scale=float(stronger_interval_meta.get("scale") or interval_meta.get("scale") or 1.0),
        )
        strengthened_metrics = _metrics_from_outputs(strengthened_outputs)
        if float(strengthened_metrics.get("cov90") or float("nan")) >= float(gated_metrics.get("cov90") or float("nan")):
            interval_meta = {
                "scale": stronger_interval_meta.get("scale"),
                "coverage": stronger_interval_meta.get("coverage"),
                "width90": stronger_interval_meta.get("width90"),
            }
            calibrated_gated_outputs = strengthened_outputs
            gated_metrics = strengthened_metrics
            coverage_guard_source = "stronger_posthoc_interval"

        if float(gated_metrics.get("cov90") or float("nan")) < float(target_cov):
            coverage_guard_triggered = True
            coverage_guard_source = "gbdt_fallback"
            calibrated_gated_outputs = {
                **eval_gbdt_outputs,
                "afno_weight": [0.0 for _ in eval_gbdt_outputs.get("y_pred") or []],
                "correctness_score": [0.0 for _ in eval_gbdt_outputs.get("y_pred") or []],
            }
            gated_metrics = _metrics_from_outputs(calibrated_gated_outputs)
            interval_meta = {
                "scale": float(eval_gbdt_outputs.get("interval_scale") or gate_meta.get("interval_scale") or 1.0),
                "coverage": float(gated_metrics.get("cov90") or float("nan")),
                "width90": float(gated_metrics.get("width90") or float("nan")),
            }
    xai_summary = _soft_gate_explanation_summary(gated_outputs)
    uncertainty_summary = _soft_gate_uncertainty_summary(gated_outputs, eval_gbdt_outputs, eval_afno_outputs)
    uncertainty_quality = _soft_gate_uncertainty_quality(
        gated_outputs,
        eval_gbdt_outputs,
        eval_afno_outputs,
        coverage_guard_triggered=coverage_guard_triggered,
    )
    gbdt_feature_summary = _gbdt_global_importance_summary(
        gbdt_bundle,
        records=explanation_records,
        feature_keys=feature_keys,
    )
    afno_feature_summary = _afno_occlusion_summary(
        records=explanation_records,
        feature_keys=feature_keys,
        snapshot=afno_snapshot,
        state_dict=afno_state_dict,
        algo=afno_algo,
        point_scale=afno_point_scale,
        point_bias=afno_point_bias,
        point_adapter=afno_point_adapter,
        regime=explanation_regime,
    )
    row = {
        "model": label,
        "rmse": gated_metrics["rmse"],
        "mae": gated_metrics["mae"],
        "nasa_score": gated_metrics["nasa_score"],
        "cov90": gated_metrics["cov90"],
        "width90": gated_metrics["width90"],
        "interval_method": "soft_gate_envelope_90",
        "gate_strategy": f"soft_temperature_gate_envelope_{mode}",
        "soft_gate_mode": mode,
        "soft_gate_temperature": gate_meta.get("temperature"),
        "soft_gate_tau": gate_meta.get("tau"),
        "soft_gate_coef_delta": gate_meta.get("coef_delta"),
        "soft_gate_coef_overlap": gate_meta.get("coef_overlap"),
        "soft_gate_coef_width": gate_meta.get("coef_width"),
        "soft_gate_coef_tail": gate_meta.get("coef_tail"),
        "soft_gate_coef_condition": gate_meta.get("coef_condition"),
        "soft_gate_coef_correctness": gate_meta.get("coef_correctness"),
        "soft_gate_condition_advantage": gate_meta.get("condition_advantage"),
        "soft_gate_avg_afno_weight": gate_meta.get("soft_gate_avg_afno_weight"),
        "soft_gate_avg_entropy": gate_meta.get("soft_gate_avg_entropy"),
        "final_interval_scale": interval_meta.get("scale"),
        "final_interval_calib_cov90": interval_meta.get("coverage"),
        "final_interval_calib_width90": interval_meta.get("width90"),
        "coverage_guard_triggered": coverage_guard_triggered,
        "coverage_guard_source": coverage_guard_source,
        "uncertainty_method": uncertainty_summary.get("method"),
        "uncertainty_components_summary": uncertainty_summary.get("components_summary"),
        "xai_method": {
            "gate": "analytic_term_decomposition_v1",
            "gbdt": gbdt_feature_summary.get("method"),
            "afno": afno_feature_summary.get("method"),
        },
        "xai_summary": {
            "top_gate_term_by_abs_contribution": xai_summary.get("top_gate_term_by_abs_contribution"),
            "top_gate_term_by_abs_leave_one_out": xai_summary.get("top_gate_term_by_abs_leave_one_out"),
            "gate_term_mean_abs": xai_summary.get("gate_term_mean_abs"),
            "gate_terms_rank_consistency": xai_summary.get("gate_terms_rank_consistency"),
            "afno_occlusion_topk_jaccard": ((afno_feature_summary.get("stability") or {}).get("topk_jaccard") if isinstance(afno_feature_summary.get("stability"), dict) else None),
            "gbdt_top_feature_global": (gbdt_feature_summary.get("top_features_global") or [{}])[0].get("feature"),
            "afno_top_feature_global": (afno_feature_summary.get("top_features_global") or [{}])[0].get("feature"),
        },
    }
    experiment = {
        "gate_strategy": f"soft_temperature_gate_envelope_{mode}",
        "soft_gate_mode": mode,
        "soft_gate_temperature": gate_meta.get("temperature"),
        "soft_gate_tau": gate_meta.get("tau"),
        "soft_gate_coef_delta": gate_meta.get("coef_delta"),
        "soft_gate_coef_overlap": gate_meta.get("coef_overlap"),
        "soft_gate_coef_width": gate_meta.get("coef_width"),
        "soft_gate_coef_tail": gate_meta.get("coef_tail"),
        "soft_gate_coef_condition": gate_meta.get("coef_condition"),
        "soft_gate_coef_correctness": gate_meta.get("coef_correctness"),
        "soft_gate_condition_advantage": gate_meta.get("condition_advantage"),
        "soft_gate_correctness_priors": gate_meta.get("correctness_priors"),
        "soft_gate_avg_afno_weight": gate_meta.get("soft_gate_avg_afno_weight"),
        "soft_gate_avg_entropy": gate_meta.get("soft_gate_avg_entropy"),
        "final_interval_method": "soft_gate_envelope_90",
        "final_interval_scale": interval_meta.get("scale"),
        "final_interval_calib_cov90": interval_meta.get("coverage"),
        "final_interval_calib_width90": interval_meta.get("width90"),
        "coverage_guard_triggered": coverage_guard_triggered,
        "coverage_guard_source": coverage_guard_source,
        "uncertainty_schema_version": "v1",
        "xai_schema_version": "v1",
        "uncertainty_method": uncertainty_summary.get("method"),
        "uncertainty_components_summary": uncertainty_summary.get("components_summary"),
        "uncertainty_components_distribution": uncertainty_summary.get("components_distribution"),
        "xai_method": {
            "gate": "analytic_term_decomposition_v1",
            "gbdt": gbdt_feature_summary.get("method"),
            "afno": afno_feature_summary.get("method"),
        },
        "gate_term_distribution": xai_summary.get("gate_term_distribution"),
        "gate_leave_one_out_distribution": xai_summary.get("gate_leave_one_out_distribution"),
        "xai_summary": {
            "top_gate_term_by_abs_contribution": xai_summary.get("top_gate_term_by_abs_contribution"),
            "top_gate_term_by_abs_leave_one_out": xai_summary.get("top_gate_term_by_abs_leave_one_out"),
            "gate_term_mean_abs": xai_summary.get("gate_term_mean_abs"),
            "gate_terms_rank_consistency": xai_summary.get("gate_terms_rank_consistency"),
            "afno_occlusion_topk_jaccard": ((afno_feature_summary.get("stability") or {}).get("topk_jaccard") if isinstance(afno_feature_summary.get("stability"), dict) else None),
            "gbdt_top_feature_global": (gbdt_feature_summary.get("top_features_global") or [{}])[0].get("feature"),
            "afno_top_feature_global": (afno_feature_summary.get("top_features_global") or [{}])[0].get("feature"),
        },
        "feature_attribution_summary": {
            "gbdt_top_features_global": gbdt_feature_summary.get("top_features_global") or [],
            "afno_top_features_global": afno_feature_summary.get("top_features_global") or [],
            "gbdt_feature_importance_mean_abs": gbdt_feature_summary.get("feature_importance_mean_abs") or {},
            "afno_feature_importance_mean_abs": afno_feature_summary.get("feature_importance_mean_abs") or {},
        },
        "xai_stability": {
            "gate_terms_rank_consistency": xai_summary.get("gate_terms_rank_consistency"),
            "afno_occlusion_topk_jaccard": ((afno_feature_summary.get("stability") or {}).get("topk_jaccard") if isinstance(afno_feature_summary.get("stability"), dict) else None),
            "afno_occlusion_rank_consistency": ((afno_feature_summary.get("stability") or {}).get("rank_consistency") if isinstance(afno_feature_summary.get("stability"), dict) else None),
        },
        "uncertainty_quality": uncertainty_quality,
        **{f"backtest_{k}": v for k, v in gated_metrics.items()},
    }
    return {
        "label": label,
        "mode": mode,
        "gate_meta": gate_meta,
        "interval_meta": interval_meta,
        "outputs": calibrated_gated_outputs,
        "metrics": gated_metrics,
        "row": row,
        "experiment": experiment,
    }


def _pareto_front(rows: list[dict[str, Any]], *, minimize_keys: list[str]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for idx, candidate in enumerate(rows):
        dominated = False
        for jdx, other in enumerate(rows):
            if idx == jdx:
                continue
            other_better_or_equal = all(_as_float(other.get(key), float("inf")) <= _as_float(candidate.get(key), float("inf")) for key in minimize_keys)
            other_strictly_better = any(_as_float(other.get(key), float("inf")) < _as_float(candidate.get(key), float("inf")) for key in minimize_keys)
            if other_better_or_equal and other_strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    front.sort(key=lambda item: tuple(_as_float(item.get(key), float("inf")) for key in minimize_keys))
    return front


def _soft_gate_pareto_summary(*, main_rows: list[dict[str, Any]], operating_rows: list[dict[str, Any]], target_cov: float) -> dict[str, Any]:
    by_label: dict[str, dict[str, Any]] = {}
    for row in main_rows:
        label = str(row.get("model") or "")
        by_label.setdefault(label, {})["main"] = row
    for row in operating_rows:
        label = str(row.get("model") or "")
        by_label.setdefault(label, {})["operating"] = row
    candidates: list[dict[str, Any]] = []
    for label, payload in by_label.items():
        main_row = payload.get("main") if isinstance(payload.get("main"), dict) else None
        operating_row = payload.get("operating") if isinstance(payload.get("operating"), dict) else None
        if main_row is None or operating_row is None:
            continue
        candidates.append(
            {
                "model": label,
                "main_rmse": main_row.get("rmse"),
                "main_nasa_score": main_row.get("nasa_score"),
                "main_cov_shortfall": _coverage_shortfall(_as_float(main_row.get("cov90"), float("nan")), target_cov),
                "operating_rmse": operating_row.get("rmse"),
                "operating_nasa_score": operating_row.get("nasa_score"),
                "operating_cov_shortfall": _coverage_shortfall(_as_float(operating_row.get("cov90"), float("nan")), target_cov),
            }
        )
    minimize_keys = ["main_rmse", "main_nasa_score", "main_cov_shortfall", "operating_rmse", "operating_nasa_score", "operating_cov_shortfall"]
    return {
        "target_cov": target_cov,
        "candidates": candidates,
        "front": _pareto_front(candidates, minimize_keys=minimize_keys),
        "minimize_keys": minimize_keys,
    }


def _gate_backtest_outputs(
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    margin: float = 0.0,
    overlap_max: float = 1.0,
    disagreement_min: float = 0.0,
    per_condition_margin: dict[str, float] | None = None,
    per_condition_params: dict[str, dict[str, float]] | None = None,
) -> dict[str, list[float]]:
    y_true = gbdt_outputs["y_true"]
    if len(y_true) != len(afno_outputs["y_true"]):
        raise ValueError("Backtest outputs must have aligned lengths for gating")
    y_pred: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    condition_keys = [str(key) for key in gbdt_outputs.get("condition_key") or []]
    for g_pred, a_pred, g_lower, a_lower, g_upper, a_upper in zip(
        gbdt_outputs["y_pred"],
        afno_outputs["y_pred"],
        gbdt_outputs["lower"],
        afno_outputs["lower"],
        gbdt_outputs["upper"],
        afno_outputs["upper"],
    ):
        idx = len(y_pred)
        condition_key = condition_keys[idx] if idx < len(condition_keys) else "global"
        condition_params = (per_condition_params or {}).get(condition_key, {})
        active_margin = float(condition_params.get("margin", (per_condition_margin or {}).get(condition_key, margin)))
        active_overlap = float(condition_params.get("overlap_max", overlap_max))
        active_disagreement = float(condition_params.get("disagreement_min", disagreement_min))
        delta = float(a_pred) - float(g_pred)
        overlap_ratio = _interval_overlap_ratio(float(g_lower), float(g_upper), float(a_lower), float(a_upper))
        disagreement = abs(delta)
        use_afno = (delta < -active_margin) and (overlap_ratio <= active_overlap or disagreement >= active_disagreement)
        if use_afno:
            y_pred.append(float(a_pred))
            lower.append(float(a_lower))
            upper.append(float(a_upper))
        else:
            y_pred.append(float(g_pred))
            lower.append(float(g_lower))
            upper.append(float(g_upper))
    return {"y_true": y_true, "y_pred": y_pred, "lower": lower, "upper": upper, "condition_key": condition_keys[: len(y_pred)]}


def _optimize_gate_strategy(
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    target_cov: float,
    use_condition_clusters: bool,
    scope_name: str = "default",
) -> dict[str, Any]:
    diffs = [abs(float(a) - float(g)) for g, a in zip(gbdt_outputs["y_pred"], afno_outputs["y_pred"]) if math.isfinite(float(a)) and math.isfinite(float(g))]
    tau_grid = sorted(set(_candidate_tau_grid(gbdt_outputs, afno_outputs) + _dense_threshold_grid(diffs, points=31)))
    disagreement_grid = sorted(set(_candidate_disagreement_grid(gbdt_outputs, afno_outputs) + _dense_threshold_grid(diffs, points=21)))
    overlap_grid = [1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    condition_keys = [str(key) for key in gbdt_outputs.get("condition_key") or []]
    primary_metric = "rmse" if scope_name == "unit_holdout" else "nasa_score"
    secondary_metrics = ("nasa_score", "width90") if scope_name == "unit_holdout" else ("rmse", "width90")
    allow_full_condition_overrides = scope_name == "unit_holdout"

    def search_best_params(
        subset_g: dict[str, list[float]],
        subset_a: dict[str, list[float]],
        *,
        tau_candidates: list[float],
        overlap_candidates: list[float],
        disagreement_candidates: list[float],
    ) -> dict[str, Any]:
        best_local: dict[str, Any] | None = None
        for local_overlap in overlap_candidates:
            for local_disagreement in disagreement_candidates:
                for tau in tau_candidates:
                    gated = _gate_backtest_outputs(
                        subset_g,
                        subset_a,
                        margin=float(tau),
                        overlap_max=float(local_overlap),
                        disagreement_min=float(local_disagreement),
                    )
                    metrics = _metrics_from_outputs(gated)
                    rank = _gate_candidate_rank(
                        metrics,
                        target_cov=target_cov,
                        primary_metric=primary_metric,
                        secondary_metrics=secondary_metrics,
                    )
                    candidate = {
                        "margin": float(tau),
                        "overlap_max": float(local_overlap),
                        "disagreement_min": float(local_disagreement),
                        "metrics": metrics,
                        "rank": rank,
                    }
                    if best_local is None or tuple(candidate["rank"]) < tuple(best_local["rank"]):
                        best_local = candidate
        return best_local or {
            "margin": 0.0,
            "overlap_max": 1.0,
            "disagreement_min": 0.0,
            "metrics": {"cov90": float("nan"), "width90": float("nan"), "rmse": float("nan"), "nasa_score": float("nan")},
            "rank": (1.0, float("inf"), float("inf"), float("inf"), float("inf")),
        }

    best: dict[str, Any] | None = None
    cluster_indices: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(condition_keys):
        cluster_indices[key].append(idx)

    def evaluate_candidate(base_params: dict[str, Any]) -> dict[str, Any]:
        global_tau = float(base_params.get("margin") or 0.0)
        overlap_max = float(base_params.get("overlap_max") or 1.0)
        disagreement_min = float(base_params.get("disagreement_min") or 0.0)
        per_condition_tau: dict[str, float] = {}
        per_condition_params: dict[str, dict[str, float]] = {}
        if use_condition_clusters and cluster_indices:
            for key, indices in cluster_indices.items():
                min_cluster_size = 6 if scope_name == "unit_holdout" else 12
                if len(indices) < min_cluster_size:
                    continue
                subset_g = _subset_outputs(gbdt_outputs, indices)
                subset_a = _subset_outputs(afno_outputs, indices)
                refined_tau = _refine_numeric_grid(tau_grid, global_tau, steps=11, width_ratio=0.2)
                refined_overlap = sorted({round(float(v), 6) for v in overlap_grid + list(np.linspace(max(0.0, overlap_max - 0.15), min(1.0, overlap_max + 0.15), num=9, dtype=float))})
                refined_disagreement = _refine_numeric_grid(disagreement_grid, disagreement_min, steps=11, width_ratio=0.2)
                local_best = search_best_params(
                    subset_g,
                    subset_a,
                    tau_candidates=refined_tau,
                    overlap_candidates=refined_overlap if allow_full_condition_overrides else [overlap_max],
                    disagreement_candidates=refined_disagreement if allow_full_condition_overrides else [disagreement_min],
                )
                per_condition_tau[key] = float(local_best.get("margin") or global_tau)
                if allow_full_condition_overrides:
                    per_condition_params[key] = {
                        "margin": float(local_best.get("margin") or global_tau),
                        "overlap_max": float(local_best.get("overlap_max") or overlap_max),
                        "disagreement_min": float(local_best.get("disagreement_min") or disagreement_min),
                    }
        gated = _gate_backtest_outputs(
            gbdt_outputs,
            afno_outputs,
            margin=global_tau,
            overlap_max=overlap_max,
            disagreement_min=disagreement_min,
            per_condition_margin=per_condition_tau,
            per_condition_params=per_condition_params,
        )
        metrics = _metrics_from_outputs(gated)
        rank = _gate_candidate_rank(
            metrics,
            target_cov=target_cov,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
        )
        return {
            "margin": float(global_tau),
            "overlap_max": float(overlap_max),
            "disagreement_min": float(disagreement_min),
            "per_condition_margin": per_condition_tau,
            "per_condition_params": per_condition_params,
            "coverage": _safe_metric_value(metrics, "cov90"),
            "rmse": _safe_metric_value(metrics, "rmse"),
            "nasa_score": _safe_metric_value(metrics, "nasa_score"),
            "width90": _safe_metric_value(metrics, "width90"),
            "rank": rank,
            "uses_condition_clusters": bool(per_condition_tau),
        }

    global_best = search_best_params(
        gbdt_outputs,
        afno_outputs,
        tau_candidates=tau_grid,
        overlap_candidates=overlap_grid,
        disagreement_candidates=disagreement_grid,
    )
    best = evaluate_candidate(global_best)

    if best is not None:
        refined_tau = _refine_numeric_grid(tau_grid, float(best.get("margin") or 0.0), steps=13, width_ratio=0.15)
        refined_overlap = sorted({round(float(v), 6) for v in overlap_grid + list(np.linspace(max(0.0, float(best.get("overlap_max") or 1.0) - 0.15), min(1.0, float(best.get("overlap_max") or 1.0) + 0.15), num=9, dtype=float))})
        refined_disagreement = _refine_numeric_grid(disagreement_grid, float(best.get("disagreement_min") or 0.0), steps=13, width_ratio=0.15)
        refined_global = search_best_params(
            gbdt_outputs,
            afno_outputs,
            tau_candidates=refined_tau,
            overlap_candidates=refined_overlap,
            disagreement_candidates=refined_disagreement,
        )
        candidate = evaluate_candidate(refined_global)
        if tuple(candidate["rank"]) < tuple(best["rank"]):
            best = candidate
    return best or {
        "margin": 0.0,
        "overlap_max": 1.0,
        "disagreement_min": 0.0,
        "per_condition_margin": {},
        "per_condition_params": {},
        "coverage": float("nan"),
        "rmse": float("nan"),
        "nasa_score": float("nan"),
        "width90": float("nan"),
        "rank": (1.0, float("inf"), float("inf"), float("inf"), float("inf")),
        "uses_condition_clusters": False,
    }


def _generate_regime_residual_adapter_candidates(
    *,
    x_windows: np.ndarray,
    y_true: np.ndarray,
    base_preds: np.ndarray,
    feature_keys: list[str],
    affine: tuple[float, float],
    adaptation_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if x_windows.size == 0 or y_true.size < 8 or not feature_keys:
        return []
    split_idx = max(4, int(round(y_true.size * 0.8)))
    split_idx = min(split_idx, y_true.size - 2)
    if split_idx <= 0:
        return []
    train_windows = x_windows[:split_idx]
    valid_windows = x_windows[split_idx:]
    train_y = y_true[:split_idx]
    valid_y = y_true[split_idx:]
    train_base = base_preds[:split_idx]
    valid_base = base_preds[split_idx:]
    if valid_y.size == 0:
        return []
    scale, bias = affine
    train_y = _sanitize_adapter_matrix(train_y, clip=_ADAPTER_RESPONSE_ABS_CLIP)
    valid_y = _sanitize_adapter_matrix(valid_y, clip=_ADAPTER_RESPONSE_ABS_CLIP)
    train_affine = _sanitize_adapter_matrix(
        np.asarray(_apply_affine(train_base.tolist(), scale=scale, bias=bias), dtype=float),
        clip=_ADAPTER_RESPONSE_ABS_CLIP,
    )
    valid_affine = _sanitize_adapter_matrix(
        np.asarray(_apply_affine(valid_base.tolist(), scale=scale, bias=bias), dtype=float),
        clip=_ADAPTER_RESPONSE_ABS_CLIP,
    )
    train_residual = _sanitize_adapter_matrix(
        train_y - train_affine,
        clip=_ADAPTER_RESIDUAL_ABS_CLIP,
    )
    valid_residual = _sanitize_adapter_matrix(
        valid_y - valid_affine,
        clip=_ADAPTER_RESIDUAL_ABS_CLIP,
    )
    baseline_train_rmse = _rmse(train_y.tolist(), train_affine.tolist())
    baseline_valid_rmse = _rmse(valid_y.tolist(), valid_affine.tolist())
    trend_window = 6
    if adaptation_rows is not None and len(adaptation_rows) == y_true.size:
        trend_signs = _series_trend_signs_from_rows(adaptation_rows, scale=scale, bias=bias, trend_window=trend_window)
    else:
        trend_signs = np.where(y_true - np.asarray(_apply_affine(base_preds.tolist(), scale=scale, bias=bias), dtype=float) >= 0.0, 1.0, -1.0)
    candidates: list[dict[str, Any]] = []
    for feature_mode in ("last", "mean", "delta", "std"):
        for max_features in (4, 6, 8):
            selected_keys = _select_regime_feature_subset(
                train_windows,
                train_residual,
                feature_keys,
                max_features=max_features,
                mode=feature_mode,
            )
            if not selected_keys:
                continue
            for use_base_pred, adapter_type in ((True, "residual_ridge_with_affine_base"), (False, "residual_ridge_summary_only")):
                train_design = _ridge_adapter_design(
                    train_windows,
                    feature_keys=feature_keys,
                    selected_keys=selected_keys,
                    base_preds=train_affine,
                    use_base_pred=use_base_pred,
                )
                valid_design = _ridge_adapter_design(
                    valid_windows,
                    feature_keys=feature_keys,
                    selected_keys=selected_keys,
                    base_preds=valid_affine,
                    use_base_pred=use_base_pred,
                )
                if train_design.shape[1] == 0 or valid_design.shape[1] == 0:
                    continue
                for standardize in (False, True):
                    train_design_fit, valid_design_fit, design_mean, design_scale = _normalize_adapter_design(
                        train_design,
                        valid_design,
                        standardize=standardize,
                    )
                    abs_train_residual = _sanitize_adapter_matrix(
                        np.abs(train_residual),
                        clip=_ADAPTER_RESIDUAL_ABS_CLIP,
                    )
                    train_signs = trend_signs[:split_idx] if trend_signs.size >= split_idx else np.where(train_residual >= 0.0, 1.0, -1.0)
                    valid_signs = trend_signs[split_idx:] if trend_signs.size >= y_true.size else np.where(valid_residual >= 0.0, 1.0, -1.0)
                    fallback_sign = 1.0 if float(np.mean(train_signs)) >= 0.0 else -1.0
                    for alpha in (1.0, 5.0, 20.0, 50.0):
                        mag_model = Ridge(alpha=alpha)
                        mag_model.fit(train_design_fit, abs_train_residual)
                        raw_mag_coef = np.asarray(mag_model.coef_, dtype=float).reshape(-1)
                        raw_mag_intercept = _safe_adapter_scalar(
                            mag_model.intercept_,
                            default=0.0,
                            clip=_ADAPTER_RESPONSE_ABS_CLIP,
                        )
                        if not np.all(np.isfinite(raw_mag_coef)):
                            continue
                        for coef_clip in (None, 0.15, 0.3, 0.6):
                            clip_value = (
                                float(coef_clip)
                                if coef_clip is not None
                                else _ADAPTER_MATRIX_ABS_CLIP
                            )
                            mag_coef = _sanitize_adapter_matrix(
                                raw_mag_coef.copy(),
                                clip=clip_value,
                            )
                            if coef_clip is not None:
                                mag_coef = np.clip(mag_coef, -float(coef_clip), float(coef_clip))
                            for magnitude_scale in (0.25, 0.5, 0.75, 1.0):
                                base_adapter = {
                                    "type": f"residual_trend_two_stage_{'with_affine_base' if use_base_pred else 'summary_only'}_bounded",
                                    "feature_mode": feature_mode,
                                    "max_features": max_features,
                                    "selected_feature_keys": selected_keys,
                                    "use_base_pred": use_base_pred,
                                    "ridge_alpha": alpha,
                                    "trend_window": trend_window,
                                    "standardize_design": standardize,
                                    "design_mean": [float(v) for v in design_mean.tolist()] if design_mean is not None else None,
                                    "design_scale": [float(v) for v in design_scale.tolist()] if design_scale is not None else None,
                                    "coef_clip": float(coef_clip) if coef_clip is not None else None,
                                    "fallback_sign": float(fallback_sign),
                                    "mag_coef": [float(v) for v in mag_coef.tolist()],
                                    "mag_intercept": raw_mag_intercept,
                                    "magnitude_scale": float(magnitude_scale),
                                    "baseline_train_rmse": baseline_train_rmse,
                                    "baseline_validation_rmse": baseline_valid_rmse,
                                }
                                for clip_q in (0.8, 0.9, 0.95):
                                    residual_clip = _nearest_rank(np.abs(train_residual).tolist(), clip_q)
                                    if not math.isfinite(residual_clip) or residual_clip <= 0.0:
                                        continue
                                    for lower_q, upper_q, sign_mode in (
                                        (0.1, 0.9, "none"),
                                        (0.2, 0.8, "none"),
                                        (0.2, 0.8, "majority"),
                                        (0.25, 0.75, "mean"),
                                    ):
                                        residual_lower, residual_upper = _residual_bounds(
                                            train_residual,
                                            lower_q=lower_q,
                                            upper_q=upper_q,
                                            abs_q=clip_q,
                                            sign_mode=sign_mode,
                                        )
                                        adapter = {
                                            **base_adapter,
                                            "clip_quantile": clip_q,
                                            "residual_clip": float(residual_clip),
                                            "residual_lower": float(residual_lower),
                                            "residual_upper": float(residual_upper),
                                            "sign_mode": sign_mode,
                                            "residual_bound_quantiles": [float(lower_q), float(upper_q)],
                                        }
                                        train_residual_pred = _predict_adapter_residuals(train_design_fit, adapter, trend_signs=train_signs)
                                        valid_residual_pred = _predict_adapter_residuals(valid_design_fit, adapter, trend_signs=valid_signs)
                                        train_pred = train_affine + train_residual_pred
                                        valid_pred = valid_affine + valid_residual_pred
                                        if not np.all(np.isfinite(train_pred)) or not np.all(np.isfinite(valid_pred)):
                                            continue
                                        candidates.append(
                                            {
                                                **adapter,
                                                "train_rmse": _rmse(train_y.tolist(), train_pred.tolist()),
                                                "validation_rmse": _rmse(valid_y.tolist(), valid_pred.tolist()),
                                            }
                                        )
    candidates.sort(key=lambda item: (float(item.get("validation_rmse") or float("inf")), float(item.get("train_rmse") or float("inf"))))
    return candidates


def _fit_regime_ridge_adapter(
    *,
    x_windows: np.ndarray,
    y_true: np.ndarray,
    base_preds: np.ndarray,
    feature_keys: list[str],
    affine: tuple[float, float],
    adaptation_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    candidates = _generate_regime_residual_adapter_candidates(
        x_windows=x_windows,
        y_true=y_true,
        base_preds=base_preds,
        feature_keys=feature_keys,
        affine=affine,
        adaptation_rows=adaptation_rows,
    )
    return candidates[0] if candidates else None


def _ridge_adapter_design(x_windows: np.ndarray, *, feature_keys: list[str], selected_keys: list[str], base_preds: np.ndarray, use_base_pred: bool) -> np.ndarray:
    summarized = _summarize_regime_windows(x_windows, feature_keys=feature_keys, selected_keys=selected_keys)
    summarized = _sanitize_adapter_matrix(summarized)
    safe_base_preds = _sanitize_adapter_matrix(np.asarray(base_preds, dtype=float).reshape(-1, 1))
    if summarized.shape[1] == 0:
        return safe_base_preds if use_base_pred else np.zeros((x_windows.shape[0], 0), dtype=float)
    return np.column_stack([safe_base_preds, summarized]) if use_base_pred else summarized


def _apply_point_adapter(
    base_pred: float,
    *,
    history: list[dict[str, object]],
    feature_keys: list[str],
    adapter: dict[str, Any] | None,
    affine: tuple[float, float],
    snapshot: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
) -> float:
    scale, bias = affine
    pred = float(scale * float(base_pred) + bias)
    if not isinstance(adapter, dict) or not str(adapter.get("type") or "").startswith("residual_"):
        return pred
    selected_keys = [str(key) for key in adapter.get("selected_feature_keys") or [] if isinstance(key, str)]
    if not selected_keys:
        return pred
    window = _window_matrix_selected(history, feature_keys=feature_keys, selected_keys=selected_keys).reshape(1, WINDOW, -1)
    design = _ridge_adapter_design(
        window,
        feature_keys=selected_keys,
        selected_keys=selected_keys,
        base_preds=np.asarray([pred], dtype=float),
        use_base_pred=bool(adapter.get("use_base_pred")),
    )
    if bool(adapter.get("standardize_design")) and design.size:
        design_mean = np.asarray(adapter.get("design_mean") or [], dtype=float)
        design_scale = np.asarray(adapter.get("design_scale") or [], dtype=float)
        if design_mean.size == design.shape[1] and design_scale.size == design.shape[1]:
            safe_scale = np.where(np.abs(design_scale) > 1e-8, design_scale, 1.0)
            design = (design - design_mean.reshape(1, -1)) / safe_scale.reshape(1, -1)
    design = _sanitize_adapter_matrix(design).reshape(-1)
    trend_signs: np.ndarray | None = None
    if str(adapter.get("type") or "").startswith("residual_trend_two_stage") and isinstance(snapshot, dict) and isinstance(state_dict, dict):
        trend_signs = np.asarray(
            [
                _history_affine_trend_sign(
                    snapshot=snapshot,
                    state_dict=state_dict,
                    history=history,
                    affine=affine,
                    trend_window=int(adapter.get("trend_window") or 6),
                )
            ],
            dtype=float,
        )
    residual = float(_predict_adapter_residuals(design, adapter, trend_signs=trend_signs).reshape(-1)[0])
    return float(pred + residual)


def _backtest_predictor(
    predictor: Any,
    *,
    records: list[dict[str, object]],
    horizon: int,
    folds: int,
    feature_keys: list[str],
    kind: str,
    snapshot: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    algo: str | None = None,
    interval_qhat: float | None = None,
) -> dict[str, float]:
    by_series = _group_records(records)
    y_true: list[float] = []
    y_pred: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for rows in by_series.values():
        n = len(rows)
        if n < WINDOW + horizon + 1:
            continue
        for fold_idx in range(folds):
            end = n - fold_idx * horizon
            start = end - horizon
            train_end = start
            if train_end <= WINDOW or start < 0 or end > n:
                break
            history = rows[:train_end]
            future = rows[start:end]
            if len(future) != horizon:
                break
            if kind == "gbdt":
                preds, lowers, uppers = _predict_gbdt_horizon(predictor, history=history, future_rows=future, feature_keys=feature_keys, horizon=horizon)
                lower_bounds.extend(lowers)
                upper_bounds.extend(uppers)
            else:
                preds = forecast_univariate_torch(
                    algo=str(algo or AFNO_ALGO),
                    snapshot=snapshot or {},
                    state_dict=state_dict or {},
                    context_records=history[-WINDOW:],
                    future_feature_rows=future,
                    horizon=horizon,
                    device=None,
                )
                if interval_qhat is not None and math.isfinite(float(interval_qhat)):
                    width = float(max(interval_qhat, 0.0))
                    lower_bounds.extend([float(value) - width for value in preds])
                    upper_bounds.extend([float(value) + width for value in preds])
            y_true.extend([_as_float(row.get("y")) for row in future])
            y_pred.extend([float(value) for value in preds])
    interval_stats = _interval_metrics(y_true, lower_bounds, upper_bounds)
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "nasa_score": _nasa_score(y_true, y_pred),
        "count": float(len(y_true)),
        "cov90": float(interval_stats["cov90"]) if interval_stats["cov90"] is not None else float("nan"),
        "width90": float(interval_stats["width90"]) if interval_stats["width90"] is not None else float("nan"),
    }


def _regime_one_step_outputs(
    predictor: Any,
    *,
    records: list[dict[str, object]],
    feature_keys: list[str],
    regime: tuple[float, float, float],
    kind: str,
    snapshot: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    algo: str | None = None,
    interval_qhat: float | None = None,
    point_scale: float = 1.0,
    point_bias: float = 0.0,
    point_adapter: dict[str, Any] | None = None,
) -> dict[str, list[float]]:
    by_series = _group_records(records)
    y_true: list[float] = []
    y_pred: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    condition_keys: list[str] = []
    tail_pos: list[float] = []
    for rows in by_series.values():
        if len(rows) <= WINDOW:
            continue
        for idx in range(WINDOW, len(rows)):
            target = rows[idx]
            if _regime_key(target) != regime:
                continue
            history = rows[:idx]
            future = [target]
            if kind == "gbdt":
                inference_bundle = dict(predictor)
                inference_bundle.setdefault("feature_keys", feature_keys)
                pred, lower, upper = shared_gbdt_pipeline.predict_rul(inference_bundle, history)
                preds = [pred]
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            else:
                raw_preds = forecast_univariate_torch(
                    algo=str(algo or AFNO_ALGO),
                    snapshot=snapshot or {},
                    state_dict=state_dict or {},
                    context_records=history[-WINDOW:],
                    future_feature_rows=future,
                    horizon=1,
                    device=None,
                )
                adapted = _apply_point_adapter(
                    float(raw_preds[0]),
                    history=history[-WINDOW:],
                    feature_keys=feature_keys,
                    adapter=point_adapter,
                    affine=(point_scale, point_bias),
                    snapshot=snapshot,
                    state_dict=state_dict,
                )
                preds = [adapted]
                if interval_qhat is not None and math.isfinite(float(interval_qhat)):
                    width = float(max(interval_qhat, 0.0))
                    lower_bounds.extend([float(preds[0]) - width])
                    upper_bounds.extend([float(preds[0]) + width])
            y_true.append(_as_float(target.get("y")))
            y_pred.append(float(preds[0]))
            condition_keys.append(_condition_cluster_key(target))
            tail_pos.append(float(idx / max(len(rows) - 1, 1)))
    return {"y_true": y_true, "y_pred": y_pred, "lower": lower_bounds, "upper": upper_bounds, "condition_key": condition_keys, "tail_pos": tail_pos}


def _evaluate_regime_one_step(
    predictor: Any,
    *,
    records: list[dict[str, object]],
    feature_keys: list[str],
    regime: tuple[float, float, float],
    kind: str,
    snapshot: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
    algo: str | None = None,
    interval_qhat: float | None = None,
    point_scale: float = 1.0,
    point_bias: float = 0.0,
    point_adapter: dict[str, Any] | None = None,
) -> dict[str, float]:
    outputs = _regime_one_step_outputs(
        predictor,
        records=records,
        feature_keys=feature_keys,
        regime=regime,
        kind=kind,
        snapshot=snapshot,
        state_dict=state_dict,
        algo=algo,
        interval_qhat=interval_qhat,
        point_scale=point_scale,
        point_bias=point_bias,
        point_adapter=point_adapter,
    )
    return _metrics_from_outputs(outputs)


def _collect_regime_one_step_pairs(
    *,
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
    records: list[dict[str, object]],
    feature_keys: list[str],
    regime: tuple[float, float, float],
) -> tuple[list[float], list[float]]:
    adaptation_rows = _collect_regime_adaptation_rows(
        snapshot=snapshot,
        state_dict=state_dict,
        records=records,
        regime=regime,
        feature_keys=feature_keys,
    )
    return [float(row["y_true"]) for row in adaptation_rows], [float(row["base_pred"]) for row in adaptation_rows]


def _train_afno_operating_variant(
    *,
    train_records: list[dict[str, object]],
    adapt_records: list[dict[str, object]],
    regime: tuple[float, float, float],
    base_snapshot: dict[str, Any] | None = None,
    base_state_dict: dict[str, Any] | None = None,
    base_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, tuple[float, float]]:
    if isinstance(base_snapshot, dict) and isinstance(base_state_dict, dict) and isinstance(base_meta, dict):
        snapshot = dict(base_snapshot)
        state_dict = dict(base_state_dict)
        meta = dict(base_meta)
    else:
        snapshot, state_dict, meta = _train_afno_variant(
            records_by_series=_group_records(train_records),
            label="afnocg3_v1_exog_w30_f24_log1p_autoexpand",
            target_transform="log1p",
            allow_structure_fallback=True,
        )
    feature_keys = meta.get("feature_keys") if isinstance(meta.get("feature_keys"), list) else []
    adaptation_rows = _collect_regime_adaptation_rows(
        snapshot=snapshot,
        state_dict=state_dict,
        records=adapt_records,
        regime=regime,
        feature_keys=feature_keys,
    )
    y_true = [float(row["y_true"]) for row in adaptation_rows]
    y_pred = [float(row["base_pred"]) for row in adaptation_rows]
    scale, bias = _fit_affine_point_calibrator(y_true, y_pred)
    calibrated_pred = _apply_affine(y_pred, scale=scale, bias=bias)
    adapter: dict[str, Any] | None = None
    selected_pred = calibrated_pred
    candidate_adapter: dict[str, Any] | None = None
    candidate_selection_rmse: float | None = None
    x_windows, adapt_y, base_preds = _build_regime_adaptation_samples(
        snapshot=snapshot,
        state_dict=state_dict,
        records=adapt_records,
        regime=regime,
        feature_keys=feature_keys,
    )
    if adapt_y.size >= 8 and base_preds.size == adapt_y.size:
        residual_adapter = _fit_regime_ridge_adapter(
            x_windows=x_windows,
            y_true=adapt_y,
            base_preds=base_preds,
            feature_keys=feature_keys,
            affine=(scale, bias),
            adaptation_rows=adaptation_rows,
        )
        if residual_adapter is not None:
            design = _ridge_adapter_design(
                x_windows,
                feature_keys=feature_keys,
                selected_keys=[str(key) for key in residual_adapter.get("selected_feature_keys") or [] if isinstance(key, str)],
                base_preds=np.asarray(calibrated_pred, dtype=float),
                use_base_pred=bool(residual_adapter.get("use_base_pred")),
            )
            if bool(residual_adapter.get("standardize_design")) and design.size:
                design_mean = np.asarray(residual_adapter.get("design_mean") or [], dtype=float)
                design_scale = np.asarray(residual_adapter.get("design_scale") or [], dtype=float)
                if design_mean.size == design.shape[1] and design_scale.size == design.shape[1]:
                    safe_scale = np.where(np.abs(design_scale) > 1e-8, design_scale, 1.0)
                    design = (design - design_mean.reshape(1, -1)) / safe_scale.reshape(1, -1)
            adapted_pred = np.asarray(calibrated_pred, dtype=float)
            if design.size:
                trend_signs = _series_trend_signs_from_rows(adaptation_rows, scale=scale, bias=bias, trend_window=int(residual_adapter.get("trend_window") or 6))
                adapted_pred = adapted_pred + _predict_adapter_residuals(design, residual_adapter, trend_signs=trend_signs)
            candidate_adapter = residual_adapter
            candidate_selection_rmse = _rmse(adapt_y.tolist(), adapted_pred.tolist())
    calibration = _recent_regime_affine_calibration(adaptation_rows, scale=scale, bias=bias)
    qhat = float(calibration.get("qhat") or 0.0)
    meta = {
        **meta,
        "selection_rmse": _rmse(y_true, selected_pred),
        "point_calibration": {"scale": scale, "bias": bias},
        "point_adapter": adapter,
        "point_strategy": "affine_only",
        "candidate_point_adapter": candidate_adapter,
        "candidate_selection_rmse": candidate_selection_rmse,
        "interval_calibration_strategy": "recent_regime_affine_only",
        "interval_calibration_count": int(calibration.get("count") or 0),
        "interval_calibration_tail_fraction": float(calibration.get("tail_fraction") or 1.0),
        "interval_calibration_coverage": float(calibration.get("coverage") or float("nan")),
        "interval_calibration_rank_quantile": float(calibration.get("rank_quantile") or 0.9),
        "selection_label": "afnocg3_v1_exog_w30_f24_log1p_autoexpand",
    }
    return snapshot, state_dict, meta, qhat, (scale, bias)


def _distribution_shift(train_records: list[dict[str, object]], backtest_records: list[dict[str, object]], *, feature_keys: list[str]) -> dict[str, Any]:
    train_y = np.asarray([_as_float(row.get("y")) for row in train_records], dtype=float)
    backtest_y = np.asarray([_as_float(row.get("y")) for row in backtest_records], dtype=float)
    feature_rows: list[dict[str, Any]] = []
    for key in feature_keys:
        train_values = np.asarray([_as_float(_as_dict(row.get("x")).get(key)) for row in train_records], dtype=float)
        backtest_values = np.asarray([_as_float(_as_dict(row.get("x")).get(key)) for row in backtest_records], dtype=float)
        ks = float(ks_2samp(train_values, backtest_values).statistic)
        feature_rows.append(
            {
                "feature": key,
                "ks_stat": ks,
                "train_mean": float(train_values.mean()) if train_values.size else None,
                "backtest_mean": float(backtest_values.mean()) if backtest_values.size else None,
            }
        )
    feature_rows.sort(key=lambda row: _as_float(row.get("ks_stat"), 0.0), reverse=True)
    return {
        "target": {
            "train_mean": float(train_y.mean()) if train_y.size else None,
            "backtest_mean": float(backtest_y.mean()) if backtest_y.size else None,
            "ks_stat": float(ks_2samp(train_y, backtest_y).statistic) if train_y.size and backtest_y.size else None,
        },
        "top_shifted_features": feature_rows[:5],
        "feature_count": len(feature_keys),
    }


def _harder_ood_splits(train_records: list[dict[str, object]], backtest_records: list[dict[str, object]]) -> dict[str, dict[str, Any]]:
    train_series = sorted(_group_records(train_records))
    holdout_n = max(2, len(train_series) // 3)
    unit_holdout_series = set(train_series[-holdout_n:])
    unit_train_records = _records_for_series(train_records, set(train_series) - unit_holdout_series)
    unit_eval_records = _records_for_series(train_records, unit_holdout_series)

    full_train_records = build_fd004_payload(split="train", task="train", horizon=15)["records"]
    full_test_records = build_fd004_payload(split="test", task="backtest", horizon=15)["records"]

    regime_counts: dict[tuple[float, float, float], int] = defaultdict(int)
    for record in full_train_records:
        regime_counts[_regime_key(record)] += 1
    ranked_regimes = sorted(regime_counts.items(), key=lambda item: item[1], reverse=True)
    selected_regime = (0.0, 0.0, 0.0)
    min_eval_len = WINDOW + 2
    for regime, _count in ranked_regimes:
        eval_grouped = _group_records(_records_for_regime(full_test_records, regime, include=True))
        if any(len(rows) >= min_eval_len for rows in eval_grouped.values()):
            selected_regime = regime
            break
    if selected_regime == (0.0, 0.0, 0.0) and ranked_regimes:
        selected_regime = ranked_regimes[0][0]
    regime_train_source = _series_subset_for_regime(full_train_records, selected_regime, min_regime_points=WINDOW + 1, max_series=8)
    regime_eval_source = _series_subset_for_regime(full_test_records, selected_regime, min_regime_points=WINDOW + 1, max_series=4)
    regime_train_records = _records_for_regime(regime_train_source, selected_regime, include=False)
    regime_eval_records = regime_eval_source

    return {
        "unit_holdout": {
            "train_records": unit_train_records,
            "eval_records": unit_eval_records,
            "train_units": sorted(set(_group_records(unit_train_records))),
            "eval_units": sorted(set(_group_records(unit_eval_records))),
        },
        "operating_condition_holdout": {
            "train_records": regime_train_records,
            "eval_records": regime_eval_records,
            "adapt_records": regime_train_source,
            "holdout_regime": list(selected_regime),
            "train_count": len(regime_train_records),
            "eval_count": len(_records_for_regime(regime_eval_source, selected_regime, include=True)),
            "eval_mode": "one_step_target_regime",
        },
    }


SOTA_ALGO_KEYS = ["bilstm_rul", "tcn_rul", "transformer_rul"]

SOTA_CACHE_VERSION = 2  # bumped: full-cycle training data + adequate epoch budget


def _sota_asymmetric_training_protocol() -> dict[str, Any]:
    return {
        "name": "sota_asymmetric_rul_v1",
        "optimizer": "adamw",
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "loss": "asymmetric_rul",
        "asym_over_penalty": 2.0,
        "asym_under_penalty": 1.0,
        "asym_max_rul": 125.0,
        "grad_clip_norm": 0.5,
        "patience": 10,
        "min_delta": 1e-4,
        "aux_loss_weight": 0.0,
    }


def _label_to_experiment_key(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return f"fd004_{slug}"


def _sota_cache_dir(label: str) -> Path:
    return ROOT / ".benchmark_cache" / "sota" / label


def _load_cached_sota(label: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    cache_dir = _sota_cache_dir(label)
    meta_path = cache_dir / "meta.json"
    snap_path = cache_dir / "snapshot.json"
    state_path = cache_dir / "state_dict.pt"
    if not (meta_path.exists() and snap_path.exists() and state_path.exists()):
        return None
    try:
        import torch as _torch
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or int(meta.get("cache_version") or 0) != SOTA_CACHE_VERSION:
            return None
        snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
        state_dict = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in _torch.load(state_path, map_location="cpu", weights_only=True).items()}
        return snapshot, state_dict, meta
    except Exception:
        return None


def _save_cached_sota(label: str, *, snapshot: dict[str, Any], state_dict: dict[str, Any], meta: dict[str, Any]) -> None:
    import torch as _torch
    cache_dir = _sota_cache_dir(label)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    tensor_state = {k: _torch.tensor(v) if isinstance(v, list) else v for k, v in state_dict.items()}
    _torch.save(tensor_state, cache_dir / "state_dict.pt")
    cache_meta = {**meta, "cache_version": SOTA_CACHE_VERSION, "training_hours": _benchmark_sota_training_hours(), "max_epochs_env": _benchmark_afno_max_epochs(), "train_profile": "fd004_full_cycles"}
    (cache_dir / "meta.json").write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_sota_variant(
    *,
    algo: str,
    records_by_series: dict[str, list[dict[str, Any]]],
    label: str,
    model_params: dict[str, Any] | None = None,
    training_protocol: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    artifact = train_univariate_torch_forecaster(
        algo=algo,
        records_by_series=records_by_series,
        training_hours=_benchmark_sota_training_hours(),
        context_len=WINDOW,
        max_exogenous_features=FEATURES,
        target_transform="log1p",
        prefer_exogenous=True,
        allow_structure_fallback=False,
        model_params=model_params,
        training_protocol=training_protocol,
        device=None,
    )
    meta: dict[str, Any] = {
        "label": label,
        "algo": algo,
        "input_dim": int(artifact.input_dim),
        "context_len": int(artifact.context_len),
        "feature_source": _as_str(artifact.snapshot.get("feature_source")),
        "target_transform": _as_str(artifact.snapshot.get("target_transform")),
        "structure_mode": _as_str(artifact.snapshot.get("structure_mode")),
        "validation_rmse": _as_float(artifact.snapshot.get("validation_rmse"), float("nan")),
        "training_protocol": _jsonable_model_params(training_protocol),
    }
    return artifact.snapshot, artifact.state_dict, meta


def _sota_benchmark_row(
    *,
    label: str,
    algo: str,
    records: list[dict[str, Any]],
    horizon: int,
    folds: int,
    feature_keys: list[str],
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    qhat = _afno_interval_qhat(snapshot)
    outputs = _standard_eval_outputs(
        object(),
        records=records,
        feature_keys=feature_keys,
        kind="afno",
        snapshot=snapshot,
        state_dict=state_dict,
        algo=algo,
        interval_qhat=qhat,
    )
    metrics = _metrics_from_outputs(outputs)
    row = {
        "model": label,
        "algo": algo,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "nasa_score": metrics["nasa_score"],
        "cov90": metrics["cov90"],
        "width90": metrics["width90"],
        "interval_method": "validation_residual_conformal_90",
        "source": "fd004_benchmark",
        "scope": "standard CMAPSS evaluation: final-cycle RUL per engine",
        "comparability": "same_protocol",
        "variant_family": "sota_comparison",
        **_point_diagnostic_row_fields(metrics),
    }
    if isinstance(snapshot.get("training_protocol"), dict) and snapshot.get("training_protocol"):
        row["training_protocol"] = snapshot.get("training_protocol")
        row["loss"] = snapshot["training_protocol"].get("loss")
    return row


def _train_afno_variant(
    *,
    records_by_series: dict[str, list[dict[str, Any]]],
    label: str,
    target_transform: str,
    allow_structure_fallback: bool,
    model_params: dict[str, Any] | None = None,
    training_protocol: dict[str, Any] | None = None,
    variant_family: str | None = None,
    ablation_stage: str | None = None,
    story_component: str | None = None,
    reuse_temporal_artifact: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reuse_snapshot: dict[str, Any] | None = None
    reuse_state_dict: dict[str, Any] | None = None
    if reuse_temporal_artifact is not None:
        reuse_snapshot, reuse_state_dict, _reuse_meta = reuse_temporal_artifact
    artifact = train_univariate_torch_forecaster(
        algo=AFNO_ALGO,
        records_by_series=records_by_series,
        training_hours=_benchmark_afno_training_hours(),
        context_len=WINDOW,
        max_exogenous_features=FEATURES,
        target_transform=target_transform,
        prefer_exogenous=True,
        allow_structure_fallback=allow_structure_fallback,
        model_params=model_params,
        training_protocol=training_protocol,
        reuse_snapshot=reuse_snapshot,
        reuse_state_dict=reuse_state_dict,
        device=None,
    )
    meta = {
        "label": label,
        "input_dim": int(artifact.input_dim),
        "context_len": int(artifact.context_len),
        "feature_source": _as_str(artifact.snapshot.get("feature_source")),
        "target_transform": _as_str(artifact.snapshot.get("target_transform")),
        "structure_mode": _as_str(artifact.snapshot.get("structure_mode")),
        "validation_rmse": _as_float(artifact.snapshot.get("validation_rmse"), float("nan")),
        "feature_keys": artifact.snapshot.get("feature_keys") if isinstance(artifact.snapshot.get("feature_keys"), list) else [],
        "variant_family": variant_family,
        "ablation_stage": ablation_stage,
        "story_component": story_component,
        "ablation_config": _jsonable_model_params(model_params),
        "training_protocol": _jsonable_model_params(training_protocol),
    }
    return artifact.snapshot, artifact.state_dict, meta


def _afno_interval_qhat(snapshot: dict[str, Any]) -> float:
    residuals = snapshot.get("pooled_residuals") if isinstance(snapshot.get("pooled_residuals"), list) else None
    if residuals:
        return _nearest_rank([abs(_as_float(v)) for v in residuals], 0.9)
    return 0.0


def _evaluate_harder_ood(
    *,
    train_records: list[dict[str, object]],
    backtest_records: list[dict[str, object]],
    horizon: int,
    operating_afno_artifact: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for split_name, split_cfg in _harder_ood_splits(train_records, backtest_records).items():
        split_train = split_cfg.get("train_records") if isinstance(split_cfg.get("train_records"), list) else []
        split_eval = split_cfg.get("eval_records") if isinstance(split_cfg.get("eval_records"), list) else []
        if not split_train or not split_eval:
            continue
        split_feature_keys = _select_feature_keys(split_train)
        if not split_feature_keys:
            continue
        split_rows: list[dict[str, Any]] = []
        split_regime = split_cfg.get("holdout_regime") if isinstance(split_cfg.get("holdout_regime"), list) else None

        gbdt_bundle, gbdt_meta = _fit_hgb(
            split_train,
            feature_keys=split_feature_keys,
            interval_calibration_targets=_standard_eval_target_ruls(split_eval),
        )
        if split_name == "operating_condition_holdout" and split_regime is not None:
            gbdt_outputs = _regime_one_step_outputs(
                gbdt_bundle,
                records=split_eval,
                feature_keys=split_feature_keys,
                regime=(float(split_regime[0]), float(split_regime[1]), float(split_regime[2])),
                kind="gbdt",
            )
        else:
            gbdt_outputs = _standard_eval_outputs(
                gbdt_bundle,
                records=split_eval,
                feature_keys=split_feature_keys,
                kind="gbdt",
            )
        gbdt_metrics = _metrics_from_outputs(gbdt_outputs)
        split_rows.append(
            {
                "model": "GBDT_w30_f24",
                "rmse": gbdt_metrics["rmse"],
                "mae": gbdt_metrics["mae"],
                "nasa_score": gbdt_metrics["nasa_score"],
                "cov90": gbdt_metrics["cov90"],
                "width90": gbdt_metrics["width90"],
                "interval_method": gbdt_meta.get("interval_method"),
            }
        )

        if split_name == "operating_condition_holdout" and split_regime is not None:
            base_snapshot: dict[str, Any] | None = None
            base_state_dict: dict[str, Any] | None = None
            base_meta: dict[str, Any] | None = None
            if operating_afno_artifact is not None:
                base_snapshot, base_state_dict, base_meta = operating_afno_artifact
            snapshot, state_dict, afno_meta, afno_qhat, point_calibrator = _train_afno_operating_variant(
                train_records=split_train,
                adapt_records=split_cfg.get("adapt_records") if isinstance(split_cfg.get("adapt_records"), list) else split_eval,
                regime=(float(split_regime[0]), float(split_regime[1]), float(split_regime[2])),
                base_snapshot=base_snapshot,
                base_state_dict=base_state_dict,
                base_meta=base_meta,
            )
            afno_outputs = _regime_one_step_outputs(
                object(),
                records=split_eval,
                feature_keys=split_feature_keys,
                regime=(float(split_regime[0]), float(split_regime[1]), float(split_regime[2])),
                kind="afno",
                snapshot=snapshot,
                state_dict=state_dict,
                algo=AFNO_ALGO,
                interval_qhat=afno_qhat,
                point_scale=point_calibrator[0],
                point_bias=point_calibrator[1],
                point_adapter=afno_meta.get("point_adapter") if isinstance(afno_meta.get("point_adapter"), dict) else None,
            )
        else:
            snapshot, state_dict, afno_meta = _train_afno_variant(
                records_by_series=_group_records(split_train),
                label="afnocg3_v1_exog_w30_f24_log1p_autoexpand",
                target_transform="log1p",
                allow_structure_fallback=True,
            )
            afno_outputs = _standard_eval_outputs(
                object(),
                records=split_eval,
                feature_keys=split_feature_keys,
                kind="afno",
                snapshot=snapshot,
                state_dict=state_dict,
                algo=AFNO_ALGO,
                interval_qhat=_afno_interval_qhat(snapshot),
            )
        afno_metrics = _metrics_from_outputs(afno_outputs)
        split_rows.append(
            {
                "model": "afnocg3_v1_exog_w30_f24_log1p_autoexpand",
                "rmse": afno_metrics["rmse"],
                "mae": afno_metrics["mae"],
                "nasa_score": afno_metrics["nasa_score"],
                "cov90": afno_metrics["cov90"],
                "width90": afno_metrics["width90"],
                "interval_method": "validation_residual_conformal_90",
                "structure_mode": afno_meta.get("structure_mode"),
                "selection_rmse": afno_meta.get("selection_rmse"),
                "point_strategy": afno_meta.get("point_strategy"),
                "interval_calibration_strategy": afno_meta.get("interval_calibration_strategy"),
                "interval_calibration_count": afno_meta.get("interval_calibration_count"),
                "interval_calibration_tail_fraction": afno_meta.get("interval_calibration_tail_fraction"),
                "interval_calibration_coverage": afno_meta.get("interval_calibration_coverage"),
                "interval_calibration_rank_quantile": afno_meta.get("interval_calibration_rank_quantile"),
                "point_adapter": afno_meta.get("point_adapter", {}).get("type") if isinstance(afno_meta.get("point_adapter"), dict) else None,
            }
        )

        if gbdt_outputs["y_true"] and len(gbdt_outputs["y_true"]) == len(afno_outputs["y_true"]):
            hybrid_weight = _hybrid_weight_from_validation(gbdt_meta, afno_meta)
            hybrid_outputs = _blend_backtest_outputs(gbdt_outputs, afno_outputs, afno_weight=hybrid_weight)
            hybrid_metrics = _metrics_from_outputs(hybrid_outputs)
            split_rows.append(
                {
                    "model": "GBDT_AFNO_hybrid_w30_f24",
                    "rmse": hybrid_metrics["rmse"],
                    "mae": hybrid_metrics["mae"],
                    "nasa_score": hybrid_metrics["nasa_score"],
                    "cov90": hybrid_metrics["cov90"],
                    "width90": hybrid_metrics["width90"],
                    "interval_method": "blended_quantile_and_conformal_90",
                    "blend_weight_afno": hybrid_weight,
                    "blend_weight_gbdt": 1.0 - hybrid_weight,
                    "blend_basis": "inverse_validation_rmse",
                }
            )
            if split_name == "operating_condition_holdout" and split_regime is not None:
                gate_records, interval_records = _gate_interval_holdout_split(
                    split_cfg.get("adapt_records") if isinstance(split_cfg.get("adapt_records"), list) else split_eval
                )
                calib_gbdt_outputs = _regime_one_step_outputs(
                    gbdt_bundle,
                    records=gate_records,
                    feature_keys=split_feature_keys,
                    regime=(float(split_regime[0]), float(split_regime[1]), float(split_regime[2])),
                    kind="gbdt",
                )
                calib_afno_outputs = _regime_one_step_outputs(
                    object(),
                    records=gate_records,
                    feature_keys=split_feature_keys,
                    regime=(float(split_regime[0]), float(split_regime[1]), float(split_regime[2])),
                    kind="afno",
                    snapshot=snapshot,
                    state_dict=state_dict,
                    algo=AFNO_ALGO,
                    interval_qhat=afno_qhat,
                    point_scale=point_calibrator[0],
                    point_bias=point_calibrator[1],
                    point_adapter=afno_meta.get("point_adapter") if isinstance(afno_meta.get("point_adapter"), dict) else None,
                )
                interval_gbdt_outputs = _regime_one_step_outputs(
                    gbdt_bundle,
                    records=interval_records,
                    feature_keys=split_feature_keys,
                    regime=(float(split_regime[0]), float(split_regime[1]), float(split_regime[2])),
                    kind="gbdt",
                )
                interval_afno_outputs = _regime_one_step_outputs(
                    object(),
                    records=interval_records,
                    feature_keys=split_feature_keys,
                    regime=(float(split_regime[0]), float(split_regime[1]), float(split_regime[2])),
                    kind="afno",
                    snapshot=snapshot,
                    state_dict=state_dict,
                    algo=AFNO_ALGO,
                    interval_qhat=afno_qhat,
                    point_scale=point_calibrator[0],
                    point_bias=point_calibrator[1],
                    point_adapter=afno_meta.get("point_adapter") if isinstance(afno_meta.get("point_adapter"), dict) else None,
                )
            else:
                _fit_records, calib_records = _gbdt_calibration_split(split_train)
                gate_records, interval_records = _gate_interval_holdout_split(calib_records)
                calib_gbdt_outputs = _backtest_outputs(gbdt_bundle, records=gate_records, horizon=horizon, folds=1, feature_keys=split_feature_keys, kind="gbdt")
                calib_afno_outputs = _backtest_outputs(object(), records=gate_records, horizon=horizon, folds=1, feature_keys=split_feature_keys, kind="afno", snapshot=snapshot, state_dict=state_dict, algo=AFNO_ALGO, interval_qhat=_afno_interval_qhat(snapshot))
                interval_gbdt_outputs = _backtest_outputs(gbdt_bundle, records=interval_records, horizon=horizon, folds=1, feature_keys=split_feature_keys, kind="gbdt")
                interval_afno_outputs = _backtest_outputs(object(), records=interval_records, horizon=horizon, folds=1, feature_keys=split_feature_keys, kind="afno", snapshot=snapshot, state_dict=state_dict, algo=AFNO_ALGO, interval_qhat=_afno_interval_qhat(snapshot))
            for variant in _soft_gate_variant_specs():
                variant_result = _evaluate_soft_gate_variant(
                    label=str(variant["label"]),
                    mode=str(variant["mode"]),
                    gbdt_bundle=gbdt_bundle,
                    feature_keys=split_feature_keys,
                    explanation_records=split_eval,
                    afno_snapshot=snapshot,
                    afno_state_dict=state_dict,
                    afno_algo=AFNO_ALGO,
                    afno_point_scale=point_calibrator[0] if split_name == "operating_condition_holdout" and split_regime is not None else 1.0,
                    afno_point_bias=point_calibrator[1] if split_name == "operating_condition_holdout" and split_regime is not None else 0.0,
                    afno_point_adapter=afno_meta.get("point_adapter") if isinstance(afno_meta.get("point_adapter"), dict) else None,
                    explanation_regime=(float(split_regime[0]), float(split_regime[1]), float(split_regime[2])) if split_name == "operating_condition_holdout" and split_regime is not None else None,
                    eval_gbdt_outputs=gbdt_outputs,
                    eval_afno_outputs=afno_outputs,
                    calib_gbdt_outputs=calib_gbdt_outputs,
                    calib_afno_outputs=calib_afno_outputs,
                    interval_gbdt_outputs=interval_gbdt_outputs,
                    interval_afno_outputs=interval_afno_outputs,
                    target_cov=0.93,
                    scope_name=split_name,
                    use_condition_clusters=True,
                )
                split_rows.append(variant_result["row"])

        results[split_name] = {
            **{key: value for key, value in split_cfg.items() if key not in {"train_records", "eval_records"}},
            "feature_count": len(split_feature_keys),
            "models": split_rows,
        }
    return results


def main() -> int:
    os.environ.setdefault("RULFM_FORECASTING_API_KEY", "dev-key")
    # Read benchmark payloads from the shared CMAPSS builder directly so this
    # script stays decoupled from FastAPI's HTTP test client and app startup.
    #
    # GBDT training uses window_size=None (all cycles per engine) so that the
    # model sees RUL labels up to MAX_RUL=125, matching the test terminal RUL
    # distribution.  The fd004_train_multiunit profile uses window_size=90 which
    # truncates training labels to y∈[0,89] and causes systematic underprediction
    # of the 44.7% of test engines whose terminal RUL exceeds 89.
    gbdt_train_records = build_fd004_payload(
        split="train", task="train", horizon=15, window_size=None,
    )["records"]
    backtest_payload = build_fd004_profile_payload("fd004_backtest_multiunit")
    # train_records retains window_size=90 for UI / profile APIs; GBDT uses
    # gbdt_train_records (full cycles) for fitting and feature selection.
    train_payload = build_fd004_profile_payload("fd004_train_multiunit")
    train_records = train_payload["records"]
    backtest_records = backtest_payload["records"]
    horizon = int(backtest_payload.get("horizon") or 5)
    folds = 2
    feature_keys = _select_feature_keys(gbdt_train_records)
    assert len(feature_keys) > 0, {"expected_features": FEATURES, "selected": feature_keys}
    train_by_series = _group_records(train_records)
    # SOTA torch models (BiLSTM / TCN / Transformer) also need full-cycle data
    # for the same reason as GBDT: window_size=90 truncates labels to y∈[0,89],
    # causing systematic underprediction of engines with terminal RUL > 89.
    sota_train_by_series = _group_records(gbdt_train_records)

    experiments: dict[str, dict[str, Any]] = {}
    rows: list[JsonDict] = []
    integrated_afno_artifact: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    temporal_afno_artifacts: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    soft_gate_main_rows: list[dict[str, Any]] = []
    benchmark_notes: list[str] = []

    with start_run(
        run_name=f"fd004-benchmark-w{WINDOW}-f{FEATURES}",
        tags={"flow": "benchmark", "dataset": "cmapss-fd004", "window": str(WINDOW), "features": str(FEATURES)},
    ):
        log_params(
            {
                "window": WINDOW,
                "features": FEATURES,
                "horizon": horizon,
                "folds": folds,
                "models": "GBDT,SOTA" if not _benchmark_should_run_afno() else "GBDT,SOTA,AFNOcG",
                "afno_enabled": _benchmark_should_run_afno(),
            }
        )

        gbdt_model, gbdt_meta = _fit_hgb(
            gbdt_train_records,
            feature_keys=feature_keys,
            interval_calibration_targets=_standard_eval_target_ruls(backtest_records),
        )
        gbdt_outputs = _standard_eval_outputs(gbdt_model, records=backtest_records, feature_keys=feature_keys, kind="gbdt")
        gbdt_metrics = _metrics_from_outputs(gbdt_outputs)
        calibration_protocol = str(gbdt_meta.get("interval_calibration_protocol") or "").strip()
        if calibration_protocol:
            benchmark_notes.append(
                f"GBDT interval calibration uses {calibration_protocol} on held-out training units instead of all sliding-window rows."
            )
        experiments["fd004_gbdt_w30_f24"] = {**gbdt_meta, **{f"backtest_{k}": v for k, v in gbdt_metrics.items()}}
        rows.append({"model": "GBDT_w30_f24", "rmse": gbdt_metrics["rmse"], "mae": gbdt_metrics["mae"], "nasa_score": gbdt_metrics["nasa_score"], "cov90": gbdt_metrics["cov90"], "width90": gbdt_metrics["width90"], "interval_method": gbdt_meta.get("interval_method"), "source": "fd004_benchmark", "scope": "standard CMAPSS evaluation: final-cycle RUL per engine", "comparability": "same_protocol", **_point_diagnostic_row_fields(gbdt_metrics)})
        log_metrics({"gbdt.rmse": gbdt_metrics["rmse"], "gbdt.nasa_score": gbdt_metrics["nasa_score"], "gbdt.cov90": gbdt_metrics["cov90"], "gbdt.width90": gbdt_metrics["width90"]})

        gbdt_phase = "gbdt-only" if _benchmark_stage() == "gbdt-only" else "gbdt-baseline"
        gbdt_summary = _build_benchmark_summary(
            feature_keys=feature_keys,
            horizon=horizon,
            folds=folds,
            rows=list(rows),
            experiments=dict(experiments),
            benchmark_notes=list(benchmark_notes),
            phase=gbdt_phase,
        )
        gbdt_data_path, gbdt_csv_path = _write_benchmark_outputs(gbdt_summary, rows)
        log_dict_artifact("fd004_benchmark_summary.partial.json", gbdt_summary)
        log_artifact(gbdt_data_path)
        log_artifact(gbdt_csv_path)
        print(f"wrote partial benchmark snapshot: {gbdt_data_path}")
        if _benchmark_stage() == "gbdt-only":
            print(json.dumps(gbdt_summary, ensure_ascii=False, indent=2))
            return 0

        # SOTA comparison: BiLSTM, TCN, TransformerRUL
        # All use a constrained protocol (AdamW + Huber + strong grad clip) for stability on FD004.
        _sota_protocol = {
            "name": "sota_constrained_v1",
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "loss": "huber",
            "huber_delta": 1.0,
            "grad_clip_norm": 0.5,
            "patience": 10,
            "min_delta": 1e-4,
            "aux_loss_weight": 0.0,
        }
        _sota_specs = [
            {"algo": "bilstm_rul",      "label": "BiLSTM_w30_f24",      "model_params": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2}},
            {"algo": "tcn_rul",         "label": "TCN_w30_f24",          "model_params": {"hidden_dim": 64, "num_layers": 4, "kernel_size": 3, "dropout": 0.2}},
            {"algo": "transformer_rul", "label": "TransformerRUL_w30_f24", "model_params": {"model_dim": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 256, "dropout": 0.1}},
            {"algo": "bilstm_rul",      "label": "BiLSTM_w30_f24_asym", "model_params": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2}, "training_protocol": _sota_asymmetric_training_protocol()},
        ]
        try:
            for sota_spec in _sota_specs:
                _sota_algo = str(sota_spec["algo"])
                _sota_label = str(sota_spec["label"])
                _sota_params = dict(sota_spec.get("model_params") or {})
                _sota_cached = _load_cached_sota(_sota_label)
                if _sota_cached is not None:
                    _sota_snap, _sota_sd, _sota_meta = _sota_cached
                else:
                    _sota_snap, _sota_sd, _sota_meta = _train_sota_variant(
                        algo=_sota_algo,
                        records_by_series=sota_train_by_series,
                        label=_sota_label,
                        model_params=_sota_params,
                        training_protocol=(
                            sota_spec.get("training_protocol")
                            if isinstance(sota_spec.get("training_protocol"), dict)
                            else _sota_protocol
                        ),
                    )
                    _save_cached_sota(
                        _sota_label,
                        snapshot=_sota_snap,
                        state_dict=_sota_sd,
                        meta=_sota_meta,
                    )
                _sota_row = _sota_benchmark_row(
                    label=_sota_label,
                    algo=_sota_algo,
                    records=backtest_records,
                    horizon=horizon,
                    folds=folds,
                    feature_keys=feature_keys,
                    snapshot=_sota_snap,
                    state_dict=_sota_sd,
                )
                experiments[_label_to_experiment_key(_sota_label)] = {
                    **_sota_meta,
                    **{f"backtest_{k}": v for k, v in _point_diagnostic_row_fields(_sota_row).items()},
                    **{
                        f"backtest_{k}": _sota_row[k]
                        for k in ("rmse", "mae", "nasa_score", "cov90", "width90")
                    },
                }
                rows.append(_sota_row)
                log_metrics(
                    {
                        f"{_sota_algo}.rmse": _sota_row["rmse"],
                        f"{_sota_algo}.nasa_score": _sota_row["nasa_score"],
                    }
                )
        except ModuleNotFoundError as exc:
            benchmark_notes.append(f"Torch-based SOTA benchmarks skipped: {exc}")

        if _benchmark_should_run_afno():
            afno_specs = [
                {
                    "label": "afnocg3_v1_exog_w30_f24",
                    "target_transform": "none",
                    "allow_structure_fallback": False,
                    "base_label": None,
                    "model_params": None,
                    "variant_family": "afno_baseline",
                    "ablation_stage": None,
                    "story_component": None,
                },
                {
                    "label": "afnocg3_v1_exog_w30_f24_log1p",
                    "target_transform": "log1p",
                    "allow_structure_fallback": False,
                    "base_label": None,
                    "model_params": None,
                    "variant_family": "afno_baseline",
                    "ablation_stage": None,
                    "story_component": None,
                },
                {
                    "label": "afnocg3_v1_exog_w30_f24_autoexpand",
                    "target_transform": "none",
                    "allow_structure_fallback": True,
                    "base_label": "afnocg3_v1_exog_w30_f24",
                    "model_params": None,
                    "variant_family": "afno_baseline",
                    "ablation_stage": None,
                    "story_component": None,
                },
                {
                    "label": "afnocg3_v1_exog_w30_f24_log1p_autoexpand",
                    "target_transform": "log1p",
                    "allow_structure_fallback": True,
                    "base_label": "afnocg3_v1_exog_w30_f24_log1p",
                    "model_params": None,
                    "variant_family": "afno_baseline",
                    "ablation_stage": None,
                    "story_component": None,
                },
                {
                    "label": "afnocg3_v1_exog_w30_f24_log1p_autoexpand_asym",
                    "target_transform": "log1p",
                    "allow_structure_fallback": True,
                    "base_label": None,
                    "model_params": None,
                    "training_protocol": _afno_asymmetric_training_protocol(),
                    "variant_family": "afno_asymmetric_loss",
                    "ablation_stage": "asymmetric_rul_loss",
                    "story_component": "low_rul_overprediction_control",
                },
                *_afno_component_ablation_specs(),
            ]
            try:
                for spec in afno_specs:
                    label = str(spec["label"])
                    target_transform = str(spec["target_transform"])
                    autoexpand = bool(spec["allow_structure_fallback"])
                    base_label = (
                        str(spec["base_label"]) if spec.get("base_label") is not None else None
                    )
                    with start_run(
                        run_name=label,
                        nested=True,
                        tags={"flow": "benchmark-train", "model_family": "AFNOcG"},
                    ):
                        cached_artifact = _load_cached_benchmark_afno(label)
                        if cached_artifact is not None:
                            snapshot, state_dict, afno_meta = cached_artifact
                        else:
                            reuse_artifact = (
                                temporal_afno_artifacts.get(str(base_label))
                                if base_label is not None
                                else None
                            )
                            snapshot, state_dict, afno_meta = _train_afno_variant(
                                records_by_series=train_by_series,
                                label=label,
                                target_transform=target_transform,
                                allow_structure_fallback=autoexpand,
                                model_params=(
                                    spec.get("model_params")
                                    if isinstance(spec.get("model_params"), dict)
                                    else None
                                ),
                                training_protocol=(
                                    spec.get("training_protocol")
                                    if isinstance(spec.get("training_protocol"), dict)
                                    else None
                                ),
                                variant_family=(
                                    str(spec.get("variant_family"))
                                    if spec.get("variant_family") is not None
                                    else None
                                ),
                                ablation_stage=(
                                    str(spec.get("ablation_stage"))
                                    if spec.get("ablation_stage") is not None
                                    else None
                                ),
                                story_component=(
                                    str(spec.get("story_component"))
                                    if spec.get("story_component") is not None
                                    else None
                                ),
                                reuse_temporal_artifact=reuse_artifact,
                            )
                            _save_cached_benchmark_afno(
                                label,
                                snapshot=snapshot,
                                state_dict=state_dict,
                                meta=afno_meta,
                            )
                        outputs = _standard_eval_outputs(
                            object(),
                            records=backtest_records,
                            feature_keys=feature_keys,
                            kind="afno",
                            snapshot=snapshot,
                            state_dict=state_dict,
                            algo=AFNO_ALGO,
                            interval_qhat=_afno_interval_qhat(snapshot),
                        )
                        metrics = _metrics_from_outputs(outputs)
                        experiments[label] = {
                            **afno_meta,
                            **{f"backtest_{k}": v for k, v in metrics.items()},
                        }
                        rows.append(_afno_benchmark_row(label=label, metrics=metrics, afno_meta=afno_meta))
                        log_params(
                            {
                                "target_transform": target_transform,
                                "structure_autoexpand": autoexpand,
                                "structure_mode": afno_meta["structure_mode"],
                            }
                        )
                        log_metrics(
                            {
                                "rmse": metrics["rmse"],
                                "nasa_score": metrics["nasa_score"],
                                "cov90": metrics["cov90"],
                                "width90": metrics["width90"],
                            }
                        )
                        log_dict_artifact(
                            f"{label}.json",
                            {"snapshot": snapshot, "meta": afno_meta, "metrics": metrics},
                        )
                        if not autoexpand:
                            temporal_afno_artifacts[label] = (snapshot, state_dict, afno_meta)
                        if label == "afnocg3_v1_exog_w30_f24_log1p_autoexpand":
                            integrated_afno_artifact = (snapshot, state_dict, afno_meta)
                            hybrid_weight = _hybrid_weight_from_validation(gbdt_meta, afno_meta)
                            hybrid_outputs = _blend_backtest_outputs(
                                gbdt_outputs,
                                outputs,
                                afno_weight=hybrid_weight,
                            )
                            hybrid_metrics = _metrics_from_outputs(hybrid_outputs)
                            hybrid_label = "gbdt_afno_hybrid_w30_f24"
                            experiments[hybrid_label] = {
                                "blend_weight_afno": hybrid_weight,
                                "blend_weight_gbdt": 1.0 - hybrid_weight,
                                "blend_basis": "inverse_validation_rmse",
                                "gbdt_validation_rmse": gbdt_meta.get("validation_rmse"),
                                "afno_validation_rmse": afno_meta.get("validation_rmse"),
                                **{f"backtest_{k}": v for k, v in hybrid_metrics.items()},
                            }
                            rows.append(
                                {
                                    "model": "GBDT_AFNO_hybrid_w30_f24",
                                    "rmse": hybrid_metrics["rmse"],
                                    "mae": hybrid_metrics["mae"],
                                    "nasa_score": hybrid_metrics["nasa_score"],
                                    "cov90": hybrid_metrics["cov90"],
                                    "width90": hybrid_metrics["width90"],
                                    "interval_method": "blended_quantile_and_conformal_90",
                                    "source": "fd004_benchmark",
                                    "scope": "standard CMAPSS evaluation: final-cycle RUL per engine",
                                    "comparability": "same_protocol",
                                    **_point_diagnostic_row_fields(hybrid_metrics),
                                }
                            )
                            log_dict_artifact(
                                f"{hybrid_label}.json",
                                {"meta": experiments[hybrid_label], "metrics": hybrid_metrics},
                            )

                            _fit_records, calib_records = _gbdt_calibration_split(train_records)
                            gate_records, interval_records = _gate_interval_holdout_split(
                                calib_records
                            )
                            calib_gbdt_outputs = _backtest_outputs(
                                gbdt_model,
                                records=gate_records,
                                horizon=horizon,
                                folds=1,
                                feature_keys=feature_keys,
                                kind="gbdt",
                            )
                            calib_afno_outputs = _backtest_outputs(
                                object(),
                                records=gate_records,
                                horizon=horizon,
                                folds=1,
                                feature_keys=feature_keys,
                                kind="afno",
                                snapshot=snapshot,
                                state_dict=state_dict,
                                algo=AFNO_ALGO,
                                interval_qhat=_afno_interval_qhat(snapshot),
                            )
                            interval_gbdt_outputs = _backtest_outputs(
                                gbdt_model,
                                records=interval_records,
                                horizon=horizon,
                                folds=1,
                                feature_keys=feature_keys,
                                kind="gbdt",
                            )
                            interval_afno_outputs = _backtest_outputs(
                                object(),
                                records=interval_records,
                                horizon=horizon,
                                folds=1,
                                feature_keys=feature_keys,
                                kind="afno",
                                snapshot=snapshot,
                                state_dict=state_dict,
                                algo=AFNO_ALGO,
                                interval_qhat=_afno_interval_qhat(snapshot),
                            )
                            for variant in _soft_gate_variant_specs():
                                variant_result = _evaluate_soft_gate_variant(
                                    label=str(variant["label"]),
                                    mode=str(variant["mode"]),
                                    gbdt_bundle=gbdt_model,
                                    feature_keys=feature_keys,
                                    explanation_records=backtest_records,
                                    afno_snapshot=snapshot,
                                    afno_state_dict=state_dict,
                                    afno_algo=AFNO_ALGO,
                                    afno_point_scale=1.0,
                                    afno_point_bias=0.0,
                                    afno_point_adapter=(
                                        afno_meta.get("point_adapter")
                                        if isinstance(afno_meta.get("point_adapter"), dict)
                                        else None
                                    ),
                                    explanation_regime=None,
                                    eval_gbdt_outputs=gbdt_outputs,
                                    eval_afno_outputs=outputs,
                                    calib_gbdt_outputs=calib_gbdt_outputs,
                                    calib_afno_outputs=calib_afno_outputs,
                                    interval_gbdt_outputs=interval_gbdt_outputs,
                                    interval_afno_outputs=interval_afno_outputs,
                                    target_cov=0.93,
                                    scope_name="main_rolling_backtest",
                                    use_condition_clusters=True,
                                )
                                gated_label = str(variant_result["label"]).lower()
                                experiments[gated_label] = {
                                    **variant_result["experiment"],
                                    "gbdt_validation_rmse": gbdt_meta.get("validation_rmse"),
                                    "afno_validation_rmse": afno_meta.get("validation_rmse"),
                                }
                                row = {
                                    **variant_result["row"],
                                    "source": "fd004_benchmark",
                                    "scope": "standard CMAPSS evaluation: final-cycle RUL per engine",
                                    "comparability": "same_protocol",
                                }
                                rows.append(row)
                                soft_gate_main_rows.append(row)
                                log_dict_artifact(
                                    f"{gated_label}.json",
                                    {
                                        "meta": experiments[gated_label],
                                        "metrics": variant_result["metrics"],
                                    },
                                )
            except ModuleNotFoundError as exc:
                benchmark_notes.append(f"Torch-based AFNO benchmarks skipped: {exc}")
        else:
            benchmark_notes.append(
                "Torch-based AFNO benchmarks skipped because RULFM_BENCHMARK_ENABLE_AFNO is not enabled."
            )

        legacy_csv = ROOT / "docs" / "references" / "fd004_legacy_extract" / "clusterwise_rul_evaluation_004.csv"
        legacy_rmse, legacy_count = _weighted_rmse_from_cluster_csv(legacy_csv)
        rows.append({"model": "Legacy PRSD_HLIL_bOCNet", "rmse": legacy_rmse, "mae": None, "nasa_score": None, "cov90": None, "width90": None, "interval_method": None, "source": "legacy_extract_clusterwise", "scope": f"legacy test evaluation aggregated from {legacy_count} rows", "comparability": "different_protocol"})

        rows.sort(key=lambda row: (_as_float(row.get("nasa_score"), float("inf")), _as_float(row.get("rmse"), float("inf"))))
        shift = _distribution_shift(train_records, backtest_records, feature_keys=feature_keys)
        harder_ood = _evaluate_harder_ood(
            train_records=train_records,
            backtest_records=backtest_records,
            horizon=horizon,
            operating_afno_artifact=integrated_afno_artifact,
        )
        operating_rows = harder_ood.get("operating_condition_holdout", {}).get("models") if isinstance(harder_ood.get("operating_condition_holdout"), dict) else []
        soft_gate_pareto = _soft_gate_pareto_summary(
            main_rows=[row for row in rows if str(row.get("model") or "").startswith("GBDT_AFNO_gated") or str(row.get("model") or "") in {"GBDT_w30_f24", "GBDT_AFNO_hybrid_w30_f24"}],
            operating_rows=operating_rows if isinstance(operating_rows, list) else [],
            target_cov=0.93,
        )
        summary = _build_benchmark_summary(
            feature_keys=feature_keys,
            horizon=horizon,
            folds=folds,
            rows=rows,
            experiments=experiments,
            benchmark_notes=benchmark_notes,
            diagnostics={
                "distribution_shift": shift,
                "uncertainty": [
                    {
                        "model": row.get("model"),
                        "cov90": row.get("cov90"),
                        "width90": row.get("width90"),
                        "interval_method": row.get("interval_method"),
                    }
                    for row in rows
                    if row.get("comparability") == "same_protocol"
                ],
                "xai_stability": [
                    {
                        "model": row.get("model"),
                        "gate_terms_rank_consistency": ((row.get("xai_summary") or {}).get("gate_terms_rank_consistency") if isinstance(row.get("xai_summary"), dict) else None),
                        "afno_occlusion_topk_jaccard": ((row.get("xai_summary") or {}).get("afno_occlusion_topk_jaccard") if isinstance(row.get("xai_summary"), dict) else None),
                    }
                    for row in rows
                    if row.get("comparability") == "same_protocol"
                ],
                "architecture_ablations": [
                    {
                        "model": row.get("model"),
                        "ablation_stage": row.get("ablation_stage"),
                        "story_component": row.get("story_component"),
                        "rmse": row.get("rmse"),
                        "nasa_score": row.get("nasa_score"),
                    }
                    for row in rows
                    if row.get("variant_family") == "afno_component_ablation"
                ],
                "uncertainty_quality": [
                    {
                        "model": experiment.get("label"),
                        **(experiment.get("uncertainty_quality") if isinstance(experiment.get("uncertainty_quality"), dict) else {}),
                    }
                    for experiment in experiments.values()
                    if str(experiment.get("label") or "").startswith("GBDT_AFNO_gated")
                ],
                "harder_ood": harder_ood,
                "soft_gate_pareto": soft_gate_pareto,
                "evaluation": {
                    "rmse": "sqrt(mean((y_hat - y)^2))",
                    "nasa_score": "error = y_hat - y; exp(-error/13)-1 for error<0 else exp(error/10)-1",
                    "cov90": "mean(lower_90 <= y <= upper_90)",
                    "width90": "mean(upper_90 - lower_90)",
                },
            },
        )

        data_path, csv_path = _write_benchmark_outputs(summary, rows)

        log_dict_artifact("fd004_benchmark_summary.json", summary)
        log_artifact(data_path)
        log_artifact(csv_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"wrote: {data_path}")
        print(f"wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
