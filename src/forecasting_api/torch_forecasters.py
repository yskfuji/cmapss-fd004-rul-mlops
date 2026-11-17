from __future__ import annotations

import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class TorchTrainedArtifact:
    algo: str
    input_dim: int
    context_len: int
    pooled_residuals: list[float]
    snapshot: dict[str, Any]
    state_dict: dict[str, Any]


def _choose_device(device: str | None = None, *, algo: str | None = None) -> str:
    if device:
        return str(device)

    env_device = os.getenv("RULFM_FORECASTING_TORCH_DEVICE", "").strip()
    if env_device:
        return env_device

    import torch

    algo_key = str(algo or "").strip().lower()
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if algo_key in {"afnocg3", "afnocg3_v1"}:
                return "cpu"
            return "mps"
    except Exception:
        pass
    return "cpu"


def _infer_context_len(n: int, requested: int | None = None) -> int:
    if requested is not None and requested >= 1:
        return max(1, min(int(requested), max(1, n - 1))) if n > 1 else 1
    if n <= 3:
        return 1
    if n <= 8:
        return 3
    if n <= 20:
        return 7
    return 14


def _record_y(record: dict[str, Any]) -> float:
    value = record.get("y")
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _record_x(record: dict[str, Any]) -> dict[str, float]:
    raw = record.get("x")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, int | float) and math.isfinite(float(value)):
            out[str(key)] = float(value)
    return out


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: str(row.get("timestamp") or ""))


def _select_feature_keys(
    records_by_series: dict[str, list[dict[str, Any]]], *, max_features: int
) -> list[str]:
    keys: set[str] = set()
    for rows in records_by_series.values():
        for row in rows:
            for key in _record_x(row):
                if key == "cycle":
                    continue
                keys.add(key)
    return sorted(keys)[: max(0, int(max_features))]


def _rows_to_feature_matrix(
    rows: list[dict[str, Any]], *, feature_keys: list[str], feature_source: str
) -> np.ndarray:
    if feature_source == "x" and feature_keys:
        mat = [[_record_x(row).get(key, 0.0) for key in feature_keys] for row in rows]
        return np.asarray(mat, dtype=float)
    return np.asarray([[_record_y(row)] for row in rows], dtype=float)


def _target_transform_name(raw: str | None) -> str:
    value = str(raw or "none").strip().lower()
    return "log1p" if value == "log1p" else "none"


def _missing_model_registry(exc: Exception) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        return str(getattr(exc, "name", "")) == "src.models.registry"
    return False


def _transform_target(values: np.ndarray, *, target_transform: str) -> np.ndarray:
    if target_transform == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    return values


def _inverse_target(values: np.ndarray, *, target_transform: str) -> np.ndarray:
    if target_transform == "log1p":
        return np.maximum(np.expm1(values), 0.0)
    return values


def _split_sequence_indices(total: int, *, min_valid: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if total <= min_valid + 8:
        idx = np.arange(total, dtype=int)
        return idx, np.zeros((0,), dtype=int)
    valid_n = max(min_valid, min(32, total // 5))
    train_end = max(1, total - valid_n)
    return np.arange(train_end, dtype=int), np.arange(train_end, total, dtype=int)


def _build_supervised_sequences(
    records_by_series: dict[str, list[dict[str, Any]]],
    *,
    context_len: int,
    feature_keys: list[str],
    feature_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for rows in records_by_series.values():
        ordered = _sorted_records(rows)
        if len(ordered) <= context_len:
            continue
        for idx in range(context_len, len(ordered)):
            context_rows = ordered[idx - context_len : idx]
            xs.append(
                _rows_to_feature_matrix(
                    context_rows, feature_keys=feature_keys, feature_source=feature_source
                )
            )
            ys.append(_record_y(ordered[idx]))
    if not xs:
        return np.zeros((0, 0, 0), dtype=float), np.zeros((0,), dtype=float)
    return np.stack(xs).astype(float), np.asarray(ys, dtype=float)


def _predict_torch_tensor(model: Any, x_tensor: Any) -> Any:
    out = model(x_tensor)
    y_pred = out[0] if isinstance(out, tuple | list) else out
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(-1)
    return y_pred


def _fit_flat_ridge(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    target_transform: str,
) -> tuple[dict[str, Any], float]:
    estimator = Ridge(alpha=1.0)
    estimator.fit(
        x_train.reshape(x_train.shape[0], -1),
        _transform_target(y_train, target_transform=target_transform),
    )
    score = float("inf")
    if x_valid.size > 0 and y_valid.size > 0:
        pred = estimator.predict(x_valid.reshape(x_valid.shape[0], -1))
        pred_inv = _inverse_target(np.asarray(pred, dtype=float), target_transform=target_transform)
        score = float(np.sqrt(np.mean((pred_inv - y_valid) ** 2)))
    return {
        "coef": [float(v) for v in np.asarray(estimator.coef_, dtype=float).reshape(-1).tolist()],
        "intercept": float(estimator.intercept_),
    }, score


def _rmse_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return float("inf")
    return float(
        np.sqrt(np.mean((np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))
    )


def _rmse_from_absolute_residuals(residuals: list[float]) -> float:
    values = np.asarray([abs(float(v)) for v in residuals if math.isfinite(float(v))], dtype=float)
    if values.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(values**2)))


class _AsymmetricRULLoss:
    """Asymmetric squared-error loss that penalises RUL overprediction more than underprediction.

    Matches the directional asymmetry of the NASA scoring function (exp(d/10) for
    overprediction vs exp(-d/13) for underprediction). The low-RUL weight term
    concentrates the overprediction penalty near the failure region.

    Args:
        over_penalty:  Multiplier applied to squared error when y_pred > y_true (default 2.0).
        under_penalty: Multiplier applied to squared error when y_pred <= y_true (default 1.0).
        max_rul:       RUL cap used to normalise the weight term (default 125.0, matching MAX_RUL).

    Note: For the aggressive reference setting (over_penalty=5.0, under_penalty=0.25) see
    docs/references/Train_model_AsymMCDver3.1.py. That ratio (20×) combined with the weight
    term risks gradient explosion at low RUL; the default 2× is more stable.
    """

    def __init__(self, over_penalty: float = 2.0, under_penalty: float = 1.0, max_rul: float = 125.0) -> None:  # noqa: E501
        self._alpha = float(over_penalty)
        self._beta = float(under_penalty)
        self._max_rul = float(max_rul)

    def __call__(self, y_pred: Any, y_true: Any) -> Any:
        import torch
        error = y_pred - y_true
        weight = ((1.0 - y_true / self._max_rul).clamp(min=0.0)) ** 2
        loss = torch.where(
            error > 0,
            self._alpha * (error ** 2) * weight,
            self._beta * (error ** 2),
        )
        return loss.mean()


def _normalize_training_protocol(training_protocol: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(training_protocol or {})
    protocol_name = str(raw.get("name") or "default")
    optimizer = str(raw.get("optimizer") or "adam").strip().lower()
    if optimizer not in {"adam", "adamw"}:
        optimizer = "adam"
    loss_name = str(raw.get("loss") or "mse").strip().lower()
    if loss_name not in {"mse", "huber", "asymmetric_rul"}:
        loss_name = "mse"
    return {
        "name": protocol_name,
        "optimizer": optimizer,
        "learning_rate": float(raw.get("learning_rate") or 1e-3),
        "weight_decay": max(0.0, float(raw.get("weight_decay") or 0.0)),
        "loss": loss_name,
        "huber_delta": max(1e-6, float(raw.get("huber_delta") or 1.0)),
        "asym_over_penalty": max(1e-6, float(raw.get("asym_over_penalty") or 2.0)),
        "asym_under_penalty": max(1e-6, float(raw.get("asym_under_penalty") or 1.0)),
        "asym_max_rul": max(1.0, float(raw.get("asym_max_rul") or 125.0)),
        "grad_clip_norm": max(0.0, float(raw.get("grad_clip_norm") or 1.0)),
        "patience": max(1, int(raw.get("patience") or 10)),
        "min_delta": max(0.0, float(raw.get("min_delta") or 1e-9)),
        "aux_loss_weight": max(0.0, float(raw.get("aux_loss_weight") or 0.0)),
    }


def train_univariate_torch_forecaster(
    *,
    algo: str,
    ys_by_series: dict[str, list[float]] | None = None,
    records_by_series: dict[str, list[dict[str, Any]]] | None = None,
    training_hours: float | None,
    device: str | None = None,
    context_len: int | None = None,
    max_exogenous_features: int = 24,
    target_transform: str = "none",
    prefer_exogenous: bool = True,
    allow_structure_fallback: bool = False,
    model_params: dict[str, Any] | None = None,
    training_protocol: dict[str, Any] | None = None,
    reuse_snapshot: dict[str, Any] | None = None,
    reuse_state_dict: dict[str, Any] | None = None,
) -> TorchTrainedArtifact:
    algo_key = str(algo or "").strip().lower()
    if algo_key not in {
        "afnocg2",
        "cifnocg2",
        "afnocg3",
        "afnocg3_v1",
        "bilstm_rul",
        "tcn_rul",
        "transformer_rul",
    }:
        raise ValueError(f"Unsupported torch algo='{algo_key}'")

    if records_by_series is None:
        records_by_series = {}
        for series_id, ys in (ys_by_series or {}).items():
            records_by_series[str(series_id)] = [{"y": float(value), "x": {}} for value in ys]

    lengths = [len(rows) for rows in records_by_series.values() if isinstance(rows, list)]
    max_len = max(lengths, default=0)
    requested_snapshot = reuse_snapshot if isinstance(reuse_snapshot, dict) else None
    ctx_len = (
        int(requested_snapshot.get("context_len"))
        if requested_snapshot and requested_snapshot.get("context_len")
        else _infer_context_len(max_len, requested=context_len)
    )
    transform_name = (
        _target_transform_name(str(requested_snapshot.get("target_transform") or target_transform))
        if requested_snapshot
        else _target_transform_name(target_transform)
    )

    if requested_snapshot and isinstance(requested_snapshot.get("feature_keys"), list):
        feature_keys = [
            str(key) for key in requested_snapshot.get("feature_keys") or [] if isinstance(key, str)
        ]
        feature_source = str(
            requested_snapshot.get("feature_source") or ("x" if feature_keys else "y")
        )
    else:
        feature_keys = (
            _select_feature_keys(records_by_series, max_features=max_exogenous_features)
            if prefer_exogenous
            else []
        )
        feature_source = "x" if feature_keys else "y"
    protocol = _normalize_training_protocol(training_protocol)
    x_all, y_all = _build_supervised_sequences(
        records_by_series,
        context_len=ctx_len,
        feature_keys=feature_keys,
        feature_source=feature_source,
    )
    if x_all.shape[0] < 1:
        raise ValueError("Insufficient training data for torch model (need >= 1 sample).")

    train_idx, valid_idx = _split_sequence_indices(int(x_all.shape[0]))
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_valid = (
        x_all[valid_idx] if valid_idx.size > 0 else np.zeros((0,) + x_all.shape[1:], dtype=float)
    )
    y_valid = y_all[valid_idx] if valid_idx.size > 0 else np.zeros((0,), dtype=float)

    input_dim = int(x_train.shape[-1]) if x_train.ndim == 3 and x_train.shape[-1] > 0 else 1
    structure_mode = "temporal"
    temporal_registry_unavailable = False
    if (
        requested_snapshot is not None
        and str(requested_snapshot.get("structure_mode") or "temporal") == "temporal"
    ):
        snapshot = dict(requested_snapshot)
        state_dict = dict(reuse_state_dict or {})
        residuals = [
            float(v)
            for v in snapshot.get("pooled_residuals") or []
            if isinstance(v, int | float) and math.isfinite(float(v))
        ]
        temporal_eval_rmse = (
            float(snapshot.get("validation_rmse"))
            if isinstance(snapshot.get("validation_rmse"), int | float)
            and math.isfinite(float(snapshot.get("validation_rmse")))
            else _rmse_from_absolute_residuals(residuals)
        )
        structure_mode = str(snapshot.get("structure_mode") or "temporal")
    else:
        try:
            import torch

            device_str = _choose_device(device, algo=algo_key)
            dev = torch.device(device_str)
            from src.models.registry import get_model_handlers

            handlers = get_model_handlers(algo_key)
            cfg = SimpleNamespace(
                MODEL_NAME=algo_key, TARGET_KIND="regression", MODEL_SEED=0, MODEL_PARAMS={}
            )
            data_shapes = {"input_dim": input_dim, "n_groups": 1}
            model = handlers.build(
                cfg, dev, data_shapes=data_shapes, artifacts=None, model_params=model_params
            )

            x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
            y_train_t = torch.tensor(
                _transform_target(y_train, target_transform=transform_name),
                dtype=torch.float32,
                device=dev,
            ).unsqueeze(-1)
            x_valid_t = (
                torch.tensor(x_valid, dtype=torch.float32, device=dev) if x_valid.size > 0 else None
            )
            y_valid_t = (
                torch.tensor(
                    _transform_target(y_valid, target_transform=transform_name),
                    dtype=torch.float32,
                    device=dev,
                ).unsqueeze(-1)
                if y_valid.size > 0
                else None
            )

            model.train()
            params = model.parameters()
            if protocol["optimizer"] == "adamw":
                opt = torch.optim.AdamW(
                    params,
                    lr=float(protocol["learning_rate"]),
                    weight_decay=float(protocol["weight_decay"]),
                )
            else:
                opt = torch.optim.Adam(
                    params,
                    lr=float(protocol["learning_rate"]),
                    weight_decay=float(protocol["weight_decay"]),
                )
            if protocol["loss"] == "huber":
                loss_fn = torch.nn.HuberLoss(delta=float(protocol["huber_delta"]))
            elif protocol["loss"] == "asymmetric_rul":
                # When target_transform="log1p" is active, y_true arrives in log-space
                # (log1p(125) ≈ 4.83).  The weight formula (1 - y_true/max_rul)^2 must
                # use the log-space cap so it varies across the full [0, log1p(max_rul)]
                # range instead of collapsing to a near-constant ≈ 0.926.
                raw_max_rul = float(protocol["asym_max_rul"])
                effective_max_rul = (
                    math.log1p(raw_max_rul)
                    if transform_name == "log1p"
                    else raw_max_rul
                )
                loss_fn = _AsymmetricRULLoss(
                    over_penalty=float(protocol["asym_over_penalty"]),
                    under_penalty=float(protocol["asym_under_penalty"]),
                    max_rul=effective_max_rul,
                )
            else:
                loss_fn = torch.nn.MSELoss()

            hours = max(0.05, float(training_hours) if training_hours is not None else 0.05)
            epochs = int(max(20, min(200, round(hours * 1200))))
            max_epochs_raw = os.getenv("RULFM_FORECASTING_TORCH_MAX_EPOCHS", "").strip()
            if max_epochs_raw:
                try:
                    epochs = max(1, min(epochs, int(max_epochs_raw)))
                except ValueError:
                    pass

            batch_size = int(
                min(256, max(16, 2 ** int(math.log2(max(16, min(len(x_train_t), 256))))))
            )
            idx = torch.arange(x_train_t.size(0), device=dev)
            best_loss = float("inf")
            patience = int(protocol["patience"])
            min_delta = float(protocol["min_delta"])
            grad_clip_norm = float(protocol["grad_clip_norm"])
            aux_loss_weight = float(protocol["aux_loss_weight"])
            bad = 0

            for _epoch in range(epochs):
                perm = idx[torch.randperm(idx.numel())]
                for start in range(0, perm.numel(), batch_size):
                    sel = perm[start : start + batch_size]
                    xb = x_train_t.index_select(0, sel)
                    yb = y_train_t.index_select(0, sel)
                    opt.zero_grad(set_to_none=True)
                    if aux_loss_weight > 0.0 and algo_key == "afnocg3_v1":
                        try:
                            forward_out = model(xb, return_aux=True)
                        except TypeError:
                            forward_out = model(xb)
                        y_pred = (
                            forward_out[0]
                            if isinstance(forward_out, tuple | list)
                            else forward_out
                        )
                        aux_penalty = None
                        aux_losses = getattr(model, "aux_losses", {})
                        if isinstance(aux_losses, dict) and aux_losses:
                            aux_terms = [
                                value.float()
                                for value in aux_losses.values()
                                if hasattr(value, "float")
                            ]
                            if aux_terms:
                                aux_penalty = torch.stack(aux_terms).sum()
                        loss = loss_fn(y_pred, yb)
                        if aux_penalty is not None:
                            loss = loss + aux_penalty * aux_loss_weight
                    else:
                        y_pred = _predict_torch_tensor(model, xb)
                        loss = loss_fn(y_pred, yb)
                    loss.backward()
                    if grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    opt.step()

                model.eval()
                with torch.no_grad():
                    eval_x = x_valid_t if x_valid_t is not None else x_train_t
                    eval_y = y_valid_t if y_valid_t is not None else y_train_t
                    eval_pred = _predict_torch_tensor(model, eval_x)
                    eval_loss = float(loss_fn(eval_pred, eval_y).detach().cpu().item())
                if eval_loss + min_delta < best_loss:
                    best_loss = eval_loss
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
                model.train()

            model.eval()
            with torch.no_grad():
                calib_x = x_valid_t if x_valid_t is not None else x_train_t
                calib_pred = (
                    _predict_torch_tensor(model, calib_x).reshape(-1).detach().cpu().numpy()
                )
            calib_truth = y_valid if y_valid.size > 0 else y_train
            calib_pred_inv = _inverse_target(
                np.asarray(calib_pred, dtype=float), target_transform=transform_name
            )
            temporal_eval_rmse = _rmse_from_predictions(calib_truth, calib_pred_inv)
            residuals = [
                float(v)
                for v in np.abs(calib_pred_inv - calib_truth).tolist()
                if math.isfinite(float(v))
            ]

            state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            extras = handlers.extras or {}
            snap_fn = extras.get("snapshot_builder")
            if callable(snap_fn):
                snapshot = snap_fn(cfg, model, context={"data_shapes": data_shapes})
                if not isinstance(snapshot, dict):
                    snapshot = {"model_name": algo_key, "input_dim": input_dim}
            else:
                snapshot = {"model_name": algo_key, "input_dim": input_dim}
        except Exception as exc:
            if not _missing_model_registry(exc):
                raise
            temporal_registry_unavailable = True
            temporal_eval_rmse = float("inf")
            residuals = []
            state_dict = {}
            snapshot = {
                "model_name": algo_key,
                "input_dim": input_dim,
                "structure_mode": "temporal_registry_unavailable",
                "missing_dependency": "src.models.registry",
            }

    if (allow_structure_fallback or temporal_registry_unavailable) and x_train.shape[0] >= 1:
        flat_state, flat_rmse = _fit_flat_ridge(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid if x_valid.size > 0 else x_train,
            y_valid=y_valid if y_valid.size > 0 else y_train,
            target_transform=transform_name,
        )
        if temporal_registry_unavailable or flat_rmse <= temporal_eval_rmse:
            structure_mode = "flat_ridge"
            snapshot = {
                "model_name": algo_key,
                "input_dim": input_dim,
                "context_len": ctx_len,
                "structure_mode": structure_mode,
                "feature_source": feature_source,
                "feature_keys": feature_keys,
                "target_transform": transform_name,
                "flat_ridge": flat_state,
            }
            if temporal_registry_unavailable:
                snapshot["missing_dependency"] = "src.models.registry"
            state_dict = {}
            eval_x = x_valid if x_valid.size > 0 else x_train
            eval_truth = y_valid if y_valid.size > 0 else y_train
            pred = np.asarray(
                eval_x.reshape(eval_x.shape[0], -1) @ np.asarray(flat_state["coef"], dtype=float)
                + float(flat_state["intercept"]),
                dtype=float,
            )
            pred = _inverse_target(pred, target_transform=transform_name)
            residuals = [
                float(v) for v in np.abs(pred - eval_truth).tolist() if math.isfinite(float(v))
            ]

    snapshot.update(
        {
            "context_len": ctx_len,
            "input_dim": input_dim,
            "feature_source": feature_source,
            "feature_keys": feature_keys,
            "target_transform": transform_name,
            "structure_mode": structure_mode,
            "validation_rmse": float(temporal_eval_rmse)
            if math.isfinite(float(temporal_eval_rmse))
            else None,
            "pooled_residuals": residuals[-500:],
            "training_protocol": protocol,
        }
    )

    return TorchTrainedArtifact(
        algo=algo_key,
        input_dim=input_dim,
        context_len=ctx_len,
        pooled_residuals=residuals[-500:],
        snapshot=snapshot,
        state_dict=state_dict,
    )


def finetune_gate_stardast2v5(
    *, model: Any, X: Any, y: Any, n_total: int, device: str
) -> None:
    import torch

    if n_total < 50:
        epochs, lr = 3, 2e-4
    elif n_total < 200:
        epochs, lr = 5, 1e-4
    else:
        epochs, lr = 10, 5e-5

    gate_params = [
        p for name, p in model.named_parameters() if "base_model" not in name and p.requires_grad
    ]
    if not gate_params or X.shape[0] == 0:
        return

    opt = torch.optim.Adam(gate_params, lr=lr, weight_decay=1e-4)
    model.train(True)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = model(X)
        yhat = (out[0] if isinstance(out, tuple | list) else out).reshape(-1)
        loss = torch.mean((yhat - y) ** 2)
        loss.backward()
        for p in gate_params:
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(gate_params, max_norm=1.0)
        opt.step()
    model.eval()


def _ensure_context_rows(
    *, context_records: list[dict[str, Any]] | None, context: list[float] | None, context_len: int
) -> list[dict[str, Any]]:
    if context_records:
        rows = [dict(row) for row in context_records]
    else:
        values = [float(v) for v in (context or [])]
        if not values:
            values = [0.0]
        rows = [{"y": value, "x": {}} for value in values]
    need = max(1, context_len)
    if len(rows) < need:
        pad = rows[0] if rows else {"y": 0.0, "x": {}}
        rows = [dict(pad) for _ in range(need - len(rows))] + rows
    return rows[-need:]


def _predict_flat_ridge_once(
    rows: list[dict[str, Any]],
    *,
    feature_keys: list[str],
    feature_source: str,
    snapshot: dict[str, Any],
) -> float:
    features = _rows_to_feature_matrix(
        rows, feature_keys=feature_keys, feature_source=feature_source
    ).reshape(-1)
    flat = snapshot.get("flat_ridge") if isinstance(snapshot.get("flat_ridge"), dict) else {}
    coef = np.asarray(flat.get("coef") or [], dtype=float)
    intercept = float(flat.get("intercept") or 0.0)
    if coef.size != features.size:
        coef = np.resize(coef, features.size)
    pred = np.asarray([float(features @ coef + intercept)], dtype=float)
    pred = _inverse_target(
        pred,
        target_transform=_target_transform_name(str(snapshot.get("target_transform") or "none")),
    )
    return float(pred[0])


def _load_torch_forecaster_model(
    *, algo_key: str, snapshot: dict[str, Any], state_dict: dict[str, Any], device: str | None
) -> tuple[Any, Any]:
    import torch

    device_str = _choose_device(device, algo=algo_key)
    dev = torch.device(device_str)
    from src.models.registry import get_model_handlers

    handlers = get_model_handlers(algo_key)
    cfg = SimpleNamespace(
        MODEL_NAME=algo_key, TARGET_KIND="regression", MODEL_SEED=0, MODEL_PARAMS={}
    )
    model = handlers.load_from_snapshot(cfg, dev, snapshot, model_params=None)
    torch_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, list | int | float | bool):
            torch_state_dict[k] = torch.tensor(v)
        else:
            torch_state_dict[k] = v
    model.load_state_dict(torch_state_dict, strict=False)
    model.eval()
    return model, dev


def _predict_torch_next_value(
    model: Any,
    *,
    rows: list[dict[str, Any]],
    context_len: int,
    feature_keys: list[str],
    feature_source: str,
    transform_name: str,
    dev: Any,
) -> float:
    import torch

    matrix = _rows_to_feature_matrix(
        rows[-context_len:], feature_keys=feature_keys, feature_source=feature_source
    )
    x_tensor = torch.tensor(matrix, dtype=torch.float32, device=dev).view(1, context_len, -1)
    with torch.no_grad():
        pred_t = _predict_torch_tensor(model, x_tensor).reshape(-1).detach().cpu().numpy()
    return float(
        _inverse_target(np.asarray([pred_t[-1]], dtype=float), target_transform=transform_name)[0]
    )


def _set_mc_dropout_enabled(model: Any, enabled: bool) -> None:
    toggle_name = "enable_final_layer_dropout" if enabled else "disable_final_layer_dropout"
    toggle = getattr(model, toggle_name, None)
    if callable(toggle):
        toggle()


def forecast_univariate_torch_with_details(
    *,
    algo: str,
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
    context: list[float] | None = None,
    context_records: list[dict[str, Any]] | None = None,
    future_feature_rows: list[dict[str, Any]] | None = None,
    horizon: int,
    device: str | None = None,
    mc_dropout_samples: int = 0,
    occlusion_feature_keys: list[str] | None = None,
    occlusion_baseline: dict[str, float] | None = None,
    occlusion_top_k: int = 5,
) -> dict[str, Any]:
    if horizon < 1:
        return {"point": [], "mc_dropout": None, "occlusion": None}

    algo_key = str(algo or "").strip().lower()
    if algo_key not in {
        "afnocg2",
        "cifnocg2",
        "afnocg3",
        "afnocg3_v1",
        "bilstm_rul",
        "tcn_rul",
        "transformer_rul",
    }:
        raise ValueError(f"Unsupported torch algo='{algo_key}'")

    context_len = int(snapshot.get("context_len") or len(context or []) or 1)
    feature_source = str(snapshot.get("feature_source") or "y")
    feature_keys = [str(key) for key in snapshot.get("feature_keys") or [] if isinstance(key, str)]
    structure_mode = str(snapshot.get("structure_mode") or "temporal")
    transform_name = _target_transform_name(str(snapshot.get("target_transform") or "none"))
    rows = _ensure_context_rows(
        context_records=context_records, context=context, context_len=context_len
    )
    future_rows = [dict(row) for row in (future_feature_rows or [])]

    if structure_mode == "flat_ridge":
        points: list[float] = []
        last_x = dict(_record_x(rows[-1])) if rows else {}
        for step_idx in range(int(horizon)):
            pred = _predict_flat_ridge_once(
                rows[-context_len:],
                feature_keys=feature_keys,
                feature_source=feature_source,
                snapshot=snapshot,
            )
            points.append(pred)
            next_x = _record_x(future_rows[step_idx]) if step_idx < len(future_rows) else last_x
            rows.append({"y": pred, "x": next_x})
        return {"point": points, "mc_dropout": None, "occlusion": None}

    model, dev = _load_torch_forecaster_model(
        algo_key=algo_key, snapshot=snapshot, state_dict=state_dict, device=device
    )

    points: list[float] = []
    per_step_occlusion: list[dict[str, Any]] = []
    last_x = dict(_record_x(rows[-1])) if rows else {}
    for step_idx in range(int(horizon)):
        pred = _predict_torch_next_value(
            model,
            rows=rows,
            context_len=context_len,
            feature_keys=feature_keys,
            feature_source=feature_source,
            transform_name=transform_name,
            dev=dev,
        )
        points.append(pred)
        requested_keys = [
            key for key in (occlusion_feature_keys or feature_keys) if key in feature_keys
        ]
        if requested_keys:
            baseline_map = {
                str(key): float((occlusion_baseline or {}).get(key, 0.0)) for key in requested_keys
            }
            local_scores: dict[str, float] = {}
            for key in requested_keys:
                occluded_rows: list[dict[str, Any]] = []
                for row in rows[-context_len:]:
                    x = dict(_record_x(row))
                    x[key] = baseline_map[key]
                    occluded_rows.append({"y": _record_y(row), "x": x})
                occluded_pred = _predict_torch_next_value(
                    model,
                    rows=occluded_rows,
                    context_len=context_len,
                    feature_keys=feature_keys,
                    feature_source=feature_source,
                    transform_name=transform_name,
                    dev=dev,
                )
                local_scores[key] = abs(float(pred) - float(occluded_pred))
            ranked = sorted(local_scores.items(), key=lambda item: item[1], reverse=True)
            per_step_occlusion.append(
                {
                    "top_features": [
                        {"feature": key, "importance": float(score), "rank": idx + 1}
                        for idx, (key, score) in enumerate(ranked[: max(1, occlusion_top_k)])
                    ],
                    "feature_importance": {key: float(score) for key, score in ranked},
                }
            )
        next_x = _record_x(future_rows[step_idx]) if step_idx < len(future_rows) else last_x
        rows.append({"y": pred, "x": next_x})

    mc_payload: dict[str, Any] | None = None
    sample_count = max(0, int(mc_dropout_samples))
    if sample_count > 0:
        paths: list[list[float]] = []
        _set_mc_dropout_enabled(model, True)
        try:
            for _ in range(sample_count):
                sample_rows = _ensure_context_rows(
                    context_records=context_records, context=context, context_len=context_len
                )
                sample_path: list[float] = []
                sample_last_x = dict(_record_x(sample_rows[-1])) if sample_rows else {}
                for step_idx in range(int(horizon)):
                    sample_pred = _predict_torch_next_value(
                        model,
                        rows=sample_rows,
                        context_len=context_len,
                        feature_keys=feature_keys,
                        feature_source=feature_source,
                        transform_name=transform_name,
                        dev=dev,
                    )
                    sample_path.append(sample_pred)
                    next_x = (
                        _record_x(future_rows[step_idx])
                        if step_idx < len(future_rows)
                        else sample_last_x
                    )
                    sample_rows.append({"y": sample_pred, "x": next_x})
                paths.append(sample_path)
        finally:
            _set_mc_dropout_enabled(model, False)
        samples_arr = (
            np.asarray(paths, dtype=float) if paths else np.zeros((0, horizon), dtype=float)
        )
        if samples_arr.size > 0:
            mc_payload = {
                "samples": int(sample_count),
                "per_step_mean": [float(v) for v in np.mean(samples_arr, axis=0).tolist()],
                "per_step_std": [float(v) for v in np.std(samples_arr, axis=0).tolist()],
                "per_step_var": [float(v) for v in np.var(samples_arr, axis=0).tolist()],
            }

    occlusion_payload: dict[str, Any] | None = None
    if per_step_occlusion:
        global_scores: dict[str, list[float]] = {}
        for step_payload in per_step_occlusion:
            for key, score in (step_payload.get("feature_importance") or {}).items():
                global_scores.setdefault(str(key), []).append(float(score))
        ranked_global = sorted(
            ((key, float(np.mean(values))) for key, values in global_scores.items() if values),
            key=lambda item: item[1],
            reverse=True,
        )
        occlusion_payload = {
            "method": "occlusion_delta_v1",
            "per_step": per_step_occlusion,
            "global": {
                "top_features": [
                    {"feature": key, "importance": float(score), "rank": idx + 1}
                    for idx, (key, score) in enumerate(ranked_global[: max(1, occlusion_top_k)])
                ],
                "feature_importance": {key: float(score) for key, score in ranked_global},
            },
        }

    return {"point": points, "mc_dropout": mc_payload, "occlusion": occlusion_payload}


def forecast_univariate_torch(
    *,
    algo: str,
    snapshot: dict[str, Any],
    state_dict: dict[str, Any],
    context: list[float] | None = None,
    context_records: list[dict[str, Any]] | None = None,
    future_feature_rows: list[dict[str, Any]] | None = None,
    horizon: int,
    device: str | None = None,
) -> list[float]:
    details = forecast_univariate_torch_with_details(
        algo=algo,
        snapshot=snapshot,
        state_dict=state_dict,
        context=context,
        context_records=context_records,
        future_feature_rows=future_feature_rows,
        horizon=horizon,
        device=device,
    )
    return [float(value) for value in details.get("point") or []]
