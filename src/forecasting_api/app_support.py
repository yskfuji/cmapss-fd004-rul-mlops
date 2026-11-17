from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def bi(en: str, ja: str) -> str:
    return f"[EN] {en}\n[JA] {ja}"


def as_dict(value: Any) -> dict[str, Any]:
    return value.copy() if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value.copy() if isinstance(value, list) else []


def as_float_list(value: Any) -> list[float]:
    return [
        float(item)
        for item in as_list(value)
        if isinstance(item, int | float) and math.isfinite(float(item))
    ]


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def model_artifact_dir(root: Path, model_id: str) -> Path:
    return (root / model_id).resolve()


def artifact_relpath(model_id: str, filename: str) -> str:
    return f"{model_id}/{filename}"


def artifact_abspath(root: Path, relpath: str) -> Path:
    return (root / relpath).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_models_from_store(store_path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(store_path)
    models = payload.get("models") if isinstance(payload.get("models"), dict) else payload
    return models.copy() if isinstance(models, dict) else {}


def save_models_to_store(store_path: Path, models: dict[str, dict[str, Any]]) -> None:
    write_json(store_path, {"models": models})


def extract_state_dict(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("state_dict", "model_state", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def try_torch_load_weights(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload if isinstance(payload, dict) else {"state_dict": payload}


def load_joblib_artifact(path: Path) -> Any:
    import joblib

    return joblib.load(path)


def load_fd004_benchmark_summary(path: Path, *, logger: Any) -> dict[str, Any]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to parse FD004 benchmark summary")
            return {
                "generated_at": None,
                "rows": [],
                "notes": ["FD004 benchmark summary exists but could not be parsed."],
            }
        if isinstance(payload, dict):
            return payload
    return {
        "generated_at": None,
        "rows": [],
        "notes": ["FD004 benchmark summary has not been generated yet."],
    }