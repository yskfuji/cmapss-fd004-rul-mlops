from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


_RAW_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{idx}" for idx in range(1, 22)],
]

_DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "CMAPSSData"
_SUPPORTED_DATASETS = ("fd001", "fd002", "fd003", "fd004")


def available_cmapss_datasets() -> list[str]:
    return list(_SUPPORTED_DATASETS)


def _normalize_dataset_id(dataset_id: str) -> str:
    value = str(dataset_id or "").strip().lower()
    if value not in _SUPPORTED_DATASETS:
        raise ValueError(f"unsupported CMAPSS dataset: {dataset_id}")
    return value


def _dataset_dir(dataset_dir: Path | None = None) -> Path:
    return dataset_dir or _DEFAULT_DATASET_DIR


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@lru_cache(maxsize=16)
def _read_split(dataset_id: str, split: str, dataset_dir_text: str) -> tuple[dict[str, Any], ...]:
    dataset_key = _normalize_dataset_id(dataset_id)
    split_name = str(split or "train").strip().lower()
    if split_name not in {"train", "test"}:
        raise ValueError(f"unsupported split: {split}")
    path = Path(dataset_dir_text) / f"{split_name}_{dataset_key.upper()}.txt"
    if not path.exists():
        raise FileNotFoundError(f"CMAPSS {dataset_key.upper()} file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            values = [float(part) for part in text.split()]
            if len(values) < len(_RAW_COLUMNS):
                raise ValueError(f"unexpected {dataset_key.upper()} column count in {path}: {len(values)}")
            row: dict[str, Any] = {}
            for idx, col in enumerate(_RAW_COLUMNS):
                val = values[idx]
                row[col] = int(val) if col in {"unit_id", "cycle"} else float(val)
            rows.append(row)
    return tuple(rows)


@lru_cache(maxsize=8)
def _read_test_terminal_rul(dataset_id: str, dataset_dir_text: str) -> dict[int, int]:
    dataset_key = _normalize_dataset_id(dataset_id)
    path = Path(dataset_dir_text) / f"RUL_{dataset_key.upper()}.txt"
    if not path.exists():
        raise FileNotFoundError(f"CMAPSS {dataset_key.upper()} RUL file not found: {path}")
    mapping: dict[int, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            mapping[idx] = int(float(text.split()[0]))
    return mapping


def _rul_by_unit(dataset_id: str, split: str, dataset_dir: Path) -> dict[int, dict[int, int]]:
    rows = _read_split(dataset_id, split, str(dataset_dir))
    by_unit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[int(row["unit_id"])].append(dict(row))

    rul_lookup: dict[int, dict[int, int]] = {}
    if split == "train":
        for unit_id, unit_rows in by_unit.items():
            max_cycle = max(int(r["cycle"]) for r in unit_rows)
            rul_lookup[unit_id] = {int(r["cycle"]): max_cycle - int(r["cycle"]) for r in unit_rows}
        return rul_lookup

    terminal_rul = _read_test_terminal_rul(dataset_id, str(dataset_dir))
    for unit_id, unit_rows in by_unit.items():
        max_cycle = max(int(r["cycle"]) for r in unit_rows)
        offset = int(terminal_rul.get(unit_id, 0))
        rul_lookup[unit_id] = {int(r["cycle"]): (max_cycle - int(r["cycle"])) + offset for r in unit_rows}
    return rul_lookup


def build_cmapss_payload(
    *,
    dataset_id: str,
    split: str = "train",
    unit_ids: list[int] | None = None,
    window_size: int | None = None,
    task: str = "forecast",
    horizon: int = 20,
    quantiles: list[float] | None = None,
    level: list[float] | None = None,
    missing_policy: str = "ignore",
    chart_type: str = "trend",
    value_unit: str = "cycles",
    max_rul: int | None = 125,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    dataset_key = _normalize_dataset_id(dataset_id)
    split_name = str(split or "train").strip().lower()
    root = _dataset_dir(dataset_dir)
    rows = _read_split(dataset_key, split_name, str(root))
    rul_lookup = _rul_by_unit(dataset_key, split_name, root)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["unit_id"])].append(dict(row))

    selected_units = [int(v) for v in (unit_ids or sorted(grouped)) if int(v) in grouped]
    if not selected_units:
        raise ValueError(f"no matching {dataset_key.upper()} units found")

    records: list[dict[str, Any]] = []
    for offset, unit_id in enumerate(selected_units):
        unit_rows = sorted(grouped[unit_id], key=lambda item: int(item["cycle"]))
        if window_size and window_size > 0:
            unit_rows = unit_rows[-int(window_size) :]
        base_ts = datetime(2008, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset)
        for row in unit_rows:
            cycle = int(row["cycle"])
            rul_value = int(rul_lookup[unit_id][cycle])
            if max_rul is not None:
                rul_value = min(rul_value, int(max_rul))
            timestamp = _iso_utc(base_ts + timedelta(hours=cycle - 1))
            x = {
                "cycle": float(cycle),
                "op_setting_1": float(row["op_setting_1"]),
                "op_setting_2": float(row["op_setting_2"]),
                "op_setting_3": float(row["op_setting_3"]),
            }
            for sensor_idx in range(1, 22):
                key = f"sensor_{sensor_idx}"
                x[key] = float(row[key])
            records.append(
                {
                    "series_id": f"{dataset_key}_engine_{unit_id:03d}",
                    "timestamp": timestamp,
                    "y": float(rul_value),
                    "x": x,
                }
            )

    feature_keys = sorted(records[0]["x"].keys()) if records else []
    return {
        "dataset": dataset_key,
        "split": split_name,
        "task": str(task or "forecast"),
        "horizon": int(horizon),
        "frequency": "1h",
        "quantiles": quantiles,
        "level": level,
        "missing_policy": str(missing_policy or "ignore"),
        "chartType": str(chart_type or "trend"),
        "value_unit": str(value_unit or "cycles"),
        "records": records,
        "meta": {
            "dataset": dataset_key,
            "unit_ids": selected_units,
            "unit_count": len(selected_units),
            "record_count": len(records),
            "feature_keys": feature_keys,
            "target": "remaining_useful_life_cycles",
            "source_files": [f"{split_name}_{dataset_key.upper()}.txt", f"RUL_{dataset_key.upper()}.txt" if split_name == "test" else None],
        },
    }


__all__ = ["available_cmapss_datasets", "build_cmapss_payload"]