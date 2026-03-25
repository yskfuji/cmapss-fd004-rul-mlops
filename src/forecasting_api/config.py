from __future__ import annotations

from pathlib import Path


def env_first(*names: str) -> str | None:
    import os

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def env_path(default_path: Path, *names: str) -> Path:
    raw = env_first(*names)
    return Path(raw) if raw else default_path


def env_bool(*names: str, default: bool = False) -> bool:
    raw = env_first(*names)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def env_int(
    *names: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw = env_first(*names)
    value = default
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value
