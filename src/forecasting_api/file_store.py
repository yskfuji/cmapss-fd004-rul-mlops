from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

fcntl: Any | None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # pragma: no cover - only triggered on non-POSIX platforms
    fcntl = None

T = TypeVar("T")


@contextmanager
def exclusive_lock(target_path: Path):
    if fcntl is None:
        # Non-POSIX platforms fall back to atomic replace without an inter-process lock.
        with contextlib.nullcontext():
            yield
        return
    lock_path = target_path.with_name(f".{target_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_json(path: Path, default: T) -> T:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    return payload  # type: ignore[return-value]


def update_json_file(path: Path, *, default: T, updater: Callable[[T], T]) -> T:
    with exclusive_lock(path):
        current = load_json(path, default)
        updated = updater(current)
        atomic_write_text(path, json.dumps(updated, indent=2, ensure_ascii=False))
        return updated
