from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ApiError(Exception):
    status_code: int
    error_code: str
    message: str
    details: dict[str, Any] | None = None
