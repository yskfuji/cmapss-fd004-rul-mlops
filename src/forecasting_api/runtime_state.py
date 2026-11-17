from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from forecasting_api.job_store import JobStore


class AppRuntimeState:
    def __init__(
        self,
        *,
        build_job_store: Callable[[], JobStore],
        load_trained_models: Callable[[], dict[str, dict[str, Any]]],
    ) -> None:
        self._build_job_store = build_job_store
        self._load_trained_models = load_trained_models
        self._job_store: JobStore | None = None
        self._trained_models: dict[str, dict[str, Any]] | None = None
        self._job_store_lock = Lock()
        self._trained_models_lock = Lock()

    def set_job_store(self, value: JobStore | None) -> JobStore | None:
        self._job_store = value
        return self._job_store

    def set_trained_models(
        self, value: dict[str, dict[str, Any]] | None
    ) -> dict[str, dict[str, Any]] | None:
        self._trained_models = value
        return self._trained_models

    def require_job_store(self, current: JobStore | None = None) -> JobStore:
        if current is not None:
            self._job_store = current
            return current
        if self._job_store is None:
            with self._job_store_lock:
                if self._job_store is None:
                    self._job_store = self._build_job_store()
        assert self._job_store is not None
        return self._job_store

    def require_trained_models(
        self,
        current: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if current is not None:
            self._trained_models = current
            return current
        if self._trained_models is None:
            with self._trained_models_lock:
                if self._trained_models is None:
                    self._trained_models = self._load_trained_models()
        assert self._trained_models is not None
        return self._trained_models