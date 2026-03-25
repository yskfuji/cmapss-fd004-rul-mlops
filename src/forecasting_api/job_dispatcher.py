from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from fastapi import BackgroundTasks

JobType = Literal["forecast", "train", "backtest"]


@dataclass(frozen=True)
class JobEnvelope:
    job_id: str
    job_type: JobType
    payload: dict[str, Any]


class JobEnqueuer(Protocol):
    def enqueue(self, *, job_id: str, job_type: JobType, payload: dict[str, Any]) -> None: ...


class JobEnqueuerFactory(Protocol):
    def __call__(self, background: BackgroundTasks) -> JobEnqueuer: ...


class InProcessBackgroundJobEnqueuer:
    def __init__(
        self,
        *,
        background: BackgroundTasks,
        run_job: Callable[[str, JobType, dict[str, Any]], None],
    ) -> None:
        self._background = background
        self._run_job = run_job

    def enqueue(self, *, job_id: str, job_type: JobType, payload: dict[str, Any]) -> None:
        self._background.add_task(self._run_job, job_id, job_type, payload)


class PersistentQueueJobEnqueuer:
    def enqueue(self, *, job_id: str, job_type: JobType, payload: dict[str, Any]) -> None:
        # Job creation already persisted the queued state in JobStore.
        # External workers claim and execute queued jobs asynchronously.
        return None


def build_inprocess_job_enqueuer(
    *,
    background: BackgroundTasks,
    run_job: Callable[[str, JobType, dict[str, Any]], None],
) -> JobEnqueuer:
    return InProcessBackgroundJobEnqueuer(background=background, run_job=run_job)


def build_persistent_job_enqueuer(*, background: BackgroundTasks | None = None) -> JobEnqueuer:
    return PersistentQueueJobEnqueuer()