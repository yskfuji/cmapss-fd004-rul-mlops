from __future__ import annotations

import os
import time

from forecasting_api import app as app_module
from forecasting_api.services import jobs_service


def _runtime_job_store():
    if app_module.JOB_STORE is None:
        app_module._set_runtime_job_store(None)
    return app_module._require_job_store()


def recover_stale_running_jobs(*, stale_after_seconds: int | None = None) -> int:
    if stale_after_seconds is None or stale_after_seconds <= 0:
        return 0
    app_module.create_app()
    job_store = _runtime_job_store()
    return int(job_store.recover_stale_running(stale_after_seconds=stale_after_seconds))


def process_next_job_once(*, stale_after_seconds: int | None = None) -> bool:
    app_module.create_app()
    job_store = _runtime_job_store()
    if stale_after_seconds is not None and stale_after_seconds > 0:
        job_store.recover_stale_running(stale_after_seconds=stale_after_seconds)
    job = job_store.claim_next_queued()
    if job is None:
        return False
    jobs_service.run_job(job.job_id, job.type, dict(job.payload or {}))
    return True


def process_queued_jobs_batch(
    *, max_jobs: int | None = None, stale_after_seconds: int | None = None
) -> int:
    processed = 0
    while True:
        if max_jobs is not None and processed >= max_jobs:
            break
        if not process_next_job_once(stale_after_seconds=stale_after_seconds):
            break
        processed += 1
    return processed


def main() -> int:
    poll_seconds = float(os.getenv("RULFM_JOB_WORKER_POLL_SECONDS", "1.0") or "1.0")
    mode = (os.getenv("RULFM_JOB_WORKER_MODE", "batch") or "batch").strip().lower()
    raw_max_jobs = (os.getenv("RULFM_JOB_WORKER_MAX_JOBS_PER_RUN", "0") or "0").strip()
    raw_recovery_timeout = (
        os.getenv("RULFM_JOB_RUNNING_TIMEOUT_SECONDS", "900") or "900"
    ).strip()
    try:
        max_jobs = int(raw_max_jobs)
    except ValueError:
        max_jobs = 0
    try:
        recovery_timeout = int(raw_recovery_timeout)
    except ValueError:
        recovery_timeout = 900
    max_jobs_per_run = max_jobs if max_jobs > 0 else None
    stale_after_seconds = recovery_timeout if recovery_timeout > 0 else None
    run_once = os.getenv("RULFM_JOB_WORKER_RUN_ONCE", "0").strip().lower() in {"1", "true", "yes"}
    if run_once:
        processed = process_next_job_once(stale_after_seconds=stale_after_seconds)
        return 0 if processed else 1
    if mode == "daemon":
        processed = process_next_job_once(stale_after_seconds=stale_after_seconds)
        while True:
            if not processed:
                time.sleep(max(0.1, poll_seconds))
            processed = process_next_job_once(stale_after_seconds=stale_after_seconds)
    processed_count = process_queued_jobs_batch(
        max_jobs=max_jobs_per_run,
        stale_after_seconds=stale_after_seconds,
    )
    return 0 if processed_count >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())