# ruff: noqa: I001
from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("RULFM_TEST_POSTGRES_DSN"),
    reason="RULFM_TEST_POSTGRES_DSN is required for PostgreSQL integration tests",
)


def test_postgres_job_store_persists_status_across_instances() -> None:
    from forecasting_api.job_store import PostgresJobStore

    dsn = os.environ["RULFM_TEST_POSTGRES_DSN"]
    first_store = PostgresJobStore(dsn)
    created = first_store.create("forecast", {})
    first_store.set_running(created.job_id, progress=0.5)
    first_store.set_succeeded(created.job_id, {"forecast_count": 3})

    second_store = PostgresJobStore(dsn)
    loaded = second_store.get(created.job_id)

    assert loaded is not None
    assert loaded.job_id == created.job_id
    assert loaded.status == "succeeded"
    assert loaded.progress == 1.0
    assert loaded.result == {"forecast_count": 3}
    assert loaded.error is None


def test_postgres_job_store_returns_none_for_missing_job() -> None:
    from forecasting_api.job_store import PostgresJobStore

    dsn = os.environ["RULFM_TEST_POSTGRES_DSN"]
    store = PostgresJobStore(dsn)

    assert store.get("missing-job-id") is None


def test_postgres_job_store_recovers_stale_running_job() -> None:
    import psycopg

    from forecasting_api.job_store import PostgresJobStore

    dsn = os.environ["RULFM_TEST_POSTGRES_DSN"]
    store = PostgresJobStore(dsn)
    created = store.create("forecast", {"horizon": 1})
    store.set_running(created.job_id, progress=0.5)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET updated_at = CURRENT_TIMESTAMP - INTERVAL '2 hours' WHERE job_id = %s",
                (created.job_id,),
            )
        conn.commit()

    recovered = store.recover_stale_running(stale_after_seconds=60)
    loaded = store.get(created.job_id)

    assert recovered >= 1
    assert loaded is not None
    assert loaded.status == "queued"
    assert loaded.progress == 0.0