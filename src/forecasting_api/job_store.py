from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable
from uuid import uuid4

JobType = Literal["forecast", "train", "backtest"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class JobRecord:
    job_id: str
    type: JobType
    status: JobStatus
    payload: dict[str, Any] | None = None
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


@runtime_checkable
class JobStore(Protocol):
    def create(self, job_type: JobType, payload: dict[str, Any]) -> JobRecord: ...
    def get(self, job_id: str) -> JobRecord | None: ...
    def claim_next_queued(self) -> JobRecord | None: ...
    def recover_stale_running(self, *, stale_after_seconds: int) -> int: ...
    def set_running(self, job_id: str, progress: float = 0.0) -> None: ...
    def set_succeeded(self, job_id: str, result: dict[str, Any]) -> None: ...
    def set_failed(self, job_id: str, error: dict[str, Any]) -> None: ...


def _stale_cutoff_utc(*, stale_after_seconds: int) -> datetime:
    bounded = max(1, int(stale_after_seconds))
    return datetime.now(UTC) - timedelta(seconds=bounded)


def _sqlite_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT,
            progress REAL,
            result_json TEXT,
            error_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        if len(row) > 1
    }
    if "payload_json" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN payload_json TEXT")
    conn.commit()


def _decode_payload(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        payload = json.loads(value)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _row_to_job(row: tuple[Any, ...] | None) -> JobRecord | None:
    if row is None:
        return None
    return JobRecord(
        job_id=str(row[0]),
        type=cast(JobType, str(row[1])),
        status=cast(JobStatus, str(row[2])),
        payload=_decode_payload(row[3]),
        progress=float(row[4]) if row[4] is not None else None,
        result=_decode_payload(row[5]),
        error=_decode_payload(row[6]),
    )


def _normalize_payload(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _postgres_json_payload(value: dict[str, Any] | None) -> Any:
    if value is None:
        return None
    try:
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL job store requires psycopg JSON support. "
            "Install requirements-lock.txt dependencies."
        ) from exc
    return Jsonb(value)


class SqliteJobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        with _connect(self._db_path) as conn:
            _ensure_schema(conn)

    def create(self, job_type: JobType, payload: dict[str, Any]) -> JobRecord:
        job = JobRecord(
            job_id=uuid4().hex,
            type=job_type,
            status="queued",
            payload=_normalize_payload(payload),
            progress=0.0,
        )
        with _connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, type, status, payload_json, progress, result_json, error_json)
                VALUES (?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    job.job_id,
                    job.type,
                    job.status,
                    json.dumps(job.payload, ensure_ascii=True) if job.payload is not None else None,
                    job.progress,
                ),
            )
            conn.commit()
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with _connect(self._db_path) as conn:
            row = conn.execute(
                """
                SELECT job_id, type, status, payload_json, progress, result_json, error_json
                FROM jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        return _row_to_job(row)

    def claim_next_queued(self) -> JobRecord | None:
        with _connect(self._db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT job_id, type, status, payload_json, progress, result_json, error_json
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC, job_id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            job_id = str(row[0])
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    progress = ?,
                    result_json = NULL,
                    error_json = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                ("running", 0.0, job_id),
            )
            conn.commit()
        claimed = self.get(job_id)
        return claimed

    def recover_stale_running(self, *, stale_after_seconds: int) -> int:
        cutoff = _sqlite_timestamp(_stale_cutoff_utc(stale_after_seconds=stale_after_seconds))
        with _connect(self._db_path) as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    progress = 0.0,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'running'
                  AND updated_at < ?
                """,
                (cutoff,),
            )
            conn.commit()
        return int(cur.rowcount or 0)

    def set_running(self, job_id: str, progress: float = 0.0) -> None:
        self._update(job_id, status="running", progress=progress, result=None, error=None)

    def set_succeeded(self, job_id: str, result: dict[str, Any]) -> None:
        self._update(job_id, status="succeeded", progress=1.0, result=result, error=None)

    def set_failed(self, job_id: str, error: dict[str, Any]) -> None:
        self._update(job_id, status="failed", progress=1.0, result=None, error=error)

    def _update(
        self,
        job_id: str,
        *,
        status: JobStatus,
        progress: float | None,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        with _connect(self._db_path) as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    progress = ?,
                    result_json = ?,
                    error_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (
                    status,
                    progress,
                    json.dumps(result, ensure_ascii=True) if result is not None else None,
                    json.dumps(error, ensure_ascii=True) if error is not None else None,
                    job_id,
                ),
            )
            conn.commit()


class PostgresJobStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        with self._connect() as conn:
            self._ensure_schema(conn)

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL job store requires psycopg. Install requirements-lock.txt dependencies."
            ) from exc
        return psycopg.connect(self._dsn)

    def _ensure_schema(self, conn: Any) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress DOUBLE PRECISION,
                    result_json JSONB,
                    error_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                ALTER TABLE jobs
                ADD COLUMN IF NOT EXISTS payload_json JSONB
                """
            )
        conn.commit()

    def create(self, job_type: JobType, payload: dict[str, Any]) -> JobRecord:
        job = JobRecord(
            job_id=uuid4().hex,
            type=job_type,
            status="queued",
            payload=_normalize_payload(payload),
            progress=0.0,
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (job_id, type, status, payload_json, progress, result_json, error_json)
                    VALUES (%s, %s, %s, %s, %s, NULL, NULL)
                    """,
                    (job.job_id, job.type, job.status, _postgres_json_payload(job.payload), job.progress),
                )
            conn.commit()
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT job_id, type, status, payload_json, progress, result_json, error_json
                    FROM jobs
                    WHERE job_id = %s
                    """,
                    (job_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return JobRecord(
            job_id=str(row[0]),
            type=cast(JobType, str(row[1])),
            status=cast(JobStatus, str(row[2])),
            payload=_normalize_payload(row[3]),
            progress=float(row[4]) if row[4] is not None else None,
            result=_normalize_payload(row[5]),
            error=_normalize_payload(row[6]),
        )

    def claim_next_queued(self) -> JobRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH next_job AS (
                        SELECT job_id
                        FROM jobs
                        WHERE status = 'queued'
                        ORDER BY created_at ASC, job_id ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE jobs
                    SET status = 'running',
                        progress = 0.0,
                        result_json = NULL,
                        error_json = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id IN (SELECT job_id FROM next_job)
                    RETURNING job_id, type, status, payload_json, progress, result_json, error_json
                    """
                )
                row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return JobRecord(
            job_id=str(row[0]),
            type=cast(JobType, str(row[1])),
            status=cast(JobStatus, str(row[2])),
            payload=_normalize_payload(row[3]),
            progress=float(row[4]) if row[4] is not None else None,
            result=_normalize_payload(row[5]),
            error=_normalize_payload(row[6]),
        )

    def recover_stale_running(self, *, stale_after_seconds: int) -> int:
        cutoff = _stale_cutoff_utc(stale_after_seconds=stale_after_seconds)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'queued',
                        progress = 0.0,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'running'
                      AND updated_at < %s
                    """,
                    (cutoff,),
                )
                rowcount = int(getattr(cur, "rowcount", 0) or 0)
            conn.commit()
        return rowcount

    def set_running(self, job_id: str, progress: float = 0.0) -> None:
        self._update(job_id, status="running", progress=progress, result=None, error=None)

    def set_succeeded(self, job_id: str, result: dict[str, Any]) -> None:
        self._update(job_id, status="succeeded", progress=1.0, result=result, error=None)

    def set_failed(self, job_id: str, error: dict[str, Any]) -> None:
        self._update(job_id, status="failed", progress=1.0, result=None, error=error)

    def _update(
        self,
        job_id: str,
        *,
        status: JobStatus,
        progress: float | None,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = %s,
                        progress = %s,
                        result_json = %s,
                        error_json = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = %s
                    """,
                    (
                        status,
                        progress,
                        _postgres_json_payload(result),
                        _postgres_json_payload(error),
                        job_id,
                    ),
                )
            conn.commit()


def build_job_store(
    *, sqlite_db_path: Path, backend: str | None = None, postgres_dsn: str | None = None
) -> JobStore:
    selected_backend = (backend or "sqlite").strip().lower()
    if selected_backend == "postgres":
        if not postgres_dsn:
            raise ValueError("postgres backend requires a PostgreSQL DSN")
        return PostgresJobStore(postgres_dsn)
    return SqliteJobStore(sqlite_db_path)