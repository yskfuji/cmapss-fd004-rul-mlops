import sys
from types import ModuleType
from datetime import UTC, datetime, timedelta

import pytest
from forecasting_api import job_store as job_store_module
from forecasting_api.job_store import PostgresJobStore, SqliteJobStore, build_job_store


def test_sqlite_job_store_persists_created_job(tmp_path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.db")

    created = store.create("forecast", {"horizon": 1})
    loaded = store.get(created.job_id)

    assert loaded is not None
    assert loaded.job_id == created.job_id
    assert loaded.type == "forecast"
    assert loaded.status == "queued"
    assert loaded.payload == {"horizon": 1}
    assert loaded.progress == 0.0


def test_sqlite_job_store_persists_success_result_across_instances(tmp_path) -> None:
    first_store = SqliteJobStore(tmp_path / "jobs.db")
    created = first_store.create("train", {"algo": "naive"})
    first_store.set_running(created.job_id, progress=0.25)
    first_store.set_succeeded(created.job_id, {"model_id": "model_123"})

    second_store = SqliteJobStore(tmp_path / "jobs.db")
    loaded = second_store.get(created.job_id)

    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.progress == 1.0
    assert loaded.result == {"model_id": "model_123"}
    assert loaded.error is None


def test_sqlite_job_store_persists_failure_payload(tmp_path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.db")
    created = store.create("backtest", {"folds": 2})
    store.set_failed(created.job_id, {"error_code": "V01", "message": "bad input"})

    loaded = store.get(created.job_id)

    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.result is None
    assert loaded.error == {"error_code": "V01", "message": "bad input"}


def test_sqlite_job_store_recovers_stale_running_job(tmp_path) -> None:
    store = SqliteJobStore(tmp_path / "jobs.db")
    created = store.create("forecast", {"horizon": 1})
    store.set_running(created.job_id, progress=0.5)

    import sqlite3

    with sqlite3.connect(str(tmp_path / "jobs.db")) as conn:
        conn.execute(
            "UPDATE jobs SET updated_at = ? WHERE job_id = ?",
            ("2000-01-01 00:00:00", created.job_id),
        )
        conn.commit()

    recovered = store.recover_stale_running(stale_after_seconds=60)
    loaded = store.get(created.job_id)

    assert recovered == 1
    assert loaded is not None
    assert loaded.status == "queued"
    assert loaded.progress == 0.0


def test_build_job_store_returns_sqlite_by_default(tmp_path) -> None:
    store = build_job_store(sqlite_db_path=tmp_path / "jobs.db")

    assert isinstance(store, SqliteJobStore)


def test_build_job_store_requires_dsn_for_postgres(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires a PostgreSQL DSN"):
        build_job_store(sqlite_db_path=tmp_path / "jobs.db", backend="postgres")


def test_build_job_store_selects_postgres_backend(monkeypatch, tmp_path) -> None:
    class _FakePostgresStore:
        def __init__(self, dsn: str) -> None:
            self.dsn = dsn

    monkeypatch.setattr(job_store_module, "PostgresJobStore", _FakePostgresStore)

    store = build_job_store(
        sqlite_db_path=tmp_path / "jobs.db",
        backend="postgres",
        postgres_dsn="postgresql://user:pass@localhost:5432/rulfm",
    )

    assert isinstance(store, _FakePostgresStore)
    assert store.dsn.endswith("/rulfm")


def test_postgres_json_payload_wraps_dict(monkeypatch) -> None:
    fake_json_module = ModuleType("psycopg.types.json")
    fake_json_module.Jsonb = lambda value: ("jsonb", value)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", fake_json_module)

    payload = job_store_module._postgres_json_payload({"ok": True})

    assert payload == ("jsonb", {"ok": True})


def test_postgres_json_payload_raises_when_json_support_missing(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "psycopg.types.json", raising=False)

    real_import = __import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psycopg.types.json":
            raise ImportError("missing psycopg.types.json")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    with pytest.raises(RuntimeError, match="requires psycopg JSON support"):
        job_store_module._postgres_json_payload({"ok": True})


def test_decode_payload_returns_none_for_invalid_or_non_dict_json() -> None:
    assert job_store_module._decode_payload("{") is None
    assert job_store_module._decode_payload('["not", "an", "object"]') is None


def test_normalize_payload_returns_none_for_non_dict_values() -> None:
    assert job_store_module._normalize_payload(None) is None
    assert job_store_module._normalize_payload(["not-a-dict"]) is None


def test_postgres_job_store_crud_with_fake_psycopg(monkeypatch) -> None:
    class _FakeCursor:
        def __init__(self, conn) -> None:
            self._conn = conn
            self.fetchone_value = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query: str, params=None) -> None:
            normalized = " ".join(query.split())
            self._conn.queries.append((normalized, params))
            if normalized.startswith("SELECT"):
                self.fetchone_value = self._conn.fetchone_value

        def fetchone(self):
            return self.fetchone_value

    class _FakeConnection:
        def __init__(self) -> None:
            self.queries: list[tuple[str, object]] = []
            self.commits = 0
            self.fetchone_value = (
                "job-123",
                "forecast",
                "succeeded",
                {"horizon": 1},
                1.0,
                {"result": 1},
                None,
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def cursor(self):
            return _FakeCursor(self)

        def commit(self) -> None:
            self.commits += 1

    fake_conn = _FakeConnection()
    fake_psycopg = ModuleType("psycopg")
    fake_psycopg.connect = lambda dsn: fake_conn
    fake_json_module = ModuleType("psycopg.types.json")
    fake_json_module.Jsonb = lambda value: ("jsonb", value)
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.types.json", fake_json_module)

    store = PostgresJobStore("postgresql://postgres:postgres@localhost:5432/test")
    created = store.create("forecast", {"horizon": 1})
    loaded = store.get("job-123")
    store.set_running(created.job_id, progress=0.25)
    store.set_succeeded(created.job_id, {"result": 2})
    store.set_failed(created.job_id, {"error_code": "V01"})
    recovered = store.recover_stale_running(stale_after_seconds=300)

    assert created.type == "forecast"
    assert loaded is not None
    assert loaded.payload == {"horizon": 1}
    assert loaded.result == {"result": 1}
    assert recovered >= 0
    assert fake_conn.commits >= 5
    assert any("INSERT INTO jobs" in query for query, _params in fake_conn.queries)
    assert any("SELECT job_id, type, status" in query for query, _params in fake_conn.queries)
    assert any("UPDATE jobs SET status = %s" in query for query, _params in fake_conn.queries)


def test_postgres_job_store_recovers_stale_running_jobs(monkeypatch) -> None:
    class _FakeCursor:
        def __init__(self, conn) -> None:
            self._conn = conn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query: str, params=None) -> None:
            normalized = " ".join(query.split())
            self._conn.queries.append((normalized, params))
            if normalized.startswith("UPDATE jobs"):
                self.rowcount = 2

        @property
        def rowcount(self) -> int:
            return self._conn.rowcount

        @rowcount.setter
        def rowcount(self, value: int) -> None:
            self._conn.rowcount = value

    class _FakeConnection:
        def __init__(self) -> None:
            self.queries: list[tuple[str, object]] = []
            self.commits = 0
            self.rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def cursor(self):
            return _FakeCursor(self)

        def commit(self) -> None:
            self.commits += 1

    fake_conn = _FakeConnection()
    fake_psycopg = ModuleType("psycopg")
    fake_psycopg.connect = lambda dsn: fake_conn
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    store = PostgresJobStore("postgresql://postgres:postgres@localhost:5432/test")
    recovered = store.recover_stale_running(stale_after_seconds=600)

    assert recovered == 2
    assert any("WHERE status = 'running'" in query for query, _ in fake_conn.queries)


def test_postgres_job_store_get_returns_none_when_job_missing(monkeypatch) -> None:
    class _FakeCursor:
        def __init__(self, conn) -> None:
            self._conn = conn
            self.fetchone_value = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, query: str, params=None) -> None:
            normalized = " ".join(query.split())
            self._conn.queries.append((normalized, params))
            if normalized.startswith("SELECT"):
                self.fetchone_value = None

        def fetchone(self):
            return self.fetchone_value

    class _FakeConnection:
        def __init__(self) -> None:
            self.queries: list[tuple[str, object]] = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def cursor(self):
            return _FakeCursor(self)

        def commit(self) -> None:
            return None

    fake_psycopg = ModuleType("psycopg")
    fake_psycopg.connect = lambda dsn: _FakeConnection()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    store = PostgresJobStore("postgresql://postgres:postgres@localhost:5432/test")

    assert store.get("missing-job") is None


def test_postgres_job_store_connect_error_is_wrapped(monkeypatch) -> None:
    fake_psycopg = ModuleType("psycopg")
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.delitem(sys.modules, "psycopg.types.json", raising=False)
    del fake_psycopg

    real_import = __import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psycopg":
            raise ImportError("missing psycopg")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    with pytest.raises(RuntimeError, match="requires psycopg"):
        PostgresJobStore("postgresql://postgres:postgres@localhost:5432/test")