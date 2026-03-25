from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

from forecasting_api.logging_config import get_logger

_LOGGER = get_logger("model_registry_store")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _postgres_json_payload(value: dict[str, Any] | None) -> Any:
    if value is None:
        return None
    try:
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL model registry requires psycopg JSON support. "
            "Install requirements-lock.txt dependencies."
        ) from exc
    return Jsonb(value)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trained_models (
            model_id TEXT PRIMARY KEY,
            created_at TEXT,
            memo TEXT,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def _sanitize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    model_id = str(entry.get("model_id") or "").strip()
    if not model_id:
        return {}
    out: dict[str, Any] = {
        "model_id": model_id,
        "created_at": entry.get("created_at"),
        "memo": entry.get("memo", "") or "",
    }
    for key in (
        "algo",
        "base_model",
        "artifact",
        "context_len",
        "input_dim",
        "pooled_residuals",
        "pooled_residuals_original_count",
        "pooled_residuals_truncated",
        "state",
    ):
        if key in entry:
            out[key] = entry.get(key)
    pooled = out.get("pooled_residuals")
    if isinstance(pooled, list):
        pooled_list = cast(list[Any], pooled)
    else:
        pooled_list = None
    if pooled_list is not None and len(pooled_list) > 500:
        _LOGGER.warning(
            "Truncating pooled residuals before persistence",
            extra={"model_id": model_id, "residual_count": len(pooled_list), "kept": 500},
        )
        out["pooled_residuals"] = pooled_list[-500:]
        out["pooled_residuals_truncated"] = True
        out["pooled_residuals_original_count"] = len(pooled_list)
    return out


def _parse_legacy_json(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw_obj: object = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw_obj, list):
        return {}
    raw = cast(list[object], raw_obj)

    out: dict[str, dict[str, Any]] = {}
    for item in raw:
        if isinstance(item, str):
            model_id = item.strip()
            if model_id:
                out[model_id] = {
                    "model_id": model_id,
                    "created_at": None,
                    "memo": "",
                }
            continue
        if isinstance(item, dict):
            sanitized = _sanitize_entry(cast(dict[str, Any], item))
            model_id = str(sanitized.get("model_id") or "").strip()
            if model_id:
                out[model_id] = sanitized
    return out


def load_models(
    *,
    db_path: Path,
    legacy_json_path: Path | None = None,
    backend: str | None = None,
    postgres_dsn: str | None = None,
) -> dict[str, dict[str, Any]]:
    selected_backend = (backend or "sqlite").strip().lower()
    if selected_backend == "postgres":
        if not postgres_dsn:
            raise ValueError("postgres backend requires a PostgreSQL DSN")
        return _load_models_postgres(postgres_dsn)

    with _connect(db_path) as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT model_id, payload_json FROM trained_models ORDER BY created_at DESC"
        ).fetchall()
        if rows:
            models: dict[str, dict[str, Any]] = {}
            for model_id, payload_json in rows:
                try:
                    payload_obj: object = json.loads(str(payload_json))
                except Exception:
                    payload_obj = {}
                if not isinstance(payload_obj, dict):
                    payload = {}
                else:
                    payload = cast(dict[str, Any], payload_obj)
                payload.setdefault("model_id", str(model_id))
                sanitized = _sanitize_entry(payload)
                key = str(sanitized.get("model_id") or "").strip()
                if key:
                    models[key] = sanitized
            return models

    if legacy_json_path is None:
        return {}

    migrated = _parse_legacy_json(legacy_json_path)
    if migrated:
        save_models(migrated, db_path=db_path, backend=backend, postgres_dsn=postgres_dsn)
    return migrated


def save_models(
    models: dict[str, dict[str, Any]],
    *,
    db_path: Path,
    backend: str | None = None,
    postgres_dsn: str | None = None,
) -> None:
    selected_backend = (backend or "sqlite").strip().lower()
    if selected_backend == "postgres":
        if not postgres_dsn:
            raise ValueError("postgres backend requires a PostgreSQL DSN")
        _save_models_postgres(models, postgres_dsn=postgres_dsn)
        return

    with _connect(db_path) as conn:
        _ensure_schema(conn)
        for entry in models.values():
            _save_entry(conn, entry)
        conn.commit()


def _save_entry(conn: sqlite3.Connection, entry: dict[str, Any]) -> None:
    sanitized = _sanitize_entry(entry)
    model_id = str(sanitized.get("model_id") or "").strip()
    if not model_id:
        return
    conn.execute(
        """
        INSERT INTO trained_models (model_id, created_at, memo, payload_json, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(model_id)
        DO UPDATE SET
            created_at=excluded.created_at,
            memo=excluded.memo,
            payload_json=excluded.payload_json,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            model_id,
            sanitized.get("created_at"),
            str(sanitized.get("memo") or ""),
            json.dumps(sanitized, ensure_ascii=True),
        ),
    )


def save_model(
    entry: dict[str, Any],
    *,
    db_path: Path,
    backend: str | None = None,
    postgres_dsn: str | None = None,
) -> None:
    selected_backend = (backend or "sqlite").strip().lower()
    if selected_backend == "postgres":
        if not postgres_dsn:
            raise ValueError("postgres backend requires a PostgreSQL DSN")
        _save_model_postgres(entry, postgres_dsn=postgres_dsn)
        return

    with _connect(db_path) as conn:
        _ensure_schema(conn)
        _save_entry(conn, entry)
        conn.commit()


def _connect_postgres(dsn: str) -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL model registry requires psycopg. "
            "Install requirements-lock.txt dependencies."
        ) from exc
    return psycopg.connect(dsn)


def _ensure_schema_postgres(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trained_models (
                model_id TEXT PRIMARY KEY,
                created_at TEXT,
                memo TEXT,
                payload_json JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    conn.commit()


def _load_models_postgres(postgres_dsn: str) -> dict[str, dict[str, Any]]:
    with _connect_postgres(postgres_dsn) as conn:
        _ensure_schema_postgres(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT model_id, payload_json FROM trained_models ORDER BY updated_at DESC"
            )
            rows = cur.fetchall()
    models: dict[str, dict[str, Any]] = {}
    for model_id, payload_obj in rows:
        if not isinstance(payload_obj, dict):
            payload = {}
        else:
            payload = cast(dict[str, Any], payload_obj)
        payload.setdefault("model_id", str(model_id))
        sanitized = _sanitize_entry(payload)
        key = str(sanitized.get("model_id") or "").strip()
        if key:
            models[key] = sanitized
    return models


def _save_models_postgres(models: dict[str, dict[str, Any]], *, postgres_dsn: str) -> None:
    with _connect_postgres(postgres_dsn) as conn:
        _ensure_schema_postgres(conn)
        with conn.cursor() as cur:
            for entry in models.values():
                _save_entry_postgres(cur, entry)
        conn.commit()


def _save_model_postgres(entry: dict[str, Any], *, postgres_dsn: str) -> None:
    with _connect_postgres(postgres_dsn) as conn:
        _ensure_schema_postgres(conn)
        with conn.cursor() as cur:
            _save_entry_postgres(cur, entry)
        conn.commit()


def _save_entry_postgres(cur: Any, entry: dict[str, Any]) -> None:
    sanitized = _sanitize_entry(entry)
    model_id = str(sanitized.get("model_id") or "").strip()
    if not model_id:
        return
    cur.execute(
        """
        INSERT INTO trained_models (model_id, created_at, memo, payload_json, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT(model_id)
        DO UPDATE SET
            created_at=excluded.created_at,
            memo=excluded.memo,
            payload_json=excluded.payload_json,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            model_id,
            sanitized.get("created_at"),
            str(sanitized.get("memo") or ""),
            _postgres_json_payload(sanitized),
        ),
    )
