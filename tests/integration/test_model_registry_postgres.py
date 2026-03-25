from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from forecasting_api.model_registry_store import load_models, save_model


def _postgres_dsn() -> str | None:
    return os.getenv("RULFM_TEST_CLOUDSQL_POSTGRES_DSN") or os.getenv("RULFM_TEST_POSTGRES_DSN")


pytestmark = pytest.mark.skipif(
    not _postgres_dsn(),
    reason=(
        "RULFM_TEST_CLOUDSQL_POSTGRES_DSN or RULFM_TEST_POSTGRES_DSN is required "
        "for PostgreSQL model registry integration tests"
    ),
)


def _purge_models(dsn: str, *model_ids: str) -> None:
    import psycopg

    with psycopg.connect(dsn) as conn:
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
            cur.execute(
                "DELETE FROM trained_models WHERE model_id = ANY(%s)",
                (list(model_ids),),
            )
        conn.commit()


def test_postgres_model_registry_persists_entries_across_instances() -> None:
    dsn = str(_postgres_dsn())
    model_id = f"registry-persist-{uuid.uuid4().hex}"
    _purge_models(dsn, model_id)
    try:
        save_model(
            {
                "model_id": model_id,
                "created_at": "2026-03-25T00:00:00+00:00",
                "memo": "cloudsql-persist-check",
                "algo": "gbdt_hgb_v1",
                "pooled_residuals": [0.1, 0.2, 0.3],
            },
            db_path=Path("/tmp/ignored.db"),
            backend="postgres",
            postgres_dsn=dsn,
        )

        loaded = load_models(
            db_path=Path("/tmp/ignored.db"),
            backend="postgres",
            postgres_dsn=dsn,
        )

        assert model_id in loaded
        assert loaded[model_id]["memo"] == "cloudsql-persist-check"
        assert loaded[model_id]["algo"] == "gbdt_hgb_v1"
        assert loaded[model_id]["pooled_residuals"] == [0.1, 0.2, 0.3]
    finally:
        _purge_models(dsn, model_id)


def test_postgres_model_registry_concurrent_upserts_keep_single_row() -> None:
    import psycopg

    dsn = str(_postgres_dsn())
    model_id = f"registry-concurrent-{uuid.uuid4().hex}"
    _purge_models(dsn, model_id)

    def _write_entry(idx: int) -> None:
        save_model(
            {
                "model_id": model_id,
                "created_at": f"2026-03-25T00:00:{idx:02d}+00:00",
                "memo": f"writer-{idx}",
                "algo": "naive",
                "pooled_residuals": [float(idx), float(idx) + 0.5],
            },
            db_path=Path("/tmp/ignored.db"),
            backend="postgres",
            postgres_dsn=dsn,
        )

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(_write_entry, range(8)))

        loaded = load_models(
            db_path=Path("/tmp/ignored.db"),
            backend="postgres",
            postgres_dsn=dsn,
        )
        assert model_id in loaded
        assert loaded[model_id]["model_id"] == model_id
        assert str(loaded[model_id]["memo"]).startswith("writer-")

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM trained_models WHERE model_id = %s",
                    (model_id,),
                )
                row_count = int(cur.fetchone()[0])

        assert row_count == 1
    finally:
        _purge_models(dsn, model_id)