from __future__ import annotations

import json
from pathlib import Path

from forecasting_api.model_registry_store import load_models, save_models


def test_save_and_load_models_with_sqlite(tmp_path: Path):
    db_path = tmp_path / "trained_models.db"
    models = {
        "model_a": {
            "model_id": "model_a",
            "created_at": "2026-03-24T00:00:00+00:00",
            "memo": "baseline",
            "algo": "naive",
            "pooled_residuals": [float(i) for i in range(600)],
        }
    }

    save_models(models, db_path=db_path)
    loaded = load_models(db_path=db_path)

    assert "model_a" in loaded
    assert loaded["model_a"]["algo"] == "naive"
    assert len(loaded["model_a"]["pooled_residuals"]) == 500
    assert loaded["model_a"]["pooled_residuals_truncated"] is True
    assert loaded["model_a"]["pooled_residuals_original_count"] == 600


def test_load_models_migrates_legacy_json(tmp_path: Path):
    db_path = tmp_path / "trained_models.db"
    legacy_json_path = tmp_path / "trained_models.json"
    legacy_payload = [
        {
            "model_id": "legacy_model",
            "created_at": "2026-03-24T00:00:00+00:00",
            "memo": "legacy",
            "algo": "ridge_lags_v1",
        }
    ]
    legacy_json_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    migrated = load_models(db_path=db_path, legacy_json_path=legacy_json_path)

    assert "legacy_model" in migrated
    assert migrated["legacy_model"]["algo"] == "ridge_lags_v1"

    loaded_from_db = load_models(db_path=db_path)
    assert "legacy_model" in loaded_from_db


def test_load_models_migrates_legacy_string_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "trained_models.db"
    legacy_json_path = tmp_path / "trained_models.json"
    legacy_json_path.write_text(json.dumps(["legacy_a", "legacy_b"]), encoding="utf-8")

    migrated = load_models(db_path=db_path, legacy_json_path=legacy_json_path)

    assert migrated["legacy_a"]["model_id"] == "legacy_a"
    assert migrated["legacy_b"]["memo"] == ""


def test_load_models_returns_empty_on_invalid_legacy_json(tmp_path: Path) -> None:
    db_path = tmp_path / "trained_models.db"
    legacy_json_path = tmp_path / "trained_models.json"
    legacy_json_path.write_text("{not-json}", encoding="utf-8")

    assert load_models(db_path=db_path, legacy_json_path=legacy_json_path) == {}


def test_save_models_skips_entries_without_model_id(tmp_path: Path) -> None:
    db_path = tmp_path / "trained_models.db"

    save_models({"broken": {"memo": "missing id"}}, db_path=db_path)

    assert load_models(db_path=db_path) == {}
