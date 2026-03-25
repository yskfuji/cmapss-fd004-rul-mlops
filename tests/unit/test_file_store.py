from __future__ import annotations

from forecasting_api import file_store


def test_load_json_returns_default_for_missing_and_invalid(tmp_path):
    missing = tmp_path / "missing.json"
    assert file_store.load_json(missing, {"ok": True}) == {"ok": True}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")
    assert file_store.load_json(invalid, [1, 2, 3]) == [1, 2, 3]

    valid = tmp_path / "valid.json"
    valid.write_text('{"ok": true}', encoding="utf-8")
    assert file_store.load_json(valid, {}) == {"ok": True}


def test_atomic_write_and_update_json_file(tmp_path):
    path = tmp_path / "store.json"
    file_store.atomic_write_text(path, '{"count": 1}')
    updated = file_store.update_json_file(
        path,
        default={"count": 0},
        updater=lambda payload: {"count": int(payload["count"]) + 1},
    )
    assert updated == {"count": 2}
    assert file_store.load_json(path, {}) == {"count": 2}


def test_exclusive_lock_falls_back_without_fcntl(monkeypatch, tmp_path):
    monkeypatch.setattr(file_store, "fcntl", None)
    with file_store.exclusive_lock(tmp_path / "example.json"):
        pass
