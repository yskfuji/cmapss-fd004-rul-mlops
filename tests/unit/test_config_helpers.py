from __future__ import annotations

from pathlib import Path

from forecasting_api.config import env_bool, env_first, env_int, env_path


def test_env_first_prefers_first_non_empty(monkeypatch):
    monkeypatch.setenv("A_KEY", "")
    monkeypatch.setenv("B_KEY", "value-b")
    monkeypatch.setenv("C_KEY", "value-c")

    assert env_first("A_KEY", "B_KEY", "C_KEY") == "value-b"


def test_env_path_uses_default_when_not_set(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("PATH_OVERRIDE", raising=False)
    default = tmp_path / "default.json"

    assert env_path(default, "PATH_OVERRIDE") == default


def test_env_bool_and_env_int_bounds(monkeypatch):
    monkeypatch.setenv("FLAG_YES", "true")
    monkeypatch.setenv("FLAG_NO", "no")
    monkeypatch.setenv("INT_BAD", "abc")
    monkeypatch.setenv("INT_LOW", "1")
    monkeypatch.setenv("INT_HIGH", "999")

    assert env_bool("FLAG_YES", default=False) is True
    assert env_bool("FLAG_NO", default=True) is False
    assert env_int("INT_BAD", default=10, min_value=5, max_value=20) == 10
    assert env_int("INT_LOW", default=10, min_value=5, max_value=20) == 5
    assert env_int("INT_HIGH", default=10, min_value=5, max_value=20) == 20
