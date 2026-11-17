from pathlib import Path

import pytest

from forecasting_api import cmapss_family


def setup_function() -> None:
    # Only the low-level file readers are cached; _rul_by_unit recomputes on demand.
    cmapss_family._read_split.cache_clear()
    cmapss_family._read_test_terminal_rul.cache_clear()


def teardown_function() -> None:
    # Keep cache cleanup symmetric so temporary split fixtures never leak across tests.
    cmapss_family._read_split.cache_clear()
    cmapss_family._read_test_terminal_rul.cache_clear()


def _write_split(path: Path, rows: list[tuple[int, int, float]]) -> None:
    lines = []
    for unit_id, cycle, sensor_base in rows:
        values = [float(unit_id), float(cycle), 1.0, 0.2, 3.0]
        values.extend(sensor_base + idx for idx in range(1, 22))
        lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_available_datasets_and_normalization() -> None:
    assert cmapss_family.available_cmapss_datasets() == ["fd001", "fd002", "fd003", "fd004"]
    assert cmapss_family._normalize_dataset_id(" FD004 ") == "fd004"
    with pytest.raises(ValueError, match="unsupported CMAPSS dataset"):
        cmapss_family._normalize_dataset_id("fd999")


def test_read_split_validates_split_and_file_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported split"):
        cmapss_family._read_split("fd004", "dev", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="file not found"):
        cmapss_family._read_split("fd004", "train", str(tmp_path))

    broken = tmp_path / "train_FD004.txt"
    broken.write_text("1 1 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected FD004 column count"):
        cmapss_family._read_split("fd004", "train", str(tmp_path))


def test_read_split_and_rul_builders_for_train_and_test(tmp_path: Path) -> None:
    _write_split(
        tmp_path / "train_FD004.txt",
        [(1, 1, 10.0), (1, 2, 11.0), (2, 1, 20.0), (2, 2, 21.0)],
    )
    _write_split(
        tmp_path / "test_FD004.txt",
        [(1, 1, 30.0), (1, 2, 31.0), (2, 1, 40.0), (2, 2, 41.0)],
    )
    (tmp_path / "RUL_FD004.txt").write_text("5\n7\n", encoding="utf-8")

    rows = cmapss_family._read_split("fd004", "train", str(tmp_path))
    assert len(rows) == 4
    assert rows[0]["unit_id"] == 1
    assert rows[0]["cycle"] == 1

    train_rul = cmapss_family._rul_by_unit("fd004", "train", tmp_path)
    assert train_rul[1][1] == 1
    assert train_rul[1][2] == 0

    test_rul = cmapss_family._rul_by_unit("fd004", "test", tmp_path)
    assert test_rul[1][1] == 6
    assert test_rul[1][2] == 5
    assert test_rul[2][1] == 8


def test_build_cmapss_payload_filters_units_windows_and_caps_rul(tmp_path: Path) -> None:
    _write_split(
        tmp_path / "train_FD004.txt",
        [(1, 1, 10.0), (1, 2, 11.0), (1, 3, 12.0), (2, 1, 20.0), (2, 2, 21.0)],
    )
    payload = cmapss_family.build_cmapss_payload(
        dataset_id="fd004",
        split="train",
        unit_ids=[1],
        window_size=2,
        horizon=15,
        quantiles=[0.1, 0.9],
        dataset_dir=tmp_path,
        max_rul=1,
    )
    assert payload["dataset"] == "fd004"
    assert payload["split"] == "train"
    assert payload["horizon"] == 15
    assert payload["quantiles"] == [0.1, 0.9]
    assert payload["meta"]["unit_ids"] == [1]
    assert payload["meta"]["record_count"] == 2
    assert all(record["series_id"] == "fd004_engine_001" for record in payload["records"])
    assert max(record["y"] for record in payload["records"]) <= 1.0
    assert payload["records"][0]["timestamp"].endswith("Z")


def test_build_cmapss_payload_rejects_missing_units(tmp_path: Path) -> None:
    _write_split(tmp_path / "train_FD004.txt", [(1, 1, 10.0), (1, 2, 11.0)])
    with pytest.raises(ValueError, match="no matching FD004 units found"):
        cmapss_family.build_cmapss_payload(
            dataset_id="fd004",
            split="train",
            unit_ids=[999],
            dataset_dir=tmp_path,
        )