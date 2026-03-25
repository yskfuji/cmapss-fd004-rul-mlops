"""Benchmark artifact regression tests.

Verifies that the committed fd004_benchmark_summary.json artifact meets
minimum quality thresholds.  Mirrors the inline regression check in
ci-stable.yml (benchmark-artifact-regression job).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "forecasting_api"
    / "data"
    / "fd004_benchmark_summary.json"
)
TARGET_ROW = "GBDT_w30_f24"


@pytest.fixture(scope="module")
def benchmark_row() -> dict:
    data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    row = next((item for item in rows if item.get("model") == TARGET_ROW), None)
    if row is None:
        pytest.fail(f"{TARGET_ROW} row missing from committed artifact")
    return row


def test_rmse_within_threshold(benchmark_row: dict) -> None:
    rmse = float(benchmark_row.get("rmse") or 999.0)
    assert rmse <= 16.0, f"RMSE regression: {rmse} > 16.0"


def test_coverage_90_within_threshold(benchmark_row: dict) -> None:
    cov90 = float(benchmark_row.get("cov90") or 0.0)
    assert cov90 >= 0.90, f"cov90 regression: {cov90} < 0.90"


def test_bias_within_threshold(benchmark_row: dict) -> None:
    bias = abs(float(benchmark_row.get("bias") or 0.0))
    assert bias <= 5.0, f"bias regression: {bias} > 5.0"
