from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cmapss_family import build_cmapss_payload

_RAW_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{idx}" for idx in range(1, 22)],
]

_DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[3] / "datasets" / "CMAPSSData"


@dataclass(frozen=True)
class Fd004Profile:
    profile: str
    split: str
    task: str
    horizon: int
    quantiles: list[float] | None
    level: list[float] | None
    missing_policy: str
    chart_type: str
    value_unit: str
    window_size: int
    unit_ids: tuple[int, ...] | None
    description_en: str
    description_ja: str


_PROFILE_CONFIG: dict[str, Fd004Profile] = {
    "fd004_rul_forecast_unit01": Fd004Profile(
        profile="fd004_rul_forecast_unit01",
        split="train",
        task="forecast",
        horizon=20,
        quantiles=[0.1, 0.5, 0.9],
        level=None,
        missing_policy="ignore",
        chart_type="trend",
        value_unit="cycles",
        window_size=80,
        unit_ids=(1,),
        description_en="Single-engine recent degradation window for forecasting remaining useful life.",
        description_ja="単一エンジンの直近劣化区間から残寿命を予測するサンプルです。",
    ),
    "fd004_rul_forecast_unit05": Fd004Profile(
        profile="fd004_rul_forecast_unit05",
        split="train",
        task="forecast",
        horizon=20,
        quantiles=[0.1, 0.5, 0.9],
        level=None,
        missing_policy="ignore",
        chart_type="trend",
        value_unit="cycles",
        window_size=80,
        unit_ids=(5,),
        description_en="Alternative single-engine degradation window to compare RUL behaviour.",
        description_ja="別エンジンの劣化区間で RUL の挙動を比較するサンプルです。",
    ),
    "fd004_train_multiunit": Fd004Profile(
        profile="fd004_train_multiunit",
        split="train",
        task="train",
        horizon=15,
        quantiles=None,
        level=None,
        missing_policy="ignore",
        chart_type="trend",
        value_unit="cycles",
        # NOTE: window_size=90 is intentional for this UI/demo profile — it limits the
        # context window to the most recent 90 cycles for dashboard display purposes.
        # The GBDT benchmark trains with window_size=None (all cycles) via
        # build_fd004_payload(split="train", window_size=None) in the benchmark script.
        # Do NOT change this to None; it would break the UI sample payload size.
        window_size=90,
        unit_ids=None,
        description_en="Multi-engine training payload with sensor channels attached as exogenous features.",
        description_ja="複数エンジンの学習用サンプルで、センサ値を説明変数として含みます。",
    ),
    "fd004_backtest_multiunit": Fd004Profile(
        profile="fd004_backtest_multiunit",
        split="test",
        task="backtest",
        horizon=15,
        quantiles=[0.1, 0.5, 0.9],
        level=None,
        missing_policy="ignore",
        chart_type="trend",
        value_unit="cycles",
        window_size=100,
        unit_ids=None,
        description_en="Backtest-oriented payload across several engines to inspect degradation error by horizon.",
        description_ja="複数エンジンのバックテスト用サンプルで、ホライズン別誤差を確認できます。",
    ),
    "fd004_failure_watchlist": Fd004Profile(
        profile="fd004_failure_watchlist",
        split="train",
        task="forecast",
        horizon=12,
        quantiles=None,
        level=[80.0, 95.0],
        missing_policy="ignore",
        chart_type="band",
        value_unit="cycles",
        window_size=60,
        unit_ids=(11, 12, 13),
        description_en="Short-horizon multi-engine watchlist for near-failure monitoring.",
        description_ja="故障近傍の監視を想定した短期マルチエンジン予測サンプルです。",
    ),
}


def available_profiles() -> list[str]:
    return sorted(_PROFILE_CONFIG)

def _dataset_dir(dataset_dir: Path | None = None) -> Path:
    return dataset_dir or _DEFAULT_DATASET_DIR


def build_fd004_payload(
    *,
    split: str = "train",
    unit_ids: list[int] | None = None,
    window_size: int | None = None,
    task: str = "forecast",
    horizon: int = 20,
    quantiles: list[float] | None = None,
    level: list[float] | None = None,
    missing_policy: str = "ignore",
    chart_type: str = "trend",
    value_unit: str = "cycles",
    max_rul: int | None = 125,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    payload = build_cmapss_payload(
        dataset_id="fd004",
        split=split,
        unit_ids=unit_ids,
        window_size=window_size,
        task=task,
        horizon=horizon,
        quantiles=quantiles,
        level=level,
        missing_policy=missing_policy,
        chart_type=chart_type,
        value_unit=value_unit,
        max_rul=max_rul,
        dataset_dir=dataset_dir,
    )
    payload.pop("dataset", None)
    return payload


def build_fd004_profile_payload(profile: str, *, dataset_dir: Path | None = None) -> dict[str, Any]:
    key = str(profile or "").strip()
    if key not in _PROFILE_CONFIG:
        raise ValueError(f"unsupported FD004 profile: {profile}")
    cfg = _PROFILE_CONFIG[key]
    payload = build_fd004_payload(
        split=cfg.split,
        unit_ids=list(cfg.unit_ids) if cfg.unit_ids is not None else None,
        window_size=cfg.window_size,
        task=cfg.task,
        horizon=cfg.horizon,
        quantiles=cfg.quantiles,
        level=cfg.level,
        missing_policy=cfg.missing_policy,
        chart_type=cfg.chart_type,
        value_unit=cfg.value_unit,
        dataset_dir=dataset_dir,
    )
    payload.update(
        {
            "profile": cfg.profile,
            "description": {"en": cfg.description_en, "ja": cfg.description_ja},
        }
    )
    return payload