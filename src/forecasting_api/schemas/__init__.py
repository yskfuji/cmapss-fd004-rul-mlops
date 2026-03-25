from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _bi(en: str, ja: str) -> str:
    return f"[EN] {en}\n[JA] {ja}"


class ErrorGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str
    prev_timestamp: str
    expected_timestamp: str
    timestamp: str


class ErrorDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    next_action: str | None = Field(
        None, description=_bi("Recommended next action for the caller.", "次の推奨アクション。")
    )
    error: str | None = Field(None, description=_bi("Single error detail.", "単一のエラー詳細。"))
    errors: list[dict[str, Any]] | None = Field(
        None,
        description=_bi("Structured validation errors.", "構造化されたバリデーションエラー。"),
    )
    algo: str | None = Field(
        None,
        description=_bi("Algorithm identifier related to the error.", "エラー対象のアルゴリズム識別子。"),
    )
    available_profiles: list[str] | None = Field(
        None,
        description=_bi("Available profile names.", "利用可能なプロファイル名。"),
    )
    gaps: list[ErrorGap] | None = Field(
        None,
        description=_bi(
            "Gap samples when missing timestamps are detected (missing_policy=error).",
            "欠損timestamp検出時のサンプル（missing_policy=error）。",
        ),
    )
    job_id: str | None = Field(None, description=_bi("Job identifier.", "ジョブID。"))
    max_points: int | None = Field(
        None,
        description=_bi("Maximum supported point count.", "許容される最大点数。"),
    )
    max_bytes: int | None = Field(
        None,
        description=_bi("Maximum supported request size in bytes.", "許容される最大リクエストサイズ。"),
    )
    limit: int | None = Field(
        None,
        description=_bi("Applied numeric limit.", "適用された数値上限。"),
    )
    missing_policy: str | None = Field(
        None,
        description=_bi("Applied missing timestamp policy.", "適用された欠損timestampポリシー。"),
    )
    model_id: str | None = Field(
        None,
        description=_bi("Model identifier related to the error.", "エラー対象のモデルID。"),
    )
    series_id: str | None = Field(
        None,
        description=_bi("Series identifier related to the error.", "エラー対象の系列ID。"),
    )
    status: str | None = Field(
        None,
        description=_bi("Job or workflow status.", "ジョブまたは処理の状態。"),
    )
    step_seconds: float | None = Field(
        None,
        description=_bi("Expected step in seconds.", "期待ステップ秒数。"),
    )
    window_seconds: int | None = Field(
        None,
        description=_bi("Rate-limit window in seconds.", "レート制限ウィンドウ秒数。"),
    )


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_code: str = Field(
        ...,
        description=_bi(
            "Stable error code (e.g., V01, V02, COST01).",
            "固定のエラーコード（例: V01, V02, COST01）。",
        ),
    )
    message: str = Field(
        ..., description=_bi("Human-readable error message.", "人向けのエラーメッセージ。")
    )
    details: ErrorDetails | None = Field(
        None, description=_bi("Optional error details.", "任意の詳細情報。")
    )
    request_id: str | None = Field(
        None,
        description=_bi("Request ID for support correlation.", "サポート照合用のリクエストID。"),
    )


class TimeSeriesRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(..., description=_bi("Series identifier.", "系列ID。"))
    timestamp: datetime = Field(
        ...,
        description=_bi(
            "Timestamp (ISO 8601). Must be ascending within each series.",
            "タイムスタンプ（ISO 8601）。系列内で昇順必須。",
        ),
    )
    y: float = Field(..., description=_bi("Target value.", "目的変数値。"))
    x: dict[str, float | bool | str] | None = Field(
        None,
        description=_bi("Optional features (key-value).", "任意の特徴量（key-value）。"),
    )


class ForecastOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    missing_policy: Literal["ignore", "error"] | None = Field(
        None,
        description=_bi(
            "How to handle missing timestamps (ignore or error).",
            "欠損timestampの扱い（ignore/error）。",
        ),
    )
    include_explanation: bool | None = Field(
        None,
        description=_bi(
            "Include explanation payloads when available.",
            "利用可能な場合に explanation payload を含める。",
        ),
    )
    include_uncertainty_components: bool | None = Field(
        None,
        description=_bi(
            "Include detailed uncertainty components when available.",
            "利用可能な場合に詳細な uncertainty component を含める。",
        ),
    )


class ForecastRequest(BaseModel):
    horizon: int = Field(
        ..., ge=1, description=_bi("Forecast horizon (steps).", "予測期間（ステップ数）。")
    )
    frequency: str | None = Field(
        None,
        description=_bi("Time interval (e.g., 1d, 1h, 15min).", "時間間隔（例: 1d, 1h, 15min）。"),
    )
    quantiles: list[float] | None = Field(
        None,
        description=_bi(
            "Quantiles between 0 and 1 (e.g., 0.1, 0.5, 0.9). Not compatible with level.",
            "0〜1の分位点（例: 0.1, 0.5, 0.9）。levelと同時指定不可。",
        ),
    )
    level: list[float] | None = Field(
        None,
        description=_bi(
            "Prediction interval levels in percent (e.g., 80, 95). Not compatible with quantiles.",
            "予測区間の信頼水準（%）（例: 80, 95）。quantilesと同時指定不可。",
        ),
    )
    model_id: str | None = Field(
        None,
        description=_bi(
            "Model ID to pin execution. Empty uses default.",
            "使用モデルIDを固定。空ならデフォルト。",
        ),
    )
    options: ForecastOptions | None = Field(
        None, description=_bi("Optional forecast options.", "予測オプション（任意）。")
    )
    data: list[TimeSeriesRecord] = Field(
        ..., description=_bi("Input time series records.", "入力時系列レコード。")
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "horizon": 3,
                    "frequency": "1d",
                    "quantiles": [0.1, 0.5, 0.9],
                    "data": [
                        {"series_id": "s1", "timestamp": "2020-01-01T00:00:00Z", "y": 10.0},
                        {"series_id": "s1", "timestamp": "2020-01-02T00:00:00Z", "y": 11.0},
                        {"series_id": "s1", "timestamp": "2020-01-03T00:00:00Z", "y": 12.0},
                    ],
                }
            ]
        },
    )


class ForecastPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(..., description=_bi("Series identifier.", "系列ID。"))
    timestamp: datetime = Field(
        ..., description=_bi("Timestamp (ISO 8601).", "タイムスタンプ（ISO 8601）。")
    )
    point: float = Field(
        ..., description=_bi("Point forecast (median/mean, etc.).", "点予測（中央値/平均など）。")
    )
    quantiles: dict[str, float] | None = Field(
        None, description=_bi("Quantile forecasts keyed by quantile.", "分位点ごとの予測。")
    )
    intervals: list[dict[str, float]] | None = Field(
        None,
        description=_bi("Prediction intervals when level is provided.", "level指定時の予測区間。"),
    )
    uncertainty: dict[str, Any] | None = Field(
        None,
        description=_bi(
            "Optional uncertainty payload for this point.", "この点の uncertainty payload（任意）。"
        ),
    )
    explanation: dict[str, Any] | None = Field(
        None,
        description=_bi(
            "Optional explanation payload for this point.", "この点の explanation payload（任意）。"
        ),
    )


class ForecastResponse(BaseModel):
    forecasts: list[ForecastPoint] = Field(..., description=_bi("Forecast results.", "予測結果。"))
    warnings: list[str] | None = Field(
        None, description=_bi("Optional warnings.", "警告（任意）。")
    )
    calibration: dict[str, Any] | None = Field(
        None,
        description=_bi(
            "Optional calibration summary (e.g., split conformal settings and qhat).",
            "任意の校正サマリ（例: split conformal の設定と qhat）。",
        ),
    )
    residuals_evidence: dict[str, Any] | None = Field(
        None,
        description=_bi(
            "Optional residual distribution summary for evidence (small histogram + quantiles).",
            "Evidence用の残差分布サマリ（小さなヒストグラム＋分位点）。",
        ),
    )
    model_explainability: dict[str, Any] | None = Field(
        None,
        description=_bi(
            "Optional explainability schema summary.", "explainability schema summary（任意）。"
        ),
    )
    uncertainty_summary: dict[str, Any] | None = Field(
        None,
        description=_bi(
            "Optional uncertainty schema summary.", "uncertainty schema summary（任意）。"
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "forecasts": [
                        {
                            "series_id": "s1",
                            "timestamp": "2020-01-04T00:00:00Z",
                            "point": 12.4,
                            "quantiles": {"0.1": 11.0, "0.5": 12.4, "0.9": 13.7},
                        }
                    ]
                }
            ]
        },
    )


class JobCreateRequest(BaseModel):
    type: Literal["forecast", "train", "backtest"] = Field(
        ...,
        description=_bi("Job type.", "ジョブ種別。"),
    )
    payload: dict[str, Any] = Field(
        ...,
        description=_bi("Request payload for the job type.", "ジョブ種別に対応するリクエスト。"),
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "type": "forecast",
                    "payload": {
                        "horizon": 3,
                        "data": [
                            {"series_id": "s1", "timestamp": "2020-01-01T00:00:00Z", "y": 1.0}
                        ],
                    },
                },
            ]
        },
    )


class JobCreateResponse(BaseModel):
    job_id: str = Field(..., description=_bi("Job ID.", "ジョブID。"))
    status: Literal["queued", "running", "succeeded", "failed"] = Field(
        ..., description=_bi("Current status.", "現在の状態。")
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"job_id": "job_123", "status": "queued"}]},
    )


class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description=_bi("Job ID.", "ジョブID。"))
    status: Literal["queued", "running", "succeeded", "failed"] = Field(
        ..., description=_bi("Current status.", "現在の状態。")
    )
    progress: float | None = Field(
        None, ge=0, le=1, description=_bi("Progress (0..1).", "進捗（0〜1）。")
    )
    error: ErrorResponse | None = Field(
        None, description=_bi("Error details when failed.", "失敗時のエラー詳細。")
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"job_id": "job_123", "status": "running", "progress": 0.4},
                {
                    "job_id": "job_123",
                    "status": "failed",
                    "progress": 1.0,
                    "error": {
                        "error_code": "V01",
                        "message": "入力が不正です",
                        "request_id": "req_abc",
                    },
                },
            ]
        },
    )


class TrainRequest(BaseModel):
    algo: str | None = Field(
        None,
        description=_bi(
            "Training algorithm (optional). Overrides base_model when set (e.g., ridge_lags_v1, gbdt_hgb_v1, naive).",
            "学習アルゴリズム（任意）。指定時は base_model より優先（例: ridge_lags_v1, gbdt_hgb_v1, naive）。",
        ),
    )

    base_model: str | None = Field(
        None, description=_bi("Base model name (optional).", "ベースモデル名（任意）。")
    )
    model_name: str | None = Field(
        None, description=_bi("New model name (optional).", "新しいモデル名（任意）。")
    )
    training_hours: float | None = Field(
        None,
        ge=0.05,
        description=_bi("Training time in hours (>= 0.05).", "学習時間（時間、0.05以上）。"),
    )
    data: list[TimeSeriesRecord] = Field(..., description=_bi("Training data.", "学習データ。"))

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "base_model": "gbdt_hgb_v1",
                    "model_name": "my-model",
                    "training_hours": 0.5,
                    "data": [
                        {
                            "series_id": "s1",
                            "timestamp": "2020-01-01T00:00:00Z",
                            "y": 10.0,
                            "x": {"sensor_1": 0.5, "sensor_2": 1.2},
                        }
                    ],
                }
            ]
        },
    )


class TrainResponse(BaseModel):
    model_id: str = Field(..., description=_bi("Generated model ID.", "生成されたモデルID。"))
    message: str | None = Field(None, description=_bi("Optional message.", "任意のメッセージ。"))

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={"examples": [{"model_id": "model_abc", "message": "accepted"}]},
    )


class CmapssFd004PreprocessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    split: Literal["train", "test"] = Field(
        "train", description=_bi("FD004 split.", "FD004 の split。")
    )
    unit_ids: list[int] | None = Field(
        None, description=_bi("Optional unit IDs.", "対象 unit_id の一覧（任意）。")
    )
    window_size: int | None = Field(
        80,
        ge=8,
        description=_bi("Recent cycles to keep per unit.", "unit ごとに保持する直近 cycle 数。"),
    )
    task: Literal["forecast", "train", "backtest"] = Field(
        "forecast",
        description=_bi("Preferred downstream task.", "この前処理結果の想定タスク。"),
    )
    horizon: int = Field(
        20, ge=1, description=_bi("Default horizon to suggest.", "提案する既定 horizon。")
    )
    quantiles: list[float] | None = Field(
        None, description=_bi("Suggested quantiles.", "提案する quantiles。")
    )
    level: list[float] | None = Field(
        None, description=_bi("Suggested interval levels.", "提案する level。")
    )
    max_rul: int | None = Field(
        125, ge=1, description=_bi("Optional RUL cap.", "RUL の上限クリップ（任意）。")
    )


class CmapssFd004PayloadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str | None = Field(
        None, description=_bi("Named profile when applicable.", "該当する場合の profile 名。")
    )
    split: Literal["train", "test"] = Field(
        ..., description=_bi("FD004 split.", "FD004 の split。")
    )
    task: Literal["forecast", "train", "backtest"] = Field(
        ..., description=_bi("Suggested task.", "提案タスク。")
    )
    horizon: int = Field(..., ge=1, description=_bi("Suggested horizon.", "提案 horizon。"))
    frequency: str = Field(
        ...,
        description=_bi(
            "Cycle interval converted to API frequency.",
            "cycle 間隔を API frequency に変換した値。",
        ),
    )
    quantiles: list[float] | None = Field(
        None, description=_bi("Suggested quantiles.", "提案 quantiles。")
    )
    level: list[float] | None = Field(None, description=_bi("Suggested levels.", "提案 level。"))
    missing_policy: str = Field(
        ..., description=_bi("Suggested missing policy.", "提案 missing_policy。")
    )
    chartType: str = Field(
        ..., description=_bi("Suggested chart type for the GUI.", "GUI 向け推奨 chartType。")
    )
    value_unit: str = Field(..., description=_bi("Suggested unit label.", "推奨単位ラベル。"))
    records: list[dict[str, Any]] = Field(
        ..., description=_bi("API-compatible time series records.", "API 互換の時系列レコード。")
    )
    meta: dict[str, Any] = Field(
        ..., description=_bi("Preprocessing metadata.", "前処理メタデータ。")
    )
    description: dict[str, str] | None = Field(
        None, description=_bi("Localized sample description.", "ローカライズ済みサンプル説明。")
    )


class ModelCatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str = Field(..., description=_bi("Model ID.", "モデルID。"))
    created_at: str | None = Field(
        None, description=_bi("Creation time (ISO 8601).", "作成時刻（ISO 8601）。")
    )
    memo: str | None = Field(None, description=_bi("Optional memo.", "任意メモ。"))


class ModelCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelCatalogEntry] = Field(
        ..., description=_bi("Trained model catalog.", "学習済みモデル一覧。")
    )


class BacktestRequest(BaseModel):
    horizon: int = Field(
        ..., ge=1, description=_bi("Forecast horizon (steps).", "予測期間（ステップ数）。")
    )
    folds: int = Field(5, ge=1, le=20, description=_bi("Rolling folds.", "ローリング分割数。"))
    metric: Literal["mae", "rmse", "mape", "smape", "mase", "wape", "nasa_score"] = Field(
        "rmse",
        description=_bi("Primary metric.", "主評価指標。"),
    )
    model_id: str | None = Field(
        None,
        description=_bi(
            "Optional model ID to evaluate (if trained).",
            "評価に使う model_id（任意、学習済みがある場合）。",
        ),
    )
    data: list[TimeSeriesRecord] = Field(
        ..., description=_bi("Backtest data.", "バックテストデータ。")
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "horizon": 3,
                    "folds": 5,
                    "metric": "rmse",
                    "data": [{"series_id": "s1", "timestamp": "2020-01-01T00:00:00Z", "y": 10.0}],
                }
            ]
        },
    )


class BacktestResponse(BaseModel):
    metrics: dict[str, float] = Field(..., description=_bi("Overall metrics.", "全体評価指標。"))
    by_series: list[dict[str, Any]] | None = Field(
        None, description=_bi("Optional per-series ranking.", "系列別ランキング（任意）。")
    )
    by_horizon: list[dict[str, Any]] | None = Field(
        None, description=_bi("Optional per-horizon metrics.", "horizon別評価（任意）。")
    )
    by_fold: list[dict[str, Any]] | None = Field(
        None, description=_bi("Optional per-fold metrics.", "fold別評価（任意）。")
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "metrics": {"rmse": 1.23},
                    "by_series": [{"series_id": "s1", "value": 1.23}],
                    "by_horizon": [{"horizon": 1, "value": 1.1}],
                    "by_fold": [{"fold": 1, "value": 1.2}],
                }
            ]
        },
    )


class DriftReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_records: list[dict[str, Any]] | None = Field(
        None,
        description=_bi(
            "Optional baseline records for one-off comparison only. They are not persisted.",
            "単発比較用の任意ベースラインレコード。保存はされません。",
        ),
    )
    candidate_records: list[dict[str, Any]] = Field(
        ...,
        description=_bi(
            "Candidate records to compare against the baseline.",
            "ベースラインと比較する候補レコード。",
        ),
    )


class DriftBaselineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_records: list[dict[str, Any]] = Field(
        ...,
        description=_bi(
            "Representative baseline records to persist for future drift checks.",
            "今後のドリフト検査に使う代表ベースラインレコード。",
        ),
    )


class DriftBaselineResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "feature_count": 3,
                    "sample_size": 186,
                    "persisted": True,
                    "feature_summaries": [
                        {
                            "feature": "sensor_1",
                            "count": 62,
                            "binning_strategy": "adaptive_equal_width_without_empty_bins",
                            "requested_bin_count": 10,
                            "selected_bin_count": 9,
                        },
                        {
                            "feature": "sensor_2",
                            "count": 62,
                            "binning_strategy": "adaptive_equal_width_without_empty_bins",
                            "requested_bin_count": 10,
                            "selected_bin_count": 6,
                        },
                    ],
                }
            ]
        },
    )

    feature_count: int
    sample_size: int
    persisted: bool
    feature_summaries: list[dict[str, Any]]


class DriftBaselineStatusResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "baseline_exists": True,
                    "feature_count": 3,
                    "sample_size": 186,
                    "sufficient_samples": True,
                    "feature_summaries": [
                        {
                            "feature": "sensor_1",
                            "count": 62,
                            "binning_strategy": "adaptive_equal_width_without_empty_bins",
                            "requested_bin_count": 10,
                            "selected_bin_count": 9,
                        }
                    ],
                }
            ]
        },
    )

    baseline_exists: bool
    feature_count: int
    sample_size: int
    sufficient_samples: bool
    feature_summaries: list[dict[str, Any]]


class DriftReportResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "severity": "medium",
                    "drift_score": 0.28,
                    "sample_size": 62,
                    "feature_reports": [
                        {
                            "feature": "sensor_2",
                            "baseline_mean": 14.88,
                            "candidate_mean": 18.42,
                            "mean_delta": 3.54,
                            "population_stability_index": 0.28,
                            "baseline_binning_strategy": "adaptive_equal_width_without_empty_bins",
                            "baseline_requested_bin_count": 10,
                            "baseline_selected_bin_count": 6,
                        }
                    ],
                }
            ]
        },
    )

    severity: str
    drift_score: float
    sample_size: int
    feature_reports: list[dict[str, Any]]
