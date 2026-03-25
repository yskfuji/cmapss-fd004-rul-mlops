# Architecture Overview

**言語:** [英語版](overview.en.md) | 日英混在

CMAPSS FD004 RUL MLOps — システム全体図とコンポーネント設計

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client / GUI                            │
│           HTTP(S)  ·  API Key auth  ·  Static SPA               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      FastAPI Application                        │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  routers/   │  │  services/  │  │      app.py core        │  │
│  │             │  │             │  │                          │  │
│  │ forecast.py │→│forecast_svc │  │  TRAINED_MODELS (dict)   │  │
│  │ backtest.py │→│backtest_svc │  │  JOB_STORE (SQLite/PG)   │  │
│  │ train.py    │→│ train_svc   │  │  Rate limiter            │  │
│  │ jobs.py     │→│  jobs_svc   │  │  OpenAPI / middleware    │  │
│  │ monitoring.py│→│monitor_svc │  │  Security middleware     │  │
│  └─────────────┘  └──────┬──────┘  └─────────────────────────┘  │
│                          │                                       │
│        ┌──────────────┴──────────────────────────────┐          │
│        │ runtime.py / runtime_state                  │          │
│        │ request_audit / request_policy / dispatcher │          │
│        │ domain/stable_models + app-owned adapters   │          │
│        └─────────────────────────────────────────────┘          │
└─────────────┬───────────────────────────────────────────────────┘
              │
    ┌─────────┴──────────────────────────────────┐
    │               Infrastructure               │
    │                                            │
    │  ┌─────────────┐   ┌────────────────────┐  │
    │  │ src/models/ │   │  src/monitoring/   │  │
    │  │             │   │                    │  │
    │  │gbdt_pipeline│   │  drift_detector.py │  │
    │  │  (910 lines)│   │  PSI-based drift   │  │
    │  │  GBDT+Ridge │   │  detection         │  │
    │  │  ensemble   │   └─────────┬──────────┘  │
    │  └──────┬──────┘             │             │
    │         │              Prometheus           │
    │         │              + Grafana            │
    │  ┌──────▼──────┐   ┌────────────────────┐  │
    │  │  MLflow     │   │       DVC          │  │
    │  │  experiment │   │  data versioning   │  │
    │  │  tracking   │   │  pipeline (dvc.yaml│  │
    │  └─────────────┘   └────────────────────┘  │
    └────────────────────────────────────────────┘
```

---

## Component Responsibilities

### `src/forecasting_api/`

| モジュール | 責務 |
|---|---|
| `app.py` | FastAPI アプリ生成、ルーター/サービスの組み立て、ランタイム依存の遅延初期化 |
| `routers/` | HTTP レイヤー。リクエスト受け取り → サービス呼び出し → レスポンス返却のみ |
| `services/` | ビジネスロジック。`runtime.py` の DI パターンで `app.py` への逆参照を排除 |
| `services/runtime.py` | `*ServiceDeps` frozen dataclass でサービスへの依存を明示的に注入 |
| `runtime_state.py` | `TRAINED_MODELS` / `JOB_STORE` の lazy state を app から分離した保持層 |
| `request_audit.py` | API request 完了イベントを `enterprise.audit.AuditEvent` 形式で JSONL 出力 |
| `request_policy.py` | `X-Tenant-Id` / `X-Connection-Type` と env 設定を使って tenant/network policy を API access path に適用 |
| `request_approval.py` | train endpoint に対する two-person approval policy を `X-Approved-By` ヘッダで適用 |
| `job_dispatcher.py` | Job 状態登録と実行ディスパッチを分離する queue-first enqueuer 境界。デフォルト public mode は worker queue、in-process adapter は互換用 |
| `job_worker.py` | queued job を claim して実行する別プロセス worker |
| `domain/stable_models.py` | 公開 stable forecast/backtest helper 群。`app.py` は互換ラッパーと composition root を担当 |
| `schemas/` | Pydantic モデル（全リクエスト/レスポンス型を `__init__.py` に集約） |
| `errors.py` | `ApiError` 例外型。services → errors の一方向依存で循環を防止 |
| `config.py` | `env_bool()`, `env_int()`, `env_path()` 環境変数ヘルパー |
| `metrics.py` | Prometheus カウンター/ヒストグラム定義 |
| `mlflow_runs.py` | MLflow 実験追跡のラッパー |
| `model_registry_store.py` | SQLite/JSON ベースのモデルレジストリ永続化 |
| `job_store.py` | 非同期ジョブ状態管理（SQLite デフォルト、PostgreSQL オプション） |
| `middleware/security_headers.py` | CSP/HSTS 等のセキュリティヘッダー付与 |

### `src/models/`

| モジュール | 責務 |
|---|---|
| `gbdt_pipeline.py` | 特徴量エンジニアリング（rolling window、位相正規化）+ GBDT 学習パイプライン（LightGBM + CatBoost アンサンブル、HGB 分位点） |
| `registry.py` | モデルバンドルの保存・読み込みインターフェース |

### `src/monitoring/`

| モジュール | 責務 |
|---|---|
| `drift_detector.py` | PSI（Population Stability Index）ベースの特徴量ドリフト検出。ベースライン要約の保存・読み込み・比較 |

### `src/enterprise/`

| モジュール | 責務 |
|---|---|
| `audit.py` | 共有 JSONL 監査イベント型。公開 API の request audit sink でも利用 |
| `iam.py` | 承認・ブレークグラス判定の最小実装。two-person approval は `request_approval.py` 経由で `/v1/train` に接続 |
| `network.py` | IP 許可リストと private 接続判定の最小実装。`request_policy.py` 経由で公開 API access path に接続 |
| `tenancy.py` | tenant_id バリデーションの最小実装。`request_policy.py` 経由で公開 API access path に接続 |
| `portability.py` | ポータビリティ要求向け監査イベント生成の最小実装 |

---

## Data Flow

### 推論リクエスト（同期）

```
POST /v1/forecast
  │
  ▼
routers/forecast.py         # リクエストバリデーション、auth、tenant/network policy
  │
  ▼
services/forecast_service.py  # ビジネスロジック（モデル選択、周波数推定）
  │
  ├── TRAINED_MODELS[model_id]    # メモリ上のモデルバンドル参照
  │
  ├── domain/stable_models.py      # stable helper 群
  │
  ├── gbdt_pipeline.predict_gbdt()  # GBDT 推論パス
  │     └── 特徴量生成 → LightGBM + CatBoost アンサンブル → 予測区間
  │
  └── ForecastResponse (Pydantic)
        └── HTTP 200 JSON
```

### 学習リクエスト（非同期）

```
POST /v1/train
  │
  ▼
routers/train.py
  │
  ▼
request_approval.py          # optional two-person approval
  │
  ▼
services/train_service.py     # ジョブ登録
  │
  ▼
JOB_STORE.create_job()        # SQLite/PostgreSQL にジョブ状態保存
  │
  ▼
job_dispatcher.JobEnqueuer.enqueue()
  │
  ▼ (default public mode: separate worker process)
job_worker.py claims queued job
  │
  ▼
gbdt_pipeline.fit_gbdt_pipeline()   # 学習ループ
  │
  ├── MLflow: log_params / log_metrics / log_artifact
  ├── model_registry_store.save_models()
  └── JOB_STORE.update_status("completed")

GET /v1/jobs/{job_id}         # ポーリングでジョブ状態確認
```

### ドリフト検出フロー

```
POST /v1/monitoring/drift/baseline   # ベースライン登録
  │
  ▼
monitoring_service.persist_drift_baseline()
  └── drift_detector.summarize_baseline()  # PSI ビン境界計算
        └── drift_baseline.json に永続化

POST /v1/monitoring/drift/report     # ドリフトレポート生成
  │
  ▼
monitoring_service.generate_drift_report()
  └── drift_detector.detect()         # PSI スコア計算
        ├── feature_scores → Prometheus メトリクス記録
  └── DriftReportResponse (severity: low / medium / high)
```

---

## Dependency Direction

```
routers/  ──▶  services/  ──▶  errors.py
    │               │               ▲
    └──▶  schemas/  │           app.py
                    ▼
               runtime.py (DI dataclasses)
                    │
              (injected from app.py at startup)
```

**設計原則**: `services/` は `app.py` を逆参照しない。依存は常に上位レイヤーから下位レイヤーへの一方向。
`errors.py` は最下層の独立モジュールで、どこからでも import 可能。
stable forecast / backtest helper 群は `domain/stable_models.py` へ移され、`app.py` には互換ラッパーと composition root が残る。
ただし学習系 helper や一部の large branch はまだ `app.py` に残っており、完全な分離は継続課題。

詳細設計の背景は [docs/adr/](../adr/) を参照。

---

## Key Technology Choices

| 技術 | 採用理由 |
|---|---|
| FastAPI | 非同期対応、Pydantic による型強制、OpenAPI 自動生成 |
| LightGBM + CatBoost ensemble | 単一モデルより汎化性能が高く、推論速度も実用的 ([ADR 0005](../adr/0005-gbdt-full-cycle-training.md)) |
| PSI drift detection | 統計的解釈がしやすく、ビン境界をベースライン時点で固定できる |
| MLflow (file-backed) | DVC との責務分離：MLflow = 実験メタデータ、DVC = データ/モデル成果物 ([ADR 0003](../adr/0003-mlflow-dvc-boundary.md)) |
| Hybrid soft-gate routing | GBDT と Torch モデルを条件によって重み付けで切り替える ([ADR 0001](../adr/0001-hybrid-soft-gate-routing.md)) |
| SQLite/PG Job Store + queue-first enqueuer | Job 状態は SQLite デフォルト・PostgreSQL オプションで保存。実行ディスパッチは `JobEnqueuer` 境界で分離し、公開デフォルトは別 worker が queued job を claim する方式。in-process `BackgroundTasks` adapter は互換モード ([ADR 0002](../adr/0002-inmemory-job-store.md)) |
| GCP Cloud Run | API と worker を分離しやすいサーバーレス実行基盤。Workload Identity Federation により CI/CD 側で長期鍵を減らせる |

---

## CI/CD Pipeline

```
push to main
  │
  ▼
ci-stable (GitHub Actions)
  ├── lint (ruff)
  ├── typecheck (mypy — critical modules + staged service modules)
  ├── security (pip-audit / npm audit / secret scan)
  ├── test (pytest -m "not experimental", coverage ≥ 72%)
  ├── benchmark (build_fd004_benchmark_summary.py)
  ├── benchmark-artifact-regression (pytest tests/regression/)
  ├── dvc-dry-run (dvc repro --dry)
  ├── mlflow-smoke (file-backed real run)
  ├── postgres-job-store-compose (docker compose integration test)
  └── e2e (Playwright)
        │
        └── (on success)
              ▼
           cd (GitHub Actions)
             ├── build Docker image → push GHCR (sha-tagged)
             └── deploy → GCP Cloud Run → smoke test /health
```

---

## Roadmap

実装済み・計画中の詳細は [README.md Architecture roadmap](../../README.md#architecture-roadmap) を参照。
