# RULFM Runbook

**言語:** [英語](operations.md) | 日本語

## まずは最小構成から

まずは、目的に合った最小構成から始めてください。

| 目的 | 必要な env / dependency | 補足 |
|---|---|---|
| 公開 benchmark をローカルで再現 | `pip install -r requirements-lock.txt` | control-plane の追加設定は不要です |
| API demo をローカルで起動 | `RULFM_FORECASTING_API_KEY`, `uvicorn forecasting_api.app:create_app --factory` | operator や metrics の認証が必要な場合のみ bearer token を追加します |
| 耐久性のあるローカル control plane | PostgreSQL DSN と compose profile、または外部 Postgres | restart をまたいで jobs と registry を保持したい場合に使います |
| Cloud Run deployment | Terraform input、Workload Identity、runtime bucket、Cloud SQL DSN、smoke-test secret | 本番相当のフル構成です |

公開 demo と benchmark 再現だけが目的であれば、最初に該当する行だけで十分です。

## ローカル起動

1. 公開版の依存関係を `pip install -r requirements-lock.txt` でインストールします。
2. torch / hybrid の実験も行う場合に限り、`pip install -r requirements-experimental-lock.txt` を追加します。
3. `RULFM_FORECASTING_API_KEY` を export します。
4. bearer auth を使う場合は `RULFM_FORECASTING_API_BEARER_TOKEN` も export します。Local Compose では `local-metrics-token` が既定値です。
5. `docker compose up --build` もしくは Uvicorn で API を起動します。
6. `dvc pull` / `dvc push` を使う前に、remote が実環境の GCS bucket を向くよう認証します。

可変 state は `runtime/` に配置され、コミット済みの benchmark / reference artifact は `src/forecasting_api/data/` に置かれます。

## 環境変数ガイド

- `RULFM_FORECASTING_API_KEY` は、最小構成のローカル API で唯一必須となる application secret です。
- `RULFM_FORECASTING_API_BEARER_TOKEN` は、bearer-only の operator access や metrics scrape が必要な場合にのみ使います。
- tenant、network、approval は上位の control です。必要な検証を行う場合にだけ有効化してください。
- PostgreSQL DSN と persistent path override は、Cloud Run や multi-instance 配置では必須です。

## 実行モード

- `RULFM_JOB_EXECUTION_BACKEND=worker` が公開版の既定です。API は job を queue に積み、別の worker process がそれを実行します。
- Local Docker Compose は `RULFM_JOB_WORKER_MODE=daemon`、Cloud Run worker は `RULFM_JOB_WORKER_MODE=batch` を使います。
- `RULFM_JOB_RUNNING_TIMEOUT_SECONDS` は stale-running recovery を制御します。
- `RULFM_JOB_EXECUTION_BACKEND=inprocess` は、互換性確保または小規模なローカル demo 向けにのみ残しています。
- `RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL=1` で `/v1/train` に two-person approval を強制できます。
- approver group / subject を設定すると、`/v1/train` は OIDC claim ベースの承認フローに切り替わります。

## デプロイメント

- Terraform は既定で Cloud Run の ingress を internal load balancer に向けます。外部公開する場合は、別途 ingress path を用意してください。
- CD workflow は deploy 後に readiness を確認し、その後 `/health` を叩きます。
- health check を load balancer URL 経由にしたい場合は、GitHub Actions に `GCP_CLOUD_RUN_HEALTHCHECK_URL` を設定します。
- Cloud Run 起動時には、mutable state path が container 内のデフォルトのままだったり、control-plane backend が PostgreSQL でなかったりする場合に fail fast します。

Terraform stack が用意するもの:

- Cloud SQL PostgreSQL instance / database / user / DSN secret
- API と worker で共有する versioned GCS runtime bucket
- API 用 Cloud Run service
- queue worker 用 Cloud Run Job
- `jobs.run` を呼ぶ Cloud Scheduler trigger

## Terraform 検証

- ローカルに Terraform CLI がなくても、Docker の `hashicorp/terraform` を使って `infra/terraform` に対する `init` と `plan` を実行できます。
- GCS backend を使う本番向けの `plan` には、backend 設定と GCP Application Default Credentials が必要です。
- 認証前の静的検証だけであれば、backend を外した一時コピーで `init` と `plan` を実行し、provider schema と resource graph の整合性を確認してください。

## Cloud SQL 統合確認

`RULFM_TEST_CLOUDSQL_POSTGRES_DSN` を設定して次を実行します。

```bash
.venv/bin/python -m pytest --no-cov \
  tests/integration/test_job_store_postgres.py \
  tests/integration/test_model_registry_postgres.py -q
```

## ヘルスとメトリクス

- `GET /health` は API が応答していることと、experimental model が有効かどうかを返します。
- `GET /metrics` は Prometheus metrics を返し、認証が必要です。
- Docker Compose 利用時、Grafana は port 3000 で利用できます。

## ドリフトワークフロー

1. `POST /v1/monitoring/drift/baseline` で baseline を保存します。
2. `POST /v1/monitoring/drift/report` で candidate observation を送ります。
3. 単発の比較では `baseline_records` を使えますが、保存済みの baseline 自体は上書きしません。
4. severity が medium / high の場合は、model promotion の前に調査してください。

## プロモーションワークフロー

1. backtest または benchmark job から評価 metric を集めます。
2. `python -m scripts.promote_model MODEL_ID --metrics-file metrics.json` を実行します。
3. stage の参照を切り替える前に、promotion registry と MLflow artifact を確認します。

## ベンチマーク実行

### GBDT プリセット

| Preset | Candidate 数 | max_iter | 用途 |
|---|---|---|---|
| `full` | 6 | 最大 600 | 論文品質の結果 |
| `fast` | 2 | 最大 90 | スモークテスト / CI |

GBDT benchmark は全 engine cycle (`window_size=None`) で学習します。公開 checkpoint だけを再生成したい場合は、次を使います。

```bash
RULFM_BENCHMARK_STAGE=gbdt-only \
RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0 \
PYTHONPATH=src python scripts/build_fd004_benchmark_summary.py
```

## インシデント対応

- `/metrics` が更新されない場合は、`RULFM_METRICS_ENABLED=1` と Prometheus scrape を確認してください。
- drift alert が増えた場合は、baseline との分布差を見て promotion を止めます。
- worker crash 後に job が `running` のまま残る場合は、timeout と worker restart を確認してください。
- benchmark RMSE が 30 を超える場合は、`train_profile` と preset 設定をまず疑ってください。

## バックアップ / リストア

- SQLite のローカル state は、`runtime/trained_models.db`、`runtime/model_artifacts/`、`runtime/request_audit_log.jsonl`、`runtime/drift_baseline.json`、`runtime/model_promotions.json` をまとめて退避します。
- Cloud Run の control plane では、Cloud SQL dump または PITR と GCS object versioning を併用します。
- restore 後は worker を再起動し、queued job が再び terminal state へ進むことを確認してください。

## ロールバック

- application rollback には、CD pipeline が出力した直前の sha-tagged GHCR image を使います。
- state rollback では、Cloud SQL snapshot と runtime bucket object version を必ず揃えてください。
- promotion rollback では、history を編集せず、新しい promotion decision を追記します。

## ランタイム / ビジネス KPI

| KPI | 目標 / 意図 |
|---|---|
| backtest_rmse | full public GBDT profile でおおむね 15 |
| cov90 | 0.90 以上 |
| bias | ±5 以内 |
| low_rul_rmse_30 | おおむね 6 |
| p95 latency | Grafana で追跡し、継続的な回帰があれば alert を上げる |
| job success rate | queued job が手動 cleanup なしで収束すること |
| drift severity counts | medium / high の増加が見られたら promotion を止める |
| promotion approval rate | governance 指標。低い場合は candidate quality を疑う |

---
英語版: [operations.md](operations.md)