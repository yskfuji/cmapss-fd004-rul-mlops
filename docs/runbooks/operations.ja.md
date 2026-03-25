# RULFM Runbook

**言語:** [英語](operations.md) | 日本語

## 最小パス優先

目的に合う最小構成から始めてください。

| 目的 | 必要な env / dependency | 補足 |
|---|---|---|
| 公開 benchmark をローカルで再現 | `pip install -r requirements-lock.txt` | control-plane 追加設定は不要 |
| API demo をローカル起動 | `RULFM_FORECASTING_API_KEY`, `uvicorn forecasting_api.app:create_app --factory` | operator や metrics 認証が必要なときだけ bearer token を追加 |
| 耐久性のあるローカル control plane | PostgreSQL DSN と compose profile か外部 Postgres | restart を跨いで jobs と registry を保持したいとき |
| Cloud Run deployment | Terraform input、Workload Identity、runtime bucket、Cloud SQL DSN、smoke-test secret | 本番相当のフルパス |

公開 demo と benchmark 再現だけが目的なら、最初の該当行で止めて構いません。

## ローカル起動

1. 公開依存を `pip install -r requirements-lock.txt` で入れる。
2. torch / hybrid 実験が必要な場合だけ `pip install -r requirements-experimental-lock.txt` を追加する。
3. `RULFM_FORECASTING_API_KEY` を export する。
4. bearer auth を使うなら `RULFM_FORECASTING_API_BEARER_TOKEN` を export する。Local Compose では `local-metrics-token` が既定です。
5. `docker compose up --build` か Uvicorn で API を起動する。
6. `dvc pull` / `dvc push` の前に、remote を実環境の GCS bucket に向けて認証する。

可変 state は `runtime/` に置かれ、commit 済み benchmark / reference artifact は `src/forecasting_api/data/` に残ります。

## 環境変数ガイド

- `RULFM_FORECASTING_API_KEY` は最小ローカル API パスで唯一必須の application secret です。
- `RULFM_FORECASTING_API_BEARER_TOKEN` は bearer-only operator access や metrics scrape が必要な場合のみ使います。
- tenant、network、approval は上級 control です。検証対象のときだけ有効化してください。
- PostgreSQL DSN と persistent path override は Cloud Run や multi-instance 配置で必須です。

## 実行モード

- `RULFM_JOB_EXECUTION_BACKEND=worker` が公開既定です。API は job を queue に積み、別 worker process がそれを実行します。
- Local Docker Compose は `RULFM_JOB_WORKER_MODE=daemon`、Cloud Run worker は `RULFM_JOB_WORKER_MODE=batch` を使います。
- `RULFM_JOB_RUNNING_TIMEOUT_SECONDS` は stale-running recovery を制御します。
- `RULFM_JOB_EXECUTION_BACKEND=inprocess` は互換または小規模ローカル demo 用にのみ残しています。
- `RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL=1` で `/v1/train` に two-person approval を強制できます。
- approver group / subject を設定すると、`/v1/train` は OIDC claim ベースの承認に切り替わります。

## デプロイメント

- Terraform は既定で Cloud Run ingress を internal load balancer に向けます。外部公開するなら別途 ingress path を用意してください。
- CD workflow は deploy 後に readiness を確認し、続けて `/health` を叩きます。
- health check を load balancer URL 経由にしたい場合は GitHub Actions に `GCP_CLOUD_RUN_HEALTHCHECK_URL` を設定します。
- Cloud Run 起動時は、mutable state path が container 内デフォルトのまま、または control-plane backend が PostgreSQL でない場合に fail fast します。

Terraform stack が用意するもの:

- Cloud SQL PostgreSQL instance / database / user / DSN secret
- API と worker で共有する versioned GCS runtime bucket
- API 用 Cloud Run service
- queue worker 用 Cloud Run Job
- `jobs.run` を呼ぶ Cloud Scheduler trigger

## Terraform 検証

- ローカルに Terraform CLI がなくても、Docker の `hashicorp/terraform` で `infra/terraform` に対して `init` と `plan` を実行できます。
- GCS backend を使う本番 `plan` には backend 設定と GCP Application Default Credentials が必要です。
- 認証前の静的検証だけなら backend を外した一時コピーで `init` と `plan` を実行し、provider schema と resource graph の整合性を確認してください。

## Cloud SQL 統合確認

`RULFM_TEST_CLOUDSQL_POSTGRES_DSN` を設定して次を実行します。

```bash
.venv/bin/python -m pytest --no-cov \
  tests/integration/test_job_store_postgres.py \
  tests/integration/test_model_registry_postgres.py -q
```

## ヘルスとメトリクス

- `GET /health` は API が応答していることと、experimental model の有効状態を返します。
- `GET /metrics` は Prometheus metrics を返し、認証が必要です。
- Docker Compose 利用時、Grafana は port 3000 で利用できます。

## ドリフトワークフロー

1. `POST /v1/monitoring/drift/baseline` で baseline を保存する。
2. `POST /v1/monitoring/drift/report` で candidate observation を送る。
3. one-off 比較では `baseline_records` を使えるが、保存済み baseline 自体は上書きしない。
4. severity が medium / high なら model promotion 前に調査する。

## プロモーションワークフロー

1. backtest または benchmark job から評価 metric を集める。
2. `python -m scripts.promote_model MODEL_ID --metrics-file metrics.json` を実行する。
3. stage 参照を変える前に promotion registry と MLflow artifact を確認する。

## ベンチマーク実行

### GBDT プリセット

| Preset | Candidate 数 | max_iter | 用途 |
|---|---|---|---|
| `full` | 6 | 最大 600 | 論文品質の結果 |
| `fast` | 2 | 最大 90 | スモークテスト / CI |

GBDT benchmark は全 engine cycle (`window_size=None`) で学習します。公開 checkpoint だけを再生成するには次を使います。

```bash
RULFM_BENCHMARK_STAGE=gbdt-only \
RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0 \
PYTHONPATH=src python scripts/build_fd004_benchmark_summary.py
```

## インシデント対応

- `/metrics` が更新されない場合は `RULFM_METRICS_ENABLED=1` と Prometheus scrape を確認する。
- drift alert が増えたら baseline との分布差を見て promotion を止める。
- worker crash 後に job が `running` のままなら timeout と worker restart を確認する。
- benchmark RMSE が 30 を超える場合は、`train_profile` と preset 設定を疑う。

## バックアップ / リストア

- SQLite ローカル state は `runtime/trained_models.db`、`runtime/model_artifacts/`、`runtime/request_audit_log.jsonl`、`runtime/drift_baseline.json`、`runtime/model_promotions.json` をまとめて退避する。
- Cloud Run control plane は Cloud SQL dump または PITR と、GCS object versioning を併用する。
- restore 後は worker を再起動し、queued job が再び terminal state へ進むことを確認する。

## ロールバック

- application rollback は CD pipeline が出力した前の sha-tagged GHCR image を使う。
- state rollback は Cloud SQL snapshot と runtime bucket object version を必ず揃える。
- promotion rollback は history を編集せず、新しい promotion decision を追記する。

## ランタイム / ビジネス KPI

| KPI | 目標 / 意図 |
|---|---|
| backtest_rmse | full public GBDT profile でおおむね 15 |
| cov90 | 0.90 以上 |
| bias | ±5 以内 |
| low_rul_rmse_30 | おおむね 6 |
| p95 latency | Grafana で追跡し、継続回帰を alert |
| job success rate | queued job が手動 cleanup なしで収束すること |
| drift severity counts | medium / high の増加は promotion を止める |
| promotion approval rate | governance 指標。低い場合は candidate quality を疑う |

---
英語版: [operations.md](operations.md)