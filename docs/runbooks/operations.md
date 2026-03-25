# RULFM Runbook

Language note: this is the English-primary operations runbook. A Japanese companion is available in `docs/runbooks/operations.ja.md`.

## Minimum paths first

Use the smallest configuration that matches your goal.

| Goal | Required env / dependencies | Notes |
|---|---|---|
| Public benchmark locally | `pip install -r requirements-lock.txt` | No control-plane extras required |
| API demo locally | `RULFM_FORECASTING_API_KEY`, `uvicorn forecasting_api.app:create_app --factory` | Add bearer token only if you want operator or metrics auth by token |
| Durable local control plane | PostgreSQL DSNs plus compose profile or external Postgres | Use when you need persisted jobs and registry across restarts |
| Cloud Run deployment | Terraform inputs, Workload Identity, runtime bucket, Cloud SQL DSNs, smoke-test secrets | Full path documented below |

If you need every knob, keep reading. If you only need a public demo or the benchmark reproduction path, stop at the first matching row and avoid carrying unnecessary settings.

## Local startup

1. Install public dependencies with `pip install -r requirements-lock.txt`.
2. Install `pip install -r requirements-experimental-lock.txt` only when you need torch or hybrid experiments.
3. Export `RULFM_FORECASTING_API_KEY`.
4. Export `RULFM_FORECASTING_API_BEARER_TOKEN` when you want bearer auth for operators, Prometheus, or CI smoke tests. Local Compose defaults it to `local-metrics-token` so Prometheus can scrape `/metrics`.
5. Start the stack with `docker compose up --build` or run the API directly with Uvicorn.
6. Before running `dvc pull` or `dvc push`, point the configured remote at the environment-specific GCS bucket and authenticate to GCP.

Mutable local state now defaults to `runtime/`. Committed benchmark and reference artifacts stay under `src/forecasting_api/data/`.

## Environment variable guide

- `RULFM_FORECASTING_API_KEY` is the only required application secret for the minimum local API path.
- `RULFM_FORECASTING_API_BEARER_TOKEN` is optional unless you want bearer-only operator access or metrics scraping by bearer token.
- Tenant, network, and approval settings are advanced controls. Enable them only when you are explicitly testing those policy surfaces.
- PostgreSQL DSNs and persistent path overrides are required for Cloud Run and any serious multi-instance deployment.

## Runtime modes

- `RULFM_JOB_EXECUTION_BACKEND=worker` is the default public mode. The API only queues jobs; a separate worker process claims and executes them.
- Local Docker Compose uses `RULFM_JOB_WORKER_MODE=daemon`; Cloud Run worker execution is `RULFM_JOB_WORKER_MODE=batch` so each Job run drains a bounded number of queued jobs and exits.
- `RULFM_JOB_RUNNING_TIMEOUT_SECONDS` controls stale-running recovery. Before the worker claims a new queued job, it re-queues `running` jobs whose `updated_at` is older than this timeout.
- `RULFM_JOB_EXECUTION_BACKEND=inprocess` remains available only for compatibility or very small local demos.
- `RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL=1` enables two-person approval on `/v1/train` via `X-Approved-By`.
- When `RULFM_FORECASTING_API_TRAIN_APPROVER_GROUPS` or `RULFM_FORECASTING_API_TRAIN_APPROVER_SUBJECTS` is set, `/v1/train` switches to OIDC claim-based approval and requires an OIDC bearer token whose `groups` or configured claim names intersect the approved set.

## Deployment

- Terraform defaults Cloud Run ingress to `INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER`; publish a load balancer or other approved ingress path before expecting external traffic.
- The CD workflow verifies Cloud Run readiness and then calls `/health` after deployment.
- Set `GCP_CLOUD_RUN_HEALTHCHECK_URL` in GitHub Actions when the health check must go through a load balancer URL instead of the service's default `run.app` URL.
- Cloud Run startup now fails fast unless runtime-mutated state paths are explicitly overridden and both control-plane backends are PostgreSQL. Set `RULFM_JOB_STORE_BACKEND=postgres`, `RULFM_JOB_STORE_POSTGRES_DSN`, `RULFM_MODEL_REGISTRY_BACKEND=postgres`, `RULFM_MODEL_REGISTRY_POSTGRES_DSN`, keep `RULFM_JOB_EXECUTION_BACKEND=worker`, and point `RULFM_MODEL_REGISTRY_DB_PATH`, `RULFM_MODEL_ARTIFACTS_ROOT`, `RULFM_FORECASTING_API_AUDIT_LOG_PATH`, `RULFM_DRIFT_BASELINE_PATH`, and `RULFM_MODEL_PROMOTION_REGISTRY_PATH` at persistent storage.
- The Terraform stack now provisions:
  - Cloud SQL PostgreSQL instance, database, user, and DSN secret
  - versioned GCS runtime bucket mounted into both API and worker
  - Cloud Run service for the API
  - Cloud Run Job for the queue worker
  - Cloud Scheduler trigger that calls `jobs.run`

### Terraform verification

- Local hostに Terraform CLI が無い場合でも、Docker で `hashicorp/terraform` を使って `infra/terraform` に対する `init` と `plan` を実行できます。
- GCS backend 付きの本番 `plan` には backend 設定と GCP Application Default Credentials が必要です。
- 認証前の静的検証は backend を外した一時コピーで `init` と `plan` を実行し、provider schema と resource graph の整合性を確認してください。

### Cloud SQL integration verification

- Cloud SQL 実インスタンスに対しては `RULFM_TEST_CLOUDSQL_POSTGRES_DSN` を設定し、次を実行します。

```bash
.venv/bin/python -m pytest --no-cov \
  tests/integration/test_job_store_postgres.py \
  tests/integration/test_model_registry_postgres.py -q
```

- `test_model_registry_postgres.py` は model registry の持続性に加えて concurrent upsert 後も単一 row が維持されることを確認します。

## Health and metrics

- `GET /health` confirms the API process is serving traffic and exposes whether experimental models are enabled.
- `GET /metrics` exposes Prometheus metrics and requires API authentication.
- Grafana is available on port 3000 when using Docker Compose.
- Application logs emitted through the `rulfm` logger are JSON. Uvicorn and other library loggers keep their own handlers unless you override them in the runtime entrypoint or log collector.

## Drift workflow

1. Persist a baseline with `POST /v1/monitoring/drift/baseline` using representative records.
2. Submit candidate observations to `POST /v1/monitoring/drift/report`.
3. Use `baseline_records` on the report endpoint only for one-off comparisons; it does not overwrite the stored baseline.
4. Investigate medium/high severity reports before promoting a model.

## Promotion workflow

1. Collect evaluation metrics from backtest or benchmark jobs.
2. Run `python -m scripts.promote_model MODEL_ID --metrics-file metrics.json`.
3. Review the promotion registry and MLflow artifact before changing stage references.

## Running the benchmark

### GBDT preset selection

The benchmark script exposes two GBDT training presets via `RULFM_BENCHMARK_GBDT_PRESET`:

| Preset | Candidates | max_iter | Use case |
|---|---|---|---|
| `full` (default) | 6 | up to 600 | Publication-quality results; ~10 min on modern hardware |
| `fast` | 2 | up to 90 | Smoke test / CI; ~1–2 min |

### Training data

The GBDT benchmark trains on all engine cycles (`window_size=None`), covering labels
y ∈ [0, 125]. Use `RULFM_BENCHMARK_STAGE=gbdt-only` to produce only the GBDT checkpoint:

```bash
RULFM_BENCHMARK_STAGE=gbdt-only \
RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0 \
PYTHONPATH=src python scripts/build_fd004_benchmark_summary.py
```

Experimental full benchmark rows require both the experimental dependency set and explicit flags:

```bash
pip install -r requirements-experimental-lock.txt
RULFM_ENABLE_EXPERIMENTAL_MODELS=1 \
RULFM_BENCHMARK_STAGE=full \
RULFM_BENCHMARK_ENABLE_AFNO=1 \
PYTHONPATH=src python scripts/build_fd004_benchmark_summary.py
```

### Expected metrics (full preset, full-cycle training)

| Metric | Expected value |
|---|---|
| backtest_rmse | ~15 |
| backtest_cov90 | ≥ 0.90 |
| backtest_bias | within ±5 |
| low_rul_rmse_30 | ~6 |

If `backtest_rmse > 30` or `backtest_bias < -20`, the most likely cause is the old
`window_size=90` profile being used for training. Verify `train_profile` in the output JSON
is `"fd004_full_cycles"`, not `"fd004_train_multiunit"`.

### Ensemble flag

`RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0` skips LightGBM + CatBoost and uses only HGB.
Set to `1` (or omit) for the full ensemble, which requires `lightgbm` and `catboost`
to be installed. Ensemble adds ~5 min but typically reduces RMSE by 5–10%.

## Incident handling

- If `/metrics` stops updating, verify `RULFM_METRICS_ENABLED=1` and Prometheus scrape health.
- If drift alerts spike, compare current feature distributions with the saved baseline and block promotion.
- If model promotion is rejected, inspect the reasons in `runtime/model_promotions.json` for local Compose or in the configured external registry path for shared environments.
- If jobs remain in `running` after a worker crash, inspect `RULFM_JOB_RUNNING_TIMEOUT_SECONDS`, verify worker restarts are healthy, and confirm the stale jobs return to `queued` before the next batch drain.
- If benchmark RMSE is unexpectedly high (>30), check `train_profile` in the summary JSON and
  ensure `RULFM_BENCHMARK_GBDT_PRESET` is not set to `fast` unintentionally.

## Backup and restore

- SQLite-backed local state: back up `runtime/trained_models.db`, `runtime/model_artifacts/`, `runtime/request_audit_log.jsonl`, `runtime/drift_baseline.json`, and `runtime/model_promotions.json` together.
- PostgreSQL-backed Cloud Run control-plane state: use a regular Cloud SQL dump or PITR schedule and verify restore by replaying `tests/integration/test_job_store_postgres.py` plus a model catalog smoke check against the restored instance.
- GCS-backed runtime artifacts: keep object versioning enabled on the runtime bucket and practice restoring `model_artifacts/`, `audit/`, `monitoring/`, and `promotion/` prefixes together.
- After restore, restart the worker and confirm queued jobs transition from `queued` to `running`.

## Rollback

- Application rollback: redeploy the previous sha-tagged GHCR image from the CD pipeline output.
- State rollback: restore the matching Cloud SQL snapshot and runtime bucket object versions together; do not roll back only DB or only artifacts.
- Promotion rollback: append a new promotion decision for the prior stage instead of editing history in place.

## Runtime and business KPIs

| KPI | Target / intent |
|---|---|
| backtest_rmse | ~15 on full public GBDT profile |
| cov90 | at least 0.90 |
| bias | within ±5 |
| low_rul_rmse_30 | ~6 |
| p95 latency | track in Grafana and alert on sustained regressions |
| job success rate | queued jobs should converge to terminal state without manual cleanup |
| drift severity counts | medium/high spikes should block promotion |
| promotion approval rate | governance metric; low rates indicate poor candidate quality |
