# cmapss-fd004-rul-mlops

[![Stable CI](https://github.com/yskfuji/cmapss-fd004-rul-mlops/actions/workflows/ci-stable.yml/badge.svg)](https://github.com/yskfuji/cmapss-fd004-rul-mlops/actions/workflows/ci-stable.yml)
[![Experimental CI](https://github.com/yskfuji/cmapss-fd004-rul-mlops/actions/workflows/ci-experimental.yml/badge.svg)](https://github.com/yskfuji/cmapss-fd004-rul-mlops/actions/workflows/ci-experimental.yml)

Public portfolio repository for NASA CMAPSS FD004 Remaining Useful Life forecasting.

**Language:** English | [Japanese](README.ja.md)

## Portfolio release

- public portfolio release tag: `0.0.0`
- version sources aligned: `pyproject.toml`, `package.json`, FastAPI OpenAPI metadata, and `CHANGELOG.md`
- public bilingual companions:
    - `README.ja.md`
    - `docs/runbooks/operations.ja.md`
    - `docs/architecture/overview.en.md`

The primary story of this repository is deliberately narrow:

- a reproducible public GBDT benchmark for FD004
- a FastAPI serving path that exposes forecast, backtest, jobs, metrics, and monitoring
- the operational controls needed to show how that benchmark would be run and governed

Enterprise-style controls such as OIDC approval, tenant policy, audit logging, drift monitoring, and job orchestration are included as supporting operational surface. They are here to demonstrate control-plane design around the public benchmark, not to claim that this repository is already a full multi-tenant production platform.

Experimental neural and hybrid models remain in the codebase for comparison only. They are opt-in and outside the default public contract unless `RULFM_ENABLE_EXPERIMENTAL_MODELS=1` is set.

## Reader guide

If you only want the public benchmark path, read these first:

- benchmark scope and metrics in this README
- operational minimums in `docs/runbooks/operations.md`
- release discipline in `docs/runbooks/release.md`
- ownership and security exception controls in `docs/governance/`

## Quality Gates

This repository includes CI, CD, and multiple test layers.

| Surface | What it covers |
|---|---|
| `.github/workflows/ci-stable.yml` | lint, security audit, typecheck, stable tests, PostgreSQL compose integration, benchmark checks, DVC dry-run, and e2e follow-up jobs |
| `.github/workflows/ci-experimental.yml` | experimental torch and hybrid model tests with focused coverage gates |
| `.github/workflows/cd.yml` | GHCR image build/publish, Cloud Run deploy, and post-deploy smoke checks |
| `tests/unit/` | unit tests for helpers, services, runtime utilities, and model adapters |
| `tests/integration/` | API, job-store, PostgreSQL, and endpoint integration tests |
| `tests/monitoring/` | drift, metrics rendering, and monitoring persistence checks |
| `tests/regression/` | benchmark artifact regression thresholds |
| `tests/frontend/` | browser-side controller tests |
| `tests/e2e/` | Playwright smoke and UI flow coverage |

If you want a quick proof point, the badges at the top of this README link directly to the stable and experimental workflows, and the paths above map to the validation layers exercised in GitHub Actions.

## What this project demonstrates

| Area | Public implementation |
|---|---|
| Benchmark model | HistGradientBoosting-based RUL forecaster with calibrated intervals |
| Feature engineering | phase-aware normalization, rolling windows, exogenous sensor features |
| Uncertainty | quantile boosting plus residual-based conformal calibration |
| Production API | FastAPI, API key / Bearer / OIDC auth, tenant/network request policy, job API, MLflow hooks |
| Operations | drift baseline/report flow, Prometheus metrics, runbook, CI |
| Frontend | browser UI with EN/JA i18n and shareable state |
| Control surface | shared audit event schema, tenant-aware request audit, and minimal policy helpers for tenancy, network, approval, and portability |

The benchmark and public API are the product surface. The control surface exists to show how release, audit, and monitoring concerns attach to that product surface.

## Public benchmark snapshot

The committed artifact at [src/forecasting_api/data/fd004_benchmark_summary.json](src/forecasting_api/data/fd004_benchmark_summary.json) is the public reference checkpoint.

- evaluation: final-cycle prediction for 237 eligible test engines
- training profile: full cycles per engine
- benchmark stage: `gbdt-only`
- preset: `full`
- ensemble flag: `RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0`

| Model | RMSE | MAE | NASA score | bias | overpred rate | cov90 | width90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GBDT_w30_f24 | 15.1594 | 10.4467 | 968.3299 | -2.2130 | 0.4473 | 0.9114 | 60.2012 |

Low-RUL slice:

| Model | RMSE <= 30 | NASA <= 30 | RMSE <= 60 | NASA <= 60 |
|---|---:|---:|---:|---:|
| GBDT_w30_f24 | 5.9604 | 37.2839 | 8.5296 | 115.1174 |

Root-cause note: earlier public snapshots trained on truncated history and collapsed the label range to roughly `[0, 89]`. The current benchmark trains on all cycles, restoring the full `[0, 125]` label range and removing the large negative-bias failure mode.

## Public architecture

![Architecture](figures/fig1_architecture.png)

Default public flow:

1. FD004 preprocessing and operating-condition-aware normalization
2. rolling feature generation from sensor history
3. HGB-based point and interval models
4. residual calibration and API serving with monitoring hooks

The public API surface is intentionally narrow:

- `ridge_lags_v1`
- `gbdt_hgb_v1`
- `naive`

Experimental torch / hybrid algorithms are opt-in only.

Implementation note: [src/forecasting_api/app.py](src/forecasting_api/app.py) is the composition root and compatibility shell. Legacy schema imports that tests or callers still access through forecasting_api.app are re-exported via [src/forecasting_api/app_compat.py](src/forecasting_api/app_compat.py), forecast/backtest/train service dependency wiring lives in dedicated runtime builder modules, and [src/forecasting_api/app_auth_facade.py](src/forecasting_api/app_auth_facade.py) now owns env-driven auth and request-setting readers. In app.py, APP_COMPAT_EXPORTS names the compatibility re-exports that stay import-stable for older callers, while APP_RUNTIME_EXPORTS names the intentionally exposed app/runtime entry points. The app module still keeps a few thin auth wrappers so existing tests and callers that monkeypatch forecasting_api.app symbols continue to work.

## Repository map

```
src/
├── models/
│   └── gbdt_pipeline.py
├── forecasting_api/
│   ├── app.py
│   ├── app_compat.py
│   ├── app_auth_facade.py
│   ├── forecast_runtime.py
│   ├── backtest_runtime.py
│   ├── domain/stable_models.py
│   ├── runtime_state.py
│   ├── request_audit.py
│   ├── request_policy.py
│   ├── job_dispatcher.py
│   ├── cmapss_fd004.py
│   ├── mlflow_runs.py
│   ├── torch_forecasters.py
│   └── static/forecasting_gui/
└── enterprise/
    ├── audit.py
    ├── iam.py
    ├── network.py
    ├── portability.py
    └── tenancy.py

scripts/
└── build_fd004_benchmark_summary.py
```

## Installation

Minimum local path for the public benchmark and API:

```bash
pip install -r requirements-lock.txt
```

Minimum runtime environment:

```bash
export RULFM_FORECASTING_API_KEY=local-demo-key
PYTHONPATH=src uvicorn forecasting_api.app:create_app --factory --port 8000
```

For the full environment matrix, PostgreSQL-backed state, Cloud Run notes, and recovery procedures, use `docs/runbooks/operations.md`. The README keeps only the minimum path and the portfolio-relevant control points.

Optional PostgreSQL-backed job state:

```bash
export RULFM_JOB_STORE_BACKEND=postgres
export RULFM_JOB_STORE_POSTGRES_DSN=postgresql://user:password@localhost:5432/rulfm
```

Optional PostgreSQL-backed model registry state:

```bash
export RULFM_MODEL_REGISTRY_BACKEND=postgres
export RULFM_MODEL_REGISTRY_POSTGRES_DSN=postgresql://user:password@localhost:5432/rulfm
```

Optional experimental models:

```bash
pip install -r requirements-experimental-lock.txt
```

## Reproduce the public benchmark

Publication-grade GBDT snapshot:

```bash
RULFM_BENCHMARK_STAGE=gbdt-only \
RULFM_BENCHMARK_GBDT_PRESET=full \
RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0 \
PYTHONPATH=src python scripts/build_fd004_benchmark_summary.py
```

CI smoke run:

```bash
RULFM_BENCHMARK_STAGE=gbdt-only \
RULFM_BENCHMARK_GBDT_PRESET=fast \
RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0 \
PYTHONPATH=src python scripts/build_fd004_benchmark_summary.py
```

Experimental full benchmark with public torch surrogates:

```bash
pip install -r requirements-experimental-lock.txt
RULFM_ENABLE_EXPERIMENTAL_MODELS=1 \
RULFM_BENCHMARK_STAGE=full \
RULFM_FORECASTING_TORCH_MAX_EPOCHS=20 \
PYTHONPATH=src python scripts/build_fd004_benchmark_summary.py
```

## Run the API

```bash
export RULFM_FORECASTING_API_KEY=your-key
PYTHONPATH=src uvicorn forecasting_api.app:create_app --factory --port 8000
```

Optional operator or metrics bearer token:

```bash
export RULFM_FORECASTING_API_BEARER_TOKEN=your-bearer-token
```

Advanced control-plane options such as tenant enforcement, network policy, train approval, and OIDC claim-based approval are documented in `docs/runbooks/operations.md`. They are intentionally kept out of the minimum quickstart so the public benchmark path is visible at a glance.

Then open http://127.0.0.1:8000/ui/forecasting/.

Health and docs:

- `/health`
- `/docs`
- `/docs/en`
- `/docs/ja`
- `/openapi.json`

The live OpenAPI schema is served at `/openapi.json` when the API is running.

## Run the stack

```bash
export RULFM_FORECASTING_API_KEY=your-api-key
export RULFM_FORECASTING_API_BEARER_TOKEN=your-bearer-token
export GF_SECURITY_ADMIN_USER=your-grafana-user
export GF_SECURITY_ADMIN_PASSWORD=your-grafana-password
docker compose up --build
```

Services:

- FastAPI: `:8000`
- Job worker: persistent queue consumer in separate container running `RULFM_JOB_WORKER_MODE=daemon`
- Prometheus: `:9090`
- Grafana: `:3000`

Local mutable state is mounted from `runtime/`; committed benchmark artifacts remain under `src/forecasting_api/data/` as reproducible reference data.

Local Compose defaults `RULFM_FORECASTING_API_BEARER_TOKEN` to `local-metrics-token` so Prometheus can scrape `/metrics` without a token mismatch. Override it explicitly when you do not want the local default.

PostgreSQL-backed JobStore integration profile:

```bash
export RULFM_TEST_POSTGRES_PASSWORD=your-test-postgres-password
docker compose --profile test up --build --abort-on-container-exit --exit-code-from job-store-postgres-test job-store-postgres-test
docker compose --profile test down --volumes
```

Cloud SQL or external PostgreSQL integration checks for both async jobs and model registry:

```bash
export RULFM_TEST_CLOUDSQL_POSTGRES_DSN='postgresql://user:password@/rulfm?host=/cloudsql/project:region:instance'
.venv/bin/python -m pytest --no-cov \
    tests/integration/test_job_store_postgres.py \
    tests/integration/test_model_registry_postgres.py -q
```

## Operational notes

- `/metrics` requires API authentication.
- API rate limiting is enabled by default for `/v1/*` and `/metrics`.
- Tune limits with `RULFM_FORECASTING_API_RATE_LIMIT_PER_WINDOW` and `RULFM_FORECASTING_API_RATE_LIMIT_WINDOW_SECONDS`.
- Trained model metadata uses `RULFM_MODEL_REGISTRY_BACKEND=sqlite|postgres`; SQLite remains the local default, while PostgreSQL is the intended multi-instance and Cloud Run backend. Training still upserts one model entry at a time instead of re-writing the full registry snapshot.
- Async job state uses `RULFM_JOB_STORE_BACKEND=sqlite|postgres`; SQLite remains the local default and PostgreSQL is available for multi-instance or shared control-plane deployments.
- When PostgreSQL is selected, set both `RULFM_JOB_STORE_POSTGRES_DSN` and `RULFM_MODEL_REGISTRY_POSTGRES_DSN` and keep the same `/v1/jobs/{job_id}` and model catalog API contracts.
- `/v1/jobs` now has an explicit split between persisted job state and execution dispatch via a queue-first enqueuer interface. The default public mode is `RULFM_JOB_EXECUTION_BACKEND=worker`, which leaves jobs queued until a separate worker process claims them. `inprocess` remains available only as an explicit compatibility mode.
- Legacy JSON registries are auto-migrated on startup when `RULFM_MODEL_REGISTRY_PATH` exists.
- CI is split into `ci-stable` and `ci-experimental`; stable now also installs torch so the public model registry tests in `tests/unit/test_models_registry.py` are part of the blocking `not experimental` profile, while feature-flagged torch/hybrid API paths remain in `ci-experimental`.
- Coverage gate is enforced at 72 for the stable profile; 80 is still monitoring until normal CI runs clear it with margin consistently. Local stable-equivalent runs have crossed 80, but the remaining concentration of logic in `src/forecasting_api/app.py` still makes that threshold noisy as a blocking gate.
- CI includes a file-backed MLflow smoke test that verifies a real run can be created and artifacts can be logged.
- Drift baseline/report endpoints are part of the public portfolio because they support production monitoring, not model experimentation.
- Stable forecast/backtest helpers now live in `src/forecasting_api/domain/stable_models.py`; `app.py` keeps compatibility wrappers because a number of tests still monkeypatch the legacy helper symbols directly.
- Auth/env configuration readers now live in `src/forecasting_api/app_auth_facade.py`; `app.py` only keeps the thin `_require_*` and expected-token wrappers that must remain monkeypatchable through the legacy module surface.
- Request audit lines are now written in the shared `enterprise.audit.AuditEvent` JSONL format and include tenant context when request policy is active, but `src/enterprise/` as a whole is still a minimal policy/control surface, not a full production IAM or multi-tenant control plane.
- Tenant and network policy checks are now wired into the `/v1/*` access path through `X-Tenant-Id`, `X-Connection-Type`, and environment-driven allowlist/private-connectivity settings.
- Two-person approval can now be enforced for `/v1/train` with `RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL=1` and `X-Approved-By` headers.
- Cloud Run Terraform defaults to internal load-balancer ingress.
- On Cloud Run, the app now refuses to start if runtime-mutated state is left on packaged in-container defaults or if `RULFM_JOB_EXECUTION_BACKEND=inprocess` is selected. Set PostgreSQL DSNs for job state and model registry, and point model artifacts, request audit, drift baseline, and promotion registry at persistent external storage before deploying.
- The Terraform stack now provisions Cloud SQL PostgreSQL, a versioned GCS runtime bucket, a Cloud Run API service, a Cloud Run Job worker, and a Cloud Scheduler trigger. The API and worker share the same durable control-plane state while keeping Compose on daemon worker mode for local development.

## Cloud Run runtime state

The production Terraform path in [infra/terraform/main.tf](infra/terraform/main.tf) externalizes all mutable runtime state:

- Cloud SQL PostgreSQL for `RULFM_JOB_STORE_POSTGRES_DSN` and `RULFM_MODEL_REGISTRY_POSTGRES_DSN`
- GCS-mounted runtime bucket for model artifacts, request audit, drift baseline, and promotion registry
- Cloud Run service for the API with Cloud SQL and GCS mounts
- Cloud Run Job for queue draining with `RULFM_JOB_WORKER_MODE=batch`
- Cloud Scheduler for periodic worker execution

Typical Terraform inputs:

```bash
terraform -chdir=infra/terraform apply \
    -var project_id=your-gcp-project \
    -var container_image=asia-northeast1-docker.pkg.dev/your-gcp-project/arcayf-forecasting/api:sha-1234 \
    -var api_key_secret_name=arcayf-forecasting-api-key \
    -var bearer_token_secret_name=arcayf-forecasting-bearer-token
```
- A sample DVC remote is committed; update it to your bucket before `dvc pull` or `dvc push`.

## Current scope and remaining work

This repository should be read as a production-style reference implementation rather than evidence of an already-operated production service.

- The Cloud Run, Cloud SQL, and GCS deployment path is implemented and guarded, but this repo does not yet include committed evidence from a real authenticated Terraform apply/plan cycle.
- PostgreSQL integration tests exist for job state and model registry, but running them still requires a reachable Cloud SQL or external PostgreSQL instance.
- Worker crash recovery is implemented with timeout-based stale-job requeue, and it still needs long-haul validation under real Cloud Run Job retry patterns.
- OIDC-backed train approval is stronger than raw approval headers, but it is not yet a full workflow with durable approval records and delegated policy administration.
- `src/forecasting_api/app.py` is now thinner, but it still carries compatibility wrappers for older imports and monkeypatch-heavy tests.
- KPI, metrics, and dashboards are in place, but the repo does not yet include sustained load-test evidence or an operational error-budget history.

## Governance and release discipline

- Coverage policy is documented in `docs/architecture/coverage-plan.md` and matches the blocking CI gate.
- Security exceptions are tracked in `docs/governance/security-exceptions.md` with rationale, mitigation, owner, and review date.
- Single-maintainer review discipline is documented in `docs/governance/review-and-ownership.md` and encoded in the PR template plus blocking CI.
- Release history starts in `CHANGELOG.md`, and the release checklist lives in `docs/runbooks/release.md`.

## Business KPIs

This repository tracks more than point RMSE. The current portfolio KPI set is:

| KPI | Why it matters | Current public source |
|---|---|---|
| RMSE / MAE / NASA score | point accuracy and ranking sensitivity | benchmark summary |
| cov90 / width90 | interval calibration vs. sharpness | benchmark summary |
| bias / overpred rate | operational failure mode visibility for maintenance planning | benchmark summary |
| low-RUL RMSE (<=30, <=60) | business-critical slice performance near failure | benchmark summary |
| request rate / p95 latency | API operability and SLO tracking | Prometheus + Grafana |
| drift severity counts | data stability and retraining trigger signal | Prometheus + drift report endpoint |
| promotion pass rate | governance quality for model release decisions | model promotion registry |

The public benchmark artifact remains the source of truth for model-quality KPIs, while Prometheus/Grafana and promotion records cover runtime and release KPIs.

## Architecture roadmap

This codebase is built toward a full MLOps package structure in staged iterations.

### Implemented (Lv1)
- [x] FastAPI router split (`routers/`)
- [x] Pydantic schema consolidation (`schemas/`)
- [x] Config helpers (`config.py`)
- [x] Error types extracted to `errors.py` — breaks circular import chain
- [x] Service layer circular dependency resolved (`services/` + `errors.py`)
- [x] Drift detection and monitoring (`monitoring/`)
- [x] CI/CD (GitHub Actions → GCP Cloud Run via Workload Identity)
- [x] MLflow experiment tracking
- [x] DVC data versioning
- [x] Benchmark regression tests (`tests/regression/`)

### Planned (Lv2)
- [ ] Data directory separation (`data/raw/`, `data/processed/`) — committed benchmark/reference artifacts still live under `src/forecasting_api/data/`; mutable runtime state now defaults to `runtime/`
- [ ] Feature engineering extraction (`src/features/`) — currently embedded in `src/models/gbdt_pipeline.py`
- [ ] Training and forecast helper separation (`src/training/`, `src/inference/`) — large helper branches still remain in `src/forecasting_api/app.py`
- [ ] Retire `src/forecasting_api/app.py` compatibility wrappers after router and service callers stop monkeypatching legacy symbols; staged plan is documented in `docs/adr/0006-app-py-compatibility-wrapper-retirement.md`
- [x] Extended documentation (`docs/architecture/overview.md`)

### Future (Lv3)
- [ ] Frontend independence (`frontend/`) — currently served as static files
- [ ] Inference pipeline (`inference/`) — predictors, postprocessors, batch/online split
- [ ] Pipeline orchestration (`pipelines/`) — end-to-end training → evaluation → deployment
- [ ] Performance tests (`tests/performance/`)

## License

MIT

---
🇯🇵 日本語版: [README.ja.md](README.ja.md)
