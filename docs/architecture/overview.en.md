# Architecture Overview

This file is the English companion for the mixed-language architecture notes. 🇯🇵 Japanese/mixed version → [overview.md](overview.md)

## System intent

The repository is centered on a narrow public contract:

- a reproducible FD004 GBDT benchmark
- a FastAPI surface for forecast, backtest, jobs, metrics, and monitoring
- enough control-plane design to demonstrate release, audit, and operational discipline

The architecture is intentionally not presented as a full multi-tenant SaaS platform. Tenant policy, approval flow, audit, drift, and portability helpers exist to demonstrate control surfaces around the benchmark and API.

## Main components

### `src/forecasting_api/`

- `app.py`: FastAPI composition root and compatibility shell
- `routers/`: HTTP layer only
- `services/`: business logic entry points
- `runtime_state.py`: lazy runtime state for trained models and job store
- `request_audit.py`, `request_policy.py`, `request_approval.py`: audit and policy controls
- `job_dispatcher.py`, `job_worker.py`: queue-first async execution boundary
- `domain/stable_models.py`: stable public forecast/backtest helpers
- `metrics.py`, `mlflow_runs.py`: observability and experiment tracking adapters

### `src/models/`

- `gbdt_pipeline.py`: feature engineering and GBDT training/inference path
- `registry.py`: public experimental torch model registry surface

### `src/monitoring/`

- `drift_detector.py`: PSI-based drift baseline and report generation

### `src/enterprise/`

- `audit.py`: shared audit event schema
- `iam.py`, `network.py`, `tenancy.py`, `portability.py`: minimal policy helpers used by the public API control path

## Data flow summary

### Forecast

`/v1/forecast` enters through the router, passes auth and policy checks, calls the forecast service, uses runtime-loaded model bundles plus stable helpers, and returns a Pydantic response payload.

### Training and jobs

`/v1/train` validates request approval, persists a queued job, and lets a worker claim and execute the training run. Metrics and artifacts are emitted through MLflow, while registry state and job status are persisted through the configured backends.

### Drift monitoring

The monitoring endpoints persist a baseline summary and later compare candidate observations against that baseline. Feature-level severity can then feed metrics, dashboards, and promotion decisions.

## Dependency direction

The intended dependency direction is one-way:

- routers call services
- services depend on schemas, errors, and injected runtime dependencies
- `app.py` wires dependencies together but should not remain the long-term implementation home for business logic

This is why compatibility wrappers are being retired in stages rather than deleted in a single rewrite.

## Technology choices

- FastAPI for typed API contracts and automatic OpenAPI generation
- HistGradientBoosting and related GBDT tooling for the public benchmark path
- MLflow for runs, metrics, and artifacts
- DVC for data lineage and reproducible benchmark flow
- SQLite locally and PostgreSQL optionally for durable shared state
- Cloud Run, Cloud SQL, and GCS as the intended production-style deployment reference

## CI/CD posture

The stable CI path covers lint, typecheck, security scanning, stable pytest, benchmark regeneration, benchmark regression, DVC dry run, MLflow smoke, PostgreSQL-backed control-plane checks, and Playwright E2E.

The release goal is not “many tools for their own sake”. The goal is to show that the public benchmark and API contract are supported by reproducibility, observability, and rollback discipline.