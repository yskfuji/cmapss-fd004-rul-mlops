# cmapss-fd004-rul-mlops

[![CI](https://github.com/yusukefujinami/cmapss-fd004-rul-mlops/actions/workflows/ci-stable.yml/badge.svg)](https://github.com/yusukefujinami/cmapss-fd004-rul-mlops/actions/workflows/ci-stable.yml)

NASA CMAPSS FD004 Remaining Useful Life 予測のための公開ポートフォリオ用リポジトリです。

**言語:** [English](README.md) | 日本語

## ポートフォリオ版

- 公開ポートフォリオ版タグ: `0.0.0`
- version の基準: `pyproject.toml`、`package.json`、FastAPI OpenAPI metadata、`CHANGELOG.md`
- 詳細 runbook:
  - `docs/runbooks/operations.md` / `docs/runbooks/operations.ja.md`
  - `docs/runbooks/release.md`
  - `docs/governance/`

## このリポジトリが示すもの

- FD004 向けの再現可能な公開 GBDT ベンチマーク
- forecast、backtest、jobs、metrics、monitoring を公開する FastAPI サービス
- その周囲にある release、audit、drift、promotion、approval といった運用統制

ポイントは、製品面の主役が「公開ベンチマークと API」であり、OIDC、tenant policy、audit logging、job orchestration はその周辺の control plane 設計を示すための補助的な実装だという点です。

## 公開ベンチマークの要点

基準 artifact は `src/forecasting_api/data/fd004_benchmark_summary.json` にコミットされています。

- 評価対象: 237 台の対象 test engine の最終サイクル予測
- 学習 profile: engine ごとの full cycle
- benchmark stage: `gbdt-only`
- preset: `full`
- ensemble flag: `RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0`

代表モデルの公開値:

- RMSE: 15.1594
- MAE: 10.4467
- NASA score: 968.3299
- bias: -2.2130
- cov90: 0.9114

過去の公開 snapshot では学習ラベル範囲が実質 `[0, 89]` に潰れており、大きな負方向 bias が出ていました。現行 benchmark は全 cycle 学習に切り替え、ラベル範囲 `[0, 125]` を回復しています。

## 最小セットアップ

```bash
pip install -r requirements-lock.txt
export RULFM_FORECASTING_API_KEY=local-demo-key
PYTHONPATH=src uvicorn forecasting_api.app:create_app --factory --port 8000
```

追加の control plane や PostgreSQL 構成は `docs/runbooks/operations.md` と `docs/runbooks/operations.ja.md` を参照してください。

## ベンチマーク再現

```bash
RULFM_BENCHMARK_STAGE=gbdt-only \
RULFM_BENCHMARK_GBDT_PRESET=full \
RULFM_BENCHMARK_ENABLE_GBDT_ENSEMBLE=0 \
PYTHONPATH=src python scripts/build_fd004_benchmark_summary.py
```

## API と監視

- API docs: `/docs`, `/docs/en`, `/docs/ja`
- health: `/health`
- metrics: `/metrics`
- UI: `/ui/forecasting/`

`/metrics` は認証付きです。公開 quickstart では最小限の API key 構成のみを示し、tenant policy や train approval などの高度な統制は runbook 側に分離しています。

## 公開上の読み方

このリポジトリは「すでに大規模本番運用されている SaaS」の証明ではなく、次を示す production-style reference implementation として読むのが正確です。

- 再現可能な benchmark
- API と非同期 job 面
- drift / metrics / promotion の観測面
- Cloud Run / Cloud SQL / GCS を前提にした control plane の設計

## 日英ドキュメント導線

| ドキュメント | 英語 | 日本語 |
|---|---|---|
| README | [README.md](README.md) | このファイル |
| 運用 Runbook | [operations.md](docs/runbooks/operations.md) | [operations.ja.md](docs/runbooks/operations.ja.md) |
| アーキテクチャ概要 | [overview.en.md](docs/architecture/overview.en.md) | [overview.md](docs/architecture/overview.md)（EN/JA 混在） |
| ADR 群 | [docs/adr/](docs/adr/) | 各 ADR ファイル内に EN/JA 併記 |
| governance / release | [docs/governance/](docs/governance/) / [release.md](docs/runbooks/release.md) | 各ファイル内に EN/JA 併記 |