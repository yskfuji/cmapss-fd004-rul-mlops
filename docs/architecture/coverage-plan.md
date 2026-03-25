# Coverage Ratchet Plan

## English

### Current baseline

- blocking stable gate: `72%`
- current posture: public GBDT/API path first, experimental torch/hybrid path second
- next ratchet candidate: `80%` once the stable workflow clears it consistently without waivers

### Why this document changed

- The blocking gate in pyproject.toml and ci-stable is already `72%`.
- This document now reflects the real operating threshold rather than an obsolete historical target.
- `80%` remains the next ratchet, but only after the stable profile proves it in normal CI runs with margin.

### What counts as high-value coverage now

1. Public API control paths that are part of the documented contract
2. Benchmark regression checks for the committed FD004 artifact
3. Operational behavior: auth, jobs, drift, promotion, metrics, persistence
4. Experimental model gates, so public and internal surfaces do not drift again

### Next ratchet candidates

| Priority | Area | Example assertion |
|---|---|---|
| 1 | `/v1/train` public `gbdt_hgb_v1` path | accepted response plus persisted artifact metadata |
| 2 | `/v1/forecast` with trained `gbdt_hgb_v1` model | non-naive predictions and interval payload |
| 3 | `/v1/backtest` with trained `gbdt_hgb_v1` model | expected metric payload shape |
| 4 | experimental model gate | experimental/hybrid algo rejected unless `RULFM_ENABLE_EXPERIMENTAL_MODELS=1` |
| 5 | benchmark artifact smoke/regression | committed JSON remains within public KPI thresholds |

### Guardrails

- Do not raise the gate based on low-signal assertions that only exercise serialization.
- Keep public-path coverage ahead of experimental-path coverage.
- Raise the threshold only in the same change that proves the new baseline is already green in CI.
- Any document, workflow, and local test command that mentions the gate must be updated in the same PR as the threshold change.

### Ownership and rule

- Every coverage-increase PR should include the before/after `pytest --cov=src --cov-report=term-missing` summary.
- Any gate change must be reflected in `.github/workflows/ci-stable.yml`, `pyproject.toml`, and this file.
- Never lower the gate. Ratchet upward only after the next milestone is already passing.

## 日本語

### 現在の基準線

- blocking stable gate: `72%`
- 現在の優先順位: public GBDT/API path を先に、experimental torch/hybrid path を後に置く
- 次の ratchet 候補: stable workflow が waiver なしで継続的に通るなら `80%`

### この文書を更新した理由

- pyproject.toml と ci-stable の blocking gate はすでに `72%` です。
- この文書は、過去の古い目標ではなく、現在の実運用基準を反映するために更新しました。
- `80%` は次の ratchet 値ですが、stable profile が通常 CI で余裕を持って通ることが前提です。

### 現時点で価値の高い coverage

1. 文書化された公開契約に含まれる API control path
2. commit 済み FD004 artifact に対する benchmark regression check
3. auth、jobs、drift、promotion、metrics、persistence といった運用挙動
4. experimental model gate。公開面と内部面のズレ再発を防ぐため

### ガードレール

- serialization だけを触る低シグナルな assertion で gate を上げない。
- 常に public path の coverage を experimental path より先行させる。
- threshold を上げるのは、新しい基準がすでに CI で green と証明された変更の中だけにする。
- gate を言及する文書、workflow、ローカル test command は同じ PR で同時更新する。

### 所有とルール

- coverage 増加 PR には `pytest --cov=src --cov-report=term-missing` の before/after 要約を付ける。
- gate 変更は `.github/workflows/ci-stable.yml`、`pyproject.toml`、この文書へ必ず反映する。
- gate は下げない。次の基準が既に通っていると示せた場合にのみ引き上げる。
