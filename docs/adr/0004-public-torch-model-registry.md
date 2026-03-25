# ADR 0004: Public Torch Model Registry

## Status / ステータス

Accepted for experimental benchmark support

experimental benchmark support 向けに採用

## Context / 背景

The proprietary temporal model registry (`src.models.registry`) contains AFNOcG3 and related
architectures that are omitted from the public repository pending patent filing.
The benchmark script (`scripts/build_fd004_benchmark_summary.py`) imports from this registry
via `torch_forecasters.py`; when the registry is absent the import raises `ModuleNotFoundError`,
causing all torch benchmark rows to fall back to a flat Ridge surrogate
(`structure_mode: flat_ridge`). This made the Huber vs asymmetric-RUL loss comparison
unobservable in the public repo — both rows reported identical metrics.

その結果、torch benchmark row がすべて flat Ridge surrogate
(`structure_mode: flat_ridge`) に fallback し、Huber と asymmetric-RUL loss の比較が
公開 repo では観測できなくなっていました。

Two options were evaluated:

**Option A — Public surrogate registry**
Create `src/models/registry.py` with BiLSTM, TCN, and Transformer RUL implementations that
satisfy the same `build` / `load_from_snapshot` / `extras` contract expected by
`torch_forecasters.py`. AFNO variants raise `ModuleNotFoundError(name="src.models.registry")`
to preserve the existing fallback path.

**Option B — Optional dependency gate**
Wrap the registry import in a try/except and skip the torch stage when the proprietary
registry is absent, emitting explicit `not_available` rows.

## Decision / 判断

Adopt **Option A** for experimental and research use, but do not treat torch rows as the default public portfolio story.

experimental / research 用には **Option A** を採用しますが、torch row を既定の公開ポートフォリオ本筋とは扱いません。

Rationale:
- BiLSTM, TCN, and Transformer are well-established public architectures with no IP concerns.
  Implementing them as surrogates allows the full benchmark pipeline to run end-to-end and
  produces meaningful, reproducible metrics for the asymmetric-loss comparison.
- Option B only documents absence; Option A produces actual results that a reviewer can
  inspect and reproduce.
- The AFNO-specific `ModuleNotFoundError` with `name="src.models.registry"` is preserved so
  `torch_forecasters.py` can distinguish "known proprietary model absent" from "unknown key"
  without code changes in the existing fallback path.

## Implementation / 実装

`src/models/registry.py` (251 lines) provides:
- `_BiLSTMRUL` — bidirectional LSTM + linear head, input `(B, T, D)` → output `(B, 1)`
- `_TCNRUL` — dilated causal conv residual blocks + global avg pool + linear head
- `_TransformerRUL` — sinusoidal PE + `TransformerEncoder` + linear head
- `_ModelHandlers` — `build` / `load_from_snapshot` / `extras` contract
- `get_model_handlers(algo_key)` — raises `ModuleNotFoundError` for AFNO variants,
  `ValueError` for unknown keys

## Consequences / 帰結

- Torch benchmark rows are reproducible when the experimental dependency set is installed.
- Public docs, OpenAPI examples, and `/health` should still describe the GBDT-first contract unless experimental models are explicitly enabled.
- Adding a new public architecture requires implementing the `build` / `load_from_snapshot` contract in this file and registering it in `_HANDLERS`.
- Proprietary architectures continue to be gated by the `ModuleNotFoundError` fallback; no changes to `torch_forecasters.py` are required when the proprietary registry is restored.

- experimental dependency を入れれば、torch benchmark row を再現できる。
- experimental model を明示的に有効化しない限り、public docs、OpenAPI 例、`/health` は GBDT-first 契約を説明し続けるべきである。
- 新しい public architecture を追加するには、この file で `build` / `load_from_snapshot` 契約を実装し `_HANDLERS` へ登録する必要がある。
- proprietary architecture は引き続き `ModuleNotFoundError` fallback で守られ、将来 proprietary registry を戻しても `torch_forecasters.py` の変更は不要である。
