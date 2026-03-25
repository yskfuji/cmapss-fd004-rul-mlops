# ADR 0006: Retire app.py Compatibility Wrappers in Stages

## Status / ステータス

Accepted

採用

## Context / 背景

`src/forecasting_api/app.py` started as both the FastAPI composition root and the main location for
forecast, training, auth, middleware, and runtime helper logic. Recent refactors moved large parts of
that behavior into smaller modules such as `auth.py`, `request_middleware.py`, `training_helpers.py`,
`runtime_state.py`, and service-layer modules.

The remaining issue is that `app.py` still exports compatibility wrappers and legacy helper symbols.
Those symbols are still referenced by tests and by some import paths that assume `forecasting_api.app`
is both the ASGI factory and the implementation module. Deleting the wrappers immediately would break
the test suite and make the refactor harder to land safely.

これらの symbol は依然として test や、`forecasting_api.app` を ASGI factory 兼 implementation module とみなす import path から参照されています。wrapper を一度に削除すると test suite が壊れ、refactor を安全に着地させにくくなります。

## Decision / 判断

Retire the `app.py` compatibility wrappers in three explicit stages instead of removing them in one
rewrite.

`app.py` の compatibility wrapper は、一度の rewrite で削除せず、3 段階で明示的に退役させます。

### Stage 1: Preserve behavior, stop new coupling

- Keep `app.py` as the composition root and compatibility surface.
- Continue extracting implementation into dedicated modules.
- Do not add new helper logic to `app.py` unless it is strictly composition-only.
- Update new tests to patch the extracted modules or service entry points instead of patching legacy
  helpers on `forecasting_api.app`.

### Stage 2: Migrate call sites and tests

- Move remaining helper branches behind explicit module boundaries.
- Replace legacy monkeypatch sites in unit and integration tests with patches against the new modules.
- Remove compatibility imports from internal callers so `app.py` is no longer an implementation
  dependency.
- Keep only the ASGI factory, route registration, and dependency wiring in `app.py`.

### Stage 3: Remove wrappers

- Delete compatibility wrappers once internal imports and tests no longer depend on them.
- Keep `create_app()` and the exported ASGI application entry point as the stable public surface.
- Treat any still-missing import migration as a release blocker for wrapper removal.

## Consequences / 帰結

- The refactor remains backward-compatible while the codebase is being decomposed.
- The repository gains a clear exit criterion for reducing `app.py` to a real composition root.
- Test maintenance work becomes part of the architectural migration, not a follow-up chore.
- `app.py` will remain larger than ideal until Stage 2 is mostly complete; that is an intentional,
  temporary trade-off to avoid a destabilizing big-bang rewrite.

- codebase 分解中も backward compatibility を保てる。
- `app.py` を本来の composition root に縮退させるための exit criterion が明確になる。
- test 保守は後回しではなく、architecture migration の一部になる。
- Stage 2 が進むまでは `app.py` が理想より大きい状態を許容するが、これは不安定な一括 rewrite を避けるための意図的な一時措置である。