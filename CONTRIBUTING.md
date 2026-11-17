# Contributing

## English

### Workflow

1. Open an issue or draft design note for non-trivial changes.
2. Keep changes focused and avoid unrelated refactors.
3. Preserve the public GBDT-first contract unless the change explicitly expands it.
4. Update tests and docs together with behavior changes.

### Local validation

Run the stable local validation path before opening a pull request:

```bash
export PYTHONPATH=src
export RULFM_FORECASTING_API_KEY=ci-test-key
.venv/bin/python -m pytest -m "not experimental" --cov=src --cov-report=term-missing --cov-fail-under=72
```

Useful focused checks:

```bash
.venv/bin/python -m mypy --explicit-package-bases --follow-imports=silent src/forecasting_api/app.py
npm run test:e2e
```

### Review expectations

- document any new environment variables
- avoid expanding compatibility wrappers unless there is a migration reason
- prefer dependency injection or focused adapters over new module globals
- keep mutable runtime state outside source directories

## 日本語

### 基本フロー

1. 影響の大きい変更は、先に issue か設計メモを作成してください。
2. 変更は小さく保ち、無関係な refactor を混ぜないでください。
3. 明示的に範囲を広げる変更でない限り、公開契約は GBDT-first を維持してください。
4. 振る舞いを変えたら、必ず test と docs を同じ変更に含めてください。

### ローカル検証

Pull Request 前には、少なくとも stable 相当の検証を実行してください。

```bash
export PYTHONPATH=src
export RULFM_FORECASTING_API_KEY=ci-test-key
.venv/bin/python -m pytest -m "not experimental" --cov=src --cov-report=term-missing --cov-fail-under=72
```

絞った確認として次も有効です。

```bash
.venv/bin/python -m mypy --explicit-package-bases --follow-imports=silent src/forecasting_api/app.py
npm run test:e2e
```

### レビュー観点

- 新しい環境変数を追加したら文書化する
- migration 理由がない限り compatibility wrapper を増やさない
- module global より dependency injection か小さな adapter を優先する
- 可変な runtime state は source directory の外に置く
