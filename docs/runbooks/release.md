# Release Runbook

## English

### Versioning rule

- Python package version and root Node harness version move together.
- The root Node package exists only for test and E2E tooling, but it is versioned so releases are traceable across the whole repository.
- Public releases are repository releases, not an npm package distribution promise.
- The current portfolio baseline is `0.0.0`.

### Release checklist

1. Update version in `pyproject.toml` and `package.json`.
2. Add a dated entry to `CHANGELOG.md`.
3. Confirm `docs/architecture/coverage-plan.md` still matches the blocking gate.
4. Confirm `docs/governance/security-exceptions.md` still reflects every ignored advisory in CI.
5. Run stable tests and the Playwright E2E suite.
6. Verify the deployment smoke script still covers health, docs, metrics, forecast, backtest, and jobs.
7. Record any known limitations explicitly in the release notes instead of burying them in code comments.

### Release evidence to keep

- stable pytest summary
- current coverage summary
- Playwright summary
- deployed smoke summary from CD
- image or commit SHA used for rollback

### Rollback rule

- Roll back by previous sha-tagged image, not by mutating release history.
- If a rollback needs state restoration, restore the matching Cloud SQL and runtime bucket snapshot together.

## 日本語

### version ルール

- Python package version と root の Node harness version は同時に更新します。
- root の Node package は test / E2E 用ですが、repository 全体の release を追跡できるよう version を持たせます。
- 公開 release は repository release であり、npm package の配布保証ではありません。
- 現在の portfolio 基準 version は `0.0.0` です。

### release checklist

1. `pyproject.toml` と `package.json` の version を更新する。
2. `CHANGELOG.md` に日付付き entry を追加する。
3. `docs/architecture/coverage-plan.md` が blocking gate と一致していることを確認する。
4. `docs/governance/security-exceptions.md` が CI で ignore している advisory をすべて反映していることを確認する。
5. stable test と Playwright E2E suite を実行する。
6. deployment smoke script が health、docs、metrics、forecast、backtest、jobs を引き続き検証していることを確認する。
7. 既知の制約は code comment に埋めず、release notes に明示する。

### 保持すべき release evidence

- stable pytest の要約
- 現在の coverage 要約
- Playwright の要約
- CD から得た deployed smoke の要約
- rollback に使う image または commit SHA

### rollback ルール

- release history を書き換えず、前の sha-tagged image に戻す。
- state の復元が必要なら、対応する Cloud SQL snapshot と runtime bucket snapshot を必ずセットで戻す。