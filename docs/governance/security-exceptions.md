# Security Exception Register

## English

Ignored dependency advisories in CI must be documented here before they are added to `.github/workflows/ci-stable.yml`.

### Active exceptions

| Vulnerability / 脆弱性 | Scope / 対象 | Decision / 判断 | Rationale / 理由 | Temporary mitigation / 暫定対策 | Review by / 再確認期限 | Owner / 所有者 |
|---|---|---|---|---|---|---|
| CVE-2025-69872 | `requirements-lock.txt` audit in stable CI | Temporarily ignored in blocking dependency audit | The current pinned dependency set still reports this advisory and the repository has not yet completed a compatible upgrade validation for the affected transitive dependency path. | Keep `pip-audit` blocking for all other advisories, keep secrets scan enabled, and re-evaluate the pinned dependency graph before the next dependency refresh. | 2026-04-30 | Repository maintainer |
| CVE-2026-4539 | `requirements-lock.txt` audit in stable CI | Temporarily ignored in blocking dependency audit | The advisory is still present in the pinned lock set and removal has not yet been validated against the stable API and benchmark path. | Keep the exception scoped to the stable lockfile audit only, keep experimental dependencies audited separately, and revisit during the next dependency upgrade pass. | 2026-04-30 | Repository maintainer |

### Rules

- No exception is added without a rationale, temporary mitigation, owner, and review date.
- Exceptions are temporary decisions, not a standing policy.
- The same PR that removes an ignore from CI must remove or close the corresponding row here.
- If the review date passes, either renew the exception with fresh evidence or remove the ignore.

## 日本語

CI で依存脆弱性 advisory を ignore する場合は、`.github/workflows/ci-stable.yml` に追加する前に、この文書へ記録しておく必要があります。

### ルール

- rationale、temporary mitigation、owner、review date の 4 つが揃わない exception は追加しない。
- exception は恒久ルールではなく、一時的な判断である。
- CI から ignore を削除する PR は、この表の対応行も削除または close する。
- review date を過ぎた場合は、新しい証拠を付けて更新するか、ignore 自体を削除する。