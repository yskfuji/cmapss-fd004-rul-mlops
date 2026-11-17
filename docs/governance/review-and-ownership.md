# Review And Ownership Model

## English

This repository is maintained by a single primary owner today. That constraint is real, so the anti-fragility mechanism cannot be "pretend there is a team".

### Current model

- CODEOWNERS is single-maintainer because the repository currently has one accountable maintainer.
- Risk is reduced with process controls rather than fake reviewer entries.
- Blocking CI, ADRs, release notes, and exception registers are treated as the minimum review surface.

### Required controls for change approval

- CI must pass before merge or release.
- Any coverage threshold change must update `pyproject.toml`, `docs/architecture/coverage-plan.md`, and CI in the same change.
- Any security exception must be recorded in `docs/governance/security-exceptions.md`.
- Any externally visible behavior change must be reflected in `CHANGELOG.md`.
- Architecture-affecting decisions should either update an existing ADR or add a new one.

### Single-maintainer operating discipline

- Use the PR template checklist even for self-review.
- Treat release notes, smoke checks, and rollback notes as part of the definition of done.
- Prefer reversible deployment changes and sha-pinned rollback targets.
- Keep public benchmark claims narrower than the evidence currently committed in the repository.

### Exit criteria for multi-reviewer mode

If this repository gains regular contributors, CODEOWNERS should move to at least two maintainers and the PR template should require one reviewer outside the author for release-affecting changes.

## 日本語

この repository は現在、単一の primary owner によって保守されています。この制約は実在するため、反脆弱性の手段を「チームがいるふり」で置き換えることはしません。

### 現在の運用モデル

- CODEOWNERS は単一 maintainer 構成です。理由は、現時点で責任を持つ maintainer が 1 名だからです。
- リスク低減は、見せかけの reviewer 指定ではなく process control で行います。
- blocking CI、ADR、release notes、exception register を最小 review surface として扱います。

### 変更承認に必須の control

- merge または release 前に CI が通っていること。
- coverage threshold を変える場合は、同じ変更で `pyproject.toml`、`docs/architecture/coverage-plan.md`、CI を更新すること。
- security exception を追加したら `docs/governance/security-exceptions.md` に記録すること。
- 外部から見える振る舞いの変更は `CHANGELOG.md` に反映すること。
- architecture に影響する判断は、既存 ADR の更新または新規 ADR 追加で残すこと。

### 単独 maintainer 時の運用規律

- self-review でも PR template checklist を使う。
- release notes、smoke check、rollback note を definition of done の一部として扱う。
- reversible な deployment change と sha 固定の rollback target を優先する。
- repository に commit 済みの証拠より広い benchmark claim を公開面でしない。

### 複数 reviewer モードへの移行条件

継続的 contributor が増えた場合、CODEOWNERS は少なくとも 2 maintainer 体制に移行し、release 影響のある変更では author 以外の reviewer を 1 名以上必須にします。