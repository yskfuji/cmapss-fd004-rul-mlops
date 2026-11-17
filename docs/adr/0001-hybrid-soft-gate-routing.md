# ADR 0001: Hybrid Soft-Gate Routing

## Status / ステータス

Superseded for public profile

公開プロファイルでは置換済み

## Context / 背景

The codebase still contains hybrid routing logic for internal experimentation, but the public repository has been repositioned around a reproducible GBDT-first benchmark and API surface.

codebase には内部実験向けの hybrid routing logic が残っていますが、公開 repository は再現可能な GBDT-first benchmark と API surface を中心に再定義されています。

## Decision / 判断

Do not present the hybrid soft-gate router as the default public inference path.

Keep the implementation in the repository for internal comparison only, and require explicit opt-in through `RULFM_ENABLE_EXPERIMENTAL_MODELS=1` before those algorithms become callable from the API.

hybrid soft-gate router を既定の公開 inference path として提示しません。

実装は内部比較用に repository に残しますが、API から呼び出せるようにするには `RULFM_ENABLE_EXPERIMENTAL_MODELS=1` による明示 opt-in を必須にします。

## Consequences / 帰結

- The public README, OpenAPI, and health contract stay aligned with the shipped benchmark artifact.
- Internal hybrid experiments remain available without being mistaken for the supported public path.
- CI and docs can treat hybrid behavior as experimental instead of a default production guarantee.

- 公開 README、OpenAPI、health contract を、出荷済み benchmark artifact と整合させたまま維持できる。
- 内部 hybrid experiment は残せるが、公開の既定 path と誤解されにくくなる。
- CI と docs は、hybrid behavior を既定本番保証ではなく experimental として扱える。
