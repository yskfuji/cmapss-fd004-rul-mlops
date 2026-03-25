# ADR 0002: In-Memory Job Store

## Status / ステータス

Accepted

採用

## Context / 背景

This portfolio is intentionally scoped to showcase production API design, not distributed task orchestration.

この portfolio は、分散 task orchestration ではなく production API design を示すことを目的に範囲を絞っています。

## Decision / 判断

Retain the in-memory job store for async demos. Do not replace it with Redis or Celery in this repository.

async demo 用には in-memory job store を維持します。この repository で Redis や Celery へ置き換えません。

## Consequences / 帰結

- The code stays self-contained and easy to run locally.
- Operational trade-offs are explicit in documentation instead of hidden behind infrastructure complexity.
- CI and containerized demos remain lightweight.

- code は self-contained のままで、ローカル実行しやすい。
- 運用上の trade-off を、複雑な infrastructure の陰ではなく documentation に明示できる。
- CI と container demo を軽量に保てる。
