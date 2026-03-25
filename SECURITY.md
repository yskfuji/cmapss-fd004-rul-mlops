# Security Policy

## English

### Supported scope

This repository is a public production-style reference for CMAPSS FD004 RUL forecasting.
The maintained public scope is the default GBDT-serving path, CI/CD workflows, and the
documented Cloud Run and PostgreSQL deployment path.

Experimental torch and hybrid paths remain best-effort and may change without notice.

### Reporting a vulnerability

Please do not open a public GitHub issue for undisclosed vulnerabilities.

Report security issues by email to the repository maintainer with:

- affected component and version or commit SHA
- reproduction steps or proof of concept
- impact assessment
- any suggested mitigation

The maintainer will acknowledge receipt, validate the report, and coordinate a fix or a
documented risk acceptance path.

### Security expectations

- secrets must not be committed to the repository
- production credentials must be supplied through Secret Manager or environment-specific secret stores
- local demo defaults are not suitable for internet-facing deployment
- Cloud Run deployments should keep authenticated ingress and externalized mutable state

## 日本語

### 対応範囲

この repository は、CMAPSS FD004 RUL forecasting のための public production-style reference です。
保守対象の公開範囲は、既定の GBDT serving path、CI/CD workflow、そして文書化済みの
Cloud Run / PostgreSQL deployment path です。

Torch 系や hybrid 系の experimental path は best-effort 扱いで、予告なく変更される場合があります。

### 脆弱性報告

未公開の脆弱性について、公開 GitHub issue を作成しないでください。

以下を添えて、repository maintainer にメールで報告してください。

- 影響を受ける component と version または commit SHA
- 再現手順または proof of concept
- 影響評価
- 提案できる暫定 mitigation

maintainer は受領を確認し、内容を検証した上で、修正または文書化された risk acceptance を調整します。

### セキュリティ前提

- secret を repository に commit しない
- 本番 credential は Secret Manager など環境別の secret store から供給する
- ローカル demo 用 default は internet-facing deployment に使わない
- Cloud Run deployment では authenticated ingress と外部化された mutable state を維持する
