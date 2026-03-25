# ADR 0003: MLflow and DVC Responsibility Boundary

## Status / ステータス

Accepted

採用

## Context / 背景

The repository uses MLflow for experiment metadata and adds DVC for reproducible data lineage.

この repository は experiment metadata に MLflow を使い、再現可能な data lineage のために DVC を追加しています。

## Decision / 判断

Use MLflow for runs, metrics, and artifacts. Use DVC for dataset lineage, pipeline reproducibility, and benchmark outputs.

run、metric、artifact は MLflow に任せ、dataset lineage、pipeline reproducibility、benchmark output は DVC に任せます。

## Consequences / 帰結

- Promotion workflows can reference MLflow metrics without overloading DVC.
- Reproducibility remains auditable through versioned pipeline stages and params.
- The two systems stay decoupled and understandable in interviews or design reviews.

- promotion workflow は、DVC に役割を過積載させずに MLflow metric を参照できる。
- 再現性は、version 化された pipeline stage と parameter を通じて監査可能なまま残る。
- 2 つの仕組みを疎結合に保てるため、面接や design review でも説明しやすい。
