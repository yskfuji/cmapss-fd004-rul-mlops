from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _bootstrap() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote an RULFM model based on evaluation metrics."
    )
    parser.add_argument("model_id")
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--target-stage", default="staging")
    return parser


def main() -> int:
    _bootstrap()
    from forecasting_api.model_promotion import promote_model

    args = build_parser().parse_args()
    metrics = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
    decision = promote_model(args.model_id, metrics, target_stage=args.target_stage)
    print(
        json.dumps(
            {
                "model_id": decision.model_id,
                "target_stage": decision.target_stage,
                "approved": decision.approved,
                "reasons": decision.reasons,
                "metrics": decision.metrics,
                "promoted_at": decision.promoted_at,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0 if decision.approved else 1


if __name__ == "__main__":
    raise SystemExit(main())
