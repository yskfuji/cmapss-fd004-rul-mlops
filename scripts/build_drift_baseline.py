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
        description="Build and persist a drift baseline from records."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--minimum-samples-per-feature", type=int, default=50)
    return parser


def main() -> int:
    _bootstrap()
    from monitoring.drift_detector import (
        DriftDetector,
        has_sufficient_baseline_samples,
        save_baseline,
    )

    args = build_parser().parse_args()
    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    detector = DriftDetector(bins=args.bins)
    baseline = detector.summarize_baseline(records)
    if not has_sufficient_baseline_samples(
        baseline,
        minimum_count=args.minimum_samples_per_feature,
    ):
        raise ValueError(
            "baseline does not meet the minimum sample count per feature; "
            "provide more reference records before persisting it"
        )
    save_baseline(baseline, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
