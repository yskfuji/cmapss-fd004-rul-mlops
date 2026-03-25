from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _bootstrap() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Request a drift report from the RULFM API."
    )
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--output", required=True)
    return parser


def post_with_retry(
    *,
    httpx_module,
    api_url: str,
    api_key: str,
    payload: dict[str, object],
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = httpx_module.post(
                f"{api_url.rstrip('/')}/v1/monitoring/drift/report",
                headers={"X-API-Key": api_key},
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            body = response.json()
            return body if isinstance(body, dict) else {"result": body}
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if (
                isinstance(status_code, int)
                and 400 <= status_code < 500
                and status_code not in {408, 429}
            ):
                raise RuntimeError(
                    f"drift report request failed with non-retryable status {status_code}"
                ) from exc
            last_error = exc
            if attempt == 2:
                break
            time.sleep(2**attempt)
    raise RuntimeError("failed to generate drift report after retries") from last_error


def main() -> int:
    _bootstrap()
    import httpx

    args = build_parser().parse_args()
    candidate_path = Path(args.candidate_file)
    if not candidate_path.exists():
        raise FileNotFoundError(f"candidate file not found: {candidate_path}")

    payload = {"candidate_records": json.loads(candidate_path.read_text(encoding="utf-8"))}
    response_json = post_with_retry(
        httpx_module=httpx,
        api_url=args.api_url,
        api_key=args.api_key,
        payload=payload,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(response_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
