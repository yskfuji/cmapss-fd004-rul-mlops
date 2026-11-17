"""Forecasting API client CLI.

This is a minimal operator-friendly CLI to call the Forecasting API.
It is intentionally separate from the NAYF CLI (training/inference entrypoint).

Examples (scheme-free base-url for public-safe docs):
  RULFM_FORECASTING_API_KEY=dev-key \
        python -m src.rulfm.forecasting_api.client_cli --base-url 127.0.0.1:8000 health
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from src.rulfm.forecasting_api.client import ApiCallError, ForecastingApiClient, load_json_file


def _print_next_action(cmd: str) -> None:
    cmd_part = f" {cmd}" if cmd else ""
    print(
        f"次アクション: python -m src.rulfm.forecasting_api.client_cli{cmd_part} --help を確認してください。",
        file=sys.stderr,
    )


def _dump_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))


def _example_forecast_request() -> dict[str, Any]:
    # Keep this minimal and schema-valid for ForecastRequest.
    # (No network calls; safe to run in CI.)
    return {
        "horizon": 2,
        "frequency": "1d",
        "level": [80, 95],
        "data": [
            {"series_id": "s1", "timestamp": "2024-01-01T00:00:00Z", "y": 10.0},
            {"series_id": "s1", "timestamp": "2024-01-02T00:00:00Z", "y": 11.0},
            {"series_id": "s1", "timestamp": "2024-01-03T00:00:00Z", "y": 12.0},
        ],
    }


def _write_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _example_job_create_request(job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": job_type, "payload": payload}


def _coerce_timeout_seconds(raw: str) -> float:
    try:
        v = float(raw)
    except ValueError:
        raise ValueError("timeout-seconds must be a number")
    if v <= 0:
        raise ValueError("timeout-seconds must be > 0")
    return v


def _coerce_poll_interval(raw: str) -> float:
    try:
        v = float(raw)
    except ValueError:
        raise ValueError("poll-interval must be a number")
    if v <= 0:
        raise ValueError("poll-interval must be > 0")
    return v


def _require_api_key(explicit: str | None) -> str:
    key = explicit or os.getenv("RULFM_FORECASTING_API_KEY")
    if not key:
        print("ERROR: RULFM_FORECASTING_API_KEY が未設定です", file=sys.stderr)
        _print_next_action(cmd="")
        raise SystemExit(2)
    return key


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--base-url",
        default=os.getenv("RULFM_FORECASTING_API_BASE_URL", "127.0.0.1:8000"),
        help="Base URL (default: 127.0.0.1:8000). Scheme optional.",
    )
    common.add_argument(
        "--api-key",
        default=None,
        help="API key (default: env RULFM_FORECASTING_API_KEY)",
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.rulfm.forecasting_api.client_cli",
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[common],
        description=(
            "Forecasting API client CLI.\n"
            "- health: quick check\n"
            "- forecast: POST /v1/forecast with JSON request\n"
            "- jobs-create: POST /v1/jobs with JSON request\n"
            "- jobs-forecast-example: print a minimal JobCreateRequest for forecast (no network)\n"
            "- jobs-get: GET /v1/jobs/{job_id}\n"
            "- jobs-result: GET /v1/jobs/{job_id}/result\n"
            "- jobs-run-forecast-example: create+wait+result (real HTTP)\n"
            "- forecast-example: print a minimal ForecastRequest JSON (no network)\n"
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="GET /health", parents=[common])

    p_ex = sub.add_parser(
        "forecast-example",
        help="Print a minimal ForecastRequest JSON (no network)",
        parents=[common],
    )
    p_ex.add_argument(
        "--out",
        default=None,
        help="Write JSON to file instead of stdout (optional).",
    )

    p_forecast = sub.add_parser("forecast", help="POST /v1/forecast", parents=[common])
    p_forecast.add_argument("--json", required=True, help="Path to ForecastRequest JSON")

    p_jobs = sub.add_parser("jobs-create", help="POST /v1/jobs", parents=[common])
    p_jobs.add_argument("--json", required=True, help="Path to JobCreateRequest JSON")

    p_jobs_ex = sub.add_parser(
        "jobs-forecast-example",
        help="Print a minimal JobCreateRequest for forecast (no network)",
        parents=[common],
    )
    p_jobs_ex.add_argument(
        "--out",
        default=None,
        help="Write JSON to file instead of stdout (optional).",
    )

    p_jobs_get = sub.add_parser("jobs-get", help="GET /v1/jobs/{job_id}", parents=[common])
    p_jobs_get.add_argument("job_id", help="Job ID")

    p_jobs_result = sub.add_parser(
        "jobs-result",
        help="GET /v1/jobs/{job_id}/result",
        parents=[common],
    )
    p_jobs_result.add_argument("job_id", help="Job ID")

    p_jobs_run = sub.add_parser(
        "jobs-run-forecast-example",
        help="Create forecast job, wait for completion, then fetch result (real HTTP)",
        parents=[common],
    )
    p_jobs_run.add_argument(
        "--timeout-seconds",
        default=os.getenv("RULFM_FORECASTING_API_JOBS_TIMEOUT_SECONDS", "10"),
        help="Overall timeout seconds (default: 10).",
    )
    p_jobs_run.add_argument(
        "--poll-interval",
        default=os.getenv("RULFM_FORECASTING_API_JOBS_POLL_INTERVAL", "0.1"),
        help="Polling interval seconds (default: 0.1).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "forecast-example":
        payload = _example_forecast_request()
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.out:
            try:
                _write_text_file(args.out, text)
            except Exception as e:
                print(f"ERROR: failed to write file: {e}", file=sys.stderr)
                return 3
        else:
            print(text, end="")
        return 0

    if args.cmd == "jobs-forecast-example":
        payload = _example_job_create_request("forecast", _example_forecast_request())
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.out:
            try:
                _write_text_file(args.out, text)
            except Exception as e:
                print(f"ERROR: failed to write file: {e}", file=sys.stderr)
                return 3
        else:
            print(text, end="")
        return 0

    api_key = _require_api_key(args.api_key)
    client = ForecastingApiClient(base_url=args.base_url, api_key=api_key)
    try:
        if args.cmd == "health":
            out = client.request("GET", "/health")
            _dump_json(out)
            return 0

        if args.cmd == "forecast":
            req = load_json_file(args.json)
            out = client.request("POST", "/v1/forecast", json_body=req)
            _dump_json(out)
            return 0

        if args.cmd == "jobs-create":
            req = load_json_file(args.json)
            out = client.request("POST", "/v1/jobs", json_body=req)
            _dump_json(out)
            return 0

        if args.cmd == "jobs-get":
            out = client.request("GET", f"/v1/jobs/{args.job_id}")
            _dump_json(out)
            return 0

        if args.cmd == "jobs-result":
            out = client.request("GET", f"/v1/jobs/{args.job_id}/result")
            _dump_json(out)
            return 0

        if args.cmd == "jobs-run-forecast-example":
            timeout_s = _coerce_timeout_seconds(args.timeout_seconds)
            poll_interval_s = _coerce_poll_interval(args.poll_interval)

            job_req = _example_job_create_request("forecast", _example_forecast_request())
            created = client.request("POST", "/v1/jobs", json_body=job_req)
            job_id = created.get("job_id") if isinstance(created, dict) else None
            if not isinstance(job_id, str) or not job_id:
                raise RuntimeError("Unexpected jobs-create response shape")

            deadline = time.monotonic() + timeout_s
            last_status: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                st = client.request("GET", f"/v1/jobs/{job_id}")
                if isinstance(st, dict):
                    last_status = st
                    status = st.get("status")
                    if status == "succeeded":
                        result = client.request("GET", f"/v1/jobs/{job_id}/result")
                        _dump_json({"job_id": job_id, "status": status, "result": result})
                        return 0
                    if status == "failed":
                        err = st.get("error")
                        if isinstance(err, dict):
                            print(f"ERROR: status=200", file=sys.stderr)
                            if err.get("error_code"):
                                print(f"error_code={err.get('error_code')}", file=sys.stderr)
                            if err.get("request_id"):
                                print(f"request_id={err.get('request_id')}", file=sys.stderr)
                            if err.get("details"):
                                print(json.dumps(err.get("details"), ensure_ascii=False), file=sys.stderr)
                        return 3
                time.sleep(poll_interval_s)

            _dump_json({"job_id": job_id, "status": "timeout", "last": last_status})
            return 3

        print("ERROR: unknown cmd", file=sys.stderr)
        return 2
    except ApiCallError as e:
        print(f"ERROR: status={e.status_code}", file=sys.stderr)
        if e.error_code:
            print(f"error_code={e.error_code}", file=sys.stderr)
        if e.request_id:
            print(f"request_id={e.request_id}", file=sys.stderr)
        if e.details:
            print(json.dumps(e.details, ensure_ascii=False), file=sys.stderr)
        return 3
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
