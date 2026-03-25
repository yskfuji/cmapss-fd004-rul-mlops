#!/usr/bin/env bash

set -euo pipefail

: "${DEPLOY_BASE_URL:?DEPLOY_BASE_URL is required}"
: "${DEPLOY_IDENTITY_TOKEN:?DEPLOY_IDENTITY_TOKEN is required}"
: "${DEPLOY_API_KEY:?DEPLOY_API_KEY is required}"
: "${DEPLOY_TENANT_ID:?DEPLOY_TENANT_ID is required}"

auth_header=("-H" "Authorization: Bearer ${DEPLOY_IDENTITY_TOKEN}")
api_headers=(
  "-H" "X-API-Key: ${DEPLOY_API_KEY}"
  "-H" "X-Tenant-Id: ${DEPLOY_TENANT_ID}"
  "-H" "X-Connection-Type: ${DEPLOY_CONNECTION_TYPE:-private}"
  "-H" "X-Forwarded-For: ${DEPLOY_X_FORWARDED_FOR:-10.0.0.10}"
  "-H" "Content-Type: application/json"
)

forecast_payload='{"horizon":1,"frequency":"1d","data":[{"series_id":"s1","timestamp":"2026-03-20T00:00:00Z","y":10.0},{"series_id":"s1","timestamp":"2026-03-21T00:00:00Z","y":11.0}]}'
backtest_payload='{"horizon":2,"folds":2,"metric":"rmse","data":[{"series_id":"s1","timestamp":"2026-03-20T00:00:00Z","y":10.0},{"series_id":"s1","timestamp":"2026-03-21T00:00:00Z","y":11.0},{"series_id":"s1","timestamp":"2026-03-22T00:00:00Z","y":12.0},{"series_id":"s1","timestamp":"2026-03-23T00:00:00Z","y":13.0},{"series_id":"s1","timestamp":"2026-03-24T00:00:00Z","y":14.0},{"series_id":"s1","timestamp":"2026-03-25T00:00:00Z","y":15.0},{"series_id":"s1","timestamp":"2026-03-26T00:00:00Z","y":16.0},{"series_id":"s1","timestamp":"2026-03-27T00:00:00Z","y":17.0}]}'
job_payload='{"type":"forecast","payload":{"horizon":1,"frequency":"1d","data":[{"series_id":"s1","timestamp":"2026-03-20T00:00:00Z","y":10.0},{"series_id":"s1","timestamp":"2026-03-21T00:00:00Z","y":11.0}]}}'

curl --fail --show-error --silent "${auth_header[@]}" "${DEPLOY_BASE_URL}/docs/en" >/dev/null

metrics_unauth_status="$(curl --silent --output /dev/null --write-out '%{http_code}' "${auth_header[@]}" "${DEPLOY_BASE_URL}/metrics")"
if [[ "${metrics_unauth_status}" != "401" ]]; then
  echo "Expected unauthenticated metrics probe to return 401, got ${metrics_unauth_status}" >&2
  exit 1
fi

curl --fail --show-error --silent "${auth_header[@]}" -H "X-API-Key: ${DEPLOY_API_KEY}" "${DEPLOY_BASE_URL}/metrics" >/dev/null

forecast_resp="$(curl --fail --show-error --silent "${auth_header[@]}" "${api_headers[@]}" -d "${forecast_payload}" "${DEPLOY_BASE_URL}/v1/forecast")"
FORECAST_RESP="${forecast_resp}" python - <<'PY'
import json
import os

payload = json.loads(os.environ["FORECAST_RESP"])
assert payload["forecasts"], payload
assert payload["forecasts"][0]["series_id"] == "s1", payload
PY

backtest_resp="$(curl --fail --show-error --silent "${auth_header[@]}" "${api_headers[@]}" -d "${backtest_payload}" "${DEPLOY_BASE_URL}/v1/backtest")"
BACKTEST_RESP="${backtest_resp}" python - <<'PY'
import json
import os

payload = json.loads(os.environ["BACKTEST_RESP"])
assert "rmse" in payload["metrics"], payload
PY

job_resp="$(curl --fail --show-error --silent "${auth_header[@]}" "${api_headers[@]}" -d "${job_payload}" "${DEPLOY_BASE_URL}/v1/jobs")"
job_id="$(JOB_RESP="${job_resp}" python - <<'PY'
import json
import os

payload = json.loads(os.environ["JOB_RESP"])
print(payload["job_id"])
PY
)"

job_status_resp="$(curl --fail --show-error --silent "${auth_header[@]}" -H "X-API-Key: ${DEPLOY_API_KEY}" -H "X-Tenant-Id: ${DEPLOY_TENANT_ID}" "${DEPLOY_BASE_URL}/v1/jobs/${job_id}")"
JOB_STATUS_RESP="${job_status_resp}" python - <<'PY'
import json
import os

payload = json.loads(os.environ["JOB_STATUS_RESP"])
assert payload["status"] in {"queued", "running", "succeeded", "failed"}, payload
PY