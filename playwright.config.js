const { defineConfig } = require("playwright/test");

module.exports = defineConfig({
  testDir: "tests/e2e",
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:8000",
    browserName: "chromium",
    headless: true,
  },
  webServer: {
    command:
      "mkdir -p .tmp && rm -f .tmp/e2e-drift-baseline.json && PATH=.venv/bin:$PATH PYTHONPATH=src RULFM_FORECASTING_API_KEY=e2e-test-key RULFM_DRIFT_BASELINE_PATH=.tmp/e2e-drift-baseline.json python -m uvicorn forecasting_api.app:create_app --factory --host 127.0.0.1 --port 8000",
    url: "http://127.0.0.1:8000/health",
    reuseExistingServer: false,
    timeout: 120_000,
  },
});