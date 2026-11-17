const { test, expect } = require("playwright/test");

const sampleForecastPayload = {
  horizon: 1,
  frequency: "1d",
  data: [
    { series_id: "s1", timestamp: "2026-03-20T00:00:00Z", y: 10.0 },
    { series_id: "s1", timestamp: "2026-03-21T00:00:00Z", y: 11.0 },
  ],
};

const sampleBacktestPayload = {
  horizon: 2,
  folds: 2,
  metric: "rmse",
  data: [
    { series_id: "s1", timestamp: "2026-03-20T00:00:00Z", y: 10.0 },
    { series_id: "s1", timestamp: "2026-03-21T00:00:00Z", y: 11.0 },
    { series_id: "s1", timestamp: "2026-03-22T00:00:00Z", y: 12.0 },
    { series_id: "s1", timestamp: "2026-03-23T00:00:00Z", y: 13.0 },
    { series_id: "s1", timestamp: "2026-03-24T00:00:00Z", y: 14.0 },
    { series_id: "s1", timestamp: "2026-03-25T00:00:00Z", y: 15.0 },
    { series_id: "s1", timestamp: "2026-03-26T00:00:00Z", y: 16.0 },
    { series_id: "s1", timestamp: "2026-03-27T00:00:00Z", y: 17.0 },
  ],
};

test("public browser smoke covers docs, auth, forecast, backtest, and jobs", async ({ page }) => {
  await page.goto("/ui/forecasting/");
  await page.waitForFunction(() => window.__ARCAYF_FORECASTING_GUI__?.ready === true);

  const result = await page.evaluate(async ({ forecastPayload, backtestPayload }) => {
    const apiKey = "e2e-test-key";

    const fetchJson = async (path, init = {}) => {
      const response = await fetch(path, init);
      const text = await response.text();
      let json = null;
      try {
        json = text ? JSON.parse(text) : null;
      } catch {
        json = text;
      }
      return { status: response.status, body: json };
    };

    const docsResponse = await fetch("/docs/en");
    const metricsUnauthorized = await fetch("/metrics");
    const metricsAuthorized = await fetch("/metrics", {
      headers: { "x-api-key": apiKey },
    });
    const metricsText = await metricsAuthorized.text();

    const forecast = await fetchJson("/v1/forecast", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
      },
      body: JSON.stringify(forecastPayload),
    });

    const backtest = await fetchJson("/v1/backtest", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
      },
      body: JSON.stringify(backtestPayload),
    });

    const jobCreate = await fetchJson("/v1/jobs", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
      },
      body: JSON.stringify({ type: "forecast", payload: forecastPayload }),
    });

    const jobStatus = await fetchJson(`/v1/jobs/${jobCreate.body.job_id}`, {
      headers: { "x-api-key": apiKey },
    });

    const jobResult = await fetchJson(`/v1/jobs/${jobCreate.body.job_id}/result`, {
      headers: { "x-api-key": apiKey },
    });

    return {
      docsStatus: docsResponse.status,
      metricsUnauthorizedStatus: metricsUnauthorized.status,
      metricsAuthorizedStatus: metricsAuthorized.status,
      metricsText,
      forecast,
      backtest,
      jobCreate,
      jobStatus,
      jobResult,
    };
  }, { forecastPayload: sampleForecastPayload, backtestPayload: sampleBacktestPayload });

  expect(result.docsStatus).toBe(200);
  expect(result.metricsUnauthorizedStatus).toBe(401);
  expect(result.metricsAuthorizedStatus).toBe(200);
  expect(result.metricsText).toContain("rulfm_http_requests_total");

  expect(result.forecast.status).toBe(200);
  expect(result.forecast.body.forecasts).toHaveLength(1);
  expect(result.forecast.body.forecasts[0].series_id).toBe("s1");

  expect(result.backtest.status).toBe(200);
  expect(result.backtest.body.metrics).toHaveProperty("rmse");

  expect(result.jobCreate.status).toBe(202);
  expect(result.jobCreate.body.job_id).toBeTruthy();
  expect(["queued", "running", "succeeded", "failed"]).toContain(result.jobStatus.body.status);
  expect(result.jobResult.status).toBe(409);
  expect(result.jobResult.body.error_code).toBe("J02");
});