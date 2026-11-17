const { test, expect } = require("playwright/test");

const forecastJobPayload = {
  horizon: 1,
  frequency: "1d",
  data: [
    { series_id: "s1", timestamp: "2026-03-20T00:00:00Z", y: 10.0 },
    { series_id: "s1", timestamp: "2026-03-21T00:00:00Z", y: 11.0 },
  ],
};

test("jobs screen flow adds, refreshes, and removes tracked jobs", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });

  await page.goto("/ui/forecasting/");
  await page.waitForFunction(() => window.__ARCAYF_FORECASTING_GUI__?.ready === true);

  await page.selectOption("#langSelect", "en");
  await page.selectOption("#densityMode", "detailed");
  await page.fill("#apiKey", "e2e-test-key");

  const jobId = await page.evaluate(async (payload) => {
    const response = await fetch("/v1/jobs", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": "e2e-test-key",
      },
      body: JSON.stringify({ type: "forecast", payload }),
    });
    const body = await response.json();
    if (response.status !== 202 || !body?.job_id) {
      throw new Error(`job seed failed: ${response.status}`);
    }
    return body.job_id;
  }, forecastJobPayload);

  await page.fill("#jobIdInput", jobId);
  await page.click("#addJobId");

  await expect(page.locator("#jobIdInput")).toHaveValue("");

  const row = page.locator("#jobsTable tbody tr", { hasText: jobId });
  await expect(row).toHaveCount(1);
  await expect(row.getByRole("button", { name: "Status" })).toBeVisible();
  await expect(row.getByRole("button", { name: "Result" })).toBeVisible();
  await expect(row.getByRole("button", { name: "Remove" })).toBeVisible();

  await page.click("#refreshJobs");
  await expect(page.locator("#jobsStatus")).toContainText("Refreshed");

  await row.getByRole("button", { name: "Status" }).click();
  await expect(row.locator("td").nth(1)).toContainText(/queued|running|succeeded|failed/i);

  await row.getByRole("button", { name: "Remove" }).click();
  await expect(page.locator("#jobsTable tbody tr", { hasText: jobId })).toHaveCount(0);
});