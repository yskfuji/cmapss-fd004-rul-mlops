const { test, expect } = require("playwright/test");

const sampleRecords = [
  {
    series_id: "engine-1",
    timestamp: "2026-03-01T00:00:00Z",
    y: 98,
    x: { sensor_1: 0.95, sensor_2: 12.1, sensor_3: 98.4 },
  },
  {
    series_id: "engine-1",
    timestamp: "2026-03-02T00:00:00Z",
    y: 97,
    x: { sensor_1: 0.98, sensor_2: 12.3, sensor_3: 99.1 },
  },
  {
    series_id: "engine-1",
    timestamp: "2026-03-03T00:00:00Z",
    y: 96,
    x: { sensor_1: 1.01, sensor_2: 12.7, sensor_3: 100.7 },
  },
  {
    series_id: "engine-1",
    timestamp: "2026-03-04T00:00:00Z",
    y: 95,
    x: { sensor_1: 1.08, sensor_2: 13.8, sensor_3: 102.2 },
  },
];

test("drift baseline flow blocks run until baseline is saved", async ({ page }) => {
  await page.goto("/ui/forecasting/");

  await page.waitForFunction(() => window.__ARCAYF_FORECASTING_GUI__?.ready === true);

  await page.evaluate(async (records) => {
    await window.__ARCAYF_FORECASTING_GUI__.setApiKey("e2e-test-key");
    await window.__ARCAYF_FORECASTING_GUI__.loadJsonRecords(records);
  }, sampleRecords);

  const blocked = await page.evaluate(async () => {
    return window.__ARCAYF_FORECASTING_GUI__.prepareDrift();
  });

  expect(blocked.task).toBe("drift");
  expect(blocked.runButtonDisabled).toBe(true);
  const blockedKeys = blocked.reasonKeys;
  expect(blockedKeys).toContain("run.block.drift_baseline");

  const saved = await page.evaluate(async () => {
    return window.__ARCAYF_FORECASTING_GUI__.saveCurrentAsDriftBaseline();
  });

  // saveCurrentAsDriftBaseline resets persisted state on each invocation before saving.
  expect(saved.persisted).toBe(true);
  expect(saved.runButtonDisabled).toBe(false);
  expect(saved.reasonKeys).not.toContain("run.block.drift_baseline");
  expect(saved.driftBaselineStatus.length).toBeGreaterThan(0);

  await expect(page.locator("#runForecast")).toBeEnabled();
});