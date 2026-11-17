const { test, expect } = require("playwright/test");

const forecastPayload = Array.from({ length: 30 }, (_, index) => ({
  series_id: "s1",
  timestamp: `2026-03-${String(index + 1).padStart(2, "0")}T00:00:00Z`,
  y: 10 + index,
}));

test("forecast screen flow validates sample input and renders downloadable results", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });

  await page.goto("/ui/forecasting/");
  await page.waitForFunction(() => window.__ARCAYF_FORECASTING_GUI__?.ready === true);

  await page.selectOption("#langSelect", "en");
  await page.selectOption("#densityMode", "detailed");
  await page.fill("#apiKey", "e2e-test-key");
  await page.click("#dataSourceBtnJson");
  await page.fill("#jsonInput", JSON.stringify(forecastPayload, null, 2));
  await page.fill("#horizon", "1");

  await expect(page.locator("#jsonInput")).not.toHaveValue("");

  await page.click("#validate");
  await expect(page.locator("#dataStatus")).not.toHaveText(/^\s*$/);
  await expect(page.locator("#runForecast")).toBeEnabled();

  await page.click("#runForecast");

  const resultRows = page.locator("#resultTable tbody tr");
  await expect(page.locator("#forecastTableWrap")).toBeVisible();
  await expect(resultRows).toHaveCount(1);
  await expect(resultRows.first().locator("td").first()).toHaveText("s1");
  await expect(page.locator("#downloadJson")).toBeEnabled();
});