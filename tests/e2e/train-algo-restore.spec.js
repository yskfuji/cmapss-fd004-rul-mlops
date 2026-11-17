const { test, expect } = require("playwright/test");

test("train algo restores from display label without exposing internal id", async ({ page }) => {
  await page.goto("/ui/forecasting/");

  await page.waitForFunction(() => window.__ARCAYF_FORECASTING_GUI__?.ready === true);
  await page.selectOption("#langSelect", "en");
  const algoDisplay = "Advanced neural v1 (Torch)";

  await expect(page.locator('#trainAlgo option[value="afnocg2"]')).toHaveText(algoDisplay);

  expect(algoDisplay).toBeTruthy();
  expect(algoDisplay).toContain("Torch");
  expect(algoDisplay).not.toMatch(/afnocg|adast|stardast/i);

  const prepared = await page.evaluate(async (display) => {
    return window.__ARCAYF_FORECASTING_GUI__.prepareTrain({
      algoDisplay: display,
      trainingHours: 0.5,
    });
  }, algoDisplay);

  expect(prepared.task).toBe("train");

  await expect(page.locator("#task")).toHaveValue("train");
  await expect(page.locator("#trainAlgo")).toHaveValue("afnocg2");
  await expect(page.locator("#trainAlgo option:checked")).toHaveText(algoDisplay);
  await expect(page.locator("#trainingHours")).toHaveValue("0.5");
});