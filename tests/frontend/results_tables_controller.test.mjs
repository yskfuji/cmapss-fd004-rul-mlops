import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

import { createResultsTablesController } from "../../src/forecasting_api/static/forecasting_gui/results_tables_controller.js";

function createFixture() {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="metricsPrimary"></div>
    <table><tbody id="metricsTableBody"><tr><td>stale</td></tr></tbody></table>
    <table><tbody id="bySeriesTableBody"><tr><td>stale</td></tr></tbody></table>
    <table><tbody id="byHorizonTableBody"><tr><td>stale</td></tr></tbody></table>
    <table><tbody id="byFoldTableBody"><tr><td>stale</td></tr></tbody></table>
    <table><tbody id="driftFeaturesTableBody"><tr><td>stale</td></tr></tbody></table>
    <table><tbody id="resultTableBody"><tr><td>stale</td></tr></tbody></table>
  </body></html>`);
  const { document } = dom.window;
  const translations = {
    "backtest.metrics_primary": "Primary {metric}",
    "backtest.metrics_primary_none": "Primary none",
  };
  const t = (key, vars = null) => {
    const raw = translations[key] || key;
    if (!vars) return raw;
    return raw.replaceAll(/\{([a-zA-Z0-9_]+)\}/g, (_m, name) => String(vars[name] ?? ""));
  };

  const controller = createResultsTablesController({
    elements: {
      metricsPrimaryEl: document.getElementById("metricsPrimary"),
      metricsTableBodyEl: document.getElementById("metricsTableBody"),
      bySeriesTableBodyEl: document.getElementById("bySeriesTableBody"),
      byHorizonTableBodyEl: document.getElementById("byHorizonTableBody"),
      byFoldTableBodyEl: document.getElementById("byFoldTableBody"),
      driftFeaturesTableBodyEl: document.getElementById("driftFeaturesTableBody"),
      resultTableBodyEl: document.getElementById("resultTableBody"),
    },
    setText: (node, text) => {
      node.textContent = String(text ?? "");
    },
    t,
  });

  return { controller, document };
}

function rowTexts(rows) {
  return [...rows].map((row) => [...row.querySelectorAll("td")].map((cell) => cell.textContent));
}

test("clear removes existing table rows and metrics summary", () => {
  const { controller, document } = createFixture();
  document.getElementById("metricsPrimary").textContent = "Primary RMSE";

  controller.clear();

  assert.equal(document.getElementById("metricsPrimary").textContent, "");
  assert.equal(document.getElementById("metricsTableBody").children.length, 0);
  assert.equal(document.getElementById("bySeriesTableBody").children.length, 0);
  assert.equal(document.getElementById("byHorizonTableBody").children.length, 0);
  assert.equal(document.getElementById("byFoldTableBody").children.length, 0);
  assert.equal(document.getElementById("driftFeaturesTableBody").children.length, 0);
  assert.equal(document.getElementById("resultTableBody").children.length, 0);
});

test("renderBacktestTables fixes highlight rows and summary text", () => {
  const { controller, document } = createFixture();
  controller.clear();

  controller.renderBacktestTables({
    highlightKey: "RMSE",
    metricTableRows: [
      { metric: "RMSE", value: "9.800", isHighlight: true },
      { metric: "MAE", value: "7.200", isHighlight: false },
    ],
    seriesTableRows: [{ rank: "1", seriesId: "unit-01", value: "8.500" }],
    horizonTableRows: [{ horizon: "6", value: "9.100" }],
    foldTableRows: [{ fold: "2", value: "10.400" }],
  });

  assert.equal(document.getElementById("metricsPrimary").textContent, "Primary RMSE");
  const metricRows = [...document.querySelectorAll("#metricsTableBody tr")];
  assert.equal(metricRows.length, 2);
  assert.equal(metricRows[0].classList.contains("metricRow"), true);
  assert.equal(metricRows[0].classList.contains("metricHighlight"), true);
  assert.deepEqual(rowTexts(metricRows), [["RMSE", "9.800"], ["MAE", "7.200"]]);
  assert.deepEqual(rowTexts(document.querySelectorAll("#bySeriesTableBody tr")), [["1", "unit-01", "8.500"]]);
  assert.deepEqual(rowTexts(document.querySelectorAll("#byHorizonTableBody tr")), [["6", "9.100"]]);
  assert.deepEqual(rowTexts(document.querySelectorAll("#byFoldTableBody tr")), [["2", "10.400"]]);
});

test("renderForecastTable and renderDriftFeatures fix cell ordering", () => {
  const { controller, document } = createFixture();
  controller.clear();

  controller.renderForecastTable([
    {
      seriesId: "unit-04",
      timestamp: "2026-04-01T00:00:00Z",
      point: "123.4",
      quantiles: "q10=120,q90=128",
      intervals: "95%",
    },
  ]);
  controller.renderDriftFeatures([
    {
      feature: "sensor_7",
      population_stability_index: 0.12345,
      baseline_requested_bin_count: 10,
      baseline_selected_bin_count: 8,
    },
  ]);

  assert.deepEqual(rowTexts(document.querySelectorAll("#resultTableBody tr")), [
    ["unit-04", "2026-04-01T00:00:00Z", "123.4", "q10=120,q90=128", "95%"],
  ]);
  assert.deepEqual(rowTexts(document.querySelectorAll("#driftFeaturesTableBody tr")), [["sensor_7", "0.123", "10", "8"]]);
});