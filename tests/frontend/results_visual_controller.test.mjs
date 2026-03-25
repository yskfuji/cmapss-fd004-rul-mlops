import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

import { createResultsVisualController } from "../../src/forecasting_api/static/forecasting_gui/results_visual_controller.js";

function setupDom() {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="resultsVisual"></div>
    <div id="resultsEmpty"></div>
    <div id="resultsVisualNote"></div>
    <svg id="resultsSparkline"></svg>
    <svg id="resultsSparklineSecondary"></svg>
    <svg id="resultsSparklineTertiary"></svg>
    <div id="resultsSecondaryChartBlock"></div>
    <div id="resultsTertiaryChartBlock"></div>
    <div id="resultsPrimaryChartTitle"></div>
    <div id="resultsSecondaryChartTitle"></div>
    <div id="resultsTertiaryChartTitle"></div>
    <div id="resultsInterpretation"></div>
    <div id="resultsInterpretIntent"></div>
    <div id="resultsInterpretWhen"></div>
    <div id="resultsInterpretCaution"></div>
    <div id="resultsSummaryNote"></div>
    <div id="resultsHighlights"></div>
    <div id="resultsHighlightSeries"></div>
    <div id="resultsHighlightPoints"></div>
    <div id="resultsHighlightBand"></div>
    <select id="resultsSeries"></select>
    <div id="resultsAxisValue"></div>
    <div id="resultsAxisTime"></div>
    <div id="resultsAxisYMax"></div>
    <div id="resultsAxisYMin"></div>
    <div id="resultsAxisXMin"></div>
    <div id="resultsAxisXMax"></div>
    <div id="resultsAxisValue2"></div>
    <div id="resultsAxisTime2"></div>
    <div id="resultsAxisYMax2"></div>
    <div id="resultsAxisYMin2"></div>
    <div id="resultsAxisXMin2"></div>
    <div id="resultsAxisXMax2"></div>
    <div id="resultsAxisValue3"></div>
    <div id="resultsAxisTime3"></div>
    <div id="resultsAxisYMax3"></div>
    <div id="resultsAxisYMin3"></div>
    <div id="resultsAxisXMin3"></div>
    <div id="resultsAxisXMax3"></div>
  </body></html>`);
  const { document } = dom.window;
  document.getElementById("resultsVisual").scrollIntoView = () => {};
  return { dom, document };
}

function createControllerFixture({ chartType = "trend", lastRunContext = null, lastRecords = [] } = {}) {
  const { document } = setupDom();
  const evidenceCalls = [];
  const visibility = new Map();
  const translations = {
    "results.axis.value": "Value",
    "results.axis.value_band": "Band width",
    "results.axis.time": "Time",
    "results.axis.count": "Count",
    "results.axis.abs_error": "Absolute error",
    "results.chart.trend": "Trend",
    "results.chart.band": "Band",
    "results.chart.points": "Points",
    "results.chart.residuals": "Residuals",
    "results.chart.point_label": "h={h}",
    "results.chart.band_label_none": "no band",
    "results.chart.band_label_quantiles": "q{low}-q{high}",
    "results.chart.band_label_level": "level {level}",
    "results.chart.title_with_context": "{chart} / {series_id} / {point_label} / {band_label}",
    "results.chart.residuals_point_label": "n={n}",
    "results.visual.note": "Visual note",
    "results.visual.preview_note": "Preview {series_id}",
    "results.interpret.trend.intent": "Intent",
    "results.interpret.trend.when": "When",
    "results.interpret.trend.caution": "Caution",
    "results.highlight.series": "Series {series_id} horizon {horizon}",
    "results.highlight.points": "n={n}, min={min}, max={max}, std={std}",
    "results.highlight.band": "band {min}..{max}",
    "results.summary.integrity_ok": "OK {series_id} {total}",
    "results.summary.integrity_mismatch": "Mismatch {series_id} {valid}/{total}",
  };

  const t = (key, vars = null) => {
    const raw = translations[key] || key;
    if (!vars) return raw;
    return raw.replaceAll(/\{([a-zA-Z0-9_]+)\}/g, (_m, name) => String(vars[name] ?? ""));
  };

  const controller = createResultsVisualController({
    elements: {
      resultsVisualEl: document.getElementById("resultsVisual"),
      resultsEmptyEl: document.getElementById("resultsEmpty"),
      resultsVisualNoteEl: document.getElementById("resultsVisualNote"),
      resultsSparklineEl: document.getElementById("resultsSparkline"),
      resultsSparklineSecondaryEl: document.getElementById("resultsSparklineSecondary"),
      resultsSparklineTertiaryEl: document.getElementById("resultsSparklineTertiary"),
      resultsSecondaryChartBlockEl: document.getElementById("resultsSecondaryChartBlock"),
      resultsTertiaryChartBlockEl: document.getElementById("resultsTertiaryChartBlock"),
      resultsPrimaryChartTitleEl: document.getElementById("resultsPrimaryChartTitle"),
      resultsSecondaryChartTitleEl: document.getElementById("resultsSecondaryChartTitle"),
      resultsTertiaryChartTitleEl: document.getElementById("resultsTertiaryChartTitle"),
      resultsInterpretationEl: document.getElementById("resultsInterpretation"),
      resultsInterpretIntentEl: document.getElementById("resultsInterpretIntent"),
      resultsInterpretWhenEl: document.getElementById("resultsInterpretWhen"),
      resultsInterpretCautionEl: document.getElementById("resultsInterpretCaution"),
      resultsSummaryNoteEl: document.getElementById("resultsSummaryNote"),
      resultsHighlightsEl: document.getElementById("resultsHighlights"),
      resultsHighlightSeriesEl: document.getElementById("resultsHighlightSeries"),
      resultsHighlightPointsEl: document.getElementById("resultsHighlightPoints"),
      resultsHighlightBandEl: document.getElementById("resultsHighlightBand"),
      resultsSeriesEl: document.getElementById("resultsSeries"),
      resultsAxisValueEl: document.getElementById("resultsAxisValue"),
      resultsAxisTimeEl: document.getElementById("resultsAxisTime"),
      resultsAxisYMaxEl: document.getElementById("resultsAxisYMax"),
      resultsAxisYMinEl: document.getElementById("resultsAxisYMin"),
      resultsAxisXMinEl: document.getElementById("resultsAxisXMin"),
      resultsAxisXMaxEl: document.getElementById("resultsAxisXMax"),
      resultsAxisValue2El: document.getElementById("resultsAxisValue2"),
      resultsAxisTime2El: document.getElementById("resultsAxisTime2"),
      resultsAxisYMax2El: document.getElementById("resultsAxisYMax2"),
      resultsAxisYMin2El: document.getElementById("resultsAxisYMin2"),
      resultsAxisXMin2El: document.getElementById("resultsAxisXMin2"),
      resultsAxisXMax2El: document.getElementById("resultsAxisXMax2"),
      resultsAxisValue3El: document.getElementById("resultsAxisValue3"),
      resultsAxisTime3El: document.getElementById("resultsAxisTime3"),
      resultsAxisYMax3El: document.getElementById("resultsAxisYMax3"),
      resultsAxisYMin3El: document.getElementById("resultsAxisYMin3"),
      resultsAxisXMin3El: document.getElementById("resultsAxisXMin3"),
      resultsAxisXMax3El: document.getElementById("resultsAxisXMax3"),
    },
    getChartType: () => chartType,
    getFrequencyValue: () => "daily",
    getHorizonValue: () => 3,
    getLastRecords: () => lastRecords,
    getLastRunContext: () => lastRunContext,
    getValueUnit: () => "cycles",
    countValidSeriesPoints: (seriesPoints) => ({
      valid: seriesPoints.filter((point) => Number.isFinite(Date.parse(String(point?.timestamp ?? ""))) && Number.isFinite(Number(point?.point))).length,
      total: seriesPoints.length,
    }),
    renderResultsEvidence: (...args) => evidenceCalls.push(args),
    setVisible: (node, visible) => {
      node.hidden = !visible;
      visibility.set(node.id, visible);
    },
    t,
  });

  return { controller, document, evidenceCalls, visibility };
}

test("syncForecastSeriesOptions keeps selected series when present", () => {
  const { controller, document } = createControllerFixture();
  const selected = controller.syncForecastSeriesOptions(
    [
      { series_id: "unit-02" },
      { series_id: "unit-01" },
      { series_id: "unit-02" },
    ],
    "unit-02",
  );

  assert.equal(selected, "unit-02");
  const options = [...document.getElementById("resultsSeries").querySelectorAll("option")].map((option) => option.value);
  assert.deepEqual(options, ["unit-01", "unit-02"]);
  assert.equal(document.getElementById("resultsSeries").value, "unit-02");
  assert.equal(document.getElementById("resultsSeries").disabled, false);
});

test("renderSelectedForecastSeries updates charts, interpretation, highlights, and evidence", () => {
  const { controller, document, evidenceCalls, visibility } = createControllerFixture({
    lastRunContext: {
      horizon: 3,
      quantiles: [0.1, 0.9],
      frequency: "daily",
    },
  });
  const forecasts = [
    { series_id: "unit-01", timestamp: "2026-01-01", point: 10, quantiles: { 0.1: 8, 0.9: 13 } },
    { series_id: "unit-01", timestamp: "2026-01-02", point: 12, quantiles: { 0.1: 9, 0.9: 15 } },
    { series_id: "unit-01", timestamp: "2026-01-03", point: 11, quantiles: { 0.1: 9, 0.9: 14 } },
  ];

  controller.renderSelectedForecastSeries({
    forecasts,
    selectedSeriesId: "unit-01",
    residualEvidence: null,
  });

  assert.match(document.getElementById("resultsPrimaryChartTitle").textContent, /Trend \/ unit-01/);
  assert.notEqual(document.getElementById("resultsSparkline").innerHTML, "");
  assert.equal(document.getElementById("resultsInterpretIntent").textContent, "Intent");
  assert.match(document.getElementById("resultsHighlightSeries").textContent, /unit-01/);
  assert.match(document.getElementById("resultsHighlightBand").textContent, /band 8\.\.15/);
  assert.equal(document.getElementById("resultsSummaryNote").textContent, "OK unit-01 3");
  assert.equal(evidenceCalls.length, 1);
  assert.equal(evidenceCalls[0][0], "forecast");
  assert.equal(visibility.get("resultsInterpretation"), true);
  assert.equal(visibility.get("resultsTertiaryChartBlock"), true);
});

test("renderInputPreviewChart shows preview note and makes the visual section visible", () => {
  const { controller, document, visibility } = createControllerFixture({
    lastRecords: [
      { y: 1 },
      { y: 2 },
    ],
  });
  const records = [
    { series_id: "unit-03", timestamp: "2026-02-01", y: 21 },
    { series_id: "unit-03", timestamp: "2026-02-02", y: 22 },
    { series_id: "unit-03", timestamp: "2026-02-03", y: 25 },
  ];

  controller.renderInputPreviewChart(records);

  assert.equal(visibility.get("resultsVisual"), true);
  assert.equal(visibility.get("resultsEmpty"), false);
  assert.equal(document.getElementById("resultsVisualNote").textContent, "Preview unit-03");
  assert.notEqual(document.getElementById("resultsSparkline").innerHTML, "");
});