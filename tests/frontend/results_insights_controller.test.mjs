import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

import { createResultsInsightsController } from "../../src/forecasting_api/static/forecasting_gui/results_insights_controller.js";

function createFixture({ benchmarkCache = null, lastResult = null, lastRunContext = null, lastRecords = [] } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <section id="resultsSummary"></section>
    <div id="resultsSummaryChips"></div>
    <div id="resultsSummaryNote"></div>
    <section id="resultsEvidence"></section>
    <div id="resultsEvidenceQuality"></div>
    <div id="resultsEvidenceUncertainty"></div>
    <div id="resultsEvidenceScope"></div>
    <div id="resultsEvidenceLimit"></div>
    <section id="resultsBenchmark"></section>
    <div id="resultsBenchmarkSummary"></div>
    <div id="resultsBenchmarkNote"></div>
    <table><tbody id="resultsBenchmarkTableBody"></tbody></table>
  </body></html>`);
  const { document } = dom.window;
  let state = { cache: benchmarkCache, pending: null };

  const translations = {
    "results.summary.model": "Model",
    "results.summary.series": "Series",
    "results.summary.horizon": "Horizon",
    "results.summary.frequency": "Frequency",
    "results.summary.quantiles": "Quantiles",
    "results.summary.level": "Level",
    "results.summary.metric": "Metric",
    "results.summary.folds": "Folds",
    "results.summary.records": "Records",
    "results.summary.training_hours": "Hours",
    "results.summary.drift_severity": "Severity",
    "results.summary.drift_selected_bin_count": "Bins",
    "results.evidence.quality_forecast": "valid={valid}/{total}",
    "results.evidence.uncertainty_forecast": "covered={covered}/{total} coverage={coverage}",
    "results.evidence.scope_forecast": "series={series} horizon={horizon} id={series_id}",
    "results.evidence.limit_assumption": "assumption",
    "results.evidence.limit_short_series": "short {total}",
    "results.evidence.limit_low_uncertainty": "low {coverage}",
    "results.evidence.limit_no_points": "no-points",
    "results.evidence.quality_backtest": "metrics={metrics} series={by_series} horizon={by_horizon} fold={by_fold}",
    "results.evidence.uncertainty_backtest": "backtest-uncertainty",
    "results.evidence.scope_backtest": "folds={folds} horizon={horizon}",
    "results.evidence.limit_backtest": "backtest-limit",
    "results.evidence.quality_train": "train {model_id}",
    "results.evidence.uncertainty_train": "train-uncertainty",
    "results.evidence.scope_train": "records={records} hours={hours}",
    "results.evidence.limit_train": "train-limit",
    "results.evidence.quality_drift": "severity={severity} features={features} score={score}",
    "results.evidence.uncertainty_drift": "feature={feature} requested={requested} selected={selected}",
    "results.evidence.scope_drift": "records={records} series={series}",
    "results.evidence.limit_drift": "drift-limit",
    "results.evidence.exog_present": "exog {keys} {coverage}",
    "results.evidence.exog_currently_ignored": "ignored",
    "results.series.note_exogenous": "no-exog",
    "results.benchmark.current_run_label": "Current run",
    "results.benchmark.current_run_source": "Local",
    "results.benchmark.current_run_scope": "Current",
    "results.benchmark.summary_leader": "Leader {model} {rmse}",
    "results.benchmark.summary_empty": "No benchmark",
    "results.benchmark.note_snapshot": "Snapshot {generated_at}",
    "results.benchmark.current_run_note": "Includes current run",
    "results.benchmark.note_run_backtest": "Run backtest",
  };

  const t = (key, vars = null) => {
    const raw = translations[key] || key;
    if (!vars) return raw;
    return raw.replaceAll(/\{([a-zA-Z0-9_]+)\}/g, (_m, name) => String(vars[name] ?? ""));
  };

  const controller = createResultsInsightsController({
    apiClient: {
      getCmapssFd004Benchmarks: async () => ({ rows: [], notes: [] }),
    },
    elements: {
      resultsSummaryEl: document.getElementById("resultsSummary"),
      resultsSummaryChipsEl: document.getElementById("resultsSummaryChips"),
      resultsSummaryNoteEl: document.getElementById("resultsSummaryNote"),
      resultsEvidenceEl: document.getElementById("resultsEvidence"),
      resultsEvidenceQualityEl: document.getElementById("resultsEvidenceQuality"),
      resultsEvidenceUncertaintyEl: document.getElementById("resultsEvidenceUncertainty"),
      resultsEvidenceScopeEl: document.getElementById("resultsEvidenceScope"),
      resultsEvidenceLimitEl: document.getElementById("resultsEvidenceLimit"),
      resultsBenchmarkEl: document.getElementById("resultsBenchmark"),
      resultsBenchmarkSummaryEl: document.getElementById("resultsBenchmarkSummary"),
      resultsBenchmarkNoteEl: document.getElementById("resultsBenchmarkNote"),
      resultsBenchmarkTableBodyEl: document.getElementById("resultsBenchmarkTableBody"),
    },
    countValidSeriesPoints: (seriesPoints) => ({
      valid: seriesPoints.filter((point) => Number.isFinite(Date.parse(String(point?.timestamp ?? ""))) && Number.isFinite(Number(point?.point))).length,
      total: seriesPoints.length,
    }),
    getBenchmarkState: () => state,
    getLastRecords: () => lastRecords,
    getLastResult: () => lastResult,
    getLastRunContext: () => lastRunContext,
    getLastTask: () => "forecast",
    getLastForecastSeriesId: () => "unit-01",
    normalizeTask: (task) => task,
    setBenchmarkState: (next) => {
      state = next;
    },
    setVisible: (node, visible) => {
      node.hidden = !visible;
    },
    t,
  });

  return { controller, document };
}

test("renderSummary fixes forecast summary chips from run context", () => {
  const { controller, document } = createFixture({
    lastRunContext: {
      model_id: "ridge-v1",
      series_count: 4,
      horizon: 6,
      frequency: "daily",
      quantiles: [0.1, 0.9],
      level: [95],
    },
  });

  controller.renderSummary("forecast");

  const chips = [...document.querySelectorAll(".chip")].map((chip) => chip.textContent);
  assert.deepEqual(chips, [
    "Modelridge-v1",
    "Series4",
    "Horizon6",
    "Frequencydaily",
    "Quantiles0.1, 0.9",
    "Level95",
  ]);
  assert.equal(document.getElementById("resultsSummary").hidden, false);
});

test("renderEvidence fixes forecast evidence wording including exogenous note", () => {
  const seriesPoints = [
    { timestamp: "2026-01-01", point: 10, quantiles: { 0.1: 9, 0.9: 12 } },
    { timestamp: "2026-01-02", point: 12, quantiles: { 0.1: 10, 0.9: 13 } },
  ];
  const { controller, document } = createFixture({
    lastResult: { calibration: null },
    lastRunContext: {
      series_count: 2,
      horizon: 2,
      algo_id: "ridge_lags_v1",
      record_count: 20,
      model_id: "ridge-v1",
    },
    lastRecords: [
      { x: { temp: 10, pressure: 20 } },
      { x: { temp: 11 } },
    ],
  });

  controller.renderEvidence("forecast", { seriesPoints });

  assert.equal(document.getElementById("resultsEvidenceQuality").textContent, "valid=2/2\ndetail: records=20, series=2, model=ridge-v1");
  assert.equal(document.getElementById("resultsEvidenceUncertainty").textContent, "covered=2/2 coverage=100");
  assert.match(document.getElementById("resultsEvidenceScope").textContent, /series=2 horizon=2 id=unit-01/);
  assert.match(document.getElementById("resultsEvidenceScope").textContent, /exog pressure, temp 100/);
  assert.match(document.getElementById("resultsEvidenceScope").textContent, /ignored/);
  assert.equal(document.getElementById("resultsEvidenceLimit").textContent, "short 2");
});

test("renderBenchmark fixes benchmark summary and rows with current run first", () => {
  const { controller, document } = createFixture({
    benchmarkCache: {
      generated_at: "2026-03-25T12:00:00Z",
      rows: [
        { model: "baseline", rmse: 14.2, mae: 10.1, source: "docs", scope: "fd004" },
        { model: "best-known", rmse: 11.3, mae: 8.4, source: "docs", scope: "fd004" },
      ],
    },
  });

  controller.renderBenchmark("backtest", { result: { metrics: { rmse: 9.8, mae: 7.2 } } });

  const rows = [...document.querySelectorAll("#resultsBenchmarkTableBody tr")].map((tr) => [...tr.querySelectorAll("td")].map((td) => td.textContent));
  assert.deepEqual(rows[0], ["1", "Current run", "9.800", "7.200", "Local", "Current"]);
  assert.equal(document.getElementById("resultsBenchmarkSummary").textContent, "Leader Current run 9.800");
  assert.equal(document.getElementById("resultsBenchmarkNote").textContent, "Snapshot 2026-03-25T12:00:00Z\nIncludes current run");
  assert.equal(document.getElementById("resultsBenchmark").hidden, false);
});