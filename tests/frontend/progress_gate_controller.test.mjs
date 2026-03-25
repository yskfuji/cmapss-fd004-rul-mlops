import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

import { createProgressGateController } from "../../src/forecasting_api/static/forecasting_gui/progress_gate_controller.js";

function createFixture({
  task = "forecast",
  connectionReady = true,
  dataReady = true,
  stepCompletion = { 1: true, 2: false, 3: false, 4: false },
  gapCount = 0,
  missing = { series_id: 0, timestamp: 0, y: 0 },
  blocked = false,
  reasons = [],
  runGateExplained = false,
  lastHealthOk = true,
  lastDataStats = { seriesCount: 4 },
  runState = { status: "idle" },
} = {}) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="checkDataInput"></div>
    <div id="checkDataRequired"></div>
    <div id="checkDataStatus"></div>
    <div id="checkDataValidate"></div>
    <div id="checkParamsHorizon"></div>
    <div id="checkParamsQuantiles"></div>
    <div id="checkParamsStatus"></div>
    <input id="horizon" value="6" />
    <input id="level" value="" />
    <select id="missingPolicy"><option value="error">error</option></select>
    <a id="nextStepLink"></a>
    <input id="quantiles" value="0.1,0.9" />
    <section id="resultsInterpretation" hidden></section>
    <section id="resultsSummary" hidden></section>
    <button id="runForecastBtn"></button>
    <div id="runStatus"></div>
    <div id="statusBillingEstimate"></div>
    <div id="statusBillingLimit"></div>
    <div id="statusConnectionApiKey"></div>
    <div id="statusConnectionHealth"></div>
    <div id="statusDataInput"></div>
    <div id="statusDataValidation"></div>
    <div id="statusParamsHorizon"></div>
    <div id="statusParamsQuantiles"></div>
  </body></html>`);
  const { document } = dom.window;
  const statusCalls = [];
  const translations = {
    "status.connection.api_key": "API key",
    "status.connection.health": "Health",
    "status.data.input": "Input",
    "status.data.validation": "Validation",
    "status.params.horizon": "Horizon",
    "status.params.quantiles": "Quantiles",
    "status.billing.limit": "Limit",
    "status.billing.estimate": "Estimate",
    "status.state.done": "Done",
    "status.state.todo": "Todo",
    "status.state.wait": "Wait",
    "status.state.check": "Check",
    "status.state.optional": "Optional",
    "status.state.reference": "Reference",
    "checklist.data.input": "Select data",
    "checklist.data.required": "Required fields",
    "checklist.data.validate": "Validate data",
    "checklist.params.horizon": "Set horizon",
    "checklist.params.quantiles": "Set uncertainty",
    "checklist.summary": "{done}/{total}",
    "next.step.ready": "Ready",
    "next.step.connection": "Connection",
    "next.step.data": "Data",
    "next.step.params": "Params",
    "next.step.results": "Results",
    "next.step.label": "Next: {step}",
    "status.run_blocked_preconditions": "Blocked",
  };
  const t = (key, vars = null) => {
    const raw = translations[key] || key;
    if (!vars) return raw;
    return raw.replaceAll(/\{([a-zA-Z0-9_]+)\}/g, (_m, name) => String(vars[name] ?? ""));
  };

  const controller = createProgressGateController({
    elements: {
      checkDataInputEl: document.getElementById("checkDataInput"),
      checkDataRequiredEl: document.getElementById("checkDataRequired"),
      checkDataStatusEl: document.getElementById("checkDataStatus"),
      checkDataValidateEl: document.getElementById("checkDataValidate"),
      checkParamsHorizonEl: document.getElementById("checkParamsHorizon"),
      checkParamsQuantilesEl: document.getElementById("checkParamsQuantiles"),
      checkParamsStatusEl: document.getElementById("checkParamsStatus"),
      horizonEl: document.getElementById("horizon"),
      levelEl: document.getElementById("level"),
      missingPolicyEl: document.getElementById("missingPolicy"),
      nextStepLinkEl: document.getElementById("nextStepLink"),
      quantilesEl: document.getElementById("quantiles"),
      resultsInterpretationEl: document.getElementById("resultsInterpretation"),
      resultsSummaryEl: document.getElementById("resultsSummary"),
      runForecastBtnEl: document.getElementById("runForecastBtn"),
      runStatusEl: document.getElementById("runStatus"),
      statusBillingEstimateEl: document.getElementById("statusBillingEstimate"),
      statusBillingLimitEl: document.getElementById("statusBillingLimit"),
      statusConnectionApiKeyEl: document.getElementById("statusConnectionApiKey"),
      statusConnectionHealthEl: document.getElementById("statusConnectionHealth"),
      statusDataInputEl: document.getElementById("statusDataInput"),
      statusDataValidationEl: document.getElementById("statusDataValidation"),
      statusParamsHorizonEl: document.getElementById("statusParamsHorizon"),
      statusParamsQuantilesEl: document.getElementById("statusParamsQuantiles"),
    },
    computeRunGateState: () => ({ blocked }),
    currentTask: () => task,
    currentDataSource: () => "json",
    getCurrentDriftBaselineState: () => ({ checked: true, exists: true }),
    getLastDataGapCount: () => gapCount,
    getLastDataMissing: () => missing,
    getLastDataStats: () => lastDataStats,
    getLastHealthOk: () => lastHealthOk,
    getRunBlockReasonKeysFromState: () => reasons.map((_, index) => `reason-${index}`),
    getRunBlockReasons: () => reasons,
    getRunGateExplained: () => runGateExplained,
    getRunState: () => runState,
    getStepCompletion: () => stepCompletion,
    hasDataForSource: () => true,
    isConnectionReady: () => connectionReady,
    isDataReady: () => dataReady,
    isDriftBaselineBusy: () => false,
    onRefreshDriftBaselineStatus: () => {},
    setAriaDisabled: (node, disabled) => {
      node.disabled = !!disabled;
      node.setAttribute("aria-disabled", disabled ? "true" : "false");
    },
    setSectionStatus: (node, labelKey, stateKey) => {
      node.textContent = `${t(labelKey)}: ${t(`status.state.${stateKey}`)}`;
    },
    setStatus: (node, text, tone = null) => {
      node.textContent = text;
      if (tone) statusCalls.push(tone);
    },
    t,
    updateResultsStatus: () => {},
  });

  return { controller, document, statusCalls };
}

test("updateSectionStatus fixes section labels and checklist summaries", () => {
  const { controller, document } = createFixture({
    gapCount: 2,
    missing: { series_id: 1, timestamp: 0, y: 0 },
  });

  controller.updateSectionStatus();

  assert.equal(document.getElementById("statusConnectionApiKey").textContent, "API key: Done");
  assert.equal(document.getElementById("statusDataValidation").textContent, "Validation: Check");
  assert.equal(document.getElementById("checkDataInput").dataset.state, "done");
  assert.equal(document.getElementById("checkDataRequired").dataset.state, "check");
  assert.equal(document.getElementById("checkDataStatus").textContent, "1/3");
  assert.equal(document.getElementById("statusBillingEstimate").textContent, "Estimate: Done");
});

test("updateNextStep fixes link target and ready wording", () => {
  const { controller, document } = createFixture({
    stepCompletion: { 1: true, 2: false, 3: false, 4: false },
  });
  controller.updateNextStep();
  assert.equal(document.getElementById("nextStepLink").textContent, "Next: Data");
  assert.equal(document.getElementById("nextStepLink").getAttribute("href"), "#dataCard");

  const readyFixture = createFixture({ stepCompletion: { 1: true, 2: true, 3: true, 4: true } });
  readyFixture.document.getElementById("resultsSummary").hidden = false;
  readyFixture.controller.updateNextStep();
  assert.equal(readyFixture.document.getElementById("nextStepLink").textContent, "Ready");
  assert.equal(readyFixture.document.getElementById("nextStepLink").getAttribute("href"), "#resultsSummary");
});

test("updateRunGate fixes blocked status text when explained", () => {
  const { controller, document, statusCalls } = createFixture({
    blocked: true,
    reasons: ["missing validation", "quota exceeded"],
    runGateExplained: true,
  });

  controller.updateRunGate({ isUiBusy: false });

  assert.equal(document.getElementById("runForecastBtn").disabled, true);
  assert.match(document.getElementById("runStatus").textContent, /Blocked/);
  assert.match(document.getElementById("runStatus").textContent, /missing validation/);
  assert.deepEqual(statusCalls, ["warn"]);
});