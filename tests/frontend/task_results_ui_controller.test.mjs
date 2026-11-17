import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

import { createTaskResultsUiController } from "../../src/forecasting_api/static/forecasting_gui/task_results_ui_controller.js";

function createFixture({ task = "forecast", lastResult = null, lastForecasts = [] } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <button id="addJobIdBtn"></button>
    <div id="backtestParams"></div>
    <input id="baseModel" />
    <button id="cancelRunBtn"></button>
    <button id="checkHealthBtn"></button>
    <button id="clearBtn"></button>
    <button id="copyLinkBtn"></button>
    <button id="copySnippetBtn"></button>
    <input id="csvFile" />
    <select id="densityMode"><option value="basic">basic</option></select>
    <button id="downloadCsv"></button>
    <button id="downloadJson"></button>
    <div id="driftBaselinePanel"></div>
    <input id="folds" />
    <div id="forecastParams"></div>
    <input id="frequency" />
    <input id="horizon" />
    <div id="horizonField"></div>
    <input id="jobIdInput" />
    <textarea id="jsonInput"></textarea>
    <select id="langSelect"><option value="en">en</option></select>
    <input id="level" />
    <div id="levelField"></div>
    <select id="metric"><option value="rmse">rmse</option></select>
    <select id="missingPolicy"><option value="error">error</option></select>
    <select id="mode"><option value="sync">sync</option><option value="job">job</option></select>
    <input id="modelId" />
    <input id="modelName" />
    <input id="quantiles" />
    <div id="quantilesField"></div>
    <button id="refreshDriftBaselineBtn"></button>
    <button id="refreshJobsBtn"></button>
    <div id="resultsCard"></div>
    <button id="runForecastBtn"></button>
    <button id="saveDriftBaselineBtn"></button>
    <button id="syncModelsBtn"></button>
    <select id="taskEl"><option value="forecast">forecast</option><option value="backtest">backtest</option><option value="train">train</option><option value="drift">drift</option></select>
    <select id="task"><option value="forecast">forecast</option><option value="backtest">backtest</option><option value="train">train</option><option value="drift">drift</option></select>
    <select id="trainAlgo"><option value="ridge">ridge</option></select>
    <input id="trainingHours" />
    <div id="trainParams"></div>
    <select id="uncertaintyMode"><option value="quantiles">quantiles</option><option value="level">level</option></select>
    <button id="validateBtn"></button>
  </body></html>`);
  const { document } = dom.window;
  document.getElementById("task").value = task;
  const calls = { refreshDrift: [], applyUncertaintyModeUi: [], idle: 0, billing: 0, aria: [] };
  const translations = {
    "action.run_train_and_forecast": "Run train",
    "action.run_drift": "Run drift",
    "action.run_backtest": "Run backtest",
    "action.run_forecast": "Run forecast",
  };
  const controller = createTaskResultsUiController({
    elements: {
      addJobIdBtnEl: document.getElementById("addJobIdBtn"),
      backtestParamsEl: document.getElementById("backtestParams"),
      baseModelEl: document.getElementById("baseModel"),
      cancelRunBtnEl: document.getElementById("cancelRunBtn"),
      checkHealthBtnEl: document.getElementById("checkHealthBtn"),
      clearBtnEl: document.getElementById("clearBtn"),
      copyLinkBtnEl: document.getElementById("copyLinkBtn"),
      copySnippetBtnEl: document.getElementById("copySnippetBtn"),
      csvFileEl: document.getElementById("csvFile"),
      densityModeEl: document.getElementById("densityMode"),
      downloadCsvEl: document.getElementById("downloadCsv"),
      downloadJsonEl: document.getElementById("downloadJson"),
      driftBaselinePanelEl: document.getElementById("driftBaselinePanel"),
      foldsEl: document.getElementById("folds"),
      forecastParamsEl: document.getElementById("forecastParams"),
      frequencyEl: document.getElementById("frequency"),
      horizonEl: document.getElementById("horizon"),
      horizonFieldEl: document.getElementById("horizonField"),
      jobIdInputEl: document.getElementById("jobIdInput"),
      jsonInputEl: document.getElementById("jsonInput"),
      langSelectEl: document.getElementById("langSelect"),
      levelEl: document.getElementById("level"),
      levelFieldEl: document.getElementById("levelField"),
      metricEl: document.getElementById("metric"),
      missingPolicyEl: document.getElementById("missingPolicy"),
      modeEl: document.getElementById("mode"),
      modelIdEl: document.getElementById("modelId"),
      modelNameEl: document.getElementById("modelName"),
      quantilesEl: document.getElementById("quantiles"),
      quantilesFieldEl: document.getElementById("quantilesField"),
      refreshDriftBaselineBtnEl: document.getElementById("refreshDriftBaselineBtn"),
      refreshJobsBtnEl: document.getElementById("refreshJobsBtn"),
      resultsCardEl: document.getElementById("resultsCard"),
      runForecastBtnEl: document.getElementById("runForecastBtn"),
      saveDriftBaselineBtnEl: document.getElementById("saveDriftBaselineBtn"),
      syncModelsBtnEl: document.getElementById("syncModelsBtn"),
      taskEl: document.getElementById("task"),
      trainAlgoEl: document.getElementById("trainAlgo"),
      trainingHoursEl: document.getElementById("trainingHours"),
      trainParamsEl: document.getElementById("trainParams"),
      uncertaintyModeEl: document.getElementById("uncertaintyMode"),
      validateBtnEl: document.getElementById("validateBtn"),
    },
    applyUncertaintyModeUi: (options) => calls.applyUncertaintyModeUi.push(options),
    currentTask: () => document.getElementById("task").value,
    getLastForecasts: () => lastForecasts,
    getLastResult: () => lastResult,
    inferUncertaintyModeFromInputs: () => "quantiles",
    isConnectionReady: () => true,
    isDriftBaselineBusy: () => false,
    normalizeUncertaintyMode: (value) => value,
    onIdle: () => {
      calls.idle += 1;
    },
    onRefreshDriftBaselineStatus: (options) => calls.refreshDrift.push(options),
    setAriaDisabled: (node, disabled) => {
      node.disabled = !!disabled;
      node.setAttribute("aria-disabled", disabled ? "true" : "false");
      calls.aria.push({ id: node.id, disabled: !!disabled });
    },
    setVisible: (node, visible) => {
      node.hidden = !visible;
    },
    t: (key) => translations[key] || key,
    updateBillingUi: () => {
      calls.billing += 1;
    },
  });

  return { calls, controller, document };
}

test("updateTaskUi fixes forecast task visibility and run label", () => {
  const { calls, controller, document } = createFixture({ task: "forecast" });
  document.getElementById("uncertaintyMode").value = "quantiles";

  controller.updateTaskUi();

  assert.equal(document.getElementById("forecastParams").hidden, false);
  assert.equal(document.getElementById("backtestParams").hidden, true);
  assert.equal(document.getElementById("trainParams").hidden, true);
  assert.equal(document.getElementById("driftBaselinePanel").hidden, true);
  assert.equal(document.getElementById("horizonField").hidden, false);
  assert.equal(document.getElementById("frequency").disabled, false);
  assert.equal(document.getElementById("quantiles").disabled, false);
  assert.equal(document.getElementById("level").disabled, true);
  assert.equal(document.getElementById("runForecastBtn").textContent, "Run forecast");
  assert.equal(calls.applyUncertaintyModeUi.length, 1);
  assert.equal(calls.billing, 1);
});

test("updateTaskUi fixes drift task state and refreshes baseline status", () => {
  const { calls, controller, document } = createFixture({ task: "drift" });
  document.getElementById("mode").value = "job";

  controller.updateTaskUi();

  assert.equal(document.getElementById("mode").value, "sync");
  assert.equal(document.getElementById("driftBaselinePanel").hidden, false);
  assert.equal(document.getElementById("horizonField").hidden, true);
  assert.equal(document.getElementById("saveDriftBaselineBtn").disabled, false);
  assert.equal(document.getElementById("refreshDriftBaselineBtn").disabled, false);
  assert.equal(document.getElementById("runForecastBtn").textContent, "Run drift");
  assert.deepEqual(calls.refreshDrift, [{ silent: true }]);
});

test("setRunUiBusy fixes busy state and restores actions on idle", () => {
  const { calls, controller, document } = createFixture({
    task: "forecast",
    lastResult: { ok: true },
    lastForecasts: [{ series_id: "unit-01" }],
  });

  controller.setRunUiBusy(true);
  assert.equal(document.getElementById("resultsCard").getAttribute("aria-busy"), "true");
  assert.equal(document.getElementById("runForecastBtn").disabled, true);
  assert.equal(document.getElementById("cancelRunBtn").disabled, false);
  assert.equal(document.getElementById("downloadJson").disabled, true);
  assert.equal(document.getElementById("downloadCsv").disabled, true);

  controller.setRunUiBusy(false);
  assert.equal(document.getElementById("resultsCard").getAttribute("aria-busy"), "false");
  assert.equal(document.getElementById("downloadJson").disabled, false);
  assert.equal(document.getElementById("downloadCsv").disabled, false);
  assert.equal(document.getElementById("copyLinkBtn").disabled, false);
  assert.equal(calls.idle, 1);
});

test("updateDownloadButtons and resetResultsActions fix result action state", () => {
  const { controller, document } = createFixture({
    task: "backtest",
    lastResult: { metrics: { rmse: 1 } },
    lastForecasts: [],
  });

  controller.updateDownloadButtons();
  assert.equal(document.getElementById("downloadJson").disabled, false);
  assert.equal(document.getElementById("downloadCsv").disabled, false);

  controller.setResultsActionPriority("train", true);
  assert.equal(document.getElementById("downloadJson").classList.contains("primary"), true);

  controller.resetResultsActions();
  assert.equal(document.getElementById("downloadJson").disabled, true);
  assert.equal(document.getElementById("downloadCsv").disabled, true);
});