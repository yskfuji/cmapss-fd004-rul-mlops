import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

import { createJobsModelsController } from "../../src/forecasting_api/static/forecasting_gui/jobs_models_controller.js";

function createFixture({ jobs = [], models = [], defaultModelId = null } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="jobsEmpty"></div>
    <table><tbody id="jobsTableBody"></tbody></table>
    <div id="modelsEmpty"></div>
    <table><tbody id="modelsTableBody"></tbody></table>
  </body></html>`);
  const { document } = dom.window;
  const calls = {
    checkJob: [],
    deleteModel: [],
    fetchJobResult: [],
    removeJob: [],
    setDefaultModel: [],
    updateModelMemo: [],
    useModel: [],
  };
  const translations = {
    "models.table.memo": "Memo",
    "models.action.use": "Use",
    "models.action.set_default": "Default",
    "models.action.delete": "Delete",
    "action.job_check": "Check",
    "action.job_result": "Result",
    "action.job_remove": "Remove",
    "jobs.status.completed": "Completed",
  };

  const controller = createJobsModelsController({
    elements: {
      jobsEmptyEl: document.getElementById("jobsEmpty"),
      jobsTableBodyEl: document.getElementById("jobsTableBody"),
      modelsEmptyEl: document.getElementById("modelsEmpty"),
      modelsTableBodyEl: document.getElementById("modelsTableBody"),
    },
    loadJobHistory: () => jobs,
    loadModelCatalog: () => models,
    getDefaultModelId: () => defaultModelId,
    formatLocalTime: (value) => `local:${value}`,
    onCheckJob: (jobId) => calls.checkJob.push(jobId),
    onDeleteModel: (modelId) => calls.deleteModel.push(modelId),
    onFetchJobResult: (jobId) => calls.fetchJobResult.push(jobId),
    onRemoveJob: (jobId) => calls.removeJob.push(jobId),
    onSetDefaultModel: (modelId) => calls.setDefaultModel.push(modelId),
    onUpdateModelMemo: (model) => calls.updateModelMemo.push(model),
    onUseModel: (model) => calls.useModel.push(model),
    setVisible: (node, visible) => {
      node.hidden = !visible;
    },
    t: (key) => translations[key] || key,
  });

  return { calls, controller, document };
}

function rowTexts(rows) {
  return [...rows].map((row) => [...row.querySelectorAll("td")].map((cell) => cell.textContent));
}

test("renderModels fixes row shape and wires actions", () => {
  const { calls, controller, document } = createFixture({
    models: [
      { model_id: "ridge-v1", created_at: "2026-03-25T10:00:00Z", memo: "stable" },
      { model_id: "ridge-v2", created_at: "2026-03-25T11:00:00Z", memo: "canary" },
    ],
    defaultModelId: "ridge-v2",
  });

  controller.renderModels();

  const rows = document.querySelectorAll("#modelsTableBody tr");
  assert.equal(rows.length, 2);
  assert.deepEqual(rowTexts(rows).map((cells) => cells.slice(0, 2)), [
    ["ridge-v1", "local:2026-03-25T10:00:00Z"],
    ["ridge-v2", "local:2026-03-25T11:00:00Z"],
  ]);
  assert.equal(rows[1].getAttribute("data-default-model"), "true");

  const memoInput = rows[0].querySelector("input");
  memoInput.value = "updated memo";
  memoInput.dispatchEvent(new document.defaultView.Event("change"));
  assert.equal(calls.updateModelMemo.length, 1);
  assert.equal(calls.updateModelMemo[0].memo, "updated memo");

  const firstButtons = rows[0].querySelectorAll("button");
  firstButtons[0].click();
  firstButtons[1].click();
  firstButtons[2].click();
  assert.equal(calls.useModel[0].model_id, "ridge-v1");
  assert.deepEqual(calls.setDefaultModel, ["ridge-v1"]);
  assert.deepEqual(calls.deleteModel, ["ridge-v1"]);
  assert.equal(firstButtons[1].disabled, false);
  assert.equal(rows[1].querySelectorAll("button")[1].disabled, true);
});

test("renderJobHistory fixes status text and action wiring", () => {
  const { calls, controller, document } = createFixture({
    jobs: [
      {
        job_id: "job-001",
        status: "completed",
        progress: 100,
        updated_at: "2026-03-25T12:00:00Z",
        request_id: "req-1",
        error_code: "E42",
        message: "done",
      },
    ],
  });

  controller.renderJobHistory({ translations: { "jobs.status.completed": "Completed" } });

  const rows = document.querySelectorAll("#jobsTableBody tr");
  assert.equal(rows.length, 1);
  assert.deepEqual(rowTexts(rows), [[
    "job-001",
    "Completed\nerror_code=E42\nrequest_id=req-1\nmessage=done",
    "100%",
    "local:2026-03-25T12:00:00Z",
    "Check Result Remove",
  ]]);

  const buttons = rows[0].querySelectorAll("button");
  buttons[0].click();
  buttons[1].click();
  buttons[2].click();
  assert.deepEqual(calls.checkJob, ["job-001"]);
  assert.deepEqual(calls.fetchJobResult, ["job-001"]);
  assert.deepEqual(calls.removeJob, ["job-001"]);
});