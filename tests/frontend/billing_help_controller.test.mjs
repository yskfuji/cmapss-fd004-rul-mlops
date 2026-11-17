import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";

import { createBillingHelpController } from "../../src/forecasting_api/static/forecasting_gui/billing_help_controller.js";

function createFixture(estimate) {
  const dom = new JSDOM(`<!doctype html><html><body>
    <div id="billingCard"></div>
    <div id="helpCard"></div>
    <div id="resultsCard"></div>
    <div id="paramsCard"></div>
    <div id="dataCard"></div>
    <input id="billingSyncMaxPoints" />
    <div id="billingQuotaStatus"></div>
    <div id="billingEstimate"></div>
    <div id="billingUsageStatus"></div>
    <div id="billingSummaryLimit"></div>
    <div id="billingSummaryEstimate"></div>
    <div id="billingSummaryRemaining"></div>
    <div id="billingMetricRemaining"></div>
    <div id="billingMetricUsed"></div>
    <div id="billingMetricLimit"></div>
    <div><div id="quotaProgressBar"></div></div>
    <div><div id="billingDetailUsedBar"></div></div>
    <div><div id="billingDetailRemainingBar"></div></div>
    <div id="billingImpactSeriesBar"></div>
    <div id="billingImpactHorizonBar"></div>
    <div id="billingImpactSeriesValue"></div>
    <div id="billingImpactHorizonValue"></div>
    <div id="quotaFlowValue"></div>
    <div><div id="quotaFlowBarFill"></div></div>
    <div id="quotaFlowAlert"></div>
    <div id="billingDetailBody"></div>
    <div id="billingWarn"></div>
    <div id="billingError"></div>
    <select id="helpTopic">
      <option value="data">data</option>
      <option value="quota">quota</option>
      <option value="support">support</option>
      <option value="results">results</option>
    </select>
    <div id="helpAdvice"></div>
    <details id="helpInputs"></details>
    <details id="helpSupport"></details>
  </body></html>`);
  const { document } = dom.window;
  const scrollCalls = [];
  for (const id of ["billingCard", "helpCard", "resultsCard", "paramsCard", "dataCard"]) {
    document.getElementById(id).scrollIntoView = (opts) => scrollCalls.push({ id, opts });
  }
  const statusCalls = [];
  let refreshCount = 0;
  const translations = {
    "help.concierge.advice.data": "Load data first",
    "help.concierge.advice.quota": "Check quota",
    "help.concierge.advice.support": "Contact support",
    "billing.points_this_month": "Points {remaining}/{limit}/{used}",
    "billing.estimate_points": "Estimate {points} {s} {h}",
    "billing.remaining_points": "Remaining {remaining}",
    "billing.remaining_na": "Remaining n/a",
    "billing.summary.limit_value": "Limit {limit}",
    "billing.summary.estimate_value": "Estimate {points}",
    "billing.summary.estimate_na": "Estimate n/a",
    "billing.summary.remaining_value": "Remain {remaining}",
    "billing.summary.remaining_na": "Remain n/a",
    "billing.metric.remaining_value": "MR {remaining}",
    "billing.metric.used_value": "MU {used}",
    "billing.metric.limit_value": "ML {limit}",
    "billing.chart.unit_points": "Units {points}",
    "billing.flow_value": "Flow {remaining} {limit} {used}",
    "billing.flow_value_na": "Flow n/a {limit}",
    "billing.flow_alert_over": "Over {remaining}",
    "billing.flow_alert_near": "Near {remaining}",
    "billing.flow_alert_ok": "Ok {remaining}",
    "billing.flow_alert_na": "No estimate",
    "billing.detail_pending": "Pending",
    "billing.detail_value": "Value {points} {s} {h}",
    "billing.detail_remaining": "Remain {remaining} {limit} {used}",
    "billing.detail_reduce": "Reduce {over} {reduce_series} {reduce_horizon}",
    "billing.detail_expand": "Expand {extra_series} {extra_horizon}",
    "billing.error_over_limit_cost01": "Over limit",
    "billing.warn_near_limit": "Near limit",
  };
  const t = (key, vars = null) => {
    const raw = translations[key] || key;
    if (!vars) return raw;
    return raw.replaceAll(/\{([a-zA-Z0-9_]+)\}/g, (_m, name) => String(vars[name] ?? ""));
  };

  const controller = createBillingHelpController({
    elements: {
      billingDetailBodyEl: document.getElementById("billingDetailBody"),
      billingDetailRemainingBarEl: document.getElementById("billingDetailRemainingBar"),
      billingDetailUsedBarEl: document.getElementById("billingDetailUsedBar"),
      billingErrorEl: document.getElementById("billingError"),
      billingEstimateEl: document.getElementById("billingEstimate"),
      billingImpactHorizonBarEl: document.getElementById("billingImpactHorizonBar"),
      billingImpactHorizonValueEl: document.getElementById("billingImpactHorizonValue"),
      billingImpactSeriesBarEl: document.getElementById("billingImpactSeriesBar"),
      billingImpactSeriesValueEl: document.getElementById("billingImpactSeriesValue"),
      billingMetricLimitEl: document.getElementById("billingMetricLimit"),
      billingMetricRemainingEl: document.getElementById("billingMetricRemaining"),
      billingMetricUsedEl: document.getElementById("billingMetricUsed"),
      billingQuotaStatusEl: document.getElementById("billingQuotaStatus"),
      billingSummaryEstimateEl: document.getElementById("billingSummaryEstimate"),
      billingSummaryLimitEl: document.getElementById("billingSummaryLimit"),
      billingSummaryRemainingEl: document.getElementById("billingSummaryRemaining"),
      billingSyncMaxPointsEl: document.getElementById("billingSyncMaxPoints"),
      billingUsageStatusEl: document.getElementById("billingUsageStatus"),
      billingWarnEl: document.getElementById("billingWarn"),
      helpAdviceEl: document.getElementById("helpAdvice"),
      helpInputsEl: document.getElementById("helpInputs"),
      helpSupportEl: document.getElementById("helpSupport"),
      helpTopicEl: document.getElementById("helpTopic"),
      quotaFlowAlertEl: document.getElementById("quotaFlowAlert"),
      quotaFlowBarFillEl: document.getElementById("quotaFlowBarFill"),
      quotaFlowValueEl: document.getElementById("quotaFlowValue"),
      quotaProgressBarEl: document.getElementById("quotaProgressBar"),
    },
    getRunPointsEstimate: () => estimate,
    refreshI18nStatuses: () => {
      refreshCount += 1;
    },
    setStatusI18n: (node, key, vars) => {
      statusCalls.push({ id: node.id, key, vars });
      node.textContent = t(key, vars);
    },
    setText: (node, text) => {
      node.textContent = String(text ?? "");
    },
    setVisible: (node, visible) => {
      node.hidden = !visible;
    },
    t,
  });

  return { controller, document, refreshCountRef: () => refreshCount, scrollCalls, statusCalls };
}

test("help advice and guide flow fix topic advice, detail open, and scroll target", () => {
  const { controller, document, scrollCalls } = createFixture({
    limit: 100,
    seriesShown: 2,
    hShown: 5,
    canEstimate: true,
    pointsEstimate: 10,
    estimateUsed: 20,
    remaining: 80,
    ratio: 0.2,
  });
  document.getElementById("helpTopic").value = "support";

  controller.updateHelpAdvice();
  controller.guideHelpFlow();

  assert.equal(document.getElementById("helpAdvice").textContent, "Contact support");
  assert.equal(document.getElementById("helpSupport").open, true);
  assert.deepEqual(scrollCalls.map((entry) => entry.id), ["helpCard"]);
  assert.equal(controller.matchHelpTopicFromText("料金 limit exceeded"), "quota");
  assert.equal(controller.matchHelpTopicFromText("request_id error"), "support");
});

test("updateBillingUi fixes metrics, bars, and warning state", () => {
  const { controller, document, refreshCountRef, statusCalls } = createFixture({
    limit: 100,
    seriesShown: 4,
    hShown: 5,
    canEstimate: true,
    pointsEstimate: 20,
    estimateUsed: 85,
    remaining: 15,
    ratio: 0.85,
  });

  controller.updateBillingUi();

  assert.equal(document.getElementById("billingSyncMaxPoints").value, "100");
  assert.equal(document.getElementById("billingEstimate").textContent, "Estimate 20 4 5");
  assert.equal(document.getElementById("billingUsageStatus").textContent, "Remaining 15");
  assert.equal(document.getElementById("billingSummaryLimit").textContent, "Limit 100");
  assert.equal(document.getElementById("billingMetricUsed").textContent, "MU 85");
  assert.equal(document.getElementById("quotaProgressBar").style.width, "15%");
  assert.equal(document.getElementById("quotaFlowBarFill").style.width, "85%");
  assert.equal(document.getElementById("quotaFlowAlert").textContent, "Near 15");
  assert.match(document.getElementById("billingDetailBody").textContent, /Expand 3 3/);
  assert.equal(document.getElementById("billingWarn").hidden, false);
  assert.equal(document.getElementById("billingWarn").textContent, "Near limit");
  assert.equal(document.getElementById("billingError").hidden, true);
  assert.equal(refreshCountRef(), 1);
  assert.equal(statusCalls[0].key, "billing.points_this_month");
});

test("updateBillingUi fixes over-limit error state", () => {
  const { controller, document } = createFixture({
    limit: 100,
    seriesShown: 4,
    hShown: 5,
    canEstimate: true,
    pointsEstimate: 20,
    estimateUsed: 110,
    remaining: -10,
    ratio: 1.1,
  });

  controller.updateBillingUi();

  assert.equal(document.getElementById("billingError").hidden, false);
  assert.equal(document.getElementById("billingError").textContent, "Over limit");
  assert.equal(document.getElementById("billingWarn").hidden, true);
  assert.match(document.getElementById("billingDetailBody").textContent, /Reduce 10 2 3/);
});