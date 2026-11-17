const NEXT_STEP_MAP = {
  1: { id: "connectionCard", key: "next.step.connection" },
  2: { id: "dataCard", key: "next.step.data" },
  3: { id: "paramsCard", key: "next.step.params" },
  4: { id: "resultsCard", key: "next.step.results" },
};

export function createProgressGateController({
  elements,
  computeRunGateState,
  currentTask,
  currentDataSource,
  getCurrentDriftBaselineState,
  getLastDataGapCount,
  getLastDataMissing,
  getLastDataStats,
  getLastHealthOk,
  getRunBlockReasonKeysFromState,
  getRunBlockReasons,
  getRunGateExplained,
  getRunState,
  getStepCompletion,
  hasDataForSource,
  isConnectionReady,
  isDataReady,
  isDriftBaselineBusy,
  onRefreshDriftBaselineStatus,
  setAriaDisabled,
  setSectionStatus,
  setStatus,
  t,
  updateResultsStatus,
}) {
  const {
    checkDataInputEl,
    checkDataRequiredEl,
    checkDataStatusEl,
    checkDataValidateEl,
    checkParamsHorizonEl,
    checkParamsQuantilesEl,
    checkParamsStatusEl,
    horizonEl,
    levelEl,
    missingPolicyEl,
    nextStepLinkEl,
    quantilesEl,
    resultsInterpretationEl,
    resultsSummaryEl,
    runForecastBtnEl,
    runStatusEl,
    statusBillingEstimateEl,
    statusBillingLimitEl,
    statusConnectionApiKeyEl,
    statusConnectionHealthEl,
    statusDataInputEl,
    statusDataValidationEl,
    statusParamsHorizonEl,
    statusParamsQuantilesEl,
  } = elements;

  function markerForState(stateKey) {
    const key = `status.short.${stateKey}`;
    const fallback = {
      done: "OK",
      check: "CHECK",
      wait: "WAIT",
      optional: "OPT",
      na: "N/A",
      todo: "TODO",
    };
    const label = t(key);
    if (label && label !== key) return label;
    return fallback[stateKey] || fallback.todo;
  }

  function formatChecklist(labelKey) {
    return t(labelKey);
  }

  function formatChecklistSummary(done, total) {
    return t("checklist.summary", { done, total });
  }

  function updateRunGate({ isUiBusy = false } = {}) {
    if (!runForecastBtnEl) return;
    const driftBaselineState = getCurrentDriftBaselineState();
    if (currentTask() === "drift" && isConnectionReady() && !driftBaselineState.checked && !isDriftBaselineBusy()) {
      void onRefreshDriftBaselineStatus({ silent: true });
    }
    const reasonKeys = getRunBlockReasonKeysFromState();
    const reasons = getRunBlockReasons();
    const { blocked } = computeRunGateState({ isUiBusy, reasonKeys });
    setAriaDisabled(runForecastBtnEl, blocked);
    if (!isUiBusy && runStatusEl) {
      const runState = getRunState();
      if (blocked && reasons.length && getRunGateExplained()) {
        setStatus(runStatusEl, `${t("status.run_blocked_preconditions")}\n- ${reasons.join("\n- ")}`, "warn");
      } else if (!runState?.status || runState.status === "idle") {
        setStatus(runStatusEl, "");
      }
    }
  }

  function updateSectionStatus() {
    setSectionStatus(statusConnectionApiKeyEl, "status.connection.api_key", isConnectionReady() ? "done" : "todo");
    if (statusConnectionHealthEl) {
      const state = isConnectionReady() ? (getLastHealthOk() ? "done" : "todo") : "wait";
      setSectionStatus(statusConnectionHealthEl, "status.connection.health", state);
    }

    const source = currentDataSource();
    const hasInput = hasDataForSource(source);
    const missingPolicy = (missingPolicyEl.value || "").trim();
    const hasGapIssue = missingPolicy === "error" && Number(getLastDataGapCount()) > 0;
    setSectionStatus(statusDataInputEl, "status.data.input", hasInput ? "done" : "todo");
    setSectionStatus(statusDataValidationEl, "status.data.validation", isDataReady() ? (hasGapIssue ? "check" : "done") : "todo");

    if (checkDataInputEl) {
      const state = hasInput ? "done" : "todo";
      checkDataInputEl.textContent = formatChecklist("checklist.data.input", state);
      checkDataInputEl.dataset.marker = markerForState(state);
      checkDataInputEl.dataset.state = state;
    }
    if (checkDataRequiredEl) {
      const missing = getLastDataMissing();
      const ok =
        missing &&
        Number(missing.series_id) === 0 &&
        Number(missing.timestamp) === 0 &&
        Number(missing.y) === 0;
      const state = isDataReady() ? (ok ? "done" : "check") : "todo";
      checkDataRequiredEl.textContent = formatChecklist("checklist.data.required", state);
      checkDataRequiredEl.dataset.marker = markerForState(state);
      checkDataRequiredEl.dataset.state = state;
    }
    if (checkDataValidateEl) {
      const state = isDataReady() ? (hasGapIssue ? "check" : "done") : "todo";
      checkDataValidateEl.textContent = formatChecklist("checklist.data.validate", state);
      checkDataValidateEl.dataset.marker = markerForState(state);
      checkDataValidateEl.dataset.state = state;
    }
    if (checkDataStatusEl) {
      const missing = getLastDataMissing();
      const states = [
        hasInput ? "done" : "todo",
        isDataReady() ? (hasGapIssue ? "check" : "done") : "todo",
        isDataReady() && missing
          ? Number(missing.series_id) === 0 && Number(missing.timestamp) === 0 && Number(missing.y) === 0
            ? "done"
            : "check"
          : "todo",
      ];
      const done = states.filter((state) => state === "done").length;
      checkDataStatusEl.textContent = formatChecklistSummary(done, states.length);
    }

    const task = currentTask();
    if (task === "train") {
      setSectionStatus(statusParamsHorizonEl, "status.params.horizon", "na");
      setSectionStatus(statusParamsQuantilesEl, "status.params.quantiles", "na");
    } else {
      const horizonValue = Number(horizonEl?.value);
      const horizonOk = Number.isFinite(horizonValue) && horizonValue >= 1;
      if (!isDataReady()) {
        setSectionStatus(statusParamsHorizonEl, "status.params.horizon", "wait");
      } else {
        setSectionStatus(statusParamsHorizonEl, "status.params.horizon", horizonOk ? "done" : "todo");
      }

      const quantiles = String(quantilesEl?.value || "").trim();
      const level = String(levelEl?.value || "").trim();
      if (quantiles && level) {
        setSectionStatus(statusParamsQuantilesEl, "status.params.quantiles", "check");
      } else if (!quantiles && !level) {
        setSectionStatus(statusParamsQuantilesEl, "status.params.quantiles", "optional");
      } else {
        setSectionStatus(statusParamsQuantilesEl, "status.params.quantiles", "done");
      }
    }

    if (checkParamsHorizonEl) {
      const horizonValue = Number(horizonEl?.value);
      const horizonOk = Number.isFinite(horizonValue) && horizonValue >= 1;
      const state = isDataReady() ? (horizonOk ? "done" : "todo") : "wait";
      checkParamsHorizonEl.textContent = formatChecklist("checklist.params.horizon", state);
      checkParamsHorizonEl.dataset.marker = markerForState(state);
      checkParamsHorizonEl.dataset.state = state;
    }
    if (checkParamsQuantilesEl) {
      const quantiles = String(quantilesEl?.value || "").trim();
      const level = String(levelEl?.value || "").trim();
      const state = quantiles && level ? "check" : "optional";
      checkParamsQuantilesEl.textContent = formatChecklist("checklist.params.quantiles", state);
      checkParamsQuantilesEl.dataset.marker = markerForState(state);
      checkParamsQuantilesEl.dataset.state = state;
    }
    if (checkParamsStatusEl) {
      const horizonValue = Number(horizonEl?.value);
      const horizonOk = Number.isFinite(horizonValue) && horizonValue >= 1;
      const quantiles = String(quantilesEl?.value || "").trim();
      const level = String(levelEl?.value || "").trim();
      const states = [
        isDataReady() ? (horizonOk ? "done" : "todo") : "wait",
        quantiles && level ? "check" : "optional",
      ];
      const done = states.filter((state) => state === "done").length;
      checkParamsStatusEl.textContent = formatChecklistSummary(done, states.length);
    }

    updateResultsStatus();
    setSectionStatus(statusBillingLimitEl, "status.billing.limit", "reference");
    const estimateOk = Number(getLastDataStats()?.seriesCount) > 0 && Number(horizonEl?.value) >= 1;
    setSectionStatus(statusBillingEstimateEl, "status.billing.estimate", estimateOk ? "done" : "wait");
  }

  function updateNextStep() {
    if (!nextStepLinkEl) return;
    const completion = getStepCompletion();
    const order = [1, 2, 3, 4];
    const next = order.find((step) => !completion[step]);
    if (!next) {
      nextStepLinkEl.textContent = t("next.step.ready");
      nextStepLinkEl.classList.add("is-complete");
      const summaryReady = !!(resultsSummaryEl && !resultsSummaryEl.hidden);
      const interpretationReady = !!(resultsInterpretationEl && !resultsInterpretationEl.hidden);
      const completedTarget = summaryReady
        ? "#resultsSummary"
        : interpretationReady
          ? "#resultsInterpretation"
          : "#resultsCard";
      nextStepLinkEl.setAttribute("href", completedTarget);
      return;
    }

    const cfg = NEXT_STEP_MAP[next];
    nextStepLinkEl.textContent = t("next.step.label", { step: t(cfg.key) });
    nextStepLinkEl.classList.remove("is-complete");
    nextStepLinkEl.setAttribute("href", `#${cfg.id}`);
  }

  function refreshI18nStatuses({ isUiBusy = false } = {}) {
    updateSectionStatus();
    updateNextStep();
    updateRunGate({ isUiBusy });
  }

  return {
    refreshI18nStatuses,
    updateNextStep,
    updateRunGate,
    updateSectionStatus,
  };
}