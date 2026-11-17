export function createTaskResultsUiController({
  elements,
  applyUncertaintyModeUi,
  currentTask,
  getLastForecasts,
  getLastResult,
  inferUncertaintyModeFromInputs,
  isConnectionReady,
  isDriftBaselineBusy,
  normalizeUncertaintyMode,
  onIdle,
  onRefreshDriftBaselineStatus,
  setAriaDisabled,
  setVisible,
  t,
  updateBillingUi,
}) {
  const {
    addJobIdBtnEl,
    backtestParamsEl,
    baseModelEl,
    cancelRunBtnEl,
    checkHealthBtnEl,
    clearBtnEl,
    copyLinkBtnEl,
    copySnippetBtnEl,
    csvFileEl,
    densityModeEl,
    downloadCsvEl,
    downloadJsonEl,
    driftBaselinePanelEl,
    foldsEl,
    forecastParamsEl,
    frequencyEl,
    horizonEl,
    horizonFieldEl,
    jobIdInputEl,
    jsonInputEl,
    langSelectEl,
    levelEl,
    levelFieldEl,
    metricEl,
    missingPolicyEl,
    modeEl,
    modelIdEl,
    modelNameEl,
    quantilesEl,
    quantilesFieldEl,
    refreshDriftBaselineBtnEl,
    refreshJobsBtnEl,
    resultsCardEl,
    runForecastBtnEl,
    saveDriftBaselineBtnEl,
    syncModelsBtnEl,
    taskEl,
    trainAlgoEl,
    trainingHoursEl,
    trainParamsEl,
    uncertaintyModeEl,
    validateBtnEl,
  } = elements;

  function updateDownloadButtons() {
    const lastResult = getLastResult();
    downloadJsonEl.disabled = !lastResult;
    const task = currentTask();
    if (task === "forecast") {
      const forecasts = Array.isArray(getLastForecasts()) ? getLastForecasts() : [];
      downloadCsvEl.disabled = forecasts.length === 0;
      return;
    }
    if (task === "backtest") {
      downloadCsvEl.disabled = !lastResult;
      return;
    }
    downloadCsvEl.disabled = true;
  }

  function setResultsActionPriority(task, hasResults) {
    const isForecast = task === "forecast";
    const isTrain = task === "train";
    downloadCsvEl.classList.toggle("primary", hasResults && isForecast);
    downloadJsonEl.classList.toggle("primary", hasResults && !isForecast);
    downloadCsvEl.classList.toggle("secondary", !hasResults || !isForecast);
    downloadJsonEl.classList.toggle("secondary", !hasResults || isForecast);
    if (isTrain && hasResults) {
      downloadJsonEl.classList.add("primary");
    }
  }

  function resetResultsActions() {
    downloadJsonEl.disabled = true;
    downloadCsvEl.disabled = true;
    setResultsActionPriority(currentTask(), false);
  }

  function updateTaskUi() {
    const task = currentTask();
    const isForecast = task === "forecast";
    const isBacktest = task === "backtest";
    const isTrain = task === "train";
    const isDrift = task === "drift";

    const jobModeOptionEl = modeEl?.querySelector('option[value="job"]');
    if (jobModeOptionEl) jobModeOptionEl.disabled = isDrift;
    if (isDrift && modeEl.value === "job") modeEl.value = "sync";

    setVisible(horizonFieldEl, !isTrain && !isDrift);
    setVisible(forecastParamsEl, isForecast);
    setVisible(backtestParamsEl, isBacktest);
    setVisible(trainParamsEl, isTrain);
    setVisible(driftBaselinePanelEl, isDrift);
    if (saveDriftBaselineBtnEl) saveDriftBaselineBtnEl.disabled = !isDrift;
    if (refreshDriftBaselineBtnEl) refreshDriftBaselineBtnEl.disabled = !isDrift;
    if (isDrift && isConnectionReady() && !isDriftBaselineBusy()) {
      void onRefreshDriftBaselineStatus({ silent: true });
    }

    horizonEl.disabled = isTrain || isDrift;
    frequencyEl.disabled = !isForecast;
    missingPolicyEl.disabled = !isForecast;
    if (uncertaintyModeEl) uncertaintyModeEl.disabled = !isForecast;

    const supported = !!uncertaintyModeEl && !!quantilesFieldEl && !!levelFieldEl;
    const mode = normalizeUncertaintyMode(supported ? uncertaintyModeEl.value : inferUncertaintyModeFromInputs());
    quantilesEl.disabled = !isForecast || mode !== "quantiles";
    levelEl.disabled = !isForecast || mode !== "level";
    if (isForecast) {
      applyUncertaintyModeUi({ applyDefaults: false, clearInactive: false });
    } else {
      setVisible(quantilesFieldEl, false);
      setVisible(levelFieldEl, false);
    }
    modelIdEl.disabled = !isForecast;

    foldsEl.disabled = !isBacktest;
    metricEl.disabled = !isBacktest;

    baseModelEl.disabled = !isTrain;
    if (trainAlgoEl) trainAlgoEl.disabled = !isTrain;
    modelNameEl.disabled = !isTrain;
    trainingHoursEl.disabled = !isTrain;

    if (runForecastBtnEl) {
      const runKey = isTrain
        ? "action.run_train_and_forecast"
        : isDrift
          ? "action.run_drift"
          : isBacktest
            ? "action.run_backtest"
            : "action.run_forecast";
      runForecastBtnEl.textContent = t(runKey);
      runForecastBtnEl.setAttribute("aria-label", t(runKey));
    }

    updateBillingUi();
  }

  function setRunUiBusy(busy) {
    if (resultsCardEl) resultsCardEl.setAttribute("aria-busy", busy ? "true" : "false");

    setAriaDisabled(runForecastBtnEl, busy);
    setAriaDisabled(cancelRunBtnEl, !busy);
    setAriaDisabled(validateBtnEl, busy);
    setAriaDisabled(clearBtnEl, busy);
    setAriaDisabled(checkHealthBtnEl, busy);
    if (syncModelsBtnEl) setAriaDisabled(syncModelsBtnEl, busy);
    if (saveDriftBaselineBtnEl) setAriaDisabled(saveDriftBaselineBtnEl, busy || currentTask() !== "drift");
    if (refreshDriftBaselineBtnEl) setAriaDisabled(refreshDriftBaselineBtnEl, busy || currentTask() !== "drift");

    langSelectEl.disabled = !!busy;
    if (densityModeEl) densityModeEl.disabled = !!busy;
    modeEl.disabled = !!busy;
    taskEl.disabled = !!busy;
    csvFileEl.disabled = !!busy;
    jsonInputEl.disabled = !!busy;
    horizonEl.disabled = !!busy;
    frequencyEl.disabled = !!busy;
    missingPolicyEl.disabled = !!busy;
    quantilesEl.disabled = !!busy;
    levelEl.disabled = !!busy;
    modelIdEl.disabled = !!busy;
    foldsEl.disabled = !!busy;
    metricEl.disabled = !!busy;
    baseModelEl.disabled = !!busy;
    if (trainAlgoEl) trainAlgoEl.disabled = !!busy;
    modelNameEl.disabled = !!busy;
    trainingHoursEl.disabled = !!busy;

    jobIdInputEl.disabled = !!busy;
    setAriaDisabled(addJobIdBtnEl, busy);
    setAriaDisabled(refreshJobsBtnEl, busy);

    if (busy) {
      downloadJsonEl.disabled = true;
      downloadCsvEl.disabled = true;
      setAriaDisabled(copyLinkBtnEl, true);
      setAriaDisabled(copySnippetBtnEl, true);
      return;
    }

    updateTaskUi();
    updateDownloadButtons();
    setAriaDisabled(copyLinkBtnEl, false);
    setAriaDisabled(copySnippetBtnEl, false);
    if (typeof onIdle === "function") onIdle();
  }

  return {
    resetResultsActions,
    setResultsActionPriority,
    setRunUiBusy,
    updateDownloadButtons,
    updateTaskUi,
  };
}