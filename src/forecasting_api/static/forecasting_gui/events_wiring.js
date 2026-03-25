export function wireCoreEvents(ctx) {
  const {
    downloadJsonEl,
    downloadCsvEl,
    validateBtnEl,
    saveDriftBaselineBtnEl,
    refreshDriftBaselineBtnEl,
    runForecastBtnEl,
    cancelRunBtnEl,
    copyLinkBtnEl,
    copySnippetBtnEl,
    copyErrorBtnEl,
    copyRequestIdBtnEl,
    clearBtnEl,
    jsonInputEl,
    csvFileEl,
    loadSampleBtnEl,
    resultsChartTypeEl,
    resultsSeriesEl,
    ackGapsEl,
    missingPolicyEl,
    frequencyEl,
    uncertaintyModeEl,
    checkHealthBtnEl,
    refreshJobsBtnEl,
    addJobIdBtnEl,
    jobIdInputEl,
    jobsStatusEl,
    horizonEl,
    taskEl,
    quantilesEl,
    levelEl,
    valueUnitEl,
    modelIdEl,
    foldsEl,
    metricEl,
    baseModelEl,
    trainAlgoEl,
    modelNameEl,
    trainingHoursEl,
    paramsSyncEl,
    rollbackDefaultModelBtnEl,
    modelsStatusEl,
    syncModelsBtnEl,
    getLastResult,
    getLastTask,
    getLastForecasts,
    getLastRecords,
    getLastInferredFrequency,
    setLastForecastSeriesId,
    setParamsLinked,
    setParamsDirty,
    currentTask,
    downloadBlob,
    backtestToCsv,
    forecastsToCsv,
    onValidate,
    onPersistDriftBaseline,
    onRefreshDriftBaseline,
    onRun,
    onCancelRun,
    onCopyLink,
    onCopySnippet,
    onCopyError,
    onCopyRequestId,
    onClear,
    onLoadSample,
    syncInputLocks,
    invalidateValidationState,
    updateWarningsForCurrentInputs,
    updateCsvFileName,
    renderSelectedForecastSeries,
    renderResult,
    updateRunGate,
    applyUncertaintyModeUi,
    updateDataAnalysis,
    onHealth,
    refreshAllJobs,
    showJobsError,
    setStatusI18n,
    upsertJobHistoryEntry,
    setStatus,
    renderJobHistory,
    markParamsDirty,
    showError,
    showWarn,
    clearResults,
    updateTaskUi,
    syncParamsFromData,
    setDefaultModelId,
    renderModels,
    syncModelsFromServer,
  } = ctx;

  downloadJsonEl.addEventListener("click", () => {
    const lastResult = getLastResult();
    if (!lastResult) return;
    const task = currentTask();
    downloadBlob(
      task === "backtest" ? "backtest_result.json" : task === "train" ? "train_result.json" : "forecast_result.json",
      "application/json",
      JSON.stringify(lastResult, null, 2),
    );
  });

  downloadCsvEl.addEventListener("click", () => {
    const lastResult = getLastResult();
    if (!lastResult) return;
    const task = currentTask();
    if (task === "backtest") {
      downloadBlob("backtest_result.csv", "text/csv", backtestToCsv(lastResult));
      return;
    }
    if (task === "train") return;
    downloadBlob("forecasts.csv", "text/csv", forecastsToCsv(lastResult));
  });

  validateBtnEl.addEventListener("click", () => void onValidate());
  if (saveDriftBaselineBtnEl) {
    saveDriftBaselineBtnEl.addEventListener("click", () => void onPersistDriftBaseline());
  }
  if (refreshDriftBaselineBtnEl) {
    refreshDriftBaselineBtnEl.addEventListener("click", () => void onRefreshDriftBaseline());
  }
  runForecastBtnEl.addEventListener("click", () => void onRun());
  cancelRunBtnEl.addEventListener("click", () => void onCancelRun());
  copyLinkBtnEl.addEventListener("click", () => void onCopyLink());
  copySnippetBtnEl.addEventListener("click", () => void onCopySnippet());
  copyErrorBtnEl.addEventListener("click", () => void onCopyError());
  copyRequestIdBtnEl.addEventListener("click", () => void onCopyRequestId());
  clearBtnEl.addEventListener("click", () => void onClear());

  jsonInputEl.addEventListener("input", () => {
    syncInputLocks();
    invalidateValidationState();
    updateWarningsForCurrentInputs();
  });

  csvFileEl.addEventListener("change", () => {
    syncInputLocks();
    invalidateValidationState();
    updateWarningsForCurrentInputs();
    updateCsvFileName();
  });

  loadSampleBtnEl.addEventListener("click", () => void onLoadSample());

  if (resultsChartTypeEl) {
    resultsChartTypeEl.addEventListener("change", () => {
      const lastTask = getLastTask();
      const lastForecasts = getLastForecasts();
      if (lastTask === "forecast" && lastForecasts.length) {
        renderSelectedForecastSeries();
        return;
      }
      const lastResult = getLastResult();
      if (lastResult && lastTask) {
        renderResult(lastTask, lastResult);
      }
    });
  }

  if (resultsSeriesEl) {
    resultsSeriesEl.addEventListener("change", () => {
      setLastForecastSeriesId(resultsSeriesEl.value);
      const lastTask = getLastTask();
      const lastForecasts = getLastForecasts();
      if (lastTask === "forecast" && lastForecasts.length) {
        renderSelectedForecastSeries();
      }
    });
  }

  if (ackGapsEl) {
    ackGapsEl.addEventListener("change", updateRunGate);
  }

  missingPolicyEl.addEventListener("change", () => {
    const lastRecords = getLastRecords();
    if (lastRecords) {
      updateDataAnalysis(lastRecords, { inferredFrequency: getLastInferredFrequency() });
    } else {
      updateRunGate();
    }
  });

  if (uncertaintyModeEl) {
    uncertaintyModeEl.addEventListener("change", () => {
      markParamsDirty();
      invalidateValidationState();
      applyUncertaintyModeUi({ applyDefaults: true, clearInactive: true });
      updateWarningsForCurrentInputs();
    });
  }

  frequencyEl.addEventListener("input", () => {
    const lastRecords = getLastRecords();
    if (lastRecords) {
      updateDataAnalysis(lastRecords, { inferredFrequency: getLastInferredFrequency() });
    }
  });

  checkHealthBtnEl.addEventListener("click", () => void onHealth());

  refreshJobsBtnEl.addEventListener("click", () =>
    void (async () => {
      try {
        await refreshAllJobs();
      } catch (e) {
        showJobsError(e);
      }
    })(),
  );

  addJobIdBtnEl.addEventListener("click", () => {
    const id = String(jobIdInputEl.value || "").trim();
    if (!id) {
      setStatusI18n(jobsStatusEl, "client.err.job_id_required");
      return;
    }
    upsertJobHistoryEntry({ job_id: id, status: null, progress: null });
    jobIdInputEl.value = "";
    setStatus(jobsStatusEl, "");
    renderJobHistory();
  });

  jobIdInputEl.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    addJobIdBtnEl.click();
  });

  horizonEl.addEventListener("input", () => {
    markParamsDirty();
    invalidateValidationState();
    updateWarningsForCurrentInputs();
  });

  taskEl.addEventListener("change", () => {
    markParamsDirty();
    invalidateValidationState();
    showError("");
    showWarn("");
    clearResults();
    updateTaskUi();
    updateWarningsForCurrentInputs();
  });

  frequencyEl.addEventListener("input", () => {
    markParamsDirty();
    invalidateValidationState();
    updateWarningsForCurrentInputs();
  });

  missingPolicyEl.addEventListener("change", () => {
    markParamsDirty();
    invalidateValidationState();
    updateWarningsForCurrentInputs();
    const lastRecords = getLastRecords();
    if (lastRecords) {
      updateDataAnalysis(lastRecords, { inferredFrequency: getLastInferredFrequency() });
    }
  });

  quantilesEl.addEventListener("input", () => {
    markParamsDirty();
    invalidateValidationState();
    updateWarningsForCurrentInputs();
  });

  levelEl.addEventListener("input", () => {
    markParamsDirty();
    invalidateValidationState();
    updateWarningsForCurrentInputs();
  });

  valueUnitEl.addEventListener("input", () => {
    markParamsDirty();
  });

  if (modelIdEl) {
    modelIdEl.addEventListener("input", () => {
      markParamsDirty();
      invalidateValidationState();
      updateWarningsForCurrentInputs();
    });
  }

  if (foldsEl) {
    foldsEl.addEventListener("input", () => {
      markParamsDirty();
      invalidateValidationState();
      updateWarningsForCurrentInputs();
    });
  }

  if (metricEl) {
    metricEl.addEventListener("change", () => {
      markParamsDirty();
      invalidateValidationState();
      updateWarningsForCurrentInputs();
    });
  }

  if (baseModelEl) {
    baseModelEl.addEventListener("input", () => {
      markParamsDirty();
      invalidateValidationState();
      updateWarningsForCurrentInputs();
    });
  }

  if (trainAlgoEl) {
    trainAlgoEl.addEventListener("change", () => {
      markParamsDirty();
      invalidateValidationState();
      updateWarningsForCurrentInputs();
    });
  }

  if (modelNameEl) {
    modelNameEl.addEventListener("input", () => {
      markParamsDirty();
      invalidateValidationState();
      updateWarningsForCurrentInputs();
    });
  }

  if (trainingHoursEl) {
    trainingHoursEl.addEventListener("input", () => {
      markParamsDirty();
      invalidateValidationState();
      updateWarningsForCurrentInputs();
    });
  }

  paramsSyncEl.addEventListener("click", () => {
    const lastRecords = getLastRecords();
    if (!lastRecords) return;
    setParamsLinked(true);
    setParamsDirty(false);
    syncParamsFromData({ records: lastRecords, inferredFrequency: getLastInferredFrequency(), force: true });
    invalidateValidationState();
    updateWarningsForCurrentInputs();
  });

  rollbackDefaultModelBtnEl.addEventListener("click", () => {
    setDefaultModelId(null);
    setStatusI18n(modelsStatusEl, "status.default_model_cleared", null, "success");
    renderModels();
  });

  if (syncModelsBtnEl) {
    syncModelsBtnEl.addEventListener("click", () => void syncModelsFromServer({ silent: false }));
  }
}

export function wireGlobalUiEvents(ctx) {
  const {
    windowObj,
    apiKeyEl,
    csvFileEl,
    jsonInputEl,
    taskEl,
    horizonEl,
    quantilesEl,
    levelEl,
    trainingHoursEl,
    billingSyncMaxPointsEl,
    valueUnitEl,
    helpTopicEl,
    helpGuideBtnEl,
    helpAskBtnEl,
    helpAskEl,
    helpAdviceEl,
    helpActionSampleEl,
    helpActionValidateEl,
    helpActionRunEl,
    helpActionSupportEl,
    helpSupportEl,
    helpCardEl,
    loadSampleBtnEl,
    validateBtnEl,
    runForecastBtnEl,
    toggleApiKeyEl,
    rememberApiKeyEl,
    clearStoredApiKeyBtnEl,
    healthStatusEl,
    updateStepNavHighlight,
    refreshAxisLabels,
    updateHelpAdvice,
    guideHelpFlow,
    matchHelpTopicFromText,
    t,
    storeApiKey,
    clearStoredApiKey,
    setLastHealthOk,
    setStatusI18n,
    updateRunGate,
  } = ctx;

  windowObj.addEventListener("scroll", updateStepNavHighlight, { passive: true });
  windowObj.addEventListener("resize", updateStepNavHighlight);

  const stepStateInputs = [
    apiKeyEl,
    csvFileEl,
    jsonInputEl,
    taskEl,
    horizonEl,
    quantilesEl,
    levelEl,
    trainingHoursEl,
    billingSyncMaxPointsEl,
  ].filter(Boolean);

  for (const node of stepStateInputs) {
    node.addEventListener("input", updateStepNavHighlight);
    node.addEventListener("change", updateStepNavHighlight);
  }

  if (valueUnitEl) {
    valueUnitEl.addEventListener("input", refreshAxisLabels);
    valueUnitEl.addEventListener("change", refreshAxisLabels);
  }

  if (helpTopicEl) {
    helpTopicEl.addEventListener("change", updateHelpAdvice);
  }

  if (helpGuideBtnEl) {
    helpGuideBtnEl.addEventListener("click", guideHelpFlow);
  }

  if (helpAskBtnEl) {
    const handleHelpAsk = () => {
      const match = matchHelpTopicFromText(helpAskEl?.value || "");
      if (match && helpTopicEl) {
        helpTopicEl.value = match;
        guideHelpFlow();
        return;
      }
      if (helpAdviceEl) {
        helpAdviceEl.textContent = t("help.concierge.ask_fallback");
      }
    };
    helpAskBtnEl.addEventListener("click", handleHelpAsk);
    if (helpAskEl) {
      helpAskEl.addEventListener("keydown", (e) => {
        if (e.key !== "Enter") return;
        e.preventDefault();
        handleHelpAsk();
      });
    }
  }

  if (helpActionSampleEl) {
    helpActionSampleEl.addEventListener("click", () => {
      if (loadSampleBtnEl) loadSampleBtnEl.click();
    });
  }

  if (helpActionValidateEl) {
    helpActionValidateEl.addEventListener("click", () => {
      if (validateBtnEl) validateBtnEl.click();
    });
  }

  if (helpActionRunEl) {
    helpActionRunEl.addEventListener("click", () => {
      if (runForecastBtnEl) runForecastBtnEl.click();
    });
  }

  if (helpActionSupportEl) {
    helpActionSupportEl.addEventListener("click", () => {
      if (helpSupportEl) helpSupportEl.open = true;
      if (helpCardEl) helpCardEl.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  apiKeyEl.addEventListener("input", () => {
    setLastHealthOk(false);
    try {
      storeApiKey(apiKeyEl.value, rememberApiKeyEl.checked);
    } catch {
      return;
    }
    if (typeof updateRunGate === "function") updateRunGate();
  });

  toggleApiKeyEl.addEventListener("click", () => {
    const isPassword = apiKeyEl.type === "password";
    apiKeyEl.type = isPassword ? "text" : "password";
    toggleApiKeyEl.textContent = isPassword ? t("action.hide_api_key") : t("action.show_api_key");
  });

  rememberApiKeyEl.addEventListener("change", () => {
    try {
      localStorage.setItem("arcayf_forecasting_api_key_persist", rememberApiKeyEl.checked ? "1" : "0");
      if (!rememberApiKeyEl.checked) {
        storeApiKey(apiKeyEl.value, false);
      } else if (apiKeyEl.value) {
        storeApiKey(apiKeyEl.value, true);
      } else {
        clearStoredApiKey();
      }
    } catch {
      return;
    }
    if (typeof updateRunGate === "function") updateRunGate();
  });

  if (clearStoredApiKeyBtnEl) {
    clearStoredApiKeyBtnEl.addEventListener("click", () => {
      clearStoredApiKey();
      apiKeyEl.value = "";
      rememberApiKeyEl.checked = false;
      try {
        localStorage.setItem("arcayf_forecasting_api_key_persist", "0");
      } catch {
        return;
      }
      setLastHealthOk(false);
      setStatusI18n(healthStatusEl, "status.api_key_cleared", null, "info");
      updateStepNavHighlight();
      if (typeof updateRunGate === "function") updateRunGate();
    });
  }
}

export function wireInitPreferenceEvents(ctx) {
  const {
    densityModeEl,
    langSelectEl,
    setDensityMode,
    updateStepNavHighlight,
    setLanguage,
  } = ctx;

  if (densityModeEl) {
    densityModeEl.addEventListener("change", () => {
      setDensityMode(densityModeEl.value, { persist: true });
      updateStepNavHighlight();
    });
  }

  langSelectEl.addEventListener("change", async () => {
    await setLanguage(langSelectEl.value, { persist: true });
  });
}