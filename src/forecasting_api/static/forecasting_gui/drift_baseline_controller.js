export function createDriftBaselineController({
  apiClient,
  driftBaselineStatusEl,
  getRecordsFromInputs,
  isConnectionReady,
  setStatusI18n,
  showError,
  showWarn,
  updateRunGate,
}) {
  let driftBaselineState = {
    checked: false,
    exists: false,
    sufficientSamples: false,
    featureCount: 0,
    sampleSize: 0,
  };
  let driftBaselineBusy = false;
  let driftBaselineRefreshPromise = null;
  let lastDriftBaselinePersisted = false;

  function setDriftBaselineState(next) {
    driftBaselineState = {
      checked: !!next?.checked,
      exists: !!next?.exists,
      sufficientSamples: !!next?.sufficientSamples,
      featureCount: Number(next?.featureCount) || 0,
      sampleSize: Number(next?.sampleSize) || 0,
    };
  }

  async function refreshStatus({ silent = false } = {}) {
    if (driftBaselineRefreshPromise) {
      if (!silent && driftBaselineStatusEl) {
        setStatusI18n(driftBaselineStatusEl, "status.drift_baseline_checking", null, "info");
      }
      return driftBaselineRefreshPromise;
    }
    driftBaselineRefreshPromise = (async () => {
      if (!isConnectionReady()) {
        setDriftBaselineState({ checked: false, exists: false, sufficientSamples: false, featureCount: 0, sampleSize: 0 });
        updateRunGate();
        return null;
      }
      driftBaselineBusy = true;
      if (!silent && driftBaselineStatusEl) {
        setStatusI18n(driftBaselineStatusEl, "status.drift_baseline_checking", null, "info");
      }
      try {
        const { body } = await apiClient.getDriftBaselineStatus();
        setDriftBaselineState({
          checked: true,
          exists: !!body?.baseline_exists,
          sufficientSamples: !!body?.sufficient_samples,
          featureCount: body?.feature_count,
          sampleSize: body?.sample_size,
        });
        if (driftBaselineStatusEl) {
          if (body?.baseline_exists) {
            setStatusI18n(
              driftBaselineStatusEl,
              "status.drift_baseline_ready",
              {
                feature_count: Number(body?.feature_count) || 0,
                sample_size: Number(body?.sample_size) || 0,
              },
              body?.sufficient_samples ? "success" : "warn",
            );
          } else {
            setStatusI18n(driftBaselineStatusEl, "status.drift_baseline_missing", null, "warn");
          }
        }
        updateRunGate();
        return body;
      } catch (error) {
        setDriftBaselineState({ checked: false, exists: false, sufficientSamples: false, featureCount: 0, sampleSize: 0 });
        if (!silent && driftBaselineStatusEl) {
          setStatusI18n(driftBaselineStatusEl, "status.drift_baseline_check_failed", null, "error");
        }
        updateRunGate();
        if (!silent) showError(error instanceof Error ? error.message : String(error || ""));
        return null;
      } finally {
        driftBaselineBusy = false;
      }
    })();
    try {
      return await driftBaselineRefreshPromise;
    } finally {
      driftBaselineRefreshPromise = null;
    }
  }

  async function onRefresh() {
    showError("");
    await refreshStatus({ silent: false });
  }

  async function onPersist() {
    showError("");
    showWarn("");
    lastDriftBaselinePersisted = false;
    if (!isConnectionReady()) {
      updateRunGate();
      return;
    }
    driftBaselineBusy = true;
    if (driftBaselineStatusEl) {
      setStatusI18n(driftBaselineStatusEl, "status.drift_baseline_saving", null, "info");
    }
    try {
      const records = await getRecordsFromInputs();
      const { body } = await apiClient.persistDriftBaseline({ baseline_records: records });
      lastDriftBaselinePersisted = !!body?.persisted;
      setDriftBaselineState({
        checked: true,
        exists: !!body?.persisted,
        sufficientSamples: (Number(body?.sample_size) || 0) >= 50,
        featureCount: body?.feature_count,
        sampleSize: body?.sample_size,
      });
      if (driftBaselineStatusEl) {
        setStatusI18n(
          driftBaselineStatusEl,
          "status.drift_baseline_saved",
          {
            feature_count: Number(body?.feature_count) || 0,
            sample_size: Number(body?.sample_size) || 0,
          },
          "success",
        );
      }
      updateRunGate();
    } catch (error) {
      if (driftBaselineStatusEl) {
        setStatusI18n(driftBaselineStatusEl, "status.drift_baseline_save_failed", null, "error");
      }
      showError(error instanceof Error ? error.message : String(error || ""));
      updateRunGate();
    } finally {
      driftBaselineBusy = false;
    }
  }

  return {
    getState() {
      return { ...driftBaselineState };
    },
    isBusy() {
      return driftBaselineBusy;
    },
    wasLastPersisted() {
      return lastDriftBaselinePersisted;
    },
    refreshStatus,
    onRefresh,
    onPersist,
  };
}