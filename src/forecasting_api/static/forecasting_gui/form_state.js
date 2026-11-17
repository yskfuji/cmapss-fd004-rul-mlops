export function normalizeTask(raw) {
  const value = String(raw || "").trim().toLowerCase();
  if (value === "drift") return "drift";
  if (value === "backtest") return "backtest";
  if (value === "train") return "train";
  return "forecast";
}

export function getParamBlockReasonKeys({ task, horizon, quantiles, level } = {}) {
  const reasons = [];
  const normalizedTask = normalizeTask(task);
  if (normalizedTask === "forecast" || normalizedTask === "backtest") {
    const horizonVal = Number(horizon);
    if (!(Number.isFinite(horizonVal) && horizonVal >= 1)) {
      reasons.push("run.block.horizon");
    }
    const q = String(quantiles || "").trim();
    const l = String(level || "").trim();
    if (q && l) {
      reasons.push("run.block.quantiles_level");
    }
  }
  return reasons;
}

export function getRunBlockReasonKeys({
  task,
  connectionReady,
  driftBaselineReady,
  dataSource,
  hasSelectedInput,
  hasOtherInput,
  lastValidatedSignature,
  validationSignature,
  paramsLinked,
  lastDataSignature,
  lastSyncedSignature,
  gapIssue,
  gapAcknowledged,
  horizon,
  quantiles,
  level,
  pointsEstimate,
  pointsLimit,
} = {}) {
  const reasons = [];
  const normalizedTask = normalizeTask(task);

  const selectedOk = !!hasSelectedInput;
  const otherOk = !!hasOtherInput;

  if (!connectionReady) reasons.push("run.block.api_key");
  if (!selectedOk) reasons.push(otherOk ? "run.block.data_source_mismatch" : "run.block.data");

  const outOfSync =
    !paramsLinked &&
    !!String(lastDataSignature || "") &&
    !!String(lastSyncedSignature || "") &&
    String(lastDataSignature) !== String(lastSyncedSignature);
  if (outOfSync) reasons.push("run.block.sync");

  reasons.push(
    ...getParamBlockReasonKeys({
      task: normalizedTask,
      horizon,
      quantiles,
      level,
    }),
  );

  // Drift checks require a persisted baseline, but low-sample baselines only warn instead of blocking.
  if (normalizedTask === "drift" && !driftBaselineReady) reasons.push("run.block.drift_baseline");

  if (gapIssue && !gapAcknowledged) reasons.push("run.block.gaps");
  return reasons;
}

export function isParamsReadyState({ task, dataReady, horizon, quantiles, level } = {}) {
  if (!dataReady) return false;
  const normalizedTask = normalizeTask(task);
  if (normalizedTask === "train" || normalizedTask === "drift") return true;

  const horizonVal = Number(horizon);
  const horizonOk = Number.isFinite(horizonVal) && horizonVal >= 1;
  if (!horizonOk) return false;

  const q = String(quantiles || "").trim();
  const l = String(level || "").trim();
  if (q && l) return false;

  return true;
}

export function computeRunGateState({ isUiBusy, reasonKeys } = {}) {
  const keys = Array.isArray(reasonKeys) ? reasonKeys : [];
  return {
    blocked: !!isUiBusy || keys.length > 0,
    reasonCount: keys.length,
  };
}