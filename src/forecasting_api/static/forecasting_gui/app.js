import {
  findSensitiveInObject,
  sanitizeRecordsForSnippet,
  sanitizeShareState,
} from "./share_security.js";
import { decodeStateFromHash, encodeStateToHash } from "./share_state_codec.js";
import {
  getInitialDensityMode as getInitialDensityModeFromPrefs,
  getInitialLang as getInitialLangFromPrefs,
  normalizeDensityMode,
  normalizeLang,
} from "./ui_prefs.js";
import { createApiClient } from "./api_client.js";
import {
  backtestToCsv,
  buildForecastTableRows,
  buildHorizonRows,
  buildBacktestViewModel,
  buildMetricTableRows,
  buildFoldRows,
  buildResultVisibility,
  buildSeriesRankRows,
  countValidSeriesPoints,
  forecastsToCsv,
  normalizeForecasts,
} from "./render_results.js";
import {
  buildSparkGrid,
  createResultsVisualController,
  extractResidualEvidence,
} from "./results_visual_controller.js";
import { createResultsInsightsController } from "./results_insights_controller.js";
import { createResultsTablesController } from "./results_tables_controller.js";
import { createJobsModelsController } from "./jobs_models_controller.js";
import { createBillingHelpController } from "./billing_help_controller.js";
import { createTaskResultsUiController } from "./task_results_ui_controller.js";
import { createProgressGateController } from "./progress_gate_controller.js";
import { createDataQualityController } from "./data_quality_controller.js";
import {
  computeRunGateState,
  getParamBlockReasonKeys,
  getRunBlockReasonKeys,
  isParamsReadyState,
  normalizeTask,
} from "./form_state.js";
import { createDriftBaselineController } from "./drift_baseline_controller.js";
import { createExplorePanelController, normalizeExploreDataset } from "./explore_panel_controller.js";
import { wireCoreEvents, wireGlobalUiEvents, wireInitPreferenceEvents } from "./events_wiring.js";

const LS_API_KEY = "arcayf_forecasting_api_key";
const LS_API_KEY_PERSIST = "arcayf_forecasting_api_key_persist";
const LS_API_KEY_SAVED_AT = "arcayf_forecasting_api_key_saved_at";
const SS_API_KEY = "arcayf_forecasting_api_key_session";
const LS_API_BASE_URL = "arcayf_forecasting_api_base_url";
const SS_API_BASE_URL_ACK = "arcayf_forecasting_api_base_url_ack";
const API_KEY_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const LS_LANG = "arcayf_forecasting_gui_lang";
const LS_JOBS = "arcayf_forecasting_gui_jobs";
const LS_MODELS = "arcayf_forecasting_gui_models";
const LS_DEFAULT_MODEL_ID = "arcayf_forecasting_gui_default_model_id";
const LS_BILLING_SYNC_MAX_POINTS = "arcayf_forecasting_gui_billing_sync_max_points";
const LS_DENSITY_MODE = "arcayf_forecasting_gui_density_mode";

const el = (id) => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element: ${id}`);
  return node;
};

const apiKeyEl = el("apiKey");
const apiBaseUrlEl = el("apiBaseUrl");
const rememberApiKeyEl = el("rememberApiKey");
const clearStoredApiKeyBtnEl = document.getElementById("clearStoredApiKey");
const toggleApiKeyEl = el("toggleApiKey");
const langSelectEl = el("langSelect");
const densityModeEl = document.getElementById("densityMode");
const modeEl = el("mode");
const taskEl = el("task");
const csvFileEl = el("csvFile");
const csvFileNameEl = document.getElementById("csvFileName");
const dataSourceEl = document.getElementById("dataSource");
const dataSourcePickerEl = document.getElementById("dataSourcePicker");
const dataSourceBtnCsvEl = document.getElementById("dataSourceBtnCsv");
const dataSourceBtnJsonEl = document.getElementById("dataSourceBtnJson");
const dataSourceBtnSampleEl = document.getElementById("dataSourceBtnSample");
const dataSourceCsvEl = document.getElementById("dataSourceCsv");
const dataSourceJsonEl = document.getElementById("dataSourceJson");
const dataActionsSampleEl = document.getElementById("dataActionsSample");
const dataActionsConfirmEl = document.getElementById("dataActionsConfirm");
const sampleTypeEl = document.getElementById("sampleType");
const jsonInputEl = el("jsonInput");
const horizonFieldEl = el("horizonField");
const horizonEl = el("horizon");
const frequencyEl = el("frequency");
const missingPolicyEl = el("missingPolicy");
const uncertaintyModeEl = document.getElementById("uncertaintyMode");
const quantilesFieldEl = document.getElementById("quantilesField");
const levelFieldEl = document.getElementById("levelField");
const quantilesEl = el("quantiles");
const levelEl = el("level");
const valueUnitEl = el("valueUnit");
const paramsLinkNoteEl = el("paramsLinkNote");
const paramsSyncEl = el("paramsSync");
const driftBaselinePanelEl = document.getElementById("driftBaselinePanel");
const saveDriftBaselineBtnEl = document.getElementById("saveDriftBaseline");
const refreshDriftBaselineBtnEl = document.getElementById("refreshDriftBaseline");
const driftBaselineStatusEl = document.getElementById("driftBaselineStatus");
const modelIdEl = el("modelId");
const foldsEl = el("folds");
const metricEl = el("metric");
const baseModelEl = el("baseModel");
const trainAlgoEl = document.getElementById("trainAlgo");
const modelNameEl = el("modelName");
const trainingHoursEl = el("trainingHours");
const errorBoxEl = el("errorBox");
const errorActionsEl = el("errorActions");
const copyErrorBtnEl = el("copyError");
const copyStatusEl = el("copyStatus");
const rawJsonEl = el("rawJson");
const metricsWrapEl = el("metricsWrap");
const metricsPrimaryEl = el("metricsPrimary");
const metricsTableBodyEl = el("metricsTable").querySelector("tbody");
const bySeriesWrapEl = el("bySeriesWrap");
const bySeriesTableBodyEl = el("bySeriesTable").querySelector("tbody");
const byHorizonWrapEl = el("byHorizonWrap");
const byHorizonTableBodyEl = el("byHorizonTable").querySelector("tbody");
const byFoldWrapEl = el("byFoldWrap");
const byFoldTableBodyEl = el("byFoldTable").querySelector("tbody");
const driftFeaturesWrapEl = el("driftFeaturesWrap");
const driftFeaturesTableBodyEl = el("driftFeaturesTable").querySelector("tbody");
const forecastTableWrapEl = el("forecastTableWrap");
const resultTableBodyEl = el("resultTable").querySelector("tbody");
const resultsGuideEl = el("resultsGuide");
const resultsLegendEl = el("resultsLegend");
const resultsTableNoteEl = el("resultsTableNote");
const resultsVisualEl = document.getElementById("resultsVisual");
const resultsSparklineEl = document.getElementById("resultsSparkline");
const resultsSparklineSecondaryEl = document.getElementById("resultsSparklineSecondary");
const resultsSparklineTertiaryEl = document.getElementById("resultsSparklineTertiary");
const resultsSecondaryChartBlockEl = document.getElementById("resultsSecondaryChartBlock");
const resultsTertiaryChartBlockEl = document.getElementById("resultsTertiaryChartBlock");
const resultsPrimaryChartTitleEl = document.getElementById("resultsPrimaryChartTitle");
const resultsSecondaryChartTitleEl = document.getElementById("resultsSecondaryChartTitle");
const resultsTertiaryChartTitleEl = document.getElementById("resultsTertiaryChartTitle");
const resultsVisualNoteEl = document.getElementById("resultsVisualNote");
const resultsChartTypeEl = document.getElementById("resultsChartType");
const resultsHighlightsEl = document.getElementById("resultsHighlights");
const resultsHighlightSeriesEl = document.getElementById("resultsHighlightSeries");
const resultsHighlightPointsEl = document.getElementById("resultsHighlightPoints");
const resultsHighlightBandEl = document.getElementById("resultsHighlightBand");
const resultsInterpretationEl = document.getElementById("resultsInterpretation");
const resultsInterpretIntentEl = document.getElementById("resultsInterpretIntent");
const resultsInterpretWhenEl = document.getElementById("resultsInterpretWhen");
const resultsInterpretCautionEl = document.getElementById("resultsInterpretCaution");
const resultsSummaryEl = document.getElementById("resultsSummary");
const resultsSummaryChipsEl = document.getElementById("resultsSummaryChips");
const resultsSummaryNoteEl = document.getElementById("resultsSummaryNote");
const resultsEvidenceEl = document.getElementById("resultsEvidence");
const resultsEvidenceQualityEl = document.getElementById("resultsEvidenceQuality");
const resultsEvidenceUncertaintyEl = document.getElementById("resultsEvidenceUncertainty");
const resultsEvidenceScopeEl = document.getElementById("resultsEvidenceScope");
const resultsEvidenceLimitEl = document.getElementById("resultsEvidenceLimit");
const resultsBenchmarkEl = document.getElementById("resultsBenchmark");
const resultsBenchmarkSummaryEl = document.getElementById("resultsBenchmarkSummary");
const resultsBenchmarkNoteEl = document.getElementById("resultsBenchmarkNote");
const resultsBenchmarkTableBodyEl = document.getElementById("resultsBenchmarkTable")?.querySelector("tbody");
const resultsSeriesEl = document.getElementById("resultsSeries");
const resultsAxisValueEl = document.getElementById("resultsAxisValue");
const resultsAxisTimeEl = document.getElementById("resultsAxisTime");
const resultsAxisYMaxEl = document.getElementById("resultsAxisYMax");
const resultsAxisYMinEl = document.getElementById("resultsAxisYMin");
const resultsAxisXMinEl = document.getElementById("resultsAxisXMin");
const resultsAxisXMaxEl = document.getElementById("resultsAxisXMax");

const resultsAxisValue2El = document.getElementById("resultsAxisValue2");
const resultsAxisTime2El = document.getElementById("resultsAxisTime2");
const resultsAxisYMax2El = document.getElementById("resultsAxisYMax2");
const resultsAxisYMin2El = document.getElementById("resultsAxisYMin2");
const resultsAxisXMin2El = document.getElementById("resultsAxisXMin2");
const resultsAxisXMax2El = document.getElementById("resultsAxisXMax2");
const resultsAxisValue3El = document.getElementById("resultsAxisValue3");
const resultsAxisTime3El = document.getElementById("resultsAxisTime3");
const resultsAxisYMax3El = document.getElementById("resultsAxisYMax3");
const resultsAxisYMin3El = document.getElementById("resultsAxisYMin3");
const resultsAxisXMin3El = document.getElementById("resultsAxisXMin3");
const resultsAxisXMax3El = document.getElementById("resultsAxisXMax3");
const resultsProgressWrapEl = document.getElementById("resultsProgress");
const resultsProgressFillEl = document.getElementById("resultsProgressFill");
const resultsProgressTextEl = document.getElementById("resultsProgressText");
const rawDetailsEl = el("rawDetails");
const dataStatusEl = el("dataStatus");
const runStatusEl = el("runStatus");
const healthStatusEl = el("healthStatus");
const resultsCardEl = document.getElementById("resultsCard");
const resultsEmptyEl = el("resultsEmpty");
const resultsTrainNoteEl = el("resultsTrainNote");
const requestIdEl = el("requestId");
const requestIdWrapEl = el("requestIdWrap");
const copyRequestIdBtnEl = el("copyRequestId");
const statusConnectionApiKeyEl = document.getElementById("statusConnectionApiKey");
const statusConnectionHealthEl = document.getElementById("statusConnectionHealth");
const statusDataInputEl = document.getElementById("statusDataInput");
const statusDataValidationEl = document.getElementById("statusDataValidation");
const checkDataInputEl = el("checkDataInput");
const checkDataRequiredEl = el("checkDataRequired");
const checkDataValidateEl = el("checkDataValidate");
const checkDataStatusEl = el("checkDataStatus");
const statusParamsHorizonEl = document.getElementById("statusParamsHorizon");
const statusParamsQuantilesEl = document.getElementById("statusParamsQuantiles");
const checkParamsHorizonEl = el("checkParamsHorizon");
const checkParamsQuantilesEl = el("checkParamsQuantiles");
const checkParamsStatusEl = el("checkParamsStatus");
const statusResultsAvailableEl = document.getElementById("statusResultsAvailable");
const statusBillingLimitEl = document.getElementById("statusBillingLimit");
const statusBillingEstimateEl = document.getElementById("statusBillingEstimate");
const billingSummaryLimitEl = document.getElementById("billingSummaryLimit");
const billingSummaryEstimateEl = document.getElementById("billingSummaryEstimate");
const billingSummaryRemainingEl = document.getElementById("billingSummaryRemaining");
const quotaProgressBarEl = document.getElementById("quotaProgressBar");
const quotaFlowValueEl = el("quotaFlowValue");
const quotaFlowBarFillEl = el("quotaFlowBarFill");
const quotaFlowAlertEl = el("quotaFlowAlert");
const helpAskEl = el("helpAsk");
const helpAskBtnEl = el("helpAskBtn");
const helpTopicEl = el("helpTopic");
const helpGuideBtnEl = el("helpGuide");
const helpAdviceEl = el("helpAdvice");
const helpInputsEl = document.getElementById("helpInputs");
const helpSupportEl = document.getElementById("helpSupport");
const helpCardEl = document.getElementById("helpCard");
const helpActionSampleEl = el("helpActionSample");
const helpActionValidateEl = el("helpActionValidate");
const helpActionRunEl = el("helpActionRun");
const helpActionSupportEl = el("helpActionSupport");
const nextStepLinkEl = document.getElementById("nextStepLink");
const validateBtnEl = el("validate");
const runForecastBtnEl = el("runForecast");
const cancelRunBtnEl = el("cancelRun");
const clearBtnEl = el("clear");
const loadSampleBtnEl = el("loadSample");
const checkHealthBtnEl = el("checkHealth");
const downloadJsonEl = el("downloadJson");
const downloadCsvEl = el("downloadCsv");
const copyLinkBtnEl = el("copyLink");
const copySnippetBtnEl = el("copySnippet");
const shareStatusEl = el("shareStatus");
const warnBoxEl = el("warnBox");
const dataAnalysisEl = el("dataAnalysis");
const analysisSummaryEl = el("analysisSummary");
const analysisFieldsEl = el("analysisFields");
const analysisTimestampsEl = el("analysisTimestamps");
const analysisGapsEl = el("analysisGaps");
const analysisGapsSampleEl = el("analysisGapsSample");
const analysisExploreEl = el("analysisExplore");
const exploreDatasetEl = el("exploreDataset");
const exploreTemplateEl = el("exploreTemplate");
const exploreChartTypeEl = el("exploreChartType");
const exploreFieldXEl = el("exploreFieldX");
const exploreFieldYEl = el("exploreFieldY");
const exploreStatusEl = el("exploreStatus");
const exploreChartEl = el("exploreChart");
const exploreSummaryTableEl = document.getElementById("exploreSummaryTable");
const dataPreviewEl = document.getElementById("dataPreview");
const dataPreviewMetaEl = document.getElementById("dataPreviewMeta");
const dataPreviewTableEl = document.getElementById("dataPreviewTable");
const dataPreviewExplainEl = document.getElementById("dataPreviewExplain");
const gapsAcknowledgeEl = document.getElementById("gapsAcknowledge");
const ackGapsEl = document.getElementById("ackGaps");
const stepProgressTextEl = document.getElementById("stepProgressText");
const stepProgressBarEl = document.getElementById("stepProgressBar");
const stepProgressWrapEl = stepProgressBarEl ? stepProgressBarEl.parentElement : null;
const stepNavLinks = Array.from(document.querySelectorAll(".stepLink"));
const docsLinkEl = document.querySelector(".headerLink");
const stepTargets = [
  { id: "connectionCard", step: 1 },
  { id: "dataCard", step: 2 },
  { id: "paramsCard", step: 3 },
  { id: "resultsCard", step: 4 },
];

const apiClient = createApiClient({
  getApiKey: () => apiKeyEl?.value || "",
  getApiBaseUrl: () => apiBaseUrlEl?.value || "",
  errorCodeToI18nKey,
  t: (key) => t(key),
});

function normalizeApiBaseUrl(raw) {
  const base = String(raw || "").trim();
  if (!base) return "";
  return base.replace(/\/+$/, "");
}

function isCrossOriginBaseUrl(baseUrl) {
  const base = normalizeApiBaseUrl(baseUrl);
  if (!base) return false;
  try {
    const u = new URL(base, window.location.href);
    return u.origin !== window.location.origin;
  } catch {
    return true;
  }
}

function loadApiBaseUrl() {
  try {
    return normalizeApiBaseUrl(localStorage.getItem(LS_API_BASE_URL) || "");
  } catch {
    return "";
  }
}

function persistApiBaseUrl(value) {
  const base = normalizeApiBaseUrl(value);
  try {
    if (base) localStorage.setItem(LS_API_BASE_URL, base);
    else localStorage.removeItem(LS_API_BASE_URL);
  } catch {
    // ignore
  }
  return base;
}

function ensureRemoteBaseUrlAcknowledged(baseUrl) {
  const base = normalizeApiBaseUrl(baseUrl);
  if (!base) return true;
  if (!isCrossOriginBaseUrl(base)) return true;
  try {
    if (sessionStorage.getItem(SS_API_BASE_URL_ACK) === base) return true;
  } catch {
    // ignore
  }
  const ok = window.confirm(t("confirm.api_base_url_remote", { base_url: base }));
  if (!ok) return false;
  try {
    sessionStorage.setItem(SS_API_BASE_URL_ACK, base);
  } catch {
    // ignore
  }
  return true;
}

function updateApiBaseUrlWarning() {
  const base = normalizeApiBaseUrl(apiBaseUrlEl?.value || "");
  if (!base) return;
  if (isCrossOriginBaseUrl(base)) {
    showWarn(t("warn.api_base_url_remote", { base_url: base }));
  }
}

function renderInputPreviewChart(records) {
  resultsVisualController.renderInputPreviewChart(records);
}

const forecastParamsEl = el("forecastParams");
const backtestParamsEl = el("backtestParams");
const trainParamsEl = el("trainParams");

const jobIdInputEl = el("jobIdInput");
const addJobIdBtnEl = el("addJobId");
const refreshJobsBtnEl = el("refreshJobs");
const jobsStatusEl = el("jobsStatus");
const jobsEmptyEl = el("jobsEmpty");
const jobsTableBodyEl = el("jobsTable").querySelector("tbody");

const defaultModelValueEl = el("defaultModelValue");
const rollbackDefaultModelBtnEl = el("rollbackDefaultModel");
const modelsStatusEl = el("modelsStatus");
const modelsEmptyEl = el("modelsEmpty");
const modelsTableBodyEl = el("modelsTable").querySelector("tbody");
const syncModelsBtnEl = document.getElementById("syncModels");

const billingSyncMaxPointsEl = document.getElementById("billingSyncMaxPoints");
const billingQuotaStatusEl = document.getElementById("billingQuotaStatus");
const billingEstimateEl = document.getElementById("billingEstimate");
const billingUsageStatusEl = document.getElementById("billingUsageStatus");
const billingDetailBodyEl = document.getElementById("billingDetailBody");
const billingWarnEl = el("billingWarn");
const billingErrorEl = el("billingError");
const billingMetricRemainingEl = document.getElementById("billingMetricRemaining");
const billingMetricUsedEl = document.getElementById("billingMetricUsed");
const billingMetricLimitEl = document.getElementById("billingMetricLimit");
const billingDetailUsedBarEl = document.getElementById("billingDetailUsedBar");
const billingDetailRemainingBarEl = document.getElementById("billingDetailRemainingBar");
const billingImpactSeriesBarEl = document.getElementById("billingImpactSeriesBar");
const billingImpactHorizonBarEl = document.getElementById("billingImpactHorizonBar");
const billingImpactSeriesValueEl = document.getElementById("billingImpactSeriesValue");
const billingImpactHorizonValueEl = document.getElementById("billingImpactHorizonValue");

const LARGE_HORIZON_THRESHOLD = 30;
const GAP_SAMPLE_MAX_SERIES = 3;
const GAP_SAMPLE_MAX_GAPS_PER_SERIES = 2;
const JOB_POLL_MAX_ATTEMPTS = 150;
const JOB_POLL_BASE_WAIT_MS = 250;
const JOB_POLL_MAX_WAIT_MS = 1200;

let lastResult = null;
let lastTask = "forecast";
let currentRunAbort = null;
let lastRequestId = null;
let lastHealthOk = false;
let lastAfnoReady = null;
let lastForecasts = [];
let lastForecastSeriesId = null;
let lastRunContext = null;
let fd004BenchmarkCache = null;
let fd004BenchmarkPending = null;
let runCostConfirmedInSession = false;
let runCostConfirmedPoints = 0;

let lastDataStats = null;
let lastDataMissing = null;
let lastRecords = null;
let lastInferredFrequency = null;
let lastDataGapCount = null;
let isUiBusy = false;
let paramsLinked = true;
let paramsDirty = false;
let lastDataSignature = null;
let lastSyncedSignature = null;
let lastValidatedSignature = null;
let isSyncingParams = false;
let runGateExplained = false;
let sampleLoadArmedForAutoChain = false;
let autoForecastAfterTrainRequested = false;
let pendingAutoForecastAfterTrain = false;
const SAMPLE_LOAD_DEFAULT_TASK = "train";
const I18N_CACHE_BUST_VERSION = "20260217-8";
const TRAIN_ALGO_DISABLED_SET = new Set(["cifnocg2"]);
const LEGACY_DATA_SOURCE_RESET_LABELS = new Set([
  "入力方法を選び直す",
  "Change input method",
  "Choose input method again",
]);
let legacyDataSourceResetObserver = null;

let csvPrecheckState = { state: "unknown", message: "" };

function generateSeries({
  series_id,
  start,
  stepMs,
  points,
  base,
  trend = 0,
  season = 0,
  seasonPeriod = 7,
  noise = 0,
  spikeAt = null,
  spike = 0,
}) {
  const out = [];
  const startMs = Date.parse(start);
  const safeStart = Number.isFinite(startMs) ? startMs : Date.now();
  for (let i = 0; i < points; i++) {
    const ts = new Date(safeStart + i * stepMs).toISOString();
    const seasonal = season ? season * Math.sin((i * 2 * Math.PI) / seasonPeriod) : 0;
    const jitter = noise ? noise * Math.sin(i * 1.7) : 0;
    const spikeValue = spikeAt !== null && i === spikeAt ? spike : 0;
    const y = Math.round((base + trend * i + seasonal + jitter + spikeValue) * 10) / 10;
    out.push({ series_id, timestamp: ts, y });
  }
  return out;
}

function generateSeriesWithGaps({
  series_id,
  start,
  stepMs,
  points,
  base,
  trend = 0,
  season = 0,
  seasonPeriod = 7,
  noise = 0,
  gapEvery = 0,
  gapSpan = 1,
}) {
  const raw = generateSeries({ series_id, start, stepMs, points, base, trend, season, seasonPeriod, noise });
  if (!gapEvery || gapEvery <= 0 || gapSpan <= 0) return raw;
  return raw.filter((_, idx) => idx % gapEvery >= gapSpan);
}

function computeDataSignature(records) {
  if (!Array.isArray(records) || records.length === 0) return "";
  let minTs = Infinity;
  let maxTs = -Infinity;
  const series = new Set();
  for (const rec of records) {
    series.add(String(rec?.series_id ?? ""));
    const ts = Date.parse(String(rec?.timestamp ?? ""));
    if (Number.isFinite(ts)) {
      if (ts < minTs) minTs = ts;
      if (ts > maxTs) maxTs = ts;
    }
  }
  return [records.length, series.size, Number.isFinite(minTs) ? minTs : "", Number.isFinite(maxTs) ? maxTs : ""].join("|");
}

function getValidationSignatureFromState() {
  const task = currentTask();
  const dataSignature = lastDataSignature || "";
  const horizon = Number(horizonEl?.value);
  const folds = Number(foldsEl?.value);
  const metric = String(metricEl?.value || "").trim().toLowerCase();
  const frequency = String(frequencyEl?.value || "").trim();
  const missing_policy = String(missingPolicyEl?.value || "").trim();
  const quantiles = String(quantilesEl?.value || "").trim();
  const level = String(levelEl?.value || "").trim();
  const model_id = String(modelIdEl?.value || "").trim();
  const base_model = String(baseModelEl?.value || "").trim();
  const model_name = String(modelNameEl?.value || "").trim();
  const training_hours = String(trainingHoursEl?.value || "").trim();
  return JSON.stringify({
    task,
    dataSignature,
    horizon,
    folds,
    metric,
    frequency,
    missing_policy,
    quantiles,
    level,
    model_id,
    base_model,
    model_name,
    training_hours,
  });
}

function invalidateValidationState() {
  dataQualityController.invalidateValidationState();
}

function updateCsvFileName() {
  if (!csvFileNameEl) return;
  const file = csvFileEl?.files?.[0];
  csvFileNameEl.textContent = file?.name || t("field.csv_none");
}

function inferUncertaintyModeFromInputs() {
  const hasLevel = !!String(levelEl?.value || "").trim();
  const hasQuantiles = !!String(quantilesEl?.value || "").trim();
  if (hasLevel && !hasQuantiles) return "level";
  return "quantiles";
}

function normalizeUncertaintyMode(value) {
  const mode = String(value || "").trim().toLowerCase();
  return mode === "level" ? "level" : "quantiles";
}

function applyUncertaintyModeUi({ applyDefaults = false, clearInactive = false } = {}) {
  if (!uncertaintyModeEl || !quantilesFieldEl || !levelFieldEl) return;
  const mode = normalizeUncertaintyMode(uncertaintyModeEl.value || inferUncertaintyModeFromInputs());
  const showQuantiles = mode === "quantiles";
  setVisible(quantilesFieldEl, showQuantiles);
  setVisible(levelFieldEl, !showQuantiles);

  if (applyDefaults) {
    if (showQuantiles && !String(quantilesEl?.value || "").trim() && quantilesEl) {
      quantilesEl.value = "0.1,0.5,0.9";
    }
    if (!showQuantiles && !String(levelEl?.value || "").trim() && levelEl) {
      levelEl.value = "80,95";
    }
  }

  if (clearInactive) {
    if (showQuantiles && levelEl) levelEl.value = "";
    if (!showQuantiles && quantilesEl) quantilesEl.value = "";
  }

  if (quantilesEl) quantilesEl.disabled = !showQuantiles;
  if (levelEl) levelEl.disabled = showQuantiles;
}

function generateNonlinearSeries({
  series_id,
  start,
  stepMs,
  points,
  base,
  growthRate = 0.02,
  curve = "exp",
  season = 0,
  seasonPeriod = 7,
  noise = 0,
}) {
  const out = [];
  const startMs = Date.parse(start);
  const safeStart = Number.isFinite(startMs) ? startMs : Date.now();
  const max = base * 6;
  for (let i = 0; i < points; i++) {
    const ts = new Date(safeStart + i * stepMs).toISOString();
    const seasonal = season ? season * Math.sin((i * 2 * Math.PI) / seasonPeriod) : 0;
    const jitter = noise ? noise * Math.sin(i * 1.3) : 0;
    let curveValue = base;
    if (curve === "logistic") {
      const k = growthRate * 6;
      const x0 = points * 0.5;
      curveValue = max / (1 + Math.exp(-k * (i - x0)));
    } else {
      curveValue = base * Math.exp(growthRate * i);
    }
    const y = Math.round((curveValue + seasonal + jitter) * 10) / 10;
    out.push({ series_id, timestamp: ts, y });
  }
  return out;
}

function generateRegimeSeries({
  series_id,
  start,
  stepMs,
  points,
  baseA,
  trendA,
  baseB,
  trendB,
  switchAt,
  season = 0,
  seasonPeriod = 7,
  noise = 0,
}) {
  const out = [];
  const startMs = Date.parse(start);
  const safeStart = Number.isFinite(startMs) ? startMs : Date.now();
  for (let i = 0; i < points; i++) {
    const ts = new Date(safeStart + i * stepMs).toISOString();
    const useB = i >= switchAt;
    const base = useB ? baseB : baseA;
    const trend = useB ? trendB : trendA;
    const seasonal = season ? season * Math.sin((i * 2 * Math.PI) / seasonPeriod) : 0;
    const jitter = noise ? noise * Math.sin(i * 1.5) : 0;
    const y = Math.round((base + trend * i + seasonal + jitter) * 10) / 10;
    out.push({ series_id, timestamp: ts, y });
  }
  return out;
}

function withRetailExogenous(records, { promoEvery = 14, promoSpan = 3 } = {}) {
  const arr = Array.isArray(records) ? records : [];
  return arr.map((rec, i) => {
    const dow = i % 7;
    const isWeekend = dow === 5 || dow === 6;
    const promo = promoEvery > 0 ? i % promoEvery < promoSpan : false;
    const temp = 18 + 8 * Math.sin((i * 2 * Math.PI) / 30);
    return {
      ...rec,
      x: {
        is_weekend: isWeekend,
        promo_flag: promo,
        temperature_c: Math.round(temp * 10) / 10,
      },
    };
  });
}

const SAMPLE_LIBRARY = {
  simple: {
    task: "forecast",
    frequency: "1d",
    horizon: 7,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2024-01-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 140,
      base: 120,
      trend: 0.6,
      season: 8,
      seasonPeriod: 7,
      noise: 2,
    }),
  },
  multi_daily: {
    task: "forecast",
    frequency: "1d",
    horizon: 14,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: [
      ...generateSeries({
        series_id: "s1",
        start: "2024-01-01T00:00:00Z",
        stepMs: 24 * 60 * 60 * 1000,
        points: 180,
        base: 200,
        trend: 0.8,
        season: 10,
        seasonPeriod: 7,
        noise: 3,
      }),
      ...generateSeries({
        series_id: "s2",
        start: "2024-01-01T00:00:00Z",
        stepMs: 24 * 60 * 60 * 1000,
        points: 180,
        base: 160,
        trend: 0.4,
        season: 6,
        seasonPeriod: 7,
        noise: 2,
      }),
    ],
  },
  gaps_hourly: {
    task: "forecast",
    frequency: "1h",
    horizon: 12,
    quantiles: "",
    missing_policy: "error",
    records: generateSeriesWithGaps({
      series_id: "s1",
      start: "2020-01-01T00:00:00Z",
      stepMs: 60 * 60 * 1000,
      points: 72,
      base: 5,
      trend: 0.05,
      season: 0.8,
      seasonPeriod: 24,
      noise: 0.2,
      gapEvery: 12,
      gapSpan: 2,
    }),
  },
  seasonal_weekly: {
    task: "forecast",
    frequency: "7d",
    horizon: 6,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2023-07-02T00:00:00Z",
      stepMs: 7 * 24 * 60 * 60 * 1000,
      points: 104,
      base: 500,
      trend: 1.2,
      season: 40,
      seasonPeriod: 4,
      noise: 6,
    }),
  },
  irregular_monthly: {
    task: "forecast",
    frequency: "30d",
    horizon: 3,
    quantiles: "",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2022-01-01T00:00:00Z",
      stepMs: 30 * 24 * 60 * 60 * 1000,
      points: 60,
      base: 900,
      trend: 3,
      season: 25,
      seasonPeriod: 6,
      noise: 5,
    }),
  },
  retail_daily_multi: {
    task: "forecast",
    frequency: "1d",
    horizon: 14,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: [
      ...generateSeries({
        series_id: "store_a",
        start: "2024-03-01T00:00:00Z",
        stepMs: 24 * 60 * 60 * 1000,
        points: 180,
        base: 320,
        trend: 0.9,
        season: 18,
        seasonPeriod: 7,
        noise: 4,
      }),
      ...generateSeries({
        series_id: "store_b",
        start: "2024-03-01T00:00:00Z",
        stepMs: 24 * 60 * 60 * 1000,
        points: 180,
        base: 240,
        trend: 0.6,
        season: 12,
        seasonPeriod: 7,
        noise: 3,
      }),
    ],
  },
  retail_daily_exogenous: {
    task: "forecast",
    frequency: "1d",
    horizon: 14,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: [
      ...withRetailExogenous(
        generateSeries({
          series_id: "store_a",
          start: "2024-03-01T00:00:00Z",
          stepMs: 24 * 60 * 60 * 1000,
          points: 180,
          base: 320,
          trend: 0.9,
          season: 18,
          seasonPeriod: 7,
          noise: 4,
        }),
      ),
      ...withRetailExogenous(
        generateSeries({
          series_id: "store_b",
          start: "2024-03-01T00:00:00Z",
          stepMs: 24 * 60 * 60 * 1000,
          points: 180,
          base: 240,
          trend: 0.6,
          season: 12,
          seasonPeriod: 7,
          noise: 3,
        }),
      ),
    ],
  },
  spike_daily: {
    task: "forecast",
    frequency: "1d",
    horizon: 10,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2024-02-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 160,
      base: 90,
      trend: 0.2,
      season: 4,
      seasonPeriod: 7,
      noise: 2,
      spikeAt: 18,
      spike: 60,
    }),
  },
  long_daily: {
    task: "forecast",
    frequency: "1d",
    horizon: 21,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2023-09-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 365,
      base: 140,
      trend: 0.3,
      season: 12,
      seasonPeriod: 7,
      noise: 2,
    }),
  },
  nonlinear_growth: {
    task: "forecast",
    frequency: "1d",
    horizon: 21,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateNonlinearSeries({
      series_id: "s1",
      start: "2023-04-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 220,
      base: 18,
      growthRate: 0.03,
      curve: "exp",
      season: 6,
      seasonPeriod: 14,
      noise: 1.6,
    }),
  },
  logistic_growth: {
    task: "forecast",
    frequency: "1d",
    horizon: 21,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateNonlinearSeries({
      series_id: "s1",
      start: "2023-04-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 220,
      base: 8,
      growthRate: 0.035,
      curve: "logistic",
      season: 5,
      seasonPeriod: 30,
      noise: 1.4,
    }),
  },
  regime_shift: {
    task: "forecast",
    frequency: "1d",
    horizon: 14,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateRegimeSeries({
      series_id: "s1",
      start: "2023-01-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 240,
      baseA: 120,
      trendA: 0.4,
      baseB: 220,
      trendB: -0.1,
      switchAt: 95,
      season: 10,
      seasonPeriod: 30,
      noise: 2.2,
    }),
  },
  seasonal_yearly: {
    task: "forecast",
    frequency: "1w",
    horizon: 12,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2021-01-03T00:00:00Z",
      stepMs: 7 * 24 * 60 * 60 * 1000,
      points: 208,
      base: 260,
      trend: 0.9,
      season: 40,
      seasonPeriod: 52,
      noise: 8,
    }),
  },
  multi_weekly: {
    task: "forecast",
    frequency: "7d",
    horizon: 8,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: [
      ...generateSeries({
        series_id: "segment_a",
        start: "2023-01-01T00:00:00Z",
        stepMs: 7 * 24 * 60 * 60 * 1000,
        points: 104,
        base: 480,
        trend: 1.4,
        season: 30,
        seasonPeriod: 13,
        noise: 6,
      }),
      ...generateSeries({
        series_id: "segment_b",
        start: "2023-01-01T00:00:00Z",
        stepMs: 7 * 24 * 60 * 60 * 1000,
        points: 104,
        base: 360,
        trend: 1.1,
        season: 24,
        seasonPeriod: 13,
        noise: 5,
      }),
    ],
  },
  linear_long_daily: {
    task: "forecast",
    frequency: "1d",
    horizon: 30,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2022-01-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 520,
      base: 110,
      trend: 0.35,
      season: 10,
      seasonPeriod: 7,
      noise: 2.2,
    }),
  },
  nonlinear_multi: {
    task: "forecast",
    frequency: "1d",
    horizon: 21,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: [
      ...generateNonlinearSeries({
        series_id: "s1",
        start: "2023-03-01T00:00:00Z",
        stepMs: 24 * 60 * 60 * 1000,
        points: 200,
        base: 14,
        growthRate: 0.03,
        curve: "exp",
        season: 6,
        seasonPeriod: 14,
        noise: 1.5,
      }),
      ...generateNonlinearSeries({
        series_id: "s2",
        start: "2023-03-01T00:00:00Z",
        stepMs: 24 * 60 * 60 * 1000,
        points: 200,
        base: 10,
        growthRate: 0.028,
        curve: "logistic",
        season: 5,
        seasonPeriod: 30,
        noise: 1.2,
      }),
    ],
  },
  backtest_short: {
    task: "backtest",
    frequency: "1d",
    horizon: 14,
    quantiles: "0.1,0.5,0.9",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2024-01-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 140,
      base: 60,
      trend: 0.4,
      season: 5,
      seasonPeriod: 7,
      noise: 2,
    }),
  },
  train_short: {
    task: "train",
    frequency: "1d",
    horizon: 7,
    quantiles: "",
    missing_policy: "ignore",
    records: generateSeries({
      series_id: "s1",
      start: "2024-01-01T00:00:00Z",
      stepMs: 24 * 60 * 60 * 1000,
      points: 140,
      base: 20,
      trend: 0.3,
      season: 2,
      seasonPeriod: 7,
      noise: 1,
    }),
  },
  train_short_exogenous: {
    task: "train",
    frequency: "1d",
    horizon: 7,
    quantiles: "",
    missing_policy: "ignore",
    records: withRetailExogenous(
      generateSeries({
        series_id: "s1",
        start: "2024-01-01T00:00:00Z",
        stepMs: 24 * 60 * 60 * 1000,
        points: 140,
        base: 20,
        trend: 0.3,
        season: 2,
        seasonPeriod: 7,
        noise: 1,
      }),
    ),
  },
};

let runState = {
  status: "idle",
  mode: null,
  jobId: null,
  progress: null,
  progressStartAt: null,
  progressStartValue: null,
  lastProgressAt: null,
  lastProgressValue: null,
  etaSeconds: null,
};

function setRunState(status, overrides = {}) {
  runState = {
    status,
    mode: null,
    jobId: null,
    progress: null,
    progressStartAt: null,
    progressStartValue: null,
    lastProgressAt: null,
    lastProgressValue: null,
    etaSeconds: null,
    ...overrides,
  };
}

let I18N = {};
const missingI18nKeys = new Set();

function isApiKeyExpired(savedAt) {
  const ts = Number(savedAt || 0);
  if (!Number.isFinite(ts) || ts <= 0) return true;
  return Date.now() - ts > API_KEY_TTL_MS;
}

function loadStoredApiKey() {
  const persist = localStorage.getItem(LS_API_KEY_PERSIST) === "1";
  if (persist) {
    const savedAt = localStorage.getItem(LS_API_KEY_SAVED_AT);
    if (isApiKeyExpired(savedAt)) {
      localStorage.removeItem(LS_API_KEY);
      localStorage.removeItem(LS_API_KEY_SAVED_AT);
      return "";
    }
    return localStorage.getItem(LS_API_KEY) || "";
  }
  return sessionStorage.getItem(SS_API_KEY) || "";
}

function storeApiKey(value, persist) {
  const v = String(value || "");
  if (persist) {
    localStorage.setItem(LS_API_KEY, v);
    localStorage.setItem(LS_API_KEY_SAVED_AT, String(Date.now()));
    sessionStorage.removeItem(SS_API_KEY);
    return;
  }
  sessionStorage.setItem(SS_API_KEY, v);
  localStorage.removeItem(LS_API_KEY);
  localStorage.removeItem(LS_API_KEY_SAVED_AT);
}

function clearStoredApiKey() {
  localStorage.removeItem(LS_API_KEY);
  localStorage.removeItem(LS_API_KEY_SAVED_AT);
  sessionStorage.removeItem(SS_API_KEY);
}

function isConnectionReady() {
  return String(apiKeyEl?.value || "").trim().length > 0;
}

function isConnectionConfirmed() {
  return isConnectionReady() && lastHealthOk;
}

function isDataReady() {
  return Number(lastDataStats?.n || 0) > 0;
}

function markParamsDirty() {
  if (isSyncingParams) return;
  if (!paramsDirty) paramsDirty = true;
  if (paramsLinked) paramsLinked = false;
  const changed = Boolean(lastDataSignature && lastSyncedSignature && lastDataSignature !== lastSyncedSignature);
  updateParamsLinkUi({ hasData: isDataReady(), changed });
}

function isParamsReady() {
  return isParamsReadyState({
    task: currentTask(),
    dataReady: isDataReady(),
    horizon: horizonEl?.value,
    quantiles: quantilesEl?.value,
    level: levelEl?.value,
  });
}

function isResultsReady() {
  return !!lastResult;
}

function isBillingReady() {
  return isDataReady();
}

function getStepCompletion() {
  return {
    1: isConnectionConfirmed(),
    2: isDataReady(),
    3: isParamsReady(),
    4: isResultsReady(),
    5: isBillingReady(),
  };
}

function stateLabel(key) {
  return t(`status.state.${key}`);
}

function setSectionStatus(node, labelKey, stateKey) {
  if (!node) return;
  const label = t(labelKey);
  node.textContent = `${label}: ${stateLabel(stateKey)}`;
}

function setVisible(node, visible) {
  if (!node) return;
  node.hidden = !visible;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setText(node, text) {
  if (!node) return;
  node.textContent = String(text || "");
}

function setStatus(node, text, tone = null) {
  if (!node) return;
  node.textContent = String(text || "");
  if (tone) {
    node.dataset.tone = String(tone);
  } else if (node.dataset?.tone) {
    delete node.dataset.tone;
  }
}

function setStatusI18n(node, key, vars = null, tone = null) {
  setStatus(node, t(key, vars), tone);
}

function setStatusComposite(node, parts, tone = null) {
  const labels = Array.isArray(parts)
    ? parts.map((part) => (part && part.key ? t(part.key, part.vars || null) : "")).filter(Boolean)
    : [];
  setStatus(node, labels.join(" / "), tone);
}

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

function formatChecklist(labelKey, stateKey) {
  const label = t(labelKey);
  return label;
}

function formatChecklistSummary(done, total) {
  return t("checklist.summary", { done, total });
}

function setResultsProgress(percent, label = "") {
  if (!resultsProgressWrapEl || !resultsProgressFillEl) return;
  const safe = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0;
  resultsProgressFillEl.style.width = `${Math.round(safe)}%`;
  resultsProgressWrapEl.setAttribute("aria-valuenow", String(Math.round(safe)));
  if (resultsProgressTextEl) {
    resultsProgressTextEl.textContent = label || "";
  }
}

function updateResultsStatus() {
  if (runState.status === "running") {
    const parts = [{ key: "status.results.running" }];
    if (runState.jobId) {
      parts.push({ key: "status.results.job_id", vars: { job_id: runState.jobId } });
    }
    if (Number.isFinite(runState.progress)) {
      const p = Math.round(runState.progress);
      parts.push({ key: "status.results.progress", vars: { progress: p } });
      if (Number.isFinite(runState.etaSeconds) && runState.etaSeconds > 0) {
        setResultsProgress(p, t("results.progress_label_eta", { progress: p, eta: formatEta(runState.etaSeconds) }));
      } else {
        setResultsProgress(p, t("results.progress_label", { progress: p }));
      }
    } else {
      setResultsProgress(35, t("results.progress_label_running"));
    }
    if (statusResultsAvailableEl) setStatusComposite(statusResultsAvailableEl, parts, "info");
    return;
  }

  if (lastResult && lastTask === currentTask()) {
    if (statusResultsAvailableEl) setStatusI18n(statusResultsAvailableEl, "status.results.ready", null, "success");
    setResultsProgress(100, t("results.progress_label_done"));
    return;
  }

  if (runState.status === "failed") {
    if (statusResultsAvailableEl) setStatusI18n(statusResultsAvailableEl, "status.results.failed", null, "error");
    setResultsProgress(0, t("results.progress_label_failed"));
    return;
  }
  if (runState.status === "cancelled") {
    if (statusResultsAvailableEl) setStatusI18n(statusResultsAvailableEl, "status.results.cancelled", null, "warn");
    setResultsProgress(0, t("results.progress_label_cancelled"));
    return;
  }
  if (isParamsReady()) {
    if (statusResultsAvailableEl) setStatusI18n(statusResultsAvailableEl, "status.results.ready_to_run", null, "info");
    setResultsProgress(0, t("results.progress_label_ready"));
  } else {
    if (statusResultsAvailableEl) setStatusI18n(statusResultsAvailableEl, "status.results.waiting", null, "info");
    setResultsProgress(0, t("results.progress_label_waiting"));
  }
}

function formatEta(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  const m = Math.floor(s / 60);
  const sec = s % 60;
  if (m <= 0) return t("results.progress_eta_secs", { sec });
  return t("results.progress_eta_min", { min: m, sec });
}

function updateRunProgress(progress) {
  const p = Number(progress);
  if (!Number.isFinite(p)) return;
  const now = Date.now();
  if (runState.progressStartAt === null) {
    runState.progressStartAt = now;
    runState.progressStartValue = p;
  }
  runState.lastProgressAt = now;
  runState.lastProgressValue = p;
  runState.progress = p;

  const startAt = runState.progressStartAt;
  const startVal = Number(runState.progressStartValue);
  if (Number.isFinite(startAt) && Number.isFinite(startVal) && p > startVal) {
    const elapsed = (now - startAt) / 1000;
    const rate = (p - startVal) / elapsed;
    if (rate > 0) {
      const remaining = Math.max(0, 100 - p);
      runState.etaSeconds = remaining / rate;
    }
  }
}

function hasGapIssue() {
  const missingPolicy = (missingPolicyEl.value || "").trim();
  return missingPolicy === "error" && Number(lastDataGapCount) > 0;
}

function selectedTrainAlgoForPreflight() {
  const raw = normalizeTrainAlgoValue(trainAlgoEl?.value);
  if (raw) return raw;
  const base = String(baseModelEl?.value || "").trim().toLowerCase();
  if (base.includes("afno")) return "afnocg2";
  return "";
}

function resolveTrainAlgoOption(raw) {
  const normalized = String(raw || "").trim().toLowerCase();
  if (!normalized) return null;
  const options = trainAlgoEl ? Array.from(trainAlgoEl.options || []) : [];
  for (const option of options) {
    const value = String(option.value || "").trim().toLowerCase();
    const label = String(option.textContent || "").trim();
    if (!value) continue;
    if (value === normalized || label.toLowerCase() === normalized) {
      return { value, label };
    }
  }
  return { value: normalized, label: "" };
}

function normalizeTrainAlgoValue(raw) {
  const resolved = resolveTrainAlgoOption(raw);
  if (!resolved?.value) return "";
  if (TRAIN_ALGO_DISABLED_SET.has(resolved.value)) return "";
  return resolved.value;
}

function getTrainAlgoDisplay(raw) {
  const resolved = resolveTrainAlgoOption(raw);
  if (!resolved?.value || TRAIN_ALGO_DISABLED_SET.has(resolved.value)) return "";
  return resolved.label || "";
}

function normalizeModelCatalogEntry(model, fallback = null) {
  const base = model && typeof model === "object" ? model : {};
  const fallbackEntry = fallback && typeof fallback === "object" ? fallback : {};
  const algo = normalizeTrainAlgoValue(base.algo || fallbackEntry.algo || "") || null;
  const algoDisplay = String(base.algo_display || fallbackEntry.algo_display || getTrainAlgoDisplay(algo) || "").trim() || null;
  return {
    model_id: String(base.model_id || fallbackEntry.model_id || "").trim(),
    created_at: String(base.created_at || fallbackEntry.created_at || nowIso()),
    memo: base.memo == null ? String(fallbackEntry.memo || "") : String(base.memo),
    algo,
    algo_display: algoDisplay,
  };
}

function lookupModelById(modelId) {
  const id = String(modelId || "").trim();
  if (!id) return null;
  return loadModelCatalog().find((model) => String(model?.model_id || "") === id) || null;
}

function isCifRestrictedSelection({ algo = "", baseModel = "" } = {}) {
  const a = String(algo || "").trim().toLowerCase();
  const b = String(baseModel || "").trim().toLowerCase();
  return a === "cifnocg2" || b.includes("cif");
}

function enforceTrainAlgoPolicy() {
  if (!trainAlgoEl) return;
  for (const disabledAlgo of TRAIN_ALGO_DISABLED_SET) {
    const opt = trainAlgoEl.querySelector(`option[value="${disabledAlgo}"]`);
    if (opt) opt.remove();
  }
  trainAlgoEl.value = normalizeTrainAlgoValue(trainAlgoEl.value);
}

function inferTorchContextLenForPreflight(maxLen) {
  const n = Number(maxLen);
  if (!Number.isFinite(n) || n <= 0) return 1;
  if (n <= 3) return 1;
  if (n <= 8) return 3;
  if (n <= 20) return 7;
  return 14;
}

function estimateTorchTrainingSamples(records) {
  const rows = Array.isArray(records) ? records : [];
  const counts = new Map();
  for (const r of rows) {
    const sid = String(r?.series_id ?? "").trim();
    if (!sid) continue;
    counts.set(sid, (counts.get(sid) || 0) + 1);
  }
  const lengths = [...counts.values()];
  const maxLen = lengths.length ? Math.max(...lengths) : 0;
  const contextLen = inferTorchContextLenForPreflight(maxLen);
  const samples = lengths.reduce((sum, n) => sum + Math.max(0, n - contextLen), 0);
  return { samples, contextLen, seriesCount: lengths.length, maxLen };
}

function getParamBlockReasons() {
  const reasonKeys = getParamBlockReasonKeys({
    task: currentTask(),
    horizon: horizonEl?.value,
    quantiles: quantilesEl?.value,
    level: levelEl?.value,
  });
  return reasonKeys.map((k) => t(k));
}

function getRunBlockReasonKeysFromState() {
  const src = currentDataSource();
  const hasJsonInput = !!(jsonInputEl.value || "").trim();
  const hasCsvInput = !!csvFileEl.files?.[0];
  const driftBaselineState = currentDriftBaselineState();
  const hasSelectedInput = hasDataForSource(src);
  const hasOtherInput =
    src === "csv" ? hasJsonInput : src === "json" ? hasCsvInput : src === "sample" ? hasCsvInput : hasJsonInput || hasCsvInput;
  const estimate = getRunPointsEstimate(currentTask());
  const reasonKeys = getRunBlockReasonKeys({
    task: currentTask(),
    connectionReady: isConnectionReady(),
    driftBaselineReady:
      currentTask() !== "drift" || (driftBaselineState.checked && driftBaselineState.exists),
    dataSource: src,
    hasSelectedInput,
    hasOtherInput,
    explicitModelId: modelIdEl?.value,
    defaultModelId: getDefaultModelId(),
    lastValidatedSignature,
    validationSignature: getValidationSignatureFromState(),
    paramsLinked,
    lastDataSignature,
    lastSyncedSignature,
    gapIssue: hasGapIssue(),
    gapAcknowledged: !!ackGapsEl?.checked,
    horizon: horizonEl?.value,
    quantiles: quantilesEl?.value,
    level: levelEl?.value,
    pointsEstimate: estimate.pointsEstimate,
    pointsLimit: estimate.limit,
  });

  const task = currentTask();
  const algo = selectedTrainAlgoForPreflight();
  if (task === "train" && isCifRestrictedSelection({ algo: trainAlgoEl?.value, baseModel: baseModelEl?.value })) {
    reasonKeys.push("run.block.cif_disabled");
  }
  if (Number(lastDataMissing?.series_id) > 0 || Number(lastDataMissing?.timestamp) > 0 || Number(lastDataMissing?.y) > 0) {
    reasonKeys.push("run.block.required_fields");
  }
  if (task === "train" && algo === "afnocg2") {
    const estimateTorch = estimateTorchTrainingSamples(lastRecords);
    if (estimateTorch.samples < 8) {
      reasonKeys.push("run.block.torch_data_insufficient");
    }
  }

  return reasonKeys;
}

function getRunBlockReasons() {
  const reasonKeys = getRunBlockReasonKeysFromState();
  const reasons = reasonKeys.map((k) => t(k));
  if (Number(lastDataMissing?.series_id) > 0 || Number(lastDataMissing?.timestamp) > 0 || Number(lastDataMissing?.y) > 0) {
    reasons.push(
      t("run.block.required_fields_detail", {
        series_id: Number(lastDataMissing?.series_id) || 0,
        timestamp: Number(lastDataMissing?.timestamp) || 0,
        y: Number(lastDataMissing?.y) || 0,
      }),
    );
  }
  return reasons;
}

const driftBaselineController = createDriftBaselineController({
  apiClient,
  driftBaselineStatusEl,
  getRecordsFromInputs,
  isConnectionReady,
  setStatusI18n,
  showError,
  showWarn,
  updateRunGate,
});

function getResidualEvidenceFromResult() {
  return extractResidualEvidence(lastResult);
}

function renderResultsEvidence(task, payload = {}) {
  resultsInsightsController.renderEvidence(task, payload);
}

const explorePanelController = createExplorePanelController({
  elements: {
    analysisExploreEl,
    exploreChartEl,
    exploreChartTypeEl,
    exploreDatasetEl,
    exploreFieldXEl,
    exploreFieldYEl,
    exploreStatusEl,
    exploreSummaryTableEl,
    exploreTemplateEl,
  },
  getLastForecasts: () => lastForecasts,
  getLastRecords: () => lastRecords,
  getResidualEvidenceFromResult,
  isDataConfirmed,
  setFieldDisabled,
  setStatus,
  setVisible,
  t,
  valueUnitEl,
  buildSparkGrid,
});

const resultsVisualController = createResultsVisualController({
  elements: {
    resultsVisualEl,
    resultsEmptyEl,
    resultsVisualNoteEl,
    resultsSparklineEl,
    resultsSparklineSecondaryEl,
    resultsSparklineTertiaryEl,
    resultsSecondaryChartBlockEl,
    resultsTertiaryChartBlockEl,
    resultsPrimaryChartTitleEl,
    resultsSecondaryChartTitleEl,
    resultsTertiaryChartTitleEl,
    resultsInterpretationEl,
    resultsInterpretIntentEl,
    resultsInterpretWhenEl,
    resultsInterpretCautionEl,
    resultsSummaryNoteEl,
    resultsHighlightsEl,
    resultsHighlightSeriesEl,
    resultsHighlightPointsEl,
    resultsHighlightBandEl,
    resultsSeriesEl,
    resultsAxisValueEl,
    resultsAxisTimeEl,
    resultsAxisYMaxEl,
    resultsAxisYMinEl,
    resultsAxisXMinEl,
    resultsAxisXMaxEl,
    resultsAxisValue2El,
    resultsAxisTime2El,
    resultsAxisYMax2El,
    resultsAxisYMin2El,
    resultsAxisXMin2El,
    resultsAxisXMax2El,
    resultsAxisValue3El,
    resultsAxisTime3El,
    resultsAxisYMax3El,
    resultsAxisYMin3El,
    resultsAxisXMin3El,
    resultsAxisXMax3El,
  },
  getChartType: () => String(resultsChartTypeEl?.value || "trend"),
  getFrequencyValue: () => String(frequencyEl?.value || ""),
  getHorizonValue: () => horizonEl?.value,
  getLastRecords: () => lastRecords,
  getLastRunContext: () => lastRunContext,
  getValueUnit: () => String(valueUnitEl?.value || ""),
  countValidSeriesPoints,
  renderResultsEvidence,
  setVisible,
  t,
});

const resultsInsightsController = createResultsInsightsController({
  apiClient,
  elements: {
    resultsSummaryEl,
    resultsSummaryChipsEl,
    resultsSummaryNoteEl,
    resultsEvidenceEl,
    resultsEvidenceQualityEl,
    resultsEvidenceUncertaintyEl,
    resultsEvidenceScopeEl,
    resultsEvidenceLimitEl,
    resultsBenchmarkEl,
    resultsBenchmarkSummaryEl,
    resultsBenchmarkNoteEl,
    resultsBenchmarkTableBodyEl,
  },
  countValidSeriesPoints,
  getBenchmarkState: () => ({ cache: fd004BenchmarkCache, pending: fd004BenchmarkPending }),
  getLastRecords: () => lastRecords,
  getLastResult: () => lastResult,
  getLastRunContext: () => lastRunContext,
  getLastTask: () => lastTask,
  getLastForecastSeriesId: () => lastForecastSeriesId,
  normalizeTask,
  setBenchmarkState: ({ cache, pending }) => {
    fd004BenchmarkCache = cache;
    fd004BenchmarkPending = pending;
  },
  setVisible,
  t,
});

const resultsTablesController = createResultsTablesController({
  elements: {
    metricsPrimaryEl,
    metricsTableBodyEl,
    bySeriesTableBodyEl,
    byHorizonTableBodyEl,
    byFoldTableBodyEl,
    driftFeaturesTableBodyEl,
    resultTableBodyEl,
  },
  setText,
  t,
});

const jobsModelsController = createJobsModelsController({
  elements: {
    jobsEmptyEl,
    jobsTableBodyEl,
    modelsEmptyEl,
    modelsTableBodyEl,
  },
  loadJobHistory,
  loadModelCatalog,
  getDefaultModelId,
  formatLocalTime,
  onCheckJob: async (jobId) => {
    try {
      setStatus(jobsStatusEl, "");
      await refreshOneJob(jobId);
    } catch (error) {
      showJobsError(error);
    }
  },
  onDeleteModel: (modelId) => {
    removeModelCatalogEntry(modelId);
    renderModels();
  },
  onFetchJobResult: async (jobId) => {
    try {
      setStatus(jobsStatusEl, "");
      await fetchOneJobResult(jobId);
    } catch (error) {
      showJobsError(error);
    }
  },
  onRemoveJob: (jobId) => {
    removeJobHistoryEntry(jobId);
    renderJobHistory();
  },
  onSetDefaultModel: (modelId) => {
    setDefaultModelId(modelId);
    renderModels();
  },
  onUpdateModelMemo: upsertModelCatalogEntry,
  onUseModel: (model) => {
    taskEl.value = "forecast";
    updateTaskUi();
    modelIdEl.value = String(model?.model_id || "");
    updateRunGate();
    setStatus(modelsStatusEl, String(model?.model_id || ""));
  },
  setVisible,
  t,
});

const billingHelpController = createBillingHelpController({
  elements: {
    billingDetailBodyEl,
    billingDetailRemainingBarEl,
    billingDetailUsedBarEl,
    billingErrorEl,
    billingEstimateEl,
    billingImpactHorizonBarEl,
    billingImpactHorizonValueEl,
    billingImpactSeriesBarEl,
    billingImpactSeriesValueEl,
    billingMetricLimitEl,
    billingMetricRemainingEl,
    billingMetricUsedEl,
    billingQuotaStatusEl,
    billingSummaryEstimateEl,
    billingSummaryLimitEl,
    billingSummaryRemainingEl,
    billingSyncMaxPointsEl,
    billingUsageStatusEl,
    billingWarnEl,
    helpAdviceEl,
    helpInputsEl,
    helpSupportEl,
    helpTopicEl,
    quotaFlowAlertEl,
    quotaFlowBarFillEl,
    quotaFlowValueEl,
    quotaProgressBarEl,
  },
  getRunPointsEstimate,
  refreshI18nStatuses,
  setStatusI18n,
  setText,
  setVisible,
  t,
});

const taskResultsUiController = createTaskResultsUiController({
  elements: {
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
  },
  applyUncertaintyModeUi,
  currentTask,
  getLastForecasts: () => lastForecasts,
  getLastResult: () => lastResult,
  inferUncertaintyModeFromInputs,
  isConnectionReady,
  isDriftBaselineBusy,
  normalizeUncertaintyMode,
  onIdle: () => {
    updateRunGate();
  },
  onRefreshDriftBaselineStatus: refreshDriftBaselineStatus,
  setAriaDisabled,
  setVisible,
  t,
  updateBillingUi,
});

const progressGateController = createProgressGateController({
  elements: {
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
  },
  computeRunGateState,
  currentTask,
  currentDataSource,
  getCurrentDriftBaselineState: currentDriftBaselineState,
  getLastDataGapCount: () => lastDataGapCount,
  getLastDataMissing: () => lastDataMissing,
  getLastDataStats: () => lastDataStats,
  getLastHealthOk: () => lastHealthOk,
  getRunBlockReasonKeysFromState,
  getRunBlockReasons,
  getRunGateExplained: () => runGateExplained,
  getRunState: () => runState,
  getStepCompletion,
  hasDataForSource,
  isConnectionReady,
  isDataReady,
  isDriftBaselineBusy,
  onRefreshDriftBaselineStatus: refreshDriftBaselineStatus,
  setAriaDisabled,
  setSectionStatus,
  setStatus,
  t,
  updateResultsStatus,
});

const dataQualityController = createDataQualityController({
  constants: {
    GAP_SAMPLE_MAX_GAPS_PER_SERIES,
    GAP_SAMPLE_MAX_SERIES,
    LARGE_HORIZON_THRESHOLD,
  },
  elements: {
    analysisExploreEl,
    analysisFieldsEl,
    analysisGapsEl,
    analysisGapsSampleEl,
    analysisSummaryEl,
    analysisTimestampsEl,
    copyErrorBtnEl,
    copyStatusEl,
    dataAnalysisEl,
    dataPreviewEl,
    dataPreviewExplainEl,
    dataPreviewMetaEl,
    dataPreviewTableEl,
    errorActionsEl,
    errorBoxEl,
    exploreChartEl,
    explorePanelController,
    exploreStatusEl,
    fieldElements: {
      ackGapsEl,
      apiKeyEl,
      billingSyncMaxPointsEl,
      csvFileEl,
      frequencyEl,
      horizonEl,
      jobIdInputEl,
      jsonInputEl,
      levelEl,
      missingPolicyEl,
      modelIdEl,
      quantilesEl,
      trainAlgoEl,
    },
    gapsAcknowledgeEl,
    requestIdEl,
    requestIdWrapEl,
    warnBoxEl,
  },
  getCurrentDataSource: currentDataSource,
  getCurrentTask: currentTask,
  getFrequencyValue: () => String(frequencyEl.value || "").trim(),
  getHorizonValue: () => horizonEl?.value,
  getLastValidatedSignature: () => lastValidatedSignature,
  getMissingPolicyValue: () => String(missingPolicyEl.value || "").trim(),
  parseErrorText,
  formatErrorForDisplay,
  countMissingRequiredFields,
  parseFrequencyToMs,
  computeGapDetailsPerSeries,
  setLastDataGapCount: (value) => {
    lastDataGapCount = value;
  },
  setLastDataMissing: (value) => {
    lastDataMissing = value;
  },
  setLastDataStats: (value) => {
    lastDataStats = value;
  },
  setLastValidatedSignature: (value) => {
    lastValidatedSignature = value;
  },
  setRunGateExplained: (value) => {
    runGateExplained = !!value;
  },
  setStatus,
  setText,
  setVisible,
  t,
  updateBillingUi,
  updateRunGate,
  updateSectionStatus,
  updateStepNavHighlight,
});

function currentDriftBaselineState() {
  return driftBaselineController.getState();
}

function isDriftBaselineBusy() {
  return driftBaselineController.isBusy();
}

function refreshDriftBaselineStatus(options = {}) {
  return driftBaselineController.refreshStatus(options);
}

function onRefreshDriftBaseline() {
  return driftBaselineController.onRefresh();
}

function onPersistDriftBaseline() {
  return driftBaselineController.onPersist();
}

function updateRunGate() {
  progressGateController.updateRunGate({ isUiBusy });
}

function updateSectionStatus() {
  progressGateController.updateSectionStatus();
}

function updateNextStep() {
  progressGateController.updateNextStep();
}

function refreshI18nStatuses() {
  progressGateController.refreshI18nStatuses({ isUiBusy });
}

function updateHelpAdvice() {
  billingHelpController.updateHelpAdvice();
}

function guideHelpFlow() {
  billingHelpController.guideHelpFlow();
}

function matchHelpTopicFromText(text) {
  return billingHelpController.matchHelpTopicFromText(text);
}

function getInitialLang() {
  return getInitialLangFromPrefs({ storageKey: LS_LANG });
}

function getInitialDensityMode() {
  return getInitialDensityModeFromPrefs({ storageKey: LS_DENSITY_MODE });
}

function setDensityMode(mode, { persist } = { persist: true }) {
  const normalized = normalizeDensityMode(mode) || "basic";
  if (densityModeEl) densityModeEl.value = normalized;
  document.body.setAttribute("data-density-mode", normalized);
  if (!persist) return;
  try {
    localStorage.setItem(LS_DENSITY_MODE, normalized);
  } catch {
    // ignore
  }
}

function t(key, vars = null) {
  let raw = I18N?.[key];
  if (typeof raw !== "string") {
    if (!missingI18nKeys.has(key)) {
      missingI18nKeys.add(key);
      console.warn(`[i18n] missing key: ${key}`);
    }
    raw = `${key} ${I18N?.["i18n.missing"] || "(untranslated)"}`;
  }
  if (!vars) return raw;
  return String(raw).replaceAll(/\{([a-zA-Z0-9_]+)\}/g, (_m, name) => {
    const v = vars[name];
    return v === undefined || v === null ? "" : String(v);
  });
}

function applyI18n() {
  if (I18N && typeof I18N["ui.page_title"] === "string") {
    document.title = t("ui.page_title");
  }

  for (const node of document.querySelectorAll("[data-i18n]")) {
    const key = node.getAttribute("data-i18n");
    if (!key) continue;
    node.textContent = t(key);
  }

  for (const node of document.querySelectorAll("[data-i18n-placeholder]")) {
    const key = node.getAttribute("data-i18n-placeholder");
    if (!key) continue;
    if (typeof node.placeholder === "string") {
      node.placeholder = t(key);
    }
  }

  for (const node of document.querySelectorAll("[data-i18n-aria-label]")) {
    const key = node.getAttribute("data-i18n-aria-label");
    if (!key) continue;
    node.setAttribute("aria-label", t(key));
  }

  for (const node of document.querySelectorAll("[data-i18n-title]")) {
    const key = node.getAttribute("data-i18n-title");
    if (!key) continue;
    node.setAttribute("title", t(key));
  }

  resultsVisualController.refreshAxisLabels();
  refreshI18nStatuses();
  updateCsvFileName();
}

function nowIso() {
  return new Date().toISOString();
}

function formatLocalTime(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (!Number.isFinite(date.getTime())) return String(value || "-");
  try {
    return new Intl.DateTimeFormat(document.documentElement.lang || undefined, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(date);
  } catch {
    return date.toLocaleString();
  }
}

function loadStoredJsonArray(key) {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveStoredJsonArray(key, items) {
  try {
    localStorage.setItem(key, JSON.stringify(Array.isArray(items) ? items : []));
  } catch {
    // ignore
  }
}

function sortEntriesNewestFirst(items, timeKeys = ["updated_at", "created_at"]) {
  const arr = Array.isArray(items) ? [...items] : [];
  arr.sort((left, right) => {
    const lhs = timeKeys
      .map((key) => Date.parse(String(left?.[key] || "")))
      .find((ts) => Number.isFinite(ts)) ?? -Infinity;
    const rhs = timeKeys
      .map((key) => Date.parse(String(right?.[key] || "")))
      .find((ts) => Number.isFinite(ts)) ?? -Infinity;
    if (rhs !== lhs) return rhs - lhs;
    return String(right?.model_id || right?.job_id || "").localeCompare(String(left?.model_id || left?.job_id || ""));
  });
  return arr;
}

function loadJobHistory() {
  return sortEntriesNewestFirst(
    loadStoredJsonArray(LS_JOBS)
      .filter((job) => job && typeof job === "object")
      .map((job) => ({
        job_id: String(job.job_id || "").trim(),
        status: job.status == null ? null : String(job.status),
        progress: Number.isFinite(Number(job.progress)) ? Number(job.progress) : null,
        error_code: job.error_code == null ? null : String(job.error_code),
        request_id: job.request_id == null ? null : String(job.request_id),
        message: job.message == null ? null : String(job.message),
        created_at: String(job.created_at || job.updated_at || nowIso()),
        updated_at: String(job.updated_at || job.created_at || nowIso()),
      }))
      .filter((job) => job.job_id),
  );
}

function saveJobHistory(jobs) {
  saveStoredJsonArray(LS_JOBS, sortEntriesNewestFirst(jobs));
}

function upsertJobHistoryEntry(job) {
  const jobId = String(job?.job_id || "").trim();
  if (!jobId) return;
  const current = loadJobHistory();
  const existing = current.find((entry) => String(entry?.job_id || "") === jobId) || null;
  const next = current.filter((entry) => String(entry?.job_id || "") !== jobId);
  next.push({
    ...(existing || {}),
    ...(job && typeof job === "object" ? job : {}),
    job_id: jobId,
    created_at: existing?.created_at || String(job?.created_at || nowIso()),
    updated_at: String(job?.updated_at || nowIso()),
  });
  saveJobHistory(next);
}

function removeJobHistoryEntry(jobId) {
  const id = String(jobId || "").trim();
  if (!id) return;
  const jobs = loadJobHistory().filter((job) => String(job?.job_id || "") !== id);
  saveJobHistory(jobs);
}

function loadModelCatalog() {
  return sortEntriesNewestFirst(
    loadStoredJsonArray(LS_MODELS)
      .filter((model) => model && typeof model === "object")
      .map((model) => normalizeModelCatalogEntry(model))
      .filter((model) => model.model_id),
    ["created_at"],
  );
}

function saveModelCatalog(models) {
  saveStoredJsonArray(
    LS_MODELS,
    sortEntriesNewestFirst(
      (Array.isArray(models) ? models : [])
        .map((model) => normalizeModelCatalogEntry(model))
        .filter((model) => model.model_id),
      ["created_at"],
    ),
  );
}

function getDefaultModelId() {
  try {
    return String(localStorage.getItem(LS_DEFAULT_MODEL_ID) || "").trim() || null;
  } catch {
    return null;
  }
}

function setDefaultModelId(modelId) {
  const normalized = String(modelId || "").trim();
  try {
    if (normalized) localStorage.setItem(LS_DEFAULT_MODEL_ID, normalized);
    else localStorage.removeItem(LS_DEFAULT_MODEL_ID);
  } catch {
    // ignore
  }
  if (defaultModelValueEl) {
    defaultModelValueEl.textContent = normalized || t("models.default_none");
  }
  if (rollbackDefaultModelBtnEl) {
    rollbackDefaultModelBtnEl.disabled = !normalized;
    rollbackDefaultModelBtnEl.setAttribute("aria-disabled", normalized ? "false" : "true");
  }
}

function upsertModelCatalogEntry(model) {
  const modelId = String(model?.model_id || "").trim();
  if (!modelId) return;
  const current = loadModelCatalog();
  const existing = current.find((entry) => String(entry?.model_id || "") === modelId) || null;
  const next = current.filter((entry) => String(entry?.model_id || "") !== modelId);
  next.push(normalizeModelCatalogEntry({ ...(existing || {}), ...(model || {}) }, existing));
  saveModelCatalog(next);
}

function removeModelCatalogEntry(modelId) {
  const id = String(modelId || "").trim();
  if (!id) return;
  const next = loadModelCatalog().filter((model) => String(model?.model_id || "") !== id);
  saveModelCatalog(next);
  if (getDefaultModelId() === id) {
    setDefaultModelId(null);
  }
}

function renderModels() {
  const defaultModelId = getDefaultModelId();
  setDefaultModelId(defaultModelId);
  jobsModelsController.renderModels();
}

async function syncModelsFromServer({ silent = false } = {}) {
  if (!String(apiKeyEl?.value || "").trim()) {
    if (!silent) setStatusI18n(modelsStatusEl, "status.models_need_key", null, "warn");
    return [];
  }
  if (!silent) setStatusI18n(modelsStatusEl, "status.models_syncing", null, "info");
  try {
    const { body } = await apiClient.listModels();
    const remoteModels = Array.isArray(body?.models) ? body.models : [];
    const localById = new Map(loadModelCatalog().map((model) => [String(model.model_id || ""), model]));
    const merged = remoteModels
      .filter((model) => model && typeof model === "object")
      .map((model) => {
        const modelId = String(model.model_id || "").trim();
        const local = localById.get(modelId) || null;
        return normalizeModelCatalogEntry({
          model_id: modelId,
          created_at: String(model.created_at || local?.created_at || nowIso()),
          memo: String(local?.memo || model.memo || ""),
          algo: local?.algo || null,
          algo_display: local?.algo_display || null,
        }, local);
      })
      .filter((model) => model.model_id);
    saveModelCatalog(merged);
    renderModels();
    if (!silent) setStatusI18n(modelsStatusEl, "status.models_synced", { count: merged.length }, "success");
    return merged;
  } catch (err) {
    if (!silent) {
      setStatusI18n(modelsStatusEl, "status.models_sync_failed", null, "error");
      showJobsError(err);
    }
    throw err;
  }
}

function updateStepNavHighlight() {
  if (!stepNavLinks.length) return;
  const offset = 140;
  const scrollY = window.scrollY + offset;
  let activeStep = stepTargets[0]?.step || 1;

  for (const target of stepTargets) {
    const node = document.getElementById(target.id);
    if (!node) continue;
    if (node.offsetTop <= scrollY) {
      activeStep = target.step;
    }
  }

  for (const link of stepNavLinks) {
    const step = Number(link.getAttribute("data-step"));
    const active = step === activeStep;
    link.classList.toggle("is-active", active);
    if (active) link.setAttribute("aria-current", "step");
    else link.removeAttribute("aria-current");
  }

  if (stepProgressTextEl) {
    stepProgressTextEl.textContent = `${activeStep}/${stepTargets.length}`;
  }
  if (stepProgressBarEl) {
    stepProgressBarEl.value = activeStep;
    stepProgressBarEl.max = stepTargets.length;
  }
  if (stepProgressWrapEl) {
    stepProgressWrapEl.setAttribute("aria-valuenow", String(activeStep));
    stepProgressWrapEl.setAttribute("aria-valuemax", String(stepTargets.length));
  }
}

function renderJobHistory() {
  jobsModelsController.renderJobHistory({ translations: I18N });
}

async function refreshOneJob(jobId) {
  const id = String(jobId || "").trim();
  if (!id) return;

  const { body: stBody, requestId: stRequestIdHeader } = await apiClient.getJob(id);

  const status = stBody?.status ?? null;
  const progress = stBody?.progress ?? null;
  const err = stBody?.error ?? null;

  upsertJobHistoryEntry({
    job_id: id,
    status,
    progress,
    error_code: err?.error_code ?? null,
    request_id: err?.request_id ?? stRequestIdHeader ?? null,
    message: err?.message ?? null,
  });
  renderJobHistory();
}

async function refreshAllJobs() {
  const jobs = loadJobHistory();
  if (jobs.length === 0) {
    setStatus(jobsStatusEl, "");
    renderJobHistory();
    return;
  }
    setStatusI18n(jobsStatusEl, "status.jobs_refreshing", null, "info");
  try {
    for (const job of jobs) {
      const id = String(job?.job_id || "").trim();
      if (!id) continue;
      try {
        await refreshOneJob(id);
      } catch (e) {
        upsertJobHistoryEntry({ job_id: id, status: job?.status ?? null, message: String(e?.message || e) });
      }
    }
    setStatusI18n(jobsStatusEl, "status.jobs_refreshed", null, "success");
  } finally {
    renderJobHistory();
  }
}

async function fetchOneJobResult(jobId) {
  const id = String(jobId || "").trim();
  if (!id) return;
  const { body: resBody } = await apiClient.getJobResult(id);
  const inferredTask =
    resBody && typeof resBody === "object" && typeof resBody.model_id === "string"
      ? "train"
      : resBody && typeof resBody === "object" && resBody.metrics
        ? "backtest"
        : "forecast";
  renderResult(inferredTask, resBody);
}

async function copyTextToClipboard(text) {
  const value = String(text || "");
  if (!value) return;

  const navClipboard = navigator?.clipboard;
  if (navClipboard && typeof navClipboard.writeText === "function") {
    await navClipboard.writeText(value);
    return;
  }

  const ta = document.createElement("textarea");
  ta.value = value;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  ta.style.top = "0";
  document.body.appendChild(ta);
  ta.select();
  ta.setSelectionRange(0, ta.value.length);
  const ok = document.execCommand("copy");
  ta.remove();
  if (!ok) throw new Error("copy failed");
}

function showError(text) {
  dataQualityController.showError(text);
}

function setShareStatus(text, kind = null) {
  setStatus(shareStatusEl, text, kind);
}

function showWarn(text) {
  dataQualityController.showWarn(text);
}

function clearFieldHighlights() {
  dataQualityController.clearFieldHighlights();
}

function markFieldError(elm) {
  if (!elm) return;
  const field = elm.closest(".field");
  if (field) field.classList.add("error");
}

function applyFieldHighlightsFromErrorText(text) {
  clearFieldHighlights();
  const parsed = parseErrorText(text);
  const code = String(parsed?.code || "").toUpperCase();
  if (!code) return;

  let firstField = null;
  const mark = (elm) => {
    if (!elm) return;
    const field = elm.closest(".field");
    if (field) {
      field.classList.add("error");
      if (!firstField) firstField = field;
    }
  };

  if (code === "A12") {
    mark(apiKeyEl);
  }
  if (code === "A21") {
    mark(billingSyncMaxPointsEl);
  }
  if (code === "V01") {
    mark(csvFileEl);
    mark(jsonInputEl);
  }
  if (code === "V02") {
    mark(frequencyEl);
  }
  if (code === "V03") {
    mark(quantilesEl);
    mark(levelEl);
  }
  if (code === "V04") {
    mark(csvFileEl);
    mark(jsonInputEl);
  }
  if (code === "V05") {
    mark(missingPolicyEl);
  }
  if (code === "V06") {
    mark(csvFileEl);
    mark(jsonInputEl);
    mark(horizonEl);
  }
  if (code === "M01" || code === "M02") {
    mark(modelIdEl);
  }
  if (code === "J01" || code === "J02") {
    mark(jobIdInputEl);
  }

  const message = String(parsed?.message || text || "").toLowerCase();
  if (message.includes("insufficient training data for torch model") || message.includes("need >= 8 samples")) {
    mark(trainAlgoEl);
    mark(csvFileEl);
    mark(jsonInputEl);
  }

  if (firstField) {
    firstField.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function setFieldDisabled(elm, disabled) {
  if (!elm) return;
  elm.disabled = !!disabled;
  const field = elm.closest(".field");
  if (field) field.classList.toggle("disabled", !!disabled);
}

function currentDataSource() {
  const raw = String(dataSourceEl?.value || "").trim().toLowerCase();
  if (raw === "json") return "json";
  if (raw === "sample") return "sample";
  if (raw === "csv") return "csv";
  return "";
}

function hasDataForSource(source) {
  const src = String(source || currentDataSource());
  if (!src) return false;
  if (src === "csv") return !!csvFileEl.files?.[0];
  if (src === "json") return !!String(jsonInputEl.value || "").trim();
  if (src === "sample") return !!String(jsonInputEl.value || "").trim();
  return false;
}

function isDataConfirmed() {
  return !!lastValidatedSignature;
}

function syncValidateButtonLabel() {
  if (!validateBtnEl) return;
  validateBtnEl.textContent = isDataConfirmed() ? t("action.unconfirm") : t("action.validate");
  if (clearBtnEl) setVisible(clearBtnEl, isDataConfirmed());
}

function canConfirmJsonText(raw) {
  const text = String(raw || "").trim();
  if (!text) return false;
  let parsed = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    return false;
  }
  if (!Array.isArray(parsed) || parsed.length < 1) return false;

  const maxCheck = Math.min(300, parsed.length);
  for (let i = 0; i < maxCheck; i++) {
    const rec = parsed[i];
    if (!rec || typeof rec !== "object") return false;
    if (!rec.series_id) return false;
    if (!rec.timestamp) return false;
    if (rec.y === null || rec.y === undefined || rec.y === "") return false;
  }
  return true;
}

function validateCsvTextLightweight(rawText) {
  const text = String(rawText || "");
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) {
    return { state: "invalid", message: t("client.err.csv_too_few_lines") };
  }
  const header = csvSplitLine(lines[0]).map((h) => h.trim());
  const idx = (name) => header.indexOf(name);
  const iSeries = idx("series_id");
  const iTs = idx("timestamp");
  const iY = idx("y");
  if (iSeries < 0 || iTs < 0 || iY < 0) {
    return { state: "invalid", message: t("client.err.csv_missing_header") };
  }

  const maxCheck = Math.min(lines.length - 1, 200);
  let hasUsableRow = false;
  for (let r = 1; r <= maxCheck; r++) {
    const cols = csvSplitLine(lines[r]);
    const series_id = String(cols[iSeries] || "").trim();
    const timestamp = String(cols[iTs] || "").trim();
    const yRaw = String(cols[iY] || "").trim();
    if (!series_id && !timestamp && !yRaw) continue;
    if (!series_id || !timestamp || yRaw === "") {
      return { state: "invalid", message: t("error.v01_required_fields") };
    }
    hasUsableRow = true;
  }
  if (!hasUsableRow) {
    return { state: "invalid", message: t("error.v01_required_fields") };
  }

  return { state: "valid", message: "" };
}

async function syncCsvPrecheckStateFromFile() {
  const file = csvFileEl?.files?.[0];
  if (!file) {
    csvPrecheckState = { state: "unknown", message: "" };
    return;
  }
  csvPrecheckState = { state: "pending", message: "" };
  try {
    const prefix = await file.slice(0, 128 * 1024).text();
    csvPrecheckState = validateCsvTextLightweight(prefix);
  } catch (e) {
    csvPrecheckState = {
      state: "invalid",
      message: String(e?.message || e || "").trim() || "CSV precheck failed",
    };
  }
}

function syncValidateButtonEnabled() {
  if (!validateBtnEl) return;
  if (isDataConfirmed()) {
    validateBtnEl.disabled = false;
    setAriaDisabled(validateBtnEl, false);
    return;
  }

  const src = currentDataSource();
  let ok = hasDataForSource(src);
  if (ok && (src === "json" || src === "sample")) {
    ok = canConfirmJsonText(jsonInputEl?.value);
  }
  if (ok && src === "csv") {
    ok = csvPrecheckState.state === "valid";
  }
  validateBtnEl.disabled = !ok;
  setAriaDisabled(validateBtnEl, !ok);
}

function setDataSource(value) {
  if (!dataSourceEl) return;
  dataSourceEl.value = String(value || "");
  dataSourceEl.dispatchEvent(new Event("change"));
}

function syncDataSourceUi() {
  const src = currentDataSource();
  const picked = !!src;

  const syncBtn = (btn, kind) => {
    if (!btn) return;
    setVisible(btn, true);
    const selected = src === kind;
    btn.classList.toggle("is-selected", selected);
    btn.classList.toggle("is-dim", picked && !selected);
    btn.setAttribute("aria-pressed", selected ? "true" : "false");
  };
  syncBtn(dataSourceBtnCsvEl, "csv");
  syncBtn(dataSourceBtnJsonEl, "json");
  syncBtn(dataSourceBtnSampleEl, "sample");

  setVisible(dataSourceCsvEl, false);
  setVisible(dataSourceJsonEl, false);
  setVisible(dataActionsSampleEl, false);
  setVisible(dataSourceCsvEl, src === "csv");
  setVisible(dataSourceJsonEl, src === "json");
  setVisible(dataActionsSampleEl, src === "sample");

  // Disable inactive inputs to prevent accidental mixing.
  setFieldDisabled(csvFileEl, src !== "csv");
  setFieldDisabled(jsonInputEl, src !== "json");

  // Confirm/Clear appears only after some data exists.
  setVisible(dataActionsConfirmEl, picked && hasDataForSource(src));
  syncValidateButtonLabel();
  syncValidateButtonEnabled();
  if (clearBtnEl) setVisible(clearBtnEl, isDataConfirmed());
}

function syncInputLocks() {
  syncDataSourceUi();
}

function removeLegacyDataSourceResetUi(root = document) {
  if (!root || typeof root.querySelectorAll !== "function") return;

  for (const node of root.querySelectorAll('#dataSourceReset, .dataSourceResetRow, [data-i18n="action.change_data_source"]')) {
    try {
      node.remove();
    } catch {
      // ignore
    }
  }

  const dataCard = document.getElementById("dataCard");
  const scope = root === document ? dataCard || root : root;
  if (!scope || typeof scope.querySelectorAll !== "function") return;

  for (const btn of scope.querySelectorAll("button")) {
    const label = String(btn.textContent || "").trim();
    if (!label || !LEGACY_DATA_SOURCE_RESET_LABELS.has(label)) continue;
    try {
      const row = btn.closest(".actions, .dataSourceResetRow");
      if (row) row.remove();
      else btn.remove();
    } catch {
      // ignore
    }
  }
}

function ensureLegacyDataSourceResetObserver() {
  if (legacyDataSourceResetObserver) return;
  if (typeof MutationObserver === "undefined") return;
  const target = document.getElementById("dataCard") || document.body;
  if (!target) return;

  legacyDataSourceResetObserver = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes || []) {
        if (!(node instanceof Element)) continue;
        removeLegacyDataSourceResetUi(node);
      }
    }
  });
  legacyDataSourceResetObserver.observe(target, { childList: true, subtree: true });
}

function showJobsError(err) {
  const msg = String(err?.message || err || "").trim();
  if (!msg) return;
  setStatus(jobsStatusEl, msg);
  showError(msg);
}

function updateWarningsForCurrentInputs() {
  dataQualityController.updateWarningsForCurrentInputs();
}

function clearResults() {
  lastResult = null;
  lastRequestId = null;
  lastRunContext = null;
  showError("");
  rawJsonEl.textContent = "";
  resultsTablesController.clear();
  setVisible(metricsWrapEl, false);
  setVisible(bySeriesWrapEl, false);
  setVisible(byHorizonWrapEl, false);
  setVisible(byFoldWrapEl, false);
  setVisible(driftFeaturesWrapEl, false);
  setVisible(forecastTableWrapEl, false);
  setVisible(resultsGuideEl, false);
  setVisible(resultsLegendEl, false);
  setVisible(resultsTableNoteEl, false);
  resultsInsightsController.clear();
  resultsVisualController.clear();
  setVisible(rawDetailsEl, false);
  setVisible(resultsEmptyEl, false);
  setVisible(resultsTrainNoteEl, false);
  setVisible(requestIdWrapEl, false);
  setText(requestIdEl, "");
  taskResultsUiController.resetResultsActions();

  for (const node of [
    forecastTableWrapEl,
    resultsGuideEl,
    metricsWrapEl,
    bySeriesWrapEl,
    byHorizonWrapEl,
    byFoldWrapEl,
    driftFeaturesWrapEl,
  ]) {
    if (node && String(node.tagName || "").toLowerCase() === "details") {
      node.open = false;
    }
  }

  updateStepNavHighlight();
  refreshI18nStatuses();
}

function clearDataAnalysis() {
  dataQualityController.clearDataAnalysis();
}


function refreshExploreFromState() {
  explorePanelController.refresh();
}

function getBillingSyncMaxPoints() {
  try {
    const raw = localStorage.getItem(LS_BILLING_SYNC_MAX_POINTS);
    const n = Number(raw);
    if (Number.isFinite(n) && Number.isInteger(n) && n >= 1) return n;
  } catch {
    // ignore
  }
  // Default matches server-side fallback in forecasting_api/app.py
  return 10000;
}

function setBillingSyncMaxPoints(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return getBillingSyncMaxPoints();
  const i = Math.max(1, Math.floor(n));
  try {
    localStorage.setItem(LS_BILLING_SYNC_MAX_POINTS, String(i));
  } catch {
    // ignore
  }
  return i;
}

function getRunPointsEstimate(taskOverride = null) {
  const limit = getBillingSyncMaxPoints();
  const used = Number(lastDataStats?.n);
  const seriesCount = Number(lastDataStats?.seriesCount);

  const seriesShown = Number.isFinite(seriesCount) && seriesCount >= 0 ? seriesCount : 0;
  const task = normalizeTask(taskOverride || currentTask());
  const horizon = task === "train" || task === "drift" ? null : Number(horizonEl.value);
  const hShown = Number.isFinite(horizon) && horizon >= 1 ? Math.floor(horizon) : 0;
  const canEstimate = seriesShown > 0 && hShown > 0;
  const pointsEstimate = canEstimate ? seriesShown * hShown : 0;
  const estimateUsed = pointsEstimate > 0 ? pointsEstimate : 0;
  const remaining = limit > 0 ? Math.max(0, limit - estimateUsed) : 0;
  const ratio = limit > 0 ? estimateUsed / limit : 0;

  return {
    limit,
    used,
    seriesShown,
    hShown,
    canEstimate,
    pointsEstimate,
    estimateUsed,
    remaining,
    ratio,
  };
}

function updateBillingUi() {
  billingHelpController.updateBillingUi();
}

function countMissingRequiredFields(records) {
  const missing = { series_id: 0, timestamp: 0, y: 0 };
  for (const rec of records) {
    if (!rec?.series_id) missing.series_id++;
    if (!rec?.timestamp) missing.timestamp++;
    if (rec?.y === null || rec?.y === undefined || rec?.y === "") missing.y++;
  }
  return missing;
}

function updateDownloadButtons() {
  taskResultsUiController.updateDownloadButtons();
}

function setResultsActionPriority(task, hasResults) {
  taskResultsUiController.setResultsActionPriority(task, hasResults);
}

function setAriaDisabled(node, disabled) {
  node.disabled = !!disabled;
  node.setAttribute("aria-disabled", disabled ? "true" : "false");
}

function currentTask() {
  return normalizeTask(taskEl.value);
}

function updateTaskUi() {
  taskResultsUiController.updateTaskUi();
}

function setRunUiBusy(busy) {
  isUiBusy = !!busy;
  taskResultsUiController.setRunUiBusy(busy);
}

function buildShareState() {
  const task = currentTask();
  const state = {
    lang: normalizeLang(langSelectEl.value) || "en",
    density: normalizeDensityMode(densityModeEl?.value) || "basic",
    mode: String(modeEl.value || "sync"),
    task,
    horizon: task === "train" || task === "drift" ? null : Number(horizonEl.value),
    frequency: (frequencyEl.value || "").trim() || null,
    missing_policy: (missingPolicyEl.value || "").trim() || null,
    quantiles: (quantilesEl.value || "").trim() || null,
    level: (levelEl.value || "").trim() || null,
    model_id: (modelIdEl.value || "").trim() || null,
    folds: task === "backtest" ? Number(foldsEl.value) : null,
    metric: task === "backtest" ? String(metricEl.value || "").trim() || null : null,
    base_model: task === "train" ? (baseModelEl.value || "").trim() || null : null,
    train_algo: task === "train" ? (trainAlgoEl?.value || "").trim() || null : null,
    model_name: task === "train" ? (modelNameEl.value || "").trim() || null : null,
    training_hours: task === "train" ? (trainingHoursEl.value || "").trim() || null : null,
  };
  return state;
}

function applyShareState(state) {
  const s = sanitizeShareState(state);
  if (!s) return;

  setDensityMode(s.density || "basic", { persist: false });

  const task = String(s.task || "forecast").toLowerCase();
  taskEl.value =
    task === "train" ? "train" : task === "backtest" ? "backtest" : task === "drift" ? "drift" : "forecast";
  updateTaskUi();

  const mode = String(s.mode || "sync").toLowerCase();
  modeEl.value = mode === "job" ? "job" : "sync";

  const horizon = Number(s.horizon);
  if (Number.isFinite(horizon) && horizon >= 1) horizonEl.value = String(horizon);

  frequencyEl.value = String(s.frequency || "");
  missingPolicyEl.value = String(s.missing_policy || missingPolicyEl.value || "ignore");
  quantilesEl.value = String(s.quantiles || "");
  levelEl.value = String(s.level || "");
  modelIdEl.value = String(s.model_id || "");

  const folds = Number(s.folds);
  if (Number.isFinite(folds) && Number.isInteger(folds)) foldsEl.value = String(folds);
  if (s.metric) metricEl.value = String(s.metric);

  baseModelEl.value = String(s.base_model || "");
  if (trainAlgoEl) {
    trainAlgoEl.value = normalizeTrainAlgoValue(String(s.train_algo || trainAlgoEl.value || ""));
    enforceTrainAlgoPolicy();
  }
  modelNameEl.value = String(s.model_name || "");
  trainingHoursEl.value = String(s.training_hours || "");

  updateWarningsForCurrentInputs();
}

async function onCopyLink() {
  try {
    if (apiKeyEl.value && apiKeyEl.value.trim()) {
      showWarn(t("warn.share_link_no_api_key"));
    } else {
      showWarn("");
    }
    const st = sanitizeShareState(buildShareState());
    const sensitiveHit = findSensitiveInObject(st, "share_state");
    if (sensitiveHit) {
      showWarn(t("warn.share_blocked_sensitive", { path: sensitiveHit.path }));
      setShareStatus(t("status.share_blocked_sensitive"), "error");
      return;
    }
    const u = new URL(window.location.href);
    u.hash = encodeStateToHash(st);
    await copyTextToClipboard(u.toString());
    setShareStatus(t("status.copied"), "success");
  } catch {
    setShareStatus(t("status.copy_failed"), "error");
  }
}

function _truncateRecordsForSnippet(records, max = 20) {
  const arr = Array.isArray(records) ? records : [];
  if (arr.length <= max) return { records: arr, truncated: false };
  return { records: arr.slice(0, max), truncated: true };
}

async function onCopySnippet() {
  try {
    if (apiKeyEl.value && apiKeyEl.value.trim()) {
      showWarn(t("warn.share_link_no_api_key"));
    } else {
      showWarn("");
    }
    setShareStatus("");
    const task = currentTask();
    const mode = String(modeEl.value || "sync").toLowerCase() === "job" ? "job" : "sync";

    let records = null;
    try {
      records = await getRecordsFromInputs();
    } catch {
      records = null;
    }
    const { records: sampleRecordsRaw, truncated } = _truncateRecordsForSnippet(records, 20);
    const sampleRecords = sanitizeRecordsForSnippet(sampleRecordsRaw, 20);

    const req =
      task === "train"
        ? buildTrainRequest(Array.isArray(sampleRecords) ? sampleRecords : [])
        : task === "backtest"
          ? buildBacktestRequest(Array.isArray(sampleRecords) ? sampleRecords : [])
          : task === "drift"
            ? buildDriftRequest(Array.isArray(sampleRecords) ? sampleRecords : [])
            : buildForecastRequest(Array.isArray(sampleRecords) ? sampleRecords : []);

    const endpoint =
      task === "backtest"
        ? "/v1/backtest"
        : task === "train"
          ? "/v1/train"
          : task === "drift"
            ? "/v1/monitoring/drift/report"
            : "/v1/forecast";

    const payloadJson = JSON.stringify(req, null, 2);
    const sensitiveHit = findSensitiveInObject(req, "payload");
    if (sensitiveHit) {
      showWarn(t("warn.share_blocked_sensitive", { path: sensitiveHit.path }));
      setShareStatus(t("status.share_blocked_sensitive"), "error");
      return;
    }
    const note = truncated
      ? "# NOTE: data is truncated to the first 20 records for readability.\n"
      : "";

    let code = "";
    code += "import json\n";
    code += "import time\n";
    code += "import urllib.request\n\n";
    code += `BASE_URL = \"${window.location.origin}\"  # adjust if needed\n`;
    code += "API_KEY = \"dev-key\"  # local dev default (make api-serve-dev); otherwise set to your ARCAYF_FORECASTING_API_KEY\n\n";
    code += note;
    code += `payload = ${payloadJson}\n\n`;

    code += "def post_json(path: str, obj: dict):\n";
    code += "    data = json.dumps(obj).encode('utf-8')\n";
    code += "    req = urllib.request.Request(\n";
    code += "        BASE_URL + path,\n";
    code += "        data=data,\n";
    code += "        method='POST',\n";
    code += "        headers={'Content-Type': 'application/json', 'X-API-Key': API_KEY},\n";
    code += "    )\n";
    code += "    with urllib.request.urlopen(req) as r:\n";
    code += "        request_id = r.headers.get('X-Request-Id')\n";
    code += "        body = json.loads(r.read().decode('utf-8'))\n";
    code += "        return body, request_id\n\n";

    if (mode === "sync") {
      code += `out, req_id = post_json('${endpoint}', payload)\n`;
      code += "print('request_id=', req_id)\n";
      code += "print(json.dumps(out, ensure_ascii=False, indent=2))\n";
    } else {
      code += "# Job mode: create -> poll -> result\n";
      code += "job, req_id = post_json('/v1/jobs', {'type': '" + task + "', 'payload': payload})\n";
      code += "print('request_id=', req_id)\n";
      code += "job_id = job.get('job_id')\n";
      code += "print('job_id=', job_id)\n";
      code += "for _ in range(40):\n";
      code += "    resp = urllib.request.urlopen(urllib.request.Request(\n";
      code += "        BASE_URL + f'/v1/jobs/{job_id}',\n";
      code += "        headers={'X-API-Key': API_KEY},\n";
      code += "    ))\n";
      code += "    request_id = resp.headers.get('X-Request-Id')\n";
      code += "    st = json.loads(resp.read().decode('utf-8'))\n";
      code += "    if request_id:\n";
      code += "        print('request_id=', request_id)\n";
      code += "    status = st.get('status')\n";
      code += "    if status in ('succeeded', 'failed'):\n";
      code += "        break\n";
      code += "    time.sleep(0.5)\n";
      code += "resp = urllib.request.urlopen(urllib.request.Request(\n";
      code += "    BASE_URL + f'/v1/jobs/{job_id}/result',\n";
      code += "    headers={'X-API-Key': API_KEY},\n";
      code += "))\n";
      code += "request_id = resp.headers.get('X-Request-Id')\n";
      code += "out = json.loads(resp.read().decode('utf-8'))\n";
      code += "print('request_id=', request_id)\n";
      code += "print(json.dumps(out, ensure_ascii=False, indent=2))\n";
    }

    await copyTextToClipboard(code);
    setShareStatus(t("status.copied"), "success");
  } catch {
    setShareStatus(t("status.copy_failed"), "error");
  }
}

function parseNumberList(s) {
  const trimmed = (s || "").trim();
  if (!trimmed) return null;
  const parts = trimmed.split(",").map((x) => x.trim()).filter(Boolean);
  const nums = parts.map((p) => Number(p));
  if (nums.some((n) => !Number.isFinite(n))) throw new Error(t("client.err.number_list"));
  return nums;
}

function csvSplitLine(line) {
  // Minimal CSV split: handles quoted fields with double quotes.
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

function parseCsv(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error(t("client.err.csv_too_few_lines"));
  const header = csvSplitLine(lines[0]).map((h) => h.trim());
  const idx = (name) => header.indexOf(name);
  const iSeries = idx("series_id");
  const iTs = idx("timestamp");
  const iY = idx("y");
  if (iSeries < 0 || iTs < 0 || iY < 0) {
    throw new Error(t("client.err.csv_missing_header"));
  }
  const exogColumns = [];
  for (let i = 0; i < header.length; i++) {
    if (i === iSeries || i === iTs || i === iY) continue;
    const key = String(header[i] || "").trim();
    if (!key) continue;
    exogColumns.push({ index: i, key: key.startsWith("x.") ? key.slice(2) : key });
  }

  const parseCsvScalar = (raw) => {
    const trimmed = String(raw ?? "").trim();
    if (!trimmed) return "";
    const lowered = trimmed.toLowerCase();
    if (lowered === "true") return true;
    if (lowered === "false") return false;
    const asNum = Number(trimmed);
    if (Number.isFinite(asNum)) return asNum;
    return trimmed;
  };

  const records = [];
  for (let r = 1; r < lines.length; r++) {
    const cols = csvSplitLine(lines[r]);
    const series_id = (cols[iSeries] || "").trim();
    const timestamp = (cols[iTs] || "").trim();
    const yRaw = (cols[iY] || "").trim();
    if (!series_id && !timestamp && !yRaw) continue;
    const y = Number(yRaw);
    const x = {};
    for (const col of exogColumns) {
      const v = parseCsvScalar(cols[col.index]);
      if (v === "") continue;
      x[col.key] = v;
    }
    const rec = { series_id, timestamp, y };
    if (Object.keys(x).length > 0) rec.x = x;
    records.push(rec);
  }
  return records;
}

async function loadI18n(lang) {
  const normalized = normalizeLang(lang) || "en";
  const origin = window.location.origin || "";
  const path = window.location.pathname || "/";
  const basePath = path.endsWith("/") ? path : `${path}/`;
  const cleanBasePath = path.endsWith(".html")
    ? path.replace(/[^/]+$/, "")
    : basePath;
  const urls = [
    new URL(`i18n/${normalized}.json`, window.location.href).toString(),
    `${cleanBasePath}i18n/${normalized}.json`,
  ];
  if (origin && origin !== "null") {
    urls.push(
      `${origin}${cleanBasePath}i18n/${normalized}.json`,
      `${origin}/ui/forecasting/i18n/${normalized}.json`,
      `${origin}/static/forecasting_gui/i18n/${normalized}.json`,
    );
  }

  const withVersion = (url) => {
    try {
      const parsed = new URL(url, window.location.href);
      parsed.searchParams.set("v", I18N_CACHE_BUST_VERSION);
      return parsed.toString();
    } catch {
      const sep = String(url).includes("?") ? "&" : "?";
      return `${url}${sep}v=${encodeURIComponent(I18N_CACHE_BUST_VERSION)}`;
    }
  };

  for (const url of [...new Set(urls.map(withVersion))]) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) continue;
      const parsed = await r.json();
      if (parsed && typeof parsed["ui.title"] === "string") return parsed;
    } catch {
      // try next
    }
  }
  return null;
}

function resolveDocsHref() {
  const origin = window.location.origin || "";
  const path = window.location.pathname || "/";
  const marker = "/ui/forecasting";
  const idx = path.indexOf(marker);
  if (origin && idx >= 0) {
    const prefix = path.slice(0, idx);
    return `${origin}${prefix}/docs`;
  }
  try {
    return new URL("docs", window.location.href).toString();
  } catch {
    return "/docs";
  }
}

async function setLanguage(lang, { persist } = { persist: true }) {
  const normalized = normalizeLang(lang) || "en";

  if (persist) {
    try {
      localStorage.setItem(LS_LANG, normalized);
    } catch {
      // ignore
    }
  }

  const dict = (await loadI18n(normalized)) || (await loadI18n("ja")) || {};
  I18N = dict;

  document.documentElement.lang = normalized;
  langSelectEl.value = normalized;
  applyI18n();
  refreshI18nStatuses();
  updateCsvFileName();
  updateStepNavHighlight();
  updateHelpAdvice();
  updateBillingUi();
  renderJobHistory();
  renderModels();
  if (isConnectionReady()) {
    void syncModelsFromServer({ silent: true });
  }
  if (lastRecords) {
    updateDataAnalysis(lastRecords, { inferredFrequency: lastInferredFrequency });
  }
  if (lastResult && lastTask) {
    renderResult(lastTask, lastResult);
  }
}

function errorCodeToI18nKey(code) {
  const c = String(code || "").toUpperCase();
  if (c === "A12") return "error.a12_auth_missing";
  if (c === "A21") return "error.a21_quota_exceeded";
  if (c === "V01") return "error.v01_required_fields";
  if (c === "V02") return "error.v02_frequency_infer_failed";
  if (c === "V03") return "error.v03_quantiles_and_level";
  if (c === "V04") return "error.v04_not_sorted";
  if (c === "V05") return "error.v05_missing_timestamps";
  if (c === "V06") return "error.v06_min_history";
  if (c === "COST01") return "error.cost01_too_large_sync";
  if (c === "J01") return "error.j01_job_not_found";
  if (c === "J02") return "error.j02_job_not_ready";
  if (c === "J03") return "error.j03_job_failed";
  if (c === "M01") return "error.m01_model_not_found";
  if (c === "M02") return "error.m02_model_not_ready";
  if (c === "S01") return "error.s01_payload_too_large";
  return null;
}

function errorCodeToSuggestionKey(code) {
  const c = String(code || "").toUpperCase();
  if (c === "A12") return "error.suggest.a12";
  if (c === "A21") return "error.suggest.a21";
  if (c === "V01") return "error.suggest.v01";
  if (c === "V02") return "error.suggest.v02";
  if (c === "V03") return "error.suggest.v03";
  if (c === "V04") return "error.suggest.v04";
  if (c === "V05") return "error.suggest.v05";
  if (c === "V06") return "error.suggest.v06";
  if (c === "COST01") return "error.suggest.cost01";
  if (c === "J01") return "error.suggest.j01";
  if (c === "J02") return "error.suggest.j02";
  if (c === "J03") return "error.suggest.j03";
  if (c === "M01") return "error.suggest.m01";
  if (c === "M02") return "error.suggest.m02";
  if (c === "S01") return "error.suggest.s01";
  return "error.suggest.generic";
}

function parseErrorText(text) {
  const lines = String(text || "")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  const out = { code: null, requestId: null, message: null, details: null, raw: text };
  let collectingDetails = false;
  for (const line of lines) {
    if (line.startsWith("error_code=")) {
      collectingDetails = false;
      out.code = line.slice("error_code=".length).trim();
      continue;
    }
    if (line.startsWith("request_id=")) {
      collectingDetails = false;
      out.requestId = line.slice("request_id=".length).trim();
      continue;
    }
    if (line.startsWith("message=")) {
      collectingDetails = false;
      out.message = line.slice("message=".length).trim();
      continue;
    }
    if (line.startsWith("details=")) {
      collectingDetails = true;
      out.details = line.slice("details=".length).trim();
      continue;
    }
    if (collectingDetails && out.details) {
      out.details = `${out.details}\n${line}`;
      continue;
    }
    if (!out.message && !line.startsWith("HTTP") && !line.startsWith("preflight") && !line.startsWith("job failed")) {
      out.message = line;
    }
  }
  return out;
}

function formatErrorDetailsForDisplay(code, detailsRaw) {
  const raw = String(detailsRaw || "").trim();
  if (!raw) return "";

  let parsed = null;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return raw;
  }

  if (String(code || "").toUpperCase() === "V01") {
    const counts = parsed?.missing_counts;
    if (counts && typeof counts === "object") {
      return `missing_counts: series_id=${Number(counts.series_id) || 0}, timestamp=${Number(counts.timestamp) || 0}, y=${Number(counts.y) || 0}`;
    }
  }
  return JSON.stringify(parsed);
}

function formatErrorForDisplay(text) {
  const parsed = parseErrorText(text);
  if (!parsed || !String(text || "").trim()) return "";

  const lines = [t("error.title")];
  if (parsed.code) lines.push(t("error.code_label", { code: parsed.code }));

  const i18nKey = errorCodeToI18nKey(parsed.code);
  if (i18nKey) {
    lines.push(t(i18nKey));
  } else if (parsed.message) {
    lines.push(t("error.message_label", { message: parsed.message }));
  } else {
    lines.push(t("error.message_label", { message: parsed.raw }));
  }

  const suggestKey = errorCodeToSuggestionKey(parsed.code);
  if (suggestKey) lines.push(t(suggestKey));

  if (String(parsed.code || "").toUpperCase() === "V01") {
    const task = currentTask();
    const algo = selectedTrainAlgoForPreflight();
    if (task === "train" && algo === "afnocg2") {
      lines.push(t("error.suggest.v01_torch"));
    }
  }

  if (parsed.details) lines.push(t("error.details_label", { details: formatErrorDetailsForDisplay(parsed.code, parsed.details) }));
  if (parsed.requestId) lines.push(t("error.request_label", { request_id: parsed.requestId }));

  return lines.filter(Boolean).join("\n");
}

function isStrictlyIncreasing(values) {
  for (let i = 1; i < values.length; i++) {
    if (!(values[i] > values[i - 1])) return false;
  }
  return true;
}

function findNonMonotonicSeriesId(records) {
  const bySeries = new Map();
  for (const rec of records) {
    const sid = String(rec?.series_id ?? "");
    if (!bySeries.has(sid)) bySeries.set(sid, []);
    bySeries.get(sid).push(String(rec?.timestamp ?? ""));
  }

  for (const [sid, tsList] of bySeries.entries()) {
    const parsed = tsList.map((ts) => Date.parse(ts));
    const allParsed = parsed.every((n) => Number.isFinite(n));
    if (allParsed) {
      if (!isStrictlyIncreasing(parsed)) return sid;
      continue;
    }
    if (!isStrictlyIncreasing(tsList)) return sid;
  }
  return null;
}

function formatFrequencyFromMs(stepMs) {
  const ms = Number(stepMs);
  if (!Number.isFinite(ms) || ms <= 0) return null;

  const day = 24 * 60 * 60 * 1000;
  const hour = 60 * 60 * 1000;
  const minute = 60 * 1000;

  if (ms % day === 0) return `${ms / day}d`;
  if (ms % hour === 0) return `${ms / hour}h`;
  if (ms % minute === 0) return `${ms / minute}min`;
  return null;
}

function inferFrequencyFromRecords(records) {
  const bySeries = new Map();
  for (const rec of records) {
    const sid = String(rec?.series_id ?? "");
    const ts = Date.parse(String(rec?.timestamp ?? ""));
    if (!Number.isFinite(ts)) continue;
    if (!bySeries.has(sid)) bySeries.set(sid, []);
    bySeries.get(sid).push(ts);
  }

  const diffs = [];
  for (const tsList of bySeries.values()) {
    const sorted = [...tsList].sort((a, b) => a - b);
    for (let i = 1; i < sorted.length; i++) {
      const d = sorted[i] - sorted[i - 1];
      if (Number.isFinite(d) && d > 0) diffs.push(d);
    }
  }

  if (diffs.length === 0) return null;

  const counts = new Map();
  for (const d of diffs) {
    counts.set(d, (counts.get(d) || 0) + 1);
  }

  let bestDiff = null;
  let bestCount = -1;
  for (const [d, c] of counts.entries()) {
    if (c > bestCount || (c === bestCount && (bestDiff === null || d < bestDiff))) {
      bestDiff = d;
      bestCount = c;
    }
  }

  return bestDiff === null ? null : formatFrequencyFromMs(bestDiff);
}

function isTimestampMonotonicPerSeries(records) {
  const bySeries = new Map();
  for (const rec of records) {
    const sid = String(rec?.series_id ?? "");
    if (!bySeries.has(sid)) bySeries.set(sid, []);
    bySeries.get(sid).push(String(rec?.timestamp ?? ""));
  }

  for (const tsList of bySeries.values()) {
    const parsed = tsList.map((ts) => Date.parse(ts));
    const allParsed = parsed.every((n) => Number.isFinite(n));
    if (allParsed) {
      if (!isStrictlyIncreasing(parsed)) return false;
      continue;
    }
    if (!isStrictlyIncreasing(tsList)) return false;
  }
  return true;
}

function parseFrequencyToMs(frequency) {
  const raw = (frequency || "").trim();
  if (!raw) return null;
  const s = raw.toLowerCase();

  if (s === "d" || s === "day") return 24 * 60 * 60 * 1000;
  if (s === "h" || s === "hr" || s === "hour") return 60 * 60 * 1000;
  if (s === "m" || s === "min" || s === "minute") return 60 * 1000;

  const m = s.match(/^([0-9]+)\s*(d|day|h|hr|hour|min|m|minute)$/);
  if (!m) return null;
  const n = Number(m[1]);
  if (!Number.isFinite(n) || n <= 0) return null;
  const unit = m[2];
  if (unit === "d" || unit === "day") return n * 24 * 60 * 60 * 1000;
  if (unit === "h" || unit === "hr" || unit === "hour") return n * 60 * 60 * 1000;
  if (unit === "min" || unit === "m" || unit === "minute") return n * 60 * 1000;
  return null;
}

function hasGapsPerSeries(records, stepMs) {
  const bySeries = new Map();
  for (const rec of records) {
    const sid = String(rec?.series_id ?? "");
    if (!bySeries.has(sid)) bySeries.set(sid, []);
    bySeries.get(sid).push(String(rec?.timestamp ?? ""));
  }

  for (const tsList of bySeries.values()) {
    const parsed = tsList.map((ts) => Date.parse(ts));
    if (!parsed.every((n) => Number.isFinite(n))) {
      // Can't reliably detect gaps without parseable timestamps.
      return false;
    }
    for (let i = 1; i < parsed.length; i++) {
      const diff = parsed[i] - parsed[i - 1];
      if (diff !== stepMs) return true;
    }
  }
  return false;
}

function countSeriesWithGaps(records, stepMs) {
  const bySeries = new Map();
  for (const rec of records) {
    const sid = String(rec?.series_id ?? "");
    const ts = Date.parse(String(rec?.timestamp ?? ""));
    if (!Number.isFinite(ts)) continue;
    if (!bySeries.has(sid)) bySeries.set(sid, []);
    bySeries.get(sid).push(ts);
  }

  let count = 0;
  for (const tsList of bySeries.values()) {
    const sorted = [...tsList].sort((a, b) => a - b);
    let hasGap = false;
    for (let i = 1; i < sorted.length; i++) {
      const diff = sorted[i] - sorted[i - 1];
      if (diff > stepMs) {
        hasGap = true;
        break;
      }
    }
    if (hasGap) count++;
  }
  return count;
}

function updateDataAnalysis(records, { inferredFrequency = null } = {}) {
  dataQualityController.updateDataAnalysis(records, { inferredFrequency });
}

function renderDataPreview(records) {
  if (!dataPreviewEl || !dataPreviewMetaEl || !dataPreviewTableEl) return;
  if (!Array.isArray(records) || records.length === 0) {
    setVisible(dataPreviewEl, false);
    return;
  }

  const xKeys = [];
  const xKeySet = new Set();
  for (const rec of records) {
    const x = rec?.x && typeof rec.x === "object" ? rec.x : null;
    if (!x) continue;
    for (const k of Object.keys(x)) {
      if (xKeySet.has(k)) continue;
      xKeySet.add(k);
      xKeys.push(k);
      if (xKeys.length >= 3) break;
    }
    if (xKeys.length >= 3) break;
  }

  dataPreviewMetaEl.innerHTML = "";
  setVisible(dataPreviewMetaEl, false);
  if (dataPreviewExplainEl) {
    dataPreviewExplainEl.textContent = "";
    setVisible(dataPreviewExplainEl, false);
  }

  const columns = ["series_id", "timestamp", "y", ...xKeys.map((k) => `x.${k}`)];
  const thead = dataPreviewTableEl.querySelector("thead");
  const tbody = dataPreviewTableEl.querySelector("tbody");
  if (!thead || !tbody) return;
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const trHead = document.createElement("tr");
  trHead.className = "dataPreviewRoleRow";
  for (const col of columns) {
    const th = document.createElement("th");
    th.className = "dataPreviewRoleCell";
    let roleLabel = t("data.preview.role.exog");
    let roleClass = "role-exog";
    if (col === "series_id") {
      roleLabel = t("data.preview.role.series");
      roleClass = "role-series";
    } else if (col === "timestamp") {
      roleLabel = t("data.preview.role.timestamp");
      roleClass = "role-time";
    } else if (col === "y") {
      roleLabel = t("data.preview.role.target");
      roleClass = "role-target";
    }
    th.innerHTML = `<span class="dataPreviewRoleTag ${roleClass}">${escapeHtml(roleLabel)}</span>`;
    trHead.appendChild(th);
  }
  thead.appendChild(trHead);

  const trCols = document.createElement("tr");
  trCols.className = "dataPreviewColumnsRow";
  for (const col of columns) {
    const th = document.createElement("th");
    th.textContent = col;
    trCols.appendChild(th);
  }
  thead.appendChild(trCols);

  const sorted = [...records].sort((a, b) => {
    const sa = String(a?.series_id ?? "");
    const sb = String(b?.series_id ?? "");
    if (sa !== sb) return sa.localeCompare(sb);
    const ta = Date.parse(String(a?.timestamp ?? ""));
    const tb = Date.parse(String(b?.timestamp ?? ""));
    if (Number.isFinite(ta) && Number.isFinite(tb)) return ta - tb;
    return String(a?.timestamp ?? "").localeCompare(String(b?.timestamp ?? ""));
  });

  const sample = sorted.slice(0, 6);
  for (const rec of sample) {
    const tr = document.createElement("tr");
    const x = rec?.x && typeof rec.x === "object" ? rec.x : null;
    for (const col of columns) {
      const td = document.createElement("td");
      if (col === "series_id") td.textContent = String(rec?.series_id ?? "");
      else if (col === "timestamp") td.textContent = String(rec?.timestamp ?? "");
      else if (col === "y") td.textContent = String(rec?.y ?? "");
      else {
        const key = col.slice(2);
        const val = x ? x[key] : "";
        td.textContent = val === undefined ? "" : String(val);
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  setVisible(dataPreviewEl, true);
}

function setParamsLinkStatus(key, vars = null, type = "info") {
  if (!paramsLinkNoteEl) return;
  setStatusI18n(paramsLinkNoteEl, key, vars, type);
}

function updateParamsLinkUi({ hasData = false, changed = false } = {}) {
  if (!paramsSyncEl) return;
  if (!hasData) {
    paramsSyncEl.hidden = true;
    setParamsLinkStatus("params.link.idle", null, "info");
    return;
  }
  if (paramsLinked) {
    paramsSyncEl.hidden = true;
    setParamsLinkStatus("params.link.synced", null, "success");
    return;
  }
  paramsSyncEl.hidden = false;
  setParamsLinkStatus(changed ? "params.link.needs_sync" : "params.link.unlinked", null, changed ? "warn" : "info");
}

function suggestHorizon(n, seriesCount) {
  const perSeries = Math.max(1, Math.round(n / Math.max(1, seriesCount)));
  const suggested = Math.round(perSeries * 0.2);
  return Math.max(2, Math.min(30, suggested));
}

function syncParamsFromData({ records, inferredFrequency, force = false } = {}) {
  if (!Array.isArray(records) || records.length === 0) {
    updateParamsLinkUi({ hasData: false, changed: false });
    return;
  }
  const n = records.length;
  const seriesCount = new Set(records.map((r) => String(r?.series_id ?? ""))).size;
  const suggestedHorizon = suggestHorizon(n, seriesCount);
  const suggestedFrequency = (inferredFrequency || frequencyEl.value || "").trim();
  const shouldApply = force || paramsLinked;

  isSyncingParams = true;
  if (shouldApply) {
    if (!frequencyEl.value || force) {
      if (suggestedFrequency) frequencyEl.value = suggestedFrequency;
    }
    if (!horizonEl.value || force) {
      horizonEl.value = String(suggestedHorizon);
    }
    if (!quantilesEl.value || force) {
      quantilesEl.value = "0.1,0.5,0.9";
    }
  }
  isSyncingParams = false;
  lastSyncedSignature = computeDataSignature(records);
  updateParamsLinkUi({ hasData: true, changed: false });
}

function computeGapDetailsPerSeries(records, stepMs, { maxSeries = 5, maxGapsPerSeries = 5 } = {}) {
  const bySeries = new Map();
  for (const rec of records) {
    const sid = String(rec?.series_id ?? "");
    const ts = Date.parse(String(rec?.timestamp ?? ""));
    if (!Number.isFinite(ts)) continue;
    if (!bySeries.has(sid)) bySeries.set(sid, []);
    bySeries.get(sid).push(ts);
  }

  const out = [];
  for (const [sid, tsList] of bySeries.entries()) {
    if (out.length >= maxSeries) break;
    const sorted = [...tsList].sort((a, b) => a - b);
    const gaps = [];
    for (let i = 1; i < sorted.length; i++) {
      const diff = sorted[i] - sorted[i - 1];
      if (diff > stepMs) {
        const missingCount = Math.max(0, Math.round(diff / stepMs) - 1);
        gaps.push({
          from: new Date(sorted[i - 1]).toISOString(),
          to: new Date(sorted[i]).toISOString(),
          missing_count: missingCount,
        });
        if (gaps.length >= maxGapsPerSeries) break;
      }
    }
    if (gaps.length > 0) {
      out.push({ series_id: sid, expected_step_ms: stepMs, gaps });
    }
  }
  return out;
}

function inferSeasonalPeriodSteps(stepMs) {
  if (!Number.isFinite(stepMs) || stepMs <= 0) return null;
  const dayMs = 24 * 60 * 60 * 1000;
  if (stepMs < dayMs) {
    const steps = Math.round(dayMs / stepMs);
    return steps >= 2 ? steps : null;
  }
  const weekSteps = Math.round((7 * dayMs) / stepMs);
  return weekSteps >= 2 ? weekSteps : null;
}

function getMinHistoryRecords(task, req) {
  const baseMin = 20;
  const stepMs = parseFrequencyToMs(req?.frequency);
  const seasonalSteps = inferSeasonalPeriodSteps(stepMs);
  const seasonalMin = seasonalSteps ? seasonalSteps * 2 : 0;
  let horizonMin = 0;
  if (task === "forecast" || task === "backtest") {
    const h = Number(req?.horizon);
    if (Number.isFinite(h) && h > 0) {
      horizonMin = Math.max(2, Math.round(h * 2));
    }
  }
  return Math.max(baseMin, seasonalMin, horizonMin);
}

function validatePreflight(records, { task, req }) {
  // V01 (required fields)
  const missingCounts = { series_id: 0, timestamp: 0, y: 0 };
  for (const rec of records) {
    if (!rec?.series_id) missingCounts.series_id++;
    if (!rec?.timestamp) missingCounts.timestamp++;
    if (rec?.y === null || rec?.y === undefined || rec?.y === "") missingCounts.y++;
  }
  if (missingCounts.series_id > 0 || missingCounts.timestamp > 0 || missingCounts.y > 0) {
    return {
      ok: false,
      error_code: "V01",
      message: t("error.v01_required_fields"),
      details: { fields: ["series_id", "timestamp", "y"], missing_counts: missingCounts },
    };
  }

  if (task === "forecast") {
    // V03 (exclusive)
    if (req.quantiles && req.level) {
      return {
        ok: false,
        error_code: "V03",
        message: t("error.v03_quantiles_and_level"),
        details: { fields: ["quantiles", "level"] },
      };
    }
  }

  // V04 (timestamp monotonic per series)
  if (!isTimestampMonotonicPerSeries(records)) {
    return {
      ok: false,
      error_code: "V04",
      message: t("error.v04_not_sorted"),
      details: { series_id: findNonMonotonicSeriesId(records) },
    };
  }

  // V06 (min history)
  const minHistory = task === "drift" ? 0 : getMinHistoryRecords(task, req);
  if (minHistory > 0) {
    const seriesCounts = new Map();
    for (const rec of records) {
      const sid = String(rec?.series_id ?? "");
      seriesCounts.set(sid, (seriesCounts.get(sid) || 0) + 1);
    }
    const shortSeries = [];
    for (const [sid, count] of seriesCounts.entries()) {
      if (count < minHistory) shortSeries.push({ series_id: sid, count });
    }
    if (shortSeries.length > 0) {
      return {
        ok: false,
        error_code: "V06",
        message: t("error.v06_min_history"),
        details: {
          min_history: minHistory,
          short_series_count: shortSeries.length,
          short_series: shortSeries.slice(0, 5),
        },
      };
    }
  }

  if (task === "forecast") {
    // V05 (gaps/missing timestamps; only when missing_policy=error)
    const missingPolicy = req?.options?.missing_policy;
    if (missingPolicy === "error") {
      const stepMs = parseFrequencyToMs(req?.frequency);
      if (stepMs) {
        if (hasGapsPerSeries(records, stepMs)) {
          return {
            ok: false,
            error_code: "V05",
            message: t("error.v05_missing_timestamps"),
            details: { gap_summary: computeGapDetailsPerSeries(records, stepMs) },
          };
        }
      }
    }
  }
  return { ok: true };
}

function buildForecastRequest(records) {
  const horizon = Number(horizonEl.value);
  if (!Number.isFinite(horizon) || horizon < 1) throw new Error(t("client.err.horizon_invalid"));

  const frequency = (frequencyEl.value || "").trim() || null;
  const missing_policy = (missingPolicyEl.value || "").trim() || null;
  const supported = !!uncertaintyModeEl && !!quantilesFieldEl && !!levelFieldEl;
  const mode = normalizeUncertaintyMode(supported ? uncertaintyModeEl.value : inferUncertaintyModeFromInputs());
  const quantiles = mode === "quantiles" ? parseNumberList(quantilesEl.value) : null;
  const level = mode === "level" ? parseNumberList(levelEl.value) : null;
  const explicitModelId = (modelIdEl.value || "").trim() || null;
  const defaultModelId = getDefaultModelId();

  const req = { horizon, data: records };
  if (frequency) req.frequency = frequency;
  if (missing_policy) req.options = { ...(req.options || {}), missing_policy };
  if (quantiles) req.quantiles = quantiles;
  if (level) req.level = level;
  if (explicitModelId || defaultModelId) req.model_id = explicitModelId || defaultModelId;
  return req;
}

function buildBacktestRequest(records) {
  const horizon = Number(horizonEl.value);
  if (!Number.isFinite(horizon) || horizon < 1) throw new Error(t("client.err.horizon_invalid"));

  const folds = Number(foldsEl.value);
  if (!Number.isFinite(folds) || !Number.isInteger(folds) || folds < 1 || folds > 20) {
    throw new Error(t("client.err.folds_invalid"));
  }

  const metric = String(metricEl.value || "rmse").trim().toLowerCase();
  const allowed = new Set(["rmse", "mae", "mape", "smape", "mase", "wape"]);
  const m = allowed.has(metric) ? metric : "rmse";

  const explicitModelId = (modelIdEl.value || "").trim() || null;
  const defaultModelId = getDefaultModelId();
  const req = { horizon, folds, metric: m, data: records };
  if (explicitModelId || defaultModelId) req.model_id = explicitModelId || defaultModelId;
  return req;
}

function buildTrainRequest(records) {
  const base_model = (baseModelEl.value || "").trim() || null;
  const algoRaw = String(trainAlgoEl?.value || "").trim().toLowerCase();
  const algoNormalized = normalizeTrainAlgoValue(algoRaw);
  const algo = algoNormalized || null;
  const model_name = (modelNameEl.value || "").trim() || null;
  const thRaw = (trainingHoursEl.value || "").trim();

  if (isCifRestrictedSelection({ algo: algoRaw, baseModel: base_model || "" })) {
    throw new Error(t("run.block.cif_disabled"));
  }

  const req = { data: records };
  if (base_model) req.base_model = base_model;
  if (algo) req.algo = algo;
  if (model_name) req.model_name = model_name;
  if (thRaw) {
    const th = Number(thRaw);
    if (!Number.isFinite(th) || th < 0.05) throw new Error(t("client.err.training_hours_invalid"));
    req.training_hours = th;
  }
  return req;
}

function buildDriftRequest(records) {
  return { candidate_records: records };
}

// (init() at bottom wires up language + listeners)

function updateForecastSeriesOptions(forecasts) {
  lastForecastSeriesId = resultsVisualController.syncForecastSeriesOptions(forecasts, lastForecastSeriesId);
}

function renderSelectedForecastSeries() {
  resultsVisualController.renderSelectedForecastSeries({
    forecasts: lastForecasts,
    selectedSeriesId: lastForecastSeriesId,
    residualEvidence: getResidualEvidenceFromResult(),
  });
}

function renderResult(task, json) {
  clearResults();
  showError("");
  lastResult = json;
  lastTask = task;
  updateStepNavHighlight();
  rawJsonEl.textContent = JSON.stringify(json, null, 2);
  setVisible(rawDetailsEl, true);
  if (lastRequestId) {
    requestIdEl.textContent = t("results.request_id", { request_id: lastRequestId });
    setVisible(requestIdWrapEl, true);
  }
  resultsInsightsController.renderSummary(task, { result: json });
  resultsInsightsController.renderEvidence(task, { result: json });
  resultsInsightsController.renderBenchmark(task, {
    result: json,
    onDeferredRefresh: () => {
      if (lastResult) resultsInsightsController.renderBenchmark(lastTask, { result: lastResult });
    },
  });

  const applyVisibility = (visibility) => {
    setVisible(forecastTableWrapEl, !!visibility.showForecastTable);
    setVisible(metricsWrapEl, !!visibility.showMetrics);
    setVisible(bySeriesWrapEl, !!visibility.showBySeries);
    setVisible(byHorizonWrapEl, !!visibility.showByHorizon);
    setVisible(byFoldWrapEl, !!visibility.showByFold);
    setVisible(driftFeaturesWrapEl, !!visibility.showDriftFeatures);
    setVisible(resultsGuideEl, !!visibility.showGuide);
    setVisible(resultsLegendEl, !!visibility.showLegend);
    setVisible(resultsTableNoteEl, !!visibility.showTableNote);
    setVisible(resultsVisualEl, !!visibility.showVisual);
    setVisible(resultsHighlightsEl, !!visibility.showHighlights);
    setVisible(resultsEmptyEl, !!visibility.showEmpty);
    setVisible(resultsTrainNoteEl, !!visibility.showTrainNote);
  };

  if (task === "train") {
    const modelId = String(json?.model_id || "").trim();
    if (modelId) {
      upsertModelCatalogEntry({
        model_id: modelId,
        created_at: nowIso(),
        memo: "",
        algo: lastRunContext?.algo_id || req.algo || null,
        algo_display: lastRunContext?.algo_display || getTrainAlgoDisplay(req.algo || "") || null,
      });
      setDefaultModelId(modelId);
      setStatusI18n(modelsStatusEl, "status.model_saved", { model_id: modelId }, "success");
      renderModels();
      void syncModelsFromServer({ silent: true });

      taskEl.value = "forecast";
      updateTaskUi();
      modelIdEl.value = modelId;
      updateRunGate();
      if (autoForecastAfterTrainRequested) {
        pendingAutoForecastAfterTrain = true;
        setStatusI18n(runStatusEl, "status.train_handoff_autorun", { model_id: modelId }, "info");
      } else {
        pendingAutoForecastAfterTrain = false;
        clearResults();
        lastResult = null;
        lastTask = "forecast";
        updateResultsStatus();
        updateStepNavHighlight();
        setStatusI18n(runStatusEl, "status.train_handoff_ready", { model_id: modelId }, "success");
      }
    }
    applyVisibility(buildResultVisibility("train"));
    resultsInsightsController.renderEvidence("train", { result: json });
    downloadJsonEl.disabled = !lastResult;
    downloadCsvEl.disabled = true;
    setResultsActionPriority(task, !!lastResult);
    clearFieldHighlights();
    return;
  }

  if (task === "drift") {
    const featureReports = Array.isArray(json?.feature_reports) ? json.feature_reports : [];
    applyVisibility(buildResultVisibility("drift", { hasDriftFeatures: featureReports.length > 0 }));
    resultsTablesController.renderDriftFeatures(featureReports);
    resultsInsightsController.renderEvidence("drift", { result: json });
    downloadJsonEl.disabled = !lastResult;
    downloadCsvEl.disabled = true;
    setResultsActionPriority(task, !!lastResult);
    clearFieldHighlights();
    return;
  }

  if (task === "backtest") {
    const { metrics, metricsEntries, highlightKey, bySeriesRows, byHorizonRows, byFoldRows } = buildBacktestViewModel(json);
    const metricTableRows = buildMetricTableRows(metricsEntries, highlightKey);
    const seriesTableRows = buildSeriesRankRows(bySeriesRows, { limit: 50 });
    const horizonTableRows = buildHorizonRows(byHorizonRows);
    const foldTableRows = buildFoldRows(byFoldRows);

    applyVisibility(
      buildResultVisibility("backtest", {
        hasMetrics: Object.keys(metrics).length > 0,
        hasBySeries: seriesTableRows.length > 0,
        hasByHorizon: horizonTableRows.length > 0,
        hasByFold: foldTableRows.length > 0,
      }),
    );
    resultsInsightsController.renderEvidence("backtest", { result: json });

    resultsTablesController.renderBacktestTables({
      highlightKey,
      metricTableRows,
      seriesTableRows,
      horizonTableRows,
      foldTableRows,
    });

    downloadJsonEl.disabled = !lastResult;
    downloadCsvEl.disabled = !lastResult;
    setResultsActionPriority(task, !!lastResult);
    clearFieldHighlights();
    return;
  }

  const forecasts = Array.isArray(json?.forecasts) ? normalizeForecasts(json.forecasts) : [];
  const forecastRows = buildForecastTableRows(forecasts);
  applyVisibility(buildResultVisibility("forecast", { hasForecasts: forecastRows.length > 0 }));

  lastForecasts = forecasts;
  updateForecastSeriesOptions(forecasts);
  resultsTablesController.renderForecastTable(forecastRows);

  if (forecastRows.length > 0) {
    renderSelectedForecastSeries();
    if (isDataConfirmed()) {
      refreshExploreFromState();
    }
  } else if (resultsSparklineEl) {
    resultsSparklineEl.innerHTML = "";
    resultsInsightsController.renderEvidence("forecast", { seriesPoints: [] });
  }
  downloadJsonEl.disabled = !lastResult;
  downloadCsvEl.disabled = forecastRows.length === 0;
  setResultsActionPriority(task, forecastRows.length > 0 || !!lastResult);
  clearFieldHighlights();
}

function downloadBlob(filename, contentType, text) {
  const blob = new Blob([text], { type: contentType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function getRecordsFromInputs() {
  const src = currentDataSource();
  if (src === "csv") {
    const file = csvFileEl.files?.[0];
    if (!file) throw new Error(t("client.err.data_required"));
    const text = await file.text();
    return parseCsv(text);
  }
  if (src === "sample") {
    const key = String(sampleTypeEl?.value || "simple");
    const sample = SAMPLE_LIBRARY[key] || SAMPLE_LIBRARY.simple;
    const records = Array.isArray(sample?.records) ? sample.records : [];
    return records.map((rec) => ({
      ...rec,
      ...(rec?.x && typeof rec.x === "object" ? { x: { ...rec.x } } : {}),
    }));
  }

  const jsonText = (jsonInputEl.value || "").trim();
  if (!jsonText) throw new Error(t("client.err.data_required"));
  const parsed = JSON.parse(jsonText);
  if (!Array.isArray(parsed)) throw new Error(t("client.err.json_must_be_array"));
  return parsed;
}

async function runSync(task, req, abortSignal) {
  const { body, requestId } = await apiClient.runTask(task, req, { signal: abortSignal });
  lastRequestId = requestId || null;
  return body;
}

function isCost01ErrorMessage(err) {
  const msg = String(err?.message ?? err ?? "");
  return /(^|\n)error_code=COST01(\n|$)/.test(msg);
}

async function runJob(task, req, abortSignal, { startedMessageKey = "info.job_started" } = {}) {
  const { body: createBody, requestId: createRequestId } = await apiClient.createJob(task, req, { signal: abortSignal });
  lastRequestId = createRequestId || null;
  const jobId = createBody?.job_id;
  if (!jobId) throw new Error(t("client.err.job_id_missing"));

  upsertJobHistoryEntry({ job_id: jobId, status: createBody?.status ?? "queued", progress: 0 });
  renderJobHistory();

  setRunState("running", { mode: "job", jobId, progress: 0 });
  updateRunProgress(0);
  updateResultsStatus();
  setStatusI18n(runStatusEl, startedMessageKey, { job_id: jobId });

  // Poll status with bounded backoff.
  let done = false;
  for (let i = 0; i < JOB_POLL_MAX_ATTEMPTS; i++) {
    if (abortSignal?.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }
    setStatusI18n(runStatusEl, "status.job_polling", { job_id: jobId, i: i + 1, total: JOB_POLL_MAX_ATTEMPTS }, "info");
    const { body: stBody, requestId: stRequestIdHeader } = await apiClient.getJob(jobId, { signal: abortSignal });
    if (stRequestIdHeader) lastRequestId = stRequestIdHeader;
    const status = stBody?.status;
    const progress = stBody?.progress;
    if (Number.isFinite(progress)) {
      runState.status = "running";
      runState.mode = "job";
      runState.jobId = jobId;
      updateRunProgress(progress);
      updateResultsStatus();
    }

    upsertJobHistoryEntry({
      job_id: jobId,
      status,
      progress: progress ?? null,
      error_code: stBody?.error?.error_code ?? null,
      request_id: stBody?.error?.request_id ?? stRequestIdHeader ?? null,
      message: stBody?.error?.message ?? null,
    });
    renderJobHistory();

    if (status === "failed") {
      const err = stBody?.error;
      const code = err?.error_code || "UNKNOWN";
      const msg = err?.message || "";
      const rid = err?.request_id || stRequestIdHeader || "";
      const i18nKey = errorCodeToI18nKey(code);
      const i18nMsg = i18nKey ? t(i18nKey) : "";
      throw new Error([`job failed`, `error_code=${code}`, msg ? `message=${msg}` : null, rid ? `request_id=${rid}` : null]
        .concat(i18nMsg ? [`i18n=${i18nKey}`, `message_i18n=${i18nMsg}`] : [])
        .filter(Boolean)
        .join("\n"));
    }
    if (status === "succeeded") {
      done = true;
      break;
    }
    const waitMs = Math.min(JOB_POLL_MAX_WAIT_MS, JOB_POLL_BASE_WAIT_MS + i * 25);
    await new Promise((r) => setTimeout(r, waitMs));
  }

  if (!done) {
    try {
      const { body: lateBody, requestId: lateRequestId } = await apiClient.getJobResult(jobId, { signal: abortSignal });
      if (lateRequestId) lastRequestId = lateRequestId;
      upsertJobHistoryEntry({ job_id: jobId, status: "succeeded", progress: 100 });
      renderJobHistory();
      return lateBody;
    } catch {
      throw new Error(t("client.err.job_poll_timeout", { job_id: jobId, total: JOB_POLL_MAX_ATTEMPTS }));
    }
  }

  const { body: resBody, requestId: resultRequestId } = await apiClient.getJobResult(jobId, { signal: abortSignal });
  lastRequestId = resultRequestId || lastRequestId;

  upsertJobHistoryEntry({ job_id: jobId, status: "succeeded", progress: 100 });
  renderJobHistory();
  return resBody;
}

async function onValidate() {
  if (isDataConfirmed()) {
    invalidateValidationState();
    clearDataAnalysis();
    clearResults();
    setStatus(dataStatusEl, "");
    showError("");
    showWarn("");
    syncInputLocks();
    updateWarningsForCurrentInputs();
    updateStepNavHighlight();
    return;
  }
  showError("");
  showWarn("");
  clearResults();
  clearDataAnalysis();
  setStatus(dataStatusEl, "");
  let records = [];
  try {
    records = await getRecordsFromInputs();
    lastRecords = records;
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err || ""));
    setStatusI18n(dataStatusEl, "status.validate_ng", null, "error");
    lastValidatedSignature = null;
    updateStepNavHighlight();
    return;
  }

  const task = currentTask();

  const seriesCount = new Set(records.map((r) => String(r?.series_id ?? ""))).size;
  let inferredFrequency = null;
  if (!(frequencyEl.value || "").trim()) {
    inferredFrequency = inferFrequencyFromRecords(records);
    if (inferredFrequency && paramsLinked) {
      frequencyEl.value = inferredFrequency;
    }
  }
  lastInferredFrequency = inferredFrequency;

  updateDataAnalysis(records, { inferredFrequency });
  lastDataSignature = computeDataSignature(records);
  if (paramsLinked) {
    syncParamsFromData({ records, inferredFrequency, force: false });
  } else {
    const changed = Boolean(lastSyncedSignature && lastDataSignature !== lastSyncedSignature);
    updateParamsLinkUi({ hasData: true, changed });
  }

  const req =
    task === "train"
      ? buildTrainRequest(records)
      : task === "backtest"
        ? buildBacktestRequest(records)
        : task === "drift"
          ? buildDriftRequest(records)
          : buildForecastRequest(records);
  const pre = validatePreflight(records, { task, req });
  if (!pre.ok) {
    const i18nKey = errorCodeToI18nKey(pre.error_code);
    const parts = [
      t("client.err.preflight_prefix"),
      `error_code=${pre.error_code}`,
      i18nKey ? `i18n=${i18nKey}` : null,
      pre.message ? `message=${pre.message}` : null,
      pre.details ? `details=${JSON.stringify(pre.details)}` : null,
    ].filter(Boolean);
    showError(parts.join("\n"));
    setStatusI18n(dataStatusEl, "status.validate_ng", null, "error");
    lastValidatedSignature = null;
    return;
  }

  const hasGapIssue = (missingPolicyEl.value || "").trim() === "error" && Number(lastDataGapCount) > 0;
  const parts = [
    { key: "status.validate_ok", vars: { n: records.length, s: seriesCount } },
    hasGapIssue ? { key: "status.validate_gaps", vars: { s: lastDataGapCount } } : null,
    inferredFrequency ? { key: "info.frequency_inferred", vars: { frequency: inferredFrequency } } : null,
  ];
  setStatusComposite(dataStatusEl, parts, hasGapIssue ? "warn" : "success");
  lastValidatedSignature = getValidationSignatureFromState();

  // Show exploration only after confirm.
  try {
    if (exploreDatasetEl) exploreDatasetEl.value = normalizeExploreDataset(exploreDatasetEl.value || "input");
    refreshExploreFromState();
  } catch {
    // ignore
  }

  syncValidateButtonLabel();

  updateWarningsForCurrentInputs();
  updateStepNavHighlight();
}

async function onRun() {
  showError("");
  showWarn("");
  clearDataAnalysis();
  setRunState("idle");
  updateResultsStatus();
  setStatus(runStatusEl, "");

  try {
    const reasons = getRunBlockReasons();
    if (reasons.length > 0) {
      runGateExplained = true;
      setStatus(runStatusEl, `${t("status.run_blocked_preconditions")}\n- ${reasons.join("\n- ")}`, "warn");
      setRunState("idle");
      updateResultsStatus();
      updateRunGate();
      return;
    }
    if (hasGapIssue() && !ackGapsEl?.checked) {
      runGateExplained = true;
      setStatusI18n(runStatusEl, "status.run_blocked_gaps", null, "warn");
      setRunState("idle");
      updateResultsStatus();
      updateRunGate();
      return;
    }
    storeApiKey(apiKeyEl.value, rememberApiKeyEl.checked);
  } catch {
    // ignore
  }

  if (currentRunAbort) {
    try {
      currentRunAbort.abort();
    } catch {
      // ignore
    }
  }
  currentRunAbort = new AbortController();

  setRunUiBusy(true);
  try {
    updateWarningsForCurrentInputs();
    const records = await getRecordsFromInputs();
    lastRecords = records;

    // Show a chart immediately so users can see the run is in progress.
    renderInputPreviewChart(records);

    const task = currentTask();
    autoForecastAfterTrainRequested = task === "train";
    pendingAutoForecastAfterTrain = false;

    let inferredFrequency = null;
    if (!(frequencyEl.value || "").trim()) {
      inferredFrequency = inferFrequencyFromRecords(records);
      if (inferredFrequency && paramsLinked) {
        frequencyEl.value = inferredFrequency;
      }
    }
    lastInferredFrequency = inferredFrequency;

    updateDataAnalysis(records, { inferredFrequency });
    lastDataSignature = computeDataSignature(records);
    if (paramsLinked) {
      syncParamsFromData({ records, inferredFrequency, force: false });
    } else {
      const changed = Boolean(lastSyncedSignature && lastDataSignature !== lastSyncedSignature);
      updateParamsLinkUi({ hasData: true, changed });
    }
    const req =
      task === "train"
        ? buildTrainRequest(records)
        : task === "backtest"
          ? buildBacktestRequest(records)
          : task === "drift"
            ? buildDriftRequest(records)
            : buildForecastRequest(records);

    const pre = validatePreflight(records, { task, req });
    if (!pre.ok) {
      const i18nKey = errorCodeToI18nKey(pre.error_code);
      const parts = [
        t("client.err.preflight_prefix"),
        `error_code=${pre.error_code}`,
        i18nKey ? `i18n=${i18nKey}` : null,
        pre.message ? `message=${pre.message}` : null,
        pre.details ? `details=${JSON.stringify(pre.details)}` : null,
      ].filter(Boolean);
      showError(parts.join("\n"));
      setStatusI18n(runStatusEl, "status.failed", null, "error");
      lastValidatedSignature = null;
      return;
    }

    const selectedModel = task === "backtest" ? lookupModelById(req.model_id) : null;
    const activeAlgoId = task === "train" ? req.algo || null : selectedModel?.algo || null;
    lastRunContext = {
      task,
      algo_id: activeAlgoId,
      algo_display: getTrainAlgoDisplay(activeAlgoId) || selectedModel?.algo_display || null,
      model_id: task === "forecast" || task === "backtest" ? req.model_id || null : null,
      horizon: task === "train" || task === "drift" ? null : req.horizon ?? null,
      frequency: task === "forecast" ? (req.frequency || inferredFrequency || null) : null,
      quantiles: task === "forecast" ? (req.quantiles || null) : null,
      level: task === "forecast" ? (req.level || null) : null,
      series_count: Number.isFinite(Number(lastDataStats?.seriesCount)) ? Number(lastDataStats.seriesCount) : null,
      record_count: Number.isFinite(Number(lastDataStats?.n)) ? Number(lastDataStats.n) : null,
      metric: task === "backtest" ? req.metric || null : null,
      folds: task === "backtest" ? req.folds || null : null,
      training_hours: task === "train" ? req.training_hours || null : null,
    };

    const estimate = getRunPointsEstimate(task);
    if (estimate.canEstimate) {
      const isOverLimit = estimate.pointsEstimate > estimate.limit;
      const requiresConfirm = isOverLimit && (!runCostConfirmedInSession || estimate.pointsEstimate > runCostConfirmedPoints);
      if (requiresConfirm) {
        const ok = window.confirm(
          t("confirm.run_cost", {
            points: estimate.pointsEstimate,
            remaining: estimate.remaining,
            limit: estimate.limit,
          }),
        );
        if (!ok) {
          setRunUiBusy(false);
          setRunState("cancelled", { mode: runState.mode, jobId: runState.jobId, progress: 0 });
          setStatusI18n(runStatusEl, "status.cancelled", null, "warn");
          updateResultsStatus();
          return;
        }
        runCostConfirmedInSession = true;
        runCostConfirmedPoints = estimate.pointsEstimate;
      }
    }

    setStatusI18n(runStatusEl, "status.running", null, "info");

    const apiBaseUrl = normalizeApiBaseUrl(apiBaseUrlEl?.value || "");
    persistApiBaseUrl(apiBaseUrl);
    updateApiBaseUrlWarning();
    if (!ensureRemoteBaseUrlAcknowledged(apiBaseUrl)) {
      setRunUiBusy(false);
      setRunState("cancelled", { mode: runState.mode, jobId: runState.jobId, progress: 0 });
      setStatusI18n(runStatusEl, "status.cancelled", null, "warn");
      updateResultsStatus();
      return;
    }

    const mode = modeEl.value;
    setRunState("running", { mode });
    updateResultsStatus();

    let result;
    if (mode === "job") {
      result = await runJob(task, req, currentRunAbort.signal, { startedMessageKey: "info.job_started" });
    } else {
      try {
        result = await runSync(task, req, currentRunAbort.signal);
      } catch (e) {
        if (isCost01ErrorMessage(e)) {
          // Contract: COST01 means "too large for sync". Auto-fallback to async jobs.
          result = await runJob(task, req, currentRunAbort.signal, { startedMessageKey: "info.job_started_large_input" });
        } else {
          throw e;
        }
      }
    }

    renderResult(task, result);

    if (task === "train" && pendingAutoForecastAfterTrain) {
      pendingAutoForecastAfterTrain = false;
      const chainTask = "forecast";
      const chainReq = buildForecastRequest(records);
      const chainEstimate = getRunPointsEstimate(chainTask);
      if (chainEstimate.canEstimate) {
        const isOverLimit = chainEstimate.pointsEstimate > chainEstimate.limit;
        const requiresConfirm =
          isOverLimit && (!runCostConfirmedInSession || chainEstimate.pointsEstimate > runCostConfirmedPoints);
        if (requiresConfirm) {
          const ok = window.confirm(
            t("confirm.run_cost", {
              points: chainEstimate.pointsEstimate,
              remaining: chainEstimate.remaining,
              limit: chainEstimate.limit,
            }),
          );
          if (!ok) {
            setStatusI18n(runStatusEl, "status.train_handoff_ready", { model_id: modelIdEl.value || "" }, "warn");
            setRunState("done", { mode, jobId: runState.jobId, progress: 100 });
            updateResultsStatus();
            return;
          }
          runCostConfirmedInSession = true;
          runCostConfirmedPoints = chainEstimate.pointsEstimate;
        }
      }

      setStatusI18n(runStatusEl, "status.auto_forecast_running", null, "info");
      setRunState("running", { mode, jobId: null, progress: 0 });
      updateResultsStatus();

      // Also show a chart immediately for the chained forecast.
      renderInputPreviewChart(records);

      let chainResult;
      if (mode === "job") {
        chainResult = await runJob(chainTask, chainReq, currentRunAbort.signal, { startedMessageKey: "info.job_started" });
      } else {
        try {
          chainResult = await runSync(chainTask, chainReq, currentRunAbort.signal);
        } catch (e) {
          if (isCost01ErrorMessage(e)) {
            chainResult = await runJob(chainTask, chainReq, currentRunAbort.signal, { startedMessageKey: "info.job_started_large_input" });
          } else {
            throw e;
          }
        }
      }

      renderResult(chainTask, chainResult);
      setStatusI18n(runStatusEl, "status.auto_forecast_done", null, "success");
      sampleLoadArmedForAutoChain = false;
      updateTaskUi();
    }

    if (task === "forecast" && (!Array.isArray(result?.forecasts) || result.forecasts.length === 0)) {
      setStatusI18n(runStatusEl, "status.done_no_points", null, "warn");
    } else {
      setStatusI18n(runStatusEl, "status.done", null, "success");
    }
    setRunState("done", { mode, jobId: runState.jobId, progress: 100 });
    updateResultsStatus();
  } catch (e) {
    const name = String(e?.name || "");
    if (name === "AbortError") {
      showError("");
      setStatusI18n(runStatusEl, "status.cancelled", null, "warn");
      setRunState("cancelled", { mode: runState.mode, jobId: runState.jobId, progress: 0 });
      updateResultsStatus();
    } else {
      showError(String(e?.message || e));
      setStatusI18n(runStatusEl, "status.failed", null, "error");
      setRunState("failed", { mode: runState.mode, jobId: runState.jobId, progress: 0 });
      updateResultsStatus();
    }
  } finally {
    setRunUiBusy(false);
    currentRunAbort = null;
    autoForecastAfterTrainRequested = false;
  }
  updateStepNavHighlight();
}

function onCancelRun() {
  if (!currentRunAbort) return;
  try {
    currentRunAbort.abort();
  } catch {
    // ignore
  }
}

async function onCopyError() {
  const text = (errorBoxEl.textContent || "").trim();
  if (!text) return;
  try {
    await copyTextToClipboard(text);
    setStatusI18n(copyStatusEl, "status.copied", null, "success");
  } catch {
    setStatusI18n(copyStatusEl, "status.copy_failed", null, "error");
  }
}

async function onCopyRequestId() {
  const text = String(lastRequestId || "").trim();
  if (!text) return;
  try {
    await copyTextToClipboard(text);
    setStatusI18n(shareStatusEl, "status.copied", null, "success");
  } catch {
    setStatusI18n(shareStatusEl, "status.copy_failed", null, "error");
  }
}

async function onHealth() {
  setStatusI18n(healthStatusEl, "status.health_checking", null, "info");
  try {
    const { body: j } = await apiClient.checkHealth();
    lastHealthOk = true;
    lastAfnoReady = String(j?.afnocg2 || "").trim().toLowerCase() === "ready";
    setStatusI18n(healthStatusEl, "status.health_ok", { status: j?.status || "" }, "success");
    void syncModelsFromServer({ silent: false });
  } catch {
    lastHealthOk = false;
    lastAfnoReady = null;
    setStatusI18n(healthStatusEl, "status.health_fail", null, "error");
  }
  updateStepNavHighlight();
}

function onClear() {
  csvFileEl.value = "";
  jsonInputEl.value = "";
  setDataSource("");
  csvPrecheckState = { state: "unknown", message: "" };
  lastRecords = null;
  lastInferredFrequency = null;
  lastDataSignature = null;
  lastSyncedSignature = null;
  lastValidatedSignature = null;
  paramsLinked = true;
  paramsDirty = false;
  sampleLoadArmedForAutoChain = true;
  autoForecastAfterTrainRequested = false;
  pendingAutoForecastAfterTrain = false;
  setRunState("idle");
  setStatus(dataStatusEl, "");
  setStatus(runStatusEl, "");
  updateParamsLinkUi({ hasData: false, changed: false });
  showError("");
  showWarn("");
  clearFieldHighlights();
  syncInputLocks();
  updateCsvFileName();
  clearDataAnalysis();
  clearResults();
}

async function onLoadSample() {
  const key = String(sampleTypeEl?.value || "fd004_rul_forecast_unit01");
  const sampleMeta = SAMPLE_LIBRARY[key] || SAMPLE_LIBRARY.fd004_rul_forecast_unit01;
  const sample = sampleMeta?.remoteProfile ? (await apiClient.getCmapssFd004Sample(sampleMeta.remoteProfile)).body : sampleMeta;
  setDataSource("sample");
  csvFileEl.value = "";
  jsonInputEl.value = JSON.stringify(sample.records, null, 2);
  lastRecords = sample.records;
  lastInferredFrequency = null;
  sampleLoadArmedForAutoChain = false;
  paramsLinked = true;
  paramsDirty = false;
  lastDataSignature = computeDataSignature(sample.records);
  lastSyncedSignature = lastDataSignature;
  setRunState("idle");
  taskEl.value = SAMPLE_LOAD_DEFAULT_TASK;
  updateTaskUi();
  horizonEl.value = String(sample.horizon || 3);
  frequencyEl.value = sample.frequency || "";
  missingPolicyEl.value = sample.missing_policy || "ignore";
  quantilesEl.value = sample.quantiles || "";
  levelEl.value = sample.level || "";
  if (valueUnitEl) valueUnitEl.value = sample.value_unit || "cycles";
  if (sample.chartType && resultsChartTypeEl) {
    resultsChartTypeEl.value = sample.chartType;
  }
  syncInputLocks();
  updateCsvFileName();
  setStatusI18n(dataStatusEl, "status.sample_loaded", null, "success");
  if (!isConnectionReady()) {
    setStatusI18n(runStatusEl, "status.sample_loaded_need_api_key", null, "warn");
  } else {
    const task = currentTask();
    const runKey = task === "train" ? "action.run_train_and_forecast" : task === "backtest" ? "action.run_backtest" : "action.run_forecast";
    setStatusI18n(runStatusEl, "status.sample_loaded_ready_run", { action: t(runKey) }, "success");
  }
  showError("");
  showWarn("");
  clearDataAnalysis();
  clearResults();
  updateParamsLinkUi({ hasData: true, changed: false });
  await onValidate();
}

jsonInputEl.addEventListener("input", () => {
  if (String(jsonInputEl.value || "").trim()) {
    if (currentDataSource() !== "sample") setDataSource("json");
  }
  sampleLoadArmedForAutoChain = false;
  updateTaskUi();
  syncInputLocks();
});

csvFileEl.addEventListener("change", async () => {
  if (csvFileEl.files?.[0]) setDataSource("csv");
  await syncCsvPrecheckStateFromFile();
  sampleLoadArmedForAutoChain = false;
  updateTaskUi();
  syncInputLocks();
  if (csvPrecheckState.state === "invalid" && csvPrecheckState.message) {
    setStatus(dataStatusEl, csvPrecheckState.message);
    showError(csvPrecheckState.message);
  }
});

if (dataSourceEl) {
  dataSourceEl.addEventListener("change", () => {
    showError("");
    showWarn("");
    invalidateValidationState();
    syncInputLocks();
    updateWarningsForCurrentInputs();
  });
}

if (dataSourcePickerEl) {
  if (dataSourceBtnCsvEl) dataSourceBtnCsvEl.addEventListener("click", () => setDataSource("csv"));
  if (dataSourceBtnJsonEl) dataSourceBtnJsonEl.addEventListener("click", () => setDataSource("json"));
  if (dataSourceBtnSampleEl) dataSourceBtnSampleEl.addEventListener("click", () => setDataSource("sample"));
}

if (exploreChartTypeEl) {
  exploreChartTypeEl.addEventListener("change", () => {
    if (exploreTemplateEl) exploreTemplateEl.value = "manual";
    refreshExploreFromState();
  });
}
if (exploreFieldXEl) {
  exploreFieldXEl.addEventListener("change", () => {
    refreshExploreFromState();
  });
}
if (exploreFieldYEl) {
  exploreFieldYEl.addEventListener("change", () => {
    refreshExploreFromState();
  });
}

if (exploreDatasetEl) {
  exploreDatasetEl.addEventListener("change", () => {
    refreshExploreFromState();
  });
}

if (exploreTemplateEl) {
  exploreTemplateEl.addEventListener("change", () => {
    refreshExploreFromState();
  });
}

wireCoreEvents({
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
  getLastResult: () => lastResult,
  getLastTask: () => lastTask,
  getLastForecasts: () => lastForecasts,
  getLastRecords: () => lastRecords,
  getLastInferredFrequency: () => lastInferredFrequency,
  setLastForecastSeriesId: (value) => {
    lastForecastSeriesId = value;
  },
  setParamsLinked: (value) => {
    paramsLinked = value;
  },
  setParamsDirty: (value) => {
    paramsDirty = value;
  },
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
});

// Initialize uncertainty mode UI (if supported).
try {
  if (uncertaintyModeEl) {
    uncertaintyModeEl.value = normalizeUncertaintyMode(
      uncertaintyModeEl.value || inferUncertaintyModeFromInputs(),
    );
  }
  applyUncertaintyModeUi({ applyDefaults: false, clearInactive: false });
} catch {
  // ignore
}

async function init() {
  enforceTrainAlgoPolicy();
  removeLegacyDataSourceResetUi(document);
  ensureLegacyDataSourceResetObserver();

  let persist = false;
  try {
    persist = localStorage.getItem(LS_API_KEY_PERSIST) === "1";
  } catch {
    persist = false;
  }
  rememberApiKeyEl.checked = persist;
  const saved = loadStoredApiKey();
  if (saved) apiKeyEl.value = saved;

  // Restore API base URL (optional)
  if (apiBaseUrlEl) {
    apiBaseUrlEl.value = loadApiBaseUrl();
    apiBaseUrlEl.addEventListener("input", () => {
      persistApiBaseUrl(apiBaseUrlEl.value);
      updateApiBaseUrlWarning();
    });
    updateApiBaseUrlWarning();
  }

  const hashState = decodeStateFromHash(window.location.hash);
  if (docsLinkEl) {
    docsLinkEl.setAttribute("href", resolveDocsHref());
  }
  const langFromHash = normalizeLang(hashState?.lang);
  await setLanguage(langFromHash || getInitialLang(), { persist: false });
  setDensityMode(hashState?.density || getInitialDensityMode(), { persist: false });
  wireInitPreferenceEvents({
    densityModeEl,
    langSelectEl,
    setDensityMode,
    updateStepNavHighlight,
    setLanguage,
  });

  updateTaskUi();
  updateParamsLinkUi({ hasData: false, changed: false });
  showError("");
  if (billingSyncMaxPointsEl) billingSyncMaxPointsEl.value = String(getBillingSyncMaxPoints());
  updateBillingUi();
  updateHelpAdvice();
  updateCsvFileName();
  renderJobHistory();
  renderModels();
  if (hashState) {
    applyShareState(hashState);
  }
  // Sync Data 3-choice UI after any restores.
  syncInputLocks();
  updateStepNavHighlight();
  requestAnimationFrame(updateStepNavHighlight);

  toggleApiKeyEl.textContent = t("action.show_api_key");
}

let forecastingGuiReady = false;

function exposeForecastingGuiHooks() {
  if (typeof window === "undefined") return;

  window.__ARCAYF_FORECASTING_GUI__ = {
    ready: forecastingGuiReady,
    error: null,
    markReady() {
      forecastingGuiReady = true;
      if (window.__ARCAYF_FORECASTING_GUI__) {
        window.__ARCAYF_FORECASTING_GUI__.ready = true;
        window.__ARCAYF_FORECASTING_GUI__.error = null;
      }
      window.dispatchEvent(new CustomEvent("arcayf:forecasting-gui-ready"));
    },
    getState() {
      return {
        ready: forecastingGuiReady,
        task: currentTask(),
        runStatus: String(runStatusEl?.textContent || ""),
        dataStatus: String(dataStatusEl?.textContent || ""),
        healthStatus: String(healthStatusEl?.textContent || ""),
        driftBaselineStatus: String(driftBaselineStatusEl?.textContent || ""),
        runButtonDisabled: !!runForecastBtnEl?.disabled,
        lastValidatedSignature,
      };
    },
    async setApiKey(value) {
      if (!apiKeyEl) return;
      apiKeyEl.value = String(value || "");
      apiKeyEl.dispatchEvent(new Event("input", { bubbles: true }));
    },
    async runHealthCheck() {
      await onHealth();
      return String(healthStatusEl?.textContent || "");
    },
    async loadJsonRecords(records) {
      setDataSource("json");
      if (jsonInputEl) {
        jsonInputEl.value = JSON.stringify(Array.isArray(records) ? records : []);
        jsonInputEl.dispatchEvent(new Event("input", { bubbles: true }));
      }
      syncInputLocks();
      invalidateValidationState();
      updateWarningsForCurrentInputs();
      await onValidate();
      return {
        dataStatus: String(dataStatusEl?.textContent || ""),
        lastValidatedSignature,
      };
    },
    async prepareTrain(options = {}) {
      const cfg = options && typeof options === "object" ? options : {};
      if (taskEl) {
        taskEl.value = "train";
        taskEl.dispatchEvent(new Event("change", { bubbles: true }));
      }
      updateTaskUi();
      const mappedAlgo = normalizeTrainAlgoValue(cfg.algo || cfg.algoDisplay || "");
      if (trainAlgoEl && mappedAlgo) {
        trainAlgoEl.value = mappedAlgo;
        trainAlgoEl.dispatchEvent(new Event("change", { bubbles: true }));
      }
      if (trainingHoursEl && cfg.trainingHours != null) {
        trainingHoursEl.value = String(cfg.trainingHours);
        trainingHoursEl.dispatchEvent(new Event("input", { bubbles: true }));
      }
      updateRunGate();
      return {
        reasons: getRunBlockReasons(),
        task: currentTask(),
      };
    },
    async prepareDrift() {
      if (taskEl) {
        taskEl.value = "drift";
        taskEl.dispatchEvent(new Event("change", { bubbles: true }));
      } else {
        updateTaskUi();
      }
      if (isConnectionReady() && !currentDriftBaselineState().checked) {
        await refreshDriftBaselineStatus({ silent: true });
      }
      updateRunGate();
      return {
        reasons: getRunBlockReasons(),
        reasonKeys: getRunBlockReasonKeysFromState(),
        task: currentTask(),
        runButtonDisabled: !!runForecastBtnEl?.disabled,
      };
    },
    async saveCurrentAsDriftBaseline() {
      await onPersistDriftBaseline();
      return {
        persisted: driftBaselineController.wasLastPersisted(),
        reasons: getRunBlockReasons(),
        reasonKeys: getRunBlockReasonKeysFromState(),
        driftBaselineStatus: String(driftBaselineStatusEl?.textContent || ""),
        runButtonDisabled: !!runForecastBtnEl?.disabled,
      };
    },
    async refreshDriftBaseline() {
      await onRefreshDriftBaseline();
      return {
        reasons: getRunBlockReasons(),
        reasonKeys: getRunBlockReasonKeysFromState(),
        driftBaselineStatus: String(driftBaselineStatusEl?.textContent || ""),
        runButtonDisabled: !!runForecastBtnEl?.disabled,
      };
    },
    getRunReasons() {
      return getRunBlockReasons();
    },
    getRunReasonKeys() {
      return getRunBlockReasonKeysFromState();
    },
    async runTrainForecast() {
      await onRun();
      return {
        runStatus: String(runStatusEl?.textContent || ""),
        resultChartReady: !!document.querySelector("#resultsSparkline polyline.sparkLine"),
      };
    },
  };
}

exposeForecastingGuiHooks();

void init()
  .then(() => {
    if (window.__ARCAYF_FORECASTING_GUI__?.markReady) {
      window.__ARCAYF_FORECASTING_GUI__.markReady();
    }
  })
  .catch((error) => {
    if (window.__ARCAYF_FORECASTING_GUI__) {
      window.__ARCAYF_FORECASTING_GUI__.error = String(error?.message || error || "init_failed");
    }
    try {
      console.error("forecasting_gui_init_failed", error);
    } catch {
      // ignore
    }
  });
wireGlobalUiEvents({
  windowObj: window,
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
  setLastHealthOk: (value) => {
    lastHealthOk = value;
  },
  setStatusI18n,
  updateRunGate,
});
