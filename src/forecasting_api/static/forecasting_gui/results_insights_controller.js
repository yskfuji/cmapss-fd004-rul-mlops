function appendChip(container, roleText, keyText) {
  if (!container) return;
  const doc = container.ownerDocument || document;
  const chip = doc.createElement("span");
  chip.className = "chip";

  const role = doc.createElement("span");
  role.className = "chipRole";
  role.textContent = String(roleText ?? "");

  const key = doc.createElement("span");
  key.className = "chipKey";
  key.textContent = String(keyText ?? "");

  chip.appendChild(role);
  chip.appendChild(key);
  container.appendChild(chip);
}

function formatBenchmarkMetric(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(3);
}

function normalizeBenchmarkRows(snapshot) {
  const rows = Array.isArray(snapshot?.rows) ? snapshot.rows : [];
  return rows
    .map((row) => ({
      model: String(row?.model || "-").trim() || "-",
      rmse: Number.isFinite(Number(row?.rmse)) ? Number(row.rmse) : null,
      mae: Number.isFinite(Number(row?.mae)) ? Number(row.mae) : null,
      source: String(row?.source || "-").trim() || "-",
      scope: String(row?.scope || "-").trim() || "-",
    }))
    .filter((row) => row.model && row.model !== "-");
}

function buildCurrentRunBenchmarkRow(task, result, t, normalizeTask) {
  if (normalizeTask(task) !== "backtest") return null;
  const metrics = result?.metrics && typeof result.metrics === "object" ? result.metrics : null;
  if (!metrics) return null;
  const rmse = Number(metrics.rmse);
  const mae = Number(metrics.mae);
  if (!Number.isFinite(rmse) && !Number.isFinite(mae)) return null;
  return {
    model: t("results.benchmark.current_run_label"),
    rmse: Number.isFinite(rmse) ? rmse : null,
    mae: Number.isFinite(mae) ? mae : null,
    source: t("results.benchmark.current_run_source"),
    scope: t("results.benchmark.current_run_scope"),
    isCurrentRun: true,
  };
}

function fillBenchmarkTable(tableBodyEl, rows) {
  if (!tableBodyEl) return;
  const doc = tableBodyEl.ownerDocument || document;
  tableBodyEl.innerHTML = "";
  rows.forEach((row, index) => {
    const tr = doc.createElement("tr");
    if (row.isCurrentRun) tr.dataset.currentRun = "true";
    const cells = [String(index + 1), row.model, formatBenchmarkMetric(row.rmse), formatBenchmarkMetric(row.mae), row.source, row.scope];
    cells.forEach((value) => {
      const td = doc.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    tableBodyEl.appendChild(tr);
  });
}

export function createResultsInsightsController({
  apiClient,
  elements,
  countValidSeriesPoints,
  getBenchmarkState,
  getLastRecords,
  getLastResult,
  getLastRunContext,
  getLastTask,
  getLastForecastSeriesId,
  normalizeTask,
  setBenchmarkState,
  setVisible,
  t,
}) {
  const {
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
  } = elements;

  function clear() {
    setVisible(resultsSummaryEl, false);
    setVisible(resultsEvidenceEl, false);
    setVisible(resultsBenchmarkEl, false);
    if (resultsSummaryChipsEl) resultsSummaryChipsEl.innerHTML = "";
    if (resultsSummaryNoteEl) resultsSummaryNoteEl.textContent = "";
    if (resultsEvidenceQualityEl) resultsEvidenceQualityEl.textContent = "";
    if (resultsEvidenceUncertaintyEl) resultsEvidenceUncertaintyEl.textContent = "";
    if (resultsEvidenceScopeEl) resultsEvidenceScopeEl.textContent = "";
    if (resultsEvidenceLimitEl) resultsEvidenceLimitEl.textContent = "";
    if (resultsBenchmarkSummaryEl) resultsBenchmarkSummaryEl.textContent = "";
    if (resultsBenchmarkNoteEl) resultsBenchmarkNoteEl.textContent = "";
    if (resultsBenchmarkTableBodyEl) resultsBenchmarkTableBodyEl.innerHTML = "";
  }

  function renderSummary(task, { result = null, runContext = null } = {}) {
    if (!resultsSummaryEl || !resultsSummaryChipsEl) return;
    const ctx = runContext || getLastRunContext();
    const src = result || getLastResult();
    if (!ctx) {
      setVisible(resultsSummaryEl, false);
      return;
    }

    resultsSummaryChipsEl.innerHTML = "";
    let chipCount = 0;
    const push = (labelKey, value) => {
      const safe = value === null || value === undefined || value === "" ? "-" : String(value);
      appendChip(resultsSummaryChipsEl, t(labelKey), safe);
      chipCount += 1;
    };

    if (task === "forecast") {
      push("results.summary.model", ctx.model_id || "-");
      push("results.summary.series", ctx.series_count);
      push("results.summary.horizon", ctx.horizon);
      push("results.summary.frequency", ctx.frequency || "-");
      push("results.summary.quantiles", Array.isArray(ctx.quantiles) && ctx.quantiles.length ? ctx.quantiles.join(", ") : "-");
      push("results.summary.level", Array.isArray(ctx.level) && ctx.level.length ? ctx.level.join(", ") : "-");
    } else if (task === "backtest") {
      push("results.summary.series", ctx.series_count);
      push("results.summary.horizon", ctx.horizon);
      push("results.summary.metric", ctx.metric || "-");
      push("results.summary.folds", ctx.folds || "-");
    } else if (task === "train") {
      push("results.summary.series", ctx.series_count);
      push("results.summary.records", ctx.record_count);
      push("results.summary.training_hours", ctx.training_hours || "-");
    } else if (task === "drift") {
      push("results.summary.records", ctx.record_count);
      push("results.summary.series", ctx.series_count);
      push("results.summary.drift_severity", src?.severity || "-");
      const top = Array.isArray(src?.feature_reports) ? src.feature_reports[0] : null;
      push("results.summary.drift_selected_bin_count", top?.baseline_selected_bin_count ?? "-");
    }

    setVisible(resultsSummaryEl, chipCount > 0);
    if (resultsSummaryNoteEl) resultsSummaryNoteEl.textContent = "";
  }

  function renderEvidence(task, { seriesPoints = null, result = null, runContext = null, records = null, selectedSeriesId = null } = {}) {
    if (!resultsEvidenceEl) return;

    const mode = normalizeTask(task);
    const src = result || getLastResult();
    const ctx = runContext || getLastRunContext();
    const inputRecords = Array.isArray(records) ? records : Array.isArray(getLastRecords()) ? getLastRecords() : [];
    const activeSeriesId = selectedSeriesId || getLastForecastSeriesId() || "-";
    let quality = "";
    let uncertainty = "";
    let scope = "";
    let limit = "";

    if (mode === "forecast") {
      const points = Array.isArray(seriesPoints) ? seriesPoints : [];
      const total = points.length;
      const { valid } = countValidSeriesPoints(points);
      const covered = points.filter((p) => {
        const hasQuantiles = p?.quantiles && typeof p.quantiles === "object" && Object.keys(p.quantiles).length > 0;
        const hasIntervals = Array.isArray(p?.intervals) && p.intervals.length > 0;
        return hasQuantiles || hasIntervals;
      }).length;
      const coverage = total > 0 ? Math.round((covered / total) * 100) : 0;

      const seriesCount = Number(ctx?.series_count);
      const horizon = Number(ctx?.horizon);
      const seriesShown = Number.isFinite(seriesCount) && seriesCount > 0 ? seriesCount : "-";
      const horizonShown = Number.isFinite(horizon) && horizon > 0 ? horizon : "-";

      quality = t("results.evidence.quality_forecast", { valid, total });
      const calib = src?.calibration && typeof src.calibration === "object" ? src.calibration : null;
      const method = calib ? String(calib?.method || "") : "";
      if (method.startsWith("split_conformal")) {
        const n = Number(calib?.residuals_n);
        const qhat = Number(calib?.qhat);
        const calibCoverageRaw = Number(calib?.coverage);
        const calibCoveragePct = Number.isFinite(calibCoverageRaw) ? Math.round(calibCoverageRaw * 100) : null;
        const scaling = String(calib?.scaling || "");
        uncertainty = t("results.evidence.uncertainty_forecast_conformal", {
          n: Number.isFinite(n) ? n : "-",
          qhat: Number.isFinite(qhat) ? qhat : "-",
          calib_coverage: calibCoveragePct === null ? "-" : calibCoveragePct,
          scaling: scaling || "-",
          covered,
          total,
          field_coverage: coverage,
        });
      } else {
        uncertainty = t("results.evidence.uncertainty_forecast", { covered, total, coverage });
      }
      scope = t("results.evidence.scope_forecast", { series: seriesShown, horizon: horizonShown, series_id: activeSeriesId });

      if (total <= 0) {
        limit = t("results.evidence.limit_no_points");
      } else if (total < 5) {
        limit = t("results.evidence.limit_short_series", { total });
      } else if (coverage < 60) {
        limit = t("results.evidence.limit_low_uncertainty", { coverage });
      } else {
        limit = t("results.evidence.limit_assumption");
      }
    } else if (mode === "backtest") {
      const metrics = src?.metrics && typeof src.metrics === "object" ? Object.keys(src.metrics).length : 0;
      const bySeries = Array.isArray(src?.by_series) ? src.by_series.length : 0;
      const byHorizon = Array.isArray(src?.by_horizon) ? src.by_horizon.length : 0;
      const byFold = Array.isArray(src?.by_fold) ? src.by_fold.length : 0;
      const folds = Number(ctx?.folds);
      const horizon = Number(ctx?.horizon);

      quality = t("results.evidence.quality_backtest", { metrics, by_series: bySeries, by_horizon: byHorizon, by_fold: byFold });
      uncertainty = t("results.evidence.uncertainty_backtest");
      scope = t("results.evidence.scope_backtest", {
        folds: Number.isFinite(folds) && folds > 0 ? folds : "-",
        horizon: Number.isFinite(horizon) && horizon > 0 ? horizon : "-",
      });
      limit = t("results.evidence.limit_backtest");
    } else if (mode === "train") {
      const modelId = String(src?.model_id || ctx?.model_id || "-");
      const recordsCount = Number(ctx?.record_count);
      const hours = Number(ctx?.training_hours);

      quality = t("results.evidence.quality_train", { model_id: modelId });
      uncertainty = t("results.evidence.uncertainty_train");
      scope = t("results.evidence.scope_train", {
        records: Number.isFinite(recordsCount) && recordsCount > 0 ? recordsCount : "-",
        hours: Number.isFinite(hours) && hours > 0 ? hours : "-",
      });
      limit = t("results.evidence.limit_train");
    } else {
      const featureReports = Array.isArray(src?.feature_reports) ? src.feature_reports : [];
      const top = featureReports[0] && typeof featureReports[0] === "object" ? featureReports[0] : null;
      const severity = String(src?.severity || "-");
      const driftScore = Number(src?.drift_score);
      const topFeature = String(top?.feature || "-");
      const requested = Number(top?.baseline_requested_bin_count);
      const selected = Number(top?.baseline_selected_bin_count);

      quality = t("results.evidence.quality_drift", {
        severity,
        features: featureReports.length,
        score: Number.isFinite(driftScore) ? driftScore.toFixed(3) : "-",
      });
      uncertainty = t("results.evidence.uncertainty_drift", {
        feature: topFeature,
        requested: Number.isFinite(requested) ? requested : "-",
        selected: Number.isFinite(selected) ? selected : "-",
      });
      scope = t("results.evidence.scope_drift", {
        records: Number(ctx?.record_count) || "-",
        series: Number(ctx?.series_count) || "-",
      });
      limit = t("results.evidence.limit_drift");
    }

    const exogKeys = new Set();
    let exogRows = 0;
    for (const rec of inputRecords.slice(0, 2000)) {
      const x = rec?.x && typeof rec.x === "object" ? rec.x : null;
      if (!x) continue;
      const keys = Object.keys(x);
      if (!keys.length) continue;
      exogRows += 1;
      for (const k of keys) {
        if (k) exogKeys.add(String(k));
      }
    }
    const exogCoverage = inputRecords.length > 0 ? Math.round((exogRows / inputRecords.length) * 100) : 0;
    const exogKeyList = [...exogKeys].sort();
    const exogShown = exogKeyList.slice(0, 6);
    const exogMore = Math.max(0, exogKeyList.length - exogShown.length);
    const exogKeysLabel = exogKeyList.length ? `${exogShown.join(", ")}${exogMore ? ` (+${exogMore})` : ""}` : "-";

    let exogLine = "";
    if (exogKeyList.length > 0) {
      exogLine = t("results.evidence.exog_present", { keys: exogKeysLabel, coverage: exogCoverage });
      const algo = String(ctx?.algo_id || "").trim().toLowerCase();
      const univariate = new Set(["naive", "ridge_lags_v1", "cifnocg2", "afnocg2", "afnocg3", "afnocg3_v1"]);
      if (!algo || univariate.has(algo)) {
        exogLine += `\n${t("results.evidence.exog_currently_ignored")}`;
      }
    } else {
      exogLine = t("results.series.note_exogenous");
    }

    if (exogLine) {
      scope = scope ? `${scope}\n${exogLine}` : exogLine;
    }

    const recCount = Number(ctx?.record_count);
    const seriesCountShown = Number(ctx?.series_count);
    const detailBits = [];
    if (Number.isFinite(recCount) && recCount > 0) detailBits.push(`records=${recCount}`);
    if (Number.isFinite(seriesCountShown) && seriesCountShown > 0) detailBits.push(`series=${seriesCountShown}`);
    if (ctx?.model_id) detailBits.push(`model=${ctx.model_id}`);
    if (detailBits.length > 0) {
      const detailLine = `detail: ${detailBits.join(", ")}`;
      quality = quality ? `${quality}\n${detailLine}` : detailLine;
    }

    if (resultsEvidenceQualityEl) resultsEvidenceQualityEl.textContent = quality;
    if (resultsEvidenceUncertaintyEl) resultsEvidenceUncertaintyEl.textContent = uncertainty;
    if (resultsEvidenceScopeEl) resultsEvidenceScopeEl.textContent = scope;
    if (resultsEvidenceLimitEl) resultsEvidenceLimitEl.textContent = limit;
    setVisible(resultsEvidenceEl, true);
  }

  function renderBenchmark(task, { result = null, onDeferredRefresh = null } = {}) {
    if (!resultsBenchmarkEl) return;

    const { cache, pending } = getBenchmarkState();
    const snapshot = cache;
    const rows = normalizeBenchmarkRows(snapshot);
    const currentRunRow = buildCurrentRunBenchmarkRow(task, result || getLastResult(), t, normalizeTask);
    const mergedRows = currentRunRow ? [currentRunRow, ...rows] : rows.slice();
    mergedRows.sort((a, b) => {
      if (a.isCurrentRun && !b.isCurrentRun) return -1;
      if (!a.isCurrentRun && b.isCurrentRun) return 1;
      const av = Number.isFinite(Number(a.rmse)) ? Number(a.rmse) : Number.POSITIVE_INFINITY;
      const bv = Number.isFinite(Number(b.rmse)) ? Number(b.rmse) : Number.POSITIVE_INFINITY;
      return av - bv;
    });

    const queueRefresh = () => {
      const current = getBenchmarkState();
      if (current.pending) return;
      const nextPending = apiClient
        .getCmapssFd004Benchmarks()
        .then((body) => {
          setBenchmarkState({ cache: body && typeof body === "object" ? body : { rows: [], notes: [] }, pending: null });
        })
        .catch(() => {
          setBenchmarkState({ cache: { rows: [], notes: [] }, pending: null });
        })
        .finally(() => {
          const latest = getLastResult();
          if (latest && typeof onDeferredRefresh === "function") onDeferredRefresh();
        });
      setBenchmarkState({ cache: current.cache, pending: nextPending });
    };

    if (!mergedRows.length) {
      if (!pending) {
        queueRefresh();
      }
      setVisible(resultsBenchmarkEl, false);
      return;
    }

    fillBenchmarkTable(resultsBenchmarkTableBodyEl, mergedRows);
    const leader = mergedRows.find((row) => Number.isFinite(Number(row.rmse)));
    if (resultsBenchmarkSummaryEl) {
      resultsBenchmarkSummaryEl.textContent = leader
        ? t("results.benchmark.summary_leader", { model: leader.model, rmse: formatBenchmarkMetric(leader.rmse) })
        : t("results.benchmark.summary_empty");
    }

    const notes = [];
    if (snapshot?.generated_at) {
      notes.push(t("results.benchmark.note_snapshot", { generated_at: snapshot.generated_at }));
    }
    if (currentRunRow) {
      notes.push(t("results.benchmark.current_run_note"));
    } else {
      notes.push(t("results.benchmark.note_run_backtest"));
    }
    if (resultsBenchmarkNoteEl) resultsBenchmarkNoteEl.textContent = notes.join("\n");
    setVisible(resultsBenchmarkEl, true);

    if (!snapshot && !pending) {
      queueRefresh();
    }
  }

  return {
    clear,
    renderBenchmark,
    renderEvidence,
    renderSummary,
  };
}