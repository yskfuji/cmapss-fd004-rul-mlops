export function createDataQualityController({
  constants,
  elements,
  getCurrentDataSource,
  getCurrentTask,
  getFrequencyValue,
  getHorizonValue,
  getLastValidatedSignature,
  getMissingPolicyValue,
  parseErrorText,
  formatErrorForDisplay,
  countMissingRequiredFields,
  parseFrequencyToMs,
  computeGapDetailsPerSeries,
  setLastDataGapCount,
  setLastDataMissing,
  setLastDataStats,
  setLastValidatedSignature,
  setRunGateExplained,
  setStatus,
  setText,
  setVisible,
  t,
  updateBillingUi,
  updateRunGate,
  updateSectionStatus,
  updateStepNavHighlight,
}) {
  const {
    GAP_SAMPLE_MAX_GAPS_PER_SERIES,
    GAP_SAMPLE_MAX_SERIES,
    LARGE_HORIZON_THRESHOLD,
  } = constants;
  const {
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
    fieldElements,
    gapsAcknowledgeEl,
    requestIdEl,
    requestIdWrapEl,
    warnBoxEl,
  } = elements;

  function clearFieldHighlights() {
    const doc = errorBoxEl?.ownerDocument || dataAnalysisEl?.ownerDocument || document;
    for (const node of doc.querySelectorAll(".field.error")) {
      node.classList.remove("error");
    }
  }

  function applyFieldHighlightsFromErrorText(text) {
    clearFieldHighlights();
    const parsed = parseErrorText(text);
    const code = String(parsed?.code || "").toUpperCase();
    if (!code) return;

    let firstField = null;
    const mark = (node) => {
      if (!node) return;
      const field = node.closest(".field");
      if (field) {
        field.classList.add("error");
        if (!firstField) firstField = field;
      }
    };

    if (code === "A12") mark(fieldElements.apiKeyEl);
    if (code === "A21") mark(fieldElements.billingSyncMaxPointsEl);
    if (code === "V01") {
      mark(fieldElements.csvFileEl);
      mark(fieldElements.jsonInputEl);
    }
    if (code === "V02") mark(fieldElements.frequencyEl);
    if (code === "V03") {
      mark(fieldElements.quantilesEl);
      mark(fieldElements.levelEl);
    }
    if (code === "V04") {
      mark(fieldElements.csvFileEl);
      mark(fieldElements.jsonInputEl);
    }
    if (code === "V05") mark(fieldElements.missingPolicyEl);
    if (code === "V06") {
      mark(fieldElements.csvFileEl);
      mark(fieldElements.jsonInputEl);
      mark(fieldElements.horizonEl);
    }
    if (code === "M01" || code === "M02") mark(fieldElements.modelIdEl);
    if (code === "J01" || code === "J02") mark(fieldElements.jobIdInputEl);

    const message = String(parsed?.message || text || "").toLowerCase();
    if (message.includes("insufficient training data for torch model") || message.includes("need >= 8 samples")) {
      mark(fieldElements.trainAlgoEl);
      mark(fieldElements.csvFileEl);
      mark(fieldElements.jsonInputEl);
    }

    if (firstField) firstField.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function showError(text) {
    const message = String(text || "").trim();
    errorBoxEl.hidden = !message;
    errorBoxEl.textContent = message ? formatErrorForDisplay(message) : "";
    if (errorActionsEl) errorActionsEl.hidden = !message;
    if (copyErrorBtnEl) copyErrorBtnEl.disabled = !message;
    setStatus(copyStatusEl, "");
    setVisible(requestIdWrapEl, false);
    setText(requestIdEl, "");
    applyFieldHighlightsFromErrorText(message);
  }

  function showWarn(text) {
    if (!warnBoxEl) return;
    warnBoxEl.hidden = !text;
    warnBoxEl.textContent = text || "";
  }

  function invalidateValidationState() {
    if (getLastValidatedSignature()) setLastValidatedSignature(null);
    if (analysisExploreEl) setVisible(analysisExploreEl, false);
    if (exploreStatusEl) setStatus(exploreStatusEl, "");
    if (exploreChartEl) exploreChartEl.innerHTML = "";
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
      for (const key of Object.keys(x)) {
        if (xKeySet.has(key)) continue;
        xKeySet.add(key);
        xKeys.push(key);
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

    const columns = ["series_id", "timestamp", "y", ...xKeys.map((key) => `x.${key}`)];
    const thead = dataPreviewTableEl.querySelector("thead");
    const tbody = dataPreviewTableEl.querySelector("tbody");
    if (!thead || !tbody) return;
    thead.innerHTML = "";
    tbody.innerHTML = "";
    const doc = dataPreviewTableEl.ownerDocument || document;

    const escapeHtml = (value) =>
      String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");

    const trHead = doc.createElement("tr");
    trHead.className = "dataPreviewRoleRow";
    for (const col of columns) {
      const th = doc.createElement("th");
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

    const trCols = doc.createElement("tr");
    trCols.className = "dataPreviewColumnsRow";
    for (const col of columns) {
      const th = doc.createElement("th");
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

    for (const rec of sorted.slice(0, 6)) {
      const tr = doc.createElement("tr");
      const x = rec?.x && typeof rec.x === "object" ? rec.x : null;
      for (const col of columns) {
        const td = doc.createElement("td");
        if (col === "series_id") td.textContent = String(rec?.series_id ?? "");
        else if (col === "timestamp") td.textContent = String(rec?.timestamp ?? "");
        else if (col === "y") td.textContent = String(rec?.y ?? "");
        else {
          const key = col.slice(2);
          const value = x ? x[key] : "";
          td.textContent = value === undefined ? "" : String(value);
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }

    setVisible(dataPreviewEl, true);
  }

  function clearDataAnalysis() {
    setVisible(dataAnalysisEl, false);
    setVisible(dataPreviewEl, false);
    setStatus(analysisSummaryEl, "");
    setStatus(analysisFieldsEl, "");
    setStatus(analysisTimestampsEl, "");
    setStatus(analysisGapsEl, "");
    setStatus(analysisGapsSampleEl, "");
    explorePanelController.clear();
    if (dataPreviewMetaEl) dataPreviewMetaEl.innerHTML = "";
    if (dataPreviewTableEl) {
      const thead = dataPreviewTableEl.querySelector("thead");
      const tbody = dataPreviewTableEl.querySelector("tbody");
      if (thead) thead.innerHTML = "";
      if (tbody) tbody.innerHTML = "";
    }
    setLastDataStats(null);
    setLastDataMissing(null);
    updateBillingUi();
    updateStepNavHighlight();
  }

  function updateWarningsForCurrentInputs() {
    const warnings = [];
    if (getCurrentTask() !== "train") {
      const horizon = Number(getHorizonValue());
      if (Number.isFinite(horizon) && horizon >= LARGE_HORIZON_THRESHOLD) {
        warnings.push(t("warn.large_horizon"));
      }
    }
    const source = getCurrentDataSource();
    const hasJson = (source === "json" || source === "sample") && !!(fieldElements.jsonInputEl?.value || "").trim();
    const hasCsv = source === "csv" && !!fieldElements.csvFileEl?.files?.[0];
    if (!hasJson && !hasCsv) {
      // no-op
    }
    showWarn(warnings.join("\n"));
    updateBillingUi();
    setRunGateExplained(false);
    updateRunGate();
  }

  function countSeriesWithGaps(records, stepMs) {
    const bySeries = new Map();
    for (const rec of records) {
      const seriesId = String(rec?.series_id ?? "");
      const ts = Date.parse(String(rec?.timestamp ?? ""));
      if (!Number.isFinite(ts)) continue;
      if (!bySeries.has(seriesId)) bySeries.set(seriesId, []);
      bySeries.get(seriesId).push(ts);
    }
    let count = 0;
    for (const timestamps of bySeries.values()) {
      const sorted = [...timestamps].sort((a, b) => a - b);
      let hasGap = false;
      for (let index = 1; index < sorted.length; index++) {
        if (sorted[index] - sorted[index - 1] > stepMs) {
          hasGap = true;
          break;
        }
      }
      if (hasGap) count += 1;
    }
    return count;
  }

  function updateDataAnalysis(records, { inferredFrequency = null } = {}) {
    if (!Array.isArray(records) || records.length === 0) {
      clearDataAnalysis();
      return;
    }

    const count = records.length;
    const seriesCount = new Set(records.map((record) => String(record?.series_id ?? ""))).size;
    setLastDataStats({ n: count, seriesCount });
    updateBillingUi();

    const parsedTimestamps = records
      .map((record) => Date.parse(String(record?.timestamp ?? "")))
      .filter((value) => Number.isFinite(value)).length;
    const frequencyValue = getFrequencyValue() || inferredFrequency || "";
    const frequencyShown = frequencyValue ? frequencyValue : "-";
    setStatus(analysisSummaryEl, t("analysis.summary", { n: count, s: seriesCount, frequency: frequencyShown }));

    const missing = countMissingRequiredFields(records);
    setLastDataMissing(missing);
    setStatus(
      analysisFieldsEl,
      t("analysis.fields", {
        series_id: missing.series_id,
        timestamp: missing.timestamp,
        y: missing.y,
      }),
    );
    setStatus(analysisTimestampsEl, t("analysis.timestamps", { ok: parsedTimestamps, n: count }));

    let gapIssue = false;
    const missingPolicy = getMissingPolicyValue();
    if (missingPolicy === "error") {
      const stepMs = parseFrequencyToMs(frequencyValue);
      if (stepMs) {
        const affected = countSeriesWithGaps(records, stepMs);
        setLastDataGapCount(affected);
        gapIssue = affected > 0;
        const gapsText = gapIssue ? t("analysis.gaps_detected", { s: affected }) : t("analysis.gaps_none");
        setStatus(analysisGapsEl, t("analysis.gaps", { gaps: gapsText }));
        if (gapIssue) {
          const details = computeGapDetailsPerSeries(records, stepMs, {
            maxSeries: GAP_SAMPLE_MAX_SERIES,
            maxGapsPerSeries: GAP_SAMPLE_MAX_GAPS_PER_SERIES,
          });
          const lines = [];
          for (const row of details) {
            const seriesId = String(row?.series_id ?? "");
            for (const gap of Array.isArray(row?.gaps) ? row.gaps : []) {
              const from = String(gap?.from ?? "");
              const to = String(gap?.to ?? "");
              const missingCount = Number(gap?.missing_count);
              const shownCount = Number.isFinite(missingCount) ? String(missingCount) : "-";
              lines.push(`series_id=${seriesId}: ${from} -> ${to} (missing_count=${shownCount})`);
            }
          }
          const samples = lines.length > 0 ? lines.join("\n") : "-";
          setStatus(analysisGapsSampleEl, t("analysis.gaps_sample", { max_series: GAP_SAMPLE_MAX_SERIES, samples }));
        } else {
          setStatus(analysisGapsSampleEl, "");
        }
      } else {
        setLastDataGapCount(null);
        setStatus(analysisGapsEl, t("analysis.gaps", { gaps: "-" }));
        setStatus(analysisGapsSampleEl, "");
      }
    } else {
      setLastDataGapCount(null);
      setStatus(analysisGapsEl, "");
      setStatus(analysisGapsSampleEl, "");
    }

    if (gapsAcknowledgeEl) setVisible(gapsAcknowledgeEl, gapIssue);
    if (fieldElements.ackGapsEl && !gapIssue) fieldElements.ackGapsEl.checked = false;
    updateRunGate();
    setVisible(dataAnalysisEl, true);
    renderDataPreview(records);
    updateSectionStatus();
  }

  return {
    clearDataAnalysis,
    clearFieldHighlights,
    invalidateValidationState,
    showError,
    showWarn,
    updateDataAnalysis,
    updateWarningsForCurrentInputs,
  };
}