export function normalizeExploreDataset(raw) {
  const v = String(raw || "").trim().toLowerCase();
  if (v === "forecast") return "forecast";
  if (v === "residual") return "residual";
  return "input";
}

export function createExplorePanelController({
  elements,
  getLastForecasts,
  getLastRecords,
  getResidualEvidenceFromResult,
  isDataConfirmed,
  setFieldDisabled,
  setStatus,
  setVisible,
  t,
  valueUnitEl,
  buildSparkGrid,
}) {
  const {
    analysisExploreEl,
    exploreChartEl,
    exploreChartTypeEl,
    exploreDatasetEl,
    exploreFieldXEl,
    exploreFieldYEl,
    exploreStatusEl,
    exploreSummaryTableEl,
    exploreTemplateEl,
  } = elements;

  function clearSummaryTable() {
    if (!exploreSummaryTableEl) return;
    const thead = exploreSummaryTableEl.querySelector("thead");
    const tbody = exploreSummaryTableEl.querySelector("tbody");
    if (thead) thead.innerHTML = "";
    if (tbody) tbody.innerHTML = "";
  }

  function clear() {
    if (analysisExploreEl) setVisible(analysisExploreEl, false);
    if (exploreStatusEl) setStatus(exploreStatusEl, "");
    if (exploreChartEl) exploreChartEl.innerHTML = "";
    clearSummaryTable();
  }

  function getExploreValue(rec, key) {
    const k = String(key || "");
    if (!k) return undefined;
    if (k.startsWith("x.")) {
      const x = rec?.x && typeof rec.x === "object" ? rec.x : null;
      const inner = k.slice(2);
      return x ? x[inner] : undefined;
    }
    return rec ? rec[k] : undefined;
  }

  function inferExploreFieldType(values) {
    const arr = values.filter((v) => v !== undefined && v !== null && v !== "");
    if (arr.length === 0) return "unknown";
    const numeric = arr.map((v) => (typeof v === "number" ? v : Number(v)));
    if (numeric.every((n) => Number.isFinite(n))) return "numeric";
    return "categorical";
  }

  function listExploreFields(records) {
    const keys = new Set(["series_id", "y"]);
    const xKeys = new Set();
    for (const rec of records.slice(0, 300)) {
      const x = rec?.x && typeof rec.x === "object" ? rec.x : null;
      if (!x) continue;
      for (const k of Object.keys(x)) xKeys.add(`x.${k}`);
    }
    for (const k of xKeys) keys.add(k);

    const out = [];
    for (const k of keys) {
      if (k === "timestamp") continue;
      const sample = [];
      for (const rec of records.slice(0, 300)) {
        sample.push(getExploreValue(rec, k));
        if (sample.length >= 50) break;
      }
      const type = inferExploreFieldType(sample);
      if (type === "unknown") continue;
      out.push({ key: k, type });
    }
    out.sort((a, b) => {
      if (a.key === "y") return -1;
      if (b.key === "y") return 1;
      if (a.key === "series_id") return -1;
      if (b.key === "series_id") return 1;
      return a.key.localeCompare(b.key);
    });
    return out;
  }

  function setSelectOptions(selectEl, options, { keepValue = true } = {}) {
    if (!selectEl) return;
    const prev = keepValue ? String(selectEl.value || "") : "";
    selectEl.innerHTML = "";
    for (const opt of options) {
      const optionEl = document.createElement("option");
      optionEl.value = opt.value;
      optionEl.textContent = opt.label;
      selectEl.appendChild(optionEl);
    }
    if (keepValue && prev && options.some((o) => o.value === prev)) {
      selectEl.value = prev;
    } else if (options.length > 0) {
      selectEl.value = options[0].value;
    }
  }

  function formatPct(p) {
    const v = Number(p);
    if (!Number.isFinite(v)) return "-";
    const pct = Math.max(0, Math.min(100, v * 100));
    if (pct >= 10) return `${pct.toFixed(0)}%`;
    return `${pct.toFixed(1)}%`;
  }

  function renderExploreSummaryTable(headers, rows) {
    if (!exploreSummaryTableEl) return;
    const thead = exploreSummaryTableEl.querySelector("thead");
    const tbody = exploreSummaryTableEl.querySelector("tbody");
    if (!thead || !tbody) return;
    thead.innerHTML = "";
    tbody.innerHTML = "";
    if (!Array.isArray(headers) || headers.length === 0 || !Array.isArray(rows) || rows.length === 0) return;

    const trh = document.createElement("tr");
    for (const header of headers) {
      const th = document.createElement("th");
      th.textContent = String(header || "");
      trh.appendChild(th);
    }
    thead.appendChild(trh);

    for (const row of rows) {
      const tr = document.createElement("tr");
      for (const cell of row) {
        const td = document.createElement("td");
        td.textContent = String(cell ?? "");
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }

  function formatExploreFieldLabel(fieldKey) {
    const k = String(fieldKey || "").trim();
    if (!k) return "-";
    if (k === "y") {
      const unit = (valueUnitEl?.value || "").trim();
      return unit ? `y (${unit})` : "y";
    }
    return k;
  }

  function renderExploreHistogram(records, fieldKey) {
    if (!exploreChartEl || !exploreStatusEl) return;
    const values = records
      .map((r) => getExploreValue(r, fieldKey))
      .map((v) => (typeof v === "number" ? v : Number(v)))
      .filter((n) => Number.isFinite(n));

    if (values.length < 2) {
      exploreChartEl.innerHTML = "";
      renderExploreSummaryTable([], []);
      setStatus(exploreStatusEl, t("analysis.explore.status_insufficient"));
      return;
    }

    const w = 640;
    const h = 220;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const bins = 20;
    const counts = new Array(bins).fill(0);
    for (const v of values) {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor(((v - min) / span) * bins)));
      counts[idx]++;
    }
    const maxC = Math.max(...counts) || 1;
    const barW = Math.max(2, Math.floor((w - 8) / bins) - 2);
    const bars = counts.map((c, i) => {
      const x = 4 + i * ((w - 8) / bins) + 1;
      const height = (c / maxC) * (h - 12);
      const y = h - height - 6;
      return `<rect class="sparkBar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW}" height="${Math.max(0, height).toFixed(1)}" />`;
    });
    exploreChartEl.innerHTML = `${buildSparkGrid(w, h)}${bars.join("")}`;
    const tableRows = counts.slice(0, 12).map((c, i) => {
      const lo = min + (span / bins) * i;
      const hi = lo + span / bins;
      return [`${lo.toFixed(3)}..${hi.toFixed(3)}`, String(c), formatPct(c / values.length)];
    });
    renderExploreSummaryTable(
      [t("analysis.explore.table.bin"), t("analysis.explore.table.count"), t("analysis.explore.table.share")],
      tableRows,
    );
    setStatus(exploreStatusEl, t("analysis.explore.status_hist", { field: formatExploreFieldLabel(fieldKey), n: values.length }));
  }

  function renderExploreCategory(records, fieldKey) {
    if (!exploreChartEl || !exploreStatusEl) return;
    const raw = records
      .map((r) => getExploreValue(r, fieldKey))
      .filter((v) => v !== undefined && v !== null && v !== "")
      .map((v) => (typeof v === "string" ? v : typeof v === "boolean" ? (v ? "true" : "false") : String(v)));
    if (raw.length < 1) {
      exploreChartEl.innerHTML = "";
      renderExploreSummaryTable([], []);
      setStatus(exploreStatusEl, t("analysis.explore.status_insufficient"));
      return;
    }
    const counts = new Map();
    for (const v of raw) counts.set(v, (counts.get(v) || 0) + 1);
    const total = raw.length;
    const rows = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const top = rows.slice(0, 10);
    const otherCount = rows.slice(10).reduce((a, [, c]) => a + c, 0);
    if (otherCount > 0) top.push([t("analysis.explore.other"), otherCount]);

    const w = 640;
    const h = 220;
    const bins = top.length;
    const barW = Math.max(8, Math.floor((w - 8) / Math.max(1, bins)) - 6);
    const bars = top.map(([, c], i) => {
      const x = 4 + i * ((w - 8) / Math.max(1, bins)) + 3;
      const height = (c / Math.max(1, total)) * (h - 12);
      const y = h - height - 6;
      return `<rect class="sparkBar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW}" height="${Math.max(0, height).toFixed(1)}" />`;
    });
    exploreChartEl.innerHTML = `${buildSparkGrid(w, h)}${bars.join("")}`;
    const tableRows = top.map(([k, c]) => [String(k), String(c), formatPct(c / total)]);
    renderExploreSummaryTable(
      [t("analysis.explore.table.category"), t("analysis.explore.table.count"), t("analysis.explore.table.share")],
      tableRows,
    );

    const topText = top
      .slice(0, 4)
      .map(([k, c]) => `${k} ${formatPct(c / total)} (${c})`)
      .join(", ");
    setStatus(exploreStatusEl, t("analysis.explore.status_category", { field: formatExploreFieldLabel(fieldKey), n: total, top: topText || "-" }));
  }

  function renderExploreScatter(records, xKey, yKey) {
    if (!exploreChartEl || !exploreStatusEl) return;
    const pts = [];
    for (const rec of records) {
      const xv = getExploreValue(rec, xKey);
      const yv = getExploreValue(rec, yKey);
      const x = typeof xv === "number" ? xv : Number(xv);
      const y = typeof yv === "number" ? yv : Number(yv);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      pts.push({ x, y });
      if (pts.length >= 800) break;
    }
    if (pts.length < 2) {
      exploreChartEl.innerHTML = "";
      renderExploreSummaryTable([], []);
      setStatus(exploreStatusEl, t("analysis.explore.status_insufficient"));
      return;
    }
    const w = 640;
    const h = 220;
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;
    const dots = pts.map((p) => {
      const cx = 4 + ((p.x - minX) / dx) * (w - 8);
      const cy = h - 4 - ((p.y - minY) / dy) * (h - 8);
      return `<circle class="exploreDot" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="2.1" />`;
    });
    exploreChartEl.innerHTML = `${buildSparkGrid(w, h)}${dots.join("")}`;
    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
    const meanY = ys.reduce((a, b) => a + b, 0) / ys.length;
    let corr = NaN;
    let num = 0;
    let denX = 0;
    let denY = 0;
    for (let i = 0; i < pts.length; i++) {
      const cx = pts[i].x - meanX;
      const cy = pts[i].y - meanY;
      num += cx * cy;
      denX += cx * cx;
      denY += cy * cy;
    }
    if (denX > 0 && denY > 0) corr = num / Math.sqrt(denX * denY);

    renderExploreSummaryTable(
      [t("analysis.explore.table.metric"), t("analysis.explore.table.value")],
      [
        ["n", String(pts.length)],
        ["x_mean", meanX.toFixed(4)],
        ["y_mean", meanY.toFixed(4)],
        ["pearson_r", Number.isFinite(corr) ? corr.toFixed(4) : "-"],
      ],
    );
    setStatus(exploreStatusEl, t("analysis.explore.status_scatter", { x: formatExploreFieldLabel(xKey), y: formatExploreFieldLabel(yKey), n: pts.length }));
  }

  function normalizeExploreTemplate(raw) {
    const v = String(raw || "").trim().toLowerCase();
    if (v === "cv_overview") return "cv_overview";
    if (v === "segment_share") return "segment_share";
    if (v === "driver_scatter") return "driver_scatter";
    return "manual";
  }

  function pickExploreTemplateConfig(template, { numeric, categorical }) {
    const numericFields = Array.isArray(numeric) ? numeric : [];
    const categoricalFields = Array.isArray(categorical) ? categorical : [];

    const yField = numericFields.find((f) => f.key === "y")?.key || numericFields[0]?.key || "";
    const firstNumeric = numericFields[0]?.key || "";
    const firstNumericNotY = numericFields.find((f) => f.key !== "y")?.key || firstNumeric;
    const firstSensor = numericFields.find((f) => f.key.startsWith("x.sensor_"))?.key || firstNumericNotY;
    const firstCategory = categoricalFields.find((f) => f.key === "series_id")?.key || categoricalFields[0]?.key || "";

    if (template === "segment_share") {
      return { mode: "category", xKey: firstCategory || "series_id", yKey: "" };
    }
    if (template === "driver_scatter") {
      return {
        mode: "scatter",
        xKey: firstSensor || firstNumericNotY || yField,
        yKey: yField || firstNumericNotY,
      };
    }
    return { mode: "hist", xKey: yField || firstNumeric, yKey: "" };
  }

  function renderExploreCvOverview(records, { numeric, categorical }) {
    if (!Array.isArray(records) || records.length === 0) return;
    const yVals = records
      .map((r) => Number(getExploreValue(r, "y")))
      .filter((n) => Number.isFinite(n));

    const catKey = categorical.find((f) => f.key === "series_id")?.key || categorical[0]?.key || "series_id";
    const catRaw = records
      .map((r) => getExploreValue(r, catKey))
      .filter((v) => v !== undefined && v !== null && v !== "")
      .map((v) => String(v));
    const catCount = new Map();
    for (const v of catRaw) catCount.set(v, (catCount.get(v) || 0) + 1);
    const catTop = [...catCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);

    const xKey = numeric.find((f) => f.key !== "y")?.key || "y";
    const pairs = [];
    for (const r of records) {
      const xv = Number(getExploreValue(r, xKey));
      const yv = Number(getExploreValue(r, "y"));
      if (Number.isFinite(xv) && Number.isFinite(yv)) pairs.push({ x: xv, y: yv });
      if (pairs.length >= 1000) break;
    }
    let corr = NaN;
    if (pairs.length > 1) {
      const xs = pairs.map((p) => p.x);
      const ys = pairs.map((p) => p.y);
      const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
      const meanY = ys.reduce((a, b) => a + b, 0) / ys.length;
      let num = 0;
      let denX = 0;
      let denY = 0;
      for (let i = 0; i < pairs.length; i++) {
        const dx = pairs[i].x - meanX;
        const dy = pairs[i].y - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
      }
      if (denX > 0 && denY > 0) corr = num / Math.sqrt(denX * denY);
    }

    const gapSeries = new Map();
    let invalidTs = 0;
    for (const rec of records) {
      const sid = String(rec?.series_id ?? "");
      const ts = Date.parse(String(rec?.timestamp ?? ""));
      if (!Number.isFinite(ts)) {
        invalidTs += 1;
        continue;
      }
      if (!gapSeries.has(sid)) gapSeries.set(sid, []);
      gapSeries.get(sid).push(ts);
    }
    let gapAlerts = 0;
    for (const arr of gapSeries.values()) {
      arr.sort((a, b) => a - b);
      if (arr.length < 4) continue;
      const diffs = [];
      for (let i = 1; i < arr.length; i++) diffs.push(arr[i] - arr[i - 1]);
      diffs.sort((a, b) => a - b);
      const med = diffs[Math.floor(diffs.length / 2)] || 0;
      if (!med) continue;
      const hasGap = diffs.some((d) => d > med * 1.5);
      if (hasGap) gapAlerts += 1;
    }

    const rows = [];
    if (yVals.length >= 2) {
      const min = Math.min(...yVals);
      const max = Math.max(...yVals);
      const bins = 8;
      const span = max - min || 1;
      const counts = new Array(bins).fill(0);
      for (const v of yVals) {
        const idx = Math.min(bins - 1, Math.max(0, Math.floor(((v - min) / span) * bins)));
        counts[idx] += 1;
      }
      const topBinIdx = counts.indexOf(Math.max(...counts));
      const lo = min + (span / bins) * topBinIdx;
      const hi = lo + span / bins;
      rows.push(["Yの度数分布", `${lo.toFixed(3)}..${hi.toFixed(3)}`, `${counts[topBinIdx]}`, formatPct(counts[topBinIdx] / yVals.length)]);
    }

    if (catTop.length) {
      const [name, count] = catTop[0];
      rows.push([`Xカテゴリ比率 (${catKey})`, String(name), String(count), formatPct(count / Math.max(1, catRaw.length))]);
    }

    rows.push(["X-Y散布要約", `${xKey} vs y`, Number.isFinite(corr) ? corr.toFixed(4) : "-", `n=${pairs.length}`]);
    rows.push(["時系列ギャップ要約", "gap_alert_series", String(gapAlerts), `invalid_ts=${invalidTs}`]);

    renderExploreSummaryTable(["観点", "軸/区間", "値", "補足"], rows);
  }

  function renderExploreChart(records) {
    if (!analysisExploreEl) return;
    if (!isDataConfirmed()) {
      setVisible(analysisExploreEl, false);
      if (exploreStatusEl) setStatus(exploreStatusEl, "");
      if (exploreChartEl) exploreChartEl.innerHTML = "";
      return;
    }
    if (!Array.isArray(records) || records.length === 0) {
      setVisible(analysisExploreEl, false);
      return;
    }
    if (!exploreChartTypeEl || !exploreFieldXEl || !exploreFieldYEl) {
      setVisible(analysisExploreEl, false);
      return;
    }
    if (!exploreChartEl || !exploreStatusEl) {
      setVisible(analysisExploreEl, false);
      return;
    }

    const fields = listExploreFields(records);
    const numeric = fields.filter((f) => f.type === "numeric");
    const categorical = fields.filter((f) => f.type === "categorical");

    const template = normalizeExploreTemplate(exploreTemplateEl?.value);
    if (template === "cv_overview") {
      const opts = numeric.map((f) => ({ value: f.key, label: f.key }));
      setSelectOptions(exploreFieldXEl, opts, { keepValue: false });
      setSelectOptions(exploreFieldYEl, [{ value: "", label: "-" }], { keepValue: false });
      setFieldDisabled(exploreFieldYEl, true);
      if (opts.length > 0) {
        const hasY = opts.some((o) => o.value === "y");
        exploreFieldXEl.value = hasY ? "y" : opts[0].value;
        renderExploreHistogram(records, String(exploreFieldXEl.value || "y"));
        renderExploreCvOverview(records, { numeric, categorical });
      } else {
        exploreChartEl.innerHTML = "";
        renderExploreSummaryTable([], []);
        setStatus(exploreStatusEl, t("analysis.explore.status_no_numeric"));
      }
      setVisible(analysisExploreEl, true);
      return;
    }

    const isManual = template === "manual";
    let mode = String(exploreChartTypeEl.value || "hist");
    let templateX = "";
    let templateY = "";
    if (!isManual) {
      const cfg = pickExploreTemplateConfig(template, { numeric, categorical });
      mode = cfg.mode;
      templateX = cfg.xKey;
      templateY = cfg.yKey;
      if (exploreChartTypeEl) exploreChartTypeEl.value = mode;
    }

    if (mode === "hist") {
      const opts = numeric.map((f) => ({ value: f.key, label: f.key }));
      setSelectOptions(exploreFieldXEl, opts, { keepValue: isManual });
      setSelectOptions(exploreFieldYEl, [{ value: "", label: "-" }], { keepValue: false });
      setFieldDisabled(exploreFieldYEl, true);
      setFieldDisabled(exploreFieldXEl, opts.length === 0);
      if (opts.length === 0) {
        exploreChartEl.innerHTML = "";
        setStatus(exploreStatusEl, t("analysis.explore.status_no_numeric"));
      } else {
        if (!isManual && templateX && opts.some((o) => o.value === templateX)) {
          exploreFieldXEl.value = templateX;
        }
        renderExploreHistogram(records, String(exploreFieldXEl.value || "y"));
      }
    } else if (mode === "category") {
      const opts = categorical.map((f) => ({ value: f.key, label: f.key }));
      setSelectOptions(exploreFieldXEl, opts, { keepValue: isManual });
      setSelectOptions(exploreFieldYEl, [{ value: "", label: "-" }], { keepValue: false });
      setFieldDisabled(exploreFieldYEl, true);
      setFieldDisabled(exploreFieldXEl, opts.length === 0);
      if (opts.length === 0) {
        exploreChartEl.innerHTML = "";
        setStatus(exploreStatusEl, t("analysis.explore.status_no_category"));
      } else {
        if (!isManual && templateX && opts.some((o) => o.value === templateX)) {
          exploreFieldXEl.value = templateX;
        }
        renderExploreCategory(records, String(exploreFieldXEl.value || "series_id"));
      }
    } else {
      const opts = numeric.map((f) => ({ value: f.key, label: f.key }));
      setSelectOptions(exploreFieldXEl, opts, { keepValue: isManual });
      setSelectOptions(exploreFieldYEl, opts, { keepValue: isManual });
      setFieldDisabled(exploreFieldYEl, opts.length === 0);
      setFieldDisabled(exploreFieldXEl, opts.length === 0);
      if (opts.length === 0) {
        exploreChartEl.innerHTML = "";
        setStatus(exploreStatusEl, t("analysis.explore.status_no_numeric"));
      } else {
        if (!isManual && templateX && opts.some((o) => o.value === templateX)) {
          exploreFieldXEl.value = templateX;
        }
        if (!isManual && templateY && opts.some((o) => o.value === templateY)) {
          exploreFieldYEl.value = templateY;
        }
        const xKey = String(exploreFieldXEl.value || "y");
        const yKey = String(exploreFieldYEl.value || "y");
        renderExploreScatter(records, xKey, yKey);
      }
    }

    setVisible(analysisExploreEl, true);
  }

  function buildExploreRecordsFromForecasts(forecasts) {
    const pts = Array.isArray(forecasts) ? forecasts : [];
    const out = [];
    for (const p of pts) {
      const series_id = String(p?.series_id ?? "");
      const timestamp = String(p?.timestamp ?? "");
      const point = Number(p?.point);
      if (!series_id || !timestamp || !Number.isFinite(point)) continue;

      let lower = null;
      let upper = null;
      if (p?.quantiles && typeof p.quantiles === "object") {
        const qVals = Object.values(p.quantiles).map((v) => Number(v)).filter((v) => Number.isFinite(v));
        if (qVals.length) {
          lower = Math.min(...qVals);
          upper = Math.max(...qVals);
        }
      }
      if ((lower === null || upper === null) && Array.isArray(p?.intervals)) {
        const intervals = p.intervals
          .map((i) => ({ level: Number(i?.level), lower: Number(i?.lower), upper: Number(i?.upper) }))
          .filter((i) => Number.isFinite(i.level) && Number.isFinite(i.lower) && Number.isFinite(i.upper));
        if (intervals.length) {
          intervals.sort((a, b) => b.level - a.level);
          lower = intervals[0].lower;
          upper = intervals[0].upper;
        }
      }
      const band_width = lower !== null && upper !== null ? Math.max(0, upper - lower) : null;

      out.push({
        series_id,
        timestamp,
        y: point,
        point,
        band_width,
      });
    }
    return out;
  }

  function buildExploreRecordsFromResidualEvidence(evidence) {
    const hist = evidence?.hist && typeof evidence.hist === "object" ? evidence.hist : null;
    const counts = Array.isArray(hist?.counts) ? hist.counts : [];
    const min = Number(hist?.min);
    const max = Number(hist?.max);
    if (!counts.length || !Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [];
    const step = (max - min) / counts.length;
    if (!(step > 0)) return [];

    const out = [];
    for (let i = 0; i < counts.length; i++) {
      const c = Number(counts[i]);
      if (!Number.isFinite(c) || c < 0) continue;
      const lo = min + i * step;
      const hi = lo + step;
      const center = lo + step / 2;
      out.push({
        series_id: "residual",
        y: c,
        x: {
          bin_index: i + 1,
          bin_center: center,
          bin_label: `${lo.toFixed(3)}..${hi.toFixed(3)}`,
        },
      });
    }
    return out;
  }

  function getExploreRecordsForSelectedDataset() {
    const dataset = normalizeExploreDataset(exploreDatasetEl?.value);
    if (dataset === "forecast") {
      return buildExploreRecordsFromForecasts(getLastForecasts());
    }
    if (dataset === "residual") {
      return buildExploreRecordsFromResidualEvidence(getResidualEvidenceFromResult());
    }
    return Array.isArray(getLastRecords()) ? getLastRecords() : [];
  }

  function syncDatasetAvailability() {
    if (!exploreDatasetEl) return;
    const lastForecasts = getLastForecasts();
    const hasForecast = Array.isArray(lastForecasts) && lastForecasts.length > 0;
    const hasResidual = !!getResidualEvidenceFromResult();
    const forecastOpt = exploreDatasetEl.querySelector('option[value="forecast"]');
    const residualOpt = exploreDatasetEl.querySelector('option[value="residual"]');
    if (forecastOpt) forecastOpt.disabled = !hasForecast;
    if (residualOpt) residualOpt.disabled = !hasResidual;
    if (forecastOpt) forecastOpt.hidden = !hasForecast;
    if (residualOpt) residualOpt.hidden = !hasResidual;

    const current = normalizeExploreDataset(exploreDatasetEl.value);
    if ((current === "forecast" && !hasForecast) || (current === "residual" && !hasResidual)) {
      exploreDatasetEl.value = "input";
    }
  }

  function refresh() {
    if (!analysisExploreEl) return;
    syncDatasetAvailability();
    if (!isDataConfirmed()) {
      setVisible(analysisExploreEl, false);
      return;
    }

    const dataset = normalizeExploreDataset(exploreDatasetEl?.value);
    const records = getExploreRecordsForSelectedDataset();
    if (dataset === "forecast" && (!Array.isArray(records) || records.length === 0)) {
      setVisible(analysisExploreEl, true);
      if (exploreStatusEl) setStatus(exploreStatusEl, t("analysis.explore.status_no_forecast"));
      if (exploreChartEl) exploreChartEl.innerHTML = "";
      renderExploreSummaryTable([], []);
      return;
    }
    if (dataset === "residual" && (!Array.isArray(records) || records.length === 0)) {
      setVisible(analysisExploreEl, true);
      if (exploreStatusEl) setStatus(exploreStatusEl, t("analysis.explore.status_no_residual"));
      if (exploreChartEl) exploreChartEl.innerHTML = "";
      renderExploreSummaryTable([], []);
      return;
    }

    try {
      renderExploreChart(records);
    } catch {
      // ignore
    }
  }

  return {
    clear,
    refresh,
  };
}