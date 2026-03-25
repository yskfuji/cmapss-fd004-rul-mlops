export function buildSparkGrid(w, h, { minY = null, maxY = null, preferZeroBaseline = false } = {}) {
  const baseStops = [0.25, 0.5, 0.75];

  let majorYStop = 0.5;
  if (
    preferZeroBaseline &&
    Number.isFinite(minY) &&
    Number.isFinite(maxY) &&
    maxY > minY &&
    minY <= 0 &&
    maxY >= 0
  ) {
    majorYStop = Math.max(0, Math.min(1, maxY / (maxY - minY)));
  }

  const yStops = [...baseStops];
  if (!yStops.some((p) => Math.abs(p - majorYStop) < 1e-3)) {
    yStops.push(majorYStop);
    yStops.sort((a, b) => a - b);
  }

  const horiz = yStops
    .map((p) => {
      const y = 4 + (h - 8) * p;
      const cls = Math.abs(p - majorYStop) < 1e-3 ? "sparkGrid sparkGridMajor" : "sparkGrid";
      return `<line class="${cls}" x1="4" y1="${y.toFixed(1)}" x2="${(w - 4).toFixed(1)}" y2="${y.toFixed(1)}" />`;
    })
    .join("");

  const vert = [0.25, 0.75]
    .map((p) => {
      const x = 4 + (w - 8) * p;
      return `<line class="sparkGrid" x1="${x.toFixed(1)}" y1="4" x2="${x.toFixed(1)}" y2="${(h - 4).toFixed(1)}" />`;
    })
    .join("");

  return `${horiz}${vert}`;
}

export function extractResidualEvidence(result) {
  const src = result && typeof result === "object" ? result : null;
  const ev = src?.residuals_evidence && typeof src.residuals_evidence === "object" ? src.residuals_evidence : null;
  if (!ev) return null;
  const hist = ev?.hist && typeof ev.hist === "object" ? ev.hist : null;
  const counts = Array.isArray(hist?.counts) ? hist.counts : null;
  if (!counts || counts.length < 2) return null;
  return ev;
}

function inferCompactNumberScale(values) {
  const nums = (Array.isArray(values) ? values : []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
  const maxAbs = nums.length ? Math.max(...nums.map((v) => Math.abs(v))) : 0;
  if (maxAbs >= 1e9) return { factor: 1e9, suffix: "B" };
  if (maxAbs >= 1e6) return { factor: 1e6, suffix: "M" };
  if (maxAbs >= 1e3) return { factor: 1e3, suffix: "k" };
  return { factor: 1, suffix: "" };
}

function formatCompactNumber(value, scale, { maximumFractionDigits = 2 } = {}) {
  const v = Number(value);
  if (!Number.isFinite(v)) return "-";
  const factor = Number(scale?.factor) || 1;
  const suffix = String(scale?.suffix || "");
  const scaled = v / factor;
  try {
    const shown = new Intl.NumberFormat(undefined, { maximumFractionDigits }).format(scaled);
    return suffix ? `${shown}${suffix}` : shown;
  } catch {
    const shown = scaled.toFixed(Math.min(2, Math.max(0, maximumFractionDigits)));
    return suffix ? `${shown}${suffix}` : shown;
  }
}

function getViewBoxSize(svgEl) {
  const vb = svgEl?.viewBox && svgEl.viewBox.baseVal ? svgEl.viewBox.baseVal : null;
  return {
    width: vb && Number.isFinite(vb.width) && vb.width > 0 ? vb.width : 320,
    height: vb && Number.isFinite(vb.height) && vb.height > 0 ? vb.height : 120,
  };
}

function formatAxisDate(ts) {
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "-";
  return d.toISOString().slice(0, 10);
}

function extractForecastValuesAndBands(records) {
  const pts = Array.isArray(records) ? records : [];
  const values = [];
  const bands = [];
  for (const rec of pts) {
    const ts = Date.parse(String(rec?.timestamp ?? ""));
    const point = Number(rec?.point);
    if (!Number.isFinite(ts) || !Number.isFinite(point)) continue;

    let lower = null;
    let upper = null;
    if (rec?.quantiles && typeof rec.quantiles === "object") {
      const qVals = Object.values(rec.quantiles).map((v) => Number(v)).filter((v) => Number.isFinite(v));
      if (qVals.length > 0) {
        lower = Math.min(...qVals);
        upper = Math.max(...qVals);
      }
    }
    if ((lower === null || upper === null) && Array.isArray(rec?.intervals)) {
      const intervals = rec.intervals
        .map((i) => ({
          level: Number(i?.level),
          lower: Number(i?.lower),
          upper: Number(i?.upper),
        }))
        .filter((i) => Number.isFinite(i.level) && Number.isFinite(i.lower) && Number.isFinite(i.upper));
      if (intervals.length > 0) {
        intervals.sort((a, b) => b.level - a.level);
        lower = intervals[0].lower;
        upper = intervals[0].upper;
      }
    }

    values.push({ ts, point });
    bands.push(lower !== null && upper !== null ? { lower, upper } : null);
  }
  return { values, bands };
}

export function createResultsVisualController({
  elements,
  getChartType,
  getFrequencyValue,
  getHorizonValue,
  getLastRecords,
  getLastRunContext,
  getValueUnit,
  countValidSeriesPoints,
  renderResultsEvidence,
  setVisible,
  t,
}) {
  const {
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
  } = elements;

  function axisValueLabel(chartType) {
    if (chartType === "cumulative") return t("results.axis.value_cumulative");
    if (chartType === "change") return t("results.axis.value_change");
    if (chartType === "index") return t("results.axis.value_index");
    if (chartType === "band") return t("results.axis.value_band");
    return t("results.axis.value");
  }

  function axisTimeLabel() {
    const base = t("results.axis.time");
    const frequency = String(getLastRunContext()?.frequency || getFrequencyValue() || "").trim();
    return frequency ? `${base} (${frequency})` : base;
  }

  function inferDomainUnitTemplate(chartType) {
    const type = String(chartType || "").trim().toLowerCase();
    if (type === "index") return "%";

    const records = Array.isArray(getLastRecords()) ? getLastRecords() : [];
    const first = records.length ? records[0] : null;
    const keys = first && typeof first === "object" ? Object.keys(first) : [];
    const joined = keys.join(" ").toLowerCase();

    if (/rul|remaining useful life|failure|degradation|health|cycle/.test(joined)) return "cycles";
    if (/ratio|share|rate|percent|pct/.test(joined)) return "%";
    if (/count|num|qty|volume|users|orders|visits/.test(joined)) return "count";

    const values = records.map((r) => Number(r?.y)).filter((v) => Number.isFinite(v));
    if (values.length >= 3) {
      const min = Math.min(...values);
      const max = Math.max(...values);
      if (min >= 0 && max <= 1) return "%";
    }
    return "";
  }

  function refreshAxisLabels(chartType = null) {
    const type = chartType || String(getChartType() || "trend");
    if (resultsAxisTimeEl) {
      resultsAxisTimeEl.textContent = axisTimeLabel();
    }
    if (resultsAxisValueEl) {
      const unit = String(getValueUnit() || "").trim() || inferDomainUnitTemplate(type);
      const base = axisValueLabel(type);
      resultsAxisValueEl.textContent = unit ? `${base} (${unit})` : base;
    }

    if (resultsVisualNoteEl && !String(resultsVisualNoteEl.textContent || "").trim()) {
      resultsVisualNoteEl.textContent = t("results.visual.note");
    }
  }

  function buildChartContextLabels() {
    const lastRunContext = getLastRunContext();
    const hRaw = lastRunContext?.horizon ?? getHorizonValue();
    const h = Math.max(1, Math.floor(Number(hRaw)));
    const point_label = Number.isFinite(h) ? t("results.chart.point_label", { h }) : t("results.chart.point_label", { h: "-" });

    const q = Array.isArray(lastRunContext?.quantiles) ? lastRunContext.quantiles : null;
    if (q && q.length) {
      const qs = q
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v) && v > 0 && v < 1)
        .sort((a, b) => a - b);
      if (qs.length >= 2) {
        const low = Math.round(qs[0] * 100);
        const high = Math.round(qs[qs.length - 1] * 100);
        return { point_label, band_label: t("results.chart.band_label_quantiles", { low, high }) };
      }
    }

    const lvl = Array.isArray(lastRunContext?.level) ? lastRunContext.level : null;
    if (lvl && lvl.length) {
      const levels = lvl
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v) && v > 0)
        .sort((a, b) => a - b);
      if (levels.length) {
        const level = Math.round(levels[levels.length - 1]);
        return { point_label, band_label: t("results.chart.band_label_level", { level }) };
      }
    }

    return { point_label, band_label: t("results.chart.band_label_none") };
  }

  function clearPrimaryAxes() {
    if (resultsAxisYMaxEl) resultsAxisYMaxEl.textContent = "";
    if (resultsAxisYMinEl) resultsAxisYMinEl.textContent = "";
    if (resultsAxisXMinEl) resultsAxisXMinEl.textContent = "";
    if (resultsAxisXMaxEl) resultsAxisXMaxEl.textContent = "";
  }

  function clearSecondaryAxes() {
    if (resultsAxisYMax2El) resultsAxisYMax2El.textContent = "";
    if (resultsAxisYMin2El) resultsAxisYMin2El.textContent = "";
    if (resultsAxisXMin2El) resultsAxisXMin2El.textContent = "";
    if (resultsAxisXMax2El) resultsAxisXMax2El.textContent = "";
  }

  function clearTertiaryAxes() {
    if (resultsAxisYMax3El) resultsAxisYMax3El.textContent = "";
    if (resultsAxisYMin3El) resultsAxisYMin3El.textContent = "";
    if (resultsAxisXMin3El) resultsAxisXMin3El.textContent = "";
    if (resultsAxisXMax3El) resultsAxisXMax3El.textContent = "";
  }

  function clear() {
    if (resultsSparklineEl) resultsSparklineEl.innerHTML = "";
    if (resultsSparklineSecondaryEl) resultsSparklineSecondaryEl.innerHTML = "";
    if (resultsSparklineTertiaryEl) resultsSparklineTertiaryEl.innerHTML = "";
    if (resultsPrimaryChartTitleEl) resultsPrimaryChartTitleEl.textContent = "";
    if (resultsSecondaryChartTitleEl) resultsSecondaryChartTitleEl.textContent = "";
    if (resultsTertiaryChartTitleEl) resultsTertiaryChartTitleEl.textContent = "";
    if (resultsVisualNoteEl) resultsVisualNoteEl.textContent = "";
    if (resultsSummaryNoteEl) resultsSummaryNoteEl.textContent = "";
    if (resultsHighlightSeriesEl) resultsHighlightSeriesEl.textContent = "";
    if (resultsHighlightPointsEl) resultsHighlightPointsEl.textContent = "";
    if (resultsHighlightBandEl) resultsHighlightBandEl.textContent = "";
    setVisible(resultsVisualEl, false);
    setVisible(resultsHighlightsEl, false);
    setVisible(resultsInterpretationEl, false);
    setVisible(resultsSecondaryChartBlockEl, false);
    setVisible(resultsTertiaryChartBlockEl, false);
    clearPrimaryAxes();
    clearSecondaryAxes();
    clearTertiaryAxes();
  }

  function renderForecastVisual(records, seriesId) {
    if (!resultsSparklineEl) return;
    const chartType = String(getChartType() || "trend");
    refreshAxisLabels(chartType);

    if (resultsPrimaryChartTitleEl) {
      const key = `results.chart.${chartType}`;
      const label = t(key);
      const chartLabel = label && label !== key ? label : chartType;
      const { point_label, band_label } = buildChartContextLabels();
      resultsPrimaryChartTitleEl.textContent = t("results.chart.title_with_context", {
        chart: chartLabel,
        series_id: String(seriesId || "").trim() || "-",
        point_label,
        band_label,
      });
    }

    const { width: w, height: h } = getViewBoxSize(resultsSparklineEl);
    const points = Array.isArray(records) ? records : [];
    if (points.length < 1) {
      resultsSparklineEl.innerHTML = "";
      clearPrimaryAxes();
      return;
    }

    const { values, bands } = extractForecastValuesAndBands(points);

    if (values.length === 1) {
      const v = values[0];
      const minX = v.ts;
      const maxX = v.ts;
      const yPad = Math.max(1, Math.abs(v.point) * 0.08);
      const minY = v.point - yPad;
      const maxY = v.point + yPad;
      const dy = maxY - minY || 1;
      const axisScale = inferCompactNumberScale([minY, maxY]);
      const grid = buildSparkGrid(w, h, { minY, maxY, preferZeroBaseline: true });

      if (resultsAxisYMaxEl) resultsAxisYMaxEl.textContent = formatCompactNumber(maxY, axisScale);
      if (resultsAxisYMinEl) resultsAxisYMinEl.textContent = formatCompactNumber(minY, axisScale);
      if (resultsAxisXMinEl) resultsAxisXMinEl.textContent = formatAxisDate(minX);
      if (resultsAxisXMaxEl) resultsAxisXMaxEl.textContent = formatAxisDate(maxX);

      const x = w / 2;
      const y = h - ((v.point - minY) / dy) * (h - 8) - 4;
      resultsSparklineEl.innerHTML = `
        ${grid}
        <circle class="sparkDot" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.6" />
      `;
      return;
    }

    if (values.length < 2) {
      resultsSparklineEl.innerHTML = "";
      clearPrimaryAxes();
      return;
    }

    const minX = Math.min(...values.map((v) => v.ts));
    const maxX = Math.max(...values.map((v) => v.ts));
    const bandLows = bands.filter(Boolean).map((b) => b.lower);
    const bandHighs = bands.filter(Boolean).map((b) => b.upper);
    let plotValues = values;
    if (chartType === "cumulative") {
      let sum = 0;
      plotValues = values.map((v) => {
        sum += v.point;
        return { ts: v.ts, point: sum };
      });
    } else if (chartType === "change") {
      plotValues = values.map((v, i) => {
        if (i === 0) return { ts: v.ts, point: 0 };
        return { ts: v.ts, point: v.point - values[i - 1].point };
      });
    } else if (chartType === "index") {
      const base = values[0]?.point;
      plotValues = values.map((v) => ({
        ts: v.ts,
        point: Number.isFinite(base) && base !== 0 ? (v.point / base) * 100 : 0,
      }));
    }

    const minPoint = Math.min(...plotValues.map((v) => v.point));
    const maxPoint = Math.max(...plotValues.map((v) => v.point));
    let minY = Math.min(minPoint, ...(bandLows.length ? bandLows : []));
    let maxY = Math.max(maxPoint, ...(bandHighs.length ? bandHighs : []));
    const yPad = (maxY - minY) * 0.08 || 1;
    minY -= yPad;
    maxY += yPad;
    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;

    const axisScale = inferCompactNumberScale([minY, maxY]);
    const grid = buildSparkGrid(w, h, { minY, maxY, preferZeroBaseline: true });
    if (resultsAxisYMaxEl) resultsAxisYMaxEl.textContent = formatCompactNumber(maxY, axisScale);
    if (resultsAxisYMinEl) resultsAxisYMinEl.textContent = formatCompactNumber(minY, axisScale);
    if (resultsAxisXMinEl) resultsAxisXMinEl.textContent = formatAxisDate(minX);
    if (resultsAxisXMaxEl) resultsAxisXMaxEl.textContent = formatAxisDate(maxX);

    if (chartType === "bars") {
      const barWidth = Math.max(2, Math.floor((w - 8) / values.length) - 2);
      const bars = values.map((v) => {
        const x = ((v.ts - minX) / dx) * (w - 8) + 4;
        const y = h - ((v.point - minY) / dy) * (h - 8) - 4;
        const height = Math.max(1, h - 4 - y);
        return `<rect class="sparkBar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth}" height="${height.toFixed(1)}" />`;
      });
      resultsSparklineEl.innerHTML = `${grid}${bars.join("")}`;
      return;
    }

    if (chartType === "area") {
      const pointCoords = plotValues.map((v) => ({
        x: ((v.ts - minX) / dx) * (w - 8) + 4,
        y: h - ((v.point - minY) / dy) * (h - 8) - 4,
      }));
      const baselineY = h - 4;
      const areaPoints = [
        ...pointCoords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`),
        `${pointCoords[pointCoords.length - 1].x.toFixed(1)},${baselineY.toFixed(1)}`,
        `${pointCoords[0].x.toFixed(1)},${baselineY.toFixed(1)}`,
      ];
      resultsSparklineEl.innerHTML = `
        ${grid}
        <polygon class="sparkArea" points="${areaPoints.join(" ")}" />
        <polyline class="sparkLine" fill="none" points="${pointCoords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}" />
      `;
      return;
    }

    if (chartType === "step") {
      const stepCoords = [];
      for (let i = 0; i < plotValues.length; i++) {
        const v = plotValues[i];
        const x = ((v.ts - minX) / dx) * (w - 8) + 4;
        const y = h - ((v.point - minY) / dy) * (h - 8) - 4;
        if (i === 0) {
          stepCoords.push(`${x.toFixed(1)},${y.toFixed(1)}`);
        } else {
          const prev = plotValues[i - 1];
          const prevY = h - ((prev.point - minY) / dy) * (h - 8) - 4;
          stepCoords.push(`${x.toFixed(1)},${prevY.toFixed(1)}`);
          stepCoords.push(`${x.toFixed(1)},${y.toFixed(1)}`);
        }
      }
      resultsSparklineEl.innerHTML = `
        ${grid}
        <polyline class="sparkStepLine" fill="none" points="${stepCoords.join(" ")}" />
      `;
      return;
    }

    if (chartType === "band") {
      const bandWidth = bands.map((b) => (b ? Math.max(0, b.upper - b.lower) : null)).filter((v) => Number.isFinite(v));
      const minBand = bandWidth.length ? Math.min(...bandWidth) : 0;
      const maxBand = bandWidth.length ? Math.max(...bandWidth) : 1;
      const ddy = maxBand - minBand || 1;
      const barWidth = Math.max(2, Math.floor((w - 8) / values.length) - 2);
      const bars = bands.map((b, i) => {
        const x = ((values[i].ts - minX) / dx) * (w - 8) + 4;
        const width = b ? Math.max(0, b.upper - b.lower) : 0;
        const height = b ? ((width - minBand) / ddy) * (h - 8) : 0;
        const y = h - height - 4;
        return `<rect class="sparkBandBar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth}" height="${height.toFixed(1)}" />`;
      });
      resultsSparklineEl.innerHTML = `${grid}${bars.join("")}`;
      return;
    }

    if (chartType === "points") {
      const xs = plotValues.map((v) => (v.ts - minX) / dx);
      const ys = plotValues.map((v) => v.point);
      const n = ys.length;
      const meanX = xs.reduce((a, b) => a + b, 0) / n;
      const meanY = ys.reduce((a, b) => a + b, 0) / n;
      let num = 0;
      let den = 0;
      for (let i = 0; i < n; i++) {
        const dxv = xs[i] - meanX;
        num += dxv * (ys[i] - meanY);
        den += dxv * dxv;
      }
      const slope = den > 0 ? num / den : 0;
      const intercept = meanY - slope * meanX;
      const residuals = ys.map((y, i) => y - (slope * xs[i] + intercept));
      const residualVar = residuals.reduce((a, b) => a + b * b, 0) / Math.max(1, n - 1);
      const residualStd = Math.sqrt(residualVar);

      const pointCoords = plotValues.map((v) => {
        const x = ((v.ts - minX) / dx) * (w - 8) + 4;
        const y = h - ((v.point - minY) / dy) * (h - 8) - 4;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      const trendCoords = plotValues.map((v) => {
        const xNorm = (v.ts - minX) / dx;
        const yVal = slope * xNorm + intercept;
        const x = xNorm * (w - 8) + 4;
        const y = h - ((yVal - minY) / dy) * (h - 8) - 4;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      const bandLowerCoords = plotValues.map((v) => {
        const xNorm = (v.ts - minX) / dx;
        const yVal = slope * xNorm + intercept - residualStd;
        const x = xNorm * (w - 8) + 4;
        const y = h - ((yVal - minY) / dy) * (h - 8) - 4;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      const bandUpperCoords = plotValues.map((v) => {
        const xNorm = (v.ts - minX) / dx;
        const yVal = slope * xNorm + intercept + residualStd;
        const x = xNorm * (w - 8) + 4;
        const y = h - ((yVal - minY) / dy) * (h - 8) - 4;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      const dots = pointCoords
        .map((p) => {
          const [x, y] = p.split(",");
          return `<circle class="sparkDot" cx="${x}" cy="${y}" r="2.4" />`;
        })
        .join("");
      resultsSparklineEl.innerHTML = `
        ${grid}
        <polygon class="sparkTrendBand" points="${bandLowerCoords.join(" ")} ${bandUpperCoords.reverse().join(" ")}" />
        <polyline class="sparkTrendLine" fill="none" points="${trendCoords.join(" ")}" />
        <polyline class="sparkLine sparkLineGhost" fill="none" points="${pointCoords.join(" ")}" />
        ${dots}
      `;
      return;
    }

    const pointCoords = [];
    const lowerCoords = [];
    const upperCoords = [];
    for (let i = 0; i < plotValues.length; i++) {
      const x = ((plotValues[i].ts - minX) / dx) * (w - 8) + 4;
      const yPoint = h - ((plotValues[i].point - minY) / dy) * (h - 8) - 4;
      pointCoords.push(`${x.toFixed(1)},${yPoint.toFixed(1)}`);
      const band = bands[i];
      if (band) {
        const yLower = h - ((band.lower - minY) / dy) * (h - 8) - 4;
        const yUpper = h - ((band.upper - minY) / dy) * (h - 8) - 4;
        lowerCoords.push(`${x.toFixed(1)},${yLower.toFixed(1)}`);
        upperCoords.push(`${x.toFixed(1)},${yUpper.toFixed(1)}`);
      }
    }

    const showBand = chartType === "trend";
    const bandPolygon =
      showBand && lowerCoords.length > 1 && upperCoords.length > 1
        ? `<polygon class="sparkBand" points="${lowerCoords.join(" ")} ${upperCoords.reverse().join(" ")}" />`
        : "";
    const dots = pointCoords
      .map((p) => {
        const [x, y] = p.split(",");
        return `<circle class="sparkDot" cx="${x}" cy="${y}" r="2.1" />`;
      })
      .join("");
    resultsSparklineEl.innerHTML = `
      ${grid}
      ${bandPolygon}
      <polyline class="sparkLine" fill="none" points="${pointCoords.join(" ")}" />
      ${dots}
    `;
  }

  function renderBandWidthSecondaryVisual(records, seriesId) {
    if (!resultsSparklineSecondaryEl || !resultsSecondaryChartBlockEl) return;

    const { values, bands } = extractForecastValuesAndBands(records);
    const widths = bands
      .map((b, i) => (b ? { ts: values[i]?.ts, width: Math.max(0, b.upper - b.lower) } : null))
      .filter((x) => x && Number.isFinite(x.ts) && Number.isFinite(x.width));

    if (widths.length < 2) {
      resultsSparklineSecondaryEl.innerHTML = "";
      setVisible(resultsSecondaryChartBlockEl, false);
      clearSecondaryAxes();
      return;
    }

    setVisible(resultsSecondaryChartBlockEl, true);
    if (resultsSecondaryChartTitleEl) {
      const chartLabel = t("results.chart.band");
      const { point_label, band_label } = buildChartContextLabels();
      resultsSecondaryChartTitleEl.textContent = t("results.chart.title_with_context", {
        chart: chartLabel,
        series_id: String(seriesId || "").trim() || "-",
        point_label,
        band_label,
      });
    }

    const { width: w, height: h } = getViewBoxSize(resultsSparklineSecondaryEl);
    const minX = Math.min(...widths.map((v) => v.ts));
    const maxX = Math.max(...widths.map((v) => v.ts));
    const minW = Math.min(...widths.map((v) => v.width));
    const maxW = Math.max(...widths.map((v) => v.width));
    const yPad = (maxW - minW) * 0.08 || 1;
    const minY = minW - yPad;
    const maxY = maxW + yPad;
    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;

    const axisScale = inferCompactNumberScale([minY, maxY]);
    if (resultsAxisValue2El) resultsAxisValue2El.textContent = t("results.axis.value_band");
    if (resultsAxisTime2El) resultsAxisTime2El.textContent = axisTimeLabel();
    if (resultsAxisYMax2El) resultsAxisYMax2El.textContent = formatCompactNumber(maxY, axisScale);
    if (resultsAxisYMin2El) resultsAxisYMin2El.textContent = formatCompactNumber(minY, axisScale);
    if (resultsAxisXMin2El) resultsAxisXMin2El.textContent = formatAxisDate(minX);
    if (resultsAxisXMax2El) resultsAxisXMax2El.textContent = formatAxisDate(maxX);

    const grid = buildSparkGrid(w, h, { minY, maxY, preferZeroBaseline: true });
    const barWidth = Math.max(2, Math.floor((w - 8) / widths.length) - 2);
    const bars = widths.map((v) => {
      const x = ((v.ts - minX) / dx) * (w - 8) + 4;
      const height = ((v.width - minY) / dy) * (h - 8);
      const y = h - height - 4;
      return `<rect class="sparkBandBar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth}" height="${Math.max(0, height).toFixed(1)}" />`;
    });
    resultsSparklineSecondaryEl.innerHTML = `${grid}${bars.join("")}`;
  }

  function renderResidualsTertiaryVisual(evidence) {
    if (!resultsSparklineTertiaryEl || !resultsTertiaryChartBlockEl) return;
    const hist = evidence?.hist && typeof evidence.hist === "object" ? evidence.hist : null;
    const counts = (Array.isArray(hist?.counts) ? hist.counts : []).map((v) => Number(v)).filter((v) => Number.isFinite(v) && v >= 0);
    const bins = counts.length;
    if (bins < 2) {
      resultsSparklineTertiaryEl.innerHTML = "";
      setVisible(resultsTertiaryChartBlockEl, false);
      clearTertiaryAxes();
      return;
    }

    setVisible(resultsTertiaryChartBlockEl, true);
    if (resultsTertiaryChartTitleEl) {
      const chartLabel = t("results.chart.residuals");
      const n = Number(evidence?.n);
      const point_label = t("results.chart.residuals_point_label", { n: Number.isFinite(n) ? n : "-" });
      resultsTertiaryChartTitleEl.textContent = t("results.chart.title_with_context", {
        chart: chartLabel,
        series_id: "-",
        point_label,
        band_label: "-",
      });
    }

    const { width: w, height: h } = getViewBoxSize(resultsSparklineTertiaryEl);
    const minX = Number(hist?.min);
    const maxX = Number(hist?.max);
    const safeMinX = Number.isFinite(minX) ? minX : 0;
    const safeMaxX = Number.isFinite(maxX) && maxX > safeMinX ? maxX : safeMinX + 1;
    const maxC = Math.max(...counts) || 1;
    const axisScale = inferCompactNumberScale([safeMinX, safeMaxX]);
    const axisScaleY = inferCompactNumberScale([0, maxC]);

    if (resultsAxisValue3El) resultsAxisValue3El.textContent = t("results.axis.count");
    if (resultsAxisTime3El) resultsAxisTime3El.textContent = t("results.axis.abs_error");
    if (resultsAxisYMax3El) resultsAxisYMax3El.textContent = formatCompactNumber(maxC, axisScaleY);
    if (resultsAxisYMin3El) resultsAxisYMin3El.textContent = formatCompactNumber(0, axisScaleY);
    if (resultsAxisXMin3El) resultsAxisXMin3El.textContent = formatCompactNumber(safeMinX, axisScale);
    if (resultsAxisXMax3El) resultsAxisXMax3El.textContent = formatCompactNumber(safeMaxX, axisScale);

    const grid = buildSparkGrid(w, h, { minY: 0, maxY: maxC, preferZeroBaseline: true });
    const barW = Math.max(2, Math.floor((w - 8) / bins) - 2);
    const bars = counts.map((c, i) => {
      const x = 4 + i * ((w - 8) / bins) + 1;
      const height = (c / maxC) * (h - 12);
      const y = h - height - 6;
      return `<rect class="sparkBandBar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW}" height="${Math.max(0, height).toFixed(1)}" />`;
    });
    resultsSparklineTertiaryEl.innerHTML = `${grid}${bars.join("")}`;
  }

  function renderPointDistributionTertiaryVisual(records, seriesId) {
    if (!resultsSparklineTertiaryEl || !resultsTertiaryChartBlockEl) return;
    const vals = (Array.isArray(records) ? records : []).map((r) => Number(r?.point)).filter((n) => Number.isFinite(n));
    if (vals.length < 2) {
      resultsSparklineTertiaryEl.innerHTML = "";
      setVisible(resultsTertiaryChartBlockEl, false);
      clearTertiaryAxes();
      return;
    }

    setVisible(resultsTertiaryChartBlockEl, true);
    if (resultsTertiaryChartTitleEl) {
      resultsTertiaryChartTitleEl.textContent = t("results.chart.title_with_context", {
        chart: t("results.chart.points"),
        series_id: String(seriesId || "").trim() || "-",
        point_label: t("results.chart.point_label", { h: vals.length }),
        band_label: "-",
      });
    }

    const { width: w, height: h } = getViewBoxSize(resultsSparklineTertiaryEl);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const span = max - min || 1;
    const bins = 16;
    const counts = new Array(bins).fill(0);
    for (const v of vals) {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor(((v - min) / span) * bins)));
      counts[idx]++;
    }
    const maxC = Math.max(...counts) || 1;
    const axisScaleY = inferCompactNumberScale([0, maxC]);
    const axisScaleX = inferCompactNumberScale([min, max]);
    if (resultsAxisValue3El) resultsAxisValue3El.textContent = t("results.axis.count");
    if (resultsAxisTime3El) resultsAxisTime3El.textContent = t("results.axis.value");
    if (resultsAxisYMax3El) resultsAxisYMax3El.textContent = formatCompactNumber(maxC, axisScaleY);
    if (resultsAxisYMin3El) resultsAxisYMin3El.textContent = formatCompactNumber(0, axisScaleY);
    if (resultsAxisXMin3El) resultsAxisXMin3El.textContent = formatCompactNumber(min, axisScaleX);
    if (resultsAxisXMax3El) resultsAxisXMax3El.textContent = formatCompactNumber(max, axisScaleX);

    const grid = buildSparkGrid(w, h, { minY: 0, maxY: maxC, preferZeroBaseline: true });
    const barW = Math.max(2, Math.floor((w - 8) / bins) - 2);
    const bars = counts.map((c, i) => {
      const x = 4 + i * ((w - 8) / bins) + 1;
      const height = (c / maxC) * (h - 12);
      const y = h - height - 6;
      return `<rect class="sparkBar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW}" height="${Math.max(0, height).toFixed(1)}" />`;
    });
    resultsSparklineTertiaryEl.innerHTML = `${grid}${bars.join("")}`;
  }

  function updateResultsInterpretation() {
    if (!resultsInterpretationEl) return;
    const chartType = String(getChartType() || "trend");
    const intentKey = `results.interpret.${chartType}.intent`;
    const whenKey = `results.interpret.${chartType}.when`;
    const cautionKey = `results.interpret.${chartType}.caution`;
    if (resultsInterpretIntentEl) resultsInterpretIntentEl.textContent = t(intentKey);
    if (resultsInterpretWhenEl) resultsInterpretWhenEl.textContent = t(whenKey);
    if (resultsInterpretCautionEl) resultsInterpretCautionEl.textContent = t(cautionKey);
    setVisible(resultsInterpretationEl, true);
  }

  function updateResultsSummaryIntegrity(seriesPoints, seriesId) {
    if (!resultsSummaryNoteEl) return;
    if (!Array.isArray(seriesPoints) || seriesPoints.length === 0) {
      resultsSummaryNoteEl.textContent = "";
      return;
    }
    const { valid, total } = countValidSeriesPoints(seriesPoints);
    if (valid < total) {
      resultsSummaryNoteEl.textContent = t("results.summary.integrity_mismatch", {
        series_id: String(seriesId || "").trim() || "-",
        total,
        valid,
      });
      return;
    }
    resultsSummaryNoteEl.textContent = t("results.summary.integrity_ok", {
      series_id: String(seriesId || "").trim() || "-",
      total,
    });
  }

  function updateHighlightCards(seriesPoints, seriesId) {
    if (resultsHighlightSeriesEl) {
      resultsHighlightSeriesEl.textContent = t("results.highlight.series", {
        series_id: String(seriesId || "").trim() || "-",
        horizon: seriesPoints.length,
      });
    }
    if (resultsHighlightPointsEl) {
      const points = seriesPoints.map((p) => Number(p?.point)).filter((v) => Number.isFinite(v));
      const min = points.length ? Math.min(...points) : null;
      const max = points.length ? Math.max(...points) : null;
      const mean = points.length ? points.reduce((a, b) => a + b, 0) / points.length : null;
      const variance = points.length > 1 && mean !== null ? points.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (points.length - 1) : null;
      const std = variance !== null ? Math.sqrt(variance) : null;
      resultsHighlightPointsEl.textContent = t("results.highlight.points", {
        n: points.length,
        min: Number.isFinite(min) ? min : "-",
        max: Number.isFinite(max) ? max : "-",
        std: Number.isFinite(std) ? std : "-",
      });
    }
    if (resultsHighlightBandEl) {
      const bands = seriesPoints
        .map((p) => {
          const extracted = extractForecastValuesAndBands([p]).bands[0];
          return extracted || null;
        })
        .filter(Boolean);
      const lows = bands.map((b) => b.lower);
      const highs = bands.map((b) => b.upper);
      const bandMin = lows.length ? Math.min(...lows) : null;
      const bandMax = highs.length ? Math.max(...highs) : null;
      resultsHighlightBandEl.textContent = t("results.highlight.band", {
        min: Number.isFinite(bandMin) ? bandMin : "-",
        max: Number.isFinite(bandMax) ? bandMax : "-",
      });
    }
  }

  function renderInputPreviewChart(records) {
    if (!Array.isArray(records) || records.length < 2 || !resultsVisualEl) return;
    const seriesIds = [...new Set(records.map((r) => String(r?.series_id ?? "").trim()))].filter(Boolean).sort();
    const seriesId = seriesIds[0] || "-";
    const points = records
      .filter((r) => String(r?.series_id ?? "") === seriesId)
      .map((r) => ({
        series_id: String(r?.series_id ?? ""),
        timestamp: String(r?.timestamp ?? ""),
        point: Number(r?.y),
      }))
      .filter((p) => Number.isFinite(Date.parse(p.timestamp)) && Number.isFinite(p.point))
      .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));

    if (points.length < 2) return;
    setVisible(resultsVisualEl, true);
    setVisible(resultsEmptyEl, false);
    if (resultsVisualNoteEl) resultsVisualNoteEl.textContent = t("results.visual.preview_note", { series_id: seriesId });
    renderForecastVisual(points, seriesId);
    try {
      resultsVisualEl.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch {
      // ignore
    }
  }

  function syncForecastSeriesOptions(forecasts, selectedSeriesId) {
    if (!resultsSeriesEl) return null;
    const doc = resultsSeriesEl.ownerDocument || document;
    const seriesIds = [...new Set((Array.isArray(forecasts) ? forecasts : []).map((f) => String(f?.series_id ?? "").trim()).filter(Boolean))]
      .sort((a, b) => a.localeCompare(b));
    resultsSeriesEl.innerHTML = "";
    for (const sid of seriesIds) {
      const opt = doc.createElement("option");
      opt.value = sid;
      opt.textContent = sid;
      resultsSeriesEl.appendChild(opt);
    }
    if (!seriesIds.length) {
      resultsSeriesEl.disabled = true;
      return null;
    }
    resultsSeriesEl.disabled = false;
    const nextSelectedSeriesId = selectedSeriesId && seriesIds.includes(selectedSeriesId) ? selectedSeriesId : seriesIds[0];
    resultsSeriesEl.value = nextSelectedSeriesId;
    return nextSelectedSeriesId;
  }

  function renderSelectedForecastSeries({ forecasts, selectedSeriesId, residualEvidence = null }) {
    const list = Array.isArray(forecasts) ? forecasts : [];
    if (!list.length) {
      if (resultsSparklineEl) resultsSparklineEl.innerHTML = "";
      return;
    }

    const seriesId = String(selectedSeriesId || list[0]?.series_id || "").trim();
    const seriesPoints = list
      .filter((f) => String(f?.series_id ?? "") === seriesId)
      .sort((a, b) => Date.parse(String(a?.timestamp ?? "")) - Date.parse(String(b?.timestamp ?? "")));

    renderForecastVisual(seriesPoints, seriesId);
    renderBandWidthSecondaryVisual(seriesPoints, seriesId);
    if (residualEvidence) {
      renderResidualsTertiaryVisual(residualEvidence);
    } else {
      renderPointDistributionTertiaryVisual(seriesPoints, seriesId);
    }
    updateResultsInterpretation();
    updateResultsSummaryIntegrity(seriesPoints, seriesId);
    renderResultsEvidence("forecast", { seriesPoints });
    updateHighlightCards(seriesPoints, seriesId);
  }

  return {
    clear,
    refreshAxisLabels,
    renderInputPreviewChart,
    renderSelectedForecastSeries,
    syncForecastSeriesOptions,
  };
}