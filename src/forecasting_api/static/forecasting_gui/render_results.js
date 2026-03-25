export function normalizeForecasts(forecasts) {
  const list = Array.isArray(forecasts) ? [...forecasts] : [];
  list.sort((a, b) => {
    const sa = String(a?.series_id ?? "");
    const sb = String(b?.series_id ?? "");
    if (sa !== sb) return sa.localeCompare(sb);
    const ta = Date.parse(String(a?.timestamp ?? ""));
    const tb = Date.parse(String(b?.timestamp ?? ""));
    if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return ta - tb;
    return String(a?.timestamp ?? "").localeCompare(String(b?.timestamp ?? ""));
  });
  return list;
}

export function getForecastSeriesIds(forecasts) {
  const out = new Set();
  for (const f of forecasts) {
    const sid = String(f?.series_id ?? "").trim();
    if (sid) out.add(sid);
  }
  return [...out].sort((a, b) => a.localeCompare(b));
}

export function forecastsToCsv(result) {
  const forecasts = Array.isArray(result?.forecasts) ? normalizeForecasts(result.forecasts) : [];
  const qKeys = new Set();
  for (const f of forecasts) {
    if (f.quantiles && typeof f.quantiles === "object") {
      for (const k of Object.keys(f.quantiles)) qKeys.add(k);
    }
  }
  const qCols = [...qKeys].sort().map((k) => `q_${k}`);
  const header = ["series_id", "timestamp", "point", ...qCols].join(",");
  const rows = [header];
  for (const f of forecasts) {
    const base = [f.series_id ?? "", f.timestamp ?? "", f.point ?? ""];
    const qs = qCols.map((c) => {
      const k = c.slice(2);
      const v = f.quantiles?.[k];
      return v === undefined || v === null ? "" : String(v);
    });
    rows.push([...base, ...qs].map((x) => String(x).replaceAll('"', '""')).map((x) => `"${x}"`).join(","));
  }
  return rows.join("\n");
}

export function backtestToCsv(result) {
  const metrics = result?.metrics && typeof result.metrics === "object" ? result.metrics : {};
  const bySeries = Array.isArray(result?.by_series) ? result.by_series : [];
  const byHorizon = Array.isArray(result?.by_horizon) ? result.by_horizon : [];
  const byFold = Array.isArray(result?.by_fold) ? result.by_fold : [];

  const header = ["section", "metric", "series_id", "horizon", "value"].join(",");
  const rows = [header];

  const pushRow = ({ section, metric, series_id = "", horizon = "", value = "" }) => {
    const vals = [section, metric, series_id, horizon, value].map((x) => String(x ?? "").replaceAll('"', '""'));
    rows.push(vals.map((x) => `"${x}"`).join(","));
  };

  for (const [metric, value] of Object.entries(metrics)) {
    pushRow({ section: "metrics", metric, value: value === null || value === undefined ? "" : value });
  }
  for (const r of bySeries) {
    pushRow({
      section: "by_series",
      metric: r?.metric ?? "",
      series_id: r?.series_id ?? "",
      value: r?.value === null || r?.value === undefined ? "" : r.value,
    });
  }
  for (const r of byHorizon) {
    pushRow({
      section: "by_horizon",
      metric: r?.metric ?? "",
      horizon: r?.horizon === null || r?.horizon === undefined ? "" : r.horizon,
      value: r?.value === null || r?.value === undefined ? "" : r.value,
    });
  }

  for (const r of byFold) {
    pushRow({
      section: "by_fold",
      metric: r?.metric ?? "",
      horizon: r?.fold === null || r?.fold === undefined ? "" : r.fold,
      value: r?.value === null || r?.value === undefined ? "" : r.value,
    });
  }

  return rows.join("\n");
}

export function buildBacktestViewModel(result) {
  const metrics = result?.metrics && typeof result.metrics === "object" ? result.metrics : {};
  const metricsEntries = Object.entries(metrics);
  const highlightKey = metricsEntries.length > 0 ? String(metricsEntries[0][0] || "") : null;

  const metricKey = Object.keys(metrics)[0] || null;
  const bySeries = Array.isArray(result?.by_series) ? result.by_series : [];
  const byHorizon = Array.isArray(result?.by_horizon) ? result.by_horizon : [];
  const byFold = Array.isArray(result?.by_fold) ? result.by_fold : [];

  const bySeriesRows = bySeries
    .map((row) => {
      const seriesId = String(row?.series_id ?? "");
      let value = null;
      if (typeof row?.value === "number") value = row.value;
      else if (metricKey && typeof row?.[metricKey] === "number") value = row[metricKey];
      return { seriesId, value };
    })
    .filter((r) => r.seriesId)
    .sort((a, b) => {
      const av = Number.isFinite(a.value) ? a.value : -Infinity;
      const bv = Number.isFinite(b.value) ? b.value : -Infinity;
      return bv - av;
    });

  const byHorizonRows = byHorizon
    .map((row) => {
      const horizon = Number(row?.horizon);
      let value = null;
      if (typeof row?.value === "number") value = row.value;
      else if (metricKey && typeof row?.[metricKey] === "number") value = row[metricKey];
      return { horizon: Number.isFinite(horizon) ? horizon : null, value };
    })
    .filter((r) => r.horizon && r.horizon >= 1)
    .sort((a, b) => a.horizon - b.horizon);

  const byFoldRows = byFold
    .map((row) => {
      const fold = Number(row?.fold);
      let value = null;
      if (typeof row?.value === "number") value = row.value;
      else if (metricKey && typeof row?.[metricKey] === "number") value = row[metricKey];
      return { fold: Number.isFinite(fold) ? fold : null, value };
    })
    .filter((r) => r.fold && r.fold >= 1)
    .sort((a, b) => a.fold - b.fold);

  return {
    metrics,
    metricsEntries,
    highlightKey,
    bySeriesRows,
    byHorizonRows,
    byFoldRows,
  };
}

export function countValidSeriesPoints(seriesPoints) {
  const points = Array.isArray(seriesPoints) ? seriesPoints : [];
  const valid = points.filter((p) => {
    const ts = Date.parse(String(p?.timestamp ?? ""));
    const point = Number(p?.point);
    return Number.isFinite(ts) && Number.isFinite(point);
  }).length;
  return { valid, total: points.length };
}

export function buildMetricTableRows(metricsEntries, highlightKey) {
  const entries = Array.isArray(metricsEntries) ? metricsEntries : [];
  return entries.map(([metric, value]) => ({
    metric: String(metric ?? ""),
    value: value === null || value === undefined ? "" : String(value),
    isHighlight: highlightKey != null && String(highlightKey) === String(metric),
  }));
}

export function buildSeriesRankRows(bySeriesRows, { limit = 50 } = {}) {
  const rows = Array.isArray(bySeriesRows) ? bySeriesRows : [];
  return rows.slice(0, Math.max(0, Number(limit) || 0)).map((row, idx) => ({
    rank: String(idx + 1),
    seriesId: String(row?.seriesId ?? ""),
    value: row?.value === null || row?.value === undefined ? "" : String(row.value),
  }));
}

export function buildHorizonRows(byHorizonRows) {
  const rows = Array.isArray(byHorizonRows) ? byHorizonRows : [];
  return rows.map((row) => ({
    horizon: String(row?.horizon ?? ""),
    value: row?.value === null || row?.value === undefined ? "" : String(row.value),
  }));
}

export function buildFoldRows(byFoldRows) {
  const rows = Array.isArray(byFoldRows) ? byFoldRows : [];
  return rows.map((row) => ({
    fold: String(row?.fold ?? ""),
    value: row?.value === null || row?.value === undefined ? "" : String(row.value),
  }));
}

export function buildForecastTableRows(forecasts) {
  const rows = Array.isArray(forecasts) ? forecasts : [];
  return rows.map((row) => ({
    seriesId: String(row?.series_id ?? ""),
    timestamp: String(row?.timestamp ?? ""),
    point: String(row?.point ?? ""),
    quantiles: row?.quantiles ? JSON.stringify(row.quantiles) : "",
    intervals: row?.intervals ? JSON.stringify(row.intervals) : "",
  }));
}

export function buildResultVisibility(
  task,
  {
    hasForecasts = false,
    hasMetrics = false,
    hasBySeries = false,
    hasByHorizon = false,
    hasByFold = false,
    hasDriftFeatures = false,
  } = {},
) {
  const mode = String(task || "forecast").toLowerCase();
  if (mode === "train") {
    return {
      showForecastTable: false,
      showMetrics: false,
      showBySeries: false,
      showByHorizon: false,
      showByFold: false,
      showDriftFeatures: false,
      showGuide: false,
      showLegend: false,
      showTableNote: false,
      showVisual: false,
      showHighlights: false,
      showEmpty: false,
      showTrainNote: true,
    };
  }
  if (mode === "backtest") {
    return {
      showForecastTable: false,
      showMetrics: true,
      showBySeries: !!hasBySeries,
      showByHorizon: !!hasByHorizon,
      showByFold: !!hasByFold,
      showDriftFeatures: false,
      showGuide: false,
      showLegend: false,
      showTableNote: false,
      showVisual: false,
      showHighlights: false,
      showEmpty: !hasMetrics,
      showTrainNote: false,
    };
  }
  if (mode === "drift") {
    return {
      showForecastTable: false,
      showMetrics: false,
      showBySeries: false,
      showByHorizon: false,
      showByFold: false,
      showDriftFeatures: !!hasDriftFeatures,
      showGuide: false,
      showLegend: false,
      showTableNote: false,
      showVisual: false,
      showHighlights: false,
      showEmpty: false,
      showTrainNote: false,
    };
  }
  return {
    showForecastTable: !!hasForecasts,
    showMetrics: false,
    showBySeries: false,
    showByHorizon: false,
    showByFold: false,
    showDriftFeatures: false,
    showGuide: !!hasForecasts,
    showLegend: !!hasForecasts,
    showTableNote: !!hasForecasts,
    showVisual: !!hasForecasts,
    showHighlights: !!hasForecasts,
    showEmpty: !hasForecasts,
    showTrainNote: false,
  };
}