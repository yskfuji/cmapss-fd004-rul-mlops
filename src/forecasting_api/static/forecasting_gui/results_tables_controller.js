export function createResultsTablesController({ elements, setText, t }) {
  const {
    metricsPrimaryEl,
    metricsTableBodyEl,
    bySeriesTableBodyEl,
    byHorizonTableBodyEl,
    byFoldTableBodyEl,
    driftFeaturesTableBodyEl,
    resultTableBodyEl,
  } = elements;

  function createCell(doc, value) {
    const cell = doc.createElement("td");
    cell.textContent = String(value ?? "");
    return cell;
  }

  function clear() {
    if (resultTableBodyEl) resultTableBodyEl.innerHTML = "";
    if (metricsTableBodyEl) metricsTableBodyEl.innerHTML = "";
    if (bySeriesTableBodyEl) bySeriesTableBodyEl.innerHTML = "";
    if (byHorizonTableBodyEl) byHorizonTableBodyEl.innerHTML = "";
    if (byFoldTableBodyEl) byFoldTableBodyEl.innerHTML = "";
    if (driftFeaturesTableBodyEl) driftFeaturesTableBodyEl.innerHTML = "";
    if (metricsPrimaryEl) setText(metricsPrimaryEl, "");
  }

  function renderDriftFeatures(featureReports) {
    if (!driftFeaturesTableBodyEl) return;
    const doc = driftFeaturesTableBodyEl.ownerDocument || document;
    for (const row of Array.isArray(featureReports) ? featureReports : []) {
      const tr = doc.createElement("tr");
      const psi = Number(row?.population_stability_index);
      tr.appendChild(createCell(doc, String(row?.feature ?? "")));
      tr.appendChild(createCell(doc, Number.isFinite(psi) ? psi.toFixed(3) : "-"));
      tr.appendChild(createCell(doc, String(row?.baseline_requested_bin_count ?? "-")));
      tr.appendChild(createCell(doc, String(row?.baseline_selected_bin_count ?? "-")));
      driftFeaturesTableBodyEl.appendChild(tr);
    }
  }

  function renderBacktestTables({ highlightKey, metricTableRows, seriesTableRows, horizonTableRows, foldTableRows }) {
    if (highlightKey) {
      setText(metricsPrimaryEl, t("backtest.metrics_primary", { metric: highlightKey }));
    } else {
      setText(metricsPrimaryEl, t("backtest.metrics_primary_none"));
    }

    if (metricsTableBodyEl) {
      const doc = metricsTableBodyEl.ownerDocument || document;
      for (const row of Array.isArray(metricTableRows) ? metricTableRows : []) {
        const tr = doc.createElement("tr");
        tr.classList.add("metricRow");
        if (row.isHighlight) tr.classList.add("metricHighlight");
        tr.appendChild(createCell(doc, row.metric));
        tr.appendChild(createCell(doc, row.value));
        metricsTableBodyEl.appendChild(tr);
      }
    }

    if (bySeriesTableBodyEl) {
      const doc = bySeriesTableBodyEl.ownerDocument || document;
      for (const row of Array.isArray(seriesTableRows) ? seriesTableRows : []) {
        const tr = doc.createElement("tr");
        tr.appendChild(createCell(doc, row.rank));
        tr.appendChild(createCell(doc, row.seriesId));
        tr.appendChild(createCell(doc, row.value));
        bySeriesTableBodyEl.appendChild(tr);
      }
    }

    if (byHorizonTableBodyEl) {
      const doc = byHorizonTableBodyEl.ownerDocument || document;
      for (const row of Array.isArray(horizonTableRows) ? horizonTableRows : []) {
        const tr = doc.createElement("tr");
        tr.appendChild(createCell(doc, row.horizon));
        tr.appendChild(createCell(doc, row.value));
        byHorizonTableBodyEl.appendChild(tr);
      }
    }

    if (byFoldTableBodyEl) {
      const doc = byFoldTableBodyEl.ownerDocument || document;
      for (const row of Array.isArray(foldTableRows) ? foldTableRows : []) {
        const tr = doc.createElement("tr");
        tr.appendChild(createCell(doc, row.fold));
        tr.appendChild(createCell(doc, row.value));
        byFoldTableBodyEl.appendChild(tr);
      }
    }
  }

  function renderForecastTable(forecastRows) {
    if (!resultTableBodyEl) return;
    const doc = resultTableBodyEl.ownerDocument || document;
    for (const row of Array.isArray(forecastRows) ? forecastRows : []) {
      const tr = doc.createElement("tr");
      tr.appendChild(createCell(doc, row.seriesId));
      tr.appendChild(createCell(doc, row.timestamp));
      tr.appendChild(createCell(doc, row.point));
      tr.appendChild(createCell(doc, row.quantiles));
      tr.appendChild(createCell(doc, row.intervals));
      resultTableBodyEl.appendChild(tr);
    }
  }

  return {
    clear,
    renderBacktestTables,
    renderDriftFeatures,
    renderForecastTable,
  };
}