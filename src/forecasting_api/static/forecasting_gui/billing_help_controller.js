const HELP_TOPIC_MAP = {
  data: { targetId: "dataCard", detail: "inputs", adviceKey: "help.concierge.advice.data" },
  run: { targetId: "paramsCard", detail: "inputs", adviceKey: "help.concierge.advice.run" },
  quota: { targetId: "billingCard", detail: null, adviceKey: "help.concierge.advice.quota" },
  results: { targetId: "resultsCard", detail: null, adviceKey: "help.concierge.advice.results" },
  support: { targetId: "helpCard", detail: "support", adviceKey: "help.concierge.advice.support" },
};

export function createBillingHelpController({
  elements,
  getRunPointsEstimate,
  refreshI18nStatuses,
  setStatusI18n,
  setText,
  setVisible,
  t,
}) {
  const {
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
  } = elements;

  function updateHelpAdvice() {
    if (!helpTopicEl || !helpAdviceEl) return;
    const key = String(helpTopicEl.value || "data");
    const cfg = HELP_TOPIC_MAP[key] || HELP_TOPIC_MAP.data;
    helpAdviceEl.textContent = t(cfg.adviceKey);
  }

  function openHelpDetail(detailKey) {
    if (detailKey === "inputs" && helpInputsEl) helpInputsEl.open = true;
    if (detailKey === "support" && helpSupportEl) helpSupportEl.open = true;
  }

  function guideHelpFlow() {
    if (!helpTopicEl) return;
    const key = String(helpTopicEl.value || "data");
    const cfg = HELP_TOPIC_MAP[key] || HELP_TOPIC_MAP.data;
    updateHelpAdvice();
    openHelpDetail(cfg.detail);
    if (cfg.targetId) {
      const doc = helpTopicEl.ownerDocument || document;
      const target = doc.getElementById(cfg.targetId);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function matchHelpTopicFromText(text) {
    const query = String(text || "").toLowerCase();
    if (!query) return null;
    if (/quota|billing|limit|料金|クォータ|上限|cost|costo1|s01/.test(query)) return "quota";
    if (/error|failed|failure|失敗|エラー|request_id|support|問い合わせ|連絡/.test(query)) return "support";
    if (/result|結果|forecast|予測|output|table|legend/.test(query)) return "results";
    if (/horizon|frequency|quantile|quantiles|level|param|設定|validate|検証/.test(query)) return "run";
    if (/csv|json|timestamp|series|入力|データ|format|schema/.test(query)) return "data";
    return null;
  }

  function updateBillingUi() {
    const {
      limit,
      seriesShown,
      hShown,
      canEstimate: allowEstimate,
      pointsEstimate,
      estimateUsed,
      remaining,
      ratio,
    } = getRunPointsEstimate();
    if (billingSyncMaxPointsEl) billingSyncMaxPointsEl.value = String(limit);

    if (billingQuotaStatusEl) {
      setStatusI18n(billingQuotaStatusEl, "billing.points_this_month", { remaining, limit, used: estimateUsed });
    }
    setText(billingEstimateEl, t("billing.estimate_points", { points: pointsEstimate, s: seriesShown, h: hShown }));
    setText(
      billingUsageStatusEl,
      pointsEstimate > 0 ? t("billing.remaining_points", { remaining }) : t("billing.remaining_na"),
    );
    if (billingSummaryLimitEl) {
      billingSummaryLimitEl.textContent = t("billing.summary.limit_value", { limit });
    }
    if (billingSummaryEstimateEl) {
      const estimateText =
        pointsEstimate > 0 ? t("billing.summary.estimate_value", { points: pointsEstimate }) : t("billing.summary.estimate_na");
      billingSummaryEstimateEl.textContent = estimateText;
    }
    if (billingSummaryRemainingEl) {
      const remainingText =
        pointsEstimate > 0 ? t("billing.summary.remaining_value", { remaining }) : t("billing.summary.remaining_na");
      billingSummaryRemainingEl.textContent = remainingText;
    }
    if (billingMetricRemainingEl) {
      billingMetricRemainingEl.textContent = pointsEstimate > 0 ? t("billing.metric.remaining_value", { remaining }) : "-";
    }
    if (billingMetricUsedEl) {
      billingMetricUsedEl.textContent = pointsEstimate > 0 ? t("billing.metric.used_value", { used: estimateUsed }) : "-";
    }
    if (billingMetricLimitEl) {
      billingMetricLimitEl.textContent = t("billing.metric.limit_value", { limit });
    }
    if (quotaProgressBarEl) {
      const percent = limit > 0 ? Math.max(0, Math.min(1, remaining / limit)) : 0;
      quotaProgressBarEl.style.width = `${Math.round(percent * 100)}%`;
      const wrap = quotaProgressBarEl.parentElement;
      if (wrap) {
        wrap.setAttribute("aria-valuenow", String(Math.round(percent * 100)));
      }
    }
    if (billingDetailUsedBarEl && billingDetailRemainingBarEl) {
      const usedPct = limit > 0 ? Math.max(0, Math.min(1, estimateUsed / limit)) : 0;
      const remainingPct = limit > 0 ? Math.max(0, Math.min(1, remaining / limit)) : 0;
      billingDetailUsedBarEl.style.width = `${Math.round(usedPct * 100)}%`;
      billingDetailRemainingBarEl.style.width = `${Math.round(remainingPct * 100)}%`;
    }
    if (billingImpactSeriesBarEl && billingImpactHorizonBarEl && billingImpactSeriesValueEl && billingImpactHorizonValueEl) {
      const unitSeries = hShown > 0 ? hShown : 0;
      const unitHorizon = seriesShown > 0 ? seriesShown : 0;
      const unitSeriesPct = limit > 0 ? Math.max(0, Math.min(1, unitSeries / limit)) : 0;
      const unitHorizonPct = limit > 0 ? Math.max(0, Math.min(1, unitHorizon / limit)) : 0;
      billingImpactSeriesBarEl.style.width = `${Math.round(unitSeriesPct * 100)}%`;
      billingImpactHorizonBarEl.style.width = `${Math.round(unitHorizonPct * 100)}%`;
      billingImpactSeriesValueEl.textContent = unitSeries > 0 ? t("billing.chart.unit_points", { points: unitSeries }) : "-";
      billingImpactHorizonValueEl.textContent = unitHorizon > 0 ? t("billing.chart.unit_points", { points: unitHorizon }) : "-";
    }
    if (quotaFlowValueEl) {
      const valueText =
        pointsEstimate > 0
          ? t("billing.flow_value", { remaining, limit, used: estimateUsed })
          : t("billing.flow_value_na", { limit });
      quotaFlowValueEl.textContent = valueText;
    }
    if (quotaFlowBarFillEl) {
      const percent = limit > 0 ? Math.max(0, Math.min(1, estimateUsed / limit)) : 0;
      quotaFlowBarFillEl.style.width = `${Math.round(percent * 100)}%`;
      const wrap = quotaFlowBarFillEl.parentElement;
      if (wrap) {
        wrap.setAttribute("aria-valuenow", String(Math.round(percent * 100)));
      }
    }

    let base = "";
    if (estimateUsed > limit) {
      base = t("billing.flow_alert_over", { remaining });
    } else if (ratio >= 0.8 && estimateUsed > 0) {
      base = t("billing.flow_alert_near", { remaining });
    } else if (estimateUsed > 0) {
      base = t("billing.flow_alert_ok", { remaining });
    } else {
      base = t("billing.flow_alert_na");
    }

    if (quotaFlowAlertEl) {
      quotaFlowAlertEl.textContent = base;
    }

    if (billingDetailBodyEl) {
      if (!allowEstimate) {
        billingDetailBodyEl.textContent = t("billing.detail_pending");
      } else {
        const extraSeries = remaining > 0 ? Math.floor(remaining / hShown) : 0;
        const extraHorizon = remaining > 0 ? Math.floor(remaining / seriesShown) : 0;
        const over = estimateUsed > limit ? Math.max(1, estimateUsed - limit) : 0;
        const reduceSeries = over > 0 ? Math.ceil(over / hShown) : 0;
        const reduceHorizon = over > 0 ? Math.ceil(over / seriesShown) : 0;
        const details = [
          t("billing.detail_value", { points: pointsEstimate, s: seriesShown, h: hShown }),
          t("billing.detail_remaining", { remaining, limit, used: estimateUsed }),
        ];
        if (over > 0) {
          details.push(
            t("billing.detail_reduce", {
              over,
              reduce_series: reduceSeries,
              reduce_horizon: reduceHorizon,
              h: hShown,
              s: seriesShown,
            }),
          );
        } else if (extraSeries > 0 || extraHorizon > 0) {
          details.push(
            t("billing.detail_expand", {
              extra_series: extraSeries,
              extra_horizon: extraHorizon,
              h: hShown,
              s: seriesShown,
            }),
          );
        }
        billingDetailBodyEl.textContent = details.filter(Boolean).join("\n");
      }
    }
    refreshI18nStatuses();

    if (estimateUsed > limit) {
      setVisible(billingErrorEl, true);
      setText(billingErrorEl, t("billing.error_over_limit_cost01"));
      setVisible(billingWarnEl, false);
      setText(billingWarnEl, "");
      return;
    }

    setVisible(billingErrorEl, false);
    setText(billingErrorEl, "");

    if (ratio >= 0.8 && estimateUsed > 0) {
      setVisible(billingWarnEl, true);
      setText(billingWarnEl, t("billing.warn_near_limit"));
      return;
    }

    setVisible(billingWarnEl, false);
    setText(billingWarnEl, "");
  }

  return {
    guideHelpFlow,
    matchHelpTopicFromText,
    updateBillingUi,
    updateHelpAdvice,
  };
}