from __future__ import annotations

from collections.abc import Callable

from forecasting_api.errors import ApiError
from forecasting_api.metrics import record_drift
from forecasting_api.schemas import (
    DriftBaselineRequest,
    DriftBaselineResponse,
    DriftBaselineStatusResponse,
    DriftReportRequest,
    DriftReportResponse,
)
from monitoring.drift_detector import (
    DriftDetector,
    drift_report_to_dict,
    has_sufficient_baseline_samples,
    has_valid_baseline,
    load_baseline,
    save_baseline,
)


def persist_drift_baseline(req: DriftBaselineRequest) -> DriftBaselineResponse:
    detector = DriftDetector()
    summary = detector.summarize_baseline(req.baseline_records)
    if not has_valid_baseline(summary):
        raise ApiError(
            status_code=400,
            error_code="V01",
            message="baseline が不正です",
            details={"next_action": "数値特徴量を含む baseline_records を指定してください"},
        )
    save_baseline(summary)
    return DriftBaselineResponse(
        feature_count=len(summary),
        sample_size=sum(int(item.get("count", 0)) for item in summary.values()),
        persisted=True,
        feature_summaries=[
            {
                "feature": name,
                "count": int(item.get("count", 0)),
                "binning_strategy": str(item.get("binning_strategy") or ""),
                "requested_bin_count": int(item.get("requested_bin_count", 0)),
                "selected_bin_count": int(item.get("selected_bin_count", 0)),
            }
            for name, item in summary.items()
        ],
    )


def get_drift_baseline_status() -> DriftBaselineStatusResponse:
    summary = load_baseline()
    baseline_exists = has_valid_baseline(summary)
    return DriftBaselineStatusResponse(
        baseline_exists=baseline_exists,
        feature_count=len(summary),
        sample_size=sum(int(item.get("count", 0)) for item in summary.values()),
        sufficient_samples=(
            has_sufficient_baseline_samples(summary, minimum_count=50) if baseline_exists else False
        ),
        feature_summaries=[
            {
                "feature": name,
                "count": int(item.get("count", 0)),
                "binning_strategy": str(item.get("binning_strategy") or ""),
                "requested_bin_count": int(item.get("requested_bin_count", 0)),
                "selected_bin_count": int(item.get("selected_bin_count", 0)),
            }
            for name, item in summary.items()
        ],
    )


def generate_drift_report(
    req: DriftReportRequest, *, log_ephemeral_baseline: Callable[[], None] | None = None
) -> DriftReportResponse:
    detector = DriftDetector()
    baseline_summary = (
        detector.summarize_baseline(req.baseline_records)
        if req.baseline_records
        else load_baseline()
    )
    if req.baseline_records and log_ephemeral_baseline is not None:
        log_ephemeral_baseline()
    if not has_valid_baseline(baseline_summary):
        raise ApiError(
            status_code=400,
            error_code="V01",
            message="baseline が未設定です",
            details={"next_action": "baseline_records を指定して再試行してください"},
        )
    report = detector.detect(baseline_summary, req.candidate_records)
    payload = drift_report_to_dict(report)
    feature_scores = {
        item["feature"]: float(item["population_stability_index"])
        for item in payload["feature_reports"]
    }
    record_drift(feature_scores, report.severity)
    return DriftReportResponse(**payload)