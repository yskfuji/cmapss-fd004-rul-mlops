from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from forecasting_api.file_store import load_json, update_json_file


def _coerce_numeric_records(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    features: dict[str, list[float]] = {}
    for record in records:
        values = record.get("x") if isinstance(record.get("x"), dict) else record
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, int | float) and math.isfinite(float(value)):
                features.setdefault(str(key), []).append(float(value))
    return features


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    population_stability_index: float
    baseline_binning_strategy: str
    baseline_requested_bin_count: int
    baseline_selected_bin_count: int


@dataclass(frozen=True)
class DriftReport:
    severity: str
    drift_score: float
    feature_reports: list[FeatureDrift]
    sample_size: int


class DriftDetector:
    def __init__(
        self,
        *,
        psi_threshold: float = 0.2,
        severe_threshold: float = 0.35,
        bins: int = 10,
    ) -> None:
        self.psi_threshold = float(psi_threshold)
        self.severe_threshold = float(severe_threshold)
        self.bins = max(2, int(bins))

    def summarize_baseline(self, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        features = _coerce_numeric_records(records)
        summary: dict[str, dict[str, Any]] = {}
        for name, values in features.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            requested_bin_count = self._effective_bin_count(int(arr.size))
            inner_edges = self._histogram_edges(arr)
            baseline_probabilities = self._histogram_probabilities(arr, inner_edges)
            summary[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "count": int(arr.size),
                "binning_strategy": "adaptive_equal_width_without_empty_bins",
                "requested_bin_count": requested_bin_count,
                "selected_bin_count": len(baseline_probabilities),
                "bin_edges": inner_edges,
                "bin_probabilities": baseline_probabilities,
            }
        return summary

    def detect(
        self,
        baseline_summary: dict[str, dict[str, Any]],
        records: list[dict[str, Any]],
    ) -> DriftReport:
        features = _coerce_numeric_records(records)
        reports: list[FeatureDrift] = []
        for name, baseline in baseline_summary.items():
            values = features.get(name) or []
            if not values:
                continue
            if not _is_valid_feature_summary(baseline):
                continue
            arr = np.asarray(values, dtype=float)
            baseline_mean = float(baseline["mean"])
            candidate_mean = float(np.mean(arr))
            mean_delta = candidate_mean - baseline_mean
            psi = self._population_stability_index(
                baseline_probabilities=[float(item) for item in baseline["bin_probabilities"]],
                candidate_probabilities=self._histogram_probabilities(
                    arr,
                    [float(item) for item in baseline["bin_edges"]],
                ),
            )
            reports.append(
                FeatureDrift(
                    feature=name,
                    baseline_mean=baseline_mean,
                    candidate_mean=candidate_mean,
                    mean_delta=mean_delta,
                    population_stability_index=psi,
                    baseline_binning_strategy=(
                        str(baseline["binning_strategy"])
                        if baseline.get("binning_strategy") is not None
                        else "adaptive_equal_width_without_empty_bins"
                    ),
                    baseline_requested_bin_count=(
                        int(baseline["requested_bin_count"])
                        if baseline.get("requested_bin_count") is not None
                        else len(baseline["bin_probabilities"])
                    ),
                    baseline_selected_bin_count=(
                        int(baseline["selected_bin_count"])
                        if baseline.get("selected_bin_count") is not None
                        else len(baseline["bin_probabilities"])
                    ),
                )
            )

        max_score = max((report.population_stability_index for report in reports), default=0.0)
        if max_score >= self.severe_threshold:
            severity = "high"
        elif max_score >= self.psi_threshold:
            severity = "medium"
        else:
            severity = "low"
        return DriftReport(
            severity=severity,
            drift_score=max_score,
            feature_reports=reports,
            sample_size=sum(len(values) for values in features.values()),
        )

    @staticmethod
    def _population_stability_index(
        *,
        baseline_probabilities: list[float],
        candidate_probabilities: list[float],
    ) -> float:
        epsilon = 1e-6
        psi = 0.0
        if len(baseline_probabilities) != len(candidate_probabilities):
            raise ValueError(
                "baseline and candidate probability bins must have identical lengths"
            )
        for baseline_ratio, candidate_ratio in zip(baseline_probabilities, candidate_probabilities):
            baseline_safe = max(float(baseline_ratio), epsilon)
            candidate_safe = max(float(candidate_ratio), epsilon)
            psi += (candidate_safe - baseline_safe) * math.log(
                candidate_safe / baseline_safe
            )
        return float(psi)

    def _histogram_edges(self, values: np.ndarray) -> list[float]:
        if values.size == 0:
            return []
        if math.isclose(float(np.min(values)), float(np.max(values)), rel_tol=0.0, abs_tol=1e-12):
            # A constant feature collapses to a single bucket.
            # PSI stays 0 unless the sensor regains variance.
            return []
        effective_bins = self._effective_bin_count(int(values.size))
        for candidate_bins in range(effective_bins, 1, -1):
            inner_edges = self._dedupe_inner_edges(
                np.histogram_bin_edges(values, bins=candidate_bins)
            )
            counts = self._histogram_counts(values, inner_edges)
            if np.all(counts > 0):
                return inner_edges
        # If no multi-bin split avoids empty buckets, collapse to a single bucket
        # instead of reintroducing epsilon-dominated empty-bin probabilities.
        return []

    def _effective_bin_count(self, sample_size: int) -> int:
        if sample_size <= 0:
            return 2
        # Keep roughly five baseline samples per bucket so PSI bins are not dominated by epsilon.
        return min(self.bins, max(2, sample_size // 5))

    @staticmethod
    def _dedupe_inner_edges(edges: np.ndarray) -> list[float]:
        deduped: list[float] = []
        for edge in edges[1:-1]:
            edge = float(edge)
            if not deduped or not math.isclose(edge, deduped[-1], rel_tol=0.0, abs_tol=1e-12):
                deduped.append(edge)
        return deduped

    @staticmethod
    def _histogram_counts(values: np.ndarray, inner_edges: list[float]) -> np.ndarray:
        bins = np.asarray([-np.inf, *inner_edges, np.inf], dtype=float)
        counts, _ = np.histogram(values, bins=bins)
        return counts.astype(float)

    @staticmethod
    def _histogram_probabilities(values: np.ndarray, inner_edges: list[float]) -> list[float]:
        counts = DriftDetector._histogram_counts(values, inner_edges)
        counts = counts.astype(float) + 1e-6
        probabilities = counts / float(np.sum(counts))
        return [float(item) for item in probabilities]


def _is_valid_feature_summary(feature_summary: dict[str, Any]) -> bool:
    edges = feature_summary.get("bin_edges")
    probs = feature_summary.get("bin_probabilities")
    return isinstance(edges, list) and isinstance(probs, list) and len(probs) == len(edges) + 1


def has_valid_baseline(summary: dict[str, dict[str, Any]]) -> bool:
    return any(_is_valid_feature_summary(item) for item in summary.values())


def has_sufficient_baseline_samples(
    summary: dict[str, dict[str, Any]],
    *,
    minimum_count: int = 50,
) -> bool:
    return any(int(item.get("count", 0)) >= minimum_count for item in summary.values())


def default_baseline_path() -> Path:
    raw = os.getenv("RULFM_DRIFT_BASELINE_PATH", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[1] / "forecasting_api" / "data" / "drift_baseline.json"


def load_baseline(path: Path | None = None) -> dict[str, dict[str, Any]]:
    baseline_path = path or default_baseline_path()
    payload: dict[str, dict[str, Any]] = load_json(baseline_path, {})
    return payload if isinstance(payload, dict) else {}


def save_baseline(summary: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    baseline_path = path or default_baseline_path()
    empty_summary: dict[str, dict[str, Any]] = {}
    update_json_file(baseline_path, default=empty_summary, updater=lambda _: summary)


def drift_report_to_dict(report: DriftReport) -> dict[str, Any]:
    return {
        "severity": report.severity,
        "drift_score": report.drift_score,
        "sample_size": report.sample_size,
        "feature_reports": [asdict(item) for item in report.feature_reports],
    }
