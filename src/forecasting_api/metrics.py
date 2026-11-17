from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "rulfm_http_requests_total",
    "Total HTTP requests handled by the API.",
    labelnames=("method", "path", "status_code"),
)
REQUEST_LATENCY = Histogram(
    "rulfm_http_request_latency_seconds",
    "Latency of HTTP requests.",
    labelnames=("method", "path"),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
MODEL_PROMOTIONS = Counter(
    "rulfm_model_promotions_total",
    "Count of model promotions by stage.",
    labelnames=("stage",),
)
DRIFT_SCORE = Gauge(
    "rulfm_feature_drift_score",
    "Feature-level drift score.",
    labelnames=("feature",),
)
DRIFT_EVENTS = Counter(
    "rulfm_drift_events_total",
    "Number of drift events detected.",
    labelnames=("severity",),
)


def metrics_enabled() -> bool:
    return os.getenv("RULFM_METRICS_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


@contextmanager
def track_request(method: str, path: str) -> Iterator[list[int]]:
    started = time.perf_counter()
    status_holder = [500]
    try:
        yield status_holder
    finally:
        if metrics_enabled():
            REQUEST_COUNT.labels(method=method, path=path, status_code=str(status_holder[0])).inc()
            REQUEST_LATENCY.labels(method=method, path=path).observe(time.perf_counter() - started)


def render_metrics() -> tuple[bytes, str]:
    payload = generate_latest()
    return payload, CONTENT_TYPE_LATEST


def record_drift(feature_scores: dict[str, float], severity: str) -> None:
    if not metrics_enabled():
        return
    for feature, score in feature_scores.items():
        DRIFT_SCORE.labels(feature=feature).set(float(score))
    DRIFT_EVENTS.labels(severity=severity).inc()


def record_model_promotion(stage: str) -> None:
    if metrics_enabled():
        MODEL_PROMOTIONS.labels(stage=stage).inc()
