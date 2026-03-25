from __future__ import annotations

import math
from typing import Any


def sigmoid(value: float) -> float:
    x = float(value)
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def quantile_nearest_rank(values: list[float], q: float) -> float:
    finite = sorted(
        float(value)
        for value in values
        if isinstance(value, int | float) and math.isfinite(float(value))
    )
    if not finite:
        return 0.0
    qq = 0.0 if q <= 0 else 1.0 if q >= 1 else float(q)
    idx = max(0, min(len(finite) - 1, int(math.ceil(qq * len(finite))) - 1))
    return float(finite[idx])


def interval_overlap_ratio(g_lower: float, g_upper: float, a_lower: float, a_upper: float) -> float:
    g_lo, g_hi = min(float(g_lower), float(g_upper)), max(float(g_lower), float(g_upper))
    a_lo, a_hi = min(float(a_lower), float(a_upper)), max(float(a_lower), float(a_upper))
    overlap = max(0.0, min(g_hi, a_hi) - max(g_lo, a_lo))
    union = max(g_hi, a_hi) - min(g_lo, a_lo)
    if union <= 1e-8:
        return 1.0
    return float(overlap / union)


def soft_gate_feature_scales(
    gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]
) -> dict[str, float]:
    pred_diffs = [
        abs(float(a) - float(g))
        for g, a in zip(
            gbdt_outputs.get("y_pred") or [], afno_outputs.get("y_pred") or [], strict=False
        )
    ]
    width_diffs = [
        abs((float(g_upper) - float(g_lower)) - (float(a_upper) - float(a_lower)))
        for g_lower, g_upper, a_lower, a_upper in zip(
            gbdt_outputs.get("lower") or [],
            gbdt_outputs.get("upper") or [],
            afno_outputs.get("lower") or [],
            afno_outputs.get("upper") or [],
            strict=False,
        )
    ]
    return {
        "delta_scale": max(quantile_nearest_rank(pred_diffs, 0.75), 1.0) if pred_diffs else 1.0,
        "width_scale": max(quantile_nearest_rank(width_diffs, 0.75), 1.0) if width_diffs else 1.0,
    }


def normalize_advantage_lookup(grouped: dict[str, list[float]]) -> dict[str, float]:
    raw = {str(key): float(sum(values) / len(values)) for key, values in grouped.items() if values}
    if not raw:
        return {}
    scale = max(quantile_nearest_rank([abs(value) for value in raw.values()], 0.9), 1e-6)
    return {key: float(max(-1.0, min(1.0, value / scale))) for key, value in raw.items()}


def condition_advantage_map(
    gbdt_outputs: dict[str, list[float]], afno_outputs: dict[str, list[float]]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for idx, condition_key in enumerate(gbdt_outputs.get("condition_key") or []):
        if idx >= len(gbdt_outputs.get("y_true") or []) or idx >= len(
            afno_outputs.get("y_true") or []
        ):
            continue
        truth = float((gbdt_outputs.get("y_true") or [])[idx])
        g_pred = float((gbdt_outputs.get("y_pred") or [])[idx])
        a_pred = float((afno_outputs.get("y_pred") or [])[idx])
        grouped.setdefault(str(condition_key), []).append(abs(truth - g_pred) - abs(truth - a_pred))
    return normalize_advantage_lookup(grouped)


def _soft_gate_bucket(value: float, edges: list[float]) -> str:
    for idx, edge in enumerate(edges):
        if float(value) <= float(edge):
            return str(idx)
    return str(len(edges))


def soft_gate_outputs(
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    temperature: float,
    tau: float,
    coef_delta: float,
    coef_overlap: float,
    coef_width: float,
    coef_tail: float,
    coef_condition: float,
    coef_correctness: float = 0.0,
    condition_advantage: dict[str, float] | None = None,
    correctness_priors: dict[str, dict[str, float]] | None = None,
) -> dict[str, list[float]]:
    if len(gbdt_outputs.get("y_true") or []) != len(afno_outputs.get("y_true") or []):
        raise ValueError("Soft gate requires aligned outputs")
    scales = soft_gate_feature_scales(gbdt_outputs, afno_outputs)
    delta_scale = float(scales["delta_scale"])
    width_scale = float(scales["width_scale"])
    safe_temperature = max(float(temperature), 1e-6)
    y_pred: list[float] = []
    afno_weight: list[float] = []
    gbdt_weight: list[float] = []
    correctness_score: list[float] = []
    gate_score: list[float] = []
    gate_term_delta: list[float] = []
    gate_term_overlap: list[float] = []
    gate_term_width: list[float] = []
    gate_term_tail: list[float] = []
    gate_term_condition: list[float] = []
    gate_term_correctness: list[float] = []
    gate_loo_delta: list[float] = []
    gate_loo_overlap: list[float] = []
    gate_loo_width: list[float] = []
    gate_loo_tail: list[float] = []
    gate_loo_condition: list[float] = []
    gate_loo_correctness: list[float] = []
    condition_keys = [str(key) for key in gbdt_outputs.get("condition_key") or []]
    tail_pos = [float(value) for value in gbdt_outputs.get("tail_pos") or []]
    priors = correctness_priors if isinstance(correctness_priors, dict) else {}
    delta_lookup = priors.get("delta") or {}
    overlap_lookup = priors.get("overlap") or {}
    tail_lookup = priors.get("tail") or {}
    width_lookup = priors.get("width") or {}
    condition_lookup = priors.get("condition") or {}
    delta_edges = [
        float(value)
        for _, value in sorted(
            ((int(k), v) for k, v in (priors.get("delta_edges") or {}).items()),
            key=lambda item: item[0],
        )
    ]
    width_edges = [
        float(value)
        for _, value in sorted(
            ((int(k), v) for k, v in (priors.get("width_edges") or {}).items()),
            key=lambda item: item[0],
        )
    ]
    overlap_edges = [0.2, 0.4, 0.6, 0.8]
    tail_edges = [0.2, 0.4, 0.6, 0.8]
    for idx, (g_pred, a_pred, g_lower, g_upper, a_lower, a_upper) in enumerate(
        zip(
            gbdt_outputs.get("y_pred") or [],
            afno_outputs.get("y_pred") or [],
            gbdt_outputs.get("lower") or [],
            gbdt_outputs.get("upper") or [],
            afno_outputs.get("lower") or [],
            afno_outputs.get("upper") or [],
            strict=False,
        )
    ):
        overlap_ratio = interval_overlap_ratio(
            float(g_lower), float(g_upper), float(a_lower), float(a_upper)
        )
        g_width = max(0.0, float(g_upper) - float(g_lower))
        a_width = max(0.0, float(a_upper) - float(a_lower))
        delta = float(a_pred) - float(g_pred)
        condition_key = condition_keys[idx] if idx < len(condition_keys) else "global"
        tail_feature = tail_pos[idx] if idx < len(tail_pos) else 0.5
        condition_feature = float((condition_advantage or {}).get(condition_key, 0.0))
        correctness_feature = 0.0
        correctness_feature += float(condition_lookup.get(condition_key, 0.0))
        correctness_feature += float(
            tail_lookup.get(_soft_gate_bucket(tail_feature, tail_edges), 0.0)
        )
        correctness_feature += float(
            delta_lookup.get(_soft_gate_bucket(abs(delta), delta_edges), 0.0)
        )
        correctness_feature += float(
            overlap_lookup.get(_soft_gate_bucket(overlap_ratio, overlap_edges), 0.0)
        )
        correctness_feature += float(
            width_lookup.get(_soft_gate_bucket(abs(g_width - a_width), width_edges), 0.0)
        )
        correctness_feature /= 5.0
        term_delta = float(coef_delta) * ((-delta - float(tau)) / delta_scale)
        term_overlap = float(coef_overlap) * (1.0 - overlap_ratio)
        term_width = float(coef_width) * ((g_width - a_width) / width_scale)
        term_tail = float(coef_tail) * (tail_feature - 0.5)
        term_condition = float(coef_condition) * condition_feature
        term_correctness = float(coef_correctness) * correctness_feature
        score = (
            term_delta + term_overlap + term_width + term_tail + term_condition + term_correctness
        )
        weight = sigmoid(score / safe_temperature)
        weight_without_delta = sigmoid((score - term_delta) / safe_temperature)
        weight_without_overlap = sigmoid((score - term_overlap) / safe_temperature)
        weight_without_width = sigmoid((score - term_width) / safe_temperature)
        weight_without_tail = sigmoid((score - term_tail) / safe_temperature)
        weight_without_condition = sigmoid((score - term_condition) / safe_temperature)
        weight_without_correctness = sigmoid((score - term_correctness) / safe_temperature)
        point_gap = float(a_pred) - float(g_pred)
        afno_weight.append(weight)
        gbdt_weight.append(1.0 - weight)
        correctness_score.append(correctness_feature)
        gate_score.append(score)
        gate_term_delta.append(term_delta)
        gate_term_overlap.append(term_overlap)
        gate_term_width.append(term_width)
        gate_term_tail.append(term_tail)
        gate_term_condition.append(term_condition)
        gate_term_correctness.append(term_correctness)
        gate_loo_delta.append((weight - weight_without_delta) * point_gap)
        gate_loo_overlap.append((weight - weight_without_overlap) * point_gap)
        gate_loo_width.append((weight - weight_without_width) * point_gap)
        gate_loo_tail.append((weight - weight_without_tail) * point_gap)
        gate_loo_condition.append((weight - weight_without_condition) * point_gap)
        gate_loo_correctness.append((weight - weight_without_correctness) * point_gap)
        y_pred.append((1.0 - weight) * float(g_pred) + weight * float(a_pred))
    return {
        "y_true": list(gbdt_outputs.get("y_true") or []),
        "y_pred": y_pred,
        "lower": list(gbdt_outputs.get("lower") or []),
        "upper": list(gbdt_outputs.get("upper") or []),
        "condition_key": condition_keys[: len(y_pred)],
        "tail_pos": tail_pos[: len(y_pred)],
        "afno_weight": afno_weight,
        "gbdt_weight": gbdt_weight,
        "gate_score": gate_score,
        "gate_term_delta": gate_term_delta,
        "gate_term_overlap": gate_term_overlap,
        "gate_term_width": gate_term_width,
        "gate_term_tail": gate_term_tail,
        "gate_term_condition": gate_term_condition,
        "gate_term_correctness": gate_term_correctness,
        "gate_loo_delta": gate_loo_delta,
        "gate_loo_overlap": gate_loo_overlap,
        "gate_loo_width": gate_loo_width,
        "gate_loo_tail": gate_loo_tail,
        "gate_loo_condition": gate_loo_condition,
        "gate_loo_correctness": gate_loo_correctness,
        "correctness_score": correctness_score,
    }


def apply_soft_gate_envelope_interval(
    soft_outputs: dict[str, list[float]],
    gbdt_outputs: dict[str, list[float]],
    afno_outputs: dict[str, list[float]],
    *,
    interval_scale: float,
) -> dict[str, list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for point_pred, g_lower, g_upper, a_lower, a_upper in zip(
        soft_outputs.get("y_pred") or [],
        gbdt_outputs.get("lower") or [],
        gbdt_outputs.get("upper") or [],
        afno_outputs.get("lower") or [],
        afno_outputs.get("upper") or [],
        strict=False,
    ):
        envelope_lower = min(float(g_lower), float(a_lower))
        envelope_upper = max(float(g_upper), float(a_upper))
        left_span = max(0.0, float(point_pred) - envelope_lower)
        right_span = max(0.0, envelope_upper - float(point_pred))
        lower.append(float(point_pred) - float(interval_scale) * left_span)
        upper.append(float(point_pred) + float(interval_scale) * right_span)
    return {**soft_outputs, "lower": lower, "upper": upper}


def soft_gate_weight_entropy(weights: list[float]) -> float:
    if not weights:
        return 0.0
    entropy_terms: list[float] = []
    for value in weights:
        prob = min(max(float(value), 1e-6), 1.0 - 1e-6)
        entropy_terms.append(-(prob * math.log(prob) + (1.0 - prob) * math.log(1.0 - prob)))
    return float(sum(entropy_terms) / len(entropy_terms)) if entropy_terms else 0.0


def gate_step_payload(
    *,
    g_pred: float,
    g_lower: float,
    g_upper: float,
    a_pred: float,
    a_lower: float,
    a_upper: float,
    gate_meta: dict[str, Any],
    condition_key: str,
    tail_pos: float,
) -> dict[str, float]:
    overlap_ratio = interval_overlap_ratio(g_lower, g_upper, a_lower, a_upper)
    g_width = max(0.0, float(g_upper) - float(g_lower))
    a_width = max(0.0, float(a_upper) - float(a_lower))
    delta = float(a_pred) - float(g_pred)
    condition_feature = (
        float((gate_meta.get("condition_advantage") or {}).get(condition_key, 0.0))
        if isinstance(gate_meta.get("condition_advantage"), dict)
        else 0.0
    )
    term_delta = float(gate_meta.get("coef_delta") or 1.0) * (
        (-delta - float(gate_meta.get("tau") or 0.0))
        / max(float(gate_meta.get("delta_scale") or 1.0), 1e-6)
    )
    term_overlap = float(gate_meta.get("coef_overlap") or 0.0) * (1.0 - overlap_ratio)
    term_width = float(gate_meta.get("coef_width") or 0.0) * (
        (g_width - a_width) / max(float(gate_meta.get("width_scale") or 1.0), 1e-6)
    )
    term_tail = float(gate_meta.get("coef_tail") or 0.0) * (float(tail_pos) - 0.5)
    term_condition = float(gate_meta.get("coef_condition") or 0.0) * condition_feature
    score = term_delta + term_overlap + term_width + term_tail + term_condition
    afno_weight = sigmoid(score / max(float(gate_meta.get("temperature") or 0.35), 1e-6))
    return {
        "score": float(score),
        "afno_weight": float(afno_weight),
        "gbdt_weight": float(1.0 - afno_weight),
        "term_delta": float(term_delta),
        "term_overlap": float(term_overlap),
        "term_width": float(term_width),
        "term_tail": float(term_tail),
        "term_condition": float(term_condition),
    }
