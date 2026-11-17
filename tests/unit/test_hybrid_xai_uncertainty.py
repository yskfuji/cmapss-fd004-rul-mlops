import pytest

from forecasting_api.hybrid_xai_uncertainty import (
    apply_soft_gate_envelope_interval,
    condition_advantage_map,
    gate_step_payload,
    interval_overlap_ratio,
    normalize_advantage_lookup,
    quantile_nearest_rank,
    sigmoid,
    soft_gate_feature_scales,
    soft_gate_outputs,
    soft_gate_weight_entropy,
)


def _aligned_outputs():
    gbdt = {
        "y_true": [10.0, 12.0, 8.0],
        "y_pred": [9.0, 11.5, 8.5],
        "lower": [8.0, 10.0, 7.0],
        "upper": [10.0, 13.0, 10.0],
        "condition_key": ["a", "a", "b"],
        "tail_pos": [0.1, 0.5, 0.9],
    }
    afno = {
        "y_true": [10.0, 12.0, 8.0],
        "y_pred": [10.5, 12.5, 7.5],
        "lower": [9.5, 11.5, 6.5],
        "upper": [11.0, 13.5, 8.5],
    }
    return gbdt, afno


def test_sigmoid_and_quantile_helpers_cover_edges() -> None:
    assert 0.5 < sigmoid(1.0) < 1.0
    assert 0.0 < sigmoid(-1.0) < 0.5
    assert quantile_nearest_rank([], 0.5) == 0.0
    assert quantile_nearest_rank([1.0, 3.0, 2.0], 0.5) == 2.0
    assert quantile_nearest_rank([1.0, 2.0, 3.0], -1.0) == 1.0
    assert quantile_nearest_rank([1.0, 2.0, 3.0], 2.0) == 3.0


def test_interval_overlap_ratio_handles_overlap_and_degenerate_union() -> None:
    assert interval_overlap_ratio(0.0, 2.0, 1.0, 3.0) == 1.0 / 3.0
    assert interval_overlap_ratio(1.0, 1.0, 1.0, 1.0) == 1.0


def test_soft_gate_feature_scales_and_advantage_maps() -> None:
    gbdt, afno = _aligned_outputs()
    scales = soft_gate_feature_scales(gbdt, afno)
    assert scales["delta_scale"] >= 1.0
    assert scales["width_scale"] >= 1.0

    normalized = normalize_advantage_lookup({"a": [2.0, 4.0], "b": [-1.0]})
    assert set(normalized) == {"a", "b"}
    assert all(-1.0 <= value <= 1.0 for value in normalized.values())

    advantage = condition_advantage_map(gbdt, afno)
    assert set(advantage) == {"a", "b"}


def test_soft_gate_outputs_requires_aligned_targets() -> None:
    gbdt, afno = _aligned_outputs()
    afno["y_true"] = [10.0]
    with pytest.raises(ValueError, match="aligned outputs"):
        soft_gate_outputs(
            gbdt,
            afno,
            temperature=0.35,
            tau=0.0,
            coef_delta=1.0,
            coef_overlap=0.5,
            coef_width=0.25,
            coef_tail=0.1,
            coef_condition=0.2,
        )


def test_soft_gate_outputs_and_interval_envelope_return_expected_shapes() -> None:
    gbdt, afno = _aligned_outputs()
    outputs = soft_gate_outputs(
        gbdt,
        afno,
        temperature=0.35,
        tau=0.0,
        coef_delta=1.0,
        coef_overlap=0.5,
        coef_width=0.25,
        coef_tail=0.1,
        coef_condition=0.2,
        coef_correctness=0.3,
        condition_advantage={"a": 0.5, "b": -0.25},
        correctness_priors={
            "delta": {"0": 0.2},
            "overlap": {"0": 0.1},
            "tail": {"0": 0.05},
            "width": {"0": -0.1},
            "condition": {"a": 0.3, "b": -0.2},
            "delta_edges": {"0": 1.0},
            "width_edges": {"0": 1.0},
        },
    )
    assert len(outputs["y_pred"]) == 3
    assert len(outputs["afno_weight"]) == 3
    assert len(outputs["gate_loo_correctness"]) == 3
    assert all(0.0 <= weight <= 1.0 for weight in outputs["afno_weight"])

    enveloped = apply_soft_gate_envelope_interval(outputs, gbdt, afno, interval_scale=1.2)
    assert len(enveloped["lower"]) == 3
    assert len(enveloped["upper"]) == 3
    assert all(lo <= hi for lo, hi in zip(enveloped["lower"], enveloped["upper"], strict=False))


def test_soft_gate_weight_entropy_and_step_payload() -> None:
    entropy = soft_gate_weight_entropy([0.2, 0.5, 0.8])
    assert entropy > 0.0
    assert soft_gate_weight_entropy([]) == 0.0

    payload = gate_step_payload(
        g_pred=9.0,
        g_lower=8.0,
        g_upper=10.0,
        a_pred=10.5,
        a_lower=9.5,
        a_upper=11.0,
        gate_meta={
            "condition_advantage": {"a": 0.5},
            "coef_delta": 1.0,
            "tau": 0.0,
            "delta_scale": 1.0,
            "coef_overlap": 0.5,
            "coef_width": 0.25,
            "width_scale": 1.0,
            "coef_tail": 0.1,
            "coef_condition": 0.2,
            "temperature": 0.35,
        },
        condition_key="a",
        tail_pos=0.8,
    )
    assert 0.0 <= payload["afno_weight"] <= 1.0
    assert payload["gbdt_weight"] == 1.0 - payload["afno_weight"]