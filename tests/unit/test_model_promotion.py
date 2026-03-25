import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest
from forecasting_api import model_promotion as model_promotion_module
from forecasting_api.model_promotion import (
    evaluate_promotion_candidate,
    load_promotion_registry,
    promote_model,
    save_promotion_registry,
)


def _patch_promotion_side_effects(monkeypatch) -> None:
    monkeypatch.setattr(model_promotion_module, "start_run", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(model_promotion_module, "log_params", lambda payload: None)
    monkeypatch.setattr(model_promotion_module, "log_metrics", lambda payload: None)
    monkeypatch.setattr(model_promotion_module, "log_dict_artifact", lambda name, payload: None)


def test_promotion_candidate_approved_when_thresholds_met():
    decision = evaluate_promotion_candidate(
        "model-1",
        {"coverage": 0.95, "rmse": 10.0, "drift_score": 0.05},
        target_stage="production",
    )
    assert decision.approved is True
    assert decision.reasons == ["approved"]


def test_promotion_candidate_rejected_on_multiple_failures():
    decision = evaluate_promotion_candidate(
        "model-1",
        {"coverage": 0.70, "rmse": 30.0, "drift_score": 0.4},
        target_stage="production",
    )
    assert decision.approved is False
    assert len(decision.reasons) == 3


def test_promotion_candidate_accepts_threshold_boundary():
    decision = evaluate_promotion_candidate(
        "model-1",
        {"coverage": 0.9, "rmse": 20.0, "drift_score": 0.2},
        target_stage="production",
    )
    assert decision.approved is True


def test_promotion_candidate_rejects_invalid_metrics():
    with pytest.raises(ValueError):
        evaluate_promotion_candidate(
            "model-1",
            {"coverage": 1.2, "rmse": -1.0, "drift_score": 0.1},
            target_stage="production",
        )


def test_promotion_candidate_rejects_missing_metric():
    with pytest.raises(ValueError, match="missing metrics"):
        evaluate_promotion_candidate(
            "model-1",
            {"coverage": 0.9, "rmse": 10.0},
            target_stage="production",
        )


def test_promotion_candidate_rejects_non_finite_metric():
    with pytest.raises(ValueError, match="metric must be finite"):
        evaluate_promotion_candidate(
            "model-1",
            {"coverage": 0.9, "rmse": float("nan"), "drift_score": 0.1},
            target_stage="production",
        )


def test_load_promotion_registry_ignores_non_list_payload(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text('{"not": "a-list"}', encoding="utf-8")
    monkeypatch.setenv("RULFM_MODEL_PROMOTION_REGISTRY_PATH", str(registry_path))
    assert load_promotion_registry() == []


def test_save_promotion_registry_roundtrip(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setenv("RULFM_MODEL_PROMOTION_REGISTRY_PATH", str(registry_path))
    entries = [{"model_id": "model-1", "approved": True}]
    save_promotion_registry(entries)
    assert load_promotion_registry() == entries


def test_promote_model_does_not_emit_metric_when_rejected(monkeypatch, tmp_path):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setenv("RULFM_MODEL_PROMOTION_REGISTRY_PATH", str(registry_path))
    _patch_promotion_side_effects(monkeypatch)
    recorded_stages: list[str] = []
    monkeypatch.setattr(
        "forecasting_api.model_promotion.record_model_promotion",
        lambda stage: recorded_stages.append(stage),
    )

    decision = promote_model(
        "model-rejected",
        {"coverage": 0.7, "rmse": 30.0, "drift_score": 0.3},
        target_stage="production",
    )

    assert decision.approved is False
    assert recorded_stages == []
    assert load_promotion_registry()[-1]["approved"] is False


def test_promotion_registry_updates_are_atomic(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setenv("RULFM_MODEL_PROMOTION_REGISTRY_PATH", str(registry_path))
    _patch_promotion_side_effects(monkeypatch)

    def _promote(index: int) -> None:
        promote_model(f"model-{index}", {"coverage": 0.95, "rmse": 5.0, "drift_score": 0.01})

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(_promote, range(12)))

    registry = load_promotion_registry()
    assert len(registry) == 12
    assert {entry["model_id"] for entry in registry} == {f"model-{index}" for index in range(12)}
    assert all(
        entry["metrics"] == {"coverage": 0.95, "rmse": 5.0, "drift_score": 0.01}
        for entry in registry
    )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 12
