from forecasting_api.model_promotion import load_promotion_registry, promote_model


def test_promote_model_persists_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("RULFM_MODEL_PROMOTION_REGISTRY_PATH", str(tmp_path / "registry.json"))
    decision = promote_model("model-1", {"coverage": 0.99, "rmse": 3.0, "drift_score": 0.01})
    assert decision.approved is True
    registry = load_promotion_registry()
    assert registry[-1]["model_id"] == "model-1"
