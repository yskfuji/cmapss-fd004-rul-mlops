import pytest
from fastapi.testclient import TestClient

from forecasting_api import app as app_module
from forecasting_api import training_helpers
from forecasting_api.app import create_app
from tests.helpers import raising_callable


def _train_payload(algo: str) -> dict[str, object]:
    return {
        "algo": algo,
        "training_hours": 0.1,
        "data": [
            {"series_id": "s1", "timestamp": "2026-03-20T00:00:00Z", "y": 10.0},
            {"series_id": "s1", "timestamp": "2026-03-21T00:00:00Z", "y": 11.0},
            {"series_id": "s1", "timestamp": "2026-03-22T00:00:00Z", "y": 12.0},
            {"series_id": "s1", "timestamp": "2026-03-23T00:00:00Z", "y": 13.0},
            {"series_id": "s1", "timestamp": "2026-03-24T00:00:00Z", "y": 14.0},
            {"series_id": "s1", "timestamp": "2026-03-25T00:00:00Z", "y": 15.0},
            {"series_id": "s1", "timestamp": "2026-03-26T00:00:00Z", "y": 16.0},
            {"series_id": "s1", "timestamp": "2026-03-27T00:00:00Z", "y": 17.0},
        ],
    }
@pytest.mark.experimental
def test_train_endpoint_maps_hybrid_value_error_to_v01(monkeypatch, patched_app_runtime):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    monkeypatch.setattr(
        training_helpers,
        "train_hybrid_entry",
        raising_callable(ValueError("hybrid data invalid")),
    )
    client = TestClient(create_app())
    response = client.post(
        "/v1/train",
        headers={"x-api-key": "test-key"},
        json=_train_payload("gbdt_afno_hybrid_v1"),
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "V01"
    assert body["details"]["error"] == "hybrid data invalid"


@pytest.mark.experimental
def test_train_endpoint_maps_torch_value_error_to_v01(monkeypatch, patched_app_runtime):
    from forecasting_api import torch_forecasters as torch_module

    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_ENABLE_EXPERIMENTAL_MODELS", "1")
    monkeypatch.setattr(
        torch_module,
        "train_univariate_torch_forecaster",
        raising_callable(ValueError("need more torch samples")),
    )
    client = TestClient(create_app())
    response = client.post(
        "/v1/train",
        headers={"x-api-key": "test-key"},
        json=_train_payload("afnocg2"),
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "V01"
    assert body["details"]["error"] == "need more torch samples"


def test_train_endpoint_rejects_experimental_algo_by_default(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    response = client.post(
        "/v1/train",
        headers={"x-api-key": "test-key"},
        json=_train_payload("afnocg2"),
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "M01"
    assert body["details"]["algo"] == "afnocg2"


def test_train_endpoint_accepts_naive_training(monkeypatch, patched_app_runtime):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    response = client.post(
        "/v1/train",
        headers={"x-api-key": "test-key"},
        json=_train_payload("naive"),
    )
    assert response.status_code == 202
    body = response.json()
    assert body["message"] == "accepted"
    assert body["model_id"].startswith("model_")


def test_train_endpoint_requires_two_person_approval_when_enabled(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL", "1")
    client = TestClient(create_app())
    response = client.post(
        "/v1/train",
        headers={"x-api-key": "test-key", "x-tenant-id": "tenant-a"},
        json=_train_payload("naive"),
    )
    assert response.status_code == 403
    body = response.json()
    assert body["error_code"] == "A16"


def test_train_endpoint_accepts_two_person_approval_when_enabled(monkeypatch, patched_app_runtime):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    monkeypatch.setenv("RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL", "1")
    client = TestClient(create_app())
    response = client.post(
        "/v1/train",
        headers={
            "x-api-key": "test-key",
            "x-tenant-id": "tenant-a",
            "x-approved-by": "alice,bob",
            "x-approval-reason": "scheduled retrain",
        },
        json=_train_payload("naive"),
    )
    assert response.status_code == 202


def test_train_endpoint_accepts_oidc_group_claim_approval(monkeypatch, patched_app_runtime):
    monkeypatch.delenv("RULFM_FORECASTING_API_KEY", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL", "1")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_ISSUER", "https://issuer.example")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_AUDIENCE", "rulfm-aud")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_JWKS_URL", "https://issuer.example/jwks.json")
    monkeypatch.setenv("RULFM_FORECASTING_API_TRAIN_APPROVER_GROUPS", "ml-approvers")
    monkeypatch.setattr(
        app_module,
        "_validate_oidc_bearer_token",
        lambda token: {"sub": "alice", "groups": ["ml-approvers", "operators"]},
    )

    client = TestClient(create_app())
    response = client.post(
        "/v1/train",
        headers={
            "authorization": "Bearer oidc-token",
            "x-tenant-id": "tenant-a",
        },
        json=_train_payload("naive"),
    )

    assert response.status_code == 202


def test_train_endpoint_rejects_oidc_principal_without_required_group(monkeypatch):
    monkeypatch.delenv("RULFM_FORECASTING_API_KEY", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("RULFM_FORECASTING_API_REQUIRE_TRAIN_APPROVAL", "1")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_ISSUER", "https://issuer.example")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_AUDIENCE", "rulfm-aud")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_JWKS_URL", "https://issuer.example/jwks.json")
    monkeypatch.setenv("RULFM_FORECASTING_API_TRAIN_APPROVER_GROUPS", "ml-approvers")
    monkeypatch.setattr(
        app_module,
        "_validate_oidc_bearer_token",
        lambda token: {"sub": "alice", "groups": ["readers"]},
    )

    client = TestClient(create_app())
    response = client.post(
        "/v1/train",
        headers={
            "authorization": "Bearer oidc-token",
            "x-tenant-id": "tenant-a",
        },
        json=_train_payload("naive"),
    )

    assert response.status_code == 403
    body = response.json()
    assert body["error_code"] == "A16"