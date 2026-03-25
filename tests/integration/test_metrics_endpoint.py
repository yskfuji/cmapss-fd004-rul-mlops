from fastapi.testclient import TestClient
from forecasting_api import app as app_module
from forecasting_api.app import create_app


def test_metrics_endpoint_exists(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    response = client.get("/metrics", headers={"x-api-key": "test-key"})
    assert response.status_code == 200
    assert "rulfm_http_requests_total" in response.text


def test_metrics_endpoint_requires_auth(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_KEY", "test-key")
    client = TestClient(create_app())
    response = client.get("/metrics")
    assert response.status_code == 401


def test_metrics_endpoint_accepts_bearer_token(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_BEARER_TOKEN", "metrics-token")
    client = TestClient(create_app())
    response = client.get("/metrics", headers={"authorization": "Bearer metrics-token"})
    assert response.status_code == 200
    assert "rulfm_http_requests_total" in response.text


def test_metrics_endpoint_rejects_wrong_bearer_token(monkeypatch):
    monkeypatch.setenv("RULFM_FORECASTING_API_BEARER_TOKEN", "metrics-token")
    client = TestClient(create_app())
    response = client.get("/metrics", headers={"authorization": "Bearer wrong-token"})
    assert response.status_code == 401


def test_metrics_endpoint_denies_when_auth_is_not_configured(monkeypatch):
    monkeypatch.delenv("RULFM_FORECASTING_API_KEY", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_OIDC_ISSUER", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_OIDC_AUDIENCE", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_OIDC_JWKS_URL", raising=False)
    client = TestClient(create_app())
    response = client.get("/metrics")
    assert response.status_code == 401
    body = response.json()
    assert body["error_code"] == "A12"
    assert "RULFM_FORECASTING_API_KEY" in body["details"]["next_action"]


def test_metrics_endpoint_accepts_oidc_bearer_token(monkeypatch):
    monkeypatch.delenv("RULFM_FORECASTING_API_KEY", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_ISSUER", "https://issuer.example")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_AUDIENCE", "rulfm-aud")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_JWKS_URL", "https://issuer.example/jwks.json")
    monkeypatch.setattr(
        app_module,
        "_validate_oidc_bearer_token",
        lambda token: {"sub": "user-123"},
    )
    client = TestClient(create_app())
    response = client.get("/metrics", headers={"authorization": "Bearer oidc-token"})
    assert response.status_code == 200
    assert "rulfm_http_requests_total" in response.text


def test_metrics_endpoint_rejects_expired_oidc_token(monkeypatch):
    """OIDC token expiry / bad-signature branch: exception in _validate_oidc_bearer_token → 401."""
    from tests.helpers import raising_callable

    monkeypatch.delenv("RULFM_FORECASTING_API_KEY", raising=False)
    monkeypatch.delenv("RULFM_FORECASTING_API_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_ISSUER", "https://issuer.example")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_AUDIENCE", "rulfm-aud")
    monkeypatch.setenv("RULFM_FORECASTING_API_OIDC_JWKS_URL", "https://issuer.example/jwks.json")
    monkeypatch.setattr(
        app_module,
        "_validate_oidc_bearer_token",
        raising_callable(Exception("Token is expired")),
    )
    client = TestClient(create_app())
    response = client.get("/metrics", headers={"authorization": "Bearer expired-token"})
    assert response.status_code == 401

