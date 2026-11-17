from fastapi.testclient import TestClient

from forecasting_api.app import create_app


def test_health_endpoint_returns_ok():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["public_profile"] == "gbdt-only"
    assert body["experimental_models"] == "disabled"
