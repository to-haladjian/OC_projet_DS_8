"""Tests for the health endpoint."""


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_structure(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert "api_version" in data
    assert "timestamp" in data
