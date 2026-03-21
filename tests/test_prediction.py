"""Tests for the prediction endpoint."""


def test_prediction_valid_payload(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "credit_approved" in data
    assert 0 <= data["default_probability"] <= 1
    assert isinstance(data["credit_approved"], bool)
    assert data["application_id"] == valid_payload["application_id"]


def test_prediction_without_application_id(client, valid_payload):
    payload = {k: v for k, v in valid_payload.items() if k != "application_id"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["application_id"] is None


def test_prediction_missing_required_field(client, valid_payload):
    payload = {k: v for k, v in valid_payload.items() if k != "amt_credit"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_out_of_range_value(client, valid_payload):
    payload = {**valid_payload, "amt_income_total": -100}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_invalid_type(client, valid_payload):
    payload = {**valid_payload, "amt_income_total": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
