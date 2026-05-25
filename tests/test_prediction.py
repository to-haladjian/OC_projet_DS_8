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


def test_prediction_responds_to_risk(client, valid_payload):
    # Arrange: same applicant, strong vs weak external credit scores
    low_risk = {**valid_payload, "ext_source_2": 0.9, "ext_source_3": 0.9}
    high_risk = {**valid_payload, "ext_source_2": 0.05, "ext_source_3": 0.05}

    # Act
    p_low = client.post("/predict", json=low_risk).json()["default_probability"]
    p_high = client.post("/predict", json=high_risk).json()["default_probability"]

    # Assert: stronger scores -> lower default risk, and not saturated near 1.0
    # (guards against the regression where the model returned ~1.0 for everyone)
    assert p_high > p_low
    assert p_low < 0.5
