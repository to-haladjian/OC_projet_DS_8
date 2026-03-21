"""Shared fixtures for all test modules."""

import os

import pytest
from fastapi.testclient import TestClient

# Ensure no DATABASE_URL so no DB connection is attempted during tests
os.environ.pop("DATABASE_URL", None)

from app.main import app


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with @pytest.mark.db when running in CI."""
    if not os.getenv("CI"):
        return
    skip_db = pytest.mark.skip(reason="database not available in CI")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip_db)


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "application_id": 100001,
        "name_contract_type": "Cash loans",
        "code_gender": "F",
        "flag_own_car": "N",
        "name_family_status": "Married",
        "name_education_type": "Secondary / secondary special",
        "amt_income_total": 202500.0,
        "amt_credit": 406597.5,
        "amt_annuity": 24700.5,
        "amt_goods_price": 351000.0,
        "cnt_fam_members": 2.0,
        "days_birth": -9461,
        "days_employed": -637,
        "ext_source_2": 0.262949,
        "ext_source_3": 0.139376,
    }
