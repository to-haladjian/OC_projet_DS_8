"""Tests for feature engineering and preprocessing.

These cover the application-table transforms that must stay faithful to Part 1
(encodings, the DAYS_EMPLOYED sentinel, engineered ratios) and the date -> days
conversion introduced for the date-based inputs.

All tests follow the Arrange-Act-Assert structure.
"""

from datetime import date, datetime

import numpy as np
import pytest

from app.services.preprocessing import (
    _coerce_date,
    _days_before,
    _feature_list,
    build_feature_dataframe,
    preprocess_application,
)


def _base_input(**overrides):
    """A minimal valid application payload, with optional field overrides."""
    data = {
        "name_contract_type": "Cash loans",
        "code_gender": "F",
        "flag_own_car": "N",
        "name_family_status": "Married",
        "name_education_type": "Secondary / secondary special",
        "amt_income_total": 200000.0,
        "amt_credit": 400000.0,
        "amt_annuity": 20000.0,
        "cnt_fam_members": 2.0,
        "birth_date": "1986-04-20",
        "employment_start_date": "2020-01-01",
    }
    data.update(overrides)
    return data


# --- date helpers ---

@pytest.mark.parametrize("value, expected", [
    (None, None),
    ("", None),
    ("2020-01-15", date(2020, 1, 15)),
    (date(2020, 1, 15), date(2020, 1, 15)),
    (datetime(2020, 1, 15, 9, 30), date(2020, 1, 15)),
])
def test_coerce_date_accepts_various_forms(value, expected):
    # Arrange / Act
    result = _coerce_date(value)

    # Assert
    assert result == expected


@pytest.mark.parametrize("value, expected", [
    (date(2020, 1, 1), -30),
    ("2020-01-01", -30),
    (None, None),
])
def test_days_before_is_negative_for_past_dates(value, expected):
    # Arrange
    reference = date(2020, 1, 31)

    # Act
    result = _days_before(value, reference)

    # Assert
    assert result == expected


# --- encodings (must match Part 1 pd.factorize order) ---

@pytest.mark.parametrize("gender, expected", [("M", 0), ("F", 1)])
def test_code_gender_encoding_matches_part1(gender, expected):
    # Arrange: Part 1 factorize (after dropping XNA) yields M=0, F=1
    data = _base_input(code_gender=gender)

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert df["CODE_GENDER"].iloc[0] == expected


@pytest.mark.parametrize("car, expected", [("N", 0), ("Y", 1)])
def test_flag_own_car_encoding(car, expected):
    # Arrange
    data = _base_input(flag_own_car=car)

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert df["FLAG_OWN_CAR"].iloc[0] == expected


def test_one_hot_columns_set_when_value_matches():
    # Arrange
    data = _base_input(
        name_contract_type="Revolving loans",
        name_family_status="Married",
        name_education_type="Higher education",
        organization_type="Self-employed",
    )

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert df["NAME_CONTRACT_TYPE_Revolving_loans"].iloc[0] == 1
    assert df["NAME_FAMILY_STATUS_Married"].iloc[0] == 1
    assert df["NAME_EDUCATION_TYPE_Higher_education"].iloc[0] == 1
    assert df["ORGANIZATION_TYPE_Self_employed"].iloc[0] == 1


def test_one_hot_columns_zero_when_value_differs():
    # Arrange
    data = _base_input(
        name_contract_type="Cash loans",
        name_family_status="Single / not married",
    )

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert df["NAME_CONTRACT_TYPE_Revolving_loans"].iloc[0] == 0
    assert df["NAME_FAMILY_STATUS_Married"].iloc[0] == 0


# --- DAYS_EMPLOYED sentinel (Part 1 maps 365243 -> NaN for imputation) ---

def test_days_employed_present_is_negative():
    # Arrange
    data = _base_input(employment_start_date="2020-01-01")

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert df["DAYS_EMPLOYED"].iloc[0] < 0


def test_days_employed_missing_is_nan():
    # Arrange: no employment date means unemployed/retired
    data = _base_input()
    data.pop("employment_start_date")

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert np.isnan(df["DAYS_EMPLOYED"].iloc[0])


def test_days_birth_is_negative():
    # Arrange
    data = _base_input(birth_date="1986-04-20")

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert df["DAYS_BIRTH"].iloc[0] < 0


# --- engineered ratios ---

def test_engineered_ratios():
    # Arrange
    data = _base_input(
        amt_income_total=200000.0, amt_credit=400000.0,
        amt_annuity=20000.0, cnt_fam_members=2.0,
    )

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert df["INCOME_CREDIT_PERC"].iloc[0] == 200000.0 / 400000.0
    assert df["PAYMENT_RATE"].iloc[0] == 20000.0 / 400000.0
    assert df["ANNUITY_INCOME_PERC"].iloc[0] == 20000.0 / 200000.0
    assert df["INCOME_PER_PERSON"].iloc[0] == 200000.0 / 2.0


# --- frame / array contracts ---

def test_build_feature_dataframe_shape_and_columns():
    # Arrange
    data = _base_input()

    # Act
    df = build_feature_dataframe(data)

    # Assert
    assert df.shape[0] == 1
    assert list(df.columns) == _feature_list
    assert not np.isinf(df.to_numpy(dtype="float64")).any()


def test_preprocess_application_returns_array_and_pops_id():
    # Arrange
    data = _base_input(application_id=4242)

    # Act
    app_id, arr = preprocess_application(data)

    # Assert
    assert app_id == 4242
    assert "application_id" not in data  # popped during preprocessing
    assert arr.shape == (1, len(_feature_list))
    assert arr.dtype == np.float32
