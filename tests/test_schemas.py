"""Tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from app.schemas.credit import LoanApplicationInput


def test_valid_input():
    data = LoanApplicationInput(
        name_contract_type="Cash loans",
        code_gender="F",
        flag_own_car="N",
        name_family_status="Married",
        name_education_type="Secondary / secondary special",
        amt_income_total=202500.0,
        amt_credit=406597.5,
        amt_annuity=24700.5,
        days_birth=-9461,
        days_employed=-637,
    )
    assert data.application_id is None
    assert data.amt_income_total == 202500.0


def test_negative_income_rejected():
    with pytest.raises(ValidationError):
        LoanApplicationInput(
            name_contract_type="Cash loans",
            code_gender="F",
            flag_own_car="N",
            name_family_status="Married",
            name_education_type="Secondary / secondary special",
            amt_income_total=-100.0,
            amt_credit=406597.5,
            amt_annuity=24700.5,
            days_birth=-9461,
            days_employed=-637,
        )


def test_ext_source_out_of_range():
    with pytest.raises(ValidationError):
        LoanApplicationInput(
            name_contract_type="Cash loans",
            code_gender="F",
            flag_own_car="N",
            name_family_status="Married",
            name_education_type="Secondary / secondary special",
            amt_income_total=202500.0,
            amt_credit=406597.5,
            amt_annuity=24700.5,
            days_birth=-9461,
            days_employed=-637,
            ext_source_2=1.5,
        )


def test_positive_days_birth_rejected():
    with pytest.raises(ValidationError):
        LoanApplicationInput(
            name_contract_type="Cash loans",
            code_gender="F",
            flag_own_car="N",
            name_family_status="Married",
            name_education_type="Secondary / secondary special",
            amt_income_total=202500.0,
            amt_credit=406597.5,
            amt_annuity=24700.5,
            days_birth=100,
            days_employed=-637,
        )
