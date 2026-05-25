"""Pydantic schemas for credit scoring input/output validation.

The model uses 100 features, 29 from the application table and 71 aggregated
from auxiliary tables (bureau, previous applications, installments, etc.).

The API accepts the application-level features. Aggregated features are
imputed with training median values during preprocessing.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LoanApplicationInput(BaseModel):
    """Input features for credit scoring prediction.

    Based on the Home Credit dataset application table.
    Categorical features are passed as strings and encoded during preprocessing.
    """

    application_id: Optional[int] = Field(
        default=None,
        description="Unique application identifier (optional)",
    )

    # --- Loan details ---
    name_contract_type: str = Field(
        ...,
        description="Type of loan: 'Cash loans' or 'Revolving loans'",
        json_schema_extra={"example": "Cash loans"},
    )
    amt_income_total: float = Field(
        ..., gt=0,
        description="Total annual income of the applicant",
        json_schema_extra={"example": 202500.0},
    )
    amt_credit: float = Field(
        ..., gt=0,
        description="Credit amount of the loan",
        json_schema_extra={"example": 406597.5},
    )
    amt_annuity: float = Field(
        ..., gt=0,
        description="Annuity amount of the loan",
        json_schema_extra={"example": 24700.5},
    )
    amt_goods_price: Optional[float] = Field(
        default=None, ge=0,
        description="Price of the goods for which the loan is given",
        json_schema_extra={"example": 351000.0},
    )

    # --- Personal information ---
    code_gender: str = Field(
        ...,
        description="Gender: 'M' or 'F'",
        json_schema_extra={"example": "F"},
    )
    flag_own_car: str = Field(
        ...,
        description="Owns a car: 'Y' or 'N'",
        json_schema_extra={"example": "N"},
    )
    name_family_status: str = Field(
        ...,
        description="Family status (e.g. 'Married', 'Single / not married')",
        json_schema_extra={"example": "Married"},
    )
    name_education_type: str = Field(
        ...,
        description="Education level (e.g. 'Higher education', 'Secondary / secondary special')",
        json_schema_extra={"example": "Secondary / secondary special"},
    )
    cnt_fam_members: float = Field(
        default=1.0, ge=1,
        description="Number of family members",
        json_schema_extra={"example": 2.0},
    )

    # --- Time-based features (real-world dates; converted to the model's
    # "days relative to application date" convention during preprocessing) ---
    birth_date: date = Field(
        ...,
        description="Applicant's date of birth (must be in the past)",
        json_schema_extra={"example": "1986-04-20"},
    )
    employment_start_date: Optional[date] = Field(
        default=None,
        description="Date current employment started; omit if unemployed/retired",
        json_schema_extra={"example": "2022-08-01"},
    )
    registration_date: Optional[date] = Field(
        default=None,
        description="Date of registration (must be in the past)",
        json_schema_extra={"example": "2014-06-15"},
    )
    id_publish_date: Optional[date] = Field(
        default=None,
        description="Date the ID document was issued (must be in the past)",
        json_schema_extra={"example": "2017-09-30"},
    )
    last_phone_change_date: Optional[date] = Field(
        default=None,
        description="Date of last phone change (must be in the past)",
        json_schema_extra={"example": "2021-11-05"},
    )

    @field_validator(
        "birth_date",
        "employment_start_date",
        "registration_date",
        "id_publish_date",
        "last_phone_change_date",
    )
    @classmethod
    def _reject_future_dates(cls, value: Optional[date]) -> Optional[date]:
        if value is not None and value > date.today():
            raise ValueError("date must not be in the future")
        return value

    # --- External data scores ---
    ext_source_2: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Normalized external data source 2 score",
        json_schema_extra={"example": 0.262949},
    )
    ext_source_3: Optional[float] = Field(
        default=None, ge=0, le=1,
        description="Normalized external data source 3 score",
        json_schema_extra={"example": 0.139376},
    )

    # --- Region and housing ---
    region_population_relative: Optional[float] = Field(
        default=None, ge=0,
        description="Normalized population of region where client lives",
        json_schema_extra={"example": 0.018801},
    )
    region_rating_client_w_city: Optional[int] = Field(
        default=None, ge=1, le=3,
        description="Rating of the region (1=best, 3=worst)",
        json_schema_extra={"example": 2},
    )
    reg_city_not_live_city: Optional[int] = Field(
        default=None, ge=0, le=1,
        description="1 if registration city differs from living city",
        json_schema_extra={"example": 0},
    )
    floorsmax_avg: Optional[float] = Field(
        default=None,
        description="Average max floors of the building",
        json_schema_extra={"example": 0.0833},
    )
    totalarea_mode: Optional[float] = Field(
        default=None,
        description="Total area mode of the building",
        json_schema_extra={"example": 0.0149},
    )
    years_beginexpluatation_medi: Optional[float] = Field(
        default=None,
        description="Years since building exploitation started (median)",
        json_schema_extra={"example": 0.9851},
    )

    # --- Social and document flags ---
    obs_30_cnt_social_circle: Optional[float] = Field(
        default=None, ge=0,
        description="Number of observable 30 DPD defaults in social circle",
        json_schema_extra={"example": 2.0},
    )
    def_30_cnt_social_circle: Optional[float] = Field(
        default=None, ge=0,
        description="Number of 30 DPD defaults in social circle",
        json_schema_extra={"example": 0.0},
    )
    amt_req_credit_bureau_qrt: Optional[float] = Field(
        default=None, ge=0,
        description="Number of credit bureau enquiries in last quarter",
        json_schema_extra={"example": 0.0},
    )
    flag_document_3: Optional[int] = Field(
        default=None, ge=0, le=1,
        description="Whether document 3 was provided",
        json_schema_extra={"example": 1},
    )
    organization_type: Optional[str] = Field(
        default=None,
        description="Type of organization where client works",
        json_schema_extra={"example": "Business Entity Type 3"},
    )


class PredictionResponse(BaseModel):
    """Response schema for credit scoring prediction."""

    api_version: str
    timestamp: str
    application_id: Optional[int] = None
    default_probability: float = Field(
        description="Probability that the applicant will default on the loan",
    )
    credit_approved: bool = Field(
        description="Whether the credit application is approved (probability < threshold)",
    )
