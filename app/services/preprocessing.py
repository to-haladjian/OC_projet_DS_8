"""Preprocessing service: feature engineering and transformation.

The model expects 100 features. Of these:
- 29 come from the application table (computed from raw API input)
- 71 are aggregations from auxiliary tables (bureau, previous apps, etc.)

For API predictions, aggregated features are set to NaN and the
preprocessing pipeline fills them with training median values.
"""

from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTEFACTS_DIR = Path(__file__).parent.parent / "artefacts"
PREPROCESSOR_PATH = ARTEFACTS_DIR / "preprocessor.joblib"
FEATURE_LIST_PATH = ARTEFACTS_DIR / "feature_list.csv"

# Sentinel DAYS_EMPLOYED value the model was trained on for clients with no
# employment (unemployed / pensioners) — used when no employment date is given.
DAYS_EMPLOYED_UNEMPLOYED = 365243

# Load preprocessor and feature list once at startup
_preprocessor = joblib.load(PREPROCESSOR_PATH)
_feature_list = pd.read_csv(FEATURE_LIST_PATH)["feature"].tolist()


def _coerce_date(value) -> date | None:
    """Normalize a date-like value to a ``date``, or ``None`` if absent.

    Accepts ``date``/``datetime`` objects, ISO date strings ("YYYY-MM-DD"),
    and Unix timestamps (as produced by Gradio's DateTime component).
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):  # must precede the date check (subclass)
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).date()
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"Unsupported date value: {value!r}")


def _days_before(value, reference: date) -> int | None:
    """Days from ``reference`` back to a past ``value`` date (negative).

    Returns ``None`` when no date is supplied. Past dates yield negative
    numbers, matching the model's "days relative to application date"
    convention.
    """
    d = _coerce_date(value)
    if d is None:
        return None
    return (d - reference).days


def build_feature_dataframe(application_data: dict) -> pd.DataFrame:
    """Map raw API input to the 100-feature DataFrame expected by the model.

    Aggregation features absent from the API payload are left as NaN; the
    preprocessing pipeline will impute them at scaling time. Returned frame
    shares its column order with the training reference set.
    """
    row = {feature: np.nan for feature in _feature_list}

    # --- Map raw application input to model features ---

    # Binary-encoded features (factorized in Part 1: first seen value = 0)
    # CODE_GENDER: F=0, M=1 (factorize order from training data)
    gender_map = {"F": 0, "M": 1}
    row["CODE_GENDER"] = gender_map.get(application_data.get("code_gender"), np.nan)

    # FLAG_OWN_CAR: N=0, Y=1
    car_map = {"N": 0, "Y": 1}
    row["FLAG_OWN_CAR"] = car_map.get(application_data.get("flag_own_car"), np.nan)

    # One-hot encoded features
    contract_type = application_data.get("name_contract_type", "")
    row["NAME_CONTRACT_TYPE_Revolving_loans"] = 1 if contract_type == "Revolving loans" else 0

    family_status = application_data.get("name_family_status", "")
    row["NAME_FAMILY_STATUS_Married"] = 1 if family_status == "Married" else 0

    education = application_data.get("name_education_type", "")
    row["NAME_EDUCATION_TYPE_Higher_education"] = 1 if education == "Higher education" else 0

    organization = application_data.get("organization_type", "")
    row["ORGANIZATION_TYPE_Self_employed"] = 1 if organization == "Self-employed" else 0

    # Numeric features (direct mapping)
    direct_mappings = {
        "AMT_GOODS_PRICE": "amt_goods_price",
        "AMT_ANNUITY": "amt_annuity",
        "EXT_SOURCE_2": "ext_source_2",
        "EXT_SOURCE_3": "ext_source_3",
        "REGION_POPULATION_RELATIVE": "region_population_relative",
        "REGION_RATING_CLIENT_W_CITY": "region_rating_client_w_city",
        "OBS_30_CNT_SOCIAL_CIRCLE": "obs_30_cnt_social_circle",
        "DEF_30_CNT_SOCIAL_CIRCLE": "def_30_cnt_social_circle",
        "AMT_REQ_CREDIT_BUREAU_QRT": "amt_req_credit_bureau_qrt",
        "REG_CITY_NOT_LIVE_CITY": "reg_city_not_live_city",
        "FLOORSMAX_AVG": "floorsmax_avg",
        "TOTALAREA_MODE": "totalarea_mode",
        "YEARS_BEGINEXPLUATATION_MEDI": "years_beginexpluatation_medi",
        "FLAG_DOCUMENT_3": "flag_document_3",
    }
    for model_feat, input_feat in direct_mappings.items():
        val = application_data.get(input_feat)
        if val is not None:
            row[model_feat] = float(val)

    # Time-based features: convert real-world dates to the model's
    # "days relative to application date" convention (past dates -> negative).
    today = date.today()
    date_mappings = {
        "DAYS_BIRTH": "birth_date",
        "DAYS_REGISTRATION": "registration_date",
        "DAYS_ID_PUBLISH": "id_publish_date",
        "DAYS_LAST_PHONE_CHANGE": "last_phone_change_date",
    }
    for model_feat, input_feat in date_mappings.items():
        days = _days_before(application_data.get(input_feat), today)
        if days is not None:
            row[model_feat] = days

    # DAYS_EMPLOYED: a missing employment date means unemployed/retired, which
    # the model represents with the 365243 sentinel rather than a negative span.
    employment_days = _days_before(application_data.get("employment_start_date"), today)
    row["DAYS_EMPLOYED"] = (
        employment_days if employment_days is not None else DAYS_EMPLOYED_UNEMPLOYED
    )

    # Engineered features (computed from raw input)
    amt_income = application_data.get("amt_income_total", 0)
    amt_credit = application_data.get("amt_credit", 0)
    amt_annuity = application_data.get("amt_annuity", 0)
    cnt_fam = application_data.get("cnt_fam_members", 1)

    if amt_credit and amt_credit > 0:
        row["INCOME_CREDIT_PERC"] = amt_income / amt_credit
        row["PAYMENT_RATE"] = amt_annuity / amt_credit
    if amt_income and amt_income > 0:
        row["ANNUITY_INCOME_PERC"] = amt_annuity / amt_income
        row["INCOME_PER_PERSON"] = amt_income / max(cnt_fam, 1)

    df = pd.DataFrame([row], columns=_feature_list)
    return df.replace([np.inf, -np.inf], np.nan)


def preprocess_application(application_data: dict) -> tuple[int | None, np.ndarray]:
    """Preprocess raw application data for model inference.

    The preprocessing pipeline (SimpleImputer + StandardScaler) was fitted
    on the 100 selected features from the training data. Missing features
    (aggregations from auxiliary tables) are imputed with training medians.

    Args:
        application_data: Raw features from the API request.

    Returns:
        (application_id, processed_feature_array)
    """
    application_id = application_data.pop("application_id", None)
    df = build_feature_dataframe(application_data)
    processed = _preprocessor.transform(df)
    return application_id, processed.astype(np.float32)
