"""ORM model for loan application (client) parameters."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from database.db_session import Base


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Optional external application identifier
    application_id = Column(Integer, nullable=True)

    # Loan details
    name_contract_type = Column(String, nullable=False)
    amt_income_total = Column(Float, nullable=False)
    amt_credit = Column(Float, nullable=False)
    amt_annuity = Column(Float, nullable=False)
    amt_goods_price = Column(Float, nullable=True)

    # Personal information
    code_gender = Column(String, nullable=False)
    flag_own_car = Column(String, nullable=False)
    name_family_status = Column(String, nullable=False)
    name_education_type = Column(String, nullable=False)
    cnt_fam_members = Column(Float, nullable=False)

    # Time-based features
    days_birth = Column(Integer, nullable=False)
    days_employed = Column(Integer, nullable=False)
    days_registration = Column(Float, nullable=True)
    days_id_publish = Column(Integer, nullable=True)
    days_last_phone_change = Column(Float, nullable=True)

    # External data scores
    ext_source_2 = Column(Float, nullable=True)
    ext_source_3 = Column(Float, nullable=True)

    # Region and housing
    region_population_relative = Column(Float, nullable=True)
    region_rating_client_w_city = Column(Integer, nullable=True)
    reg_city_not_live_city = Column(Integer, nullable=True)
    floorsmax_avg = Column(Float, nullable=True)
    totalarea_mode = Column(Float, nullable=True)
    years_beginexpluatation_medi = Column(Float, nullable=True)

    # Social and document flags
    obs_30_cnt_social_circle = Column(Float, nullable=True)
    def_30_cnt_social_circle = Column(Float, nullable=True)
    amt_req_credit_bureau_qrt = Column(Float, nullable=True)
    flag_document_3 = Column(Integer, nullable=True)
    organization_type = Column(String, nullable=True)
