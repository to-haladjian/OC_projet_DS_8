"""ORM model for prediction logs."""

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String

from database.db_session import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    input_features = Column(JSON)
    default_probability = Column(Float)
    credit_approved = Column(Boolean)
    execution_time_ms = Column(Float)
    model_version = Column(String)
