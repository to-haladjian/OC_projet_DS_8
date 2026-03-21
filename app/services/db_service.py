"""Database service for logging predictions to Supabase PostgreSQL."""

import logging

from app.core.config import API_VERSION
from database.db_session import SessionLocal
from database.models.prediction_log import PredictionLog

logger = logging.getLogger(__name__)


def save_prediction_log(
    input_features: dict,
    default_probability: float,
    credit_approved: bool,
    execution_time_ms: float,
) -> None:
    """Save a prediction result to the database."""
    if SessionLocal is None:
        return

    try:
        with SessionLocal() as session:
            log = PredictionLog(
                input_features=input_features,
                default_probability=default_probability,
                credit_approved=credit_approved,
                execution_time_ms=execution_time_ms,
                model_version=API_VERSION,
            )
            session.add(log)
            session.commit()
    except Exception:
        logger.exception("Failed to save prediction log to database")


def get_recent_predictions(limit: int = 100) -> list[dict]:
    """Retrieve recent prediction logs from the database."""
    if SessionLocal is None:
        return []

    with SessionLocal() as session:
        logs = (
            session.query(PredictionLog)
            .order_by(PredictionLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "input_features": log.input_features,
                "default_probability": log.default_probability,
                "credit_approved": log.credit_approved,
                "execution_time_ms": log.execution_time_ms,
                "model_version": log.model_version,
            }
            for log in logs
        ]
