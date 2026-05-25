"""Credit scoring prediction endpoint."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import API_VERSION
from app.schemas.credit import LoanApplicationInput, PredictionResponse
from app.services.db_service import save_prediction_log
from app.services.prediction_service import predict_credit_default

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict credit default",
    description=(
        "Predict whether a loan applicant will default on their loan. "
        "Returns the default probability and approval decision."
    ),
)
def predict(application: LoanApplicationInput):
    start = time.perf_counter()

    # mode="json" serializes date fields to ISO strings so they survive both
    # preprocessing and the JSON prediction-log column.
    features = application.model_dump(mode="json")
    application_id, default_probability, credit_approved = predict_credit_default(features)

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Structured JSON logging
    logger.info(
        "prediction_completed",
        extra={
            "extra": {
                "event": "prediction",
                "application_id": application_id,
                "default_probability": round(default_probability, 6),
                "credit_approved": credit_approved,
                "execution_time_ms": round(elapsed_ms, 2),
            }
        },
    )

    # Save to database (no-op if DATABASE_URL is not set)
    save_prediction_log(
        input_features=features,
        default_probability=default_probability,
        credit_approved=credit_approved,
        execution_time_ms=elapsed_ms,
    )

    return PredictionResponse(
        api_version=API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        application_id=application_id,
        default_probability=round(default_probability, 6),
        credit_approved=credit_approved,
    )
