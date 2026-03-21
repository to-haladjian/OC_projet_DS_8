"""Health check endpoint."""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import API_VERSION
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    description="Returns the API status, version, and current timestamp.",
    responses={
        200: {
            "description": "API is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "api_version": "1.0.0",
                        "timestamp": "2026-03-21T12:00:00+00:00",
                    }
                }
            },
        }
    },
)
def health():
    return {
        "status": "healthy",
        "api_version": API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
