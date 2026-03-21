"""Health endpoint response schema."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    api_version: str
    timestamp: str
