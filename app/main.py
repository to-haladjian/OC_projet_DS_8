"""Credit Scoring API - FastAPI application with Gradio UI."""

import logging
import os

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.core.config import API_VERSION
from app.core.logging import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Scoring API",
    description=(
        "API for predicting loan default probability. "
        "Receive applicant data, return credit approval decision."
    ),
    version=API_VERSION,
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


# Register routers
from app.routers import health, prediction  # noqa: E402

app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Credit Scoring"])

# Mount Gradio UI
from app.gradio_ui import demo  # noqa: E402

app = gr.mount_gradio_app(app, demo, path="/gradio")

# Database router (optional, only if DATABASE_URL is set)
if os.getenv("DATABASE_URL"):
    logger.info("Database logging enabled (DATABASE_URL is set)")
else:
    logger.info("Database logging disabled (DATABASE_URL not set)")

logger.info("Credit Scoring API v%s started", API_VERSION)
