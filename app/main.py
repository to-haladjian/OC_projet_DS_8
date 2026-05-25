"""Credit Scoring API - FastAPI application with Gradio UI."""

import logging
import os

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import API_VERSION
from app.core.logging import setup_logging

load_dotenv()
setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Scoring API",
    description=(
        "API for predicting loan default probability. "
        "Receive applicant data, return credit approval decision."
    ),
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a clean 500 instead of leaking a stack trace to the client."""
    logger.exception("Unhandled error processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Register routers
from app.routers import health, prediction  # noqa: E402

app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Credit Scoring"])

# Mount Gradio UI at root so HF Space opens it directly
from app.gradio_ui import demo  # noqa: E402

app = gr.mount_gradio_app(app, demo, path="/")

if os.getenv("DATABASE_URL"):
    from database.db_session import init_db  # noqa: E402

    init_db()  # create app_logs / prediction_logs if missing
    logger.info("Database logging enabled (DATABASE_URL is set)")
else:
    logger.info("Database logging disabled (DATABASE_URL not set)")

logger.info("Credit Scoring API v%s started", API_VERSION)
