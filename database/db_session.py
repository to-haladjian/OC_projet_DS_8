"""Database session management for Supabase PostgreSQL."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
else:
    engine = None
    SessionLocal = None


def init_db() -> None:
    """Create any missing tables (no-op without a database).

    Idempotent: ``checkfirst`` skips tables that already exist, so the
    pre-existing ``prediction_logs`` is left untouched.
    """
    if engine is None:
        return
    # Import models so they register on Base.metadata before create_all.
    from database.models import app_log, prediction_log  # noqa: F401

    Base.metadata.create_all(engine, checkfirst=True)
