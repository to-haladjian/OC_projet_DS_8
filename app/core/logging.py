"""Structured JSON logging configuration."""

import json
import logging
import threading
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry, default=str)


class SupabaseLogHandler(logging.Handler):
    """Persist log records to the Supabase ``app_logs`` table.

    No-op when DATABASE_URL is unset. Never raises (a failed log write must not
    break the request) and guards against reentrancy so a log emitted while
    writing a log cannot recurse into another database write.
    """

    _local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._local, "writing", False):
            return
        self._local.writing = True
        try:
            from database.db_session import SessionLocal
            from database.models.app_log import AppLog

            if SessionLocal is None:
                return
            context = getattr(record, "extra", None)
            with SessionLocal() as session:
                session.add(AppLog(
                    timestamp=datetime.fromtimestamp(record.created, timezone.utc),
                    level=record.levelname,
                    module=record.module,
                    message=record.getMessage(),
                    context=context if isinstance(context, dict) else None,
                ))
                session.commit()
        except Exception:
            pass  # logging must never break the application
        finally:
            self._local.writing = False


def setup_logging() -> None:
    """Set up structured JSON logging (stdout + optional Supabase persistence)."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler, SupabaseLogHandler()],
    )

    # Silence verbose libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("gradio").setLevel(logging.WARNING)
    # Keep DB-engine chatter out of the handler that writes to the DB.
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
