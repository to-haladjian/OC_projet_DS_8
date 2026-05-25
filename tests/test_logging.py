"""Tests for structured logging and the Supabase log handler.

DATABASE_URL is unset during tests (see conftest), so the handler and init_db
run their no-op paths; the key guarantee is that neither ever raises.
All tests follow the Arrange-Act-Assert structure.
"""

import json
import logging

from app.core.logging import JsonFormatter, SupabaseLogHandler


def _make_record(extra=None):
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello %s", args=("world",), exc_info=None,
    )
    if extra is not None:
        record.extra = extra
    return record


def test_json_formatter_emits_valid_json_with_expected_keys():
    # Arrange
    record = _make_record()

    # Act
    parsed = json.loads(JsonFormatter().format(record))

    # Assert
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "hello world"
    assert {"timestamp", "level", "module", "message"} <= parsed.keys()


def test_json_formatter_merges_extra_context():
    # Arrange
    record = _make_record(extra={"event": "prediction", "credit_approved": True})

    # Act
    parsed = json.loads(JsonFormatter().format(record))

    # Assert
    assert parsed["event"] == "prediction"
    assert parsed["credit_approved"] is True


def test_supabase_handler_is_noop_without_database():
    # Arrange: SessionLocal is None (no DATABASE_URL in tests)
    handler = SupabaseLogHandler()
    record = _make_record(extra={"k": "v"})

    # Act / Assert: emitting must not raise
    handler.emit(record)


def test_supabase_handler_reentrancy_guard_returns_early():
    # Arrange: simulate being inside a log-triggered DB write
    handler = SupabaseLogHandler()
    handler._local.writing = True
    record = _make_record()

    # Act
    handler.emit(record)  # must short-circuit without error

    # Assert
    assert handler._local.writing is True  # guard left as-is, no reset
    handler._local.writing = False  # cleanup


def test_init_db_is_noop_without_engine():
    # Arrange
    from database.db_session import init_db

    # Act / Assert: no database configured -> returns without raising
    assert init_db() is None
