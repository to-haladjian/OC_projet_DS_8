"""ORM model for application logs."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text

from database.db_session import Base


class AppLog(Base):
    __tablename__ = "app_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    level = Column(String)
    module = Column(String)
    message = Column(Text)
    context = Column(JSON, nullable=True)
