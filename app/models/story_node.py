"""Story node model (Spec §12.1, §5.4)."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.database import Base


class StoryNode(Base):
    __tablename__ = "story_nodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("story_sessions.id"), nullable=True)
    scenario_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    node_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
