"""Published scenario model (Spec §12.1, §15).

When a draft is published, its graph is copied into this table so
the runtime can list and load scenarios without touching the
authoring tables.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.database import Base


class PublishedScenario(Base):
    """A published story scenario available at runtime (Spec §15)."""

    __tablename__ = "published_scenarios"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    draft_id: Mapped[str | None] = mapped_column(
        ForeignKey("story_drafts.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    genre: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    graph_json: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
