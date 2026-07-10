"""Story draft version model (Spec §12.2).

Each version captures a snapshot of the outline and graph at a particular
point in the authoring cycle.  Versions are append-only — repairs and
reviews create new versions rather than mutating old ones.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.database import Base

if TYPE_CHECKING:
    from app.models.story_draft import StoryDraft


class StoryDraftVersion(Base):
    """An immutable snapshot of a draft's outline + graph."""

    __tablename__ = "story_draft_versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    draft_id: Mapped[str] = mapped_column(
        ForeignKey("story_drafts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    outline_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    graph_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str] = mapped_column(String(128), default="agent", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── relationship ────────────────────────────────────────────────
    draft: Mapped["StoryDraft"] = relationship(back_populates="versions")
