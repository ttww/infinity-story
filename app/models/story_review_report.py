"""Story review report model (Spec §12.2).

Stores the output of a critic-agent review pass: a numeric score
and a structured list of issues.  Each report is linked to a specific
draft version so the review history is traceable.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.database import Base

if TYPE_CHECKING:
    from app.models.story_draft import StoryDraft


class StoryReviewReport(Base):
    """Critic review report for a draft version."""

    __tablename__ = "story_review_reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    draft_id: Mapped[str] = mapped_column(
        ForeignKey("story_drafts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    version_id: Mapped[str | None] = mapped_column(
        ForeignKey("story_draft_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    issues_json: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    # ── relationship ────────────────────────────────────────────────
    draft: Mapped["StoryDraft"] = relationship(back_populates="review_reports")
