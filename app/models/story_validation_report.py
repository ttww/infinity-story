"""Story validation report model (Spec §12.2).

Stores the output of the deterministic validation service:
``is_valid`` flag plus structured ``errors_json`` and ``warnings_json``
arrays.  Like review reports, each validation report is linked to a
specific draft version for traceability.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.database import Base

if TYPE_CHECKING:
    from app.models.story_draft import StoryDraft


class StoryValidationReport(Base):
    """Deterministic validation report for a draft version."""

    __tablename__ = "story_validation_reports"

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
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    errors_json: Mapped[str] = mapped_column(Text, nullable=False)
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    # ── relationship ────────────────────────────────────────────────
    draft: Mapped["StoryDraft"] = relationship(back_populates="validation_reports")
