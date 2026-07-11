"""Story draft model (Spec §12.2).

Represents the top-level entity of the authoring pipeline.
Each draft accumulates versions, jobs, review reports, and validation
reports as it progresses through the lifecycle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.enums import DraftStatus
from app.persistence.database import Base

if TYPE_CHECKING:
    from app.models.story_draft_version import StoryDraftVersion
    from app.models.story_generation_job import StoryGenerationJob
    from app.models.story_review_report import StoryReviewReport
    from app.models.story_validation_report import StoryValidationReport


class StoryDraft(Base):
    """A story draft in the authoring pipeline."""

    __tablename__ = "story_drafts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    genre: Mapped[str] = mapped_column(String(64), nullable=False)
    tone: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str] = mapped_column(String(16), default="de", nullable=False)
    target_age: Mapped[str] = mapped_column(String(16), default="16+", nullable=False)
    brief_json: Mapped[str] = mapped_column(Text, nullable=False)
    # ── story config: sentence + connection bounds ─────────────────────
    min_sentences_per_node: Mapped[int] = mapped_column(
        Integer, server_default="3", default=3, nullable=False,
    )
    max_sentences_per_node: Mapped[int] = mapped_column(
        Integer, server_default="8", default=8, nullable=False,
    )
    min_node_connections: Mapped[int] = mapped_column(
        Integer, server_default="2", default=2, nullable=False,
    )
    max_node_connections: Mapped[int] = mapped_column(
        Integer, server_default="5", default=5, nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32), default=DraftStatus.DRAFT.value, nullable=False, index=True,
    )
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # ── relationships ───────────────────────────────────────────────
    versions: Mapped[list["StoryDraftVersion"]] = relationship(
        back_populates="draft",
        cascade="all, delete-orphan",
        order_by="StoryDraftVersion.version_number",
        lazy="selectin",
    )
    jobs: Mapped[list["StoryGenerationJob"]] = relationship(
        back_populates="draft",
        cascade="all, delete-orphan",
        order_by="StoryGenerationJob.created_at",
        lazy="selectin",
    )
    review_reports: Mapped[list["StoryReviewReport"]] = relationship(
        back_populates="draft",
        cascade="all, delete-orphan",
        order_by="StoryReviewReport.created_at",
        lazy="selectin",
    )
    validation_reports: Mapped[list["StoryValidationReport"]] = relationship(
        back_populates="draft",
        cascade="all, delete-orphan",
        order_by="StoryValidationReport.created_at",
        lazy="selectin",
    )

    # ── convenience ─────────────────────────────────────────────────
    @property
    def status_enum(self) -> DraftStatus:
        """Return the status as a typed enum."""
        return DraftStatus(self.status)

    def can_transition_to(self, target: DraftStatus) -> bool:
        """Check whether the draft can move to *target* status."""
        return self.status_enum.can_transition_to(target)

    def latest_version(self) -> "StoryDraftVersion | None":
        """Return the most recent version, or *None* if there are none."""
        return self.version_records[-1] if self.version_records else None

    @property
    def version_records(self) -> list["StoryDraftVersion"]:
        """Return versions sorted by version_number (ascending)."""
        return sorted(self.versions, key=lambda v: v.version_number)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a lightweight summary suitable for list views."""
        return {
            "id": self.id,
            "title": self.title,
            "genre": self.genre,
            "tone": self.tone,
            "language": self.language,
            "target_age": self.target_age,
            "status": self.status,
            "quality_score": self.quality_score,
            "min_sentences_per_node": self.min_sentences_per_node,
            "max_sentences_per_node": self.max_sentences_per_node,
            "min_node_connections": self.min_node_connections,
            "max_node_connections": self.max_node_connections,
            "version_count": len(self.versions),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
        }
