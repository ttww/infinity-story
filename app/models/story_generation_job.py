"""Story generation job model (Spec §12.2).

Tracks each asynchronous operation in the authoring pipeline
(outline generation, graph generation, review, repair, validation,
publishing).  ``token_usage_json`` stores provider-specific usage
metadata for cost analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.enums import JobStatus, JobType
from app.persistence.database import Base

if TYPE_CHECKING:
    from app.models.story_draft import StoryDraft


class StoryGenerationJob(Base):
    """A single generation job within the authoring pipeline."""

    __tablename__ = "story_generation_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    draft_id: Mapped[str] = mapped_column(
        ForeignKey("story_drafts.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=JobStatus.PENDING.value, nullable=False, index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_usage_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    # ── relationship ────────────────────────────────────────────────
    draft: Mapped["StoryDraft"] = relationship(back_populates="jobs")

    # ── convenience ─────────────────────────────────────────────────
    @property
    def job_type_enum(self) -> JobType:
        return JobType(self.job_type)

    @property
    def status_enum(self) -> JobStatus:
        return JobStatus(self.status)

    def is_terminal(self) -> bool:
        """Return *True* if the job has reached a final state."""
        return self.status in (
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        )
