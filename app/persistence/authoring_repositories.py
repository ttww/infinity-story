"""Authoring-specific repository layer (Spec §12.2, §7).

Encapsulates all DB queries for the authoring pipeline:
drafts, versions, jobs, review reports, and validation reports.
Services call these repositories instead of writing raw queries.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import DraftStatus, JobStatus
from app.models.story_draft import StoryDraft
from app.models.story_draft_version import StoryDraftVersion
from app.models.story_generation_job import StoryGenerationJob
from app.models.story_review_report import StoryReviewReport
from app.models.story_validation_report import StoryValidationReport


# ── helpers ──────────────────────────────────────────────────────────

def _new_id(prefix: str = "") -> str:
    """Return a short unique ID (prefix + hex uuid4)."""
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dump(data: Any) -> str:
    """Serialise *data* to compact JSON (used for *_json columns)."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


# ── DraftRepository ────────────────────────────────────────────────────

class StoryDraftRepository:
    """CRUD + workflow queries for :class:`StoryDraft`."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        title: str,
        genre: str,
        tone: str,
        language: str = "de",
        target_age: str = "16+",
        brief: dict[str, Any] | None = None,
        draft_id: str | None = None,
    ) -> StoryDraft:
        """Create a new draft in ``DRAFT`` status."""
        draft = StoryDraft(
            id=draft_id or _new_id("draft_"),
            title=title,
            genre=genre,
            tone=tone,
            language=language,
            target_age=target_age,
            brief_json=_dump(brief or {}),
            status=DraftStatus.DRAFT.value,
        )
        self.session.add(draft)
        await self.session.commit()
        await self.session.refresh(draft)
        return draft

    async def get_by_id(self, draft_id: str) -> StoryDraft | None:
        """Return a single draft with all eager-loaded relationships."""
        stmt = (
            select(StoryDraft)
            .where(StoryDraft.id == draft_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        *,
        status: DraftStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[StoryDraft]:
        """List drafts, optionally filtered by status."""
        stmt = select(StoryDraft).offset(offset).limit(limit)
        if status is not None:
            stmt = stmt.where(StoryDraft.status == status.value)
        stmt = stmt.order_by(StoryDraft.created_at.desc())
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_status(
        self,
        draft_id: str,
        new_status: DraftStatus,
        *,
        quality_score: float | None = None,
    ) -> StoryDraft | None:
        """Transition a draft to *new_status*.

        Validates the transition using :meth:`DraftStatus.can_transition_to`.
        Sets ``approved_at`` / ``published_at`` timestamps automatically.
        """
        draft = await self.get_by_id(draft_id)
        if draft is None:
            return None

        current = DraftStatus(draft.status)
        if not current.can_transition_to(new_status):
            raise ValueError(
                f"Invalid draft status transition: {current.value!r} → {new_status.value!r}"
            )

        now = _now()
        values: dict[str, Any] = {"status": new_status.value}
        if quality_score is not None:
            values["quality_score"] = quality_score
        if new_status == DraftStatus.APPROVED and draft.approved_at is None:
            values["approved_at"] = now
        if new_status == DraftStatus.PUBLISHED and draft.published_at is None:
            values["published_at"] = now

        stmt = (
            update(StoryDraft)
            .where(StoryDraft.id == draft_id)
            .values(**values)
        )
        await self.session.execute(stmt)
        await self.session.commit()
        await self.session.refresh(draft)
        return draft

    async def delete(self, draft_id: str) -> bool:
        """Delete a draft and all cascade children."""
        draft = await self.get_by_id(draft_id)
        if draft is None:
            return False
        await self.session.delete(draft)
        await self.session.commit()
        return True


# ── VersionRepository ───────────────────────────────────────────────────

class StoryDraftVersionRepository:
    """Append-only version management for drafts."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        draft_id: str,
        graph: dict[str, Any],
        outline: dict[str, Any] | None = None,
        created_by: str = "agent",
        notes: str | None = None,
        version_id: str | None = None,
    ) -> StoryDraftVersion:
        """Append a new version (auto-incrementing version_number)."""
        # Determine next version_number
        existing = await self.list_by_draft(draft_id)
        next_num = max((v.version_number for v in existing), default=0) + 1
        version = StoryDraftVersion(
            id=version_id or _new_id("ver_"),
            draft_id=draft_id,
            version_number=next_num,
            outline_json=_dump(outline) if outline is not None else None,
            graph_json=_dump(graph),
            created_by=created_by,
            notes=notes,
        )
        self.session.add(version)
        await self.session.commit()
        await self.session.refresh(version)
        return version

    async def get_by_id(self, version_id: str) -> StoryDraftVersion | None:
        result = await self.session.execute(
            select(StoryDraftVersion).where(StoryDraftVersion.id == version_id)
        )
        return result.scalar_one_or_none()

    async def list_by_draft(
        self, draft_id: str,
    ) -> Sequence[StoryDraftVersion]:
        """Return all versions for a draft, ordered by version_number."""
        result = await self.session.execute(
            select(StoryDraftVersion)
            .where(StoryDraftVersion.draft_id == draft_id)
            .order_by(StoryDraftVersion.version_number)
        )
        return result.scalars().all()

    async def latest_for_draft(
        self, draft_id: str,
    ) -> StoryDraftVersion | None:
        """Return the latest version for a draft, or *None*."""
        versions = await self.list_by_draft(draft_id)
        return versions[-1] if versions else None

    def parse_graph(self, version: StoryDraftVersion) -> dict[str, Any]:
        """Decode the stored ``graph_json`` into a dict."""
        return json.loads(version.graph_json)

    def parse_outline(self, version: StoryDraftVersion) -> dict[str, Any] | None:
        """Decode ``outline_json`` or return *None* if not present."""
        if version.outline_json is None:
            return None
        return json.loads(version.outline_json)


# ── JobRepository ───────────────────────────────────────────────────────

class StoryGenerationJobRepository:
    """CRUD for generation jobs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        draft_id: str,
        job_type: str,
        llm_provider: str | None = None,
        job_id: str | None = None,
    ) -> StoryGenerationJob:
        """Create a pending job."""
        job = StoryGenerationJob(
            id=job_id or _new_id("job_"),
            draft_id=draft_id,
            job_type=job_type,
            status=JobStatus.PENDING.value,
            llm_provider=llm_provider,
        )
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def get_by_id(self, job_id: str) -> StoryGenerationJob | None:
        result = await self.session.execute(
            select(StoryGenerationJob).where(StoryGenerationJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def list_by_draft(
        self, draft_id: str,
    ) -> Sequence[StoryGenerationJob]:
        result = await self.session.execute(
            select(StoryGenerationJob)
            .where(StoryGenerationJob.draft_id == draft_id)
            .order_by(StoryGenerationJob.created_at)
        )
        return result.scalars().all()

    async def mark_running(self, job_id: str) -> StoryGenerationJob | None:
        job = await self.get_by_id(job_id)
        if job is None:
            return None
        job.status = JobStatus.RUNNING.value
        job.started_at = _now()
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def mark_completed(
        self,
        job_id: str,
        *,
        token_usage: dict[str, Any] | None = None,
    ) -> StoryGenerationJob | None:
        job = await self.get_by_id(job_id)
        if job is None:
            return None
        job.status = JobStatus.COMPLETED.value
        job.finished_at = _now()
        if token_usage is not None:
            job.token_usage_json = _dump(token_usage)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def mark_failed(
        self,
        job_id: str,
        *,
        error_message: str,
    ) -> StoryGenerationJob | None:
        job = await self.get_by_id(job_id)
        if job is None:
            return None
        job.status = JobStatus.FAILED.value
        job.finished_at = _now()
        job.error_message = error_message
        await self.session.commit()
        await self.session.refresh(job)
        return job


# ── ReviewReportRepository ──────────────────────────────────────────────

class StoryReviewReportRepository:
    """Append-only review reports."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        draft_id: str,
        score: float,
        issues: list[dict[str, Any]],
        version_id: str | None = None,
        summary: str | None = None,
        report_id: str | None = None,
    ) -> StoryReviewReport:
        report = StoryReviewReport(
            id=report_id or _new_id("review_"),
            draft_id=draft_id,
            version_id=version_id,
            score=score,
            issues_json=_dump(issues),
            summary=summary,
        )
        self.session.add(report)
        await self.session.commit()
        await self.session.refresh(report)
        return report

    async def list_by_draft(
        self, draft_id: str,
    ) -> Sequence[StoryReviewReport]:
        result = await self.session.execute(
            select(StoryReviewReport)
            .where(StoryReviewReport.draft_id == draft_id)
            .order_by(StoryReviewReport.created_at)
        )
        return result.scalars().all()

    def parse_issues(self, report: StoryReviewReport) -> list[dict[str, Any]]:
        return json.loads(report.issues_json)


# ── ValidationReportRepository ──────────────────────────────────────────

class StoryValidationReportRepository:
    """Append-only validation reports."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        draft_id: str,
        is_valid: bool,
        errors: list[str],
        warnings: list[str],
        version_id: str | None = None,
        report_id: str | None = None,
    ) -> StoryValidationReport:
        report = StoryValidationReport(
            id=report_id or _new_id("val_"),
            draft_id=draft_id,
            version_id=version_id,
            is_valid=is_valid,
            errors_json=_dump(errors),
            warnings_json=_dump(warnings),
        )
        self.session.add(report)
        await self.session.commit()
        await self.session.refresh(report)
        return report

    async def list_by_draft(
        self, draft_id: str,
    ) -> Sequence[StoryValidationReport]:
        result = await self.session.execute(
            select(StoryValidationReport)
            .where(StoryValidationReport.draft_id == draft_id)
            .order_by(StoryValidationReport.created_at)
        )
        return result.scalars().all()

    async def latest_for_draft(
        self, draft_id: str,
    ) -> StoryValidationReport | None:
        reports = await self.list_by_draft(draft_id)
        return reports[-1] if reports else None

    @staticmethod
    def parse_errors(report: StoryValidationReport) -> list[str]:
        return json.loads(report.errors_json)

    @staticmethod
    def parse_warnings(report: StoryValidationReport) -> list[str]:
        return json.loads(report.warnings_json)
