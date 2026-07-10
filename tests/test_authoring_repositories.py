"""Test authoring repository layer (Spec §12.2).

Exercises StoryDraftRepository, StoryDraftVersionRepository,
StoryGenerationJobRepository, StoryReviewReportRepository, and
StoryValidationReportRepository against an in-memory SQLite DB.
"""

import json

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import DraftStatus, JobStatus, JobType
from app.persistence.authoring_repositories import (
    StoryDraftRepository,
    StoryDraftVersionRepository,
    StoryGenerationJobRepository,
    StoryReviewReportRepository,
    StoryValidationReportRepository,
)


@pytest.fixture
async def db_session():
    """Provide an isolated async session with fresh schema.

    Each test gets its own in-memory database so there is no cross-test
    data leakage.  We create a brand-new engine per test.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        future=True,
        echo=False,
    )

    # Enable SQLite foreign-key enforcement (required for ON DELETE CASCADE)
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fk(dbapi_conn, _conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # Create all tables in this engine
    import app.models  # noqa: F401 — register models on Base.metadata
    from app.persistence.database import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with factory() as session:
        yield session

    await engine.dispose()


# ── StoryDraftRepository ────────────────────────────────────────────────

class TestStoryDraftRepository:
    @pytest.mark.asyncio
    async def test_create_draft(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(
            title="Test Story",
            genre="science_fiction",
            tone="dark_mystery",
        )
        assert draft.id.startswith("draft_")
        assert draft.title == "Test Story"
        assert draft.genre == "science_fiction"
        assert draft.status == DraftStatus.DRAFT.value
        assert draft.quality_score is None
        assert draft.approved_at is None
        assert draft.published_at is None
        # brief_json should be valid JSON
        brief = json.loads(draft.brief_json)
        assert brief == {}

    @pytest.mark.asyncio
    async def test_create_with_brief(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(
            title="Helios",
            genre="science_fiction",
            tone="dark_mystery",
            brief={"node_count": 25, "themes": ["space"]},
        )
        brief = json.loads(draft.brief_json)
        assert brief["node_count"] == 25
        assert "space" in brief["themes"]

    @pytest.mark.asyncio
    async def test_get_by_id(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(title="Find Me", genre="horror", tone="eerie")
        fetched = await repo.get_by_id(draft.id)
        assert fetched is not None
        assert fetched.title == "Find Me"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, db_session):
        repo = StoryDraftRepository(db_session)
        assert await repo.get_by_id("nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_all(self, db_session):
        repo = StoryDraftRepository(db_session)
        await repo.create(title="A", genre="g1", tone="t1")
        await repo.create(title="B", genre="g2", tone="t2")
        drafts = await repo.list_all()
        assert len(drafts) == 2

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, db_session):
        repo = StoryDraftRepository(db_session)
        d1 = await repo.create(title="Draft", genre="g", tone="t")
        await repo.update_status(d1.id, DraftStatus.GENERATING)
        d2 = await repo.create(title="Other", genre="g", tone="t")
        # d1 is generating, d2 is draft
        generating = await repo.list_all(status=DraftStatus.GENERATING)
        assert len(generating) == 1
        assert generating[0].title == "Draft"
        drafts = await repo.list_all(status=DraftStatus.DRAFT)
        assert len(drafts) == 1
        assert drafts[0].title == "Other"

    @pytest.mark.asyncio
    async def test_update_status_valid_transition(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(title="T", genre="g", tone="t")
        updated = await repo.update_status(draft.id, DraftStatus.GENERATING)
        assert updated.status == DraftStatus.GENERATING.value

    @pytest.mark.asyncio
    async def test_update_status_sets_approved_at(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(title="T", genre="g", tone="t")
        await repo.update_status(draft.id, DraftStatus.GENERATING)
        await repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
        await repo.update_status(draft.id, DraftStatus.VALIDATED)
        approved = await repo.update_status(draft.id, DraftStatus.APPROVED)
        assert approved.approved_at is not None

    @pytest.mark.asyncio
    async def test_update_status_sets_published_at(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(title="T", genre="g", tone="t")
        await repo.update_status(draft.id, DraftStatus.GENERATING)
        await repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
        await repo.update_status(draft.id, DraftStatus.VALIDATED)
        await repo.update_status(draft.id, DraftStatus.APPROVED)
        published = await repo.update_status(draft.id, DraftStatus.PUBLISHED)
        assert published.published_at is not None

    @pytest.mark.asyncio
    async def test_update_status_invalid_transition(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(title="T", genre="g", tone="t")
        with pytest.raises(ValueError, match="Invalid draft status transition"):
            await repo.update_status(draft.id, DraftStatus.PUBLISHED)

    @pytest.mark.asyncio
    async def test_update_status_with_quality_score(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(title="T", genre="g", tone="t")
        updated = await repo.update_status(
            draft.id, DraftStatus.GENERATING, quality_score=8.5,
        )
        assert updated.quality_score == 8.5

    @pytest.mark.asyncio
    async def test_update_status_not_found(self, db_session):
        repo = StoryDraftRepository(db_session)
        assert await repo.update_status("nonexistent", DraftStatus.GENERATING) is None

    @pytest.mark.asyncio
    async def test_delete(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(title="Del", genre="g", tone="t")
        assert await repo.delete(draft.id) is True
        assert await repo.get_by_id(draft.id) is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, db_session):
        repo = StoryDraftRepository(db_session)
        assert await repo.delete("nonexistent") is False

    @pytest.mark.asyncio
    async def test_to_summary_dict(self, db_session):
        repo = StoryDraftRepository(db_session)
        draft = await repo.create(title="Summary", genre="g", tone="t")
        summary = draft.to_summary_dict()
        assert summary["id"] == draft.id
        assert summary["title"] == "Summary"
        assert summary["status"] == "draft"
        assert summary["version_count"] == 0


# ── StoryDraftVersionRepository ─────────────────────────────────────────

class TestStoryDraftVersionRepository:
    @pytest.mark.asyncio
    async def test_create_version(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        graph = {"nodes": {"n1": {"id": "n1", "type": "start"}}}
        version = await ver_repo.create(draft_id=draft.id, graph=graph)
        assert version.id.startswith("ver_")
        assert version.version_number == 1
        assert version.draft_id == draft.id

    @pytest.mark.asyncio
    async def test_version_number_auto_increment(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        v1 = await ver_repo.create(draft_id=draft.id, graph={"nodes": {}})
        v2 = await ver_repo.create(draft_id=draft.id, graph={"nodes": {}})
        v3 = await ver_repo.create(draft_id=draft.id, graph={"nodes": {}})
        assert v1.version_number == 1
        assert v2.version_number == 2
        assert v3.version_number == 3

    @pytest.mark.asyncio
    async def test_create_with_outline(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        version = await ver_repo.create(
            draft_id=draft.id,
            graph={"nodes": {}},
            outline={"premise": "Test"},
        )
        outline = ver_repo.parse_outline(version)
        assert outline is not None
        assert outline["premise"] == "Test"

    @pytest.mark.asyncio
    async def test_parse_outline_none(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        version = await ver_repo.create(draft_id=draft.id, graph={"nodes": {}})
        assert ver_repo.parse_outline(version) is None

    @pytest.mark.asyncio
    async def test_list_by_draft(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        await ver_repo.create(draft_id=draft.id, graph={})
        await ver_repo.create(draft_id=draft.id, graph={})

        versions = await ver_repo.list_by_draft(draft.id)
        assert len(versions) == 2
        assert versions[0].version_number == 1
        assert versions[1].version_number == 2

    @pytest.mark.asyncio
    async def test_latest_for_draft(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        await ver_repo.create(draft_id=draft.id, graph={})
        await ver_repo.create(draft_id=draft.id, graph={})

        latest = await ver_repo.latest_for_draft(draft.id)
        assert latest is not None
        assert latest.version_number == 2

    @pytest.mark.asyncio
    async def test_latest_for_draft_none(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")
        ver_repo = StoryDraftVersionRepository(db_session)
        assert await ver_repo.latest_for_draft(draft.id) is None

    @pytest.mark.asyncio
    async def test_parse_graph(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        graph = {"nodes": {"n1": {"type": "start"}}, "start_node_id": "n1"}
        version = await ver_repo.create(draft_id=draft.id, graph=graph)
        parsed = ver_repo.parse_graph(version)
        assert parsed["start_node_id"] == "n1"
        assert "n1" in parsed["nodes"]


# ── StoryGenerationJobRepository ────────────────────────────────────────

class TestStoryGenerationJobRepository:
    @pytest.mark.asyncio
    async def test_create_job(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="J", genre="g", tone="t")

        job_repo = StoryGenerationJobRepository(db_session)
        job = await job_repo.create(
            draft_id=draft.id,
            job_type=JobType.OUTLINE.value,
        )
        assert job.id.startswith("job_")
        assert job.status == JobStatus.PENDING.value
        assert job.job_type == "outline"

    @pytest.mark.asyncio
    async def test_mark_running(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="J", genre="g", tone="t")

        job_repo = StoryGenerationJobRepository(db_session)
        job = await job_repo.create(draft_id=draft.id, job_type=JobType.GRAPH.value)
        running = await job_repo.mark_running(job.id)
        assert running.status == JobStatus.RUNNING.value
        assert running.started_at is not None

    @pytest.mark.asyncio
    async def test_mark_completed(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="J", genre="g", tone="t")

        job_repo = StoryGenerationJobRepository(db_session)
        job = await job_repo.create(draft_id=draft.id, job_type=JobType.REVIEW.value)
        await job_repo.mark_running(job.id)
        completed = await job_repo.mark_completed(
            job.id, token_usage={"prompt_tokens": 100, "completion_tokens": 50}
        )
        assert completed.status == JobStatus.COMPLETED.value
        assert completed.finished_at is not None
        usage = json.loads(completed.token_usage_json)
        assert usage["prompt_tokens"] == 100

    @pytest.mark.asyncio
    async def test_mark_failed(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="J", genre="g", tone="t")

        job_repo = StoryGenerationJobRepository(db_session)
        job = await job_repo.create(draft_id=draft.id, job_type=JobType.GRAPH.value)
        await job_repo.mark_running(job.id)
        failed = await job_repo.mark_failed(job.id, error_message="OOM")
        assert failed.status == JobStatus.FAILED.value
        assert failed.error_message == "OOM"
        assert failed.finished_at is not None

    @pytest.mark.asyncio
    async def test_list_by_draft(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="J", genre="g", tone="t")

        job_repo = StoryGenerationJobRepository(db_session)
        await job_repo.create(draft_id=draft.id, job_type=JobType.OUTLINE.value)
        await job_repo.create(draft_id=draft.id, job_type=JobType.GRAPH.value)

        jobs = await job_repo.list_by_draft(draft.id)
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_is_terminal(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="J", genre="g", tone="t")

        job_repo = StoryGenerationJobRepository(db_session)
        job = await job_repo.create(draft_id=draft.id, job_type=JobType.OUTLINE.value)
        assert job.is_terminal() is False
        await job_repo.mark_completed(job.id)
        refreshed = await job_repo.get_by_id(job.id)
        assert refreshed.is_terminal() is True


# ── StoryReviewReportRepository ──────────────────────────────────────────

class TestStoryReviewReportRepository:
    @pytest.mark.asyncio
    async def test_create_review_report(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="R", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        version = await ver_repo.create(draft_id=draft.id, graph={"nodes": {}})

        review_repo = StoryReviewReportRepository(db_session)
        report = await review_repo.create(
            draft_id=draft.id,
            version_id=version.id,
            score=7.5,
            issues=[
                {"severity": "high", "node_id": "n3", "problem": "Weak climax"},
            ],
            summary="Good but needs stronger act 3",
        )
        assert report.id.startswith("review_")
        assert report.score == 7.5
        issues = review_repo.parse_issues(report)
        assert len(issues) == 1
        assert issues[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_list_by_draft(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="R", genre="g", tone="t")

        review_repo = StoryReviewReportRepository(db_session)
        await review_repo.create(draft_id=draft.id, score=6.0, issues=[])
        await review_repo.create(draft_id=draft.id, score=7.0, issues=[])

        reports = await review_repo.list_by_draft(draft.id)
        assert len(reports) == 2


# ── StoryValidationReportRepository ──────────────────────────────────────

class TestStoryValidationReportRepository:
    @pytest.mark.asyncio
    async def test_create_validation_report(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        version = await ver_repo.create(draft_id=draft.id, graph={"nodes": {}})

        val_repo = StoryValidationReportRepository(db_session)
        report = await val_repo.create(
            draft_id=draft.id,
            version_id=version.id,
            is_valid=False,
            errors=["Broken reference in node_002"],
            warnings=["Unreachable node: node_005"],
        )
        assert report.id.startswith("val_")
        assert report.is_valid is False
        errors = val_repo.parse_errors(report)
        warnings = val_repo.parse_warnings(report)
        assert len(errors) == 1
        assert len(warnings) == 1
        assert "Broken reference" in errors[0]

    @pytest.mark.asyncio
    async def test_list_by_draft(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        val_repo = StoryValidationReportRepository(db_session)
        await val_repo.create(draft_id=draft.id, is_valid=True, errors=[], warnings=[])
        await val_repo.create(draft_id=draft.id, is_valid=False, errors=["e"], warnings=[])

        reports = await val_repo.list_by_draft(draft.id)
        assert len(reports) == 2

    @pytest.mark.asyncio
    async def test_latest_for_draft(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")

        val_repo = StoryValidationReportRepository(db_session)
        await val_repo.create(draft_id=draft.id, is_valid=True, errors=[], warnings=[])
        await val_repo.create(draft_id=draft.id, is_valid=False, errors=["e"], warnings=[])

        latest = await val_repo.latest_for_draft(draft.id)
        assert latest is not None
        assert latest.is_valid is False

    @pytest.mark.asyncio
    async def test_latest_for_draft_none(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="V", genre="g", tone="t")
        val_repo = StoryValidationReportRepository(db_session)
        assert await val_repo.latest_for_draft(draft.id) is None


# ── Cascade delete ─────────────────────────────────────────────────────

class TestCascadeDelete:
    @pytest.mark.asyncio
    async def test_delete_draft_cascades_versions(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="C", genre="g", tone="t")

        ver_repo = StoryDraftVersionRepository(db_session)
        await ver_repo.create(draft_id=draft.id, graph={})
        await draft_repo.delete(draft.id)

        # Use a fresh query (expiring the session) to verify cascade
        db_session.expire_all()
        versions = await ver_repo.list_by_draft(draft.id)
        assert len(versions) == 0

    @pytest.mark.asyncio
    async def test_delete_draft_cascades_jobs(self, db_session):
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(title="C", genre="g", tone="t")

        job_repo = StoryGenerationJobRepository(db_session)
        await job_repo.create(draft_id=draft.id, job_type=JobType.OUTLINE.value)
        await draft_repo.delete(draft.id)

        db_session.expire_all()
        jobs = await job_repo.list_by_draft(draft.id)
        assert len(jobs) == 0


# ── Full workflow ───────────────────────────────────────────────────────

class TestFullAuthoringWorkflow:
    @pytest.mark.asyncio
    async def test_complete_lifecycle(self, db_session):
        """Exercise the full draft lifecycle: draft → generating →
        needs_review → validated → approved → published."""
        draft_repo = StoryDraftRepository(db_session)
        ver_repo = StoryDraftVersionRepository(db_session)
        job_repo = StoryGenerationJobRepository(db_session)
        review_repo = StoryReviewReportRepository(db_session)
        val_repo = StoryValidationReportRepository(db_session)

        # 1. Create draft
        draft = await draft_repo.create(
            title="Helios", genre="science_fiction", tone="dark_mystery",
            brief={"node_count": 25},
        )
        assert draft.status == "draft"

        # 2. Start generation
        job = await job_repo.create(
            draft_id=draft.id, job_type=JobType.OUTLINE.value,
        )
        await job_repo.mark_running(job.id)
        await draft_repo.update_status(draft.id, DraftStatus.GENERATING)
        await job_repo.mark_completed(job.id, token_usage={"prompt_tokens": 500})

        # 3. Create first version
        v1 = await ver_repo.create(
            draft_id=draft.id,
            graph={"nodes": {"n1": {"type": "start"}}},
            outline={"premise": "Test"},
        )

        # 4. Review
        await draft_repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
        review = await review_repo.create(
            draft_id=draft.id, version_id=v1.id, score=6.5,
            issues=[{"severity": "medium", "problem": "Weak act 2"}],
        )
        assert review.score == 6.5

        # 5. Repair (creates new version)
        await draft_repo.update_status(draft.id, DraftStatus.NEEDS_REPAIR)
        v2 = await ver_repo.create(
            draft_id=draft.id, graph={"nodes": {"n1": {"type": "start"}, "n2": {"type": "end"}}},
            notes="Added ending node",
        )
        assert v2.version_number == 2

        # 6. Re-review → validated
        await draft_repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
        await draft_repo.update_status(draft.id, DraftStatus.VALIDATED, quality_score=8.0)
        val_report = await val_repo.create(
            draft_id=draft.id, version_id=v2.id,
            is_valid=True, errors=[], warnings=[],
        )
        assert val_report.is_valid is True

        # 7. Approve
        approved = await draft_repo.update_status(draft.id, DraftStatus.APPROVED)
        assert approved.approved_at is not None
        assert approved.quality_score == 8.0

        # 8. Publish
        published = await draft_repo.update_status(draft.id, DraftStatus.PUBLISHED)
        assert published.published_at is not None
        assert published.status == "published"

        # Verify all data is retrievable
        versions = await ver_repo.list_by_draft(draft.id)
        assert len(versions) == 2
        jobs = await job_repo.list_by_draft(draft.id)
        assert len(jobs) == 1
        reviews = await review_repo.list_by_draft(draft.id)
        assert len(reviews) == 1
        validations = await val_repo.list_by_draft(draft.id)
        assert len(validations) == 1
