"""Test Pydantic schemas for the authoring API (Spec §12.2, §13.2)."""

import pytest
from pydantic import ValidationError

from app.story.authoring_schemas import (
    DraftActionResponse,
    DraftCreateResponse,
    DraftDetailResponse,
    DraftSummaryResponse,
    GraphResponse,
    JobResponse,
    ReviewReportResponse,
    StoryBriefCreate,
    ValidationReportResponse,
    VersionResponse,
)


class TestStoryBriefCreate:
    def test_valid_brief(self):
        brief = StoryBriefCreate(
            title="Test", genre="scifi", tone="dark",
        )
        assert brief.title == "Test"
        assert brief.language == "de"
        assert brief.target_age == "16+"
        assert brief.node_count == 25
        assert brief.ending_count == 3
        assert brief.branching_level == "medium"

    # ── story config fields ────────────────────────────────────────────

    def test_default_sentence_and_connection_fields(self):
        """New config fields have sensible defaults when omitted."""
        brief = StoryBriefCreate(title="T", genre="g", tone="t")
        assert brief.min_sentences_per_node == 3
        assert brief.max_sentences_per_node == 8
        assert brief.min_node_connections == 2
        assert brief.max_node_connections == 5

    def test_custom_sentence_and_connection_fields(self):
        """New config fields accept custom values."""
        brief = StoryBriefCreate(
            title="T", genre="g", tone="t",
            min_sentences_per_node=5,
            max_sentences_per_node=12,
            min_node_connections=1,
            max_node_connections=8,
        )
        assert brief.min_sentences_per_node == 5
        assert brief.max_sentences_per_node == 12
        assert brief.min_node_connections == 1
        assert brief.max_node_connections == 8

    def test_sentence_bounds_validation(self):
        """min/max sentence fields enforce bounds."""
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", min_sentences_per_node=0)
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", max_sentences_per_node=0)
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", min_sentences_per_node=100)

    def test_connection_bounds_validation(self):
        """min/max connection fields enforce bounds."""
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", min_node_connections=-1)
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", max_node_connections=100)

    def test_config_fields_in_storage_dict(self):
        """to_storage_dict includes the new config fields."""
        brief = StoryBriefCreate(
            title="T", genre="g", tone="t",
            min_sentences_per_node=4, max_sentences_per_node=10,
        )
        d = brief.to_storage_dict()
        assert d["min_sentences_per_node"] == 4
        assert d["max_sentences_per_node"] == 10
        assert d["min_node_connections"] == 2
        assert d["max_node_connections"] == 5

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="", genre="g", tone="t")

    def test_node_count_bounds(self):
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", node_count=1)
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", node_count=500)

    def test_ending_count_bounds(self):
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", ending_count=0)
        with pytest.raises(ValidationError):
            StoryBriefCreate(title="T", genre="g", tone="t", ending_count=50)

    def test_to_storage_dict(self):
        brief = StoryBriefCreate(title="T", genre="g", tone="t", themes=["x"])
        d = brief.to_storage_dict()
        assert d["title"] == "T"
        assert d["themes"] == ["x"]
        assert "notes" in d


class TestDraftSummaryResponse:
    def test_minimal(self):
        r = DraftSummaryResponse(id="d1", title="T", genre="g", tone="t", language="de", target_age="16+", status="draft")
        assert r.quality_score is None
        assert r.version_count == 0


class TestDraftCreateResponse:
    def test_with_job(self):
        r = DraftCreateResponse(draft_id="d1", status="generating", job_id="j1")
        assert r.job_id == "j1"

    def test_without_job(self):
        r = DraftCreateResponse(draft_id="d1", status="draft")
        assert r.job_id is None


class TestVersionResponse:
    def test_basic(self):
        v = VersionResponse(
            id="v1", version_number=1, created_by="agent",
            created_at="2026-01-01T00:00:00",
        )
        assert v.has_outline is False
        assert v.has_graph is True


class TestDraftDetailResponse:
    def test_with_versions(self):
        r = DraftDetailResponse(
            id="d1", title="T", genre="g", tone="t",
            language="de", target_age="16+", status="draft",
            versions=[
                VersionResponse(
                    id="v1", version_number=1, created_by="agent",
                    created_at="2026-01-01T00:00:00",
                ),
            ],
        )
        assert len(r.versions) == 1
        assert r.versions[0].version_number == 1


class TestGraphResponse:
    def test_basic(self):
        r = GraphResponse(draft_id="d1", graph={"nodes": {}})
        assert r.version_id is None
        assert r.graph == {"nodes": {}}


class TestReviewReportResponse:
    def test_basic(self):
        r = ReviewReportResponse(
            id="r1", draft_id="d1", score=7.5,
            issues=[{"severity": "high"}],
            created_at="2026-01-01T00:00:00",
        )
        assert r.score == 7.5
        assert len(r.issues) == 1


class TestValidationReportResponse:
    def test_valid(self):
        r = ValidationReportResponse(
            id="v1", draft_id="d1", is_valid=True,
            errors=[], warnings=[],
            created_at="2026-01-01T00:00:00",
        )
        assert r.is_valid is True

    def test_invalid(self):
        r = ValidationReportResponse(
            id="v2", draft_id="d2", is_valid=False,
            errors=["broken ref"], warnings=["unreachable"],
            created_at="2026-01-01T00:00:00",
        )
        assert r.is_valid is False
        assert len(r.errors) == 1


class TestJobResponse:
    def test_pending(self):
        j = JobResponse(id="j1", draft_id="d1", job_type="outline", status="pending")
        assert j.started_at is None
        assert j.token_usage is None

    def test_completed_with_usage(self):
        j = JobResponse(
            id="j1", draft_id="d1", job_type="outline", status="completed",
            token_usage={"prompt_tokens": 100},
        )
        assert j.token_usage["prompt_tokens"] == 100


class TestDraftActionResponse:
    def test_basic(self):
        r = DraftActionResponse(draft_id="d1", status="approved", message="OK")
        assert r.message == "OK"
