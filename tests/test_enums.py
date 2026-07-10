"""Test draft status enums and transitions (Spec §12.2)."""

import pytest

from app.models.enums import DraftStatus, JobStatus, JobType, ReviewSeverity


class TestDraftStatus:
    """DraftStatus enum and transition logic."""

    def test_all_values_present(self):
        """All nine statuses from the spec must exist."""
        values = {s.value for s in DraftStatus}
        assert values == {
            "draft", "generating", "needs_review", "needs_repair",
            "validated", "approved", "published", "rejected", "failed",
        }

    def test_is_str_enum(self):
        """DraftStatus must be a str subclass for SQLAlchemy compatibility."""
        assert DraftStatus.DRAFT == "draft"
        assert isinstance(DraftStatus.DRAFT, str)

    def test_valid_transition_draft_to_generating(self):
        assert DraftStatus.DRAFT.can_transition_to(DraftStatus.GENERATING) is True

    def test_valid_transition_generating_to_needs_review(self):
        assert DraftStatus.GENERATING.can_transition_to(DraftStatus.NEEDS_REVIEW) is True

    def test_valid_transition_needs_review_to_needs_repair(self):
        assert DraftStatus.NEEDS_REVIEW.can_transition_to(DraftStatus.NEEDS_REPAIR) is True

    def test_valid_transition_needs_review_to_validated(self):
        assert DraftStatus.NEEDS_REVIEW.can_transition_to(DraftStatus.VALIDATED) is True

    def test_valid_transition_validated_to_approved(self):
        assert DraftStatus.VALIDATED.can_transition_to(DraftStatus.APPROVED) is True

    def test_valid_transition_approved_to_published(self):
        assert DraftStatus.APPROVED.can_transition_to(DraftStatus.PUBLISHED) is True

    def test_valid_transition_needs_repair_to_needs_review(self):
        assert DraftStatus.NEEDS_REPAIR.can_transition_to(DraftStatus.NEEDS_REVIEW) is True

    def test_valid_transition_validated_to_needs_repair(self):
        assert DraftStatus.VALIDATED.can_transition_to(DraftStatus.NEEDS_REPAIR) is True

    def test_invalid_transition_draft_to_published(self):
        """Cannot skip from draft directly to published."""
        assert DraftStatus.DRAFT.can_transition_to(DraftStatus.PUBLISHED) is False

    def test_invalid_transition_published_to_anything(self):
        """Published is terminal."""
        for target in DraftStatus:
            if target == DraftStatus.PUBLISHED:
                continue
            assert DraftStatus.PUBLISHED.can_transition_to(target) is False

    def test_invalid_transition_rejected_to_anything(self):
        """Rejected is terminal."""
        for target in DraftStatus:
            if target == DraftStatus.REJECTED:
                continue
            assert DraftStatus.REJECTED.can_transition_to(target) is False

    def test_invalid_transition_failed_to_anything(self):
        """Failed is terminal."""
        for target in DraftStatus:
            if target == DraftStatus.FAILED:
                continue
            assert DraftStatus.FAILED.can_transition_to(target) is False

    def test_any_status_can_fail(self):
        """Every non-terminal status should allow transition to FAILED or REJECTED."""
        terminal = {DraftStatus.PUBLISHED, DraftStatus.REJECTED, DraftStatus.FAILED}
        for s in DraftStatus:
            if s in terminal:
                continue
            allowed = s.valid_transitions().get(s, frozenset())
            assert DraftStatus.FAILED in allowed or DraftStatus.REJECTED in allowed, (
                f"{s.value} has no path to failure"
            )


class TestJobStatus:
    def test_values(self):
        assert {s.value for s in JobStatus} == {
            "pending", "running", "completed", "failed", "cancelled",
        }

    def test_is_str_enum(self):
        assert JobStatus.PENDING == "pending"


class TestJobType:
    def test_values(self):
        assert {s.value for s in JobType} == {
            "outline", "graph", "review", "repair", "validate", "publish",
        }


class TestReviewSeverity:
    def test_values(self):
        assert {s.value for s in ReviewSeverity} == {
            "info", "low", "medium", "high",
        }
