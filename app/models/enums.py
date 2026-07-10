"""Enums for the authoring workflow (Spec §12.2).

Centralises all status strings so the rest of the codebase references
typed constants instead of raw strings.
"""

from __future__ import annotations

from enum import Enum


class DraftStatus(str, Enum):
    """Lifecycle of a story draft (Spec §12.2).

    Valid transitions (enforced by :class:`StoryDraftRepository`):

        draft → generating → needs_review → needs_repair → needs_review ...
        needs_review → validated → approved → published
        any → rejected | failed
    """

    DRAFT = "draft"
    GENERATING = "generating"
    NEEDS_REVIEW = "needs_review"
    NEEDS_REPAIR = "needs_repair"
    VALIDATED = "validated"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    FAILED = "failed"

    # ── transition helpers ──────────────────────────────────────────
    @classmethod
    def valid_transitions(cls) -> dict["DraftStatus", frozenset["DraftStatus"]]:
        """Return the allowed forward-transition map."""
        return {
            cls.DRAFT: frozenset({cls.GENERATING, cls.FAILED, cls.REJECTED}),
            cls.GENERATING: frozenset({
                cls.NEEDS_REVIEW,
                cls.FAILED,
                cls.DRAFT,
            }),
            cls.NEEDS_REVIEW: frozenset({
                cls.NEEDS_REPAIR,
                cls.VALIDATED,
                cls.REJECTED,
                cls.FAILED,
            }),
            cls.NEEDS_REPAIR: frozenset({
                cls.NEEDS_REVIEW,
                cls.FAILED,
            }),
            cls.VALIDATED: frozenset({
                cls.APPROVED,
                cls.NEEDS_REPAIR,
                cls.REJECTED,
            }),
            cls.APPROVED: frozenset({cls.PUBLISHED, cls.NEEDS_REPAIR, cls.REJECTED, cls.FAILED}),
            cls.PUBLISHED: frozenset(),   # terminal
            cls.REJECTED: frozenset(),    # terminal
            cls.FAILED: frozenset(),      # terminal
        }

    def can_transition_to(self, target: "DraftStatus") -> bool:
        """Return *True* if *self* → *target* is an allowed transition."""
        return target in self.valid_transitions().get(self, frozenset())


class JobStatus(str, Enum):
    """Lifecycle of a story generation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Kinds of generation jobs in the authoring pipeline."""

    OUTLINE = "outline"
    GRAPH = "graph"
    REVIEW = "review"
    REPAIR = "repair"
    VALIDATE = "validate"
    PUBLISH = "publish"


class ReviewSeverity(str, Enum):
    """Severity levels used inside ``review_report.issues_json``."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
