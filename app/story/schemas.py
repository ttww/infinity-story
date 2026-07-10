"""Pydantic schemas for story-related API and internal data exchange."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class WorldState(BaseModel):
    """Structured world state (Spec §5.5)."""
    genre: str = ""
    tone: str = ""
    current_location: str = ""
    main_character_name: str = ""
    inventory: list[str] = Field(default_factory=list)
    relationships: dict[str, float] = Field(default_factory=dict)
    open_mysteries: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)


class GeneratedSceneResponse(BaseModel):
    """API-facing representation of a generated scene."""
    scene_text: str
    choices: list[dict[str, str]] = Field(default_factory=list)
    state_updates: dict[str, Any] = Field(default_factory=dict)
    suggested_next_node: str | None = None


class StoryBriefSchema(BaseModel):
    """Story brief for authoring (Spec §7.1)."""
    title: str
    genre: str
    tone: str
    language: str = "de"
    target_age: str = "16+"
    node_count: int = 25
    ending_count: int = 3
    branching_level: str = "medium"
    themes: list[str] = Field(default_factory=list)
    forbidden_content: list[str] = Field(default_factory=list)
    notes: str | None = None


class OutlineSchema(BaseModel):
    """High-level story outline (Spec §7.2)."""
    premise: str
    main_conflict: str
    core_mystery: str
    main_characters: list[dict[str, Any]] = Field(default_factory=list)
    endings: list[str] = Field(default_factory=list)


# ── Critic review schemas (Spec §7.4, §14.2) ───────────────────────


class CriticIssue(BaseModel):
    """A single issue identified by the story critic agent."""
    severity: str = Field(
        ...,
        description="One of: high, medium, low, info",
    )
    node_id: str | None = Field(
        default=None,
        description="Node ID where the issue occurs, or null for graph-wide issues.",
    )
    problem: str = Field(
        ...,
        min_length=1,
        description="Concise description of the issue.",
    )
    suggestion: str = Field(
        ...,
        min_length=1,
        description="Actionable fix for the issue.",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        allowed = {"high", "medium", "low", "info"}
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(
                f"severity must be one of {allowed}, got '{v}'"
            )
        return v_lower


class CriticReviewReport(BaseModel):
    """Full review report from the story critic agent (Spec §7.4, §14.2).

    Output shape::

        {
            "score": float,
            "issues": [CriticIssue, ...],
            "repair_suggestions": [str, ...],
            "summary": str
        }
    """
    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Overall quality score (0.0–10.0). 7.0+ is publishable.",
    )
    issues: list[CriticIssue] = Field(default_factory=list)
    repair_suggestions: list[str] = Field(default_factory=list)
    summary: str = ""

    @property
    def high_severity_issues(self) -> list[CriticIssue]:
        """Issues with severity 'high' — must be fixed before publication."""
        return [i for i in self.issues if i.severity == "high"]

    @property
    def is_publishable(self) -> bool:
        """True if score >= 7.0 AND no high-severity issues (Spec §15)."""
        return self.score >= 7.0 and len(self.high_severity_issues) == 0
