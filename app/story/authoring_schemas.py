"""Pydantic schemas for the authoring API (Spec §12.2, §13.2).

These schemas define the request/response bodies for all
``/api/admin/story-drafts`` endpoints and the internal data
transfer objects used by the authoring services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Persona schema (narrative characters) ─────────────────────────────

class Persona(BaseModel):
    """A named character in the story."""
    name: str = Field(..., min_length=1, max_length=64)
    role: str = Field(default="", max_length=128,
                      description="Rolle im Story-Kontext (z.B. 'Pilot', 'Ärztin')")
    pronouns: str = Field(default="er", max_length=16)
    description: str = Field(default="", max_length=500,
                             description="Hintergrund, Motivation, Besonderheiten")


# ── Request schemas ────────────────────────────────────────────────────

class StoryBriefCreate(BaseModel):
    """Request body for POST /api/admin/story-drafts (Spec §7.1)."""
    title: str = Field(..., min_length=1, max_length=256)
    genre: str = Field(..., min_length=1, max_length=64)
    tone: str = Field(..., min_length=1, max_length=64)
    language: str = Field(default="de", max_length=16)
    target_age: str = Field(default="16+", max_length=16)
    node_count: int = Field(default=25, ge=3, le=200)
    ending_count: int = Field(default=3, ge=1, le=20)
    branching_level: str = Field(default="medium", max_length=32)
    # ── story config: sentence + connection bounds ─────────────────────
    min_sentences_per_node: int = Field(default=3, ge=1, le=50)
    max_sentences_per_node: int = Field(default=8, ge=1, le=100)
    min_node_connections: int = Field(default=2, ge=0, le=20)
    max_node_connections: int = Field(default=5, ge=0, le=50)
    themes: list[str] = Field(default_factory=list)
    forbidden_content: list[str] = Field(default_factory=list)
    notes: str | None = None
    # ── protagonist / narrative voice ───────────────────────────────────
    protagonist_name: str = Field(
        default="", max_length=64,
        description="Name des Hauptcharakters. Leer = automatisch generieren.",
    )
    protagonist_pronouns: str = Field(
        default="er", max_length=16,
        description="Pronomen des Hauptcharakters: 'er', 'sie', 'es'.",
    )
    protagonist_description: str = Field(
        default="", max_length=500,
        description="Kurzbeschreibung des Hauptcharakters.",
    )
    personas: list[Persona] = Field(
        default_factory=list,
        description="Nebencharaktere mit Namen, Rolle, Pronomen, Beschreibung. "
                    "Der Hauptcharakter wird über protagonist_name definiert.",
    )

    def to_storage_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable dict stored in ``brief_json``."""
        return self.model_dump()


# ── Response schemas ───────────────────────────────────────────────────

class DraftSummaryResponse(BaseModel):
    """Lightweight draft summary for list views."""
    id: str
    title: str
    genre: str
    tone: str
    language: str
    target_age: str
    status: str
    quality_score: float | None = None
    min_sentences_per_node: int = 3
    max_sentences_per_node: int = 8
    min_node_connections: int = 2
    max_node_connections: int = 5
    version_count: int = 0
    created_at: datetime | None = None
    approved_at: datetime | None = None
    published_at: datetime | None = None


class DraftCreateResponse(BaseModel):
    """Response for POST /api/admin/story-drafts."""
    draft_id: str
    status: str
    job_id: str | None = None


class VersionResponse(BaseModel):
    """A single version inside a draft detail response."""
    id: str
    version_number: int
    created_by: str
    created_at: datetime
    notes: str | None = None
    has_outline: bool = False
    has_graph: bool = True


class DraftDetailResponse(BaseModel):
    """Full draft detail with versions and latest reports."""
    id: str
    title: str
    genre: str
    tone: str
    language: str
    target_age: str
    status: str
    quality_score: float | None = None
    brief: dict[str, Any] = Field(default_factory=dict)
    min_sentences_per_node: int = 3
    max_sentences_per_node: int = 8
    min_node_connections: int = 2
    max_node_connections: int = 5
    created_at: datetime | None = None
    updated_at: datetime | None = None
    approved_at: datetime | None = None
    published_at: datetime | None = None
    versions: list[VersionResponse] = Field(default_factory=list)


class GraphResponse(BaseModel):
    """Story graph for a draft version."""
    draft_id: str
    version_id: str | None = None
    version_number: int | None = None
    graph: dict[str, Any] = Field(default_factory=dict)


class ReviewReportResponse(BaseModel):
    """Critic review report."""
    id: str
    draft_id: str
    version_id: str | None = None
    score: float
    issues: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None
    created_at: datetime


class ValidationReportResponse(BaseModel):
    """Deterministic validation report."""
    id: str
    draft_id: str
    version_id: str | None = None
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime


class JobResponse(BaseModel):
    """Generation job status."""
    id: str
    draft_id: str
    job_type: str
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    token_usage: dict[str, Any] | None = None


# ── Action response ────────────────────────────────────────────────────

class DraftActionResponse(BaseModel):
    """Generic response for status-changing endpoints."""
    draft_id: str
    status: str
    message: str | None = None
