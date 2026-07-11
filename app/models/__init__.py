"""Pydantic models for the Runtime Story System."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── ORM model imports (registers tables on Base.metadata) ──────────────
# These must be imported so that Base.metadata.create_all() in
# database.init_db() sees all tables.  We import the modules (not the
# names) to avoid name collisions with the Pydantic models below.
from app.models.user import User as _UserORM  # noqa: F401
from app.models.story_session import StorySession as _StorySessionORM  # noqa: F401
from app.models.story_node import StoryNode as _StoryNodeORM  # noqa: F401
from app.models.message import Message as _MessageORM  # noqa: F401
from app.models.story_draft import StoryDraft as _StoryDraftORM  # noqa: F401
from app.models.story_draft_version import StoryDraftVersion as _StoryDraftVersionORM  # noqa: F401
from app.models.story_generation_job import StoryGenerationJob as _StoryGenerationJobORM  # noqa: F401
from app.models.story_review_report import StoryReviewReport as _StoryReviewReportORM  # noqa: F401
from app.models.story_validation_report import StoryValidationReport as _StoryValidationReportORM  # noqa: F401
from app.models.published_scenario import PublishedScenario as _PublishedScenarioORM  # noqa: F401


class SessionStatus(str, Enum):
    new = "new"
    selecting_scenario = "selecting_scenario"
    collecting_parameters = "collecting_parameters"
    running = "running"
    paused = "paused"
    completed = "completed"
    cancelled = "cancelled"


class DraftStatus(str, Enum):
    draft = "draft"
    generating = "generating"
    needs_review = "needs_review"
    needs_repair = "needs_repair"
    validated = "validated"
    approved = "approved"
    published = "published"
    rejected = "rejected"
    failed = "failed"


class JobType(str, Enum):
    generate_outline = "generate_outline"
    generate_graph = "generate_graph"
    critic_review = "critic_review"
    repair_graph = "repair_graph"
    validate_graph = "validate_graph"
    publish_story = "publish_story"


class StoryMode(str, Enum):
    guided = "guided"
    simple_free_input = "simple_free_input"
    free = "free"
    auto = "auto"
    group = "group"


class User(BaseModel):
    id: str
    channel_user_id: str
    plan: str = "free"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class WorldState(BaseModel):
    genre: str = ""
    tone: str = ""
    current_location: str = ""
    main_character_name: str = ""
    inventory: list[str] = Field(default_factory=list)
    relationships: dict[str, float] = Field(default_factory=dict)
    open_mysteries: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)
    custom: dict[str, Any] = Field(default_factory=dict)


class UserSession(BaseModel):
    id: str
    user_id: str
    scenario_id: str | None = None
    current_node_id: str | None = None
    world_state: WorldState = Field(default_factory=WorldState)
    status: SessionStatus = SessionStatus.new
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Message(BaseModel):
    id: str
    session_id: str
    direction: str
    text: str
    created_at: datetime | None = None


class GeneratedScene(BaseModel):
    scene_text: str
    choices: list[dict] = Field(default_factory=list)
    state_updates: dict = Field(default_factory=dict)
    suggested_next_node: str | None = None
    node_id: str | None = None


class Choice(BaseModel):
    id: str
    label: str
    next_node_id: str | None = None
    state_updates: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", description="Why this choice matters / narrative justification")


class StoryNode(BaseModel):
    id: str
    title: str = ""
    type: str = "scene"
    act: int = 1
    scene_goal: str = ""
    scene_text: str = Field(default="", description="Pre-written narrative text for this scene (overlays LLM-generated text at runtime)")
    location: str = ""
    characters: list[str] = Field(default_factory=list)
    mood: str = ""
    known_facts: list[str] = Field(default_factory=list)
    reveals: list[str] = Field(default_factory=list)
    choices: list[Choice] = Field(default_factory=list)
    next_node_id: str | None = Field(default=None, description="Node to auto-advance to when choices is empty (auto_advance mode)")
    auto_advance_delay_ms: int | None = Field(default=None, description="Delay in ms before auto-advancing; absent = use renderer default")
    quality_notes: list[str] = Field(default_factory=list)
    state_updates: dict[str, Any] = Field(default_factory=dict)
    is_start: bool = False
    is_end: bool = False


class StoryGraph(BaseModel):
    nodes: dict[str, StoryNode] = Field(default_factory=dict)
    start_node_id: str | None = None
    title: str = ""
    genre: str = ""
    tone: str = ""


class StoryBrief(BaseModel):
    title: str
    genre: str = "science_fiction"
    tone: str = "dark_mystery"
    language: str = "de"
    target_age: str = "16+"
    node_count: int = 25
    ending_count: int = 3
    branching_level: str = "medium"
    # ── story config: sentence + connection bounds ─────────────────────
    min_sentences_per_node: int = 3
    max_sentences_per_node: int = 8
    min_node_connections: int = 2
    max_node_connections: int = 5
    themes: list[str] = Field(default_factory=list)
    forbidden_content: list[str] = Field(default_factory=list)
    notes: str = ""


class StoryOutline(BaseModel):
    premise: str = ""
    main_conflict: str = ""
    core_mystery: str = ""
    main_characters: list[dict] = Field(default_factory=list)
    endings: list[str] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    severity: str = "medium"
    node_id: str | None = None
    problem: str = ""
    suggestion: str = ""


class ReviewReport(BaseModel):
    score: float = 0.0
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = ""


class ValidationReport(BaseModel):
    is_valid: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MessageRequest(BaseModel):
    channel: str = "whatsapp_mock"
    user_id: str
    message: str


class MessageResponse(BaseModel):
    messages: list[str]
    session_id: str | None = None
    session_status: str | None = None


class ScenarioInfo(BaseModel):
    id: str
    title: str
    genre: str | None = None

