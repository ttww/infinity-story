"""Tests for the Story Critic Agent (Spec §7.4, §14.2).

Verifies:
  - review() returns the correct schema (score, issues, repair_suggestions, summary)
  - Schema enforcement via Pydantic (invalid JSON, schema mismatch)
  - Retry on invalid JSON or schema mismatch
  - StoryCriticError after exhausting retries
  - is_publishable / has_high_severity_issues helpers
  - CriticReviewReport and CriticIssue Pydantic schemas
  - build_critic_user_prompt produces the right structure
  - CRITIC_SYSTEM_PROMPT contains all 13 criteria
  - Mock LLM integration
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from app.services.llm_service import LLMResponse, LLMService, LLMResponseError
from app.services.story_critic_agent import StoryCriticAgent, StoryCriticError
from app.story.prompts import CRITIC_SYSTEM_PROMPT, build_critic_user_prompt
from app.story.schemas import CriticIssue, CriticReviewReport


# ── Test fixtures ───────────────────────────────────────────────────

SAMPLE_OUTLINE: dict[str, Any] = {
    "premise": "A crew on Mars receives a signal from a dead colleague.",
    "main_conflict": "Uncover the truth vs. the AI's manipulation.",
    "core_mystery": "Why is a dead crew member's voice coming from the station?",
    "main_characters": [
        {"name": "Mira", "role": "Comms officer", "secret": "Knows the signal source."},
        {"name": "Captain Rao", "role": "Commander", "secret": "Ordered the evacuation."},
    ],
    "endings": ["The AI reveals it trapped the crew."],
}

SAMPLE_GRAPH: dict[str, Any] = {
    "start_node_id": "node_001",
    "nodes": {
        "node_001": {
            "id": "node_001", "title": "Start", "type": "start",
            "scene_goal": "Discover a signal.",
            "choices": [{"id": "c1", "label": "Go", "next_node_id": "node_002"}],
            "quality_notes": ["start"], "is_start": True, "is_end": False,
        },
        "node_002": {
            "id": "node_002", "title": "End", "type": "end",
            "scene_goal": "Resolve the mystery.",
            "choices": [], "quality_notes": ["end"],
            "is_start": False, "is_end": True,
        },
    },
}

GOOD_REVIEW: dict[str, Any] = {
    "score": 8.0,
    "issues": [
        {"severity": "low", "node_id": "node_001",
         "problem": "Pacing could be tighter.",
         "suggestion": "Add a time-pressure element."},
    ],
    "repair_suggestions": ["Tighten node_001 pacing."],
    "summary": "Solid graph with good pacing.",
}

HIGH_SEVERITY_REVIEW: dict[str, Any] = {
    "score": 5.5,
    "issues": [
        {"severity": "high", "node_id": "node_001",
         "problem": "Core mystery revealed too early.",
         "suggestion": "Move reveal to act 3."},
        {"severity": "medium", "node_id": "node_002",
         "problem": "Choice A and B lead to same outcome.",
         "suggestion": "Differentiate the consequences."},
    ],
    "repair_suggestions": ["Move reveal from node_001 to node_005."],
    "summary": "Premature reveal issue needs fixing.",
}


class _ScriptedLLMService(LLMService):
    """Mock LLM that returns pre-scripted JSON responses.

    Each call to generate_json pops the next response from the
    ``responses`` list.  If a response is an Exception, it is raised.
    """

    provider_name = "scripted_mock"

    def __init__(self, responses: list[Any] | None = None) -> None:
        # Skip parent __init__ which calls get_settings() — we don't
        # need usage tracking or budget checks for these unit tests.
        self._responses: list[Any] = responses or []
        self._call_count = 0
        self.call_args: list[dict[str, Any]] = []
        self.usage = None  # not needed for tests

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.call_args.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        self._call_count += 1
        if not self._responses:
            return {"score": 7.0, "issues": [], "repair_suggestions": [], "summary": ""}
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, str):
            raise LLMResponseError(f"Invalid JSON: {resp}")
        return resp

    async def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Not used — generate_json is overridden directly."""
        raise NotImplementedError("Use generate_json directly")


# ── Prompt tests ───────────────────────────────────────────────────


def test_critic_system_prompt_contains_all_13_criteria():
    """CRITIC_SYSTEM_PROMPT must mention all 13 review criteria (Spec §7.4)."""
    p = CRITIC_SYSTEM_PROMPT.lower()
    expected_keywords = [
        "premise",
        "conflict",
        "turning point",
        "decision relevance",
        "consequence",
        "dead end",
        "end reachab",
        "secret reveal",
        "character consist",
        "logic error",
        "linear",
        "audience fit",
        "safety",
    ]
    for kw in expected_keywords:
        assert kw in p, f"CRITIC_SYSTEM_PROMPT missing criterion: '{kw}'"


def test_critic_system_prompt_mentions_score_range():
    """Prompt must specify the 0.0–10.0 scoring range."""
    p = CRITIC_SYSTEM_PROMPT
    assert "0.0" in p and "10.0" in p
    assert "7.0" in p  # publishable threshold


def test_critic_system_prompt_mentions_severities():
    """Prompt must list all four severity levels."""
    p = CRITIC_SYSTEM_PROMPT.lower()
    for sev in ("high", "medium", "low", "info"):
        assert sev in p, f"Missing severity: '{sev}'"


def test_critic_system_prompt_mentions_repair_suggestions():
    """Prompt must request repair_suggestions in output (Spec §14.2)."""
    assert "repair_suggestions" in CRITIC_SYSTEM_PROMPT


def test_build_critic_user_prompt_contains_outline_and_graph():
    """User prompt must include both the outline and the graph JSON."""
    prompt = build_critic_user_prompt(SAMPLE_OUTLINE, SAMPLE_GRAPH)
    assert "STORY OUTLINE" in prompt
    assert "STORY GRAPH" in prompt
    assert "TASK" in prompt
    # Check key outline content is present
    assert "Mira" in prompt
    # Check key graph content is present
    assert "node_001" in prompt


def test_build_critic_user_prompt_is_json_serialised():
    """The prompt should embed JSON-serialised outline and graph."""
    prompt = build_critic_user_prompt(SAMPLE_OUTLINE, SAMPLE_GRAPH)
    # The JSON should be parseable from the prompt
    assert '"premise"' in prompt
    assert '"nodes"' in prompt
    assert '"start_node_id"' in prompt


# ── Pydantic schema tests ──────────────────────────────────────────


def test_critic_issue_valid():
    """CriticIssue should accept all valid severity levels."""
    for sev in ("high", "medium", "low", "info"):
        issue = CriticIssue(
            severity=sev, node_id="node_001",
            problem="Test problem", suggestion="Test suggestion",
        )
        assert issue.severity == sev


def test_critic_issue_invalid_severity():
    """CriticIssue should reject unknown severity values."""
    with pytest.raises(ValidationError):
        CriticIssue(
            severity="critical", node_id="node_001",
            problem="Test", suggestion="Fix",
        )


def test_critic_issue_severity_case_insensitive():
    """Severity should be normalised to lowercase."""
    issue = CriticIssue(
        severity="HIGH", node_id="n1",
        problem="Bad", suggestion="Fix it",
    )
    assert issue.severity == "high"


def test_critic_issue_node_id_optional():
    """node_id should be optional (None for graph-wide issues)."""
    issue = CriticIssue(
        severity="info", node_id=None,
        problem="Graph-wide issue", suggestion="Global fix",
    )
    assert issue.node_id is None


def test_critic_issue_empty_problem_rejected():
    """Empty problem string should be rejected."""
    with pytest.raises(ValidationError):
        CriticIssue(severity="low", problem="", suggestion="Fix")


def test_critic_issue_empty_suggestion_rejected():
    """Empty suggestion string should be rejected."""
    with pytest.raises(ValidationError):
        CriticIssue(severity="low", problem="Bad", suggestion="")


def test_review_report_valid():
    """CriticReviewReport should validate a well-formed review."""
    report = CriticReviewReport.model_validate(GOOD_REVIEW)
    assert report.score == 8.0
    assert len(report.issues) == 1
    assert report.issues[0].severity == "low"
    assert len(report.repair_suggestions) == 1
    assert report.summary == "Solid graph with good pacing."


def test_review_report_score_out_of_range():
    """Score > 10.0 should be rejected."""
    with pytest.raises(ValidationError):
        CriticReviewReport(score=11.0, issues=[])


def test_review_report_negative_score_rejected():
    """Score < 0.0 should be rejected."""
    with pytest.raises(ValidationError):
        CriticReviewReport(score=-1.0, issues=[])


def test_review_report_high_severity_issues():
    """high_severity_issues property should filter correctly."""
    report = CriticReviewReport.model_validate(HIGH_SEVERITY_REVIEW)
    highs = report.high_severity_issues
    assert len(highs) == 1
    assert highs[0].severity == "high"


def test_review_report_is_publishable_true():
    """Score >= 7.0 and no high issues → publishable."""
    report = CriticReviewReport.model_validate(GOOD_REVIEW)
    assert report.is_publishable is True


def test_review_report_is_publishable_false_low_score():
    """Score < 7.0 → not publishable."""
    report = CriticReviewReport(
        score=6.5, issues=[], repair_suggestions=[], summary="",
    )
    assert report.is_publishable is False


def test_review_report_is_publishable_false_high_severity():
    """High severity issue → not publishable even with good score."""
    report = CriticReviewReport.model_validate(HIGH_SEVERITY_REVIEW)
    assert report.is_publishable is False


def test_review_report_defaults():
    """Missing optional fields should get defaults."""
    report = CriticReviewReport(score=7.5)
    assert report.issues == []
    assert report.repair_suggestions == []
    assert report.summary == ""


def test_review_report_round_trip():
    """model_dump → model_validate should be identity."""
    report = CriticReviewReport.model_validate(GOOD_REVIEW)
    dumped = report.model_dump()
    report2 = CriticReviewReport.model_validate(dumped)
    assert report2.score == report.score
    assert len(report2.issues) == len(report.issues)


# ── StoryCriticAgent async tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_review_returns_valid_schema():
    """review() should return a dict matching the CriticReviewReport schema."""
    llm = _ScriptedLLMService(responses=[GOOD_REVIEW])
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert isinstance(result, dict)
    assert result["score"] == 8.0
    assert len(result["issues"]) == 1
    assert result["issues"][0]["severity"] == "low"
    assert len(result["repair_suggestions"]) == 1
    assert result["summary"] == "Solid graph with good pacing."


@pytest.mark.asyncio
async def test_review_passes_correct_prompts():
    """review() should pass CRITIC_SYSTEM_PROMPT and build_critic_user_prompt output."""
    llm = _ScriptedLLMService(responses=[GOOD_REVIEW])
    agent = StoryCriticAgent(llm=llm)
    await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert llm._call_count == 1
    args = llm.call_args[0]
    assert args["system_prompt"] == CRITIC_SYSTEM_PROMPT
    assert "STORY OUTLINE" in args["user_prompt"]
    assert "STORY GRAPH" in args["user_prompt"]


@pytest.mark.asyncio
async def test_review_retries_on_invalid_json():
    """review() should retry when LLM returns invalid JSON."""
    llm = _ScriptedLLMService(responses=[
        "not valid json",  # first attempt: LLMResponseError
        GOOD_REVIEW,        # second attempt: success
    ])
    agent = StoryCriticAgent(llm=llm, max_schema_retries=2)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert llm._call_count == 2
    assert result["score"] == 8.0


@pytest.mark.asyncio
async def test_review_retries_on_schema_mismatch():
    """review() should retry when LLM returns schema-mismatched JSON."""
    bad_review = {"score": "not a float", "issues": "not a list"}
    llm = _ScriptedLLMService(responses=[bad_review, GOOD_REVIEW])
    agent = StoryCriticAgent(llm=llm, max_schema_retries=2)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert llm._call_count == 2
    assert result["score"] == 8.0


@pytest.mark.asyncio
async def test_review_raises_after_max_retries():
    """review() should raise StoryCriticError after exhausting retries."""
    llm = _ScriptedLLMService(responses=[
        "bad json", "bad json", "bad json",
    ])
    agent = StoryCriticAgent(llm=llm, max_schema_retries=2)
    with pytest.raises(StoryCriticError):
        await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)
    assert llm._call_count == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_review_raises_on_score_out_of_range():
    """Schema validation should reject score > 10.0 and trigger retry."""
    bad_review = {"score": 15.0, "issues": []}
    llm = _ScriptedLLMService(responses=[bad_review, GOOD_REVIEW])
    agent = StoryCriticAgent(llm=llm, max_schema_retries=2)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)
    assert llm._call_count == 2
    assert result["score"] == 8.0


@pytest.mark.asyncio
async def test_review_with_no_issues():
    """A perfect graph should produce a review with no issues."""
    perfect = {"score": 10.0, "issues": [], "repair_suggestions": [], "summary": "Perfect."}
    llm = _ScriptedLLMService(responses=[perfect])
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert result["score"] == 10.0
    assert len(result["issues"]) == 0
    assert result["summary"] == "Perfect."


@pytest.mark.asyncio
async def test_review_with_high_severity():
    """Review with high-severity issues should be detectable."""
    llm = _ScriptedLLMService(responses=[HIGH_SEVERITY_REVIEW])
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert result["score"] == 5.5
    assert StoryCriticAgent.has_high_severity_issues(result)
    assert not StoryCriticAgent.is_publishable(result)


@pytest.mark.asyncio
async def test_review_normalises_severity_case():
    """Severity values should be normalised to lowercase."""
    review = {
        "score": 7.0,
        "issues": [
            {"severity": "HIGH", "node_id": "n1",
             "problem": "Bad", "suggestion": "Fix"},
        ],
        "repair_suggestions": [],
        "summary": "",
    }
    llm = _ScriptedLLMService(responses=[review])
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert result["issues"][0]["severity"] == "high"


@pytest.mark.asyncio
async def test_review_normalises_missing_repair_suggestions():
    """Missing repair_suggestions should default to empty list."""
    review = {
        "score": 7.0,
        "issues": [],
        "summary": "Good.",
        # repair_suggestions missing
    }
    llm = _ScriptedLLMService(responses=[review])
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert result["repair_suggestions"] == []


@pytest.mark.asyncio
async def test_review_normalises_missing_summary():
    """Missing summary should default to empty string."""
    review = {"score": 7.0, "issues": []}
    llm = _ScriptedLLMService(responses=[review])
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert result["summary"] == ""


@pytest.mark.asyncio
async def test_review_with_null_node_id():
    """Issues with null node_id should be accepted (graph-wide issues)."""
    review = {
        "score": 6.0,
        "issues": [
            {"severity": "medium", "node_id": None,
             "problem": "Too linear overall", "suggestion": "Add branching"},
        ],
        "repair_suggestions": ["Add more branches in act 2"],
        "summary": "Too linear.",
    }
    llm = _ScriptedLLMService(responses=[review])
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert result["issues"][0]["node_id"] is None


# ── Static helper tests ────────────────────────────────────────────


def test_has_high_severity_issues_true():
    """Should detect high-severity issues in a review dict."""
    assert StoryCriticAgent.has_high_severity_issues(HIGH_SEVERITY_REVIEW) is True


def test_has_high_severity_issues_false():
    """Should return False when no high-severity issues exist."""
    assert StoryCriticAgent.has_high_severity_issues(GOOD_REVIEW) is False


def test_has_high_severity_issues_empty():
    """Empty issues list → no high-severity issues."""
    review = {"score": 7.0, "issues": []}
    assert StoryCriticAgent.has_high_severity_issues(review) is False


def test_is_publishable_true():
    """Score >= 7.0 and no high issues → publishable."""
    assert StoryCriticAgent.is_publishable(GOOD_REVIEW) is True


def test_is_publishable_false_low_score():
    """Score < 7.0 → not publishable."""
    review = {"score": 6.5, "issues": []}
    assert StoryCriticAgent.is_publishable(review) is False


def test_is_publishable_false_high_severity():
    """High-severity issue → not publishable regardless of score."""
    assert StoryCriticAgent.is_publishable(HIGH_SEVERITY_REVIEW) is False


def test_is_publishable_custom_threshold():
    """Custom min_score threshold should work."""
    review = {"score": 8.0, "issues": []}
    assert StoryCriticAgent.is_publishable(review, min_score=8.0) is True
    assert StoryCriticAgent.is_publishable(review, min_score=9.0) is False


def test_is_publishable_invalid_score():
    """Non-numeric score should be treated as 0.0."""
    review = {"score": "invalid", "issues": []}
    assert StoryCriticAgent.is_publishable(review) is False


# ── MockLLMService integration ─────────────────────────────────────


@pytest.mark.asyncio
async def test_review_with_mock_llm_service():
    """review() should work with the real MockLLMService."""
    from app.services.llm_service import MockLLMService
    llm = MockLLMService()
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert isinstance(result["score"], (int, float))
    assert isinstance(result["issues"], list)
    assert isinstance(result["repair_suggestions"], list)
    assert isinstance(result["summary"], str)


@pytest.mark.asyncio
async def test_mock_llm_review_has_repair_suggestions():
    """MockLLMService should return repair_suggestions in the review."""
    from app.services.llm_service import MockLLMService
    llm = MockLLMService()
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(SAMPLE_OUTLINE, SAMPLE_GRAPH)

    assert len(result["repair_suggestions"]) > 0


@pytest.mark.asyncio
async def test_review_with_helios_graph_and_outline():
    """review() should work with the full Helios example data."""
    from app.services.story_authoring_agent import (
        DummyStoryAuthoringAgent,
    )
    from app.services.llm_service import MockLLMService

    authoring = DummyStoryAuthoringAgent()
    outline = await authoring.generate_outline({})
    graph = await authoring.generate_graph({})

    llm = MockLLMService()
    agent = StoryCriticAgent(llm=llm)
    result = await agent.review(outline, graph)

    assert 0.0 <= result["score"] <= 10.0
    assert isinstance(result["issues"], list)
    assert isinstance(result["repair_suggestions"], list)
