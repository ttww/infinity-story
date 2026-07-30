"""Integration tests for story text validator wired into generation pipeline.

Tests that the validator is correctly integrated into:
1. LLMService.generate_scene (runtime scene generation)
2. StoryAuthoringAgent.generate_graph (authoring-time graph generation)

Acceptance criteria (from task t_e38ab052):
- Story text containing 'node_002' is caught by the pipeline and does not
  silently pass through to the user.
- Clean story text passes through the pipeline unchanged.
- Validation failures are visible in logs with enough detail to identify
  which pattern triggered the rejection.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm_service import LLMResponse, LLMService, MockLLMService
from app.services.story_authoring_agent import StoryAuthoringAgent, _find_marker_violations
from app.services.story_orchestrator import GeneratedScene, StoryContext
from app.story.story_text_validator import validate_story_text_markers


# ── Helpers ────────────────────────────────────────────────────────


def _make_llm_response(text: str) -> LLMResponse:
    """Build a minimal LLMResponse for testing."""
    return LLMResponse(
        text=text,
        provider="test",
        model="test-model",
        input_tokens=10,
        output_tokens=10,
        cost_usd=0.0,
        elapsed_seconds=0.001,
    )


def _make_scene_json(scene_text: str, **extra: Any) -> str:
    """Build a JSON string for a scene response."""
    data = {
        "scene_text": scene_text,
        "choices": [],
        "state_updates": {},
        "suggested_next_node": None,
    }
    data.update(extra)
    return json.dumps(data, ensure_ascii=False)


# ── Runtime pipeline: LLMService.generate_scene ──────────────────


class TestGenerateSceneValidation:
    """Tests for the validator wired into LLMService.generate_scene."""

    @pytest.mark.asyncio
    async def test_clean_text_passes_through_unchanged(self):
        """Clean story text passes through the pipeline unchanged."""
        clean_text = (
            "Die Tür öffnete sich langsam. Ein kalter Wind strich über ihr Gesicht. "
            "Sie zögerte, dann trat sie ein."
        )

        class TestLLM(LLMService):
            provider_name = "test"

            async def _complete(self, system_prompt, user_prompt, **kwargs):
                return _make_llm_response(_make_scene_json(clean_text))

        llm = TestLLM()
        ctx = StoryContext(
            session_id="s1",
            current_node={
                "id": "node_001",
                "title": "Test",
                "scene_goal": "Test goal",
                "location": "Test location",
                "choices": [],
            },
            world_state={"language": "de"},
        )

        scene = await llm.generate_scene(ctx)
        assert scene.scene_text == clean_text
        # Verify it passes validation
        result = validate_story_text_markers(scene.scene_text)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_marker_text_triggers_retry(self):
        """Scene text containing 'node_002' triggers re-generation."""
        dirty_text = "Du siehst node_002 im Text. Das ist ein Problem."
        clean_text = "Du siehst einen dunklen Gang vor dir. Das Licht flackert."

        # Build a mock LLMService that returns dirty text first, then clean
        call_count = 0

        class TestLLM(LLMService):
            provider_name = "test"

            async def _complete(self, system_prompt, user_prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _make_llm_response(_make_scene_json(dirty_text))
                return _make_llm_response(_make_scene_json(clean_text))

        llm = TestLLM()
        ctx = StoryContext(
            session_id="s1",
            current_node={
                "id": "node_001",
                "title": "Test",
                "scene_goal": "Test goal",
                "location": "Test location",
                "choices": [],
            },
            world_state={"language": "de"},
        )

        scene = await llm.generate_scene(ctx)

        # Should have been called twice (original + retry)
        assert call_count == 2
        # The returned scene should have clean text
        result = validate_story_text_markers(scene.scene_text)
        assert result.passed is True, f"Expected clean text, got: {scene.scene_text}"

    @pytest.mark.asyncio
    async def test_marker_failure_logged_with_details(self, caplog):
        """Validation failures are visible in logs with pattern details."""
        # Use enough sentences to pass sentence validation.
        dirty_text = (
            "Sie geht zu node_002 und findet Klarheit. "
            "Der Gang ist dunkel und eng. "
            "Sie hört ein Flüstern."
        )

        class TestLLM(LLMService):
            provider_name = "test"

            async def _complete(self, system_prompt, user_prompt, **kwargs):
                # Always return dirty text — forces escalation path
                return _make_llm_response(_make_scene_json(dirty_text))

        llm = TestLLM()
        ctx = StoryContext(
            session_id="s1",
            current_node={
                "id": "node_001",
                "title": "Test",
                "scene_goal": "Test goal",
                "location": "Test location",
                "choices": [],
            },
            world_state={"language": "de"},
        )

        with caplog.at_level(logging.WARNING, logger="app.services.llm_service"):
            scene = await llm.generate_scene(ctx)

        # Log should contain pattern details
        log_text = " ".join(r.getMessage() for r in caplog.records)
        assert "internal marker validation" in log_text.lower()
        # Should mention at least one pattern name
        assert "node_id" in log_text or "json_field" in log_text

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_validator_error(self):
        """If the validator throws, the pipeline degrades gracefully."""
        clean_text = "Die Sonne schien hell über den Bergen."

        class TestLLM(LLMService):
            provider_name = "test"

            async def _complete(self, system_prompt, user_prompt, **kwargs):
                return _make_llm_response(_make_scene_json(clean_text))

        llm = TestLLM()
        ctx = StoryContext(
            session_id="s1",
            current_node={
                "id": "node_001",
                "title": "Test",
                "scene_goal": "Test goal",
                "location": "Test location",
                "choices": [],
            },
            world_state={"language": "de"},
        )

        # Patch validate_story_text_markers to raise
        with patch(
            "app.story.story_text_validator.validate_story_text_markers",
            side_effect=RuntimeError("Simulated validator crash"),
        ):
            # Should NOT raise — should degrade gracefully
            scene = await llm.generate_scene(ctx)

        # Original scene should be returned
        assert scene.scene_text == clean_text

    @pytest.mark.asyncio
    async def test_node_002_does_not_silently_pass(self):
        """Text with 'node_002' must not silently pass through to the user."""
        # Use enough sentences to pass sentence validation.
        dirty_text = (
            "Gehe zu node_002 um fortzufahren. "
            "Der Flur ist dunkel. "
            "Du zögerst."
        )
        clean_text = (
            "Gehe durch die Tür, um fortzufahren. "
            "Der Flur ist dunkel. "
            "Du zögerst."
        )

        call_count = 0

        class TestLLM(LLMService):
            provider_name = "test"

            async def _complete(self, system_prompt, user_prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _make_llm_response(_make_scene_json(dirty_text))
                return _make_llm_response(_make_scene_json(clean_text))

        llm = TestLLM()
        ctx = StoryContext(
            session_id="s1",
            current_node={
                "id": "node_001",
                "title": "Test",
                "scene_goal": "Test goal",
                "location": "Test location",
                "choices": [],
            },
            world_state={"language": "de"},
        )

        scene = await llm.generate_scene(ctx)
        # The final text must NOT contain node_002
        assert "node_002" not in scene.scene_text
        assert "node_" not in scene.scene_text


# ── Authoring pipeline: StoryAuthoringAgent.generate_graph ───────


class TestGenerateGraphValidation:
    """Tests for the validator wired into StoryAuthoringAgent.generate_graph."""

    def test_find_marker_violations_clean_graph(self):
        """No violations in a clean graph."""
        graph = {
            "nodes": {
                "node_001": {
                    "scene_text": "Die Sonne geht auf. Es ist ein neuer Tag.",
                    "scene_goal": "Einführung der Szene",
                },
                "node_002": {
                    "scene_text": "Der Wald ist dunkel und still.",
                    "scene_goal": "Spannung aufbauen",
                },
            }
        }
        violations = _find_marker_violations(graph)
        assert violations == []

    def test_find_marker_violations_dirty_graph(self):
        """Violations detected when node text contains markers."""
        graph = {
            "nodes": {
                "node_001": {
                    "scene_text": "Du betrittst node_002 und siehst die Szene.",
                    "scene_goal": "Einführung der Szene",
                },
                "node_002": {
                    "scene_text": "Der Wald ist dunkel und still.",
                    "scene_goal": "Spannung aufbauen, scene_goal: wichtig",
                },
            }
        }
        violations = _find_marker_violations(graph)
        assert len(violations) == 2
        # First violation: node_id in scene_text
        assert violations[0]["node_id"] == "node_001"
        assert violations[0]["pattern_name"] == "node_id"
        assert violations[0]["matched_text"] == "node_002"
        # Second violation: json_field in scene_goal
        assert violations[1]["node_id"] == "node_002"
        assert violations[1]["pattern_name"] == "json_field"

    def test_find_marker_violations_empty_graph(self):
        """Empty graph returns no violations."""
        assert _find_marker_violations({"nodes": {}}) == []
        assert _find_marker_violations({}) == []

    def test_find_marker_violations_non_dict_nodes(self):
        """Non-dict nodes field returns no violations (graceful)."""
        assert _find_marker_violations({"nodes": "not a dict"}) == []

    @pytest.mark.asyncio
    async def test_generate_graph_retries_on_marker_violation(self, caplog):
        """generate_graph retries when node text contains internal markers."""
        # Graphs use scene_text that has enough sentences to pass sentence
        # validation but contains internal markers.
        dirty_scene = (
            "Du stehst vor node_002 und zögerst. "
            "Der Wind heult in den Bäumen. "
            "Es ist kalt und dunkel."
        )
        clean_scene = (
            "Du stehst vor der Tür und zögerst. "
            "Der Wind heult in den Bäumen. "
            "Es ist kalt und dunkel."
        )
        dirty_graph = {
            "nodes": {
                "node_001": {
                    "title": "Start",
                    "type": "start",
                    "scene_text": dirty_scene,
                    "scene_goal": (
                        "Beginn der Geschichte. Einführung der Szene. "
                        "Der Protagonist erwacht."
                    ),
                    "choices": [],
                    "is_start": True,
                    "is_end": True,
                },
            },
            "start_node_id": "node_001",
        }
        clean_graph = {
            "nodes": {
                "node_001": {
                    "title": "Start",
                    "type": "start",
                    "scene_text": clean_scene,
                    "scene_goal": (
                        "Beginn der Geschichte. Einführung der Szene. "
                        "Der Protagonist erwacht."
                    ),
                    "choices": [],
                    "is_start": True,
                    "is_end": True,
                },
            },
            "start_node_id": "node_001",
        }

        call_count = 0

        class TestLLM(LLMService):
            provider_name = "test"

            async def generate_json(self, system_prompt, user_prompt, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return dirty_graph
                return clean_graph

            async def _complete(self, *args, **kwargs):
                return _make_llm_response(_make_scene_json("Clean text."))

        agent = StoryAuthoringAgent(llm=TestLLM())

        with caplog.at_level(logging.WARNING, logger="app.services.story_authoring_agent"):
            result = await agent.generate_graph(
                outline={"title": "Test", "genre": "test", "tone": "test"},
                max_retries=2,
            )

        # Should have retried
        assert call_count >= 2
        # Final result should have clean text
        node = result["nodes"]["node_001"]
        assert "node_" not in node["scene_text"]
        # Log should mention marker violations
        log_text = " ".join(r.getMessage() for r in caplog.records)
        assert "marker violation" in log_text.lower()

    @pytest.mark.asyncio
    async def test_generate_graph_accepts_best_effort(self, caplog):
        """When markers persist after retries, graph is accepted with warning."""
        dirty_scene = (
            "Du stehst vor node_002 und zögerst. "
            "Der Wind heult in den Bäumen. "
            "Es ist kalt und dunkel."
        )
        dirty_graph = {
            "nodes": {
                "node_001": {
                    "title": "Start",
                    "type": "start",
                    "scene_text": dirty_scene,
                    "scene_goal": (
                        "Beginn der Geschichte. Einführung der Szene. "
                        "Der Protagonist erwacht."
                    ),
                    "choices": [],
                    "is_start": True,
                    "is_end": True,
                },
            },
            "start_node_id": "node_001",
        }

        class TestLLM(LLMService):
            provider_name = "test"

            async def generate_json(self, system_prompt, user_prompt, **kwargs):
                # Always return dirty graph
                return dirty_graph

            async def _complete(self, *args, **kwargs):
                return _make_llm_response(_make_scene_json("Clean text."))

        agent = StoryAuthoringAgent(llm=TestLLM())

        with caplog.at_level(logging.WARNING, logger="app.services.story_authoring_agent"):
            result = await agent.generate_graph(
                outline={"title": "Test", "genre": "test", "tone": "test"},
                max_retries=1,
            )

        # Should still return the graph (best effort)
        assert "nodes" in result
        # Log should mention accepting best effort
        log_text = " ".join(r.getMessage() for r in caplog.records)
        assert "best effort" in log_text.lower() or "marker violation" in log_text.lower()


# ── Non-destructive behavior ─────────────────────────────────────


class TestNonDestructiveBehavior:
    """Verify the integration is non-destructive."""

    @pytest.mark.asyncio
    async def test_validator_error_does_not_block_output(self):
        """If the validator crashes, story output is not blocked."""
        clean_text = "Die Sonne schien hell über den Bergen."

        class TestLLM(LLMService):
            provider_name = "test"

            async def _complete(self, system_prompt, user_prompt, **kwargs):
                return _make_llm_response(_make_scene_json(clean_text))

        llm = TestLLM()
        ctx = StoryContext(
            session_id="s1",
            current_node={
                "id": "node_001",
                "title": "Test",
                "scene_goal": "Test goal",
                "location": "Test location",
                "choices": [],
            },
            world_state={"language": "de"},
        )

        with patch(
            "app.story.story_text_validator.validate_story_text_markers",
            side_effect=ValueError("Crash"),
        ):
            scene = await llm.generate_scene(ctx)

        # Scene should still be returned
        assert scene.scene_text == clean_text

    @pytest.mark.asyncio
    async def test_regen_failure_returns_original_scene(self):
        """If re-generation fails, the original scene is returned."""
        # Use enough sentences to pass sentence validation so we reach
        # the marker validation path.
        dirty_text = (
            "Gehe zu node_002 im Dunkeln. "
            "Der Flur ist lang und leer. "
            "Du hörst ein Flüstern."
        )

        class TestLLM(LLMService):
            provider_name = "test"

            async def _complete(self, system_prompt, user_prompt, **kwargs):
                # First call returns dirty text, second call (with correction
                # appended) raises.
                if "IMPORTANT" not in system_prompt:
                    return _make_llm_response(_make_scene_json(dirty_text))
                raise RuntimeError("LLM service down")

        llm = TestLLM()
        ctx = StoryContext(
            session_id="s1",
            current_node={
                "id": "node_001",
                "title": "Test",
                "scene_goal": "Test goal",
                "location": "Test location",
                "choices": [],
            },
            world_state={"language": "de"},
        )

        scene = await llm.generate_scene(ctx)
        # Should return the original (dirty) scene, not crash
        assert scene.scene_text == dirty_text
