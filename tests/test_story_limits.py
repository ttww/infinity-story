"""Tests for sentence and connection limit enforcement in story generation.

Tests cover:
  - count_sentences: basic counting, edge cases
  - validate_node_sentences: min/max bounds, boundary cases
  - validate_node_connections: end nodes, auto-advance, regular nodes
  - find_violating_nodes: graph-level detection
  - adjust_node_connections: trimming excess, padding insufficient
  - enforce_graph_limits: full graph post-processing
  - StoryAuthoringAgent.generate_graph: retry on sentence violations
  - LLMService.generate_scene: retry on sentence count violation
"""

from __future__ import annotations

import json
import pytest

from app.story.limits import (
    count_sentences,
    validate_node_sentences,
    validate_node_connections,
    find_violating_nodes,
    adjust_node_connections,
    enforce_graph_limits,
)
from app.story.prompts import (
    build_graph_system_prompt,
    build_scene_system_prompt,
    GRAPH_SYSTEM_PROMPT,
    SCENE_SYSTEM_PROMPT,
)


# ── count_sentences tests ───────────────────────────────────────

class TestCountSentences:
    def test_three_sentences(self):
        assert count_sentences("Hallo. Wie geht es dir? Mir geht es gut!") == 3

    def test_empty_string(self):
        assert count_sentences("") == 0

    def test_no_terminator(self):
        assert count_sentences("Ein Satz ohne Punkt") == 0

    def test_single_sentence(self):
        assert count_sentences("Dies ist ein Satz.") == 1

    def test_exclamation_marks(self):
        assert count_sentences("Achtung! Stop! Warte!") == 3

    def test_mixed_punctuation(self):
        text = "Der Hund bellt. Die Katze schreit! Wer ist das?"
        assert count_sentences(text) == 3

    def test_ellipsis_not_counted_extra(self):
        text = "Er dachte nach... Dann ging er."
        assert count_sentences(text) == 2


# ── validate_node_sentences tests ───────────────────────────────

class TestValidateNodeSentences:
    def _node(self, goal: str) -> dict:
        return {"scene_goal": goal}

    def test_exactly_min(self):
        goal = "Ein Satz. Noch ein Satz. Und ein dritter."
        assert validate_node_sentences(self._node(goal), 3, 8) is True

    def test_exactly_max(self):
        parts = [f"Satz {i}." for i in range(8)]
        goal = " ".join(parts)
        assert validate_node_sentences(self._node(goal), 3, 8) is True

    def test_below_min(self):
        goal = "Nur ein Satz."
        assert validate_node_sentences(self._node(goal), 3, 8) is False

    def test_above_max(self):
        parts = [f"Satz {i}." for i in range(10)]
        goal = " ".join(parts)
        assert validate_node_sentences(self._node(goal), 3, 8) is False

    def test_empty_goal(self):
        assert validate_node_sentences(self._node(""), 3, 8) is False

    def test_within_range(self):
        parts = [f"Satz {i}." for i in range(5)]
        goal = " ".join(parts)
        assert validate_node_sentences(self._node(goal), 3, 8) is True


# ── validate_node_connections tests ─────────────────────────────

class TestValidateNodeConnections:
    def _choice(self, idx=0):
        return {"id": f"c{idx}", "label": f"Wahl {idx}", "next_node_id": "node_002"}

    def _node(self, choices=None, is_end=False, next_node_id=None, node_type="scene"):
        return {
            "choices": choices or [],
            "is_end": is_end,
            "type": node_type,
            "next_node_id": next_node_id,
        }

    def test_end_node_zero_choices(self):
        node = self._node(choices=[], is_end=True)
        assert validate_node_connections(node, 2, 5) is True

    def test_end_node_with_choices_is_invalid(self):
        node = self._node(choices=[self._choice()], is_end=True)
        assert validate_node_connections(node, 2, 5) is False

    def test_auto_advance_node(self):
        node = self._node(choices=[], next_node_id="node_002")
        assert validate_node_connections(node, 2, 5) is True

    def test_exactly_min_connections(self):
        choices = [self._choice(i) for i in range(2)]
        node = self._node(choices=choices)
        assert validate_node_connections(node, 2, 5) is True

    def test_exactly_max_connections(self):
        choices = [self._choice(i) for i in range(5)]
        node = self._node(choices=choices)
        assert validate_node_connections(node, 2, 5) is True

    def test_below_min_connections(self):
        choices = [self._choice(0)]
        node = self._node(choices=choices)
        assert validate_node_connections(node, 2, 5) is False

    def test_above_max_connections(self):
        choices = [self._choice(i) for i in range(6)]
        node = self._node(choices=choices)
        assert validate_node_connections(node, 2, 5) is False


# ── find_violating_nodes tests ──────────────────────────────────

class TestFindViolatingNodes:
    def _graph(self, nodes: dict) -> dict:
        return {"nodes": nodes, "start_node_id": list(nodes.keys())[0] if nodes else None}

    def test_clean_graph_no_violations(self):
        nodes = {
            "n1": {
                "scene_goal": "Eins. Zwei. Drei. Vier.",
                "choices": [{"id": "c1", "label": "A", "next_node_id": "n2"},
                            {"id": "c2", "label": "B", "next_node_id": "n3"}],
                "is_end": False, "type": "scene", "next_node_id": None,
            },
            "n2": {
                "scene_goal": "Ende. Schluss. Vorbei. Fertig.",
                "choices": [], "is_end": True, "type": "end", "next_node_id": None,
            },
        }
        graph = self._graph(nodes)
        violations = find_violating_nodes(graph, 3, 8, 2, 5)
        assert violations == []

    def test_sentence_violation_detected(self):
        nodes = {
            "n1": {
                "scene_goal": "Nur ein Satz.",
                "choices": [{"id": "c1"}, {"id": "c2"}],
                "is_end": False, "type": "scene", "next_node_id": None,
            },
        }
        graph = self._graph(nodes)
        violations = find_violating_nodes(graph, 3, 8, 2, 5)
        assert len(violations) == 1
        assert violations[0]["violation"] == "sentences"
        assert violations[0]["node_id"] == "n1"

    def test_connection_violation_detected(self):
        nodes = {
            "n1": {
                "scene_goal": "Eins. Zwei. Drei.",
                "choices": [{"id": f"c{i}"} for i in range(7)],
                "is_end": False, "type": "scene", "next_node_id": None,
            },
        }
        graph = self._graph(nodes)
        violations = find_violating_nodes(graph, 3, 8, 2, 5)
        assert len(violations) == 1
        assert violations[0]["violation"] == "connections"

    def test_both_violations_on_same_node(self):
        nodes = {
            "n1": {
                "scene_goal": "Nur ein Satz.",
                "choices": [{"id": "c1"} for _ in range(7)],
                "is_end": False, "type": "scene", "next_node_id": None,
            },
        }
        graph = self._graph(nodes)
        violations = find_violating_nodes(graph, 3, 8, 2, 5)
        assert len(violations) == 2


# ── adjust_node_connections tests ────────────────────────────────

class TestAdjustNodeConnections:
    def test_trims_excess_choices(self):
        node = {
            "choices": [{"id": f"c{i}", "label": f"W{i}", "next_node_id": "n2"} for i in range(7)],
            "is_end": False, "type": "scene", "next_node_id": None,
        }
        result = adjust_node_connections(node, 2, 5, ["n1", "n2"])
        assert len(result["choices"]) == 5

    def test_pads_insufficient_choices(self):
        node = {
            "choices": [{"id": "c0", "label": "W0", "next_node_id": "n2"}],
            "is_end": False, "type": "scene", "next_node_id": None,
        }
        result = adjust_node_connections(node, 2, 5, ["n1", "n2"])
        assert len(result["choices"]) == 2

    def test_end_node_unchanged(self):
        node = {
            "choices": [], "is_end": True, "type": "end", "next_node_id": None,
            "scene_goal": "Ende.",
        }
        result = adjust_node_connections(node, 2, 5, ["n1"])
        assert result["choices"] == []

    def test_auto_advance_unchanged(self):
        node = {
            "choices": [], "is_end": False, "type": "scene", "next_node_id": "n2",
            "scene_goal": "Uebergang.",
        }
        result = adjust_node_connections(node, 2, 5, ["n1", "n2"])
        assert result["choices"] == []

    def test_within_bounds_unchanged(self):
        choices = [{"id": f"c{i}", "label": f"W{i}", "next_node_id": "n2"} for i in range(3)]
        node = {
            "choices": choices, "is_end": False, "type": "scene", "next_node_id": None,
        }
        result = adjust_node_connections(node, 2, 5, ["n1", "n2"])
        assert len(result["choices"]) == 3
        assert result["choices"][0]["id"] == "c0"


# ── enforce_graph_limits tests ──────────────────────────────────

class TestEnforceGraphLimits:
    def test_trims_all_over_connection_nodes(self):
        graph = {
            "nodes": {
                "n1": {
                    "choices": [{"id": f"c{i}"} for i in range(7)],
                    "is_end": False, "type": "scene", "next_node_id": None,
                    "scene_goal": "Eins. Zwei. Drei.",
                },
                "n2": {
                    "choices": [{"id": f"c{i}"} for i in range(8)],
                    "is_end": False, "type": "scene", "next_node_id": None,
                    "scene_goal": "Eins. Zwei. Drei.",
                },
            },
            "start_node_id": "n1",
        }
        result = enforce_graph_limits(graph, 3, 8, 2, 5)
        for nid, node in result["nodes"].items():
            assert len(node["choices"]) <= 5

    def test_end_nodes_preserved(self):
        graph = {
            "nodes": {
                "n1": {
                    "choices": [{"id": "c1", "label": "A", "next_node_id": "n2"},
                                {"id": "c2", "label": "B", "next_node_id": "n2"}],
                    "is_end": False, "type": "scene", "next_node_id": None,
                    "scene_goal": "Eins. Zwei. Drei.",
                },
                "n2": {
                    "choices": [], "is_end": True, "type": "end", "next_node_id": None,
                    "scene_goal": "Ende. Schluss. Vorbei.",
                },
            },
            "start_node_id": "n1",
        }
        result = enforce_graph_limits(graph, 3, 8, 2, 5)
        assert result["nodes"]["n2"]["choices"] == []
        assert len(result["nodes"]["n1"]["choices"]) == 2


# ── Prompt builder tests ────────────────────────────────────────

class TestPromptBuilders:
    def test_graph_prompt_contains_custom_limits(self):
        prompt = build_graph_system_prompt(min_sentences=5, max_sentences=12, min_connections=3, max_connections=7)
        assert "5" in prompt and "12" in prompt
        assert "3" in prompt and "7" in prompt

    def test_graph_prompt_default_matches_constant(self):
        prompt = build_graph_system_prompt()
        assert prompt == GRAPH_SYSTEM_PROMPT

    def test_scene_prompt_contains_custom_limits(self):
        prompt = build_scene_system_prompt(min_sentences=4, max_sentences=10)
        assert "4" in prompt and "10" in prompt

    def test_scene_prompt_default_matches_constant(self):
        assert build_scene_system_prompt() == SCENE_SYSTEM_PROMPT


# ── StoryAuthoringAgent integration tests ───────────────────────

class TestAuthoringAgentLimitEnforcement:
    @pytest.mark.asyncio
    async def test_generate_graph_trims_excess_connections(self):
        """When MockLLM returns a graph with nodes that have too many choices,
        enforce_graph_limits should trim them to max_node_connections."""
        from app.services.story_authoring_agent import StoryAuthoringAgent
        from app.services.llm_service import MockLLMService

        agent = StoryAuthoringAgent(MockLLMService())
        # _MOCK_GRAPH has node_006 with 3 choices (within 2-5).
        # Use tight limits: min=2, max=2 to force trimming on node_006.
        outline = {"premise": "Test", "main_conflict": "Test", "core_mystery": "Test",
                   "main_characters": [], "endings": []}
        graph = await agent.generate_graph(
            outline, min_sentences=1, max_sentences=50,
            min_node_connections=2, max_node_connections=2,
        )
        for nid, node in graph["nodes"].items():
            is_end = node.get("is_end", False) or node.get("type") == "end"
            if not is_end and node.get("choices"):
                assert len(node["choices"]) <= 2, f"{nid} has {len(node['choices'])} choices"

    @pytest.mark.asyncio
    async def test_generate_graph_retries_on_sentence_violation(self):
        """When the LLM returns a graph with sentence violations, the agent
        should retry.  MockLLMService always returns the same graph, so
        after max_retries the agent accepts the best effort."""
        from app.services.story_authoring_agent import StoryAuthoringAgent
        from app.services.llm_service import MockLLMService

        agent = StoryAuthoringAgent(MockLLMService())
        outline = {"premise": "Test", "main_conflict": "Test", "core_mystery": "Test",
                   "main_characters": [], "endings": []}
        # Set min_sentences very high so the mock graph always violates
        graph = await agent.generate_graph(
            outline, min_sentences=50, max_sentences=100,
            min_node_connections=2, max_node_connections=5,
            max_retries=1,
        )
        # Graph should still be returned (best effort after retries)
        assert "nodes" in graph
        assert len(graph["nodes"]) > 0

    @pytest.mark.asyncio
    async def test_generate_graph_passes_limits_to_prompt(self):
        """The system prompt sent to the LLM should contain the custom limits."""
        from app.services.story_authoring_agent import StoryAuthoringAgent
        from app.services.llm_service import MockLLMService, LLMResponse
        import time

        class CaptureMockLLM(MockLLMService):
            captured_system_prompt: str = ""
            async def _complete(self, system_prompt, user_prompt, **kwargs):
                self.captured_system_prompt = system_prompt
                return await super()._complete(system_prompt, user_prompt, **kwargs)

        llm = CaptureMockLLM()
        agent = StoryAuthoringAgent(llm)
        outline = {"premise": "Test"}
        await agent.generate_graph(
            outline, min_sentences=7, max_sentences=14,
            min_node_connections=3, max_node_connections=6,
        )
        assert "7" in llm.captured_system_prompt
        assert "14" in llm.captured_system_prompt
        assert "3" in llm.captured_system_prompt
        assert "6" in llm.captured_system_prompt


# ── generate_scene sentence validation tests ─────────────────────

class TestGenerateSceneSentenceValidation:
    @pytest.mark.asyncio
    async def test_scene_retry_on_too_few_sentences(self):
        """When the LLM returns scene_text with too few sentences, generate_scene
        should retry.  MockLLMService returns a short mock scene, so with
        a high min_sentences it should trigger a retry (2 LLM calls)."""
        from app.services.llm_service import MockLLMService
        from app.services.story_orchestrator import StoryContext

        call_count = 0

        class CountingMockLLM(MockLLMService):
            async def _complete(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                return await super()._complete(*args, **kwargs)

        llm = CountingMockLLM()
        node = {
            "id": "node_001", "title": "Test", "scene_goal": "Test goal.",
            "location": "Here", "characters": [], "reveals": [],
            "choices": [{"id": "c1", "label": "A", "next_node_id": "n2"}],
        }
        ctx = StoryContext(
            session_id="s1", current_node=node,
            world_state={"genre": "scifi", "tone": "dark"},
        )
        # Mock scene text is short — set min_sentences=10 to force retry
        scene = await llm.generate_scene(ctx, min_sentences=10, max_sentences=20)
        assert call_count == 2  # initial + 1 retry

    @pytest.mark.asyncio
    async def test_scene_no_retry_when_within_bounds(self):
        """When scene_text is within sentence bounds, no retry happens."""
        from app.services.llm_service import MockLLMService
        from app.services.story_orchestrator import StoryContext

        call_count = 0

        class CountingMockLLM(MockLLMService):
            async def _complete(self, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                return await super()._complete(*args, **kwargs)

        llm = CountingMockLLM()
        node = {
            "id": "node_001", "title": "Test", "scene_goal": "Test goal.",
            "location": "Here", "characters": [], "reveals": [],
            "choices": [{"id": "c1", "label": "A", "next_node_id": "n2"}],
        }
        ctx = StoryContext(
            session_id="s1", current_node=node,
            world_state={"genre": "scifi", "tone": "dark"},
        )
        # Mock scene text: "[Mock scene] Knoten: ..." — 1-2 sentences.
        # With min=1, max=10 it should be within bounds.
        scene = await llm.generate_scene(ctx, min_sentences=1, max_sentences=10)
        assert call_count == 1  # no retry needed

    @pytest.mark.asyncio
    async def test_scene_passes_limits_to_system_prompt(self):
        """The system prompt should contain the custom sentence limits."""
        from app.services.llm_service import MockLLMService
        from app.services.story_orchestrator import StoryContext

        class CaptureMockLLM(MockLLMService):
            captured_system_prompt: str = ""
            async def _complete(self, system_prompt, user_prompt, **kwargs):
                self.captured_system_prompt = system_prompt
                return await super()._complete(system_prompt, user_prompt, **kwargs)

        llm = CaptureMockLLM()
        node = {
            "id": "n1", "title": "T", "scene_goal": "G.", "location": "L",
            "characters": [], "reveals": [],
            "choices": [{"id": "c1", "label": "A", "next_node_id": "n2"}],
        }
        ctx = StoryContext(session_id="s1", current_node=node, world_state={})
        await llm.generate_scene(ctx, min_sentences=5, max_sentences=12)
        assert "5" in llm.captured_system_prompt
        assert "12" in llm.captured_system_prompt
