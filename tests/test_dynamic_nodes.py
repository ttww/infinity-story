"""Tests for dynamic node rendering and transition logic.

Tests the three node modes introduced by the dynamic node schema:
- auto_advance (0 choices, next_node_id set)
- single_path (1 choice, implicit continuation)
- multi_choice (2+ choices, explicit buttons)
- ending (0 choices, no next_node_id, or is_end=True)

Also tests:
- derive_node_mode() utility
- Orchestrator handling of auto_advance nodes
- ChoiceInterpreter with extended letter support (a-z)
- Graph serialization round-trip of new fields
- Validation rules for auto_advance nodes
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("MIN_NODE_COUNT", "3")
os.environ.setdefault("MIN_ENDING_COUNT", "1")

import pytest

from app.models import StoryNode, Choice
from app.story.graph import (
    derive_node_mode,
    graph_to_dict,
    load_graph_from_dict,
)
from app.services.story_orchestrator import (
    ChoiceInterpreter,
    StoryContext,
    StoryOrchestrator,
)
from app.services.llm_service import MockLLMService
from app.services.story_validation_service import StoryValidationService


# ── derive_node_mode tests ──────────────────────────────────────


class TestDeriveNodeMode:
    """Test the mode derivation logic."""

    def test_zero_choices_with_next_is_auto_advance(self):
        mode = derive_node_mode([], is_end=False, next_node_id="node_002")
        assert mode == "auto_advance"

    def test_zero_choices_no_next_is_ending(self):
        mode = derive_node_mode([], is_end=False, next_node_id=None)
        assert mode == "ending"

    def test_zero_choices_is_end_is_ending(self):
        mode = derive_node_mode([], is_end=True, next_node_id="node_002")
        assert mode == "ending"

    def test_one_choice_is_single_path(self):
        mode = derive_node_mode(
            [{"id": "c1", "label": "Weiter", "next_node_id": "n2"}],
            is_end=False,
            next_node_id=None,
        )
        assert mode == "single_path"

    def test_two_choices_is_multi_choice(self):
        mode = derive_node_mode(
            [
                {"id": "c1", "label": "A", "next_node_id": "n2"},
                {"id": "c2", "label": "B", "next_node_id": "n3"},
            ],
            is_end=False,
            next_node_id=None,
        )
        assert mode == "multi_choice"

    def test_four_choices_is_multi_choice(self):
        mode = derive_node_mode(
            [
                {"id": "c1", "label": "A", "next_node_id": "n2"},
                {"id": "c2", "label": "B", "next_node_id": "n3"},
                {"id": "c3", "label": "C", "next_node_id": "n4"},
                {"id": "c4", "label": "D", "next_node_id": "n5"},
            ],
            is_end=False,
            next_node_id=None,
        )
        assert mode == "multi_choice"

    def test_is_end_overrides_choices(self):
        """Even with choices, is_end=True means ending."""
        mode = derive_node_mode(
            [{"id": "c1", "label": "A", "next_node_id": "n2"}],
            is_end=True,
            next_node_id=None,
        )
        assert mode == "ending"


# ── Model tests ─────────────────────────────────────────────────


class TestStoryNodeModel:
    """Test the StoryNode Pydantic model with new fields."""

    def test_next_node_id_defaults_to_none(self):
        node = StoryNode(id="n1")
        assert node.next_node_id is None

    def test_auto_advance_delay_ms_defaults_to_none(self):
        node = StoryNode(id="n1")
        assert node.auto_advance_delay_ms is None

    def test_is_start_defaults_false(self):
        node = StoryNode(id="n1")
        assert node.is_start is False

    def test_is_end_defaults_false(self):
        node = StoryNode(id="n1")
        assert node.is_end is False

    def test_node_with_next_node_id(self):
        node = StoryNode(id="n1", next_node_id="n2", auto_advance_delay_ms=3000)
        assert node.next_node_id == "n2"
        assert node.auto_advance_delay_ms == 3000

    def test_node_with_is_start_is_end(self):
        node = StoryNode(id="n1", is_start=True, is_end=False)
        assert node.is_start is True
        assert node.is_end is False


# ── Graph serialization round-trip tests ────────────────────────


class TestGraphSerialization:
    """Test that new fields survive graph_to_dict / load_graph_from_dict."""

    def test_round_trip_next_node_id(self):
        from app.models import StoryGraph
        node = StoryNode(id="n1", next_node_id="n2", auto_advance_delay_ms=4000)
        graph = StoryGraph(nodes={"n1": node}, start_node_id="n1")
        d = graph_to_dict(graph)
        assert d["nodes"]["n1"]["next_node_id"] == "n2"
        assert d["nodes"]["n1"]["auto_advance_delay_ms"] == 4000

        # Round-trip back
        graph2 = load_graph_from_dict(d)
        assert graph2.nodes["n1"].next_node_id == "n2"
        assert graph2.nodes["n1"].auto_advance_delay_ms == 4000

    def test_round_trip_is_start_is_end(self):
        from app.models import StoryGraph
        node = StoryNode(id="n1", is_start=True, is_end=False)
        graph = StoryGraph(nodes={"n1": node}, start_node_id="n1")
        d = graph_to_dict(graph)
        assert d["nodes"]["n1"]["is_start"] is True
        assert d["nodes"]["n1"]["is_end"] is False

        graph2 = load_graph_from_dict(d)
        assert graph2.nodes["n1"].is_start is True
        assert graph2.nodes["n1"].is_end is False

    def test_round_trip_preserves_choices(self):
        from app.models import StoryGraph
        choices = [
            Choice(id="c1", label="A", next_node_id="n2"),
            Choice(id="c2", label="B", next_node_id="n3"),
        ]
        node = StoryNode(id="n1", choices=choices)
        graph = StoryGraph(nodes={"n1": node}, start_node_id="n1")
        d = graph_to_dict(graph)
        graph2 = load_graph_from_dict(d)
        assert len(graph2.nodes["n1"].choices) == 2
        assert graph2.nodes["n1"].choices[0].id == "c1"


# ── ChoiceInterpreter extended letter support ───────────────────


class TestChoiceInterpreterExtended:
    """Test that letter selection works beyond a-d."""

    def test_letter_e_works(self):
        choices = [
            {"id": "a", "label": "A", "next_node_id": "n1"},
            {"id": "b", "label": "B", "next_node_id": "n2"},
            {"id": "c", "label": "C", "next_node_id": "n3"},
            {"id": "d", "label": "D", "next_node_id": "n4"},
            {"id": "e", "label": "E", "next_node_id": "n5"},
        ]
        cid, is_free = ChoiceInterpreter.interpret("e", choices)
        assert cid == "e"
        assert is_free is False

    def test_letter_e_uppercase(self):
        choices = [
            {"id": "a", "label": "A", "next_node_id": "n1"},
            {"id": "b", "label": "B", "next_node_id": "n2"},
            {"id": "c", "label": "C", "next_node_id": "n3"},
            {"id": "d", "label": "D", "next_node_id": "n4"},
            {"id": "e", "label": "E", "next_node_id": "n5"},
        ]
        cid, is_free = ChoiceInterpreter.interpret("E", choices)
        assert cid == "e"
        assert is_free is False

    def test_letter_beyond_range_is_free(self):
        choices = [
            {"id": "a", "label": "A", "next_node_id": "n1"},
        ]
        cid, is_free = ChoiceInterpreter.interpret("z", choices)
        assert cid is None
        assert is_free is True


# ── Orchestrator auto_advance tests ─────────────────────────────


class TestOrchestratorAutoAdvance:
    """Test the orchestrator's handling of auto_advance nodes."""

    @pytest.mark.asyncio
    async def test_auto_advance_node_advances(self):
        """A node with 0 choices and next_node_id should auto-advance."""
        orchestrator = StoryOrchestrator(MockLLMService())

        # Build a scenario with an auto_advance node
        scenario = {
            "nodes": {
                "auto_node": {
                    "id": "auto_node",
                    "title": "Flur",
                    "type": "scene",
                    "scene_goal": "Ein dunkler Flur.",
                    "location": "Flur",
                    "characters": [],
                    "reveals": [],
                    "choices": [],
                    "next_node_id": "next_node",
                    "is_end": False,
                },
                "next_node": {
                    "id": "next_node",
                    "title": "Tür",
                    "type": "scene",
                    "scene_goal": "Eine verschlossene Tür.",
                    "location": "Tür",
                    "characters": [],
                    "reveals": [],
                    "choices": [
                        {"id": "open", "label": "Öffnen", "next_node_id": None}
                    ],
                    "is_end": False,
                },
            },
            "start_node_id": "auto_node",
        }

        ctx = orchestrator.build_context(
            session_id="test-session",
            node=scenario["nodes"]["auto_node"],
            world_state={"genre": "test"},
            user_input="__auto_advance__",
            scenario_id="test",
        )
        result = await orchestrator.process_user_input(ctx, scenario=scenario)

        assert result.next_node_id == "next_node"
        assert result.is_ending is False
        assert result.mode == "single_path"  # next_node has 1 choice

    @pytest.mark.asyncio
    async def test_ending_node_returns_ending_mode(self):
        """A node with 0 choices and no next_node_id should be ending."""
        orchestrator = StoryOrchestrator(MockLLMService())

        scenario = {
            "nodes": {
                "end_node": {
                    "id": "end_node",
                    "title": "Ende",
                    "type": "end",
                    "scene_goal": "Die Geschichte endet.",
                    "location": "Ende",
                    "characters": [],
                    "reveals": [],
                    "choices": [],
                    "is_end": True,
                },
            },
            "start_node_id": "end_node",
        }

        ctx = orchestrator.build_context(
            session_id="test-session",
            node=scenario["nodes"]["end_node"],
            world_state={"genre": "test"},
            user_input="something",
            scenario_id="test",
        )
        result = await orchestrator.process_user_input(ctx, scenario=scenario)

        assert result.is_ending is True
        assert result.mode == "ending"

    @pytest.mark.asyncio
    async def test_opening_scene_includes_mode(self):
        """generate_opening_scene should include the mode in the result."""
        orchestrator = StoryOrchestrator(MockLLMService())

        node = {
            "id": "n1",
            "title": "Start",
            "type": "start",
            "scene_goal": "Beginning.",
            "location": "Here",
            "characters": [],
            "reveals": [],
            "choices": [
                {"id": "a", "label": "A", "next_node_id": "n2"},
                {"id": "b", "label": "B", "next_node_id": "n3"},
            ],
            "is_start": True,
            "is_end": False,
        }

        ctx = orchestrator.build_context(
            session_id="test-session",
            node=node,
            world_state={"genre": "test"},
        )
        result = await orchestrator.generate_opening_scene(ctx)

        assert result.mode == "multi_choice"


# ── Validation tests for auto_advance nodes ─────────────────────


class TestValidationAutoAdvance:
    """Test validation rules for auto_advance and ending nodes."""

    @pytest.mark.asyncio
    async def test_auto_advance_without_next_node_id_is_error(self):
        """A non-optional node with 0 choices and no next_node_id is an error."""
        graph = {
            "nodes": {
                "start": {
                    "id": "start",
                    "type": "start",
                    "is_start": True,
                    "scene_goal": "Start",
                    "choices": [
                        {"id": "c1", "label": "Go", "next_node_id": "bad"}
                    ],
                    "quality_notes": ["note"],
                },
                "bad": {
                    "id": "bad",
                    "type": "scene",
                    "is_end": False,
                    "scene_goal": "No choices, no next",
                    "choices": [],
                    "quality_notes": ["note"],
                },
            },
            "start_node_id": "start",
        }
        service = StoryValidationService()
        result = await service.validate(graph)
        assert not result["is_valid"]
        assert any("no next_node_id" in e or "not marked as an ending" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_auto_advance_with_next_node_id_is_valid(self):
        """A node with 0 choices and next_node_id is valid (auto_advance)."""
        graph = {
            "nodes": {
                "start": {
                    "id": "start",
                    "type": "start",
                    "is_start": True,
                    "scene_goal": "Start",
                    "choices": [],
                    "next_node_id": "next",
                    "quality_notes": ["note"],
                },
                "next": {
                    "id": "next",
                    "type": "end",
                    "is_end": True,
                    "scene_goal": "End",
                    "choices": [],
                    "quality_notes": ["note"],
                },
            },
            "start_node_id": "start",
        }
        service = StoryValidationService()
        result = await service.validate(graph)
        assert result["is_valid"], result["errors"]

    @pytest.mark.asyncio
    async def test_ending_via_zero_choices_no_next(self):
        """A node with 0 choices and no next_node_id is an implicit ending."""
        graph = {
            "nodes": {
                "start": {
                    "id": "start",
                    "type": "start",
                    "is_start": True,
                    "scene_goal": "Start",
                    "choices": [],
                    "next_node_id": "end",
                    "quality_notes": ["note"],
                },
                "end": {
                    "id": "end",
                    "type": "end",
                    "is_end": True,
                    "scene_goal": "End",
                    "choices": [],
                    "next_node_id": None,
                    "quality_notes": ["note"],
                },
            },
            "start_node_id": "start",
        }
        service = StoryValidationService()
        result = await service.validate(graph)
        # "end" node has 0 choices, no next_node_id, is_end=True → valid ending
        assert result["is_valid"], result["errors"]

    @pytest.mark.asyncio
    async def test_node_level_next_node_id_reference_checked(self):
        """Node-level next_node_id referencing a missing node is an error."""
        graph = {
            "nodes": {
                "start": {
                    "id": "start",
                    "type": "start",
                    "is_start": True,
                    "scene_goal": "Start",
                    "choices": [],
                    "next_node_id": "nonexistent",
                    "quality_notes": ["note"],
                },
            },
            "start_node_id": "start",
        }
        service = StoryValidationService()
        result = await service.validate(graph)
        assert not result["is_valid"]
        assert any("nonexistent" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_auto_advance_chain_reachable(self):
        """BFS should follow node-level next_node_id for reachability."""
        graph = {
            "nodes": {
                "start": {
                    "id": "start",
                    "type": "start",
                    "is_start": True,
                    "scene_goal": "Start",
                    "choices": [],
                    "next_node_id": "middle",
                    "quality_notes": ["note"],
                },
                "middle": {
                    "id": "middle",
                    "type": "scene",
                    "is_end": False,
                    "scene_goal": "Middle",
                    "choices": [],
                    "next_node_id": "end",
                    "quality_notes": ["note"],
                },
                "end": {
                    "id": "end",
                    "type": "end",
                    "is_end": True,
                    "scene_goal": "End",
                    "choices": [],
                    "quality_notes": ["note"],
                },
            },
            "start_node_id": "start",
        }
        service = StoryValidationService()
        result = await service.validate(graph)
        assert result["is_valid"], result["errors"]
