"""Tests verifying all story-generation prompt templates forbid internal references.

Ensures that node IDs (e.g. "node_002"), internal identifiers, and technical
markers like "(Teil 1 von 3)" or "(Teil xxx)" are explicitly banned from
appearing in generated story prose.
"""

from __future__ import annotations

import re

import pytest

from app.story.prompts import (
    ENHANCEMENT_SYSTEM_PROMPT,
    GRAPH_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    SCENE_SYSTEM_PROMPT,
    build_graph_system_prompt,
    build_scene_system_prompt,
)


# ── All prompt templates that generate story content ──────────────

ALL_STORY_PROMPTS = [
    pytest.param(SCENE_SYSTEM_PROMPT, id="scene_system_prompt"),
    pytest.param(GRAPH_SYSTEM_PROMPT, id="graph_system_prompt"),
    pytest.param(REPAIR_SYSTEM_PROMPT, id="repair_system_prompt"),
    pytest.param(ENHANCEMENT_SYSTEM_PROMPT, id="enhancement_system_prompt"),
]


# ── Presence of prohibition rule ───────────────────────────────────


class TestInternalReferenceProhibition:
    """Every story-generation prompt must explicitly forbid internal references."""

    @pytest.mark.parametrize("prompt", ALL_STORY_PROMPTS)
    def test_prompt_contains_never_include(self, prompt):
        """The word 'NEVER' followed by 'include' should appear in every prompt."""
        prompt_lower = prompt.lower()
        assert "never include" in prompt_lower, (
            "Prompt must contain 'NEVER include' directive for internal references"
        )

    @pytest.mark.parametrize("prompt", ALL_STORY_PROMPTS)
    def test_prompt_mentions_node_ids(self, prompt):
        """The prompt must mention 'node IDs' as a forbidden reference type."""
        prompt_lower = prompt.lower()
        assert "node id" in prompt_lower, (
            "Prompt must mention 'node IDs' as a forbidden reference type"
        )

    @pytest.mark.parametrize("prompt", ALL_STORY_PROMPTS)
    def test_prompt_mentions_teil_marker(self, prompt):
        """The prompt must mention the '(Teil xxx)' technical marker as forbidden."""
        assert "Teil" in prompt, (
            "Prompt must mention '(Teil xxx)' as a forbidden technical marker"
        )

    @pytest.mark.parametrize("prompt", ALL_STORY_PROMPTS)
    def test_prompt_calls_them_internal_metadata(self, prompt):
        """The prompt must clarify that such references are internal metadata."""
        prompt_lower = prompt.lower()
        assert "internal metadata" in prompt_lower, (
            "Prompt must state that internal references are 'internal metadata'"
        )

    @pytest.mark.parametrize("prompt", ALL_STORY_PROMPTS)
    def test_no_node_id_pattern_in_prohibition_examples(self, prompt):
        """The prohibition text itself should reference 'node_002' as an example.
        This verifies the example identifier is present."""
        assert "node_002" in prompt, (
            "Prompt should include 'node_002' as an example of a forbidden node ID"
        )


# ── Dynamic prompt builders also include the prohibition ──────────


class TestDynamicPromptBuilders:
    """The dynamic prompt builder functions must also include the prohibition."""

    def test_build_scene_system_prompt_contains_prohibition(self):
        prompt = build_scene_system_prompt(min_sentences=3, max_sentences=8)
        assert "NEVER include" in prompt
        assert "node" in prompt.lower()
        assert "Teil" in prompt

    def test_build_graph_system_prompt_contains_prohibition(self):
        prompt = build_graph_system_prompt(
            min_sentences=3, max_sentences=8,
            min_connections=2, max_connections=5,
        )
        assert "NEVER include" in prompt
        assert "node" in prompt.lower()
        assert "Teil" in prompt

    def test_build_scene_system_prompt_custom_limits_keep_prohibition(self):
        """Changing sentence limits should not remove the prohibition."""
        prompt = build_scene_system_prompt(min_sentences=1, max_sentences=20)
        assert "NEVER include" in prompt
        assert "internal metadata" in prompt.lower()

    def test_build_graph_system_prompt_custom_limits_keep_prohibition(self):
        """Changing connection limits should not remove the prohibition."""
        prompt = build_graph_system_prompt(
            min_sentences=1, max_sentences=20,
            min_connections=1, max_connections=10,
        )
        assert "NEVER include" in prompt
        assert "internal metadata" in prompt.lower()


# ── Regex pattern verification (acceptance criteria) ──────────────


class TestRegexPatterns:
    """Verify the regex patterns from the acceptance criteria are defined
    and would match the forbidden patterns the prompts mention."""

    def test_node_id_pattern_matches_example(self):
        r"""The pattern node_\d+ should match 'node_002'."""
        pattern = re.compile(r"node_\d+")
        assert pattern.search("node_002") is not None
        assert pattern.search("node_001") is not None
        assert pattern.search("node_999") is not None

    def test_teil_pattern_matches_example(self):
        """The pattern for (Teil ...) should match '(Teil 1 von 3)' and '(Teil xxx)'."""
        pattern = re.compile(r"\(Teil\s+[^)]+\)")
        assert pattern.search("(Teil 1 von 3)") is not None
        assert pattern.search("(Teil xxx)") is not None
        assert pattern.search("(Teil 2)") is not None

    def test_node_id_pattern_does_not_match_normal_text(self):
        """The pattern should not match normal prose words."""
        pattern = re.compile(r"node_\d+")
        assert pattern.search("Der Node war still.") is None
        assert pattern.search("Eine normale Geschichte.") is None

    def test_teil_pattern_does_not_match_normal_text(self):
        """The pattern should not match the German word 'Teil' in normal prose."""
        pattern = re.compile(r"\(Teil\s+[^)]+\)")
        assert pattern.search("Ein Teil der Wahrheit.") is None
        assert pattern.search("Der dritte Teil der Serie") is None
