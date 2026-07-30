"""Tests for story text marker validation (task t_3323ded4)."""

from __future__ import annotations

from app.story.story_text_validator import (
    INTERNAL_MARKER_PATTERNS,
    MarkerMatch,
    ValidationResult,
    validate_story_text_markers,
)


# ── Acceptance criteria ────────────────────────────────────────────


class TestAcceptanceCriteria:
    """Direct acceptance-criteria checks from the task spec."""

    def test_fails_on_node_002(self):
        """Input containing 'node_002' must return passed=False."""
        result = validate_story_text_markers("Siehst du node_002 im Text?")
        assert result.passed is False
        assert len(result.matches) == 1
        assert result.matches[0].matched_text == "node_002"
        assert result.matches[0].pattern_name == "node_id"

    def test_fails_on_teil_abc(self):
        """Input containing '(Teil abc)' must return passed=False."""
        result = validate_story_text_markers("Die Wahrheit (Teil abc) zeigt sich.")
        assert result.passed is False
        assert len(result.matches) == 1
        assert result.matches[0].matched_text == "(Teil abc)"
        assert result.matches[0].pattern_name == "teil_marker"

    def test_passes_clean_prose(self):
        """Clean narrative prose returns passed=True with empty matches."""
        result = validate_story_text_markers(
            "Die Tür öffnete sich langsam. Ein kalter Wind strich über ihr Gesicht. "
            "Sie zögerte, dann trat sie ein."
        )
        assert result.passed is True
        assert result.matches == []

    def test_function_is_importable(self):
        """The function must be importable from its module."""
        assert callable(validate_story_text_markers)

    def test_function_has_docstring(self):
        """The function must be documented with a docstring."""
        assert validate_story_text_markers.__doc__ is not None
        assert "internal" in validate_story_text_markers.__doc__.lower()


# ── Pattern coverage ──────────────────────────────────────────────


class TestPatternCoverage:
    """Verify all required and additional patterns are detected."""

    def test_node_id_variants(self):
        """node_001, node_14, node_999 etc. are all detected."""
        for sample in ["node_001", "node_14", "node_999", "node_002"]:
            result = validate_story_text_markers(f"Text {sample} end")
            assert result.passed is False
            assert result.matches[0].matched_text == sample

    def test_teil_marker_variants(self):
        """(Teil abc), (Teil XYZ), (Teil 1 von 3) are all detected."""
        for sample in ["(Teil abc)", "(Teil XYZ)", "(Teil 1 von 3)", "(Teil 2)"]:
            result = validate_story_text_markers(f"Text {sample} end")
            assert result.passed is False
            assert result.matches[0].matched_text == sample

    def test_json_field_leak_detected(self):
        """Raw JSON field names like 'scene_goal:' are flagged."""
        result = validate_story_text_markers("Die scene_goal: war klar.")
        assert result.passed is False
        assert result.matches[0].pattern_name == "json_field"

    def test_suggested_next_node_detected(self):
        """Bare 'suggested_next_node' references are flagged."""
        result = validate_story_text_markers("suggested_next_node: node_003")
        assert result.passed is False
        assert any(m.pattern_name == "suggested_next_node" for m in result.matches)


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: empty input, multiple matches, offsets, false positives."""

    def test_empty_string_passes(self):
        result = validate_story_text_markers("")
        assert result.passed is True
        assert result.matches == []

    def test_none_like_input_passes(self):
        """Falsy input (empty string) should not crash."""
        result = validate_story_text_markers("")
        assert result.passed is True

    def test_multiple_matches_collected(self):
        """Multiple markers in one text are all captured."""
        result = validate_story_text_markers(
            "Siehe node_001 und (Teil 2) sowie node_003."
        )
        assert result.passed is False
        assert len(result.matches) == 3

    def test_matches_sorted_by_offset(self):
        """Matches should be sorted by character offset (left-to-right)."""
        result = validate_story_text_markers("node_003 kommt vor node_001")
        offsets = [m.offset for m in result.matches]
        assert offsets == sorted(offsets)

    def test_offset_is_correct(self):
        """The offset field reports the correct character position."""
        text = "Hallo node_002!"
        idx = text.index("node_002")
        result = validate_story_text_markers(text)
        assert result.matches[0].offset == idx

    def test_normal_teil_not_flagged(self):
        """The German word 'Teil' in normal prose is not flagged."""
        result = validate_story_text_markers("Ein Teil der Wahrheit.")
        assert result.passed is True

    def test_normal_node_not_flagged(self):
        """The English word 'Node' without underscore+digits is not flagged."""
        result = validate_story_text_markers("Der Node war still.")
        assert result.passed is True


# ── Extensibility ──────────────────────────────────────────────────


class TestExtensibility:
    """The pattern list should be extensible without modifying the function."""

    def test_pattern_list_is_list_of_tuples(self):
        """INTERNAL_MARKER_PATTERNS is a list of (name, pattern) tuples."""
        for entry in INTERNAL_MARKER_PATTERNS:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], str)
            assert hasattr(entry[1], "finditer")

    def test_adding_pattern_is_detected_without_function_change(self):
        """Adding a pattern to the list is picked up automatically."""
        import re as _re
        from app.story import story_text_validator as mod

        original = mod.INTERNAL_MARKER_PATTERNS
        mod.INTERNAL_MARKER_PATTERNS.append(
            ("test_marker", _re.compile(r"TESTMARK_\d+"))
        )
        try:
            result = mod.validate_story_text_markers("Hier TESTMARK_42!")
            assert result.passed is False
            assert result.matches[0].pattern_name == "test_marker"
        finally:
            mod.INTERNAL_MARKER_PATTERNS[:] = original


# ── Result structure ──────────────────────────────────────────────


class TestResultStructure:
    """Validate the shape of ValidationResult and MarkerMatch."""

    def test_as_dict_shape(self):
        """as_dict() returns a dict with 'passed' and 'matches' keys."""
        result = validate_story_text_markers("node_001")
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "passed" in d
        assert "matches" in d
        assert d["passed"] is False
        match_dict = d["matches"][0]
        assert set(match_dict.keys()) == {"pattern_name", "pattern", "matched_text", "offset"}

    def test_marker_match_fields(self):
        """MarkerMatch has pattern_name, pattern, matched_text, offset."""
        result = validate_story_text_markers("node_001")
        m = result.matches[0]
        assert isinstance(m, MarkerMatch)
        assert m.pattern_name == "node_id"
        assert m.pattern == r"node_\d+"
        assert m.matched_text == "node_001"
        assert isinstance(m.offset, int)

    def test_validation_result_default_factory(self):
        """ValidationResult can be constructed with default empty matches."""
        r = ValidationResult(passed=True)
        assert r.matches == []


# ── Task t_5cd38d8a: explicit acceptance-criteria tests ────────────


class TestTaskAcceptanceCases:
    """The five explicit test cases from task t_5cd38d8a.

    These are self-contained and map 1:1 to the task spec, ensuring
    each required scenario is covered by its own named test — even if
    the general TestPatternCoverage / TestEdgeCases classes already
    exercise similar inputs.
    """

    def test_case1_node_002_detected(self):
        """Feed story text containing 'node_002' — assert passed=False
        and the matches array includes an entry for the node_\\d+ pattern."""
        result = validate_story_text_markers(
            "Du gehst durch den Wald und siehst node_002 leuchten."
        )
        assert result.passed is False
        node_matches = [m for m in result.matches if m.pattern_name == "node_id"]
        assert len(node_matches) == 1
        assert node_matches[0].matched_text == "node_002"

    def test_case2_teil_abc_detected(self):
        """Feed story text containing '(Teil abc)' — assert passed=False
        and the matches array includes an entry for the (Teil \\s+\\w+) pattern."""
        result = validate_story_text_markers(
            "Die Wahrheit (Teil abc) offenbarte sich langsam."
        )
        assert result.passed is False
        teil_matches = [m for m in result.matches if m.pattern_name == "teil_marker"]
        assert len(teil_matches) == 1
        assert teil_matches[0].matched_text == "(Teil abc)"

    def test_case3_both_markers_detected(self):
        """Feed text containing both 'node_002' and '(Teil abc)' — assert
        both are detected and reported."""
        result = validate_story_text_markers(
            "Siehst du node_002 im Text? Die Wahrheit (Teil abc) zeigt sich."
        )
        assert result.passed is False
        pattern_names = {m.pattern_name for m in result.matches}
        assert "node_id" in pattern_names
        assert "teil_marker" in pattern_names
        node_match = next(m for m in result.matches if m.pattern_name == "node_id")
        teil_match = next(m for m in result.matches if m.pattern_name == "teil_marker")
        assert node_match.matched_text == "node_002"
        assert teil_match.matched_text == "(Teil abc)"

    def test_case4_clean_prose_passes(self):
        """Feed clean narrative prose (no programmatic markers) — assert
        passed=True and empty matches array."""
        result = validate_story_text_markers(
            "Die alte Eiche krachte. Ein Schatten huschte über den Pfad. "
            "Sie hielt den Atem an und lauschte. Nur der Wind raschelte "
            "in den Blättern. Vorsichtig schritt sie vorwärts, den Bogen "
            "im Anschlag, bereit für alles, was kommen mochte."
        )
        assert result.passed is True
        assert result.matches == []

    def test_case5_numbers_without_node_prefix_pass(self):
        """Numbers without 'node_' prefix should pass (no over-matching)."""
        result = validate_story_text_markers(
            "Es waren 002 Soldaten und 14 Pferde auf dem Hof."
        )
        assert result.passed is True
        assert result.matches == []

    def test_case5_node_without_number_passes(self):
        """'node_' without a trailing number should pass (no over-matching)."""
        result = validate_story_text_markers(
            "Das node_ ist keine gültige Referenz."
        )
        assert result.passed is True
        assert result.matches == []

    def test_case5_lowercase_teil_without_parens_passes(self):
        """Lowercase 'teil' without parentheses should pass (no over-matching)."""
        result = validate_story_text_markers(
            "Das ist nur ein teil der Geschichte, mehr nicht."
        )
        assert result.passed is True
        assert result.matches == []
