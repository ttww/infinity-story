"""Test deterministic story graph validation (Spec §7.6).

Covers every check listed in the task body:
 - next_node_id existence
 - exactly one start node
 - at least one end node
 - all end nodes reachable
 - no unreachable mandatory nodes
 - no broken references
 - no duplicate IDs
 - no empty choice labels
 - no node without scene_goal
 - state updates valid JSON
 - graph serialisable
 - no cycles without progress
 - quality metadata present
"""

import pytest

from app.services.story_validation_service import StoryValidationService


# ── helpers ────────────────────────────────────────────────────────

def _svc() -> StoryValidationService:
    return StoryValidationService()


def _node(
    nid: str,
    *,
    type: str = "scene",
    scene_goal: str = "A goal",
    choices: list[dict] | None = None,
    is_start: bool = False,
    is_end: bool = False,
    quality_notes: list[str] | None = None,
    state_updates: dict | None = None,
) -> dict:
    """Build a minimal valid node dict."""
    node: dict = {
        "id": nid,
        "type": type,
        "scene_goal": scene_goal,
        "choices": choices or [],
        "quality_notes": quality_notes if quality_notes is not None else ["ok"],
    }
    if is_start:
        node["is_start"] = True
        node["type"] = "start"
    if is_end:
        node["is_end"] = True
        node["type"] = "end"
    return node


def _valid_graph() -> dict:
    """A minimal well-formed graph that passes all checks."""
    return {
        "nodes": {
            "start": _node(
                "start", is_start=True, scene_goal="Intro",
                quality_notes=["good start"],
                choices=[
                    {"id": "c1", "label": "Go", "next_node_id": "mid",
                     "state_updates": {"visited_mid": True}},
                ],
            ),
            "mid": _node(
                "mid", scene_goal="Middle",
                quality_notes=["decent middle"],
                choices=[
                    {"id": "c2", "label": "Proceed", "next_node_id": "end"},
                ],
            ),
            "end": _node(
                "end", is_end=True, scene_goal="Finale",
                quality_notes=["satisfying end"],
                choices=[],
            ),
        },
    }


# ── happy path ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_graph_passes():
    """A well-formed graph should pass all validation."""
    result = await _svc().validate(_valid_graph())
    assert result["is_valid"] is True
    assert result["errors"] == []
    assert all(result["checks"].values()), result["checks"]


# ── Check 1 & 6: next_node_id existence / broken references ───────

@pytest.mark.asyncio
async def test_broken_reference():
    """Choice pointing to non-existent node should fail."""
    graph = _valid_graph()
    graph["nodes"]["start"]["choices"][0]["next_node_id"] = "nonexistent"
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("missing node" in e for e in result["errors"])
    assert result["checks"]["references_valid"] is False


@pytest.mark.asyncio
async def test_all_next_node_ids_exist():
    """All next_node_id references should resolve."""
    graph = _valid_graph()
    # Add an extra valid choice
    graph["nodes"]["start"]["choices"].append(
        {"id": "c1b", "label": "Alt", "next_node_id": "end",
         "state_updates": {}}
    )
    result = await _svc().validate(graph)
    assert result["is_valid"] is True
    assert result["checks"]["references_valid"] is True


# ── Check 2: exactly one start node ────────────────────────────────

@pytest.mark.asyncio
async def test_missing_start_node():
    """Graph without a start node should fail."""
    graph = _valid_graph()
    graph["nodes"]["start"]["is_start"] = False
    graph["nodes"]["start"]["type"] = "scene"
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("No start node" in e for e in result["errors"])
    assert result["checks"]["exactly_one_start"] is False


@pytest.mark.asyncio
async def test_multiple_start_nodes():
    """Graph with multiple start nodes should fail."""
    graph = _valid_graph()
    graph["nodes"]["mid"]["is_start"] = True
    graph["nodes"]["mid"]["type"] = "start"
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("Multiple start nodes" in e for e in result["errors"])
    assert result["checks"]["exactly_one_start"] is False


# ── Check 3: at least one end node ─────────────────────────────────

@pytest.mark.asyncio
async def test_no_end_node():
    """Graph without any end node should fail."""
    graph = _valid_graph()
    graph["nodes"]["end"]["is_end"] = False
    graph["nodes"]["end"]["type"] = "scene"
    graph["nodes"]["end"]["choices"] = [
        {"id": "c3", "label": "Loop", "next_node_id": "start"}
    ]
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("No end node" in e for e in result["errors"])
    assert result["checks"]["at_least_one_end"] is False


# ── Check 4: all ends reachable ────────────────────────────────────

@pytest.mark.asyncio
async def test_unreachable_end_node():
    """An end node not reachable from start should fail."""
    graph = _valid_graph()
    # Add a disconnected end node
    graph["nodes"]["end2"] = _node(
        "end2", is_end=True, scene_goal="Alt end",
        quality_notes=["alt"],
        choices=[],
    )
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("not reachable" in e and "end2" in e for e in result["errors"])
    assert result["checks"]["ends_reachable"] is False


@pytest.mark.asyncio
async def test_all_ends_reachable():
    """Multiple reachable end nodes should pass."""
    graph = _valid_graph()
    # Add a second path to a second end, keeping the original path through mid
    graph["nodes"]["start"]["choices"].append(
        {"id": "c1b", "label": "Path B", "next_node_id": "end2",
         "state_updates": {}}
    )
    graph["nodes"]["end2"] = _node(
        "end2", is_end=True, scene_goal="Alt end",
        quality_notes=["alt"],
        choices=[],
    )
    result = await _svc().validate(graph)
    assert result["is_valid"] is True, result["errors"]
    assert result["checks"]["ends_reachable"] is True


# ── Check 5: no unreachable mandatory nodes ───────────────────────

@pytest.mark.asyncio
async def test_unreachable_mandatory_node():
    """An unreachable non-optional node should fail."""
    graph = _valid_graph()
    graph["nodes"]["orphan"] = _node(
        "orphan", scene_goal="Orphaned",
        quality_notes=["orphan"],
        choices=[],
    )
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("Unreachable mandatory" in e and "orphan" in e
               for e in result["errors"])
    assert result["checks"]["no_unreachable_mandatory"] is False


@pytest.mark.asyncio
async def test_unreachable_optional_node_is_warning():
    """An unreachable optional node should not cause failure."""
    graph = _valid_graph()
    graph["nodes"]["bonus"] = _node(
        "bonus", scene_goal="Bonus",
        quality_notes=["bonus"],
        choices=[],
    )
    graph["nodes"]["bonus"]["optional"] = True
    result = await _svc().validate(graph)
    assert result["is_valid"] is True, result["errors"]
    assert result["checks"]["no_unreachable_mandatory"] is True


# ── Check 7: no duplicate IDs ──────────────────────────────────────

@pytest.mark.asyncio
async def test_duplicate_id_field_mismatch():
    """Node with id field different from its key should fail."""
    graph = _valid_graph()
    graph["nodes"]["mid"]["id"] = "wrong_id"
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("mismatched id" in e for e in result["errors"])
    assert result["checks"]["no_duplicate_ids"] is False


# ── Check 8: no empty choice labels ────────────────────────────────

@pytest.mark.asyncio
async def test_empty_choice_label():
    """Choice with empty label should fail."""
    graph = _valid_graph()
    graph["nodes"]["start"]["choices"][0]["label"] = ""
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("empty label" in e for e in result["errors"])
    assert result["checks"]["no_empty_labels"] is False


@pytest.mark.asyncio
async def test_whitespace_only_label():
    """Choice with whitespace-only label should fail."""
    graph = _valid_graph()
    graph["nodes"]["start"]["choices"][0]["label"] = "   "
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("empty label" in e for e in result["errors"])


# ── Check 9: no node without scene_goal ────────────────────────────

@pytest.mark.asyncio
async def test_missing_scene_goal():
    """Node without scene_goal should fail."""
    graph = _valid_graph()
    graph["nodes"]["mid"]["scene_goal"] = ""
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("no scene_goal" in e and "mid" in e
               for e in result["errors"])
    assert result["checks"]["all_have_scene_goal"] is False


@pytest.mark.asyncio
async def test_whitespace_only_scene_goal():
    """Node with whitespace-only scene_goal should fail."""
    graph = _valid_graph()
    graph["nodes"]["mid"]["scene_goal"] = "  "
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("no scene_goal" in e for e in result["errors"])


# ── Check 10: state updates valid JSON ─────────────────────────────

@pytest.mark.asyncio
async def test_non_dict_state_updates():
    """A non-dict state_updates should fail."""
    graph = _valid_graph()
    graph["nodes"]["start"]["choices"][0]["state_updates"] = "not a dict"
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("non-dict state_updates" in e for e in result["errors"])
    assert result["checks"]["state_updates_valid"] is False


@pytest.mark.asyncio
async def test_non_serialisable_state_updates():
    """A non-dict state_updates should fail the state_updates check.

    Note: truly non-JSON-serialisable values (like sets) in the graph
    are caught first by the graph_serialisable check.  The state_updates
    check specifically catches non-dict values.
    """
    graph = _valid_graph()
    # A list is JSON-serialisable but not a valid state_updates type
    graph["nodes"]["start"]["choices"][0]["state_updates"] = [1, 2, 3]
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("non-dict state_updates" in e for e in result["errors"])
    assert result["checks"]["state_updates_valid"] is False


@pytest.mark.asyncio
async def test_missing_state_updates_is_ok():
    """A choice without state_updates key should be fine."""
    graph = _valid_graph()
    del graph["nodes"]["start"]["choices"][0]["state_updates"]
    result = await _svc().validate(graph)
    assert result["is_valid"] is True, result["errors"]
    assert result["checks"]["state_updates_valid"] is True


# ── Check 11: graph serialisable ───────────────────────────────────

@pytest.mark.asyncio
async def test_non_serialisable_graph():
    """A graph with non-serialisable content should fail."""
    graph = _valid_graph()
    graph["nodes"]["start"]["_bad"] = {1, 2, 3}  # set is not JSON-serialisable
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("not JSON-serialisable" in e for e in result["errors"])
    assert result["checks"]["graph_serialisable"] is False


# ── Check 12: no cycles without progress ───────────────────────────

@pytest.mark.asyncio
async def test_stagnant_cycle_fails():
    """A cycle with no state_updates on any edge should fail."""
    graph = {
        "nodes": {
            "n1": _node(
                "n1", is_start=True, scene_goal="Start",
                quality_notes=["n1"],
                choices=[
                    {"id": "a", "label": "Loop", "next_node_id": "n2",
                     "state_updates": {}},
                ],
            ),
            "n2": _node(
                "n2", scene_goal="Loop",
                quality_notes=["n2"],
                choices=[
                    {"id": "b", "label": "Back", "next_node_id": "n1",
                     "state_updates": {}},
                ],
            ),
            "n3": _node(
                "n3", is_end=True, scene_goal="End",
                quality_notes=["n3"],
                choices=[],
            ),
        },
    }
    # n1 → n2 → n1 is a stagnant cycle (no state_updates on either edge)
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("Stagnant cycle" in e for e in result["errors"])
    assert result["checks"]["no_stagnant_cycles"] is False


@pytest.mark.asyncio
async def test_progress_cycle_passes():
    """A cycle where at least one edge carries state_updates is OK."""
    graph = {
        "nodes": {
            "n1": _node(
                "n1", is_start=True, scene_goal="Start",
                quality_notes=["n1"],
                choices=[
                    {"id": "a", "label": "Loop", "next_node_id": "n2",
                     "state_updates": {"loop_count": 1}},
                ],
            ),
            "n2": _node(
                "n2", scene_goal="Loop",
                quality_notes=["n2"],
                choices=[
                    {"id": "b", "label": "Back", "next_node_id": "n1",
                     "state_updates": {}},
                    {"id": "c", "label": "Exit", "next_node_id": "n3",
                     "state_updates": {}},
                ],
            ),
            "n3": _node(
                "n3", is_end=True, scene_goal="End",
                quality_notes=["n3"],
                choices=[],
            ),
        },
    }
    result = await _svc().validate(graph)
    assert result["is_valid"] is True, result["errors"]
    assert result["checks"]["no_stagnant_cycles"] is True


# ── Check 13: quality metadata ─────────────────────────────────────

@pytest.mark.asyncio
async def test_all_quality_notes_empty_fails():
    """If every node has quality_notes but all are empty, fail."""
    graph = _valid_graph()
    for n in graph["nodes"].values():
        n["quality_notes"] = []
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    assert any("quality_notes" in e for e in result["errors"])
    assert result["checks"]["quality_metadata_present"] is False


@pytest.mark.asyncio
async def test_missing_quality_notes_warns():
    """Nodes missing the quality_notes key should produce a warning."""
    graph = _valid_graph()
    del graph["nodes"]["mid"]["quality_notes"]
    result = await _svc().validate(graph)
    assert result["is_valid"] is True  # warning, not error
    assert any("quality_notes" in w for w in result["warnings"])


# ── edge cases ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_graph_fails():
    """A graph with no nodes should fail."""
    result = await _svc().validate({"nodes": {}})
    assert result["is_valid"] is False
    assert any("no nodes" in e.lower() for e in result["errors"])


@pytest.mark.asyncio
async def test_no_choices_on_start():
    """A start node with no choices leading anywhere is technically valid
    if it is also an end node, but otherwise the end won't be reachable."""
    graph = {
        "nodes": {
            "start": _node(
                "start", is_start=True, is_end=True,
                scene_goal="Start and end",
                quality_notes=["solo"],
                choices=[],
            ),
        },
    }
    result = await _svc().validate(graph)
    assert result["is_valid"] is True, result["errors"]


@pytest.mark.asyncio
async def test_check_keys_present():
    """The result should always contain a 'checks' dict with all keys."""
    result = await _svc().validate(_valid_graph())
    expected_keys = {
        "graph_serialisable",
        "has_nodes",
        "no_duplicate_ids",
        "references_valid",
        "exactly_one_start",
        "at_least_one_end",
        "no_empty_labels",
        "all_have_scene_goal",
        "state_updates_valid",
        "quality_metadata_present",
        "ends_reachable",
        "no_unreachable_mandatory",
        "no_stagnant_cycles",
    }
    assert expected_keys.issubset(set(result["checks"].keys())), (
        f"Missing checks: {expected_keys - set(result['checks'].keys())}"
    )


@pytest.mark.asyncio
async def test_start_node_id_top_level():
    """Graph with start_node_id at top level should be honoured."""
    graph = _valid_graph()
    # Remove is_start from the node, set top-level start_node_id
    graph["nodes"]["start"]["is_start"] = False
    graph["nodes"]["start"]["type"] = "scene"
    graph["start_node_id"] = "start"
    result = await _svc().validate(graph)
    assert result["is_valid"] is True, result["errors"]
    assert result["checks"]["exactly_one_start"] is True


@pytest.mark.asyncio
async def test_multiple_errors_reported():
    """Multiple independent errors should all be reported."""
    graph = {
        "nodes": {
            "n1": _node(
                "n1", scene_goal="",  # missing scene_goal
                quality_notes=["n1"],
                choices=[
                    {"id": "c1", "label": "", "next_node_id": "missing",
                     "state_updates": "bad"},  # empty label + broken ref + bad su
                ],
            ),
        },
    }
    result = await _svc().validate(graph)
    assert result["is_valid"] is False
    # Should have at least: no start, no end, missing scene_goal,
    # empty label, broken ref, non-dict state_updates
    assert len(result["errors"]) >= 5, result["errors"]
