"""Test scenario loading and helios demo scenario."""

import json
from pathlib import Path


def test_helios_scenario_file_exists():
    """The helios.json demo scenario should exist and be valid JSON."""
    scenario_path = Path(__file__).parent.parent / "app" / "story" / "scenarios" / "helios.json"
    assert scenario_path.exists(), f"Scenario file not found: {scenario_path}"
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    assert data["id"] == "helios"
    assert data["title"] == "Signal von Helios"
    assert len(data["nodes"]) >= 5
    assert data["start_node_id"] == "node_001"


def test_helios_scenario_has_start_and_end():
    """The helios scenario must have at least one start and one end node."""
    scenario_path = Path(__file__).parent.parent / "app" / "story" / "scenarios" / "helios.json"
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    has_start = any(n.get("is_start") or n.get("type") == "start" for n in data["nodes"].values())
    has_end = any(n.get("is_end") or n.get("type") == "end" for n in data["nodes"].values())
    assert has_start, "No start node in helios scenario"
    assert has_end, "No end node in helios scenario"


def test_helios_choices_reference_valid_nodes():
    """All choice.next_node_id values should reference existing nodes."""
    scenario_path = Path(__file__).parent.parent / "app" / "story" / "scenarios" / "helios.json"
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    node_ids = set(data["nodes"].keys())
    for nid, node in data["nodes"].items():
        for choice in node.get("choices", []):
            next_id = choice.get("next_node_id")
            if next_id:
                assert next_id in node_ids, (
                    f"Node '{nid}' choice '{choice.get('id')}' "
                    f"references missing node '{next_id}'"
                )
