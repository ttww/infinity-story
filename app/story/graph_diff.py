"""Graph diff computation for incremental story editing (Spec §8.1.6).

Compares two graph versions and produces a structured diff showing:
- Added nodes
- Removed nodes
- Modified nodes (field-level changes)
- Added/removed/modified choices within nodes
- Changed start_node_id
- Changed top-level fields (title, genre, tone)
"""

from __future__ import annotations

from typing import Any


# Fields that are compared at the node level
_NODE_FIELDS = (
    "title", "type", "act", "scene_goal", "scene_text", "location",
    "characters", "mood", "known_facts", "reveals",
    "quality_notes", "state_updates",
)

# Fields that are compared at the choice level
_CHOICE_FIELDS = ("id", "label", "next_node_id", "state_updates", "rationale")


def compute_graph_diff(
    old_graph: dict[str, Any],
    new_graph: dict[str, Any],
) -> dict[str, Any]:
    """Compute a structured diff between two graph versions.

    Returns a dict with keys:
        - added_nodes: list of node ids present only in new
        - removed_nodes: list of node ids present only in old
        - modified_nodes: list of {node_id, changes: [{field, old, new}]}
        - added_choices: list of {node_id, choice}
        - removed_choices: list of {node_id, choice_id}
        - modified_choices: list of {node_id, choice_id, changes: [{field, old, new}]}
        - graph_changes: list of {field, old, new} for top-level fields
        - summary: human-readable summary string
    """
    old_nodes = old_graph.get("nodes", {})
    new_nodes = new_graph.get("nodes", {})
    old_ids = set(old_nodes.keys())
    new_ids = set(new_nodes.keys())

    added_nodes = sorted(new_ids - old_ids)
    removed_nodes = sorted(old_ids - new_ids)
    common_nodes = old_ids & new_ids

    # ── Node-level modifications ──
    modified_nodes = []
    for nid in sorted(common_nodes):
        old_node = old_nodes[nid] or {}
        new_node = new_nodes[nid] or {}
        changes = _diff_fields(old_node, new_node, _NODE_FIELDS)
        if changes:
            modified_nodes.append({"node_id": nid, "changes": changes})

    # ── Choice-level modifications ──
    added_choices = []
    removed_choices = []
    modified_choices = []

    for nid in sorted(common_nodes):
        old_node = old_nodes[nid] or {}
        new_node = new_nodes[nid] or {}
        old_choices = {c.get("id", f"_{i}"): c for i, c in enumerate(old_node.get("choices", []) or [])}
        new_choices = {c.get("id", f"_{i}"): c for i, c in enumerate(new_node.get("choices", []) or [])}

        for cid in sorted(set(new_choices) - set(old_choices)):
            added_choices.append({"node_id": nid, "choice": new_choices[cid]})

        for cid in sorted(set(old_choices) - set(new_choices)):
            removed_choices.append({"node_id": nid, "choice_id": cid})

        for cid in sorted(set(old_choices) & set(new_choices)):
            changes = _diff_fields(old_choices[cid], new_choices[cid], _CHOICE_FIELDS)
            if changes:
                modified_choices.append({
                    "node_id": nid,
                    "choice_id": cid,
                    "changes": changes,
                })

    # ── Top-level graph fields ──
    _GRAPH_FIELDS = ("title", "genre", "tone", "start_node_id")
    graph_changes = _diff_fields(old_graph, new_graph, _GRAPH_FIELDS)

    # ── Summary ──
    parts = []
    if added_nodes:
        parts.append(f"+{len(added_nodes)} nodes")
    if removed_nodes:
        parts.append(f"-{len(removed_nodes)} nodes")
    if modified_nodes:
        parts.append(f"~{len(modified_nodes)} nodes")
    if added_choices:
        parts.append(f"+{len(added_choices)} choices")
    if removed_choices:
        parts.append(f"-{len(removed_choices)} choices")
    if modified_choices:
        parts.append(f"~{len(modified_choices)} choices")
    if graph_changes:
        parts.append(f"~{len(graph_changes)} graph fields")
    summary = ", ".join(parts) if parts else "no changes"

    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "modified_nodes": modified_nodes,
        "added_choices": added_choices,
        "removed_choices": removed_choices,
        "modified_choices": modified_choices,
        "graph_changes": graph_changes,
        "summary": summary,
    }


def _diff_fields(
    old: dict[str, Any],
    new: dict[str, Any],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Compare specified fields between old and new dicts."""
    changes = []
    for field in fields:
        old_val = old.get(field)
        new_val = new.get(field)
        if old_val != new_val:
            changes.append({
                "field": field,
                "old": old_val,
                "new": new_val,
            })
    return changes
