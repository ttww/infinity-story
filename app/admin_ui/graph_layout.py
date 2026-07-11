"""Graph layout computation for SVG-based story graph visualization.

Computes node positions using a layered hierarchical layout (Sugiyama-style
simplified) that arranges nodes by their act/depth from the start node.
Edges are drawn as bezier curves between nodes.

Spec §8.1.3: Graph Visualization with start/end nodes marked,
problematic nodes highlighted, edges, zoom/pan, click for details.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any


# Layout constants
NODE_WIDTH = 180
NODE_HEIGHT = 60
HORIZONTAL_GAP = 60
VERTICAL_GAP = 120
LAYER_GAP = 220
TOP_PADDING = 40
LEFT_PADDING = 40
MIN_SVG_WIDTH = 800
MIN_SVG_HEIGHT = 400


def compute_layout(graph_data: dict[str, Any]) -> dict[str, Any]:
    """Compute SVG layout positions for all nodes in the graph.

    Returns a dict with:
        - nodes: list of {id, title, type, x, y, width, height, act, ...}
        - edges: list of {from, to, label, x1, y1, x2, y2, cx1, cy1, cx2, cy2}
        - svg_width, svg_height: canvas dimensions
        - start_node_id: the start node id
        - end_node_ids: list of end node ids
        - problematic_node_ids: nodes flagged by review/validation issues
    """
    nodes_data = graph_data.get("nodes", {})
    if not nodes_data:
        return {
            "nodes": [],
            "edges": [],
            "svg_width": MIN_SVG_WIDTH,
            "svg_height": MIN_SVG_HEIGHT,
            "start_node_id": graph_data.get("start_node_id"),
            "end_node_ids": [],
            "problematic_node_ids": [],
        }

    # ── 1. Build adjacency list and compute layers (BFS from start) ──
    start_node_id = graph_data.get("start_node_id")
    if not start_node_id:
        # Find by type
        for nid, node in nodes_data.items():
            if isinstance(node, dict) and (node.get("type") == "start" or node.get("is_start")):
                start_node_id = nid
                break
    if not start_node_id and nodes_data:
        start_node_id = next(iter(nodes_data))

    # BFS to assign layers (depth from start)
    layers: dict[str, int] = {}
    if start_node_id and start_node_id in nodes_data:
        queue = deque([(start_node_id, 0)])
        layers[start_node_id] = 0
        while queue:
            nid, depth = queue.popleft()
            node = nodes_data.get(nid, {})
            for choice in node.get("choices", []) or []:
                next_id = choice.get("next_node_id")
                if next_id and next_id in nodes_data and next_id not in layers:
                    # Don't override if already assigned a shorter path
                    layers[next_id] = depth + 1
                    queue.append((next_id, depth + 1))

    # Any nodes not reachable get put in a deep layer
    max_layer = max(layers.values()) if layers else 0
    for nid in nodes_data:
        if nid not in layers:
            max_layer += 1
            layers[nid] = max_layer

    # Also use 'act' to refine: group by act if layers are ambiguous
    # Nodes in the same layer get positioned vertically
    layer_nodes: dict[int, list[str]] = defaultdict(list)
    for nid, layer in layers.items():
        layer_nodes[layer].append(nid)

    # Sort nodes within layers by act then title for deterministic layout
    for layer in layer_nodes:
        layer_nodes[layer].sort(
            key=lambda nid: (
                nodes_data[nid].get("act", 1) if isinstance(nodes_data[nid], dict) else 1,
                nodes_data[nid].get("title", "") if isinstance(nodes_data[nid], dict) else "",
            )
        )

    # ── 2. Assign x,y positions ──
    node_positions: dict[str, dict[str, float]] = {}
    max_x = 0.0
    max_y = 0.0

    for layer_idx in sorted(layer_nodes.keys()):
        nodes_in_layer = layer_nodes[layer_idx]
        x = LEFT_PADDING + layer_idx * LAYER_GAP
        for i, nid in enumerate(nodes_in_layer):
            y = TOP_PADDING + i * VERTICAL_GAP
            node_positions[nid] = {"x": x, "y": y}
            max_x = max(max_x, x + NODE_WIDTH)
            max_y = max(max_y, y + NODE_HEIGHT)

    # ── 3. Identify start, end, and problematic nodes ──
    end_node_ids = []
    for nid, node in nodes_data.items():
        if not isinstance(node, dict):
            continue
        if node.get("type") in ("end", "ending") or node.get("is_end"):
            end_node_ids.append(nid)

    # Problematic nodes: those with empty scene_goal, empty quality_notes,
    # dangling references, or referenced in review issues
    problematic_node_ids = _identify_problematic_nodes(nodes_data)

    # ── 4. Build node list for SVG ──
    svg_nodes = []
    for nid, node in nodes_data.items():
        if not isinstance(node, dict):
            continue
        pos = node_positions.get(nid, {"x": 0, "y": 0})
        node_type = node.get("type", "scene")
        is_start = nid == start_node_id or node.get("is_start", False) or node_type == "start"
        is_end = node.get("is_end", False) or node_type in ("end", "ending")
        is_problematic = nid in problematic_node_ids

        svg_nodes.append({
            "id": nid,
            "title": node.get("title", nid),
            "type": node_type,
            "act": node.get("act", 1),
            "x": pos["x"],
            "y": pos["y"],
            "width": NODE_WIDTH,
            "height": NODE_HEIGHT,
            "is_start": is_start,
            "is_end": is_end,
            "is_problematic": is_problematic,
            "choice_count": len(node.get("choices", []) or []),
            "location": node.get("location", ""),
        })

    # ── 5. Build edge list for SVG ──
    svg_edges = []
    for nid, node in nodes_data.items():
        if not isinstance(node, dict):
            continue
        from_pos = node_positions.get(nid)
        if not from_pos:
            continue
        for choice in node.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            next_id = choice.get("next_node_id")
            if not next_id or next_id not in nodes_data:
                continue
            to_pos = node_positions.get(next_id)
            if not to_pos:
                continue

            x1 = from_pos["x"] + NODE_WIDTH  # right edge of source
            y1 = from_pos["y"] + NODE_HEIGHT / 2
            x2 = to_pos["x"]  # left edge of target
            y2 = to_pos["y"] + NODE_HEIGHT / 2

            # Bezier control points for smooth curves
            dx = x2 - x1
            cx1 = x1 + dx * 0.5
            cy1 = y1
            cx2 = x2 - dx * 0.5
            cy2 = y2

            svg_edges.append({
                "from": nid,
                "to": next_id,
                "label": choice.get("label", ""),
                "choice_id": choice.get("id", ""),
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "cx1": cx1, "cy1": cy1,
                "cx2": cx2, "cy2": cy2,
                "dangling": next_id not in nodes_data,
            })

    svg_width = max(max_x + 40, MIN_SVG_WIDTH)
    svg_height = max(max_y + 40, MIN_SVG_HEIGHT)

    return {
        "nodes": svg_nodes,
        "edges": svg_edges,
        "svg_width": svg_width,
        "svg_height": svg_height,
        "start_node_id": start_node_id,
        "end_node_ids": end_node_ids,
        "problematic_node_ids": list(problematic_node_ids),
    }


def _identify_problematic_nodes(nodes_data: dict[str, Any]) -> set[str]:
    """Identify nodes with technical issues (Spec §8.1.3: problematische Knoten).

    A node is problematic if it has:
    - Empty scene_goal
    - Empty quality_notes
    - Dangling choice references (next_node_id points to non-existent node)
    - Dangling node-level next_node_id (auto_advance pointing to missing node)
    """
    problematic = set()
    node_ids = set(nodes_data.keys())

    for nid, node in nodes_data.items():
        if not isinstance(node, dict):
            problematic.add(nid)
            continue

        # Empty scene_goal
        goal = node.get("scene_goal", "")
        if not goal or not str(goal).strip():
            problematic.add(nid)
            continue

        # Dangling references
        for choice in node.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            next_id = choice.get("next_node_id")
            if next_id and next_id not in node_ids:
                problematic.add(nid)
                break

        # Dangling node-level next_node_id (auto_advance nodes)
        node_next = node.get("next_node_id")
        if node_next and node_next not in node_ids:
            problematic.add(nid)
            continue

        # Empty label on choices
        for choice in node.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            label = choice.get("label", "")
            if not label or not str(label).strip():
                problematic.add(nid)
                break

    return problematic


def enrich_node_detail(
    node_id: str,
    node_data: dict[str, Any],
    review_issues: list[dict[str, Any]] | None = None,
    validation_errors: list[str] | None = None,
    validation_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a comprehensive node detail dict for the Node Detail Panel.

    Spec §8.1.4: Node Detail Panel shows:
    ID, Titel, Typ, Akt, Ort, Szenenziel, Figuren, Reveals,
    Choices, State-Updates, Kritiker-Kommentare, technische Warnings.
    """
    review_issues = review_issues or []
    validation_errors = validation_errors or []
    validation_warnings = validation_warnings or []

    # Filter review issues that mention this node
    node_review_issues = []
    for issue in review_issues:
        if not isinstance(issue, dict):
            continue
        issue_node = issue.get("node_id")
        if issue_node == node_id or issue_node == node_data.get("id"):
            node_review_issues.append(issue)

    # Filter validation errors/warnings that mention this node
    node_tech_warnings = []
    for err in validation_errors:
        if isinstance(err, str) and node_id in err:
            node_tech_warnings.append({"severity": "error", "message": err})
    for warn in validation_warnings:
        if isinstance(warn, str) and node_id in warn:
            node_tech_warnings.append({"severity": "warning", "message": warn})

    # Also check for structural issues on this node
    choices = node_data.get("choices", []) or []
    quality_notes = node_data.get("quality_notes", []) or []
    scene_goal = node_data.get("scene_goal", "")

    if not scene_goal or not str(scene_goal).strip():
        node_tech_warnings.append({"severity": "error", "message": "Missing scene_goal."})
    if not quality_notes:
        node_tech_warnings.append({"severity": "warning", "message": "No quality_notes defined."})

    # Check for dangling references
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        next_id = choice.get("next_node_id")
        label = choice.get("label", "")
        if not label or not str(label).strip():
            node_tech_warnings.append({
                "severity": "error",
                "message": f"Choice '{choice.get('id', '?')}' has empty label.",
            })

    # Build choice list with state updates
    choice_details = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        choice_details.append({
            "id": choice.get("id", ""),
            "label": choice.get("label", ""),
            "next_node_id": choice.get("next_node_id"),
            "state_updates": choice.get("state_updates", {}),
            "rationale": choice.get("rationale", ""),
        })

    # State updates on the node itself
    state_updates = node_data.get("state_updates", {})

    return {
        "id": node_data.get("id", node_id),
        "title": node_data.get("title", ""),
        "type": node_data.get("type", "scene"),
        "act": node_data.get("act", 1),
        "location": node_data.get("location", ""),
        "scene_goal": node_data.get("scene_goal", ""),
        "scene_text": node_data.get("scene_text", ""),
        "characters": node_data.get("characters", []),
        "mood": node_data.get("mood", ""),
        "known_facts": node_data.get("known_facts", []),
        "reveals": node_data.get("reveals", []),
        "choices": choice_details,
        "state_updates": state_updates,
        "quality_notes": quality_notes,
        "review_issues": node_review_issues,
        "tech_warnings": node_tech_warnings,
        "is_start": node_data.get("is_start", False) or node_data.get("type") == "start",
        "is_end": node_data.get("is_end", False) or node_data.get("type") in ("end", "ending"),
    }
