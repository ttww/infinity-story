"""Validation helpers for story node sentence and connection limits.

Used by the authoring agent and the scene generator to enforce the
configured min/max sentences per node and min/max outgoing connections.
"""

from __future__ import annotations

import re
from typing import Any


def count_sentences(text: str) -> int:
    """Count the number of sentences in *text*.

    A sentence ends with '.', '!', or '?' followed by whitespace, end of
    string, or a closing quote/paren after the punctuation.  Text without
    any sentence-ending punctuation is counted as 0 (not well-formed).

    >>> count_sentences("Hallo. Wie geht es dir? Mir geht es gut!")
    3
    >>> count_sentences("Ein Satz ohne Punkt")
    0
    >>> count_sentences("")
    0
    """
    if not text or not text.strip():
        return 0
    # Find all sentence-ending punctuation marks
    matches = re.findall(r'[.!?]+(?:["\')\]]?)(?:\s+|$)', text.strip())
    return len(matches)


def validate_node_sentences(
    node: dict[str, Any],
    min_sentences: int,
    max_sentences: int,
) -> bool:
    """Return True if the node's scene_goal sentence count is within bounds."""
    text = node.get("scene_goal", "")
    count = count_sentences(text)
    return min_sentences <= count <= max_sentences


def validate_node_connections(
    node: dict[str, Any],
    min_connections: int,
    max_connections: int,
) -> bool:
    """Return True if the node's outgoing choice count is within bounds.

    End nodes (is_end=True or type=="end") and auto-advance nodes
    (choices empty with next_node_id) are always valid regardless of
    min_connections — they are exempt by design.
    """
    is_end = node.get("is_end", False) or node.get("type") == "end"
    choices = node.get("choices", [])
    has_next = bool(node.get("next_node_id"))

    # End nodes: 0 choices is correct
    if is_end:
        return len(choices) == 0

    # Auto-advance nodes: 0 choices + next_node_id is valid
    if len(choices) == 0 and has_next:
        return True

    # Regular nodes: must be within [min, max]
    return min_connections <= len(choices) <= max_connections


def find_violating_nodes(
    graph: dict[str, Any],
    min_sentences: int,
    max_sentences: int,
    min_connections: int,
    max_connections: int,
) -> list[dict[str, str]]:
    """Return a list of violation records for nodes that break limits.

    Each record: {"node_id": ..., "violation": "sentences" | "connections", "detail": ...}
    """
    violations: list[dict[str, str]] = []
    nodes = graph.get("nodes", {})
    for nid, node in nodes.items():
        if not validate_node_sentences(node, min_sentences, max_sentences):
            count = count_sentences(node.get("scene_goal", ""))
            violations.append({
                "node_id": nid,
                "violation": "sentences",
                "detail": f"sentence_count={count}, required [{min_sentences},{max_sentences}]",
            })
        if not validate_node_connections(node, min_connections, max_connections):
            choices = node.get("choices", [])
            violations.append({
                "node_id": nid,
                "violation": "connections",
                "detail": f"choice_count={len(choices)}, required [{min_connections},{max_connections}]",
            })
    return violations


def adjust_node_connections(
    node: dict[str, Any],
    min_connections: int,
    max_connections: int,
    all_node_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Adjust a node's choices to be within [min, max] connections.

    If the node has too many choices, the extras are removed.
    If the node has too few (but >0), placeholder choices are added
    pointing to existing nodes (if all_node_ids is provided) or
    to "node_unknown" otherwise.

    End nodes and auto-advance nodes are returned unchanged.
    """
    import copy
    result = copy.deepcopy(node)

    is_end = result.get("is_end", False) or result.get("type") == "end"
    choices = result.get("choices", [])
    has_next = bool(result.get("next_node_id"))

    if is_end:
        return result
    if len(choices) == 0 and has_next:
        return result

    # Trim if too many
    if len(choices) > max_connections:
        result["choices"] = choices[:max_connections]
        return result

    # Pad if too few
    if len(choices) < min_connections and len(choices) > 0:
        # Only pad if there's at least 1 choice — otherwise the node
        # is likely an auto-advance or end node that was mislabeled
        target = all_node_ids[0] if all_node_ids else "node_unknown"
        while len(result["choices"]) < min_connections:
            idx = len(result["choices"])
            result["choices"].append({
                "id": f"auto_c{idx}",
                "label": f"Weiter {idx + 1}",
                "next_node_id": target,
            })

    return result


def enforce_graph_limits(
    graph: dict[str, Any],
    min_sentences: int,
    max_sentences: int,
    min_connections: int,
    max_connections: int,
) -> dict[str, Any]:
    """Post-process a generated graph: adjust connection counts on all nodes.

    Returns a new graph dict with corrected choice lists.  Sentence counts
    cannot be auto-fixed (they require LLM regeneration) — use
    find_violating_nodes to detect them and retry.
    """
    import copy
    result = copy.deepcopy(graph)
    nodes = result.get("nodes", {})
    node_ids = list(nodes.keys())

    for nid, node in nodes.items():
        nodes[nid] = adjust_node_connections(
            node, min_connections, max_connections, node_ids
        )

    return result


# ── Node text trimming / extending ────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split *text* into a list of sentence strings (with terminators)."""
    if not text or not text.strip():
        return []
    matches = list(re.finditer(r'[^.!?]*[.!?]+(?:["\')\]]?)?\s*', text.strip()))
    if not matches:
        return [text.strip()]
    return [m.group(0).strip() for m in matches if m.group(0).strip()]


def trim_node_text(text: str, max_sentences: int) -> str:
    """Trim *text* to at most *max_sentences* sentences."""
    if not text:
        return text
    sents = _split_sentences(text)
    if len(sents) <= max_sentences:
        return text
    return " ".join(sents[:max_sentences])


def extend_node_text(text: str, min_sentences: int, node_title: str = "") -> str:
    """Extend *text* so it has at least *min_sentences* sentences.

    If the text already meets the minimum, it is returned unchanged.
    Otherwise placeholder sentences are appended to reach the target count.
    """
    if not text:
        text = ""
    sents = _split_sentences(text)
    current = len(sents)
    if current >= min_sentences:
        return text

    title_ref = node_title or "Diese Szene"
    needed = min_sentences - current
    if current == 0:
        # No well-formed sentences at all — add a full set
        parts = [f"{title_ref} beginnt."]
        for i in range(needed - 1):
            parts.append(f"Die Handlung nimmt weiter ihren Lauf.")
        return " ".join(parts)

    parts = list(sents)
    for i in range(needed):
        parts.append(f"Weitere Details ergänzen den Verlauf der Szene.")
    return " ".join(parts)


def adjust_node_sentences(
    node: dict[str, Any],
    min_sentences: int,
    max_sentences: int,
) -> dict[str, Any]:
    """Return a copy of *node* with scene_goal text adjusted to fit bounds.

    - If the text has too many sentences, it is trimmed.
    - If the text has too few sentences (but >0), placeholder sentences
      are appended to reach the minimum.

    End nodes (is_end or type=="end") are returned unchanged.
    """
    import copy
    result = copy.deepcopy(node)

    is_end = result.get("is_end", False) or result.get("type") == "end"
    if is_end:
        return result

    text = result.get("scene_goal", "")
    count = count_sentences(text)

    if count == 0:
        # No well-formed text — nothing to trim/extend deterministically
        return result

    if count > max_sentences:
        result["scene_goal"] = trim_node_text(text, max_sentences)
    elif count < min_sentences:
        result["scene_goal"] = extend_node_text(text, min_sentences, result.get("title", ""))

    return result


def enforce_graph_sentence_limits(
    graph: dict[str, Any],
    min_sentences: int,
    max_sentences: int,
) -> dict[str, Any]:
    """Post-process a graph: trim/extend scene_goal text on all nodes.

    Returns a new graph dict with adjusted scene_goal fields.
    """
    import copy
    result = copy.deepcopy(graph)
    nodes = result.get("nodes", {})
    for nid, node in nodes.items():
        nodes[nid] = adjust_node_sentences(node, min_sentences, max_sentences)
    return result


def preview_limit_adjustments(
    graph: dict[str, Any],
    min_sentences: int,
    max_sentences: int,
    min_connections: int,
    max_connections: int,
) -> dict[str, Any]:
    """Preview what would change if limits are enforced on the graph.

    Returns a dict with:
      - "violations": list of nodes that currently violate limits
      - "sentence_changes": list of {node_id, old_count, new_count, action}
      - "connection_changes": list of {node_id, old_count, new_count, action}
      - "summary": {total_nodes, nodes_to_change, sentence_fixes, connection_fixes}
    """
    violations = find_violating_nodes(
        graph, min_sentences, max_sentences,
        min_connections, max_connections,
    )

    nodes = graph.get("nodes", {})
    node_ids = list(nodes.keys())

    sentence_changes: list[dict[str, Any]] = []
    for nid, node in nodes.items():
        old_text = node.get("scene_goal", "")
        old_count = count_sentences(old_text)
        adjusted = adjust_node_sentences(node, min_sentences, max_sentences)
        new_text = adjusted.get("scene_goal", "")
        new_count = count_sentences(new_text)
        if old_count != new_count:
            action = "trimmed" if new_count < old_count else "extended"
            sentence_changes.append({
                "node_id": nid,
                "old_count": old_count,
                "new_count": new_count,
                "action": action,
            })

    connection_changes: list[dict[str, Any]] = []
    for nid, node in nodes.items():
        old_count = len(node.get("choices", []) or [])
        adjusted = adjust_node_connections(node, min_connections, max_connections, node_ids)
        new_count = len(adjusted.get("choices", []) or [])
        if old_count != new_count:
            action = "trimmed" if new_count < old_count else "extended"
            connection_changes.append({
                "node_id": nid,
                "old_count": old_count,
                "new_count": new_count,
                "action": action,
            })

    return {
        "violations": violations,
        "sentence_changes": sentence_changes,
        "connection_changes": connection_changes,
        "summary": {
            "total_nodes": len(nodes),
            "nodes_to_change": len(set(
                [c["node_id"] for c in sentence_changes] +
                [c["node_id"] for c in connection_changes]
            )),
            "sentence_fixes": len(sentence_changes),
            "connection_fixes": len(connection_changes),
        },
    }


def apply_limit_adjustments(
    graph: dict[str, Any],
    min_sentences: int,
    max_sentences: int,
    min_connections: int,
    max_connections: int,
    mode: str = "auto",
) -> dict[str, Any]:
    """Apply limit adjustments to a graph.

    *mode* controls the strategy:
      - "auto": trim/extend text + adjust connections (no LLM)
      - "connections_only": only fix connection counts, leave text unchanged
      - "sentences_only": only fix sentence counts, leave connections unchanged

    Returns a new graph dict with all adjustments applied.
    """
    import copy
    result = copy.deepcopy(graph)
    nodes = result.get("nodes", {})
    node_ids = list(nodes.keys())

    if mode in ("auto", "sentences_only"):
        for nid, node in nodes.items():
            nodes[nid] = adjust_node_sentences(node, min_sentences, max_sentences)

    if mode in ("auto", "connections_only"):
        for nid, node in nodes.items():
            nodes[nid] = adjust_node_connections(
                nodes[nid], min_connections, max_connections, node_ids
            )

    return result
