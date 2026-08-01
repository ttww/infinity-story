"""Deterministic Story Validation Service (Spec §7.6, §14.4).

NO LLM — pure deterministic checks on graph structure.

All checks defined in Spec §7.6:

 1.  All ``next_node_id`` references point to existing nodes.
 2.  Exactly one start node.
 3.  At least one end node.
 4.  All end nodes are reachable from the start node.
 5.  No unreachable mandatory (non-optional) nodes.
 6.  No broken references (dangling ``next_node_id``).
 7.  No duplicate node IDs.
 8.  No empty choice labels.
 9.  No node without a ``scene_goal``.
 10. State updates in choices are valid JSON.
 11. Graph is JSON-serialisable.
 12. No cycles without progress (every cycle must advance at least
     one state variable or flag).
 13. Quality metadata (``quality_notes``) present on each node.

The service accepts a ``graph`` dict (as stored in
``StoryDraftVersion.graph_json``) and returns::

    {
        "is_valid": bool,
        "errors":   list[str],
        "warnings": list[str],
        "checks":   dict[str, bool],   # per-check pass/fail
    }
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


def _describe_unreachable_pieces(
    nodes: dict[str, Any], unreachable: list[str],
) -> str:
    """Build a human-readable description of disconnected subgraphs."""
    _ = nodes  # unused, kept for API compatibility
    return (f"Die Knoten {unreachable} sind nicht vom Startknoten aus "
            f"erreichbar. Das deutet auf einen fehlenden Übergang (Choice) hin.")


class StoryValidationService:
    """Validates technical correctness of a story graph (Spec §7.6).

    Fully deterministic — no LLM calls.  Each check is independent
    so that a single failure does not mask other issues.
    """

    async def validate(self, graph: dict[str, Any]) -> dict[str, Any]:
        """Run all deterministic checks.

        Parameters
        ----------
        graph
            The story graph dict.  Expected shape::

                {
                    "nodes": {
                        "<node_id>": {
                            "id": str,
                            "type": str,           # "start"|"scene"|"end"
                            "is_start": bool,
                            "is_end": bool,
                            "scene_goal": str,
                            "choices": [
                                {
                                    "id": str,
                                    "label": str,
                                    "next_node_id": str | None,
                                    "state_updates": dict[str, Any],
                                },
                                ...
                            ],
                            "quality_notes": list[str],
                        },
                        ...
                    },
                    "start_node_id": str | None,
                }

        Returns
        -------
        dict
            ``{"is_valid": bool, "errors": [...], "warnings": [...], "checks": {...}}``
        """
        errors: list[str] = []
        warnings: list[str] = []
        checks: dict[str, bool] = {}

        # ── 11. Graph is JSON-serialisable ──────────────────────────
        checks["graph_serialisable"] = self._check_serialisable(graph, errors)

        if not checks["graph_serialisable"]:
            # Cannot proceed meaningfully — the graph isn't even serialisable
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "checks": checks,
            }

        nodes = graph.get("nodes", {})
        if not nodes:
            errors.append("Graph has no nodes.")
            checks["has_nodes"] = False
            return {
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "checks": checks,
            }
        checks["has_nodes"] = True

        node_ids = set(nodes.keys())

        # ── 7. Duplicate IDs ────────────────────────────────────────
        # With a dict, keys are unique by construction.  But the *value*
        # of each node's "id" field might differ from its key, or the
        # graph may have arrived as a list.  We check that all node "id"
        # fields match their keys and are unique.
        checks["no_duplicate_ids"] = self._check_duplicate_ids(nodes, errors)

        # ── 1 & 6. Broken references + next_node_id existence ───────
        checks["references_valid"] = self._check_references(
            nodes, node_ids, errors,
        )

        # ── 14. Auto-advance nodes need next_node_id ───────────────
        checks["auto_advance_valid"] = self._check_auto_advance(
            nodes, errors,
        )

        # ── 2. Exactly one start node ───────────────────────────────
        start_nodes = self._find_start_nodes(nodes, graph)
        checks["exactly_one_start"] = self._check_single_start(
            start_nodes, errors, warnings,
        )

        # ── 3. At least one end node ────────────────────────────────
        end_nodes = self._find_end_nodes(nodes)
        checks["at_least_one_end"] = self._check_at_least_one_end(
            end_nodes, errors,
        )

        # ── 8. No empty choice labels ───────────────────────────────
        checks["no_empty_labels"] = self._check_choice_labels(
            nodes, errors,
        )

        # ── 9. No node without scene_goal ───────────────────────────
        checks["all_have_scene_goal"] = self._check_scene_goals(
            nodes, errors,
        )

        # ── 10. State updates are valid JSON ────────────────────────
        checks["state_updates_valid"] = self._check_state_updates(
            nodes, errors,
        )

        # ── 13. Quality metadata present ────────────────────────────
        checks["quality_metadata_present"] = self._check_quality_metadata(
            nodes, errors, warnings,
        )

        # ── 4 & 5. Reachability ─────────────────────────────────────
        reachable: set[str] = set()
        if start_nodes:
            reachable = self._bfs_reachable(nodes, start_nodes[0])
        checks["ends_reachable"] = self._check_ends_reachable(
            end_nodes, reachable, errors,
        )
        checks["no_unreachable_mandatory"] = self._check_unreachable_mandatory(
            nodes, reachable, errors, warnings,
        )
        # ── 15. No disconnected subgraphs ───────────────────────────
        checks["no_disconnected_subgraphs"] = self._check_disconnected_subgraphs(
            nodes, reachable, errors,
        )

        # ── 12. No cycles without progress ──────────────────────────
        checks["no_stagnant_cycles"] = self._check_stagnant_cycles(
            nodes, errors, warnings,
        )

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    # ── Check 11: JSON serialisable ────────────────────────────────

    @staticmethod
    def _check_serialisable(
        graph: dict[str, Any], errors: list[str],
    ) -> bool:
        """Verify the graph can be round-tripped through ``json``."""
        try:
            json.dumps(graph, ensure_ascii=False)
            return True
        except (TypeError, ValueError) as exc:
            errors.append(f"Graph is not JSON-serialisable: {exc}")
            return False

    # ── Check 7: Duplicate IDs ─────────────────────────────────────

    @staticmethod
    def _check_duplicate_ids(
        nodes: dict[str, Any], errors: list[str],
    ) -> bool:
        """Detect duplicate or mismatched node IDs."""
        seen_ids: set[str] = set()
        ok = True
        for key, node in nodes.items():
            nid = node.get("id", key) if isinstance(node, dict) else key
            # mismatch between dict key and id field
            if nid != key:
                errors.append(
                    f"Node key '{key}' has mismatched id field '{nid}'."
                )
                ok = False
            if nid in seen_ids:
                errors.append(f"Duplicate node id: '{nid}'.")
                ok = False
            seen_ids.add(nid)
        return ok

    # ── Check 1 & 6: References ────────────────────────────────────

    @staticmethod
    def _check_references(
        nodes: dict[str, Any],
        node_ids: set[str],
        errors: list[str],
    ) -> bool:
        """Ensure every ``next_node_id`` points to an existing node.

        Checks both choice-level and node-level next_node_id.
        """
        ok = True
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                errors.append(f"Node '{nid}' is not a dict.")
                ok = False
                continue
            for choice in node.get("choices", []) or []:
                if not isinstance(choice, dict):
                    errors.append(
                        f"Node '{nid}' has a choice that is not a dict."
                    )
                    ok = False
                    continue
                next_id = choice.get("next_node_id")
                if next_id is not None and next_id not in node_ids:
                    errors.append(
                        f"Node '{nid}' choice '{choice.get('id', '?')}' "
                        f"references missing node '{next_id}'."
                    )
                    ok = False
            # Node-level next_node_id (auto_advance nodes)
            node_next = node.get("next_node_id")
            if node_next is not None and node_next not in node_ids:
                errors.append(
                    f"Node '{nid}' has next_node_id '{node_next}' "
                    f"which references a missing node."
                )
                ok = False
        return ok

    # ── Check 14: Auto-advance validation ──────────────────────────

    @staticmethod
    def _check_auto_advance(
        nodes: dict[str, Any], errors: list[str],
    ) -> bool:
        """Validate auto_advance nodes (0 choices).

        A non-optional node with 0 choices must either:
        - Be an ending (is_end=True or type="end"), or
        - Have next_node_id pointing to an existing node.

        A node with 0 choices, no next_node_id, and is_end=False
        is flagged as an error: the author should either mark it
        as an ending (is_end=True) or provide a next_node_id.

        Optional nodes are skipped (they may be unreachable bonus content).
        """
        ok = True
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            # Skip optional nodes
            if node.get("optional", False):
                continue
            choices = node.get("choices", []) or []
            if len(choices) > 0:
                continue  # Not an auto_advance node
            is_end = node.get("is_end", False) or node.get("type") == "end"
            next_id = node.get("next_node_id")
            if not is_end and not next_id:
                errors.append(
                    f"Node '{nid}' has no choices, no next_node_id, "
                    f"and is not marked as an ending. Either set is_end=true "
                    f"or provide a next_node_id for auto-advancing."
                )
                ok = False
        return ok

    # ── Check 2: Start node ────────────────────────────────────────

    @staticmethod
    def _find_start_nodes(
        nodes: dict[str, Any], graph: dict[str, Any],
    ) -> list[str]:
        """Return node IDs that are marked as start."""
        start_nodes = [
            nid for nid, n in nodes.items()
            if isinstance(n, dict) and (
                n.get("type") == "start" or n.get("is_start", False)
            )
        ]
        # Also honour top-level start_node_id
        top_start = graph.get("start_node_id")
        if top_start and top_start not in start_nodes:
            start_nodes.append(top_start)
        return start_nodes

    @staticmethod
    def _check_single_start(
        start_nodes: list[str],
        errors: list[str],
        warnings: list[str],
    ) -> bool:
        """Ensure exactly one start node."""
        if len(start_nodes) == 0:
            errors.append("No start node found.")
            return False
        if len(start_nodes) > 1:
            errors.append(
                f"Multiple start nodes found: {start_nodes}. "
                f"Exactly one is required."
            )
            return False
        return True

    # ── Check 3: End nodes ─────────────────────────────────────────

    @staticmethod
    def _find_end_nodes(nodes: dict[str, Any]) -> list[str]:
        return [
            nid for nid, n in nodes.items()
            if isinstance(n, dict) and (
                n.get("type") == "end" or n.get("is_end", False)
            )
        ]

    @staticmethod
    def _check_at_least_one_end(
        end_nodes: list[str], errors: list[str],
    ) -> bool:
        if not end_nodes:
            errors.append("No end node found.")
            return False
        return True

    # ── Check 8: Choice labels ─────────────────────────────────────

    @staticmethod
    def _check_choice_labels(
        nodes: dict[str, Any], errors: list[str],
    ) -> bool:
        """No choice may have an empty or whitespace-only label."""
        ok = True
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            for choice in node.get("choices", []) or []:
                if not isinstance(choice, dict):
                    continue
                label = choice.get("label", "")
                if not label or not str(label).strip():
                    errors.append(
                        f"Node '{nid}' choice '{choice.get('id', '?')}' "
                        f"has an empty label."
                    )
                    ok = False
        return ok

    # ── Check 9: Scene goals ───────────────────────────────────────

    @staticmethod
    def _check_scene_goals(
        nodes: dict[str, Any], errors: list[str],
    ) -> bool:
        """Every node must have a non-empty ``scene_goal``."""
        ok = True
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            goal = node.get("scene_goal", "")
            if not goal or not str(goal).strip():
                errors.append(f"Node '{nid}' has no scene_goal.")
                ok = False
        return ok

    # ── Check 10: State updates ────────────────────────────────────

    @staticmethod
    def _check_state_updates(
        nodes: dict[str, Any], errors: list[str],
    ) -> bool:
        """Every choice's ``state_updates`` must be a valid JSON dict."""
        ok = True
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            for choice in node.get("choices", []) or []:
                if not isinstance(choice, dict):
                    continue
                su = choice.get("state_updates")
                if su is None:
                    continue  # missing is OK — defaults to {}
                if not isinstance(su, dict):
                    errors.append(
                        f"Node '{nid}' choice '{choice.get('id', '?')}' "
                        f"has non-dict state_updates."
                    )
                    ok = False
                    continue
                # Verify it's JSON-serialisable (covers sets, datetimes, etc.)
                try:
                    json.dumps(su)
                except (TypeError, ValueError) as exc:
                    errors.append(
                        f"Node '{nid}' choice '{choice.get('id', '?')}' "
                        f"has non-serialisable state_updates: {exc}"
                    )
                    ok = False
        return ok

    # ── Check 13: Quality metadata ─────────────────────────────────

    @staticmethod
    def _check_quality_metadata(
        nodes: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> bool:
        """Each node should carry ``quality_notes``.

        Missing ``quality_notes`` is a *warning*, not an error —
        but an entirely absent ``quality_notes`` key on every node
        is an error (quality metadata is required by the spec).
        """
        has_any = False
        missing: list[str] = []
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            qn = node.get("quality_notes")
            if qn is None:
                missing.append(nid)
            elif isinstance(qn, list) and len(qn) > 0:
                has_any = True
            elif isinstance(qn, str) and qn.strip():
                has_any = True
        if missing:
            warnings.append(
                f"Nodes without quality_notes: {missing}"
            )
        if not has_any and not missing:
            # Every node has the key but all are empty
            errors.append(
                "No node carries any quality_notes content."
            )
            return False
        return True

    # ── Check 4: Ends reachable ────────────────────────────────────

    @staticmethod
    def _check_ends_reachable(
        end_nodes: list[str],
        reachable: set[str],
        errors: list[str],
    ) -> bool:
        """All end nodes must be reachable from the start."""
        ok = True
        for en in end_nodes:
            if en not in reachable:
                errors.append(
                    f"End node '{en}' is not reachable from the start node."
                )
                ok = False
        return ok

    # ── Check 5: No unreachable mandatory nodes ────────────────────

    @staticmethod
    def _check_unreachable_mandatory(
        nodes: dict[str, Any],
        reachable: set[str],
        errors: list[str],
        warnings: list[str],
    ) -> bool:
        """Flag nodes that are unreachable and not marked optional."""
        ok = True
        for nid, node in nodes.items():
            if nid in reachable:
                continue
            if isinstance(node, dict) and node.get("optional", False):
                continue  # optional nodes are allowed to be unreachable
            errors.append(f"Unreachable mandatory node: '{nid}'.")
            ok = False
        return ok

    # ── Check 15: No disconnected subgraphs ───────────────────────

    @staticmethod
    def _check_disconnected_subgraphs(
        nodes: dict[str, Any],
        reachable: set[str],
        errors: list[str],
    ) -> bool:
        """Detect nodes that form a disconnected subgraph.

        Unlike check 5 (unreachable_mandatory), this check flags
        EVERY node not reachable from the start, including nodes
        marked as ``optional``.  A graph in two or more pieces is
        always a design problem — optional or not.
        """
        unreachable = [nid for nid in nodes if nid not in reachable]
        if unreachable:
            piece_info = _describe_unreachable_pieces(nodes, unreachable)
            errors.append(
                f"Graph is broken into disconnected pieces. "
                f"Nodes not reachable from start: {unreachable}. "
                f"{piece_info}"
            )
            return False
        return True

    # ── Check 12: No cycles without progress ───────────────────────

    @staticmethod
    def _check_stagnant_cycles(
        nodes: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> bool:
        """Detect cycles that do not advance any state variable.

        A cycle is "stagnant" if you can loop through it without any
        choice on the cycle path carrying a non-empty ``state_updates``
        dict.  We detect this via DFS: for each back-edge discovered,
        we check whether at least one edge on the cycle path carries
        state updates.  If none do, it's an error.
        """
        # Build adjacency list
        adj: dict[str, list[tuple[str, str, dict[str, Any]]]] = (
            defaultdict(list)
        )
        for nid, node in nodes.items():
            if not isinstance(node, dict):
                continue
            for choice in node.get("choices", []) or []:
                if not isinstance(choice, dict):
                    continue
                next_id = choice.get("next_node_id")
                if next_id is None:
                    continue
                su = choice.get("state_updates") or {}
                adj[nid].append((
                    next_id,
                    choice.get("id", "?"),
                    su if isinstance(su, dict) else {},
                ))
            # Also follow node-level next_node_id (auto_advance nodes)
            node_next = node.get("next_node_id")
            if node_next is not None:
                su = node.get("state_updates") or {}
                adj[nid].append((
                    node_next,
                    "__auto_advance__",
                    su if isinstance(su, dict) else {},
                ))

        ok = True
        visited: set[str] = set()
        stack: set[str] = set()
        # path_nodes and path_edges track the current DFS path
        path_nodes: list[str] = []
        path_edges: list[list[tuple[str, str, dict[str, Any]]]] = []

        def _dfs(u: str) -> None:
            nonlocal ok
            visited.add(u)
            stack.add(u)
            path_nodes.append(u)
            path_edges.append([])

            for (v, cid, su) in adj.get(u, []):
                path_edges[-1].append((v, cid, su))
                if v in stack:
                    # Found a back-edge u→v — cycle detected
                    # Find the cycle path from v to u
                    cycle_start_idx = path_nodes.index(v)
                    cycle_nodes = path_nodes[cycle_start_idx:]
                    # Collect all edges along the cycle
                    cycle_edges: list[tuple[str, str, dict[str, Any]]] = []
                    for i in range(cycle_start_idx, len(path_nodes)):
                        # edges from path_nodes[i] that lead to next in cycle
                        target = (
                            cycle_nodes[i - cycle_start_idx + 1]
                            if i + 1 < len(path_nodes)
                            else v  # last node's edge back to v
                        )
                        for edge in path_edges[i]:
                            if edge[0] == target:
                                cycle_edges.append(edge)
                                break
                    # Check if any edge on the cycle has state updates
                    has_progress = any(
                        len(edge[2]) > 0 for edge in cycle_edges
                    )
                    if not has_progress:
                        errors.append(
                            f"Stagnant cycle detected: "
                            f"{' → '.join(cycle_nodes)} → {v}. "
                            f"No choice on this cycle carries "
                            f"state_updates."
                        )
                        ok = False
                elif v not in visited:
                    _dfs(v)

                # Remove the edge we just added from path
                path_edges[-1].pop()

            path_nodes.pop()
            path_edges.pop()
            stack.discard(u)

        for nid in nodes:
            if nid not in visited:
                _dfs(nid)

        return ok

    # ── BFS reachability ───────────────────────────────────────────

    @staticmethod
    def _bfs_reachable(
        nodes: dict[str, Any], start_id: str,
    ) -> set[str]:
        """Return the set of node IDs reachable from *start_id*."""
        visited: set[str] = set()
        queue: list[str] = [start_id]
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            node = nodes.get(nid, {})
            if not isinstance(node, dict):
                continue
            # Follow choice-level edges
            for choice in node.get("choices", []) or []:
                if not isinstance(choice, dict):
                    continue
                next_id = choice.get("next_node_id")
                if next_id and next_id not in visited:
                    queue.append(next_id)
            # Follow node-level next_node_id (auto_advance nodes)
            node_next = node.get("next_node_id")
            if node_next and node_next not in visited:
                queue.append(node_next)
        return visited
