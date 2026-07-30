"""Tests for node update/regeneration when parameters change (t_598dd355).

Tests cover:
  - trim_node_text: trimming excess sentences
  - extend_node_text: extending insufficient sentences
  - adjust_node_sentences: node-level sentence adjustment
  - enforce_graph_sentence_limits: graph-level sentence enforcement
  - preview_limit_adjustments: previewing changes before applying
  - apply_limit_adjustments: applying changes with different modes
  - POST /admin/draft/{id}/check-limits: endpoint test
  - POST /admin/draft/{id}/apply-limits: endpoint test (new version created)
  - POST /admin/draft/{id}/settings: settings card contains check-limits button
"""

from __future__ import annotations

import pytest

from app.story.limits import (
    count_sentences,
    trim_node_text,
    extend_node_text,
    adjust_node_sentences,
    enforce_graph_sentence_limits,
    preview_limit_adjustments,
    apply_limit_adjustments,
    find_violating_nodes,
)


# ── trim_node_text tests ──────────────────────────────────────────


class TestTrimNodeText:
    def test_trims_excess_sentences(self):
        text = "Erster Satz. Zweiter Satz. Dritter Satz. Vierter Satz."
        result = trim_node_text(text, 2)
        assert count_sentences(result) == 2
        assert result == "Erster Satz. Zweiter Satz."

    def test_no_trim_when_within_limit(self):
        text = "Ein Satz. Noch ein Satz."
        result = trim_node_text(text, 5)
        assert result == text

    def test_exact_limit(self):
        text = "A. B. C."
        result = trim_node_text(text, 3)
        assert count_sentences(result) == 3

    def test_empty_text(self):
        assert trim_node_text("", 3) == ""

    def test_trims_to_one(self):
        text = "First. Second. Third."
        result = trim_node_text(text, 1)
        assert count_sentences(result) == 1


# ── extend_node_text tests ────────────────────────────────────────


class TestExtendNodeText:
    def test_extends_with_placeholder_sentences(self):
        text = "Ein Satz."
        result = extend_node_text(text, 3)
        assert count_sentences(result) == 3

    def test_no_extend_when_meets_minimum(self):
        text = "A. B. C."
        result = extend_node_text(text, 3)
        assert result == text

    def test_extend_empty_text(self):
        result = extend_node_text("", 2, "Test Node")
        assert count_sentences(result) == 2

    def test_extend_uses_node_title(self):
        result = extend_node_text("", 1, "Der Anfang")
        assert "Der Anfang" in result

    def test_extend_multiple_needed(self):
        text = "Eins."
        result = extend_node_text(text, 5)
        assert count_sentences(result) == 5

    def test_extend_does_not_produce_teil_markers(self):
        """Placeholder sentences must never contain (Teil ...) markers."""
        from app.story.story_text_validator import validate_story_text_markers
        text = "Ein Satz."
        result = extend_node_text(text, 5)
        val = validate_story_text_markers(result)
        assert val.passed, f"Internal marker found in placeholder: {[m.matched_text for m in val.matches]}"

    def test_extend_empty_does_not_produce_teil_markers(self):
        """Empty-text extension must also be clean of internal markers."""
        from app.story.story_text_validator import validate_story_text_markers
        result = extend_node_text("", 4, "Test Node")
        val = validate_story_text_markers(result)
        assert val.passed, f"Internal marker found in placeholder: {[m.matched_text for m in val.matches]}"


# ── adjust_node_sentences tests ───────────────────────────────────


class TestAdjustNodeSentences:
    def test_trims_too_many(self):
        node = {
            "id": "n1",
            "scene_goal": "A. B. C. D. E.",
            "type": "scene",
        }
        result = adjust_node_sentences(node, 2, 3)
        assert count_sentences(result["scene_goal"]) == 3

    def test_extends_too_few(self):
        node = {
            "id": "n2",
            "scene_goal": "Nur ein Satz.",
            "type": "scene",
        }
        result = adjust_node_sentences(node, 3, 5)
        assert count_sentences(result["scene_goal"]) == 3

    def test_no_change_when_within_bounds(self):
        node = {
            "id": "n3",
            "scene_goal": "A. B. C.",
            "type": "scene",
        }
        result = adjust_node_sentences(node, 2, 5)
        assert result["scene_goal"] == node["scene_goal"]

    def test_end_node_unchanged(self):
        node = {
            "id": "n4",
            "scene_goal": "A. B. C. D. E.",
            "type": "end",
        }
        result = adjust_node_sentences(node, 2, 3)
        assert result["scene_goal"] == "A. B. C. D. E."

    def test_is_end_node_unchanged(self):
        node = {
            "id": "n5",
            "scene_goal": "A. B. C. D. E.",
            "is_end": True,
        }
        result = adjust_node_sentences(node, 2, 3)
        assert result["scene_goal"] == "A. B. C. D. E."

    def test_zero_sentence_text_unchanged(self):
        node = {
            "id": "n6",
            "scene_goal": "Text ohne Punkt",
            "type": "scene",
        }
        result = adjust_node_sentences(node, 3, 5)
        assert result["scene_goal"] == "Text ohne Punkt"


# ── enforce_graph_sentence_limits tests ───────────────────────────


class TestEnforceGraphSentenceLimits:
    def test_trims_all_nodes(self):
        graph = {
            "nodes": {
                "n1": {"scene_goal": "A. B. C. D.", "type": "scene"},
                "n2": {"scene_goal": "X. Y. Z.", "type": "scene"},
            }
        }
        result = enforce_graph_sentence_limits(graph, 1, 2)
        for nid, node in result["nodes"].items():
            assert count_sentences(node["scene_goal"]) <= 2

    def test_extends_all_nodes(self):
        graph = {
            "nodes": {
                "n1": {"scene_goal": "A.", "type": "scene"},
                "n2": {"scene_goal": "B.", "type": "scene"},
            }
        }
        result = enforce_graph_sentence_limits(graph, 3, 5)
        for nid, node in result["nodes"].items():
            assert count_sentences(node["scene_goal"]) >= 3

    def test_does_not_modify_original(self):
        graph = {
            "nodes": {
                "n1": {"scene_goal": "A. B. C. D.", "type": "scene"},
            }
        }
        original_text = graph["nodes"]["n1"]["scene_goal"]
        _ = enforce_graph_sentence_limits(graph, 1, 2)
        assert graph["nodes"]["n1"]["scene_goal"] == original_text


# ── preview_limit_adjustments tests ───────────────────────────────


class TestPreviewLimitAdjustments:
    def test_no_changes_needed(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C.",
                    "choices": [
                        {"id": "c1", "next_node_id": "n2"},
                        {"id": "c2", "next_node_id": "n2"},
                    ],
                    "type": "scene",
                },
                "n2": {
                    "scene_goal": "X. Y. Z.",
                    "choices": [],
                    "type": "end",
                },
            }
        }
        preview = preview_limit_adjustments(graph, 3, 5, 2, 4)
        assert preview["summary"]["nodes_to_change"] == 0
        assert len(preview["sentence_changes"]) == 0
        assert len(preview["connection_changes"]) == 0

    def test_detects_sentence_changes(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C. D. E. F.",
                    "choices": [
                        {"id": "c1", "next_node_id": "n2"},
                        {"id": "c2", "next_node_id": "n2"},
                    ],
                    "type": "scene",
                },
            }
        }
        preview = preview_limit_adjustments(graph, 2, 3, 2, 4)
        assert len(preview["sentence_changes"]) == 1
        assert preview["sentence_changes"][0]["node_id"] == "n1"
        assert preview["sentence_changes"][0]["action"] == "trimmed"
        assert preview["sentence_changes"][0]["old_count"] == 6
        assert preview["sentence_changes"][0]["new_count"] == 3

    def test_detects_connection_changes(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C.",
                    "choices": [
                        {"id": "c1", "next_node_id": "n2"},
                        {"id": "c2", "next_node_id": "n2"},
                        {"id": "c3", "next_node_id": "n2"},
                        {"id": "c4", "next_node_id": "n2"},
                        {"id": "c5", "next_node_id": "n2"},
                    ],
                    "type": "scene",
                },
            }
        }
        preview = preview_limit_adjustments(graph, 2, 5, 2, 3)
        assert len(preview["connection_changes"]) == 1
        assert preview["connection_changes"][0]["node_id"] == "n1"
        assert preview["connection_changes"][0]["action"] == "trimmed"
        assert preview["connection_changes"][0]["old_count"] == 5
        assert preview["connection_changes"][0]["new_count"] == 3

    def test_detects_both_sentence_and_connection(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B.",
                    "choices": [
                        {"id": "c1", "next_node_id": "n2"},
                    ],
                    "type": "scene",
                },
            }
        }
        preview = preview_limit_adjustments(graph, 3, 5, 2, 4)
        assert len(preview["sentence_changes"]) == 1
        assert preview["sentence_changes"][0]["action"] == "extended"
        assert len(preview["connection_changes"]) == 1
        assert preview["connection_changes"][0]["action"] == "extended"

    def test_summary_counts(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C. D.",
                    "choices": [{"id": "c1", "next_node_id": "n2"}],
                    "type": "scene",
                },
                "n2": {
                    "scene_goal": "X. Y.",
                    "choices": [{"id": "c2", "next_node_id": "n1"}],
                    "type": "scene",
                },
            }
        }
        preview = preview_limit_adjustments(graph, 3, 3, 2, 2)
        assert preview["summary"]["total_nodes"] == 2
        assert preview["summary"]["nodes_to_change"] == 2
        assert preview["summary"]["sentence_fixes"] == 2  # n1 trimmed, n2 extended


# ── apply_limit_adjustments tests ──────────────────────────────────


class TestApplyLimitAdjustments:
    def test_auto_mode_fixes_both(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C. D. E.",
                    "choices": [
                        {"id": "c1", "next_node_id": "n2"},
                        {"id": "c2", "next_node_id": "n2"},
                        {"id": "c3", "next_node_id": "n2"},
                        {"id": "c4", "next_node_id": "n2"},
                    ],
                    "type": "scene",
                },
            }
        }
        result = apply_limit_adjustments(graph, 2, 3, 2, 3, mode="auto")
        node = result["nodes"]["n1"]
        assert count_sentences(node["scene_goal"]) == 3
        assert len(node["choices"]) == 3

    def test_sentences_only_mode(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C. D. E.",
                    "choices": [
                        {"id": "c1", "next_node_id": "n2"},
                        {"id": "c2", "next_node_id": "n2"},
                        {"id": "c3", "next_node_id": "n2"},
                        {"id": "c4", "next_node_id": "n2"},
                    ],
                    "type": "scene",
                },
            }
        }
        result = apply_limit_adjustments(graph, 2, 3, 2, 3, mode="sentences_only")
        node = result["nodes"]["n1"]
        assert count_sentences(node["scene_goal"]) == 3
        assert len(node["choices"]) == 4  # connections unchanged

    def test_connections_only_mode(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C. D. E.",
                    "choices": [
                        {"id": "c1", "next_node_id": "n2"},
                        {"id": "c2", "next_node_id": "n2"},
                        {"id": "c3", "next_node_id": "n2"},
                        {"id": "c4", "next_node_id": "n2"},
                    ],
                    "type": "scene",
                },
            }
        }
        result = apply_limit_adjustments(graph, 2, 3, 2, 3, mode="connections_only")
        node = result["nodes"]["n1"]
        assert count_sentences(node["scene_goal"]) == 5  # text unchanged
        assert len(node["choices"]) == 3

    def test_all_nodes_comply_after_apply(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C. D. E. F.",
                    "choices": [{"id": "c1", "next_node_id": "n2"}],
                    "type": "scene",
                },
                "n2": {
                    "scene_goal": "X.",
                    "choices": [
                        {"id": "c2", "next_node_id": "n1"},
                        {"id": "c3", "next_node_id": "n1"},
                        {"id": "c4", "next_node_id": "n1"},
                        {"id": "c5", "next_node_id": "n1"},
                        {"id": "c6", "next_node_id": "n1"},
                    ],
                    "type": "scene",
                },
            }
        }
        result = apply_limit_adjustments(graph, 2, 4, 2, 3, mode="auto")
        violations = find_violating_nodes(result, 2, 4, 2, 3)
        assert len(violations) == 0

    def test_does_not_modify_original(self):
        graph = {
            "nodes": {
                "n1": {
                    "scene_goal": "A. B. C. D. E.",
                    "choices": [{"id": "c1", "next_node_id": "n2"}],
                    "type": "scene",
                },
            }
        }
        original_goal = graph["nodes"]["n1"]["scene_goal"]
        _ = apply_limit_adjustments(graph, 2, 3, 2, 3)
        assert graph["nodes"]["n1"]["scene_goal"] == original_goal


# ── Endpoint tests ────────────────────────────────────────────────


class TestLimitEndpoints:
    """Tests for POST /admin/draft/{id}/check-limits and /apply-limits."""

    @pytest.fixture
    async def admin_client(self):
        from app.admin_ui.app import admin_app
        from app.persistence.database import init_db, close_db
        await init_db()
        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    async def _create_draft(self, admin_client) -> str:
        resp = await admin_client.post(
            "/new",
            data={
                "title": "Limit Test Story",
                "genre": "mystery",
                "tone": "dark",
                "language": "de",
                "target_age": "16+",
                "node_count": "10",
                "ending_count": "2",
                "branching_level": "medium",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        location = resp.headers.get("location", "")
        return location.rsplit("/", 1)[-1]

    @pytest.mark.asyncio
    async def test_check_limits_returns_preview(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        payload = {
            "min_sentences_per_node": 3,
            "max_sentences_per_node": 8,
            "min_node_connections": 2,
            "max_node_connections": 4,
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/check-limits",
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "preview" in data
        assert "summary" in data["preview"]

    @pytest.mark.asyncio
    async def test_apply_limits_creates_new_version(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        payload = {
            "min_sentences_per_node": 3,
            "max_sentences_per_node": 8,
            "min_node_connections": 2,
            "max_node_connections": 4,
            "mode": "auto",
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/apply-limits",
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "version_id" in data
        assert "version_number" in data
        assert isinstance(data["remaining_violations"], list)

    @pytest.mark.asyncio
    async def test_apply_limits_with_mode_sentences_only(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        payload = {
            "min_sentences_per_node": 3,
            "max_sentences_per_node": 8,
            "min_node_connections": 2,
            "max_node_connections": 4,
            "mode": "sentences_only",
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/apply-limits",
            json=payload,
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_check_limits_404_nonexistent_draft(self, admin_client):
        payload = {
            "min_sentences_per_node": 3,
            "max_sentences_per_node": 8,
            "min_node_connections": 2,
            "max_node_connections": 4,
        }
        resp = await admin_client.post(
            "/draft/nonexistent-id/check-limits",
            json=payload,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_apply_limits_404_nonexistent_draft(self, admin_client):
        payload = {
            "min_sentences_per_node": 3,
            "max_sentences_per_node": 8,
            "min_node_connections": 2,
            "max_node_connections": 4,
        }
        resp = await admin_client.post(
            "/draft/nonexistent-id/apply-limits",
            json=payload,
        )
        assert resp.status_code == 404


class TestSettingsCardHasCheckLimitsButton:
    """Verify the settings card includes the 'Auf Nodes anwenden' button."""

    @pytest.fixture
    async def admin_client(self):
        from app.admin_ui.app import admin_app
        from app.persistence.database import init_db, close_db
        await init_db()
        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    @pytest.mark.asyncio
    async def test_detail_page_has_check_limits_button(self, admin_client):
        # Create a draft
        resp = await admin_client.post(
            "/new",
            data={
                "title": "UI Test",
                "genre": "mystery",
                "tone": "dark",
                "language": "de",
                "target_age": "16+",
                "node_count": "5",
                "ending_count": "1",
                "branching_level": "low",
            },
            follow_redirects=False,
        )
        draft_id = resp.headers.get("location", "").rsplit("/", 1)[-1]

        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        html = resp.text
        assert "btn-check-limits" in html
        assert "checkNodeLimits" in html
        assert "limit-modal-overlay" in html
        assert "applyNodeLimits" in html
