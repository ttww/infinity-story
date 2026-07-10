"""Publishing service (Spec §6.3, §7.7, §15).

Handles the transition from approved draft to published scenario.
Copies the approved graph into the ``published_scenarios`` table so
that the runtime can list and load it without touching authoring
tables.

Publish quality thresholds (Spec §15):
  - ``draft.quality_score >= 7.0``
  - latest ``validation.is_valid == True``
  - no high-severity issues in the latest review report
  - at least 15 nodes (configurable via ``min_node_count``)
  - at least 2 endings (configurable via ``min_ending_count``)
  - all end nodes reachable from the start node
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.enums import DraftStatus
from app.models.published_scenario import PublishedScenario
from app.persistence.authoring_repositories import (
    StoryDraftRepository,
    StoryDraftVersionRepository,
    StoryReviewReportRepository,
    StoryValidationReportRepository,
)
from app.services.story_validation_service import StoryValidationService

logger = logging.getLogger(__name__)


class PublishingService:
    """Publishes approved story drafts to the runtime scenario store."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._draft_repo = StoryDraftRepository(session)
        self._version_repo = StoryDraftVersionRepository(session)
        self._review_repo = StoryReviewReportRepository(session)
        self._validation_repo = StoryValidationReportRepository(session)

    # ── Quality thresholds (Spec §15) ───────────────────────────────

    async def can_publish(self, draft_id: str) -> tuple[bool, list[str]]:
        """Check whether a draft meets all publish criteria (Spec §15).

        Returns ``(True, [])`` if publishable, otherwise
        ``(False, [reasons])``.
        """
        settings = get_settings()
        reasons: list[str] = []

        draft = await self._draft_repo.get_by_id(draft_id)
        if draft is None:
            return False, [f"Draft {draft_id} not found."]

        if draft.status != DraftStatus.APPROVED.value:
            reasons.append(
                f"Draft must be in 'approved' status, got '{draft.status}'."
            )

        # Latest version
        latest_version = await self._version_repo.latest_for_draft(draft_id)
        if latest_version is None:
            reasons.append("Draft has no versions — nothing to publish.")
            return False, reasons

        graph = self._version_repo.parse_graph(latest_version)
        nodes = graph.get("nodes", {})

        # ── Minimum node count ──────────────────────────────────────
        if len(nodes) < settings.min_node_count:
            reasons.append(
                f"Graph has {len(nodes)} nodes, minimum is {settings.min_node_count}."
            )

        # ── Minimum ending count ────────────────────────────────────
        end_nodes = [
            nid for nid, n in nodes.items()
            if isinstance(n, dict) and (
                n.get("is_end") or n.get("type") == "end"
            )
        ]
        if len(end_nodes) < settings.min_ending_count:
            reasons.append(
                f"Graph has {len(end_nodes)} endings, minimum is {settings.min_ending_count}."
            )

        # ── Quality score check ─────────────────────────────────────
        # Spec §15: quality_score >= 7.0.  A draft without a quality
        # score has not been reviewed and must not be published.
        if draft.quality_score is None:
            reasons.append(
                "Draft has no quality_score — run a critic review first."
            )
        elif draft.quality_score < settings.min_quality_score:
            reasons.append(
                f"Quality score {draft.quality_score} below threshold {settings.min_quality_score}."
            )

        # ── Latest validation must exist and be valid ───────────────
        latest_validation = await self._validation_repo.latest_for_draft(draft_id)
        if latest_validation is None:
            reasons.append(
                "No validation report found — run deterministic validation first."
            )
        elif not latest_validation.is_valid:
            reasons.append("Latest validation report has errors — graph is invalid.")

        # ── Latest review must not have high-severity issues ────────
        review_reports = await self._review_repo.list_by_draft(draft_id)
        if not review_reports:
            reasons.append(
                "No review report found — run a critic review first."
            )
        else:
            latest_review = review_reports[-1]
            issues = self._review_repo.parse_issues(latest_review)
            high_issues = [
                i for i in issues
                if isinstance(i, dict) and i.get("severity") == "high"
            ]
            if high_issues:
                reasons.append(
                    f"Latest review has {len(high_issues)} high-severity issues."
                )

        # ── All end nodes must be reachable from start ──────────────
        if end_nodes:
            start_nodes = StoryValidationService._find_start_nodes(nodes, graph)
            if start_nodes:
                reachable = StoryValidationService._bfs_reachable(
                    nodes, start_nodes[0],
                )
                unreachable_ends = [
                    en for en in end_nodes if en not in reachable
                ]
                if unreachable_ends:
                    reasons.append(
                        f"End nodes not reachable from start: {unreachable_ends}."
                    )
            else:
                reasons.append("No start node found — cannot verify reachability.")

        return len(reasons) == 0, reasons

    # ── Publish ─────────────────────────────────────────────────────

    async def publish(self, draft_id: str) -> dict[str, Any]:
        """Copy the approved graph into the published_scenarios table.

        Returns a summary dict with the draft_id, scenario_id, and
        the number of nodes published.
        """
        can, reasons = await self.can_publish(draft_id)
        if not can:
            raise ValueError(
                f"Cannot publish draft {draft_id}: {'; '.join(reasons)}"
            )

        draft = await self._draft_repo.get_by_id(draft_id)
        assert draft is not None  # checked in can_publish

        latest_version = await self._version_repo.latest_for_draft(draft_id)
        assert latest_version is not None

        graph = self._version_repo.parse_graph(latest_version)
        nodes = graph.get("nodes", {})

        scenario_id = f"published_{draft_id}"
        published_count = len(nodes)

        # Delete any previously published scenario for this draft
        existing = await self.session.execute(
            select(PublishedScenario).where(
                PublishedScenario.id == scenario_id
            )
        )
        for old in existing.scalars().all():
            await self.session.delete(old)

        # Create the published scenario row
        scenario = PublishedScenario(
            id=scenario_id,
            draft_id=draft_id,
            title=draft.title,
            genre=draft.genre,
            graph_json=json.dumps(graph, ensure_ascii=False),
        )
        self.session.add(scenario)

        await self.session.commit()

        # Transition draft to PUBLISHED
        await self._draft_repo.update_status(
            draft_id, DraftStatus.PUBLISHED
        )

        logger.info(
            "Published draft %s as scenario %s (%d nodes)",
            draft_id, scenario_id, published_count,
        )

        return {
            "draft_id": draft_id,
            "scenario_id": scenario_id,
            "version_id": latest_version.id,
            "nodes_published": published_count,
        }
