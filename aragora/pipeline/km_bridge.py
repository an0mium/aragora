"""KnowledgeMound bidirectional bridge for pipeline precedent queries."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PipelineKMBridge:
    """Queries KnowledgeMound for similar past goals/actions.

    Provides bidirectional integration between the pipeline and KM:
    - Forward: query KM for similar past goals and actions (precedents)
    - Backward: store completed pipeline results for future queries
    """

    def __init__(self, knowledge_mound: Any | None = None):
        self._km = knowledge_mound
        if self._km is None:
            try:
                from aragora.knowledge.mound.core import get_knowledge_mound

                self._km = get_knowledge_mound()
            except (ImportError, Exception):
                logger.debug("KnowledgeMound not available for pipeline bridge")

    @property
    def available(self) -> bool:
        """Whether the KnowledgeMound backend is available."""
        return self._km is not None

    def query_similar_goals(self, goal_graph: Any) -> dict[str, list[dict[str, Any]]]:
        """Query KM for similar past goals.

        Args:
            goal_graph: GoalGraph with goals to search for precedents

        Returns:
            Dict mapping goal IDs to lists of similar past goal dicts
        """
        if not self.available:
            return {}
        results: dict[str, list[dict[str, Any]]] = {}
        for goal in goal_graph.goals:
            try:
                matches = self._km.search(
                    query=goal.title,
                    limit=3,
                    min_similarity=0.5,
                )
                results[goal.id] = [
                    {
                        "title": getattr(m, "title", str(m)),
                        "similarity": getattr(m, "similarity", 0.0),
                        "outcome": getattr(m, "metadata", {}).get(
                            "outcome", "unknown"
                        ),
                    }
                    for m in (matches if matches else [])
                ]
            except (AttributeError, TypeError, Exception):
                results[goal.id] = []
        return results

    def query_similar_actions(
        self, actions_canvas: Any
    ) -> dict[str, list[dict[str, Any]]]:
        """Query KM for similar past action plans.

        Args:
            actions_canvas: Canvas with action nodes to search for precedents

        Returns:
            Dict mapping node IDs to lists of similar past action dicts
        """
        if not self.available:
            return {}
        results: dict[str, list[dict[str, Any]]] = {}
        for node_id, node in actions_canvas.nodes.items():
            try:
                matches = self._km.search(
                    query=node.label,
                    limit=3,
                    min_similarity=0.5,
                )
                results[node_id] = [
                    {
                        "title": getattr(m, "title", str(m)),
                        "similarity": getattr(m, "similarity", 0.0),
                        "outcome": getattr(m, "metadata", {}).get(
                            "outcome", "unknown"
                        ),
                    }
                    for m in (matches if matches else [])
                ]
            except (AttributeError, TypeError, Exception):
                results[node_id] = []
        return results

    def enrich_with_precedents(
        self, goal_graph: Any, precedents: dict[str, list[dict[str, Any]]]
    ) -> Any:
        """Add precedent data to goal metadata.

        Args:
            goal_graph: GoalGraph to enrich
            precedents: Dict mapping goal IDs to precedent lists

        Returns:
            The enriched goal_graph (modified in place)
        """
        for goal in goal_graph.goals:
            if goal.id in precedents and precedents[goal.id]:
                goal.metadata["precedents"] = precedents[goal.id]
        return goal_graph

    def store_pipeline_result(self, result: Any) -> bool:
        """Store completed pipeline in KM for future queries.

        Args:
            result: PipelineResult to store

        Returns:
            True if stored successfully, False otherwise
        """
        if not self.available:
            return False
        try:
            from aragora.knowledge.mound.adapters.decision_plan_adapter import (
                DecisionPlanAdapter,
            )

            adapter = DecisionPlanAdapter(self._km)
            adapter.store(result.to_dict())
            return True
        except (ImportError, AttributeError, Exception) as e:
            logger.debug("Failed to store pipeline result in KM: %s", e)
            return False
