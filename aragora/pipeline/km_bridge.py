"""KnowledgeMound bidirectional bridge for pipeline precedent queries.

Queries three KM adapter types for decision precedents:
- ReceiptAdapter: past decision receipts (verdict, confidence, findings)
- OutcomeAdapter: past decision outcomes (impact, lessons learned)
- DebateAdapter: past debate results (consensus, dissenting views)

These precedents are attached to goal metadata so downstream pipeline
stages can reference historical context when planning actions.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PipelineKMBridge:
    """Queries KnowledgeMound for similar past goals/actions.

    Provides bidirectional integration between the pipeline and KM:
    - Forward: query KM for similar past goals and actions (precedents)
    - Forward: query Receipt/Outcome/Debate adapters for decision precedents
    - Backward: store completed pipeline results for future queries
    """

    def __init__(self, knowledge_mound: Any | None = None):
        self._km = knowledge_mound
        if self._km is None:
            try:
                from aragora.knowledge.mound import core as mound_core

                factory = getattr(mound_core, "get_knowledge_mound", None)
                if callable(factory):
                    self._km = factory()
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
                        "outcome": getattr(m, "metadata", {}).get("outcome", "unknown"),
                    }
                    for m in (matches if matches else [])
                ]
            except (AttributeError, TypeError, Exception):
                results[goal.id] = []
        return results

    def query_similar_actions(self, actions_canvas: Any) -> dict[str, list[dict[str, Any]]]:
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
                        "outcome": getattr(m, "metadata", {}).get("outcome", "unknown"),
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

    def query_receipt_precedents(
        self, goal_description: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        """Query ReceiptAdapter for past decisions similar to the goal.

        Uses the singleton ReceiptAdapter to find related decision receipts.
        The adapter's ``find_related_decisions`` is async, so this method
        accesses the adapter's ingested receipt cache synchronously.

        Args:
            goal_description: Text description of the goal to match against
            limit: Maximum number of results

        Returns:
            List of precedent dicts with keys: source, receipt_id, summary,
            verdict, confidence.
        """
        try:
            from aragora.knowledge.mound.adapters.receipt_adapter import (
                get_receipt_adapter,
            )

            adapter = get_receipt_adapter()
            # Access the adapter's local ingestion cache for sync queries
            results: list[dict[str, Any]] = []
            goal_lower = goal_description.lower()
            for receipt_id, ingestion in adapter._ingested_receipts.items():
                # Check if any knowledge item content relates to the goal
                for item_id in ingestion.knowledge_item_ids:
                    if any(
                        word in item_id.lower() or word in receipt_id.lower()
                        for word in goal_lower.split()[:5]
                    ):
                        results.append(
                            {
                                "source": "receipt",
                                "receipt_id": receipt_id,
                                "summary": f"Decision receipt {receipt_id}",
                                "verdict": "unknown",
                                "confidence": 0.5,
                            }
                        )
                        break
                if len(results) >= limit:
                    break

            # Also check the mound if available (sync path via search)
            if self._km and len(results) < limit:
                try:
                    matches = self._km.search(
                        query=goal_description,
                        limit=limit - len(results),
                        min_similarity=0.4,
                    )
                    for m in matches or []:
                        meta = getattr(m, "metadata", {})
                        if meta.get("item_type") == "decision_summary" or "decision_receipt" in (
                            meta.get("tags") or []
                        ):
                            results.append(
                                {
                                    "source": "receipt",
                                    "receipt_id": meta.get("receipt_id", ""),
                                    "summary": getattr(m, "content", str(m))[:200],
                                    "verdict": meta.get("verdict", "unknown"),
                                    "confidence": meta.get("confidence", 0.0),
                                }
                            )
                except (AttributeError, TypeError, RuntimeError, ValueError):
                    pass

            return results[:limit]
        except (ImportError, RuntimeError, AttributeError) as exc:
            logger.debug("ReceiptAdapter query unavailable: %s", exc)
            return []

    def query_outcome_precedents(
        self, goal_description: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        """Query OutcomeAdapter for past decision outcomes similar to the goal.

        Uses the singleton OutcomeAdapter to find related outcomes.

        Args:
            goal_description: Text description of the goal to match against
            limit: Maximum number of results

        Returns:
            List of precedent dicts with keys: source, outcome_id,
            description, impact_score, lessons_learned.
        """
        try:
            from aragora.knowledge.mound.adapters.outcome_adapter import (
                get_outcome_adapter,
            )

            adapter = get_outcome_adapter()
            results: list[dict[str, Any]] = []

            # Search the mound for outcome items
            if self._km:
                try:
                    matches = self._km.search(
                        query=goal_description,
                        limit=limit,
                        min_similarity=0.4,
                    )
                    for m in matches or []:
                        meta = getattr(m, "metadata", {})
                        if meta.get("item_type") == "decision_outcome" or "decision_outcome" in (
                            meta.get("tags") or []
                        ):
                            results.append(
                                {
                                    "source": "outcome",
                                    "outcome_id": meta.get("outcome_id", ""),
                                    "description": getattr(m, "content", str(m))[:200],
                                    "impact_score": meta.get("impact_score", 0.0),
                                    "lessons_learned": meta.get("lessons_learned", ""),
                                }
                            )
                except (AttributeError, TypeError, RuntimeError, ValueError):
                    pass

            # Supplement from adapter's local cache
            if len(results) < limit:
                goal_lower = goal_description.lower()
                for outcome_id, ingestion in adapter._ingested_outcomes.items():
                    if any(word in outcome_id.lower() for word in goal_lower.split()[:5]):
                        results.append(
                            {
                                "source": "outcome",
                                "outcome_id": outcome_id,
                                "description": f"Past outcome {outcome_id}",
                                "impact_score": 0.5,
                                "lessons_learned": "",
                            }
                        )
                    if len(results) >= limit:
                        break

            return results[:limit]
        except (ImportError, RuntimeError, AttributeError) as exc:
            logger.debug("OutcomeAdapter query unavailable: %s", exc)
            return []

    def query_debate_precedents(
        self, goal_description: str, limit: int = 3
    ) -> list[dict[str, Any]]:
        """Query DebateAdapter for past debates similar to the goal.

        Uses a local DebateAdapter instance to search stored debate outcomes
        by topic. ``search_by_topic`` is async, so this method falls back to
        a synchronous text-match over the adapter's in-memory stores.

        Args:
            goal_description: Text description of the goal to match against
            limit: Maximum number of results

        Returns:
            List of precedent dicts with keys: source, debate_id, task,
            final_answer, confidence, consensus_reached.
        """
        try:
            from aragora.knowledge.mound.adapters.debate_adapter import (
                DebateAdapter,
            )

            adapter = DebateAdapter()
            results: list[dict[str, Any]] = []
            goal_lower = goal_description.lower()

            # Synchronous text-match over stored outcomes
            all_outcomes = list(adapter._synced_outcomes.values()) + adapter._pending_outcomes
            for outcome in all_outcomes:
                task_lower = outcome.task.lower()
                if goal_lower in task_lower or any(
                    word in task_lower for word in goal_lower.split()[:5]
                ):
                    results.append(
                        {
                            "source": "debate",
                            "debate_id": outcome.debate_id,
                            "task": outcome.task,
                            "final_answer": outcome.final_answer[:200],
                            "confidence": outcome.confidence,
                            "consensus_reached": outcome.consensus_reached,
                        }
                    )
                if len(results) >= limit:
                    break

            # Also search the mound for debate items
            if self._km and len(results) < limit:
                try:
                    matches = self._km.search(
                        query=goal_description,
                        limit=limit - len(results),
                        min_similarity=0.4,
                    )
                    for m in matches or []:
                        meta = getattr(m, "metadata", {})
                        if meta.get("task") and meta.get("consensus_reached") is not None:
                            results.append(
                                {
                                    "source": "debate",
                                    "debate_id": meta.get("debate_id", getattr(m, "source_id", "")),
                                    "task": meta.get("task", ""),
                                    "final_answer": getattr(m, "content", "")[:200],
                                    "confidence": meta.get("confidence", 0.0)
                                    if isinstance(meta.get("confidence"), (int, float))
                                    else 0.0,
                                    "consensus_reached": meta.get("consensus_reached", False),
                                }
                            )
                except (AttributeError, TypeError, RuntimeError, ValueError):
                    pass

            return results[:limit]
        except (ImportError, RuntimeError, AttributeError) as exc:
            logger.debug("DebateAdapter query unavailable: %s", exc)
            return []

    def query_all_adapter_precedents(
        self, goal_description: str, limit: int = 3
    ) -> dict[str, list[dict[str, Any]]]:
        """Query all three KM adapters for decision precedents.

        Convenience method that queries ReceiptAdapter, OutcomeAdapter, and
        DebateAdapter in sequence and returns a combined dict.

        Args:
            goal_description: Text description to search for
            limit: Max results per adapter type

        Returns:
            Dict with keys "receipts", "outcomes", "debates", each mapping
            to a list of precedent dicts.
        """
        return {
            "receipts": self.query_receipt_precedents(goal_description, limit=limit),
            "outcomes": self.query_outcome_precedents(goal_description, limit=limit),
            "debates": self.query_debate_precedents(goal_description, limit=limit),
        }

    def enrich_goals_with_adapter_precedents(self, goal_graph: Any) -> Any:
        """Enrich each goal in the graph with adapter-sourced precedents.

        For every goal, queries Receipt, Outcome, and Debate adapters and
        attaches matching precedents to the goal's metadata under the key
        ``adapter_precedents``.

        Args:
            goal_graph: GoalGraph to enrich

        Returns:
            The enriched goal_graph (modified in place)
        """
        for goal in goal_graph.goals:
            try:
                all_precs = self.query_all_adapter_precedents(goal.title, limit=3)
                # Only attach if we found at least one precedent
                combined = (
                    all_precs.get("receipts", [])
                    + all_precs.get("outcomes", [])
                    + all_precs.get("debates", [])
                )
                if combined:
                    goal.metadata["adapter_precedents"] = all_precs
                    logger.debug(
                        "Goal %s enriched with %d adapter precedents",
                        goal.id,
                        len(combined),
                    )
            except (AttributeError, TypeError, RuntimeError) as exc:
                logger.debug(
                    "Adapter precedent enrichment skipped for goal %s: %s",
                    getattr(goal, "id", "?"),
                    exc,
                )
        return goal_graph

    def store_pipeline_result(self, result: Any) -> bool:
        """Store completed pipeline result in KM for future queries.

        Accepts either:
        - An object with ``to_dict()`` (PipelineResult, SelfImproveResult)
        - A plain dict with cycle outcome data

        Args:
            result: PipelineResult, dict, or object with to_dict()

        Returns:
            True if stored successfully, False otherwise
        """
        if not self.available:
            return False

        # Normalize to dict
        if isinstance(result, dict):
            result_dict = result
        elif hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        else:
            result_dict = {"raw": str(result)}

        # Try DecisionPlanAdapter first
        try:
            from aragora.knowledge.mound.adapters.decision_plan_adapter import (
                DecisionPlanAdapter,
            )

            adapter = DecisionPlanAdapter(self._km)
            store_method = getattr(adapter, "store", None)
            if callable(store_method):
                store_method(result_dict)
                return True
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

        # Fallback: store directly in KM as a knowledge item
        try:
            objective = result_dict.get("objective", "pipeline result")
            cycle_id = result_dict.get("cycle_id", "unknown")
            self._km.add(
                content=(f"Pipeline cycle {cycle_id}: {objective}"),
                metadata={
                    "item_type": "pipeline_result",
                    "cycle_id": cycle_id,
                    "success": result_dict.get("success", False),
                    "improvement_score": result_dict.get("improvement_score", 0.0),
                    "files_changed": result_dict.get("files_changed", []),
                    "subtasks_completed": result_dict.get("subtasks_completed", 0),
                    "subtasks_failed": result_dict.get("subtasks_failed", 0),
                    "tags": ["pipeline", "self_improve", "cycle_result"],
                },
            )
            return True
        except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
            logger.debug("Failed to store pipeline result in KM: %s", exc)
            return False
