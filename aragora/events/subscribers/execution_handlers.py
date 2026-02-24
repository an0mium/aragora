"""
Execution outcome event handlers.

Subscribes to PLAN_COMPLETED/PLAN_FAILED events to:
- Update ELO ratings for participating agents based on execution outcomes
- Feed outcomes to MetaLearner for knowledge improvement

This creates a feedback loop: agents who propose plans that actually succeed
get higher ELO, while agents whose plans fail get downranked.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from aragora.events.types import StreamEvent

logger = logging.getLogger(__name__)

# ELO adjustments for execution outcomes
_ELO_SUCCESS_BOOST = 5
_ELO_FAILURE_PENALTY = -3


class ExecutionHandlersMixin:
    """Mixin providing execution outcome event handlers.

    This mixin requires the implementing class to provide:
    - stats: dict - Handler statistics tracking
    - _is_km_handler_enabled(handler_name: str) -> bool - Feature flag check
    """

    stats: dict
    _is_km_handler_enabled: Callable[[str], bool]

    def _handle_plan_completed(self, event: StreamEvent) -> None:
        """Handle PLAN_COMPLETED events.

        Boosts ELO for agents whose proposals were in the winning plan.
        Optionally invokes MetaLearner to learn from the outcome.
        """
        start = time.time()

        try:
            data = event.data
            plan_id = data.get("plan_id", "")
            debate_id = data.get("debate_id", "")
            workspace_id = data.get("workspace_id", "")
            tasks_completed = data.get("tasks_completed", 0)
            tasks_total = data.get("tasks_total", 0)

            if not debate_id:
                return

            # Update ELO for participating agents
            self._update_elo_from_outcome(
                debate_id=debate_id,
                success=True,
                boost=_ELO_SUCCESS_BOOST,
            )

            # Feed outcome to MetaLearner
            self._feed_meta_learner(
                debate_id=debate_id,
                plan_id=plan_id,
                success=True,
                workspace_id=workspace_id,
                tasks_completed=tasks_completed,
                tasks_total=tasks_total,
                lessons=data.get("lessons", []),
            )

            self.stats.setdefault("plan_completed", {"events": 0, "errors": 0})
            self.stats["plan_completed"]["events"] += 1

            latency_ms = (time.time() - start) * 1000
            logger.debug(
                "PLAN_COMPLETED handler: plan=%s latency=%.1fms",
                plan_id,
                latency_ms,
            )

        except (KeyError, TypeError, AttributeError, ValueError, RuntimeError) as e:
            logger.error("PLAN_COMPLETED handler error: %s", e)
            self.stats.setdefault("plan_completed", {"events": 0, "errors": 0})
            self.stats["plan_completed"]["errors"] += 1

    def _handle_plan_failed(self, event: StreamEvent) -> None:
        """Handle PLAN_FAILED events.

        Applies ELO penalty to agents whose proposals led to a failed plan.
        """
        start = time.time()

        try:
            data = event.data
            plan_id = data.get("plan_id", "")
            debate_id = data.get("debate_id", "")
            workspace_id = data.get("workspace_id", "")
            error = data.get("error", "")

            if not debate_id:
                return

            # Apply ELO penalty
            self._update_elo_from_outcome(
                debate_id=debate_id,
                success=False,
                boost=_ELO_FAILURE_PENALTY,
            )

            # Feed failure to MetaLearner
            self._feed_meta_learner(
                debate_id=debate_id,
                plan_id=plan_id,
                success=False,
                workspace_id=workspace_id,
                error=error,
                lessons=data.get("lessons", []),
            )

            self.stats.setdefault("plan_failed", {"events": 0, "errors": 0})
            self.stats["plan_failed"]["events"] += 1

            latency_ms = (time.time() - start) * 1000
            logger.debug(
                "PLAN_FAILED handler: plan=%s latency=%.1fms",
                plan_id,
                latency_ms,
            )

        except (KeyError, TypeError, AttributeError, ValueError, RuntimeError) as e:
            logger.error("PLAN_FAILED handler error: %s", e)
            self.stats.setdefault("plan_failed", {"events": 0, "errors": 0})
            self.stats["plan_failed"]["errors"] += 1

    @staticmethod
    def _update_elo_from_outcome(
        debate_id: str,
        success: bool,
        boost: int,
    ) -> None:
        """Update ELO ratings for agents who participated in the debate.

        Args:
            debate_id: The debate whose agents should be updated
            success: Whether the plan execution succeeded
            boost: ELO adjustment (positive for success, negative for failure)
        """
        try:
            from aragora.ranking.elo import EloSystem

            elo = EloSystem()

            # Look up agents from the debate's match history
            recent = elo.get_recent_matches(limit=50)
            debate_agents: set[str] = set()
            for match in recent:
                match_debate = ""
                if isinstance(match, dict):
                    match_debate = match.get("debate_id", "")
                    participants = match.get("participants", [])
                elif hasattr(match, "debate_id"):
                    match_debate = getattr(match, "debate_id", "")
                    participants = getattr(match, "participants", [])
                else:
                    continue

                if match_debate == debate_id:
                    if isinstance(participants, list):
                        debate_agents.update(participants)
                    elif isinstance(participants, str):
                        debate_agents.add(participants)

            if not debate_agents:
                logger.debug("No agents found for debate %s, skipping ELO update", debate_id)
                return

            # Record an execution-outcome match
            scores = {agent: (1.0 if success else 0.0) for agent in debate_agents}
            elo.record_match(
                debate_id=f"{debate_id}_execution",
                participants=list(debate_agents),
                scores=scores,
                domain="execution_outcome",
            )

            logger.debug(
                "ELO updated for %d agents from debate %s: %+d",
                len(debate_agents),
                debate_id,
                boost,
            )

        except ImportError:
            logger.debug("EloSystem not available for execution feedback")
        except (RuntimeError, TypeError, AttributeError, ValueError, KeyError) as e:
            logger.debug("ELO update from execution outcome failed: %s", e)

    @staticmethod
    def _feed_meta_learner(
        debate_id: str,
        plan_id: str,
        success: bool,
        workspace_id: str = "",
        tasks_completed: int = 0,
        tasks_total: int = 0,
        error: str = "",
        lessons: list[str] | None = None,
    ) -> None:
        """Feed execution outcome to MetaLearner for knowledge improvement."""
        try:
            from aragora.knowledge.bridges import KnowledgeBridgeHub

            from aragora.knowledge.mound import KnowledgeMound

            mound = KnowledgeMound(workspace_id=workspace_id or "default")
            hub = KnowledgeBridgeHub(mound)
            if not hasattr(hub, "meta_learner") or hub.meta_learner is None:
                return

            await hub.meta_learner.capture_learning_summary(
                summary={
                    "debate_id": debate_id,
                    "plan_id": plan_id,
                    "success": success,
                    "workspace_id": workspace_id,
                    "tasks_completed": tasks_completed,
                    "tasks_total": tasks_total,
                    "error": error,
                    "lessons": lessons or [],
                }
            )
            logger.debug("MetaLearner fed outcome for plan %s", plan_id)

            # Emit META_LEARNING_EVALUATED event
            try:
                from aragora.events.types import StreamEvent, StreamEventType

                from aragora.server.stream.emitter import SyncEventEmitter

                emitter: SyncEventEmitter | None = None
                if emitter is not None:
                    emitter.emit(
                        StreamEvent(
                            type=StreamEventType.META_LEARNING_EVALUATED,
                            data={
                                "plan_id": plan_id,
                                "debate_id": debate_id,
                                "success": success,
                                "tasks_completed": tasks_completed,
                                "tasks_total": tasks_total,
                            },
                        )
                    )
            except (ImportError, AttributeError, TypeError):
                pass

        except ImportError:
            logger.debug("KnowledgeBridgeHub not available for MetaLearner feedback")
        except (RuntimeError, TypeError, AttributeError, ValueError) as e:
            logger.debug("MetaLearner feedback failed: %s", e)
