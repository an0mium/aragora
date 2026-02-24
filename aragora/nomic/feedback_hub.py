"""Unified Feedback Routing Hub -- the nervous system of self-improvement.

Collects signals from ALL subsystems and routes them to appropriate
improvement targets.  Acts as the central routing nexus that ties
together every feedback loop in the platform:

    Source                     Target(s)
    ------                     ---------
    user_feedback          ->  ImprovementQueue (via FeedbackAnalyzer)
    gauntlet               ->  ImprovementQueue (via GauntletAutoImprove)
    introspection          ->  ImprovementQueue + Genesis evolution
    debate_outcomes        ->  KnowledgeMound + ELO rating update
    knowledge_contradictions -> ImprovementQueue
    pulse_stale_topics     ->  Pulse refresh queue

Routing is tracked for full auditability and statistics.

Usage:
    from aragora.nomic.feedback_hub import get_feedback_hub

    hub = get_feedback_hub()
    result = hub.route("user_feedback", {"feedback_id": "f-123", ...})
    stats = hub.stats()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Maximum history entries retained in memory
_MAX_HISTORY = 500


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    """Result of routing a single payload."""

    source: str
    targets_hit: list[str] = field(default_factory=list)
    targets_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    routed_at: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        """True if at least one target was hit and none failed."""
        return len(self.targets_hit) > 0 and len(self.targets_failed) == 0

    @property
    def partial(self) -> bool:
        """True if some targets hit but some failed."""
        return len(self.targets_hit) > 0 and len(self.targets_failed) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "targets_hit": self.targets_hit,
            "targets_failed": self.targets_failed,
            "errors": self.errors,
            "routed_at": self.routed_at,
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Known source types
# ---------------------------------------------------------------------------

KNOWN_SOURCES = frozenset(
    {
        "user_feedback",
        "gauntlet",
        "introspection",
        "debate_outcomes",
        "knowledge_contradictions",
        "pulse_stale_topics",
    }
)


# ---------------------------------------------------------------------------
# FeedbackHub
# ---------------------------------------------------------------------------


class FeedbackHub:
    """Central routing nexus for all feedback signals.

    For each known source type, the hub defines a routing strategy that
    dispatches the payload to one or more targets.  Targets are resolved
    via lazy imports so missing modules degrade gracefully rather than
    crashing the hub.

    Thread-safe: all mutations to counters and history are protected by
    a lock.

    Args:
        max_history: Maximum routing history entries to retain.
    """

    def __init__(self, *, max_history: int = _MAX_HISTORY) -> None:
        self._max_history = max_history

        # Routing history (most recent first)
        self._history: list[RouteResult] = []

        # Counters
        self._total_routed: int = 0
        self._by_source: dict[str, int] = defaultdict(int)
        self._by_target: dict[str, int] = defaultdict(int)
        self._failures: int = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, source: str, payload: dict[str, Any]) -> RouteResult:
        """Route a feedback payload from *source* to its target(s).

        Args:
            source: One of the KNOWN_SOURCES strings.
            payload: Arbitrary data dict specific to the source.

        Returns:
            RouteResult describing which targets were hit/failed.

        Raises:
            ValueError: If *source* is not a recognised source type.
        """
        if source not in KNOWN_SOURCES:
            raise ValueError(
                f"Unknown feedback source {source!r}. "
                f"Known sources: {sorted(KNOWN_SOURCES)}"
            )

        router = _ROUTE_TABLE.get(source)
        if router is None:
            raise ValueError(f"No routing strategy registered for {source!r}")

        result = router(self, payload)

        # Record stats
        with self._lock:
            self._total_routed += 1
            self._by_source[source] += 1
            for t in result.targets_hit:
                self._by_target[t] += 1
            if result.targets_failed:
                self._failures += len(result.targets_failed)

            self._history.insert(0, result)
            if len(self._history) > self._max_history:
                self._history = self._history[: self._max_history]

        logger.info(
            "feedback_hub_routed source=%s hit=%s failed=%s",
            source,
            result.targets_hit,
            result.targets_failed,
        )

        return result

    def stats(self) -> dict[str, Any]:
        """Return routing statistics snapshot."""
        with self._lock:
            return {
                "total_routed": self._total_routed,
                "total_failures": self._failures,
                "by_source": dict(self._by_source),
                "by_target": dict(self._by_target),
                "history_size": len(self._history),
                "known_sources": sorted(KNOWN_SOURCES),
            }

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent routing history entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of RouteResult dicts, most recent first.
        """
        with self._lock:
            return [r.to_dict() for r in self._history[:limit]]

    def reset(self) -> None:
        """Reset all counters and history (for testing)."""
        with self._lock:
            self._history.clear()
            self._total_routed = 0
            self._by_source.clear()
            self._by_target.clear()
            self._failures = 0

    # ------------------------------------------------------------------
    # Routing strategies (one per source)
    # ------------------------------------------------------------------

    def _route_user_feedback(self, payload: dict[str, Any]) -> RouteResult:
        """user_feedback -> ImprovementQueue (via FeedbackAnalyzer).

        Delegates to the existing ``FeedbackAnalyzer.process_new_feedback()``
        pipeline.  We do NOT duplicate its logic -- just trigger it.
        """
        result = RouteResult(source="user_feedback")

        try:
            from aragora.nomic.feedback_analyzer import FeedbackAnalyzer

            analyzer = FeedbackAnalyzer()
            analysis = analyzer.process_new_feedback(
                limit=payload.get("limit", 50),
            )
            result.targets_hit.append("improvement_queue")
            payload["_analysis_result"] = {
                "goals_created": analysis.goals_created,
                "feedback_processed": analysis.feedback_processed,
            }
        except ImportError:
            result.targets_failed.append("improvement_queue")
            result.errors.append("FeedbackAnalyzer not available")
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            result.targets_failed.append("improvement_queue")
            result.errors.append(f"FeedbackAnalyzer failed: {type(exc).__name__}")
            logger.warning("feedback_hub_user_feedback_error: %s", exc)

        return result

    def _route_gauntlet(self, payload: dict[str, Any]) -> RouteResult:
        """gauntlet -> ImprovementQueue (via GauntletAutoImprove).

        Delegates to the existing ``GauntletAutoImprove.on_run_complete()``
        pipeline.  Expects ``payload["gauntlet_result"]`` to be a
        GauntletResult object (or mock with the right attrs).
        """
        result = RouteResult(source="gauntlet")

        try:
            from aragora.gauntlet.auto_improve import GauntletAutoImprove

            auto = GauntletAutoImprove(
                enabled=True,
                max_goals_per_run=payload.get("max_goals_per_run", 5),
            )
            gauntlet_result = payload.get("gauntlet_result")
            if gauntlet_result is None:
                result.targets_failed.append("improvement_queue")
                result.errors.append("Missing gauntlet_result in payload")
                return result

            auto_result = auto.on_run_complete(gauntlet_result)
            result.targets_hit.append("improvement_queue")
            payload["_auto_improve_result"] = {
                "goals_queued": auto_result.goals_queued,
            }
        except ImportError:
            result.targets_failed.append("improvement_queue")
            result.errors.append("GauntletAutoImprove not available")
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError) as exc:
            result.targets_failed.append("improvement_queue")
            result.errors.append(f"GauntletAutoImprove failed: {type(exc).__name__}")
            logger.warning("feedback_hub_gauntlet_error: %s", exc)

        return result

    def _route_introspection(self, payload: dict[str, Any]) -> RouteResult:
        """introspection -> ImprovementQueue + Genesis evolution.

        Pushes low-performing agent signals to the ImprovementQueue AND
        triggers Genesis evolution breeding for underperforming agents.
        """
        result = RouteResult(source="introspection")

        # Target 1: ImprovementQueue
        try:
            from aragora.nomic.feedback_orchestrator import (
                ImprovementGoal,
                ImprovementQueue,
            )

            agent_name = payload.get("agent_name", "unknown")
            success_rate = payload.get("success_rate", 0.0)
            description = payload.get(
                "description",
                f"Agent '{agent_name}' underperforming (rate={success_rate:.0%})",
            )

            queue = ImprovementQueue()
            queue.push(
                ImprovementGoal(
                    goal=description,
                    source="introspection",
                    priority=max(0.0, min(1.0, 1.0 - success_rate)),
                    context={
                        "agent_name": agent_name,
                        "success_rate": success_rate,
                        **{k: v for k, v in payload.items() if k not in (
                            "agent_name", "success_rate", "description",
                        )},
                    },
                )
            )
            result.targets_hit.append("improvement_queue")
        except ImportError:
            result.targets_failed.append("improvement_queue")
            result.errors.append("ImprovementQueue not available")
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            result.targets_failed.append("improvement_queue")
            result.errors.append(f"ImprovementQueue push failed: {type(exc).__name__}")
            logger.warning("feedback_hub_introspection_queue_error: %s", exc)

        # Target 2: Genesis evolution
        try:
            from aragora.genesis.breeding import PopulationManager

            agent_type = payload.get("agent_type", "claude")
            manager = PopulationManager()
            population = manager.get_or_create_population([agent_type])
            evolved = manager.evolve_population(population)
            result.targets_hit.append("genesis_evolution")
            payload["_evolved_genomes"] = len(evolved.genomes) if evolved else 0
        except ImportError:
            # Genesis is optional; graceful skip
            result.targets_failed.append("genesis_evolution")
            result.errors.append("Genesis breeding not available")
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError) as exc:
            result.targets_failed.append("genesis_evolution")
            result.errors.append(f"Genesis evolution failed: {type(exc).__name__}")
            logger.warning("feedback_hub_introspection_genesis_error: %s", exc)

        return result

    def _route_debate_outcomes(self, payload: dict[str, Any]) -> RouteResult:
        """debate_outcomes -> KnowledgeMound + ELO update.

        Persists debate results to KM for cross-debate learning and
        updates agent ELO ratings.
        """
        result = RouteResult(source="debate_outcomes")

        debate_id = payload.get("debate_id", "")

        # Target 1: KnowledgeMound
        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound(workspace_id="default")
            # Store debate outcome as a knowledge item (simplified sync path)
            if hasattr(mound, "store_dict"):
                mound.store_dict(
                    {
                        "type": "debate_outcome",
                        "debate_id": debate_id,
                        "conclusion": payload.get("conclusion", ""),
                        "consensus": payload.get("consensus_reached", False),
                        "confidence": payload.get("confidence", 0.0),
                    }
                )
            result.targets_hit.append("knowledge_mound")
        except ImportError:
            result.targets_failed.append("knowledge_mound")
            result.errors.append("KnowledgeMound not available")
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError) as exc:
            result.targets_failed.append("knowledge_mound")
            result.errors.append(f"KM storage failed: {type(exc).__name__}")
            logger.warning("feedback_hub_debate_km_error: %s", exc)

        # Target 2: ELO rating update
        try:
            from aragora.ranking.elo import EloSystem

            agents = payload.get("agents_participated", [])
            winner = payload.get("winner_agent")
            if agents and winner:
                elo = EloSystem()
                for agent in agents:
                    if agent != winner:
                        elo.record_match(winner=winner, loser=agent)
                result.targets_hit.append("elo_update")
            else:
                # No clear winner -- nothing to update
                result.targets_hit.append("elo_update")
        except ImportError:
            result.targets_failed.append("elo_update")
            result.errors.append("EloSystem not available")
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError) as exc:
            result.targets_failed.append("elo_update")
            result.errors.append(f"ELO update failed: {type(exc).__name__}")
            logger.warning("feedback_hub_debate_elo_error: %s", exc)

        return result

    def _route_knowledge_contradictions(
        self, payload: dict[str, Any]
    ) -> RouteResult:
        """knowledge_contradictions -> ImprovementQueue.

        Converts knowledge contradiction signals into improvement goals
        so the Nomic Loop can resolve them.
        """
        result = RouteResult(source="knowledge_contradictions")

        try:
            from aragora.nomic.feedback_orchestrator import (
                ImprovementGoal,
                ImprovementQueue,
            )

            contradiction_type = payload.get("contradiction_type", "unknown")
            item_a = payload.get("item_a_id", "?")
            item_b = payload.get("item_b_id", "?")
            severity = payload.get("severity", "medium")
            conflict_score = payload.get("conflict_score", 0.5)

            description = (
                f"Resolve KM contradiction ({severity}): "
                f"{contradiction_type} conflict between {item_a} and {item_b} "
                f"(score={conflict_score:.2f})"
            )

            priority_map = {"critical": 0.95, "high": 0.8, "medium": 0.6, "low": 0.3}
            priority = priority_map.get(severity, 0.5)

            queue = ImprovementQueue()
            queue.push(
                ImprovementGoal(
                    goal=description,
                    source="knowledge_contradiction",
                    priority=priority,
                    context={
                        "contradiction_type": contradiction_type,
                        "item_a_id": item_a,
                        "item_b_id": item_b,
                        "severity": severity,
                        "conflict_score": conflict_score,
                    },
                )
            )
            result.targets_hit.append("improvement_queue")
        except ImportError:
            result.targets_failed.append("improvement_queue")
            result.errors.append("ImprovementQueue not available")
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            result.targets_failed.append("improvement_queue")
            result.errors.append(
                f"Contradiction goal push failed: {type(exc).__name__}"
            )
            logger.warning("feedback_hub_contradiction_error: %s", exc)

        return result

    def _route_pulse_stale_topics(self, payload: dict[str, Any]) -> RouteResult:
        """pulse_stale_topics -> Pulse refresh queue.

        Signals the PulseManager to re-fetch stale topics from their
        platforms so the system stays current.
        """
        result = RouteResult(source="pulse_stale_topics")

        try:
            from aragora.pulse.ingestor import PulseManager

            manager = PulseManager()
            platforms = payload.get("platforms")  # None = all
            limit = payload.get("limit_per_platform", 5)

            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    topics = pool.submit(
                        asyncio.run,
                        manager.get_trending_topics(
                            platforms=platforms, limit_per_platform=limit
                        ),
                    ).result(timeout=30)
            else:
                topics = asyncio.run(
                    manager.get_trending_topics(
                        platforms=platforms, limit_per_platform=limit
                    )
                )

            result.targets_hit.append("pulse_refresh")
            payload["_refreshed_topics"] = len(topics) if topics else 0
        except ImportError:
            result.targets_failed.append("pulse_refresh")
            result.errors.append("PulseManager not available")
        except (RuntimeError, ValueError, TypeError, OSError, TimeoutError) as exc:
            result.targets_failed.append("pulse_refresh")
            result.errors.append(f"Pulse refresh failed: {type(exc).__name__}")
            logger.warning("feedback_hub_pulse_refresh_error: %s", exc)

        return result


# ---------------------------------------------------------------------------
# Route table  (source name -> method)
# ---------------------------------------------------------------------------

_ROUTE_TABLE: dict[str, Any] = {
    "user_feedback": FeedbackHub._route_user_feedback,
    "gauntlet": FeedbackHub._route_gauntlet,
    "introspection": FeedbackHub._route_introspection,
    "debate_outcomes": FeedbackHub._route_debate_outcomes,
    "knowledge_contradictions": FeedbackHub._route_knowledge_contradictions,
    "pulse_stale_topics": FeedbackHub._route_pulse_stale_topics,
}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_hub_singleton: FeedbackHub | None = None
_hub_lock = threading.Lock()


def get_feedback_hub() -> FeedbackHub:
    """Get or create the module-level FeedbackHub singleton.

    Returns:
        The singleton FeedbackHub instance.
    """
    global _hub_singleton
    if _hub_singleton is None:
        with _hub_lock:
            if _hub_singleton is None:
                _hub_singleton = FeedbackHub()
    return _hub_singleton


__all__ = [
    "FeedbackHub",
    "KNOWN_SOURCES",
    "RouteResult",
    "get_feedback_hub",
]
