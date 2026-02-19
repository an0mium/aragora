"""Meta-Planner for debate-driven goal prioritization.

Takes high-level business objectives and uses multi-agent debate to
determine which areas should be improved first.

Includes cross-cycle learning: queries past Nomic Loop outcomes from
the Knowledge Mound to inform planning and avoid repeating failures.

Usage:
    from aragora.nomic.meta_planner import MetaPlanner

    planner = MetaPlanner()
    goals = await planner.prioritize_work(
        objective="Maximize utility for SME businesses",
        available_tracks=[Track.SME, Track.QA],
    )

    for goal in goals:
        print(f"{goal.track}: {goal.description} ({goal.estimated_impact})")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Track(Enum):
    """Development tracks for domain-based routing."""

    SME = "sme"
    DEVELOPER = "developer"
    SELF_HOSTED = "self_hosted"
    QA = "qa"
    CORE = "core"
    SECURITY = "security"


@dataclass
class PrioritizedGoal:
    """A prioritized improvement goal."""

    id: str
    track: Track
    description: str
    rationale: str
    estimated_impact: str  # high, medium, low
    priority: int  # 1 = highest
    focus_areas: list[str] = field(default_factory=list)
    file_hints: list[str] = field(default_factory=list)


@dataclass
class HistoricalLearning:
    """Learning from a past Nomic cycle."""

    cycle_id: str
    objective: str
    was_success: bool
    lesson: str
    relevance: float  # 0-1, how relevant to current objective


@dataclass
class PlanningContext:
    """Context for meta-planning decisions."""

    recent_issues: list[str] = field(default_factory=list)
    test_failures: list[str] = field(default_factory=list)
    user_feedback: list[str] = field(default_factory=list)
    recent_changes: list[str] = field(default_factory=list)
    # Cross-cycle learning
    historical_learnings: list[HistoricalLearning] = field(default_factory=list)
    past_failures_to_avoid: list[str] = field(default_factory=list)
    past_successes_to_build_on: list[str] = field(default_factory=list)
    # CI feedback
    ci_failures: list[str] = field(default_factory=list)
    ci_flaky_tests: list[str] = field(default_factory=list)
    # Debate-sourced improvement suggestions
    recent_improvements: list[dict[str, Any]] = field(default_factory=list)
    # Codebase metrics snapshot (from MetricsCollector)
    metric_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaPlannerConfig:
    """Configuration for MetaPlanner."""

    agents: list[str] = field(default_factory=lambda: ["claude", "gemini", "deepseek"])
    debate_rounds: int = 2
    max_goals: int = 5
    consensus_threshold: float = 0.6
    # Cross-cycle learning
    enable_cross_cycle_learning: bool = True
    max_similar_cycles: int = 3
    min_cycle_similarity: float = 0.3
    # Quick mode: skip debate, use heuristic for concrete goals
    quick_mode: bool = False
    # Use introspection data to select agents for planning debates
    use_introspection_selection: bool = True
    # Trickster: detect hollow consensus in self-improvement debates
    enable_trickster: bool = True
    trickster_sensitivity: float = 0.7
    # Convergence detection for semantic consensus
    enable_convergence: bool = True
    # Generate DecisionReceipts for self-improvement decisions
    enable_receipts: bool = True
    # Inject codebase metrics into planning context
    enable_metrics_collection: bool = True
    # Scan mode: prioritize from codebase signals without LLM calls
    scan_mode: bool = False
    # Generate self-explanations for planning decisions
    explain_decisions: bool = True
    # Use business context to re-rank goals by impact
    use_business_context: bool = True


class MetaPlanner:
    """Debate-driven goal prioritization.

    Uses multi-agent debate to determine which areas should be improved
    to best achieve a high-level objective.
    """

    def __init__(self, config: MetaPlannerConfig | None = None):
        self.config = config or MetaPlannerConfig()

    async def prioritize_work(
        self,
        objective: str,
        available_tracks: list[Track] | None = None,
        constraints: list[str] | None = None,
        context: PlanningContext | None = None,
    ) -> list[PrioritizedGoal]:
        """Use multi-agent debate to prioritize work.

        Args:
            objective: High-level business objective
            available_tracks: Which tracks can be worked on
            constraints: Constraints like "no breaking changes"
            context: Additional context (issues, feedback, etc.)

        Returns:
            List of prioritized goals ordered by priority
        """
        available_tracks = available_tracks or list(Track)
        constraints = constraints or []
        context = context or PlanningContext()

        logger.info(
            f"meta_planner_started objective={objective[:100]} tracks={[t.value for t in available_tracks]}"
        )

        # Quick mode: skip debate entirely, use heuristic
        if self.config.quick_mode:
            logger.info("meta_planner_quick_mode using heuristic prioritization")
            return self._heuristic_prioritize(objective, available_tracks)

        # Scan mode: prioritize from codebase signals without LLM calls
        if self.config.scan_mode:
            logger.info("meta_planner_scan_mode using codebase signals")
            return await self._scan_prioritize(objective, available_tracks)

        # Inject codebase metrics for data-driven planning
        if self.config.enable_metrics_collection:
            context = self._enrich_context_with_metrics(context)

        # Cross-cycle learning: Query past similar cycles
        if self.config.enable_cross_cycle_learning:
            context = await self._enrich_context_with_history(objective, available_tracks, context)

        # Inject findings from cross-agent learning bus
        if self.config.enable_cross_cycle_learning:
            self._inject_learning_bus_findings(context)

        # Inject debate-sourced improvement suggestions
        try:
            from aragora.nomic.improvement_queue import get_improvement_queue
            queue = get_improvement_queue()
            suggestions = queue.peek(10)
            if suggestions:
                context.recent_improvements = [
                    {"task": s.task, "category": s.category, "confidence": s.confidence}
                    for s in suggestions
                ]
                logger.info("meta_planner_injected_improvements count=%d", len(suggestions))
        except ImportError:
            pass

        # Auto-discover actionable items from codebase signals
        try:
            from aragora.compat.openclaw.next_steps_runner import NextStepsRunner

            runner = NextStepsRunner(
                repo_path=".",
                scan_code=True,
                scan_issues=False,  # Skip GitHub API calls
                scan_prs=False,
                scan_tests=False,
                scan_deps=False,
                scan_docs=True,
                limit=20,
            )
            scan_result = await runner.scan()
            if scan_result.steps:
                # Feed high-priority items into planning context
                for step in scan_result.steps[:10]:
                    if step.priority in ("critical", "high"):
                        if step.source == "test-failure" and step.title not in context.test_failures:
                            context.test_failures.append(step.title)
                        elif step.title not in context.recent_issues:
                            context.recent_issues.append(
                                f"[{step.category}] {step.title}"
                            )
                logger.info(
                    "meta_planner_injected_next_steps count=%d",
                    min(len(scan_result.steps), 10),
                )
        except ImportError:
            pass
        except (OSError, RuntimeError, ValueError) as e:
            logger.debug(f"NextStepsRunner scan skipped: {e}")

        try:
            from aragora.debate.orchestrator import Arena, DebateProtocol
            from aragora.core import Environment

            # Build debate topic
            topic = self._build_debate_topic(objective, available_tracks, constraints, context)

            # Select agents: use introspection ranking if available
            agent_types = (
                self._select_agents_by_introspection(objective)
                if self.config.use_introspection_selection
                else self.config.agents
            )

            # Create agents using get_secret pattern for API key resolution
            agents = []
            for agent_type in agent_types:
                try:
                    agent = self._create_agent(agent_type)
                    if agent is not None:
                        agents.append(agent)
                except (RuntimeError, OSError, ConnectionError, TimeoutError, ValueError) as e:
                    logger.warning(f"Could not create agent {agent_type}: {e}")

            if not agents:
                logger.warning("No agents available, using heuristic prioritization")
                return self._heuristic_prioritize(objective, available_tracks)

            # Run debate with Trickster and convergence detection
            env = Environment(task=topic)
            protocol = DebateProtocol(
                rounds=self.config.debate_rounds,
                consensus="weighted",
                enable_trickster=self.config.enable_trickster,
                trickster_sensitivity=self.config.trickster_sensitivity,
                convergence_detection=self.config.enable_convergence,
            )

            arena = Arena(env, agents, protocol)
            result = await arena.run()

            # Generate DecisionReceipt for audit trail
            self._generate_receipt(result)

            # Parse goals from debate result
            goals = self._parse_goals_from_debate(result, available_tracks, objective)

            # Re-rank goals using business context
            if self.config.use_business_context:
                goals = self._rerank_with_business_context(goals)

            # Self-explanation: annotate goals with rationale from debate
            self._explain_planning_decision(result, goals)

            logger.info(
                f"meta_planner_completed goal_count={len(goals)} objectives={[g.description[:50] for g in goals]}"
            )

            return goals

        except ImportError as e:
            logger.warning(f"Debate infrastructure not available: {e}")
            return self._heuristic_prioritize(objective, available_tracks)
        except (RuntimeError, OSError, ValueError) as e:
            logger.exception(f"Meta-planning failed: {e}")
            return self._heuristic_prioritize(objective, available_tracks)

    def _enrich_context_with_metrics(self, context: PlanningContext) -> PlanningContext:
        """Enrich planning context with codebase metrics.

        Instantiates MetricsCollector, runs a synchronous collection of test/lint/size
        metrics, and injects the snapshot into PlanningContext.metric_snapshot so the
        debate topic can include hard numbers.

        Args:
            context: Existing planning context

        Returns:
            Enriched PlanningContext with metric_snapshot populated
        """
        try:
            from aragora.nomic.metrics_collector import MetricsCollector, MetricsCollectorConfig

            config = MetricsCollectorConfig(
                test_args=["-x", "-q", "--tb=no", "--timeout=60"],
                test_timeout=120,
            )
            collector = MetricsCollector(config)

            # Synchronous collection of size + lint (skip tests for speed in planning)
            from aragora.nomic.metrics_collector import MetricSnapshot
            import time

            snapshot = MetricSnapshot(timestamp=time.time())
            try:
                collector._collect_size_metrics(snapshot, None)
            except (OSError, ValueError) as e:
                logger.debug("metrics_size_collection_failed: %s", e)
            try:
                collector._collect_lint_metrics(snapshot, None)
            except (OSError, ValueError, Exception) as e:  # noqa: BLE001
                logger.debug("metrics_lint_collection_failed: %s", e)

            context.metric_snapshot = snapshot.to_dict()

            # Inject notable issues into recent_issues for debate visibility
            if snapshot.lint_errors > 0:
                context.recent_issues.append(
                    f"[metrics] {snapshot.lint_errors} lint errors detected"
                )
            if snapshot.tests_failed > 0:
                context.recent_issues.append(
                    f"[metrics] {snapshot.tests_failed} test failures detected"
                )

            logger.info(
                "metrics_enrichment_complete files=%d lines=%d lint_errors=%d",
                snapshot.files_count,
                snapshot.total_lines,
                snapshot.lint_errors,
            )

        except ImportError:
            logger.debug("MetricsCollector not available, skipping metrics enrichment")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Failed to enrich context with metrics: %s", e)

        return context

    async def _enrich_context_with_history(
        self,
        objective: str,
        tracks: list[Track],
        context: PlanningContext,
    ) -> PlanningContext:
        """Enrich planning context with learnings from past cycles.

        Queries the Knowledge Mound for similar past cycles and extracts
        relevant learnings to inform the current planning session.

        Args:
            objective: Current planning objective
            tracks: Available tracks
            context: Existing planning context

        Returns:
            Enriched PlanningContext with historical learnings
        """
        try:
            from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
                get_nomic_cycle_adapter,
            )

            adapter = get_nomic_cycle_adapter()
            track_names = [t.value for t in tracks]

            similar_cycles = await adapter.find_similar_cycles(
                objective=objective,
                tracks=track_names,
                limit=self.config.max_similar_cycles,
                min_similarity=self.config.min_cycle_similarity,
            )

            if similar_cycles:
                logger.info(
                    f"cross_cycle_learning found={len(similar_cycles)} cycles "
                    f"for objective={objective[:50]}"
                )

            for cycle in similar_cycles:
                # Add what worked
                for success in cycle.what_worked:
                    context.past_successes_to_build_on.append(f"[{cycle.objective[:30]}] {success}")
                    context.historical_learnings.append(
                        HistoricalLearning(
                            cycle_id=cycle.cycle_id,
                            objective=cycle.objective,
                            was_success=True,
                            lesson=success,
                            relevance=cycle.similarity,
                        )
                    )

                # Add what failed (important to avoid!)
                for failure in cycle.what_failed:
                    context.past_failures_to_avoid.append(f"[{cycle.objective[:30]}] {failure}")
                    context.historical_learnings.append(
                        HistoricalLearning(
                            cycle_id=cycle.cycle_id,
                            objective=cycle.objective,
                            was_success=False,
                            lesson=failure,
                            relevance=cycle.similarity,
                        )
                    )

            # Query high-ROI goal types for smarter prioritization
            try:
                high_roi = await adapter.find_high_roi_goal_types(limit=5)
                for roi_entry in high_roi:
                    if roi_entry.get("avg_improvement_score", 0) > 0.3:
                        context.past_successes_to_build_on.append(
                            f"[high_roi] Pattern '{roi_entry['pattern']}' "
                            f"avg_improvement={roi_entry['avg_improvement_score']:.2f} "
                            f"({roi_entry['cycle_count']} cycles)"
                        )
                if high_roi:
                    logger.info(
                        "high_roi_patterns loaded=%d for planning", len(high_roi)
                    )
            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                logger.debug("High-ROI query failed: %s", e)

            # Query recurring failures to avoid
            try:
                recurring = await adapter.find_recurring_failures(min_occurrences=2, limit=5)
                for failure in recurring:
                    tracks_str = ", ".join(failure.get("affected_tracks", [])[:3])
                    context.past_failures_to_avoid.append(
                        f"[recurring_failure] '{failure['pattern']}' "
                        f"({failure['occurrences']}x"
                        f"{', tracks: ' + tracks_str if tracks_str else ''})"
                    )
                if recurring:
                    logger.info(
                        "recurring_failures loaded=%d for planning", len(recurring)
                    )
            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                logger.debug("Recurring failures query failed: %s", e)

        except ImportError:
            logger.debug("Nomic cycle adapter not available, skipping history enrichment")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to enrich context with history: {e}")

        # Also query PlanStore for recent pipeline outcomes
        try:
            from aragora.pipeline.plan_store import get_plan_store

            store = get_plan_store()
            outcomes = store.get_recent_outcomes(limit=5)

            for outcome in outcomes:
                status = outcome.get("status", "unknown")
                task = outcome.get("task", "unknown task")
                exec_error = outcome.get("execution_error")

                if status in ("completed",) and not exec_error:
                    context.past_successes_to_build_on.append(
                        f"[pipeline] {task[:60]}"
                    )
                elif status in ("failed", "rejected") or exec_error:
                    error_msg = ""
                    if exec_error and isinstance(exec_error, dict):
                        error_msg = f": {exec_error.get('message', '')[:80]}"
                    context.past_failures_to_avoid.append(
                        f"[pipeline:{status}] {task[:60]}{error_msg}"
                    )

            if outcomes:
                logger.info(
                    f"pipeline_feedback loaded={len(outcomes)} outcomes "
                    f"for planning"
                )
        except ImportError:
            logger.debug("PlanStore not available, skipping pipeline feedback")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to load pipeline outcomes: {e}")

        # Outcome tracker feedback: inject regression data from past cycles
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            regressions = NomicOutcomeTracker.get_regression_history(limit=5)
            for reg in regressions:
                regressed = ", ".join(reg["regressed_metrics"])
                context.past_failures_to_avoid.append(
                    f"[outcome_regression] Cycle {reg['cycle_id'][:8]} regressed: {regressed} "
                    f"(recommendation: {reg['recommendation']})"
                )
            if regressions:
                logger.info("outcome_feedback loaded=%d regressions for planning", len(regressions))
        except ImportError:
            logger.debug("OutcomeTracker not available, skipping regression feedback")
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(f"Failed to load outcome regressions: {e}")

        # --- Calibration data enrichment ---
        try:
            from aragora.ranking.elo import get_elo_store
            from aragora.agents.calibration import CalibrationTracker

            elo = get_elo_store()
            calibration = CalibrationTracker()

            # Get agents ranked by calibration quality
            cal_leaders = calibration.get_leaderboard(metric="brier", limit=5)
            if cal_leaders:
                well_calibrated = [
                    name for name, score in cal_leaders if score < 0.25
                ]
                if well_calibrated:
                    context.past_successes_to_build_on.append(
                        f"[calibration] Well-calibrated agents: "
                        f"{', '.join(well_calibrated[:3])} (Brier < 0.25)"
                    )

            # Get domain-specific performance for relevant tracks
            all_ratings = elo.get_all_ratings()
            underperformers = []
            for rating in all_ratings[:10]:  # Top 10 agents by ELO
                if rating.calibration_total >= 5:
                    brier = rating.calibration_brier_score
                    if brier > 0.35:
                        underperformers.append(
                            f"{rating.agent_name} (Brier={brier:.2f})"
                        )

            if underperformers:
                context.past_failures_to_avoid.append(
                    f"[calibration] Overconfident agents needing "
                    f"improvement: {', '.join(underperformers[:3])}"
                )

            logger.info(
                "calibration_enrichment leaders=%d underperformers=%d",
                len(cal_leaders) if cal_leaders else 0,
                len(underperformers),
            )
        except ImportError:
            logger.debug("Calibration subsystems not available")
        except (RuntimeError, ValueError, OSError, AttributeError) as e:
            logger.warning(f"Failed to enrich with calibration data: {e}")

        return context

    def _generate_receipt(self, result: Any) -> None:
        """Generate a DecisionReceipt from a debate result and persist to KM.

        Creates an audit-ready receipt from the self-improvement debate and
        ingests it into the Knowledge Mound via the ReceiptAdapter so future
        cycles can query past self-improvement decisions.

        Args:
            result: DebateResult from Arena.run()
        """
        if not self.config.enable_receipts:
            return

        try:
            from aragora.export.decision_receipt import DecisionReceipt

            receipt = DecisionReceipt.from_debate_result(result)
            logger.info(
                "meta_planner_receipt_generated receipt_id=%s verdict=%s",
                receipt.receipt_id,
                receipt.verdict,
            )

            # Persist receipt to Knowledge Mound via ReceiptAdapter
            self._ingest_receipt_to_km(receipt)

        except ImportError:
            logger.debug("DecisionReceipt not available, skipping receipt generation")
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to generate receipt from debate result: %s", e)

    def _ingest_receipt_to_km(self, receipt: Any) -> None:
        """Ingest a DecisionReceipt into the Knowledge Mound.

        Uses the ReceiptAdapter to store the receipt so future Nomic cycles
        can query past self-improvement decisions for context.

        Args:
            receipt: DecisionReceipt to ingest
        """
        try:
            from aragora.knowledge.mound.adapters.receipt_adapter import ReceiptAdapter

            adapter = ReceiptAdapter()
            # Fire-and-forget: schedule async ingestion without blocking
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    adapter.ingest_receipt(
                        receipt,
                        tags=["nomic_loop", "self_improvement", "meta_planner"],
                    )
                )
                logger.info(
                    "meta_planner_receipt_km_ingestion_scheduled receipt_id=%s",
                    receipt.receipt_id,
                )
            except RuntimeError:
                logger.debug("No event loop, skipping async receipt KM ingestion")

        except ImportError:
            logger.debug("ReceiptAdapter not available, skipping KM ingestion")
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to ingest receipt to KM: %s", e)

    def _create_agent(self, agent_type: str) -> Any:
        """Create an agent using get_secret pattern for API key resolution.

        Falls back through multiple import paths to maximize compatibility.
        """
        try:
            from aragora.agents import create_agent
            from aragora.agents.base import AgentType

            return create_agent(cast(AgentType, agent_type))
        except ImportError:
            pass

        # Fallback: try direct agent construction
        try:
            from aragora.config.secrets import get_secret

            # Map agent type to required API key
            key_map = {
                "claude": "ANTHROPIC_API_KEY",
                "anthropic-api": "ANTHROPIC_API_KEY",
                "openai-api": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "deepseek": "OPENROUTER_API_KEY",
                "grok": "XAI_API_KEY",
            }
            required_key = key_map.get(agent_type)
            if required_key and not get_secret(required_key):
                logger.debug(f"No API key for {agent_type}, skipping")
                return None
        except ImportError:
            pass

        logger.debug(f"Could not create agent {agent_type} via any path")
        return None

    def _select_agents_by_introspection(self, domain: str) -> list[str]:
        """Select agents using introspection data for better planning quality.

        Ranks available agents by their reputation_score + calibration_score
        for the given domain, preferring agents with proven track records.

        Args:
            domain: The planning domain or objective keyword to match expertise.

        Returns:
            List of agent names ranked by introspection scores, or the static
            config list if introspection is unavailable.
        """
        try:
            from aragora.introspection.api import get_agent_introspection

            scored_agents: list[tuple[str, float]] = []
            for agent_name in self.config.agents:
                snapshot = get_agent_introspection(agent_name)
                # Combined score: reputation + calibration, with domain expertise bonus
                score = snapshot.reputation_score + snapshot.calibration_score
                if domain and snapshot.top_expertise:
                    domain_lower = domain.lower()
                    if any(domain_lower in exp.lower() or exp.lower() in domain_lower
                           for exp in snapshot.top_expertise):
                        score += 0.2  # Domain expertise bonus
                scored_agents.append((agent_name, score))

            if not scored_agents:
                return self.config.agents

            # Sort by score descending
            scored_agents.sort(key=lambda x: x[1], reverse=True)
            selected = [name for name, _ in scored_agents]

            logger.info(
                "introspection_agent_selection domain=%s selected=%s",
                domain[:50] if domain else "general",
                selected,
            )
            return selected

        except ImportError:
            logger.debug("Introspection API not available, using static agent list")
            return self.config.agents
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.debug("Introspection selection failed: %s", e)
            return self.config.agents

    def _explain_planning_decision(
        self,
        result: Any,
        goals: list[PrioritizedGoal],
    ) -> None:
        """Generate a self-explanation for the planning decision.

        Builds an explanation from the debate result and attaches the summary
        as rationale to each goal. Persists the explanation to KM.

        Args:
            result: DebateResult from Arena.run()
            goals: Parsed PrioritizedGoal list to annotate with rationale
        """
        if not self.config.explain_decisions:
            return

        try:
            from aragora.explainability.builder import ExplanationBuilder

            builder = ExplanationBuilder()

            # build() is async, but we schedule it fire-and-forget
            import asyncio

            async def _explain() -> None:
                try:
                    decision = await builder.build(result)
                    summary = builder.generate_summary(decision)
                    for goal in goals:
                        if not goal.rationale:
                            goal.rationale = summary[:500]

                    # Persist explanation to KM
                    self._persist_explanation_to_km(summary, goals)
                except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                    logger.debug("Self-explanation build failed: %s", exc)

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_explain())
            except RuntimeError:
                logger.debug("No event loop, skipping async self-explanation")

        except ImportError:
            logger.debug("ExplanationBuilder not available, skipping self-explanation")
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.debug("Self-explanation unavailable: %s", e)

    def _persist_explanation_to_km(
        self,
        summary: str,
        goals: list[PrioritizedGoal],
    ) -> None:
        """Persist a planning explanation to the Knowledge Mound."""
        try:
            from aragora.knowledge.mound.adapters.receipt_adapter import ReceiptAdapter

            adapter = ReceiptAdapter()
            import asyncio

            async def _ingest() -> None:
                try:
                    from aragora.knowledge.mound.core import KnowledgeItem

                    item = KnowledgeItem(
                        content=summary[:2000],
                        source="meta_planner_explanation",
                        tags=["self_explanation", "meta_planner"]
                        + [g.track.value for g in goals[:5]],
                    )
                    await adapter.ingest(item)
                except (ImportError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
                    logger.debug("KM explanation ingestion failed: %s", exc)

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_ingest())
            except RuntimeError:
                pass

        except ImportError:
            logger.debug("ReceiptAdapter not available for explanation persistence")
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.debug("Explanation KM persistence failed: %s", e)

    def _rerank_with_business_context(self, goals: list[PrioritizedGoal]) -> list[PrioritizedGoal]:
        """Re-rank goals using business impact scoring."""
        try:
            from aragora.nomic.business_context import BusinessContext

            ctx = BusinessContext()
            scored_goals = []
            for goal in goals:
                score = ctx.score_goal(
                    goal=goal.description,
                    file_paths=goal.file_hints,
                    metadata={"focus_areas": goal.focus_areas},
                )
                scored_goals.append((goal, score.total))

            # Sort by business score (descending), re-assign priorities
            scored_goals.sort(key=lambda x: x[1], reverse=True)
            for i, (goal, _score) in enumerate(scored_goals):
                goal.priority = i + 1

            reranked = [g for g, _ in scored_goals]
            logger.info(
                "meta_planner_business_rerank reranked=%s",
                [(g.description[:40], g.priority) for g in reranked],
            )
            return reranked
        except ImportError:
            logger.debug("BusinessContext not available, skipping re-ranking")
            return goals
        except (RuntimeError, ValueError) as e:
            logger.warning("Business context re-ranking failed: %s", e)
            return goals

    def _inject_learning_bus_findings(self, context: PlanningContext) -> None:
        """Inject recent findings from the cross-agent learning bus."""
        try:
            from aragora.nomic.learning_bus import LearningBus

            bus = LearningBus.get_instance()
            findings = bus.get_findings()
            if not findings:
                return

            for finding in findings:
                if finding.severity == "critical":
                    if finding.description not in context.recent_issues:
                        context.recent_issues.append(
                            f"[learning_bus:{finding.topic}] {finding.description}"
                        )
                elif finding.topic == "test_failure":
                    if finding.description not in context.test_failures:
                        context.test_failures.append(finding.description)

            logger.info(
                "meta_planner_injected_learning_bus findings=%d critical=%d",
                len(findings),
                sum(1 for f in findings if f.severity == "critical"),
            )
        except ImportError:
            pass

    def _build_debate_topic(
        self,
        objective: str,
        tracks: list[Track],
        constraints: list[str],
        context: PlanningContext,
    ) -> str:
        """Build the debate topic for meta-planning."""
        track_names = ", ".join(t.value for t in tracks)

        topic = f"""You are planning improvements for the Aragora project.

OBJECTIVE: {objective}

AVAILABLE TRACKS (domains you can work on):
{track_names}

Track descriptions:
- SME: Small business features, dashboard, user workspace
- Developer: SDKs, API, documentation
- Self-Hosted: Docker, deployment, backup/restore
- QA: Tests, CI/CD, code quality
- Core: Debate engine, agents, memory (requires approval)
- Security: Vulnerability scanning, auth hardening, secrets, OWASP compliance

CONSTRAINTS:
{chr(10).join(f"- {c}" for c in constraints) if constraints else "- None specified"}

"""
        if context.recent_issues:
            topic += f"""
RECENT ISSUES:
{chr(10).join(f"- {issue}" for issue in context.recent_issues[:5])}
"""

        if context.test_failures:
            topic += f"""
FAILING TESTS:
{chr(10).join(f"- {failure}" for failure in context.test_failures[:5])}
"""

        # Add CI feedback
        if context.ci_failures:
            topic += f"""
CI FAILURES (recent CI pipeline failures to address):
{chr(10).join(f"- {f}" for f in context.ci_failures[:5])}
"""

        if context.ci_flaky_tests:
            topic += f"""
FLAKY TESTS (intermittent CI failures to stabilize):
{chr(10).join(f"- {t}" for t in context.ci_flaky_tests[:5])}
"""

        # Add historical learnings (cross-cycle learning)
        if context.past_successes_to_build_on:
            topic += f"""
PAST SUCCESSES TO BUILD ON (from similar cycles):
{chr(10).join(f"- {s}" for s in context.past_successes_to_build_on[:5])}
"""

        if context.past_failures_to_avoid:
            topic += f"""
PAST FAILURES TO AVOID (learn from these mistakes):
{chr(10).join(f"- {f}" for f in context.past_failures_to_avoid[:5])}
"""

        # Add codebase metrics for data-driven planning
        if context.metric_snapshot:
            snap = context.metric_snapshot
            metric_lines = ["CODEBASE METRICS (current state):"]
            if snap.get("files_count"):
                metric_lines.append(f"- Python files: {snap['files_count']}")
            if snap.get("total_lines"):
                metric_lines.append(f"- Total lines: {snap['total_lines']:,}")
            if snap.get("tests_passed") or snap.get("tests_failed"):
                passed = snap.get("tests_passed", 0)
                failed = snap.get("tests_failed", 0)
                total = passed + failed + snap.get("tests_errors", 0)
                rate = passed / total if total > 0 else 0
                metric_lines.append(
                    f"- Tests: {passed}/{total} passing ({rate:.0%} pass rate)"
                )
            if snap.get("lint_errors"):
                metric_lines.append(f"- Lint errors: {snap['lint_errors']}")
            if snap.get("test_coverage") is not None:
                metric_lines.append(f"- Test coverage: {snap['test_coverage']:.0%}")
            if len(metric_lines) > 1:
                topic += "\n" + "\n".join(metric_lines) + "\n"

        # Inject relevant deliberation templates to ground abstract objectives
        try:
            from aragora.deliberation.templates.registry import match_templates

            matched = match_templates(objective, limit=3)
            if matched:
                topic += "\nRELEVANT DELIBERATION TEMPLATES (use these as inspiration):\n"
                for tmpl in matched:
                    topic += (
                        f"- {tmpl.name}: {tmpl.description} "
                        f"(category={tmpl.category.value}, "
                        f"tags={', '.join(tmpl.tags[:4])})\n"
                    )
        except ImportError:
            pass

        topic += """
YOUR TASK:
Propose 3-5 specific improvement goals that would best achieve the objective.
For each goal, specify:
1. Which track it belongs to
2. A clear, actionable description
3. Why this should be prioritized (rationale)
4. Expected impact: high, medium, or low

Format your response as a numbered list with clear structure.
Consider dependencies and order goals by priority.
"""
        if context.past_failures_to_avoid:
            topic += """
IMPORTANT: Avoid repeating past failures listed above. Learn from history.
"""
        return topic

    def _parse_goals_from_debate(
        self,
        debate_result: Any,
        available_tracks: list[Track],
        objective: str,
    ) -> list[PrioritizedGoal]:
        """Parse prioritized goals from debate consensus."""
        goals = []

        # Get consensus text from debate result
        consensus_text = ""
        if hasattr(debate_result, "consensus") and debate_result.consensus:
            consensus_text = str(debate_result.consensus)
        elif hasattr(debate_result, "final_response"):
            consensus_text = str(debate_result.final_response)
        elif hasattr(debate_result, "responses") and debate_result.responses:
            consensus_text = str(debate_result.responses[-1])

        if not consensus_text:
            return self._heuristic_prioritize(objective, available_tracks)

        # Parse numbered items from the consensus
        lines = consensus_text.split("\n")
        current_goal: dict[str, Any] = {}
        goal_id = 0

        for line in lines:
            line = line.strip()

            # Detect numbered items (1., 2., etc.) or bullet points
            if re.match(r"^[\d]+[\.\)]\s+", line) or re.match(r"^[-*]\s+", line):
                # Save previous goal if exists
                if current_goal.get("description"):
                    goals.append(self._build_goal(current_goal, goal_id, available_tracks))
                    goal_id += 1

                # Start new goal
                current_goal = {
                    "description": re.sub(r"^[\d]+[\.\)]\s+|^[-*]\s+", "", line),
                    "track": None,
                    "rationale": "",
                    "impact": "medium",
                }

            elif current_goal:
                # Parse track from line
                for track in Track:
                    if track.value.lower() in line.lower():
                        current_goal["track"] = track
                        break

                # Parse impact
                if "high" in line.lower() and "impact" in line.lower():
                    current_goal["impact"] = "high"
                elif "low" in line.lower() and "impact" in line.lower():
                    current_goal["impact"] = "low"

                # Accumulate rationale
                if "because" in line.lower() or "rationale" in line.lower():
                    current_goal["rationale"] = line

        # Don't forget last goal
        if current_goal.get("description"):
            goals.append(self._build_goal(current_goal, goal_id, available_tracks))

        # Limit to max goals
        goals = goals[: self.config.max_goals]

        # If no goals parsed, fall back to heuristics
        if not goals:
            return self._heuristic_prioritize(objective, available_tracks)

        return goals

    def _build_goal(
        self,
        goal_dict: dict[str, Any],
        priority: int,
        available_tracks: list[Track],
    ) -> PrioritizedGoal:
        """Build a PrioritizedGoal from parsed data."""
        # Default track based on keywords if not explicitly set
        track = goal_dict.get("track")
        if not track:
            track = self._infer_track(goal_dict["description"], available_tracks)

        return PrioritizedGoal(
            id=f"goal_{priority}",
            track=track,
            description=goal_dict["description"],
            rationale=goal_dict.get("rationale", ""),
            estimated_impact=goal_dict.get("impact", "medium"),
            priority=priority + 1,
        )

    def _infer_track(self, description: str, available_tracks: list[Track]) -> Track:
        """Infer track from goal description."""
        desc_lower = description.lower()

        track_keywords = {
            Track.SME: ["dashboard", "user", "ui", "frontend", "workspace", "admin"],
            Track.DEVELOPER: ["sdk", "api", "documentation", "client", "package"],
            Track.SELF_HOSTED: ["docker", "deploy", "backup", "ops", "kubernetes"],
            Track.QA: ["test", "ci", "coverage", "quality", "e2e", "playwright"],
            Track.CORE: ["debate", "agent", "consensus", "arena", "memory"],
            Track.SECURITY: [
                "security",
                "auth",
                "vuln",
                "secret",
                "owasp",
                "encrypt",
                "csrf",
                "xss",
                "injection",
            ],
        }

        for track, keywords in track_keywords.items():
            if track in available_tracks:
                if any(kw in desc_lower for kw in keywords):
                    return track

        # Default to first available track
        return available_tracks[0] if available_tracks else Track.DEVELOPER

    async def _scan_prioritize(
        self,
        objective: str,
        available_tracks: list[Track],
    ) -> list[PrioritizedGoal]:
        """Prioritize from codebase signals without any LLM calls.

        Gathers six signal sources:
        1. ``git log`` — recently changed files mapped to tracks
        2. ``CodebaseIndexer`` — untested modules
        3. ``OutcomeTracker`` — past regression patterns
        4. ``.pytest_cache`` — last-run test failures
        5. ``ruff`` — lint violations
        6. ``grep`` — TODO/FIXME/HACK comments

        Each signal contributes a candidate goal. Goals are ranked by signal
        count (more signals = higher priority).

        Args:
            objective: High-level objective (used to seed descriptions).
            available_tracks: Tracks that can receive work.

        Returns:
            List of PrioritizedGoal sorted by priority.
        """
        import subprocess

        track_signals: dict[str, list[str]] = {t.value: [] for t in available_tracks}

        # Signal 1: Recent git changes → map files to tracks
        try:
            git_result = subprocess.run(
                ["git", "log", "--oneline", "--name-only", "-20"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=".",
            )
            if git_result.returncode == 0:
                for line in git_result.stdout.splitlines():
                    line = line.strip()
                    if not line or line[0].isalnum() and " " in line:
                        continue  # Skip commit messages
                    track = self._file_to_track(line, available_tracks)
                    if track and track.value in track_signals:
                        track_signals[track.value].append(f"recent_change: {line}")
        except (subprocess.TimeoutExpired, OSError):
            pass

        # Signal 2: Untested modules from CodebaseIndexer
        try:
            from aragora.nomic.codebase_indexer import CodebaseIndexer

            indexer = CodebaseIndexer(repo_path=".")
            stats = await indexer.index()
            for module in indexer._modules:
                test_paths = indexer._test_map.get(str(module.path), [])
                if not test_paths:
                    track = self._file_to_track(str(module.path), available_tracks)
                    if track and track.value in track_signals:
                        track_signals[track.value].append(
                            f"untested: {module.path}"
                        )
        except (ImportError, RuntimeError, ValueError, OSError):
            pass

        # Signal 3: Past regression patterns
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            regressions = NomicOutcomeTracker.get_regression_history(limit=10)
            for reg in regressions:
                for metric in reg.get("regressed_metrics", []):
                    # Map regression metrics to tracks
                    if "test" in metric.lower() or "coverage" in metric.lower():
                        if Track.QA.value in track_signals:
                            track_signals[Track.QA.value].append(
                                f"regression: {metric}"
                            )
                    elif "token" in metric.lower():
                        if Track.CORE.value in track_signals:
                            track_signals[Track.CORE.value].append(
                                f"regression: {metric}"
                            )
        except (ImportError, RuntimeError, ValueError, OSError):
            pass

        # Signal 4: pytest last-run failures
        try:
            import json as _json
            from pathlib import Path as _P

            lastfailed_path = _P(".pytest_cache/v/cache/lastfailed")
            if lastfailed_path.exists():
                failed = _json.loads(lastfailed_path.read_text())
                for node_id in list(failed.keys())[:20]:
                    # Extract file path from node ID (e.g. "tests/foo.py::TestBar::test_baz")
                    test_file = node_id.split("::")[0] if "::" in node_id else node_id
                    track = self._file_to_track(test_file, available_tracks)
                    if track and track.value in track_signals:
                        track_signals[track.value].append(
                            f"test_failure: {node_id}"
                        )
        except (OSError, ValueError, _json.JSONDecodeError):
            pass

        # Signal 5: ruff lint violations
        try:
            ruff_result = subprocess.run(
                ["ruff", "check", "--quiet", "--output-format=concise", "."],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=".",
            )
            if ruff_result.stdout:
                for i, line in enumerate(ruff_result.stdout.splitlines()):
                    if i >= 20:
                        break
                    # Format: "path/to/file.py:42:1 E501 ..."
                    parts = line.split(":", 1)
                    if parts:
                        track = self._file_to_track(parts[0], available_tracks)
                        if track and track.value in track_signals:
                            track_signals[track.value].append(
                                f"lint: {line.strip()[:100]}"
                            )
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass

        # Signal 6: TODO/FIXME/HACK comments
        try:
            todo_result = subprocess.run(
                ["grep", "-rn", r"TODO\|FIXME\|HACK", "aragora/",
                 "--include=*.py", "-l"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=".",
            )
            if todo_result.returncode == 0 and todo_result.stdout:
                for i, filepath in enumerate(todo_result.stdout.splitlines()):
                    if i >= 10:
                        break
                    filepath = filepath.strip()
                    if filepath:
                        track = self._file_to_track(filepath, available_tracks)
                        if track and track.value in track_signals:
                            track_signals[track.value].append(
                                f"todo: {filepath}"
                            )
        except (subprocess.TimeoutExpired, OSError):
            pass

        # Build goals from signals, ranked by signal count
        ranked = sorted(
            track_signals.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )

        goals: list[PrioritizedGoal] = []
        for priority, (track_name, signals) in enumerate(ranked, start=1):
            if not signals:
                continue

            try:
                track = Track(track_name)
            except ValueError:
                continue

            # Build a description from the top signals
            top_signals = signals[:3]
            signal_summary = "; ".join(top_signals)
            description = (
                f"[{objective[:40]}] {track_name}: "
                f"{len(signals)} signals ({signal_summary})"
            )

            # Enrich with file excerpts for grounded execution
            excerpts = self._gather_file_excerpts(top_signals)
            if excerpts:
                excerpt_text = "\n".join(
                    f"--- {p} ---\n{s[:500]}" for p, s in excerpts.items()
                )
                description += f"\n\nRelevant source:\n{excerpt_text}"

            goals.append(
                PrioritizedGoal(
                    id=f"scan_{priority - 1}",
                    track=track,
                    description=description[:2000],
                    rationale=f"Scan mode: {len(signals)} codebase signals detected",
                    estimated_impact="high" if len(signals) >= 5 else "medium",
                    priority=priority,
                )
            )

        if not goals:
            logger.info("scan_mode_no_signals falling back to heuristic")
            return self._heuristic_prioritize(objective, available_tracks)

        # Re-rank goals using business context
        if self.config.use_business_context:
            goals = self._rerank_with_business_context(goals)

        logger.info(
            "scan_mode_complete goals=%d signals=%d",
            len(goals),
            sum(len(s) for s in track_signals.values()),
        )
        return goals[: self.config.max_goals]

    def _file_to_track(
        self, filepath: str, available_tracks: list[Track]
    ) -> Track | None:
        """Map a file path to a development track."""
        fp = filepath.lower()
        mapping = {
            Track.QA: ["tests/", "test_", "conftest"],
            Track.SME: ["dashboard", "frontend", "live/", "workspace"],
            Track.DEVELOPER: ["sdk", "client", "aragora_sdk/"],
            Track.SELF_HOSTED: ["deploy/", "docker", "k8s", "kubernetes"],
            Track.SECURITY: ["security/", "auth/", "rbac/", "encryption"],
            Track.CORE: ["debate/", "agents/", "memory/", "consensus"],
        }
        for track, patterns in mapping.items():
            if track in available_tracks and any(p in fp for p in patterns):
                return track
        return available_tracks[0] if available_tracks else None

    @staticmethod
    def _gather_file_excerpts(
        signals: list[str],
        max_files: int = 3,
        max_chars_per_file: int = 1500,
        max_total_chars: int = 5000,
    ) -> dict[str, str]:
        """Extract file paths from signal strings and read excerpts.

        Provides real source code context to ground goals instead of
        relying solely on signal labels.

        Args:
            signals: Signal strings like ``"recent_change: aragora/foo.py"``.
            max_files: Maximum number of files to read.
            max_chars_per_file: Max characters per file excerpt.
            max_total_chars: Max total characters across all excerpts.

        Returns:
            Dict mapping file path to truncated content.
        """
        import re
        from pathlib import Path as _P

        # Extract file paths from signal strings
        path_re = re.compile(r"(?:aragora|tests|scripts)/\S+\.py")
        paths: list[str] = []
        for sig in signals:
            match = path_re.search(sig)
            if match and match.group() not in paths:
                paths.append(match.group())
            if len(paths) >= max_files:
                break

        result: dict[str, str] = {}
        total = 0
        for path in paths:
            try:
                content = _P(path).read_text(errors="replace")[:max_chars_per_file]
                if total + len(content) > max_total_chars:
                    content = content[: max_total_chars - total]
                if content:
                    result[path] = content
                    total += len(content)
                if total >= max_total_chars:
                    break
            except OSError:
                continue

        return result

    def _gather_codebase_hints(
        self,
        objective: str,
        available_tracks: list[Track],
    ) -> dict[Track, list[str]]:
        """Gather codebase file hints using CodebaseIndexer (synchronous).

        Queries the codebase index for modules relevant to the objective
        and maps results to tracks via _file_to_track().

        Returns:
            Mapping of Track → list of relevant file paths.
        """
        try:
            from aragora.nomic.codebase_indexer import CodebaseIndexer

            indexer = CodebaseIndexer(repo_path=".")
            # Synchronous scan of already-indexed modules (lightweight)
            for source_dir in indexer.source_dirs:
                source_path = indexer.repo_path / source_dir
                if not source_path.is_dir():
                    continue
                for py_file in sorted(source_path.rglob("*.py")):
                    if len(indexer._modules) >= indexer.max_modules:
                        break
                    if py_file.name.startswith("_") and py_file.name != "__init__.py":
                        continue
                    try:
                        info = indexer._analyze_module(py_file)
                        if info:
                            indexer._modules.append(info)
                    except (SyntaxError, UnicodeDecodeError):
                        continue

            # Keyword-match modules against objective
            obj_lower = objective.lower()
            hints: dict[Track, list[str]] = {}
            for module in indexer._modules:
                searchable = module.to_km_entry()["searchable_text"].lower()
                if any(word in searchable for word in obj_lower.split()):
                    track = self._file_to_track(module.path, available_tracks)
                    if track:
                        hints.setdefault(track, []).append(module.path)

            return hints
        except (ImportError, RuntimeError, ValueError, OSError):
            return {}

    def _heuristic_prioritize(
        self,
        objective: str,
        available_tracks: list[Track],
    ) -> list[PrioritizedGoal]:
        """Fallback heuristic prioritization when debate is unavailable."""
        # Gather codebase hints before keyword matching
        file_hints = self._gather_codebase_hints(objective, available_tracks)

        goals = []
        obj_lower = objective.lower()

        # Generate goals based on objective keywords
        if "sme" in obj_lower or "small business" in obj_lower:
            if Track.SME in available_tracks:
                goals.append(
                    PrioritizedGoal(
                        id="goal_0",
                        track=Track.SME,
                        description="Improve dashboard usability for small business users",
                        rationale="Directly addresses SME utility objective",
                        estimated_impact="high",
                        priority=1,
                    )
                )

            if Track.QA in available_tracks:
                goals.append(
                    PrioritizedGoal(
                        id="goal_1",
                        track=Track.QA,
                        description="Add E2E tests for critical user flows",
                        rationale="Ensures reliability for SME users",
                        estimated_impact="medium",
                        priority=2,
                    )
                )

        # Security-focused objectives
        if any(kw in obj_lower for kw in ["security", "harden", "vuln", "audit"]):
            if Track.SECURITY in available_tracks:
                goals.append(
                    PrioritizedGoal(
                        id=f"goal_{len(goals)}",
                        track=Track.SECURITY,
                        description="Run security scanner and address critical findings",
                        rationale="Security hardening is critical for production",
                        estimated_impact="high",
                        priority=1,
                        focus_areas=["auth", "secrets", "input validation"],
                    )
                )

        # Add default goals for available tracks
        priority = len(goals) + 1
        for track in available_tracks:
            if not any(g.track == track for g in goals):
                goals.append(
                    PrioritizedGoal(
                        id=f"goal_{priority - 1}",
                        track=track,
                        description=f"Improve {track.value} track capabilities",
                        rationale="Supports overall project health",
                        estimated_impact="medium",
                        priority=priority,
                    )
                )
                priority += 1

        # Enrich goals with codebase file hints
        for goal in goals:
            track_hints = file_hints.get(goal.track, [])
            if track_hints:
                goal.file_hints = track_hints[:10]  # Cap at 10 files per goal

        # Re-rank goals using business context
        if self.config.use_business_context:
            goals = self._rerank_with_business_context(goals)

        # Apply self-correction priority adjustments if available
        goals = self._apply_self_correction_adjustments(goals)

        return goals[: self.config.max_goals]

    def _apply_self_correction_adjustments(
        self,
        goals: list[PrioritizedGoal],
    ) -> list[PrioritizedGoal]:
        """Re-rank goals using self-correction engine priority adjustments.

        Queries the SelfCorrectionEngine for track-level adjustments and
        uses them to boost or demote goals. Higher adjustment = higher
        priority (lower number).
        """
        try:
            from aragora.nomic.self_correction import SelfCorrectionEngine

            engine = SelfCorrectionEngine()

            # Query past outcomes from Knowledge Mound
            past_outcomes = self._get_past_outcomes()
            if not past_outcomes:
                return goals

            report = engine.analyze_patterns(past_outcomes)
            adjustments = engine.compute_priority_adjustments(report)

            if not adjustments:
                return goals

            # Apply adjustments: multiply priority by inverse of adjustment
            # (higher adjustment = more important = lower priority number)
            for goal in goals:
                track_key = goal.track.value
                adj = adjustments.get(track_key, 1.0)
                # Adjusted priority: divide by adjustment factor so boosted
                # tracks get lower (higher priority) numbers
                goal.priority = max(1, round(goal.priority / adj))

            # Re-sort by adjusted priority
            goals.sort(key=lambda g: g.priority)

            # Re-assign sequential priority numbers
            for i, goal in enumerate(goals):
                goal.priority = i + 1

            logger.info(
                "self_correction_adjustments_applied tracks=%s",
                {g.track.value: adjustments.get(g.track.value, 1.0) for g in goals},
            )
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Self-correction adjustments unavailable: %s", e)

        return goals

    def _get_past_outcomes(self) -> list[dict[str, Any]]:
        """Retrieve past orchestration outcomes for self-correction analysis."""
        try:
            from aragora.nomic.cycle_store import get_recent_cycles

            cycles = get_recent_cycles(limit=20)
            outcomes: list[dict[str, Any]] = []
            for cycle in cycles:
                for contrib in getattr(cycle, "agent_contributions", []):
                    outcomes.append({
                        "track": getattr(contrib, "domain", "unknown"),
                        "success": getattr(contrib, "was_success", False),
                        "agent": getattr(contrib, "agent_name", "unknown"),
                        "timestamp": getattr(cycle, "timestamp", None),
                    })
            return outcomes
        except (ImportError, RuntimeError, TypeError, ValueError) as e:
            logger.debug("Past outcomes unavailable: %s", e)
            return []

    def record_outcome(
        self,
        goal_outcomes: list[dict[str, Any]],
        objective: str = "",
    ) -> dict[str, dict[str, Any]]:
        """Record improvement outcomes and log success rates by track.

        Closes the feedback loop: after goals are executed, this method
        aggregates results by Track and logs structured success rates so
        the next planning cycle can learn from what worked.

        Args:
            goal_outcomes: List of dicts with keys:
                - track: str (Track value like "sme", "qa", "core")
                - success: bool
                - description: str (optional)
                - error: str (optional, for failures)
            objective: The original planning objective

        Returns:
            Dict mapping track name to {attempted, succeeded, failed, rate}
        """
        track_stats: dict[str, dict[str, Any]] = {}

        for outcome in goal_outcomes:
            track = outcome.get("track", "unknown")
            if track not in track_stats:
                track_stats[track] = {
                    "attempted": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "failures": [],
                }

            track_stats[track]["attempted"] += 1
            if outcome.get("success"):
                track_stats[track]["succeeded"] += 1
            else:
                track_stats[track]["failed"] += 1
                error = outcome.get("error", outcome.get("description", "unknown"))
                track_stats[track]["failures"].append(error)

        # Compute rates and log
        total_attempted = 0
        total_succeeded = 0
        for track, stats in track_stats.items():
            rate = stats["succeeded"] / stats["attempted"] if stats["attempted"] > 0 else 0.0
            stats["rate"] = rate
            total_attempted += stats["attempted"]
            total_succeeded += stats["succeeded"]

            logger.info(
                "meta_planner_track_outcome track=%s attempted=%d succeeded=%d "
                "failed=%d rate=%.2f",
                track,
                stats["attempted"],
                stats["succeeded"],
                stats["failed"],
                rate,
            )

        overall_rate = total_succeeded / total_attempted if total_attempted > 0 else 0.0
        logger.info(
            "meta_planner_outcome_summary objective=%s total_attempted=%d "
            "total_succeeded=%d overall_rate=%.2f tracks=%d",
            objective[:80] if objective else "unspecified",
            total_attempted,
            total_succeeded,
            overall_rate,
            len(track_stats),
        )

        # Persist to KM for cross-cycle learning
        self._persist_outcome_to_km(goal_outcomes, objective, track_stats)

        return track_stats

    def _persist_outcome_to_km(
        self,
        goal_outcomes: list[dict[str, Any]],
        objective: str,
        track_stats: dict[str, dict[str, Any]],
    ) -> None:
        """Persist outcomes to Knowledge Mound via NomicCycleAdapter."""
        try:
            from datetime import datetime, timezone

            from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
                CycleStatus,
                GoalOutcome,
                NomicCycleOutcome,
                get_nomic_cycle_adapter,
            )

            total = sum(s["attempted"] for s in track_stats.values())
            succeeded = sum(s["succeeded"] for s in track_stats.values())
            failed = sum(s["failed"] for s in track_stats.values())

            if total == 0:
                return

            if failed == 0:
                status = CycleStatus.SUCCESS
            elif succeeded == 0:
                status = CycleStatus.FAILED
            else:
                status = CycleStatus.PARTIAL

            now = datetime.now(timezone.utc)
            outcomes = []
            for o in goal_outcomes:
                outcomes.append(
                    GoalOutcome(
                        goal_id=o.get("goal_id", ""),
                        description=o.get("description", ""),
                        track=o.get("track", "unknown"),
                        status=CycleStatus.SUCCESS if o.get("success") else CycleStatus.FAILED,
                        error=o.get("error"),
                        learnings=o.get("learnings", []),
                    )
                )

            cycle = NomicCycleOutcome(
                cycle_id=f"meta_{now.strftime('%Y%m%d_%H%M%S')}",
                objective=objective,
                status=status,
                started_at=now,
                completed_at=now,
                goal_outcomes=outcomes,
                goals_attempted=total,
                goals_succeeded=succeeded,
                goals_failed=failed,
                tracks_affected=list(track_stats.keys()),
            )

            adapter = get_nomic_cycle_adapter()
            # Fire-and-forget: don't block on async KM write
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(adapter.ingest_cycle_outcome(cycle))
            except RuntimeError:
                # No running loop - skip async persistence
                logger.debug("No event loop, skipping async KM persistence")

        except ImportError:
            logger.debug("NomicCycleAdapter not available, skipping KM persistence")
        except (ValueError, TypeError, OSError, AttributeError, KeyError) as e:
            logger.warning("Failed to persist outcome to KM: %s", e)


__all__ = [
    "MetaPlanner",
    "MetaPlannerConfig",
    "PrioritizedGoal",
    "PlanningContext",
    "HistoricalLearning",
    "Track",
]
