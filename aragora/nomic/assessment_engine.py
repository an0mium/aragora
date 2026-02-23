"""Autonomous Assessment Engine — unified codebase health scoring.

Combines multiple signal sources (strategic scanner, metrics collector,
outcome tracker, improvement queue, feedback bridge) into a single
CodebaseHealthReport with ranked improvement candidates.

No LLM calls — pure static analysis and signal aggregation.

Usage:
    engine = AutonomousAssessmentEngine()
    report = await engine.assess()
    print(f"Health: {report.health_score:.2f}")
    for candidate in report.improvement_candidates[:5]:
        print(f"  [{candidate.priority:.2f}] {candidate.description}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SignalSource:
    """A source of improvement signals."""

    name: str
    weight: float  # 0.0-1.0, relative importance
    findings: list[Any] = field(default_factory=list)
    error: str | None = None  # Set if source failed to produce findings


@dataclass
class ImprovementCandidate:
    """A ranked improvement candidate derived from signal sources."""

    description: str
    priority: float  # 0.0-1.0, higher is more urgent
    source: str  # Which signal source identified this
    files: list[str] = field(default_factory=list)
    category: str = "general"  # "test", "lint", "complexity", "todo", "regression", "feedback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "priority": self.priority,
            "source": self.source,
            "files": self.files,
            "category": self.category,
        }


@dataclass
class CodebaseHealthReport:
    """Complete codebase health assessment."""

    health_score: float  # 0.0-1.0, higher is healthier
    signal_sources: list[SignalSource] = field(default_factory=list)
    improvement_candidates: list[ImprovementCandidate] = field(default_factory=list)
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    assessment_duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "health_score": self.health_score,
            "signal_sources": [
                {
                    "name": s.name,
                    "weight": s.weight,
                    "findings_count": len(s.findings),
                    "error": s.error,
                }
                for s in self.signal_sources
            ],
            "improvement_candidates": [c.to_dict() for c in self.improvement_candidates],
            "metrics_snapshot": self.metrics_snapshot,
            "assessment_duration_seconds": self.assessment_duration_seconds,
        }


class AutonomousAssessmentEngine:
    """Unified codebase assessment combining all signal sources.

    Collects findings from:
    1. StrategicScanner — untested modules, complexity hotspots, stale TODOs
    2. MetricsCollector — test/lint/coverage metrics
    3. OutcomeTracker — regression history from past cycles
    4. ImprovementQueue — queued feedback goals from previous cycles
    5. OutcomeFeedbackBridge — calibration and systematic error findings

    Aggregates into a single health score and ranked improvement candidates.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._weights = weights or {
            "scanner": 0.3,
            "metrics": 0.25,
            "regressions": 0.2,
            "queue": 0.15,
            "feedback": 0.1,
        }

    async def assess(self) -> CodebaseHealthReport:
        """Run full codebase assessment. No LLM calls."""
        start = time.time()
        sources: list[SignalSource] = []
        candidates: list[ImprovementCandidate] = []
        metrics: dict[str, Any] = {}

        # 1. Strategic Scanner signals
        scanner_source = self._collect_scanner_signals()
        sources.append(scanner_source)
        candidates.extend(self._scanner_to_candidates(scanner_source))

        # 2. Metrics Collector signals
        metrics_source = await self._collect_metrics_signals()
        sources.append(metrics_source)
        candidates.extend(self._metrics_to_candidates(metrics_source))
        if metrics_source.findings:
            metrics.update(metrics_source.findings[0] if metrics_source.findings else {})

        # 3. Regression history
        regression_source = self._collect_regression_signals()
        sources.append(regression_source)
        candidates.extend(self._regressions_to_candidates(regression_source))

        # 4. Improvement queue
        queue_source = self._collect_queue_signals()
        sources.append(queue_source)
        candidates.extend(self._queue_to_candidates(queue_source))

        # 5. Feedback bridge
        feedback_source = self._collect_feedback_signals()
        sources.append(feedback_source)
        candidates.extend(self._feedback_to_candidates(feedback_source))

        # Sort candidates by priority (highest first)
        candidates.sort(key=lambda c: c.priority, reverse=True)

        # Compute health score
        health_score = self._compute_health_score(sources, candidates)

        duration = time.time() - start
        report = CodebaseHealthReport(
            health_score=health_score,
            signal_sources=sources,
            improvement_candidates=candidates,
            metrics_snapshot=metrics,
            assessment_duration_seconds=duration,
        )

        logger.info(
            "assessment_complete health=%.2f candidates=%d duration=%.1fs",
            health_score,
            len(candidates),
            duration,
        )

        return report

    # --- Signal collection ---

    def _collect_scanner_signals(self) -> SignalSource:
        """Collect signals from StrategicScanner."""
        source = SignalSource(name="scanner", weight=self._weights.get("scanner", 0.3))
        try:
            from aragora.nomic.strategic_scanner import StrategicScanner

            scanner = StrategicScanner()
            assessment = scanner.scan()
            source.findings = assessment.findings
        except ImportError:
            source.error = "StrategicScanner not available"
        except (RuntimeError, ValueError, OSError) as e:
            source.error = f"Scanner failed: {type(e).__name__}"
            logger.debug("Scanner signal collection failed: %s", e)
        return source

    async def _collect_metrics_signals(self) -> SignalSource:
        """Collect signals from MetricsCollector."""
        source = SignalSource(name="metrics", weight=self._weights.get("metrics", 0.25))
        try:
            from aragora.nomic.metrics_collector import MetricsCollector

            collector = MetricsCollector()
            snapshot = await collector.collect_baseline("assessment")
            source.findings = [snapshot.to_dict()]
        except ImportError:
            source.error = "MetricsCollector not available"
        except (RuntimeError, ValueError, OSError) as e:
            source.error = f"Metrics collection failed: {type(e).__name__}"
            logger.debug("Metrics signal collection failed: %s", e)
        return source

    def _collect_regression_signals(self) -> SignalSource:
        """Collect regression history from OutcomeTracker."""
        source = SignalSource(name="regressions", weight=self._weights.get("regressions", 0.2))
        try:
            from aragora.nomic.outcome_tracker import NomicOutcomeTracker

            history = NomicOutcomeTracker.get_regression_history()
            source.findings = history
        except ImportError:
            source.error = "OutcomeTracker not available"
        except (RuntimeError, ValueError, OSError) as e:
            source.error = f"Regression history failed: {type(e).__name__}"
            logger.debug("Regression signal collection failed: %s", e)
        return source

    def _collect_queue_signals(self) -> SignalSource:
        """Collect queued improvement suggestions from ImprovementQueue."""
        source = SignalSource(name="queue", weight=self._weights.get("queue", 0.15))
        try:
            from aragora.nomic.improvement_queue import get_improvement_queue

            queue = get_improvement_queue()
            items = queue.peek(20)
            source.findings = [
                {
                    "goal": item.suggestion,
                    "description": item.task,
                    "category": item.category,
                    "priority": item.confidence,
                }
                for item in items
            ]
        except ImportError:
            source.error = "ImprovementQueue not available"
        except (RuntimeError, ValueError, OSError) as e:
            source.error = f"Queue retrieval failed: {type(e).__name__}"
            logger.debug("Queue signal collection failed: %s", e)
        return source

    def _collect_feedback_signals(self) -> SignalSource:
        """Collect calibration findings from OutcomeFeedbackBridge."""
        source = SignalSource(name="feedback", weight=self._weights.get("feedback", 0.1))
        try:
            from aragora.nomic.outcome_feedback import OutcomeFeedbackBridge

            bridge = OutcomeFeedbackBridge()
            goals = bridge.generate_improvement_goals()
            source.findings = [
                {
                    "description": goal.description,
                    "priority": goal.severity,
                    "files": [],
                    "category": "feedback",
                }
                for goal in goals
            ]
        except ImportError:
            source.error = "OutcomeFeedbackBridge not available"
        except (RuntimeError, ValueError, OSError) as e:
            source.error = f"Feedback collection failed: {type(e).__name__}"
            logger.debug("Feedback signal collection failed: %s", e)
        return source

    # --- Signal to candidate conversion ---

    def _scanner_to_candidates(self, source: SignalSource) -> list[ImprovementCandidate]:
        """Convert scanner findings to improvement candidates."""
        candidates: list[ImprovementCandidate] = []
        if source.error or not source.findings:
            return candidates

        for finding in source.findings:
            if isinstance(finding, dict):
                candidates.append(
                    ImprovementCandidate(
                        description=finding.get("description", str(finding)),
                        priority=self._severity_to_priority(finding.get("severity", "medium")),
                        source="scanner",
                        files=[finding["file_path"]] if "file_path" in finding else [],
                        category=finding.get("category", "general"),
                    )
                )
            elif hasattr(finding, "description"):
                candidates.append(
                    ImprovementCandidate(
                        description=getattr(finding, "description", str(finding)),
                        priority=self._severity_to_priority(getattr(finding, "severity", "medium")),
                        source="scanner",
                        files=[getattr(finding, "file_path", "")]
                        if hasattr(finding, "file_path")
                        else [],
                        category=getattr(finding, "category", "general"),
                    )
                )

        return candidates

    def _metrics_to_candidates(self, source: SignalSource) -> list[ImprovementCandidate]:
        """Convert metrics findings to improvement candidates."""
        candidates: list[ImprovementCandidate] = []
        if source.error or not source.findings:
            return candidates

        for snapshot in source.findings:
            if not isinstance(snapshot, dict):
                continue

            # High test failure rate
            fail_count = snapshot.get("tests_failed", 0)
            if fail_count > 0:
                candidates.append(
                    ImprovementCandidate(
                        description=f"Fix {fail_count} failing test(s)",
                        priority=0.9,
                        source="metrics",
                        category="test",
                    )
                )

            # Lint violations
            lint_count = snapshot.get("lint_errors", 0)
            if lint_count > 10:
                candidates.append(
                    ImprovementCandidate(
                        description=f"Reduce {lint_count} lint violations",
                        priority=0.5 if lint_count < 50 else 0.7,
                        source="metrics",
                        category="lint",
                    )
                )

        return candidates

    def _regressions_to_candidates(self, source: SignalSource) -> list[ImprovementCandidate]:
        """Convert regression history to improvement candidates."""
        candidates: list[ImprovementCandidate] = []
        if source.error or not source.findings:
            return candidates

        for regression in source.findings:
            if isinstance(regression, dict):
                metrics = regression.get("regressed_metrics", [])
                desc = f"Fix regression in cycle {regression.get('cycle_id', '?')}: {', '.join(metrics)}"
                candidates.append(
                    ImprovementCandidate(
                        description=regression.get("description", desc),
                        priority=0.85,
                        source="regressions",
                        files=regression.get("files", []),
                        category="regression",
                    )
                )

        return candidates

    def _queue_to_candidates(self, source: SignalSource) -> list[ImprovementCandidate]:
        """Convert queued improvement goals to candidates."""
        candidates: list[ImprovementCandidate] = []
        if source.error or not source.findings:
            return candidates

        for item in source.findings:
            if isinstance(item, dict):
                candidates.append(
                    ImprovementCandidate(
                        description=item.get("goal", item.get("description", str(item))),
                        priority=float(item.get("priority", 0.6)),
                        source="queue",
                        files=item.get("files", []),
                        category=item.get("category", "feedback"),
                    )
                )

        return candidates

    def _feedback_to_candidates(self, source: SignalSource) -> list[ImprovementCandidate]:
        """Convert feedback bridge findings to candidates."""
        candidates: list[ImprovementCandidate] = []
        if source.error or not source.findings:
            return candidates

        for finding in source.findings:
            if isinstance(finding, dict):
                candidates.append(
                    ImprovementCandidate(
                        description=finding.get("description", str(finding)),
                        priority=float(finding.get("priority", 0.65)),
                        source="feedback",
                        files=finding.get("files", []),
                        category="feedback",
                    )
                )

        return candidates

    # --- Health score computation ---

    def _compute_health_score(
        self,
        sources: list[SignalSource],
        candidates: list[ImprovementCandidate],
    ) -> float:
        """Compute overall health score from sources and candidates.

        Score is 1.0 (perfectly healthy) minus penalty for issues found.
        Each candidate reduces the score based on its priority and source weight.
        """
        if not candidates:
            # No issues found — report high health but not perfect
            # (we may have failed to collect some signals)
            failed_sources = sum(1 for s in sources if s.error)
            return max(0.5, 1.0 - (failed_sources * 0.1))

        total_penalty = 0.0
        for candidate in candidates:
            # Find the source weight
            source_weight = 1.0
            for s in sources:
                if s.name == candidate.source:
                    source_weight = s.weight
                    break
            total_penalty += candidate.priority * source_weight * 0.05  # Diminishing penalty

        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, 1.0 - total_penalty))

    @staticmethod
    def _severity_to_priority(severity: str) -> float:
        """Convert severity string to priority float."""
        mapping = {
            "critical": 0.95,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
            "info": 0.2,
        }
        return mapping.get(severity.lower(), 0.5)


__all__ = [
    "AutonomousAssessmentEngine",
    "CodebaseHealthReport",
    "ImprovementCandidate",
    "SignalSource",
]
