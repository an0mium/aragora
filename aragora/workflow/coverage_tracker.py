"""
Workflow Coverage Tracker.

Provides systematic tracking of workflow test coverage across:
- Step types (task, agent, parallel, conditional, loop, etc.)
- Execution patterns (sequential, parallel, hive_mind, etc.)
- Templates (legal, healthcare, accounting, etc.)
- Configuration combinations (timeouts, checkpoints, retries)

Usage:
    from aragora.workflow.coverage_tracker import WorkflowCoverageTracker

    tracker = WorkflowCoverageTracker()

    # Track step execution
    tracker.track_step("task", "test_basic_task")
    tracker.track_step("parallel", "test_parallel_execution")

    # Track pattern usage
    tracker.track_pattern("sequential", "test_basic_flow")
    tracker.track_pattern("hive_mind", "test_parallel_agents")

    # Get coverage report
    report = tracker.get_report()
    print(f"Step coverage: {report['step_coverage']:.1%}")
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Known step types that should be covered
KNOWN_STEP_TYPES = frozenset(
    {
        "task",
        "agent",
        "parallel",
        "conditional",
        "loop",
        "debate",
        "quick_debate",
        "decision",
        "memory_read",
        "memory_write",
        "human_checkpoint",
        "wait",
        "end",
    }
)

# Known execution patterns
KNOWN_PATTERNS = frozenset(
    {
        "sequential",
        "parallel",
        "conditional",
        "loop",
        "hive_mind",
        "hierarchical",
        "review_cycle",
        "dialectic",
        "map_reduce",
    }
)

# Known template categories
KNOWN_TEMPLATES = frozenset(
    {
        "legal_contract_review",
        "legal_due_diligence",
        "healthcare_hipaa_compliance",
        "healthcare_clinical_review",
        "accounting_financial_audit",
        "software_code_review",
        "software_security_audit",
        "academic_citation_verification",
        "regulatory_compliance_assessment",
        "finance_investment_analysis",
        "general_research",
    }
)

# Key configuration dimensions
KNOWN_CONFIG_DIMENSIONS = frozenset(
    {
        "checkpoint_enabled",
        "checkpoint_disabled",
        "timeout_short",
        "timeout_long",
        "retry_enabled",
        "retry_disabled",
        "resource_limits_enabled",
        "resource_limits_disabled",
        "parallel_limit_1",
        "parallel_limit_5",
        "parallel_unlimited",
    }
)


@dataclass
class CoverageEntry:
    """A single coverage tracking entry."""

    component: str
    test_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "test_name": self.test_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CoverageReport:
    """Workflow coverage report."""

    step_coverage: float
    pattern_coverage: float
    template_coverage: float
    config_coverage: float
    covered_steps: Set[str]
    covered_patterns: Set[str]
    covered_templates: Set[str]
    covered_configs: Set[str]
    missing_steps: Set[str]
    missing_patterns: Set[str]
    missing_templates: Set[str]
    missing_configs: Set[str]
    total_tests: int
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def overall_coverage(self) -> float:
        """Calculate overall coverage as weighted average."""
        # Weight: steps 40%, patterns 25%, templates 20%, configs 15%
        return (
            self.step_coverage * 0.40
            + self.pattern_coverage * 0.25
            + self.template_coverage * 0.20
            + self.config_coverage * 0.15
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_coverage": self.step_coverage,
            "pattern_coverage": self.pattern_coverage,
            "template_coverage": self.template_coverage,
            "config_coverage": self.config_coverage,
            "overall_coverage": self.overall_coverage,
            "covered_steps": sorted(self.covered_steps),
            "covered_patterns": sorted(self.covered_patterns),
            "covered_templates": sorted(self.covered_templates),
            "covered_configs": sorted(self.covered_configs),
            "missing_steps": sorted(self.missing_steps),
            "missing_patterns": sorted(self.missing_patterns),
            "missing_templates": sorted(self.missing_templates),
            "missing_configs": sorted(self.missing_configs),
            "total_tests": self.total_tests,
            "generated_at": self.generated_at.isoformat(),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Workflow Coverage Report",
            "=" * 50,
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Coverage by Dimension:",
            f"  Step Types:    {self.step_coverage:6.1%} ({len(self.covered_steps)}/{len(KNOWN_STEP_TYPES)})",
            f"  Patterns:      {self.pattern_coverage:6.1%} ({len(self.covered_patterns)}/{len(KNOWN_PATTERNS)})",
            f"  Templates:     {self.template_coverage:6.1%} ({len(self.covered_templates)}/{len(KNOWN_TEMPLATES)})",
            f"  Configs:       {self.config_coverage:6.1%} ({len(self.covered_configs)}/{len(KNOWN_CONFIG_DIMENSIONS)})",
            "",
            f"  OVERALL:       {self.overall_coverage:6.1%}",
            "",
            f"Total Tests Tracked: {self.total_tests}",
        ]

        if self.missing_steps:
            lines.extend(
                [
                    "",
                    "Missing Step Coverage:",
                    *[f"  - {s}" for s in sorted(self.missing_steps)],
                ]
            )

        if self.missing_patterns:
            lines.extend(
                [
                    "",
                    "Missing Pattern Coverage:",
                    *[f"  - {p}" for p in sorted(self.missing_patterns)],
                ]
            )

        if self.missing_templates:
            lines.extend(
                [
                    "",
                    "Missing Template Coverage:",
                    *[f"  - {t}" for t in sorted(self.missing_templates)],
                ]
            )

        if self.missing_configs:
            lines.extend(
                [
                    "",
                    "Missing Config Coverage:",
                    *[f"  - {c}" for c in sorted(self.missing_configs)],
                ]
            )

        return "\n".join(lines)


class WorkflowCoverageTracker:
    """
    Tracks workflow test coverage across multiple dimensions.

    Thread-safe singleton that accumulates coverage data across test runs.
    """

    _instance: Optional["WorkflowCoverageTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "WorkflowCoverageTracker":
        """Create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False  # type: ignore[has-type]
            return cls._instance

    def __init__(self) -> None:
        """Initialize tracker (only once due to singleton)."""
        if self._initialized:  # type: ignore[has-type]
            return

        self._data_lock = threading.Lock()
        self._step_coverage: Dict[str, List[CoverageEntry]] = defaultdict(list)
        self._pattern_coverage: Dict[str, List[CoverageEntry]] = defaultdict(list)
        self._template_coverage: Dict[str, List[CoverageEntry]] = defaultdict(list)
        self._config_coverage: Dict[str, List[CoverageEntry]] = defaultdict(list)
        self._test_names: Set[str] = set()
        self._initialized = True

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._data_lock:
            self._step_coverage.clear()
            self._pattern_coverage.clear()
            self._template_coverage.clear()
            self._config_coverage.clear()
            self._test_names.clear()

    def track_step(
        self, step_type: str, test_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track coverage of a step type."""
        with self._data_lock:
            entry = CoverageEntry(
                component=step_type,
                test_name=test_name,
                metadata=metadata or {},
            )
            self._step_coverage[step_type].append(entry)
            self._test_names.add(test_name)
            logger.debug(f"Tracked step '{step_type}' in test '{test_name}'")

    def track_pattern(
        self, pattern: str, test_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track coverage of an execution pattern."""
        with self._data_lock:
            entry = CoverageEntry(
                component=pattern,
                test_name=test_name,
                metadata=metadata or {},
            )
            self._pattern_coverage[pattern].append(entry)
            self._test_names.add(test_name)
            logger.debug(f"Tracked pattern '{pattern}' in test '{test_name}'")

    def track_template(
        self, template: str, test_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track coverage of a workflow template."""
        with self._data_lock:
            entry = CoverageEntry(
                component=template,
                test_name=test_name,
                metadata=metadata or {},
            )
            self._template_coverage[template].append(entry)
            self._test_names.add(test_name)
            logger.debug(f"Tracked template '{template}' in test '{test_name}'")

    def track_config(
        self, config_dimension: str, test_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track coverage of a configuration dimension."""
        with self._data_lock:
            entry = CoverageEntry(
                component=config_dimension,
                test_name=test_name,
                metadata=metadata or {},
            )
            self._config_coverage[config_dimension].append(entry)
            self._test_names.add(test_name)
            logger.debug(f"Tracked config '{config_dimension}' in test '{test_name}'")

    def get_report(self) -> CoverageReport:
        """Generate a comprehensive coverage report."""
        with self._data_lock:
            covered_steps = set(self._step_coverage.keys())
            covered_patterns = set(self._pattern_coverage.keys())
            covered_templates = set(self._template_coverage.keys())
            covered_configs = set(self._config_coverage.keys())

            missing_steps = KNOWN_STEP_TYPES - covered_steps
            missing_patterns = KNOWN_PATTERNS - covered_patterns
            missing_templates = KNOWN_TEMPLATES - covered_templates
            missing_configs = KNOWN_CONFIG_DIMENSIONS - covered_configs

            step_coverage = len(covered_steps) / len(KNOWN_STEP_TYPES) if KNOWN_STEP_TYPES else 1.0
            pattern_coverage = (
                len(covered_patterns) / len(KNOWN_PATTERNS) if KNOWN_PATTERNS else 1.0
            )
            template_coverage = (
                len(covered_templates) / len(KNOWN_TEMPLATES) if KNOWN_TEMPLATES else 1.0
            )
            config_coverage = (
                len(covered_configs) / len(KNOWN_CONFIG_DIMENSIONS)
                if KNOWN_CONFIG_DIMENSIONS
                else 1.0
            )

            return CoverageReport(
                step_coverage=step_coverage,
                pattern_coverage=pattern_coverage,
                template_coverage=template_coverage,
                config_coverage=config_coverage,
                covered_steps=covered_steps,
                covered_patterns=covered_patterns,
                covered_templates=covered_templates,
                covered_configs=covered_configs,
                missing_steps=missing_steps,  # type: ignore[arg-type]
                missing_patterns=missing_patterns,  # type: ignore[arg-type]
                missing_templates=missing_templates,  # type: ignore[arg-type]
                missing_configs=missing_configs,  # type: ignore[arg-type]
                total_tests=len(self._test_names),
            )

    def save_report(self, path: Optional[Path] = None) -> Path:
        """Save coverage report to file."""
        if path is None:
            path = Path(".coverage/workflow_coverage.json")

        path.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_report()
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved workflow coverage report to {path}")
        return path

    def print_summary(self) -> None:
        """Print coverage summary to stdout."""
        report = self.get_report()
        print(report.summary())  # noqa: T201


# Global tracker instance
_tracker: Optional[WorkflowCoverageTracker] = None


def get_tracker() -> WorkflowCoverageTracker:
    """Get the global workflow coverage tracker."""
    global _tracker
    if _tracker is None:
        _tracker = WorkflowCoverageTracker()
    return _tracker


def track_step(step_type: str, test_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to track step coverage."""
    get_tracker().track_step(step_type, test_name, metadata)


def track_pattern(pattern: str, test_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to track pattern coverage."""
    get_tracker().track_pattern(pattern, test_name, metadata)


def track_template(
    template: str, test_name: str, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to track template coverage."""
    get_tracker().track_template(template, test_name, metadata)


def track_config(
    config_dimension: str, test_name: str, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to track config coverage."""
    get_tracker().track_config(config_dimension, test_name, metadata)


def get_coverage_report() -> CoverageReport:
    """Get the current coverage report."""
    return get_tracker().get_report()


def print_coverage_summary() -> None:
    """Print coverage summary."""
    get_tracker().print_summary()
