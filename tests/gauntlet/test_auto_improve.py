"""Tests for Gauntlet Auto-Improve feedback loop.

Verifies that GauntletAutoImprove correctly:
- Converts high/critical findings into improvement goals
- Filters out low/medium severity findings
- Rate-limits to max N goals per run
- Is disabled by default (opt-in)
- Handles empty findings gracefully
- Skips already-queued findings (deduplication)
- Logs queued goals for auditability
- Respects cooldown between triggers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.gauntlet.auto_improve import (
    AutoImproveResult,
    GauntletAutoImprove,
)


# =============================================================================
# Test helpers
# =============================================================================


@dataclass
class FakeGauntletResult:
    """Minimal GauntletResult-like object for testing."""

    gauntlet_id: str = "gauntlet-test-001"
    findings: list[Any] = field(default_factory=list)


def _make_finding(
    *,
    finding_id: str = "f-1",
    severity: str = "high",
    category: str = "security",
    description: str = "SQL injection in login handler",
    recommendation: str = "Use parameterized queries",
    location: str = "aragora/server/auth.py",
) -> dict:
    """Create a finding dict compatible with improvement_bridge."""
    return {
        "id": finding_id,
        "severity": severity,
        "category": category,
        "description": description,
        "recommendation": recommendation,
        "location": location,
    }


def _make_result_with_findings(
    findings: list[dict],
    gauntlet_id: str = "gauntlet-test-001",
) -> FakeGauntletResult:
    """Create a FakeGauntletResult with the given findings."""
    return FakeGauntletResult(
        gauntlet_id=gauntlet_id,
        findings=findings,
    )


def _patch_queue():
    """Patch the ImprovementQueue to avoid SQLite writes."""
    return patch(
        "aragora.nomic.feedback_orchestrator.ImprovementQueue",
        return_value=MagicMock(),
    )


# =============================================================================
# Core behavior tests
# =============================================================================


class TestAutoImproveDisabledByDefault:
    """Verify auto-improve is opt-in."""

    def test_disabled_by_default(self):
        """Auto-improve should be disabled by default."""
        auto = GauntletAutoImprove()
        assert auto.enabled is False

    def test_disabled_returns_empty_result(self):
        """When disabled, on_run_complete returns an empty result."""
        auto = GauntletAutoImprove(enabled=False)
        result = auto.on_run_complete(
            _make_result_with_findings([_make_finding(severity="critical")])
        )
        assert result.goals_queued == 0
        assert result.error is None

    def test_enabled_explicit(self):
        """Auto-improve can be explicitly enabled."""
        auto = GauntletAutoImprove(enabled=True)
        assert auto.enabled is True


class TestHighSeverityFindings:
    """Verify high/critical findings become goals."""

    def test_critical_finding_becomes_goal(self):
        """Critical severity findings should be queued as goals."""
        with _patch_queue():
            auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)
            gauntlet_result = _make_result_with_findings([
                _make_finding(finding_id="f-crit", severity="critical"),
            ])

            result = auto.on_run_complete(gauntlet_result)

            assert result.goals_queued == 1
            assert result.error is None

    def test_high_finding_becomes_goal(self):
        """High severity findings should be queued as goals."""
        with _patch_queue():
            auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)
            gauntlet_result = _make_result_with_findings([
                _make_finding(finding_id="f-high", severity="high"),
            ])

            result = auto.on_run_complete(gauntlet_result)

            assert result.goals_queued == 1
            assert result.error is None

    def test_multiple_high_critical_findings(self):
        """Multiple high/critical findings should all be queued."""
        with _patch_queue():
            auto = GauntletAutoImprove(
                enabled=True, max_goals_per_run=10, cooldown_seconds=0
            )
            gauntlet_result = _make_result_with_findings([
                _make_finding(finding_id="f-1", severity="critical"),
                _make_finding(finding_id="f-2", severity="high"),
                _make_finding(finding_id="f-3", severity="critical"),
            ])

            result = auto.on_run_complete(gauntlet_result)

            assert result.goals_queued == 3
            assert len(result.goal_descriptions) == 3


class TestLowSeverityFiltered:
    """Verify low/medium severity findings are filtered out."""

    def test_low_severity_filtered(self):
        """Low severity findings should not produce goals."""
        with _patch_queue():
            auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)
            gauntlet_result = _make_result_with_findings([
                _make_finding(finding_id="f-low", severity="low"),
            ])

            result = auto.on_run_complete(gauntlet_result)

            assert result.goals_queued == 0

    def test_medium_severity_filtered_with_high_min(self):
        """Medium severity findings should be filtered when min_severity='high'."""
        with _patch_queue():
            auto = GauntletAutoImprove(
                enabled=True, min_severity="high", cooldown_seconds=0
            )
            gauntlet_result = _make_result_with_findings([
                _make_finding(finding_id="f-med", severity="medium"),
            ])

            result = auto.on_run_complete(gauntlet_result)

            assert result.goals_queued == 0

    def test_mixed_severities_only_high_critical_queued(self):
        """Mixed findings should only queue high/critical ones."""
        with _patch_queue():
            auto = GauntletAutoImprove(
                enabled=True, max_goals_per_run=10, cooldown_seconds=0
            )
            gauntlet_result = _make_result_with_findings([
                _make_finding(finding_id="f-crit", severity="critical"),
                _make_finding(finding_id="f-low", severity="low"),
                _make_finding(finding_id="f-high", severity="high"),
                _make_finding(finding_id="f-med", severity="medium"),
                _make_finding(finding_id="f-info", severity="info"),
            ])

            result = auto.on_run_complete(gauntlet_result)

            # Only critical and high should be queued
            assert result.goals_queued == 2


class TestRateLimiting:
    """Verify rate limiting caps goals per run."""

    def test_rate_limit_caps_at_max(self):
        """Should cap at max_goals_per_run goals."""
        with _patch_queue():
            auto = GauntletAutoImprove(
                enabled=True, max_goals_per_run=3, cooldown_seconds=0
            )
            gauntlet_result = _make_result_with_findings([
                _make_finding(finding_id=f"f-{i}", severity="critical")
                for i in range(8)
            ])

            result = auto.on_run_complete(gauntlet_result)

            assert result.goals_queued == 3
            assert result.goals_skipped_rate_limit == 3
            # Note: findings_to_goals was called with max_goals=6 (3*2),
            # so bridge may cap before rate limit; 8 findings -> bridge returns 6
            # -> 3 queued, 3 rate-limited

    def test_default_max_is_five(self):
        """Default max_goals_per_run should be 5."""
        with _patch_queue():
            auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)
            gauntlet_result = _make_result_with_findings([
                _make_finding(finding_id=f"f-{i}", severity="high")
                for i in range(12)
            ])

            result = auto.on_run_complete(gauntlet_result)

            assert result.goals_queued == 5
            assert result.goals_skipped_rate_limit == 5
            # Bridge over-fetches with max_goals=10, all 10 are high severity,
            # 5 queued + 5 rate-limited


class TestEmptyFindings:
    """Verify empty findings produce no goals."""

    def test_empty_findings_no_goals(self):
        """Empty findings list should produce zero goals."""
        auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)
        gauntlet_result = _make_result_with_findings([])

        result = auto.on_run_complete(gauntlet_result)

        assert result.goals_queued == 0
        assert result.error is None

    def test_none_findings_no_goals(self):
        """Result with no findings attribute should produce zero goals."""
        auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)

        @dataclass
        class EmptyResult:
            gauntlet_id: str = "gauntlet-none"

        result = auto.on_run_complete(EmptyResult())

        assert result.goals_queued == 0


class TestDeduplication:
    """Verify already-queued findings are skipped."""

    def test_duplicate_findings_skipped(self):
        """Same finding ID should not be queued twice."""
        with _patch_queue():
            auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)

            findings = [
                _make_finding(finding_id="f-dup", severity="critical"),
            ]

            # First run - should queue
            result1 = auto.on_run_complete(_make_result_with_findings(findings))
            assert result1.goals_queued == 1

            # Second run with same finding - should skip
            result2 = auto.on_run_complete(
                _make_result_with_findings(findings, gauntlet_id="gauntlet-test-002")
            )
            assert result2.goals_queued == 0
            assert result2.goals_skipped_duplicate == 1

    def test_new_findings_still_queued_after_dedup(self):
        """New findings should still be queued even when some are duplicates."""
        with _patch_queue():
            auto = GauntletAutoImprove(
                enabled=True, max_goals_per_run=10, cooldown_seconds=0
            )

            # First run
            result1 = auto.on_run_complete(
                _make_result_with_findings([
                    _make_finding(finding_id="f-1", severity="critical"),
                ])
            )
            assert result1.goals_queued == 1

            # Second run with one old + one new finding
            result2 = auto.on_run_complete(
                _make_result_with_findings(
                    [
                        _make_finding(finding_id="f-1", severity="critical"),
                        _make_finding(finding_id="f-2", severity="high"),
                    ],
                    gauntlet_id="gauntlet-test-002",
                )
            )
            assert result2.goals_queued == 1
            assert result2.goals_skipped_duplicate == 1

    def test_reset_clears_dedup_state(self):
        """reset() should clear duplicate tracking state."""
        with _patch_queue():
            auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)

            findings = [_make_finding(finding_id="f-reset", severity="critical")]

            # First run
            result1 = auto.on_run_complete(_make_result_with_findings(findings))
            assert result1.goals_queued == 1

            # Reset
            auto.reset()

            # Same finding after reset - should be queued again
            result2 = auto.on_run_complete(
                _make_result_with_findings(findings, gauntlet_id="gauntlet-002")
            )
            assert result2.goals_queued == 1


class TestCooldown:
    """Verify cooldown between triggers."""

    def test_cooldown_blocks_rapid_triggers(self):
        """Rapid successive triggers should be blocked by cooldown."""
        auto = GauntletAutoImprove(enabled=True, cooldown_seconds=300)

        findings = [_make_finding(severity="critical")]

        # First call sets the timer
        auto.on_run_complete(_make_result_with_findings(findings))

        # Second call should be blocked by cooldown
        result2 = auto.on_run_complete(
            _make_result_with_findings(findings, gauntlet_id="gauntlet-002")
        )
        assert result2.error == "cooldown_active"
        assert result2.goals_queued == 0

    def test_zero_cooldown_allows_rapid_triggers(self):
        """Zero cooldown should allow immediate re-triggers."""
        with _patch_queue():
            auto = GauntletAutoImprove(enabled=True, cooldown_seconds=0)

            # Two rapid triggers should both work
            result1 = auto.on_run_complete(
                _make_result_with_findings([
                    _make_finding(finding_id="f-1", severity="critical"),
                ])
            )
            result2 = auto.on_run_complete(
                _make_result_with_findings(
                    [_make_finding(finding_id="f-2", severity="critical")],
                    gauntlet_id="gauntlet-002",
                )
            )

            assert result1.goals_queued == 1
            assert result2.goals_queued == 1


class TestAutoImproveResult:
    """Test the AutoImproveResult dataclass."""

    def test_total_findings_processed(self):
        """total_findings_processed should sum all categories."""
        result = AutoImproveResult(
            goals_queued=3,
            goals_skipped_severity=2,
            goals_skipped_duplicate=1,
            goals_skipped_rate_limit=4,
        )
        assert result.total_findings_processed == 10

    def test_empty_result(self):
        """Default result should have zero counts."""
        result = AutoImproveResult()
        assert result.goals_queued == 0
        assert result.total_findings_processed == 0
        assert result.error is None
        assert result.goal_descriptions == []


class TestRunnerIntegration:
    """Verify the runner wiring works."""

    def test_runner_accepts_auto_improve_param(self):
        """GauntletRunner should accept auto_improve parameter."""
        from aragora.gauntlet.runner import GauntletRunner

        runner = GauntletRunner(auto_improve=True)
        assert runner._auto_improve is True

    def test_runner_auto_improve_defaults_false(self):
        """GauntletRunner should default auto_improve to False."""
        from aragora.gauntlet.runner import GauntletRunner

        runner = GauntletRunner()
        assert runner._auto_improve is False


class TestGoalDescriptions:
    """Verify goal descriptions are captured for auditability."""

    def test_goal_descriptions_populated(self):
        """Goal descriptions should be captured in the result."""
        with _patch_queue():
            auto = GauntletAutoImprove(
                enabled=True, max_goals_per_run=10, cooldown_seconds=0
            )
            gauntlet_result = _make_result_with_findings([
                _make_finding(
                    finding_id="f-1",
                    severity="critical",
                    description="XSS in admin panel",
                    category="security",
                ),
            ])

            result = auto.on_run_complete(gauntlet_result)

            assert result.goals_queued == 1
            assert len(result.goal_descriptions) == 1
            assert "XSS in admin panel" in result.goal_descriptions[0]
