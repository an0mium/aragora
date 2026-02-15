"""Tests for pre-debate compliance policy check.

Covers:
- CRITICAL issue blocks debate (allowed=False)
- HIGH severity logs warning but allows
- No monitor available = skip (allowed=True)
- Multiple frameworks checked
- check_pre_debate_compliance returns correct structure
- Domain filtering works
- Error in monitor doesn't block debate
- Integration with orchestrator_runner budget flow
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.extensions import (
    ComplianceCheckResult,
    check_pre_debate_compliance,
)


@dataclass
class _FakeRule:
    name: str = "test-rule"
    description: str = "Test compliance rule"
    severity: str = "high"
    enabled: bool = True
    domains: list[str] | None = None
    task_pattern: str | None = None


@dataclass
class _FakeFramework:
    name: str = "SOC2"
    rules: list = field(default_factory=list)
    restricted_domains: list[str] | None = None
    blocked_task_patterns: list[str] = field(default_factory=list)


class TestComplianceCheckResult:
    """Test ComplianceCheckResult dataclass."""

    def test_defaults(self):
        result = ComplianceCheckResult()
        assert result.allowed is True
        assert result.issues == []
        assert result.warnings == []
        assert result.frameworks_checked == []


class TestCheckPreDebateCompliance:
    """Test check_pre_debate_compliance function."""

    def test_critical_blocks_debate(self):
        """CRITICAL severity rule makes allowed=False."""
        rule = _FakeRule(
            name="pii-disclosure",
            description="PII disclosure not allowed",
            severity="critical",
            task_pattern="share personal data",
        )
        fw = _FakeFramework(name="GDPR", rules=[rule])
        monitor = MagicMock()
        monitor.active_frameworks = [fw]

        result = check_pre_debate_compliance(
            debate_id="test-001",
            task="Please share personal data from users",
            domain="general",
            compliance_monitor=monitor,
        )

        assert result.allowed is False
        assert len(result.issues) == 1
        assert "pii-disclosure" in result.issues[0]
        assert "GDPR" in result.frameworks_checked

    def test_high_severity_warns_but_allows(self):
        """HIGH severity rule adds warning but doesn't block."""
        rule = _FakeRule(
            name="data-retention",
            description="Data retention policy check",
            severity="high",
            task_pattern="delete records",
        )
        fw = _FakeFramework(name="SOC2", rules=[rule])
        monitor = MagicMock()
        monitor.active_frameworks = [fw]

        result = check_pre_debate_compliance(
            debate_id="test-002",
            task="Should we delete records older than 30 days?",
            domain="general",
            compliance_monitor=monitor,
        )

        assert result.allowed is True
        assert len(result.warnings) == 1
        assert "data-retention" in result.warnings[0]

    def test_no_monitor_skips(self):
        """When no monitor available, returns allowed=True."""
        with patch.dict("sys.modules", {"aragora.compliance.monitor": None}):
            result = check_pre_debate_compliance(
                debate_id="test-003",
                task="Any task",
                compliance_monitor=None,
            )
        assert result.allowed is True
        assert result.frameworks_checked == []

    def test_multiple_frameworks_checked(self):
        """All active frameworks are evaluated."""
        fw1 = _FakeFramework(name="GDPR", rules=[])
        fw2 = _FakeFramework(name="SOC2", rules=[])
        fw3 = _FakeFramework(name="HIPAA", rules=[])
        monitor = MagicMock()
        monitor.active_frameworks = [fw1, fw2, fw3]

        result = check_pre_debate_compliance(
            debate_id="test-004",
            task="Regular debate task",
            compliance_monitor=monitor,
        )

        assert result.allowed is True
        assert result.frameworks_checked == ["GDPR", "SOC2", "HIPAA"]

    def test_domain_filtering(self):
        """Rules with domain restrictions only apply to matching domains."""
        rule = _FakeRule(
            name="hipaa-check",
            description="HIPAA-specific check",
            severity="critical",
            domains=["healthcare"],
            task_pattern="patient",
        )
        fw = _FakeFramework(name="HIPAA", rules=[rule])
        monitor = MagicMock()
        monitor.active_frameworks = [fw]

        # Non-healthcare domain: rule should not apply
        result = check_pre_debate_compliance(
            debate_id="test-005",
            task="Discuss patient data handling",
            domain="financial",
            compliance_monitor=monitor,
        )
        assert result.allowed is True

        # Healthcare domain: rule should apply
        result = check_pre_debate_compliance(
            debate_id="test-005b",
            task="Discuss patient data handling",
            domain="healthcare",
            compliance_monitor=monitor,
        )
        assert result.allowed is False

    def test_error_in_monitor_does_not_block(self):
        """Exception in compliance check adds warning but allows debate."""
        monitor = MagicMock()
        monitor.active_frameworks = MagicMock(side_effect=RuntimeError("monitor error"))
        # active_frameworks as property that raises
        type(monitor).active_frameworks = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("monitor error"))
        )

        result = check_pre_debate_compliance(
            debate_id="test-006",
            task="Any task",
            compliance_monitor=monitor,
        )

        assert result.allowed is True
        assert len(result.warnings) == 1
        assert "error" in result.warnings[0].lower()

    def test_disabled_rules_skipped(self):
        """Disabled rules are not evaluated."""
        rule = _FakeRule(
            name="disabled-rule",
            severity="critical",
            task_pattern="anything",
            enabled=False,
        )
        fw = _FakeFramework(name="SOC2", rules=[rule])
        monitor = MagicMock()
        monitor.active_frameworks = [fw]

        result = check_pre_debate_compliance(
            debate_id="test-007",
            task="anything goes here",
            compliance_monitor=monitor,
        )

        assert result.allowed is True
        assert len(result.issues) == 0

    def test_restricted_domain_blocks(self):
        """Framework with restricted_domains + blocked_task_patterns blocks matching tasks."""
        fw = _FakeFramework(
            name="Export-Control",
            restricted_domains=["defense"],
            blocked_task_patterns=["classified", "export"],
        )
        monitor = MagicMock()
        monitor.active_frameworks = [fw]

        result = check_pre_debate_compliance(
            debate_id="test-008",
            task="Discuss classified information export",
            domain="defense",
            compliance_monitor=monitor,
        )

        assert result.allowed is False
        assert len(result.issues) >= 1
