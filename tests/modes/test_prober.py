"""
Tests for adversarial capability prober.

Tests cover:
- CapabilityProber class
- ProbeBeforePromote class
- generate_probe_report_markdown function
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.modes.prober import (
    CapabilityProber,
    ProbeBeforePromote,
    generate_probe_report_markdown,
)
from aragora.modes.probes import (
    ProbeResult,
    ProbeType,
    VulnerabilityReport,
    VulnerabilitySeverity,
)


class TestCapabilityProber:
    """Tests for CapabilityProber class."""

    def test_init_defaults(self):
        """Initializes with default values."""
        prober = CapabilityProber()

        assert prober.elo_system is None
        assert prober.elo_penalty_multiplier == 5.0
        assert prober._probe_counter == 0

    def test_init_with_elo_system(self):
        """Initializes with ELO system."""
        elo = MagicMock()
        prober = CapabilityProber(elo_system=elo, elo_penalty_multiplier=10.0)

        assert prober.elo_system is elo
        assert prober.elo_penalty_multiplier == 10.0

    def test_has_strategies(self):
        """Has probe strategies defined."""
        prober = CapabilityProber()

        assert prober.STRATEGIES is not None
        assert len(prober.STRATEGIES) > 0

    @pytest.mark.asyncio
    async def test_probe_agent_basic(self):
        """Probes an agent and returns report."""
        prober = CapabilityProber()
        agent = MagicMock(name="test-agent")

        async def mock_run(agent, prompt):
            return "I agree with that assessment."

        report = await prober.probe_agent(
            target_agent=agent,
            run_agent_fn=mock_run,
            probe_types=[ProbeType.SYCOPHANCY],
            probes_per_type=1,
        )

        assert report is not None
        assert isinstance(report, VulnerabilityReport)
        assert report.probes_run >= 1

    @pytest.mark.asyncio
    async def test_probe_agent_multiple_types(self):
        """Probes with multiple probe types."""
        prober = CapabilityProber()
        agent = MagicMock(name="test-agent")

        async def mock_run(agent, prompt):
            return "Test response"

        report = await prober.probe_agent(
            target_agent=agent,
            run_agent_fn=mock_run,
            probe_types=[ProbeType.SYCOPHANCY, ProbeType.CONTRADICTION],
            probes_per_type=2,
        )

        assert report.probes_run >= 4  # 2 types * 2 probes each

    @pytest.mark.asyncio
    async def test_probe_agent_all_types(self):
        """Probes with all types when none specified."""
        prober = CapabilityProber()
        agent = MagicMock(name="test-agent")

        async def mock_run(agent, prompt):
            return "Test response"

        report = await prober.probe_agent(
            target_agent=agent,
            run_agent_fn=mock_run,
            probes_per_type=1,
        )

        # Should have probed with all available types
        assert report.probes_run >= len(prober.STRATEGIES)

    @pytest.mark.asyncio
    async def test_probe_agent_with_context(self):
        """Probes with conversation context."""
        prober = CapabilityProber()
        agent = MagicMock(name="test-agent")

        context = [
            MagicMock(content="Previous message"),
        ]

        async def mock_run(agent, prompt):
            return "Response considering context"

        report = await prober.probe_agent(
            target_agent=agent,
            run_agent_fn=mock_run,
            probe_types=[ProbeType.PERSISTENCE],
            probes_per_type=1,
            context=context,
        )

        assert report is not None

    @pytest.mark.asyncio
    async def test_probe_agent_handles_exceptions(self):
        """Handles exceptions from agent gracefully."""
        prober = CapabilityProber()
        agent = MagicMock(name="failing-agent")

        async def failing_run(agent, prompt):
            raise RuntimeError("Agent failed")

        report = await prober.probe_agent(
            target_agent=agent,
            run_agent_fn=failing_run,
            probe_types=[ProbeType.SYCOPHANCY],
            probes_per_type=1,
        )

        # Should still return report, error captured
        assert report is not None
        assert report.probes_run >= 1

    def test_generate_report(self):
        """Generates report from probe results."""
        prober = CapabilityProber()

        results = [
            ProbeResult(
                probe_id="p1",
                probe_type=ProbeType.SYCOPHANCY,
                target_agent="agent",
                probe_prompt="Test",
                agent_response="Response",
                vulnerability_found=True,
                severity=VulnerabilitySeverity.HIGH,
            ),
            ProbeResult(
                probe_id="p2",
                probe_type=ProbeType.CONTRADICTION,
                target_agent="agent",
                probe_prompt="Test2",
                agent_response="Response2",
                vulnerability_found=False,
            ),
        ]

        by_type = {
            ProbeType.SYCOPHANCY.value: [results[0]],
            ProbeType.CONTRADICTION.value: [results[1]],
        }

        report = prober._generate_report("test-agent", results, by_type)

        assert report.target_agent == "test-agent"
        assert report.probes_run == 2
        assert report.vulnerabilities_found == 1
        assert report.high_count == 1

    def test_generate_report_vulnerability_rate(self):
        """Report calculates vulnerability rate correctly."""
        prober = CapabilityProber()

        results = [
            ProbeResult(
                probe_id=f"p{i}",
                probe_type=ProbeType.SYCOPHANCY,
                target_agent="agent",
                probe_prompt="",
                agent_response="",
                vulnerability_found=(i < 3),  # 3 vulnerabilities
            )
            for i in range(5)
        ]

        report = prober._generate_report("agent", results, {})

        assert report.vulnerability_rate == 0.6  # 3/5

    def test_generate_report_severity_counts(self):
        """Report counts severities correctly."""
        prober = CapabilityProber()

        results = [
            ProbeResult(
                probe_id="p1",
                probe_type=ProbeType.SYCOPHANCY,
                target_agent="agent",
                probe_prompt="",
                agent_response="",
                vulnerability_found=True,
                severity=VulnerabilitySeverity.CRITICAL,
            ),
            ProbeResult(
                probe_id="p2",
                probe_type=ProbeType.CONTRADICTION,
                target_agent="agent",
                probe_prompt="",
                agent_response="",
                vulnerability_found=True,
                severity=VulnerabilitySeverity.HIGH,
            ),
            ProbeResult(
                probe_id="p3",
                probe_type=ProbeType.PERSISTENCE,
                target_agent="agent",
                probe_prompt="",
                agent_response="",
                vulnerability_found=True,
                severity=VulnerabilitySeverity.MEDIUM,
            ),
        ]

        report = prober._generate_report("agent", results, {})

        assert report.critical_count == 1
        assert report.high_count == 1
        assert report.medium_count == 1
        assert report.low_count == 0

    def test_generate_report_recommendations(self):
        """Report includes recommendations for vulnerabilities."""
        prober = CapabilityProber()

        results = [
            ProbeResult(
                probe_id="p1",
                probe_type=ProbeType.HALLUCINATION,
                target_agent="agent",
                probe_prompt="",
                agent_response="",
                vulnerability_found=True,
                severity=VulnerabilitySeverity.CRITICAL,
            ),
        ]

        report = prober._generate_report("agent", results, {})

        assert len(report.recommendations) >= 1
        assert "CRITICAL" in report.recommendations[0]

    def test_generate_report_elo_penalty(self):
        """Report calculates ELO penalty."""
        prober = CapabilityProber(elo_penalty_multiplier=5.0)

        results = [
            ProbeResult(
                probe_id="p1",
                probe_type=ProbeType.SYCOPHANCY,
                target_agent="agent",
                probe_prompt="",
                agent_response="",
                vulnerability_found=True,
                severity=VulnerabilitySeverity.CRITICAL,  # 30 points
            ),
        ]

        report = prober._generate_report("agent", results, {})

        # 30 * 5.0 / 10 = 15
        assert report.elo_penalty == 15.0

    @pytest.mark.asyncio
    async def test_probe_agent_applies_elo_penalty(self):
        """Applies ELO penalty when system is available."""
        elo = MagicMock()
        rating = MagicMock(elo=1500)
        elo.get_rating.return_value = rating

        prober = CapabilityProber(elo_system=elo)
        agent = MagicMock()
        agent.name = "test-agent"

        async def mock_run(agent, prompt):
            # Response that triggers sycophancy detection
            return "Yes, you're absolutely right! I completely agree!"

        report = await prober.probe_agent(
            target_agent=agent,
            run_agent_fn=mock_run,
            probe_types=[ProbeType.SYCOPHANCY],
            probes_per_type=1,
        )

        # If vulnerabilities found with penalty, ELO should be applied
        if report.elo_penalty > 0:
            elo.get_rating.assert_called_with("test-agent")


class TestProbeBeforePromote:
    """Tests for ProbeBeforePromote class."""

    @pytest.fixture
    def mock_elo_system(self):
        """Create mock ELO system."""
        elo = MagicMock()
        rating = MagicMock(elo=1500)
        elo.get_rating.return_value = rating
        return elo

    @pytest.fixture
    def mock_prober(self):
        """Create mock prober."""
        prober = MagicMock(spec=CapabilityProber)
        return prober

    def test_init(self, mock_elo_system, mock_prober):
        """Initializes with parameters."""
        gate = ProbeBeforePromote(
            elo_system=mock_elo_system,
            prober=mock_prober,
            max_vulnerability_rate=0.2,
            max_critical=0,
        )

        assert gate.elo_system is mock_elo_system
        assert gate.prober is mock_prober
        assert gate.max_vulnerability_rate == 0.2
        assert gate.max_critical == 0
        assert gate.pending_promotions == {}

    @pytest.mark.asyncio
    async def test_check_promotion_approved(self, mock_elo_system, mock_prober):
        """Approves promotion when probing passes."""
        # Mock clean report
        clean_report = VulnerabilityReport(
            report_id="r1",
            target_agent="agent",
            probes_run=10,
            vulnerabilities_found=1,
            by_type={},
            vulnerability_rate=0.1,
            critical_count=0,
            high_count=1,
        )
        mock_prober.probe_agent = AsyncMock(return_value=clean_report)

        gate = ProbeBeforePromote(
            elo_system=mock_elo_system,
            prober=mock_prober,
            max_vulnerability_rate=0.2,
            max_critical=0,
        )

        agent = MagicMock()
        agent.name = "agent"

        async def mock_run(a, p):
            return "response"

        approved, report = await gate.check_promotion(agent, mock_run, 50.0)

        assert approved is True
        assert report is clean_report
        # ELO gain should be applied
        mock_elo_system.get_rating.assert_called_with("agent")

    @pytest.mark.asyncio
    async def test_check_promotion_rejected_high_vuln_rate(self, mock_elo_system, mock_prober):
        """Rejects promotion when vulnerability rate too high."""
        bad_report = VulnerabilityReport(
            report_id="r1",
            target_agent="agent",
            probes_run=10,
            vulnerabilities_found=5,
            by_type={},
            vulnerability_rate=0.5,  # 50% > 20% threshold
            critical_count=0,
        )
        mock_prober.probe_agent = AsyncMock(return_value=bad_report)

        gate = ProbeBeforePromote(
            elo_system=mock_elo_system,
            prober=mock_prober,
            max_vulnerability_rate=0.2,
            max_critical=0,
        )

        agent = MagicMock()
        agent.name = "agent"

        async def mock_run(a, p):
            return "response"

        approved, report = await gate.check_promotion(agent, mock_run, 50.0)

        assert approved is False
        # Pending promotion stored
        assert "agent" in gate.pending_promotions
        assert gate.pending_promotions["agent"] == 50.0

    @pytest.mark.asyncio
    async def test_check_promotion_rejected_critical(self, mock_elo_system, mock_prober):
        """Rejects promotion when critical vulnerabilities found."""
        critical_report = VulnerabilityReport(
            report_id="r1",
            target_agent="agent",
            probes_run=10,
            vulnerabilities_found=1,
            by_type={},
            vulnerability_rate=0.1,
            critical_count=1,  # > 0 threshold
        )
        mock_prober.probe_agent = AsyncMock(return_value=critical_report)

        gate = ProbeBeforePromote(
            elo_system=mock_elo_system,
            prober=mock_prober,
            max_vulnerability_rate=0.2,
            max_critical=0,
        )

        agent = MagicMock()
        agent.name = "agent"

        async def mock_run(a, p):
            return "response"

        approved, report = await gate.check_promotion(agent, mock_run, 50.0)

        assert approved is False

    @pytest.mark.asyncio
    async def test_retry_promotion(self, mock_elo_system, mock_prober):
        """Retries pending promotion."""
        # First rejection
        bad_report = VulnerabilityReport(
            report_id="r1",
            target_agent="agent",
            probes_run=10,
            vulnerabilities_found=5,
            by_type={},
            vulnerability_rate=0.5,
            critical_count=0,
        )
        mock_prober.probe_agent = AsyncMock(return_value=bad_report)

        gate = ProbeBeforePromote(
            elo_system=mock_elo_system,
            prober=mock_prober,
        )

        agent = MagicMock()
        agent.name = "agent"

        async def mock_run(a, p):
            return "response"

        await gate.check_promotion(agent, mock_run, 50.0)
        assert "agent" in gate.pending_promotions

        # Now mock clean report for retry
        clean_report = VulnerabilityReport(
            report_id="r2",
            target_agent="agent",
            probes_run=10,
            vulnerabilities_found=1,
            by_type={},
            vulnerability_rate=0.1,
            critical_count=0,
        )
        mock_prober.probe_agent = AsyncMock(return_value=clean_report)

        approved, report = await gate.retry_promotion(agent, mock_run)

        assert approved is True
        assert "agent" not in gate.pending_promotions

    @pytest.mark.asyncio
    async def test_retry_promotion_no_pending(self, mock_elo_system, mock_prober):
        """Retry with no pending promotion returns True."""
        gate = ProbeBeforePromote(
            elo_system=mock_elo_system,
            prober=mock_prober,
        )

        agent = MagicMock()
        agent.name = "new-agent"

        async def mock_run(a, p):
            return "response"

        approved, report = await gate.retry_promotion(agent, mock_run)

        assert approved is True
        assert report is None


class TestGenerateProbeReportMarkdown:
    """Tests for generate_probe_report_markdown function."""

    def test_basic_report(self):
        """Generates basic markdown report."""
        report = VulnerabilityReport(
            report_id="test-123",
            target_agent="claude",
            probes_run=10,
            vulnerabilities_found=3,
            by_type={},
            vulnerability_rate=0.3,
            critical_count=1,
            high_count=1,
            medium_count=1,
            low_count=0,
            elo_penalty=15.0,
        )

        markdown = generate_probe_report_markdown(report)

        assert "# Capability Probe Report: claude" in markdown
        assert "test-123" in markdown
        assert "Probes Run | 10" in markdown
        assert "Vulnerabilities | 3" in markdown
        assert "Vulnerability Rate | 30.0%" in markdown
        assert "ELO Penalty | 15.0" in markdown

    def test_report_with_recommendations(self):
        """Includes recommendations in report."""
        report = VulnerabilityReport(
            report_id="r1",
            target_agent="agent",
            probes_run=5,
            vulnerabilities_found=2,
            by_type={},
            vulnerability_rate=0.4,
            recommendations=["Fix sycophancy", "Improve persistence"],
        )

        markdown = generate_probe_report_markdown(report)

        assert "## Recommendations" in markdown
        assert "Fix sycophancy" in markdown
        assert "Improve persistence" in markdown

    def test_report_with_probe_details(self):
        """Includes probe type details."""
        vulnerability = ProbeResult(
            probe_id="p1",
            probe_type=ProbeType.SYCOPHANCY,
            target_agent="agent",
            probe_prompt="Test",
            agent_response="Agreed!",
            vulnerability_found=True,
            vulnerability_description="Agent agreed without evidence",
            severity=VulnerabilitySeverity.HIGH,
            evidence="Response: Agreed!",
        )

        report = VulnerabilityReport(
            report_id="r1",
            target_agent="agent",
            probes_run=1,
            vulnerabilities_found=1,
            by_type={ProbeType.SYCOPHANCY.value: [vulnerability]},
            vulnerability_rate=1.0,
        )

        markdown = generate_probe_report_markdown(report)

        assert "## Details by Probe Type" in markdown
        assert "Sycophancy" in markdown
        assert "Agent agreed without evidence" in markdown
        assert "HIGH" in markdown

    def test_report_severity_breakdown(self):
        """Includes severity breakdown."""
        report = VulnerabilityReport(
            report_id="r1",
            target_agent="agent",
            probes_run=10,
            vulnerabilities_found=6,
            by_type={},
            vulnerability_rate=0.6,
            critical_count=2,
            high_count=2,
            medium_count=1,
            low_count=1,
        )

        markdown = generate_probe_report_markdown(report)

        assert "### Severity Breakdown" in markdown
        assert "Critical: 2" in markdown
        assert "High: 2" in markdown
        assert "Medium: 1" in markdown
        assert "Low: 1" in markdown
