"""Tests for red team ELO integration."""

import pytest
from unittest.mock import MagicMock, patch

from aragora.ranking.redteam import (
    RedTeamResult,
    VulnerabilitySummary,
    RedTeamIntegrator,
    K_FACTOR,
)


class TestRedTeamResult:
    """Test RedTeamResult dataclass."""

    def test_create_result(self):
        """Test creating a red team result."""
        result = RedTeamResult(
            agent_name="claude",
            robustness_score=0.85,
            successful_attacks=2,
            total_attacks=10,
            critical_vulnerabilities=0,
            session_id="rt_001",
            elo_change=8.5,
        )

        assert result.agent_name == "claude"
        assert result.robustness_score == 0.85
        assert result.successful_attacks == 2
        assert result.total_attacks == 10
        assert result.critical_vulnerabilities == 0
        assert result.session_id == "rt_001"
        assert result.elo_change == 8.5

    def test_create_result_no_session(self):
        """Test result without session ID."""
        result = RedTeamResult(
            agent_name="gpt",
            robustness_score=0.3,
            successful_attacks=7,
            total_attacks=10,
            critical_vulnerabilities=3,
            session_id=None,
            elo_change=-15.0,
        )

        assert result.session_id is None
        assert result.elo_change == -15.0


class TestVulnerabilitySummary:
    """Test VulnerabilitySummary dataclass."""

    def test_create_summary(self):
        """Test creating a vulnerability summary."""
        summary = VulnerabilitySummary(
            redteam_sessions=5,
            total_elo_impact=-25.0,
            last_session="rt_005",
        )

        assert summary.redteam_sessions == 5
        assert summary.total_elo_impact == -25.0
        assert summary.last_session == "rt_005"

    def test_create_summary_no_sessions(self):
        """Test summary with no sessions."""
        summary = VulnerabilitySummary(
            redteam_sessions=0,
            total_elo_impact=0.0,
            last_session=None,
        )

        assert summary.redteam_sessions == 0
        assert summary.last_session is None


class TestRedTeamIntegratorInit:
    """Test RedTeamIntegrator initialization."""

    def test_init(self):
        """Test initialization."""
        mock_elo = MagicMock()
        integrator = RedTeamIntegrator(mock_elo)

        assert integrator.elo_system is mock_elo


class TestRedTeamIntegratorRecordResult:
    """Test record_result method."""

    @pytest.fixture
    def integrator(self):
        """Create integrator with mocked ELO system."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1200.0
        mock_elo.get_rating.return_value = mock_rating
        return RedTeamIntegrator(mock_elo), mock_elo, mock_rating

    def test_robust_agent_gets_boost(self, integrator):
        """Test robust agent (score >= 0.8) gets ELO boost."""
        integ, mock_elo, mock_rating = integrator

        elo_change = integ.record_result(
            agent_name="claude",
            robustness_score=0.9,
            successful_attacks=1,
            total_attacks=10,
        )

        # Should be positive (boost)
        assert elo_change > 0
        # K_FACTOR * 0.3 * 0.9 = 32 * 0.3 * 0.9 = 8.64
        expected = K_FACTOR * 0.3 * 0.9
        assert abs(elo_change - expected) < 0.01

        # Rating should have been updated
        mock_elo._save_rating.assert_called_once()

    def test_borderline_robust(self, integrator):
        """Test borderline robust agent (exactly 0.8)."""
        integ, mock_elo, mock_rating = integrator

        elo_change = integ.record_result(
            agent_name="claude",
            robustness_score=0.8,
            successful_attacks=2,
            total_attacks=10,
        )

        # Score of exactly 0.8 should get boost
        assert elo_change > 0

    def test_neutral_agent_no_change(self, integrator):
        """Test neutral agent (0.5-0.8) gets no change."""
        integ, mock_elo, mock_rating = integrator

        elo_change = integ.record_result(
            agent_name="claude",
            robustness_score=0.7,
            successful_attacks=3,
            total_attacks=10,
        )

        assert elo_change == 0.0
        # Should NOT save rating (no change)
        mock_elo._save_rating.assert_not_called()

    def test_vulnerable_agent_gets_penalty(self, integrator):
        """Test vulnerable agent (score < 0.5) gets penalty."""
        integ, mock_elo, mock_rating = integrator

        elo_change = integ.record_result(
            agent_name="claude",
            robustness_score=0.3,
            successful_attacks=7,
            total_attacks=10,
            critical_vulnerabilities=0,
        )

        assert elo_change < 0
        mock_elo._save_rating.assert_called_once()

    def test_critical_vulnerabilities_increase_penalty(self, integrator):
        """Test critical vulnerabilities increase penalty."""
        integ, mock_elo, mock_rating = integrator

        # Without critical vulns
        elo_change_no_critical = integ.record_result(
            agent_name="agent1",
            robustness_score=0.3,
            successful_attacks=5,
            total_attacks=10,
            critical_vulnerabilities=0,
        )

        # Reset mock
        mock_elo.reset_mock()

        # With critical vulns
        elo_change_critical = integ.record_result(
            agent_name="agent2",
            robustness_score=0.3,
            successful_attacks=5,
            total_attacks=10,
            critical_vulnerabilities=3,  # 3 critical = -6 additional
        )

        # Critical vulns should make penalty worse (more negative)
        assert elo_change_critical < elo_change_no_critical

    def test_penalty_capped_at_minus_30(self, integrator):
        """Test penalty is capped at -30."""
        integ, mock_elo, mock_rating = integrator

        elo_change = integ.record_result(
            agent_name="claude",
            robustness_score=0.1,
            successful_attacks=10,
            total_attacks=10,  # 100% vulnerability
            critical_vulnerabilities=20,  # Lots of critical
        )

        # Should be capped
        assert elo_change >= -30

    def test_zero_total_attacks(self, integrator):
        """Test handling zero total attacks."""
        integ, mock_elo, mock_rating = integrator

        elo_change = integ.record_result(
            agent_name="claude",
            robustness_score=0.3,
            successful_attacks=0,
            total_attacks=0,  # Edge case
        )

        # vulnerability_rate = 0 when total_attacks = 0
        # Should still apply penalty based on robustness
        assert elo_change <= 0

    def test_records_elo_history(self, integrator):
        """Test that ELO history is recorded."""
        integ, mock_elo, mock_rating = integrator

        integ.record_result(
            agent_name="claude",
            robustness_score=0.9,
            successful_attacks=1,
            total_attacks=10,
            session_id="rt_test",
        )

        mock_elo._record_elo_history.assert_called_once()
        # Should include "redteam_" prefix with session ID
        call_args = mock_elo._record_elo_history.call_args[0]
        assert call_args[0] == "claude"
        assert "redteam_rt_test" in call_args[2]

    def test_records_history_without_session_id(self, integrator):
        """Test ELO history recorded without session ID."""
        integ, mock_elo, mock_rating = integrator

        integ.record_result(
            agent_name="claude",
            robustness_score=0.9,
            successful_attacks=1,
            total_attacks=10,
            session_id=None,
        )

        mock_elo._record_elo_history.assert_called_once()
        call_args = mock_elo._record_elo_history.call_args[0]
        assert call_args[2] == "redteam"


class TestRedTeamIntegratorGetVulnerabilitySummary:
    """Test get_vulnerability_summary method."""

    def test_no_history(self):
        """Test summary with no red team history."""
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = []

        integrator = RedTeamIntegrator(mock_elo)
        summary = integrator.get_vulnerability_summary("claude")

        assert summary.redteam_sessions == 0
        assert summary.total_elo_impact == 0.0
        assert summary.last_session is None

    def test_with_redteam_history(self):
        """Test summary with red team history entries."""
        mock_elo = MagicMock()
        # History format: (timestamp/debate_id, elo)
        mock_elo.get_elo_history.return_value = [
            ("redteam_rt_003", 1190.0),  # Most recent
            ("debate_123", 1195.0),
            ("redteam_rt_002", 1195.0),
            ("debate_122", 1200.0),
            ("redteam_rt_001", 1200.0),
        ]

        integrator = RedTeamIntegrator(mock_elo)
        summary = integrator.get_vulnerability_summary("claude")

        assert summary.redteam_sessions == 3
        # Note: most recent is first in the list (we reverse to process chronologically)
        assert (
            summary.last_session == "redteam_rt_001"
        )  # Earliest session becomes last_session in reversed iteration

    def test_no_redteam_entries(self):
        """Test summary with history but no red team entries."""
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = [
            ("debate_123", 1210.0),
            ("debate_122", 1205.0),
            ("debate_121", 1200.0),
        ]

        integrator = RedTeamIntegrator(mock_elo)
        summary = integrator.get_vulnerability_summary("claude")

        assert summary.redteam_sessions == 0
        assert summary.total_elo_impact == 0.0


class TestRedTeamIntegratorCalculateEloAdjustment:
    """Test calculate_elo_adjustment method."""

    @pytest.fixture
    def integrator(self):
        """Create integrator."""
        mock_elo = MagicMock()
        return RedTeamIntegrator(mock_elo)

    def test_robust_adjustment(self, integrator):
        """Test adjustment for robust agent."""
        adjustment = integrator.calculate_elo_adjustment(
            robustness_score=0.9,
            vulnerability_rate=0.1,
        )

        expected = K_FACTOR * 0.3 * 0.9
        assert abs(adjustment - expected) < 0.01

    def test_exact_threshold_robust(self, integrator):
        """Test adjustment at exact 0.8 threshold."""
        adjustment = integrator.calculate_elo_adjustment(
            robustness_score=0.8,
            vulnerability_rate=0.2,
        )

        expected = K_FACTOR * 0.3 * 0.8
        assert abs(adjustment - expected) < 0.01

    def test_neutral_adjustment(self, integrator):
        """Test adjustment for neutral agent (0.5-0.8)."""
        adjustment = integrator.calculate_elo_adjustment(
            robustness_score=0.6,
            vulnerability_rate=0.4,
        )

        assert adjustment == 0.0

    def test_borderline_neutral(self, integrator):
        """Test adjustment at exactly 0.5 (borderline)."""
        adjustment = integrator.calculate_elo_adjustment(
            robustness_score=0.5,
            vulnerability_rate=0.5,
        )

        # 0.5 is in neutral range (not < 0.5)
        assert adjustment == 0.0

    def test_vulnerable_adjustment(self, integrator):
        """Test adjustment for vulnerable agent."""
        adjustment = integrator.calculate_elo_adjustment(
            robustness_score=0.3,
            vulnerability_rate=0.7,
        )

        assert adjustment < 0
        # Should be: -(K_FACTOR * 0.5 * 0.7) = -(32 * 0.5 * 0.7) = -11.2
        expected = -(K_FACTOR * 0.5 * 0.7)
        assert abs(adjustment - expected) < 0.01

    def test_vulnerable_with_critical(self, integrator):
        """Test adjustment includes critical vulnerability penalty."""
        adjustment = integrator.calculate_elo_adjustment(
            robustness_score=0.3,
            vulnerability_rate=0.5,
            critical_vulnerabilities=5,
        )

        # Base: -(K_FACTOR * 0.5 * 0.5) = -8
        # Critical: -5 * 2 = -10
        # Total: -18
        base = K_FACTOR * 0.5 * 0.5
        critical = 5 * 2
        expected = -(base + critical)
        assert abs(adjustment - expected) < 0.01

    def test_vulnerable_capped(self, integrator):
        """Test vulnerable adjustment is capped at -30."""
        adjustment = integrator.calculate_elo_adjustment(
            robustness_score=0.1,
            vulnerability_rate=1.0,  # 100% vulnerable
            critical_vulnerabilities=20,  # Lots of critical
        )

        assert adjustment == -30  # Capped


class TestRedTeamIntegration:
    """Integration tests for red team workflow."""

    def test_full_workflow(self):
        """Test complete red team workflow."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1200.0
        mock_elo.get_rating.return_value = mock_rating

        integrator = RedTeamIntegrator(mock_elo)

        # Preview the impact
        preview = integrator.calculate_elo_adjustment(
            robustness_score=0.9,
            vulnerability_rate=0.1,
        )
        assert preview > 0

        # Record the result
        actual = integrator.record_result(
            agent_name="claude",
            robustness_score=0.9,
            successful_attacks=1,
            total_attacks=10,
            session_id="test_session",
        )

        # Preview and actual should match
        assert abs(preview - actual) < 0.01

    def test_multiple_sessions(self):
        """Test multiple red team sessions accumulate."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1200.0
        mock_elo.get_rating.return_value = mock_rating

        integrator = RedTeamIntegrator(mock_elo)

        # Session 1: Robust
        change1 = integrator.record_result(
            agent_name="claude",
            robustness_score=0.9,
            successful_attacks=1,
            total_attacks=10,
        )
        assert change1 > 0

        # Session 2: Vulnerable
        change2 = integrator.record_result(
            agent_name="claude",
            robustness_score=0.3,
            successful_attacks=7,
            total_attacks=10,
        )
        assert change2 < 0

        # Both should have been recorded
        assert mock_elo._save_rating.call_count == 2
