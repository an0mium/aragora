"""
Tests for ProbeFilter class in aragora.routing.probe_filter.

Tests cover:
- ProbeProfile dataclass methods
- ProbeFilter initialization
- Probe data loading from filesystem
- Agent filtering based on vulnerability rates
- Team score calculation
- Role recommendation based on probe history
- Cache management with TTL and mtime invalidation
- Edge cases and error handling
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.routing.probe_filter import ProbeFilter, ProbeProfile


# =============================================================================
# TestProbeProfile - Dataclass Tests
# =============================================================================


class TestProbeProfileCreation:
    """Tests for ProbeProfile creation and defaults."""

    def test_minimal_creation(self):
        """Should create with just agent_name."""
        profile = ProbeProfile(agent_name="test_agent")

        assert profile.agent_name == "test_agent"
        assert profile.vulnerability_rate == 0.0
        assert profile.probe_score == 1.0
        assert profile.total_probes == 0
        assert profile.critical_count == 0
        assert profile.high_count == 0
        assert profile.medium_count == 0
        assert profile.low_count == 0

    def test_full_creation(self):
        """Should accept all fields."""
        profile = ProbeProfile(
            agent_name="full_agent",
            vulnerability_rate=0.3,
            probe_score=0.7,
            critical_count=2,
            high_count=5,
            medium_count=10,
            low_count=20,
            dominant_weakness="prompt_injection",
            weakness_counts={"prompt_injection": 5, "jailbreak": 3},
            total_probes=100,
            last_probe_date="2025-01-15T10:00:00Z",
            days_since_probe=5,
            report_count=3,
            recommendations=["Improve input validation"],
        )

        assert profile.vulnerability_rate == 0.3
        assert profile.dominant_weakness == "prompt_injection"
        assert len(profile.recommendations) == 1


class TestProbeProfileMethods:
    """Tests for ProbeProfile methods."""

    def test_is_stale_fresh_data(self):
        """is_stale should return False for recent data."""
        profile = ProbeProfile(agent_name="test", days_since_probe=3)

        assert profile.is_stale(max_days=7) is False

    def test_is_stale_old_data(self):
        """is_stale should return True for old data."""
        profile = ProbeProfile(agent_name="test", days_since_probe=10)

        assert profile.is_stale(max_days=7) is True

    def test_is_stale_default_threshold(self):
        """is_stale should use 7 days as default."""
        profile = ProbeProfile(agent_name="test", days_since_probe=999)

        assert profile.is_stale() is True

    def test_is_high_risk_below_threshold(self):
        """is_high_risk should return False below threshold."""
        profile = ProbeProfile(agent_name="test", vulnerability_rate=0.3)

        assert profile.is_high_risk(threshold=0.4) is False

    def test_is_high_risk_above_threshold(self):
        """is_high_risk should return True above threshold."""
        profile = ProbeProfile(agent_name="test", vulnerability_rate=0.5)

        assert profile.is_high_risk(threshold=0.4) is True

    def test_has_critical_issues_none(self):
        """has_critical_issues should return False with no criticals."""
        profile = ProbeProfile(agent_name="test", critical_count=0)

        assert profile.has_critical_issues() is False

    def test_has_critical_issues_present(self):
        """has_critical_issues should return True with criticals."""
        profile = ProbeProfile(agent_name="test", critical_count=1)

        assert profile.has_critical_issues() is True

    def test_to_dict(self):
        """to_dict should serialize profile to dict."""
        profile = ProbeProfile(
            agent_name="test",
            vulnerability_rate=0.2,
            probe_score=0.8,
            critical_count=1,
            total_probes=50,
        )

        data = profile.to_dict()

        assert data["agent_name"] == "test"
        assert data["vulnerability_rate"] == 0.2
        assert data["probe_score"] == 0.8
        assert data["critical_count"] == 1
        assert data["total_probes"] == 50


# =============================================================================
# TestProbeFilterInit - Initialization Tests
# =============================================================================


class TestProbeFilterInit:
    """Tests for ProbeFilter initialization."""

    def test_default_initialization(self):
        """Should initialize with default paths."""
        probe_filter = ProbeFilter()

        assert probe_filter.nomic_dir == Path(".nomic")
        assert probe_filter.probes_dir == Path(".nomic/probes")
        assert probe_filter.cache_ttl_seconds == 300

    def test_custom_nomic_dir(self):
        """Should accept custom nomic directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            assert probe_filter.nomic_dir == Path(temp_dir)
            assert probe_filter.probes_dir == Path(temp_dir) / "probes"

    def test_custom_cache_ttl(self):
        """Should accept custom cache TTL."""
        probe_filter = ProbeFilter(cache_ttl_seconds=600)

        assert probe_filter.cache_ttl_seconds == 600


# =============================================================================
# TestProbeFilterLoadProfile - Profile Loading
# =============================================================================


class TestProbeFilterLoadProfile:
    """Tests for ProbeFilter profile loading."""

    def test_get_agent_profile_no_data(self):
        """Should return default profile for agent without data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            profile = probe_filter.get_agent_profile("nonexistent")

            assert profile.agent_name == "nonexistent"
            assert profile.probe_score == 1.0
            assert profile.total_probes == 0

    def test_get_agent_profile_with_data(self):
        """Should load profile from probe reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create probe data
            probes_dir = Path(temp_dir) / "probes" / "test_agent"
            probes_dir.mkdir(parents=True)

            report = {
                "probes_run": 10,
                "vulnerabilities_found": 3,
                "breakdown": {
                    "critical": 1,
                    "high": 1,
                    "medium": 1,
                    "low": 0,
                },
                "by_type": {
                    "prompt_injection": [{"vulnerability_found": True}],
                },
                "recommendations": ["Improve input validation"],
                "created_at": datetime.now().isoformat(),
            }
            (probes_dir / "report_001.json").write_text(json.dumps(report))

            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            profile = probe_filter.get_agent_profile("test_agent")

            assert profile.agent_name == "test_agent"
            assert profile.total_probes == 10
            assert profile.vulnerability_rate == 0.3
            assert profile.probe_score == 0.7
            assert profile.critical_count == 1
            assert profile.high_count == 1

    def test_aggregates_multiple_reports(self):
        """Should aggregate metrics from multiple reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "multi_report"
            probes_dir.mkdir(parents=True)

            # First report
            (probes_dir / "report_001.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 2,
                        "breakdown": {"critical": 0, "high": 1, "medium": 1, "low": 0},
                        "by_type": {},
                        "created_at": "2025-01-01T00:00:00Z",
                    }
                )
            )

            # Second report
            (probes_dir / "report_002.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 4,
                        "breakdown": {"critical": 2, "high": 1, "medium": 1, "low": 0},
                        "by_type": {},
                        "created_at": "2025-01-10T00:00:00Z",
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            profile = probe_filter.get_agent_profile("multi_report")

            # Should sum across reports
            assert profile.total_probes == 20
            assert profile.vulnerability_rate == 0.3  # 6/20
            assert profile.critical_count == 2
            assert profile.high_count == 2
            assert profile.report_count == 2

    def test_finds_dominant_weakness(self):
        """Should identify dominant weakness type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "weakness_test"
            probes_dir.mkdir(parents=True)

            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 5,
                        "breakdown": {},
                        "by_type": {
                            "prompt_injection": [
                                {"vulnerability_found": True},
                                {"vulnerability_found": True},
                                {"vulnerability_found": True},
                            ],
                            "jailbreak": [
                                {"vulnerability_found": True},
                                {"vulnerability_found": False},
                            ],
                        },
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            profile = probe_filter.get_agent_profile("weakness_test")

            assert profile.dominant_weakness == "prompt_injection"
            assert profile.weakness_counts["prompt_injection"] == 3

    def test_handles_malformed_report(self):
        """Should handle malformed JSON reports gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "malformed"
            probes_dir.mkdir(parents=True)

            (probes_dir / "bad_report.json").write_text("not valid json")

            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            profile = probe_filter.get_agent_profile("malformed")

            # Should return default profile
            assert profile.probe_score == 1.0
            assert profile.total_probes == 0


# =============================================================================
# TestProbeFilterCaching - Cache Behavior
# =============================================================================


class TestProbeFilterCaching:
    """Tests for ProbeFilter caching behavior."""

    def test_caches_profile(self):
        """Should cache loaded profiles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir, cache_ttl_seconds=300)

            profile1 = probe_filter.get_agent_profile("test")
            profile2 = probe_filter.get_agent_profile("test")

            # Should be the same object
            assert profile1 is profile2

    def test_cache_respects_ttl(self):
        """Should expire cache after TTL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir, cache_ttl_seconds=0)

            profile1 = probe_filter.get_agent_profile("test")
            time.sleep(0.01)
            profile2 = probe_filter.get_agent_profile("test")

            # Should be different objects (reloaded after TTL)
            assert profile1 is not profile2

    def test_cache_invalidation_on_file_change(self):
        """Should invalidate cache when probe files are modified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "mtime_test"
            probes_dir.mkdir(parents=True)

            # Initial report
            report_file = probes_dir / "report.json"
            report_file.write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 2,
                        "breakdown": {},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir, cache_ttl_seconds=300)

            profile1 = probe_filter.get_agent_profile("mtime_test")
            assert profile1.vulnerability_rate == 0.2

            # Modify the report
            time.sleep(0.1)  # Ensure different mtime
            report_file.write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 5,
                        "breakdown": {},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            profile2 = probe_filter.get_agent_profile("mtime_test")
            # Should be reloaded with new data
            assert profile2.vulnerability_rate == 0.5

    def test_clear_cache(self):
        """clear_cache should empty the cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            probe_filter.get_agent_profile("test1")
            probe_filter.get_agent_profile("test2")

            assert len(probe_filter._profile_cache) == 2

            probe_filter.clear_cache()

            assert len(probe_filter._profile_cache) == 0


# =============================================================================
# TestProbeFilterFilterAgents - Agent Filtering
# =============================================================================


class TestProbeFilterFilterAgents:
    """Tests for ProbeFilter.filter_agents() method."""

    def test_includes_agents_without_probe_data(self):
        """Should include agents without probe data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            filtered = probe_filter.filter_agents(
                candidates=["unknown1", "unknown2"],
                max_vulnerability_rate=0.3,
            )

            assert filtered == ["unknown1", "unknown2"]

    def test_filters_by_vulnerability_rate(self):
        """Should filter agents exceeding vulnerability rate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create high-vulnerability agent
            probes_dir = Path(temp_dir) / "probes" / "risky"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 5,  # 50%
                        "breakdown": {},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            filtered = probe_filter.filter_agents(
                candidates=["safe", "risky"],
                max_vulnerability_rate=0.3,
            )

            assert "safe" in filtered
            assert "risky" not in filtered

    def test_excludes_critical_when_requested(self):
        """Should exclude agents with critical issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "critical_agent"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 1,  # Low rate
                        "breakdown": {"critical": 1},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            filtered = probe_filter.filter_agents(
                candidates=["safe", "critical_agent"],
                max_vulnerability_rate=0.5,
                exclude_critical=True,
            )

            assert "safe" in filtered
            assert "critical_agent" not in filtered

    def test_excludes_stale_when_requested(self):
        """Should exclude agents with stale probe data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "stale_agent"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 1,
                        "breakdown": {},
                        "by_type": {},
                        "created_at": "2020-01-01T00:00:00Z",  # Very old
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            filtered = probe_filter.filter_agents(
                candidates=["fresh", "stale_agent"],
                max_vulnerability_rate=0.5,
                exclude_stale=True,
                stale_days=30,
            )

            assert "fresh" in filtered
            assert "stale_agent" not in filtered


# =============================================================================
# TestProbeFilterTeamScores - Score Calculation
# =============================================================================


class TestProbeFilterTeamScores:
    """Tests for ProbeFilter.get_team_scores() method."""

    def test_base_score_for_unknown_agents(self):
        """Should return base score for agents without data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            scores = probe_filter.get_team_scores(
                candidates=["unknown1", "unknown2"],
                base_score=1.0,
            )

            assert scores["unknown1"] == 1.0
            assert scores["unknown2"] == 1.0

    def test_scores_based_on_probe_score(self):
        """Should calculate scores based on probe performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "probed"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 3,  # 30% = 0.7 score
                        "breakdown": {},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            scores = probe_filter.get_team_scores(["probed"])

            assert scores["probed"] == pytest.approx(0.7, rel=0.01)

    def test_critical_penalty(self):
        """Should apply penalty for critical issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "critical"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 2,  # 20% = 0.8 score
                        "breakdown": {"critical": 1},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            scores = probe_filter.get_team_scores(["critical"])

            # 0.8 * 0.5 = 0.4 (critical penalty)
            assert scores["critical"] == pytest.approx(0.4, rel=0.01)

    def test_stale_data_blends_to_base(self):
        """Should blend stale data toward base score."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "stale"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 1,  # 10% = 0.9 score
                        "breakdown": {},
                        "by_type": {},
                        "created_at": "2020-01-01T00:00:00Z",  # Very old
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            scores = probe_filter.get_team_scores(["stale"], base_score=1.0)

            # Score should be blended: (0.9 + 1.0) / 2 = 0.95
            assert scores["stale"] == pytest.approx(0.95, rel=0.01)

    def test_minimum_score(self):
        """Should not go below minimum score of 0.1."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "terrible"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 10,  # 100% = 0 score
                        "breakdown": {"critical": 5},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            scores = probe_filter.get_team_scores(["terrible"])

            # Should be clamped to minimum
            assert scores["terrible"] >= 0.1


# =============================================================================
# TestProbeFilterRoleRecommendation - Role Recommendations
# =============================================================================


class TestProbeFilterRoleRecommendation:
    """Tests for ProbeFilter.get_role_recommendation() method."""

    def test_no_data_returns_proposer(self):
        """Should recommend proposer for agents without data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            role = probe_filter.get_role_recommendation("unknown")

            assert role == "proposer"

    def test_critical_issues_returns_observer(self):
        """Should recommend observer for agents with critical issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "critical"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 1,
                        "breakdown": {"critical": 1},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            role = probe_filter.get_role_recommendation("critical")

            assert role == "observer"

    def test_high_vulnerability_returns_critic(self):
        """Should recommend critic for high-vulnerability agents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "risky"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 5,  # 50%
                        "breakdown": {},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            role = probe_filter.get_role_recommendation("risky")

            assert role == "critic"

    def test_moderate_vulnerability_returns_proposer(self):
        """Should recommend proposer for moderate vulnerability agents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "moderate"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 3,  # 30%
                        "breakdown": {},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            role = probe_filter.get_role_recommendation("moderate")

            assert role == "proposer"

    def test_low_vulnerability_returns_judge(self):
        """Should recommend judge for low-vulnerability agents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "reliable"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 1,  # 10%
                        "breakdown": {},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            role = probe_filter.get_role_recommendation("reliable")

            assert role == "judge"


# =============================================================================
# TestProbeFilterGetAllProfiles - Bulk Operations
# =============================================================================


class TestProbeFilterGetAllProfiles:
    """Tests for ProbeFilter.get_all_profiles() method."""

    def test_returns_empty_for_nonexistent_dir(self):
        """Should return empty dict if probes dir doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            profiles = probe_filter.get_all_profiles()

            assert profiles == {}

    def test_returns_all_agent_profiles(self):
        """Should return profiles for all agents with data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes"
            probes_dir.mkdir(parents=True)

            for agent_name in ["agent1", "agent2", "agent3"]:
                agent_dir = probes_dir / agent_name
                agent_dir.mkdir()
                (agent_dir / "report.json").write_text(
                    json.dumps(
                        {
                            "probes_run": 10,
                            "vulnerabilities_found": 1,
                            "breakdown": {},
                            "by_type": {},
                            "created_at": datetime.now().isoformat(),
                        }
                    )
                )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            profiles = probe_filter.get_all_profiles()

            assert len(profiles) == 3
            assert "agent1" in profiles
            assert "agent2" in profiles
            assert "agent3" in profiles


# =============================================================================
# TestProbeFilterFormatSummary - Summary Formatting
# =============================================================================


class TestProbeFilterFormatSummary:
    """Tests for ProbeFilter.format_summary() method."""

    def test_empty_returns_message(self):
        """Should return message for no probe data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            summary = probe_filter.format_summary()

            assert "No probe data available" in summary

    def test_includes_agent_info(self):
        """Should include agent information in summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "test_agent"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 2,
                        "breakdown": {"critical": 0, "high": 1},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)

            summary = probe_filter.format_summary()

            assert "test_agent" in summary
            assert "80%" in summary or "0.8" in summary  # probe score


# =============================================================================
# TestProbeFilterEdgeCases - Edge Cases
# =============================================================================


class TestProbeFilterEdgeCases:
    """Tests for edge cases in ProbeFilter."""

    def test_handles_empty_probe_directory(self):
        """Should handle empty probe directory for agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "empty_agent"
            probes_dir.mkdir(parents=True)

            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            profile = probe_filter.get_agent_profile("empty_agent")

            # Should return default profile
            assert profile.probe_score == 1.0
            assert profile.total_probes == 0

    def test_handles_invalid_date_format(self):
        """Should handle invalid date format in reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "bad_date"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 1,
                        "breakdown": {},
                        "by_type": {},
                        "created_at": "not a valid date",
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            profile = probe_filter.get_agent_profile("bad_date")

            # Should still load but with default days_since_probe
            assert profile.total_probes == 10
            assert profile.days_since_probe == 999

    def test_handles_missing_breakdown_key(self):
        """Should handle missing breakdown key in reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "no_breakdown"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 10,
                        "vulnerabilities_found": 1,
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            profile = probe_filter.get_agent_profile("no_breakdown")

            # Should still load
            assert profile.total_probes == 10

    def test_handles_zero_probes_run(self):
        """Should handle reports with zero probes run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "zero_probes"
            probes_dir.mkdir(parents=True)
            (probes_dir / "report.json").write_text(
                json.dumps(
                    {
                        "probes_run": 0,
                        "vulnerabilities_found": 0,
                        "breakdown": {},
                        "by_type": {},
                        "created_at": datetime.now().isoformat(),
                    }
                )
            )

            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            profile = probe_filter.get_agent_profile("zero_probes")

            # Should handle division by zero
            assert profile.vulnerability_rate == 0.0
            assert profile.probe_score == 1.0
