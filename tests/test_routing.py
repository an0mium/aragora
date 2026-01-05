"""
Tests for the routing module: AgentSelector, ProbeFilter, and related components.

Tests agent selection, probe-based filtering, team composition, and bench management.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

import pytest

from aragora.routing.probe_filter import ProbeProfile, ProbeFilter
from aragora.routing.selection import (
    AgentProfile,
    AgentSelector,
    TaskRequirements,
    TeamComposition,
)


# =============================================================================
# ProbeProfile Tests
# =============================================================================


class TestProbeProfile:
    """Tests for ProbeProfile dataclass."""

    def test_probe_profile_creation(self):
        """Test basic ProbeProfile creation with required fields."""
        profile = ProbeProfile(agent_name="agent_1")
        assert profile.agent_name == "agent_1"
        assert profile.vulnerability_rate == 0.0
        assert profile.probe_score == 1.0
        assert profile.total_probes == 0
        assert profile.critical_count == 0

    def test_probe_profile_with_all_fields(self):
        """Test ProbeProfile with all optional fields."""
        profile = ProbeProfile(
            agent_name="agent_2",
            vulnerability_rate=0.25,
            probe_score=0.75,
            critical_count=2,
            high_count=5,
            medium_count=10,
            low_count=20,
            dominant_weakness="prompt_injection",
            total_probes=100,
            last_probe_date="2025-01-01",
            days_since_probe=5,
            report_count=3,
        )
        assert profile.vulnerability_rate == 0.25
        assert profile.probe_score == 0.75
        assert profile.critical_count == 2
        assert profile.high_count == 5
        assert profile.total_probes == 100
        assert profile.dominant_weakness == "prompt_injection"

    def test_is_stale_fresh_probe(self):
        """Test is_stale returns False for recent probes."""
        profile = ProbeProfile(
            agent_name="agent_1",
            days_since_probe=3,
        )
        # Default staleness threshold is 7 days
        assert not profile.is_stale(max_days=7)

    def test_is_stale_old_probe(self):
        """Test is_stale returns True for old probes."""
        profile = ProbeProfile(
            agent_name="agent_1",
            days_since_probe=14,
        )
        assert profile.is_stale(max_days=7)

    def test_is_stale_default_very_old(self):
        """Test is_stale with default days_since_probe (999)."""
        profile = ProbeProfile(agent_name="agent_1")
        assert profile.is_stale(max_days=7)

    def test_is_high_risk_low_vulnerability(self):
        """Test is_high_risk returns False for low vulnerability agents."""
        profile = ProbeProfile(
            agent_name="agent_1",
            vulnerability_rate=0.1,
        )
        assert not profile.is_high_risk(threshold=0.4)

    def test_is_high_risk_high_vulnerability(self):
        """Test is_high_risk returns True for high vulnerability agents."""
        profile = ProbeProfile(
            agent_name="agent_1",
            vulnerability_rate=0.5,
        )
        assert profile.is_high_risk(threshold=0.4)

    def test_is_high_risk_default_threshold(self):
        """Test is_high_risk with default threshold of 0.4."""
        profile = ProbeProfile(agent_name="agent_1", vulnerability_rate=0.45)
        assert profile.is_high_risk()

    def test_has_critical_issues_none(self):
        """Test has_critical_issues returns False with no critical failures."""
        profile = ProbeProfile(
            agent_name="agent_1",
            critical_count=0,
        )
        assert not profile.has_critical_issues()

    def test_has_critical_issues_present(self):
        """Test has_critical_issues returns True with critical failures."""
        profile = ProbeProfile(
            agent_name="agent_1",
            critical_count=3,
        )
        assert profile.has_critical_issues()

    def test_to_dict(self):
        """Test ProbeProfile serialization to dict."""
        profile = ProbeProfile(
            agent_name="agent_1",
            vulnerability_rate=0.2,
            probe_score=0.8,
            critical_count=1,
            total_probes=50,
        )
        data = profile.to_dict()
        assert data["agent_name"] == "agent_1"
        assert data["vulnerability_rate"] == 0.2
        assert data["probe_score"] == 0.8
        assert data["critical_count"] == 1
        assert data["total_probes"] == 50


# =============================================================================
# ProbeFilter Tests
# =============================================================================


class TestProbeFilter:
    """Tests for ProbeFilter class."""

    def test_probe_filter_creation(self):
        """Test ProbeFilter initialization."""
        filter = ProbeFilter()
        assert filter is not None
        assert hasattr(filter, "_profile_cache")

    def test_probe_filter_with_custom_nomic_dir(self):
        """Test ProbeFilter with custom nomic directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filter = ProbeFilter(nomic_dir=temp_dir)
            assert filter.nomic_dir == Path(temp_dir)

    def test_get_agent_profile_new_agent(self):
        """Test get_agent_profile creates profile for new agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filter = ProbeFilter(nomic_dir=temp_dir)
            profile = filter.get_agent_profile("new_agent")
            assert profile.agent_name == "new_agent"
            # New agents without probe data have default scores
            assert profile.probe_score == 1.0
            assert profile.total_probes == 0

    def test_get_agent_profile_cached(self):
        """Test get_agent_profile returns cached profile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filter = ProbeFilter(nomic_dir=temp_dir, cache_ttl_seconds=300)

            # First call loads/creates profile
            profile1 = filter.get_agent_profile("agent_1")

            # Second call should return cached version
            profile2 = filter.get_agent_profile("agent_1")

            # Should be the same object due to caching
            assert profile1 is profile2

    def test_filter_agents_no_probe_data(self):
        """Test filter_agents includes agents without probe data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filter = ProbeFilter(nomic_dir=temp_dir)

            # Agents without probe data should be included
            filtered = filter.filter_agents(
                candidates=["agent_1", "agent_2", "agent_3"],
                max_vulnerability_rate=0.3,
            )
            assert len(filtered) == 3

    def test_filter_agents_with_probe_data(self):
        """Test filter_agents filters based on probe data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create probe data directory structure
            probes_dir = Path(temp_dir) / "probes" / "risky_agent"
            probes_dir.mkdir(parents=True)

            # Create a probe report
            probe_data = {
                "probes_run": 10,
                "vulnerabilities_found": 6,
                "breakdown": {"critical": 0, "high": 2, "medium": 3, "low": 1},
                "by_type": {},
                "created_at": datetime.now().isoformat(),
            }
            (probes_dir / "probe_001.json").write_text(json.dumps(probe_data))

            filter = ProbeFilter(nomic_dir=temp_dir)

            # Risky agent should be filtered out (60% vulnerability > 30% threshold)
            filtered = filter.filter_agents(
                candidates=["safe_agent", "risky_agent"],
                max_vulnerability_rate=0.3,
            )
            assert "safe_agent" in filtered
            assert "risky_agent" not in filtered

    def test_filter_agents_exclude_critical(self):
        """Test filter_agents can exclude agents with critical issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "critical_agent"
            probes_dir.mkdir(parents=True)

            probe_data = {
                "probes_run": 10,
                "vulnerabilities_found": 2,
                "breakdown": {"critical": 1, "high": 1, "medium": 0, "low": 0},
                "by_type": {},
                "created_at": datetime.now().isoformat(),
            }
            (probes_dir / "probe_001.json").write_text(json.dumps(probe_data))

            filter = ProbeFilter(nomic_dir=temp_dir)

            # With exclude_critical=True, agent should be filtered
            filtered = filter.filter_agents(
                candidates=["safe_agent", "critical_agent"],
                max_vulnerability_rate=0.5,
                exclude_critical=True,
            )
            assert "safe_agent" in filtered
            assert "critical_agent" not in filtered

    def test_get_team_scores(self):
        """Test get_team_scores returns scores for each agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filter = ProbeFilter(nomic_dir=temp_dir)

            scores = filter.get_team_scores(["agent_1", "agent_2"])

            # Both agents should have base score (no probe data)
            assert "agent_1" in scores
            assert "agent_2" in scores
            assert scores["agent_1"] == 1.0  # base_score default
            assert scores["agent_2"] == 1.0

    def test_get_team_scores_with_probe_data(self):
        """Test get_team_scores adjusts based on probe data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "probed_agent"
            probes_dir.mkdir(parents=True)

            probe_data = {
                "probes_run": 10,
                "vulnerabilities_found": 3,
                "breakdown": {"critical": 0, "high": 1, "medium": 2, "low": 0},
                "by_type": {},
                "created_at": datetime.now().isoformat(),
            }
            (probes_dir / "probe_001.json").write_text(json.dumps(probe_data))

            filter = ProbeFilter(nomic_dir=temp_dir)

            scores = filter.get_team_scores(["probed_agent", "unprobed_agent"])

            # Probed agent has 30% vulnerability = 70% probe score
            assert scores["probed_agent"] == pytest.approx(0.7, rel=0.01)
            assert scores["unprobed_agent"] == 1.0

    def test_get_role_recommendation_no_data(self):
        """Test role recommendation for agent without probe data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filter = ProbeFilter(nomic_dir=temp_dir)

            role = filter.get_role_recommendation("new_agent")
            assert role == "proposer"

    def test_get_role_recommendation_high_vulnerability(self):
        """Test role recommendation for high vulnerability agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "vulnerable_agent"
            probes_dir.mkdir(parents=True)

            probe_data = {
                "probes_run": 10,
                "vulnerabilities_found": 5,
                "breakdown": {"critical": 0, "high": 2, "medium": 3, "low": 0},
                "by_type": {},
                "created_at": datetime.now().isoformat(),
            }
            (probes_dir / "probe_001.json").write_text(json.dumps(probe_data))

            filter = ProbeFilter(nomic_dir=temp_dir)

            role = filter.get_role_recommendation("vulnerable_agent")
            assert role == "critic"  # 50% vulnerability > 40% threshold

    def test_clear_cache(self):
        """Test cache clearing."""
        filter = ProbeFilter()

        # Load a profile
        filter.get_agent_profile("agent_1")
        assert len(filter._profile_cache) == 1

        # Clear cache
        filter.clear_cache()
        assert len(filter._profile_cache) == 0


# =============================================================================
# AgentProfile Tests
# =============================================================================


class TestAgentProfile:
    """Tests for AgentProfile dataclass."""

    def test_agent_profile_creation(self):
        """Test basic AgentProfile creation."""
        profile = AgentProfile(
            name="TestAgent",
            agent_type="claude",
        )
        assert profile.name == "TestAgent"
        assert profile.agent_type == "claude"
        assert profile.elo_rating == 1500

    def test_agent_profile_with_all_fields(self):
        """Test AgentProfile with all fields."""
        profile = AgentProfile(
            name="FullAgent",
            agent_type="gemini",
            elo_rating=1700,
            domain_ratings={"coding": 1800, "math": 1600},
            expertise={"coding": 0.9, "math": 0.7},
            traits=["analytical", "thorough"],
            availability=0.95,
            cost_factor=1.5,
            latency_ms=500,
            success_rate=0.85,
            probe_score=0.9,
            calibration_score=0.8,
        )
        assert profile.elo_rating == 1700
        assert profile.domain_ratings["coding"] == 1800
        assert profile.expertise["coding"] == 0.9
        assert "analytical" in profile.traits
        assert profile.probe_score == 0.9

    def test_overall_score_calculation(self):
        """Test overall_score combines ELO, probes, and calibration."""
        profile = AgentProfile(
            name="GoodAgent",
            agent_type="claude",
            elo_rating=1800,  # High ELO
            success_rate=0.9,
            probe_score=0.95,
            calibration_score=0.9,
            latency_ms=500,
            cost_factor=1.0,
        )
        score = profile.overall_score
        assert score > 0.5
        assert score <= 1.0

    def test_overall_score_critical_probe_penalty(self):
        """Test overall_score applies penalty for critical probe issues."""
        base_profile = AgentProfile(
            name="Agent",
            agent_type="claude",
            elo_rating=1600,
            success_rate=0.8,
            probe_score=0.8,
            has_critical_probes=False,
        )
        critical_profile = AgentProfile(
            name="Agent",
            agent_type="claude",
            elo_rating=1600,
            success_rate=0.8,
            probe_score=0.8,
            has_critical_probes=True,
        )

        # Critical probe should reduce score by 30%
        assert critical_profile.overall_score < base_profile.overall_score
        assert critical_profile.overall_score == pytest.approx(
            base_profile.overall_score * 0.7, rel=0.01
        )

    def test_overall_score_overconfidence_penalty(self):
        """Test overall_score applies penalty for overconfident agents."""
        calibrated = AgentProfile(
            name="Calibrated",
            agent_type="claude",
            elo_rating=1600,
            is_overconfident=False,
        )
        overconfident = AgentProfile(
            name="Overconfident",
            agent_type="claude",
            elo_rating=1600,
            is_overconfident=True,
        )

        # Overconfidence should reduce score
        assert overconfident.overall_score < calibrated.overall_score


# =============================================================================
# TaskRequirements Tests
# =============================================================================


class TestTaskRequirements:
    """Tests for TaskRequirements dataclass."""

    def test_task_requirements_creation(self):
        """Test basic TaskRequirements creation."""
        reqs = TaskRequirements(
            task_id="task_001",
            description="Test task",
            primary_domain="coding",
        )
        assert reqs.task_id == "task_001"
        assert reqs.primary_domain == "coding"
        assert reqs.min_agents == 2
        assert reqs.max_agents == 5

    def test_task_requirements_with_secondary_domains(self):
        """Test TaskRequirements with secondary domains."""
        reqs = TaskRequirements(
            task_id="task_002",
            description="Multi-domain task",
            primary_domain="coding",
            secondary_domains=["math", "logic"],
        )
        assert "math" in reqs.secondary_domains
        assert "logic" in reqs.secondary_domains

    def test_task_requirements_with_traits(self):
        """Test TaskRequirements with required traits."""
        reqs = TaskRequirements(
            task_id="task_003",
            description="Security review",
            primary_domain="security",
            required_traits=["thorough", "security-focused"],
        )
        assert "thorough" in reqs.required_traits

    def test_task_requirements_quality_priority(self):
        """Test TaskRequirements quality vs speed priority."""
        quality_reqs = TaskRequirements(
            task_id="quality_task",
            description="High quality task",
            primary_domain="coding",
            quality_priority=0.9,
        )
        speed_reqs = TaskRequirements(
            task_id="speed_task",
            description="Fast task",
            primary_domain="coding",
            quality_priority=0.1,
        )
        assert quality_reqs.quality_priority > speed_reqs.quality_priority


# =============================================================================
# TeamComposition Tests
# =============================================================================


class TestTeamComposition:
    """Tests for TeamComposition dataclass."""

    def test_team_composition_creation(self):
        """Test basic TeamComposition creation."""
        agents = [
            AgentProfile(name="Agent1", agent_type="claude"),
            AgentProfile(name="Agent2", agent_type="gemini"),
        ]
        team = TeamComposition(
            team_id="team_001",
            task_id="task_001",
            agents=agents,
            roles={"Agent1": "proposer", "Agent2": "critic"},
            expected_quality=0.8,
            expected_cost=2.0,
            diversity_score=0.6,
            rationale="Selected top 2 agents",
        )
        assert len(team.agents) == 2
        assert team.roles["Agent1"] == "proposer"
        assert team.expected_quality == 0.8


# =============================================================================
# AgentSelector Tests
# =============================================================================


class TestAgentSelector:
    """Tests for AgentSelector class."""

    def test_agent_selector_creation(self):
        """Test AgentSelector initialization."""
        selector = AgentSelector()
        assert selector is not None
        assert hasattr(selector, "agent_pool")
        assert hasattr(selector, "bench")

    def test_register_agent(self):
        """Test registering an agent with the selector."""
        selector = AgentSelector()
        profile = AgentProfile(name="Agent1", agent_type="claude", elo_rating=1500)
        selector.register_agent(profile)
        assert "Agent1" in selector.agent_pool

    def test_register_multiple_agents(self):
        """Test registering multiple agents."""
        selector = AgentSelector()
        for i in range(5):
            profile = AgentProfile(
                name=f"Agent{i}",
                agent_type="claude",
                elo_rating=1500 + i * 100,
            )
            selector.register_agent(profile)
        assert len(selector.agent_pool) == 5

    def test_remove_agent(self):
        """Test removing an agent from the pool."""
        selector = AgentSelector()
        profile = AgentProfile(name="Agent1", agent_type="claude")
        selector.register_agent(profile)

        selector.remove_agent("Agent1")
        assert "Agent1" not in selector.agent_pool

    def test_move_to_bench(self):
        """Test moving an agent to the bench."""
        selector = AgentSelector()
        profile = AgentProfile(name="Agent1", agent_type="claude")
        selector.register_agent(profile)

        selector.move_to_bench("Agent1")
        assert "Agent1" in selector.bench

    def test_promote_from_bench(self):
        """Test promoting an agent from the bench."""
        selector = AgentSelector()
        profile = AgentProfile(name="Agent1", agent_type="claude")
        selector.register_agent(profile)
        selector.move_to_bench("Agent1")

        selector.promote_from_bench("Agent1")
        assert "Agent1" not in selector.bench

    def test_select_team_basic(self):
        """Test basic team selection."""
        selector = AgentSelector()

        # Register several agents
        for i in range(5):
            profile = AgentProfile(
                name=f"Agent{i}",
                agent_type="claude",
                elo_rating=1500 + i * 50,
            )
            selector.register_agent(profile)

        reqs = TaskRequirements(
            task_id="test_task",
            description="Test",
            primary_domain="coding",
            min_agents=2,
            max_agents=3,
        )

        team = selector.select_team(reqs)
        assert len(team.agents) >= 2
        assert len(team.agents) <= 3
        assert team.task_id == "test_task"

    def test_select_team_excludes_benched(self):
        """Test team selection excludes benched agents."""
        selector = AgentSelector()

        active = AgentProfile(name="Active", agent_type="claude", elo_rating=1600)
        benched = AgentProfile(name="Benched", agent_type="claude", elo_rating=1700)

        selector.register_agent(active)
        selector.register_agent(benched)
        selector.move_to_bench("Benched")

        reqs = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="coding",
            min_agents=1,
            max_agents=2,
        )

        team = selector.select_team(reqs)
        agent_names = [a.name for a in team.agents]
        assert "Benched" not in agent_names

    def test_select_team_assigns_roles(self):
        """Test team selection assigns appropriate roles."""
        selector = AgentSelector()

        for i in range(3):
            profile = AgentProfile(
                name=f"Agent{i}",
                agent_type="claude",
                elo_rating=1500,
                expertise={"coding": 0.5 + i * 0.2},
            )
            selector.register_agent(profile)

        reqs = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="coding",
            min_agents=2,
            max_agents=3,
        )

        team = selector.select_team(reqs)
        assert len(team.roles) == len(team.agents)
        # Should have at least a proposer
        assert "proposer" in team.roles.values()

    def test_refresh_probe_scores(self):
        """Test refreshing probe scores from ProbeFilter."""
        selector = AgentSelector()

        profile = AgentProfile(
            name="TestAgent",
            agent_type="claude",
            probe_score=0.5,  # Initial low score
        )
        selector.register_agent(profile)

        # Create mock probe filter
        mock_filter = MagicMock(spec=ProbeFilter)
        mock_probe_profile = ProbeProfile(
            agent_name="TestAgent",
            vulnerability_rate=0.1,
            probe_score=0.9,
            total_probes=10,
        )
        mock_filter.get_agent_profile.return_value = mock_probe_profile

        selector.set_probe_filter(mock_filter)

        # Agent's probe score should be updated
        assert selector.agent_pool["TestAgent"].probe_score == 0.9

    def test_get_probe_adjusted_score(self):
        """Test score adjustment based on probe reliability."""
        with tempfile.TemporaryDirectory() as temp_dir:
            probes_dir = Path(temp_dir) / "probes" / "risky_agent"
            probes_dir.mkdir(parents=True)

            probe_data = {
                "probes_run": 10,
                "vulnerabilities_found": 5,
                "breakdown": {"critical": 0, "high": 2, "medium": 3, "low": 0},
                "by_type": {},
                "created_at": datetime.now().isoformat(),
            }
            (probes_dir / "probe_001.json").write_text(json.dumps(probe_data))

            filter = ProbeFilter(nomic_dir=temp_dir)
            selector = AgentSelector(probe_filter=filter)

            profile = AgentProfile(name="risky_agent", agent_type="claude")
            selector.register_agent(profile)

            # Base score of 1.0 should be reduced for risky agent
            adjusted = selector.get_probe_adjusted_score("risky_agent", 1.0)
            assert adjusted < 1.0

    def test_get_leaderboard(self):
        """Test getting agent leaderboard."""
        selector = AgentSelector()

        for i, elo in enumerate([1400, 1600, 1800, 1500, 1700]):
            profile = AgentProfile(
                name=f"Agent{i}",
                agent_type="claude",
                elo_rating=elo,
            )
            selector.register_agent(profile)

        leaderboard = selector.get_leaderboard(limit=3)
        assert len(leaderboard) == 3
        # Should be sorted by overall score (higher ELO = higher score)
        assert leaderboard[0]["elo"] >= leaderboard[1]["elo"]

    def test_get_recommendations(self):
        """Test getting agent recommendations for a task."""
        selector = AgentSelector()

        coder = AgentProfile(
            name="Coder",
            agent_type="claude",
            elo_rating=1600,
            expertise={"coding": 0.9, "math": 0.5},
        )
        math_expert = AgentProfile(
            name="MathExpert",
            agent_type="gemini",
            elo_rating=1600,
            expertise={"coding": 0.5, "math": 0.9},
        )
        selector.register_agent(coder)
        selector.register_agent(math_expert)

        reqs = TaskRequirements(
            task_id="coding_task",
            description="Implement algorithm",
            primary_domain="coding",
        )

        recommendations = selector.get_recommendations(reqs, limit=2)
        assert len(recommendations) == 2
        # Coder should be ranked higher for coding task
        assert recommendations[0]["name"] == "Coder"


# =============================================================================
# Integration Tests
# =============================================================================


class TestRoutingIntegration:
    """Integration tests for the routing module."""

    def test_full_selection_pipeline(self):
        """Test complete agent selection pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup probe filter
            probe_filter = ProbeFilter(nomic_dir=temp_dir)
            selector = AgentSelector(probe_filter=probe_filter)

            # Register diverse agents
            agents_config = [
                ("Claude", "claude", 1700, 0.9),
                ("Gemini", "gemini", 1600, 0.8),
                ("Grok", "grok", 1500, 0.7),
                ("DeepSeek", "deepseek", 1550, 0.85),
            ]

            for name, agent_type, elo, expertise in agents_config:
                profile = AgentProfile(
                    name=name,
                    agent_type=agent_type,
                    elo_rating=elo,
                    expertise={"coding": expertise},
                )
                selector.register_agent(profile)

            # Create task requirements
            reqs = TaskRequirements(
                task_id="code_review",
                description="Review code for security issues",
                primary_domain="coding",
                min_agents=2,
                max_agents=3,
            )

            # Select team
            team = selector.select_team(reqs)

            # Verify team composition
            assert len(team.agents) >= 2
            assert len(team.agents) <= 3
            assert team.expected_quality > 0
            assert len(team.roles) == len(team.agents)

    def test_bench_recovery_workflow(self):
        """Test agent bench and recovery workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            selector = AgentSelector()

            # Register agent
            profile = AgentProfile(
                name="TestAgent",
                agent_type="claude",
                elo_rating=1500,
            )
            selector.register_agent(profile)

            # Move to bench (simulating poor probe results)
            selector.move_to_bench("TestAgent")
            assert "TestAgent" in selector.bench

            # Create task - benched agent should not be selected
            reqs = TaskRequirements(
                task_id="test",
                description="Test",
                primary_domain="coding",
                min_agents=1,
                max_agents=1,
            )

            # Should raise error - only agent is benched
            with pytest.raises(ValueError):
                selector.select_team(reqs)

            # Promote from bench
            selector.promote_from_bench("TestAgent")
            assert "TestAgent" not in selector.bench

            # Now selection should work
            team = selector.select_team(reqs)
            assert len(team.agents) == 1

    def test_hybrid_role_assignment(self):
        """Test hybrid model architecture role assignment."""
        selector = AgentSelector()

        # Register agents matching hybrid architecture
        agents = [
            AgentProfile(name="gemini-pro", agent_type="gemini", elo_rating=1600),
            AgentProfile(name="claude-sonnet", agent_type="claude", elo_rating=1700),
            AgentProfile(name="codex", agent_type="codex", elo_rating=1500),
            AgentProfile(name="grok-2", agent_type="grok", elo_rating=1550),
        ]

        for agent in agents:
            selector.register_agent(agent)

        # Test design phase roles
        design_roles = selector.assign_hybrid_roles(agents, "design")
        assert design_roles.get("gemini-pro") == "design_lead"
        assert design_roles.get("claude-sonnet") == "architecture_critic"

        # Test implement phase roles
        impl_roles = selector.assign_hybrid_roles(agents, "implement")
        assert impl_roles.get("claude-sonnet") == "implementer"

        # Test verify phase roles
        verify_roles = selector.assign_hybrid_roles(agents, "verify")
        assert verify_roles.get("codex") == "verification_lead"

    def test_selection_history_tracking(self):
        """Test that selection history is properly tracked."""
        selector = AgentSelector()

        for i in range(3):
            profile = AgentProfile(name=f"Agent{i}", agent_type="claude")
            selector.register_agent(profile)

        reqs = TaskRequirements(
            task_id="tracked_task",
            description="Test",
            primary_domain="coding",
            min_agents=2,
            max_agents=2,
        )

        # Initial history should be empty
        assert len(selector.get_selection_history()) == 0

        # Select team
        selector.select_team(reqs)

        # History should have one entry
        history = selector.get_selection_history()
        assert len(history) == 1
        assert history[0]["task_id"] == "tracked_task"
        assert len(history[0]["selected"]) == 2
