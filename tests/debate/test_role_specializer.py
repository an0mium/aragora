"""Tests for A-HMAD role specialization system.

Covers:
- RoleType enum, dataclasses (AgentCapability, RoleRequirement, RoleAssignment, TeamComposition, AHMADConfig)
- AHMADRoleSpecializer: topic analysis, role assignment, scoring, metrics
- Convenience functions: create_role_specializer, quick_assign_roles
"""

from __future__ import annotations

import pytest

from aragora.debate.role_specializer import (
    AHMADConfig,
    AHMADRoleSpecializer,
    AgentCapability,
    RoleAssignment,
    RoleRequirement,
    RoleType,
    TeamComposition,
    create_role_specializer,
    quick_assign_roles,
)


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_role_type_values(self):
        assert RoleType.PROPOSER.value == "proposer"
        assert RoleType.DEVIL_ADVOCATE.value == "devil_advocate"
        assert len(RoleType) == 8

    def test_agent_capability_defaults(self):
        cap = AgentCapability(agent_id="claude")
        assert cap.elo_rating == 1500.0
        assert cap.brier_score == 0.25
        assert cap.domain_scores == {}
        assert cap.role_performance == {}

    def test_role_requirement_defaults(self):
        req = RoleRequirement(role_type=RoleType.CRITIC, importance=0.8)
        assert req.domain_preference is None
        assert req.min_agents == 1
        assert req.max_agents == 1

    def test_role_assignment_defaults(self):
        a = RoleAssignment(agent_id="gpt", role=RoleType.PROPOSER, confidence=0.9, reasoning="test")
        assert a.is_fallback is False

    def test_team_composition_defaults(self):
        tc = TeamComposition(
            assignments=[], diversity_score=0.5, coverage_score=1.0, total_capability_score=0.8
        )
        assert tc.assignment_time_ms == 0.0

    def test_ahmad_config_defaults(self):
        cfg = AHMADConfig()
        assert cfg.elo_weight == 0.3
        assert cfg.calibration_weight == 0.25
        assert cfg.domain_weight == 0.25
        assert cfg.role_history_weight == 0.2
        assert len(cfg.default_roles) == 3
        assert RoleType.PROPOSER in cfg.default_roles
        assert "fact" in cfg.topic_role_triggers
        assert "proposer" in cfg.static_fallback


# ---------------------------------------------------------------------------
# AHMADRoleSpecializer - analyze_topic
# ---------------------------------------------------------------------------


class TestAnalyzeTopic:
    @pytest.fixture
    def specializer(self) -> AHMADRoleSpecializer:
        return AHMADRoleSpecializer()

    def test_default_roles_always_included(self, specializer):
        roles = specializer.analyze_topic("some generic topic")
        role_types = {r.role_type for r in roles}
        assert RoleType.PROPOSER in role_types
        assert RoleType.CRITIC in role_types
        assert RoleType.SYNTHESIZER in role_types

    def test_fact_keyword_triggers_fact_checker(self, specializer):
        roles = specializer.analyze_topic("Verify the facts about climate change")
        role_types = {r.role_type for r in roles}
        assert RoleType.FACT_CHECKER in role_types

    def test_debate_keyword_triggers_devil_advocate(self, specializer):
        roles = specializer.analyze_topic("A controversial stance on AI regulation")
        role_types = {r.role_type for r in roles}
        assert RoleType.DEVIL_ADVOCATE in role_types

    def test_technical_keyword_triggers_domain_expert(self, specializer):
        roles = specializer.analyze_topic("Technical analysis of database indexing")
        role_types = {r.role_type for r in roles}
        assert RoleType.DOMAIN_EXPERT in role_types

    def test_conflict_keyword_triggers_mediator(self, specializer):
        roles = specializer.analyze_topic("Resolve the conflict between teams")
        role_types = {r.role_type for r in roles}
        assert RoleType.MEDIATOR in role_types

    def test_importance_higher_for_early_keywords(self, specializer):
        """Keywords in first 50 chars get importance 0.8, later get 0.6."""
        roles = specializer.analyze_topic("fact checking is important for this topic")
        fact_checker = [r for r in roles if r.role_type == RoleType.FACT_CHECKER][0]
        assert fact_checker.importance == 0.8

    def test_domain_preference_set(self, specializer):
        roles = specializer.analyze_topic("Compare approaches", domain="economics")
        for role in roles:
            assert role.domain_preference == "economics"

    def test_sorted_by_importance(self, specializer):
        roles = specializer.analyze_topic("Verify facts and compare approaches technically")
        importances = [r.importance for r in roles]
        assert importances == sorted(importances, reverse=True)

    def test_no_duplicate_roles(self, specializer):
        roles = specializer.analyze_topic("Verify and fact check this")
        role_types = [r.role_type for r in roles]
        assert len(role_types) == len(set(role_types))


# ---------------------------------------------------------------------------
# AHMADRoleSpecializer - assign_roles
# ---------------------------------------------------------------------------


class TestAssignRoles:
    @pytest.fixture
    def specializer(self) -> AHMADRoleSpecializer:
        return AHMADRoleSpecializer()

    def test_empty_agents_returns_empty(self, specializer):
        roles = [RoleRequirement(role_type=RoleType.PROPOSER, importance=0.8)]
        team = specializer.assign_roles(roles=roles, available_agents=[])
        assert team.assignments == []
        assert team.diversity_score == 0.0

    def test_assigns_agents_to_roles(self, specializer):
        roles = specializer.analyze_topic("Generic debate")
        agents = ["claude", "gpt-4", "gemini"]
        team = specializer.assign_roles(roles=roles, available_agents=agents)
        assert len(team.assignments) > 0
        assert all(a.agent_id in agents for a in team.assignments)

    def test_elo_scores_influence_assignment(self, specializer):
        """Agent with higher ELO should score higher."""
        roles = [RoleRequirement(role_type=RoleType.PROPOSER, importance=0.9)]
        team = specializer.assign_roles(
            roles=roles,
            available_agents=["low_elo", "high_elo"],
            elo_scores={"low_elo": 1100, "high_elo": 1900},
        )
        assert team.assignments[0].agent_id == "high_elo"

    def test_calibration_scores_influence_assignment(self, specializer):
        """Agent with better (lower) Brier score should be preferred."""
        roles = [RoleRequirement(role_type=RoleType.PROPOSER, importance=0.9)]
        team = specializer.assign_roles(
            roles=roles,
            available_agents=["bad_cal", "good_cal"],
            elo_scores={"bad_cal": 1500, "good_cal": 1500},
            calibration_scores={"bad_cal": 0.5, "good_cal": 0.05},
        )
        assert team.assignments[0].agent_id == "good_cal"

    def test_domain_scores_influence_assignment(self, specializer):
        """Agent with domain expertise should be preferred for domain roles."""
        roles = [
            RoleRequirement(
                role_type=RoleType.DOMAIN_EXPERT,
                importance=0.9,
                domain_preference="economics",
            )
        ]
        team = specializer.assign_roles(
            roles=roles,
            available_agents=["generic", "econ_expert"],
            elo_scores={"generic": 1500, "econ_expert": 1500},
            domain_scores={"econ_expert": {"economics": 0.95}, "generic": {"economics": 0.1}},
        )
        assert team.assignments[0].agent_id == "econ_expert"

    def test_diversity_penalty_avoids_duplicate_agents(self, specializer):
        """When agent already assigned, it gets penalized for subsequent roles."""
        roles = [
            RoleRequirement(role_type=RoleType.PROPOSER, importance=0.9),
            RoleRequirement(role_type=RoleType.CRITIC, importance=0.8),
        ]
        team = specializer.assign_roles(
            roles=roles,
            available_agents=["claude", "gpt"],
            elo_scores={"claude": 1600, "gpt": 1580},
        )
        # Should assign different agents to each role due to diversity penalty
        assigned = [a.agent_id for a in team.assignments]
        assert len(set(assigned)) == 2

    def test_fallback_flag_for_low_scores(self, specializer):
        """Assignments with score < 0.3 are marked as fallback."""
        config = AHMADConfig(
            elo_weight=0.0,
            calibration_weight=0.0,
            domain_weight=0.0,
            role_history_weight=0.0,
        )
        specializer = AHMADRoleSpecializer(config=config)
        roles = [RoleRequirement(role_type=RoleType.PROPOSER, importance=0.9)]
        team = specializer.assign_roles(
            roles=roles,
            available_agents=["agent1"],
        )
        assert team.assignments[0].is_fallback is True

    def test_static_fallback_used_for_unknown_agents(self, specializer):
        """Static fallback boosts score for agents matching the fallback list."""
        roles = [RoleRequirement(role_type=RoleType.PROPOSER, importance=0.9)]
        # "claude" is in the static fallback for proposer
        team = specializer.assign_roles(
            roles=roles,
            available_agents=["claude", "unknown_agent"],
        )
        assert team.assignments[0].agent_id == "claude"

    def test_assignment_recorded_in_history(self, specializer):
        assert len(specializer._assignment_history) == 0
        roles = specializer.analyze_topic("test")
        specializer.assign_roles(roles=roles, available_agents=["claude"])
        assert len(specializer._assignment_history) == 1

    def test_assignment_time_recorded(self, specializer):
        roles = specializer.analyze_topic("test")
        team = specializer.assign_roles(roles=roles, available_agents=["claude"])
        assert team.assignment_time_ms >= 0


# ---------------------------------------------------------------------------
# Scoring internals
# ---------------------------------------------------------------------------


class TestScoringInternals:
    @pytest.fixture
    def specializer(self) -> AHMADRoleSpecializer:
        return AHMADRoleSpecializer()

    def test_generate_reasoning_strong(self, specializer):
        reason = specializer._generate_reasoning("claude", RoleType.PROPOSER, 0.8)
        assert "Strong match" in reason

    def test_generate_reasoning_good(self, specializer):
        reason = specializer._generate_reasoning("claude", RoleType.PROPOSER, 0.55)
        assert "Good match" in reason

    def test_generate_reasoning_adequate(self, specializer):
        reason = specializer._generate_reasoning("claude", RoleType.PROPOSER, 0.35)
        assert "Adequate match" in reason

    def test_generate_reasoning_fallback(self, specializer):
        reason = specializer._generate_reasoning("claude", RoleType.PROPOSER, 0.1)
        assert "Fallback" in reason

    def test_diversity_empty_assignments(self, specializer):
        assert specializer._calculate_diversity([]) == 0.0

    def test_diversity_unique_agents(self, specializer):
        assignments = [
            RoleAssignment(agent_id="a", role=RoleType.PROPOSER, confidence=0.9, reasoning=""),
            RoleAssignment(agent_id="b", role=RoleType.CRITIC, confidence=0.9, reasoning=""),
        ]
        score = specializer._calculate_diversity(assignments)
        assert score > 0.5  # Both agents unique, both roles unique

    def test_diversity_duplicate_agents(self, specializer):
        assignments = [
            RoleAssignment(agent_id="a", role=RoleType.PROPOSER, confidence=0.9, reasoning=""),
            RoleAssignment(agent_id="a", role=RoleType.CRITIC, confidence=0.9, reasoning=""),
        ]
        score = specializer._calculate_diversity(assignments)
        unique_score = specializer._calculate_diversity(
            [
                RoleAssignment(agent_id="a", role=RoleType.PROPOSER, confidence=0.9, reasoning=""),
                RoleAssignment(agent_id="b", role=RoleType.CRITIC, confidence=0.9, reasoning=""),
            ]
        )
        assert score < unique_score

    def test_coverage_empty_requirements(self, specializer):
        assert specializer._calculate_coverage([], []) == 1.0

    def test_coverage_full(self, specializer):
        reqs = [RoleRequirement(role_type=RoleType.PROPOSER, importance=0.9)]
        assignments = [
            RoleAssignment(agent_id="a", role=RoleType.PROPOSER, confidence=0.9, reasoning=""),
        ]
        assert specializer._calculate_coverage(assignments, reqs) == 1.0

    def test_coverage_partial(self, specializer):
        reqs = [
            RoleRequirement(role_type=RoleType.PROPOSER, importance=0.9),
            RoleRequirement(role_type=RoleType.CRITIC, importance=0.8),
        ]
        assignments = [
            RoleAssignment(agent_id="a", role=RoleType.PROPOSER, confidence=0.9, reasoning=""),
        ]
        assert specializer._calculate_coverage(assignments, reqs) == 0.5

    def test_total_capability_empty(self, specializer):
        assert specializer._calculate_total_capability([]) == 0.0

    def test_total_capability_mean(self, specializer):
        assignments = [
            RoleAssignment(agent_id="a", role=RoleType.PROPOSER, confidence=0.8, reasoning=""),
            RoleAssignment(agent_id="b", role=RoleType.CRITIC, confidence=0.6, reasoning=""),
        ]
        cap = specializer._calculate_total_capability(assignments)
        assert abs(cap - 0.7) < 0.01


# ---------------------------------------------------------------------------
# Performance recording and metrics
# ---------------------------------------------------------------------------


class TestPerformanceAndMetrics:
    @pytest.fixture
    def specializer(self) -> AHMADRoleSpecializer:
        return AHMADRoleSpecializer()

    def test_record_performance(self, specializer):
        specializer.record_performance("claude", RoleType.PROPOSER, 0.9)
        assert len(specializer._agent_role_history["claude"]["proposer"]) == 1
        assert specializer._agent_role_history["claude"]["proposer"][0] == 0.9

    def test_record_performance_caps_at_50(self, specializer):
        for i in range(60):
            specializer.record_performance("claude", RoleType.PROPOSER, float(i) / 60)
        assert len(specializer._agent_role_history["claude"]["proposer"]) == 50

    def test_role_history_influences_scoring(self, specializer):
        """Agent with good role history should score higher."""
        for _ in range(5):
            specializer.record_performance("expert", RoleType.PROPOSER, 0.95)
            specializer.record_performance("novice", RoleType.PROPOSER, 0.3)

        roles = [RoleRequirement(role_type=RoleType.PROPOSER, importance=0.9)]
        team = specializer.assign_roles(
            roles=roles,
            available_agents=["expert", "novice"],
            elo_scores={"expert": 1500, "novice": 1500},
        )
        assert team.assignments[0].agent_id == "expert"

    def test_reset(self, specializer):
        specializer.record_performance("claude", RoleType.PROPOSER, 0.9)
        roles = specializer.analyze_topic("test")
        specializer.assign_roles(roles=roles, available_agents=["claude"])

        specializer.reset()
        assert len(specializer._assignment_history) == 0
        assert len(specializer._agent_role_history) == 0

    def test_metrics_empty(self, specializer):
        m = specializer.get_metrics()
        assert m["total_assignments"] == 0
        assert m["avg_diversity"] == 0.0

    def test_metrics_after_assignments(self, specializer):
        roles = specializer.analyze_topic("test topic")
        specializer.assign_roles(roles=roles, available_agents=["claude", "gpt"])
        specializer.assign_roles(roles=roles, available_agents=["claude", "gemini"])

        m = specializer.get_metrics()
        assert m["total_assignments"] == 2
        assert m["avg_diversity"] > 0
        assert m["avg_coverage"] > 0
        assert "avg_time_ms" in m
        assert "agents_tracked" in m


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_create_role_specializer(self):
        s = create_role_specializer(min_diversity=0.8)
        assert s.config.min_diversity_score == 0.8
        assert isinstance(s, AHMADRoleSpecializer)

    def test_create_with_kwargs(self):
        s = create_role_specializer(min_diversity=0.5, elo_weight=0.5)
        assert s.config.elo_weight == 0.5

    def test_quick_assign_roles(self):
        result = quick_assign_roles(
            topic="Compare technical approaches",
            available_agents=["claude", "gpt", "gemini"],
        )
        assert len(result) > 0
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        # All agent IDs should be from available list
        assert all(agent in ["claude", "gpt", "gemini"] for agent, _ in result)

    def test_quick_assign_with_elo(self):
        result = quick_assign_roles(
            topic="fact checking exercise",
            available_agents=["claude", "gpt"],
            elo_scores={"claude": 1800, "gpt": 1200},
        )
        assert len(result) > 0
        # Claude should be preferred due to higher ELO
        first_agent = result[0][0]
        assert first_agent == "claude"
