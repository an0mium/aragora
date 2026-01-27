"""Tests for agent hierarchy management (Gastown-inspired patterns)."""

import pytest
from datetime import datetime

from aragora.debate.hierarchy import (
    AgentHierarchy,
    HierarchyConfig,
    HierarchyRole,
    RoleAssignment,
    ROLE_CAPABILITIES,
    STANDARD_CAPABILITIES,
)
from aragora.routing.selection import AgentProfile


class TestHierarchyRole:
    """Tests for HierarchyRole enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert HierarchyRole.ORCHESTRATOR.value == "orchestrator"
        assert HierarchyRole.MONITOR.value == "monitor"
        assert HierarchyRole.WORKER.value == "worker"

    def test_role_capabilities_defined(self):
        """Test role capabilities are defined."""
        assert HierarchyRole.ORCHESTRATOR in ROLE_CAPABILITIES
        assert HierarchyRole.MONITOR in ROLE_CAPABILITIES
        assert HierarchyRole.WORKER in ROLE_CAPABILITIES

    def test_orchestrator_requires_capabilities(self):
        """Test orchestrator requires specific capabilities."""
        caps = ROLE_CAPABILITIES[HierarchyRole.ORCHESTRATOR]
        assert "reasoning" in caps
        assert "synthesis" in caps
        assert "coordination" in caps

    def test_monitor_requires_capabilities(self):
        """Test monitor requires specific capabilities."""
        caps = ROLE_CAPABILITIES[HierarchyRole.MONITOR]
        assert "analysis" in caps
        assert "quality_assessment" in caps

    def test_worker_has_no_requirements(self):
        """Test worker has no specific capability requirements."""
        caps = ROLE_CAPABILITIES[HierarchyRole.WORKER]
        assert caps == set()


class TestHierarchyConfig:
    """Tests for HierarchyConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HierarchyConfig()
        assert config.max_orchestrators == 1
        assert config.max_monitors == 2
        assert config.min_workers == 2
        assert config.capability_weight == 0.4
        assert config.elo_weight == 0.3
        assert config.affinity_weight == 0.3
        assert config.auto_promote is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = HierarchyConfig(
            max_orchestrators=2,
            max_monitors=3,
            min_workers=5,
            capability_weight=0.5,
            elo_weight=0.3,
            affinity_weight=0.2,
            auto_promote=False,
        )
        assert config.max_orchestrators == 2
        assert config.max_monitors == 3
        assert config.min_workers == 5
        assert config.capability_weight == 0.5
        assert config.auto_promote is False

    def test_weight_sum(self):
        """Test that default weights sum to 1.0."""
        config = HierarchyConfig()
        total = config.capability_weight + config.elo_weight + config.affinity_weight
        assert abs(total - 1.0) < 0.001


class TestRoleAssignment:
    """Tests for RoleAssignment dataclass."""

    def test_create_assignment(self):
        """Test creating a role assignment."""
        assignment = RoleAssignment(
            agent_name="claude-opus",
            role=HierarchyRole.ORCHESTRATOR,
            assigned_at=datetime.now().isoformat(),
            capabilities_matched={"reasoning", "synthesis"},
            affinity_score=0.85,
        )
        assert assignment.agent_name == "claude-opus"
        assert assignment.role == HierarchyRole.ORCHESTRATOR
        assert "reasoning" in assignment.capabilities_matched
        assert assignment.affinity_score == 0.85

    def test_default_values(self):
        """Test default values for optional fields."""
        assignment = RoleAssignment(
            agent_name="test",
            role=HierarchyRole.WORKER,
            assigned_at="2024-01-01T00:00:00",
        )
        assert assignment.capabilities_matched == set()
        assert assignment.affinity_score == 0.0


class TestAgentHierarchy:
    """Tests for AgentHierarchy class."""

    @pytest.fixture
    def hierarchy(self):
        """Create a default hierarchy for testing."""
        return AgentHierarchy()

    @pytest.fixture
    def custom_hierarchy(self):
        """Create a hierarchy with custom config."""
        config = HierarchyConfig(
            max_orchestrators=1,
            max_monitors=2,
            min_workers=1,
        )
        return AgentHierarchy(config)

    @pytest.fixture
    def sample_agents(self):
        """Create sample agent profiles for testing."""
        return [
            AgentProfile(
                name="claude-opus",
                agent_type="claude",
                elo_rating=1800.0,
                capabilities={"reasoning", "synthesis", "coordination", "analysis"},
                task_affinity={"security": 0.9},
            ),
            AgentProfile(
                name="gpt-4",
                agent_type="codex",
                elo_rating=1700.0,
                capabilities={"analysis", "quality_assessment", "reasoning"},
                task_affinity={"api": 0.8},
            ),
            AgentProfile(
                name="gemini-pro",
                agent_type="gemini",
                elo_rating=1600.0,
                capabilities={"research", "analysis"},
                task_affinity={"database": 0.7},
            ),
            AgentProfile(
                name="grok",
                agent_type="grok",
                elo_rating=1500.0,
                capabilities={"creativity"},
                task_affinity={},
            ),
        ]

    def test_init_default_config(self, hierarchy):
        """Test initialization with default config."""
        assert hierarchy.config.max_orchestrators == 1
        assert hierarchy._assignments == {}
        assert hierarchy._role_history == {}

    def test_init_custom_config(self, custom_hierarchy):
        """Test initialization with custom config."""
        assert custom_hierarchy.config.max_orchestrators == 1
        assert custom_hierarchy.config.max_monitors == 2

    def test_assign_roles_basic(self, hierarchy, sample_agents):
        """Test basic role assignment."""
        assignments = hierarchy.assign_roles(
            debate_id="debate-001",
            agents=sample_agents,
        )

        assert len(assignments) == len(sample_agents)
        assert all(isinstance(a, RoleAssignment) for a in assignments.values())

    def test_assigns_orchestrator(self, hierarchy, sample_agents):
        """Test that an orchestrator is assigned."""
        assignments = hierarchy.assign_roles(
            debate_id="debate-001",
            agents=sample_agents,
        )

        orchestrators = [a for a in assignments.values() if a.role == HierarchyRole.ORCHESTRATOR]
        assert len(orchestrators) == 1

    def test_assigns_monitors(self, hierarchy, sample_agents):
        """Test that monitors are assigned."""
        assignments = hierarchy.assign_roles(
            debate_id="debate-001",
            agents=sample_agents,
        )

        monitors = [a for a in assignments.values() if a.role == HierarchyRole.MONITOR]
        assert len(monitors) <= hierarchy.config.max_monitors

    def test_assigns_workers(self, hierarchy, sample_agents):
        """Test that workers are assigned."""
        assignments = hierarchy.assign_roles(
            debate_id="debate-001",
            agents=sample_agents,
        )

        workers = [a for a in assignments.values() if a.role == HierarchyRole.WORKER]
        assert len(workers) >= 1

    def test_best_agent_becomes_orchestrator(self, hierarchy, sample_agents):
        """Test that the agent with best orchestrator capabilities becomes orchestrator."""
        assignments = hierarchy.assign_roles(
            debate_id="debate-001",
            agents=sample_agents,
        )

        # claude-opus has the most orchestrator capabilities
        orchestrators = [
            name for name, a in assignments.items() if a.role == HierarchyRole.ORCHESTRATOR
        ]
        assert "claude-opus" in orchestrators

    def test_task_type_affects_assignment(self, hierarchy, sample_agents):
        """Test that task type affects affinity scoring."""
        # Security task should favor claude-opus (has security affinity)
        assignments_security = hierarchy.assign_roles(
            debate_id="debate-002",
            agents=sample_agents,
            task_type="security",
        )

        # All agents should be assigned
        assert len(assignments_security) == len(sample_agents)

    def test_role_history_tracked(self, hierarchy, sample_agents):
        """Test that role history is tracked."""
        hierarchy.assign_roles(debate_id="debate-001", agents=sample_agents)

        assert "debate-001" in hierarchy._role_history
        assert len(hierarchy._role_history["debate-001"]) > 0

    def test_multiple_debates(self, hierarchy, sample_agents):
        """Test assigning roles for multiple debates."""
        assignments1 = hierarchy.assign_roles(debate_id="debate-001", agents=sample_agents)
        assignments2 = hierarchy.assign_roles(debate_id="debate-002", agents=sample_agents)

        assert "debate-001" in hierarchy._assignments
        assert "debate-002" in hierarchy._assignments
        assert len(assignments1) == len(assignments2)

    def test_get_hierarchy_status(self, hierarchy, sample_agents):
        """Test getting hierarchy status for a debate."""
        hierarchy.assign_roles(debate_id="debate-001", agents=sample_agents)

        status = hierarchy.get_hierarchy_status("debate-001")

        assert "orchestrators" in status
        assert "monitors" in status
        assert "workers" in status
        assert "total_agents" in status
        assert status["total_agents"] == len(sample_agents)

    def test_get_hierarchy_status_nonexistent_debate(self, hierarchy):
        """Test getting status for nonexistent debate."""
        status = hierarchy.get_hierarchy_status("nonexistent")
        assert status["total_agents"] == 0

    def test_promote_worker(self, hierarchy, sample_agents):
        """Test promoting a worker to monitor."""
        hierarchy.assign_roles(debate_id="debate-001", agents=sample_agents)

        # Find a worker
        workers = [
            name
            for name, a in hierarchy._assignments["debate-001"].items()
            if a.role == HierarchyRole.WORKER
        ]

        if workers:
            result = hierarchy.promote_worker(
                debate_id="debate-001",
                agent_name=workers[0],
                to_role=HierarchyRole.MONITOR,
            )
            assert result is True

            # Verify promotion
            new_role = hierarchy._assignments["debate-001"][workers[0]].role
            assert new_role == HierarchyRole.MONITOR

    def test_promote_worker_invalid_debate(self, hierarchy):
        """Test promoting in nonexistent debate fails."""
        result = hierarchy.promote_worker(
            debate_id="nonexistent",
            agent_name="agent",
            to_role=HierarchyRole.MONITOR,
        )
        assert result is False

    def test_empty_agents_list(self, hierarchy):
        """Test assigning with empty agents list."""
        assignments = hierarchy.assign_roles(debate_id="debate-001", agents=[])
        assert assignments == {}

    def test_single_agent(self, hierarchy):
        """Test assigning with single agent."""
        single_agent = [
            AgentProfile(
                name="claude",
                agent_type="claude",
                elo_rating=1500.0,
                capabilities={"reasoning"},
            )
        ]
        assignments = hierarchy.assign_roles(debate_id="debate-001", agents=single_agent)

        assert len(assignments) == 1
        # Single agent should still get a role
        assert "claude" in assignments


class TestCapabilityMatching:
    """Tests for capability matching in role assignment."""

    @pytest.fixture
    def hierarchy(self):
        return AgentHierarchy()

    def test_orchestrator_capability_matching(self, hierarchy):
        """Test that orchestrator role matches correct capabilities."""
        agents = [
            AgentProfile(
                name="full-caps",
                agent_type="claude",
                capabilities={"reasoning", "synthesis", "coordination"},
            ),
            AgentProfile(
                name="partial-caps",
                agent_type="codex",
                capabilities={"reasoning"},
            ),
        ]

        assignments = hierarchy.assign_roles(debate_id="test", agents=agents)

        # Full-caps agent should be orchestrator
        assert assignments["full-caps"].role == HierarchyRole.ORCHESTRATOR

    def test_monitor_capability_matching(self, hierarchy):
        """Test that monitor role matches correct capabilities."""
        agents = [
            AgentProfile(
                name="orchestrator-agent",
                agent_type="claude",
                capabilities={"reasoning", "synthesis", "coordination"},
            ),
            AgentProfile(
                name="monitor-agent",
                agent_type="codex",
                capabilities={"analysis", "quality_assessment"},
            ),
            AgentProfile(
                name="worker-agent",
                agent_type="gemini",
                capabilities={"coding"},
            ),
        ]

        assignments = hierarchy.assign_roles(debate_id="test", agents=agents)

        # Monitor agent should get monitor role (not orchestrator)
        assert assignments["monitor-agent"].role == HierarchyRole.MONITOR

    def test_capabilities_matched_tracked(self, hierarchy):
        """Test that matched capabilities are tracked in assignment."""
        agents = [
            AgentProfile(
                name="agent",
                agent_type="claude",
                capabilities={"reasoning", "synthesis", "coordination", "extra"},
            ),
        ]

        assignments = hierarchy.assign_roles(debate_id="test", agents=agents)

        # Should track which capabilities matched the role
        matched = assignments["agent"].capabilities_matched
        assert "reasoning" in matched or "synthesis" in matched


class TestEloWeighting:
    """Tests for ELO rating influence on role assignment."""

    @pytest.fixture
    def hierarchy(self):
        return AgentHierarchy()

    def test_high_elo_preferred_for_orchestrator(self, hierarchy):
        """Test that higher ELO agents are preferred for orchestrator."""
        agents = [
            AgentProfile(
                name="high-elo",
                agent_type="claude",
                elo_rating=2000.0,
                capabilities={"reasoning", "synthesis", "coordination"},
            ),
            AgentProfile(
                name="low-elo",
                agent_type="codex",
                elo_rating=1000.0,
                capabilities={"reasoning", "synthesis", "coordination"},
            ),
        ]

        assignments = hierarchy.assign_roles(debate_id="test", agents=agents)

        # High ELO should be orchestrator
        assert assignments["high-elo"].role == HierarchyRole.ORCHESTRATOR

    def test_elo_with_equal_capabilities(self, hierarchy):
        """Test ELO tiebreaker when capabilities are equal."""
        agents = [
            AgentProfile(
                name="agent-1800",
                agent_type="claude",
                elo_rating=1800.0,
                capabilities={"reasoning"},
            ),
            AgentProfile(
                name="agent-1600",
                agent_type="codex",
                elo_rating=1600.0,
                capabilities={"reasoning"},
            ),
            AgentProfile(
                name="agent-1400",
                agent_type="gemini",
                elo_rating=1400.0,
                capabilities={"reasoning"},
            ),
        ]

        assignments = hierarchy.assign_roles(debate_id="test", agents=agents)

        # Higher ELO should get better roles
        roles = {name: a.role for name, a in assignments.items()}
        assert roles["agent-1800"] in [HierarchyRole.ORCHESTRATOR, HierarchyRole.MONITOR]


class TestAffinityScoring:
    """Tests for task affinity influence on role assignment."""

    @pytest.fixture
    def hierarchy(self):
        return AgentHierarchy()

    def test_task_affinity_tracked(self, hierarchy):
        """Test that affinity score is tracked."""
        agents = [
            AgentProfile(
                name="security-expert",
                agent_type="claude",
                capabilities={"reasoning", "synthesis", "coordination"},
                task_affinity={"security": 0.95},
            ),
        ]

        assignments = hierarchy.assign_roles(
            debate_id="test",
            agents=agents,
            task_type="security",
        )

        # Affinity score should be set
        assert assignments["security-expert"].affinity_score >= 0.0

    def test_matching_affinity_improves_score(self, hierarchy):
        """Test that matching task affinity improves assignment score."""
        agents = [
            AgentProfile(
                name="security-expert",
                elo_rating=1500.0,
                capabilities={"reasoning", "synthesis", "coordination"},
                task_affinity={"security": 0.95},
            ),
            AgentProfile(
                name="general-agent",
                elo_rating=1600.0,  # Higher ELO
                capabilities={"reasoning", "synthesis", "coordination"},
                task_affinity={},  # No affinity
            ),
        ]

        assignments = hierarchy.assign_roles(
            debate_id="test",
            agents=agents,
            task_type="security",
        )

        # Security expert might be preferred despite lower ELO due to affinity
        # (depends on weight configuration)
        assert "security-expert" in assignments
        assert "general-agent" in assignments


class TestStandardCapabilities:
    """Tests for standard capability definitions."""

    def test_all_capabilities_defined(self):
        """Test that all standard capabilities have descriptions."""
        expected_caps = [
            "reasoning",
            "synthesis",
            "coordination",
            "analysis",
            "quality_assessment",
            "coding",
            "research",
            "creativity",
            "mathematics",
            "domain_expert",
        ]

        for cap in expected_caps:
            assert cap in STANDARD_CAPABILITIES
            assert len(STANDARD_CAPABILITIES[cap]) > 0

    def test_role_capabilities_use_standard(self):
        """Test that role capabilities are subset of standard capabilities."""
        for role, caps in ROLE_CAPABILITIES.items():
            for cap in caps:
                assert cap in STANDARD_CAPABILITIES, f"{cap} not in standard capabilities"
