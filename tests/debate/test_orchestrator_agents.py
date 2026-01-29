"""Tests for orchestrator_agents.py - Agent selection, quality filtering, and hierarchy helpers.

Tests cover:
- select_debate_team: ML delegation, performance selection, fallbacks
- filter_responses_by_quality: Quality gates, error handling
- should_terminate_early: Consensus estimation
- init_agent_hierarchy: Hierarchy initialization
- assign_hierarchy_roles: Role assignment to agents
- get_fabric_agents_sync: Synchronous fabric agent retrieval
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent
from aragora.debate.protocol import DebateProtocol


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, name: str = "mock-agent", model: str = "mock-model"):
        super().__init__(name=name, model=model, role="proposer")
        self.provider = "test-provider"
        self.elo_rating = 1500.0
        self.capabilities = {"reasoning", "debate"}
        self.task_affinity = {"general": 0.8}

    async def generate(self, prompt: str, context: list = None) -> str:
        return "Test response"

    async def critique(self, proposal: str, task: str, context: list = None, target_agent: str = None):
        return MagicMock()

    async def vote(self, proposals: dict, task: str):
        return MagicMock()


# =============================================================================
# Tests for select_debate_team
# =============================================================================


class TestSelectDebateTeam:
    """Tests for select_debate_team function."""

    @pytest.fixture
    def agents(self):
        return [MockAgent(name=f"agent-{i}") for i in range(3)]

    @pytest.fixture
    def env(self):
        env = MagicMock()
        env.task = "Test debate task"
        return env

    @pytest.fixture
    def extract_domain_fn(self):
        return lambda: "general"

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=3)

    @pytest.fixture
    def mock_ml_strategy(self):
        strategy = MagicMock()
        strategy.select_agents = MagicMock(return_value=[MockAgent(name="ml-selected")])
        return strategy

    @pytest.fixture
    def mock_agent_pool(self):
        pool = MagicMock()
        pool.select_team = MagicMock(return_value=[MockAgent(name="pool-selected")])
        return pool

    def test_returns_original_agents_when_ml_delegation_disabled(
        self, agents, env, extract_domain_fn, protocol, mock_agent_pool
    ):
        """When ML delegation is disabled, return original agents."""
        from aragora.debate.orchestrator_agents import select_debate_team

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=False,
            ml_delegation_strategy=None,
            protocol=protocol,
            use_performance_selection=False,
            agent_pool=mock_agent_pool,
        )
        assert result == agents

    def test_ml_delegation_selected_agents_on_success(
        self, agents, env, extract_domain_fn, protocol, mock_ml_strategy, mock_agent_pool
    ):
        """When ML delegation succeeds, return ML-selected agents."""
        from aragora.debate.orchestrator_agents import select_debate_team

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=True,
            ml_delegation_strategy=mock_ml_strategy,
            protocol=protocol,
            use_performance_selection=False,
            agent_pool=mock_agent_pool,
        )
        assert len(result) == 1
        assert result[0].name == "ml-selected"
        mock_ml_strategy.select_agents.assert_called_once()

    def test_ml_delegation_fallback_on_value_error(
        self, agents, env, extract_domain_fn, protocol, mock_ml_strategy, mock_agent_pool
    ):
        """When ML delegation raises ValueError, fall back to original agents."""
        from aragora.debate.orchestrator_agents import select_debate_team

        mock_ml_strategy.select_agents.side_effect = ValueError("Invalid selection")

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=True,
            ml_delegation_strategy=mock_ml_strategy,
            protocol=protocol,
            use_performance_selection=False,
            agent_pool=mock_agent_pool,
        )
        assert result == agents

    def test_ml_delegation_fallback_on_type_error(
        self, agents, env, extract_domain_fn, protocol, mock_ml_strategy, mock_agent_pool
    ):
        """When ML delegation raises TypeError, fall back to original agents."""
        from aragora.debate.orchestrator_agents import select_debate_team

        mock_ml_strategy.select_agents.side_effect = TypeError("Type mismatch")

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=True,
            ml_delegation_strategy=mock_ml_strategy,
            protocol=protocol,
            use_performance_selection=False,
            agent_pool=mock_agent_pool,
        )
        assert result == agents

    def test_ml_delegation_fallback_on_key_error(
        self, agents, env, extract_domain_fn, protocol, mock_ml_strategy, mock_agent_pool
    ):
        """When ML delegation raises KeyError, fall back to original agents."""
        from aragora.debate.orchestrator_agents import select_debate_team

        mock_ml_strategy.select_agents.side_effect = KeyError("Missing key")

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=True,
            ml_delegation_strategy=mock_ml_strategy,
            protocol=protocol,
            use_performance_selection=False,
            agent_pool=mock_agent_pool,
        )
        assert result == agents

    def test_ml_delegation_fallback_on_unexpected_error(
        self, agents, env, extract_domain_fn, protocol, mock_ml_strategy, mock_agent_pool
    ):
        """When ML delegation raises unexpected exception, fall back to original agents."""
        from aragora.debate.orchestrator_agents import select_debate_team

        mock_ml_strategy.select_agents.side_effect = RuntimeError("Unexpected error")

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=True,
            ml_delegation_strategy=mock_ml_strategy,
            protocol=protocol,
            use_performance_selection=False,
            agent_pool=mock_agent_pool,
        )
        assert result == agents

    def test_performance_selection_when_ml_disabled(
        self, agents, env, extract_domain_fn, protocol, mock_agent_pool
    ):
        """When ML is disabled but performance selection is enabled, use agent pool."""
        from aragora.debate.orchestrator_agents import select_debate_team

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=False,
            ml_delegation_strategy=None,
            protocol=protocol,
            use_performance_selection=True,
            agent_pool=mock_agent_pool,
        )
        assert len(result) == 1
        assert result[0].name == "pool-selected"
        mock_agent_pool.select_team.assert_called_once()

    def test_ml_delegation_priority_over_performance(
        self, agents, env, extract_domain_fn, protocol, mock_ml_strategy, mock_agent_pool
    ):
        """ML delegation takes priority over performance selection."""
        from aragora.debate.orchestrator_agents import select_debate_team

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=True,
            ml_delegation_strategy=mock_ml_strategy,
            protocol=protocol,
            use_performance_selection=True,
            agent_pool=mock_agent_pool,
        )
        assert result[0].name == "ml-selected"
        mock_agent_pool.select_team.assert_not_called()

    def test_fallback_to_performance_after_ml_failure(
        self, agents, env, extract_domain_fn, protocol, mock_ml_strategy, mock_agent_pool
    ):
        """After ML delegation failure, fall back to performance selection if enabled."""
        from aragora.debate.orchestrator_agents import select_debate_team

        mock_ml_strategy.select_agents.side_effect = ValueError("ML failed")

        result = select_debate_team(
            agents=agents,
            env=env,
            extract_domain_fn=extract_domain_fn,
            enable_ml_delegation=True,
            ml_delegation_strategy=mock_ml_strategy,
            protocol=protocol,
            use_performance_selection=True,
            agent_pool=mock_agent_pool,
        )
        assert result[0].name == "pool-selected"


# =============================================================================
# Tests for filter_responses_by_quality
# =============================================================================


class TestFilterResponsesByQuality:
    """Tests for filter_responses_by_quality function."""

    @pytest.fixture
    def responses(self):
        return [
            ("agent-1", "High quality response with detailed analysis"),
            ("agent-2", "Medium quality response"),
            ("agent-3", "Low quality"),
        ]

    @pytest.fixture
    def mock_quality_gate(self):
        gate = MagicMock()
        gate.filter_responses = MagicMock(
            return_value=[("agent-1", "High quality response with detailed analysis")]
        )
        return gate

    def test_returns_all_responses_when_gates_disabled(self, responses):
        """When quality gates are disabled, return all responses."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        result = filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=False,
            ml_quality_gate=MagicMock(),
            task="Test task",
        )
        assert result == responses

    def test_returns_all_responses_when_gate_is_none(self, responses):
        """When quality gate is None, return all responses."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        result = filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=True,
            ml_quality_gate=None,
            task="Test task",
        )
        assert result == responses

    def test_filters_responses_on_success(self, responses, mock_quality_gate):
        """When quality gate succeeds, return filtered responses."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        result = filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=True,
            ml_quality_gate=mock_quality_gate,
            task="Test task",
        )
        assert len(result) == 1
        assert result[0][0] == "agent-1"

    def test_fallback_on_value_error(self, responses, mock_quality_gate):
        """When quality gate raises ValueError, return all responses."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        mock_quality_gate.filter_responses.side_effect = ValueError("Invalid")

        result = filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=True,
            ml_quality_gate=mock_quality_gate,
            task="Test task",
        )
        assert result == responses

    def test_fallback_on_type_error(self, responses, mock_quality_gate):
        """When quality gate raises TypeError, return all responses."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        mock_quality_gate.filter_responses.side_effect = TypeError("Type mismatch")

        result = filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=True,
            ml_quality_gate=mock_quality_gate,
            task="Test task",
        )
        assert result == responses

    def test_fallback_on_key_error(self, responses, mock_quality_gate):
        """When quality gate raises KeyError, return all responses."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        mock_quality_gate.filter_responses.side_effect = KeyError("Missing")

        result = filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=True,
            ml_quality_gate=mock_quality_gate,
            task="Test task",
        )
        assert result == responses

    def test_fallback_on_attribute_error(self, responses, mock_quality_gate):
        """When quality gate raises AttributeError, return all responses."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        mock_quality_gate.filter_responses.side_effect = AttributeError("No attr")

        result = filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=True,
            ml_quality_gate=mock_quality_gate,
            task="Test task",
        )
        assert result == responses

    def test_fallback_on_unexpected_error(self, responses, mock_quality_gate):
        """When quality gate raises unexpected error, return all responses."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        mock_quality_gate.filter_responses.side_effect = RuntimeError("Unexpected")

        result = filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=True,
            ml_quality_gate=mock_quality_gate,
            task="Test task",
        )
        assert result == responses

    def test_handles_empty_responses(self, mock_quality_gate):
        """Handle empty response list gracefully."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        mock_quality_gate.filter_responses.return_value = []

        result = filter_responses_by_quality(
            responses=[],
            enable_quality_gates=True,
            ml_quality_gate=mock_quality_gate,
            task="Test task",
        )
        assert result == []

    def test_uses_context_parameter(self, responses, mock_quality_gate):
        """Quality gate receives context parameter."""
        from aragora.debate.orchestrator_agents import filter_responses_by_quality

        filter_responses_by_quality(
            responses=responses,
            enable_quality_gates=True,
            ml_quality_gate=mock_quality_gate,
            task="Test task",
            context="Custom context",
        )

        mock_quality_gate.filter_responses.assert_called_once_with(
            responses, context="Custom context"
        )


# =============================================================================
# Tests for should_terminate_early
# =============================================================================


class TestShouldTerminateEarly:
    """Tests for should_terminate_early function."""

    @pytest.fixture
    def responses(self):
        return [("agent-1", "Response 1"), ("agent-2", "Response 2")]

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=5)

    @pytest.fixture
    def mock_estimator(self):
        estimator = MagicMock()
        estimator.should_terminate_early = MagicMock(return_value=True)
        return estimator

    def test_returns_false_when_estimation_disabled(self, responses, protocol):
        """When consensus estimation is disabled, return False."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        result = should_terminate_early(
            responses=responses,
            current_round=2,
            enable_consensus_estimation=False,
            ml_consensus_estimator=MagicMock(),
            protocol=protocol,
            task="Test task",
        )
        assert result is False

    def test_returns_false_when_estimator_is_none(self, responses, protocol):
        """When estimator is None, return False."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        result = should_terminate_early(
            responses=responses,
            current_round=2,
            enable_consensus_estimation=True,
            ml_consensus_estimator=None,
            protocol=protocol,
            task="Test task",
        )
        assert result is False

    def test_true_when_consensus_detected(self, responses, protocol, mock_estimator):
        """When estimator recommends termination, return True."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        result = should_terminate_early(
            responses=responses,
            current_round=2,
            enable_consensus_estimation=True,
            ml_consensus_estimator=mock_estimator,
            protocol=protocol,
            task="Test task",
        )
        assert result is True

    def test_false_when_no_consensus(self, responses, protocol, mock_estimator):
        """When estimator returns False, return False."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        mock_estimator.should_terminate_early.return_value = False

        result = should_terminate_early(
            responses=responses,
            current_round=2,
            enable_consensus_estimation=True,
            ml_consensus_estimator=mock_estimator,
            protocol=protocol,
            task="Test task",
        )
        assert result is False

    def test_catches_value_error_returns_false(self, responses, protocol, mock_estimator):
        """When estimator raises ValueError, return False."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        mock_estimator.should_terminate_early.side_effect = ValueError("Invalid")

        result = should_terminate_early(
            responses=responses,
            current_round=2,
            enable_consensus_estimation=True,
            ml_consensus_estimator=mock_estimator,
            protocol=protocol,
            task="Test task",
        )
        assert result is False

    def test_catches_type_error_returns_false(self, responses, protocol, mock_estimator):
        """When estimator raises TypeError, return False."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        mock_estimator.should_terminate_early.side_effect = TypeError("Type mismatch")

        result = should_terminate_early(
            responses=responses,
            current_round=2,
            enable_consensus_estimation=True,
            ml_consensus_estimator=mock_estimator,
            protocol=protocol,
            task="Test task",
        )
        assert result is False

    def test_catches_key_error_returns_false(self, responses, protocol, mock_estimator):
        """When estimator raises KeyError, return False."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        mock_estimator.should_terminate_early.side_effect = KeyError("Missing")

        result = should_terminate_early(
            responses=responses,
            current_round=2,
            enable_consensus_estimation=True,
            ml_consensus_estimator=mock_estimator,
            protocol=protocol,
            task="Test task",
        )
        assert result is False

    def test_catches_unexpected_error_returns_false(self, responses, protocol, mock_estimator):
        """When estimator raises unexpected error, return False."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        mock_estimator.should_terminate_early.side_effect = RuntimeError("Unexpected")

        result = should_terminate_early(
            responses=responses,
            current_round=2,
            enable_consensus_estimation=True,
            ml_consensus_estimator=mock_estimator,
            protocol=protocol,
            task="Test task",
        )
        assert result is False

    def test_passes_correct_parameters(self, responses, protocol, mock_estimator):
        """Estimator receives correct parameters."""
        from aragora.debate.orchestrator_agents import should_terminate_early

        should_terminate_early(
            responses=responses,
            current_round=3,
            enable_consensus_estimation=True,
            ml_consensus_estimator=mock_estimator,
            protocol=protocol,
            task="Test task",
        )

        mock_estimator.should_terminate_early.assert_called_once_with(
            responses=responses,
            current_round=3,
            total_rounds=5,
            context="Test task",
        )


# =============================================================================
# Tests for init_agent_hierarchy
# =============================================================================


class TestInitAgentHierarchy:
    """Tests for init_agent_hierarchy function."""

    def test_returns_none_when_disabled(self):
        """When hierarchy is disabled, return None."""
        from aragora.debate.orchestrator_agents import init_agent_hierarchy

        result = init_agent_hierarchy(
            enable_agent_hierarchy=False,
            hierarchy_config=None,
        )
        assert result is None

    def test_creates_hierarchy_with_default_config(self):
        """When enabled without config, create hierarchy with defaults."""
        from aragora.debate.orchestrator_agents import init_agent_hierarchy

        result = init_agent_hierarchy(
            enable_agent_hierarchy=True,
            hierarchy_config=None,
        )
        assert result is not None

    def test_creates_hierarchy_with_custom_config(self):
        """When enabled with config, create hierarchy with custom settings."""
        from aragora.debate.hierarchy import HierarchyConfig
        from aragora.debate.orchestrator_agents import init_agent_hierarchy

        config = HierarchyConfig(max_orchestrators=2, max_monitors=3)
        result = init_agent_hierarchy(
            enable_agent_hierarchy=True,
            hierarchy_config=config,
        )
        assert result is not None


# =============================================================================
# Tests for assign_hierarchy_roles
# =============================================================================


class TestAssignHierarchyRoles:
    """Tests for assign_hierarchy_roles function."""

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.debate_id = "test-debate-123"
        ctx.agents = [MockAgent(name=f"agent-{i}") for i in range(3)]
        ctx.hierarchy_assignments = {}
        return ctx

    @pytest.fixture
    def mock_hierarchy(self):
        hierarchy = MagicMock()
        hierarchy.assign_roles = MagicMock(return_value={
            "agent-0": MagicMock(role=MagicMock(value="orchestrator")),
            "agent-1": MagicMock(role=MagicMock(value="worker")),
            "agent-2": MagicMock(role=MagicMock(value="worker")),
        })
        return hierarchy

    def test_skips_when_hierarchy_disabled(self, mock_ctx):
        """When hierarchy is disabled, do nothing."""
        from aragora.debate.orchestrator_agents import assign_hierarchy_roles

        assign_hierarchy_roles(
            ctx=mock_ctx,
            enable_agent_hierarchy=False,
            hierarchy=MagicMock(),
        )
        assert mock_ctx.hierarchy_assignments == {}

    def test_skips_when_hierarchy_is_none(self, mock_ctx):
        """When hierarchy is None, do nothing."""
        from aragora.debate.orchestrator_agents import assign_hierarchy_roles

        assign_hierarchy_roles(
            ctx=mock_ctx,
            enable_agent_hierarchy=True,
            hierarchy=None,
        )
        assert mock_ctx.hierarchy_assignments == {}

    def test_assigns_roles_to_agents(self, mock_ctx, mock_hierarchy):
        """When enabled with hierarchy, assign roles to agents."""
        from aragora.debate.orchestrator_agents import assign_hierarchy_roles

        assign_hierarchy_roles(
            ctx=mock_ctx,
            enable_agent_hierarchy=True,
            hierarchy=mock_hierarchy,
        )
        assert mock_ctx.hierarchy_assignments != {}
        mock_hierarchy.assign_roles.assert_called_once()

    def test_stores_roles_in_context(self, mock_ctx, mock_hierarchy):
        """Assignments are stored in context."""
        from aragora.debate.orchestrator_agents import assign_hierarchy_roles

        assign_hierarchy_roles(
            ctx=mock_ctx,
            enable_agent_hierarchy=True,
            hierarchy=mock_hierarchy,
        )
        assert "agent-0" in mock_ctx.hierarchy_assignments

    def test_handles_import_error_gracefully(self, mock_ctx, mock_hierarchy):
        """When AgentProfile import fails, handle gracefully."""
        from aragora.debate.orchestrator_agents import assign_hierarchy_roles

        with patch(
            "aragora.debate.orchestrator_agents.AgentProfile",
            side_effect=ImportError("Not found"),
        ):
            assign_hierarchy_roles(
                ctx=mock_ctx,
                enable_agent_hierarchy=True,
                hierarchy=mock_hierarchy,
            )
        # Should not raise, assignments should be empty
        assert mock_ctx.hierarchy_assignments == {}

    def test_handles_value_error_gracefully(self, mock_ctx, mock_hierarchy):
        """When role assignment raises ValueError, handle gracefully."""
        from aragora.debate.orchestrator_agents import assign_hierarchy_roles

        mock_hierarchy.assign_roles.side_effect = ValueError("Invalid")

        assign_hierarchy_roles(
            ctx=mock_ctx,
            enable_agent_hierarchy=True,
            hierarchy=mock_hierarchy,
        )
        assert mock_ctx.hierarchy_assignments == {}


# =============================================================================
# Tests for get_fabric_agents_sync
# =============================================================================


class TestGetFabricAgentsSync:
    """Tests for get_fabric_agents_sync function."""

    @pytest.fixture
    def mock_fabric(self):
        fabric = MagicMock()
        mock_pool = MagicMock()
        mock_pool.current_agents = ["agent-1", "agent-2", "agent-3"]
        mock_pool.model = "test-model"
        fabric.get_pool = AsyncMock(return_value=mock_pool)
        return fabric

    @pytest.fixture
    def mock_fabric_config(self):
        config = MagicMock()
        config.pool_id = "test-pool"
        config.max_agents = 2
        return config

    def test_raises_runtime_error_in_async_context(self, mock_fabric, mock_fabric_config):
        """When called in async context, raise RuntimeError."""
        from aragora.debate.orchestrator_agents import get_fabric_agents_sync

        async def test_in_async():
            with pytest.raises(RuntimeError, match="Cannot use sync helper"):
                get_fabric_agents_sync(mock_fabric, mock_fabric_config)

        asyncio.run(test_in_async())

    def test_raises_value_error_when_pool_not_found(self, mock_fabric_config):
        """When pool not found, raise ValueError."""
        from aragora.debate.orchestrator_agents import get_fabric_agents_sync

        mock_fabric = MagicMock()
        mock_fabric.get_pool = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not found"):
            get_fabric_agents_sync(mock_fabric, mock_fabric_config)

    def test_respects_max_agents_limit(self, mock_fabric, mock_fabric_config):
        """Respects max_agents configuration."""
        from aragora.debate.orchestrator_agents import get_fabric_agents_sync

        with patch("aragora.debate.orchestrator_agents.FabricAgentAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            result = get_fabric_agents_sync(mock_fabric, mock_fabric_config)

        # max_agents is 2, so should only create 2 adapters
        assert len(result) == 2
