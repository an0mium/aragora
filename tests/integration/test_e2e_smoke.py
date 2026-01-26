"""
End-to-End Smoke Tests.

Quick verification tests that core functionality works end-to-end.
These tests run fast and verify critical paths before deployment.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.core import Environment, DebateProtocol
from aragora.debate.orchestrator import Arena
from aragora.server.handlers.rlm import RLMContextHandler
from aragora.server.handlers.admin.health import HealthHandler


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agents():
    """Create minimal mock agents for smoke tests."""
    agents = []
    for name in ["agent_a", "agent_b"]:
        agent = MagicMock()
        agent.name = name
        agent.generate = AsyncMock(return_value=f"Response from {name}")
        agent.get_metrics = MagicMock(return_value={})
        agents.append(agent)
    return agents


@pytest.fixture
def simple_environment():
    """Create a simple test environment."""
    return Environment(task="Test smoke question")


@pytest.fixture
def handler_context():
    """Create a mock handler context."""
    return {
        "storage": None,
        "nomic_dir": None,
    }


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    return handler


# =============================================================================
# Core Debate Flow Smoke Tests
# =============================================================================


class TestCoreDebateFlowSmoke:
    """Smoke tests for core debate functionality."""

    @pytest.mark.asyncio
    async def test_arena_initializes(self, mock_agents, simple_environment):
        """Verify Arena can be initialized with minimal config."""
        protocol = DebateProtocol(rounds=1)
        arena = Arena(simple_environment, mock_agents, protocol)

        assert arena is not None
        # Arena stores agents internally
        assert hasattr(arena, "agents") or hasattr(arena, "_agents")

    @pytest.mark.asyncio
    async def test_arena_from_config_initializes(self, mock_agents, simple_environment):
        """Verify Arena.from_config factory works."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        arena = Arena.from_config(
            environment=simple_environment,
            agents=mock_agents,
            config=config,
        )

        assert arena is not None
        # Just verify the factory method returns an Arena instance
        assert isinstance(arena, Arena)


# =============================================================================
# Handler Smoke Tests
# =============================================================================


class TestHandlerSmoke:
    """Smoke tests for critical handlers."""

    @pytest.mark.asyncio
    async def test_health_handler_responds(self, handler_context, mock_http_handler):
        """Verify health endpoint responds."""
        handler = HealthHandler(handler_context)
        result = await handler.handle("/api/v1/health", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "status" in body

    def test_rlm_handler_initializes(self, handler_context):
        """Verify RLM handler can be initialized."""
        handler = RLMContextHandler(handler_context)
        assert handler is not None
        assert handler.can_handle("/api/v1/rlm/stats")

    @pytest.mark.asyncio
    async def test_rlm_strategies_endpoint(self, handler_context, mock_http_handler):
        """Verify RLM strategies endpoint works."""
        handler = RLMContextHandler(handler_context)
        result = await handler.handle("/api/v1/rlm/strategies", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "strategies" in body
        assert "auto" in body["strategies"]


# =============================================================================
# Data Model Smoke Tests
# =============================================================================


class TestDataModelSmoke:
    """Smoke tests for core data models."""

    def test_environment_creates(self):
        """Verify Environment model creates correctly."""
        env = Environment(task="Test task", context={"key": "value"})
        assert env.task == "Test task"
        assert env.context == {"key": "value"}

    def test_debate_protocol_creates(self):
        """Verify DebateProtocol creates with defaults."""
        protocol = DebateProtocol()
        assert protocol.rounds > 0
        assert protocol.consensus in ["majority", "unanimous", "judge"]

    def test_debate_protocol_custom(self):
        """Verify DebateProtocol accepts custom values."""
        protocol = DebateProtocol(
            rounds=5,
            consensus="unanimous",
            early_stopping=True,
        )
        assert protocol.rounds == 5
        assert protocol.consensus == "unanimous"
        assert protocol.early_stopping is True


# =============================================================================
# Feature Validator Smoke Tests
# =============================================================================


class TestFeatureValidatorSmoke:
    """Smoke tests for feature validation."""

    def test_validator_imports(self):
        """Verify feature validator can be imported."""
        from aragora.debate.feature_validator import (
            validate_feature_dependencies,
            validate_and_warn,
            FEATURE_DEPENDENCIES,
        )

        assert validate_feature_dependencies is not None
        assert validate_and_warn is not None
        assert len(FEATURE_DEPENDENCIES) > 0

    def test_validator_runs_on_config(self):
        """Verify validator runs without error on default config."""
        from aragora.debate.arena_config import ArenaConfig
        from aragora.debate.feature_validator import validate_feature_dependencies

        config = ArenaConfig()
        result = validate_feature_dependencies(config)

        assert result is not None
        assert hasattr(result, "valid")
        assert hasattr(result, "warnings")
        assert hasattr(result, "errors")


# =============================================================================
# Integration Smoke Tests
# =============================================================================


class TestIntegrationSmoke:
    """Quick integration smoke tests."""

    def test_knowledge_source_enum_has_debate(self):
        """Verify KnowledgeSource.DEBATE exists."""
        from aragora.knowledge.unified.types import KnowledgeSource

        assert hasattr(KnowledgeSource, "DEBATE")
        assert KnowledgeSource.DEBATE.value == "debate"

    def test_arena_config_to_kwargs(self):
        """Verify ArenaConfig.to_arena_kwargs works."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(
            enable_performance_monitor=True,
        )
        kwargs = config.to_arena_kwargs()

        assert isinstance(kwargs, dict)
        assert kwargs.get("enable_performance_monitor") is True

    @pytest.mark.asyncio
    async def test_full_debate_smoke(self, mock_agents, simple_environment):
        """Quick smoke test of full debate flow."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        protocol = DebateProtocol(rounds=1, timeout_seconds=30)

        arena = Arena.from_config(
            environment=simple_environment,
            agents=mock_agents,
            protocol=protocol,
            config=config,
        )

        # Just verify initialization succeeds
        assert arena is not None
        assert arena.protocol.rounds == 1
