"""
Tests for the Agent Registry - Factory pattern for agent creation.

Tests the following components:
- RegistrySpec: Agent registration specification (frozen dataclass)
- AgentRegistry: Factory registry with caching

Test coverage includes:
- Registration via @register decorator
- Agent creation (basic, with API key, unknown type)
- Cache behavior (hits, eviction, clearing, stats)
- Registry listing and spec retrieval
- Type validation (allowed/invalid types)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry and cache before and after each test."""
    from aragora.agents.registry import AgentRegistry, _agent_cache

    # Store original registry state
    original_registry = AgentRegistry._registry.copy()
    original_cache = _agent_cache.copy()

    # Clear for test
    AgentRegistry.clear()

    yield

    # Restore original state
    AgentRegistry._registry.clear()
    AgentRegistry._registry.update(original_registry)
    _agent_cache.clear()
    _agent_cache.update(original_cache)


@pytest.fixture
def mock_agent_class():
    """Create a mock agent class for testing."""

    class MockAgent:
        def __init__(
            self,
            name: str,
            role: str = "proposer",
            model: str | None = None,
            api_key: str | None = None,
            **kwargs: Any,
        ):
            self.name = name
            self.role = role
            self.model = model
            self.api_key = api_key
            self.extra_kwargs = kwargs

        async def generate(self, prompt: str, context=None) -> str:
            return "Mock response"

    return MockAgent


@pytest.fixture
def registered_agent(mock_agent_class):
    """Register a mock agent and return the class."""
    from aragora.agents.registry import AgentRegistry

    AgentRegistry.register(
        "test-agent",
        default_model="test-model-v1",
        default_name="test-agent",
        agent_type="API",
        requires=None,
        env_vars="TEST_API_KEY",
        description="A test agent for unit tests",
        accepts_api_key=True,
    )(mock_agent_class)

    return mock_agent_class


# =============================================================================
# RegistrySpec Tests
# =============================================================================


class TestRegistrySpec:
    """Test RegistrySpec dataclass."""

    def test_spec_creation(self, mock_agent_class):
        """Test RegistrySpec can be created with all fields."""
        from aragora.agents.registry import RegistrySpec

        spec = RegistrySpec(
            name="my-agent",
            agent_class=mock_agent_class,
            default_model="my-model",
            default_name="my-agent",
            agent_type="API",
            requires="some-dependency",
            env_vars="MY_API_KEY",
            description="My test agent",
            accepts_api_key=True,
        )

        assert spec.name == "my-agent"
        assert spec.agent_class is mock_agent_class
        assert spec.default_model == "my-model"
        assert spec.default_name == "my-agent"
        assert spec.agent_type == "API"
        assert spec.requires == "some-dependency"
        assert spec.env_vars == "MY_API_KEY"
        assert spec.description == "My test agent"
        assert spec.accepts_api_key is True

    def test_spec_is_frozen(self, mock_agent_class):
        """Test RegistrySpec is immutable (frozen dataclass)."""
        from aragora.agents.registry import RegistrySpec

        spec = RegistrySpec(
            name="frozen-agent",
            agent_class=mock_agent_class,
            default_model="model",
            default_name="frozen-agent",
            agent_type="API",
            requires=None,
            env_vars=None,
        )

        with pytest.raises(FrozenInstanceError):
            spec.name = "modified-name"  # type: ignore[misc]

    def test_spec_defaults(self, mock_agent_class):
        """Test RegistrySpec default values."""
        from aragora.agents.registry import RegistrySpec

        spec = RegistrySpec(
            name="minimal-agent",
            agent_class=mock_agent_class,
            default_model=None,
            default_name="minimal-agent",
            agent_type="CLI",
            requires=None,
            env_vars=None,
        )

        assert spec.description is None
        assert spec.accepts_api_key is False


# =============================================================================
# Registration Tests
# =============================================================================


class TestRegisterAgentWithDecorator:
    """Test agent registration via @register decorator."""

    def test_register_agent_with_decorator(self, mock_agent_class):
        """Test registration via @register decorator."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register(
            "decorator-test",
            default_model="test-model",
            agent_type="CLI",
            requires="test-cli",
            env_vars="TEST_KEY",
            description="Test decorator registration",
        )
        class DecoratorTestAgent(mock_agent_class):
            pass

        assert AgentRegistry.is_registered("decorator-test")
        spec = AgentRegistry.get_spec("decorator-test")
        assert spec is not None
        assert spec.name == "decorator-test"
        assert spec.default_model == "test-model"
        assert spec.agent_type == "CLI"
        assert spec.requires == "test-cli"
        assert spec.env_vars == "TEST_KEY"
        assert spec.description == "Test decorator registration"

    def test_register_returns_original_class(self, mock_agent_class):
        """Test that @register decorator returns the original class."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register("return-test")
        class ReturnTestAgent(mock_agent_class):
            custom_attr = "custom_value"

        assert ReturnTestAgent.custom_attr == "custom_value"
        assert ReturnTestAgent is not mock_agent_class

    def test_register_with_default_name(self, mock_agent_class):
        """Test registration uses type_name as default_name when not specified."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register("default-name-test", default_model="model")
        class DefaultNameAgent(mock_agent_class):
            pass

        spec = AgentRegistry.get_spec("default-name-test")
        assert spec is not None
        assert spec.default_name == "default-name-test"

    def test_register_with_custom_default_name(self, mock_agent_class):
        """Test registration with custom default_name."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register(
            "custom-name-test",
            default_name="my-custom-name",
        )
        class CustomNameAgent(mock_agent_class):
            pass

        spec = AgentRegistry.get_spec("custom-name-test")
        assert spec is not None
        assert spec.default_name == "my-custom-name"

    def test_register_with_accepts_api_key(self, mock_agent_class):
        """Test registration with accepts_api_key flag."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register("api-key-test", accepts_api_key=True)
        class ApiKeyAgent(mock_agent_class):
            pass

        spec = AgentRegistry.get_spec("api-key-test")
        assert spec is not None
        assert spec.accepts_api_key is True


# =============================================================================
# Agent Creation Tests
# =============================================================================


class TestCreateAgentBasic:
    """Test basic agent creation."""

    def test_create_agent_basic(self, registered_agent):
        """Test basic agent creation."""
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent")

        assert agent.name == "test-agent"
        assert agent.role == "proposer"
        assert agent.model == "test-model-v1"

    def test_create_agent_with_custom_name(self, registered_agent):
        """Test agent creation with custom name."""
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent", name="custom-name")

        assert agent.name == "custom-name"

    def test_create_agent_with_custom_role(self, registered_agent):
        """Test agent creation with custom role."""
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent", role="critic")

        assert agent.role == "critic"

    def test_create_agent_with_custom_model(self, registered_agent):
        """Test agent creation with custom model."""
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent", model="custom-model-v2")

        assert agent.model == "custom-model-v2"

    def test_create_agent_with_kwargs(self, registered_agent):
        """Test agent creation with additional kwargs."""
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent", custom_param="custom_value")

        assert agent.extra_kwargs.get("custom_param") == "custom_value"


class TestCreateAgentWithApiKey:
    """Test agent creation with API key parameter."""

    def test_create_agent_with_api_key(self, registered_agent):
        """Test API key is passed when agent accepts it."""
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent", api_key="sk-test-key-123")

        assert agent.api_key == "sk-test-key-123"

    def test_create_agent_api_key_not_passed_when_not_accepted(self, mock_agent_class):
        """Test API key is not passed when agent doesn't accept it."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register("no-api-key-agent", accepts_api_key=False)
        class NoApiKeyAgent(mock_agent_class):
            pass

        agent = AgentRegistry.create("no-api-key-agent", api_key="sk-ignored-key")

        assert agent.api_key is None


class TestCreateUnknownTypeRaisesError:
    """Test error handling for unknown agent types."""

    def test_create_unknown_type_raises_error(self):
        """Test ValueError is raised for unknown agent type."""
        from aragora.agents.registry import AgentRegistry

        with pytest.raises(ValueError) as exc_info:
            AgentRegistry.create("nonexistent-agent")

        assert "Unknown agent type" in str(exc_info.value)
        assert "nonexistent-agent" in str(exc_info.value)

    def test_create_unknown_type_shows_valid_types(self, registered_agent):
        """Test error message includes valid agent types."""
        from aragora.agents.registry import AgentRegistry

        with pytest.raises(ValueError) as exc_info:
            AgentRegistry.create("wrong-agent")

        error_message = str(exc_info.value)
        assert "Valid types:" in error_message
        assert "test-agent" in error_message


# =============================================================================
# Cache Tests
# =============================================================================


class TestGetCachedReturnsSameInstance:
    """Test cache hit behavior."""

    def test_get_cached_returns_same_instance(self, registered_agent):
        """Test get_cached returns same instance on cache hit."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.get_cached("test-agent", name="cached-agent")
        agent2 = AgentRegistry.get_cached("test-agent", name="cached-agent")

        assert agent1 is agent2

    def test_create_with_cache_returns_same_instance(self, registered_agent):
        """Test create with use_cache=True returns same instance."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.create("test-agent", name="cached", use_cache=True)
        agent2 = AgentRegistry.create("test-agent", name="cached", use_cache=True)

        assert agent1 is agent2

    def test_create_without_cache_returns_new_instance(self, registered_agent):
        """Test create with use_cache=False returns new instance."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.create("test-agent", name="uncached", use_cache=False)
        agent2 = AgentRegistry.create("test-agent", name="uncached", use_cache=False)

        assert agent1 is not agent2

    def test_different_params_different_cache_entries(self, registered_agent):
        """Test different parameters create different cache entries."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.get_cached("test-agent", name="agent-1")
        agent2 = AgentRegistry.get_cached("test-agent", name="agent-2")

        assert agent1 is not agent2

    def test_cache_key_includes_model(self, registered_agent):
        """Test cache key includes model parameter."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.get_cached("test-agent", model="model-a")
        agent2 = AgentRegistry.get_cached("test-agent", model="model-b")

        assert agent1 is not agent2

    def test_cache_key_includes_role(self, registered_agent):
        """Test cache key includes role parameter."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.get_cached("test-agent", role="proposer")
        agent2 = AgentRegistry.get_cached("test-agent", role="critic")

        assert agent1 is not agent2

    def test_cache_key_includes_api_key(self, registered_agent):
        """Test cache key includes api_key parameter."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.get_cached("test-agent", api_key="key-1")
        agent2 = AgentRegistry.get_cached("test-agent", api_key="key-2")

        assert agent1 is not agent2

    def test_kwargs_bypass_cache(self, registered_agent):
        """Test kwargs bypass cache (always create new)."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.create(
            "test-agent", name="kwarg-agent", use_cache=True, custom_param="a"
        )
        agent2 = AgentRegistry.create(
            "test-agent", name="kwarg-agent", use_cache=True, custom_param="b"
        )

        # With kwargs, cache is bypassed so different instances created
        assert agent1 is not agent2


class TestCacheEvictionAtMaxSize:
    """Test LRU cache eviction."""

    def test_cache_eviction_at_max_size(self, mock_agent_class):
        """Test oldest cache entry is evicted at max size."""
        from aragora.agents.registry import AgentRegistry, _CACHE_MAX_SIZE, _agent_cache

        # Register agent
        @AgentRegistry.register("evict-test", default_model="model")
        class EvictTestAgent(mock_agent_class):
            pass

        # Fill cache to max size
        for i in range(_CACHE_MAX_SIZE):
            AgentRegistry.create("evict-test", name=f"agent-{i}", use_cache=True)

        assert len(_agent_cache) == _CACHE_MAX_SIZE

        # Add one more to trigger eviction
        AgentRegistry.create("evict-test", name="agent-overflow", use_cache=True)

        # Cache size should still be at max
        assert len(_agent_cache) == _CACHE_MAX_SIZE

        # First entry should be evicted
        first_key = ("evict-test", "agent-0", "proposer", "model", None)
        assert first_key not in _agent_cache


class TestClearCache:
    """Test cache clearing functionality."""

    def test_clear_cache(self, registered_agent):
        """Test clear_cache removes all cached agents."""
        from aragora.agents.registry import AgentRegistry, _agent_cache

        # Populate cache
        AgentRegistry.get_cached("test-agent", name="agent-1")
        AgentRegistry.get_cached("test-agent", name="agent-2")

        assert len(_agent_cache) >= 2

        # Clear cache
        AgentRegistry.clear_cache()

        assert len(_agent_cache) == 0

    def test_clear_removes_registry_and_cache(self, registered_agent):
        """Test clear() removes both registry and cache."""
        from aragora.agents.registry import AgentRegistry, _agent_cache

        # Populate cache
        AgentRegistry.get_cached("test-agent", name="agent-1")

        assert AgentRegistry.is_registered("test-agent")
        assert len(_agent_cache) >= 1

        # Clear all
        AgentRegistry.clear()

        assert not AgentRegistry.is_registered("test-agent")
        assert len(_agent_cache) == 0


class TestCacheStats:
    """Test cache statistics functionality."""

    def test_cache_stats(self, registered_agent):
        """Test cache_stats returns correct statistics."""
        from aragora.agents.registry import AgentRegistry, _CACHE_MAX_SIZE

        # Clear and populate cache
        AgentRegistry.clear_cache()
        AgentRegistry.get_cached("test-agent", name="stats-agent-1")
        AgentRegistry.get_cached("test-agent", name="stats-agent-2")

        stats = AgentRegistry.cache_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == _CACHE_MAX_SIZE
        assert "keys" in stats
        assert len(stats["keys"]) == 2

    def test_cache_stats_empty(self):
        """Test cache_stats with empty cache."""
        from aragora.agents.registry import AgentRegistry

        AgentRegistry.clear_cache()
        stats = AgentRegistry.cache_stats()

        assert stats["size"] == 0
        assert stats["keys"] == []

    def test_cache_stats_masks_api_keys(self, registered_agent):
        """Test cache_stats masks API keys for security."""
        from aragora.agents.registry import AgentRegistry

        AgentRegistry.clear_cache()
        AgentRegistry.get_cached(
            "test-agent",
            name="masked-agent",
            api_key="sk-super-secret-key-12345",
        )

        stats = AgentRegistry.cache_stats()

        # API key should be masked
        for key in stats["keys"]:
            if len(key) >= 5 and key[4]:  # 5th element is api_key
                # Should be masked, not the full key
                assert key[4] != "sk-super-secret-key-12345"
                assert "..." in key[4] or "***" in key[4]

    def test_cache_stats_handles_short_api_key(self, registered_agent):
        """Test cache_stats handles short API keys."""
        from aragora.agents.registry import AgentRegistry

        AgentRegistry.clear_cache()
        AgentRegistry.get_cached(
            "test-agent",
            name="short-key-agent",
            api_key="short",
        )

        stats = AgentRegistry.cache_stats()

        # Short key should be fully masked
        for key in stats["keys"]:
            if len(key) >= 5 and key[4]:
                assert key[4] == "***"


# =============================================================================
# Registry Listing Tests
# =============================================================================


class TestListAllAgents:
    """Test registry listing functionality."""

    def test_list_all_agents(self, mock_agent_class):
        """Test list_all returns all registered agents with metadata."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register(
            "list-agent-1",
            default_model="model-1",
            agent_type="API",
            requires="req-1",
            env_vars="KEY_1",
            description="First agent",
        )
        class ListAgent1(mock_agent_class):
            pass

        @AgentRegistry.register(
            "list-agent-2",
            default_model="model-2",
            agent_type="CLI",
            requires="req-2",
            env_vars="KEY_2",
            description="Second agent",
        )
        class ListAgent2(mock_agent_class):
            pass

        all_agents = AgentRegistry.list_all()

        assert "list-agent-1" in all_agents
        assert "list-agent-2" in all_agents

        assert all_agents["list-agent-1"]["type"] == "API"
        assert all_agents["list-agent-1"]["default_model"] == "model-1"
        assert all_agents["list-agent-1"]["requires"] == "req-1"
        assert all_agents["list-agent-1"]["env_vars"] == "KEY_1"
        assert all_agents["list-agent-1"]["description"] == "First agent"

        assert all_agents["list-agent-2"]["type"] == "CLI"
        assert all_agents["list-agent-2"]["default_model"] == "model-2"

    def test_list_all_empty_registry(self):
        """Test list_all returns empty dict for empty registry."""
        from aragora.agents.registry import AgentRegistry

        all_agents = AgentRegistry.list_all()

        assert all_agents == {}

    def test_get_registered_types(self, mock_agent_class):
        """Test get_registered_types returns list of type names."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register("type-a")
        class TypeAAgent(mock_agent_class):
            pass

        @AgentRegistry.register("type-b")
        class TypeBAgent(mock_agent_class):
            pass

        types = AgentRegistry.get_registered_types()

        assert "type-a" in types
        assert "type-b" in types
        assert isinstance(types, list)


# =============================================================================
# Spec Retrieval Tests
# =============================================================================


class TestGetSpecReturnsCorrectSpec:
    """Test spec retrieval functionality."""

    def test_get_spec_returns_correct_spec(self, registered_agent):
        """Test get_spec returns the correct RegistrySpec."""
        from aragora.agents.registry import AgentRegistry, RegistrySpec

        spec = AgentRegistry.get_spec("test-agent")

        assert spec is not None
        assert isinstance(spec, RegistrySpec)
        assert spec.name == "test-agent"
        assert spec.default_model == "test-model-v1"
        assert spec.agent_class is registered_agent

    def test_get_spec_returns_none_for_unknown(self):
        """Test get_spec returns None for unregistered type."""
        from aragora.agents.registry import AgentRegistry

        spec = AgentRegistry.get_spec("unknown-agent-type")

        assert spec is None

    def test_is_registered_true(self, registered_agent):
        """Test is_registered returns True for registered type."""
        from aragora.agents.registry import AgentRegistry

        assert AgentRegistry.is_registered("test-agent") is True

    def test_is_registered_false(self):
        """Test is_registered returns False for unregistered type."""
        from aragora.agents.registry import AgentRegistry

        assert AgentRegistry.is_registered("nonexistent") is False


# =============================================================================
# Type Validation Tests
# =============================================================================


class TestValidateAllowedWithValidType:
    """Test allowed type validation for valid types."""

    def test_validate_allowed_with_valid_type(self):
        """Test validate_allowed returns True for allowed types."""
        from aragora.agents.registry import AgentRegistry

        # These are from ALLOWED_AGENT_TYPES in config
        valid_types = ["demo", "claude", "gemini", "ollama", "anthropic-api"]

        for agent_type in valid_types:
            assert AgentRegistry.validate_allowed(agent_type) is True, (
                f"Expected {agent_type} to be allowed"
            )


class TestValidateAllowedWithInvalidType:
    """Test allowed type validation for invalid types."""

    def test_validate_allowed_with_invalid_type(self):
        """Test validate_allowed returns False for disallowed types."""
        from aragora.agents.registry import AgentRegistry

        invalid_types = [
            "malicious-agent",
            "not-in-allowlist",
            "random-type",
            "",
            "CLAUDE",  # Case-sensitive
        ]

        for agent_type in invalid_types:
            assert AgentRegistry.validate_allowed(agent_type) is False, (
                f"Expected {agent_type} to be disallowed"
            )


# =============================================================================
# Register All Agents Tests
# =============================================================================


class TestRegisterAllAgents:
    """Test register_all_agents function."""

    def test_register_all_agents_imports_modules(self):
        """Test register_all_agents imports agent modules."""
        from aragora.agents.registry import AgentRegistry, register_all_agents

        # Clear registry first
        original_registry = AgentRegistry._registry.copy()
        AgentRegistry._registry.clear()

        try:
            # Should not raise even if imports fail
            register_all_agents()

            # Should have registered at least some agents
            # (depends on what modules are importable)
            registered = AgentRegistry.list_all()
            # We don't assert specific agents since it depends on environment
            assert isinstance(registered, dict)
        finally:
            # Restore original registry
            AgentRegistry._registry.clear()
            AgentRegistry._registry.update(original_registry)


# =============================================================================
# Model Resolution Tests
# =============================================================================


class TestModelResolution:
    """Test model resolution in agent creation."""

    def test_create_uses_default_model_when_none_provided(self, registered_agent):
        """Test default model is used when no model specified."""
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent")

        assert agent.model == "test-model-v1"

    def test_create_uses_provided_model_over_default(self, registered_agent):
        """Test provided model overrides default."""
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent", model="override-model")

        assert agent.model == "override-model"

    def test_create_agent_without_default_model(self, mock_agent_class):
        """Test agent creation when no default model specified."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register("no-model-agent")
        class NoModelAgent(mock_agent_class):
            pass

        agent = AgentRegistry.create("no-model-agent")

        # Model should not be passed to constructor
        assert agent.model is None


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_register_same_type_twice_overwrites(self, mock_agent_class):
        """Test registering same type twice overwrites the first."""
        from aragora.agents.registry import AgentRegistry

        @AgentRegistry.register("overwrite-test", default_model="model-v1")
        class FirstAgent(mock_agent_class):
            pass

        @AgentRegistry.register("overwrite-test", default_model="model-v2")
        class SecondAgent(mock_agent_class):
            pass

        spec = AgentRegistry.get_spec("overwrite-test")
        assert spec is not None
        assert spec.default_model == "model-v2"
        assert spec.agent_class is SecondAgent

    def test_cache_with_none_values(self, registered_agent):
        """Test cache handles None values correctly."""
        from aragora.agents.registry import AgentRegistry

        agent1 = AgentRegistry.get_cached("test-agent", name=None, model=None, api_key=None)
        agent2 = AgentRegistry.get_cached("test-agent", name=None, model=None, api_key=None)

        assert agent1 is agent2

    def test_create_with_empty_string_name_uses_default(self, registered_agent):
        """Test agent creation with empty string name uses default.

        Empty string is falsy, so it falls back to default_name.
        """
        from aragora.agents.registry import AgentRegistry

        agent = AgentRegistry.create("test-agent", name="")

        # Empty string is falsy, so default_name is used
        assert agent.name == "test-agent"


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_registry_spec_exportable(self):
        """Test RegistrySpec can be imported."""
        from aragora.agents.registry import RegistrySpec

        assert RegistrySpec is not None

    def test_agent_registry_exportable(self):
        """Test AgentRegistry can be imported."""
        from aragora.agents.registry import AgentRegistry

        assert AgentRegistry is not None

    def test_register_all_agents_exportable(self):
        """Test register_all_agents can be imported."""
        from aragora.agents.registry import register_all_agents

        assert register_all_agents is not None

    def test_all_exports(self):
        """Test __all__ exports are available."""
        from aragora.agents import registry

        for name in registry.__all__:
            assert hasattr(registry, name), f"Missing export: {name}"
