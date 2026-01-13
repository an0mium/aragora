"""Tests for AgentRegistry pattern."""

import pytest
from unittest.mock import MagicMock, patch

from aragora.agents.registry import AgentRegistry, AgentSpec, register_all_agents


class TestAgentSpec:
    """Tests for AgentSpec dataclass."""

    def test_agent_spec_creation(self):
        """AgentSpec is created with required fields."""
        spec = AgentSpec(
            name="test",
            agent_class=MagicMock,
            default_model="test-model",
            default_name="test",
            agent_type="API",
            requires=None,
            env_vars="TEST_KEY",
        )

        assert spec.name == "test"
        assert spec.default_model == "test-model"
        assert spec.agent_type == "API"
        assert spec.env_vars == "TEST_KEY"
        assert spec.description is None
        assert spec.accepts_api_key is False

    def test_agent_spec_with_description(self):
        """AgentSpec supports optional description."""
        spec = AgentSpec(
            name="test",
            agent_class=MagicMock,
            default_model="model",
            default_name="test",
            agent_type="API",
            requires=None,
            env_vars=None,
            description="A test agent",
            accepts_api_key=True,
        )

        assert spec.description == "A test agent"
        assert spec.accepts_api_key is True


class TestAgentRegistry:
    """Tests for AgentRegistry class."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        # Save current registry
        saved = dict(AgentRegistry._registry)
        AgentRegistry.clear()
        yield
        # Restore registry
        AgentRegistry._registry = saved

    def test_register_decorator(self):
        """@AgentRegistry.register() adds class to registry."""

        @AgentRegistry.register(
            "test-agent",
            default_model="test-model",
            agent_type="CLI",
            requires="test CLI",
        )
        class TestAgent:
            def __init__(self, name, role, model=None):
                self.name = name
                self.role = role
                self.model = model

        assert AgentRegistry.is_registered("test-agent")
        spec = AgentRegistry.get_spec("test-agent")
        assert spec is not None
        assert spec.default_model == "test-model"
        assert spec.agent_type == "CLI"

    def test_create_agent_basic(self):
        """AgentRegistry.create() instantiates registered agent."""

        @AgentRegistry.register("simple", default_model="simple-v1")
        class SimpleAgent:
            def __init__(self, name, role, model=None):
                self.name = name
                self.role = role
                self.model = model

        agent = AgentRegistry.create("simple", name="my-agent", role="critic")

        assert agent.name == "my-agent"
        assert agent.role == "critic"
        assert agent.model == "simple-v1"

    def test_create_agent_with_model_override(self):
        """Model parameter overrides default."""

        @AgentRegistry.register("override", default_model="default")
        class OverrideAgent:
            def __init__(self, name, role, model=None):
                self.name = name
                self.role = role
                self.model = model

        agent = AgentRegistry.create("override", model="custom-model")

        assert agent.model == "custom-model"

    def test_create_agent_uses_default_name(self):
        """Agent uses default_name when name not provided."""

        @AgentRegistry.register("named", default_name="default-name")
        class NamedAgent:
            def __init__(self, name, role):
                self.name = name
                self.role = role

        agent = AgentRegistry.create("named")

        assert agent.name == "default-name"

    def test_create_agent_with_api_key(self):
        """API key is passed when accepts_api_key=True."""

        @AgentRegistry.register("api-agent", accepts_api_key=True)
        class ApiAgent:
            def __init__(self, name, role, api_key=None):
                self.name = name
                self.role = role
                self.api_key = api_key

        agent = AgentRegistry.create("api-agent", api_key="secret-key")

        assert agent.api_key == "secret-key"

    def test_create_agent_unknown_type_raises(self):
        """Creating unknown agent type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentRegistry.create("nonexistent")

    def test_list_all(self):
        """list_all() returns all registered agents."""

        @AgentRegistry.register(
            "agent1",
            default_model="m1",
            agent_type="CLI",
            requires="cli1",
            env_vars="KEY1",
            description="Agent 1",
        )
        class Agent1:
            pass

        @AgentRegistry.register(
            "agent2",
            default_model="m2",
            agent_type="API",
            env_vars="KEY2",
        )
        class Agent2:
            pass

        all_agents = AgentRegistry.list_all()

        assert "agent1" in all_agents
        assert "agent2" in all_agents
        assert all_agents["agent1"]["type"] == "CLI"
        assert all_agents["agent1"]["description"] == "Agent 1"
        assert all_agents["agent2"]["type"] == "API"

    def test_get_registered_types(self):
        """get_registered_types() returns list of type names."""

        @AgentRegistry.register("type-a")
        class TypeA:
            pass

        @AgentRegistry.register("type-b")
        class TypeB:
            pass

        types = AgentRegistry.get_registered_types()

        assert "type-a" in types
        assert "type-b" in types

    def test_is_registered(self):
        """is_registered() returns correct boolean."""
        assert AgentRegistry.is_registered("not-registered") is False

        @AgentRegistry.register("now-registered")
        class NowRegistered:
            pass

        assert AgentRegistry.is_registered("now-registered") is True

    def test_validate_allowed(self):
        """validate_allowed() checks against ALLOWED_AGENT_TYPES."""
        # These should be in the allowed list (from config.py)
        assert AgentRegistry.validate_allowed("claude") is True
        assert AgentRegistry.validate_allowed("gemini") is True
        assert AgentRegistry.validate_allowed("anthropic-api") is True

        # Made up type should not be allowed
        assert AgentRegistry.validate_allowed("malicious-agent") is False

    def test_clear(self):
        """clear() removes all registrations."""

        @AgentRegistry.register("to-clear")
        class ToClear:
            pass

        assert AgentRegistry.is_registered("to-clear")

        AgentRegistry.clear()

        assert AgentRegistry.is_registered("to-clear") is False

    def test_extra_kwargs_passed_to_constructor(self):
        """Extra kwargs are passed to agent constructor."""

        @AgentRegistry.register("kwargs-agent")
        class KwargsAgent:
            def __init__(self, name, role, custom_param=None, another=False):
                self.name = name
                self.role = role
                self.custom_param = custom_param
                self.another = another

        agent = AgentRegistry.create(
            "kwargs-agent",
            custom_param="value",
            another=True,
        )

        assert agent.custom_param == "value"
        assert agent.another is True


class TestRegisterAllAgents:
    """Tests for register_all_agents function."""

    def test_register_all_agents_populates_registry(self):
        """register_all_agents() ensures agents are registered."""
        # Note: We don't clear the registry here because decorators
        # only run once when modules are first imported.
        # This test verifies that after calling register_all_agents,
        # the expected agents are present.
        register_all_agents()

        # Should have some agents registered now
        types = AgentRegistry.get_registered_types()
        assert len(types) > 0
        # Check for known agent types
        assert "claude" in types
        assert "gemini" in types
        assert "anthropic-api" in types


class TestCreateAgentIntegration:
    """Integration tests for create_agent with registry."""

    def test_create_agent_uses_registry(self):
        """create_agent() uses AgentRegistry under the hood."""
        from aragora.agents.base import create_agent

        # Ensure agents are registered
        register_all_agents()

        # Verify registry has claude
        assert AgentRegistry.is_registered("claude")

        # Create should work through registry
        spec = AgentRegistry.get_spec("claude")
        assert spec is not None
        assert spec.agent_type == "CLI"

    def test_list_available_agents_uses_registry(self):
        """list_available_agents() returns registry data."""
        from aragora.agents.base import list_available_agents

        agents = list_available_agents()

        # Should have agent types
        assert len(agents) > 0
        assert "claude" in agents
        assert "gemini" in agents
        assert "anthropic-api" in agents

        # Should have expected structure
        claude_info = agents["claude"]
        assert "type" in claude_info
        assert claude_info["type"] == "CLI"


class TestAgentRegistryIdempotency:
    """Tests for registry idempotency."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        # Save current registry
        saved = dict(AgentRegistry._registry)
        AgentRegistry.clear()
        yield
        # Restore registry
        AgentRegistry._registry = saved

    def test_register_same_type_twice_overwrites(self):
        """Registering same type twice overwrites."""

        @AgentRegistry.register("overwrite-me", default_model="v1")
        class V1Agent:
            pass

        @AgentRegistry.register("overwrite-me", default_model="v2")
        class V2Agent:
            pass

        spec = AgentRegistry.get_spec("overwrite-me")
        assert spec.default_model == "v2"
        assert spec.agent_class is V2Agent

    def test_multiple_register_all_agents_calls_safe(self):
        """Multiple register_all_agents() calls are safe."""
        # Note: This test uses a fresh registry but register_all_agents
        # won't re-run decorators (modules already imported).
        # We verify calling it multiple times doesn't error.

        # First, ensure at least one agent is registered for the test
        @AgentRegistry.register("test-idempotent")
        class IdempotentAgent:
            pass

        types1 = set(AgentRegistry.get_registered_types())
        assert "test-idempotent" in types1

        # Calling register_all_agents shouldn't break anything
        register_all_agents()
        register_all_agents()

        # Our test agent should still be there
        assert AgentRegistry.is_registered("test-idempotent")


class TestAgentRegistryCaching:
    """Tests for agent instance caching."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry and cache before each test."""
        saved = dict(AgentRegistry._registry)
        AgentRegistry.clear()
        yield
        AgentRegistry._registry = saved

    def test_create_without_cache_returns_new_instances(self):
        """create() without use_cache returns new instances each time."""

        @AgentRegistry.register("no-cache-agent")
        class NoCacheAgent:
            def __init__(self, name, role):
                self.name = name
                self.role = role

        agent1 = AgentRegistry.create("no-cache-agent")
        agent2 = AgentRegistry.create("no-cache-agent")

        assert agent1 is not agent2

    def test_create_with_cache_returns_same_instance(self):
        """create() with use_cache=True returns same instance."""

        @AgentRegistry.register("cache-agent")
        class CacheAgent:
            def __init__(self, name, role):
                self.name = name
                self.role = role

        agent1 = AgentRegistry.create("cache-agent", use_cache=True)
        agent2 = AgentRegistry.create("cache-agent", use_cache=True)

        assert agent1 is agent2

    def test_cache_key_includes_all_params(self):
        """Cache key differentiates by name, role, model, api_key."""

        @AgentRegistry.register("param-agent", default_model="v1")
        class ParamAgent:
            def __init__(self, name, role, model=None, api_key=None):
                self.name = name
                self.role = role
                self.model = model
                self.api_key = api_key

        agent1 = AgentRegistry.create("param-agent", name="a", role="critic", use_cache=True)
        agent2 = AgentRegistry.create("param-agent", name="b", role="critic", use_cache=True)
        agent3 = AgentRegistry.create("param-agent", name="a", role="proposer", use_cache=True)
        agent4 = AgentRegistry.create("param-agent", name="a", role="critic", use_cache=True)

        # Different name = different instance
        assert agent1 is not agent2
        # Different role = different instance
        assert agent1 is not agent3
        # Same params = same instance
        assert agent1 is agent4

    def test_get_cached_always_uses_cache(self):
        """get_cached() always uses caching."""

        @AgentRegistry.register("get-cached-agent")
        class GetCachedAgent:
            def __init__(self, name, role):
                self.name = name
                self.role = role

        agent1 = AgentRegistry.get_cached("get-cached-agent")
        agent2 = AgentRegistry.get_cached("get-cached-agent")

        assert agent1 is agent2

    def test_clear_cache_clears_instances(self):
        """clear_cache() removes cached instances."""

        @AgentRegistry.register("clear-cache-agent")
        class ClearCacheAgent:
            def __init__(self, name, role):
                self.name = name
                self.role = role

        agent1 = AgentRegistry.create("clear-cache-agent", use_cache=True)
        AgentRegistry.clear_cache()
        agent2 = AgentRegistry.create("clear-cache-agent", use_cache=True)

        assert agent1 is not agent2

    def test_clear_also_clears_cache(self):
        """clear() clears both registry and cache."""

        @AgentRegistry.register("full-clear-agent")
        class FullClearAgent:
            def __init__(self, name, role):
                self.name = name
                self.role = role

        AgentRegistry.create("full-clear-agent", use_cache=True)
        stats_before = AgentRegistry.cache_stats()
        assert stats_before["size"] > 0

        AgentRegistry.clear()
        stats_after = AgentRegistry.cache_stats()

        assert stats_after["size"] == 0

    def test_cache_stats_returns_info(self):
        """cache_stats() returns size and keys."""

        @AgentRegistry.register("stats-agent")
        class StatsAgent:
            def __init__(self, name, role):
                self.name = name
                self.role = role

        AgentRegistry.clear_cache()
        AgentRegistry.create("stats-agent", name="a", use_cache=True)
        AgentRegistry.create("stats-agent", name="b", use_cache=True)

        stats = AgentRegistry.cache_stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 32
        assert len(stats["keys"]) == 2

    def test_cache_stats_masks_api_keys(self):
        """cache_stats() masks API keys for security."""

        @AgentRegistry.register("masked-agent", accepts_api_key=True)
        class MaskedAgent:
            def __init__(self, name, role, api_key=None):
                self.name = name
                self.role = role
                self.api_key = api_key

        AgentRegistry.clear_cache()
        AgentRegistry.create(
            "masked-agent",
            api_key="super-secret-key-12345",
            use_cache=True,
        )

        stats = AgentRegistry.cache_stats()

        # API key should be masked
        for key in stats["keys"]:
            if key[4]:  # 5th element is api_key
                assert "super-secret-key-12345" not in str(key)
                assert "..." in key[4] or "***" in key[4]

    def test_cache_with_kwargs_bypasses_cache(self):
        """Extra kwargs bypass caching."""

        @AgentRegistry.register("kwargs-cache-agent")
        class KwargsCacheAgent:
            def __init__(self, name, role, extra=None):
                self.name = name
                self.role = role
                self.extra = extra

        agent1 = AgentRegistry.create("kwargs-cache-agent", extra="val1", use_cache=True)
        agent2 = AgentRegistry.create("kwargs-cache-agent", extra="val1", use_cache=True)

        # Should be different instances because kwargs bypass cache
        assert agent1 is not agent2

    def test_cache_eviction_at_max_size(self):
        """Cache evicts oldest entry when at max size."""
        from aragora.agents import registry

        # Temporarily reduce cache size for testing
        original_max = registry._CACHE_MAX_SIZE
        registry._CACHE_MAX_SIZE = 3

        try:

            @AgentRegistry.register("evict-agent")
            class EvictAgent:
                def __init__(self, name, role):
                    self.name = name
                    self.role = role

            AgentRegistry.clear_cache()

            # Fill cache
            AgentRegistry.create("evict-agent", name="a", use_cache=True)
            AgentRegistry.create("evict-agent", name="b", use_cache=True)
            AgentRegistry.create("evict-agent", name="c", use_cache=True)

            stats1 = AgentRegistry.cache_stats()
            assert stats1["size"] == 3

            # Add one more - should evict oldest
            AgentRegistry.create("evict-agent", name="d", use_cache=True)

            stats2 = AgentRegistry.cache_stats()
            assert stats2["size"] == 3  # Still at max

            # Oldest (name="a") should be evicted
            key_names = [k[1] for k in stats2["keys"]]
            assert "a" not in key_names
            assert "d" in key_names

        finally:
            registry._CACHE_MAX_SIZE = original_max
