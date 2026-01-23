"""Tests for circuit breaker configuration."""

import os
import pytest
from unittest.mock import patch

from aragora.resilience_config import (
    CircuitBreakerConfig,
    PROVIDER_CONFIGS,
    get_circuit_breaker_config,
    register_agent_config,
    unregister_agent_config,
    get_registered_agent_configs,
    clear_agent_configs,
)
from aragora.resilience import CircuitBreaker, get_circuit_breaker


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 60.0
        assert config.half_open_max_calls == 3

    def test_custom_values(self):
        """Config accepts custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout_seconds=120.0,
            half_open_max_calls=5,
        )
        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.timeout_seconds == 120.0
        assert config.half_open_max_calls == 5

    def test_validation_failure_threshold(self):
        """Validates failure_threshold >= 1."""
        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            CircuitBreakerConfig(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be at least 1"):
            CircuitBreakerConfig(failure_threshold=-1)

    def test_validation_success_threshold(self):
        """Validates success_threshold >= 1."""
        with pytest.raises(ValueError, match="success_threshold must be at least 1"):
            CircuitBreakerConfig(success_threshold=0)

    def test_validation_timeout_seconds(self):
        """Validates timeout_seconds > 0."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            CircuitBreakerConfig(timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            CircuitBreakerConfig(timeout_seconds=-10)

    def test_validation_half_open_max_calls(self):
        """Validates half_open_max_calls >= 1."""
        with pytest.raises(ValueError, match="half_open_max_calls must be at least 1"):
            CircuitBreakerConfig(half_open_max_calls=0)

    def test_immutable(self):
        """Config is immutable (frozen dataclass)."""
        config = CircuitBreakerConfig()
        with pytest.raises(AttributeError):
            config.failure_threshold = 10

    def test_with_overrides_single(self):
        """with_overrides creates new config with single override."""
        base = CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60.0)
        new_config = base.with_overrides(failure_threshold=10)

        assert new_config.failure_threshold == 10
        assert new_config.timeout_seconds == 60.0  # unchanged
        assert base.failure_threshold == 5  # original unchanged

    def test_with_overrides_multiple(self):
        """with_overrides handles multiple overrides."""
        base = CircuitBreakerConfig()
        new_config = base.with_overrides(
            failure_threshold=2,
            timeout_seconds=30.0,
            success_threshold=1,
        )

        assert new_config.failure_threshold == 2
        assert new_config.timeout_seconds == 30.0
        assert new_config.success_threshold == 1
        assert new_config.half_open_max_calls == 3  # unchanged default

    def test_with_overrides_none_preserves(self):
        """with_overrides(None) preserves original values."""
        base = CircuitBreakerConfig(failure_threshold=10)
        new_config = base.with_overrides(failure_threshold=None)

        assert new_config.failure_threshold == 10


class TestProviderConfigs:
    """Tests for provider configuration registry."""

    def test_default_provider_exists(self):
        """Default provider config exists."""
        assert "default" in PROVIDER_CONFIGS

    def test_anthropic_config(self):
        """Anthropic has specific config."""
        config = PROVIDER_CONFIGS["anthropic"]
        assert config.failure_threshold == 3
        assert config.timeout_seconds == 30.0

    def test_openai_config(self):
        """OpenAI has specific config."""
        config = PROVIDER_CONFIGS["openai"]
        assert config.failure_threshold == 5
        assert config.timeout_seconds == 60.0

    def test_all_providers_have_valid_config(self):
        """All registered providers have valid configs."""
        for name, config in PROVIDER_CONFIGS.items():
            assert config.failure_threshold >= 1
            assert config.success_threshold >= 1
            assert config.timeout_seconds > 0
            assert config.half_open_max_calls >= 1


class TestGetCircuitBreakerConfig:
    """Tests for get_circuit_breaker_config function."""

    def setup_method(self):
        """Clear agent configs before each test."""
        clear_agent_configs()

    def teardown_method(self):
        """Clear agent configs and env vars after each test."""
        clear_agent_configs()
        for key in [
            "ARAGORA_CB_FAILURE_THRESHOLD",
            "ARAGORA_CB_SUCCESS_THRESHOLD",
            "ARAGORA_CB_TIMEOUT_SECONDS",
            "ARAGORA_CB_HALF_OPEN_MAX_CALLS",
        ]:
            os.environ.pop(key, None)

    def test_default_config(self):
        """Returns default config when no provider specified."""
        config = get_circuit_breaker_config()
        default = PROVIDER_CONFIGS["default"]
        assert config.failure_threshold == default.failure_threshold
        assert config.timeout_seconds == default.timeout_seconds

    def test_provider_config(self):
        """Returns provider-specific config."""
        config = get_circuit_breaker_config(provider="anthropic")
        assert config.failure_threshold == 3
        assert config.timeout_seconds == 30.0

    def test_provider_case_insensitive(self):
        """Provider lookup is case-insensitive."""
        config1 = get_circuit_breaker_config(provider="ANTHROPIC")
        config2 = get_circuit_breaker_config(provider="anthropic")
        assert config1 == config2

    def test_unknown_provider_returns_default(self):
        """Unknown provider returns default config."""
        config = get_circuit_breaker_config(provider="unknown_provider")
        default = PROVIDER_CONFIGS["default"]
        assert config == default

    def test_env_override_failure_threshold(self):
        """Environment variable overrides failure_threshold."""
        os.environ["ARAGORA_CB_FAILURE_THRESHOLD"] = "2"
        config = get_circuit_breaker_config(provider="openai")

        assert config.failure_threshold == 2  # overridden
        assert config.timeout_seconds == 60.0  # from openai config

    def test_env_override_timeout_seconds(self):
        """Environment variable overrides timeout_seconds."""
        os.environ["ARAGORA_CB_TIMEOUT_SECONDS"] = "120.5"
        config = get_circuit_breaker_config()

        assert config.timeout_seconds == 120.5

    def test_env_override_success_threshold(self):
        """Environment variable overrides success_threshold."""
        os.environ["ARAGORA_CB_SUCCESS_THRESHOLD"] = "5"
        config = get_circuit_breaker_config()

        assert config.success_threshold == 5

    def test_env_override_half_open_max_calls(self):
        """Environment variable overrides half_open_max_calls."""
        os.environ["ARAGORA_CB_HALF_OPEN_MAX_CALLS"] = "10"
        config = get_circuit_breaker_config()

        assert config.half_open_max_calls == 10

    def test_env_override_invalid_ignored(self):
        """Invalid environment variable values are ignored."""
        os.environ["ARAGORA_CB_FAILURE_THRESHOLD"] = "not_a_number"
        config = get_circuit_breaker_config()
        default = PROVIDER_CONFIGS["default"]

        assert config.failure_threshold == default.failure_threshold

    def test_env_override_float_for_int_ignored(self):
        """Float value for int env var is ignored."""
        os.environ["ARAGORA_CB_FAILURE_THRESHOLD"] = "3.5"
        config = get_circuit_breaker_config()
        default = PROVIDER_CONFIGS["default"]

        # "3.5" cannot be parsed as int, so default is used
        assert config.failure_threshold == default.failure_threshold

    def test_agent_config_takes_priority(self):
        """Agent-specific config takes priority over provider."""
        register_agent_config(
            "my-agent",
            CircuitBreakerConfig(failure_threshold=100, timeout_seconds=1.0),
        )

        config = get_circuit_breaker_config(
            provider="anthropic",
            agent_name="my-agent",
        )

        assert config.failure_threshold == 100
        assert config.timeout_seconds == 1.0

    def test_env_overrides_agent_config(self):
        """Environment variables override even agent-specific configs."""
        register_agent_config(
            "my-agent",
            CircuitBreakerConfig(failure_threshold=100),
        )
        os.environ["ARAGORA_CB_FAILURE_THRESHOLD"] = "1"

        config = get_circuit_breaker_config(agent_name="my-agent")

        assert config.failure_threshold == 1  # env override wins


class TestAgentConfigRegistry:
    """Tests for agent-specific configuration registry."""

    def setup_method(self):
        """Clear agent configs before each test."""
        clear_agent_configs()

    def teardown_method(self):
        """Clear agent configs after each test."""
        clear_agent_configs()

    def test_register_and_retrieve(self):
        """Can register and retrieve agent config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        register_agent_config("test-agent", config)

        retrieved = get_circuit_breaker_config(agent_name="test-agent")
        assert retrieved.failure_threshold == 10

    def test_unregister_returns_true_if_existed(self):
        """unregister_agent_config returns True if agent was registered."""
        register_agent_config("test-agent", CircuitBreakerConfig())
        assert unregister_agent_config("test-agent") is True

    def test_unregister_returns_false_if_not_existed(self):
        """unregister_agent_config returns False if agent wasn't registered."""
        assert unregister_agent_config("nonexistent") is False

    def test_get_registered_agent_configs(self):
        """Can get all registered agent configs."""
        config1 = CircuitBreakerConfig(failure_threshold=1)
        config2 = CircuitBreakerConfig(failure_threshold=2)

        register_agent_config("agent1", config1)
        register_agent_config("agent2", config2)

        configs = get_registered_agent_configs()
        assert len(configs) == 2
        assert configs["agent1"].failure_threshold == 1
        assert configs["agent2"].failure_threshold == 2

    def test_get_registered_returns_copy(self):
        """get_registered_agent_configs returns a copy."""
        register_agent_config("test", CircuitBreakerConfig())
        configs = get_registered_agent_configs()
        configs["new"] = CircuitBreakerConfig()

        # Original registry should be unchanged
        assert "new" not in get_registered_agent_configs()

    def test_clear_agent_configs(self):
        """clear_agent_configs removes all registered configs."""
        register_agent_config("agent1", CircuitBreakerConfig())
        register_agent_config("agent2", CircuitBreakerConfig())

        clear_agent_configs()

        assert len(get_registered_agent_configs()) == 0


class TestCircuitBreakerWithConfig:
    """Tests for CircuitBreaker with configurable thresholds."""

    def test_from_config(self):
        """CircuitBreaker.from_config creates properly configured instance."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout_seconds=120.0,
            half_open_max_calls=4,
        )
        cb = CircuitBreaker.from_config(config, name="test")

        assert cb.name == "test"
        assert cb.failure_threshold == 10
        assert cb.half_open_success_threshold == 5
        assert cb.cooldown_seconds == 120.0
        assert cb.half_open_max_calls == 4

    def test_from_config_preserves_reference(self):
        """from_config stores reference to original config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker.from_config(config)

        assert cb.config is config

    def test_to_dict_includes_config(self):
        """to_dict includes configuration values."""
        cb = CircuitBreaker(
            failure_threshold=10,
            cooldown_seconds=120.0,
            half_open_success_threshold=5,
            half_open_max_calls=4,
        )
        data = cb.to_dict()

        assert "config" in data
        assert data["config"]["failure_threshold"] == 10
        assert data["config"]["cooldown_seconds"] == 120.0
        assert data["config"]["half_open_success_threshold"] == 5
        assert data["config"]["half_open_max_calls"] == 4


class TestGetCircuitBreakerWithConfig:
    """Tests for get_circuit_breaker function with config support."""

    def setup_method(self):
        """Clear global state before each test."""
        from aragora.resilience import reset_all_circuit_breakers, _circuit_breakers

        reset_all_circuit_breakers()
        _circuit_breakers.clear()
        clear_agent_configs()

    def teardown_method(self):
        """Clear global state after each test."""
        from aragora.resilience import _circuit_breakers

        _circuit_breakers.clear()
        clear_agent_configs()
        for key in [
            "ARAGORA_CB_FAILURE_THRESHOLD",
            "ARAGORA_CB_TIMEOUT_SECONDS",
        ]:
            os.environ.pop(key, None)

    def test_with_provider(self):
        """get_circuit_breaker with provider uses provider config."""
        cb = get_circuit_breaker("test-anthropic", provider="anthropic")

        assert cb.failure_threshold == 3  # anthropic default
        assert cb.cooldown_seconds == 30.0  # anthropic default

    def test_with_explicit_config(self):
        """get_circuit_breaker with explicit config."""
        config = CircuitBreakerConfig(failure_threshold=99, timeout_seconds=999.0)
        cb = get_circuit_breaker("test-custom", config=config)

        assert cb.failure_threshold == 99
        assert cb.cooldown_seconds == 999.0

    def test_legacy_parameters(self):
        """get_circuit_breaker with legacy parameters."""
        cb = get_circuit_breaker(
            "test-legacy",
            failure_threshold=7,
            cooldown_seconds=70.0,
        )

        assert cb.failure_threshold == 7
        assert cb.cooldown_seconds == 70.0

    def test_env_override_via_get_circuit_breaker(self):
        """Environment overrides work through get_circuit_breaker."""
        os.environ["ARAGORA_CB_FAILURE_THRESHOLD"] = "1"

        cb = get_circuit_breaker("test-env", provider="openai")

        # OpenAI default is 5, but env override should give us 1
        assert cb.failure_threshold == 1

    def test_config_priority_over_provider(self):
        """Explicit config takes priority over provider."""
        config = CircuitBreakerConfig(failure_threshold=50)
        cb = get_circuit_breaker(
            "test-priority",
            provider="anthropic",  # would give 3
            config=config,  # should give 50
        )

        assert cb.failure_threshold == 50

    def test_shared_circuit_breaker(self):
        """Same name returns same circuit breaker."""
        cb1 = get_circuit_breaker("shared", provider="anthropic")
        cb2 = get_circuit_breaker("shared", provider="openai")  # different provider

        # Should return same instance (first one created)
        assert cb1 is cb2
        assert cb1.failure_threshold == 3  # anthropic config from first call


class TestCircuitBreakerBehaviorWithConfig:
    """Tests that circuit breaker behavior respects config values."""

    def test_opens_at_configured_threshold(self):
        """Circuit opens at configured failure_threshold."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker.from_config(config)

        cb.record_failure()
        assert cb.get_status() == "closed"

        cb.record_failure()
        assert cb.get_status() == "open"

    def test_cooldown_uses_configured_timeout(self):
        """Cooldown respects configured timeout_seconds."""
        import time

        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        cb = CircuitBreaker.from_config(config)

        cb.record_failure()
        assert cb.get_status() == "open"
        assert cb.can_proceed() is False

        # Wait for timeout
        time.sleep(0.15)

        assert cb.can_proceed() is True

    def test_entity_mode_with_configured_threshold(self):
        """Entity mode respects configured threshold."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker.from_config(config)

        cb.record_failure("agent-a")
        assert cb.get_status("agent-a") == "closed"

        cb.record_failure("agent-a")
        assert cb.get_status("agent-a") == "open"

        # Other entities unaffected
        assert cb.get_status("agent-b") == "closed"
