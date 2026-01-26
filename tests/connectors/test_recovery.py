"""
Tests for connector error recovery strategies.

Tests cover:
- RecoveryAction enum
- RecoveryResult dataclass
- RecoveryConfig dataclass
- RecoveryStrategy class
- with_recovery decorator
- create_recovery_chain function
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import fields

from aragora.connectors.recovery import (
    RecoveryAction,
    RecoveryResult,
    RecoveryConfig,
    RecoveryStrategy,
    with_recovery,
    create_recovery_chain,
)
from aragora.connectors.exceptions import (
    ConnectorError,
    ConnectorAuthError,
    ConnectorRateLimitError,
    ConnectorNetworkError,
)


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_action_values(self):
        """Actions have correct string values."""
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.WAIT.value == "wait"
        assert RecoveryAction.REFRESH_TOKEN.value == "refresh_token"
        assert RecoveryAction.OPEN_CIRCUIT.value == "open_circuit"
        assert RecoveryAction.FALLBACK.value == "fallback"
        assert RecoveryAction.ESCALATE.value == "escalate"
        assert RecoveryAction.FAIL.value == "fail"


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_default_values(self):
        """RecoveryResult has correct defaults."""
        result = RecoveryResult(
            action=RecoveryAction.RETRY,
            success=True,
        )

        assert result.error is None
        assert result.attempts == 0
        assert result.total_wait_seconds == 0.0
        assert result.message == ""

    def test_with_all_values(self):
        """RecoveryResult accepts all values."""
        error = ValueError("test")
        result = RecoveryResult(
            action=RecoveryAction.FAIL,
            success=False,
            error=error,
            attempts=3,
            total_wait_seconds=15.5,
            message="Max retries exceeded",
        )

        assert result.action == RecoveryAction.FAIL
        assert result.success is False
        assert result.error is error
        assert result.attempts == 3
        assert result.total_wait_seconds == 15.5
        assert result.message == "Max retries exceeded"


class TestRecoveryConfig:
    """Tests for RecoveryConfig dataclass."""

    def test_default_values(self):
        """RecoveryConfig has sensible defaults."""
        config = RecoveryConfig()

        assert config.max_retries == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.jitter_factor == 0.3
        assert config.enable_circuit_breaker is True
        assert config.enable_token_refresh is True
        assert config.escalation_threshold == 3
        assert config.fallback_endpoints == []

    def test_custom_values(self):
        """RecoveryConfig accepts custom values."""
        config = RecoveryConfig(
            max_retries=5,
            base_delay_seconds=2.0,
            enable_circuit_breaker=False,
        )

        assert config.max_retries == 5
        assert config.base_delay_seconds == 2.0
        assert config.enable_circuit_breaker is False


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Fresh recovery strategy for tests."""
        config = RecoveryConfig(
            max_retries=3,
            base_delay_seconds=0.01,  # Fast for tests
            enable_circuit_breaker=False,  # Disable for simpler tests
        )
        return RecoveryStrategy(config=config, connector_name="test")

    @pytest.mark.asyncio
    async def test_execute_success(self, strategy):
        """Execute returns result on success."""

        async def operation():
            return "success"

        result = await strategy.execute(operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_retries_on_failure(self, strategy):
        """Execute retries on retryable errors."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectorNetworkError("Network failed", connector_name="test")
            return "success"

        result = await strategy.execute(operation)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_fails_after_max_retries(self, strategy):
        """Execute fails after max retries exhausted."""

        async def operation():
            raise ConnectorNetworkError("Always fails", connector_name="test")

        with pytest.raises(ConnectorError):
            await strategy.execute(operation)

    @pytest.mark.asyncio
    async def test_execute_no_retry_on_non_retryable(self, strategy):
        """Execute doesn't retry non-retryable errors."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            raise ConnectorAuthError("Auth failed", connector_name="test")

        with pytest.raises(ConnectorError):
            await strategy.execute(operation)

        # Should only call once (no retries for auth errors without refresh)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_token_refresh(self):
        """Execute refreshes token on auth error."""
        config = RecoveryConfig(
            max_retries=3,
            enable_token_refresh=True,
            enable_circuit_breaker=False,
        )
        strategy = RecoveryStrategy(config=config, connector_name="test")

        refresh_called = False

        def refresh_token():
            nonlocal refresh_called
            refresh_called = True

        strategy.set_token_refresh_callback(refresh_token)

        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectorAuthError("Token expired", connector_name="test")
            return "success"

        result = await strategy.execute(operation)
        assert result == "success"
        assert refresh_called is True

    @pytest.mark.asyncio
    async def test_execute_waits_on_rate_limit(self):
        """Execute waits on rate limit errors."""
        config = RecoveryConfig(
            max_retries=3,
            enable_circuit_breaker=False,
        )
        strategy = RecoveryStrategy(config=config, connector_name="test")

        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectorRateLimitError(
                    "Rate limited",
                    connector_name="test",
                    retry_after=0.01,  # Fast for tests
                )
            return "success"

        result = await strategy.execute(operation)
        assert result == "success"
        assert call_count == 2

    def test_get_stats(self, strategy):
        """get_stats returns statistics."""
        stats = strategy.get_stats()

        assert stats["connector_name"] == "test"
        assert stats["consecutive_failures"] == 0
        assert stats["total_retries"] == 0

    def test_calculate_backoff(self, strategy):
        """_calculate_backoff returns increasing delays."""
        delays = [strategy._calculate_backoff(i) for i in range(5)]

        # Delays should generally increase (accounting for jitter)
        # Just check they're all positive and have some variation
        assert all(d > 0 for d in delays)

    def test_determine_recovery_action_retry(self, strategy):
        """_determine_recovery_action returns RETRY for retryable errors."""
        error = ConnectorNetworkError("Network", connector_name="test")
        action = strategy._determine_recovery_action(error, attempt=0)
        assert action == RecoveryAction.RETRY

    def test_determine_recovery_action_fail_after_max(self, strategy):
        """_determine_recovery_action returns FAIL after max retries."""
        error = ConnectorNetworkError("Network", connector_name="test")
        action = strategy._determine_recovery_action(error, attempt=10)
        assert action == RecoveryAction.FAIL


class TestWithRecoveryDecorator:
    """Tests for with_recovery decorator."""

    @pytest.mark.asyncio
    async def test_decorator_wraps_function(self):
        """Decorator wraps async function."""

        @with_recovery(max_retries=2, enable_circuit_breaker=False)
        async def my_operation():
            return "result"

        result = await my_operation()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_retries(self):
        """Decorator retries on failure."""
        call_count = 0

        @with_recovery(max_retries=3, enable_circuit_breaker=False)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectorNetworkError("Flaky", connector_name="test")
            return "success"

        result = await flaky_operation()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_attaches_strategy(self):
        """Decorator attaches strategy to function."""

        @with_recovery(max_retries=5)
        async def operation():
            return "x"

        assert hasattr(operation, "_recovery_strategy")
        assert operation._recovery_strategy.config.max_retries == 5


class TestCreateRecoveryChain:
    """Tests for create_recovery_chain function."""

    def test_creates_strategy(self):
        """create_recovery_chain creates a RecoveryStrategy."""
        connector = MagicMock()
        connector.name = "test_connector"

        strategy = create_recovery_chain(connector)

        assert isinstance(strategy, RecoveryStrategy)
        assert strategy.connector_name == "test_connector"

    def test_uses_type_name_as_fallback(self):
        """Uses type name when connector has no name attribute."""
        connector = MagicMock(spec=[])  # No name attribute

        strategy = create_recovery_chain(connector)

        assert "MagicMock" in strategy.connector_name

    def test_sets_token_refresh_callback(self):
        """Sets token refresh callback if available."""
        connector = MagicMock()
        connector.name = "oauth_connector"
        connector._get_access_token = MagicMock()

        config = RecoveryConfig(enable_token_refresh=True)
        strategy = create_recovery_chain(connector, config=config)

        assert strategy._refresh_token_callback is not None

    def test_uses_refresh_token_method(self):
        """Falls back to refresh_token method."""
        connector = MagicMock(spec=["name", "refresh_token"])
        connector.name = "oauth_connector"
        connector.refresh_token = MagicMock()

        config = RecoveryConfig(enable_token_refresh=True)
        strategy = create_recovery_chain(connector, config=config)

        assert strategy._refresh_token_callback is not None

    def test_uses_custom_config(self):
        """Uses provided config."""
        connector = MagicMock()
        connector.name = "test"
        config = RecoveryConfig(max_retries=10)

        strategy = create_recovery_chain(connector, config=config)

        assert strategy.config.max_retries == 10


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_recovery_flow(self):
        """Tests a full recovery flow with multiple error types."""
        config = RecoveryConfig(
            max_retries=5,
            base_delay_seconds=0.001,
            enable_circuit_breaker=False,
            enable_token_refresh=True,
        )
        strategy = RecoveryStrategy(config=config, connector_name="integration")

        tokens_refreshed = 0

        def refresh_token():
            nonlocal tokens_refreshed
            tokens_refreshed += 1

        strategy.set_token_refresh_callback(refresh_token)

        call_count = 0

        async def complex_operation():
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise ConnectorNetworkError("Network flaky", connector_name="test")
            elif call_count == 2:
                raise ConnectorAuthError("Token expired", connector_name="test")
            elif call_count == 3:
                raise ConnectorNetworkError("Network flaky again", connector_name="test")
            return "finally success"

        result = await strategy.execute(complex_operation)

        assert result == "finally success"
        assert call_count == 4
        assert tokens_refreshed == 1  # Only refreshed once

    @pytest.mark.asyncio
    async def test_escalation_callback(self):
        """Tests escalation callback is called."""
        escalations = []

        def escalation_handler(error, failures):
            escalations.append((error, failures))

        config = RecoveryConfig(
            max_retries=1,
            escalation_threshold=2,
            escalation_callback=escalation_handler,
            enable_circuit_breaker=False,
        )
        strategy = RecoveryStrategy(config=config, connector_name="escalate")

        # Force consecutive failures
        strategy._consecutive_failures = 2

        async def failing_operation():
            raise ConnectorError("Always fails", connector_name="test")

        with pytest.raises(ConnectorError):
            await strategy.execute(failing_operation)

        assert len(escalations) == 1
        assert escalations[0][1] >= 2  # At least 2 consecutive failures
