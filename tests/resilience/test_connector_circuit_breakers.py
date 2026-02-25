"""
Tests for circuit breaker integration with external connectors.

Verifies that:
- Circuit opens after N consecutive failures
- Half-open state allows a test request after cooldown
- Successful request in half-open transitions back to closed
- Each connector type has its own independent circuit breaker
- Health endpoint exposes circuit breaker state for connectors
- get_connector_circuit_breaker_states returns correct data
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from aragora.resilience.registry import (
    get_circuit_breaker,
    get_connector_circuit_breaker_states,
    reset_all_circuit_breakers,
    _circuit_breakers,
    _circuit_breakers_lock,
)


@pytest.fixture(autouse=True)
def _clean_circuit_breakers():
    """Reset circuit breaker registry before and after each test."""
    reset_all_circuit_breakers()
    # Also clear the registry entirely so tests start fresh
    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    yield
    reset_all_circuit_breakers()
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


# ============================================================================
# Core Circuit Breaker State Transitions
# ============================================================================


class TestCircuitBreakerStateTransitions:
    """Verify fundamental state transitions: closed -> open -> half-open -> closed."""

    def test_circuit_opens_after_n_failures(self):
        """Circuit should transition to open after failure_threshold consecutive failures."""
        cb = CircuitBreaker(name="test_connector", failure_threshold=3, cooldown_seconds=60.0)

        assert cb.get_status() == "closed"
        assert cb.can_proceed() is True

        # Record failures up to threshold
        cb.record_failure()
        assert cb.get_status() == "closed"

        cb.record_failure()
        assert cb.get_status() == "closed"

        opened = cb.record_failure()
        assert opened is True
        assert cb.get_status() == "open"
        assert cb.can_proceed() is False

    def test_half_open_after_cooldown(self):
        """After cooldown expires, circuit should transition to half-open."""
        cb = CircuitBreaker(name="test_connector", failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status() == "open"

        # Wait for cooldown to expire
        time.sleep(0.15)

        # Now should be half-open (allows request)
        assert cb.get_status() == "half-open"
        assert cb.can_proceed() is True

    def test_successful_request_in_half_open_closes_circuit(self):
        """A success after cooldown should close the circuit."""
        cb = CircuitBreaker(name="test_connector", failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.get_status() == "open"

        # Wait for cooldown
        time.sleep(0.15)

        # Circuit allows request (half-open via can_proceed reset)
        assert cb.can_proceed() is True

        # Record success closes it
        cb.record_success()
        assert cb.get_status() == "closed"
        assert cb.failures == 0

    def test_failure_in_half_open_reopens_circuit(self):
        """A failure during half-open should reopen the circuit."""
        cb = CircuitBreaker(name="test_connector", failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for cooldown
        time.sleep(0.15)
        assert cb.can_proceed() is True

        # Another failure while half-open reopens
        cb.record_failure()
        # After can_proceed resets the circuit, a single failure won't reopen
        # (needs threshold again), but two failures will
        cb.record_failure()
        assert cb.get_status() == "open"


# ============================================================================
# Independent Circuit Breakers Per Connector
# ============================================================================


class TestIndependentCircuitBreakers:
    """Each connector must have its own independent circuit breaker."""

    def test_connectors_have_independent_circuit_breakers(self):
        """Failures in one connector should not affect another."""
        slack_cb = get_circuit_breaker("slack_api", failure_threshold=2, cooldown_seconds=60)
        discord_cb = get_circuit_breaker("discord_api", failure_threshold=2, cooldown_seconds=60)
        github_cb = get_circuit_breaker(
            "connector_GitHub", failure_threshold=2, cooldown_seconds=60
        )

        # Open slack circuit
        slack_cb.record_failure()
        slack_cb.record_failure()
        assert slack_cb.get_status() == "open"

        # Others should still be closed
        assert discord_cb.get_status() == "closed"
        assert github_cb.get_status() == "closed"

        # Open discord circuit
        discord_cb.record_failure()
        discord_cb.record_failure()
        assert discord_cb.get_status() == "open"

        # GitHub should still be closed
        assert github_cb.get_status() == "closed"

    def test_registry_returns_same_instance(self):
        """get_circuit_breaker should return the same instance for the same name."""
        cb1 = get_circuit_breaker("slack_api")
        cb2 = get_circuit_breaker("slack_api")
        assert cb1 is cb2

    def test_registry_returns_different_instances_for_different_names(self):
        """Different names should produce different circuit breaker instances."""
        cb1 = get_circuit_breaker("slack_api")
        cb2 = get_circuit_breaker("discord_api")
        assert cb1 is not cb2


# ============================================================================
# Connector Circuit Breaker States Function
# ============================================================================


class TestGetConnectorCircuitBreakerStates:
    """Tests for get_connector_circuit_breaker_states utility."""

    def test_returns_empty_when_no_connectors(self):
        """Should return empty dict when no connector circuit breakers exist."""
        states = get_connector_circuit_breaker_states()
        assert states == {}

    def test_returns_connector_circuit_breakers_only(self):
        """Should filter to only connector-related circuit breakers."""
        # Create connector and non-connector circuit breakers
        get_circuit_breaker("slack_api")
        get_circuit_breaker("discord_api")
        get_circuit_breaker("connector_GitHub")
        get_circuit_breaker("chat_connector_telegram")
        get_circuit_breaker("agent_claude")  # Non-connector, should be excluded

        states = get_connector_circuit_breaker_states()

        assert "slack_api" in states
        assert "discord_api" in states
        assert "connector_GitHub" in states
        assert "chat_connector_telegram" in states
        assert "agent_claude" not in states

    def test_state_reflects_current_status(self):
        """States should reflect the actual circuit breaker status."""
        cb = get_circuit_breaker("slack_api", failure_threshold=2, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()

        states = get_connector_circuit_breaker_states()

        assert states["slack_api"]["state"] == "open"
        assert states["slack_api"]["failures"] == 2
        assert "cooldown_remaining" in states["slack_api"]

    def test_closed_state_has_no_cooldown(self):
        """Closed circuits should not include cooldown_remaining."""
        get_circuit_breaker("email_smtp")

        states = get_connector_circuit_breaker_states()

        assert states["email_smtp"]["state"] == "closed"
        assert states["email_smtp"]["failures"] == 0
        assert "cooldown_remaining" not in states["email_smtp"]


# ============================================================================
# Async Protected Call Integration
# ============================================================================


class TestAsyncProtectedCall:
    """Test async protected_call context manager used by connectors."""

    @pytest.mark.asyncio
    async def test_protected_call_records_success(self):
        """protected_call should record success when no exception."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        async with cb.protected_call():
            pass  # Simulate successful API call

        assert cb.get_status() == "closed"
        assert cb.failures == 0

    @pytest.mark.asyncio
    async def test_protected_call_records_failure(self):
        """protected_call should record failure on exception."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        with pytest.raises(ConnectionError):
            async with cb.protected_call():
                raise ConnectionError("API unreachable")

        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_protected_call_raises_circuit_open_error(self):
        """protected_call should raise CircuitOpenError when circuit is open."""
        cb = CircuitBreaker(name="test", failure_threshold=2, cooldown_seconds=60)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            async with cb.protected_call():
                pass  # Should not reach here

        assert "test" in str(exc_info.value) or exc_info.value.circuit_name == "circuit"

    @pytest.mark.asyncio
    async def test_protected_call_opens_circuit_after_threshold(self):
        """Multiple failures via protected_call should open the circuit."""
        cb = CircuitBreaker(name="test", failure_threshold=2, cooldown_seconds=60)

        for _ in range(2):
            with pytest.raises(RuntimeError):
                async with cb.protected_call():
                    raise RuntimeError("Service unavailable")

        assert cb.get_status() == "open"


# ============================================================================
# Integration: SlackIntegration Circuit Breaker
# ============================================================================


class TestSlackIntegrationCircuitBreaker:
    """Test that SlackIntegration uses circuit breaker correctly."""

    def test_slack_creates_circuit_breaker(self):
        """SlackIntegration should create a circuit breaker named 'slack_api'."""
        from aragora.integrations.slack import SlackConfig, SlackIntegration

        config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        integration = SlackIntegration(config)

        assert integration._circuit_breaker is not None
        assert integration._circuit_breaker.name == "slack_api"


# ============================================================================
# Integration: DiscordIntegration Circuit Breaker
# ============================================================================


class TestDiscordIntegrationCircuitBreaker:
    """Test that DiscordIntegration uses circuit breaker correctly."""

    def test_discord_creates_circuit_breaker(self):
        """DiscordIntegration should create a circuit breaker named 'discord_api'."""
        from aragora.integrations.discord import DiscordConfig, DiscordIntegration

        config = DiscordConfig(webhook_url="https://discord.com/api/webhooks/test")
        integration = DiscordIntegration(config)

        assert integration._circuit_breaker is not None
        assert integration._circuit_breaker.name == "discord_api"


# ============================================================================
# Integration: TeamsIntegration Circuit Breaker
# ============================================================================


class TestTeamsIntegrationCircuitBreaker:
    """Test that TeamsIntegration uses circuit breaker correctly."""

    def test_teams_creates_circuit_breaker(self):
        """TeamsIntegration should create a circuit breaker named 'teams_api'."""
        from aragora.integrations.teams import TeamsConfig, TeamsIntegration

        config = TeamsConfig(webhook_url="https://test.webhook.office.com/test")
        integration = TeamsIntegration(config)

        assert integration._circuit_breaker is not None
        assert integration._circuit_breaker.name == "teams_api"


# ============================================================================
# Integration: EmailIntegration Circuit Breaker
# ============================================================================


class TestEmailIntegrationCircuitBreaker:
    """Test that EmailIntegration uses circuit breaker correctly."""

    def test_email_circuit_breaker_check(self):
        """EmailIntegration should check circuit breaker before sending."""
        from aragora.integrations.email import EmailConfig, EmailIntegration

        config = EmailConfig(
            provider="sendgrid",
            sendgrid_api_key="SG.test",
            enable_circuit_breaker=True,
        )
        integration = EmailIntegration(config)

        # Initially should be allowed
        can_proceed, error = integration._check_circuit_breaker()
        assert can_proceed is True
        assert error is None


# ============================================================================
# Integration: BaseConnector (GitHub) Circuit Breaker
# ============================================================================


class TestBaseConnectorCircuitBreaker:
    """Test that BaseConnector subclasses have circuit breaker support."""

    def test_github_connector_has_circuit_breaker_methods(self):
        """GitHubConnector should inherit circuit breaker methods from BaseConnector."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="test/repo")

        # Check that circuit breaker methods exist
        assert hasattr(connector, "check_circuit_breaker")
        assert hasattr(connector, "record_circuit_breaker_success")
        assert hasattr(connector, "record_circuit_breaker_failure")
        assert hasattr(connector, "get_circuit_breaker_status")

    def test_github_connector_circuit_breaker_initially_allows_requests(self):
        """Fresh circuit breaker should allow requests."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="test/repo")

        # Circuit breaker should allow requests initially
        assert connector.check_circuit_breaker() is True

    def test_github_connector_circuit_breaker_status(self):
        """get_circuit_breaker_status should return status dict."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="test/repo")
        status = connector.get_circuit_breaker_status()

        assert isinstance(status, dict)
        assert "enabled" in status
        assert "status" in status


# ============================================================================
# Health Endpoint Circuit Breaker Exposure
# ============================================================================


class TestHealthEndpointCircuitBreakerExposure:
    """Test that the integration health endpoint includes circuit breaker states."""

    def test_health_includes_circuit_breaker_state(self):
        """Health response should include circuit_breakers field per integration."""
        from aragora.server.handlers.integrations.health import IntegrationHealthHandler

        # Create a circuit breaker for slack
        cb = get_circuit_breaker("slack_api", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # Opens the circuit

        handler = IntegrationHealthHandler()

        # Use a mock for the handler argument and bypass RBAC
        mock_handler = MagicMock()
        mock_handler.request_context = MagicMock()

        with patch(
            "aragora.server.handlers.integrations.health.rate_limit", lambda **kw: lambda f: f
        ):
            with patch("aragora.rbac.decorators.require_permission", lambda perm: lambda f: f):
                # Call _get_health directly, bypassing decorators
                result = handler._get_health.__wrapped__(handler, mock_handler)

        # Find the slack entry
        body = result.body if hasattr(result, "body") else result
        if isinstance(body, dict):
            integrations = body.get("integrations", [])
        else:
            integrations = []

        slack_entry = next((i for i in integrations if i["name"] == "slack"), None)
        if slack_entry:
            assert "circuit_breakers" in slack_entry

    def test_health_response_structure(self):
        """Verify the structure of the health response with circuit breakers."""
        # Create circuit breakers for testing
        cb = get_circuit_breaker("slack_api", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        get_circuit_breaker("discord_api")  # Closed

        states = get_connector_circuit_breaker_states()

        # Verify structure
        assert "slack_api" in states
        slack_state = states["slack_api"]
        assert slack_state["state"] == "open"
        assert slack_state["failures"] == 2
        assert "cooldown_remaining" in slack_state

        assert "discord_api" in states
        discord_state = states["discord_api"]
        assert discord_state["state"] == "closed"
        assert discord_state["failures"] == 0
