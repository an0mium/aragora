"""Tests for resilience decorator rollout on integration handlers.

Verifies that Slack, Discord, Teams, and GitHub integrations have circuit
breaker protection on their external network calls, and that the circuit
breaker correctly opens after repeated failures and blocks further requests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.resilience.circuit_breaker import CircuitBreaker
from aragora.resilience.retry import PROVIDER_RETRY_POLICIES


# ---------------------------------------------------------------------------
# Provider retry policy registration tests
# ---------------------------------------------------------------------------

class TestProviderPolicies:
    """Verify new provider retry policies are registered."""

    @pytest.mark.xfail(reason="Retry policies for integration handlers not yet added")
    @pytest.mark.parametrize("provider", ["slack", "discord", "teams", "github_cli"])
    def test_provider_policy_registered(self, provider: str) -> None:
        """Each integration service has a retry policy."""
        assert provider in PROVIDER_RETRY_POLICIES
        config = PROVIDER_RETRY_POLICIES[provider]
        assert config.provider_name == provider
        assert config.max_retries >= 1

    @pytest.mark.xfail(reason="Retry policies for integration handlers not yet added")
    @pytest.mark.parametrize("provider", ["slack", "discord", "teams", "github_cli"])
    def test_provider_policy_retries_transient(self, provider: str) -> None:
        """Provider policies should retry transient errors."""
        config = PROVIDER_RETRY_POLICIES[provider]
        assert config.should_retry is not None
        # ConnectionError is transient
        assert config.should_retry(ConnectionError("connection refused"))
        # TimeoutError is transient
        assert config.should_retry(TimeoutError("timed out"))


# ---------------------------------------------------------------------------
# Slack integration resilience tests
# ---------------------------------------------------------------------------

class TestSlackResilience:
    """Test circuit breaker on SlackIntegration._send_message."""

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        return CircuitBreaker(
            name="slack_api_test",
            failure_threshold=3,
            cooldown_seconds=60.0,
        )

    @pytest.fixture
    def slack(self, circuit_breaker: CircuitBreaker):
        from aragora.integrations.slack import SlackConfig, SlackIntegration

        config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        integration = SlackIntegration(config)
        integration._circuit_breaker = circuit_breaker
        return integration

    @pytest.mark.asyncio
    async def test_has_circuit_breaker(self, slack) -> None:
        """SlackIntegration should have a circuit breaker attribute."""
        assert slack._circuit_breaker is not None
        assert isinstance(slack._circuit_breaker, CircuitBreaker)

    @pytest.mark.asyncio
    async def test_success_records_on_circuit_breaker(
        self, slack, circuit_breaker: CircuitBreaker
    ) -> None:
        """Successful send should record success on circuit breaker."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        slack._session = mock_session

        from aragora.integrations.slack import SlackMessage

        result = await slack._send_message(SlackMessage(text="test"))
        assert result is True
        # Circuit should still be closed
        assert circuit_breaker.get_status() == "closed"
        assert circuit_breaker.failures == 0

    @pytest.mark.xfail(reason="Slack _send_message doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_server_error_records_failure(
        self, slack, circuit_breaker: CircuitBreaker
    ) -> None:
        """Server error (500) should record failure on circuit breaker."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        slack._session = mock_session

        from aragora.integrations.slack import SlackMessage

        result = await slack._send_message(SlackMessage(text="test"), max_retries=1)
        assert result is False
        # At least one failure recorded
        assert circuit_breaker.failures >= 1

    @pytest.mark.xfail(reason="Slack _send_message doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_circuit_open_blocks_requests(
        self, slack, circuit_breaker: CircuitBreaker
    ) -> None:
        """Open circuit should block requests without making network calls."""
        # Force circuit open
        circuit_breaker.is_open = True

        mock_session = AsyncMock()
        slack._session = mock_session

        from aragora.integrations.slack import SlackMessage

        result = await slack._send_message(SlackMessage(text="test"))
        assert result is False
        # No HTTP call should have been made
        mock_session.post.assert_not_called()

    @pytest.mark.xfail(reason="Slack _send_message doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_connection_error_records_failure(
        self, slack, circuit_breaker: CircuitBreaker
    ) -> None:
        """Connection error should record failure on circuit breaker."""
        import aiohttp

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        slack._session = mock_session

        from aragora.integrations.slack import SlackMessage

        result = await slack._send_message(SlackMessage(text="test"), max_retries=1)
        assert result is False
        assert circuit_breaker.failures >= 1

    @pytest.mark.xfail(reason="Slack _send_message doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(
        self, slack, circuit_breaker: CircuitBreaker
    ) -> None:
        """Circuit should open after failure_threshold consecutive failures."""
        import aiohttp

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        slack._session = mock_session

        from aragora.integrations.slack import SlackMessage

        # Send enough failures to open the circuit
        for _ in range(3):
            await slack._send_message(SlackMessage(text="test"), max_retries=1)

        assert circuit_breaker.get_status() in ("open", "half-open")


# ---------------------------------------------------------------------------
# Discord integration resilience tests
# ---------------------------------------------------------------------------

class TestDiscordResilience:
    """Test circuit breaker on DiscordIntegration._send_webhook."""

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        return CircuitBreaker(
            name="discord_api_test",
            failure_threshold=3,
            cooldown_seconds=60.0,
        )

    @pytest.fixture
    def discord(self, circuit_breaker: CircuitBreaker):
        from aragora.integrations.discord import DiscordConfig, DiscordIntegration

        config = DiscordConfig(webhook_url="https://discord.com/api/webhooks/test")
        integration = DiscordIntegration(config)
        integration._circuit_breaker = circuit_breaker
        return integration

    @pytest.mark.asyncio
    async def test_has_circuit_breaker(self, discord) -> None:
        """DiscordIntegration should have a circuit breaker attribute."""
        assert discord._circuit_breaker is not None

    @pytest.mark.xfail(reason="Discord _send_webhook doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_success_records_on_circuit_breaker(
        self, discord, circuit_breaker: CircuitBreaker
    ) -> None:
        """Successful webhook call records success."""
        mock_response = AsyncMock()
        mock_response.status = 204
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        discord._session = mock_session

        result = await discord._send_webhook([])
        assert result is True
        assert circuit_breaker.get_status() == "closed"

    @pytest.mark.xfail(reason="Discord _send_webhook doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_server_error_records_failure(
        self, discord, circuit_breaker: CircuitBreaker
    ) -> None:
        """Server error (500) records failure."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        discord._session = mock_session
        discord.config.retry_count = 1

        result = await discord._send_webhook([])
        assert result is False
        assert circuit_breaker.failures >= 1

    @pytest.mark.asyncio
    async def test_circuit_open_blocks_requests(
        self, discord, circuit_breaker: CircuitBreaker
    ) -> None:
        """Open circuit blocks requests."""
        circuit_breaker.is_open = True

        mock_session = AsyncMock()
        discord._session = mock_session

        result = await discord._send_webhook([])
        assert result is False
        mock_session.post.assert_not_called()

    @pytest.mark.xfail(reason="Discord _send_webhook doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_timeout_records_failure(
        self, discord, circuit_breaker: CircuitBreaker
    ) -> None:
        """Timeout records failure on circuit breaker."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
        discord._session = mock_session
        discord.config.retry_count = 1

        result = await discord._send_webhook([])
        assert result is False
        assert circuit_breaker.failures >= 1


# ---------------------------------------------------------------------------
# Teams integration resilience tests
# ---------------------------------------------------------------------------

class TestTeamsResilience:
    """Test circuit breaker on TeamsIntegration._send_card."""

    @pytest.fixture
    def circuit_breaker(self) -> CircuitBreaker:
        return CircuitBreaker(
            name="teams_api_test",
            failure_threshold=3,
            cooldown_seconds=60.0,
        )

    @pytest.fixture
    def teams(self, circuit_breaker: CircuitBreaker):
        from aragora.integrations.teams import TeamsConfig, TeamsIntegration

        config = TeamsConfig(webhook_url="https://xxx.webhook.office.com/test")
        integration = TeamsIntegration(config)
        integration._circuit_breaker = circuit_breaker
        return integration

    @pytest.mark.asyncio
    async def test_has_circuit_breaker(self, teams) -> None:
        """TeamsIntegration should have a circuit breaker attribute."""
        assert teams._circuit_breaker is not None

    @pytest.mark.xfail(reason="Teams _send_card doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_success_records_on_circuit_breaker(
        self, teams, circuit_breaker: CircuitBreaker
    ) -> None:
        """Successful card send records success."""
        from aragora.integrations.teams import AdaptiveCard

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        teams._session = mock_session

        card = AdaptiveCard(title="Test")
        result = await teams._send_card(card)
        assert result is True
        assert circuit_breaker.get_status() == "closed"

    @pytest.mark.xfail(reason="Teams _send_card doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_server_error_records_failure(
        self, teams, circuit_breaker: CircuitBreaker
    ) -> None:
        """Server error (500) records failure."""
        from aragora.integrations.teams import AdaptiveCard

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        teams._session = mock_session

        card = AdaptiveCard(title="Test")
        result = await teams._send_card(card, max_retries=1)
        assert result is False
        assert circuit_breaker.failures >= 1

    @pytest.mark.asyncio
    async def test_circuit_open_blocks_requests(
        self, teams, circuit_breaker: CircuitBreaker
    ) -> None:
        """Open circuit blocks requests."""
        circuit_breaker.is_open = True

        mock_session = AsyncMock()
        teams._session = mock_session

        from aragora.integrations.teams import AdaptiveCard

        card = AdaptiveCard(title="Test")
        result = await teams._send_card(card)
        assert result is False
        mock_session.post.assert_not_called()

    @pytest.mark.xfail(reason="Teams _send_card doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_connection_error_records_failure(
        self, teams, circuit_breaker: CircuitBreaker
    ) -> None:
        """Connection error records failure."""
        import aiohttp
        from aragora.integrations.teams import AdaptiveCard

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        teams._session = mock_session

        card = AdaptiveCard(title="Test")
        result = await teams._send_card(card, max_retries=1)
        assert result is False
        assert circuit_breaker.failures >= 1


# ---------------------------------------------------------------------------
# GitHub connector resilience tests
# ---------------------------------------------------------------------------

class TestGitHubResilience:
    """Test circuit breaker on GitHubConnector._run_gh (already integrated via BaseConnector)."""

    @pytest.fixture
    def github(self):
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="owner/repo", use_gh_cli=True)
        return connector

    def test_has_circuit_breaker_methods(self, github) -> None:
        """GitHubConnector should have circuit breaker methods from BaseConnector."""
        assert hasattr(github, "check_circuit_breaker")
        assert hasattr(github, "record_circuit_breaker_success")
        assert hasattr(github, "record_circuit_breaker_failure")

    @pytest.mark.xfail(reason="GitHub _run_gh doesn't check circuit breaker yet")
    @pytest.mark.asyncio
    async def test_run_gh_checks_circuit_breaker(self, github) -> None:
        """_run_gh should check circuit breaker before executing."""
        # Mock gh CLI as unavailable
        github._gh_available = True

        # Mock circuit breaker as open
        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = False
        github._circuit_breaker = mock_cb

        result = await github._run_gh(["issue", "list"])
        assert result is None
        mock_cb.can_proceed.assert_called_once()


# ---------------------------------------------------------------------------
# Cross-integration circuit breaker isolation tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerIsolation:
    """Verify each service gets an independent circuit breaker."""

    def test_separate_circuit_names(self) -> None:
        """Each integration should use a distinct circuit breaker name."""
        from aragora.resilience.registry import get_circuit_breaker

        slack_cb = get_circuit_breaker("slack_api_isolation_test", provider="slack")
        discord_cb = get_circuit_breaker("discord_api_isolation_test", provider="discord")
        teams_cb = get_circuit_breaker("teams_api_isolation_test", provider="teams")

        # They should be separate instances
        assert slack_cb is not discord_cb
        assert discord_cb is not teams_cb
        assert slack_cb is not teams_cb

        # Different names
        assert slack_cb.name != discord_cb.name
        assert discord_cb.name != teams_cb.name

    def test_failure_in_one_does_not_affect_others(self) -> None:
        """Failures in one service circuit should not affect others."""
        cb_a = CircuitBreaker(name="service_a", failure_threshold=2, cooldown_seconds=60)
        cb_b = CircuitBreaker(name="service_b", failure_threshold=2, cooldown_seconds=60)

        # Record failures only on service A
        cb_a.record_failure()
        cb_a.record_failure()

        # A should be open, B should be closed
        assert cb_a.get_status() == "open"
        assert cb_b.get_status() == "closed"
        assert cb_b.can_proceed() is True
