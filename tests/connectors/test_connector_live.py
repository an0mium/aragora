"""Live integration tests for connector API calls.

These tests only run when real sandbox API keys are present in the
environment. They are skipped gracefully when keys are absent.

Run with:
    pytest tests/connectors/test_connector_live.py -v -m integration

Set environment variables before running:
    QUICKBOOKS_ACCESS_TOKEN, QUICKBOOKS_REALM_ID
    SENDGRID_API_KEY
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
    INSTAGRAM_ACCESS_TOKEN
    TRELLO_API_KEY, TRELLO_TOKEN
"""

from __future__ import annotations

import os

import pytest


def _has_env(*keys: str) -> bool:
    """Check if all required environment variables are set."""
    return all(os.environ.get(k) for k in keys)


# ---------------------------------------------------------------------------
# QuickBooks
# ---------------------------------------------------------------------------

_quickbooks_configured = pytest.mark.skipif(
    not _has_env("QUICKBOOKS_ACCESS_TOKEN", "QUICKBOOKS_REALM_ID"),
    reason="QuickBooks sandbox credentials not configured",
)


@pytest.mark.integration
@_quickbooks_configured
class TestQuickBooksLive:
    """Live tests against QuickBooks Online sandbox."""

    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        from aragora.connectors.accounting.quickbooks import QuickBooksConnector

        connector = QuickBooksConnector()
        results = await connector.search("invoice")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_fetch_nonexistent_returns_none(self):
        from aragora.connectors.accounting.quickbooks import QuickBooksConnector

        connector = QuickBooksConnector()
        result = await connector.fetch("99999999")
        # Nonexistent ID should return None or raise gracefully
        assert result is None or hasattr(result, "id")


# ---------------------------------------------------------------------------
# SendGrid
# ---------------------------------------------------------------------------

_sendgrid_configured = pytest.mark.skipif(
    not _has_env("SENDGRID_API_KEY"),
    reason="SendGrid API key not configured",
)


@pytest.mark.integration
@_sendgrid_configured
class TestSendGridLive:
    """Live tests against SendGrid API."""

    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        from aragora.connectors.communication.sendgrid import SendGridConnector

        connector = SendGridConnector()
        results = await connector.search("test")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Twilio
# ---------------------------------------------------------------------------

_twilio_configured = pytest.mark.skipif(
    not _has_env("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
    reason="Twilio credentials not configured",
)


@pytest.mark.integration
@_twilio_configured
class TestTwilioLive:
    """Live tests against Twilio API."""

    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        from aragora.connectors.communication.twilio import TwilioConnector

        connector = TwilioConnector()
        results = await connector.search("message")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Instagram
# ---------------------------------------------------------------------------

_instagram_configured = pytest.mark.skipif(
    not _has_env("INSTAGRAM_ACCESS_TOKEN"),
    reason="Instagram access token not configured",
)


@pytest.mark.integration
@_instagram_configured
class TestInstagramLive:
    """Live tests against Instagram Graph API."""

    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        from aragora.connectors.social.instagram import InstagramConnector

        connector = InstagramConnector()
        results = await connector.search("post")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Trello
# ---------------------------------------------------------------------------

_trello_configured = pytest.mark.skipif(
    not _has_env("TRELLO_API_KEY", "TRELLO_TOKEN"),
    reason="Trello credentials not configured",
)


@pytest.mark.integration
@_trello_configured
class TestTrelloLive:
    """Live tests against Trello API."""

    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        from aragora.connectors.productivity.trello import TrelloConnector

        connector = TrelloConnector()
        results = await connector.search("card")
        assert isinstance(results, list)
