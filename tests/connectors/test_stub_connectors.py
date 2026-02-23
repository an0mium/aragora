"""Tests for connector stub implementations.

Validates that QuickBooks, SendGrid, Twilio, Instagram, and Trello connectors
correctly implement search() and fetch() with proper API calls, sanitization,
and unconfigured fallback behavior.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


CONNECTORS = [
    (
        "quickbooks",
        "aragora.connectors.accounting.quickbooks",
        "QuickBooksConnector",
        {
            "QUICKBOOKS_CLIENT_ID": "x",
            "QUICKBOOKS_CLIENT_SECRET": "x",
            "QUICKBOOKS_ACCESS_TOKEN": "tok",
            "QUICKBOOKS_REALM_ID": "123",
        },
    ),
    (
        "sendgrid",
        "aragora.connectors.communication.sendgrid",
        "SendGridConnector",
        {"SENDGRID_API_KEY": "SG.test"},
    ),
    (
        "twilio",
        "aragora.connectors.communication.twilio",
        "TwilioConnector",
        {"TWILIO_ACCOUNT_SID": "AC123", "TWILIO_AUTH_TOKEN": "tok"},
    ),
    (
        "instagram",
        "aragora.connectors.social.instagram",
        "InstagramConnector",
        {"INSTAGRAM_ACCESS_TOKEN": "tok"},
    ),
    (
        "trello",
        "aragora.connectors.productivity.trello",
        "TrelloConnector",
        {"TRELLO_API_KEY": "key", "TRELLO_TOKEN": "tok"},
    ),
]


def _make_connector(module: str, cls_name: str):
    """Import and instantiate a connector class."""
    mod = __import__(module, fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    return cls()


@pytest.mark.parametrize("name,module,cls_name,env_vars", CONNECTORS)
class TestConnectorUnconfigured:
    """Test that unconfigured connectors return empty/None."""

    @pytest.mark.asyncio
    async def test_unconfigured_search_returns_empty(
        self, name, module, cls_name, env_vars, monkeypatch
    ):
        for key in env_vars:
            monkeypatch.delenv(key, raising=False)
        connector = _make_connector(module, cls_name)
        result = await connector.search("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_unconfigured_fetch_returns_none(
        self, name, module, cls_name, env_vars, monkeypatch
    ):
        for key in env_vars:
            monkeypatch.delenv(key, raising=False)
        connector = _make_connector(module, cls_name)
        result = await connector.fetch("test-id")
        assert result is None


@pytest.mark.parametrize("name,module,cls_name,env_vars", CONNECTORS)
class TestConnectorSanitization:
    """Test that empty/malicious queries are rejected."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(
        self, name, module, cls_name, env_vars, monkeypatch
    ):
        for key, val in env_vars.items():
            monkeypatch.setenv(key, val)
        connector = _make_connector(module, cls_name)
        # Query that sanitizes to empty (only special chars)
        result = await connector.search("!@#$%^&*()")
        assert result == []


class TestQuickBooksSearch:
    """Test QuickBooks search with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, monkeypatch):
        monkeypatch.setenv("QUICKBOOKS_CLIENT_ID", "cid")
        monkeypatch.setenv("QUICKBOOKS_CLIENT_SECRET", "csec")
        monkeypatch.setenv("QUICKBOOKS_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("QUICKBOOKS_REALM_ID", "realm1")

        from aragora.connectors.accounting.quickbooks import QuickBooksConnector

        connector = QuickBooksConnector()

        mock_response = {
            "QueryResponse": {
                "Invoice": [
                    {
                        "Id": "101",
                        "DocNumber": "INV-001",
                        "TotalAmt": 500.0,
                        "CustomerRef": {"name": "Acme Corp"},
                    }
                ]
            }
        }

        async def _mock_retry(func, op):
            return mock_response

        monkeypatch.setattr(connector, "_request_with_retry", _mock_retry)
        results = await connector.search("invoice")
        assert len(results) == 1
        assert results[0].id == "qb_inv_101"
        assert "INV-001" in results[0].content
        assert results[0].metadata["amount"] == 500.0

    @pytest.mark.asyncio
    async def test_fetch_returns_evidence(self, monkeypatch):
        monkeypatch.setenv("QUICKBOOKS_CLIENT_ID", "cid")
        monkeypatch.setenv("QUICKBOOKS_CLIENT_SECRET", "csec")
        monkeypatch.setenv("QUICKBOOKS_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("QUICKBOOKS_REALM_ID", "realm1")

        from aragora.connectors.accounting.quickbooks import QuickBooksConnector

        connector = QuickBooksConnector()

        mock_response = {
            "Invoice": {
                "Id": "101",
                "DocNumber": "INV-001",
                "TotalAmt": 750.0,
                "CustomerRef": {"name": "Beta Inc"},
            }
        }

        async def _mock_retry(func, op):
            return mock_response

        monkeypatch.setattr(connector, "_request_with_retry", _mock_retry)
        result = await connector.fetch("101")
        assert result is not None
        assert result.id == "qb_inv_101"
        assert result.metadata["customer"] == "Beta Inc"


class TestSendGridSearch:
    """Test SendGrid search with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, monkeypatch):
        monkeypatch.setenv("SENDGRID_API_KEY", "SG.test")

        from aragora.connectors.communication.sendgrid import SendGridConnector

        connector = SendGridConnector()

        mock_response = {
            "messages": [
                {
                    "msg_id": "abc123",
                    "subject": "Welcome!",
                    "to_email": "user@example.com",
                    "status": "delivered",
                }
            ]
        }

        async def _mock_retry(func, op):
            return mock_response

        monkeypatch.setattr(connector, "_request_with_retry", _mock_retry)
        results = await connector.search("welcome")
        assert len(results) == 1
        assert results[0].id == "sg_msg_abc123"
        assert "user@example.com" in results[0].content


class TestTwilioSearch:
    """Test Twilio search with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, monkeypatch):
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")

        from aragora.connectors.communication.twilio import TwilioConnector

        connector = TwilioConnector()

        mock_response = {
            "messages": [
                {
                    "sid": "SM123",
                    "body": "Hello there",
                    "from": "+15551234567",
                    "to": "+15559876543",
                    "status": "delivered",
                }
            ]
        }

        async def _mock_retry(func, op):
            return mock_response

        monkeypatch.setattr(connector, "_request_with_retry", _mock_retry)
        results = await connector.search("+15559876543")
        assert len(results) == 1
        assert results[0].id == "twilio_msg_SM123"
        assert "Hello there" in results[0].content


class TestInstagramSearch:
    """Test Instagram search with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, monkeypatch):
        monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "tok")

        from aragora.connectors.social.instagram import InstagramConnector

        connector = InstagramConnector()

        mock_response = {
            "data": [
                {
                    "id": "17890",
                    "caption": "Beautiful sunset photo",
                    "media_type": "IMAGE",
                    "timestamp": "2026-01-15T12:00:00+0000",
                    "permalink": "https://www.instagram.com/p/abc",
                }
            ]
        }

        async def _mock_retry(func, op):
            return mock_response

        monkeypatch.setattr(connector, "_request_with_retry", _mock_retry)
        results = await connector.search("sunset")
        assert len(results) == 1
        assert results[0].id == "ig_media_17890"
        assert results[0].source_type == SourceType.WEB_SEARCH
        assert "sunset" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_search_filters_by_query(self, monkeypatch):
        monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "tok")

        from aragora.connectors.social.instagram import InstagramConnector

        connector = InstagramConnector()

        mock_response = {
            "data": [
                {
                    "id": "1",
                    "caption": "Beach day",
                    "media_type": "IMAGE",
                    "timestamp": "2026-01-15T12:00:00+0000",
                    "permalink": "https://www.instagram.com/p/1",
                },
                {
                    "id": "2",
                    "caption": "Mountain hike sunset",
                    "media_type": "IMAGE",
                    "timestamp": "2026-01-15T12:00:00+0000",
                    "permalink": "https://www.instagram.com/p/2",
                },
            ]
        }

        async def _mock_retry(func, op):
            return mock_response

        monkeypatch.setattr(connector, "_request_with_retry", _mock_retry)
        results = await connector.search("sunset")
        assert len(results) == 1
        assert results[0].id == "ig_media_2"


class TestTrelloSearch:
    """Test Trello search with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, monkeypatch):
        monkeypatch.setenv("TRELLO_API_KEY", "key")
        monkeypatch.setenv("TRELLO_TOKEN", "tok")

        from aragora.connectors.productivity.trello import TrelloConnector

        connector = TrelloConnector()

        mock_response = {
            "cards": [
                {
                    "id": "card1",
                    "name": "Fix login bug",
                    "desc": "Users cannot login with SSO",
                    "shortUrl": "https://trello.com/c/abc",
                    "board": {"name": "Sprint Board"},
                }
            ]
        }

        async def _mock_retry(func, op):
            return mock_response

        monkeypatch.setattr(connector, "_request_with_retry", _mock_retry)
        results = await connector.search("login")
        assert len(results) == 1
        assert results[0].id == "trello_card_card1"
        assert "SSO" in results[0].content
        assert results[0].metadata["board_name"] == "Sprint Board"

    @pytest.mark.asyncio
    async def test_fetch_returns_evidence(self, monkeypatch):
        monkeypatch.setenv("TRELLO_API_KEY", "key")
        monkeypatch.setenv("TRELLO_TOKEN", "tok")

        from aragora.connectors.productivity.trello import TrelloConnector

        connector = TrelloConnector()

        mock_response = {
            "id": "card1",
            "name": "Fix login bug",
            "desc": "Users cannot login with SSO",
            "shortUrl": "https://trello.com/c/abc",
        }

        async def _mock_retry(func, op):
            return mock_response

        monkeypatch.setattr(connector, "_request_with_retry", _mock_retry)
        result = await connector.fetch("card1")
        assert result is not None
        assert result.id == "trello_card_card1"
        assert result.title == "Fix login bug"
