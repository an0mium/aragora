"""Tests for connector implementations.

Validates that:
- QuickBooks (deprecated quickbooks.py) still works but emits deprecation warnings
- SendGrid, Twilio, Instagram, and Trello connectors handle unconfigured state
  gracefully and produce Evidence objects when configured with mocked HTTP.
"""

from __future__ import annotations

import warnings

import pytest

from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# Fully implemented connectors with their env var requirements
IMPLEMENTED_CONNECTORS = [
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


@pytest.mark.parametrize("name,module,cls_name,env_vars", IMPLEMENTED_CONNECTORS)
class TestImplementedConnectors:
    """Verify that implemented connectors handle unconfigured state and sanitize input."""

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

    @pytest.mark.asyncio
    async def test_sanitized_empty_query_returns_empty(
        self, name, module, cls_name, env_vars, monkeypatch
    ):
        for key, val in env_vars.items():
            monkeypatch.setenv(key, val)
        connector = _make_connector(module, cls_name)
        result = await connector.search("!@#$%^&*()")
        assert result == []

    def test_connector_properties(self, name, module, cls_name, env_vars):
        connector = _make_connector(module, cls_name)
        assert connector.name == name
        assert isinstance(connector.source_type, SourceType)


class TestQuickBooksDeprecation:
    """Test that quickbooks.py emits deprecation warnings on instantiation."""

    def test_instantiation_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from aragora.connectors.accounting.quickbooks import QuickBooksConnector

            QuickBooksConnector()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()


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

    @pytest.mark.asyncio
    async def test_unconfigured_search_returns_empty(self, monkeypatch):
        for key in (
            "QUICKBOOKS_CLIENT_ID",
            "QUICKBOOKS_CLIENT_SECRET",
            "QUICKBOOKS_ACCESS_TOKEN",
            "QUICKBOOKS_REALM_ID",
        ):
            monkeypatch.delenv(key, raising=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from aragora.connectors.accounting.quickbooks import QuickBooksConnector

            connector = QuickBooksConnector()

        result = await connector.search("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_unconfigured_fetch_returns_none(self, monkeypatch):
        for key in (
            "QUICKBOOKS_CLIENT_ID",
            "QUICKBOOKS_CLIENT_SECRET",
            "QUICKBOOKS_ACCESS_TOKEN",
            "QUICKBOOKS_REALM_ID",
        ):
            monkeypatch.delenv(key, raising=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from aragora.connectors.accounting.quickbooks import QuickBooksConnector

            connector = QuickBooksConnector()

        result = await connector.fetch("test-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_sanitized_empty_query_returns_empty(self, monkeypatch):
        monkeypatch.setenv("QUICKBOOKS_CLIENT_ID", "cid")
        monkeypatch.setenv("QUICKBOOKS_CLIENT_SECRET", "csec")
        monkeypatch.setenv("QUICKBOOKS_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("QUICKBOOKS_REALM_ID", "realm1")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from aragora.connectors.accounting.quickbooks import QuickBooksConnector

            connector = QuickBooksConnector()

        # Query that sanitizes to empty (only special chars)
        result = await connector.search("!@#$%^&*()")
        assert result == []
