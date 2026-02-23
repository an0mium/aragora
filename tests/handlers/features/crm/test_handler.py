"""Tests for the CRM handler.

Covers all routes and behavior of the CRMHandler class:
- can_handle() routing
- GET /api/v1/crm/status - circuit breaker status
- GET /api/v1/crm/platforms - list supported platforms
- POST /api/v1/crm/connect - connect a platform
- DELETE /api/v1/crm/{platform} - disconnect a platform
- GET /api/v1/crm/contacts - list all contacts
- GET /api/v1/crm/{platform}/contacts - list platform contacts
- GET /api/v1/crm/{platform}/contacts/{id} - get single contact
- POST /api/v1/crm/{platform}/contacts - create contact
- PUT /api/v1/crm/{platform}/contacts/{id} - update contact
- GET /api/v1/crm/companies - list all companies
- GET /api/v1/crm/{platform}/companies - list platform companies
- GET /api/v1/crm/{platform}/companies/{id} - get single company
- POST /api/v1/crm/{platform}/companies - create company
- GET /api/v1/crm/deals - list all deals
- GET /api/v1/crm/{platform}/deals - list platform deals
- GET /api/v1/crm/{platform}/deals/{id} - get single deal
- POST /api/v1/crm/{platform}/deals - create deal
- GET /api/v1/crm/pipeline - pipeline summary
- POST /api/v1/crm/sync-lead - lead sync
- POST /api/v1/crm/enrich - contact enrichment
- POST /api/v1/crm/search - cross-platform search
- Circuit breaker rejection (503)
- Validation errors (400)
- Platform not connected (404)
- Endpoint not found (404)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.crm.handler import (
    CRMHandler,
    _platform_credentials,
    _platform_connectors,
)
from aragora.server.handlers.features.crm.circuit_breaker import reset_crm_circuit_breaker
from aragora.server.handlers.features.crm.models import SUPPORTED_PLATFORMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: dict) -> dict:
    """Extract body dict from a HandlerResult dict."""
    if isinstance(result, dict):
        b = result.get("body", result)
        return b
    return json.loads(result.body)


def _status(result: dict) -> int:
    """Extract HTTP status code from a HandlerResult dict."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock Request
# ---------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock async HTTP request for CRMHandler."""

    method: str = "GET"
    path: str = "/"
    query: dict[str, str] = field(default_factory=dict)
    _body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    async def json(self) -> dict[str, Any]:
        return self._body or {}

    async def body(self) -> bytes:
        return json.dumps(self._body or {}).encode()

    async def read(self) -> bytes:
        return json.dumps(self._body or {}).encode()


def _req(
    method: str = "GET",
    path: str = "/api/v1/crm/platforms",
    query: dict | None = None,
    body: dict | None = None,
) -> MockRequest:
    """Shortcut to create a MockRequest."""
    return MockRequest(method=method, path=path, query=query or {}, _body=body or {})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a CRMHandler instance."""
    return CRMHandler({})


@pytest.fixture(autouse=True)
def _clean_crm_state():
    """Reset module-level CRM state before and after each test."""
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_crm_circuit_breaker()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_crm_circuit_breaker()


def _connect_hubspot():
    """Insert hubspot credentials into the module-level store."""
    _platform_credentials["hubspot"] = {
        "credentials": {"access_token": "tok-123"},
        "connected_at": "2026-02-23T00:00:00+00:00",
    }


def _open_circuit_breaker():
    """Force the circuit breaker into the OPEN state."""
    from aragora.server.handlers.features.crm.circuit_breaker import get_crm_circuit_breaker

    cb = get_crm_circuit_breaker()
    # Record enough failures to trip it
    for _ in range(cb.failure_threshold + 1):
        cb.record_failure()


# ---------------------------------------------------------------------------
# can_handle Tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_handles_crm_platforms(self, handler):
        assert handler.can_handle("/api/v1/crm/platforms") is True

    def test_handles_crm_connect(self, handler):
        assert handler.can_handle("/api/v1/crm/connect") is True

    def test_handles_crm_status(self, handler):
        assert handler.can_handle("/api/v1/crm/status") is True

    def test_handles_crm_contacts(self, handler):
        assert handler.can_handle("/api/v1/crm/contacts") is True

    def test_handles_platform_contacts(self, handler):
        assert handler.can_handle("/api/v1/crm/hubspot/contacts") is True

    def test_handles_crm_companies(self, handler):
        assert handler.can_handle("/api/v1/crm/companies") is True

    def test_handles_crm_deals(self, handler):
        assert handler.can_handle("/api/v1/crm/deals") is True

    def test_handles_crm_pipeline(self, handler):
        assert handler.can_handle("/api/v1/crm/pipeline") is True

    def test_handles_crm_sync_lead(self, handler):
        assert handler.can_handle("/api/v1/crm/sync-lead") is True

    def test_handles_crm_enrich(self, handler):
        assert handler.can_handle("/api/v1/crm/enrich") is True

    def test_handles_crm_search(self, handler):
        assert handler.can_handle("/api/v1/crm/search") is True

    def test_does_not_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_does_not_handle_health(self, handler):
        assert handler.can_handle("/api/v1/health") is False

    def test_does_not_handle_partial_crm(self, handler):
        assert handler.can_handle("/api/v1/crm") is False


# ---------------------------------------------------------------------------
# ROUTES definition
# ---------------------------------------------------------------------------


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_is_not_empty(self, handler):
        assert len(handler.ROUTES) > 0

    def test_routes_contain_key_paths(self, handler):
        expected = [
            "/api/v1/crm/platforms",
            "/api/v1/crm/connect",
            "/api/v1/crm/status",
            "/api/v1/crm/contacts",
            "/api/v1/crm/companies",
            "/api/v1/crm/deals",
            "/api/v1/crm/pipeline",
            "/api/v1/crm/sync-lead",
            "/api/v1/crm/enrich",
            "/api/v1/crm/search",
        ]
        for path in expected:
            assert path in handler.ROUTES, f"Missing route: {path}"


# ---------------------------------------------------------------------------
# GET /api/v1/crm/status
# ---------------------------------------------------------------------------


class TestStatus:
    """Tests for GET /api/v1/crm/status."""

    @pytest.mark.asyncio
    async def test_status_healthy_no_platforms(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/status"))
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["connected_platforms"] == []
        assert body["connected_count"] == 0
        assert "supported_platforms" in body

    @pytest.mark.asyncio
    async def test_status_with_connected_platform(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(_req(path="/api/v1/crm/status"))
        assert _status(result) == 200
        body = _body(result)
        assert "hubspot" in body["connected_platforms"]
        assert body["connected_count"] == 1

    @pytest.mark.asyncio
    async def test_status_includes_circuit_breaker_info(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/status"))
        body = _body(result)
        assert "circuit_breaker" in body
        assert body["circuit_breaker"]["state"] == "closed"


# ---------------------------------------------------------------------------
# GET /api/v1/crm/platforms
# ---------------------------------------------------------------------------


class TestListPlatforms:
    """Tests for GET /api/v1/crm/platforms."""

    @pytest.mark.asyncio
    async def test_list_platforms_returns_all(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/platforms"))
        assert _status(result) == 200
        body = _body(result)
        ids = [p["id"] for p in body["platforms"]]
        assert "hubspot" in ids
        assert "salesforce" in ids
        assert "pipedrive" in ids

    @pytest.mark.asyncio
    async def test_list_platforms_shows_connection_status(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(_req(path="/api/v1/crm/platforms"))
        body = _body(result)
        hubspot = next(p for p in body["platforms"] if p["id"] == "hubspot")
        assert hubspot["connected"] is True
        salesforce = next(p for p in body["platforms"] if p["id"] == "salesforce")
        assert salesforce["connected"] is False

    @pytest.mark.asyncio
    async def test_list_platforms_connected_count(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/platforms"))
        body = _body(result)
        assert body["connected_count"] == 0

    @pytest.mark.asyncio
    async def test_list_platforms_coming_soon_flag(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/platforms"))
        body = _body(result)
        salesforce = next(p for p in body["platforms"] if p["id"] == "salesforce")
        assert salesforce["coming_soon"] is True

    @pytest.mark.asyncio
    async def test_list_platforms_includes_features(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/platforms"))
        body = _body(result)
        hubspot = next(p for p in body["platforms"] if p["id"] == "hubspot")
        assert "contacts" in hubspot["features"]


# ---------------------------------------------------------------------------
# POST /api/v1/crm/connect
# ---------------------------------------------------------------------------


class TestConnectPlatform:
    """Tests for POST /api/v1/crm/connect."""

    @pytest.mark.asyncio
    async def test_connect_hubspot_success(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/connect",
                body={"platform": "hubspot", "credentials": {"access_token": "tok-abc"}},
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "hubspot"
        assert "connected_at" in body
        assert "hubspot" in _platform_credentials

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler):
        result = await handler.handle_request(
            _req(method="POST", path="/api/v1/crm/connect", body={"credentials": {}})
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_unsupported_platform(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/connect",
                body={"platform": "zoho", "credentials": {"key": "val"}},
            )
        )
        assert _status(result) == 400
        assert "Unsupported" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_connect_coming_soon_platform(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/connect",
                body={
                    "platform": "salesforce",
                    "credentials": {
                        "client_id": "c",
                        "client_secret": "s",
                        "refresh_token": "r",
                        "instance_url": "https://example.com",
                    },
                },
            )
        )
        assert _status(result) == 400
        assert "coming soon" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/connect",
                body={"platform": "hubspot", "credentials": {}},
            )
        )
        assert _status(result) == 400
        assert "Credentials" in _body(result).get("error", "") or "required" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_connect_missing_required_credential_field(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/connect",
                body={"platform": "hubspot", "credentials": {"wrong_key": "val"}},
            )
        )
        assert _status(result) == 400
        assert "access_token" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_connect_credential_too_long(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/connect",
                body={"platform": "hubspot", "credentials": {"access_token": "x" * 2000}},
            )
        )
        assert _status(result) == 400
        assert "too long" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_connect_invalid_platform_format(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/connect",
                body={"platform": "bad platform!", "credentials": {"key": "val"}},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_empty_body(self, handler):
        result = await handler.handle_request(
            _req(method="POST", path="/api/v1/crm/connect", body={})
        )
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# DELETE /api/v1/crm/{platform}
# ---------------------------------------------------------------------------


class TestDisconnectPlatform:
    """Tests for DELETE /api/v1/crm/{platform}."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(method="DELETE", path="/api/v1/crm/hubspot")
        )
        assert _status(result) == 200
        assert "hubspot" not in _platform_credentials

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler):
        result = await handler.handle_request(
            _req(method="DELETE", path="/api/v1/crm/hubspot")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_disconnect_closes_connector(self, handler):
        _connect_hubspot()
        mock_conn = AsyncMock()
        mock_conn.close = AsyncMock()
        _platform_connectors["hubspot"] = mock_conn

        result = await handler.handle_request(
            _req(method="DELETE", path="/api/v1/crm/hubspot")
        )
        assert _status(result) == 200
        mock_conn.close.assert_awaited_once()
        assert "hubspot" not in _platform_connectors


# ---------------------------------------------------------------------------
# GET /api/v1/crm/contacts (all platforms)
# ---------------------------------------------------------------------------


class TestListAllContacts:
    """Tests for GET /api/v1/crm/contacts."""

    @pytest.mark.asyncio
    async def test_list_all_contacts_empty(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/contacts"))
        assert _status(result) == 200
        body = _body(result)
        assert body["contacts"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_contacts_invalid_email_filter(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/contacts", query={"email": "not-an-email"})
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_all_contacts_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler.handle_request(_req(path="/api/v1/crm/contacts"))
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/{platform}/contacts
# ---------------------------------------------------------------------------


class TestListPlatformContacts:
    """Tests for GET /api/v1/crm/{platform}/contacts."""

    @pytest.mark.asyncio
    async def test_list_platform_contacts_success(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/contacts")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["contacts"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_platform_contacts_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/contacts")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_list_platform_contacts_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/contacts")
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/{platform}/contacts/{id}
# ---------------------------------------------------------------------------


class TestGetContact:
    """Tests for GET /api/v1/crm/{platform}/contacts/{id}."""

    @pytest.mark.asyncio
    async def test_get_contact_not_found(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/contacts/c123")
        )
        # The stub returns 404 "Contact not found"
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_contact_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/contacts/c123")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_contact_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/contacts/c123")
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# POST /api/v1/crm/{platform}/contacts
# ---------------------------------------------------------------------------


class TestCreateContact:
    """Tests for POST /api/v1/crm/{platform}/contacts."""

    @pytest.mark.asyncio
    async def test_create_contact_success(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/contacts",
                body={"email": "alice@example.com", "first_name": "Alice", "last_name": "Smith"},
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["contact"]["email"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_create_contact_missing_email(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/contacts",
                body={"first_name": "Alice"},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_contact_invalid_email(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/contacts",
                body={"email": "not-valid"},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_contact_name_too_long(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/contacts",
                body={"email": "a@b.com", "first_name": "x" * 200},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_contact_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/contacts",
                body={"email": "a@b.com"},
            )
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_contact_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/contacts",
                body={"email": "a@b.com"},
            )
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# PUT /api/v1/crm/{platform}/contacts/{id}
# ---------------------------------------------------------------------------


class TestUpdateContact:
    """Tests for PUT /api/v1/crm/{platform}/contacts/{id}."""

    @pytest.mark.asyncio
    async def test_update_contact_success(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="PUT",
                path="/api/v1/crm/hubspot/contacts/c123",
                body={"first_name": "Bob"},
            )
        )
        assert _status(result) == 200
        assert _body(result).get("success") is True

    @pytest.mark.asyncio
    async def test_update_contact_invalid_email(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="PUT",
                path="/api/v1/crm/hubspot/contacts/c123",
                body={"email": "bad-email"},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_contact_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(
                method="PUT",
                path="/api/v1/crm/hubspot/contacts/c123",
                body={"first_name": "Bob"},
            )
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_contact_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(
                method="PUT",
                path="/api/v1/crm/hubspot/contacts/c123",
                body={"first_name": "Bob"},
            )
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/companies (all platforms)
# ---------------------------------------------------------------------------


class TestListAllCompanies:
    """Tests for GET /api/v1/crm/companies."""

    @pytest.mark.asyncio
    async def test_list_all_companies_empty(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/companies"))
        assert _status(result) == 200
        body = _body(result)
        assert body["companies"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_companies_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler.handle_request(_req(path="/api/v1/crm/companies"))
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/{platform}/companies
# ---------------------------------------------------------------------------


class TestListPlatformCompanies:
    """Tests for GET /api/v1/crm/{platform}/companies."""

    @pytest.mark.asyncio
    async def test_list_platform_companies_no_connector(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(path="/api/v1/crm/hubspot/companies")
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["companies"] == []
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_list_platform_companies_with_connector(self, handler):
        _connect_hubspot()
        mock_company = MagicMock()
        mock_company.id = "co1"
        mock_company.properties = {"name": "Acme"}
        mock_company.created_at = None

        mock_conn = AsyncMock()
        mock_conn.get_companies = AsyncMock(return_value=[mock_company])
        _platform_connectors["hubspot"] = mock_conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies")
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["companies"]) == 1
        assert body["companies"][0]["name"] == "Acme"

    @pytest.mark.asyncio
    async def test_list_platform_companies_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_list_platform_companies_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies")
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/{platform}/companies/{id}
# ---------------------------------------------------------------------------


class TestGetCompany:
    """Tests for GET /api/v1/crm/{platform}/companies/{id}."""

    @pytest.mark.asyncio
    async def test_get_company_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies/co123")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_company_no_connector(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(path="/api/v1/crm/hubspot/companies/co123")
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_company_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies/co123")
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# POST /api/v1/crm/{platform}/companies
# ---------------------------------------------------------------------------


class TestCreateCompany:
    """Tests for POST /api/v1/crm/{platform}/companies."""

    @pytest.mark.asyncio
    async def test_create_company_missing_name(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/companies",
                body={"domain": "example.com"},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_company_name_too_long(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/companies",
                body={"name": "x" * 300},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_company_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/companies",
                body={"name": "Acme"},
            )
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_company_no_connector(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(
                    method="POST",
                    path="/api/v1/crm/hubspot/companies",
                    body={"name": "Acme"},
                )
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_company_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/companies",
                body={"name": "Acme"},
            )
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/deals (all platforms)
# ---------------------------------------------------------------------------


class TestListAllDeals:
    """Tests for GET /api/v1/crm/deals."""

    @pytest.mark.asyncio
    async def test_list_all_deals_empty(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 200
        body = _body(result)
        assert body["deals"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_deals_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/{platform}/deals
# ---------------------------------------------------------------------------


class TestListPlatformDeals:
    """Tests for GET /api/v1/crm/{platform}/deals."""

    @pytest.mark.asyncio
    async def test_list_platform_deals_no_connector(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(path="/api/v1/crm/hubspot/deals")
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["deals"] == []
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_list_platform_deals_with_connector(self, handler):
        _connect_hubspot()
        mock_deal = MagicMock()
        mock_deal.id = "d1"
        mock_deal.properties = {"dealname": "Deal1", "amount": "1000", "dealstage": "proposal"}
        mock_deal.created_at = None

        mock_conn = AsyncMock()
        mock_conn.get_deals = AsyncMock(return_value=[mock_deal])
        _platform_connectors["hubspot"] = mock_conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["deals"]) == 1

    @pytest.mark.asyncio
    async def test_list_platform_deals_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_list_platform_deals_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/{platform}/deals/{id}
# ---------------------------------------------------------------------------


class TestGetDeal:
    """Tests for GET /api/v1/crm/{platform}/deals/{id}."""

    @pytest.mark.asyncio
    async def test_get_deal_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d123")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_deal_no_connector(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(path="/api/v1/crm/hubspot/deals/d123")
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_deal_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d123")
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# POST /api/v1/crm/{platform}/deals
# ---------------------------------------------------------------------------


class TestCreateDeal:
    """Tests for POST /api/v1/crm/{platform}/deals."""

    @pytest.mark.asyncio
    async def test_create_deal_missing_name(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"stage": "proposal"},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_deal_missing_stage(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Big Deal"},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_deal_negative_amount(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal", "amount": -100},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_deal_invalid_amount(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal", "amount": "not-a-number"},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_deal_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_deal_no_connector(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(
                    method="POST",
                    path="/api/v1/crm/hubspot/deals",
                    body={"name": "Deal", "stage": "proposal"},
                )
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_deal_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /api/v1/crm/pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    """Tests for GET /api/v1/crm/pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_empty(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/pipeline"))
        assert _status(result) == 200
        body = _body(result)
        assert body["pipelines"] == []
        assert body["total_deals"] == 0
        assert body["total_pipeline_value"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_with_platform_filter_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/pipeline", query={"platform": "hubspot"})
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_pipeline_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler.handle_request(_req(path="/api/v1/crm/pipeline"))
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_pipeline_invalid_platform_filter(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/pipeline", query={"platform": "bad platform!"})
        )
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/crm/sync-lead
# ---------------------------------------------------------------------------


class TestSyncLead:
    """Tests for POST /api/v1/crm/sync-lead."""

    @pytest.mark.asyncio
    async def test_sync_lead_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/sync-lead",
                body={
                    "platform": "hubspot",
                    "source": "linkedin",
                    "lead": {"email": "lead@example.com"},
                },
            )
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_sync_lead_missing_email(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/sync-lead",
                body={"platform": "hubspot", "lead": {}},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_lead_no_connector(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(
                    method="POST",
                    path="/api/v1/crm/sync-lead",
                    body={
                        "platform": "hubspot",
                        "lead": {"email": "lead@example.com"},
                    },
                )
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_sync_lead_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/sync-lead",
                body={
                    "platform": "hubspot",
                    "lead": {"email": "lead@example.com"},
                },
            )
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_sync_lead_with_connector_creates_new(self, handler):
        _connect_hubspot()
        mock_contact = MagicMock()
        mock_contact.id = "c1"
        mock_contact.properties = {"email": "lead@example.com", "firstname": "New"}
        mock_contact.created_at = None
        mock_contact.updated_at = None
        mock_conn = AsyncMock()
        mock_conn.get_contact_by_email = AsyncMock(return_value=None)
        mock_conn.create_contact = AsyncMock(return_value=mock_contact)
        _platform_connectors["hubspot"] = mock_conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/sync-lead",
                body={
                    "platform": "hubspot",
                    "source": "form",
                    "lead": {"email": "lead@example.com", "first_name": "New"},
                },
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["action"] == "created"
        assert body["source"] == "form"

    @pytest.mark.asyncio
    async def test_sync_lead_with_connector_updates_existing(self, handler):
        _connect_hubspot()
        existing_contact = MagicMock()
        existing_contact.id = "c-exist"
        existing_contact.properties = {"email": "lead@example.com"}
        existing_contact.created_at = None
        existing_contact.updated_at = None

        updated_contact = MagicMock()
        updated_contact.id = "c-exist"
        updated_contact.properties = {"email": "lead@example.com", "firstname": "Updated"}
        updated_contact.created_at = None
        updated_contact.updated_at = None

        mock_conn = AsyncMock()
        mock_conn.get_contact_by_email = AsyncMock(return_value=existing_contact)
        mock_conn.update_contact = AsyncMock(return_value=updated_contact)
        _platform_connectors["hubspot"] = mock_conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/sync-lead",
                body={
                    "platform": "hubspot",
                    "lead": {"email": "lead@example.com"},
                },
            )
        )
        assert _status(result) == 200
        assert _body(result)["action"] == "updated"


# ---------------------------------------------------------------------------
# POST /api/v1/crm/enrich
# ---------------------------------------------------------------------------


class TestEnrichContact:
    """Tests for POST /api/v1/crm/enrich."""

    @pytest.mark.asyncio
    async def test_enrich_success(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/enrich",
                body={"email": "user@example.com"},
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["email"] == "user@example.com"
        assert body["enriched"] is False
        assert "available_providers" in body

    @pytest.mark.asyncio
    async def test_enrich_missing_email(self, handler):
        result = await handler.handle_request(
            _req(method="POST", path="/api/v1/crm/enrich", body={})
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_enrich_invalid_email(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/enrich",
                body={"email": "invalid"},
            )
        )
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/crm/search
# ---------------------------------------------------------------------------


class TestSearchCRM:
    """Tests for POST /api/v1/crm/search."""

    @pytest.mark.asyncio
    async def test_search_no_platforms_connected(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "acme"},
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["query"] == "acme"
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_search_invalid_object_type(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "acme", "types": ["invalid"]},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_limit_too_high(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "acme", "limit": 500},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_limit_too_low(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "acme", "limit": 0},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_limit_not_int(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "acme", "limit": "ten"},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_query_too_long(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "x" * 300},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "acme"},
            )
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_search_with_specific_types(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "acme", "types": ["contacts"]},
            )
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_default_types(self, handler):
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "test"},
            )
        )
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# 404 for unknown endpoints
# ---------------------------------------------------------------------------


class TestEndpointNotFound:
    """Tests for unmatched routes."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/nonexistent")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_platforms(self, handler):
        result = await handler.handle_request(
            _req(method="DELETE", path="/api/v1/crm/platforms")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_on_connect_endpoint(self, handler):
        result = await handler.handle_request(
            _req(method="GET", path="/api/v1/crm/connect")
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


class TestNormalization:
    """Tests for HubSpot entity normalization methods."""

    def test_normalize_hubspot_contact(self, handler):
        mock_contact = MagicMock()
        mock_contact.id = "c1"
        mock_contact.properties = {
            "email": "alice@acme.com",
            "firstname": "Alice",
            "lastname": "Jones",
            "phone": "555-1234",
            "company": "Acme Inc",
            "jobtitle": "CEO",
            "lifecyclestage": "customer",
            "hs_lead_status": "OPEN",
            "hubspot_owner_id": "owner1",
        }
        mock_contact.created_at = None
        mock_contact.updated_at = None
        result = handler._normalize_hubspot_contact(mock_contact)
        assert result["id"] == "c1"
        assert result["platform"] == "hubspot"
        assert result["email"] == "alice@acme.com"
        assert result["first_name"] == "Alice"
        assert result["last_name"] == "Jones"
        assert result["full_name"] == "Alice Jones"
        assert result["phone"] == "555-1234"
        assert result["company"] == "Acme Inc"
        assert result["job_title"] == "CEO"

    def test_normalize_hubspot_contact_no_names(self, handler):
        mock_contact = MagicMock()
        mock_contact.id = "c2"
        mock_contact.properties = {}
        mock_contact.created_at = None
        mock_contact.updated_at = None
        result = handler._normalize_hubspot_contact(mock_contact)
        assert result["full_name"] is None

    def test_normalize_hubspot_company(self, handler):
        mock_company = MagicMock()
        mock_company.id = "co1"
        mock_company.properties = {
            "name": "Acme",
            "domain": "acme.com",
            "industry": "Tech",
            "numberofemployees": "50",
            "annualrevenue": "1000000",
            "hubspot_owner_id": "owner1",
        }
        mock_company.created_at = None
        result = handler._normalize_hubspot_company(mock_company)
        assert result["id"] == "co1"
        assert result["name"] == "Acme"
        assert result["employee_count"] == 50
        assert result["annual_revenue"] == 1000000.0

    def test_normalize_hubspot_company_invalid_numbers(self, handler):
        mock_company = MagicMock()
        mock_company.id = "co2"
        mock_company.properties = {
            "name": "Acme",
            "numberofemployees": "not-a-number",
            "annualrevenue": "also-not",
        }
        mock_company.created_at = None
        result = handler._normalize_hubspot_company(mock_company)
        assert result["employee_count"] is None
        assert result["annual_revenue"] is None

    def test_normalize_hubspot_deal(self, handler):
        mock_deal = MagicMock()
        mock_deal.id = "d1"
        mock_deal.properties = {
            "dealname": "Big Deal",
            "amount": "50000",
            "dealstage": "proposal",
            "pipeline": "default",
            "closedate": "2026-03-01",
            "hubspot_owner_id": "owner1",
        }
        mock_deal.created_at = None
        result = handler._normalize_hubspot_deal(mock_deal)
        assert result["id"] == "d1"
        assert result["name"] == "Big Deal"
        assert result["amount"] == 50000.0
        assert result["stage"] == "proposal"

    def test_normalize_hubspot_deal_invalid_amount(self, handler):
        mock_deal = MagicMock()
        mock_deal.id = "d2"
        mock_deal.properties = {"amount": "invalid"}
        mock_deal.created_at = None
        result = handler._normalize_hubspot_deal(mock_deal)
        assert result["amount"] is None

    def test_map_lead_to_hubspot(self, handler):
        lead = {
            "email": "lead@acme.com",
            "first_name": "Lead",
            "last_name": "User",
            "phone": "555-9999",
            "company": "Acme",
            "job_title": "Manager",
        }
        result = handler._map_lead_to_hubspot(lead, "linkedin")
        assert result["email"] == "lead@acme.com"
        assert result["firstname"] == "Lead"
        assert result["lastname"] == "User"
        assert result["lifecyclestage"] == "lead"
        assert result["hs_lead_status"] == "NEW"
        assert result["hs_analytics_source"] == "linkedin"


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_get_required_credentials_hubspot(self, handler):
        assert handler._get_required_credentials("hubspot") == ["access_token"]

    def test_get_required_credentials_salesforce(self, handler):
        creds = handler._get_required_credentials("salesforce")
        assert "client_id" in creds
        assert "client_secret" in creds

    def test_get_required_credentials_unknown(self, handler):
        assert handler._get_required_credentials("unknown") == []

    @pytest.mark.asyncio
    async def test_get_connector_no_credentials(self, handler):
        result = await handler._get_connector("hubspot")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_connector_returns_cached(self, handler):
        mock_conn = MagicMock()
        _platform_connectors["hubspot"] = mock_conn
        result = await handler._get_connector("hubspot")
        assert result is mock_conn

    def test_json_response_format(self, handler):
        result = handler._json_response(200, {"key": "val"})
        assert result["status_code"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["body"]["key"] == "val"

    def test_error_response_format(self, handler):
        result = handler._error_response(400, "bad request")
        assert result["status_code"] == 400
        assert result["body"]["error"] == "bad request"

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "crm"


# ---------------------------------------------------------------------------
# RBAC Permission Tests (opt out of auto-auth)
# ---------------------------------------------------------------------------


class TestRBACPermissions:
    """Tests that verify permission checks are invoked for protected endpoints."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_connect_requires_configure_permission(self):
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/connect",
                body={"platform": "hubspot", "credentials": {"access_token": "tok"}},
            )
        )
        # Without auth, should get 401 or 403
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_disconnect_requires_configure_permission(self):
        _connect_hubspot()
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(method="DELETE", path="/api/v1/crm/hubspot")
        )
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_create_contact_requires_write_permission(self):
        _connect_hubspot()
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/contacts",
                body={"email": "a@b.com"},
            )
        )
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_list_contacts_requires_read_permission(self):
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(path="/api/v1/crm/contacts")
        )
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_search_requires_read_permission(self):
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(method="POST", path="/api/v1/crm/search", body={"query": "x"})
        )
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_status_does_not_require_auth(self):
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(path="/api/v1/crm/status")
        )
        # Status is a public endpoint
        assert _status(result) == 200

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_list_platforms_does_not_require_auth(self):
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(path="/api/v1/crm/platforms")
        )
        assert _status(result) == 200
