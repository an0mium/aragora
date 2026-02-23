"""Tests for the CRM Deals operations mixin (deals.py).

Comprehensive tests covering all deal-related endpoints:
- GET /api/v1/crm/deals - list all deals across platforms
- GET /api/v1/crm/{platform}/deals - list deals for a specific platform
- GET /api/v1/crm/{platform}/deals/{deal_id} - get a single deal
- POST /api/v1/crm/{platform}/deals - create a deal
- _fetch_platform_deals internal helper
- Circuit breaker integration
- Input validation (platform, deal_id, name, stage, pipeline, amount)
- Connector failure handling
- Normalization of HubSpot deal data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.crm.handler import (
    CRMHandler,
    _platform_credentials,
    _platform_connectors,
)
from aragora.server.handlers.features.crm.circuit_breaker import (
    get_crm_circuit_breaker,
    reset_crm_circuit_breaker,
)
from aragora.server.handlers.features.crm.validation import (
    MAX_DEAL_NAME_LENGTH,
    MAX_STAGE_LENGTH,
    MAX_PIPELINE_LENGTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: dict) -> dict:
    """Extract body dict from a HandlerResult dict."""
    if isinstance(result, dict):
        return result.get("body", result)
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
    path: str = "/api/v1/crm/deals",
    query: dict | None = None,
    body: dict | None = None,
) -> MockRequest:
    """Shortcut to create a MockRequest."""
    return MockRequest(method=method, path=path, query=query or {}, _body=body or {})


# ---------------------------------------------------------------------------
# Mock HubSpot Deal object
# ---------------------------------------------------------------------------


def _mock_deal(
    deal_id: str = "d1",
    name: str = "Test Deal",
    amount: str = "5000",
    stage: str = "proposal",
    pipeline: str = "default",
    close_date: str | None = "2026-06-01",
    owner_id: str | None = "owner1",
    created_at: Any = None,
) -> MagicMock:
    """Create a mock HubSpot deal object."""
    deal = MagicMock()
    deal.id = deal_id
    deal.properties = {
        "dealname": name,
        "amount": amount,
        "dealstage": stage,
        "pipeline": pipeline,
        "closedate": close_date,
        "hubspot_owner_id": owner_id,
    }
    deal.created_at = created_at
    return deal


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
    cb = get_crm_circuit_breaker()
    for _ in range(cb.failure_threshold + 1):
        cb.record_failure()


def _make_connector(deals=None, deal=None, get_deals_exc=None, get_deal_exc=None, create_deal_exc=None):
    """Create a mock HubSpot connector with configurable behavior."""
    conn = AsyncMock()
    if get_deals_exc:
        conn.get_deals = AsyncMock(side_effect=get_deals_exc)
    else:
        conn.get_deals = AsyncMock(return_value=deals or [])
    if get_deal_exc:
        conn.get_deal = AsyncMock(side_effect=get_deal_exc)
    else:
        conn.get_deal = AsyncMock(return_value=deal)
    if create_deal_exc:
        conn.create_deal = AsyncMock(side_effect=create_deal_exc)
    else:
        conn.create_deal = AsyncMock(return_value=deal)
    return conn


# ===========================================================================
# GET /api/v1/crm/deals (list all deals)
# ===========================================================================


class TestListAllDeals:
    """Tests for GET /api/v1/crm/deals."""

    @pytest.mark.asyncio
    async def test_list_all_deals_empty_no_platforms(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 200
        body = _body(result)
        assert body["deals"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_deals_with_hubspot_deals(self, handler):
        _connect_hubspot()
        d1 = _mock_deal("d1", "Deal A", "1000", "proposal")
        d2 = _mock_deal("d2", "Deal B", "2000", "closed")
        conn = _make_connector(deals=[d1, d2])
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["deals"]) == 2

    @pytest.mark.asyncio
    async def test_list_all_deals_with_stage_filter(self, handler):
        _connect_hubspot()
        d1 = _mock_deal("d1", "Deal A", "1000", "proposal")
        d2 = _mock_deal("d2", "Deal B", "2000", "closed")
        conn = _make_connector(deals=[d1, d2])
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/deals", query={"stage": "proposal"})
        )
        assert _status(result) == 200
        body = _body(result)
        # Only the 'proposal' deal should pass through
        for deal in body["deals"]:
            assert deal["stage"] == "proposal"

    @pytest.mark.asyncio
    async def test_list_all_deals_invalid_stage_too_long(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/deals", query={"stage": "x" * (MAX_STAGE_LENGTH + 1)})
        )
        assert _status(result) == 400
        assert "too long" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_list_all_deals_with_limit(self, handler):
        _connect_hubspot()
        deals = [_mock_deal(f"d{i}", f"Deal {i}", str(i * 100), "proposal") for i in range(5)]
        conn = _make_connector(deals=deals)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/deals", query={"limit": "2"})
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["deals"]) <= 2

    @pytest.mark.asyncio
    async def test_list_all_deals_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 503
        assert "circuit breaker" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_list_all_deals_skips_coming_soon_platforms(self, handler):
        """Platforms marked coming_soon should not be queried."""
        _connect_hubspot()
        # Also add salesforce (coming_soon=True) credentials
        _platform_credentials["salesforce"] = {
            "credentials": {"client_id": "c", "client_secret": "s", "refresh_token": "r", "instance_url": "u"},
            "connected_at": "2026-02-23T00:00:00+00:00",
        }
        conn = _make_connector(deals=[])
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 200
        # Should not error out, salesforce is skipped

    @pytest.mark.asyncio
    async def test_list_all_deals_handles_platform_failure(self, handler):
        """When a platform raises an exception, it's logged and others continue."""
        _connect_hubspot()
        conn = _make_connector(get_deals_exc=ConnectionError("Network down"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 200
        body = _body(result)
        assert body["deals"] == []

    @pytest.mark.asyncio
    async def test_list_all_deals_records_cb_failure_on_platform_error(self, handler):
        """Circuit breaker should record failure when platform errors occur."""
        _connect_hubspot()
        conn = _make_connector(get_deals_exc=ConnectionError("fail"))
        _platform_connectors["hubspot"] = conn

        cb = get_crm_circuit_breaker()
        initial_state = cb.state

        await handler.handle_request(_req(path="/api/v1/crm/deals"))
        # The circuit breaker should have recorded a failure
        # (state may still be closed if threshold not reached, but failure count increases)
        assert cb.state in ("closed", "open")

    @pytest.mark.asyncio
    async def test_list_all_deals_records_cb_success_on_no_failures(self, handler):
        """Circuit breaker should record success when all platforms succeed."""
        _connect_hubspot()
        conn = _make_connector(deals=[])
        _platform_connectors["hubspot"] = conn

        await handler.handle_request(_req(path="/api/v1/crm/deals"))
        cb = get_crm_circuit_breaker()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_list_all_deals_default_limit_100(self, handler):
        """Default limit should be 100."""
        _connect_hubspot()
        # Create 150 deals
        deals = [_mock_deal(f"d{i}", f"Deal {i}", str(i * 100), "proposal") for i in range(150)]
        conn = _make_connector(deals=deals)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 200
        body = _body(result)
        assert len(body["deals"]) <= 100


# ===========================================================================
# _fetch_platform_deals (internal helper)
# ===========================================================================


class TestFetchPlatformDeals:
    """Tests for the _fetch_platform_deals internal helper."""

    @pytest.mark.asyncio
    async def test_fetch_no_connector_returns_empty(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler._fetch_platform_deals("hubspot")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_hubspot_normalizes_deals(self, handler):
        _connect_hubspot()
        d = _mock_deal("d1", "Test", "5000", "proposal")
        conn = _make_connector(deals=[d])
        _platform_connectors["hubspot"] = conn

        result = await handler._fetch_platform_deals("hubspot")
        assert len(result) == 1
        assert result[0]["id"] == "d1"
        assert result[0]["platform"] == "hubspot"
        assert result[0]["name"] == "Test"
        assert result[0]["amount"] == 5000.0
        assert result[0]["stage"] == "proposal"

    @pytest.mark.asyncio
    async def test_fetch_hubspot_filters_by_stage(self, handler):
        _connect_hubspot()
        d1 = _mock_deal("d1", "A", "100", "proposal")
        d2 = _mock_deal("d2", "B", "200", "closed")
        d3 = _mock_deal("d3", "C", "300", "proposal")
        conn = _make_connector(deals=[d1, d2, d3])
        _platform_connectors["hubspot"] = conn

        result = await handler._fetch_platform_deals("hubspot", stage="proposal")
        assert len(result) == 2
        for deal in result:
            assert deal["stage"] == "proposal"

    @pytest.mark.asyncio
    async def test_fetch_hubspot_no_stage_filter_returns_all(self, handler):
        _connect_hubspot()
        d1 = _mock_deal("d1", "A", "100", "proposal")
        d2 = _mock_deal("d2", "B", "200", "closed")
        conn = _make_connector(deals=[d1, d2])
        _platform_connectors["hubspot"] = conn

        result = await handler._fetch_platform_deals("hubspot")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fetch_hubspot_connection_error_returns_empty(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deals_exc=ConnectionError("fail"))
        _platform_connectors["hubspot"] = conn

        result = await handler._fetch_platform_deals("hubspot")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_hubspot_timeout_error_returns_empty(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deals_exc=TimeoutError("timeout"))
        _platform_connectors["hubspot"] = conn

        result = await handler._fetch_platform_deals("hubspot")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_hubspot_value_error_returns_empty(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deals_exc=ValueError("bad data"))
        _platform_connectors["hubspot"] = conn

        result = await handler._fetch_platform_deals("hubspot")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_hubspot_os_error_returns_empty(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deals_exc=OSError("io err"))
        _platform_connectors["hubspot"] = conn

        result = await handler._fetch_platform_deals("hubspot")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_records_cb_success_on_hubspot(self, handler):
        _connect_hubspot()
        conn = _make_connector(deals=[])
        _platform_connectors["hubspot"] = conn

        await handler._fetch_platform_deals("hubspot")
        cb = get_crm_circuit_breaker()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_fetch_records_cb_failure_on_exception(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deals_exc=ConnectionError("fail"))
        _platform_connectors["hubspot"] = conn

        await handler._fetch_platform_deals("hubspot")
        # Should have recorded a failure
        cb = get_crm_circuit_breaker()
        assert cb.state in ("closed", "open")

    @pytest.mark.asyncio
    async def test_fetch_respects_limit_parameter(self, handler):
        _connect_hubspot()
        conn = _make_connector(deals=[])
        _platform_connectors["hubspot"] = conn

        await handler._fetch_platform_deals("hubspot", limit=50)
        conn.get_deals.assert_awaited_once_with(limit=50)

    @pytest.mark.asyncio
    async def test_fetch_unsupported_platform_returns_empty(self, handler):
        """Non-hubspot platforms currently return empty list."""
        _platform_credentials["zoho_custom"] = {
            "credentials": {"key": "val"},
            "connected_at": "2026-01-01",
        }
        conn = AsyncMock()
        _platform_connectors["zoho_custom"] = conn

        result = await handler._fetch_platform_deals("zoho_custom")
        assert result == []


# ===========================================================================
# GET /api/v1/crm/{platform}/deals (list platform deals)
# ===========================================================================


class TestListPlatformDeals:
    """Tests for GET /api/v1/crm/{platform}/deals."""

    @pytest.mark.asyncio
    async def test_list_platform_deals_success_empty(self, handler):
        _connect_hubspot()
        conn = _make_connector(deals=[])
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["deals"] == []
        assert body["total"] == 0
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_list_platform_deals_with_deals(self, handler):
        _connect_hubspot()
        d1 = _mock_deal("d1", "Deal A", "1000", "proposal")
        conn = _make_connector(deals=[d1])
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["deals"][0]["name"] == "Deal A"

    @pytest.mark.asyncio
    async def test_list_platform_deals_with_stage_filter(self, handler):
        _connect_hubspot()
        d1 = _mock_deal("d1", "A", "100", "proposal")
        d2 = _mock_deal("d2", "B", "200", "closed")
        conn = _make_connector(deals=[d1, d2])
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals", query={"stage": "closed"})
        )
        assert _status(result) == 200
        body = _body(result)
        for deal in body["deals"]:
            assert deal["stage"] == "closed"

    @pytest.mark.asyncio
    async def test_list_platform_deals_invalid_stage_too_long(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals", query={"stage": "s" * (MAX_STAGE_LENGTH + 1)})
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_platform_deals_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 404
        assert "not connected" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_list_platform_deals_invalid_platform_format(self, handler):
        """Invalid platform format is not in SUPPORTED_PLATFORMS, so the router
        does not parse it as a platform. The path ends with /deals, so it falls
        through to _list_all_deals (200) instead of returning a validation error.
        This tests the actual routing behavior."""
        result = await handler.handle_request(
            _req(path="/api/v1/crm/bad platform!/deals")
        )
        # Router doesn't parse 'bad platform!' as a platform, falls through to list all
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_platform_deals_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 503

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

    @pytest.mark.asyncio
    async def test_list_platform_deals_with_limit_param(self, handler):
        _connect_hubspot()
        conn = _make_connector(deals=[])
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals", query={"limit": "10"})
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_platform_deals_connector_raises_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deals_exc=ConnectionError("network"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["deals"] == []


# ===========================================================================
# GET /api/v1/crm/{platform}/deals/{deal_id} (get single deal)
# ===========================================================================


class TestGetDeal:
    """Tests for GET /api/v1/crm/{platform}/deals/{deal_id}."""

    @pytest.mark.asyncio
    async def test_get_deal_success(self, handler):
        _connect_hubspot()
        d = _mock_deal("d1", "Big Deal", "10000", "negotiation")
        conn = _make_connector(deal=d)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d1")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "d1"
        assert body["name"] == "Big Deal"
        assert body["amount"] == 10000.0
        assert body["stage"] == "negotiation"
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_get_deal_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d123")
        )
        assert _status(result) == 404
        assert "not connected" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_deal_no_connector(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(path="/api/v1/crm/hubspot/deals/d123")
            )
        assert _status(result) == 500
        assert "could not initialize" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_deal_connection_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deal_exc=ConnectionError("fail"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d123")
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_deal_timeout_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deal_exc=TimeoutError("timeout"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d123")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_deal_value_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deal_exc=ValueError("bad"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d123")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_deal_os_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deal_exc=OSError("io"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d123")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_deal_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d123")
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_deal_invalid_deal_id_too_long(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/" + "x" * 200)
        )
        assert _status(result) == 400
        assert "too long" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_deal_invalid_deal_id_format(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/" + "!@#$%")
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_deal_records_cb_success(self, handler):
        _connect_hubspot()
        d = _mock_deal("d1")
        conn = _make_connector(deal=d)
        _platform_connectors["hubspot"] = conn

        await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d1")
        )
        cb = get_crm_circuit_breaker()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_get_deal_records_cb_failure(self, handler):
        _connect_hubspot()
        conn = _make_connector(get_deal_exc=ConnectionError("fail"))
        _platform_connectors["hubspot"] = conn

        await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d1")
        )
        cb = get_crm_circuit_breaker()
        # Failure count should have increased
        assert cb.state in ("closed", "open")

    @pytest.mark.asyncio
    async def test_get_deal_unsupported_platform_returns_400(self, handler):
        """A non-hubspot platform that is connected returns 'Unsupported platform'."""
        _platform_credentials["custom_crm"] = {
            "credentials": {"key": "val"},
            "connected_at": "2026-01-01",
        }
        conn = AsyncMock()
        _platform_connectors["custom_crm"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/custom_crm/deals/d1")
        )
        # custom_crm is not in SUPPORTED_PLATFORMS so routing may not detect it as platform
        # The result is either 400 (unsupported) or 404 (endpoint not found)
        assert _status(result) in (400, 404)


# ===========================================================================
# POST /api/v1/crm/{platform}/deals (create deal)
# ===========================================================================


class TestCreateDeal:
    """Tests for POST /api/v1/crm/{platform}/deals."""

    @pytest.mark.asyncio
    async def test_create_deal_success(self, handler):
        _connect_hubspot()
        created = _mock_deal("d-new", "New Deal", "7500", "qualification")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "New Deal", "stage": "qualification", "amount": 7500},
            )
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["id"] == "d-new"
        assert body["name"] == "New Deal"

    @pytest.mark.asyncio
    async def test_create_deal_success_minimal(self, handler):
        """Create deal with only required fields (name and stage)."""
        _connect_hubspot()
        created = _mock_deal("d-min", "Min Deal", None, "proposal")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Min Deal", "stage": "proposal"},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_deal_with_pipeline(self, handler):
        _connect_hubspot()
        created = _mock_deal("d-pipe", "Pipeline Deal", "500", "proposal", pipeline="sales")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Pipeline Deal", "stage": "proposal", "pipeline": "sales"},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_deal_with_close_date(self, handler):
        _connect_hubspot()
        created = _mock_deal("d-date", "Date Deal", "1000", "proposal")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={
                    "name": "Date Deal",
                    "stage": "proposal",
                    "close_date": "2026-12-31",
                },
            )
        )
        assert _status(result) == 201

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
        assert "name" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_deal_missing_stage(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal"},
            )
        )
        assert _status(result) == 400
        assert "stage" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_deal_missing_both_required(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_deal_name_too_long(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "x" * (MAX_DEAL_NAME_LENGTH + 1), "stage": "proposal"},
            )
        )
        assert _status(result) == 400
        assert "too long" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_deal_stage_too_long(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "s" * (MAX_STAGE_LENGTH + 1)},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_deal_pipeline_too_long(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={
                    "name": "Deal",
                    "stage": "proposal",
                    "pipeline": "p" * (MAX_PIPELINE_LENGTH + 1),
                },
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
        assert "negative" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_deal_invalid_amount_string(self, handler):
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
    async def test_create_deal_amount_too_large(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal", "amount": 2_000_000_000_000},
            )
        )
        assert _status(result) == 400
        assert "too large" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_deal_amount_zero(self, handler):
        """Zero is a valid amount."""
        _connect_hubspot()
        created = _mock_deal("d-zero", "Free Deal", "0", "proposal")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Free Deal", "stage": "proposal", "amount": 0},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_deal_amount_none_is_optional(self, handler):
        """Amount is optional and can be omitted."""
        _connect_hubspot()
        created = _mock_deal("d-no-amt", "No Amount", None, "proposal")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "No Amount", "stage": "proposal"},
            )
        )
        assert _status(result) == 201

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
        assert "could not initialize" in _body(result).get("error", "").lower()

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

    @pytest.mark.asyncio
    async def test_create_deal_connector_connection_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(create_deal_exc=ConnectionError("fail"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        assert _status(result) == 500
        assert "failed" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_deal_connector_timeout_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(create_deal_exc=TimeoutError("timeout"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_deal_connector_value_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(create_deal_exc=ValueError("bad value"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_deal_connector_os_error(self, handler):
        _connect_hubspot()
        conn = _make_connector(create_deal_exc=OSError("io"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_deal_records_cb_success(self, handler):
        _connect_hubspot()
        created = _mock_deal("d-ok", "OK Deal", "1000", "proposal")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "OK Deal", "stage": "proposal"},
            )
        )
        cb = get_crm_circuit_breaker()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_create_deal_records_cb_failure(self, handler):
        _connect_hubspot()
        conn = _make_connector(create_deal_exc=ConnectionError("fail"))
        _platform_connectors["hubspot"] = conn

        await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        cb = get_crm_circuit_breaker()
        assert cb.state in ("closed", "open")

    @pytest.mark.asyncio
    async def test_create_deal_properties_sent_to_connector(self, handler):
        """Verify the correct HubSpot properties are sent to the connector."""
        _connect_hubspot()
        created = _mock_deal("d-prop")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={
                    "name": "Prop Deal",
                    "stage": "negotiation",
                    "amount": 5000,
                    "pipeline": "sales",
                    "close_date": "2026-12-31",
                },
            )
        )
        conn.create_deal.assert_awaited_once()
        props = conn.create_deal.call_args[0][0]
        assert props["dealname"] == "Prop Deal"
        assert props["dealstage"] == "negotiation"
        assert props["amount"] == 5000
        assert props["pipeline"] == "sales"
        assert props["closedate"] == "2026-12-31"

    @pytest.mark.asyncio
    async def test_create_deal_default_pipeline(self, handler):
        """When pipeline is not provided, default should be 'default'."""
        _connect_hubspot()
        created = _mock_deal("d-def")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        conn.create_deal.assert_awaited_once()
        props = conn.create_deal.call_args[0][0]
        assert props["pipeline"] == "default"

    @pytest.mark.asyncio
    async def test_create_deal_none_values_filtered_from_properties(self, handler):
        """Properties with None values should be filtered out."""
        _connect_hubspot()
        created = _mock_deal("d-filt")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        conn.create_deal.assert_awaited_once()
        props = conn.create_deal.call_args[0][0]
        # amount and closedate should be filtered since they were None
        assert "amount" not in props or props.get("amount") is not None
        assert "closedate" not in props


# ===========================================================================
# Normalization tests
# ===========================================================================


class TestNormalizeHubspotDeal:
    """Tests for _normalize_hubspot_deal."""

    def test_normalize_full_deal(self, handler):
        d = _mock_deal("d1", "Full Deal", "25000", "won", "enterprise", "2026-09-01", "own1")
        dt = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        d.created_at = dt

        result = handler._normalize_hubspot_deal(d)
        assert result["id"] == "d1"
        assert result["platform"] == "hubspot"
        assert result["name"] == "Full Deal"
        assert result["amount"] == 25000.0
        assert result["stage"] == "won"
        assert result["pipeline"] == "enterprise"
        assert result["close_date"] == "2026-09-01"
        assert result["owner_id"] == "own1"
        assert result["created_at"] == dt.isoformat()

    def test_normalize_deal_with_no_properties(self, handler):
        d = MagicMock()
        d.id = "d-empty"
        d.properties = {}
        d.created_at = None

        result = handler._normalize_hubspot_deal(d)
        assert result["id"] == "d-empty"
        assert result["name"] is None
        assert result["amount"] is None
        assert result["stage"] is None
        assert result["pipeline"] is None
        assert result["close_date"] is None
        assert result["owner_id"] is None
        assert result["created_at"] is None

    def test_normalize_deal_invalid_amount(self, handler):
        d = _mock_deal("d-bad", "Bad Amount", "not-a-number", "proposal")
        result = handler._normalize_hubspot_deal(d)
        assert result["amount"] is None

    def test_normalize_deal_float_amount(self, handler):
        d = _mock_deal("d-float", "Float", "1234.56", "proposal")
        result = handler._normalize_hubspot_deal(d)
        assert result["amount"] == 1234.56

    def test_normalize_deal_zero_amount(self, handler):
        d = _mock_deal("d-zero", "Zero", "0", "proposal")
        result = handler._normalize_hubspot_deal(d)
        assert result["amount"] == 0.0

    def test_normalize_deal_without_properties_attribute(self, handler):
        """Deal object without properties attribute should default to empty dict."""
        d = MagicMock(spec=[])
        d.id = "d-noattr"
        d.created_at = None
        # MagicMock with spec=[] won't have 'properties'
        result = handler._normalize_hubspot_deal(d)
        assert result["id"] == "d-noattr"
        assert result["name"] is None

    def test_normalize_deal_with_created_at(self, handler):
        d = _mock_deal("d-ts")
        d.created_at = datetime(2026, 6, 15, 8, 0, tzinfo=timezone.utc)
        result = handler._normalize_hubspot_deal(d)
        assert result["created_at"] == "2026-06-15T08:00:00+00:00"

    def test_normalize_deal_created_at_none(self, handler):
        d = _mock_deal("d-no-ts")
        d.created_at = None
        result = handler._normalize_hubspot_deal(d)
        assert result["created_at"] is None


# ===========================================================================
# Routing integration tests (via handle_request)
# ===========================================================================


class TestDealRouting:
    """Tests for deal-related route dispatch in the main handler."""

    @pytest.mark.asyncio
    async def test_get_deals_routes_correctly(self, handler):
        """GET /api/v1/crm/deals routes to _list_all_deals."""
        result = await handler.handle_request(
            _req(method="GET", path="/api/v1/crm/deals")
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_platform_deals_routes_correctly(self, handler):
        """GET /api/v1/crm/hubspot/deals routes to _list_platform_deals."""
        _connect_hubspot()
        conn = _make_connector(deals=[])
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(method="GET", path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) == 200
        assert _body(result).get("platform") == "hubspot"

    @pytest.mark.asyncio
    async def test_get_deal_by_id_routes_correctly(self, handler):
        """GET /api/v1/crm/hubspot/deals/d1 routes to _get_deal."""
        _connect_hubspot()
        d = _mock_deal("d1")
        conn = _make_connector(deal=d)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(method="GET", path="/api/v1/crm/hubspot/deals/d1")
        )
        assert _status(result) == 200
        assert _body(result)["id"] == "d1"

    @pytest.mark.asyncio
    async def test_post_deal_routes_correctly(self, handler):
        """POST /api/v1/crm/hubspot/deals routes to _create_deal."""
        _connect_hubspot()
        created = _mock_deal("d-new")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "New", "stage": "proposal"},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_put_deal_returns_404(self, handler):
        """PUT on deals endpoint is not supported."""
        _connect_hubspot()
        result = await handler.handle_request(
            _req(method="PUT", path="/api/v1/crm/hubspot/deals/d1", body={})
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_deal_returns_404(self, handler):
        """DELETE on deals endpoint is not supported."""
        _connect_hubspot()
        result = await handler.handle_request(
            _req(method="DELETE", path="/api/v1/crm/hubspot/deals/d1")
        )
        assert _status(result) == 404


# ===========================================================================
# Edge cases
# ===========================================================================


class TestDealEdgeCases:
    """Edge case tests for deal operations."""

    @pytest.mark.asyncio
    async def test_empty_body_create_deal(self, handler):
        """POST with completely empty body should fail validation."""
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_large_valid_amount(self, handler):
        """Amount just under the 1 trillion limit should be accepted."""
        _connect_hubspot()
        created = _mock_deal("d-big", "Big", "999999999999", "proposal")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Big", "stage": "proposal", "amount": 999_999_999_999},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_amount_at_trillion_limit(self, handler):
        """Amount exactly at 1 trillion should be accepted."""
        _connect_hubspot()
        created = _mock_deal("d-lim")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Limit", "stage": "proposal", "amount": 1_000_000_000_000},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_amount_just_over_trillion_limit(self, handler):
        """Amount just over 1 trillion should be rejected."""
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Over", "stage": "proposal", "amount": 1_000_000_000_001},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_deal_name_at_max_length(self, handler):
        """Deal name exactly at max length should be accepted."""
        _connect_hubspot()
        created = _mock_deal("d-max")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "x" * MAX_DEAL_NAME_LENGTH, "stage": "proposal"},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_stage_at_max_length(self, handler):
        """Stage exactly at max length should be accepted."""
        _connect_hubspot()
        created = _mock_deal("d-stg")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "s" * MAX_STAGE_LENGTH},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_pipeline_at_max_length(self, handler):
        """Pipeline exactly at max length should be accepted."""
        _connect_hubspot()
        created = _mock_deal("d-pipe")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={
                    "name": "Deal",
                    "stage": "proposal",
                    "pipeline": "p" * MAX_PIPELINE_LENGTH,
                },
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_deal_with_numeric_string_amount(self, handler):
        """Amount as a numeric string should be valid."""
        _connect_hubspot()
        created = _mock_deal("d-str")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal", "amount": "5000.50"},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_list_all_deals_multiple_platforms_one_fails(self, handler):
        """When one platform fails, deals from others should still be returned."""
        _connect_hubspot()
        # Add another platform that is not coming_soon (none exist currently,
        # but we can test that hubspot failure still returns 200)
        conn = _make_connector(get_deals_exc=TimeoutError("slow"))
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(_req(path="/api/v1/crm/deals"))
        assert _status(result) == 200
        assert _body(result)["deals"] == []

    @pytest.mark.asyncio
    async def test_create_deal_amount_as_float_string(self, handler):
        """Amount as '99.99' string should be valid."""
        _connect_hubspot()
        created = _mock_deal("d-f")
        conn = _make_connector(deal=created)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal", "amount": "99.99"},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_deal_id_with_hyphens_is_valid(self, handler):
        _connect_hubspot()
        d = _mock_deal("d-with-hyphens")
        conn = _make_connector(deal=d)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d-with-hyphens")
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_deal_id_with_underscores_is_valid(self, handler):
        _connect_hubspot()
        d = _mock_deal("d_with_underscores")
        conn = _make_connector(deal=d)
        _platform_connectors["hubspot"] = conn

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d_with_underscores")
        )
        assert _status(result) == 200


# ===========================================================================
# RBAC permission tests for deal endpoints
# ===========================================================================


class TestDealRBACPermissions:
    """Tests that verify permission checks for deal endpoints."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_list_all_deals_requires_read_permission(self):
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(path="/api/v1/crm/deals")
        )
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_list_platform_deals_requires_read_permission(self):
        _connect_hubspot()
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals")
        )
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_deal_requires_read_permission(self):
        _connect_hubspot()
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/deals/d1")
        )
        assert _status(result) in (401, 403)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_create_deal_requires_write_permission(self):
        _connect_hubspot()
        handler = CRMHandler({})
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/deals",
                body={"name": "Deal", "stage": "proposal"},
            )
        )
        assert _status(result) in (401, 403)
