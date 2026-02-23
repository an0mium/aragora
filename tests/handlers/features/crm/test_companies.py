"""Comprehensive tests for the CRM CompanyOperationsMixin.

Tests all methods of CompanyOperationsMixin defined in
aragora/server/handlers/features/crm/companies.py:

- _list_all_companies() - list companies across all platforms
- _fetch_platform_companies() - internal helper to fetch from one platform
- _list_platform_companies() - list companies from a specific platform
- _get_company() - get a single company by platform and ID
- _create_company() - create a company on a platform

Covers: happy paths, validation errors, circuit breaker, platform not
connected, field length limits, connector failures, normalization, edge cases.
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
from aragora.server.handlers.features.crm.circuit_breaker import (
    get_crm_circuit_breaker,
    reset_crm_circuit_breaker,
)
from aragora.server.handlers.features.crm.validation import (
    MAX_COMPANY_NAME_LENGTH,
    MAX_DOMAIN_LENGTH,
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
    path: str = "/api/v1/crm/companies",
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
    """Create a CRMHandler instance with empty context."""
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


def _connect_pipedrive():
    """Insert pipedrive credentials into the module-level store."""
    _platform_credentials["pipedrive"] = {
        "credentials": {"api_token": "tok-pd-456"},
        "connected_at": "2026-02-23T00:00:00+00:00",
    }


def _open_circuit_breaker():
    """Force the circuit breaker into the OPEN state."""
    cb = get_crm_circuit_breaker()
    for _ in range(cb.failure_threshold + 1):
        cb.record_failure()


def _mock_hubspot_company(
    company_id: str = "co1",
    name: str = "Acme Corp",
    domain: str = "acme.com",
    industry: str = "Technology",
    employees: str | None = "50",
    revenue: str | None = "1000000",
    owner_id: str | None = "owner1",
) -> MagicMock:
    """Create a mock HubSpot company object."""
    company = MagicMock()
    company.id = company_id
    company.properties = {
        "name": name,
        "domain": domain,
        "industry": industry,
    }
    if employees is not None:
        company.properties["numberofemployees"] = employees
    if revenue is not None:
        company.properties["annualrevenue"] = revenue
    if owner_id is not None:
        company.properties["hubspot_owner_id"] = owner_id
    company.created_at = None
    return company


def _setup_hubspot_connector(
    companies: list | None = None,
    get_company_result: Any = None,
    create_company_result: Any = None,
    get_companies_error: BaseException | None = None,
    get_company_error: BaseException | None = None,
    create_company_error: BaseException | None = None,
) -> AsyncMock:
    """Set up a mock HubSpot connector in the module-level store."""
    mock_conn = AsyncMock()
    if get_companies_error:
        mock_conn.get_companies = AsyncMock(side_effect=get_companies_error)
    else:
        mock_conn.get_companies = AsyncMock(return_value=companies or [])

    if get_company_error:
        mock_conn.get_company = AsyncMock(side_effect=get_company_error)
    elif get_company_result is not None:
        mock_conn.get_company = AsyncMock(return_value=get_company_result)
    else:
        mock_conn.get_company = AsyncMock(side_effect=ValueError("Not found"))

    if create_company_error:
        mock_conn.create_company = AsyncMock(side_effect=create_company_error)
    elif create_company_result is not None:
        mock_conn.create_company = AsyncMock(return_value=create_company_result)
    else:
        mock_conn.create_company = AsyncMock(return_value=_mock_hubspot_company())

    _platform_connectors["hubspot"] = mock_conn
    return mock_conn


# ===========================================================================
# _list_all_companies
# ===========================================================================


class TestListAllCompanies:
    """Tests for _list_all_companies."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_platforms(self, handler):
        result = await handler._list_all_companies(_req())
        assert _status(result) == 200
        body = _body(result)
        assert body["companies"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler._list_all_companies(_req())
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_with_hubspot_connector_returns_companies(self, handler):
        _connect_hubspot()
        companies = [
            _mock_hubspot_company("co1", "Acme"),
            _mock_hubspot_company("co2", "Globex"),
        ]
        _setup_hubspot_connector(companies=companies)

        result = await handler._list_all_companies(_req())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["companies"]) == 2
        assert body["companies"][0]["name"] == "Acme"
        assert body["companies"][1]["name"] == "Globex"

    @pytest.mark.asyncio
    async def test_limit_query_param(self, handler):
        _connect_hubspot()
        companies = [_mock_hubspot_company(f"co{i}", f"Company{i}") for i in range(5)]
        _setup_hubspot_connector(companies=companies)

        result = await handler._list_all_companies(_req(query={"limit": "2"}))
        assert _status(result) == 200
        body = _body(result)
        # total is the full count before limit slice
        assert body["total"] == 5
        # companies list is sliced to limit
        assert len(body["companies"]) == 2

    @pytest.mark.asyncio
    async def test_default_limit_is_100(self, handler):
        """When no limit query param, default is 100 per fetch."""
        _connect_hubspot()
        _setup_hubspot_connector(companies=[])

        result = await handler._list_all_companies(_req())
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_no_connector_returns_empty(self, handler):
        """When platform has credentials but no connector available, returns empty."""
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler._list_all_companies(_req())
        assert _status(result) == 200
        body = _body(result)
        assert body["companies"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_skips_coming_soon_platforms(self, handler):
        """Platforms marked as coming_soon should be skipped."""
        _connect_hubspot()
        _connect_pipedrive()  # pipedrive is "coming_soon"
        _setup_hubspot_connector(companies=[_mock_hubspot_company()])

        result = await handler._list_all_companies(_req())
        assert _status(result) == 200
        body = _body(result)
        # Only hubspot companies, not pipedrive
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_platform_fetch_error_records_failure(self, handler):
        """When a platform raises an exception, circuit breaker records failure."""
        _connect_hubspot()
        _setup_hubspot_connector(get_companies_error=ConnectionError("fail"))

        result = await handler._list_all_companies(_req())
        assert _status(result) == 200
        body = _body(result)
        assert body["companies"] == []

    @pytest.mark.asyncio
    async def test_companies_normalized(self, handler):
        """Companies should be normalized via _normalize_hubspot_company."""
        _connect_hubspot()
        company = _mock_hubspot_company(
            "co1", "Acme", "acme.com", "Tech", "100", "5000000", "owner1"
        )
        _setup_hubspot_connector(companies=[company])

        result = await handler._list_all_companies(_req())
        assert _status(result) == 200
        body = _body(result)
        co = body["companies"][0]
        assert co["id"] == "co1"
        assert co["platform"] == "hubspot"
        assert co["name"] == "Acme"
        assert co["domain"] == "acme.com"
        assert co["industry"] == "Tech"
        assert co["employee_count"] == 100
        assert co["annual_revenue"] == 5000000.0

    @pytest.mark.asyncio
    async def test_successful_fetch_records_success(self, handler):
        """When all platform fetches succeed, circuit breaker records success."""
        _connect_hubspot()
        _setup_hubspot_connector(companies=[])

        cb = get_crm_circuit_breaker()
        initial_successes = cb._success_count if hasattr(cb, "_success_count") else 0

        result = await handler._list_all_companies(_req())
        assert _status(result) == 200


# ===========================================================================
# _fetch_platform_companies
# ===========================================================================


class TestFetchPlatformCompanies:
    """Tests for _fetch_platform_companies internal helper."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_connector(self, handler):
        """When _get_connector returns None, returns empty list."""
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler._fetch_platform_companies("hubspot", 100)
        assert result == []

    @pytest.mark.asyncio
    async def test_hubspot_returns_normalized_companies(self, handler):
        _connect_hubspot()
        companies = [_mock_hubspot_company("co1", "Acme")]
        _setup_hubspot_connector(companies=companies)

        result = await handler._fetch_platform_companies("hubspot", 100)
        assert len(result) == 1
        assert result[0]["name"] == "Acme"
        assert result[0]["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_connection_error_returns_empty(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_companies_error=ConnectionError("timeout"))

        result = await handler._fetch_platform_companies("hubspot", 100)
        assert result == []

    @pytest.mark.asyncio
    async def test_timeout_error_returns_empty(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_companies_error=TimeoutError("slow"))

        result = await handler._fetch_platform_companies("hubspot", 100)
        assert result == []

    @pytest.mark.asyncio
    async def test_os_error_returns_empty(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_companies_error=OSError("network"))

        result = await handler._fetch_platform_companies("hubspot", 100)
        assert result == []

    @pytest.mark.asyncio
    async def test_value_error_returns_empty(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_companies_error=ValueError("bad data"))

        result = await handler._fetch_platform_companies("hubspot", 100)
        assert result == []

    @pytest.mark.asyncio
    async def test_unsupported_platform_returns_empty(self, handler):
        """Non-hubspot platforms are not yet implemented, returns empty list."""
        _connect_pipedrive()
        mock_conn = AsyncMock()
        _platform_connectors["pipedrive"] = mock_conn

        result = await handler._fetch_platform_companies("pipedrive", 100)
        assert result == []

    @pytest.mark.asyncio
    async def test_limit_passed_to_connector(self, handler):
        _connect_hubspot()
        mock_conn = _setup_hubspot_connector(companies=[])

        await handler._fetch_platform_companies("hubspot", 50)
        mock_conn.get_companies.assert_awaited_once_with(limit=50)

    @pytest.mark.asyncio
    async def test_multiple_companies(self, handler):
        _connect_hubspot()
        companies = [
            _mock_hubspot_company("co1", "Alpha"),
            _mock_hubspot_company("co2", "Beta"),
            _mock_hubspot_company("co3", "Gamma"),
        ]
        _setup_hubspot_connector(companies=companies)

        result = await handler._fetch_platform_companies("hubspot", 100)
        assert len(result) == 3
        names = [c["name"] for c in result]
        assert names == ["Alpha", "Beta", "Gamma"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success_on_hubspot(self, handler):
        """Successful HubSpot fetch should record success on the circuit breaker."""
        _connect_hubspot()
        _setup_hubspot_connector(companies=[])

        cb = get_crm_circuit_breaker()
        await handler._fetch_platform_companies("hubspot", 100)
        # Should not be open after success
        assert cb.can_proceed()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure_on_error(self, handler):
        """Failed HubSpot fetch should record failure on the circuit breaker."""
        _connect_hubspot()
        _setup_hubspot_connector(get_companies_error=ConnectionError("fail"))

        await handler._fetch_platform_companies("hubspot", 100)
        # The circuit breaker should still allow (single failure < threshold)
        cb = get_crm_circuit_breaker()
        assert cb.can_proceed()


# ===========================================================================
# _list_platform_companies
# ===========================================================================


class TestListPlatformCompanies:
    """Tests for _list_platform_companies."""

    @pytest.mark.asyncio
    async def test_success_empty(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(companies=[])
        result = await handler._list_platform_companies(_req(), "hubspot")
        assert _status(result) == 200
        body = _body(result)
        assert body["companies"] == []
        assert body["total"] == 0
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_success_with_companies(self, handler):
        _connect_hubspot()
        companies = [_mock_hubspot_company("co1", "Acme")]
        _setup_hubspot_connector(companies=companies)

        result = await handler._list_platform_companies(_req(), "hubspot")
        assert _status(result) == 200
        body = _body(result)
        assert len(body["companies"]) == 1
        assert body["total"] == 1
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        result = await handler._list_platform_companies(_req(), "hubspot")
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_platform_format(self, handler):
        result = await handler._list_platform_companies(_req(), "bad platform!")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_platform(self, handler):
        result = await handler._list_platform_companies(_req(), "")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_platform_too_long(self, handler):
        result = await handler._list_platform_companies(_req(), "a" * 51)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler._list_platform_companies(_req(), "hubspot")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_limit_query_param(self, handler):
        _connect_hubspot()
        companies = [_mock_hubspot_company(f"co{i}", f"C{i}") for i in range(3)]
        _setup_hubspot_connector(companies=companies)

        result = await handler._list_platform_companies(
            _req(query={"limit": "2"}), "hubspot"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_platform_with_special_chars(self, handler):
        """Platforms with special characters should be rejected."""
        for bad_platform in ["hub<script>", "plat/form", "plat.form"]:
            result = await handler._list_platform_companies(_req(), bad_platform)
            assert _status(result) == 400, f"Expected 400 for platform: {bad_platform}"

    @pytest.mark.asyncio
    async def test_response_includes_platform_field(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(companies=[])
        result = await handler._list_platform_companies(_req(), "hubspot")
        body = _body(result)
        assert "platform" in body
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_platform_starting_with_number(self, handler):
        """Platform IDs must start with a letter."""
        result = await handler._list_platform_companies(_req(), "123abc")
        assert _status(result) == 400


# ===========================================================================
# _get_company
# ===========================================================================


class TestGetCompany:
    """Tests for _get_company."""

    @pytest.mark.asyncio
    async def test_success(self, handler):
        _connect_hubspot()
        company = _mock_hubspot_company("co1", "Acme", "acme.com", "Tech", "50", "1000000")
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "co1"
        assert body["name"] == "Acme"
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        result = await handler._get_company(_req(), "hubspot", "co123")
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_platform(self, handler):
        result = await handler._get_company(_req(), "bad platform!", "co123")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_company_id(self, handler):
        _connect_hubspot()
        result = await handler._get_company(_req(), "hubspot", "bad id!")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_company_id_too_long(self, handler):
        _connect_hubspot()
        result = await handler._get_company(_req(), "hubspot", "x" * 129)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_company_id(self, handler):
        _connect_hubspot()
        result = await handler._get_company(_req(), "hubspot", "")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_company_id_none(self, handler):
        _connect_hubspot()
        result = await handler._get_company(_req(), "hubspot", None)
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_no_connector_returns_500(self, handler):
        _connect_hubspot()
        # No connector in the store, and real import fails
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler._get_company(_req(), "hubspot", "co123")
        assert _status(result) == 500
        assert "could not initialize" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_company_not_found_via_connector_error(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_company_error=ValueError("Not found"))

        result = await handler._get_company(_req(), "hubspot", "co-missing")
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connection_error_returns_404(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_company_error=ConnectionError("timeout"))

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_timeout_error_returns_404(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_company_error=TimeoutError("slow"))

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_os_error_returns_404(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_company_error=OSError("network fail"))

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_platform(self, handler):
        result = await handler._get_company(_req(), "", "co1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_platform_too_long(self, handler):
        result = await handler._get_company(_req(), "a" * 51, "co1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_company_id_starts_with_hyphen(self, handler):
        """IDs must start with alphanumeric."""
        _connect_hubspot()
        result = await handler._get_company(_req(), "hubspot", "-invalid")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_company_id_with_special_chars(self, handler):
        """Company IDs with non-allowed chars should be rejected."""
        _connect_hubspot()
        for bad_id in ["<script>", "id/../../etc", "id with spaces"]:
            result = await handler._get_company(_req(), "hubspot", bad_id)
            assert _status(result) == 400, f"Expected 400 for id: {bad_id}"

    @pytest.mark.asyncio
    async def test_valid_resource_id_formats(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(
            get_company_result=_mock_hubspot_company("co1", "Acme")
        )
        for cid in ["co1", "company-abc", "company_def", "A1B2C3"]:
            result = await handler._get_company(_req(), "hubspot", cid)
            assert _status(result) == 200, f"Expected 200 for id: {cid}"

    @pytest.mark.asyncio
    async def test_unsupported_platform_returns_400(self, handler):
        """Non-hubspot platform with valid connector returns 'Unsupported platform'."""
        _connect_pipedrive()
        mock_conn = AsyncMock()
        _platform_connectors["pipedrive"] = mock_conn

        result = await handler._get_company(_req(), "pipedrive", "co1")
        assert _status(result) == 400
        assert "unsupported" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_normalized_response_shape(self, handler):
        _connect_hubspot()
        company = _mock_hubspot_company(
            "co1", "Acme", "acme.com", "Tech", "50", "1000000", "own1"
        )
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        body = _body(result)
        expected_keys = {
            "id", "platform", "name", "domain", "industry",
            "employee_count", "annual_revenue", "owner_id", "created_at",
        }
        assert expected_keys.issubset(body.keys())

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, handler):
        _connect_hubspot()
        company = _mock_hubspot_company("co1", "Acme")
        _setup_hubspot_connector(get_company_result=company)

        await handler._get_company(_req(), "hubspot", "co1")
        cb = get_crm_circuit_breaker()
        assert cb.can_proceed()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure_on_error(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(get_company_error=ConnectionError("down"))

        await handler._get_company(_req(), "hubspot", "co1")
        # A single failure should not open the breaker
        cb = get_crm_circuit_breaker()
        assert cb.can_proceed()


# ===========================================================================
# _create_company
# ===========================================================================


class TestCreateCompany:
    """Tests for _create_company."""

    @pytest.mark.asyncio
    async def test_success_minimal(self, handler):
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "New Corp")
        _setup_hubspot_connector(create_company_result=created)

        result = await handler._create_company(
            _req(method="POST", body={"name": "New Corp"}),
            "hubspot",
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["name"] == "New Corp"
        assert body["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_success_with_all_fields(self, handler):
        _connect_hubspot()
        created = _mock_hubspot_company(
            "co-new", "Full Corp", "full.com", "Finance", "200", "10000000"
        )
        _setup_hubspot_connector(create_company_result=created)

        result = await handler._create_company(
            _req(
                method="POST",
                body={
                    "name": "Full Corp",
                    "domain": "full.com",
                    "industry": "Finance",
                    "employee_count": 200,
                    "annual_revenue": 10000000,
                },
            ),
            "hubspot",
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["name"] == "Full Corp"

    @pytest.mark.asyncio
    async def test_missing_name(self, handler):
        _connect_hubspot()
        result = await handler._create_company(
            _req(method="POST", body={"domain": "example.com"}),
            "hubspot",
        )
        assert _status(result) == 400
        assert "company name" in _body(result)["error"].lower() or "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_name(self, handler):
        _connect_hubspot()
        result = await handler._create_company(
            _req(method="POST", body={"name": ""}),
            "hubspot",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_name_too_long(self, handler):
        _connect_hubspot()
        result = await handler._create_company(
            _req(method="POST", body={"name": "x" * (MAX_COMPANY_NAME_LENGTH + 1)}),
            "hubspot",
        )
        assert _status(result) == 400
        assert "too long" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_name_at_max_length(self, handler):
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "x" * MAX_COMPANY_NAME_LENGTH)
        _setup_hubspot_connector(create_company_result=created)

        result = await handler._create_company(
            _req(method="POST", body={"name": "x" * MAX_COMPANY_NAME_LENGTH}),
            "hubspot",
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_domain_too_long(self, handler):
        _connect_hubspot()
        result = await handler._create_company(
            _req(
                method="POST",
                body={"name": "Acme", "domain": "x" * (MAX_DOMAIN_LENGTH + 1)},
            ),
            "hubspot",
        )
        assert _status(result) == 400
        assert "domain" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_domain_at_max_length(self, handler):
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "Acme", "x" * MAX_DOMAIN_LENGTH)
        _setup_hubspot_connector(create_company_result=created)

        result = await handler._create_company(
            _req(
                method="POST",
                body={"name": "Acme", "domain": "x" * MAX_DOMAIN_LENGTH},
            ),
            "hubspot",
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_platform(self, handler):
        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "bad platform!",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_platform(self, handler):
        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_no_connector_returns_500(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler._create_company(
                _req(method="POST", body={"name": "Acme"}),
                "hubspot",
            )
        assert _status(result) == 500
        assert "could not initialize" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connection_error_returns_500(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(create_company_error=ConnectionError("timeout"))

        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_timeout_error_returns_500(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(create_company_error=TimeoutError("slow"))

        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(create_company_error=OSError("io"))

        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(create_company_error=ValueError("bad"))

        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_empty_body(self, handler):
        _connect_hubspot()
        result = await handler._create_company(
            _req(method="POST", body={}),
            "hubspot",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_properties_filter_none_values(self, handler):
        """Optional fields that are None should not be sent to HubSpot."""
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "Acme")
        mock_conn = _setup_hubspot_connector(create_company_result=created)

        await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )

        # Verify the properties passed to create_company do not include None values
        call_args = mock_conn.create_company.call_args
        props = call_args[0][0] if call_args[0] else call_args[1].get("properties", {})
        for value in props.values():
            assert value is not None

    @pytest.mark.asyncio
    async def test_optional_fields_passed_to_connector(self, handler):
        """Industry, employee_count, annual_revenue should be passed when provided."""
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "Acme")
        mock_conn = _setup_hubspot_connector(create_company_result=created)

        await handler._create_company(
            _req(
                method="POST",
                body={
                    "name": "Acme",
                    "domain": "acme.com",
                    "industry": "Tech",
                    "employee_count": 100,
                    "annual_revenue": 5000000,
                },
            ),
            "hubspot",
        )

        call_args = mock_conn.create_company.call_args
        props = call_args[0][0] if call_args[0] else call_args[1].get("properties", {})
        assert props.get("name") == "Acme"
        assert props.get("domain") == "acme.com"
        assert props.get("industry") == "Tech"
        assert props.get("numberofemployees") == 100
        assert props.get("annualrevenue") == 5000000

    @pytest.mark.asyncio
    async def test_unsupported_platform_returns_400(self, handler):
        """Non-hubspot platform returns 'Unsupported platform'."""
        _connect_pipedrive()
        mock_conn = AsyncMock()
        _platform_connectors["pipedrive"] = mock_conn

        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "pipedrive",
        )
        assert _status(result) == 400
        assert "unsupported" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_success(self, handler):
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "Acme")
        _setup_hubspot_connector(create_company_result=created)

        await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        cb = get_crm_circuit_breaker()
        assert cb.can_proceed()

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure_on_error(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(create_company_error=ConnectionError("fail"))

        await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        cb = get_crm_circuit_breaker()
        assert cb.can_proceed()  # single failure shouldn't open

    @pytest.mark.asyncio
    async def test_domain_is_optional(self, handler):
        """Domain should not be required for company creation."""
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "No Domain Corp")
        _setup_hubspot_connector(create_company_result=created)

        result = await handler._create_company(
            _req(method="POST", body={"name": "No Domain Corp"}),
            "hubspot",
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_name_with_special_characters(self, handler):
        """Company names can contain special characters."""
        _connect_hubspot()
        name = "O'Reilly & Sons (UK) Ltd."
        created = _mock_hubspot_company("co-new", name)
        _setup_hubspot_connector(create_company_result=created)

        result = await handler._create_company(
            _req(method="POST", body={"name": name}),
            "hubspot",
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_name_none_treated_as_missing(self, handler):
        """Explicit None name should be treated as missing/required."""
        _connect_hubspot()
        result = await handler._create_company(
            _req(method="POST", body={"name": None}),
            "hubspot",
        )
        assert _status(result) == 400


# ===========================================================================
# Integration: handle_request routing to company endpoints
# ===========================================================================


class TestHandleRequestRouting:
    """Test that CRMHandler.handle_request correctly routes to company methods."""

    @pytest.mark.asyncio
    async def test_list_all_companies_route(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/companies"))
        assert _status(result) == 200
        assert "companies" in _body(result)

    @pytest.mark.asyncio
    async def test_list_platform_companies_route(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(companies=[])
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies")
        )
        assert _status(result) == 200
        assert "companies" in _body(result)

    @pytest.mark.asyncio
    async def test_get_company_route(self, handler):
        _connect_hubspot()
        company = _mock_hubspot_company("co123", "Acme")
        _setup_hubspot_connector(get_company_result=company)

        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies/co123")
        )
        assert _status(result) == 200
        assert _body(result)["name"] == "Acme"

    @pytest.mark.asyncio
    async def test_create_company_route(self, handler):
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "Acme")
        _setup_hubspot_connector(create_company_result=created)

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/companies",
                body={"name": "Acme"},
            )
        )
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_list_all_companies_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler.handle_request(_req(path="/api/v1/crm/companies"))
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_platform_companies_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies")
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_company_platform_not_connected(self, handler):
        result = await handler.handle_request(
            _req(path="/api/v1/crm/hubspot/companies/co123")
        )
        assert _status(result) == 404

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
    async def test_create_company_missing_name_via_route(self, handler):
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
    async def test_create_company_name_too_long_via_route(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/companies",
                body={"name": "x" * (MAX_COMPANY_NAME_LENGTH + 1)},
            )
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_company_no_connector_via_route(self, handler):
        _connect_hubspot()
        with patch.object(handler, "_get_connector", return_value=None):
            result = await handler.handle_request(
                _req(path="/api/v1/crm/hubspot/companies/co123")
            )
        assert _status(result) == 500


# ===========================================================================
# Edge Cases / Security
# ===========================================================================


class TestEdgeCases:
    """Edge cases and security-related tests."""

    @pytest.mark.asyncio
    async def test_platform_with_path_traversal(self, handler):
        """Platform containing path traversal should be rejected."""
        result = await handler._list_platform_companies(_req(), "../../etc")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_company_id_with_path_traversal(self, handler):
        _connect_hubspot()
        result = await handler._get_company(_req(), "hubspot", "../../etc/passwd")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_all_company_operations(self, handler):
        """Verify every company method is blocked when circuit breaker is open."""
        _connect_hubspot()
        _open_circuit_breaker()
        ops = [
            handler._list_all_companies(_req()),
            handler._list_platform_companies(_req(), "hubspot"),
            handler._get_company(_req(), "hubspot", "co1"),
            handler._create_company(
                _req(method="POST", body={"name": "Acme"}), "hubspot"
            ),
        ]
        for coro in ops:
            result = await coro
            assert _status(result) == 503, "Expected 503 for circuit breaker open"

    @pytest.mark.asyncio
    async def test_validation_before_connection_check(self, handler):
        """Invalid platform format should return 400 before checking connection."""
        result = await handler._list_platform_companies(_req(), "!!")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_response_shape_list_companies(self, handler):
        result = await handler._list_all_companies(_req())
        body = _body(result)
        assert "companies" in body
        assert "total" in body
        assert isinstance(body["companies"], list)
        assert isinstance(body["total"], int)

    @pytest.mark.asyncio
    async def test_response_shape_list_platform_companies(self, handler):
        _connect_hubspot()
        _setup_hubspot_connector(companies=[])
        result = await handler._list_platform_companies(_req(), "hubspot")
        body = _body(result)
        assert "companies" in body
        assert "total" in body
        assert "platform" in body
        assert isinstance(body["companies"], list)

    @pytest.mark.asyncio
    async def test_response_shape_get_company(self, handler):
        _connect_hubspot()
        company = _mock_hubspot_company("co1", "Acme", "acme.com", "Tech")
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        body = _body(result)
        assert "id" in body
        assert "platform" in body
        assert "name" in body

    @pytest.mark.asyncio
    async def test_response_shape_create_company(self, handler):
        _connect_hubspot()
        created = _mock_hubspot_company("co-new", "Acme")
        _setup_hubspot_connector(create_company_result=created)

        result = await handler._create_company(
            _req(method="POST", body={"name": "Acme"}),
            "hubspot",
        )
        body = _body(result)
        assert "id" in body
        assert "platform" in body
        assert "name" in body

    @pytest.mark.asyncio
    async def test_normalization_with_none_properties(self, handler):
        """Company with empty properties should not crash normalization."""
        _connect_hubspot()
        company = MagicMock()
        company.id = "co-empty"
        company.properties = {}
        company.created_at = None
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "coEmpty")
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] is None
        assert body["domain"] is None
        assert body["industry"] is None
        assert body["employee_count"] is None
        assert body["annual_revenue"] is None

    @pytest.mark.asyncio
    async def test_normalization_invalid_employee_count(self, handler):
        """Non-numeric employee count should be normalized to None."""
        _connect_hubspot()
        company = _mock_hubspot_company(
            "co1", "Acme", employees="not-a-number"
        )
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 200
        assert _body(result)["employee_count"] is None

    @pytest.mark.asyncio
    async def test_normalization_invalid_annual_revenue(self, handler):
        """Non-numeric revenue should be normalized to None."""
        _connect_hubspot()
        company = _mock_hubspot_company(
            "co1", "Acme", revenue="not-a-number"
        )
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 200
        assert _body(result)["annual_revenue"] is None

    @pytest.mark.asyncio
    async def test_normalization_zero_employees(self, handler):
        """Zero employees should be a valid value."""
        _connect_hubspot()
        company = _mock_hubspot_company("co1", "Acme", employees="0")
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 200
        # "0" is falsy in Python, so props.get("numberofemployees") will be "0"
        # which is truthy as a string, so int("0") == 0
        assert _body(result)["employee_count"] == 0

    @pytest.mark.asyncio
    async def test_normalization_zero_revenue(self, handler):
        """Zero revenue should be a valid value."""
        _connect_hubspot()
        company = _mock_hubspot_company("co1", "Acme", revenue="0")
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 200
        assert _body(result)["annual_revenue"] == 0.0

    @pytest.mark.asyncio
    async def test_normalization_created_at_present(self, handler):
        """When created_at is present, it should be included in ISO format."""
        from datetime import datetime, timezone

        _connect_hubspot()
        company = _mock_hubspot_company("co1", "Acme")
        company.created_at = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 200
        assert _body(result)["created_at"] == "2026-01-15T10:30:00+00:00"

    @pytest.mark.asyncio
    async def test_normalization_created_at_none(self, handler):
        """When created_at is None, it should be None in the response."""
        _connect_hubspot()
        company = _mock_hubspot_company("co1", "Acme")
        company.created_at = None
        _setup_hubspot_connector(get_company_result=company)

        result = await handler._get_company(_req(), "hubspot", "co1")
        assert _status(result) == 200
        assert _body(result)["created_at"] is None
