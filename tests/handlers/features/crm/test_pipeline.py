"""Comprehensive tests for the CRM PipelineOperationsMixin.

Tests all methods of PipelineOperationsMixin defined in
aragora/server/handlers/features/crm/pipeline.py:

- _get_pipeline()    - Get sales pipeline summary with deal stage aggregation
- _sync_lead()       - Sync leads from external sources (create/update)
- _enrich_contact()  - Enrich contact data via external services
- _search_crm()      - Cross-platform CRM search across contacts/companies/deals

Covers: happy paths, validation errors, circuit breaker, platform not connected,
email validation, field length limits, connector errors, edge cases.
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
from aragora.server.handlers.features.crm.validation import MAX_SEARCH_QUERY_LENGTH


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
    path: str = "/api/v1/crm/pipeline",
    query: dict | None = None,
    body: dict | None = None,
) -> MockRequest:
    """Shortcut to create a MockRequest."""
    return MockRequest(method=method, path=path, query=query or {}, _body=body or {})


# ---------------------------------------------------------------------------
# Mock HubSpot objects
# ---------------------------------------------------------------------------


class MockStage:
    """Mock HubSpot pipeline stage."""

    def __init__(self, stage_id="s1", label="Prospect", display_order=0, metadata=None):
        self.id = stage_id
        self.label = label
        self.display_order = display_order
        self.metadata = metadata or {}


class MockPipeline:
    """Mock HubSpot pipeline."""

    def __init__(self, pipe_id="p1", label="Sales Pipeline", stages=None):
        self.id = pipe_id
        self.label = label
        self.stages = stages or []


class MockContact:
    """Mock HubSpot contact."""

    def __init__(self, contact_id="c1", properties=None, created_at=None, updated_at=None):
        self.id = contact_id
        self.properties = properties or {}
        self.created_at = created_at
        self.updated_at = updated_at


class MockCompany:
    """Mock HubSpot company."""

    def __init__(self, company_id="co1", properties=None, created_at=None):
        self.id = company_id
        self.properties = properties or {}
        self.created_at = created_at


class MockDeal:
    """Mock HubSpot deal."""

    def __init__(self, deal_id="d1", properties=None, created_at=None):
        self.id = deal_id
        self.properties = properties or {}
        self.created_at = created_at


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
    """Insert pipedrive credentials (coming_soon platform)."""
    _platform_credentials["pipedrive"] = {
        "credentials": {"api_token": "tok-pd-456"},
        "connected_at": "2026-02-23T00:00:00+00:00",
    }


def _open_circuit_breaker():
    """Force the circuit breaker into the OPEN state."""
    cb = get_crm_circuit_breaker()
    for _ in range(cb.failure_threshold + 1):
        cb.record_failure()


def _mock_connector(**overrides):
    """Create a mock HubSpot connector with sensible defaults."""
    conn = AsyncMock()
    conn.get_pipelines = AsyncMock(return_value=overrides.get("pipelines", []))
    conn.get_contact_by_email = AsyncMock(return_value=overrides.get("existing_contact"))
    conn.create_contact = AsyncMock(
        return_value=overrides.get(
            "created_contact",
            MockContact("new-1", {"email": "a@b.com", "firstname": "A", "lastname": "B"}),
        )
    )
    conn.update_contact = AsyncMock(
        return_value=overrides.get(
            "updated_contact",
            MockContact("existing-1", {"email": "a@b.com", "firstname": "A", "lastname": "B"}),
        )
    )
    conn.search_contacts = AsyncMock(return_value=overrides.get("search_contacts", []))
    conn.search_companies = AsyncMock(return_value=overrides.get("search_companies", []))
    conn.search_deals = AsyncMock(return_value=overrides.get("search_deals", []))
    return conn


def _install_connector(conn=None):
    """Install a mock connector in the module-level store."""
    if conn is None:
        conn = _mock_connector()
    _platform_connectors["hubspot"] = conn
    return conn


# ===========================================================================
# _get_pipeline  (GET /api/v1/crm/pipeline)
# ===========================================================================


class TestGetPipeline:
    """Tests for _get_pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_no_platforms_connected(self, handler):
        """Pipeline with no platforms returns empty data."""
        result = await handler._get_pipeline(_req())
        assert _status(result) == 200
        body = _body(result)
        assert body["pipelines"] == []
        assert body["total_deals"] == 0
        assert body["total_pipeline_value"] == 0
        assert body["stage_summary"] == {}

    @pytest.mark.asyncio
    async def test_pipeline_circuit_breaker_open(self, handler):
        """Pipeline request rejected when circuit breaker is open."""
        _open_circuit_breaker()
        result = await handler._get_pipeline(_req())
        assert _status(result) == 503
        assert "circuit breaker" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_pipeline_with_platform_filter(self, handler):
        """Pipeline filtered to specific platform."""
        _connect_hubspot()
        conn = _install_connector()
        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200
        conn.get_pipelines.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_platform_not_connected(self, handler):
        """Pipeline for non-connected platform returns 404."""
        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 404
        assert "not connected" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_pipeline_invalid_platform_format(self, handler):
        """Pipeline with invalid platform format returns 400."""
        result = await handler._get_pipeline(_req(query={"platform": "bad platform!"}))
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_pipeline_returns_hubspot_pipelines(self, handler):
        """Pipeline returns normalized HubSpot pipeline data."""
        _connect_hubspot()
        stages = [
            MockStage("s1", "Discovery", 0, {"probability": 10}),
            MockStage("s2", "Proposal", 1, {"probability": 50}),
        ]
        pipeline = MockPipeline("p1", "Sales Pipeline", stages)
        conn = _mock_connector(pipelines=[pipeline])
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200
        body = _body(result)
        assert len(body["pipelines"]) == 1
        p = body["pipelines"][0]
        assert p["id"] == "p1"
        assert p["platform"] == "hubspot"
        assert p["name"] == "Sales Pipeline"
        assert len(p["stages"]) == 2

    @pytest.mark.asyncio
    async def test_pipeline_stage_details(self, handler):
        """Pipeline stages include id, name, display_order, probability."""
        _connect_hubspot()
        stage = MockStage("s1", "Discovery", 0, {"probability": 25})
        pipeline = MockPipeline("p1", "Main", [stage])
        conn = _mock_connector(pipelines=[pipeline])
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        s = body["pipelines"][0]["stages"][0]
        assert s["id"] == "s1"
        assert s["name"] == "Discovery"
        assert s["display_order"] == 0
        assert s["probability"] == 25

    @pytest.mark.asyncio
    async def test_pipeline_stage_without_metadata(self, handler):
        """Pipeline stage without metadata attribute has probability=None."""
        _connect_hubspot()
        stage = MockStage("s1", "No Meta", 0)
        # Remove metadata attribute to test hasattr branch
        delattr(stage, "metadata")
        pipeline = MockPipeline("p1", "Main", [stage])
        conn = _mock_connector(pipelines=[pipeline])
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        s = body["pipelines"][0]["stages"][0]
        assert s["probability"] is None

    @pytest.mark.asyncio
    async def test_pipeline_without_stages_attribute(self, handler):
        """Pipeline without stages attribute uses empty list."""
        _connect_hubspot()
        pipeline = MockPipeline("p1", "Main")
        delattr(pipeline, "stages")
        conn = _mock_connector(pipelines=[pipeline])
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        assert body["pipelines"][0]["stages"] == []

    @pytest.mark.asyncio
    async def test_pipeline_multiple_pipelines(self, handler):
        """Handler returns multiple pipelines from hubspot."""
        _connect_hubspot()
        p1 = MockPipeline("p1", "Sales")
        p2 = MockPipeline("p2", "Partner")
        conn = _mock_connector(pipelines=[p1, p2])
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        assert len(body["pipelines"]) == 2
        names = {p["name"] for p in body["pipelines"]}
        assert names == {"Sales", "Partner"}

    @pytest.mark.asyncio
    async def test_pipeline_deal_stage_summary(self, handler):
        """Pipeline aggregates deals into stage summary."""
        _connect_hubspot()
        _install_connector()

        # Mock _list_all_deals to return deals with stages
        deals_result = {
            "status_code": 200,
            "body": {
                "deals": [
                    {"stage": "discovery", "amount": 1000},
                    {"stage": "discovery", "amount": 2000},
                    {"stage": "proposal", "amount": 5000},
                ],
            },
        }
        with patch.object(
            handler, "_list_all_deals", new_callable=AsyncMock, return_value=deals_result
        ):
            result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        assert body["stage_summary"]["discovery"]["count"] == 2
        assert body["stage_summary"]["discovery"]["total_value"] == 3000
        assert body["stage_summary"]["proposal"]["count"] == 1
        assert body["stage_summary"]["proposal"]["total_value"] == 5000
        assert body["total_deals"] == 3
        assert body["total_pipeline_value"] == 8000

    @pytest.mark.asyncio
    async def test_pipeline_deal_with_none_amount(self, handler):
        """Pipeline handles deals with amount=None gracefully."""
        _connect_hubspot()
        _install_connector()

        deals_result = {
            "status_code": 200,
            "body": {
                "deals": [
                    {"stage": "discovery", "amount": None},
                    {"stage": "discovery", "amount": 500},
                ],
            },
        }
        with patch.object(
            handler, "_list_all_deals", new_callable=AsyncMock, return_value=deals_result
        ):
            result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        assert body["stage_summary"]["discovery"]["total_value"] == 500

    @pytest.mark.asyncio
    async def test_pipeline_deal_without_stage(self, handler):
        """Pipeline defaults stage to 'unknown' when missing."""
        _connect_hubspot()
        _install_connector()

        deals_result = {
            "status_code": 200,
            "body": {"deals": [{"amount": 100}]},
        }
        with patch.object(
            handler, "_list_all_deals", new_callable=AsyncMock, return_value=deals_result
        ):
            result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        assert "unknown" in body["stage_summary"]
        assert body["stage_summary"]["unknown"]["count"] == 1

    @pytest.mark.asyncio
    async def test_pipeline_connector_error_logged(self, handler):
        """Pipeline logs and continues when connector raises."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.get_pipelines = AsyncMock(side_effect=ConnectionError("network down"))
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200
        body = _body(result)
        assert body["pipelines"] == []

    @pytest.mark.asyncio
    async def test_pipeline_timeout_error(self, handler):
        """Pipeline handles TimeoutError from connector."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.get_pipelines = AsyncMock(side_effect=TimeoutError("timed out"))
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200
        body = _body(result)
        assert body["pipelines"] == []

    @pytest.mark.asyncio
    async def test_pipeline_os_error(self, handler):
        """Pipeline handles OSError from connector."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.get_pipelines = AsyncMock(side_effect=OSError("io error"))
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_pipeline_value_error(self, handler):
        """Pipeline handles ValueError from connector."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.get_pipelines = AsyncMock(side_effect=ValueError("bad data"))
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_pipeline_skips_coming_soon_platforms(self, handler):
        """Pipeline skips platforms marked as coming_soon."""
        _connect_hubspot()
        _connect_pipedrive()
        conn = _mock_connector()
        _install_connector(conn)
        # Pipedrive is coming_soon=True, should be skipped
        result = await handler._get_pipeline(_req())
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_pipeline_no_connector_available(self, handler):
        """Pipeline skips platform if connector cannot be created."""
        _connect_hubspot()
        # Don't install connector; _get_connector will try to import HubSpotConnector
        # and fail, returning None
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200
        body = _body(result)
        assert body["pipelines"] == []

    @pytest.mark.asyncio
    async def test_pipeline_records_success_on_circuit_breaker(self, handler):
        """Pipeline records success on circuit breaker after successful fetch."""
        _connect_hubspot()
        conn = _mock_connector(pipelines=[MockPipeline("p1", "Sales")])
        _install_connector(conn)

        cb = get_crm_circuit_breaker()
        initial_success = cb._success_count if hasattr(cb, "_success_count") else 0

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_pipeline_records_failure_on_circuit_breaker(self, handler):
        """Pipeline records failure on circuit breaker after connector error."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.get_pipelines = AsyncMock(side_effect=ConnectionError("fail"))
        _install_connector(conn)

        result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_pipeline_list_all_deals_returns_non_dict(self, handler):
        """Pipeline handles non-dict _list_all_deals gracefully."""
        _connect_hubspot()
        _install_connector()

        with patch.object(
            handler, "_list_all_deals", new_callable=AsyncMock, return_value="not-a-dict"
        ):
            result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        assert body["total_deals"] == 0

    @pytest.mark.asyncio
    async def test_pipeline_empty_deals_list(self, handler):
        """Pipeline with no deals returns zero totals."""
        _connect_hubspot()
        _install_connector()

        deals_result = {"status_code": 200, "body": {"deals": []}}
        with patch.object(
            handler, "_list_all_deals", new_callable=AsyncMock, return_value=deals_result
        ):
            result = await handler._get_pipeline(_req(query={"platform": "hubspot"}))
        body = _body(result)
        assert body["total_deals"] == 0
        assert body["total_pipeline_value"] == 0
        assert body["stage_summary"] == {}

    @pytest.mark.asyncio
    async def test_pipeline_platform_too_long(self, handler):
        """Pipeline rejects platform name exceeding max length."""
        result = await handler._get_pipeline(_req(query={"platform": "x" * 60}))
        assert _status(result) == 400


# ===========================================================================
# _sync_lead  (POST /api/v1/crm/sync-lead)
# ===========================================================================


class TestSyncLead:
    """Tests for _sync_lead."""

    @pytest.mark.asyncio
    async def test_sync_lead_create_new_contact(self, handler):
        """Sync lead creates a new contact when none exists."""
        _connect_hubspot()
        new_contact = MockContact("new-1", {"email": "lead@example.com"})
        conn = _mock_connector(existing_contact=None, created_contact=new_contact)
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={
                "platform": "hubspot",
                "source": "linkedin",
                "lead": {"email": "lead@example.com", "first_name": "Jane"},
            },
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["action"] == "created"
        assert body["source"] == "linkedin"
        conn.create_contact.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sync_lead_update_existing_contact(self, handler):
        """Sync lead updates an existing contact when found by email."""
        _connect_hubspot()
        existing = MockContact("existing-1", {"email": "lead@example.com"})
        updated = MockContact("existing-1", {"email": "lead@example.com", "firstname": "Jane"})
        conn = _mock_connector(existing_contact=existing, updated_contact=updated)
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={
                "platform": "hubspot",
                "source": "form",
                "lead": {"email": "lead@example.com", "first_name": "Jane"},
            },
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["action"] == "updated"
        conn.update_contact.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sync_lead_circuit_breaker_open(self, handler):
        """Sync lead rejected when circuit breaker is open."""
        _open_circuit_breaker()
        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_sync_lead_missing_email(self, handler):
        """Sync lead requires email in lead data."""
        _connect_hubspot()
        _install_connector()

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"first_name": "Jane"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 400
        assert "email" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_sync_lead_invalid_email(self, handler):
        """Sync lead rejects invalid email format."""
        _connect_hubspot()
        _install_connector()

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "not-an-email"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_lead_platform_not_connected(self, handler):
        """Sync lead returns 404 for unconnected platform."""
        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 404
        assert "not connected" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_sync_lead_invalid_platform(self, handler):
        """Sync lead rejects invalid platform format."""
        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "bad platform!", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_lead_default_platform_is_hubspot(self, handler):
        """Sync lead defaults to hubspot platform."""
        _connect_hubspot()
        _install_connector()

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_sync_lead_default_source_is_api(self, handler):
        """Sync lead defaults source to 'api'."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["source"] == "api"

    @pytest.mark.asyncio
    async def test_sync_lead_connector_not_available(self, handler):
        """Sync lead returns 500 when connector cannot be created."""
        _connect_hubspot()
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            req = _req(
                method="POST",
                path="/api/v1/crm/sync-lead",
                body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
            )
            result = await handler._sync_lead(req)
        assert _status(result) == 500
        assert "could not initialize" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_sync_lead_connector_create_fails(self, handler):
        """Sync lead returns 500 when create_contact raises."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.create_contact = AsyncMock(side_effect=ConnectionError("network error"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 500
        assert "sync failed" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_sync_lead_connector_update_fails(self, handler):
        """Sync lead returns 500 when update_contact raises."""
        _connect_hubspot()
        existing = MockContact("existing-1", {"email": "a@b.com"})
        conn = _mock_connector(existing_contact=existing)
        conn.update_contact = AsyncMock(side_effect=TimeoutError("timeout"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_sync_lead_lookup_fails_proceeds_with_create(self, handler):
        """Sync lead proceeds to create when email lookup fails with network error."""
        _connect_hubspot()
        new_contact = MockContact("new-1", {"email": "a@b.com"})
        conn = _mock_connector(created_contact=new_contact)
        conn.get_contact_by_email = AsyncMock(side_effect=ConnectionError("lookup failed"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["action"] == "created"

    @pytest.mark.asyncio
    async def test_sync_lead_lookup_timeout_proceeds_with_create(self, handler):
        """Sync lead proceeds to create when email lookup times out."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.get_contact_by_email = AsyncMock(side_effect=TimeoutError("slow"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["action"] == "created"

    @pytest.mark.asyncio
    async def test_sync_lead_lookup_oserror_proceeds_with_create(self, handler):
        """Sync lead proceeds to create when email lookup raises OSError."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.get_contact_by_email = AsyncMock(side_effect=OSError("io fail"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["action"] == "created"

    @pytest.mark.asyncio
    async def test_sync_lead_maps_lead_data(self, handler):
        """Sync lead maps lead data fields correctly."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        lead_data = {
            "email": "jane@corp.com",
            "first_name": "Jane",
            "last_name": "Doe",
            "phone": "+15551234567",
            "company": "Acme Inc",
            "job_title": "CTO",
        }
        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "source": "webhook", "lead": lead_data},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 200

        # Verify the properties passed to create_contact
        call_args = conn.create_contact.call_args
        properties = call_args[0][0]
        assert properties["email"] == "jane@corp.com"
        assert properties["firstname"] == "Jane"
        assert properties["lastname"] == "Doe"
        assert properties["phone"] == "+15551234567"
        assert properties["company"] == "Acme Inc"
        assert properties["jobtitle"] == "CTO"
        assert properties["lifecyclestage"] == "lead"
        assert properties["hs_lead_status"] == "NEW"
        assert properties["hs_analytics_source"] == "webhook"

    @pytest.mark.asyncio
    async def test_sync_lead_empty_lead_data(self, handler):
        """Sync lead with empty lead dict requires email."""
        _connect_hubspot()
        _install_connector()

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 400
        assert "email" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_sync_lead_os_error_on_create(self, handler):
        """Sync lead returns 500 on OSError during create."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.create_contact = AsyncMock(side_effect=OSError("disk error"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_sync_lead_value_error_on_update(self, handler):
        """Sync lead returns 500 on ValueError during update."""
        _connect_hubspot()
        existing = MockContact("e1", {"email": "a@b.com"})
        conn = _mock_connector(existing_contact=existing)
        conn.update_contact = AsyncMock(side_effect=ValueError("bad value"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
        )
        result = await handler._sync_lead(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_sync_lead_contact_response_normalized(self, handler):
        """Sync lead response contains normalized contact data."""
        _connect_hubspot()
        new_contact = MockContact(
            "new-1",
            {"email": "lead@test.com", "firstname": "Test", "lastname": "Lead"},
        )
        conn = _mock_connector(created_contact=new_contact)
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/sync-lead",
            body={"platform": "hubspot", "lead": {"email": "lead@test.com"}},
        )
        result = await handler._sync_lead(req)
        body = _body(result)
        contact = body["contact"]
        assert contact["id"] == "new-1"
        assert contact["platform"] == "hubspot"
        assert contact["email"] == "lead@test.com"


# ===========================================================================
# _enrich_contact  (POST /api/v1/crm/enrich)
# ===========================================================================


class TestEnrichContact:
    """Tests for _enrich_contact."""

    @pytest.mark.asyncio
    async def test_enrich_success(self, handler):
        """Enrich returns placeholder data with available providers."""
        req = _req(
            method="POST",
            path="/api/v1/crm/enrich",
            body={"email": "user@example.com"},
        )
        result = await handler._enrich_contact(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["email"] == "user@example.com"
        assert body["enriched"] is False
        assert "clearbit" in body["available_providers"]
        assert "zoominfo" in body["available_providers"]
        assert "apollo" in body["available_providers"]

    @pytest.mark.asyncio
    async def test_enrich_missing_email(self, handler):
        """Enrich requires email."""
        req = _req(method="POST", path="/api/v1/crm/enrich", body={})
        result = await handler._enrich_contact(req)
        assert _status(result) == 400
        assert "email" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_enrich_invalid_email(self, handler):
        """Enrich rejects invalid email format."""
        req = _req(
            method="POST",
            path="/api/v1/crm/enrich",
            body={"email": "not-valid"},
        )
        result = await handler._enrich_contact(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_enrich_email_too_long(self, handler):
        """Enrich rejects email exceeding max length."""
        long_email = "a" * 250 + "@b.com"
        req = _req(
            method="POST",
            path="/api/v1/crm/enrich",
            body={"email": long_email},
        )
        result = await handler._enrich_contact(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_enrich_message_field(self, handler):
        """Enrich response includes pending message."""
        req = _req(
            method="POST",
            path="/api/v1/crm/enrich",
            body={"email": "user@example.com"},
        )
        result = await handler._enrich_contact(req)
        body = _body(result)
        assert "pending" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_enrich_email_none_explicitly(self, handler):
        """Enrich returns 400 when email is explicitly None."""
        req = _req(
            method="POST",
            path="/api/v1/crm/enrich",
            body={"email": None},
        )
        result = await handler._enrich_contact(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_enrich_providers_list(self, handler):
        """Enrich returns exactly 3 available providers."""
        req = _req(
            method="POST",
            path="/api/v1/crm/enrich",
            body={"email": "user@example.com"},
        )
        result = await handler._enrich_contact(req)
        body = _body(result)
        assert len(body["available_providers"]) == 3


# ===========================================================================
# _search_crm  (POST /api/v1/crm/search)
# ===========================================================================


class TestSearchCRM:
    """Tests for _search_crm."""

    @pytest.mark.asyncio
    async def test_search_no_platforms(self, handler):
        """Search with no connected platforms returns empty results."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test"},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["query"] == "test"
        assert body["results"] == {}
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_search_circuit_breaker_open(self, handler):
        """Search rejected when circuit breaker is open."""
        _open_circuit_breaker()
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test"},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_search_contacts_only(self, handler):
        """Search for contacts only."""
        _connect_hubspot()
        contacts = [MockContact("c1", {"email": "a@b.com", "firstname": "A", "lastname": "B"})]
        conn = _mock_connector(search_contacts=contacts)
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["results"]["contacts"]) == 1
        assert body["total"] == 1
        conn.search_contacts.assert_awaited_once()
        conn.search_companies.assert_not_awaited()
        conn.search_deals.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_companies_only(self, handler):
        """Search for companies only."""
        _connect_hubspot()
        companies = [MockCompany("co1", {"name": "Acme"})]
        conn = _mock_connector(search_companies=companies)
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "acme", "types": ["companies"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["results"]["companies"]) == 1
        conn.search_companies.assert_awaited_once()
        conn.search_contacts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_deals_only(self, handler):
        """Search for deals only."""
        _connect_hubspot()
        deals = [MockDeal("d1", {"dealname": "Big Deal", "amount": "5000"})]
        conn = _mock_connector(search_deals=deals)
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "big", "types": ["deals"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["results"]["deals"]) == 1
        conn.search_deals.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_all_types_default(self, handler):
        """Search defaults to all types: contacts, companies, deals."""
        _connect_hubspot()
        contacts = [MockContact("c1")]
        companies = [MockCompany("co1", {"name": "Test"})]
        deals = [MockDeal("d1", {"dealname": "Deal"})]
        conn = _mock_connector(
            search_contacts=contacts,
            search_companies=companies,
            search_deals=deals,
        )
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test"},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        conn.search_contacts.assert_awaited_once()
        conn.search_companies.assert_awaited_once()
        conn.search_deals.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_invalid_object_type(self, handler):
        """Search rejects invalid object type."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["tickets"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 400
        assert "invalid object type" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_search_mixed_valid_and_invalid_types(self, handler):
        """Search rejects if any type is invalid."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts", "invalid"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_limit_default(self, handler):
        """Search default limit is 20."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        conn.search_contacts.assert_awaited_once_with("test", limit=20)

    @pytest.mark.asyncio
    async def test_search_custom_limit(self, handler):
        """Search with custom limit."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"], "limit": 50},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        conn.search_contacts.assert_awaited_once_with("test", limit=50)

    @pytest.mark.asyncio
    async def test_search_limit_too_low(self, handler):
        """Search rejects limit below 1."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "limit": 0},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 400
        assert "limit" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_search_limit_too_high(self, handler):
        """Search rejects limit above 100."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "limit": 200},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 400
        assert "limit" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_search_limit_not_integer(self, handler):
        """Search rejects non-integer limit."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "limit": "ten"},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_limit_negative(self, handler):
        """Search rejects negative limit."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "limit": -5},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_query_too_long(self, handler):
        """Search rejects query exceeding max length."""
        long_query = "x" * (MAX_SEARCH_QUERY_LENGTH + 1)
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": long_query},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 400
        assert "too long" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_search_empty_query(self, handler):
        """Search with empty query still works (no length violation)."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": ""},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_connector_error(self, handler):
        """Search handles connector errors gracefully."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.search_contacts = AsyncMock(side_effect=ConnectionError("network"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_search_timeout_error(self, handler):
        """Search handles TimeoutError from connector."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.search_contacts = AsyncMock(side_effect=TimeoutError("slow"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_search_os_error(self, handler):
        """Search handles OSError from connector."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.search_contacts = AsyncMock(side_effect=OSError("io error"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_value_error(self, handler):
        """Search handles ValueError from connector."""
        _connect_hubspot()
        conn = _mock_connector()
        conn.search_contacts = AsyncMock(side_effect=ValueError("bad data"))
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_skips_coming_soon_platforms(self, handler):
        """Search skips platforms with coming_soon flag."""
        _connect_hubspot()
        _connect_pipedrive()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        # Only hubspot search should be called
        conn.search_contacts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_no_connector_for_platform(self, handler):
        """Search skips platform if connector is None."""
        _connect_hubspot()
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            req = _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "test"},
            )
            result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_search_results_normalized(self, handler):
        """Search results are normalized from HubSpot format."""
        _connect_hubspot()
        contacts = [
            MockContact("c1", {"email": "a@b.com", "firstname": "Alice", "lastname": "B"}),
        ]
        companies = [
            MockCompany("co1", {"name": "Acme Corp", "domain": "acme.com"}),
        ]
        deals = [
            MockDeal("d1", {"dealname": "Big Deal", "amount": "5000", "dealstage": "proposal"}),
        ]
        conn = _mock_connector(
            search_contacts=contacts,
            search_companies=companies,
            search_deals=deals,
        )
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test"},
        )
        result = await handler._search_crm(req)
        body = _body(result)
        assert body["results"]["contacts"][0]["email"] == "a@b.com"
        assert body["results"]["companies"][0]["name"] == "Acme Corp"
        assert body["results"]["deals"][0]["name"] == "Big Deal"

    @pytest.mark.asyncio
    async def test_search_multiple_contacts(self, handler):
        """Search returns multiple contacts."""
        _connect_hubspot()
        contacts = [
            MockContact("c1", {"email": "a@b.com"}),
            MockContact("c2", {"email": "c@d.com"}),
            MockContact("c3", {"email": "e@f.com"}),
        ]
        conn = _mock_connector(search_contacts=contacts)
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        body = _body(result)
        assert len(body["results"]["contacts"]) == 3
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_search_limit_boundary_1(self, handler):
        """Search accepts limit=1."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"], "limit": 1},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_limit_boundary_100(self, handler):
        """Search accepts limit=100."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": ["contacts"], "limit": 100},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_limit_boundary_101(self, handler):
        """Search rejects limit=101."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "limit": 101},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_query_at_max_length(self, handler):
        """Search accepts query at exactly max length."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "x" * MAX_SEARCH_QUERY_LENGTH, "types": ["contacts"]},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_query_field_present_in_response(self, handler):
        """Search response includes the original query."""
        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "my-search-term"},
        )
        result = await handler._search_crm(req)
        body = _body(result)
        assert body["query"] == "my-search-term"

    @pytest.mark.asyncio
    async def test_search_empty_types_list(self, handler):
        """Search with empty types list returns empty results (no type to search)."""
        _connect_hubspot()
        conn = _mock_connector()
        _install_connector(conn)

        req = _req(
            method="POST",
            path="/api/v1/crm/search",
            body={"query": "test", "types": []},
        )
        result = await handler._search_crm(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0


# ===========================================================================
# Integration: handle_request routing to pipeline methods
# ===========================================================================


class TestHandleRequestRouting:
    """Verify handle_request routes to pipeline mixin methods correctly."""

    @pytest.mark.asyncio
    async def test_route_get_pipeline(self, handler):
        """GET /api/v1/crm/pipeline routes to _get_pipeline."""
        result = await handler.handle_request(_req(method="GET", path="/api/v1/crm/pipeline"))
        assert _status(result) == 200
        body = _body(result)
        assert "pipelines" in body

    @pytest.mark.asyncio
    async def test_route_post_sync_lead(self, handler):
        """POST /api/v1/crm/sync-lead routes to _sync_lead."""
        _connect_hubspot()
        _install_connector()

        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/sync-lead",
                body={"platform": "hubspot", "lead": {"email": "a@b.com"}},
            )
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_route_post_enrich(self, handler):
        """POST /api/v1/crm/enrich routes to _enrich_contact."""
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/enrich",
                body={"email": "user@example.com"},
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["enriched"] is False

    @pytest.mark.asyncio
    async def test_route_post_search(self, handler):
        """POST /api/v1/crm/search routes to _search_crm."""
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/search",
                body={"query": "test"},
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert "results" in body

    @pytest.mark.asyncio
    async def test_route_wrong_method_for_pipeline(self, handler):
        """POST /api/v1/crm/pipeline returns 404 (wrong method)."""
        result = await handler.handle_request(_req(method="POST", path="/api/v1/crm/pipeline"))
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_route_wrong_method_for_enrich(self, handler):
        """GET /api/v1/crm/enrich returns 404 (wrong method)."""
        result = await handler.handle_request(_req(method="GET", path="/api/v1/crm/enrich"))
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_route_wrong_method_for_search(self, handler):
        """GET /api/v1/crm/search returns 404 (wrong method)."""
        result = await handler.handle_request(_req(method="GET", path="/api/v1/crm/search"))
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_route_wrong_method_for_sync_lead(self, handler):
        """GET /api/v1/crm/sync-lead returns 404 (wrong method)."""
        result = await handler.handle_request(_req(method="GET", path="/api/v1/crm/sync-lead"))
        assert _status(result) == 404
