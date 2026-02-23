"""Comprehensive tests for the CRM ContactOperationsMixin.

Tests all methods of ContactOperationsMixin defined in
aragora/server/handlers/features/crm/contacts.py:

- _contacts_stub_enabled() - stub detection via config
- _contacts_unavailable() - 503 stub response
- _list_all_contacts() - list contacts across all platforms
- _list_platform_contacts() - list contacts from a specific platform
- _get_contact() - get a single contact by platform and ID
- _create_contact() - create a contact on a platform
- _update_contact() - update a contact on a platform
- _delete_contact() - delete a contact from a platform

Covers: happy paths, validation errors, circuit breaker, stub mode,
platform not connected, email validation, field length limits, edge cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.server.handlers.features.crm.handler import (
    CRMHandler,
    _platform_credentials,
    _platform_connectors,
)
from aragora.server.handlers.features.crm.circuit_breaker import reset_crm_circuit_breaker
from aragora.server.handlers.features.crm.validation import (
    MAX_EMAIL_LENGTH,
    MAX_JOB_TITLE_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PHONE_LENGTH,
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
    path: str = "/api/v1/crm/contacts",
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


@pytest.fixture
def stub_handler():
    """Create a CRMHandler with contacts_stub enabled."""
    return CRMHandler({"config": {"contacts_stub": True}})


@pytest.fixture
def contacts_enabled_handler():
    """Create a CRMHandler with contacts_enabled=True (stub disabled)."""
    return CRMHandler({"config": {"contacts_enabled": True}})


@pytest.fixture
def contacts_disabled_handler():
    """Create a CRMHandler with contacts_enabled=False (stub enabled)."""
    return CRMHandler({"config": {"contacts_enabled": False}})


@pytest.fixture
def rate_limit_handler():
    """Create a CRMHandler with rate_limit_enabled=True (legacy stub mode)."""
    return CRMHandler({"config": {"rate_limit_enabled": True}})


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
    from aragora.server.handlers.features.crm.circuit_breaker import get_crm_circuit_breaker

    cb = get_crm_circuit_breaker()
    for _ in range(cb.failure_threshold + 1):
        cb.record_failure()


# ===========================================================================
# _contacts_stub_enabled
# ===========================================================================


class TestContactsStubEnabled:
    """Tests for the _contacts_stub_enabled config detection."""

    def test_default_is_false(self, handler):
        assert handler._contacts_stub_enabled() is False

    def test_contacts_stub_true(self, stub_handler):
        assert stub_handler._contacts_stub_enabled() is True

    def test_contacts_stub_false(self):
        h = CRMHandler({"config": {"contacts_stub": False}})
        assert h._contacts_stub_enabled() is False

    def test_contacts_enabled_true_means_stub_disabled(self, contacts_enabled_handler):
        assert contacts_enabled_handler._contacts_stub_enabled() is False

    def test_contacts_enabled_false_means_stub_enabled(self, contacts_disabled_handler):
        assert contacts_disabled_handler._contacts_stub_enabled() is True

    def test_rate_limit_enabled_legacy_stub(self, rate_limit_handler):
        assert rate_limit_handler._contacts_stub_enabled() is True

    def test_rate_limit_disabled_no_stub(self):
        h = CRMHandler({"config": {"rate_limit_enabled": False}})
        assert h._contacts_stub_enabled() is False

    def test_contacts_stub_takes_priority_over_contacts_enabled(self):
        h = CRMHandler({"config": {"contacts_stub": True, "contacts_enabled": True}})
        assert h._contacts_stub_enabled() is True

    def test_contacts_enabled_takes_priority_over_rate_limit(self):
        h = CRMHandler({"config": {"contacts_enabled": True, "rate_limit_enabled": True}})
        assert h._contacts_stub_enabled() is False

    def test_no_ctx_attribute(self):
        h = CRMHandler.__new__(CRMHandler)
        # When ctx is missing, getattr defaults to {}
        assert h._contacts_stub_enabled() is False

    def test_config_is_not_dict(self):
        h = CRMHandler({"config": "not-a-dict"})
        assert h._contacts_stub_enabled() is False

    def test_empty_config(self):
        h = CRMHandler({"config": {}})
        assert h._contacts_stub_enabled() is False


# ===========================================================================
# _contacts_unavailable
# ===========================================================================


class TestContactsUnavailable:
    """Tests for the _contacts_unavailable stub response."""

    def test_returns_503(self, handler):
        result = handler._contacts_unavailable()
        assert _status(result) == 503

    def test_error_message(self, handler):
        result = handler._contacts_unavailable()
        assert "not available" in _body(result)["error"]

    def test_body_has_error_key(self, handler):
        result = handler._contacts_unavailable()
        assert "error" in _body(result)


# ===========================================================================
# _list_all_contacts
# ===========================================================================


class TestListAllContacts:
    """Tests for _list_all_contacts."""

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, handler):
        result = await handler._list_all_contacts(_req())
        assert _status(result) == 200
        body = _body(result)
        assert body["contacts"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_no_email_filter_is_valid(self, handler):
        result = await handler._list_all_contacts(_req())
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_valid_email_filter(self, handler):
        result = await handler._list_all_contacts(_req(query={"email": "user@example.com"}))
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invalid_email_filter(self, handler):
        result = await handler._list_all_contacts(_req(query={"email": "not-an-email"}))
        assert _status(result) == 400
        assert "email" in _body(result)["error"].lower() or "Invalid" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_email_too_long(self, handler):
        long_email = "a" * (MAX_EMAIL_LENGTH + 1) + "@example.com"
        result = await handler._list_all_contacts(_req(query={"email": long_email}))
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_stub_enabled_returns_503(self, stub_handler):
        result = await stub_handler._list_all_contacts(_req())
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _open_circuit_breaker()
        result = await handler._list_all_contacts(_req())
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_request_without_query_attribute(self, handler):
        """When request has no query attr, email is None and passes validation."""

        class BareRequest:
            pass

        result = await handler._list_all_contacts(BareRequest())
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_stub_checked_before_circuit_breaker(self, stub_handler):
        _open_circuit_breaker()
        result = await stub_handler._list_all_contacts(_req())
        # Stub takes priority -- returns its own 503 message
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]


# ===========================================================================
# _list_platform_contacts
# ===========================================================================


class TestListPlatformContacts:
    """Tests for _list_platform_contacts."""

    @pytest.mark.asyncio
    async def test_success_empty(self, handler):
        _connect_hubspot()
        result = await handler._list_platform_contacts(_req(), "hubspot")
        assert _status(result) == 200
        body = _body(result)
        assert body["contacts"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        result = await handler._list_platform_contacts(_req(), "hubspot")
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_platform_format(self, handler):
        result = await handler._list_platform_contacts(_req(), "bad platform!")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_platform(self, handler):
        result = await handler._list_platform_contacts(_req(), "")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_platform_too_long(self, handler):
        result = await handler._list_platform_contacts(_req(), "a" * 51)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_valid_email_filter(self, handler):
        _connect_hubspot()
        result = await handler._list_platform_contacts(
            _req(query={"email": "user@example.com"}), "hubspot"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invalid_email_filter(self, handler):
        _connect_hubspot()
        result = await handler._list_platform_contacts(
            _req(query={"email": "bad-email"}), "hubspot"
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler._list_platform_contacts(_req(), "hubspot")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stub_enabled(self, stub_handler):
        _connect_hubspot()
        result = await stub_handler._list_platform_contacts(_req(), "hubspot")
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_request_without_query_attr(self, handler):
        _connect_hubspot()

        class BareRequest:
            pass

        result = await handler._list_platform_contacts(BareRequest(), "hubspot")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_multiple_platforms_connected(self, handler):
        _connect_hubspot()
        _connect_pipedrive()
        r1 = await handler._list_platform_contacts(_req(), "hubspot")
        r2 = await handler._list_platform_contacts(_req(), "pipedrive")
        assert _status(r1) == 200
        assert _status(r2) == 200


# ===========================================================================
# _get_contact
# ===========================================================================


class TestGetContact:
    """Tests for _get_contact."""

    @pytest.mark.asyncio
    async def test_contact_not_found(self, handler):
        _connect_hubspot()
        result = await handler._get_contact(_req(), "hubspot", "c123")
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        result = await handler._get_contact(_req(), "hubspot", "c123")
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_platform(self, handler):
        result = await handler._get_contact(_req(), "bad platform!", "c123")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_contact_id(self, handler):
        _connect_hubspot()
        result = await handler._get_contact(_req(), "hubspot", "bad id!")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_contact_id_too_long(self, handler):
        _connect_hubspot()
        result = await handler._get_contact(_req(), "hubspot", "x" * 129)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_contact_id_treated_as_none(self, handler):
        """Empty string contact_id -- validate_resource_id rejects it."""
        _connect_hubspot()
        result = await handler._get_contact(_req(), "hubspot", "")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler._get_contact(_req(), "hubspot", "c123")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stub_enabled(self, stub_handler):
        _connect_hubspot()
        result = await stub_handler._get_contact(_req(), "hubspot", "c123")
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_contact_id_none_stub_enabled(self, stub_handler):
        """Legacy stub callers pass (request, platform) only with contact_id=None."""
        result = await stub_handler._get_contact(_req(), "hubspot", None)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_contact_id_none_stub_disabled(self, handler):
        result = await handler._get_contact(_req(), "hubspot", None)
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_valid_resource_id_formats(self, handler):
        _connect_hubspot()
        # Alphanumeric with hyphens and underscores
        for cid in ["c123", "contact-abc", "contact_def", "A1B2C3"]:
            result = await handler._get_contact(_req(), "hubspot", cid)
            # Always 404 "Contact not found" since stub returns empty
            assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_contact_id_starts_with_hyphen(self, handler):
        """IDs must start with alphanumeric."""
        _connect_hubspot()
        result = await handler._get_contact(_req(), "hubspot", "-invalid")
        assert _status(result) == 400


# ===========================================================================
# _create_contact
# ===========================================================================


class TestCreateContact:
    """Tests for _create_contact."""

    @pytest.mark.asyncio
    async def test_success(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "alice@example.com", "first_name": "Alice", "last_name": "Smith"}),
            "hubspot",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["contact"]["email"] == "alice@example.com"
        assert body["contact"]["first_name"] == "Alice"
        assert body["contact"]["last_name"] == "Smith"

    @pytest.mark.asyncio
    async def test_email_only(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "bob@example.com"}),
            "hubspot",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["contact"]["email"] == "bob@example.com"
        assert body["contact"]["first_name"] is None
        assert body["contact"]["last_name"] is None

    @pytest.mark.asyncio
    async def test_missing_email(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"first_name": "Alice"}),
            "hubspot",
        )
        assert _status(result) == 400
        assert "email" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_email(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "not-valid"}),
            "hubspot",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_email_too_long(self, handler):
        _connect_hubspot()
        long_email = "a" * 250 + "@example.com"
        result = await handler._create_contact(
            _req(body={"email": long_email}),
            "hubspot",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_first_name_too_long(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "a@b.com", "first_name": "x" * (MAX_NAME_LENGTH + 1)}),
            "hubspot",
        )
        assert _status(result) == 400
        assert "First name" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_last_name_too_long(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "a@b.com", "last_name": "x" * (MAX_NAME_LENGTH + 1)}),
            "hubspot",
        )
        assert _status(result) == 400
        assert "Last name" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_job_title_too_long(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "a@b.com", "job_title": "x" * (MAX_JOB_TITLE_LENGTH + 1)}),
            "hubspot",
        )
        assert _status(result) == 400
        assert "Job title" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_phone_too_long(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "a@b.com", "phone": "1" * (MAX_PHONE_LENGTH + 1)}),
            "hubspot",
        )
        assert _status(result) == 400
        assert "Phone" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_all_optional_fields(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(
                body={
                    "email": "full@example.com",
                    "first_name": "Full",
                    "last_name": "Contact",
                    "job_title": "CEO",
                    "phone": "+1-555-0199",
                }
            ),
            "hubspot",
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        result = await handler._create_contact(
            _req(body={"email": "a@b.com"}),
            "hubspot",
        )
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_platform(self, handler):
        result = await handler._create_contact(
            _req(body={"email": "a@b.com"}),
            "bad platform!",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_platform(self, handler):
        result = await handler._create_contact(
            _req(body={"email": "a@b.com"}),
            "",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler._create_contact(
            _req(body={"email": "a@b.com"}),
            "hubspot",
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stub_enabled(self, stub_handler):
        _connect_hubspot()
        result = await stub_handler._create_contact(
            _req(body={"email": "a@b.com"}),
            "hubspot",
        )
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_empty_body(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={}),
            "hubspot",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_first_name_at_max_length(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "a@b.com", "first_name": "x" * MAX_NAME_LENGTH}),
            "hubspot",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_phone_at_max_length(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "a@b.com", "phone": "1" * MAX_PHONE_LENGTH}),
            "hubspot",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_job_title_at_max_length(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "a@b.com", "job_title": "T" * MAX_JOB_TITLE_LENGTH}),
            "hubspot",
        )
        assert _status(result) == 200


# ===========================================================================
# _update_contact
# ===========================================================================


class TestUpdateContact:
    """Tests for _update_contact."""

    @pytest.mark.asyncio
    async def test_success_basic(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={"first_name": "Bob"}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_update_with_valid_email(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={"email": "bob@new.com"}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_update_with_invalid_email(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={"email": "not-valid"}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_email_not_required(self, handler):
        """Email is optional on update (required=False)."""
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={"first_name": "Bob"}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_update_empty_body(self, handler):
        """Empty update body should succeed (no required fields)."""
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_first_name_too_long(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={"first_name": "x" * (MAX_NAME_LENGTH + 1)}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 400
        assert "First name" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_last_name_too_long(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={"last_name": "x" * (MAX_NAME_LENGTH + 1)}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 400
        assert "Last name" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_job_title_too_long(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={"job_title": "x" * (MAX_JOB_TITLE_LENGTH + 1)}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 400
        assert "Job title" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_phone_too_long(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(
            _req(body={"phone": "1" * (MAX_PHONE_LENGTH + 1)}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 400
        assert "Phone" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        result = await handler._update_contact(
            _req(body={"first_name": "Bob"}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_invalid_platform(self, handler):
        result = await handler._update_contact(
            _req(body={"first_name": "Bob"}),
            "bad platform!",
            "c123",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler._update_contact(
            _req(body={"first_name": "Bob"}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stub_enabled(self, stub_handler):
        _connect_hubspot()
        result = await stub_handler._update_contact(
            _req(body={"first_name": "Bob"}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_update_all_fields(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(
            _req(
                body={
                    "email": "updated@example.com",
                    "first_name": "Updated",
                    "last_name": "Name",
                    "job_title": "CTO",
                    "phone": "+1-555-0000",
                }
            ),
            "hubspot",
            "c123",
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_update_email_too_long(self, handler):
        _connect_hubspot()
        long_email = "a" * 250 + "@example.com"
        result = await handler._update_contact(
            _req(body={"email": long_email}),
            "hubspot",
            "c123",
        )
        assert _status(result) == 400


# ===========================================================================
# _delete_contact
# ===========================================================================


class TestDeleteContact:
    """Tests for _delete_contact."""

    @pytest.mark.asyncio
    async def test_success(self, handler):
        _connect_hubspot()
        result = await handler._delete_contact(_req(), "hubspot", "c123")
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        result = await handler._delete_contact(_req(), "hubspot", "c123")
        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_platform(self, handler):
        result = await handler._delete_contact(_req(), "bad platform!", "c123")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_platform(self, handler):
        result = await handler._delete_contact(_req(), "", "c123")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_contact_id(self, handler):
        _connect_hubspot()
        result = await handler._delete_contact(_req(), "hubspot", "bad id!")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_contact_id(self, handler):
        _connect_hubspot()
        result = await handler._delete_contact(_req(), "hubspot", "")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_contact_id_too_long(self, handler):
        _connect_hubspot()
        result = await handler._delete_contact(_req(), "hubspot", "x" * 129)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, handler):
        _connect_hubspot()
        _open_circuit_breaker()
        result = await handler._delete_contact(_req(), "hubspot", "c123")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_stub_enabled(self, stub_handler):
        _connect_hubspot()
        result = await stub_handler._delete_contact(_req(), "hubspot", "c123")
        assert _status(result) == 503
        assert "not available" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_valid_id_formats(self, handler):
        _connect_hubspot()
        for cid in ["abc123", "a-b-c", "a_b_c", "A1"]:
            result = await handler._delete_contact(_req(), "hubspot", cid)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_contact_id_starts_with_hyphen(self, handler):
        _connect_hubspot()
        result = await handler._delete_contact(_req(), "hubspot", "-invalid")
        assert _status(result) == 400


# ===========================================================================
# Integration: handle_request routing to contact endpoints
# ===========================================================================


class TestHandleRequestRouting:
    """Test that CRMHandler.handle_request correctly routes to contact methods."""

    @pytest.mark.asyncio
    async def test_list_all_contacts_route(self, handler):
        result = await handler.handle_request(_req(path="/api/v1/crm/contacts"))
        assert _status(result) == 200
        assert "contacts" in _body(result)

    @pytest.mark.asyncio
    async def test_list_platform_contacts_route(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(_req(path="/api/v1/crm/hubspot/contacts"))
        assert _status(result) == 200
        assert "contacts" in _body(result)

    @pytest.mark.asyncio
    async def test_get_contact_route(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(_req(path="/api/v1/crm/hubspot/contacts/c123"))
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_contact_route(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="POST",
                path="/api/v1/crm/hubspot/contacts",
                body={"email": "new@example.com"},
            )
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_update_contact_route(self, handler):
        _connect_hubspot()
        result = await handler.handle_request(
            _req(
                method="PUT",
                path="/api/v1/crm/hubspot/contacts/c123",
                body={"first_name": "Updated"},
            )
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    async def test_delete_contact_no_route(self, handler):
        """DELETE on contacts/{id} is not wired in handle_request routing."""
        _connect_hubspot()
        result = await handler.handle_request(
            _req(method="DELETE", path="/api/v1/crm/hubspot/contacts/c123")
        )
        # The router doesn't have a DELETE handler for contacts, returns 404
        assert _status(result) == 404


# ===========================================================================
# Edge cases / Security
# ===========================================================================


class TestEdgeCases:
    """Edge cases and security-related tests."""

    @pytest.mark.asyncio
    async def test_platform_with_special_chars(self, handler):
        """Platform IDs with special characters should be rejected."""
        for bad_platform in ["hub<script>", "plat/form", "plat.form", "123start"]:
            result = await handler._list_platform_contacts(_req(), bad_platform)
            assert _status(result) == 400, f"Expected 400 for platform: {bad_platform}"

    @pytest.mark.asyncio
    async def test_contact_id_with_special_chars(self, handler):
        """Contact IDs with non-allowed chars should be rejected."""
        _connect_hubspot()
        for bad_id in ["<script>", "id/../../etc", "id with spaces"]:
            result = await handler._get_contact(_req(), "hubspot", bad_id)
            assert _status(result) == 400, f"Expected 400 for id: {bad_id}"

    @pytest.mark.asyncio
    async def test_email_with_xss_payload(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "<script>alert(1)</script>@evil.com"}),
            "hubspot",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_contacts_enabled_config_false(self, contacts_disabled_handler):
        _connect_hubspot()
        result = await contacts_disabled_handler._list_all_contacts(_req())
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_rate_limit_legacy_stub(self, rate_limit_handler):
        _connect_hubspot()
        result = await rate_limit_handler._create_contact(
            _req(body={"email": "a@b.com"}),
            "hubspot",
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_contacts_enabled_true_allows_operations(self, contacts_enabled_handler):
        _connect_hubspot()
        result = await contacts_enabled_handler._list_all_contacts(_req())
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_stub_mode_blocks_all_operations(self, stub_handler):
        """Verify every mixin method is blocked in stub mode."""
        _connect_hubspot()
        ops = [
            stub_handler._list_all_contacts(_req()),
            stub_handler._list_platform_contacts(_req(), "hubspot"),
            stub_handler._get_contact(_req(), "hubspot", "c1"),
            stub_handler._create_contact(_req(body={"email": "a@b.com"}), "hubspot"),
            stub_handler._update_contact(_req(body={}), "hubspot", "c1"),
            stub_handler._delete_contact(_req(), "hubspot", "c1"),
        ]
        for coro in ops:
            result = await coro
            assert _status(result) == 503, "Expected 503 for stub operation"

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_all_operations(self, handler):
        """Verify every mixin method is blocked when circuit breaker open."""
        _connect_hubspot()
        _open_circuit_breaker()
        ops = [
            handler._list_all_contacts(_req()),
            handler._list_platform_contacts(_req(), "hubspot"),
            handler._get_contact(_req(), "hubspot", "c1"),
            handler._create_contact(_req(body={"email": "a@b.com"}), "hubspot"),
            handler._update_contact(_req(body={}), "hubspot", "c1"),
            handler._delete_contact(_req(), "hubspot", "c1"),
        ]
        for coro in ops:
            result = await coro
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_platform_validation_before_connection_check(self, handler):
        """Invalid platform format should return 400 before checking connection."""
        result = await handler._list_platform_contacts(_req(), "!!")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_response_shape_list_contacts(self, handler):
        result = await handler._list_all_contacts(_req())
        body = _body(result)
        assert "contacts" in body
        assert "total" in body
        assert isinstance(body["contacts"], list)
        assert isinstance(body["total"], int)

    @pytest.mark.asyncio
    async def test_response_shape_create_contact(self, handler):
        _connect_hubspot()
        result = await handler._create_contact(
            _req(body={"email": "shape@test.com"}),
            "hubspot",
        )
        body = _body(result)
        assert "success" in body
        assert "contact" in body
        assert "email" in body["contact"]

    @pytest.mark.asyncio
    async def test_response_shape_delete_contact(self, handler):
        _connect_hubspot()
        result = await handler._delete_contact(_req(), "hubspot", "c1")
        body = _body(result)
        assert "success" in body
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_response_shape_update_contact(self, handler):
        _connect_hubspot()
        result = await handler._update_contact(_req(body={}), "hubspot", "c1")
        body = _body(result)
        assert "success" in body
        assert body["success"] is True
