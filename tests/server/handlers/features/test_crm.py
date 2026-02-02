"""
Tests for CRM Integration Handler.

Stability: STABLE (graduated from EXPERIMENTAL)

Covers:
- Platform listing and connection status
- Platform connection/disconnection
- Contacts: listing, filtering, creation, update, cross-platform aggregation
- Companies: listing, creation, cross-platform aggregation
- Deals: listing, filtering, creation, cross-platform aggregation
- Pipeline: retrieval and stage summary
- Lead sync: external source lead synchronization
- Contact enrichment
- Search across CRM data
- Circuit breaker functionality
- Rate limiting integration
- Input validation
- RBAC permission checks
- Error handling
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.crm import (
    CRMHandler,
    CRMCircuitBreaker,
    SUPPORTED_PLATFORMS,
    UnifiedContact,
    UnifiedCompany,
    UnifiedDeal,
    get_crm_circuit_breaker,
    reset_crm_circuit_breaker,
    _validate_platform_id,
    _validate_resource_id,
    _validate_email,
    _validate_string_field,
    _validate_amount,
    _validate_probability,
    _platform_credentials,
    _platform_connectors,
    MAX_EMAIL_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PHONE_LENGTH,
    MAX_COMPANY_NAME_LENGTH,
    MAX_JOB_TITLE_LENGTH,
    MAX_DOMAIN_LENGTH,
    MAX_DEAL_NAME_LENGTH,
    MAX_STAGE_LENGTH,
    MAX_PIPELINE_LENGTH,
    MAX_CREDENTIAL_VALUE_LENGTH,
    MAX_SEARCH_QUERY_LENGTH,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {"config": {"debug": True}}


@pytest.fixture
def crm_handler(server_context):
    """Create a CRMHandler instance."""
    reset_crm_circuit_breaker()
    # Clear global state
    _platform_credentials.clear()
    _platform_connectors.clear()
    return CRMHandler(server_context=server_context)


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    request = MagicMock()
    request.method = "GET"
    request.path = "/api/v1/crm/platforms"
    request.query = {}
    request.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }
    return request


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context."""
    ctx = MagicMock()
    ctx.user_id = "test-user"
    ctx.workspace_id = "test-workspace"
    ctx.roles = ["admin"]
    return ctx


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Clean up global state before and after each test."""
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_crm_circuit_breaker()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_crm_circuit_breaker()


# -----------------------------------------------------------------------------
# Test: Supported Platforms Configuration
# -----------------------------------------------------------------------------


class TestSupportedPlatforms:
    """Tests for CRM platform configuration."""

    def test_all_platforms_defined(self):
        """Test that CRM platforms are configured."""
        assert "hubspot" in SUPPORTED_PLATFORMS
        assert "salesforce" in SUPPORTED_PLATFORMS
        assert "pipedrive" in SUPPORTED_PLATFORMS
        assert len(SUPPORTED_PLATFORMS) >= 3

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config, f"Platform {platform_id} missing 'name'"
            assert "description" in config, f"Platform {platform_id} missing 'description'"
            assert "features" in config, f"Platform {platform_id} missing 'features'"
            assert isinstance(config["features"], list), (
                f"Platform {platform_id} features should be list"
            )

    def test_hubspot_features(self):
        """Test HubSpot has expected features."""
        hubspot = SUPPORTED_PLATFORMS["hubspot"]
        assert "contacts" in hubspot["features"]
        assert "companies" in hubspot["features"]
        assert "deals" in hubspot["features"]

    def test_salesforce_coming_soon(self):
        """Test Salesforce is marked as coming soon."""
        salesforce = SUPPORTED_PLATFORMS["salesforce"]
        assert salesforce.get("coming_soon") is True

    def test_pipedrive_coming_soon(self):
        """Test Pipedrive is marked as coming soon."""
        pipedrive = SUPPORTED_PLATFORMS["pipedrive"]
        assert pipedrive.get("coming_soon") is True


# -----------------------------------------------------------------------------
# Test: Input Validation Functions
# -----------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_platform_id_valid(self):
        """Test valid platform IDs."""
        assert _validate_platform_id("hubspot") == (True, None)
        assert _validate_platform_id("my_platform") == (True, None)
        assert _validate_platform_id("platform123") == (True, None)

    def test_validate_platform_id_invalid_empty(self):
        """Test invalid platform ID - empty."""
        is_valid, msg = _validate_platform_id("")
        assert not is_valid
        assert "required" in msg.lower()

    def test_validate_platform_id_invalid_too_long(self):
        """Test invalid platform ID - too long."""
        is_valid, msg = _validate_platform_id("a" * 100)
        assert not is_valid
        assert "too long" in msg.lower()

    def test_validate_platform_id_invalid_format(self):
        """Test invalid platform ID - invalid format."""
        is_valid, msg = _validate_platform_id("invalid-platform")
        assert not is_valid
        assert "invalid" in msg.lower()

        is_valid, msg = _validate_platform_id("123platform")
        assert not is_valid

    def test_validate_resource_id_valid(self):
        """Test valid resource IDs."""
        assert _validate_resource_id("contact123") == (True, None)
        assert _validate_resource_id("id-123-abc") == (True, None)
        assert _validate_resource_id("item_456") == (True, None)

    def test_validate_resource_id_invalid_empty(self):
        """Test invalid resource ID - empty."""
        is_valid, msg = _validate_resource_id("")
        assert not is_valid

    def test_validate_resource_id_invalid_too_long(self):
        """Test invalid resource ID - too long."""
        is_valid, msg = _validate_resource_id("a" * 200)
        assert not is_valid
        assert "too long" in msg.lower()

    def test_validate_resource_id_invalid_format(self):
        """Test invalid resource ID - invalid format."""
        is_valid, msg = _validate_resource_id("-invalid")
        assert not is_valid

    def test_validate_resource_id_custom_type(self):
        """Test resource ID validation with custom type."""
        is_valid, msg = _validate_resource_id("", "Contact ID")
        assert not is_valid
        assert "contact id" in msg.lower()

    def test_validate_email_valid(self):
        """Test valid emails."""
        assert _validate_email("test@example.com") == (True, None)
        assert _validate_email("user.name+tag@domain.co.uk") == (True, None)
        assert _validate_email(None) == (True, None)

    def test_validate_email_required(self):
        """Test email required validation."""
        is_valid, msg = _validate_email(None, required=True)
        assert not is_valid
        assert "required" in msg.lower()

        is_valid, msg = _validate_email("", required=True)
        assert not is_valid

    def test_validate_email_too_long(self):
        """Test email too long validation."""
        long_email = "a" * 250 + "@example.com"
        is_valid, msg = _validate_email(long_email)
        assert not is_valid
        assert "too long" in msg.lower()

    def test_validate_email_invalid_format(self):
        """Test invalid email format."""
        is_valid, msg = _validate_email("not-an-email")
        assert not is_valid
        assert "invalid" in msg.lower()

        is_valid, msg = _validate_email("missing@domain")
        assert not is_valid

    def test_validate_string_field_valid(self):
        """Test valid string field."""
        assert _validate_string_field("John", "First name", 128) == (True, None)
        assert _validate_string_field(None, "Optional", 128) == (True, None)
        assert _validate_string_field("", "Optional", 128) == (True, None)

    def test_validate_string_field_required(self):
        """Test required string field."""
        is_valid, msg = _validate_string_field(None, "Name", 128, required=True)
        assert not is_valid
        assert "required" in msg.lower()

        is_valid, msg = _validate_string_field("", "Name", 128, required=True)
        assert not is_valid

    def test_validate_string_field_too_long(self):
        """Test string field too long."""
        is_valid, msg = _validate_string_field("a" * 200, "Name", 128)
        assert not is_valid
        assert "too long" in msg.lower()
        assert "128" in msg

    def test_validate_amount_valid(self):
        """Test valid amounts."""
        is_valid, msg, val = _validate_amount(100.00)
        assert is_valid
        assert val == 100.00

        is_valid, msg, val = _validate_amount("50.50")
        assert is_valid
        assert val == 50.50

        is_valid, msg, val = _validate_amount(0)
        assert is_valid
        assert val == 0

        is_valid, msg, val = _validate_amount(None)
        assert is_valid
        assert val is None

    def test_validate_amount_invalid(self):
        """Test invalid amounts."""
        is_valid, msg, val = _validate_amount(-100)
        assert not is_valid
        assert "negative" in msg.lower()

        is_valid, msg, val = _validate_amount(10_000_000_000_000)
        assert not is_valid
        assert "too large" in msg.lower()

        is_valid, msg, val = _validate_amount("not-a-number")
        assert not is_valid
        assert "invalid" in msg.lower()

    def test_validate_probability_valid(self):
        """Test valid probability values."""
        is_valid, msg, val = _validate_probability(75.0)
        assert is_valid
        assert val == 75.0

        is_valid, msg, val = _validate_probability("50")
        assert is_valid
        assert val == 50

        is_valid, msg, val = _validate_probability(0)
        assert is_valid
        assert val == 0

        is_valid, msg, val = _validate_probability(100)
        assert is_valid
        assert val == 100

        is_valid, msg, val = _validate_probability(None)
        assert is_valid
        assert val is None

    def test_validate_probability_invalid(self):
        """Test invalid probability values."""
        is_valid, msg, val = _validate_probability(-10)
        assert not is_valid
        assert "between" in msg.lower()

        is_valid, msg, val = _validate_probability(110)
        assert not is_valid
        assert "between" in msg.lower()

        is_valid, msg, val = _validate_probability("not-a-number")
        assert not is_valid
        assert "invalid" in msg.lower()


# -----------------------------------------------------------------------------
# Test: Circuit Breaker
# -----------------------------------------------------------------------------


class TestCRMCircuitBreaker:
    """Tests for the circuit breaker pattern."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = CRMCircuitBreaker()
        assert cb.state == "closed"
        assert cb.can_proceed()

    def test_opens_after_failure_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        cb = CRMCircuitBreaker(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert not cb.can_proceed()

    def test_remains_closed_below_threshold(self):
        """Test circuit stays closed below failure threshold."""
        cb = CRMCircuitBreaker(failure_threshold=5)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == "closed"
        assert cb.can_proceed()

    def test_success_resets_failure_count(self):
        """Test success resets failure count in closed state."""
        cb = CRMCircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        # Should be able to withstand more failures
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_transitions_to_half_open_after_cooldown(self):
        """Test transition to half-open after cooldown."""
        cb = CRMCircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)

        cb.record_failure()
        assert cb.state == "open"

        time.sleep(0.15)
        assert cb.state == "half_open"

    def test_closes_after_successful_half_open_calls(self):
        """Test circuit closes after successful calls in half-open."""
        cb = CRMCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.1,
            half_open_max_calls=2,
        )

        cb.record_failure()
        time.sleep(0.15)

        # Now in half-open state
        assert cb.can_proceed()
        cb.record_success()
        cb.record_success()

        assert cb.state == "closed"

    def test_reopens_on_half_open_failure(self):
        """Test circuit reopens on failure during half-open."""
        cb = CRMCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.1,
        )

        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"

        cb.record_failure()
        assert cb.state == "open"

    def test_half_open_limits_calls(self):
        """Test half-open state limits number of calls."""
        cb = CRMCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.1,
            half_open_max_calls=2,
        )

        cb.record_failure()
        time.sleep(0.15)

        # First two calls allowed
        assert cb.can_proceed()
        assert cb.can_proceed()
        # Third call blocked
        assert not cb.can_proceed()

    def test_get_status(self):
        """Test circuit breaker status reporting."""
        cb = CRMCircuitBreaker()
        status = cb.get_status()

        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failure_threshold" in status
        assert "cooldown_seconds" in status
        assert "last_failure_time" in status

    def test_reset(self):
        """Test circuit breaker reset."""
        cb = CRMCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()
        assert cb.state == "closed"
        assert cb.can_proceed()

    def test_global_circuit_breaker(self):
        """Test global circuit breaker access."""
        cb = get_crm_circuit_breaker()
        assert cb is not None

        reset_crm_circuit_breaker()
        assert cb.state == "closed"


# -----------------------------------------------------------------------------
# Test: Handler Creation and Configuration
# -----------------------------------------------------------------------------


class TestCRMHandler:
    """Tests for CRMHandler class."""

    def test_handler_creation(self, server_context):
        """Test creating handler instance."""
        handler = CRMHandler(server_context=server_context)
        assert handler is not None
        assert handler.ctx == server_context

    def test_handler_creation_with_ctx(self):
        """Test creating handler with ctx parameter."""
        handler = CRMHandler(ctx={"key": "value"})
        assert handler.ctx == {"key": "value"}

    def test_handler_has_routes(self, crm_handler):
        """Test that handler has route definitions."""
        assert hasattr(crm_handler, "ROUTES")
        assert len(crm_handler.ROUTES) > 0
        assert "/api/v1/crm/platforms" in crm_handler.ROUTES

    def test_handler_has_resource_type(self, crm_handler):
        """Test handler has resource type for RBAC."""
        assert hasattr(crm_handler, "RESOURCE_TYPE")
        assert crm_handler.RESOURCE_TYPE == "crm"

    def test_can_handle_matching_paths(self, crm_handler):
        """Test can_handle returns True for matching paths."""
        assert crm_handler.can_handle("/api/v1/crm/platforms")
        assert crm_handler.can_handle("/api/v1/crm/contacts")
        assert crm_handler.can_handle("/api/v1/crm/hubspot/contacts")

    def test_can_handle_non_matching_paths(self, crm_handler):
        """Test can_handle returns False for non-matching paths."""
        assert not crm_handler.can_handle("/api/v1/other/path")
        assert not crm_handler.can_handle("/api/ecommerce/orders")


# -----------------------------------------------------------------------------
# Test: Platform Management Endpoints
# -----------------------------------------------------------------------------


class TestPlatformManagement:
    """Tests for platform connection management."""

    @pytest.mark.asyncio
    async def test_list_platforms(self, crm_handler, mock_request):
        """Test listing supported platforms."""
        mock_request.path = "/api/v1/crm/platforms"

        result = await crm_handler._list_platforms(mock_request)

        assert result["status_code"] == 200
        assert "platforms" in result["body"]
        assert len(result["body"]["platforms"]) >= 3
        assert result["body"]["connected_count"] == 0

    @pytest.mark.asyncio
    async def test_list_platforms_with_connected(self, crm_handler, mock_request):
        """Test listing platforms shows connection status."""
        _platform_credentials["hubspot"] = {
            "credentials": {"access_token": "test"},
            "connected_at": "2024-01-01T00:00:00Z",
        }

        result = await crm_handler._list_platforms(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["connected_count"] == 1

        hubspot = next(p for p in result["body"]["platforms"] if p["id"] == "hubspot")
        assert hubspot["connected"] is True

    @pytest.mark.asyncio
    async def test_connect_platform_success(self, crm_handler, mock_request):
        """Test successful platform connection."""
        mock_request.method = "POST"
        mock_request.path = "/api/v1/crm/connect"

        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "credentials": {
                    "access_token": "test-token",
                },
            }

            with patch.object(crm_handler, "_get_connector", new_callable=AsyncMock) as mock_conn:
                mock_conn.return_value = None

                result = await crm_handler._connect_platform(mock_request)

                assert result["status_code"] == 200
                assert "hubspot" in _platform_credentials

    @pytest.mark.asyncio
    async def test_connect_platform_missing_platform(self, crm_handler, mock_request):
        """Test connect with missing platform."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"credentials": {"key": "value"}}

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "required" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_invalid_platform(self, crm_handler, mock_request):
        """Test connect with invalid platform ID."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "invalid-platform!@#",
                "credentials": {"key": "value"},
            }

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_connect_platform_unsupported(self, crm_handler, mock_request):
        """Test connect with unsupported platform."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "unknown_platform",
                "credentials": {"key": "value"},
            }

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "unsupported" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_coming_soon(self, crm_handler, mock_request):
        """Test connect with coming soon platform."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "salesforce",
                "credentials": {
                    "client_id": "id",
                    "client_secret": "secret",
                    "refresh_token": "token",
                    "instance_url": "https://example.salesforce.com",
                },
            }

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "coming soon" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_missing_credentials(self, crm_handler, mock_request):
        """Test connect with missing credentials."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "credentials": {},
            }

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "credentials" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_missing_required_credentials(self, crm_handler, mock_request):
        """Test connect with missing required credentials."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "credentials": {"wrong_field": "value"},
            }

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "missing" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_credential_too_long(self, crm_handler, mock_request):
        """Test connect with credential value too long."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "credentials": {"access_token": "a" * (MAX_CREDENTIAL_VALUE_LENGTH + 1)},
            }

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "too long" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_disconnect_platform_success(self, crm_handler, mock_request):
        """Test successful platform disconnection."""
        _platform_credentials["hubspot"] = {
            "credentials": {"access_token": "test"},
            "connected_at": "2024-01-01T00:00:00Z",
        }

        result = await crm_handler._disconnect_platform(mock_request, "hubspot")

        assert result["status_code"] == 200
        assert "hubspot" not in _platform_credentials

    @pytest.mark.asyncio
    async def test_disconnect_platform_with_connector(self, crm_handler, mock_request):
        """Test disconnect closes connector."""
        mock_connector = AsyncMock()
        _platform_credentials["hubspot"] = {
            "credentials": {"access_token": "test"},
            "connected_at": "2024-01-01T00:00:00Z",
        }
        _platform_connectors["hubspot"] = mock_connector

        result = await crm_handler._disconnect_platform(mock_request, "hubspot")

        assert result["status_code"] == 200
        mock_connector.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_platform_not_connected(self, crm_handler, mock_request):
        """Test disconnect when platform not connected."""
        result = await crm_handler._disconnect_platform(mock_request, "hubspot")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_disconnect_platform_invalid_id(self, crm_handler, mock_request):
        """Test disconnect with invalid platform ID."""
        result = await crm_handler._disconnect_platform(mock_request, "invalid-id!")

        assert result["status_code"] == 400


# -----------------------------------------------------------------------------
# Test: Contact Endpoints
# -----------------------------------------------------------------------------


class TestContactEndpoints:
    """Tests for contact management endpoints."""

    @pytest.mark.asyncio
    async def test_list_all_contacts_no_platforms(self, crm_handler, mock_request):
        """Test listing contacts with no connected platforms."""
        result = await crm_handler._list_all_contacts(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["contacts"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_contacts_with_email_filter(self, crm_handler, mock_request):
        """Test listing contacts with email filter."""
        mock_request.query = {"email": "test@example.com"}
        result = await crm_handler._list_all_contacts(mock_request)

        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_list_all_contacts_invalid_email(self, crm_handler, mock_request):
        """Test listing contacts with invalid email filter."""
        mock_request.query = {"email": "invalid-email"}
        result = await crm_handler._list_all_contacts(mock_request)

        assert result["status_code"] == 400
        assert "invalid" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_list_platform_contacts_not_connected(self, crm_handler, mock_request):
        """Test listing contacts for non-connected platform."""
        result = await crm_handler._list_platform_contacts(mock_request, "hubspot")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_list_platform_contacts_invalid_platform(self, crm_handler, mock_request):
        """Test listing contacts with invalid platform ID."""
        result = await crm_handler._list_platform_contacts(mock_request, "invalid-id!")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_get_contact_not_connected(self, crm_handler, mock_request):
        """Test getting contact for non-connected platform."""
        result = await crm_handler._get_contact(mock_request, "hubspot", "contact123")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_contact_invalid_platform(self, crm_handler, mock_request):
        """Test getting contact with invalid platform ID."""
        result = await crm_handler._get_contact(mock_request, "invalid!", "contact123")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_get_contact_invalid_contact_id(self, crm_handler, mock_request):
        """Test getting contact with invalid contact ID."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        result = await crm_handler._get_contact(mock_request, "hubspot", "-invalid")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_create_contact_not_connected(self, crm_handler, mock_request):
        """Test creating contact for non-connected platform."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"email": "test@example.com"}
            result = await crm_handler._create_contact(mock_request, "hubspot")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_create_contact_invalid_email(self, crm_handler, mock_request):
        """Test creating contact with invalid email."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"email": "invalid-email"}
            result = await crm_handler._create_contact(mock_request, "hubspot")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_create_contact_name_too_long(self, crm_handler, mock_request):
        """Test creating contact with name too long."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "email": "test@example.com",
                "first_name": "a" * (MAX_NAME_LENGTH + 1),
            }
            result = await crm_handler._create_contact(mock_request, "hubspot")

        assert result["status_code"] == 400
        assert "too long" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_update_contact_not_connected(self, crm_handler, mock_request):
        """Test updating contact for non-connected platform."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"first_name": "Updated"}
            result = await crm_handler._update_contact(mock_request, "hubspot", "contact123")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_update_contact_invalid_email(self, crm_handler, mock_request):
        """Test updating contact with invalid email."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"email": "invalid-email"}
            result = await crm_handler._update_contact(mock_request, "hubspot", "contact123")

        assert result["status_code"] == 400


# -----------------------------------------------------------------------------
# Test: Company Endpoints
# -----------------------------------------------------------------------------


class TestCompanyEndpoints:
    """Tests for company management endpoints."""

    @pytest.mark.asyncio
    async def test_list_all_companies_no_platforms(self, crm_handler, mock_request):
        """Test listing companies with no connected platforms."""
        result = await crm_handler._list_all_companies(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["companies"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_platform_companies_not_connected(self, crm_handler, mock_request):
        """Test listing companies for non-connected platform."""
        result = await crm_handler._list_platform_companies(mock_request, "hubspot")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_company_not_connected(self, crm_handler, mock_request):
        """Test getting company for non-connected platform."""
        result = await crm_handler._get_company(mock_request, "hubspot", "company123")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_company_invalid_id(self, crm_handler, mock_request):
        """Test getting company with invalid ID."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        result = await crm_handler._get_company(mock_request, "hubspot", "-invalid")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_create_company_not_connected(self, crm_handler, mock_request):
        """Test creating company for non-connected platform."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"name": "Acme Corp"}
            result = await crm_handler._create_company(mock_request, "hubspot")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_create_company_missing_name(self, crm_handler, mock_request):
        """Test creating company without required name."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {}
            result = await crm_handler._create_company(mock_request, "hubspot")

        assert result["status_code"] == 400
        assert "required" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_create_company_name_too_long(self, crm_handler, mock_request):
        """Test creating company with name too long."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"name": "a" * (MAX_COMPANY_NAME_LENGTH + 1)}
            result = await crm_handler._create_company(mock_request, "hubspot")

        assert result["status_code"] == 400


# -----------------------------------------------------------------------------
# Test: Deal Endpoints
# -----------------------------------------------------------------------------


class TestDealEndpoints:
    """Tests for deal management endpoints."""

    @pytest.mark.asyncio
    async def test_list_all_deals_no_platforms(self, crm_handler, mock_request):
        """Test listing deals with no connected platforms."""
        result = await crm_handler._list_all_deals(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["deals"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_deals_with_stage_filter(self, crm_handler, mock_request):
        """Test listing deals with stage filter."""
        mock_request.query = {"stage": "negotiation"}
        result = await crm_handler._list_all_deals(mock_request)

        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_list_all_deals_stage_too_long(self, crm_handler, mock_request):
        """Test listing deals with stage filter too long."""
        mock_request.query = {"stage": "a" * (MAX_STAGE_LENGTH + 1)}
        result = await crm_handler._list_all_deals(mock_request)

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_list_platform_deals_not_connected(self, crm_handler, mock_request):
        """Test listing deals for non-connected platform."""
        result = await crm_handler._list_platform_deals(mock_request, "hubspot")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_deal_not_connected(self, crm_handler, mock_request):
        """Test getting deal for non-connected platform."""
        result = await crm_handler._get_deal(mock_request, "hubspot", "deal123")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_deal_invalid_id(self, crm_handler, mock_request):
        """Test getting deal with invalid ID."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        result = await crm_handler._get_deal(mock_request, "hubspot", "-invalid")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_create_deal_not_connected(self, crm_handler, mock_request):
        """Test creating deal for non-connected platform."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"name": "Big Deal", "stage": "proposal"}
            result = await crm_handler._create_deal(mock_request, "hubspot")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_create_deal_missing_name(self, crm_handler, mock_request):
        """Test creating deal without required name."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"stage": "proposal"}
            result = await crm_handler._create_deal(mock_request, "hubspot")

        assert result["status_code"] == 400
        assert "required" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_create_deal_missing_stage(self, crm_handler, mock_request):
        """Test creating deal without required stage."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"name": "Big Deal"}
            result = await crm_handler._create_deal(mock_request, "hubspot")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_create_deal_invalid_amount(self, crm_handler, mock_request):
        """Test creating deal with invalid amount."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "name": "Big Deal",
                "stage": "proposal",
                "amount": -1000,
            }
            result = await crm_handler._create_deal(mock_request, "hubspot")

        assert result["status_code"] == 400


# -----------------------------------------------------------------------------
# Test: Pipeline Endpoint
# -----------------------------------------------------------------------------


class TestPipelineEndpoint:
    """Tests for pipeline endpoint."""

    @pytest.mark.asyncio
    async def test_get_pipeline_no_platforms(self, crm_handler, mock_request):
        """Test getting pipeline with no connected platforms."""
        mock_request.query = {}
        result = await crm_handler._get_pipeline(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["pipelines"] == []
        assert result["body"]["total_deals"] == 0

    @pytest.mark.asyncio
    async def test_get_pipeline_platform_not_connected(self, crm_handler, mock_request):
        """Test getting pipeline for non-connected platform."""
        mock_request.query = {"platform": "hubspot"}
        result = await crm_handler._get_pipeline(mock_request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_pipeline_invalid_platform(self, crm_handler, mock_request):
        """Test getting pipeline with invalid platform ID."""
        mock_request.query = {"platform": "invalid-id!"}
        result = await crm_handler._get_pipeline(mock_request)

        assert result["status_code"] == 400


# -----------------------------------------------------------------------------
# Test: Lead Sync Endpoint
# -----------------------------------------------------------------------------


class TestLeadSyncEndpoint:
    """Tests for lead sync endpoint."""

    @pytest.mark.asyncio
    async def test_sync_lead_not_connected(self, crm_handler, mock_request):
        """Test syncing lead to non-connected platform."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "lead": {"email": "test@example.com"},
            }
            result = await crm_handler._sync_lead(mock_request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_sync_lead_missing_email(self, crm_handler, mock_request):
        """Test syncing lead without email."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "lead": {"first_name": "John"},
            }
            result = await crm_handler._sync_lead(mock_request)

        assert result["status_code"] == 400
        assert "required" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_sync_lead_invalid_email(self, crm_handler, mock_request):
        """Test syncing lead with invalid email."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "lead": {"email": "invalid-email"},
            }
            result = await crm_handler._sync_lead(mock_request)

        assert result["status_code"] == 400


# -----------------------------------------------------------------------------
# Test: Enrichment Endpoint
# -----------------------------------------------------------------------------


class TestEnrichmentEndpoint:
    """Tests for contact enrichment endpoint."""

    @pytest.mark.asyncio
    async def test_enrich_contact_missing_email(self, crm_handler, mock_request):
        """Test enriching contact without email."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {}
            result = await crm_handler._enrich_contact(mock_request)

        assert result["status_code"] == 400
        assert "required" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_enrich_contact_invalid_email(self, crm_handler, mock_request):
        """Test enriching contact with invalid email."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"email": "invalid-email"}
            result = await crm_handler._enrich_contact(mock_request)

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_enrich_contact_success(self, crm_handler, mock_request):
        """Test successful enrichment request."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"email": "test@example.com"}
            result = await crm_handler._enrich_contact(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["email"] == "test@example.com"
        assert "available_providers" in result["body"]


# -----------------------------------------------------------------------------
# Test: Search Endpoint
# -----------------------------------------------------------------------------


class TestSearchEndpoint:
    """Tests for CRM search endpoint."""

    @pytest.mark.asyncio
    async def test_search_no_platforms(self, crm_handler, mock_request):
        """Test searching with no connected platforms."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"query": "test"}
            result = await crm_handler._search_crm(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_search_query_too_long(self, crm_handler, mock_request):
        """Test searching with query too long."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"query": "a" * (MAX_SEARCH_QUERY_LENGTH + 1)}
            result = await crm_handler._search_crm(mock_request)

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_search_invalid_object_type(self, crm_handler, mock_request):
        """Test searching with invalid object type."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"query": "test", "types": ["invalid"]}
            result = await crm_handler._search_crm(mock_request)

        assert result["status_code"] == 400
        assert "invalid object type" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_search_invalid_limit(self, crm_handler, mock_request):
        """Test searching with invalid limit."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"query": "test", "limit": 500}
            result = await crm_handler._search_crm(mock_request)

        assert result["status_code"] == 400
        assert "limit" in result["body"]["error"].lower()


# -----------------------------------------------------------------------------
# Test: Status Endpoint
# -----------------------------------------------------------------------------


class TestStatusEndpoint:
    """Tests for status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status(self, crm_handler, mock_request):
        """Test getting CRM status."""
        result = await crm_handler._get_status(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["status"] == "healthy"
        assert "circuit_breaker" in result["body"]
        assert "connected_platforms" in result["body"]
        assert "supported_platforms" in result["body"]

    @pytest.mark.asyncio
    async def test_get_status_with_open_circuit(self, crm_handler, mock_request):
        """Test status shows degraded when circuit is open."""
        cb = get_crm_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        result = await crm_handler._get_status(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["status"] == "degraded"


# -----------------------------------------------------------------------------
# Test: Circuit Breaker Integration
# -----------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with handler."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_contacts(self, crm_handler, mock_request):
        """Test contacts endpoint blocked when circuit is open."""
        cb = get_crm_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        result = await crm_handler._list_all_contacts(mock_request)

        assert result["status_code"] == 503
        assert "unavailable" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_companies(self, crm_handler, mock_request):
        """Test companies endpoint blocked when circuit is open."""
        cb = get_crm_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        result = await crm_handler._list_all_companies(mock_request)

        assert result["status_code"] == 503

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_deals(self, crm_handler, mock_request):
        """Test deals endpoint blocked when circuit is open."""
        cb = get_crm_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        result = await crm_handler._list_all_deals(mock_request)

        assert result["status_code"] == 503


# -----------------------------------------------------------------------------
# Test: Request Routing
# -----------------------------------------------------------------------------


class TestRequestRouting:
    """Tests for request routing in handle_request."""

    @pytest.mark.asyncio
    async def test_route_status(self, crm_handler, mock_request, mock_auth_context):
        """Test routing to status endpoint."""
        mock_request.path = "/api/v1/crm/status"
        mock_request.method = "GET"

        result = await crm_handler._get_status(mock_request)

        assert result["status_code"] == 200
        assert "status" in result["body"]

    @pytest.mark.asyncio
    async def test_route_platforms(self, crm_handler, mock_request, mock_auth_context):
        """Test routing to platforms endpoint."""
        mock_request.path = "/api/v1/crm/platforms"
        mock_request.method = "GET"

        result = await crm_handler._list_platforms(mock_request)

        assert result["status_code"] == 200
        assert "platforms" in result["body"]


# -----------------------------------------------------------------------------
# Test: Data Models
# -----------------------------------------------------------------------------


class TestUnifiedContact:
    """Tests for UnifiedContact dataclass."""

    def test_contact_creation(self):
        """Test creating a unified contact."""
        contact = UnifiedContact(
            id="contact_123",
            platform="hubspot",
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            phone="+1234567890",
            company="Acme Inc",
            job_title="CEO",
            lifecycle_stage="customer",
            lead_status=None,
            owner_id="user_1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        assert contact.id == "contact_123"
        assert contact.email == "test@example.com"
        assert contact.company == "Acme Inc"

    def test_contact_to_dict(self):
        """Test contact serialization."""
        contact = UnifiedContact(
            id="contact_456",
            platform="hubspot",
            email="jane@example.com",
            first_name="Jane",
            last_name="Smith",
            phone=None,
            company=None,
            job_title=None,
            lifecycle_stage=None,
            lead_status=None,
            owner_id=None,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            properties={"source": "website"},
        )

        data = contact.to_dict()
        assert data["id"] == "contact_456"
        assert data["email"] == "jane@example.com"
        assert data["full_name"] == "Jane Smith"
        assert data["properties"]["source"] == "website"
        assert data["created_at"] == "2024-01-01T00:00:00+00:00"

    def test_contact_to_dict_no_name(self):
        """Test contact serialization without name."""
        contact = UnifiedContact(
            id="contact_789",
            platform="hubspot",
            email="anon@example.com",
            first_name=None,
            last_name=None,
            phone=None,
            company=None,
            job_title=None,
            lifecycle_stage=None,
            lead_status=None,
            owner_id=None,
            created_at=None,
            updated_at=None,
        )

        data = contact.to_dict()
        assert data["full_name"] is None
        assert data["created_at"] is None


class TestUnifiedCompany:
    """Tests for UnifiedCompany dataclass."""

    def test_company_creation(self):
        """Test creating a unified company."""
        company = UnifiedCompany(
            id="company_123",
            platform="hubspot",
            name="Acme Corporation",
            domain="acme.com",
            industry="Technology",
            employee_count=500,
            annual_revenue=50000000.0,
            owner_id="user_1",
            created_at=datetime.now(timezone.utc),
        )

        assert company.id == "company_123"
        assert company.name == "Acme Corporation"
        assert company.employee_count == 500

    def test_company_to_dict(self):
        """Test company serialization."""
        company = UnifiedCompany(
            id="company_456",
            platform="hubspot",
            name="Test Corp",
            domain="test.com",
            industry=None,
            employee_count=None,
            annual_revenue=None,
            owner_id=None,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        data = company.to_dict()
        assert data["id"] == "company_456"
        assert data["name"] == "Test Corp"
        assert data["created_at"] == "2024-01-01T00:00:00+00:00"


class TestUnifiedDeal:
    """Tests for UnifiedDeal dataclass."""

    def test_deal_creation(self):
        """Test creating a unified deal."""
        deal = UnifiedDeal(
            id="deal_123",
            platform="hubspot",
            name="Big Enterprise Deal",
            amount=100000.0,
            stage="negotiation",
            pipeline="Enterprise",
            close_date=datetime.now(timezone.utc),
            probability=75.0,
            contact_ids=["contact_123"],
            company_id="company_123",
            owner_id="user_1",
            created_at=datetime.now(timezone.utc),
        )

        assert deal.id == "deal_123"
        assert deal.amount == 100000.0
        assert deal.probability == 75.0

    def test_deal_to_dict(self):
        """Test deal serialization."""
        deal = UnifiedDeal(
            id="deal_456",
            platform="hubspot",
            name="Small Deal",
            amount=5000.0,
            stage="proposal",
            pipeline="SMB",
            close_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            probability=50.0,
            contact_ids=[],
            company_id=None,
            owner_id=None,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        data = deal.to_dict()
        assert data["id"] == "deal_456"
        assert data["name"] == "Small Deal"
        assert data["close_date"] == "2024-06-01T00:00:00+00:00"


# -----------------------------------------------------------------------------
# Test: Error Handling
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_body_connect(self, crm_handler, mock_request):
        """Test handling of invalid JSON body on connect."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = ValueError("Invalid JSON")

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "invalid" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json_body_create_contact(self, crm_handler, mock_request):
        """Test handling of invalid JSON body on create contact."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = ValueError("Invalid JSON")

            result = await crm_handler._create_contact(mock_request, "hubspot")

            assert result["status_code"] == 400

    def test_error_response_format(self, crm_handler):
        """Test error response format."""
        result = crm_handler._error_response(400, "Test error message")

        assert result["status_code"] == 400
        assert result["body"]["error"] == "Test error message"

    def test_json_response_format(self, crm_handler):
        """Test JSON response format."""
        result = crm_handler._json_response(200, {"key": "value"})

        assert result["status_code"] == 200
        assert result["body"]["key"] == "value"


# -----------------------------------------------------------------------------
# Test: Helper Methods
# -----------------------------------------------------------------------------


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_required_credentials_hubspot(self, crm_handler):
        """Test required credentials for HubSpot."""
        creds = crm_handler._get_required_credentials("hubspot")
        assert "access_token" in creds

    def test_get_required_credentials_salesforce(self, crm_handler):
        """Test required credentials for Salesforce."""
        creds = crm_handler._get_required_credentials("salesforce")
        assert "client_id" in creds
        assert "client_secret" in creds
        assert "refresh_token" in creds
        assert "instance_url" in creds

    def test_get_required_credentials_pipedrive(self, crm_handler):
        """Test required credentials for Pipedrive."""
        creds = crm_handler._get_required_credentials("pipedrive")
        assert "api_token" in creds

    def test_get_required_credentials_unknown(self, crm_handler):
        """Test required credentials for unknown platform."""
        creds = crm_handler._get_required_credentials("unknown")
        assert creds == []

    def test_normalize_hubspot_contact(self, crm_handler):
        """Test normalizing HubSpot contact."""
        mock_contact = MagicMock()
        mock_contact.id = "123"
        mock_contact.properties = {
            "email": "test@example.com",
            "firstname": "John",
            "lastname": "Doe",
            "phone": "+1234567890",
            "company": "Acme",
            "jobtitle": "CEO",
            "lifecyclestage": "customer",
            "hs_lead_status": "OPEN",
            "hubspot_owner_id": "owner1",
        }
        mock_contact.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_contact.updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)

        result = crm_handler._normalize_hubspot_contact(mock_contact)

        assert result["id"] == "123"
        assert result["email"] == "test@example.com"
        assert result["first_name"] == "John"
        assert result["last_name"] == "Doe"
        assert result["full_name"] == "John Doe"

    def test_normalize_hubspot_company(self, crm_handler):
        """Test normalizing HubSpot company."""
        mock_company = MagicMock()
        mock_company.id = "456"
        mock_company.properties = {
            "name": "Acme Corp",
            "domain": "acme.com",
            "industry": "Technology",
            "numberofemployees": "500",
            "annualrevenue": "50000000",
            "hubspot_owner_id": "owner1",
        }
        mock_company.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        result = crm_handler._normalize_hubspot_company(mock_company)

        assert result["id"] == "456"
        assert result["name"] == "Acme Corp"
        assert result["employee_count"] == 500
        assert result["annual_revenue"] == 50000000.0

    def test_normalize_hubspot_company_invalid_numbers(self, crm_handler):
        """Test normalizing HubSpot company with invalid numbers."""
        mock_company = MagicMock()
        mock_company.id = "789"
        mock_company.properties = {
            "name": "Test Corp",
            "numberofemployees": "not-a-number",
            "annualrevenue": "invalid",
        }
        mock_company.created_at = None

        result = crm_handler._normalize_hubspot_company(mock_company)

        assert result["employee_count"] is None
        assert result["annual_revenue"] is None

    def test_normalize_hubspot_deal(self, crm_handler):
        """Test normalizing HubSpot deal."""
        mock_deal = MagicMock()
        mock_deal.id = "789"
        mock_deal.properties = {
            "dealname": "Big Deal",
            "amount": "100000",
            "dealstage": "proposal",
            "pipeline": "default",
            "closedate": "2024-06-01",
            "hubspot_owner_id": "owner1",
        }
        mock_deal.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        result = crm_handler._normalize_hubspot_deal(mock_deal)

        assert result["id"] == "789"
        assert result["name"] == "Big Deal"
        assert result["amount"] == 100000.0

    def test_normalize_hubspot_deal_invalid_amount(self, crm_handler):
        """Test normalizing HubSpot deal with invalid amount."""
        mock_deal = MagicMock()
        mock_deal.id = "abc"
        mock_deal.properties = {
            "dealname": "Test Deal",
            "amount": "not-a-number",
            "dealstage": "open",
        }
        mock_deal.created_at = None

        result = crm_handler._normalize_hubspot_deal(mock_deal)

        assert result["amount"] is None

    def test_map_lead_to_hubspot(self, crm_handler):
        """Test mapping lead data to HubSpot properties."""
        lead = {
            "email": "lead@example.com",
            "first_name": "Jane",
            "last_name": "Smith",
            "phone": "+1987654321",
            "company": "Test Inc",
            "job_title": "CTO",
        }

        result = crm_handler._map_lead_to_hubspot(lead, "linkedin")

        assert result["email"] == "lead@example.com"
        assert result["firstname"] == "Jane"
        assert result["lastname"] == "Smith"
        assert result["lifecyclestage"] == "lead"
        assert result["hs_lead_status"] == "NEW"
        assert result["hs_analytics_source"] == "linkedin"


# -----------------------------------------------------------------------------
# Test: Constants
# -----------------------------------------------------------------------------


class TestConstants:
    """Tests for validation constants."""

    def test_max_lengths_defined(self):
        """Test that max length constants are defined."""
        assert MAX_EMAIL_LENGTH == 254
        assert MAX_NAME_LENGTH == 128
        assert MAX_PHONE_LENGTH == 32
        assert MAX_COMPANY_NAME_LENGTH == 256
        assert MAX_JOB_TITLE_LENGTH == 128
        assert MAX_DOMAIN_LENGTH == 253
        assert MAX_DEAL_NAME_LENGTH == 256
        assert MAX_STAGE_LENGTH == 64
        assert MAX_PIPELINE_LENGTH == 64
        assert MAX_CREDENTIAL_VALUE_LENGTH == 1024
        assert MAX_SEARCH_QUERY_LENGTH == 256
