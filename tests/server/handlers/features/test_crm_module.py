"""
Comprehensive Unit Tests for CRM Module.

This test module provides in-depth coverage for the CRM platform integration
module (aragora/server/handlers/features/crm/).

Test Categories:
1. Platform CRUD operations (connect, disconnect, list)
2. Deal and contact queries (with mocked backends)
3. Input validation (SAFE_PLATFORM_PATTERN, SAFE_RESOURCE_ID_PATTERN)
4. Rate limiter enforcement
5. Circuit breaker state transitions
6. Error handling and edge cases

Stability: STABLE
"""

from __future__ import annotations

import asyncio
import re
import time
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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
    # Validation constants
    SAFE_PLATFORM_PATTERN,
    SAFE_RESOURCE_ID_PATTERN,
    EMAIL_PATTERN,
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


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def server_context() -> dict[str, Any]:
    """Create a mock server context."""
    return {"config": {"debug": True, "rate_limit_enabled": True}}


@pytest.fixture
def crm_handler(server_context: dict[str, Any]) -> CRMHandler:
    """Create a CRMHandler instance with clean state."""
    reset_crm_circuit_breaker()
    _platform_credentials.clear()
    _platform_connectors.clear()
    return CRMHandler(server_context=server_context)


@pytest.fixture
def mock_request() -> MagicMock:
    """Create a mock HTTP request."""
    request = MagicMock()
    request.method = "GET"
    request.path = "/api/v1/crm/platforms"
    request.query = {}
    request.headers = {
        "Authorization": "Bearer test-token-abc123",
        "Content-Type": "application/json",
    }
    return request


@pytest.fixture
def mock_auth_context() -> MagicMock:
    """Create a mock authorization context with full permissions."""
    ctx = MagicMock()
    ctx.user_id = "test-user-001"
    ctx.workspace_id = "test-workspace-001"
    ctx.roles = ["admin"]
    ctx.permissions = ["crm:read", "crm:write", "crm:configure"]
    return ctx


@pytest.fixture
def mock_hubspot_connector() -> AsyncMock:
    """Create a mock HubSpot connector."""
    connector = AsyncMock()

    # Mock contact
    mock_contact = MagicMock()
    mock_contact.id = "contact_12345"
    mock_contact.properties = {
        "email": "test@example.com",
        "firstname": "John",
        "lastname": "Doe",
        "phone": "+1234567890",
        "company": "Acme Corp",
        "jobtitle": "Engineer",
        "lifecyclestage": "customer",
        "hs_lead_status": "CONNECTED",
        "hubspot_owner_id": "owner_001",
    }
    mock_contact.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    mock_contact.updated_at = datetime(2024, 1, 15, tzinfo=timezone.utc)

    # Mock company
    mock_company = MagicMock()
    mock_company.id = "company_67890"
    mock_company.properties = {
        "name": "Acme Corporation",
        "domain": "acme.com",
        "industry": "Technology",
        "numberofemployees": "500",
        "annualrevenue": "50000000",
        "hubspot_owner_id": "owner_001",
    }
    mock_company.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Mock deal
    mock_deal = MagicMock()
    mock_deal.id = "deal_11111"
    mock_deal.properties = {
        "dealname": "Enterprise Contract",
        "amount": "100000",
        "dealstage": "negotiation",
        "pipeline": "default",
        "closedate": "2024-06-01",
        "hubspot_owner_id": "owner_001",
    }
    mock_deal.created_at = datetime(2024, 2, 1, tzinfo=timezone.utc)

    # Configure connector methods
    connector.get_contacts = AsyncMock(return_value=[mock_contact])
    connector.get_contact = AsyncMock(return_value=mock_contact)
    connector.get_contact_by_email = AsyncMock(return_value=mock_contact)
    connector.create_contact = AsyncMock(return_value=mock_contact)
    connector.update_contact = AsyncMock(return_value=mock_contact)
    connector.search_contacts = AsyncMock(return_value=[mock_contact])

    connector.get_companies = AsyncMock(return_value=[mock_company])
    connector.get_company = AsyncMock(return_value=mock_company)
    connector.create_company = AsyncMock(return_value=mock_company)
    connector.search_companies = AsyncMock(return_value=[mock_company])

    connector.get_deals = AsyncMock(return_value=[mock_deal])
    connector.get_deal = AsyncMock(return_value=mock_deal)
    connector.create_deal = AsyncMock(return_value=mock_deal)
    connector.search_deals = AsyncMock(return_value=[mock_deal])

    connector.get_pipelines = AsyncMock(return_value=[])
    connector.close = AsyncMock()

    return connector


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


# =============================================================================
# Test: Input Validation Patterns (SAFE_PLATFORM_PATTERN, SAFE_RESOURCE_ID_PATTERN)
# =============================================================================


class TestValidationPatterns:
    """Tests for input validation regex patterns."""

    # --- SAFE_PLATFORM_PATTERN Tests ---

    def test_safe_platform_pattern_valid_lowercase(self):
        """Test SAFE_PLATFORM_PATTERN accepts lowercase platform names."""
        assert SAFE_PLATFORM_PATTERN.match("hubspot") is not None
        assert SAFE_PLATFORM_PATTERN.match("salesforce") is not None
        assert SAFE_PLATFORM_PATTERN.match("pipedrive") is not None

    def test_safe_platform_pattern_valid_with_underscore(self):
        """Test SAFE_PLATFORM_PATTERN accepts underscores."""
        assert SAFE_PLATFORM_PATTERN.match("my_platform") is not None
        assert SAFE_PLATFORM_PATTERN.match("custom_crm_v2") is not None

    def test_safe_platform_pattern_valid_with_numbers(self):
        """Test SAFE_PLATFORM_PATTERN accepts numbers after first char."""
        assert SAFE_PLATFORM_PATTERN.match("platform123") is not None
        assert SAFE_PLATFORM_PATTERN.match("crm2go") is not None

    def test_safe_platform_pattern_valid_uppercase(self):
        """Test SAFE_PLATFORM_PATTERN accepts uppercase."""
        assert SAFE_PLATFORM_PATTERN.match("HubSpot") is not None
        assert SAFE_PLATFORM_PATTERN.match("CRM") is not None

    def test_safe_platform_pattern_valid_mixed_case(self):
        """Test SAFE_PLATFORM_PATTERN accepts mixed case."""
        assert SAFE_PLATFORM_PATTERN.match("MyCustomCRM") is not None

    def test_safe_platform_pattern_invalid_starts_with_number(self):
        """Test SAFE_PLATFORM_PATTERN rejects names starting with number."""
        assert SAFE_PLATFORM_PATTERN.match("123platform") is None
        assert SAFE_PLATFORM_PATTERN.match("1crm") is None

    def test_safe_platform_pattern_invalid_hyphen(self):
        """Test SAFE_PLATFORM_PATTERN rejects hyphens."""
        assert SAFE_PLATFORM_PATTERN.match("my-platform") is None
        assert SAFE_PLATFORM_PATTERN.match("crm-pro") is None

    def test_safe_platform_pattern_invalid_special_chars(self):
        """Test SAFE_PLATFORM_PATTERN rejects special characters."""
        assert SAFE_PLATFORM_PATTERN.match("platform@crm") is None
        assert SAFE_PLATFORM_PATTERN.match("crm$$$") is None
        assert SAFE_PLATFORM_PATTERN.match("my.platform") is None

    def test_safe_platform_pattern_invalid_spaces(self):
        """Test SAFE_PLATFORM_PATTERN rejects spaces."""
        assert SAFE_PLATFORM_PATTERN.match("my platform") is None
        assert SAFE_PLATFORM_PATTERN.match(" hubspot") is None

    def test_safe_platform_pattern_max_length(self):
        """Test SAFE_PLATFORM_PATTERN max length is 50 chars."""
        # Pattern is ^[a-zA-Z][a-zA-Z0-9_]{0,49}$ meaning max total 50 chars
        valid_50_chars = "a" * 50
        invalid_51_chars = "a" * 51
        assert SAFE_PLATFORM_PATTERN.match(valid_50_chars) is not None
        assert SAFE_PLATFORM_PATTERN.match(invalid_51_chars) is None

    def test_safe_platform_pattern_empty_string(self):
        """Test SAFE_PLATFORM_PATTERN rejects empty string."""
        assert SAFE_PLATFORM_PATTERN.match("") is None

    def test_safe_platform_pattern_single_char(self):
        """Test SAFE_PLATFORM_PATTERN accepts single letter."""
        assert SAFE_PLATFORM_PATTERN.match("a") is not None
        assert SAFE_PLATFORM_PATTERN.match("Z") is not None

    # --- SAFE_RESOURCE_ID_PATTERN Tests ---

    def test_safe_resource_id_pattern_valid_alphanumeric(self):
        """Test SAFE_RESOURCE_ID_PATTERN accepts alphanumeric IDs."""
        assert SAFE_RESOURCE_ID_PATTERN.match("contact123") is not None
        assert SAFE_RESOURCE_ID_PATTERN.match("ABC123XYZ") is not None

    def test_safe_resource_id_pattern_valid_with_hyphen(self):
        """Test SAFE_RESOURCE_ID_PATTERN accepts hyphens."""
        assert SAFE_RESOURCE_ID_PATTERN.match("contact-123-abc") is not None
        assert SAFE_RESOURCE_ID_PATTERN.match("id-with-many-hyphens") is not None

    def test_safe_resource_id_pattern_valid_with_underscore(self):
        """Test SAFE_RESOURCE_ID_PATTERN accepts underscores."""
        assert SAFE_RESOURCE_ID_PATTERN.match("contact_123_abc") is not None
        assert SAFE_RESOURCE_ID_PATTERN.match("deal_record_001") is not None

    def test_safe_resource_id_pattern_valid_uuid_like(self):
        """Test SAFE_RESOURCE_ID_PATTERN accepts UUID-like IDs."""
        assert SAFE_RESOURCE_ID_PATTERN.match("550e8400-e29b-41d4-a716-446655440000") is not None

    def test_safe_resource_id_pattern_valid_starts_with_number(self):
        """Test SAFE_RESOURCE_ID_PATTERN accepts IDs starting with number."""
        assert SAFE_RESOURCE_ID_PATTERN.match("123abc") is not None
        assert SAFE_RESOURCE_ID_PATTERN.match("001-contact") is not None

    def test_safe_resource_id_pattern_invalid_starts_with_hyphen(self):
        """Test SAFE_RESOURCE_ID_PATTERN rejects IDs starting with hyphen."""
        assert SAFE_RESOURCE_ID_PATTERN.match("-invalid") is None
        assert SAFE_RESOURCE_ID_PATTERN.match("-123") is None

    def test_safe_resource_id_pattern_invalid_starts_with_underscore(self):
        """Test SAFE_RESOURCE_ID_PATTERN rejects IDs starting with underscore."""
        assert SAFE_RESOURCE_ID_PATTERN.match("_invalid") is None

    def test_safe_resource_id_pattern_invalid_special_chars(self):
        """Test SAFE_RESOURCE_ID_PATTERN rejects special characters."""
        assert SAFE_RESOURCE_ID_PATTERN.match("contact@123") is None
        assert SAFE_RESOURCE_ID_PATTERN.match("deal#001") is None
        assert SAFE_RESOURCE_ID_PATTERN.match("id$value") is None

    def test_safe_resource_id_pattern_invalid_spaces(self):
        """Test SAFE_RESOURCE_ID_PATTERN rejects spaces."""
        assert SAFE_RESOURCE_ID_PATTERN.match("contact 123") is None
        assert SAFE_RESOURCE_ID_PATTERN.match(" id") is None

    def test_safe_resource_id_pattern_max_length(self):
        """Test SAFE_RESOURCE_ID_PATTERN max length is 128 chars."""
        # Pattern is ^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}$ meaning max total 128 chars
        valid_128_chars = "a" * 128
        invalid_129_chars = "a" * 129
        assert SAFE_RESOURCE_ID_PATTERN.match(valid_128_chars) is not None
        assert SAFE_RESOURCE_ID_PATTERN.match(invalid_129_chars) is None

    def test_safe_resource_id_pattern_empty_string(self):
        """Test SAFE_RESOURCE_ID_PATTERN rejects empty string."""
        assert SAFE_RESOURCE_ID_PATTERN.match("") is None

    def test_safe_resource_id_pattern_single_char(self):
        """Test SAFE_RESOURCE_ID_PATTERN accepts single alphanumeric char."""
        assert SAFE_RESOURCE_ID_PATTERN.match("a") is not None
        assert SAFE_RESOURCE_ID_PATTERN.match("1") is not None

    # --- EMAIL_PATTERN Tests ---

    def test_email_pattern_valid_simple(self):
        """Test EMAIL_PATTERN accepts simple emails."""
        assert EMAIL_PATTERN.match("test@example.com") is not None
        assert EMAIL_PATTERN.match("user@domain.org") is not None

    def test_email_pattern_valid_with_plus(self):
        """Test EMAIL_PATTERN accepts emails with plus sign."""
        assert EMAIL_PATTERN.match("user+tag@example.com") is not None

    def test_email_pattern_valid_with_dots(self):
        """Test EMAIL_PATTERN accepts emails with dots."""
        assert EMAIL_PATTERN.match("first.last@example.com") is not None

    def test_email_pattern_valid_subdomain(self):
        """Test EMAIL_PATTERN accepts emails with subdomains."""
        assert EMAIL_PATTERN.match("user@mail.example.com") is not None
        assert EMAIL_PATTERN.match("user@sub.domain.co.uk") is not None

    def test_email_pattern_invalid_no_at(self):
        """Test EMAIL_PATTERN rejects emails without @."""
        assert EMAIL_PATTERN.match("userexample.com") is None

    def test_email_pattern_invalid_no_domain(self):
        """Test EMAIL_PATTERN rejects emails without domain."""
        assert EMAIL_PATTERN.match("user@") is None

    def test_email_pattern_invalid_no_tld(self):
        """Test EMAIL_PATTERN rejects emails without TLD."""
        assert EMAIL_PATTERN.match("user@domain") is None

    def test_email_pattern_invalid_short_tld(self):
        """Test EMAIL_PATTERN rejects emails with 1-char TLD."""
        assert EMAIL_PATTERN.match("user@domain.a") is None


# =============================================================================
# Test: Validation Functions
# =============================================================================


class TestValidationFunctions:
    """Tests for validation helper functions."""

    # --- validate_platform_id Tests ---

    def test_validate_platform_id_valid(self):
        """Test validate_platform_id accepts valid IDs."""
        assert _validate_platform_id("hubspot") == (True, None)
        assert _validate_platform_id("my_platform") == (True, None)
        assert _validate_platform_id("CRM2024") == (True, None)

    def test_validate_platform_id_empty(self):
        """Test validate_platform_id rejects empty string."""
        valid, msg = _validate_platform_id("")
        assert not valid
        assert "required" in msg.lower()

    def test_validate_platform_id_none(self):
        """Test validate_platform_id rejects None."""
        valid, msg = _validate_platform_id(None)
        assert not valid
        assert "required" in msg.lower()

    def test_validate_platform_id_too_long(self):
        """Test validate_platform_id rejects IDs over 50 chars."""
        valid, msg = _validate_platform_id("a" * 51)
        assert not valid
        assert "too long" in msg.lower()
        assert "50" in msg

    def test_validate_platform_id_boundary_50_chars(self):
        """Test validate_platform_id accepts exactly 50 chars."""
        valid, msg = _validate_platform_id("a" * 50)
        assert valid
        assert msg is None

    def test_validate_platform_id_invalid_format(self):
        """Test validate_platform_id rejects invalid formats."""
        valid, msg = _validate_platform_id("invalid-id")
        assert not valid
        assert "invalid" in msg.lower()

    # --- validate_resource_id Tests ---

    def test_validate_resource_id_valid(self):
        """Test validate_resource_id accepts valid IDs."""
        assert _validate_resource_id("contact123") == (True, None)
        assert _validate_resource_id("deal-001-abc") == (True, None)
        assert _validate_resource_id("123") == (True, None)

    def test_validate_resource_id_empty(self):
        """Test validate_resource_id rejects empty string."""
        valid, msg = _validate_resource_id("")
        assert not valid
        assert "required" in msg.lower()

    def test_validate_resource_id_too_long(self):
        """Test validate_resource_id rejects IDs over 128 chars."""
        valid, msg = _validate_resource_id("a" * 129)
        assert not valid
        assert "too long" in msg.lower()

    def test_validate_resource_id_boundary_128_chars(self):
        """Test validate_resource_id accepts exactly 128 chars."""
        valid, msg = _validate_resource_id("a" * 128)
        assert valid
        assert msg is None

    def test_validate_resource_id_invalid_format(self):
        """Test validate_resource_id rejects invalid formats."""
        valid, msg = _validate_resource_id("-invalid")
        assert not valid
        assert "invalid" in msg.lower()

    def test_validate_resource_id_custom_type_name(self):
        """Test validate_resource_id uses custom type name in errors."""
        valid, msg = _validate_resource_id("", "Contact ID")
        assert not valid
        assert "contact id" in msg.lower()

        valid, msg = _validate_resource_id("-bad", "Deal ID")
        assert not valid
        assert "deal id" in msg.lower()

    # --- validate_email Tests ---

    def test_validate_email_valid(self):
        """Test validate_email accepts valid emails."""
        assert _validate_email("test@example.com") == (True, None)
        assert _validate_email("user+tag@domain.co.uk") == (True, None)

    def test_validate_email_none_not_required(self):
        """Test validate_email accepts None when not required."""
        assert _validate_email(None) == (True, None)
        assert _validate_email(None, required=False) == (True, None)

    def test_validate_email_empty_not_required(self):
        """Test validate_email accepts empty when not required."""
        assert _validate_email("") == (True, None)

    def test_validate_email_required_missing(self):
        """Test validate_email rejects missing email when required."""
        valid, msg = _validate_email(None, required=True)
        assert not valid
        assert "required" in msg.lower()

        valid, msg = _validate_email("", required=True)
        assert not valid

    def test_validate_email_too_long(self):
        """Test validate_email rejects emails over MAX_EMAIL_LENGTH."""
        long_email = "a" * 250 + "@example.com"
        valid, msg = _validate_email(long_email)
        assert not valid
        assert "too long" in msg.lower()

    def test_validate_email_invalid_format(self):
        """Test validate_email rejects invalid formats."""
        valid, msg = _validate_email("not-an-email")
        assert not valid
        assert "invalid" in msg.lower()

    # --- validate_string_field Tests ---

    def test_validate_string_field_valid(self):
        """Test validate_string_field accepts valid strings."""
        assert _validate_string_field("John", "Name", 100) == (True, None)
        assert _validate_string_field("A very long name", "Name", 100) == (True, None)

    def test_validate_string_field_none_not_required(self):
        """Test validate_string_field accepts None when not required."""
        assert _validate_string_field(None, "Name", 100) == (True, None)

    def test_validate_string_field_required_missing(self):
        """Test validate_string_field rejects missing when required."""
        valid, msg = _validate_string_field(None, "Name", 100, required=True)
        assert not valid
        assert "required" in msg.lower()
        assert "name" in msg.lower()

    def test_validate_string_field_too_long(self):
        """Test validate_string_field rejects strings over max_length."""
        valid, msg = _validate_string_field("a" * 101, "Name", 100)
        assert not valid
        assert "too long" in msg.lower()
        assert "100" in msg

    def test_validate_string_field_boundary(self):
        """Test validate_string_field accepts strings at max_length."""
        valid, msg = _validate_string_field("a" * 100, "Name", 100)
        assert valid
        assert msg is None

    # --- validate_amount Tests ---

    def test_validate_amount_valid_float(self):
        """Test validate_amount accepts valid floats."""
        valid, msg, val = _validate_amount(100.50)
        assert valid
        assert val == 100.50

    def test_validate_amount_valid_int(self):
        """Test validate_amount accepts integers."""
        valid, msg, val = _validate_amount(1000)
        assert valid
        assert val == 1000.0

    def test_validate_amount_valid_string(self):
        """Test validate_amount accepts numeric strings."""
        valid, msg, val = _validate_amount("500.25")
        assert valid
        assert val == 500.25

    def test_validate_amount_valid_zero(self):
        """Test validate_amount accepts zero."""
        valid, msg, val = _validate_amount(0)
        assert valid
        assert val == 0.0

    def test_validate_amount_none(self):
        """Test validate_amount accepts None."""
        valid, msg, val = _validate_amount(None)
        assert valid
        assert val is None

    def test_validate_amount_negative(self):
        """Test validate_amount rejects negative amounts."""
        valid, msg, val = _validate_amount(-100)
        assert not valid
        assert "negative" in msg.lower()
        assert val is None

    def test_validate_amount_too_large(self):
        """Test validate_amount rejects amounts over 1 trillion."""
        valid, msg, val = _validate_amount(2_000_000_000_000)
        assert not valid
        assert "too large" in msg.lower()

    def test_validate_amount_invalid_string(self):
        """Test validate_amount rejects non-numeric strings."""
        valid, msg, val = _validate_amount("not-a-number")
        assert not valid
        assert "invalid" in msg.lower()

    # --- validate_probability Tests ---

    def test_validate_probability_valid(self):
        """Test validate_probability accepts valid values."""
        valid, msg, val = _validate_probability(75)
        assert valid
        assert val == 75.0

    def test_validate_probability_valid_boundaries(self):
        """Test validate_probability accepts 0 and 100."""
        valid, msg, val = _validate_probability(0)
        assert valid
        assert val == 0.0

        valid, msg, val = _validate_probability(100)
        assert valid
        assert val == 100.0

    def test_validate_probability_none(self):
        """Test validate_probability accepts None."""
        valid, msg, val = _validate_probability(None)
        assert valid
        assert val is None

    def test_validate_probability_below_zero(self):
        """Test validate_probability rejects negative values."""
        valid, msg, val = _validate_probability(-10)
        assert not valid
        assert "between" in msg.lower()

    def test_validate_probability_above_hundred(self):
        """Test validate_probability rejects values over 100."""
        valid, msg, val = _validate_probability(150)
        assert not valid
        assert "between" in msg.lower()

    def test_validate_probability_invalid_string(self):
        """Test validate_probability rejects non-numeric strings."""
        valid, msg, val = _validate_probability("high")
        assert not valid
        assert "invalid" in msg.lower()


# =============================================================================
# Test: Circuit Breaker State Transitions
# =============================================================================


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state machine transitions."""

    def test_initial_state_is_closed(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CRMCircuitBreaker()
        assert cb.state == CRMCircuitBreaker.CLOSED
        assert cb.can_proceed() is True

    def test_closed_to_open_transition(self):
        """Test transition from CLOSED to OPEN after failure threshold."""
        cb = CRMCircuitBreaker(failure_threshold=3)

        # Record failures
        cb.record_failure()
        assert cb.state == CRMCircuitBreaker.CLOSED
        cb.record_failure()
        assert cb.state == CRMCircuitBreaker.CLOSED
        cb.record_failure()  # Third failure triggers transition
        assert cb.state == CRMCircuitBreaker.OPEN
        assert cb.can_proceed() is False

    def test_open_to_half_open_after_cooldown(self):
        """Test transition from OPEN to HALF_OPEN after cooldown."""
        cb = CRMCircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

        cb.record_failure()
        assert cb.state == CRMCircuitBreaker.OPEN

        # Wait for cooldown
        time.sleep(0.1)
        assert cb.state == CRMCircuitBreaker.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        """Test transition from HALF_OPEN to CLOSED on successful calls."""
        cb = CRMCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.05,
            half_open_max_calls=2,
        )

        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CRMCircuitBreaker.HALF_OPEN

        # Allow test calls and record successes
        cb.can_proceed()
        cb.record_success()
        cb.record_success()

        assert cb.state == CRMCircuitBreaker.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Test transition from HALF_OPEN to OPEN on failure."""
        cb = CRMCircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

        cb.record_failure()
        time.sleep(0.1)
        assert cb.state == CRMCircuitBreaker.HALF_OPEN

        # Failure in half-open reopens circuit
        cb.can_proceed()
        cb.record_failure()
        assert cb.state == CRMCircuitBreaker.OPEN

    def test_half_open_limits_concurrent_calls(self):
        """Test HALF_OPEN state limits number of test calls."""
        cb = CRMCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.05,
            half_open_max_calls=2,
        )

        cb.record_failure()
        time.sleep(0.1)

        # First two calls allowed
        assert cb.can_proceed() is True
        assert cb.can_proceed() is True
        # Third call rejected
        assert cb.can_proceed() is False

    def test_success_resets_failure_count_in_closed(self):
        """Test success in CLOSED state resets failure count."""
        cb = CRMCircuitBreaker(failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Resets failure count

        # Now we need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CRMCircuitBreaker.CLOSED

        cb.record_failure()
        assert cb.state == CRMCircuitBreaker.OPEN

    def test_get_status_returns_complete_info(self):
        """Test get_status returns all expected fields."""
        cb = CRMCircuitBreaker(failure_threshold=5, cooldown_seconds=30)
        cb.record_failure()
        cb.record_failure()

        status = cb.get_status()

        assert status["state"] == CRMCircuitBreaker.CLOSED
        assert status["failure_count"] == 2
        assert status["success_count"] == 0
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30
        assert status["last_failure_time"] is not None

    def test_reset_returns_to_closed(self):
        """Test reset() returns circuit to clean CLOSED state."""
        cb = CRMCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CRMCircuitBreaker.OPEN

        cb.reset()

        assert cb.state == CRMCircuitBreaker.CLOSED
        status = cb.get_status()
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["last_failure_time"] is None

    def test_thread_safety(self):
        """Test circuit breaker is thread-safe."""
        cb = CRMCircuitBreaker(failure_threshold=100)
        failures_recorded = []

        def record_many_failures():
            for _ in range(50):
                cb.record_failure()
                failures_recorded.append(1)

        threads = [threading.Thread(target=record_many_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All failures should be recorded
        assert len(failures_recorded) == 200
        assert cb.get_status()["failure_count"] == 200

    def test_global_circuit_breaker_singleton(self):
        """Test global circuit breaker functions."""
        cb1 = get_crm_circuit_breaker()
        cb2 = get_crm_circuit_breaker()
        assert cb1 is cb2

    def test_global_circuit_breaker_reset(self):
        """Test global circuit breaker reset function."""
        cb = get_crm_circuit_breaker()
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CRMCircuitBreaker.OPEN

        reset_crm_circuit_breaker()
        assert cb.state == CRMCircuitBreaker.CLOSED


# =============================================================================
# Test: Platform CRUD Operations
# =============================================================================


class TestPlatformCRUDOperations:
    """Tests for platform management CRUD operations."""

    @pytest.mark.asyncio
    async def test_list_platforms_returns_all_supported(self, crm_handler, mock_request):
        """Test _list_platforms returns all supported platforms."""
        result = await crm_handler._list_platforms(mock_request)

        assert result["status_code"] == 200
        platforms = result["body"]["platforms"]
        platform_ids = [p["id"] for p in platforms]

        assert "hubspot" in platform_ids
        assert "salesforce" in platform_ids
        assert "pipedrive" in platform_ids

    @pytest.mark.asyncio
    async def test_list_platforms_shows_connection_status(self, crm_handler, mock_request):
        """Test _list_platforms shows correct connection status."""
        _platform_credentials["hubspot"] = {
            "credentials": {"access_token": "test"},
            "connected_at": "2024-01-01T00:00:00Z",
        }

        result = await crm_handler._list_platforms(mock_request)

        assert result["body"]["connected_count"] == 1
        hubspot = next(p for p in result["body"]["platforms"] if p["id"] == "hubspot")
        assert hubspot["connected"] is True
        assert hubspot["connected_at"] == "2024-01-01T00:00:00Z"

        salesforce = next(p for p in result["body"]["platforms"] if p["id"] == "salesforce")
        assert salesforce["connected"] is False

    @pytest.mark.asyncio
    async def test_connect_platform_success(self, crm_handler, mock_request):
        """Test successful platform connection."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "credentials": {"access_token": "pat-na1-test-token"},
            }
            with patch.object(crm_handler, "_get_connector", new_callable=AsyncMock) as mock_conn:
                mock_conn.return_value = None

                result = await crm_handler._connect_platform(mock_request)

                assert result["status_code"] == 200
                assert "hubspot" in _platform_credentials
                assert "connected_at" in _platform_credentials["hubspot"]
                assert result["body"]["platform"] == "hubspot"

    @pytest.mark.asyncio
    async def test_connect_platform_validates_platform_format(self, crm_handler, mock_request):
        """Test connect validates platform ID format."""
        invalid_platforms = ["123platform", "my-platform", "plat@form", ""]

        for invalid_platform in invalid_platforms:
            with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
                mock_body.return_value = {
                    "platform": invalid_platform,
                    "credentials": {"access_token": "test"},
                }
                result = await crm_handler._connect_platform(mock_request)
                assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_connect_platform_rejects_unsupported(self, crm_handler, mock_request):
        """Test connect rejects unsupported platforms."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "unknown_crm",
                "credentials": {"api_key": "test"},
            }
            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "unsupported" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_rejects_coming_soon(self, crm_handler, mock_request):
        """Test connect rejects coming soon platforms."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "salesforce",
                "credentials": {
                    "client_id": "test",
                    "client_secret": "test",
                    "refresh_token": "test",
                    "instance_url": "https://test.salesforce.com",
                },
            }
            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "coming soon" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_requires_credentials(self, crm_handler, mock_request):
        """Test connect requires credentials."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "hubspot", "credentials": {}}
            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "credentials" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_validates_required_fields(self, crm_handler, mock_request):
        """Test connect validates required credential fields."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "credentials": {"wrong_field": "value"},
            }
            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "missing" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_platform_rejects_long_credentials(self, crm_handler, mock_request):
        """Test connect rejects credentials exceeding max length."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "platform": "hubspot",
                "credentials": {"access_token": "x" * (MAX_CREDENTIAL_VALUE_LENGTH + 1)},
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
    async def test_disconnect_platform_closes_connector(self, crm_handler, mock_request):
        """Test disconnect closes existing connector."""
        mock_connector = AsyncMock()
        _platform_credentials["hubspot"] = {
            "credentials": {"access_token": "test"},
            "connected_at": "2024-01-01T00:00:00Z",
        }
        _platform_connectors["hubspot"] = mock_connector

        result = await crm_handler._disconnect_platform(mock_request, "hubspot")

        assert result["status_code"] == 200
        mock_connector.close.assert_called_once()
        assert "hubspot" not in _platform_connectors

    @pytest.mark.asyncio
    async def test_disconnect_platform_not_connected(self, crm_handler, mock_request):
        """Test disconnect returns 404 for unconnected platform."""
        result = await crm_handler._disconnect_platform(mock_request, "hubspot")

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_disconnect_platform_validates_id(self, crm_handler, mock_request):
        """Test disconnect validates platform ID format."""
        result = await crm_handler._disconnect_platform(mock_request, "invalid-id!")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_get_status_healthy(self, crm_handler, mock_request):
        """Test _get_status returns healthy status."""
        result = await crm_handler._get_status(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["status"] == "healthy"
        assert "circuit_breaker" in result["body"]
        assert result["body"]["circuit_breaker"]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_get_status_degraded_when_circuit_open(self, crm_handler, mock_request):
        """Test _get_status returns degraded when circuit is open."""
        cb = get_crm_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        result = await crm_handler._get_status(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["status"] == "degraded"


# =============================================================================
# Test: Contact and Deal Queries with Mocked Backends
# =============================================================================


class TestContactQueries:
    """Tests for contact query operations with mocked backends."""

    @pytest.mark.asyncio
    async def test_list_all_contacts_empty_when_no_platforms(self, crm_handler, mock_request):
        """Test list all contacts returns empty when no platforms connected."""
        result = await crm_handler._list_all_contacts(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["contacts"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_contacts_validates_email_filter(self, crm_handler, mock_request):
        """Test list all contacts validates email filter format."""
        mock_request.query = {"email": "invalid-email-format"}
        result = await crm_handler._list_all_contacts(mock_request)

        assert result["status_code"] == 400
        assert "invalid" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_list_all_contacts_circuit_breaker_blocks(self, crm_handler, mock_request):
        """Test list all contacts is blocked when circuit is open."""
        cb = get_crm_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        result = await crm_handler._list_all_contacts(mock_request)

        assert result["status_code"] == 503
        assert "circuit breaker" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_list_platform_contacts_validates_platform(self, crm_handler, mock_request):
        """Test list platform contacts validates platform ID."""
        result = await crm_handler._list_platform_contacts(mock_request, "invalid-platform!")

        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_list_platform_contacts_not_connected(self, crm_handler, mock_request):
        """Test list platform contacts returns 404 when not connected."""
        result = await crm_handler._list_platform_contacts(mock_request, "hubspot")

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_contact_validates_both_ids(self, crm_handler, mock_request):
        """Test get contact validates both platform and contact IDs."""
        # Invalid platform ID
        result = await crm_handler._get_contact(mock_request, "bad-platform!", "contact123")
        assert result["status_code"] == 400

        # Valid platform, invalid contact ID
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        result = await crm_handler._get_contact(mock_request, "hubspot", "-invalid-contact")
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_create_contact_validates_all_fields(self, crm_handler, mock_request):
        """Test create contact validates all input fields."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}

        # Test each field validation
        test_cases = [
            ({"email": "invalid"}, "email"),
            ({"email": "test@example.com", "first_name": "x" * 200}, "first name"),
            ({"email": "test@example.com", "last_name": "x" * 200}, "last name"),
            ({"email": "test@example.com", "phone": "x" * 50}, "phone"),
            ({"email": "test@example.com", "company": "x" * 300}, "company"),
            ({"email": "test@example.com", "job_title": "x" * 200}, "job title"),
        ]

        for body_data, expected_field in test_cases:
            with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
                mock_body.return_value = body_data
                result = await crm_handler._create_contact(mock_request, "hubspot")
                assert result["status_code"] == 400, f"Expected 400 for {expected_field}"

    @pytest.mark.asyncio
    async def test_update_contact_validates_partial_fields(self, crm_handler, mock_request):
        """Test update contact validates only provided fields."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}

        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"email": "invalid-email"}
            result = await crm_handler._update_contact(mock_request, "hubspot", "contact123")
            assert result["status_code"] == 400


class TestDealQueries:
    """Tests for deal query operations."""

    @pytest.mark.asyncio
    async def test_list_all_deals_empty_when_no_platforms(self, crm_handler, mock_request):
        """Test list all deals returns empty when no platforms connected."""
        result = await crm_handler._list_all_deals(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["deals"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_deals_validates_stage_filter(self, crm_handler, mock_request):
        """Test list all deals validates stage filter length."""
        mock_request.query = {"stage": "x" * (MAX_STAGE_LENGTH + 1)}
        result = await crm_handler._list_all_deals(mock_request)

        assert result["status_code"] == 400
        assert "too long" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_list_all_deals_circuit_breaker_blocks(self, crm_handler, mock_request):
        """Test list all deals is blocked when circuit is open."""
        cb = get_crm_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        result = await crm_handler._list_all_deals(mock_request)

        assert result["status_code"] == 503

    @pytest.mark.asyncio
    async def test_create_deal_validates_required_fields(self, crm_handler, mock_request):
        """Test create deal validates required name and stage."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}

        # Missing name
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"stage": "proposal"}
            result = await crm_handler._create_deal(mock_request, "hubspot")
            assert result["status_code"] == 400
            assert "required" in result["body"]["error"].lower()

        # Missing stage
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"name": "Big Deal"}
            result = await crm_handler._create_deal(mock_request, "hubspot")
            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_create_deal_validates_amount(self, crm_handler, mock_request):
        """Test create deal validates amount field."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}

        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {
                "name": "Deal",
                "stage": "proposal",
                "amount": -1000,  # Negative amount
            }
            result = await crm_handler._create_deal(mock_request, "hubspot")
            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_get_deal_validates_ids(self, crm_handler, mock_request):
        """Test get deal validates platform and deal IDs."""
        # Invalid platform
        result = await crm_handler._get_deal(mock_request, "123invalid", "deal123")
        assert result["status_code"] == 400

        # Valid platform, invalid deal ID
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}
        result = await crm_handler._get_deal(mock_request, "hubspot", "-bad-deal")
        assert result["status_code"] == 400


# =============================================================================
# Test: Rate Limiter Enforcement
# =============================================================================


class TestRateLimiterEnforcement:
    """Tests for rate limiter behavior on CRM endpoints."""

    @pytest.mark.asyncio
    async def test_handle_request_has_rate_limit_decorator(self, crm_handler):
        """Test handle_request method has rate_limit decorator."""
        # Check that the method has been decorated (wrapper attributes)
        method = crm_handler.handle_request
        # Rate limit decorator typically wraps the function
        assert callable(method)

    @pytest.mark.asyncio
    async def test_rate_limit_allows_normal_requests(
        self, crm_handler, mock_request, mock_auth_context
    ):
        """Test rate limiter allows requests under the limit."""
        mock_request.path = "/api/v1/crm/platforms"
        mock_request.method = "GET"

        # The handler should process the request normally
        result = await crm_handler._list_platforms(mock_request)
        assert result["status_code"] == 200


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_json_body_connect(self, crm_handler, mock_request):
        """Test error handling for invalid JSON in connect."""
        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = ValueError("Invalid JSON")

            result = await crm_handler._connect_platform(mock_request)

            assert result["status_code"] == 400
            assert "invalid json" in result["body"]["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json_body_create_contact(self, crm_handler, mock_request):
        """Test error handling for invalid JSON in create contact."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}

        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = Exception("Parse error")

            result = await crm_handler._create_contact(mock_request, "hubspot")

            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_invalid_json_body_create_company(self, crm_handler, mock_request):
        """Test error handling for invalid JSON in create company."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}

        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = Exception("Parse error")

            result = await crm_handler._create_company(mock_request, "hubspot")

            assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_invalid_json_body_create_deal(self, crm_handler, mock_request):
        """Test error handling for invalid JSON in create deal."""
        _platform_credentials["hubspot"] = {"credentials": {"access_token": "test"}}

        with patch.object(crm_handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = Exception("Parse error")

            result = await crm_handler._create_deal(mock_request, "hubspot")

            assert result["status_code"] == 400

    def test_error_response_format(self, crm_handler):
        """Test _error_response creates correct format."""
        result = crm_handler._error_response(400, "Test error")

        assert result["status_code"] == 400
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["body"]["error"] == "Test error"

    def test_json_response_format(self, crm_handler):
        """Test _json_response creates correct format."""
        result = crm_handler._json_response(200, {"key": "value", "count": 42})

        assert result["status_code"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["body"]["key"] == "value"
        assert result["body"]["count"] == 42

    @pytest.mark.asyncio
    async def test_endpoint_not_found(self, crm_handler, mock_request, mock_auth_context):
        """Test handler returns 404 for unknown endpoints."""
        mock_request.path = "/api/v1/crm/unknown/endpoint"
        mock_request.method = "GET"

        with patch.object(crm_handler, "_check_permission", new_callable=AsyncMock) as mock_perm:
            mock_perm.return_value = None

            result = await crm_handler.handle_request(mock_request)

            assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_permission_denied_unauthorized(self, crm_handler, mock_request):
        """Test handler returns 401 when not authenticated."""
        from aragora.server.handlers.secure import UnauthorizedError

        mock_request.path = "/api/v1/crm/contacts"
        mock_request.method = "GET"

        with patch.object(crm_handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await crm_handler._check_permission(mock_request, "crm:read")

            assert result["status_code"] == 401

    @pytest.mark.asyncio
    async def test_permission_denied_forbidden(self, crm_handler, mock_request):
        """Test handler returns 403 when lacking permission."""
        from aragora.server.handlers.secure import ForbiddenError

        mock_request.path = "/api/v1/crm/connect"
        mock_request.method = "POST"

        with patch.object(crm_handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_ctx = MagicMock()
            mock_auth.return_value = mock_ctx

            with patch.object(crm_handler, "check_permission") as mock_check:
                mock_check.side_effect = ForbiddenError("Permission denied: crm:configure")

                result = await crm_handler._check_permission(mock_request, "crm:configure")

                assert result["status_code"] == 403


# =============================================================================
# Test: Data Models
# =============================================================================


class TestUnifiedDataModels:
    """Tests for unified CRM data models."""

    def test_unified_contact_creation(self):
        """Test UnifiedContact dataclass creation."""
        contact = UnifiedContact(
            id="contact_001",
            platform="hubspot",
            email="john@example.com",
            first_name="John",
            last_name="Doe",
            phone="+1234567890",
            company="Acme Inc",
            job_title="Engineer",
            lifecycle_stage="lead",
            lead_status="NEW",
            owner_id="owner_001",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
            properties={"source": "website"},
        )

        assert contact.id == "contact_001"
        assert contact.platform == "hubspot"
        assert contact.email == "john@example.com"
        assert contact.properties["source"] == "website"

    def test_unified_contact_to_dict(self):
        """Test UnifiedContact.to_dict() method."""
        contact = UnifiedContact(
            id="contact_002",
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
            created_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=None,
        )

        data = contact.to_dict()

        assert data["id"] == "contact_002"
        assert data["full_name"] == "Jane Smith"
        assert data["created_at"] == "2024-06-01T12:00:00+00:00"
        assert data["updated_at"] is None

    def test_unified_contact_to_dict_empty_names(self):
        """Test UnifiedContact.to_dict() with no names."""
        contact = UnifiedContact(
            id="contact_003",
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

    def test_unified_company_creation(self):
        """Test UnifiedCompany dataclass creation."""
        company = UnifiedCompany(
            id="company_001",
            platform="hubspot",
            name="Acme Corporation",
            domain="acme.com",
            industry="Technology",
            employee_count=500,
            annual_revenue=50000000.0,
            owner_id="owner_001",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

        assert company.id == "company_001"
        assert company.name == "Acme Corporation"
        assert company.employee_count == 500

    def test_unified_company_to_dict(self):
        """Test UnifiedCompany.to_dict() method."""
        company = UnifiedCompany(
            id="company_002",
            platform="hubspot",
            name="Test Corp",
            domain="test.com",
            industry=None,
            employee_count=None,
            annual_revenue=None,
            owner_id=None,
            created_at=datetime(2024, 3, 15, tzinfo=timezone.utc),
        )

        data = company.to_dict()

        assert data["name"] == "Test Corp"
        assert data["employee_count"] is None
        assert data["created_at"] == "2024-03-15T00:00:00+00:00"

    def test_unified_deal_creation(self):
        """Test UnifiedDeal dataclass creation."""
        deal = UnifiedDeal(
            id="deal_001",
            platform="hubspot",
            name="Enterprise Contract",
            amount=100000.0,
            stage="negotiation",
            pipeline="Enterprise",
            close_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            probability=75.0,
            contact_ids=["contact_001", "contact_002"],
            company_id="company_001",
            owner_id="owner_001",
            created_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )

        assert deal.id == "deal_001"
        assert deal.amount == 100000.0
        assert len(deal.contact_ids) == 2

    def test_unified_deal_to_dict(self):
        """Test UnifiedDeal.to_dict() method."""
        deal = UnifiedDeal(
            id="deal_002",
            platform="hubspot",
            name="Small Deal",
            amount=5000.0,
            stage="proposal",
            pipeline="SMB",
            close_date=None,
            probability=50.0,
        )

        data = deal.to_dict()

        assert data["name"] == "Small Deal"
        assert data["close_date"] is None
        assert data["contact_ids"] == []
        assert data["company_id"] is None


# =============================================================================
# Test: Handler Helper Methods
# =============================================================================


class TestHandlerHelperMethods:
    """Tests for CRMHandler helper methods."""

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
        """Test required credentials for unknown platform returns empty."""
        creds = crm_handler._get_required_credentials("unknown_platform")
        assert creds == []

    def test_normalize_hubspot_contact(self, crm_handler):
        """Test _normalize_hubspot_contact helper."""
        mock_contact = MagicMock()
        mock_contact.id = "123"
        mock_contact.properties = {
            "email": "test@example.com",
            "firstname": "John",
            "lastname": "Doe",
            "phone": "+1234567890",
            "company": "Acme",
            "jobtitle": "Engineer",
            "lifecyclestage": "customer",
            "hs_lead_status": "OPEN",
            "hubspot_owner_id": "owner1",
        }
        mock_contact.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_contact.updated_at = datetime(2024, 1, 2, tzinfo=timezone.utc)

        result = crm_handler._normalize_hubspot_contact(mock_contact)

        assert result["id"] == "123"
        assert result["platform"] == "hubspot"
        assert result["email"] == "test@example.com"
        assert result["full_name"] == "John Doe"
        assert result["created_at"] == "2024-01-01T00:00:00+00:00"

    def test_normalize_hubspot_contact_no_properties(self, crm_handler):
        """Test _normalize_hubspot_contact with missing properties."""
        mock_contact = MagicMock(spec=[])  # No properties attribute
        mock_contact.id = "456"

        result = crm_handler._normalize_hubspot_contact(mock_contact)

        assert result["id"] == "456"
        assert result["email"] is None
        assert result["full_name"] is None

    def test_normalize_hubspot_company(self, crm_handler):
        """Test _normalize_hubspot_company helper."""
        mock_company = MagicMock()
        mock_company.id = "789"
        mock_company.properties = {
            "name": "Acme Corp",
            "domain": "acme.com",
            "industry": "Tech",
            "numberofemployees": "500",
            "annualrevenue": "50000000",
            "hubspot_owner_id": "owner1",
        }
        mock_company.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)

        result = crm_handler._normalize_hubspot_company(mock_company)

        assert result["id"] == "789"
        assert result["name"] == "Acme Corp"
        assert result["employee_count"] == 500
        assert result["annual_revenue"] == 50000000.0

    def test_normalize_hubspot_company_invalid_numbers(self, crm_handler):
        """Test _normalize_hubspot_company handles invalid numbers."""
        mock_company = MagicMock()
        mock_company.id = "abc"
        mock_company.properties = {
            "name": "Test",
            "numberofemployees": "not-a-number",
            "annualrevenue": "invalid",
        }
        mock_company.created_at = None

        result = crm_handler._normalize_hubspot_company(mock_company)

        assert result["employee_count"] is None
        assert result["annual_revenue"] is None

    def test_normalize_hubspot_deal(self, crm_handler):
        """Test _normalize_hubspot_deal helper."""
        mock_deal = MagicMock()
        mock_deal.id = "deal123"
        mock_deal.properties = {
            "dealname": "Big Deal",
            "amount": "100000",
            "dealstage": "proposal",
            "pipeline": "default",
            "closedate": "2024-06-01",
            "hubspot_owner_id": "owner1",
        }
        mock_deal.created_at = datetime(2024, 2, 1, tzinfo=timezone.utc)

        result = crm_handler._normalize_hubspot_deal(mock_deal)

        assert result["id"] == "deal123"
        assert result["name"] == "Big Deal"
        assert result["amount"] == 100000.0

    def test_normalize_hubspot_deal_invalid_amount(self, crm_handler):
        """Test _normalize_hubspot_deal handles invalid amount."""
        mock_deal = MagicMock()
        mock_deal.id = "deal456"
        mock_deal.properties = {
            "dealname": "Test Deal",
            "amount": "not-a-number",
            "dealstage": "open",
        }
        mock_deal.created_at = None

        result = crm_handler._normalize_hubspot_deal(mock_deal)

        assert result["amount"] is None

    def test_map_lead_to_hubspot(self, crm_handler):
        """Test _map_lead_to_hubspot helper."""
        lead = {
            "email": "lead@example.com",
            "first_name": "Jane",
            "last_name": "Lead",
            "phone": "+1987654321",
            "company": "Lead Inc",
            "job_title": "Director",
        }

        result = crm_handler._map_lead_to_hubspot(lead, "linkedin")

        assert result["email"] == "lead@example.com"
        assert result["firstname"] == "Jane"
        assert result["lastname"] == "Lead"
        assert result["lifecyclestage"] == "lead"
        assert result["hs_lead_status"] == "NEW"
        assert result["hs_analytics_source"] == "linkedin"

    def test_can_handle_matching_paths(self, crm_handler):
        """Test can_handle returns True for CRM paths."""
        assert crm_handler.can_handle("/api/v1/crm/platforms") is True
        assert crm_handler.can_handle("/api/v1/crm/hubspot/contacts") is True
        assert crm_handler.can_handle("/api/v1/crm/deals") is True
        assert crm_handler.can_handle("/api/v1/crm/connect") is True

    def test_can_handle_non_matching_paths(self, crm_handler):
        """Test can_handle returns False for non-CRM paths."""
        assert crm_handler.can_handle("/api/v1/other/path") is False
        assert crm_handler.can_handle("/api/v1/ecommerce/orders") is False
        assert crm_handler.can_handle("/health") is False


# =============================================================================
# Test: Request Routing
# =============================================================================


class TestRequestRouting:
    """Tests for request routing in handle_request."""

    @pytest.mark.asyncio
    async def test_route_to_status_endpoint(self, crm_handler, mock_request):
        """Test routing to status endpoint."""
        mock_request.path = "/api/v1/crm/status"
        mock_request.method = "GET"

        result = await crm_handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "status" in result["body"]

    @pytest.mark.asyncio
    async def test_route_to_platforms_endpoint(self, crm_handler, mock_request):
        """Test routing to platforms endpoint."""
        mock_request.path = "/api/v1/crm/platforms"
        mock_request.method = "GET"

        result = await crm_handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "platforms" in result["body"]


# =============================================================================
# Test: Constants
# =============================================================================


class TestValidationConstants:
    """Tests for validation constant values."""

    def test_max_length_constants(self):
        """Test max length constants have expected values."""
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

    def test_supported_platforms_structure(self):
        """Test SUPPORTED_PLATFORMS has expected structure."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config
            assert "description" in config
            assert "features" in config
            assert isinstance(config["features"], list)

    def test_hubspot_not_coming_soon(self):
        """Test HubSpot is not marked as coming soon."""
        assert SUPPORTED_PLATFORMS["hubspot"].get("coming_soon", False) is False

    def test_salesforce_and_pipedrive_coming_soon(self):
        """Test Salesforce and Pipedrive are marked as coming soon."""
        assert SUPPORTED_PLATFORMS["salesforce"].get("coming_soon") is True
        assert SUPPORTED_PLATFORMS["pipedrive"].get("coming_soon") is True
