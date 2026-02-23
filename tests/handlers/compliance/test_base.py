"""Comprehensive tests for compliance base utilities (aragora/server/handlers/compliance/_base.py).

Covers:
- extract_user_id_from_headers(): None headers, empty headers, missing auth,
  non-Bearer auth, API key tokens (ara_ prefix), JWT token decoding (success,
  ImportError, ValueError, AttributeError), fallback behavior
- parse_timestamp(): None/empty input, unix timestamps (int and float),
  ISO 8601 strings, ISO with Z suffix, ISO with timezone offset,
  invalid strings, non-numeric non-ISO strings, edge cases
- __all__ exports: verifies all expected symbols are listed
- Module-level logger: correct module name
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.compliance._base import (
    extract_user_id_from_headers,
    parse_timestamp,
)


# ---------------------------------------------------------------------------
# extract_user_id_from_headers tests
# ---------------------------------------------------------------------------


class TestExtractUserIdFromHeaders:
    """Tests for extract_user_id_from_headers."""

    def test_none_headers_returns_default(self):
        """None headers should return the default 'compliance_api'."""
        result = extract_user_id_from_headers(None)
        assert result == "compliance_api"

    def test_empty_dict_returns_default(self):
        """Empty dict should return the default 'compliance_api'."""
        result = extract_user_id_from_headers({})
        assert result == "compliance_api"

    def test_no_authorization_header_returns_default(self):
        """Headers without Authorization should return default."""
        result = extract_user_id_from_headers({"Content-Type": "application/json"})
        assert result == "compliance_api"

    def test_empty_authorization_header(self):
        """Empty Authorization header should return default."""
        result = extract_user_id_from_headers({"Authorization": ""})
        assert result == "compliance_api"

    def test_non_bearer_authorization(self):
        """Non-Bearer auth (e.g. Basic) should return default."""
        result = extract_user_id_from_headers({"Authorization": "Basic abc123"})
        assert result == "compliance_api"

    def test_bearer_only_no_token(self):
        """'Bearer ' with empty token should fall through to default."""
        result = extract_user_id_from_headers({"Authorization": "Bearer "})
        assert result == "compliance_api"

    def test_api_key_token_short(self):
        """API key token (ara_ prefix) returns truncated key identifier."""
        result = extract_user_id_from_headers({"Authorization": "Bearer ara_abcdefgh"})
        # "ara_abcdefgh" is exactly 12 chars, so token[:12] keeps all of it
        assert result == "api_key:ara_abcdefgh..."

    def test_api_key_token_long(self):
        """API key token with long key returns first 12 chars."""
        token = "ara_" + "x" * 50
        result = extract_user_id_from_headers({"Authorization": f"Bearer {token}"})
        # token[:12] = "ara_xxxxxxxx"
        assert result == "api_key:ara_xxxxxxxx..."

    def test_api_key_exact_12_chars(self):
        """API key exactly 12 chars should work correctly."""
        token = "ara_12345678"  # 12 chars
        result = extract_user_id_from_headers({"Authorization": f"Bearer {token}"})
        assert result == f"api_key:{token}..."

    def test_api_key_shorter_than_12_chars(self):
        """API key shorter than 12 chars uses what's available."""
        token = "ara_abc"  # 7 chars
        result = extract_user_id_from_headers({"Authorization": f"Bearer {token}"})
        assert result == f"api_key:{token}..."

    @patch("aragora.billing.auth.tokens.validate_access_token")
    def test_jwt_valid_token_returns_user_id(self, mock_validate):
        """Valid JWT with user_id should return the user_id."""
        mock_payload = SimpleNamespace(user_id="user-42")
        mock_validate.return_value = mock_payload
        result = extract_user_id_from_headers(
            {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.test.sig"}
        )
        assert result == "user-42"

    @patch("aragora.billing.auth.tokens.validate_access_token")
    def test_jwt_valid_token_none_user_id(self, mock_validate):
        """JWT payload with None user_id should fall through to default."""
        mock_payload = SimpleNamespace(user_id=None)
        mock_validate.return_value = mock_payload
        result = extract_user_id_from_headers(
            {"Authorization": "Bearer some_jwt_token"}
        )
        assert result == "compliance_api"

    @patch("aragora.billing.auth.tokens.validate_access_token")
    def test_jwt_returns_none_payload(self, mock_validate):
        """If validate_access_token returns None, fall through to default."""
        mock_validate.return_value = None
        result = extract_user_id_from_headers(
            {"Authorization": "Bearer some_jwt_token"}
        )
        assert result == "compliance_api"

    def test_jwt_import_error_falls_back(self):
        """ImportError during JWT validation falls back to default."""
        with patch.dict(
            "sys.modules",
            {"aragora.billing.auth.tokens": None},
        ):
            result = extract_user_id_from_headers(
                {"Authorization": "Bearer some_jwt_token"}
            )
            assert result == "compliance_api"

    @patch("aragora.billing.auth.tokens.validate_access_token")
    def test_jwt_value_error_falls_back(self, mock_validate):
        """ValueError during JWT validation falls back to default."""
        mock_validate.side_effect = ValueError("bad token")
        result = extract_user_id_from_headers(
            {"Authorization": "Bearer bad_token"}
        )
        assert result == "compliance_api"

    @patch("aragora.billing.auth.tokens.validate_access_token")
    def test_jwt_attribute_error_falls_back(self, mock_validate):
        """AttributeError during JWT validation falls back to default."""
        mock_validate.side_effect = AttributeError("no user_id")
        result = extract_user_id_from_headers(
            {"Authorization": "Bearer some_token"}
        )
        assert result == "compliance_api"

    def test_lowercase_authorization_header(self):
        """Lowercase 'authorization' header key should also be recognized."""
        result = extract_user_id_from_headers(
            {"authorization": "Bearer ara_test1234567890"}
        )
        assert result == "api_key:ara_test1234..."

    def test_both_cases_prefers_capitalized(self):
        """When both Authorization and authorization exist, capitalize wins."""
        result = extract_user_id_from_headers({
            "Authorization": "Bearer ara_UPPERCASE",
            "authorization": "Bearer ara_lowercase",
        })
        # The code checks "Authorization" first via .get()
        assert result == "api_key:ara_UPPERCAS..."

    def test_bearer_prefix_case_sensitive(self):
        """'bearer ' (lowercase) should NOT match - code checks 'Bearer '."""
        result = extract_user_id_from_headers({"Authorization": "bearer token123"})
        assert result == "compliance_api"


# ---------------------------------------------------------------------------
# parse_timestamp tests
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    """Tests for parse_timestamp."""

    def test_none_returns_none(self):
        """None input returns None."""
        assert parse_timestamp(None) is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert parse_timestamp("") is None

    def test_unix_timestamp_integer(self):
        """Integer unix timestamp parses correctly."""
        result = parse_timestamp("1700000000")
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is not None
        assert result == datetime.fromtimestamp(1700000000, tz=timezone.utc)

    def test_unix_timestamp_float(self):
        """Float unix timestamp with milliseconds parses correctly."""
        result = parse_timestamp("1700000000.123")
        assert result is not None
        assert result.tzinfo == timezone.utc
        expected = datetime.fromtimestamp(1700000000.123, tz=timezone.utc)
        assert result == expected

    def test_unix_timestamp_zero(self):
        """Unix epoch zero (1970-01-01) parses correctly."""
        result = parse_timestamp("0")
        assert result is not None
        assert result == datetime.fromtimestamp(0, tz=timezone.utc)

    def test_unix_timestamp_negative(self):
        """Negative unix timestamp (before epoch) parses correctly."""
        result = parse_timestamp("-86400")
        assert result is not None
        assert result == datetime.fromtimestamp(-86400, tz=timezone.utc)

    def test_iso_format_basic(self):
        """Standard ISO 8601 format parses correctly."""
        result = parse_timestamp("2024-01-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_iso_format_with_z_suffix(self):
        """ISO 8601 with Z (Zulu) suffix parses correctly."""
        result = parse_timestamp("2024-06-15T12:00:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 12

    def test_iso_format_date_only(self):
        """Date-only ISO string parses correctly."""
        result = parse_timestamp("2024-03-20")
        assert result is not None
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 20

    def test_iso_format_with_microseconds(self):
        """ISO 8601 with microseconds parses correctly."""
        result = parse_timestamp("2024-01-15T10:30:00.123456+00:00")
        assert result is not None
        assert result.microsecond == 123456

    def test_iso_format_with_timezone_offset(self):
        """ISO 8601 with non-UTC timezone offset parses correctly."""
        result = parse_timestamp("2024-01-15T10:30:00+05:30")
        assert result is not None
        assert result.hour == 10
        assert result.minute == 30

    def test_invalid_string_returns_none(self):
        """Non-parseable string returns None."""
        assert parse_timestamp("not-a-timestamp") is None

    def test_random_text_returns_none(self):
        """Random text returns None."""
        assert parse_timestamp("hello world") is None

    def test_partial_iso_returns_none_or_parses(self):
        """Incomplete ISO-like strings may fail gracefully."""
        result = parse_timestamp("2024-13-40")
        # Invalid month/day - should return None
        assert result is None

    def test_iso_without_time(self):
        """ISO date without time component parses."""
        result = parse_timestamp("2025-12-31")
        assert result is not None
        assert result.year == 2025
        assert result.month == 12
        assert result.day == 31

    def test_very_large_unix_timestamp(self):
        """Very large unix timestamp parses (far future)."""
        result = parse_timestamp("4102444800")  # 2100-01-01
        assert result is not None
        assert result.year == 2100


# ---------------------------------------------------------------------------
# __all__ exports tests
# ---------------------------------------------------------------------------


class TestAllExports:
    """Tests for __all__ listing in _base.py."""

    def test_all_exports_defined(self):
        """__all__ should be a list containing expected exports."""
        import aragora.server.handlers.compliance._base as base_mod

        assert hasattr(base_mod, "__all__")
        assert isinstance(base_mod.__all__, list)
        assert len(base_mod.__all__) > 0

    def test_all_contains_handler_base_types(self):
        """__all__ should include BaseHandler and HandlerResult."""
        import aragora.server.handlers.compliance._base as base_mod

        assert "BaseHandler" in base_mod.__all__
        assert "HandlerResult" in base_mod.__all__

    def test_all_contains_response_helpers(self):
        """__all__ should include error_response and json_response."""
        import aragora.server.handlers.compliance._base as base_mod

        assert "error_response" in base_mod.__all__
        assert "json_response" in base_mod.__all__

    def test_all_contains_storage_getters(self):
        """__all__ should include storage access functions."""
        import aragora.server.handlers.compliance._base as base_mod

        assert "get_audit_store" in base_mod.__all__
        assert "get_receipt_store" in base_mod.__all__

    def test_all_contains_privacy_deletion(self):
        """__all__ should include privacy/deletion utilities."""
        import aragora.server.handlers.compliance._base as base_mod

        assert "get_deletion_scheduler" in base_mod.__all__
        assert "get_legal_hold_manager" in base_mod.__all__
        assert "get_deletion_coordinator" in base_mod.__all__

    def test_all_contains_local_utilities(self):
        """__all__ should include locally-defined utilities."""
        import aragora.server.handlers.compliance._base as base_mod

        assert "extract_user_id_from_headers" in base_mod.__all__
        assert "parse_timestamp" in base_mod.__all__
        assert "logger" in base_mod.__all__

    def test_all_contains_standard_library_reexports(self):
        """__all__ should include standard library re-exports."""
        import aragora.server.handlers.compliance._base as base_mod

        for name in ("hashlib", "html", "json", "logging", "datetime",
                      "timezone", "timedelta", "Any", "Optional"):
            assert name in base_mod.__all__, f"{name} missing from __all__"

    def test_all_symbols_are_accessible(self):
        """Every symbol in __all__ should be importable from the module."""
        import aragora.server.handlers.compliance._base as base_mod

        for name in base_mod.__all__:
            assert hasattr(base_mod, name), f"{name} in __all__ but not accessible on module"

    def test_all_contains_rbac_items(self):
        """__all__ should include RBAC-related imports."""
        import aragora.server.handlers.compliance._base as base_mod

        assert "PermissionDeniedError" in base_mod.__all__
        assert "require_permission" in base_mod.__all__

    def test_all_contains_observability(self):
        """__all__ should include track_handler from observability."""
        import aragora.server.handlers.compliance._base as base_mod

        assert "track_handler" in base_mod.__all__

    def test_all_contains_rate_limit(self):
        """__all__ should include rate_limit decorator."""
        import aragora.server.handlers.compliance._base as base_mod

        assert "rate_limit" in base_mod.__all__


# ---------------------------------------------------------------------------
# Module-level logger tests
# ---------------------------------------------------------------------------


class TestModuleLogger:
    """Tests for the module-level logger."""

    def test_logger_exists(self):
        """Module should have a logger attribute."""
        import aragora.server.handlers.compliance._base as base_mod

        assert hasattr(base_mod, "logger")

    def test_logger_is_logging_instance(self):
        """logger should be a logging.Logger instance."""
        import aragora.server.handlers.compliance._base as base_mod

        assert isinstance(base_mod.logger, logging.Logger)

    def test_logger_name(self):
        """logger should be named after the module."""
        import aragora.server.handlers.compliance._base as base_mod

        assert base_mod.logger.name == "aragora.server.handlers.compliance._base"
