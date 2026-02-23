"""Comprehensive tests for DevOps validation helpers.

Tests cover all public functions and constants in
aragora/server/handlers/features/devops/validation.py:

- validate_pagerduty_id: PagerDuty ID format validation
- validate_urgency: Urgency normalization
- validate_string_field: Generic string field validation
- validate_id_list: List-of-IDs validation
- Constants: patterns, frozensets, max lengths
- Backward-compatible aliases (_-prefixed variants)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aragora.server.handlers.features.devops.validation import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NOTE_CONTENT_LENGTH,
    MAX_RESOLUTION_LENGTH,
    MAX_SOURCE_INCIDENT_IDS,
    MAX_TITLE_LENGTH,
    MAX_USER_IDS,
    PAGERDUTY_ID_PATTERN,
    VALID_INCIDENT_STATUSES,
    VALID_URGENCIES,
    _validate_id_list,
    _validate_pagerduty_id,
    _validate_string_field,
    _validate_urgency,
    validate_id_list,
    validate_pagerduty_id,
    validate_string_field,
    validate_urgency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    """Verify exported constants have expected values."""

    def test_pagerduty_id_pattern_matches_alphanumeric(self):
        assert PAGERDUTY_ID_PATTERN.match("ABC123")

    def test_pagerduty_id_pattern_rejects_special_chars(self):
        assert PAGERDUTY_ID_PATTERN.match("abc!@#") is None

    def test_valid_urgencies(self):
        assert VALID_URGENCIES == frozenset({"high", "low"})

    def test_valid_incident_statuses(self):
        assert VALID_INCIDENT_STATUSES == frozenset({"triggered", "acknowledged", "resolved"})

    def test_max_title_length(self):
        assert MAX_TITLE_LENGTH == 500

    def test_max_description_length(self):
        assert MAX_DESCRIPTION_LENGTH == 10000

    def test_max_note_content_length(self):
        assert MAX_NOTE_CONTENT_LENGTH == 5000

    def test_max_resolution_length(self):
        assert MAX_RESOLUTION_LENGTH == 2000

    def test_max_user_ids(self):
        assert MAX_USER_IDS == 20

    def test_max_source_incident_ids(self):
        assert MAX_SOURCE_INCIDENT_IDS == 50


# ============================================================================
# validate_pagerduty_id
# ============================================================================


class TestValidatePagerdutyId:
    """Tests for validate_pagerduty_id."""

    def test_valid_alphanumeric_id(self):
        is_valid, err = validate_pagerduty_id("ABC123")
        assert is_valid is True
        assert err is None

    def test_valid_lowercase_id(self):
        is_valid, err = validate_pagerduty_id("abcdef")
        assert is_valid is True
        assert err is None

    def test_valid_uppercase_id(self):
        is_valid, err = validate_pagerduty_id("PXYZ")
        assert is_valid is True
        assert err is None

    def test_valid_numeric_only(self):
        is_valid, err = validate_pagerduty_id("12345")
        assert is_valid is True
        assert err is None

    def test_single_character_id(self):
        is_valid, err = validate_pagerduty_id("A")
        assert is_valid is True
        assert err is None

    def test_max_length_id(self):
        """20 chars should be accepted."""
        is_valid, err = validate_pagerduty_id("A" * 20)
        assert is_valid is True
        assert err is None

    def test_too_long_id(self):
        """21 chars should be rejected."""
        is_valid, err = validate_pagerduty_id("A" * 21)
        assert is_valid is False
        assert "too long" in err

    def test_empty_string(self):
        is_valid, err = validate_pagerduty_id("")
        assert is_valid is False
        assert "required" in err

    def test_none_value(self):
        # None is falsy, triggers "required" branch
        is_valid, err = validate_pagerduty_id(None)
        assert is_valid is False
        assert "required" in err

    def test_special_characters_rejected(self):
        is_valid, err = validate_pagerduty_id("ABC-123")
        assert is_valid is False
        assert "invalid format" in err

    def test_spaces_rejected(self):
        is_valid, err = validate_pagerduty_id("ABC 123")
        assert is_valid is False
        assert "invalid format" in err

    def test_underscore_rejected(self):
        is_valid, err = validate_pagerduty_id("ABC_123")
        assert is_valid is False
        assert "invalid format" in err

    def test_custom_field_name_in_error(self):
        is_valid, err = validate_pagerduty_id("", field_name="service_id")
        assert is_valid is False
        assert "service_id" in err

    def test_default_field_name(self):
        is_valid, err = validate_pagerduty_id("")
        assert "id" in err

    def test_non_string_integer(self):
        """Integer triggers 'must be a string' because isinstance check fails."""
        is_valid, err = validate_pagerduty_id(12345)
        assert is_valid is False
        assert "must be a string" in err

    def test_non_string_list(self):
        is_valid, err = validate_pagerduty_id(["abc"])
        assert is_valid is False
        # list is truthy and not a string
        assert "must be a string" in err

    def test_dot_rejected(self):
        is_valid, err = validate_pagerduty_id("ABC.123")
        assert is_valid is False
        assert "invalid format" in err

    def test_slash_rejected(self):
        is_valid, err = validate_pagerduty_id("ABC/123")
        assert is_valid is False
        assert "invalid format" in err


# ============================================================================
# validate_urgency
# ============================================================================


class TestValidateUrgency:
    """Tests for validate_urgency."""

    def test_high(self):
        assert validate_urgency("high") == "high"

    def test_low(self):
        assert validate_urgency("low") == "low"

    def test_none_defaults_to_high(self):
        assert validate_urgency(None) == "high"

    def test_uppercase_normalized(self):
        assert validate_urgency("HIGH") == "high"

    def test_mixed_case(self):
        assert validate_urgency("Low") == "low"

    def test_whitespace_stripped(self):
        assert validate_urgency("  high  ") == "high"

    def test_invalid_defaults_to_high(self):
        assert validate_urgency("critical") == "high"

    def test_empty_string_defaults_to_high(self):
        assert validate_urgency("") == "high"

    def test_numeric_input_converted_to_string(self):
        """Non-string gets str() called, then won't match urgencies."""
        assert validate_urgency(123) == "high"

    def test_medium_not_valid(self):
        assert validate_urgency("medium") == "high"


# ============================================================================
# validate_string_field
# ============================================================================


class TestValidateStringField:
    """Tests for validate_string_field."""

    def test_valid_string(self):
        val, err = validate_string_field("hello", "title")
        assert val == "hello"
        assert err is None

    def test_whitespace_stripped(self):
        val, err = validate_string_field("  hello  ", "title")
        assert val == "hello"
        assert err is None

    def test_none_optional_returns_none(self):
        val, err = validate_string_field(None, "title", required=False)
        assert val is None
        assert err is None

    def test_none_required_returns_error(self):
        val, err = validate_string_field(None, "title", required=True)
        assert val is None
        assert "required" in err

    def test_empty_string_optional(self):
        val, err = validate_string_field("", "title", required=False)
        assert val is None
        assert err is None

    def test_empty_string_required(self):
        val, err = validate_string_field("", "title", required=True)
        assert val is None
        assert "required" in err

    def test_exceeds_max_length(self):
        val, err = validate_string_field("x" * 501, "title", max_length=500)
        assert val is None
        assert "exceeds maximum length of 500" in err

    def test_exactly_max_length(self):
        val, err = validate_string_field("x" * 500, "title", max_length=500)
        assert val == "x" * 500
        assert err is None

    def test_custom_max_length(self):
        val, err = validate_string_field("hello world", "note", max_length=5)
        assert val is None
        assert "exceeds maximum length of 5" in err

    def test_non_string_converted(self):
        """Integer is converted via str()."""
        val, err = validate_string_field(42, "count")
        assert val == "42"
        assert err is None

    def test_non_string_boolean_converted(self):
        val, err = validate_string_field(True, "flag")
        assert val == "True"
        assert err is None

    def test_non_string_float_converted(self):
        val, err = validate_string_field(3.14, "ratio")
        assert val == "3.14"
        assert err is None

    def test_non_string_list_converted(self):
        val, err = validate_string_field([1, 2], "items")
        assert val == "[1, 2]"
        assert err is None

    def test_field_name_in_error(self):
        _, err = validate_string_field(None, "description", required=True)
        assert "description" in err

    def test_whitespace_only_after_strip_within_max(self):
        """A whitespace-only string that strips to empty string with length 0."""
        val, err = validate_string_field("   ", "title")
        assert val == ""
        assert err is None

    def test_default_max_length_is_500(self):
        """Default max_length is 500."""
        val, err = validate_string_field("x" * 501, "title")
        assert val is None
        assert "500" in err

    def test_unicode_string(self):
        val, err = validate_string_field("hello world", "title")
        assert val == "hello world"
        assert err is None

    def test_long_description_field(self):
        val, err = validate_string_field(
            "a" * MAX_DESCRIPTION_LENGTH, "description", max_length=MAX_DESCRIPTION_LENGTH
        )
        assert val == "a" * MAX_DESCRIPTION_LENGTH
        assert err is None

    def test_long_description_exceeds(self):
        val, err = validate_string_field(
            "a" * (MAX_DESCRIPTION_LENGTH + 1), "description", max_length=MAX_DESCRIPTION_LENGTH
        )
        assert val is None
        assert "exceeds" in err


# ============================================================================
# validate_id_list
# ============================================================================


class TestValidateIdList:
    """Tests for validate_id_list."""

    def test_none_returns_none(self):
        val, err = validate_id_list(None, "user_ids")
        assert val is None
        assert err is None

    def test_valid_list(self):
        val, err = validate_id_list(["ABC123", "DEF456"], "user_ids")
        assert val == ["ABC123", "DEF456"]
        assert err is None

    def test_not_a_list(self):
        val, err = validate_id_list("not_a_list", "user_ids")
        assert val is None
        assert "must be a list" in err

    def test_dict_not_a_list(self):
        val, err = validate_id_list({"key": "val"}, "user_ids")
        assert val is None
        assert "must be a list" in err

    def test_exceeds_max_items(self):
        ids = [f"ID{i}" for i in range(21)]
        val, err = validate_id_list(ids, "user_ids", max_items=20)
        assert val is None
        assert "exceeds maximum of 20" in err

    def test_exactly_max_items(self):
        ids = [f"ID{i}" for i in range(20)]
        val, err = validate_id_list(ids, "user_ids", max_items=20)
        assert val == [f"ID{i}" for i in range(20)]
        assert err is None

    def test_empty_list(self):
        val, err = validate_id_list([], "user_ids")
        assert val == []
        assert err is None

    def test_invalid_id_in_list(self):
        val, err = validate_id_list(["ABC123", "INVALID-ID"], "user_ids")
        assert val is None
        assert "invalid format" in err

    def test_invalid_id_reports_index(self):
        val, err = validate_id_list(["GOOD", "BAD!"], "ids")
        assert val is None
        assert "ids[1]" in err

    def test_numeric_items_converted(self):
        """Integer items get str() applied via validate_pagerduty_id."""
        val, err = validate_id_list([123, 456], "ids")
        assert val == ["123", "456"]
        assert err is None

    def test_custom_max_items(self):
        ids = ["A", "B", "C"]
        val, err = validate_id_list(ids, "ids", max_items=2)
        assert val is None
        assert "exceeds maximum of 2" in err

    def test_default_max_items_is_20(self):
        ids = [f"ID{i}" for i in range(21)]
        val, err = validate_id_list(ids, "ids")
        assert val is None
        assert "20" in err

    def test_tuple_not_a_list(self):
        val, err = validate_id_list(("A", "B"), "ids")
        assert val is None
        assert "must be a list" in err

    def test_item_too_long(self):
        val, err = validate_id_list(["A" * 21], "ids")
        assert val is None
        assert "too long" in err

    def test_item_with_special_chars(self):
        val, err = validate_id_list(["OK", "NO@WAY"], "ids")
        assert val is None
        assert "invalid format" in err

    def test_single_item_list(self):
        val, err = validate_id_list(["ABC"], "ids")
        assert val == ["ABC"]
        assert err is None

    def test_large_valid_list(self):
        """50 items with custom max_items=50 should pass."""
        ids = [f"ID{i:04d}" for i in range(50)]
        val, err = validate_id_list(ids, "source_incidents", max_items=50)
        assert len(val) == 50
        assert err is None

    def test_max_source_incident_ids_constant(self):
        """Verify MAX_SOURCE_INCIDENT_IDS works as max_items."""
        ids = [f"ID{i}" for i in range(MAX_SOURCE_INCIDENT_IDS + 1)]
        val, err = validate_id_list(ids, "src", max_items=MAX_SOURCE_INCIDENT_IDS)
        assert val is None
        assert str(MAX_SOURCE_INCIDENT_IDS) in err


# ============================================================================
# Backward-compatible aliases
# ============================================================================


class TestBackwardCompatibleAliases:
    """Verify underscore-prefixed aliases point to the same functions."""

    def test_validate_pagerduty_id_alias(self):
        assert _validate_pagerduty_id is validate_pagerduty_id

    def test_validate_urgency_alias(self):
        assert _validate_urgency is validate_urgency

    def test_validate_string_field_alias(self):
        assert _validate_string_field is validate_string_field

    def test_validate_id_list_alias(self):
        assert _validate_id_list is validate_id_list


# ============================================================================
# Edge cases and integration-style tests
# ============================================================================


class TestEdgeCases:
    """Cross-cutting edge cases."""

    def test_pagerduty_pattern_anchored(self):
        """Pattern is anchored with ^ and $ -- no partial matches."""
        assert PAGERDUTY_ID_PATTERN.match("ABC") is not None
        # fullmatch semantics via ^...$
        assert PAGERDUTY_ID_PATTERN.match("ABC\n123") is None

    def test_urgency_frozenset_is_immutable(self):
        with pytest.raises(AttributeError):
            VALID_URGENCIES.add("medium")

    def test_incident_statuses_frozenset_is_immutable(self):
        with pytest.raises(AttributeError):
            VALID_INCIDENT_STATUSES.add("pending")

    def test_string_field_max_length_zero(self):
        """max_length=0 means only empty strings pass length check."""
        val, err = validate_string_field("a", "f", max_length=0)
        assert val is None
        assert "exceeds maximum length" in err

    def test_id_list_max_items_zero(self):
        """max_items=0 means any non-empty list fails."""
        val, err = validate_id_list(["A"], "ids", max_items=0)
        assert val is None
        assert "exceeds maximum of 0" in err

    def test_validate_string_field_strips_then_checks_length(self):
        """Whitespace is stripped before length check."""
        # 498 chars of content + leading/trailing spaces
        padded = "  " + "x" * 498 + "  "
        val, err = validate_string_field(padded, "t", max_length=500)
        assert val == "x" * 498
        assert err is None

    def test_validate_pagerduty_id_zero_string(self):
        """The string '0' is a valid alphanumeric ID."""
        is_valid, err = validate_pagerduty_id("0")
        assert is_valid is True
        assert err is None
