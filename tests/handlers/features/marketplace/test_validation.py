"""Comprehensive tests for marketplace input validation functions and constants.

Covers every function and constant in:
    aragora/server/handlers/features/marketplace/validation.py

Functions tested:
- _validate_id              (ID validation with label)
- _validate_template_id     (backward-compatible template ID alias)
- _validate_deployment_id   (backward-compatible deployment ID alias)
- _validate_pagination      (limit/offset from query dict)
- _clamp_pagination         (direct limit/offset clamping)
- _validate_rating_value    (rating 1-5)
- _validate_rating          (alias for _validate_rating_value)
- _validate_review_internal (review string with sanitization)
- _validate_review          (backward-compatible 2-tuple alias)
- _validate_deployment_name_internal (name with sanitization + fallback)
- _validate_deployment_name (backward-compatible 2-tuple alias)
- _validate_config          (config dict size check)
- _validate_search_query    (search string with sanitization)
- _validate_category        (category enum lookup)
- _validate_category_filter (category returning enum .value string)
- Constants                 (regex, limits, aliases)
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.features.marketplace.models import TemplateCategory
from aragora.server.handlers.features.marketplace.validation import (
    DEFAULT_LIMIT,
    MAX_CONFIG_KEYS,
    MAX_CONFIG_SIZE,
    MAX_DEPLOYMENT_NAME_LENGTH,
    MAX_LIMIT,
    MAX_OFFSET,
    MAX_RATING,
    MAX_REVIEW_LENGTH,
    MAX_SEARCH_QUERY_LENGTH,
    MAX_TEMPLATE_NAME_LENGTH,
    MIN_LIMIT,
    MIN_RATING,
    SAFE_ID_PATTERN,
    SAFE_TEMPLATE_ID_PATTERN,
    _clamp_pagination,
    _validate_category,
    _validate_category_filter,
    _validate_config,
    _validate_deployment_id,
    _validate_deployment_name,
    _validate_deployment_name_internal,
    _validate_id,
    _validate_pagination,
    _validate_rating,
    _validate_rating_value,
    _validate_review,
    _validate_review_internal,
    _validate_search_query,
    _validate_template_id,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Verify all exported constants have expected values."""

    def test_safe_id_pattern_matches_alphanumeric(self):
        assert SAFE_ID_PATTERN.match("abc123")

    def test_safe_id_pattern_allows_hyphens_and_underscores(self):
        assert SAFE_ID_PATTERN.match("my-template_1")

    def test_safe_id_pattern_rejects_empty(self):
        assert not SAFE_ID_PATTERN.match("")

    def test_safe_id_pattern_rejects_leading_hyphen(self):
        assert not SAFE_ID_PATTERN.match("-leading")

    def test_safe_id_pattern_rejects_leading_underscore(self):
        assert not SAFE_ID_PATTERN.match("_leading")

    def test_safe_id_pattern_rejects_special_chars(self):
        assert not SAFE_ID_PATTERN.match("has space")
        assert not SAFE_ID_PATTERN.match("has@sign")
        assert not SAFE_ID_PATTERN.match("has/slash")

    def test_safe_id_pattern_max_length_128(self):
        # 1 leading char + 127 trailing chars = 128 total
        assert SAFE_ID_PATTERN.match("a" * 128)
        assert not SAFE_ID_PATTERN.match("a" * 129)

    def test_safe_id_pattern_single_char(self):
        # Pattern is ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$ so 1 char is valid
        assert SAFE_ID_PATTERN.match("a")
        assert SAFE_ID_PATTERN.match("Z")
        assert SAFE_ID_PATTERN.match("0")

    def test_safe_template_id_pattern_is_alias(self):
        assert SAFE_TEMPLATE_ID_PATTERN is SAFE_ID_PATTERN

    def test_max_template_name_length(self):
        assert MAX_TEMPLATE_NAME_LENGTH == 200

    def test_max_deployment_name_length(self):
        assert MAX_DEPLOYMENT_NAME_LENGTH == 200

    def test_max_review_length(self):
        assert MAX_REVIEW_LENGTH == 2000

    def test_max_search_query_length(self):
        assert MAX_SEARCH_QUERY_LENGTH == 500

    def test_max_config_keys(self):
        assert MAX_CONFIG_KEYS == 50

    def test_max_config_size_equals_max_config_keys(self):
        assert MAX_CONFIG_SIZE == MAX_CONFIG_KEYS

    def test_min_rating(self):
        assert MIN_RATING == 1

    def test_max_rating(self):
        assert MAX_RATING == 5

    def test_default_limit(self):
        assert DEFAULT_LIMIT == 50

    def test_min_limit(self):
        assert MIN_LIMIT == 1

    def test_max_limit(self):
        assert MAX_LIMIT == 200

    def test_max_offset(self):
        assert MAX_OFFSET == 10000


# =============================================================================
# _validate_id Tests
# =============================================================================


class TestValidateId:
    """Tests for _validate_id(value, label)."""

    def test_valid_simple_id(self):
        valid, err = _validate_id("my-template-1", "Template ID")
        assert valid is True
        assert err == ""

    def test_valid_alphanumeric_only(self):
        valid, err = _validate_id("abc123", "ID")
        assert valid is True
        assert err == ""

    def test_valid_with_underscores(self):
        valid, err = _validate_id("a_b_c", "ID")
        assert valid is True
        assert err == ""

    def test_valid_single_character(self):
        valid, err = _validate_id("x", "ID")
        assert valid is True
        assert err == ""

    def test_valid_max_length_128(self):
        valid, err = _validate_id("a" * 128, "ID")
        assert valid is True
        assert err == ""

    def test_empty_string(self):
        valid, err = _validate_id("", "Template ID")
        assert valid is False
        assert "Template ID" in err
        assert "required" in err

    def test_none_value(self):
        valid, err = _validate_id(None, "Template ID")
        assert valid is False
        assert "required" in err

    def test_too_long(self):
        valid, err = _validate_id("a" * 129, "Deployment ID")
        assert valid is False
        assert "128" in err

    def test_invalid_chars_space(self):
        valid, err = _validate_id("has space", "ID")
        assert valid is False
        assert "invalid characters" in err

    def test_invalid_chars_at_sign(self):
        valid, err = _validate_id("user@host", "ID")
        assert valid is False
        assert "invalid characters" in err

    def test_invalid_chars_slash(self):
        valid, err = _validate_id("path/to/thing", "ID")
        assert valid is False
        assert "invalid characters" in err

    def test_invalid_leading_hyphen(self):
        valid, err = _validate_id("-leading", "ID")
        assert valid is False
        assert "invalid characters" in err

    def test_invalid_leading_underscore(self):
        valid, err = _validate_id("_leading", "ID")
        assert valid is False
        assert "invalid characters" in err

    def test_custom_label_in_error(self):
        valid, err = _validate_id("", "Custom Label")
        assert valid is False
        assert "Custom Label" in err

    def test_default_label_is_id(self):
        valid, err = _validate_id("")
        assert valid is False
        assert "ID is required" in err

    def test_non_string_integer(self):
        valid, err = _validate_id(123, "ID")
        assert valid is False
        assert "required" in err

    def test_non_string_list(self):
        valid, err = _validate_id(["a"], "ID")
        assert valid is False
        assert "required" in err


# =============================================================================
# _validate_template_id Tests (backward-compatible alias)
# =============================================================================


class TestValidateTemplateId:
    """Tests for _validate_template_id(value) -> (bool, error_or_None)."""

    def test_valid_returns_true_none(self):
        valid, err = _validate_template_id("my-template")
        assert valid is True
        assert err is None

    def test_empty_returns_false_with_error(self):
        valid, err = _validate_template_id("")
        assert valid is False
        assert err is not None
        assert "Template ID" in err

    def test_too_long(self):
        valid, err = _validate_template_id("t" * 129)
        assert valid is False
        assert err is not None

    def test_invalid_chars(self):
        valid, err = _validate_template_id("bad!id")
        assert valid is False
        assert err is not None

    def test_none_value(self):
        valid, err = _validate_template_id(None)
        assert valid is False
        assert err is not None


# =============================================================================
# _validate_deployment_id Tests (backward-compatible alias)
# =============================================================================


class TestValidateDeploymentId:
    """Tests for _validate_deployment_id(value) -> (bool, error_or_None)."""

    def test_valid_returns_true_none(self):
        valid, err = _validate_deployment_id("deploy-abc-123")
        assert valid is True
        assert err is None

    def test_empty_returns_false_with_error(self):
        valid, err = _validate_deployment_id("")
        assert valid is False
        assert err is not None
        assert "Deployment ID" in err

    def test_too_long(self):
        valid, err = _validate_deployment_id("d" * 129)
        assert valid is False
        assert err is not None

    def test_invalid_chars(self):
        valid, err = _validate_deployment_id("bad id")
        assert valid is False
        assert err is not None

    def test_none_value(self):
        valid, err = _validate_deployment_id(None)
        assert valid is False
        assert err is not None


# =============================================================================
# _validate_pagination Tests
# =============================================================================


class TestValidatePagination:
    """Tests for _validate_pagination(query) -> (limit, offset, error)."""

    def test_defaults_when_empty(self):
        limit, offset, err = _validate_pagination({})
        assert limit == DEFAULT_LIMIT
        assert offset == 0
        assert err == ""

    def test_explicit_limit_and_offset(self):
        limit, offset, err = _validate_pagination({"limit": "10", "offset": "20"})
        assert limit == 10
        assert offset == 20
        assert err == ""

    def test_integer_values_in_query(self):
        limit, offset, err = _validate_pagination({"limit": 25, "offset": 50})
        assert limit == 25
        assert offset == 50
        assert err == ""

    def test_limit_clamped_to_min(self):
        limit, offset, err = _validate_pagination({"limit": "0"})
        assert limit == MIN_LIMIT
        assert err == ""

    def test_limit_clamped_to_max(self):
        limit, offset, err = _validate_pagination({"limit": "999"})
        assert limit == MAX_LIMIT
        assert err == ""

    def test_offset_clamped_to_zero(self):
        limit, offset, err = _validate_pagination({"offset": "-5"})
        assert offset == 0
        assert err == ""

    def test_offset_clamped_to_max(self):
        limit, offset, err = _validate_pagination({"offset": "99999"})
        assert offset == MAX_OFFSET
        assert err == ""

    def test_invalid_limit_string(self):
        limit, offset, err = _validate_pagination({"limit": "abc"})
        assert limit == DEFAULT_LIMIT
        assert offset == 0
        assert "limit must be an integer" in err

    def test_invalid_offset_string(self):
        limit, offset, err = _validate_pagination({"offset": "xyz"})
        assert limit == DEFAULT_LIMIT
        assert offset == 0
        assert "offset must be an integer" in err

    def test_limit_none_type(self):
        limit, offset, err = _validate_pagination({"limit": None})
        assert limit == DEFAULT_LIMIT
        assert offset == 0
        assert "limit must be an integer" in err

    def test_offset_none_type(self):
        # When limit is valid but offset is None
        limit, offset, err = _validate_pagination({"limit": "10", "offset": None})
        assert limit == DEFAULT_LIMIT
        assert offset == 0
        assert "offset must be an integer" in err

    def test_boundary_limit_1(self):
        limit, offset, err = _validate_pagination({"limit": "1"})
        assert limit == 1
        assert err == ""

    def test_boundary_limit_200(self):
        limit, offset, err = _validate_pagination({"limit": "200"})
        assert limit == 200
        assert err == ""

    def test_negative_limit(self):
        limit, offset, err = _validate_pagination({"limit": "-1"})
        assert limit == MIN_LIMIT
        assert err == ""


# =============================================================================
# _clamp_pagination Tests
# =============================================================================


class TestClampPagination:
    """Tests for _clamp_pagination(limit, offset) -> (int, int)."""

    def test_normal_values(self):
        assert _clamp_pagination(10, 20) == (10, 20)

    def test_limit_below_min(self):
        limit, offset = _clamp_pagination(0, 0)
        assert limit == MIN_LIMIT

    def test_limit_above_max(self):
        limit, offset = _clamp_pagination(300, 0)
        assert limit == MAX_LIMIT

    def test_offset_negative(self):
        limit, offset = _clamp_pagination(10, -10)
        assert offset == 0

    def test_offset_above_max(self):
        limit, offset = _clamp_pagination(10, 20000)
        assert offset == MAX_OFFSET

    def test_none_limit_uses_default(self):
        limit, offset = _clamp_pagination(None, 0)
        assert limit == DEFAULT_LIMIT

    def test_none_offset_uses_zero(self):
        limit, offset = _clamp_pagination(10, None)
        assert offset == 0

    def test_string_values_converted(self):
        limit, offset = _clamp_pagination("15", "25")
        assert limit == 15
        assert offset == 25

    def test_invalid_string_limit_uses_default(self):
        limit, offset = _clamp_pagination("bad", 0)
        assert limit == DEFAULT_LIMIT

    def test_invalid_string_offset_uses_zero(self):
        limit, offset = _clamp_pagination(10, "bad")
        assert offset == 0

    def test_both_none(self):
        limit, offset = _clamp_pagination(None, None)
        assert limit == DEFAULT_LIMIT
        assert offset == 0

    def test_boundary_min_limit(self):
        limit, offset = _clamp_pagination(MIN_LIMIT, 0)
        assert limit == MIN_LIMIT

    def test_boundary_max_limit(self):
        limit, offset = _clamp_pagination(MAX_LIMIT, 0)
        assert limit == MAX_LIMIT

    def test_boundary_max_offset(self):
        limit, offset = _clamp_pagination(10, MAX_OFFSET)
        assert offset == MAX_OFFSET

    def test_float_values(self):
        # float is int-convertible via int()
        limit, offset = _clamp_pagination(10.9, 5.5)
        assert limit == 10
        assert offset == 5


# =============================================================================
# _validate_rating_value Tests
# =============================================================================


class TestValidateRatingValue:
    """Tests for _validate_rating_value(value) -> (bool, int, error)."""

    def test_valid_minimum_rating(self):
        valid, val, err = _validate_rating_value(1)
        assert valid is True
        assert val == 1
        assert err == ""

    def test_valid_maximum_rating(self):
        valid, val, err = _validate_rating_value(5)
        assert valid is True
        assert val == 5
        assert err == ""

    def test_valid_mid_rating(self):
        valid, val, err = _validate_rating_value(3)
        assert valid is True
        assert val == 3
        assert err == ""

    def test_none_value(self):
        valid, val, err = _validate_rating_value(None)
        assert valid is False
        assert val == 0
        assert "required" in err.lower()

    def test_non_integer_string(self):
        valid, val, err = _validate_rating_value("3")
        assert valid is False
        assert "integer" in err.lower()

    def test_non_integer_float(self):
        valid, val, err = _validate_rating_value(3.5)
        assert valid is False
        assert "integer" in err.lower()

    def test_below_min(self):
        valid, val, err = _validate_rating_value(0)
        assert valid is False
        assert str(MIN_RATING) in err
        assert str(MAX_RATING) in err

    def test_above_max(self):
        valid, val, err = _validate_rating_value(6)
        assert valid is False
        assert str(MIN_RATING) in err
        assert str(MAX_RATING) in err

    def test_negative_rating(self):
        valid, val, err = _validate_rating_value(-1)
        assert valid is False

    def test_large_positive(self):
        valid, val, err = _validate_rating_value(100)
        assert valid is False

    def test_boolean_is_not_int_for_type_check(self):
        # In Python, bool is a subclass of int so True/False pass isinstance(x, int)
        # This test documents that behavior
        valid, val, err = _validate_rating_value(True)
        assert valid is True  # True == 1 which is in range
        assert val == True

    def test_all_valid_ratings(self):
        for r in range(MIN_RATING, MAX_RATING + 1):
            valid, val, err = _validate_rating_value(r)
            assert valid is True
            assert val == r
            assert err == ""


# =============================================================================
# _validate_rating Tests (alias)
# =============================================================================


class TestValidateRating:
    """Tests for _validate_rating (alias of _validate_rating_value)."""

    def test_valid_rating(self):
        valid, val, err = _validate_rating(4)
        assert valid is True
        assert val == 4
        assert err == ""

    def test_invalid_none(self):
        valid, val, err = _validate_rating(None)
        assert valid is False

    def test_invalid_out_of_range(self):
        valid, val, err = _validate_rating(0)
        assert valid is False

    def test_invalid_type(self):
        valid, val, err = _validate_rating("5")
        assert valid is False


# =============================================================================
# _validate_review_internal Tests
# =============================================================================


class TestValidateReviewInternal:
    """Tests for _validate_review_internal(value) -> (bool, sanitized, error)."""

    def test_none_is_valid(self):
        valid, val, err = _validate_review_internal(None)
        assert valid is True
        assert val is None
        assert err == ""

    def test_valid_short_review(self):
        valid, val, err = _validate_review_internal("Great template!")
        assert valid is True
        assert val is not None
        assert err == ""

    def test_review_is_sanitized(self):
        valid, val, err = _validate_review_internal("  hello  ")
        assert valid is True
        # sanitize_string strips whitespace
        assert val == "hello"
        assert err == ""

    def test_non_string_integer(self):
        valid, val, err = _validate_review_internal(123)
        assert valid is False
        assert val is None
        assert "string" in err.lower()

    def test_non_string_list(self):
        valid, val, err = _validate_review_internal(["review"])
        assert valid is False
        assert val is None

    def test_too_long_review(self):
        valid, val, err = _validate_review_internal("x" * (MAX_REVIEW_LENGTH + 1))
        assert valid is False
        assert val is None
        assert str(MAX_REVIEW_LENGTH) in err

    def test_exact_max_length(self):
        valid, val, err = _validate_review_internal("x" * MAX_REVIEW_LENGTH)
        assert valid is True
        assert err == ""

    def test_empty_string_is_valid(self):
        valid, val, err = _validate_review_internal("")
        assert valid is True
        # sanitize_string("") -> ""
        assert err == ""

    def test_unicode_review(self):
        valid, val, err = _validate_review_internal("Great template! Merci beaucoup!")
        assert valid is True
        assert err == ""


# =============================================================================
# _validate_review Tests (backward-compatible alias)
# =============================================================================


class TestValidateReview:
    """Tests for _validate_review(value) -> (bool, error_or_None)."""

    def test_valid_returns_true_none(self):
        valid, err = _validate_review("Nice!")
        assert valid is True
        assert err is None

    def test_none_returns_true_none(self):
        valid, err = _validate_review(None)
        assert valid is True
        assert err is None

    def test_non_string_returns_false(self):
        valid, err = _validate_review(42)
        assert valid is False
        assert err is not None

    def test_too_long_returns_false(self):
        valid, err = _validate_review("y" * (MAX_REVIEW_LENGTH + 1))
        assert valid is False
        assert err is not None


# =============================================================================
# _validate_deployment_name_internal Tests
# =============================================================================


class TestValidateDeploymentNameInternal:
    """Tests for _validate_deployment_name_internal(value, fallback)."""

    def test_none_returns_fallback(self):
        valid, val, err = _validate_deployment_name_internal(None, "default-name")
        assert valid is True
        assert val == "default-name"
        assert err == ""

    def test_valid_name(self):
        valid, val, err = _validate_deployment_name_internal("my-deploy", "fb")
        assert valid is True
        assert val == "my-deploy"
        assert err == ""

    def test_non_string_integer(self):
        valid, val, err = _validate_deployment_name_internal(123, "fb")
        assert valid is False
        assert val == ""
        assert "string" in err.lower()

    def test_non_string_list(self):
        valid, val, err = _validate_deployment_name_internal(["name"], "fb")
        assert valid is False
        assert val == ""

    def test_too_long(self):
        valid, val, err = _validate_deployment_name_internal(
            "n" * (MAX_DEPLOYMENT_NAME_LENGTH + 1), "fb"
        )
        assert valid is False
        assert val == ""
        assert str(MAX_DEPLOYMENT_NAME_LENGTH) in err

    def test_exact_max_length(self):
        valid, val, err = _validate_deployment_name_internal("n" * MAX_DEPLOYMENT_NAME_LENGTH, "fb")
        assert valid is True
        assert err == ""

    def test_whitespace_only_returns_fallback(self):
        # sanitize_string("   ") -> "" which is falsy, so fallback is used
        valid, val, err = _validate_deployment_name_internal("   ", "fb")
        assert valid is True
        assert val == "fb"
        assert err == ""

    def test_name_is_sanitized(self):
        valid, val, err = _validate_deployment_name_internal("  trimmed  ", "fb")
        assert valid is True
        assert val == "trimmed"
        assert err == ""

    def test_empty_string_returns_fallback(self):
        valid, val, err = _validate_deployment_name_internal("", "fb")
        assert valid is True
        assert val == "fb"
        assert err == ""


# =============================================================================
# _validate_deployment_name Tests (backward-compatible alias)
# =============================================================================


class TestValidateDeploymentName:
    """Tests for _validate_deployment_name(value, fallback) -> (bool, error_or_None)."""

    def test_valid_returns_true_none(self):
        valid, err = _validate_deployment_name("deploy-name", "fb")
        assert valid is True
        assert err is None

    def test_none_returns_true_none(self):
        valid, err = _validate_deployment_name(None, "fb")
        assert valid is True
        assert err is None

    def test_non_string_returns_false(self):
        valid, err = _validate_deployment_name(123, "fb")
        assert valid is False
        assert err is not None

    def test_too_long_returns_false(self):
        valid, err = _validate_deployment_name("x" * (MAX_DEPLOYMENT_NAME_LENGTH + 1), "fb")
        assert valid is False
        assert err is not None

    def test_default_fallback_empty(self):
        valid, err = _validate_deployment_name(None)
        assert valid is True
        assert err is None


# =============================================================================
# _validate_config Tests
# =============================================================================


class TestValidateConfig:
    """Tests for _validate_config(value) -> (bool, dict, error)."""

    def test_none_returns_empty_dict(self):
        valid, val, err = _validate_config(None)
        assert valid is True
        assert val == {}
        assert err == ""

    def test_valid_empty_dict(self):
        valid, val, err = _validate_config({})
        assert valid is True
        assert val == {}
        assert err == ""

    def test_valid_dict_with_data(self):
        config = {"key1": "value1", "key2": 42}
        valid, val, err = _validate_config(config)
        assert valid is True
        assert val == config
        assert err == ""

    def test_non_dict_string(self):
        valid, val, err = _validate_config("not a dict")
        assert valid is False
        assert val == {}
        assert "dictionary" in err.lower()

    def test_non_dict_list(self):
        valid, val, err = _validate_config([1, 2, 3])
        assert valid is False
        assert val == {}

    def test_non_dict_integer(self):
        valid, val, err = _validate_config(42)
        assert valid is False
        assert val == {}

    def test_too_many_keys(self):
        big_config = {f"key_{i}": f"val_{i}" for i in range(MAX_CONFIG_SIZE + 1)}
        valid, val, err = _validate_config(big_config)
        assert valid is False
        assert val == {}
        assert str(MAX_CONFIG_SIZE) in err

    def test_exact_max_keys(self):
        config = {f"key_{i}": f"val_{i}" for i in range(MAX_CONFIG_SIZE)}
        valid, val, err = _validate_config(config)
        assert valid is True
        assert val == config
        assert err == ""

    def test_one_less_than_max_keys(self):
        config = {f"key_{i}": f"val_{i}" for i in range(MAX_CONFIG_SIZE - 1)}
        valid, val, err = _validate_config(config)
        assert valid is True
        assert err == ""

    def test_nested_dict_counts_only_top_level_keys(self):
        config = {"outer": {"inner1": "a", "inner2": "b"}}
        valid, val, err = _validate_config(config)
        assert valid is True
        assert val == config


# =============================================================================
# _validate_search_query Tests
# =============================================================================


class TestValidateSearchQuery:
    """Tests for _validate_search_query(value) -> (bool, sanitized, error)."""

    def test_none_returns_empty(self):
        valid, val, err = _validate_search_query(None)
        assert valid is True
        assert val == ""
        assert err == ""

    def test_empty_string_returns_empty(self):
        valid, val, err = _validate_search_query("")
        assert valid is True
        assert val == ""
        assert err == ""

    def test_valid_query(self):
        valid, val, err = _validate_search_query("accounting")
        assert valid is True
        assert val == "accounting"
        assert err == ""

    def test_query_is_lowercased(self):
        valid, val, err = _validate_search_query("ACCOUNTING")
        assert valid is True
        assert val == "accounting"

    def test_query_is_sanitized_and_lowered(self):
        valid, val, err = _validate_search_query("  HELLO  ")
        assert valid is True
        assert val == "hello"

    def test_non_string_integer(self):
        valid, val, err = _validate_search_query(123)
        assert valid is False
        assert val == ""
        assert "string" in err.lower()

    def test_non_string_list(self):
        valid, val, err = _validate_search_query(["query"])
        assert valid is False
        assert val == ""

    def test_too_long(self):
        valid, val, err = _validate_search_query("q" * (MAX_SEARCH_QUERY_LENGTH + 1))
        assert valid is False
        assert val == ""
        assert str(MAX_SEARCH_QUERY_LENGTH) in err

    def test_exact_max_length(self):
        valid, val, err = _validate_search_query("q" * MAX_SEARCH_QUERY_LENGTH)
        assert valid is True
        assert err == ""

    def test_mixed_case_query(self):
        valid, val, err = _validate_search_query("Legal Templates")
        assert valid is True
        assert val == "legal templates"


# =============================================================================
# _validate_category Tests
# =============================================================================


class TestValidateCategory:
    """Tests for _validate_category(value) -> (bool, TemplateCategory|None, error)."""

    def test_none_returns_none_category(self):
        valid, cat, err = _validate_category(None)
        assert valid is True
        assert cat is None
        assert err == ""

    def test_empty_string_returns_none_category(self):
        valid, cat, err = _validate_category("")
        assert valid is True
        assert cat is None
        assert err == ""

    def test_valid_category_accounting(self):
        valid, cat, err = _validate_category("accounting")
        assert valid is True
        assert cat == TemplateCategory.ACCOUNTING
        assert err == ""

    def test_valid_category_legal(self):
        valid, cat, err = _validate_category("legal")
        assert valid is True
        assert cat == TemplateCategory.LEGAL

    def test_valid_category_healthcare(self):
        valid, cat, err = _validate_category("healthcare")
        assert valid is True
        assert cat == TemplateCategory.HEALTHCARE

    def test_valid_category_software(self):
        valid, cat, err = _validate_category("software")
        assert valid is True
        assert cat == TemplateCategory.SOFTWARE

    def test_valid_category_regulatory(self):
        valid, cat, err = _validate_category("regulatory")
        assert valid is True
        assert cat == TemplateCategory.REGULATORY

    def test_valid_category_academic(self):
        valid, cat, err = _validate_category("academic")
        assert valid is True
        assert cat == TemplateCategory.ACADEMIC

    def test_valid_category_finance(self):
        valid, cat, err = _validate_category("finance")
        assert valid is True
        assert cat == TemplateCategory.FINANCE

    def test_valid_category_general(self):
        valid, cat, err = _validate_category("general")
        assert valid is True
        assert cat == TemplateCategory.GENERAL

    def test_valid_category_devops(self):
        valid, cat, err = _validate_category("devops")
        assert valid is True
        assert cat == TemplateCategory.DEVOPS

    def test_valid_category_marketing(self):
        valid, cat, err = _validate_category("marketing")
        assert valid is True
        assert cat == TemplateCategory.MARKETING

    def test_case_insensitive_uppercase(self):
        valid, cat, err = _validate_category("ACCOUNTING")
        assert valid is True
        assert cat == TemplateCategory.ACCOUNTING

    def test_case_insensitive_mixed(self):
        valid, cat, err = _validate_category("LeGaL")
        assert valid is True
        assert cat == TemplateCategory.LEGAL

    def test_invalid_category(self):
        valid, cat, err = _validate_category("nonexistent")
        assert valid is False
        assert cat is None
        assert "Invalid category" in err
        assert "Must be one of" in err

    def test_invalid_category_error_lists_all_valid(self):
        valid, cat, err = _validate_category("bogus")
        assert valid is False
        # All categories should appear in error message
        for tc in TemplateCategory:
            assert tc.value in err

    def test_non_string_type(self):
        valid, cat, err = _validate_category(42)
        assert valid is False
        assert cat is None
        assert "string" in err.lower()

    def test_non_string_list(self):
        valid, cat, err = _validate_category(["legal"])
        assert valid is False
        assert cat is None

    def test_all_template_categories_are_valid(self):
        for tc in TemplateCategory:
            valid, cat, err = _validate_category(tc.value)
            assert valid is True, f"Category {tc.value} should be valid"
            assert cat == tc


# =============================================================================
# _validate_category_filter Tests
# =============================================================================


class TestValidateCategoryFilter:
    """Tests for _validate_category_filter(value) -> (bool, str|None, error)."""

    def test_none_returns_none(self):
        valid, val, err = _validate_category_filter(None)
        assert valid is True
        assert val is None
        assert err == ""

    def test_empty_string_returns_none(self):
        valid, val, err = _validate_category_filter("")
        assert valid is True
        assert val is None
        assert err == ""

    def test_valid_category_returns_string_value(self):
        valid, val, err = _validate_category_filter("legal")
        assert valid is True
        assert val == "legal"
        assert err == ""

    def test_valid_category_case_insensitive(self):
        valid, val, err = _validate_category_filter("HEALTHCARE")
        assert valid is True
        assert val == "healthcare"

    def test_invalid_category(self):
        valid, val, err = _validate_category_filter("invalid")
        assert valid is False
        assert val is None
        assert "Invalid category" in err

    def test_non_string_type(self):
        valid, val, err = _validate_category_filter(99)
        assert valid is False
        assert val is None

    def test_all_categories_return_their_value(self):
        for tc in TemplateCategory:
            valid, val, err = _validate_category_filter(tc.value)
            assert valid is True
            assert val == tc.value


# =============================================================================
# Integration / Cross-Function Tests
# =============================================================================


class TestCrossFunctionIntegration:
    """Tests that verify interactions between multiple validation functions."""

    def test_validate_id_consistency_with_template_and_deployment(self):
        """_validate_template_id and _validate_deployment_id should accept the same IDs."""
        test_ids = ["abc", "a-b-c", "test_123", "a" * 128]
        for tid in test_ids:
            t_valid, _ = _validate_template_id(tid)
            d_valid, _ = _validate_deployment_id(tid)
            assert t_valid == d_valid, f"Inconsistent for id={tid!r}"

    def test_validate_id_consistency_rejection(self):
        """Both aliases should reject the same invalid IDs."""
        bad_ids = ["", None, "a" * 129, "bad id", "-leading"]
        for bad in bad_ids:
            t_valid, _ = _validate_template_id(bad)
            d_valid, _ = _validate_deployment_id(bad)
            assert t_valid is False
            assert d_valid is False

    def test_pagination_uses_clamp(self):
        """_validate_pagination should clamp the same way as _clamp_pagination."""
        limit, offset, err = _validate_pagination({"limit": "0", "offset": "-1"})
        cl, co = _clamp_pagination(0, -1)
        assert limit == cl
        assert offset == co

    def test_rating_alias_matches_rating_value(self):
        """_validate_rating should produce same results as _validate_rating_value."""
        for v in [None, "x", 0, 1, 3, 5, 6]:
            r1 = _validate_rating_value(v)
            r2 = _validate_rating(v)
            assert r1 == r2, f"Mismatch for value={v!r}"

    def test_review_alias_matches_internal(self):
        """_validate_review should agree with _validate_review_internal on validity."""
        cases = [None, "ok", 123, "x" * (MAX_REVIEW_LENGTH + 1)]
        for c in cases:
            internal_valid, _, _ = _validate_review_internal(c)
            alias_valid, _ = _validate_review(c)
            assert internal_valid == alias_valid, f"Mismatch for value={c!r}"

    def test_deployment_name_alias_matches_internal(self):
        """_validate_deployment_name should agree with _validate_deployment_name_internal."""
        cases = [None, "ok", 123, "x" * (MAX_DEPLOYMENT_NAME_LENGTH + 1)]
        for c in cases:
            internal_valid, _, _ = _validate_deployment_name_internal(c, "fb")
            alias_valid, _ = _validate_deployment_name(c, "fb")
            assert internal_valid == alias_valid, f"Mismatch for value={c!r}"


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Additional edge cases for thorough coverage."""

    def test_id_with_only_digits(self):
        valid, err = _validate_id("123456")
        assert valid is True

    def test_id_with_uppercase(self):
        valid, err = _validate_id("MyTemplate")
        assert valid is True

    def test_id_with_all_hyphens_after_first(self):
        valid, err = _validate_id("a----")
        assert valid is True

    def test_id_with_all_underscores_after_first(self):
        valid, err = _validate_id("a____")
        assert valid is True

    def test_id_mixed_case_hyphens_underscores(self):
        valid, err = _validate_id("aB-cD_eF-0")
        assert valid is True

    def test_config_with_zero_keys(self):
        valid, val, err = _validate_config({})
        assert valid is True
        assert val == {}

    def test_config_with_none_values(self):
        config = {"key": None}
        valid, val, err = _validate_config(config)
        assert valid is True
        assert val == config

    def test_search_query_with_special_chars(self):
        valid, val, err = _validate_search_query("hello & world")
        assert valid is True
        assert "hello & world" == val

    def test_review_with_newlines(self):
        valid, val, err = _validate_review_internal("line1\nline2\nline3")
        assert valid is True
        assert err == ""

    def test_deployment_name_truncated_by_sanitize(self):
        # sanitize_string truncates to max_length; name at exactly max_length is fine
        name = "a" * MAX_DEPLOYMENT_NAME_LENGTH
        valid, val, err = _validate_deployment_name_internal(name, "fb")
        assert valid is True
        assert len(val) <= MAX_DEPLOYMENT_NAME_LENGTH

    def test_clamp_pagination_with_exactly_min_and_max(self):
        limit, offset = _clamp_pagination(MIN_LIMIT, 0)
        assert limit == MIN_LIMIT
        limit, offset = _clamp_pagination(MAX_LIMIT, MAX_OFFSET)
        assert limit == MAX_LIMIT
        assert offset == MAX_OFFSET

    def test_category_filter_returns_string_not_enum(self):
        """_validate_category_filter returns enum .value (str), not TemplateCategory."""
        valid, val, err = _validate_category_filter("general")
        assert isinstance(val, str)
        assert not isinstance(val, TemplateCategory)

    def test_category_returns_enum_not_string(self):
        """_validate_category returns TemplateCategory, not a bare string."""
        valid, cat, err = _validate_category("general")
        assert isinstance(cat, TemplateCategory)
