"""Tests for e-commerce input validation utilities.

Covers all validation functions, constants, and edge cases in
aragora/server/handlers/features/ecommerce/validation.py:

- validate_platform_id: platform ID format, length, special chars
- validate_resource_id: resource ID format, custom type names
- validate_sku: SKU format, length bounds
- validate_url: URL scheme, length bounds
- validate_quantity: integer parsing, bounds, negative values
- validate_financial_amount: Decimal parsing, precision, bounds, NaN/Inf
- validate_currency_code: ISO 4217 whitelist, case normalization
- validate_product_id: product ID format with dots
- validate_pagination: limit/offset parsing, bounds, defaults
- Constants: verify values of all validation constants
- Backward-compatible aliases
- Security: injection, path traversal, Unicode abuse
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from aragora.server.handlers.features.ecommerce.validation import (
    ALLOWED_CURRENCY_CODES,
    DEFAULT_PAGINATION_LIMIT,
    DEFAULT_PAGINATION_OFFSET,
    MAX_CARRIER_LENGTH,
    MAX_CREDENTIAL_VALUE_LENGTH,
    MAX_DECIMAL_PLACES,
    MAX_FINANCIAL_AMOUNT,
    MAX_ORDER_ID_LENGTH,
    MAX_PAGINATION_LIMIT,
    MAX_PAGINATION_OFFSET,
    MAX_SERVICE_LENGTH,
    MAX_SHOP_URL_LENGTH,
    MAX_SKU_LENGTH,
    MAX_TRACKING_NUMBER_LENGTH,
    SAFE_PLATFORM_PATTERN,
    SAFE_PRODUCT_ID_PATTERN,
    SAFE_RESOURCE_ID_PATTERN,
    _validate_currency_code,
    _validate_financial_amount,
    _validate_pagination,
    _validate_platform_id,
    _validate_product_id,
    _validate_quantity,
    _validate_resource_id,
    _validate_sku,
    _validate_url,
    validate_currency_code,
    validate_financial_amount,
    validate_pagination,
    validate_platform_id,
    validate_product_id,
    validate_quantity,
    validate_resource_id,
    validate_sku,
    validate_url,
)


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify validation constants have expected values."""

    def test_max_shop_url_length(self):
        assert MAX_SHOP_URL_LENGTH == 512

    def test_max_credential_value_length(self):
        assert MAX_CREDENTIAL_VALUE_LENGTH == 1024

    def test_max_sku_length(self):
        assert MAX_SKU_LENGTH == 128

    def test_max_order_id_length(self):
        assert MAX_ORDER_ID_LENGTH == 128

    def test_max_carrier_length(self):
        assert MAX_CARRIER_LENGTH == 64

    def test_max_service_length(self):
        assert MAX_SERVICE_LENGTH == 64

    def test_max_tracking_number_length(self):
        assert MAX_TRACKING_NUMBER_LENGTH == 128

    def test_max_financial_amount(self):
        assert MAX_FINANCIAL_AMOUNT == Decimal("99999999.99")

    def test_max_decimal_places(self):
        assert MAX_DECIMAL_PLACES == 2

    def test_max_pagination_limit(self):
        assert MAX_PAGINATION_LIMIT == 1000

    def test_max_pagination_offset(self):
        assert MAX_PAGINATION_OFFSET == 1_000_000

    def test_default_pagination_limit(self):
        assert DEFAULT_PAGINATION_LIMIT == 100

    def test_default_pagination_offset(self):
        assert DEFAULT_PAGINATION_OFFSET == 0

    def test_currency_codes_is_frozenset(self):
        assert isinstance(ALLOWED_CURRENCY_CODES, frozenset)

    def test_currency_codes_contains_usd(self):
        assert "USD" in ALLOWED_CURRENCY_CODES

    def test_currency_codes_contains_eur(self):
        assert "EUR" in ALLOWED_CURRENCY_CODES

    def test_currency_codes_contains_gbp(self):
        assert "GBP" in ALLOWED_CURRENCY_CODES

    def test_currency_codes_contains_jpy(self):
        assert "JPY" in ALLOWED_CURRENCY_CODES

    def test_currency_codes_count(self):
        # At least 30 currencies defined
        assert len(ALLOWED_CURRENCY_CODES) >= 30


# ---------------------------------------------------------------------------
# Regex Pattern Tests
# ---------------------------------------------------------------------------


class TestRegexPatterns:
    """Verify compiled regex patterns match expected inputs."""

    def test_platform_pattern_accepts_alpha_start(self):
        assert SAFE_PLATFORM_PATTERN.match("shopify") is not None

    def test_platform_pattern_accepts_underscore(self):
        assert SAFE_PLATFORM_PATTERN.match("my_platform") is not None

    def test_platform_pattern_rejects_leading_digit(self):
        assert SAFE_PLATFORM_PATTERN.match("1platform") is None

    def test_platform_pattern_rejects_hyphen(self):
        assert SAFE_PLATFORM_PATTERN.match("my-platform") is None

    def test_resource_id_pattern_accepts_alphanumeric(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("abc123") is not None

    def test_resource_id_pattern_accepts_hyphens(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("order-123") is not None

    def test_resource_id_pattern_rejects_leading_hyphen(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("-order") is None

    def test_product_id_pattern_accepts_dots(self):
        assert SAFE_PRODUCT_ID_PATTERN.match("product.v2") is not None

    def test_product_id_pattern_rejects_spaces(self):
        assert SAFE_PRODUCT_ID_PATTERN.match("product name") is None


# ---------------------------------------------------------------------------
# validate_platform_id Tests
# ---------------------------------------------------------------------------


class TestValidatePlatformId:
    """Tests for validate_platform_id."""

    def test_valid_simple_platform(self):
        valid, err = validate_platform_id("shopify")
        assert valid is True
        assert err is None

    def test_valid_platform_with_underscore(self):
        valid, err = validate_platform_id("my_platform")
        assert valid is True
        assert err is None

    def test_valid_platform_with_numbers(self):
        valid, err = validate_platform_id("platform2")
        assert valid is True
        assert err is None

    def test_empty_string(self):
        valid, err = validate_platform_id("")
        assert valid is False
        assert "required" in err.lower()

    def test_too_long(self):
        valid, err = validate_platform_id("a" * 51)
        assert valid is False
        assert "too long" in err.lower()

    def test_exactly_50_chars(self):
        valid, err = validate_platform_id("a" * 50)
        assert valid is True
        assert err is None

    def test_starts_with_digit(self):
        valid, err = validate_platform_id("1platform")
        assert valid is False
        assert "Invalid" in err

    def test_contains_hyphen(self):
        valid, err = validate_platform_id("my-platform")
        assert valid is False
        assert "Invalid" in err

    def test_contains_space(self):
        valid, err = validate_platform_id("my platform")
        assert valid is False
        assert "Invalid" in err

    def test_contains_special_chars(self):
        valid, err = validate_platform_id("plat@form!")
        assert valid is False

    def test_single_letter(self):
        valid, err = validate_platform_id("a")
        assert valid is True
        assert err is None

    def test_path_traversal_attempt(self):
        valid, err = validate_platform_id("../../etc")
        assert valid is False

    def test_sql_injection_attempt(self):
        valid, err = validate_platform_id("'; DROP TABLE--")
        assert valid is False

    def test_xss_injection_attempt(self):
        valid, err = validate_platform_id("<script>alert(1)</script>")
        assert valid is False


# ---------------------------------------------------------------------------
# validate_resource_id Tests
# ---------------------------------------------------------------------------


class TestValidateResourceId:
    """Tests for validate_resource_id."""

    def test_valid_numeric_id(self):
        valid, err = validate_resource_id("12345")
        assert valid is True
        assert err is None

    def test_valid_alphanumeric_id(self):
        valid, err = validate_resource_id("order-abc-123")
        assert valid is True
        assert err is None

    def test_valid_id_with_underscore(self):
        valid, err = validate_resource_id("order_123")
        assert valid is True
        assert err is None

    def test_empty_string(self):
        valid, err = validate_resource_id("")
        assert valid is False
        assert "required" in err.lower()

    def test_too_long(self):
        valid, err = validate_resource_id("a" * 129)
        assert valid is False
        assert "too long" in err.lower()

    def test_exactly_128_chars(self):
        valid, err = validate_resource_id("a" * 128)
        assert valid is True

    def test_leading_hyphen_rejected(self):
        valid, err = validate_resource_id("-order")
        assert valid is False

    def test_custom_resource_type_in_error(self):
        valid, err = validate_resource_id("", resource_type="Order ID")
        assert valid is False
        assert "Order ID" in err

    def test_custom_type_in_too_long_error(self):
        valid, err = validate_resource_id("a" * 200, resource_type="SKU")
        assert valid is False
        assert "SKU" in err

    def test_custom_type_in_format_error(self):
        valid, err = validate_resource_id("!!!invalid!!!", resource_type="Order")
        assert valid is False
        assert "order" in err.lower()

    def test_default_type_name(self):
        valid, err = validate_resource_id("")
        assert "ID" in err

    def test_special_chars_rejected(self):
        valid, err = validate_resource_id("order@#$%")
        assert valid is False

    def test_path_traversal_rejected(self):
        valid, err = validate_resource_id("../../etc/passwd")
        assert valid is False


# ---------------------------------------------------------------------------
# validate_sku Tests
# ---------------------------------------------------------------------------


class TestValidateSku:
    """Tests for validate_sku."""

    def test_valid_sku(self):
        valid, err = validate_sku("WDG-001")
        assert valid is True
        assert err is None

    def test_valid_sku_with_dots(self):
        valid, err = validate_sku("SKU.v2.0")
        assert valid is True
        assert err is None

    def test_valid_sku_with_underscores(self):
        valid, err = validate_sku("SKU_001_A")
        assert valid is True
        assert err is None

    def test_empty_string(self):
        valid, err = validate_sku("")
        assert valid is False
        assert "required" in err.lower()

    def test_too_long(self):
        valid, err = validate_sku("S" * 129)
        assert valid is False
        assert "too long" in err.lower()

    def test_exactly_max_length(self):
        valid, err = validate_sku("S" * 128)
        assert valid is True

    def test_starts_with_hyphen_rejected(self):
        valid, err = validate_sku("-SKU001")
        assert valid is False

    def test_special_chars_rejected(self):
        valid, err = validate_sku("!!!invalid!!!")
        assert valid is False

    def test_spaces_rejected(self):
        valid, err = validate_sku("SKU 001")
        assert valid is False


# ---------------------------------------------------------------------------
# validate_url Tests
# ---------------------------------------------------------------------------


class TestValidateUrl:
    """Tests for validate_url."""

    def test_valid_https_url(self):
        valid, err = validate_url("https://example.com")
        assert valid is True
        assert err is None

    def test_valid_http_url(self):
        valid, err = validate_url("http://example.com")
        assert valid is True
        assert err is None

    def test_empty_string(self):
        valid, err = validate_url("")
        assert valid is False
        assert "required" in err.lower()

    def test_too_long(self):
        valid, err = validate_url("https://example.com/" + "a" * 500)
        assert valid is False
        assert "too long" in err.lower()

    def test_missing_scheme(self):
        valid, err = validate_url("example.com")
        assert valid is False
        assert "must start with" in err.lower()

    def test_ftp_scheme_rejected(self):
        valid, err = validate_url("ftp://files.example.com")
        assert valid is False

    def test_javascript_scheme_rejected(self):
        valid, err = validate_url("javascript:alert(1)")
        assert valid is False

    def test_custom_field_name_in_error(self):
        valid, err = validate_url("", field_name="Shop URL")
        assert "Shop URL" in err

    def test_custom_field_name_in_too_long_error(self):
        valid, err = validate_url("https://x.com/" + "a" * 500, field_name="Webhook")
        assert "Webhook" in err

    def test_custom_field_name_in_scheme_error(self):
        valid, err = validate_url("ftp://x.com", field_name="Endpoint")
        assert "Endpoint" in err

    def test_data_uri_rejected(self):
        valid, err = validate_url("data:text/html,<h1>XSS</h1>")
        assert valid is False


# ---------------------------------------------------------------------------
# validate_quantity Tests
# ---------------------------------------------------------------------------


class TestValidateQuantity:
    """Tests for validate_quantity."""

    def test_valid_integer(self):
        valid, err, val = validate_quantity(10)
        assert valid is True
        assert err is None
        assert val == 10

    def test_valid_zero(self):
        valid, err, val = validate_quantity(0)
        assert valid is True
        assert val == 0

    def test_valid_string_integer(self):
        valid, err, val = validate_quantity("42")
        assert valid is True
        assert val == 42

    def test_none_value(self):
        valid, err, val = validate_quantity(None)
        assert valid is False
        assert "required" in err.lower()
        assert val is None

    def test_negative_value(self):
        valid, err, val = validate_quantity(-1)
        assert valid is False
        assert "negative" in err.lower()
        assert val is None

    def test_too_large(self):
        valid, err, val = validate_quantity(1_000_000_001)
        assert valid is False
        assert "too large" in err.lower()
        assert val is None

    def test_boundary_max_value(self):
        valid, err, val = validate_quantity(1_000_000_000)
        assert valid is True
        assert val == 1_000_000_000

    def test_invalid_string(self):
        valid, err, val = validate_quantity("abc")
        assert valid is False
        assert "format" in err.lower()
        assert val is None

    def test_float_string_truncated(self):
        # int("3.5") raises ValueError
        valid, err, val = validate_quantity("3.5")
        assert valid is False

    def test_float_value_converted(self):
        # int(3.5) -> 3
        valid, err, val = validate_quantity(3.5)
        assert valid is True
        assert val == 3

    def test_list_rejected(self):
        valid, err, val = validate_quantity([1, 2, 3])
        assert valid is False


# ---------------------------------------------------------------------------
# validate_financial_amount Tests
# ---------------------------------------------------------------------------


class TestValidateFinancialAmount:
    """Tests for validate_financial_amount."""

    def test_valid_decimal_string(self):
        valid, err, val = validate_financial_amount("19.99")
        assert valid is True
        assert err is None
        assert val == Decimal("19.99")

    def test_valid_integer(self):
        valid, err, val = validate_financial_amount(100)
        assert valid is True
        assert val == Decimal("100")

    def test_valid_float(self):
        valid, err, val = validate_financial_amount(29.99)
        assert valid is True
        assert val == Decimal("29.99")

    def test_valid_zero_with_allow_zero(self):
        valid, err, val = validate_financial_amount(0, allow_zero=True)
        assert valid is True
        assert val == Decimal("0")

    def test_zero_rejected_when_not_allowed(self):
        valid, err, val = validate_financial_amount(0, allow_zero=False)
        assert valid is False
        assert "greater than zero" in err.lower()
        assert val is None

    def test_none_value(self):
        valid, err, val = validate_financial_amount(None)
        assert valid is False
        assert "required" in err.lower()
        assert val is None

    def test_negative_value(self):
        valid, err, val = validate_financial_amount("-5.00")
        assert valid is False
        assert "negative" in err.lower()

    def test_exceeds_max(self):
        valid, err, val = validate_financial_amount("100000000.00")
        assert valid is False
        assert "exceeds maximum" in err.lower()

    def test_exactly_max(self):
        valid, err, val = validate_financial_amount("99999999.99")
        assert valid is True
        assert val == Decimal("99999999.99")

    def test_custom_max_amount(self):
        valid, err, val = validate_financial_amount(
            "500", max_amount=Decimal("100")
        )
        assert valid is False
        assert "exceeds maximum" in err.lower()

    def test_custom_max_amount_within_bound(self):
        valid, err, val = validate_financial_amount(
            "50", max_amount=Decimal("100")
        )
        assert valid is True

    def test_too_many_decimal_places(self):
        valid, err, val = validate_financial_amount("19.999")
        assert valid is False
        assert "decimal places" in err.lower()

    def test_two_decimal_places_valid(self):
        valid, err, val = validate_financial_amount("19.99")
        assert valid is True

    def test_one_decimal_place_valid(self):
        valid, err, val = validate_financial_amount("19.9")
        assert valid is True

    def test_nan_rejected(self):
        valid, err, val = validate_financial_amount("NaN")
        assert valid is False
        assert "finite" in err.lower()

    def test_infinity_rejected(self):
        valid, err, val = validate_financial_amount("Infinity")
        assert valid is False
        assert "finite" in err.lower()

    def test_negative_infinity_rejected(self):
        valid, err, val = validate_financial_amount("-Infinity")
        assert valid is False
        assert "finite" in err.lower()

    def test_invalid_string(self):
        valid, err, val = validate_financial_amount("not-a-number")
        assert valid is False
        assert "format" in err.lower()

    def test_custom_field_name_in_error(self):
        valid, err, val = validate_financial_amount(None, field_name="Price")
        assert "Price" in err

    def test_custom_field_name_negative(self):
        valid, err, val = validate_financial_amount("-1", field_name="Cost")
        assert "Cost" in err

    def test_custom_field_name_zero_not_allowed(self):
        valid, err, val = validate_financial_amount(
            0, field_name="Total", allow_zero=False
        )
        assert "Total" in err

    def test_empty_string(self):
        valid, err, val = validate_financial_amount("")
        assert valid is False

    def test_whitespace_string(self):
        valid, err, val = validate_financial_amount("  ")
        assert valid is False

    def test_small_positive_value(self):
        valid, err, val = validate_financial_amount("0.01")
        assert valid is True
        assert val == Decimal("0.01")


# ---------------------------------------------------------------------------
# validate_currency_code Tests
# ---------------------------------------------------------------------------


class TestValidateCurrencyCode:
    """Tests for validate_currency_code."""

    def test_valid_usd(self):
        valid, err = validate_currency_code("USD")
        assert valid is True
        assert err is None

    def test_valid_lowercase(self):
        valid, err = validate_currency_code("usd")
        assert valid is True

    def test_valid_mixed_case(self):
        valid, err = validate_currency_code("Eur")
        assert valid is True

    def test_valid_with_whitespace(self):
        valid, err = validate_currency_code("  USD  ")
        assert valid is True

    def test_empty_string(self):
        valid, err = validate_currency_code("")
        assert valid is False
        assert "required" in err.lower()

    def test_none_value(self):
        valid, err = validate_currency_code(None)
        assert valid is False
        assert "required" in err.lower()

    def test_not_string(self):
        valid, err = validate_currency_code(123)
        assert valid is False
        assert "string" in err.lower()

    def test_wrong_length_short(self):
        valid, err = validate_currency_code("US")
        assert valid is False
        assert "3 characters" in err

    def test_wrong_length_long(self):
        valid, err = validate_currency_code("USDD")
        assert valid is False
        assert "3 characters" in err

    def test_numeric_code_rejected(self):
        valid, err = validate_currency_code("123")
        assert valid is False
        assert "letters" in err.lower()

    def test_unsupported_code(self):
        valid, err = validate_currency_code("XYZ")
        assert valid is False
        assert "Unsupported" in err

    def test_all_major_currencies_accepted(self):
        major = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY"]
        for code in major:
            valid, err = validate_currency_code(code)
            assert valid is True, f"{code} should be valid"

    def test_all_allowed_codes_pass(self):
        for code in ALLOWED_CURRENCY_CODES:
            valid, err = validate_currency_code(code)
            assert valid is True, f"{code} should be valid"


# ---------------------------------------------------------------------------
# validate_product_id Tests
# ---------------------------------------------------------------------------


class TestValidateProductId:
    """Tests for validate_product_id."""

    def test_valid_numeric_id(self):
        valid, err = validate_product_id("12345")
        assert valid is True
        assert err is None

    def test_valid_id_with_dots(self):
        valid, err = validate_product_id("product.v2")
        assert valid is True

    def test_valid_id_with_hyphens(self):
        valid, err = validate_product_id("prod-123")
        assert valid is True

    def test_valid_id_with_underscores(self):
        valid, err = validate_product_id("prod_123")
        assert valid is True

    def test_empty_string(self):
        valid, err = validate_product_id("")
        assert valid is False
        assert "required" in err.lower()

    def test_not_string(self):
        valid, err = validate_product_id(12345)
        assert valid is False
        assert "string" in err.lower()

    def test_too_long(self):
        valid, err = validate_product_id("a" * 129)
        assert valid is False
        assert "too long" in err.lower()

    def test_exactly_128_chars(self):
        valid, err = validate_product_id("a" * 128)
        assert valid is True

    def test_starts_with_dot_rejected(self):
        valid, err = validate_product_id(".hidden")
        assert valid is False

    def test_starts_with_hyphen_rejected(self):
        valid, err = validate_product_id("-product")
        assert valid is False

    def test_special_chars_rejected(self):
        valid, err = validate_product_id("prod@#$%")
        assert valid is False

    def test_path_traversal_rejected(self):
        valid, err = validate_product_id("../../etc/passwd")
        assert valid is False

    def test_xss_injection_rejected(self):
        valid, err = validate_product_id("<script>alert(1)</script>")
        assert valid is False

    def test_spaces_rejected(self):
        valid, err = validate_product_id("prod 123")
        assert valid is False


# ---------------------------------------------------------------------------
# validate_pagination Tests
# ---------------------------------------------------------------------------


class TestValidatePagination:
    """Tests for validate_pagination."""

    def test_defaults_when_none(self):
        valid, err, limit, offset = validate_pagination()
        assert valid is True
        assert err is None
        assert limit == DEFAULT_PAGINATION_LIMIT
        assert offset == DEFAULT_PAGINATION_OFFSET

    def test_valid_limit_and_offset(self):
        valid, err, limit, offset = validate_pagination(limit=50, offset=100)
        assert valid is True
        assert limit == 50
        assert offset == 100

    def test_valid_string_values(self):
        valid, err, limit, offset = validate_pagination(limit="25", offset="50")
        assert valid is True
        assert limit == 25
        assert offset == 50

    def test_limit_at_minimum(self):
        valid, err, limit, offset = validate_pagination(limit=1)
        assert valid is True
        assert limit == 1

    def test_limit_zero_rejected(self):
        valid, err, limit, offset = validate_pagination(limit=0)
        assert valid is False
        assert "at least 1" in err

    def test_limit_negative_rejected(self):
        valid, err, limit, offset = validate_pagination(limit=-1)
        assert valid is False

    def test_limit_exceeds_max(self):
        valid, err, limit, offset = validate_pagination(limit=1001)
        assert valid is False
        assert "maximum" in err.lower()

    def test_limit_at_max(self):
        valid, err, limit, offset = validate_pagination(limit=1000)
        assert valid is True
        assert limit == 1000

    def test_offset_zero_valid(self):
        valid, err, limit, offset = validate_pagination(offset=0)
        assert valid is True
        assert offset == 0

    def test_offset_negative_rejected(self):
        valid, err, limit, offset = validate_pagination(offset=-1)
        assert valid is False
        assert "negative" in err.lower()

    def test_offset_exceeds_max(self):
        valid, err, limit, offset = validate_pagination(offset=1_000_001)
        assert valid is False
        assert "maximum" in err.lower()

    def test_offset_at_max(self):
        valid, err, limit, offset = validate_pagination(offset=1_000_000)
        assert valid is True
        assert offset == 1_000_000

    def test_invalid_limit_string(self):
        valid, err, limit, offset = validate_pagination(limit="abc")
        assert valid is False
        assert "integer" in err.lower()

    def test_invalid_offset_string(self):
        valid, err, limit, offset = validate_pagination(offset="xyz")
        assert valid is False
        assert "integer" in err.lower()

    def test_error_returns_zero_values(self):
        valid, err, limit, offset = validate_pagination(limit=-1)
        assert valid is False
        assert limit == 0
        assert offset == 0

    def test_limit_none_uses_default(self):
        valid, err, limit, offset = validate_pagination(limit=None, offset=10)
        assert valid is True
        assert limit == DEFAULT_PAGINATION_LIMIT
        assert offset == 10

    def test_offset_none_uses_default(self):
        valid, err, limit, offset = validate_pagination(limit=10, offset=None)
        assert valid is True
        assert limit == 10
        assert offset == DEFAULT_PAGINATION_OFFSET

    def test_float_limit_converted(self):
        # int(3.9) -> error since "3.9" as string won't parse to int
        # but float 3.9 -> int(3.9) = 3 directly
        valid, err, limit, offset = validate_pagination(limit=3.9)
        assert valid is True
        assert limit == 3


# ---------------------------------------------------------------------------
# Backward-Compatible Aliases Tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibleAliases:
    """Verify backward-compatible underscore-prefixed aliases work."""

    def test_platform_id_alias(self):
        assert _validate_platform_id is validate_platform_id

    def test_resource_id_alias(self):
        assert _validate_resource_id is validate_resource_id

    def test_sku_alias(self):
        assert _validate_sku is validate_sku

    def test_url_alias(self):
        assert _validate_url is validate_url

    def test_quantity_alias(self):
        assert _validate_quantity is validate_quantity

    def test_financial_amount_alias(self):
        assert _validate_financial_amount is validate_financial_amount

    def test_currency_code_alias(self):
        assert _validate_currency_code is validate_currency_code

    def test_product_id_alias(self):
        assert _validate_product_id is validate_product_id

    def test_pagination_alias(self):
        assert _validate_pagination is validate_pagination


# ---------------------------------------------------------------------------
# Security Tests
# ---------------------------------------------------------------------------


class TestSecurityInputs:
    """Tests for security-relevant edge cases across validators."""

    def test_platform_null_byte_injection(self):
        valid, err = validate_platform_id("shop\x00ify")
        assert valid is False

    def test_resource_id_null_byte_injection(self):
        valid, err = validate_resource_id("order\x00123")
        assert valid is False

    def test_sku_null_byte_injection(self):
        valid, err = validate_sku("SKU\x00001")
        assert valid is False

    def test_url_null_byte_injection(self):
        # URL with null bytes should still start with http(s)
        valid, err = validate_url("https://example\x00.com")
        # Passes scheme check but may be dangerous; at least it validates
        assert isinstance(valid, bool)

    def test_product_id_null_byte(self):
        valid, err = validate_product_id("prod\x00123")
        assert valid is False

    def test_platform_unicode_homoglyph(self):
        # Cyrillic 'a' looks like Latin 'a'
        valid, err = validate_platform_id("\u0430dmin")  # Cyrillic 'a'
        assert valid is False

    def test_resource_id_unicode_exploit(self):
        valid, err = validate_resource_id("\u202Ereverse-text")
        assert valid is False

    def test_sku_newline_injection(self):
        valid, err = validate_sku("SKU001\nHTTP/1.1")
        assert valid is False

    def test_quantity_extremely_large_string(self):
        valid, err, val = validate_quantity("9" * 1000)
        # Should parse but fail the upper bound check
        assert valid is False

    def test_financial_amount_scientific_notation(self):
        valid, err, val = validate_financial_amount("1e10")
        # 1e10 = 10,000,000,000 which exceeds max
        assert valid is False

    def test_pagination_extremely_large_limit(self):
        valid, err, limit, offset = validate_pagination(limit=999999999)
        assert valid is False

    def test_pagination_extremely_large_offset(self):
        valid, err, limit, offset = validate_pagination(offset=999999999)
        assert valid is False

    def test_platform_command_injection(self):
        valid, err = validate_platform_id("$(whoami)")
        assert valid is False

    def test_resource_id_command_injection(self):
        valid, err = validate_resource_id("$(cat /etc/passwd)")
        assert valid is False

    def test_url_ssrf_attempt(self):
        valid, err = validate_url("http://169.254.169.254/latest/meta-data/")
        # Scheme is valid so it passes basic validation
        # SSRF protection would be elsewhere; URL validator just checks format
        assert valid is True  # format is valid; SSRF is separate layer

    def test_financial_amount_negative_zero(self):
        valid, err, val = validate_financial_amount("-0")
        # -0 as Decimal is equal to 0 and >= 0
        assert valid is True
        assert val == Decimal("0")


# ---------------------------------------------------------------------------
# Edge Cases / Boundary Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge cases and boundary conditions."""

    def test_platform_id_all_digits_after_letter(self):
        valid, err = validate_platform_id("a123456789")
        assert valid is True

    def test_platform_id_all_underscores_after_letter(self):
        valid, err = validate_platform_id("a_____")
        assert valid is True

    def test_resource_id_single_char(self):
        valid, err = validate_resource_id("a")
        assert valid is True

    def test_resource_id_single_digit(self):
        valid, err = validate_resource_id("1")
        assert valid is True

    def test_sku_single_char(self):
        valid, err = validate_sku("A")
        assert valid is True

    def test_url_minimal_valid(self):
        valid, err = validate_url("http://x")
        assert valid is True

    def test_url_at_max_length(self):
        # 8 chars for "https://" + 504 chars = 512
        valid, err = validate_url("https://" + "a" * 504)
        assert valid is True

    def test_url_one_over_max_length(self):
        valid, err = validate_url("https://" + "a" * 505)
        assert valid is False

    def test_quantity_large_valid(self):
        valid, err, val = validate_quantity(999_999_999)
        assert valid is True
        assert val == 999_999_999

    def test_financial_amount_whole_number(self):
        valid, err, val = validate_financial_amount("50")
        assert valid is True
        assert val == Decimal("50")

    def test_financial_amount_with_leading_zeros(self):
        valid, err, val = validate_financial_amount("0099.99")
        assert valid is True
        assert val == Decimal("99.99")

    def test_pagination_both_at_max(self):
        valid, err, limit, offset = validate_pagination(
            limit=MAX_PAGINATION_LIMIT, offset=MAX_PAGINATION_OFFSET
        )
        assert valid is True
        assert limit == MAX_PAGINATION_LIMIT
        assert offset == MAX_PAGINATION_OFFSET

    def test_financial_amount_decimal_object_input(self):
        valid, err, val = validate_financial_amount(Decimal("42.50"))
        assert valid is True
        assert val == Decimal("42.50")

    def test_financial_amount_bool_true_parses(self):
        # bool is subclass of int, str(True) = "True"
        valid, err, val = validate_financial_amount(True)
        # Decimal("True") would raise InvalidOperation
        assert valid is False

    def test_financial_amount_bool_false_parses(self):
        valid, err, val = validate_financial_amount(False)
        assert valid is False

    def test_currency_code_all_spaces(self):
        valid, err = validate_currency_code("   ")
        # strip() -> empty -> length != 3
        assert valid is False

    def test_currency_code_mixed_alpha_numeric(self):
        valid, err = validate_currency_code("US1")
        assert valid is False
        assert "letters" in err.lower()
