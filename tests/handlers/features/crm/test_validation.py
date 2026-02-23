"""Comprehensive tests for the CRM validation module.

Tests all constants and public functions in
aragora/server/handlers/features/crm/validation.py:

Constants:
- SAFE_PLATFORM_PATTERN, SAFE_RESOURCE_ID_PATTERN, EMAIL_PATTERN
- MAX_EMAIL_LENGTH, MAX_NAME_LENGTH, MAX_PHONE_LENGTH, MAX_COMPANY_NAME_LENGTH
- MAX_JOB_TITLE_LENGTH, MAX_DOMAIN_LENGTH, MAX_DEAL_NAME_LENGTH
- MAX_STAGE_LENGTH, MAX_PIPELINE_LENGTH, MAX_CREDENTIAL_VALUE_LENGTH
- MAX_SEARCH_QUERY_LENGTH

Functions:
- validate_platform_id(platform) -> (bool, str|None)
- validate_resource_id(resource_id, resource_type) -> (bool, str|None)
- validate_email(email, required) -> (bool, str|None)
- validate_string_field(value, field_name, max_length, required) -> (bool, str|None)
- validate_amount(amount) -> (bool, str|None, float|None)
- validate_probability(probability) -> (bool, str|None, float|None)

Covers: happy paths, boundary values, edge cases, error messages, type coercion.
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.features.crm.validation import (
    EMAIL_PATTERN,
    MAX_COMPANY_NAME_LENGTH,
    MAX_CREDENTIAL_VALUE_LENGTH,
    MAX_DEAL_NAME_LENGTH,
    MAX_DOMAIN_LENGTH,
    MAX_EMAIL_LENGTH,
    MAX_JOB_TITLE_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PHONE_LENGTH,
    MAX_PIPELINE_LENGTH,
    MAX_SEARCH_QUERY_LENGTH,
    MAX_STAGE_LENGTH,
    SAFE_PLATFORM_PATTERN,
    SAFE_RESOURCE_ID_PATTERN,
    validate_amount,
    validate_email,
    validate_platform_id,
    validate_probability,
    validate_resource_id,
    validate_string_field,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Verify constant values match documented limits."""

    def test_max_email_length(self):
        assert MAX_EMAIL_LENGTH == 254

    def test_max_name_length(self):
        assert MAX_NAME_LENGTH == 128

    def test_max_phone_length(self):
        assert MAX_PHONE_LENGTH == 32

    def test_max_company_name_length(self):
        assert MAX_COMPANY_NAME_LENGTH == 256

    def test_max_job_title_length(self):
        assert MAX_JOB_TITLE_LENGTH == 128

    def test_max_domain_length(self):
        assert MAX_DOMAIN_LENGTH == 253

    def test_max_deal_name_length(self):
        assert MAX_DEAL_NAME_LENGTH == 256

    def test_max_stage_length(self):
        assert MAX_STAGE_LENGTH == 64

    def test_max_pipeline_length(self):
        assert MAX_PIPELINE_LENGTH == 64

    def test_max_credential_value_length(self):
        assert MAX_CREDENTIAL_VALUE_LENGTH == 1024

    def test_max_search_query_length(self):
        assert MAX_SEARCH_QUERY_LENGTH == 256


# =============================================================================
# Pattern Tests
# =============================================================================


class TestSafePlatformPattern:
    """Test SAFE_PLATFORM_PATTERN regex."""

    def test_simple_alpha(self):
        assert SAFE_PLATFORM_PATTERN.match("salesforce")

    def test_mixed_case(self):
        assert SAFE_PLATFORM_PATTERN.match("HubSpot")

    def test_alpha_with_underscores(self):
        assert SAFE_PLATFORM_PATTERN.match("my_crm")

    def test_alpha_with_digits(self):
        assert SAFE_PLATFORM_PATTERN.match("crm2")

    def test_mixed_all(self):
        assert SAFE_PLATFORM_PATTERN.match("My_CRM_v2")

    def test_rejects_leading_digit(self):
        assert not SAFE_PLATFORM_PATTERN.match("2crm")

    def test_rejects_leading_underscore(self):
        assert not SAFE_PLATFORM_PATTERN.match("_crm")

    def test_rejects_hyphen(self):
        assert not SAFE_PLATFORM_PATTERN.match("my-crm")

    def test_rejects_space(self):
        assert not SAFE_PLATFORM_PATTERN.match("my crm")

    def test_max_length_50(self):
        # 1 alpha + 49 more = 50 total
        assert SAFE_PLATFORM_PATTERN.match("a" * 50)

    def test_rejects_over_50(self):
        assert not SAFE_PLATFORM_PATTERN.match("a" * 51)

    def test_rejects_empty(self):
        assert not SAFE_PLATFORM_PATTERN.match("")

    def test_single_letter(self):
        assert SAFE_PLATFORM_PATTERN.match("x")


class TestSafeResourceIdPattern:
    """Test SAFE_RESOURCE_ID_PATTERN regex."""

    def test_simple_alpha(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("contact123")

    def test_leading_digit(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("1234")

    def test_hyphens(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("abc-def-ghi")

    def test_underscores(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("abc_def_ghi")

    def test_mixed(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("contact_123-abc")

    def test_max_length_128(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("a" * 128)

    def test_rejects_over_128(self):
        assert not SAFE_RESOURCE_ID_PATTERN.match("a" * 129)

    def test_rejects_leading_hyphen(self):
        assert not SAFE_RESOURCE_ID_PATTERN.match("-abc")

    def test_rejects_leading_underscore(self):
        assert not SAFE_RESOURCE_ID_PATTERN.match("_abc")

    def test_rejects_space(self):
        assert not SAFE_RESOURCE_ID_PATTERN.match("abc def")

    def test_rejects_empty(self):
        assert not SAFE_RESOURCE_ID_PATTERN.match("")

    def test_single_char(self):
        assert SAFE_RESOURCE_ID_PATTERN.match("x")


class TestEmailPattern:
    """Test EMAIL_PATTERN regex."""

    def test_basic_email(self):
        assert EMAIL_PATTERN.match("user@example.com")

    def test_plus_addressing(self):
        assert EMAIL_PATTERN.match("user+tag@example.com")

    def test_dots_in_local(self):
        assert EMAIL_PATTERN.match("first.last@example.com")

    def test_subdomain(self):
        assert EMAIL_PATTERN.match("user@mail.example.com")

    def test_percent_in_local(self):
        assert EMAIL_PATTERN.match("user%name@example.com")

    def test_rejects_no_at(self):
        assert not EMAIL_PATTERN.match("userexample.com")

    def test_rejects_no_domain(self):
        assert not EMAIL_PATTERN.match("user@")

    def test_rejects_no_tld(self):
        assert not EMAIL_PATTERN.match("user@example")

    def test_rejects_single_char_tld(self):
        assert not EMAIL_PATTERN.match("user@example.c")

    def test_two_char_tld(self):
        assert EMAIL_PATTERN.match("user@example.co")


# =============================================================================
# validate_platform_id Tests
# =============================================================================


class TestValidatePlatformId:
    """Test validate_platform_id function."""

    def test_valid_platform(self):
        valid, err = validate_platform_id("salesforce")
        assert valid is True
        assert err is None

    def test_valid_with_underscores(self):
        valid, err = validate_platform_id("my_crm")
        assert valid is True
        assert err is None

    def test_valid_with_digits(self):
        valid, err = validate_platform_id("crm2")
        assert valid is True
        assert err is None

    def test_empty_string(self):
        valid, err = validate_platform_id("")
        assert valid is False
        assert err == "Platform is required"

    def test_too_long(self):
        valid, err = validate_platform_id("a" * 51)
        assert valid is False
        assert "too long" in err

    def test_invalid_format_hyphen(self):
        valid, err = validate_platform_id("my-crm")
        assert valid is False
        assert "Invalid platform format" in err

    def test_invalid_format_leading_digit(self):
        valid, err = validate_platform_id("2crm")
        assert valid is False
        assert "Invalid platform format" in err

    def test_invalid_format_special_chars(self):
        valid, err = validate_platform_id("crm@platform")
        assert valid is False
        assert "Invalid platform format" in err

    def test_exactly_50_chars(self):
        valid, err = validate_platform_id("a" * 50)
        assert valid is True
        assert err is None

    def test_single_letter(self):
        valid, err = validate_platform_id("X")
        assert valid is True
        assert err is None


# =============================================================================
# validate_resource_id Tests
# =============================================================================


class TestValidateResourceId:
    """Test validate_resource_id function."""

    def test_valid_id(self):
        valid, err = validate_resource_id("contact123")
        assert valid is True
        assert err is None

    def test_valid_with_hyphens(self):
        valid, err = validate_resource_id("abc-123-def")
        assert valid is True
        assert err is None

    def test_valid_with_underscores(self):
        valid, err = validate_resource_id("abc_123_def")
        assert valid is True
        assert err is None

    def test_empty_string_default_type(self):
        valid, err = validate_resource_id("")
        assert valid is False
        assert err == "ID is required"

    def test_empty_string_custom_type(self):
        valid, err = validate_resource_id("", "Contact ID")
        assert valid is False
        assert err == "Contact ID is required"

    def test_too_long_default_type(self):
        valid, err = validate_resource_id("a" * 129)
        assert valid is False
        assert "too long" in err
        assert "ID" in err

    def test_too_long_custom_type(self):
        valid, err = validate_resource_id("a" * 129, "Deal ID")
        assert valid is False
        assert "Deal ID too long" in err

    def test_invalid_format_default_type(self):
        valid, err = validate_resource_id("abc def")
        assert valid is False
        assert "Invalid id format" in err

    def test_invalid_format_custom_type(self):
        valid, err = validate_resource_id("abc def", "Contact ID")
        assert valid is False
        assert "Invalid contact id format" in err

    def test_exactly_128_chars(self):
        valid, err = validate_resource_id("a" * 128)
        assert valid is True
        assert err is None

    def test_leading_digit(self):
        valid, err = validate_resource_id("123abc")
        assert valid is True
        assert err is None

    def test_leading_hyphen_invalid(self):
        valid, err = validate_resource_id("-abc")
        assert valid is False

    def test_leading_underscore_invalid(self):
        valid, err = validate_resource_id("_abc")
        assert valid is False


# =============================================================================
# validate_email Tests
# =============================================================================


class TestValidateEmail:
    """Test validate_email function."""

    def test_valid_email(self):
        valid, err = validate_email("user@example.com")
        assert valid is True
        assert err is None

    def test_none_not_required(self):
        valid, err = validate_email(None, required=False)
        assert valid is True
        assert err is None

    def test_none_required(self):
        valid, err = validate_email(None, required=True)
        assert valid is False
        assert err == "Email is required"

    def test_empty_not_required(self):
        valid, err = validate_email("", required=False)
        assert valid is True
        assert err is None

    def test_empty_required(self):
        valid, err = validate_email("", required=True)
        assert valid is False
        assert err == "Email is required"

    def test_too_long(self):
        long_email = "a" * 250 + "@b.co"
        valid, err = validate_email(long_email)
        assert valid is False
        assert "too long" in err
        assert str(MAX_EMAIL_LENGTH) in err

    def test_invalid_format_no_at(self):
        valid, err = validate_email("userexample.com")
        assert valid is False
        assert "Invalid email format" in err

    def test_invalid_format_no_domain(self):
        valid, err = validate_email("user@")
        assert valid is False
        assert "Invalid email format" in err

    def test_valid_plus_addressing(self):
        valid, err = validate_email("user+tag@example.com")
        assert valid is True
        assert err is None

    def test_valid_dots(self):
        valid, err = validate_email("first.last@example.com")
        assert valid is True
        assert err is None

    def test_exactly_max_length_valid(self):
        # Build email exactly at MAX_EMAIL_LENGTH with a valid format
        local = "a" * (MAX_EMAIL_LENGTH - len("@example.com"))
        email = f"{local}@example.com"
        assert len(email) == MAX_EMAIL_LENGTH
        valid, err = validate_email(email)
        assert valid is True
        assert err is None

    def test_one_over_max_length(self):
        local = "a" * (MAX_EMAIL_LENGTH - len("@example.com") + 1)
        email = f"{local}@example.com"
        assert len(email) == MAX_EMAIL_LENGTH + 1
        valid, err = validate_email(email)
        assert valid is False


# =============================================================================
# validate_string_field Tests
# =============================================================================


class TestValidateStringField:
    """Test validate_string_field function."""

    def test_valid_value(self):
        valid, err = validate_string_field("John Doe", "Name", MAX_NAME_LENGTH)
        assert valid is True
        assert err is None

    def test_none_not_required(self):
        valid, err = validate_string_field(None, "Name", MAX_NAME_LENGTH, required=False)
        assert valid is True
        assert err is None

    def test_none_required(self):
        valid, err = validate_string_field(None, "Name", MAX_NAME_LENGTH, required=True)
        assert valid is False
        assert err == "Name is required"

    def test_empty_not_required(self):
        valid, err = validate_string_field("", "Name", MAX_NAME_LENGTH, required=False)
        assert valid is True
        assert err is None

    def test_empty_required(self):
        valid, err = validate_string_field("", "Name", MAX_NAME_LENGTH, required=True)
        assert valid is False
        assert err == "Name is required"

    def test_too_long(self):
        valid, err = validate_string_field("a" * 129, "Name", MAX_NAME_LENGTH)
        assert valid is False
        assert "Name too long" in err
        assert str(MAX_NAME_LENGTH) in err

    def test_exactly_max_length(self):
        valid, err = validate_string_field("a" * MAX_NAME_LENGTH, "Name", MAX_NAME_LENGTH)
        assert valid is True
        assert err is None

    def test_one_over_max_length(self):
        valid, err = validate_string_field("a" * (MAX_NAME_LENGTH + 1), "Name", MAX_NAME_LENGTH)
        assert valid is False

    def test_custom_field_name_in_error(self):
        valid, err = validate_string_field(None, "Job Title", MAX_JOB_TITLE_LENGTH, required=True)
        assert "Job Title is required" in err

    def test_custom_max_length(self):
        valid, err = validate_string_field("a" * 11, "Code", 10)
        assert valid is False
        assert "max 10 characters" in err

    def test_phone_length_limit(self):
        valid, err = validate_string_field("+" + "1" * MAX_PHONE_LENGTH, "Phone", MAX_PHONE_LENGTH)
        assert valid is False

    def test_company_name_length_limit(self):
        valid, err = validate_string_field(
            "x" * (MAX_COMPANY_NAME_LENGTH + 1),
            "Company Name",
            MAX_COMPANY_NAME_LENGTH,
        )
        assert valid is False

    def test_deal_name_length_limit(self):
        valid, err = validate_string_field(
            "x" * MAX_DEAL_NAME_LENGTH,
            "Deal Name",
            MAX_DEAL_NAME_LENGTH,
        )
        assert valid is True

    def test_stage_length_limit(self):
        valid, err = validate_string_field(
            "x" * (MAX_STAGE_LENGTH + 1),
            "Stage",
            MAX_STAGE_LENGTH,
        )
        assert valid is False

    def test_pipeline_length_limit(self):
        valid, err = validate_string_field(
            "x" * MAX_PIPELINE_LENGTH,
            "Pipeline",
            MAX_PIPELINE_LENGTH,
        )
        assert valid is True

    def test_credential_value_length_limit(self):
        valid, err = validate_string_field(
            "x" * (MAX_CREDENTIAL_VALUE_LENGTH + 1),
            "Credential",
            MAX_CREDENTIAL_VALUE_LENGTH,
        )
        assert valid is False

    def test_search_query_length_limit(self):
        valid, err = validate_string_field(
            "x" * MAX_SEARCH_QUERY_LENGTH,
            "Query",
            MAX_SEARCH_QUERY_LENGTH,
        )
        assert valid is True

    def test_domain_length_limit(self):
        valid, err = validate_string_field(
            "x" * (MAX_DOMAIN_LENGTH + 1),
            "Domain",
            MAX_DOMAIN_LENGTH,
        )
        assert valid is False


# =============================================================================
# validate_amount Tests
# =============================================================================


class TestValidateAmount:
    """Test validate_amount function."""

    def test_none_returns_valid(self):
        valid, err, val = validate_amount(None)
        assert valid is True
        assert err is None
        assert val is None

    def test_zero(self):
        valid, err, val = validate_amount(0)
        assert valid is True
        assert err is None
        assert val == 0.0

    def test_positive_integer(self):
        valid, err, val = validate_amount(100)
        assert valid is True
        assert err is None
        assert val == 100.0

    def test_positive_float(self):
        valid, err, val = validate_amount(99.99)
        assert valid is True
        assert err is None
        assert val == 99.99

    def test_string_number(self):
        valid, err, val = validate_amount("250.50")
        assert valid is True
        assert err is None
        assert val == 250.50

    def test_negative_amount(self):
        valid, err, val = validate_amount(-1)
        assert valid is False
        assert "cannot be negative" in err
        assert val is None

    def test_negative_float(self):
        valid, err, val = validate_amount(-0.01)
        assert valid is False
        assert "cannot be negative" in err
        assert val is None

    def test_too_large(self):
        valid, err, val = validate_amount(1_000_000_000_001)
        assert valid is False
        assert "too large" in err
        assert val is None

    def test_exactly_one_trillion(self):
        valid, err, val = validate_amount(1_000_000_000_000)
        assert valid is True
        assert err is None
        assert val == 1_000_000_000_000.0

    def test_invalid_string(self):
        valid, err, val = validate_amount("not_a_number")
        assert valid is False
        assert "Invalid amount format" in err
        assert val is None

    def test_invalid_type_list(self):
        valid, err, val = validate_amount([100])
        assert valid is False
        assert "Invalid amount format" in err
        assert val is None

    def test_invalid_type_dict(self):
        valid, err, val = validate_amount({"value": 100})
        assert valid is False
        assert "Invalid amount format" in err
        assert val is None

    def test_small_positive_float(self):
        valid, err, val = validate_amount(0.001)
        assert valid is True
        assert val == pytest.approx(0.001)

    def test_boolean_true_coerces(self):
        # bool is a subclass of int, float(True) == 1.0
        valid, err, val = validate_amount(True)
        assert valid is True
        assert val == 1.0

    def test_boolean_false_coerces(self):
        valid, err, val = validate_amount(False)
        assert valid is True
        assert val == 0.0


# =============================================================================
# validate_probability Tests
# =============================================================================


class TestValidateProbability:
    """Test validate_probability function."""

    def test_none_returns_valid(self):
        valid, err, val = validate_probability(None)
        assert valid is True
        assert err is None
        assert val is None

    def test_zero(self):
        valid, err, val = validate_probability(0)
        assert valid is True
        assert err is None
        assert val == 0.0

    def test_hundred(self):
        valid, err, val = validate_probability(100)
        assert valid is True
        assert err is None
        assert val == 100.0

    def test_fifty_percent(self):
        valid, err, val = validate_probability(50)
        assert valid is True
        assert err is None
        assert val == 50.0

    def test_fractional(self):
        valid, err, val = validate_probability(33.33)
        assert valid is True
        assert val == pytest.approx(33.33)

    def test_string_number(self):
        valid, err, val = validate_probability("75.5")
        assert valid is True
        assert val == 75.5

    def test_negative(self):
        valid, err, val = validate_probability(-1)
        assert valid is False
        assert "between 0 and 100" in err
        assert val is None

    def test_over_hundred(self):
        valid, err, val = validate_probability(101)
        assert valid is False
        assert "between 0 and 100" in err
        assert val is None

    def test_slightly_over_hundred(self):
        valid, err, val = validate_probability(100.01)
        assert valid is False
        assert "between 0 and 100" in err

    def test_slightly_under_zero(self):
        valid, err, val = validate_probability(-0.01)
        assert valid is False
        assert "between 0 and 100" in err

    def test_invalid_string(self):
        valid, err, val = validate_probability("high")
        assert valid is False
        assert "Invalid probability format" in err
        assert val is None

    def test_invalid_type_list(self):
        valid, err, val = validate_probability([50])
        assert valid is False
        assert "Invalid probability format" in err

    def test_invalid_type_dict(self):
        valid, err, val = validate_probability({"value": 50})
        assert valid is False
        assert "Invalid probability format" in err

    def test_boundary_zero_float(self):
        valid, err, val = validate_probability(0.0)
        assert valid is True
        assert val == 0.0

    def test_boundary_hundred_float(self):
        valid, err, val = validate_probability(100.0)
        assert valid is True
        assert val == 100.0


# =============================================================================
# __all__ Exports Test
# =============================================================================


class TestModuleExports:
    """Test that __all__ exposes expected symbols."""

    def test_all_constants_exported(self):
        from aragora.server.handlers.features.crm import validation

        expected_constants = [
            "SAFE_PLATFORM_PATTERN",
            "SAFE_RESOURCE_ID_PATTERN",
            "EMAIL_PATTERN",
            "MAX_EMAIL_LENGTH",
            "MAX_NAME_LENGTH",
            "MAX_PHONE_LENGTH",
            "MAX_COMPANY_NAME_LENGTH",
            "MAX_JOB_TITLE_LENGTH",
            "MAX_DOMAIN_LENGTH",
            "MAX_DEAL_NAME_LENGTH",
            "MAX_STAGE_LENGTH",
            "MAX_PIPELINE_LENGTH",
            "MAX_CREDENTIAL_VALUE_LENGTH",
            "MAX_SEARCH_QUERY_LENGTH",
        ]
        for name in expected_constants:
            assert name in validation.__all__, f"{name} missing from __all__"

    def test_all_functions_exported(self):
        from aragora.server.handlers.features.crm import validation

        expected_functions = [
            "validate_platform_id",
            "validate_resource_id",
            "validate_email",
            "validate_string_field",
            "validate_amount",
            "validate_probability",
        ]
        for name in expected_functions:
            assert name in validation.__all__, f"{name} missing from __all__"

    def test_no_unexpected_exports(self):
        from aragora.server.handlers.features.crm import validation

        expected_count = 14 + 6  # 14 constants + 6 functions
        assert len(validation.__all__) == expected_count
