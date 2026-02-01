"""Tests for aragora.gauntlet.errors module.

Covers GauntletErrorCode enum, GauntletError dataclass, ERRORS dict,
and gauntlet_error_response helper.
"""

from __future__ import annotations

import re

import pytest

from aragora.gauntlet.errors import (
    ERRORS,
    GauntletError,
    GauntletErrorCode,
    gauntlet_error_response,
)


# ---------------------------------------------------------------------------
# GauntletErrorCode enum tests
# ---------------------------------------------------------------------------


class TestGauntletErrorCodeEnum:
    """Tests for the GauntletErrorCode enum."""

    def test_total_member_count(self):
        """Enum should have exactly 31 members (8+5+6+7+5)."""
        assert len(GauntletErrorCode) == 31

    def test_is_str_enum(self):
        """Each member should be an instance of str."""
        for member in GauntletErrorCode:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_all_codes_follow_gauntlet_xxx_format(self):
        """Every value must match GAUNTLET_<three digits>."""
        pattern = re.compile(r"^GAUNTLET_\d{3}$")
        for member in GauntletErrorCode:
            assert pattern.match(member.value), (
                f"{member.name} has unexpected value format: {member.value}"
            )

    # -- 1XX Input Validation (8 members) --

    INPUT_VALIDATION_CODES = [
        ("INVALID_INPUT", "GAUNTLET_100"),
        ("INPUT_TOO_LARGE", "GAUNTLET_101"),
        ("INVALID_INPUT_TYPE", "GAUNTLET_102"),
        ("INVALID_PERSONA", "GAUNTLET_103"),
        ("INVALID_AGENTS", "GAUNTLET_104"),
        ("INVALID_PROFILE", "GAUNTLET_105"),
        ("MISSING_REQUIRED_FIELD", "GAUNTLET_106"),
        ("INVALID_FORMAT", "GAUNTLET_107"),
    ]

    @pytest.mark.parametrize("name, value", INPUT_VALIDATION_CODES)
    def test_input_validation_codes(self, name: str, value: str):
        member = GauntletErrorCode[name]
        assert member.value == value

    def test_input_validation_count(self):
        """There should be exactly 8 input validation codes (1XX)."""
        codes_1xx = [m for m in GauntletErrorCode if m.value.startswith("GAUNTLET_1")]
        assert len(codes_1xx) == 8

    # -- 2XX Auth (5 members) --

    AUTH_CODES = [
        ("NOT_AUTHENTICATED", "GAUNTLET_200"),
        ("INSUFFICIENT_PERMISSIONS", "GAUNTLET_201"),
        ("TOKEN_EXPIRED", "GAUNTLET_202"),
        ("INVALID_API_KEY", "GAUNTLET_203"),
        ("RBAC_DENIED", "GAUNTLET_204"),
    ]

    @pytest.mark.parametrize("name, value", AUTH_CODES)
    def test_auth_codes(self, name: str, value: str):
        member = GauntletErrorCode[name]
        assert member.value == value

    def test_auth_count(self):
        """There should be exactly 5 auth codes (2XX)."""
        codes_2xx = [m for m in GauntletErrorCode if m.value.startswith("GAUNTLET_2")]
        assert len(codes_2xx) == 5

    # -- 3XX Resource (6 members) --

    RESOURCE_CODES = [
        ("GAUNTLET_NOT_FOUND", "GAUNTLET_300"),
        ("RECEIPT_NOT_FOUND", "GAUNTLET_301"),
        ("PERSONA_NOT_FOUND", "GAUNTLET_302"),
        ("QUOTA_EXCEEDED", "GAUNTLET_303"),
        ("RATE_LIMITED", "GAUNTLET_304"),
        ("RESOURCE_LOCKED", "GAUNTLET_305"),
    ]

    @pytest.mark.parametrize("name, value", RESOURCE_CODES)
    def test_resource_codes(self, name: str, value: str):
        member = GauntletErrorCode[name]
        assert member.value == value

    def test_resource_count(self):
        """There should be exactly 6 resource codes (3XX)."""
        codes_3xx = [m for m in GauntletErrorCode if m.value.startswith("GAUNTLET_3")]
        assert len(codes_3xx) == 6

    # -- 4XX Execution (7 members) --

    EXECUTION_CODES = [
        ("GAUNTLET_FAILED", "GAUNTLET_400"),
        ("AGENT_UNAVAILABLE", "GAUNTLET_401"),
        ("EXECUTION_TIMEOUT", "GAUNTLET_402"),
        ("CONSENSUS_FAILED", "GAUNTLET_403"),
        ("VERIFICATION_FAILED", "GAUNTLET_404"),
        ("INCOMPLETE_RESULT", "GAUNTLET_405"),
        ("NOT_COMPLETED", "GAUNTLET_406"),
    ]

    @pytest.mark.parametrize("name, value", EXECUTION_CODES)
    def test_execution_codes(self, name: str, value: str):
        member = GauntletErrorCode[name]
        assert member.value == value

    def test_execution_count(self):
        """There should be exactly 7 execution codes (4XX)."""
        codes_4xx = [m for m in GauntletErrorCode if m.value.startswith("GAUNTLET_4")]
        assert len(codes_4xx) == 7

    # -- 5XX System (5 members) --

    SYSTEM_CODES = [
        ("INTERNAL_ERROR", "GAUNTLET_500"),
        ("STORAGE_ERROR", "GAUNTLET_501"),
        ("SERVICE_UNAVAILABLE", "GAUNTLET_502"),
        ("CONFIGURATION_ERROR", "GAUNTLET_503"),
        ("SIGNING_ERROR", "GAUNTLET_504"),
    ]

    @pytest.mark.parametrize("name, value", SYSTEM_CODES)
    def test_system_codes(self, name: str, value: str):
        member = GauntletErrorCode[name]
        assert member.value == value

    def test_system_count(self):
        """There should be exactly 5 system codes (5XX)."""
        codes_5xx = [m for m in GauntletErrorCode if m.value.startswith("GAUNTLET_5")]
        assert len(codes_5xx) == 5

    def test_string_equality(self):
        """Because GauntletErrorCode is a str enum, members equal their value."""
        assert GauntletErrorCode.INVALID_INPUT == "GAUNTLET_100"
        assert GauntletErrorCode.INTERNAL_ERROR == "GAUNTLET_500"


# ---------------------------------------------------------------------------
# GauntletError dataclass tests
# ---------------------------------------------------------------------------


class TestGauntletError:
    """Tests for the GauntletError dataclass."""

    def test_creation_with_defaults(self):
        """http_status defaults to 400, details defaults to None."""
        err = GauntletError(
            code=GauntletErrorCode.INVALID_INPUT,
            message="bad input",
        )
        assert err.code == GauntletErrorCode.INVALID_INPUT
        assert err.message == "bad input"
        assert err.details is None
        assert err.http_status == 400

    def test_creation_with_all_fields(self):
        err = GauntletError(
            code=GauntletErrorCode.STORAGE_ERROR,
            message="disk full",
            details={"disk": "/dev/sda1"},
            http_status=500,
        )
        assert err.code == GauntletErrorCode.STORAGE_ERROR
        assert err.message == "disk full"
        assert err.details == {"disk": "/dev/sda1"}
        assert err.http_status == 500

    def test_to_dict_without_details(self):
        """When details is None, the output dict should not contain a 'details' key."""
        err = GauntletError(
            code=GauntletErrorCode.GAUNTLET_NOT_FOUND,
            message="not found",
        )
        result = err.to_dict()
        assert result == {
            "error": True,
            "code": "GAUNTLET_300",
            "message": "not found",
        }
        assert "details" not in result

    def test_to_dict_with_details(self):
        """When details are present, they should appear in the output dict."""
        err = GauntletError(
            code=GauntletErrorCode.INPUT_TOO_LARGE,
            message="too large",
            details={"max_size_kb": 1024},
        )
        result = err.to_dict()
        assert result == {
            "error": True,
            "code": "GAUNTLET_101",
            "message": "too large",
            "details": {"max_size_kb": 1024},
        }

    def test_to_dict_with_empty_dict_details(self):
        """An empty dict is falsy, so 'details' key should be absent."""
        err = GauntletError(
            code=GauntletErrorCode.INVALID_INPUT,
            message="test",
            details={},
        )
        result = err.to_dict()
        assert "details" not in result

    def test_to_dict_always_has_error_true(self):
        """Every to_dict output must have error=True."""
        for code in GauntletErrorCode:
            err = GauntletError(code=code, message="test")
            assert err.to_dict()["error"] is True


# ---------------------------------------------------------------------------
# ERRORS dict tests
# ---------------------------------------------------------------------------


class TestErrorsDict:
    """Tests for the pre-defined ERRORS dictionary."""

    EXPECTED_KEYS = [
        "invalid_input",
        "input_too_large",
        "invalid_input_type",
        "invalid_persona",
        "not_authenticated",
        "insufficient_permissions",
        "gauntlet_not_found",
        "receipt_not_found",
        "quota_exceeded",
        "rate_limited",
        "not_completed",
        "execution_timeout",
        "internal_error",
        "storage_error",
    ]

    def test_errors_dict_has_14_entries(self):
        assert len(ERRORS) == 14

    def test_errors_dict_keys(self):
        """All expected keys are present."""
        assert set(ERRORS.keys()) == set(self.EXPECTED_KEYS)

    def test_all_entries_are_gauntlet_error_instances(self):
        for key, err in ERRORS.items():
            assert isinstance(err, GauntletError), f"ERRORS[{key!r}] is not a GauntletError"

    def test_all_entries_have_valid_error_codes(self):
        for key, err in ERRORS.items():
            assert isinstance(err.code, GauntletErrorCode), (
                f"ERRORS[{key!r}].code is not a GauntletErrorCode"
            )

    def test_all_entries_have_positive_http_status(self):
        for key, err in ERRORS.items():
            assert 400 <= err.http_status <= 599, (
                f"ERRORS[{key!r}].http_status={err.http_status} is outside 4xx-5xx range"
            )

    def test_all_entries_have_nonempty_message(self):
        for key, err in ERRORS.items():
            assert err.message, f"ERRORS[{key!r}].message is empty"

    # Specific http_status assertions
    @pytest.mark.parametrize(
        "key, expected_status",
        [
            ("not_authenticated", 401),
            ("gauntlet_not_found", 404),
            ("receipt_not_found", 404),
            ("quota_exceeded", 429),
            ("rate_limited", 429),
            ("internal_error", 500),
            ("storage_error", 500),
            ("input_too_large", 413),
            ("execution_timeout", 408),
            ("insufficient_permissions", 403),
            ("invalid_input", 400),
            ("not_completed", 400),
        ],
    )
    def test_specific_http_status(self, key: str, expected_status: int):
        assert ERRORS[key].http_status == expected_status

    def test_input_too_large_has_base_details(self):
        """input_too_large should have max_size_kb in its details."""
        err = ERRORS["input_too_large"]
        assert err.details is not None
        assert "max_size_kb" in err.details
        assert err.details["max_size_kb"] == 1024

    def test_invalid_input_type_has_valid_types_detail(self):
        """invalid_input_type should list valid types."""
        err = ERRORS["invalid_input_type"]
        assert err.details is not None
        assert "valid_types" in err.details
        assert isinstance(err.details["valid_types"], list)
        assert len(err.details["valid_types"]) > 0


# ---------------------------------------------------------------------------
# gauntlet_error_response tests
# ---------------------------------------------------------------------------


class TestGauntletErrorResponse:
    """Tests for the gauntlet_error_response helper function."""

    def test_returns_tuple_of_dict_and_int(self):
        body, status = gauntlet_error_response("invalid_input")
        assert isinstance(body, dict)
        assert isinstance(status, int)

    def test_valid_key_returns_correct_code_and_status(self):
        body, status = gauntlet_error_response("gauntlet_not_found")
        assert body["error"] is True
        assert body["code"] == "GAUNTLET_300"
        assert body["message"] == "Gauntlet run not found"
        assert status == 404

    def test_valid_key_no_details_by_default(self):
        """For entries without base details, output should have no 'details' key."""
        body, _ = gauntlet_error_response("invalid_input")
        assert "details" not in body

    def test_unknown_key_falls_back_to_internal_error(self):
        body, status = gauntlet_error_response("nonexistent_key")
        assert body["code"] == GauntletErrorCode.INTERNAL_ERROR.value
        assert status == 500

    def test_unknown_key_returns_internal_error_message(self):
        body, _ = gauntlet_error_response("this_does_not_exist")
        assert body["message"] == ERRORS["internal_error"].message

    def test_message_override_replaces_default_message(self):
        body, status = gauntlet_error_response(
            "invalid_input",
            message_override="Custom validation message",
        )
        assert body["message"] == "Custom validation message"
        # Code and status should remain unchanged
        assert body["code"] == "GAUNTLET_100"
        assert status == 400

    def test_message_override_on_fallback(self):
        """message_override should work even when key is unknown (fallback)."""
        body, status = gauntlet_error_response(
            "unknown_key",
            message_override="Overridden fallback",
        )
        assert body["message"] == "Overridden fallback"
        assert body["code"] == "GAUNTLET_500"
        assert status == 500

    def test_details_passed_through(self):
        body, _ = gauntlet_error_response(
            "gauntlet_not_found",
            details={"id": "gauntlet-123"},
        )
        assert body["details"] == {"id": "gauntlet-123"}

    def test_details_merged_with_base_details(self):
        """input_too_large has base details (max_size_kb); extra details should merge."""
        body, status = gauntlet_error_response(
            "input_too_large",
            details={"actual_size_kb": 2048},
        )
        assert status == 413
        assert body["details"]["max_size_kb"] == 1024
        assert body["details"]["actual_size_kb"] == 2048

    def test_details_override_base_details_on_conflict(self):
        """When provided details conflict with base, provided details should win."""
        body, _ = gauntlet_error_response(
            "input_too_large",
            details={"max_size_kb": 512},
        )
        # Provided details are merged after base, so they overwrite
        assert body["details"]["max_size_kb"] == 512

    def test_no_details_when_neither_base_nor_extra(self):
        """When base has no details and none provided, key should be absent."""
        body, _ = gauntlet_error_response("not_authenticated")
        assert "details" not in body

    def test_base_details_present_without_extra(self):
        """When base has details but none provided, base details should appear."""
        body, _ = gauntlet_error_response("input_too_large")
        assert "details" in body
        assert body["details"]["max_size_kb"] == 1024

    def test_response_always_has_error_true(self):
        """Every response should have error=True."""
        for key in ERRORS:
            body, _ = gauntlet_error_response(key)
            assert body["error"] is True

    def test_response_codes_match_errors_dict(self):
        """Each key should produce the code from the corresponding ERRORS entry."""
        for key, err in ERRORS.items():
            body, status = gauntlet_error_response(key)
            assert body["code"] == err.code.value
            assert status == err.http_status

    def test_invalid_input_type_merges_details(self):
        """invalid_input_type has base details (valid_types); extra details merge."""
        body, status = gauntlet_error_response(
            "invalid_input_type",
            details={"provided_type": "unknown"},
        )
        assert status == 400
        assert "valid_types" in body["details"]
        assert body["details"]["provided_type"] == "unknown"
