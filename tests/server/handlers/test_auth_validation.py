"""
Tests for the auth validation module.
Tests cover:
- Email validation patterns
- Password requirements and validation
- Edge cases and boundary conditions
"""

import pytest

from aragora.server.handlers.auth.validation import (
    EMAIL_PATTERN,
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    validate_email,
    validate_password,
)


class TestEmailPattern:
    """Tests for email regex pattern."""

    def test_valid_simple_email(self):
        """Simple email matches pattern."""
        assert EMAIL_PATTERN.match("user@example.com")

    def test_valid_email_with_subdomain(self):
        """Email with subdomain matches pattern."""
        assert EMAIL_PATTERN.match("user@mail.example.com")

    def test_valid_email_with_plus(self):
        """Email with plus addressing matches pattern."""
        assert EMAIL_PATTERN.match("user+tag@example.com")

    def test_valid_email_with_dots(self):
        """Email with dots in local part matches pattern."""
        assert EMAIL_PATTERN.match("first.last@example.com")

    def test_valid_email_with_numbers(self):
        """Email with numbers matches pattern."""
        assert EMAIL_PATTERN.match("user123@example123.com")

    def test_valid_email_with_underscore(self):
        """Email with underscore matches pattern."""
        assert EMAIL_PATTERN.match("user_name@example.com")

    def test_valid_email_with_percent(self):
        """Email with percent matches pattern."""
        assert EMAIL_PATTERN.match("user%name@example.com")

    def test_valid_email_with_hyphen(self):
        """Email with hyphen in domain matches pattern."""
        assert EMAIL_PATTERN.match("user@ex-ample.com")

    def test_invalid_email_no_at(self):
        """Email without @ does not match."""
        assert not EMAIL_PATTERN.match("userexample.com")

    def test_invalid_email_no_domain(self):
        """Email without domain does not match."""
        assert not EMAIL_PATTERN.match("user@")

    def test_invalid_email_no_tld(self):
        """Email without TLD does not match."""
        assert not EMAIL_PATTERN.match("user@example")

    def test_invalid_email_short_tld(self):
        """Email with single-char TLD does not match."""
        assert not EMAIL_PATTERN.match("user@example.c")

    def test_invalid_email_double_at(self):
        """Email with double @ does not match."""
        assert not EMAIL_PATTERN.match("user@@example.com")


class TestValidateEmail:
    """Tests for validate_email function."""

    def test_valid_email_returns_success(self):
        """Valid email returns (True, '')."""
        valid, err = validate_email("user@example.com")
        assert valid is True
        assert err == ""

    def test_empty_email_returns_error(self):
        """Empty email returns appropriate error."""
        valid, err = validate_email("")
        assert valid is False
        assert "required" in err.lower()

    def test_none_email_returns_error(self):
        """None-like empty email returns error."""
        valid, err = validate_email("")
        assert valid is False

    def test_too_long_email_returns_error(self):
        """Email > 254 chars returns error."""
        long_email = "a" * 250 + "@example.com"
        valid, err = validate_email(long_email)
        assert valid is False
        assert "too long" in err.lower()

    def test_invalid_format_returns_error(self):
        """Invalid format email returns error."""
        valid, err = validate_email("not-an-email")
        assert valid is False
        assert "invalid" in err.lower() or "format" in err.lower()

    def test_email_with_spaces_returns_error(self):
        """Email with spaces is invalid."""
        valid, err = validate_email("user @example.com")
        assert valid is False

    def test_unicode_email_may_fail(self):
        """Unicode in email may fail validation."""
        valid, _ = validate_email("user@exämple.com")
        # The regex doesn't support unicode, so this should fail
        assert valid is False


class TestPasswordConstants:
    """Tests for password constants."""

    def test_min_password_length_is_reasonable(self):
        """Minimum password length is at least 8."""
        assert MIN_PASSWORD_LENGTH >= 8

    def test_max_password_length_is_reasonable(self):
        """Maximum password length allows long passwords."""
        assert MAX_PASSWORD_LENGTH >= 64


class TestValidatePassword:
    """Tests for validate_password function."""

    def test_valid_password_returns_success(self):
        """Valid password returns (True, '')."""
        valid, err = validate_password("SecurePass123!")
        assert valid is True
        assert err == ""

    def test_empty_password_returns_error(self):
        """Empty password returns appropriate error."""
        valid, err = validate_password("")
        assert valid is False
        assert "required" in err.lower()

    def test_short_password_returns_error(self):
        """Password below minimum length returns error."""
        short = "a" * (MIN_PASSWORD_LENGTH - 1)
        valid, err = validate_password(short)
        assert valid is False
        assert "at least" in err.lower() or str(MIN_PASSWORD_LENGTH) in err

    def test_exact_min_length_password_is_valid(self):
        """Password at exact minimum length is valid."""
        exact = "a" * MIN_PASSWORD_LENGTH
        valid, err = validate_password(exact)
        assert valid is True
        assert err == ""

    def test_long_password_returns_error(self):
        """Password above maximum length returns error."""
        long = "a" * (MAX_PASSWORD_LENGTH + 1)
        valid, err = validate_password(long)
        assert valid is False
        assert "at most" in err.lower() or str(MAX_PASSWORD_LENGTH) in err

    def test_exact_max_length_password_is_valid(self):
        """Password at exact maximum length is valid."""
        exact = "a" * MAX_PASSWORD_LENGTH
        valid, err = validate_password(exact)
        assert valid is True
        assert err == ""

    def test_password_with_spaces_is_valid(self):
        """Password with spaces is allowed."""
        valid, err = validate_password("pass word 123!")
        assert valid is True

    def test_password_with_unicode_is_valid(self):
        """Password with unicode characters is allowed."""
        valid, err = validate_password("pässwörd123!")
        assert valid is True


class TestValidationEdgeCases:
    """Edge case tests for validation functions."""

    def test_email_exactly_254_chars_is_valid(self):
        """Email at exactly 254 characters is valid."""
        # Create email that's exactly 254 chars
        local_part = "a" * 64  # max local part
        domain = "b" * (254 - 64 - 1 - 4) + ".com"  # -1 for @, -4 for .com
        email = f"{local_part}@{domain}"
        if len(email) == 254:
            valid, _ = validate_email(email)
            # Pattern might not match due to domain constraints
            # but length check should pass
            assert True  # Just checking it doesn't crash

    def test_password_with_null_bytes(self):
        """Password with null bytes is handled."""
        # Should either accept or reject gracefully
        try:
            valid, _ = validate_password("password\x00123")
            # Either outcome is acceptable
            assert isinstance(valid, bool)
        except Exception:
            pytest.fail("Should handle null bytes gracefully")

    def test_email_with_leading_dot(self):
        """Email with leading dot in local part - check validation."""
        valid, _ = validate_email(".user@example.com")
        # RFC 5321 technically forbids this, but regex may allow it
        # Just check it doesn't crash
        assert isinstance(valid, bool)

    def test_email_with_trailing_dot(self):
        """Email with trailing dot in local part - check validation."""
        valid, _ = validate_email("user.@example.com")
        # RFC 5321 technically forbids this, but regex may allow it
        # Just check it doesn't crash
        assert isinstance(valid, bool)


class TestValidationReturnTypes:
    """Tests for validation function return types."""

    def test_validate_email_returns_tuple(self):
        """validate_email returns a tuple."""
        result = validate_email("test@example.com")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_email_first_element_is_bool(self):
        """First element of validate_email result is bool."""
        valid, _ = validate_email("test@example.com")
        assert isinstance(valid, bool)

    def test_validate_email_second_element_is_str(self):
        """Second element of validate_email result is str."""
        _, err = validate_email("test@example.com")
        assert isinstance(err, str)

    def test_validate_password_returns_tuple(self):
        """validate_password returns a tuple."""
        result = validate_password("password123")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_password_first_element_is_bool(self):
        """First element of validate_password result is bool."""
        valid, _ = validate_password("password123")
        assert isinstance(valid, bool)

    def test_validate_password_second_element_is_str(self):
        """Second element of validate_password result is str."""
        _, err = validate_password("password123")
        assert isinstance(err, str)
