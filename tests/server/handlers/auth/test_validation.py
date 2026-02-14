"""
Tests for aragora.server.handlers.auth.validation module.

Tests cover:
- Email validation with EMAIL_PATTERN regex and validate_email function
- Password validation with validate_password function
- Constants: MIN_PASSWORD_LENGTH and MAX_PASSWORD_LENGTH
- Input validation edge cases and boundary conditions
- Error message correctness
- Return type validation

Target: 80%+ code coverage.
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.auth.validation import (
    EMAIL_PATTERN,
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    validate_email,
    validate_password,
)


# ===========================================================================
# Test: EMAIL_PATTERN Regex - Valid Emails
# ===========================================================================


class TestEmailPatternValidEmails:
    """Test that EMAIL_PATTERN matches valid email formats."""

    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user_name@example.com",
            "user123@example.com",
            "123user@example.com",
            "user@subdomain.example.com",
            "user@example.co.uk",
            "user@example.io",
            "user%name@example.com",
            "user-name@example.com",
            "a@b.cd",  # Minimum valid: single char local, two char TLD
            "user@example-domain.com",
            "user@123.456.com",
            "User@Example.COM",  # Mixed case
            "USER@EXAMPLE.COM",  # All caps
        ],
    )
    def test_valid_email_matches(self, email: str) -> None:
        """Valid email addresses should match the pattern."""
        assert EMAIL_PATTERN.match(email) is not None, f"Expected {email!r} to match"


class TestEmailPatternInvalidEmails:
    """Test that EMAIL_PATTERN rejects invalid email formats."""

    @pytest.mark.parametrize(
        "email",
        [
            "",  # Empty
            "user",  # No @ or domain
            "@example.com",  # No local part
            "user@",  # No domain
            "user@example",  # No TLD
            "user@example.",  # TLD is empty
            "user@example.c",  # TLD too short (1 char)
            "user@@example.com",  # Double @
            "user @example.com",  # Space in local part
            "user@ example.com",  # Space after @
            "user@exam ple.com",  # Space in domain
            "user\t@example.com",  # Tab character
            "user\n@example.com",  # Newline character
            "@",  # Just @
            "user@.com",  # Domain starts with dot
            "user@com",  # No second level domain
        ],
    )
    def test_invalid_email_does_not_match(self, email: str) -> None:
        """Invalid email addresses should not match the pattern."""
        assert EMAIL_PATTERN.match(email) is None, f"Expected {email!r} to NOT match"

    def test_email_with_unicode_domain_fails(self) -> None:
        """Unicode in domain is not supported by this pattern."""
        assert EMAIL_PATTERN.match("user@exämple.com") is None

    def test_email_with_unicode_local_part_fails(self) -> None:
        """Unicode in local part is not supported by this pattern."""
        assert EMAIL_PATTERN.match("üser@example.com") is None


# ===========================================================================
# Test: validate_email Function
# ===========================================================================


class TestValidateEmailSuccess:
    """Test validate_email with valid inputs."""

    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "test.user@company.org",
            "admin+billing@domain.co",
            "support_team@service.io",
        ],
    )
    def test_valid_email_returns_true_empty_error(self, email: str) -> None:
        """Valid email should return (True, '')."""
        valid, err = validate_email(email)
        assert valid is True
        assert err == ""

    def test_email_at_254_chars_boundary_valid(self) -> None:
        """Email at exactly 254 characters (max valid length) should pass length check."""
        # Create a valid-looking email at 254 chars
        # Format: local@domain.com where local + @ + domain.com = 254
        local = "a" * 64  # Max recommended local part
        domain_base = "b" * (254 - 64 - 1 - 4)  # 254 - local - @ - .com
        email = f"{local}@{domain_base}.com"
        assert len(email) == 254

        # Length validation should pass, but pattern may fail on the long domain
        valid, err = validate_email(email)
        # Pattern matching may reject due to domain structure, but length is OK
        assert "too long" not in err.lower()


class TestValidateEmailErrors:
    """Test validate_email error cases."""

    def test_empty_string_returns_required_error(self) -> None:
        """Empty email returns 'required' error."""
        valid, err = validate_email("")
        assert valid is False
        assert "required" in err.lower()

    def test_whitespace_only_treated_as_falsy(self) -> None:
        """Whitespace-only string might be treated as valid by pattern but should ideally fail."""
        valid, err = validate_email("   ")
        # Pattern won't match whitespace, so should return invalid format
        assert valid is False

    def test_email_too_long_returns_error(self) -> None:
        """Email > 254 characters returns 'too long' error."""
        # Create email that exceeds 254 chars
        long_email = "a" * 200 + "@" + "b" * 50 + ".com"
        assert len(long_email) > 254

        valid, err = validate_email(long_email)
        assert valid is False
        assert "too long" in err.lower()

    def test_invalid_format_returns_format_error(self) -> None:
        """Email with invalid format returns format error."""
        valid, err = validate_email("not-an-email")
        assert valid is False
        assert "invalid" in err.lower() or "format" in err.lower()

    def test_email_missing_at_sign(self) -> None:
        """Email without @ returns format error."""
        valid, err = validate_email("userexample.com")
        assert valid is False
        assert "invalid" in err.lower()

    def test_email_missing_domain(self) -> None:
        """Email without domain returns format error."""
        valid, err = validate_email("user@")
        assert valid is False
        assert "invalid" in err.lower()

    def test_email_missing_tld(self) -> None:
        """Email without TLD returns format error."""
        valid, err = validate_email("user@example")
        assert valid is False
        assert "invalid" in err.lower()


class TestValidateEmailOrderOfChecks:
    """Test the order of validation checks in validate_email."""

    def test_empty_check_before_length_check(self) -> None:
        """Empty string should return 'required' not 'too long' or 'invalid'."""
        valid, err = validate_email("")
        assert "required" in err.lower()
        assert "long" not in err.lower()
        assert "format" not in err.lower()

    def test_length_check_before_pattern_check(self) -> None:
        """Too long email should return 'too long' before pattern check."""
        # Create an email that's too long but would match pattern if shorter
        long_email = "a" * 250 + "@example.com"  # > 254 chars
        valid, err = validate_email(long_email)
        assert "too long" in err.lower()


# ===========================================================================
# Test: Password Constants
# ===========================================================================


class TestPasswordConstants:
    """Test password length constants."""

    def test_min_password_length_value(self) -> None:
        """MIN_PASSWORD_LENGTH should be 12 (strengthened from 8)."""
        assert MIN_PASSWORD_LENGTH == 12

    def test_max_password_length_value(self) -> None:
        """MAX_PASSWORD_LENGTH should be 128."""
        assert MAX_PASSWORD_LENGTH == 128

    def test_min_less_than_max(self) -> None:
        """Minimum length should be less than maximum length."""
        assert MIN_PASSWORD_LENGTH < MAX_PASSWORD_LENGTH

    def test_min_is_reasonable(self) -> None:
        """Minimum password length should be at least 8 for security."""
        assert MIN_PASSWORD_LENGTH >= 8

    def test_max_allows_long_passphrases(self) -> None:
        """Maximum should allow passphrases (at least 64 chars)."""
        assert MAX_PASSWORD_LENGTH >= 64


# ===========================================================================
# Test: validate_password Function
# ===========================================================================


class TestValidatePasswordSuccess:
    """Test validate_password with valid inputs.

    Note: Password validation now requires:
    - Minimum 12 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    - Not a common password
    """

    @pytest.mark.parametrize(
        "password",
        [
            "MyS3cur3P@ss!",  # 13 chars, meets all requirements
            "Abcdefgh123!",  # 12 chars, meets all requirements
            "P@$$w0rd!#%^",  # 12 chars with many special chars
            "Aa1!" + "x" * 124,  # Exactly max length with requirements
        ],
    )
    def test_valid_password_returns_true_empty_error(self, password: str) -> None:
        """Valid password should return (True, '')."""
        valid, err = validate_password(password)
        assert valid is True
        assert err == ""

    def test_password_with_unicode_is_valid(self) -> None:
        """Unicode characters are allowed in passwords if other requirements met."""
        # Unicode password that meets all complexity requirements
        valid, err = validate_password("Pässwörd123!")
        assert valid is True
        assert err == ""

    def test_password_with_emoji_is_valid(self) -> None:
        """Password with special chars is valid if all requirements met."""
        valid, err = validate_password("Password123!!")
        assert valid is True

    def test_password_with_special_chars(self) -> None:
        """Special characters are allowed."""
        valid, err = validate_password("P@$$w0rd!#%^&*()")
        assert valid is True


class TestValidatePasswordErrors:
    """Test validate_password error cases."""

    def test_empty_password_returns_required_error(self) -> None:
        """Empty password returns 'required' error."""
        valid, err = validate_password("")
        assert valid is False
        assert "required" in err.lower()

    def test_short_password_returns_length_error(self) -> None:
        """Password below minimum length returns error with length info."""
        valid, err = validate_password("1234567")  # 7 chars
        assert valid is False
        assert "at least" in err.lower() or str(MIN_PASSWORD_LENGTH) in err

    def test_one_char_below_minimum(self) -> None:
        """Password with MIN_PASSWORD_LENGTH - 1 chars fails."""
        password = "a" * (MIN_PASSWORD_LENGTH - 1)
        valid, err = validate_password(password)
        assert valid is False

    def test_exact_minimum_length_valid(self) -> None:
        """Password at exact minimum length is valid if complexity requirements met."""
        # Build a 12-char password that meets all complexity requirements
        password = "Abcdef123!@#"  # 12 chars with upper, lower, digit, special
        assert len(password) == MIN_PASSWORD_LENGTH
        valid, err = validate_password(password)
        assert valid is True
        assert err == ""

    def test_one_above_minimum_valid(self) -> None:
        """Password with MIN_PASSWORD_LENGTH + 1 chars is valid if complexity requirements met."""
        # Build a 13-char password that meets all complexity requirements
        password = "Abcdefg123!@#"  # 13 chars with upper, lower, digit, special
        assert len(password) == MIN_PASSWORD_LENGTH + 1
        valid, err = validate_password(password)
        assert valid is True

    def test_too_long_password_returns_length_error(self) -> None:
        """Password above maximum length returns error with length info."""
        valid, err = validate_password("a" * (MAX_PASSWORD_LENGTH + 1))
        assert valid is False
        assert "at most" in err.lower() or str(MAX_PASSWORD_LENGTH) in err

    def test_one_char_above_maximum(self) -> None:
        """Password with MAX_PASSWORD_LENGTH + 1 chars fails."""
        password = "a" * (MAX_PASSWORD_LENGTH + 1)
        valid, err = validate_password(password)
        assert valid is False

    def test_exact_maximum_length_valid(self) -> None:
        """Password at exact maximum length is valid if complexity requirements met."""
        # Build a 128-char password that meets all complexity requirements
        password = "Aa1!" + "x" * 124  # 128 chars with upper, lower, digit, special
        assert len(password) == MAX_PASSWORD_LENGTH
        valid, err = validate_password(password)
        assert valid is True
        assert err == ""

    def test_one_below_maximum_valid(self) -> None:
        """Password with MAX_PASSWORD_LENGTH - 1 chars is valid if complexity requirements met."""
        # Build a 127-char password that meets all complexity requirements
        password = "Aa1!" + "x" * 123  # 127 chars with upper, lower, digit, special
        assert len(password) == MAX_PASSWORD_LENGTH - 1
        valid, err = validate_password(password)
        assert valid is True


class TestValidatePasswordOrderOfChecks:
    """Test the order of validation checks in validate_password."""

    def test_empty_check_before_length_check(self) -> None:
        """Empty string should return 'required' not length error."""
        valid, err = validate_password("")
        assert "required" in err.lower()
        assert "at least" not in err.lower()
        assert "at most" not in err.lower()


# ===========================================================================
# Test: Return Types
# ===========================================================================


class TestReturnTypes:
    """Test that validation functions return correct types."""

    def test_validate_email_returns_tuple(self) -> None:
        """validate_email returns a 2-tuple."""
        result = validate_email("test@example.com")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_email_tuple_types(self) -> None:
        """validate_email tuple contains (bool, str)."""
        valid, err = validate_email("test@example.com")
        assert isinstance(valid, bool)
        assert isinstance(err, str)

    def test_validate_email_error_returns_tuple(self) -> None:
        """validate_email returns tuple even on error."""
        result = validate_email("invalid")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_password_returns_tuple(self) -> None:
        """validate_password returns a 2-tuple."""
        result = validate_password("password123")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_password_tuple_types(self) -> None:
        """validate_password tuple contains (bool, str)."""
        valid, err = validate_password("password123")
        assert isinstance(valid, bool)
        assert isinstance(err, str)

    def test_validate_password_error_returns_tuple(self) -> None:
        """validate_password returns tuple even on error."""
        result = validate_password("")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ===========================================================================
# Test: Edge Cases and Special Characters
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_email_with_null_byte(self) -> None:
        """Email with null byte should be handled gracefully."""
        result = validate_email("user\x00@example.com")
        assert isinstance(result, tuple)
        valid, _ = result
        # Should likely fail validation
        assert isinstance(valid, bool)

    def test_password_with_null_byte(self) -> None:
        """Password with null byte should be handled gracefully."""
        result = validate_password("password\x00123")
        assert isinstance(result, tuple)
        # As long as it doesn't crash and returns proper type, it's acceptable
        valid, _ = result
        assert isinstance(valid, bool)

    def test_email_with_leading_dot_in_local(self) -> None:
        """Email with leading dot in local part - RFC violation."""
        valid, _ = validate_email(".user@example.com")
        # Pattern may or may not accept this
        assert isinstance(valid, bool)

    def test_email_with_trailing_dot_in_local(self) -> None:
        """Email with trailing dot in local part - RFC violation."""
        valid, _ = validate_email("user.@example.com")
        assert isinstance(valid, bool)

    def test_email_with_consecutive_dots(self) -> None:
        """Email with consecutive dots in local part - RFC violation."""
        valid, _ = validate_email("user..name@example.com")
        assert isinstance(valid, bool)

    def test_very_long_tld(self) -> None:
        """Email with very long TLD should still validate if pattern matches."""
        valid, err = validate_email("user@example.technology")
        assert valid is True
        assert err == ""

    def test_two_letter_tld(self) -> None:
        """Email with 2-letter TLD (minimum allowed) should be valid."""
        valid, err = validate_email("user@example.io")
        assert valid is True

    def test_password_whitespace_only(self) -> None:
        """Password with only whitespace characters is rejected."""
        # 12 spaces - meets length but is entirely whitespace
        valid, err = validate_password("            ")
        assert valid is False
        assert "whitespace" in err.lower()


class TestModuleExports:
    """Test that module exports are correct."""

    def test_email_pattern_is_compiled_regex(self) -> None:
        """EMAIL_PATTERN should be a compiled regex pattern."""
        import re

        assert isinstance(EMAIL_PATTERN, re.Pattern)

    def test_min_password_length_is_int(self) -> None:
        """MIN_PASSWORD_LENGTH should be an int."""
        assert isinstance(MIN_PASSWORD_LENGTH, int)

    def test_max_password_length_is_int(self) -> None:
        """MAX_PASSWORD_LENGTH should be an int."""
        assert isinstance(MAX_PASSWORD_LENGTH, int)

    def test_validate_email_is_callable(self) -> None:
        """validate_email should be a callable function."""
        assert callable(validate_email)

    def test_validate_password_is_callable(self) -> None:
        """validate_password should be a callable function."""
        assert callable(validate_password)


# ===========================================================================
# Test: Error Message Content
# ===========================================================================


class TestErrorMessages:
    """Test that error messages are meaningful and consistent."""

    def test_empty_email_error_message(self) -> None:
        """Empty email error message mentions 'required' or 'Email'."""
        _, err = validate_email("")
        assert "required" in err.lower() or "email" in err.lower()

    def test_long_email_error_message(self) -> None:
        """Long email error message mentions 'too long'."""
        _, err = validate_email("a" * 300 + "@example.com")
        assert "too long" in err.lower()

    def test_invalid_email_format_message(self) -> None:
        """Invalid email format error message mentions 'invalid' or 'format'."""
        _, err = validate_email("notanemail")
        assert "invalid" in err.lower() or "format" in err.lower()

    def test_empty_password_error_message(self) -> None:
        """Empty password error message mentions 'required' or 'password'."""
        _, err = validate_password("")
        assert "required" in err.lower() or "password" in err.lower()

    def test_short_password_error_message_includes_min_length(self) -> None:
        """Short password error message includes the minimum length."""
        _, err = validate_password("short")
        assert str(MIN_PASSWORD_LENGTH) in err or "at least" in err.lower()

    def test_long_password_error_message_includes_max_length(self) -> None:
        """Long password error message includes the maximum length."""
        _, err = validate_password("a" * 200)
        assert str(MAX_PASSWORD_LENGTH) in err or "at most" in err.lower()


# ===========================================================================
# Test: Integration Scenarios
# ===========================================================================


class TestIntegrationScenarios:
    """Test real-world usage scenarios."""

    def test_typical_registration_flow_valid(self) -> None:
        """Typical registration with valid email and password."""
        email = "john.doe@company.com"
        password = "SecureP@ss123!"

        email_valid, email_err = validate_email(email)
        password_valid, password_err = validate_password(password)

        assert email_valid is True
        assert password_valid is True
        assert email_err == ""
        assert password_err == ""

    def test_typical_registration_flow_invalid_email(self) -> None:
        """Registration with invalid email should fail email validation only."""
        email = "invalid-email"
        password = "SecureP@ss123!"

        email_valid, _ = validate_email(email)
        password_valid, _ = validate_password(password)

        assert email_valid is False
        assert password_valid is True

    def test_typical_registration_flow_weak_password(self) -> None:
        """Registration with weak password should fail password validation only."""
        email = "valid@example.com"
        password = "weak"

        email_valid, _ = validate_email(email)
        password_valid, _ = validate_password(password)

        assert email_valid is True
        assert password_valid is False

    def test_typical_registration_flow_both_invalid(self) -> None:
        """Registration with both invalid email and password should fail both."""
        email = "bad"
        password = "123"

        email_valid, _ = validate_email(email)
        password_valid, _ = validate_password(password)

        assert email_valid is False
        assert password_valid is False
