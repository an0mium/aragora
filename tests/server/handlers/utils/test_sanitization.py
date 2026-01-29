"""
Tests for response sanitization utilities.

Tests cover:
- sanitize_response basic functionality
- Recursive sanitization
- Redact mode vs remove mode
- Specialized sanitizers (user, integration, payment)
- sanitize_output decorator
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.utils.sanitization import (
    RESPONSE_SENSITIVE_FIELDS,
    sanitize_response,
    sanitize_user_response,
    sanitize_integration_response,
    sanitize_payment_response,
    sanitize_output,
)


class TestSanitizeResponse:
    """Tests for sanitize_response function."""

    def test_removes_password_fields(self):
        """Should remove password-related fields."""
        data = {
            "id": "user-123",
            "email": "user@example.com",
            "password": "secret123",
            "password_hash": "abc123hash",
            "password_salt": "xyz789",
        }

        result = sanitize_response(data)

        assert result["id"] == "user-123"
        assert result["email"] == "user@example.com"
        assert "password" not in result
        assert "password_hash" not in result
        assert "password_salt" not in result

    def test_removes_token_fields(self):
        """Should remove token-related fields."""
        data = {
            "user_id": "123",
            "access_token": "eyJ...",
            "refresh_token": "xyz...",
            "api_key": "sk_live_...",
        }

        result = sanitize_response(data)

        assert result["user_id"] == "123"
        assert "access_token" not in result
        assert "refresh_token" not in result
        assert "api_key" not in result

    def test_removes_secret_fields(self):
        """Should remove secret-related fields."""
        data = {
            "name": "Integration",
            "client_id": "abc123",
            "client_secret": "super-secret",
            "webhook_secret": "whsec_...",
        }

        result = sanitize_response(data)

        assert result["name"] == "Integration"
        assert result["client_id"] == "abc123"
        assert "client_secret" not in result
        assert "webhook_secret" not in result

    def test_removes_mfa_fields(self):
        """Should remove MFA-related fields."""
        data = {
            "id": "user-123",
            "mfa_enabled": True,
            "mfa_secret": "JBSWY3DPEHPK3PXP",
            "totp_secret": "base32secret",
            "mfa_backup_codes": ["12345678", "87654321"],
        }

        result = sanitize_response(data)

        assert result["id"] == "user-123"
        assert result["mfa_enabled"] is True
        assert "mfa_secret" not in result
        assert "totp_secret" not in result
        assert "mfa_backup_codes" not in result

    def test_preserves_safe_fields(self):
        """Should preserve non-sensitive fields."""
        data = {
            "id": "123",
            "name": "Test User",
            "email": "test@example.com",
            "created_at": "2025-01-01T00:00:00Z",
            "role": "admin",
        }

        result = sanitize_response(data)

        assert result == data

    def test_handles_none(self):
        """Should handle None input."""
        assert sanitize_response(None) is None

    def test_handles_empty_dict(self):
        """Should handle empty dict."""
        assert sanitize_response({}) == {}

    def test_handles_non_dict(self):
        """Should pass through non-dict values."""
        assert sanitize_response("string") == "string"
        assert sanitize_response(123) == 123
        assert sanitize_response(True) is True


class TestRecursiveSanitization:
    """Tests for recursive sanitization."""

    def test_sanitizes_nested_dicts(self):
        """Should sanitize nested dictionaries."""
        data = {
            "user": {
                "id": "123",
                "password_hash": "abc",
            },
            "settings": {
                "theme": "dark",
                "api_key": "sk_...",
            },
        }

        result = sanitize_response(data)

        assert result["user"]["id"] == "123"
        assert "password_hash" not in result["user"]
        assert result["settings"]["theme"] == "dark"
        assert "api_key" not in result["settings"]

    def test_sanitizes_lists_of_dicts(self):
        """Should sanitize lists of dictionaries."""
        data = {
            "users": [
                {"id": "1", "password": "pass1"},
                {"id": "2", "password": "pass2"},
            ]
        }

        result = sanitize_response(data)

        assert len(result["users"]) == 2
        assert result["users"][0]["id"] == "1"
        assert "password" not in result["users"][0]
        assert result["users"][1]["id"] == "2"
        assert "password" not in result["users"][1]

    def test_handles_deeply_nested(self):
        """Should handle deeply nested structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": "safe",
                        "secret": "hidden",
                    }
                }
            }
        }

        result = sanitize_response(data)

        assert result["level1"]["level2"]["level3"]["data"] == "safe"
        assert "secret" not in result["level1"]["level2"]["level3"]

    def test_no_recursive_when_disabled(self):
        """Should not recurse when recursive=False."""
        data = {
            "user": {
                "id": "123",
                "password": "secret",
            },
            "password": "top-level",
        }

        result = sanitize_response(data, recursive=False)

        # Top-level removed
        assert "password" not in result
        # Nested NOT removed
        assert result["user"]["password"] == "secret"

    def test_removes_nested_sensitive_containers(self):
        """Should remove entire 'credentials' containers."""
        data = {
            "name": "Service",
            "credentials": {
                "username": "admin",
                "password": "secret",
            },
        }

        result = sanitize_response(data)

        assert result["name"] == "Service"
        assert "credentials" not in result


class TestRedactMode:
    """Tests for redact mode."""

    def test_redact_replaces_values(self):
        """Should replace sensitive values with redact string."""
        data = {
            "id": "123",
            "password": "secret",
            "api_key": "sk_live_...",
        }

        result = sanitize_response(data, redact_value="[REDACTED]")

        assert result["id"] == "123"
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"

    def test_redact_custom_value(self):
        """Should use custom redact value."""
        data = {"password": "secret"}

        result = sanitize_response(data, redact_value="***")
        assert result["password"] == "***"

    def test_redact_nested(self):
        """Should redact nested fields."""
        data = {
            "user": {
                "password": "secret",
            }
        }

        result = sanitize_response(data, redact_value="[HIDDEN]")

        assert result["user"]["password"] == "[HIDDEN]"


class TestAdditionalFields:
    """Tests for additional fields parameter."""

    def test_removes_additional_fields(self):
        """Should remove additional specified fields."""
        data = {
            "id": "123",
            "internal_notes": "Do not share",
            "debug_info": {"trace": "..."},
        }

        result = sanitize_response(data, additional_fields={"internal_notes", "debug_info"})

        assert result["id"] == "123"
        assert "internal_notes" not in result
        assert "debug_info" not in result

    def test_combines_with_default_fields(self):
        """Should combine additional with default sensitive fields."""
        data = {
            "id": "123",
            "password": "secret",
            "custom_secret": "also-secret",
        }

        result = sanitize_response(data, additional_fields={"custom_secret"})

        assert result["id"] == "123"
        assert "password" not in result
        assert "custom_secret" not in result


class TestSanitizeUserResponse:
    """Tests for sanitize_user_response."""

    def test_removes_password_fields(self):
        """Should remove all password-related fields."""
        user_data = {
            "id": "user-123",
            "email": "user@example.com",
            "password_hash": "hash",
            "password_salt": "salt",
            "password_reset_token": "token",
            "password_reset_expires": "2025-01-01",
        }

        result = sanitize_user_response(user_data)

        assert result["id"] == "user-123"
        assert result["email"] == "user@example.com"
        assert "password_hash" not in result
        assert "password_salt" not in result
        assert "password_reset_token" not in result
        assert "password_reset_expires" not in result

    def test_removes_session_tokens(self):
        """Should remove session-related tokens."""
        user_data = {
            "id": "user-123",
            "session_token": "sess_...",
            "email_verification_token": "verify_...",
        }

        result = sanitize_user_response(user_data)

        assert result["id"] == "user-123"
        assert "session_token" not in result
        assert "email_verification_token" not in result


class TestSanitizeIntegrationResponse:
    """Tests for sanitize_integration_response."""

    def test_removes_oauth_tokens(self):
        """Should remove OAuth tokens."""
        integration = {
            "id": "int-123",
            "name": "Slack",
            "oauth_access_token": "xoxb-...",
            "oauth_refresh_token": "xoxr-...",
        }

        result = sanitize_integration_response(integration)

        assert result["id"] == "int-123"
        assert result["name"] == "Slack"
        assert "oauth_access_token" not in result
        assert "oauth_refresh_token" not in result

    def test_removes_api_secrets(self):
        """Should remove API secrets."""
        integration = {
            "id": "int-123",
            "api_key": "visible",  # This is in default sensitive fields
            "api_secret": "hidden",
            "verification_token": "vtoken",
        }

        result = sanitize_integration_response(integration)

        assert result["id"] == "int-123"
        assert "api_key" not in result
        assert "api_secret" not in result
        assert "verification_token" not in result


class TestSanitizePaymentResponse:
    """Tests for sanitize_payment_response."""

    def test_removes_card_details(self):
        """Should remove full card details."""
        payment = {
            "id": "pay-123",
            "amount": 100.00,
            "card_number": "4111111111111111",
            "cvv": "123",
        }

        result = sanitize_payment_response(payment)

        assert result["id"] == "pay-123"
        assert result["amount"] == 100.00
        assert "card_number" not in result
        assert "cvv" not in result

    def test_removes_bank_details(self):
        """Should remove bank account details."""
        payment = {
            "id": "pay-123",
            "account_number": "1234567890",
            "routing_number": "021000021",
        }

        result = sanitize_payment_response(payment)

        assert result["id"] == "pay-123"
        assert "account_number" not in result
        assert "routing_number" not in result


class TestSanitizeOutputDecorator:
    """Tests for sanitize_output decorator."""

    def test_decorator_on_sync_function(self):
        """Should sanitize sync function output."""

        @sanitize_output()
        def get_user():
            return {"id": "123", "password": "secret"}

        result = get_user()

        assert result["id"] == "123"
        assert "password" not in result

    @pytest.mark.asyncio
    async def test_decorator_on_async_function(self):
        """Should sanitize async function output."""

        @sanitize_output()
        async def get_user_async():
            return {"id": "123", "api_key": "sk_..."}

        result = await get_user_async()

        assert result["id"] == "123"
        assert "api_key" not in result

    def test_decorator_with_additional_fields(self):
        """Should respect additional_fields parameter."""

        @sanitize_output(additional_fields={"internal_id"})
        def get_data():
            return {"name": "Test", "internal_id": "int-123"}

        result = get_data()

        assert result["name"] == "Test"
        assert "internal_id" not in result

    def test_decorator_with_redact_value(self):
        """Should respect redact_value parameter."""

        @sanitize_output(redact_value="***")
        def get_data():
            return {"name": "Test", "secret": "hidden"}

        result = get_data()

        assert result["name"] == "Test"
        assert result["secret"] == "***"

    def test_decorator_passes_through_non_dict(self):
        """Should pass through non-dict return values."""

        @sanitize_output()
        def get_string():
            return "just a string"

        result = get_string()
        assert result == "just a string"

    def test_decorator_handles_list(self):
        """Should sanitize list return values."""

        @sanitize_output()
        def get_users():
            return [
                {"id": "1", "password": "pass1"},
                {"id": "2", "password": "pass2"},
            ]

        result = get_users()

        assert len(result) == 2
        assert "password" not in result[0]
        assert "password" not in result[1]


class TestSensitiveFieldsList:
    """Tests for RESPONSE_SENSITIVE_FIELDS constant."""

    def test_contains_password_fields(self):
        """Should contain password-related fields."""
        assert "password" in RESPONSE_SENSITIVE_FIELDS
        assert "password_hash" in RESPONSE_SENSITIVE_FIELDS
        assert "password_salt" in RESPONSE_SENSITIVE_FIELDS

    def test_contains_token_fields(self):
        """Should contain token-related fields."""
        assert "access_token" in RESPONSE_SENSITIVE_FIELDS
        assert "refresh_token" in RESPONSE_SENSITIVE_FIELDS
        assert "api_key" in RESPONSE_SENSITIVE_FIELDS

    def test_contains_secret_fields(self):
        """Should contain secret-related fields."""
        assert "client_secret" in RESPONSE_SENSITIVE_FIELDS
        assert "webhook_secret" in RESPONSE_SENSITIVE_FIELDS
        assert "signing_secret" in RESPONSE_SENSITIVE_FIELDS

    def test_contains_platform_specific(self):
        """Should contain platform-specific credentials."""
        assert "slack_signing_secret" in RESPONSE_SENSITIVE_FIELDS
        assert "stripe_secret_key" in RESPONSE_SENSITIVE_FIELDS
        assert "telegram_token" in RESPONSE_SENSITIVE_FIELDS

    def test_is_frozenset(self):
        """Should be immutable frozenset."""
        assert isinstance(RESPONSE_SENSITIVE_FIELDS, frozenset)
