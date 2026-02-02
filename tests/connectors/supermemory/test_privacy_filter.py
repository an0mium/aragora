"""Tests for Supermemory privacy filter."""

import pytest

from aragora.connectors.supermemory.privacy_filter import (
    PrivacyFilter,
    PrivacyFilterConfig,
)


class TestPrivacyFilter:
    """Test PrivacyFilter class."""

    def test_filter_api_keys(self):
        """Test filtering API key patterns."""
        filter = PrivacyFilter()

        # OpenAI-style key
        content = "My key is sk-abcdefghijklmnopqrstuvwxyz123456"
        filtered = filter.filter(content)
        assert "sk-abcdef" not in filtered
        assert "[REDACTED_SK_KEY]" in filtered

        # Supermemory key
        content = "API key: sm_test_abcdefghijklmnopqrstuvwxyz1234567890ABCDEF"
        filtered = filter.filter(content)
        assert "sm_test_" not in filtered
        assert "[REDACTED_SM_KEY]" in filtered

    def test_filter_bearer_tokens(self):
        """Test filtering bearer tokens."""
        filter = PrivacyFilter()

        content = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload"
        filtered = filter.filter(content)
        assert "eyJhbGci" not in filtered
        assert "Bearer [REDACTED_TOKEN]" in filtered

    def test_filter_github_tokens(self):
        """Test filtering GitHub tokens."""
        filter = PrivacyFilter()

        # Personal access token - ghp_ + exactly 36 alphanumeric
        # 36 chars: "aA1bB2cC3dD4eE5fF6gG7hH8iI9jJ0kK1lL2m"
        content = "GitHub: ghp_aA1bB2cC3dD4eE5fF6gG7hH8iI9jJ0kK1lL2m"
        filtered = filter.filter(content)
        assert "ghp_aA1bB2" not in filtered
        assert "[REDACTED" in filtered

        # OAuth token - gho_ + exactly 36 alphanumeric
        content = "GitHub OAuth: gho_aA1bB2cC3dD4eE5fF6gG7hH8iI9jJ0kK1lL2m"
        filtered = filter.filter(content)
        assert "gho_aA1bB2" not in filtered
        assert "[REDACTED" in filtered

    def test_filter_passwords(self):
        """Test filtering password patterns."""
        filter = PrivacyFilter()

        content = 'config = {"password": "supersecret123"}'
        filtered = filter.filter(content)
        assert "supersecret123" not in filtered
        assert "password=[REDACTED]" in filtered

        content = "secret='my_secret_value'"
        filtered = filter.filter(content)
        assert "my_secret_value" not in filtered

    def test_filter_connection_strings(self):
        """Test filtering database connection strings."""
        filter = PrivacyFilter()

        # PostgreSQL
        content = "DATABASE_URL=postgres://user:pass@localhost:5432/db"
        filtered = filter.filter(content)
        assert "postgres://" not in filtered
        assert "[REDACTED_DB_URL]" in filtered

        # MongoDB
        content = "MONGO_URI=mongodb://user:pass@cluster.mongodb.net/db"
        filtered = filter.filter(content)
        assert "mongodb://" not in filtered

        # Redis
        content = "REDIS_URL=redis://default:password@redis-host:6379"
        filtered = filter.filter(content)
        assert "redis://" not in filtered

    def test_filter_env_vars(self):
        """Test filtering environment variable patterns."""
        filter = PrivacyFilter()

        content = "export ANTHROPIC_API_KEY=sk-ant-api03-..."
        filtered = filter.filter(content)
        assert "sk-ant" not in filtered
        assert "ANTHROPIC_API_KEY=[REDACTED]" in filtered

        content = "OPENAI_API_KEY=sk-proj-abcdefg"
        filtered = filter.filter(content)
        assert "sk-proj" not in filtered

    def test_filter_preserves_safe_content(self):
        """Test that safe content is preserved."""
        filter = PrivacyFilter()

        content = "The debate concluded with 85% confidence. No secrets here."
        filtered = filter.filter(content)
        assert filtered == content

    def test_filter_multiple_patterns(self):
        """Test filtering multiple sensitive patterns in one string."""
        filter = PrivacyFilter()

        content = """
        API_KEY=sk-1234567890123456789012345678901234
        TOKEN=Bearer eyJhbGciOiJIUzI1NiJ9.test
        DB=postgres://user:pass@localhost/db
        """
        filtered = filter.filter(content)

        assert "sk-" not in filtered
        assert "eyJhbGci" not in filtered
        assert "postgres://" not in filtered

    def test_filter_empty_content(self):
        """Test filtering empty content."""
        filter = PrivacyFilter()

        assert filter.filter("") == ""
        assert filter.filter(None) is None


class TestPrivacyFilterConfig:
    """Test PrivacyFilterConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PrivacyFilterConfig()

        assert config.filter_api_keys is True
        assert config.filter_tokens is True
        assert config.filter_passwords is True
        assert config.filter_emails is False
        assert config.filter_phone_numbers is False

    def test_disabled_filters(self):
        """Test disabling specific filters."""
        config = PrivacyFilterConfig(
            filter_api_keys=False,
            filter_tokens=False,
            filter_passwords=False,
        )
        filter = PrivacyFilter(config)

        # These should NOT be filtered when disabled
        content = "sk-abcdefghijklmnopqrstuvwxyz123456"
        filtered = filter.filter(content)
        assert filtered == content

    def test_pii_filters_opt_in(self):
        """Test PII filters are opt-in."""
        # Default config should NOT filter emails
        default_filter = PrivacyFilter()
        content = "Contact: test@example.com"
        assert default_filter.filter(content) == content

        # With emails enabled
        config = PrivacyFilterConfig(filter_emails=True)
        pii_filter = PrivacyFilter(config)
        filtered = pii_filter.filter(content)
        assert "test@example.com" not in filtered
        assert "[REDACTED_EMAIL]" in filtered

    def test_phone_filter_opt_in(self):
        """Test phone number filtering is opt-in."""
        # Default should NOT filter
        default_filter = PrivacyFilter()
        content = "Call me at 555-123-4567"
        assert default_filter.filter(content) == content

        # With phone enabled
        config = PrivacyFilterConfig(filter_phone_numbers=True)
        phone_filter = PrivacyFilter(config)
        filtered = phone_filter.filter(content)
        assert "555-123-4567" not in filtered
        assert "[REDACTED_PHONE]" in filtered

    def test_custom_patterns(self):
        """Test custom pattern filtering."""
        config = PrivacyFilterConfig(
            custom_patterns=[
                (r"ACME-\d{6}", "[ACME_ID]"),
                (r"internal://[^\s]+", "[INTERNAL_URL]"),
            ]
        )
        filter = PrivacyFilter(config)

        content = "Reference: ACME-123456 at internal://docs/secret"
        filtered = filter.filter(content)

        assert "ACME-123456" not in filtered
        assert "[ACME_ID]" in filtered
        assert "internal://docs/secret" not in filtered
        assert "[INTERNAL_URL]" in filtered


class TestPrivacyFilterMetadata:
    """Test metadata filtering."""

    def test_filter_metadata_basic(self):
        """Test filtering metadata dictionary."""
        filter = PrivacyFilter()

        metadata = {
            "debate_id": "123",
            "api_key": "sk-secret123456789012345678901234",
            "content": "Bearer token_here",
        }
        filtered = filter.filter_metadata(metadata)

        assert filtered["debate_id"] == "123"
        assert "sk-secret" not in filtered.get("api_key", "")
        assert filtered["api_key"] == "[REDACTED]"  # Key name triggers redaction

    def test_filter_metadata_nested(self):
        """Test filtering nested metadata."""
        filter = PrivacyFilter()

        metadata = {
            "outer": {
                "password": "secret123",
                "nested": {
                    "token": "abc123",
                },
            }
        }
        filtered = filter.filter_metadata(metadata)

        assert filtered["outer"]["password"] == "[REDACTED]"
        assert filtered["outer"]["nested"]["token"] == "[REDACTED]"

    def test_filter_metadata_empty(self):
        """Test filtering empty metadata."""
        filter = PrivacyFilter()

        assert filter.filter_metadata({}) == {}
        assert filter.filter_metadata(None) is None

    def test_filter_metadata_list_values(self):
        """Test filtering metadata with list values."""
        filter = PrivacyFilter()

        metadata = {
            "items": [
                "normal text",
                "sk-secret12345678901234567890123456",  # 32 chars after sk-
            ]
        }
        filtered = filter.filter_metadata(metadata)

        assert filtered["items"][0] == "normal text"
        assert "sk-secret" not in filtered["items"][1]


class TestPrivacyFilterIsSafe:
    """Test is_safe method."""

    def test_is_safe_clean_content(self):
        """Test is_safe returns True for clean content."""
        filter = PrivacyFilter()

        assert filter.is_safe("This is clean content.") is True
        assert filter.is_safe("Debate outcome: consensus reached.") is True

    def test_is_safe_sensitive_content(self):
        """Test is_safe returns False for sensitive content."""
        filter = PrivacyFilter()

        assert filter.is_safe("Key: sk-abcdefghijklmnopqrstuvwxyz123456") is False
        assert filter.is_safe("Bearer eyJhbGciOiJIUzI1NiJ9") is False
        assert filter.is_safe("postgres://user:pass@host/db") is False
