"""Tests for output filter (PII/secret redaction)."""

import pytest

from aragora.gateway.security.output_filter import (
    OutputFilter,
    SensitiveDataType,
    SensitivePattern,
    RedactionResult,
)


class TestSensitiveDataType:
    """Tests for SensitiveDataType enum."""

    def test_data_type_values(self):
        """Test data type values."""
        assert SensitiveDataType.API_KEY.value == "api_key"
        assert SensitiveDataType.AWS_KEY.value == "aws_key"
        assert SensitiveDataType.CREDIT_CARD.value == "credit_card"
        assert SensitiveDataType.SSN.value == "ssn"
        assert SensitiveDataType.EMAIL.value == "email"


class TestSensitivePattern:
    """Tests for SensitivePattern."""

    def test_pattern_creation(self):
        """Test pattern creation."""
        pattern = SensitivePattern(
            name="test_pattern",
            pattern=r"SECRET-\d+",
            data_type=SensitiveDataType.CUSTOM,
        )
        assert pattern.name == "test_pattern"
        assert pattern.enabled is True
        assert pattern.priority == 0


class TestRedactionResult:
    """Tests for RedactionResult."""

    def test_result_creation(self):
        """Test result creation."""
        result = RedactionResult(
            original_length=100,
            redacted_length=80,
            redaction_count=3,
            redacted_types={"api_key": 2, "ssn": 1},
        )
        assert result.original_length == 100
        assert result.redacted_length == 80
        assert result.redaction_count == 3


class TestOutputFilter:
    """Tests for OutputFilter."""

    def test_default_patterns_loaded(self):
        """Test that default patterns are loaded."""
        filter_obj = OutputFilter()
        patterns = filter_obj.list_patterns()
        assert len(patterns) > 0

        # Check some expected patterns
        pattern_names = [p["name"] for p in patterns]
        assert "openai_api_key" in pattern_names
        assert "aws_access_key" in pattern_names
        assert "credit_card" in pattern_names

    @pytest.mark.asyncio
    async def test_redact_openai_api_key(self):
        """Test redacting OpenAI API key."""
        filter_obj = OutputFilter()
        text = "Use this key: sk-abc123def456ghi789jkl012mno345"

        redacted, result = await filter_obj.redact(text)

        assert "sk-abc123" not in redacted
        assert "[REDACTED:api_key]" in redacted
        assert result.redaction_count == 1

    @pytest.mark.asyncio
    async def test_redact_anthropic_api_key(self):
        """Test redacting Anthropic API key."""
        filter_obj = OutputFilter()
        text = "My API key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz"

        redacted, result = await filter_obj.redact(text)

        assert "sk-ant-api03" not in redacted
        assert "[REDACTED:api_key]" in redacted
        assert result.redaction_count >= 1

    @pytest.mark.asyncio
    async def test_redact_aws_access_key(self):
        """Test redacting AWS access key."""
        filter_obj = OutputFilter()
        text = "AWS Key: AKIAIOSFODNN7EXAMPLE"

        redacted, result = await filter_obj.redact(text)

        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED:aws_key]" in redacted

    @pytest.mark.asyncio
    async def test_redact_credit_card_visa(self):
        """Test redacting Visa credit card."""
        filter_obj = OutputFilter()
        text = "Card: 4111111111111111"

        redacted, result = await filter_obj.redact(text)

        assert "4111111111111111" not in redacted
        assert "[REDACTED:credit_card]" in redacted

    @pytest.mark.asyncio
    async def test_redact_credit_card_mastercard(self):
        """Test redacting Mastercard credit card."""
        filter_obj = OutputFilter()
        text = "Card: 5500000000000004"

        redacted, result = await filter_obj.redact(text)

        assert "5500000000000004" not in redacted
        assert "[REDACTED:credit_card]" in redacted

    @pytest.mark.asyncio
    async def test_redact_ssn(self):
        """Test redacting SSN."""
        filter_obj = OutputFilter()
        text = "SSN: 123-45-6789"

        redacted, result = await filter_obj.redact(text)

        assert "123-45-6789" not in redacted
        assert "[REDACTED:ssn]" in redacted

    @pytest.mark.asyncio
    async def test_redact_jwt_token(self):
        """Test redacting JWT token."""
        filter_obj = OutputFilter()
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Token: {jwt}"

        redacted, result = await filter_obj.redact(text)

        assert jwt not in redacted
        assert "[REDACTED:jwt_token]" in redacted

    @pytest.mark.asyncio
    async def test_redact_private_key(self):
        """Test redacting private key."""
        filter_obj = OutputFilter()
        text = """-----BEGIN RSA PRIVATE KEY-----
MIICXgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu
KUpRKfFLfRYC9AIKjbJTWit+CqvjWYzvQwECAwEAAQ==
-----END RSA PRIVATE KEY-----"""

        redacted, result = await filter_obj.redact(text)

        assert "MIICXgIBAAJBAKj34" not in redacted
        assert "[REDACTED:private_key]" in redacted

    @pytest.mark.asyncio
    async def test_redact_multiple_sensitive_items(self):
        """Test redacting multiple sensitive items."""
        filter_obj = OutputFilter()
        text = """
        API Key: sk-abc123def456ghi789jkl012mno345
        SSN: 123-45-6789
        Card: 4111111111111111
        """

        redacted, result = await filter_obj.redact(text)

        assert result.redaction_count >= 3
        assert "api_key" in result.redacted_types
        assert "ssn" in result.redacted_types
        assert "credit_card" in result.redacted_types

    @pytest.mark.asyncio
    async def test_no_redaction_for_clean_text(self):
        """Test no redaction for clean text."""
        filter_obj = OutputFilter()
        text = "This is a normal message without any sensitive data."

        redacted, result = await filter_obj.redact(text)

        assert redacted == text
        assert result.redaction_count == 0

    @pytest.mark.asyncio
    async def test_email_disabled_by_default(self):
        """Test that email redaction is disabled by default."""
        filter_obj = OutputFilter()
        text = "Email: user@example.com"

        redacted, result = await filter_obj.redact(text)

        # Email should not be redacted by default
        assert "user@example.com" in redacted

    @pytest.mark.asyncio
    async def test_enable_email_redaction(self):
        """Test enabling email redaction."""
        filter_obj = OutputFilter()
        filter_obj.enable_pattern("email")
        text = "Email: user@example.com"

        redacted, result = await filter_obj.redact(text)

        assert "user@example.com" not in redacted
        assert "[REDACTED:email]" in redacted

    def test_add_custom_pattern(self):
        """Test adding a custom pattern."""
        filter_obj = OutputFilter()
        filter_obj.add_pattern(
            SensitivePattern(
                name="internal_id",
                pattern=r"ACME-\d{10}",
                data_type=SensitiveDataType.CUSTOM,
            )
        )

        patterns = filter_obj.list_patterns()
        pattern_names = [p["name"] for p in patterns]
        assert "internal_id" in pattern_names

    @pytest.mark.asyncio
    async def test_redact_custom_pattern(self):
        """Test redacting with custom pattern."""
        filter_obj = OutputFilter()
        filter_obj.add_pattern(
            SensitivePattern(
                name="internal_id",
                pattern=r"ACME-\d{10}",
                data_type=SensitiveDataType.CUSTOM,
            )
        )
        text = "Internal ID: ACME-1234567890"

        redacted, result = await filter_obj.redact(text)

        assert "ACME-1234567890" not in redacted
        assert "[REDACTED:custom]" in redacted

    def test_disable_pattern(self):
        """Test disabling a pattern."""
        filter_obj = OutputFilter()
        result = filter_obj.disable_pattern("credit_card")
        assert result is True

        patterns = filter_obj.list_patterns()
        cc_pattern = next(p for p in patterns if p["name"] == "credit_card")
        assert cc_pattern["enabled"] is False

    @pytest.mark.asyncio
    async def test_disabled_pattern_not_redacted(self):
        """Test that disabled patterns don't redact."""
        filter_obj = OutputFilter()
        filter_obj.disable_pattern("credit_card")
        text = "Card: 4111111111111111"

        redacted, result = await filter_obj.redact(text)

        assert "4111111111111111" in redacted  # Not redacted

    def test_create_strict_filter(self):
        """Test creating strict filter with all patterns enabled."""
        filter_obj = OutputFilter.create_strict()
        patterns = filter_obj.list_patterns()

        # All patterns should be enabled
        for pattern in patterns:
            assert pattern["enabled"] is True

    def test_create_minimal_filter(self):
        """Test creating minimal filter with only critical patterns."""
        filter_obj = OutputFilter.create_minimal()
        patterns = filter_obj.list_patterns()

        # Should only have critical patterns
        pattern_types = {p["data_type"] for p in patterns}
        assert "api_key" in pattern_types
        assert "private_key" in pattern_types
        # Should not have non-critical patterns
        assert "email" not in pattern_types
        assert "phone" not in pattern_types
