"""
Tests for aragora.skills.builtin.data_extraction module.

Covers:
- DataExtractionSkill manifest and initialization
- Email extraction
- URL extraction
- Phone number extraction
- Date extraction
- Money/currency extraction
- Number and percentage extraction
- JSON extraction
- Key-value extraction
- Custom pattern extraction
"""

from __future__ import annotations

from typing import Any

import pytest

from aragora.skills.base import SkillContext
from aragora.skills.builtin.data_extraction import DataExtractionSkill, ExtractionResult


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def skill() -> DataExtractionSkill:
    """Create a data extraction skill for testing."""
    return DataExtractionSkill()


@pytest.fixture
def context() -> SkillContext:
    """Create a context for testing."""
    return SkillContext(user_id="user123")


@pytest.fixture
def sample_text() -> str:
    """Sample text with various extractable data."""
    return """
    Contact John Smith at john.smith@example.com or call (555) 123-4567.
    Our website is https://www.example.com/products?id=123.
    The meeting is scheduled for January 15, 2024 at 2:30 PM.
    The project costs $1,500.50 USD and represents a 25% increase.
    Configuration: {"debug": true, "timeout": 30}
    Status: active
    Count: 42
    """


# =============================================================================
# ExtractionResult Tests
# =============================================================================


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_create_extraction_result(self):
        """Test creating an extraction result."""
        result = ExtractionResult(
            type="email",
            value="test@example.com",
            start=0,
            end=16,
        )

        assert result.type == "email"
        assert result.value == "test@example.com"
        assert result.start == 0
        assert result.end == 16

    def test_extraction_result_defaults(self):
        """Test extraction result default values."""
        result = ExtractionResult(
            type="test",
            value="value",
            start=0,
            end=5,
        )

        assert result.confidence == 1.0
        assert result.metadata == {}

    def test_to_dict(self):
        """Test converting extraction result to dict."""
        result = ExtractionResult(
            type="email",
            value="test@example.com",
            start=0,
            end=16,
            confidence=0.95,
            metadata={"domain": "example.com"},
        )

        data = result.to_dict()

        assert data["type"] == "email"
        assert data["value"] == "test@example.com"
        assert data["confidence"] == 0.95
        assert data["metadata"]["domain"] == "example.com"


# =============================================================================
# DataExtractionSkill Manifest Tests
# =============================================================================


class TestDataExtractionSkillManifest:
    """Tests for DataExtractionSkill manifest."""

    def test_manifest_name(self, skill: DataExtractionSkill):
        """Test manifest name."""
        assert skill.manifest.name == "data_extraction"

    def test_manifest_version(self, skill: DataExtractionSkill):
        """Test manifest version."""
        assert skill.manifest.version == "1.0.0"

    def test_manifest_input_schema(self, skill: DataExtractionSkill):
        """Test manifest input schema."""
        schema = skill.manifest.input_schema

        assert "text" in schema
        assert schema["text"]["type"] == "string"
        assert schema["text"]["required"] is True

        assert "extract_types" in schema
        assert "custom_patterns" in schema

    def test_manifest_debate_compatible(self, skill: DataExtractionSkill):
        """Test skill is debate compatible."""
        assert skill.manifest.debate_compatible is True


# =============================================================================
# Email Extraction Tests
# =============================================================================


class TestEmailExtraction:
    """Tests for email extraction."""

    def test_extract_simple_email(self, skill: DataExtractionSkill):
        """Test extracting simple email."""
        text = "Contact us at test@example.com"
        results = skill._extract_emails(text)

        assert len(results) == 1
        assert results[0].value == "test@example.com"

    def test_extract_multiple_emails(self, skill: DataExtractionSkill):
        """Test extracting multiple emails."""
        text = "Email john@example.com or jane@test.org"
        results = skill._extract_emails(text)

        assert len(results) == 2
        values = [r.value for r in results]
        assert "john@example.com" in values
        assert "jane@test.org" in values

    def test_extract_email_with_subdomain(self, skill: DataExtractionSkill):
        """Test extracting email with subdomain."""
        text = "Contact admin@mail.example.com"
        results = skill._extract_emails(text)

        assert len(results) == 1
        assert results[0].value == "admin@mail.example.com"

    def test_extract_email_with_plus(self, skill: DataExtractionSkill):
        """Test extracting email with plus addressing."""
        text = "Email user+tag@example.com"
        results = skill._extract_emails(text)

        assert len(results) == 1
        assert results[0].value == "user+tag@example.com"


# =============================================================================
# URL Extraction Tests
# =============================================================================


class TestURLExtraction:
    """Tests for URL extraction."""

    def test_extract_simple_url(self, skill: DataExtractionSkill):
        """Test extracting simple URL."""
        text = "Visit https://example.com"
        results = skill._extract_urls(text)

        assert len(results) == 1
        assert results[0].value == "https://example.com"

    def test_extract_url_with_path(self, skill: DataExtractionSkill):
        """Test extracting URL with path."""
        text = "See https://example.com/path/to/page"
        results = skill._extract_urls(text)

        assert len(results) == 1
        assert "/path/to/page" in results[0].value

    def test_extract_url_with_query(self, skill: DataExtractionSkill):
        """Test extracting URL with query parameters."""
        text = "Link: https://example.com/search?q=test&page=1"
        results = skill._extract_urls(text)

        assert len(results) == 1
        assert "q=test" in results[0].value

    def test_extract_url_metadata(self, skill: DataExtractionSkill):
        """Test URL metadata extraction."""
        text = "Visit https://www.example.com/page"
        results = skill._extract_urls(text)

        assert len(results) == 1
        assert "domain" in results[0].metadata


# =============================================================================
# Phone Number Extraction Tests
# =============================================================================


class TestPhoneExtraction:
    """Tests for phone number extraction."""

    def test_extract_us_phone(self, skill: DataExtractionSkill):
        """Test extracting US phone number."""
        text = "Call (555) 123-4567"
        results = skill._extract_phones(text)

        assert len(results) == 1
        assert "555" in results[0].value

    def test_extract_phone_with_country_code(self, skill: DataExtractionSkill):
        """Test extracting phone with country code."""
        text = "Call +1-555-123-4567"
        results = skill._extract_phones(text)

        assert len(results) == 1

    def test_phone_normalization(self, skill: DataExtractionSkill):
        """Test phone number normalization."""
        text = "Call (555) 123-4567"
        results = skill._extract_phones(text)

        assert len(results) == 1
        assert "normalized" in results[0].metadata


# =============================================================================
# Date Extraction Tests
# =============================================================================


class TestDateExtraction:
    """Tests for date extraction."""

    def test_extract_iso_date(self, skill: DataExtractionSkill):
        """Test extracting ISO format date."""
        text = "Date: 2024-01-15"
        results = skill._extract_dates(text)

        assert len(results) >= 1
        values = [r.value for r in results]
        assert any("2024-01-15" in v for v in values)

    def test_extract_us_date(self, skill: DataExtractionSkill):
        """Test extracting US format date."""
        text = "Date: 01/15/2024"
        results = skill._extract_dates(text)

        assert len(results) >= 1

    def test_extract_written_date(self, skill: DataExtractionSkill):
        """Test extracting written date."""
        text = "Meeting on January 15, 2024"
        results = skill._extract_dates(text)

        assert len(results) >= 1
        values = [r.value for r in results]
        assert any("January" in v for v in values)

    def test_extract_time(self, skill: DataExtractionSkill):
        """Test extracting time."""
        text = "Meeting at 2:30 PM"
        results = skill._extract_dates(text)

        assert len(results) >= 1
        values = [r.value for r in results]
        assert any("2:30" in v for v in values)


# =============================================================================
# Money Extraction Tests
# =============================================================================


class TestMoneyExtraction:
    """Tests for money/currency extraction."""

    def test_extract_usd_dollar_sign(self, skill: DataExtractionSkill):
        """Test extracting USD with dollar sign."""
        text = "Cost: $1,500.50"
        results = skill._extract_money(text)

        assert len(results) >= 1
        assert results[0].metadata.get("currency") == "USD"

    def test_extract_money_amount(self, skill: DataExtractionSkill):
        """Test extracting money amount value."""
        text = "Cost: $1,500.50"
        results = skill._extract_money(text)

        assert len(results) >= 1
        assert results[0].metadata.get("amount") == 1500.50

    def test_extract_money_words(self, skill: DataExtractionSkill):
        """Test extracting money with words."""
        text = "Cost is 100 dollars"
        results = skill._extract_money(text)

        assert len(results) >= 1


# =============================================================================
# Number and Percentage Extraction Tests
# =============================================================================


class TestNumberExtraction:
    """Tests for number extraction."""

    def test_extract_integer(self, skill: DataExtractionSkill):
        """Test extracting integer."""
        text = "There are 42 items"
        results = skill._extract_numbers(text)

        values = [r.value for r in results]
        assert "42" in values

    def test_extract_decimal(self, skill: DataExtractionSkill):
        """Test extracting decimal number."""
        text = "Value is 3.14159"
        results = skill._extract_numbers(text)

        values = [r.value for r in results]
        assert any("3.14" in v for v in values)


class TestPercentageExtraction:
    """Tests for percentage extraction."""

    def test_extract_percentage(self, skill: DataExtractionSkill):
        """Test extracting percentage."""
        text = "Growth of 25%"
        results = skill._extract_percentages(text)

        assert len(results) == 1
        assert results[0].value == "25%"

    def test_extract_decimal_percentage(self, skill: DataExtractionSkill):
        """Test extracting decimal percentage."""
        text = "Increase of 12.5%"
        results = skill._extract_percentages(text)

        assert len(results) == 1
        assert results[0].metadata.get("numeric_value") == 12.5


# =============================================================================
# JSON Extraction Tests
# =============================================================================


class TestJSONExtraction:
    """Tests for JSON extraction."""

    def test_extract_json_object(self, skill: DataExtractionSkill):
        """Test extracting JSON object."""
        text = 'Config: {"debug": true, "timeout": 30}'
        results = skill._extract_json(text)

        assert len(results) >= 1
        assert results[0].metadata.get("json_type") == "object"

    def test_extract_json_array(self, skill: DataExtractionSkill):
        """Test extracting JSON array."""
        text = "Items: [1, 2, 3]"
        results = skill._extract_json(text)

        assert len(results) >= 1
        assert results[0].metadata.get("json_type") == "array"

    def test_json_parsed_value(self, skill: DataExtractionSkill):
        """Test JSON is properly parsed."""
        text = 'Data: {"key": "value"}'
        results = skill._extract_json(text)

        assert len(results) >= 1
        parsed = results[0].metadata.get("parsed")
        assert parsed == {"key": "value"}


# =============================================================================
# Key-Value Extraction Tests
# =============================================================================


class TestKeyValueExtraction:
    """Tests for key-value extraction."""

    def test_extract_colon_separated(self, skill: DataExtractionSkill):
        """Test extracting colon-separated key-value."""
        text = "Status: active"
        results = skill._extract_key_values(text)

        assert len(results) >= 1
        assert any(r.metadata.get("key") == "Status" for r in results)

    def test_extract_equals_separated(self, skill: DataExtractionSkill):
        """Test extracting equals-separated key-value."""
        text = "count = 42"
        results = skill._extract_key_values(text)

        assert len(results) >= 1


# =============================================================================
# Custom Pattern Extraction Tests
# =============================================================================


class TestCustomPatternExtraction:
    """Tests for custom pattern extraction."""

    def test_extract_custom_pattern(self, skill: DataExtractionSkill):
        """Test extracting with custom pattern."""
        text = "Order ID: ORD-12345"
        results = skill._extract_custom(text, r"ORD-\d+", "order_id")

        assert len(results) == 1
        assert results[0].value == "ORD-12345"

    def test_custom_pattern_with_groups(self, skill: DataExtractionSkill):
        """Test custom pattern with capture groups."""
        text = "Version: v1.2.3"
        results = skill._extract_custom(text, r"v(\d+)\.(\d+)\.(\d+)", "version")

        assert len(results) == 1
        assert results[0].metadata.get("groups") == ("1", "2", "3")


# =============================================================================
# Full Execution Tests
# =============================================================================


class TestDataExtractionExecution:
    """Tests for full skill execution."""

    @pytest.mark.asyncio
    async def test_execute_missing_text(self, skill: DataExtractionSkill, context: SkillContext):
        """Test execution fails without text."""
        result = await skill.execute({}, context)

        assert result.success is False
        assert "text" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_all_types(
        self, skill: DataExtractionSkill, context: SkillContext, sample_text: str
    ):
        """Test extraction of all types."""
        result = await skill.execute({"text": sample_text}, context)

        assert result.success is True
        extractions = result.data["extractions"]

        # Should have various types
        assert "emails" in extractions
        assert "urls" in extractions
        assert "phone_numbers" in extractions

    @pytest.mark.asyncio
    async def test_execute_specific_types(
        self, skill: DataExtractionSkill, context: SkillContext, sample_text: str
    ):
        """Test extraction of specific types only."""
        result = await skill.execute(
            {"text": sample_text, "extract_types": ["email", "url"]},
            context,
        )

        assert result.success is True
        extractions = result.data["extractions"]

        assert "emails" in extractions
        assert "urls" in extractions
        # Should not have other types
        assert "money" not in extractions

    @pytest.mark.asyncio
    async def test_execute_with_custom_patterns(
        self, skill: DataExtractionSkill, context: SkillContext
    ):
        """Test execution with custom patterns."""
        text = "Order: ORD-12345, Product: PRD-67890"
        result = await skill.execute(
            {
                "text": text,
                "extract_types": [],
                "custom_patterns": {
                    "order": r"ORD-\d+",
                    "product": r"PRD-\d+",
                },
            },
            context,
        )

        assert result.success is True
        extractions = result.data["extractions"]

        assert "custom_order" in extractions
        assert "custom_product" in extractions

    @pytest.mark.asyncio
    async def test_execute_include_positions(
        self, skill: DataExtractionSkill, context: SkillContext
    ):
        """Test extraction with positions included."""
        text = "Email: test@example.com"
        result = await skill.execute(
            {"text": text, "extract_types": ["email"], "include_positions": True},
            context,
        )

        assert result.success is True
        emails = result.data["extractions"]["emails"]
        assert len(emails) > 0
        assert "start" in emails[0]
        assert "end" in emails[0]


# =============================================================================
# SKILLS Registration Tests
# =============================================================================


class TestSkillsRegistration:
    """Tests for SKILLS module-level list."""

    def test_skills_list_exists(self):
        """Test SKILLS list exists in module."""
        from aragora.skills.builtin import data_extraction

        assert hasattr(data_extraction, "SKILLS")

    def test_skills_list_contains_skill(self):
        """Test SKILLS list contains DataExtractionSkill."""
        from aragora.skills.builtin.data_extraction import SKILLS

        assert len(SKILLS) == 1
        assert isinstance(SKILLS[0], DataExtractionSkill)
