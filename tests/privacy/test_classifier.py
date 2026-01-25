"""Tests for privacy sensitivity classifier."""

from __future__ import annotations

import pytest
from datetime import datetime

from aragora.privacy.classifier import (
    SensitivityLevel,
    SensitivityIndicator,
    ClassificationConfig,
    IndicatorMatch,
    ClassificationResult,
    SensitivityClassifier,
    get_classifier,
)


# ============================================================================
# SensitivityLevel Tests
# ============================================================================


class TestSensitivityLevel:
    """Tests for SensitivityLevel enum."""

    def test_enum_values(self):
        """Test all sensitivity levels are defined."""
        assert SensitivityLevel.PUBLIC == "public"
        assert SensitivityLevel.INTERNAL == "internal"
        assert SensitivityLevel.CONFIDENTIAL == "confidential"
        assert SensitivityLevel.RESTRICTED == "restricted"
        assert SensitivityLevel.TOP_SECRET == "top_secret"

    def test_enum_ordering(self):
        """Test enum values can be ordered by index."""
        levels = list(SensitivityLevel)
        # PUBLIC should be first (lowest sensitivity)
        assert levels.index(SensitivityLevel.PUBLIC) < levels.index(SensitivityLevel.INTERNAL)
        assert levels.index(SensitivityLevel.INTERNAL) < levels.index(SensitivityLevel.CONFIDENTIAL)
        assert levels.index(SensitivityLevel.CONFIDENTIAL) < levels.index(
            SensitivityLevel.RESTRICTED
        )
        assert levels.index(SensitivityLevel.RESTRICTED) < levels.index(SensitivityLevel.TOP_SECRET)

    def test_str_inheritance(self):
        """Test SensitivityLevel inherits from str."""
        assert isinstance(SensitivityLevel.PUBLIC, str)
        assert SensitivityLevel.PUBLIC == "public"


# ============================================================================
# SensitivityIndicator Tests
# ============================================================================


class TestSensitivityIndicator:
    """Tests for SensitivityIndicator dataclass."""

    def test_minimal_indicator(self):
        """Test creating indicator with required fields only."""
        indicator = SensitivityIndicator(
            name="test",
            pattern=r"\btest\b",
            level=SensitivityLevel.INTERNAL,
        )
        assert indicator.name == "test"
        assert indicator.pattern == r"\btest\b"
        assert indicator.level == SensitivityLevel.INTERNAL
        assert indicator.confidence == 0.8  # default
        assert indicator.description == ""  # default

    def test_full_indicator(self):
        """Test creating indicator with all fields."""
        indicator = SensitivityIndicator(
            name="secret_key",
            pattern=r"secret_key=[A-Za-z0-9]+",
            level=SensitivityLevel.RESTRICTED,
            confidence=0.95,
            description="Secret key pattern",
        )
        assert indicator.name == "secret_key"
        assert indicator.confidence == 0.95
        assert indicator.description == "Secret key pattern"


# ============================================================================
# ClassificationConfig Tests
# ============================================================================


class TestClassificationConfig:
    """Tests for ClassificationConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = ClassificationConfig()
        assert config.default_level == SensitivityLevel.INTERNAL
        assert config.min_confidence == 0.6
        assert config.use_llm is False
        assert config.llm_model == "claude-sonnet-4-20250514"
        assert config.custom_indicators == []

    def test_custom_config(self):
        """Test custom configuration."""
        custom_indicator = SensitivityIndicator(
            name="custom",
            pattern=r"CUSTOM-\d+",
            level=SensitivityLevel.CONFIDENTIAL,
        )
        config = ClassificationConfig(
            default_level=SensitivityLevel.PUBLIC,
            min_confidence=0.7,
            use_llm=True,
            custom_indicators=[custom_indicator],
        )
        assert config.default_level == SensitivityLevel.PUBLIC
        assert config.min_confidence == 0.7
        assert config.use_llm is True
        assert len(config.custom_indicators) == 1


# ============================================================================
# IndicatorMatch Tests
# ============================================================================


class TestIndicatorMatch:
    """Tests for IndicatorMatch dataclass."""

    def test_match_creation(self):
        """Test creating an indicator match."""
        indicator = SensitivityIndicator(
            name="ssn",
            pattern=r"\d{3}-\d{2}-\d{4}",
            level=SensitivityLevel.CONFIDENTIAL,
        )
        match = IndicatorMatch(
            indicator=indicator,
            match_text="123-45-6789",
            position=50,
            confidence=0.9,
        )
        assert match.indicator == indicator
        assert match.match_text == "123-45-6789"
        assert match.position == 50
        assert match.confidence == 0.9


# ============================================================================
# ClassificationResult Tests
# ============================================================================


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_minimal_result(self):
        """Test minimal classification result."""
        result = ClassificationResult(
            level=SensitivityLevel.INTERNAL,
            confidence=0.75,
        )
        assert result.level == SensitivityLevel.INTERNAL
        assert result.confidence == 0.75
        assert isinstance(result.classified_at, datetime)
        assert result.indicators_found == []
        assert result.pii_detected is False
        assert result.secrets_detected is False
        assert result.document_id == ""
        assert result.content_length == 0
        assert result.classification_method == "rule_based"

    def test_full_result(self):
        """Test classification result with all fields."""
        indicator = SensitivityIndicator(
            name="ssn",
            pattern=r"\d{3}-\d{2}-\d{4}",
            level=SensitivityLevel.CONFIDENTIAL,
        )
        match = IndicatorMatch(
            indicator=indicator,
            match_text="123-45-6789",
            position=0,
            confidence=0.9,
        )
        result = ClassificationResult(
            level=SensitivityLevel.CONFIDENTIAL,
            confidence=0.9,
            indicators_found=[match],
            pii_detected=True,
            secrets_detected=False,
            document_id="doc-123",
            content_length=500,
            classification_method="rule_based",
        )
        assert result.pii_detected is True
        assert result.document_id == "doc-123"
        assert len(result.indicators_found) == 1

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ClassificationResult(
            level=SensitivityLevel.CONFIDENTIAL,
            confidence=0.9,
            pii_detected=True,
            secrets_detected=False,
            document_id="doc-123",
            content_length=500,
        )
        d = result.to_dict()
        assert d["level"] == "confidential"
        assert d["confidence"] == 0.9
        assert d["pii_detected"] is True
        assert d["secrets_detected"] is False
        assert d["indicators_found"] == 0
        assert d["classification_method"] == "rule_based"
        assert "classified_at" in d


# ============================================================================
# SensitivityClassifier Tests
# ============================================================================


class TestSensitivityClassifier:
    """Tests for SensitivityClassifier class."""

    def test_default_initialization(self):
        """Test classifier with default config."""
        classifier = SensitivityClassifier()
        assert classifier.config.default_level == SensitivityLevel.INTERNAL
        # Should have default indicators loaded
        assert len(classifier._indicators) >= 10

    def test_custom_config(self):
        """Test classifier with custom config."""
        custom_indicator = SensitivityIndicator(
            name="company_id",
            pattern=r"ACME-\d{6}",
            level=SensitivityLevel.INTERNAL,
        )
        config = ClassificationConfig(
            default_level=SensitivityLevel.PUBLIC,
            custom_indicators=[custom_indicator],
        )
        classifier = SensitivityClassifier(config)
        assert classifier.config.default_level == SensitivityLevel.PUBLIC
        # Default + 1 custom indicator
        assert len(classifier._indicators) > len(SensitivityClassifier.DEFAULT_INDICATORS)

    def test_pattern_compilation(self):
        """Test regex patterns are compiled."""
        classifier = SensitivityClassifier()
        # All default indicators should have compiled patterns
        for indicator in SensitivityClassifier.DEFAULT_INDICATORS:
            assert indicator.name in classifier._compiled_patterns


class TestClassifyMethod:
    """Tests for the classify() method."""

    @pytest.fixture
    def classifier(self):
        """Create a classifier for tests."""
        return SensitivityClassifier()

    @pytest.mark.asyncio
    async def test_classify_no_matches(self, classifier):
        """Test classification with no sensitive content."""
        result = await classifier.classify("This is a normal public document.")
        assert result.level == SensitivityLevel.INTERNAL  # default
        assert result.confidence == 0.5
        assert result.indicators_found == []
        assert result.pii_detected is False
        assert result.secrets_detected is False

    @pytest.mark.asyncio
    async def test_classify_ssn(self, classifier):
        """Test SSN detection."""
        content = "Customer SSN is 123-45-6789 for records."
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert result.pii_detected is True
        assert any(m.indicator.name == "ssn" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_credit_card(self, classifier):
        """Test credit card detection."""
        content = "Payment with card 4532-1234-5678-9012"
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert result.pii_detected is True
        assert any(m.indicator.name == "credit_card" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_api_key(self, classifier):
        """Test API key detection."""
        content = "Configuration: api_key='sk-1234567890abcdefghijklmnop'"
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.RESTRICTED
        assert result.secrets_detected is True
        assert any(m.indicator.name == "api_keys" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_private_key(self, classifier):
        """Test private key detection."""
        content = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.RESTRICTED
        assert result.secrets_detected is True
        assert any(m.indicator.name == "private_keys" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_password(self, classifier):
        """Test password detection."""
        content = 'Database connection: password="supersecret123"'
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.RESTRICTED
        assert result.secrets_detected is True
        assert any(m.indicator.name == "database_credentials" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_top_secret(self, classifier):
        """Test top secret classification markers."""
        content = "This document is TOP SECRET and contains NOFORN information."
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.TOP_SECRET
        assert result.confidence >= 0.9
        assert any(m.indicator.name == "national_security" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_medical_data(self, classifier):
        """Test medical/HIPAA data detection."""
        content = "Patient diagnosis shows elevated blood pressure. Medical record attached."
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert any(m.indicator.name == "medical_record" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_financial_data(self, classifier):
        """Test financial data detection."""
        content = "Wire transfer to bank account. IBAN: DE89370400440532013000"
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert any(m.indicator.name == "financial_data" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_email_addresses(self, classifier):
        """Test email address detection."""
        content = "Contact us at support@example.com for assistance."
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.INTERNAL
        assert any(m.indicator.name == "email_addresses" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_internal_markers(self, classifier):
        """Test internal use markers."""
        content = "INTERNAL ONLY: This document is confidential and proprietary."
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.INTERNAL
        assert any(m.indicator.name == "internal_only" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_phone_number(self, classifier):
        """Test phone number detection."""
        content = "Call us at (555) 123-4567 or +1-800-555-0199"
        result = await classifier.classify(content)
        assert any(m.indicator.name == "phone_number" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_dob(self, classifier):
        """Test date of birth detection."""
        content = "Date of Birth: 01/15/1990"
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert any(m.indicator.name == "date_of_birth" for m in result.indicators_found)

    @pytest.mark.asyncio
    async def test_classify_highest_level_wins(self, classifier):
        """Test that highest sensitivity level wins when multiple patterns match."""
        content = """
        Customer SSN: 123-45-6789
        API Key: api_key='sk-1234567890abcdefghijklmnop'
        """
        result = await classifier.classify(content)
        # RESTRICTED (api_key) beats CONFIDENTIAL (ssn)
        assert result.level == SensitivityLevel.RESTRICTED
        assert result.pii_detected is True
        assert result.secrets_detected is True

    @pytest.mark.asyncio
    async def test_classify_with_document_id(self, classifier):
        """Test classification stores document ID."""
        result = await classifier.classify(
            "Test content",
            document_id="doc-abc-123",
        )
        assert result.document_id == "doc-abc-123"

    @pytest.mark.asyncio
    async def test_classify_content_length(self, classifier):
        """Test content length is recorded."""
        content = "A" * 500
        result = await classifier.classify(content)
        assert result.content_length == 500

    @pytest.mark.asyncio
    async def test_classify_truncates_match_text(self, classifier):
        """Test that match text is truncated for safety."""
        # Create content with a very long match
        long_email = "a" * 200 + "@example.com"
        content = f"Contact: {long_email}"
        result = await classifier.classify(content)

        # Match text should be truncated to 100 chars
        for match in result.indicators_found:
            assert len(match.match_text) <= 100


class TestClassifyDocument:
    """Tests for classify_document() method."""

    @pytest.fixture
    def classifier(self):
        return SensitivityClassifier()

    @pytest.mark.asyncio
    async def test_classify_document_basic(self, classifier):
        """Test basic document classification."""
        doc = {
            "content": "Customer SSN: 123-45-6789",
            "id": "doc-123",
            "metadata": {"source": "upload"},
        }
        result = await classifier.classify_document(doc)
        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert result.document_id == "doc-123"

    @pytest.mark.asyncio
    async def test_classify_document_missing_fields(self, classifier):
        """Test document with missing optional fields."""
        doc = {"content": "Public information"}
        result = await classifier.classify_document(doc)
        assert result.document_id == ""

    @pytest.mark.asyncio
    async def test_classify_document_empty_content(self, classifier):
        """Test document with empty content."""
        doc = {"id": "doc-456"}
        result = await classifier.classify_document(doc)
        assert result.level == SensitivityLevel.INTERNAL  # default
        assert result.content_length == 0


class TestBatchClassify:
    """Tests for batch_classify() method."""

    @pytest.fixture
    def classifier(self):
        return SensitivityClassifier()

    @pytest.mark.asyncio
    async def test_batch_classify(self, classifier):
        """Test batch classification of multiple documents."""
        docs = [
            {"content": "Public announcement", "id": "doc-1"},
            {"content": "SSN: 123-45-6789", "id": "doc-2"},
            {"content": "api_key='sk-abcdefghij1234567890'", "id": "doc-3"},
        ]
        results = await classifier.batch_classify(docs)

        assert len(results) == 3
        assert results[0].level == SensitivityLevel.INTERNAL  # default
        assert results[1].level == SensitivityLevel.CONFIDENTIAL
        assert results[2].level == SensitivityLevel.RESTRICTED

    @pytest.mark.asyncio
    async def test_batch_classify_empty(self, classifier):
        """Test batch classification with empty list."""
        results = await classifier.batch_classify([])
        assert results == []


class TestIndicatorManagement:
    """Tests for add_indicator and remove_indicator."""

    def test_add_indicator(self):
        """Test adding a custom indicator."""
        classifier = SensitivityClassifier()
        initial_count = len(classifier._indicators)

        new_indicator = SensitivityIndicator(
            name="custom_pattern",
            pattern=r"CUSTOM-[A-Z]{4}-\d{4}",
            level=SensitivityLevel.CONFIDENTIAL,
            confidence=0.85,
        )
        classifier.add_indicator(new_indicator)

        assert len(classifier._indicators) == initial_count + 1
        assert "custom_pattern" in classifier._compiled_patterns

    def test_add_invalid_pattern(self):
        """Test adding indicator with invalid regex pattern."""
        classifier = SensitivityClassifier()

        invalid_indicator = SensitivityIndicator(
            name="invalid",
            pattern=r"[invalid(regex",  # Invalid regex
            level=SensitivityLevel.INTERNAL,
        )
        # Should not raise, just log warning
        classifier.add_indicator(invalid_indicator)

        # Indicator added but pattern not compiled
        assert any(i.name == "invalid" for i in classifier._indicators)
        assert "invalid" not in classifier._compiled_patterns

    def test_remove_indicator(self):
        """Test removing an indicator."""
        classifier = SensitivityClassifier()
        initial_count = len(classifier._indicators)

        # Remove a default indicator
        classifier.remove_indicator("ssn")

        assert len(classifier._indicators) == initial_count - 1
        assert "ssn" not in classifier._compiled_patterns
        assert not any(i.name == "ssn" for i in classifier._indicators)

    def test_remove_nonexistent_indicator(self):
        """Test removing indicator that doesn't exist."""
        classifier = SensitivityClassifier()
        initial_count = len(classifier._indicators)

        # Should not raise
        classifier.remove_indicator("nonexistent")

        assert len(classifier._indicators) == initial_count

    @pytest.mark.asyncio
    async def test_added_indicator_works(self):
        """Test that added indicator is used in classification."""
        classifier = SensitivityClassifier()

        new_indicator = SensitivityIndicator(
            name="project_code",
            pattern=r"PROJECT-ALPHA-\d+",
            level=SensitivityLevel.RESTRICTED,
            confidence=0.95,
        )
        classifier.add_indicator(new_indicator)

        result = await classifier.classify("Reference: PROJECT-ALPHA-12345")
        assert result.level == SensitivityLevel.RESTRICTED
        assert any(m.indicator.name == "project_code" for m in result.indicators_found)


class TestGetLevelPolicy:
    """Tests for get_level_policy() method."""

    @pytest.fixture
    def classifier(self):
        return SensitivityClassifier()

    def test_public_policy(self, classifier):
        """Test PUBLIC level policy."""
        policy = classifier.get_level_policy(SensitivityLevel.PUBLIC)
        assert policy["encryption_required"] is False
        assert policy["access_logging"] is False
        assert policy["retention_days"] is None
        assert policy["sharing_allowed"] is True
        assert policy["export_allowed"] is True

    def test_internal_policy(self, classifier):
        """Test INTERNAL level policy."""
        policy = classifier.get_level_policy(SensitivityLevel.INTERNAL)
        assert policy["encryption_required"] is False
        assert policy["access_logging"] is True
        assert policy["retention_days"] == 365
        assert policy["sharing_allowed"] is True
        assert policy["export_allowed"] is True

    def test_confidential_policy(self, classifier):
        """Test CONFIDENTIAL level policy."""
        policy = classifier.get_level_policy(SensitivityLevel.CONFIDENTIAL)
        assert policy["encryption_required"] is True
        assert policy["access_logging"] is True
        assert policy["retention_days"] == 90
        assert policy["sharing_allowed"] is False
        assert policy["export_allowed"] is False

    def test_restricted_policy(self, classifier):
        """Test RESTRICTED level policy."""
        policy = classifier.get_level_policy(SensitivityLevel.RESTRICTED)
        assert policy["encryption_required"] is True
        assert policy["access_logging"] is True
        assert policy["retention_days"] == 30
        assert policy["sharing_allowed"] is False
        assert policy["export_allowed"] is False
        assert policy["approval_required"] is True

    def test_top_secret_policy(self, classifier):
        """Test TOP_SECRET level policy."""
        policy = classifier.get_level_policy(SensitivityLevel.TOP_SECRET)
        assert policy["encryption_required"] is True
        assert policy["access_logging"] is True
        assert policy["retention_days"] == 7
        assert policy["sharing_allowed"] is False
        assert policy["export_allowed"] is False
        assert policy["approval_required"] is True
        assert policy["mfa_required"] is True


class TestGetClassifier:
    """Tests for get_classifier() global function."""

    def test_get_classifier_default(self):
        """Test getting default classifier."""
        import aragora.privacy.classifier as module

        # Reset global
        module._classifier = None

        classifier = get_classifier()
        assert isinstance(classifier, SensitivityClassifier)

    def test_get_classifier_returns_same_instance(self):
        """Test that get_classifier returns singleton."""
        import aragora.privacy.classifier as module

        # Reset global
        module._classifier = None

        classifier1 = get_classifier()
        classifier2 = get_classifier()
        assert classifier1 is classifier2

    def test_get_classifier_with_config(self):
        """Test get_classifier with custom config on first call."""
        import aragora.privacy.classifier as module

        # Reset global
        module._classifier = None

        config = ClassificationConfig(default_level=SensitivityLevel.PUBLIC)
        classifier = get_classifier(config)
        assert classifier.config.default_level == SensitivityLevel.PUBLIC


class TestMinConfidenceThreshold:
    """Tests for minimum confidence filtering."""

    @pytest.mark.asyncio
    async def test_low_confidence_uses_default(self):
        """Test that low confidence matches fall back to default level."""
        config = ClassificationConfig(min_confidence=0.9)
        classifier = SensitivityClassifier(config)

        # Email addresses have 0.6 confidence, below 0.9 threshold
        result = await classifier.classify("Contact: test@example.com")
        # Should fall back to default since no confident matches
        assert result.level == SensitivityLevel.INTERNAL
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_high_confidence_classified(self):
        """Test that high confidence matches are used."""
        config = ClassificationConfig(min_confidence=0.9)
        classifier = SensitivityClassifier(config)

        # Private keys have 0.95 confidence, above 0.9 threshold
        result = await classifier.classify("-----BEGIN RSA PRIVATE KEY-----")
        assert result.level == SensitivityLevel.RESTRICTED
        assert result.confidence >= 0.9


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def classifier(self):
        return SensitivityClassifier()

    @pytest.mark.asyncio
    async def test_empty_content(self, classifier):
        """Test classification of empty content."""
        result = await classifier.classify("")
        assert result.level == SensitivityLevel.INTERNAL
        assert result.content_length == 0
        assert result.indicators_found == []

    @pytest.mark.asyncio
    async def test_whitespace_only(self, classifier):
        """Test classification of whitespace-only content."""
        result = await classifier.classify("   \n\t\n   ")
        assert result.level == SensitivityLevel.INTERNAL
        assert result.indicators_found == []

    @pytest.mark.asyncio
    async def test_unicode_content(self, classifier):
        """Test classification with unicode content."""
        result = await classifier.classify("客户信息 123-45-6789 保密")
        # Should still detect SSN pattern
        assert result.level == SensitivityLevel.CONFIDENTIAL

    @pytest.mark.asyncio
    async def test_multiple_ssn_matches(self, classifier):
        """Test content with multiple SSN matches."""
        content = "SSN1: 123-45-6789, SSN2: 987-65-4321, SSN3: 111-22-3333"
        result = await classifier.classify(content)

        ssn_matches = [m for m in result.indicators_found if m.indicator.name == "ssn"]
        assert len(ssn_matches) == 3

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, classifier):
        """Test case insensitive pattern matching."""
        result1 = await classifier.classify("API_KEY='test12345678901234567890'")
        result2 = await classifier.classify("api_key='test12345678901234567890'")

        assert result1.secrets_detected is True
        assert result2.secrets_detected is True

    @pytest.mark.asyncio
    async def test_multiline_content(self, classifier):
        """Test classification of multiline content."""
        content = """
        Document Title

        Customer Information:
        SSN: 123-45-6789

        End of document.
        """
        result = await classifier.classify(content)
        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert result.pii_detected is True
