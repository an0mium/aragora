"""Tests for data classification policy enforcement.

Tests cover:
- DataClassification enum levels and ordering
- ClassificationPolicy dataclass and serialization
- PIIDetection and ValidationResult dataclasses
- DataClassifier.classify() rule-based classification
- DataClassifier.get_policy() per-level policy lookup
- DataClassifier.validate_handling() operation validation
- DataClassifier.scan_for_pii() regex-based PII detection
- Default policies (encryption, audit, retention, regions, consent)
"""

from __future__ import annotations

import pytest

from aragora.compliance.data_classification import (
    DEFAULT_POLICIES,
    ClassificationPolicy,
    DataClassification,
    DataClassifier,
    Operation,
    PIIDetection,
    ValidationResult,
)


# =============================================================================
# DataClassification Enum Tests
# =============================================================================


class TestDataClassification:
    """Test DataClassification enum."""

    def test_all_levels_exist(self):
        assert DataClassification.PUBLIC.value == "public"
        assert DataClassification.INTERNAL.value == "internal"
        assert DataClassification.CONFIDENTIAL.value == "confidential"
        assert DataClassification.RESTRICTED.value == "restricted"
        assert DataClassification.PII.value == "pii"

    def test_level_count(self):
        assert len(DataClassification) == 5

    def test_string_enum(self):
        assert str(DataClassification.PUBLIC) == "DataClassification.PUBLIC"
        assert DataClassification("public") is DataClassification.PUBLIC


class TestOperation:
    """Test Operation enum."""

    def test_all_operations_exist(self):
        assert Operation.READ.value == "read"
        assert Operation.WRITE.value == "write"
        assert Operation.EXPORT.value == "export"
        assert Operation.SHARE.value == "share"
        assert Operation.DELETE.value == "delete"
        assert Operation.ARCHIVE.value == "archive"


# =============================================================================
# ClassificationPolicy Tests
# =============================================================================


class TestClassificationPolicy:
    """Test ClassificationPolicy dataclass."""

    def test_creation_with_defaults(self):
        policy = ClassificationPolicy(classification=DataClassification.PUBLIC)
        assert policy.classification == DataClassification.PUBLIC
        assert policy.encryption_required is False
        assert policy.audit_logging is False
        assert policy.retention_days == 365
        assert policy.allowed_regions == []
        assert policy.requires_consent is False
        assert policy.allowed_operations == []

    def test_creation_with_values(self):
        policy = ClassificationPolicy(
            classification=DataClassification.RESTRICTED,
            encryption_required=True,
            audit_logging=True,
            retention_days=30,
            allowed_regions=["us"],
            requires_consent=True,
            allowed_operations=["read", "delete"],
        )
        assert policy.encryption_required is True
        assert policy.retention_days == 30
        assert policy.allowed_regions == ["us"]

    def test_to_dict(self):
        policy = ClassificationPolicy(
            classification=DataClassification.CONFIDENTIAL,
            encryption_required=True,
            audit_logging=True,
            retention_days=180,
        )
        d = policy.to_dict()
        assert d["classification"] == "confidential"
        assert d["encryption_required"] is True
        assert d["audit_logging"] is True
        assert d["retention_days"] == 180

    def test_frozen(self):
        """ClassificationPolicy should be immutable."""
        policy = ClassificationPolicy(classification=DataClassification.PUBLIC)
        with pytest.raises(AttributeError):
            policy.encryption_required = True  # type: ignore[misc]


# =============================================================================
# PIIDetection Tests
# =============================================================================


class TestPIIDetection:
    """Test PIIDetection dataclass."""

    def test_creation(self):
        det = PIIDetection(type="email", start=0, end=15, confidence=0.95)
        assert det.type == "email"
        assert det.start == 0
        assert det.end == 15
        assert det.confidence == 0.95

    def test_default_confidence(self):
        det = PIIDetection(type="phone", start=5, end=17)
        assert det.confidence == 0.9

    def test_to_dict(self):
        det = PIIDetection(type="ssn", start=10, end=21, confidence=0.95)
        d = det.to_dict()
        assert d == {"type": "ssn", "start": 10, "end": 21, "confidence": 0.95}


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_allowed(self):
        result = ValidationResult(allowed=True)
        assert result.allowed is True
        assert result.violations == []
        assert result.recommendations == []

    def test_not_allowed(self):
        result = ValidationResult(
            allowed=False,
            violations=["Encryption required"],
            recommendations=["Use AES-256"],
        )
        assert result.allowed is False
        assert len(result.violations) == 1
        assert len(result.recommendations) == 1

    def test_to_dict(self):
        result = ValidationResult(allowed=True, violations=[], recommendations=["audit"])
        d = result.to_dict()
        assert d["allowed"] is True
        assert d["recommendations"] == ["audit"]


# =============================================================================
# Default Policies Tests
# =============================================================================


class TestDefaultPolicies:
    """Test the DEFAULT_POLICIES mapping."""

    def test_all_levels_have_policies(self):
        for level in DataClassification:
            assert level in DEFAULT_POLICIES

    def test_public_policy(self):
        policy = DEFAULT_POLICIES[DataClassification.PUBLIC]
        assert policy.encryption_required is False
        assert policy.audit_logging is False
        assert policy.requires_consent is False

    def test_pii_policy(self):
        policy = DEFAULT_POLICIES[DataClassification.PII]
        assert policy.encryption_required is True
        assert policy.audit_logging is True
        assert policy.requires_consent is True
        assert policy.retention_days == 90

    def test_confidential_has_region_restrictions(self):
        policy = DEFAULT_POLICIES[DataClassification.CONFIDENTIAL]
        assert len(policy.allowed_regions) > 0

    def test_internal_no_share(self):
        """INTERNAL data should not allow 'share' by default."""
        policy = DEFAULT_POLICIES[DataClassification.INTERNAL]
        assert "share" not in policy.allowed_operations

    def test_restricted_requires_consent(self):
        policy = DEFAULT_POLICIES[DataClassification.RESTRICTED]
        assert policy.requires_consent is True
        assert policy.encryption_required is True


# =============================================================================
# DataClassifier.classify() Tests
# =============================================================================


class TestClassify:
    """Test DataClassifier.classify()."""

    def setup_method(self):
        self.classifier = DataClassifier()

    def test_public_by_default(self):
        """Data with no sensitive markers should be PUBLIC."""
        level = self.classifier.classify({"title": "Hello World"})
        assert level == DataClassification.PUBLIC

    def test_pii_by_field_name(self):
        """A key named 'email' should trigger PII classification."""
        level = self.classifier.classify({"email": "user@example.com"})
        assert level == DataClassification.PII

    def test_pii_by_value_scan(self):
        """PII regex detection in values should elevate to PII."""
        level = self.classifier.classify({"note": "Reach me at user@example.com"})
        assert level == DataClassification.PII

    def test_restricted_by_field_name(self):
        level = self.classifier.classify({"api_key": "sk-1234567890"})
        assert level == DataClassification.RESTRICTED

    def test_confidential_by_field_name(self):
        level = self.classifier.classify({"salary": "120000"})
        assert level == DataClassification.CONFIDENTIAL

    def test_internal_by_context(self):
        level = self.classifier.classify({"data": "quarterly numbers"}, context="internal")
        assert level == DataClassification.INTERNAL

    def test_highest_sensitivity_wins(self):
        """When multiple levels match, the most sensitive should win."""
        level = self.classifier.classify(
            {"salary": "120000", "ssn": "123-45-6789"},
        )
        assert level == DataClassification.PII

    def test_context_triggers_classification(self):
        """Context string should also be checked for keywords."""
        level = self.classifier.classify({"value": "100"}, context="financial report")
        assert level == DataClassification.CONFIDENTIAL

    def test_empty_data_is_public(self):
        level = self.classifier.classify({})
        assert level == DataClassification.PUBLIC

    def test_phone_in_value_triggers_pii(self):
        level = self.classifier.classify({"info": "Call 555-123-4567"})
        assert level == DataClassification.PII

    def test_ssn_in_value_triggers_pii(self):
        level = self.classifier.classify({"notes": "SSN is 123-45-6789"})
        assert level == DataClassification.PII


# =============================================================================
# DataClassifier.get_policy() Tests
# =============================================================================


class TestGetPolicy:
    """Test DataClassifier.get_policy()."""

    def setup_method(self):
        self.classifier = DataClassifier()

    def test_returns_correct_policy_for_each_level(self):
        for level in DataClassification:
            policy = self.classifier.get_policy(level)
            assert policy.classification == level

    def test_custom_policies(self):
        custom = {
            DataClassification.PUBLIC: ClassificationPolicy(
                classification=DataClassification.PUBLIC,
                encryption_required=True,  # stricter than default
            ),
        }
        # Merge with defaults for other levels
        merged = DEFAULT_POLICIES.copy()
        merged.update(custom)
        classifier = DataClassifier(policies=merged)
        assert classifier.get_policy(DataClassification.PUBLIC).encryption_required is True


# =============================================================================
# DataClassifier.validate_handling() Tests
# =============================================================================


class TestValidateHandling:
    """Test DataClassifier.validate_handling()."""

    def setup_method(self):
        self.classifier = DataClassifier()

    def test_public_read_allowed(self):
        result = self.classifier.validate_handling(
            data={"title": "Hello"},
            classification=DataClassification.PUBLIC,
            operation="read",
        )
        assert result.allowed is True
        assert result.violations == []

    def test_internal_share_blocked(self):
        """INTERNAL data should not allow 'share'."""
        result = self.classifier.validate_handling(
            data={"info": "internal doc"},
            classification=DataClassification.INTERNAL,
            operation="share",
        )
        assert result.allowed is False
        assert any("not allowed" in v for v in result.violations)

    def test_restricted_without_encryption(self):
        result = self.classifier.validate_handling(
            data={"secret": "value"},
            classification=DataClassification.RESTRICTED,
            operation="read",
            is_encrypted=False,
        )
        assert result.allowed is False
        assert any("Encryption" in v for v in result.violations)

    def test_restricted_with_encryption_and_consent(self):
        result = self.classifier.validate_handling(
            data={"secret": "value"},
            classification=DataClassification.RESTRICTED,
            operation="read",
            is_encrypted=True,
            has_consent=True,
        )
        assert result.allowed is True

    def test_pii_without_consent(self):
        result = self.classifier.validate_handling(
            data={"email": "user@example.com"},
            classification=DataClassification.PII,
            operation="read",
            is_encrypted=True,
            has_consent=False,
        )
        assert result.allowed is False
        assert any("consent" in v.lower() for v in result.violations)

    def test_confidential_disallowed_region(self):
        result = self.classifier.validate_handling(
            data={"financial": "data"},
            classification=DataClassification.CONFIDENTIAL,
            operation="read",
            is_encrypted=True,
            region="cn",
        )
        assert result.allowed is False
        assert any("Region" in v for v in result.violations)

    def test_confidential_allowed_region(self):
        result = self.classifier.validate_handling(
            data={"financial": "data"},
            classification=DataClassification.CONFIDENTIAL,
            operation="read",
            is_encrypted=True,
            region="us",
        )
        assert result.allowed is True

    def test_no_region_check_when_region_not_provided(self):
        """If no region is given, the region check should be skipped."""
        result = self.classifier.validate_handling(
            data={"financial": "data"},
            classification=DataClassification.CONFIDENTIAL,
            operation="read",
            is_encrypted=True,
        )
        assert result.allowed is True

    def test_audit_logging_recommendation(self):
        """Policies with audit_logging should recommend logging."""
        result = self.classifier.validate_handling(
            data={"info": "doc"},
            classification=DataClassification.INTERNAL,
            operation="read",
        )
        assert any("audit" in r.lower() for r in result.recommendations)

    def test_public_no_audit_recommendation(self):
        """PUBLIC policy has no audit logging, so no audit recommendation."""
        result = self.classifier.validate_handling(
            data={"title": "Hello"},
            classification=DataClassification.PUBLIC,
            operation="read",
        )
        assert not any("audit" in r.lower() for r in result.recommendations)

    def test_export_blocked_for_restricted(self):
        """RESTRICTED does not allow export."""
        result = self.classifier.validate_handling(
            data={"secret": "value"},
            classification=DataClassification.RESTRICTED,
            operation="export",
            is_encrypted=True,
            has_consent=True,
        )
        assert result.allowed is False
        assert any("export" in v.lower() for v in result.violations)

    def test_multiple_violations(self):
        """Multiple policy violations should all be reported."""
        result = self.classifier.validate_handling(
            data={"secret": "value"},
            classification=DataClassification.RESTRICTED,
            operation="export",
            is_encrypted=False,
            has_consent=False,
            region="cn",
        )
        assert result.allowed is False
        # Should have at least: operation, encryption, region, consent
        assert len(result.violations) >= 3


# =============================================================================
# DataClassifier.scan_for_pii() Tests
# =============================================================================


class TestScanForPII:
    """Test DataClassifier.scan_for_pii()."""

    def setup_method(self):
        self.classifier = DataClassifier()

    def test_detect_email(self):
        detections = self.classifier.scan_for_pii("Contact john@example.com for info")
        types = [d.type for d in detections]
        assert "email" in types
        email_det = next(d for d in detections if d.type == "email")
        assert email_det.confidence >= 0.9

    def test_detect_phone(self):
        detections = self.classifier.scan_for_pii("Call 555-123-4567 now")
        types = [d.type for d in detections]
        assert "phone" in types

    def test_detect_ssn(self):
        detections = self.classifier.scan_for_pii("SSN: 123-45-6789")
        types = [d.type for d in detections]
        assert "ssn" in types

    def test_detect_multiple(self):
        text = "Email john@acme.com, SSN 123-45-6789, phone 555-867-5309"
        detections = self.classifier.scan_for_pii(text)
        types = {d.type for d in detections}
        assert "email" in types
        assert "ssn" in types

    def test_no_pii(self):
        detections = self.classifier.scan_for_pii("This is a clean sentence.")
        assert detections == []

    def test_positions_are_correct(self):
        text = "My email is test@example.com ok"
        detections = self.classifier.scan_for_pii(text)
        email_det = next(d for d in detections if d.type == "email")
        assert text[email_det.start : email_det.end] == "test@example.com"

    def test_empty_string(self):
        assert self.classifier.scan_for_pii("") == []

    def test_phone_without_dashes(self):
        detections = self.classifier.scan_for_pii("Call 5551234567 today")
        types = [d.type for d in detections]
        assert "phone" in types

    def test_phone_with_dots(self):
        detections = self.classifier.scan_for_pii("Phone: 555.123.4567")
        types = [d.type for d in detections]
        assert "phone" in types
