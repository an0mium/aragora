"""Tests for data classification policy enforcement.

Tests cover:
- DataClassification enum levels and ordering
- ClassificationPolicy dataclass and serialization
- PIIDetection and ValidationResult dataclasses
- DataClassifier.classify() rule-based classification
- DataClassifier.get_policy() per-level policy lookup
- DataClassifier.validate_handling() operation validation
- DataClassifier.scan_for_pii() regex-based PII detection
- DataClassifier.tag() classification metadata generation
- DataClassifier.get_active_policy() full policy summary
- ClassificationMetadata dataclass and serialization
- EnforcementResult dataclass and serialization
- PolicyEnforcer.enforce_access() cross-context enforcement
- PolicyEnforcer.audit_label() audit log labeling
- PolicyEnforcer.classify_debate_result() debate result enrichment
- PolicyEnforcer.classify_knowledge_item() knowledge item enrichment
- PolicyEnforcer.classify_api_response() API response enrichment
- sensitivity_index() ordering utility
- Default policies (encryption, audit, retention, regions, consent)
"""

from __future__ import annotations

import pytest

from aragora.compliance.data_classification import (
    DEFAULT_POLICIES,
    SENSITIVITY_ORDER,
    ClassificationMetadata,
    ClassificationPolicy,
    DataClassification,
    DataClassifier,
    EnforcementResult,
    Operation,
    PIIDetection,
    PolicyEnforcer,
    ValidationResult,
    sensitivity_index,
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
        level = self.classifier.classify({"api_key": "sk-abcXyz"})
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


# =============================================================================
# Sensitivity Ordering Tests
# =============================================================================


class TestSensitivityOrder:
    """Test sensitivity ordering utilities."""

    def test_order_list(self):
        assert SENSITIVITY_ORDER == list(DataClassification)

    def test_sensitivity_index_public_is_lowest(self):
        assert sensitivity_index(DataClassification.PUBLIC) == 0

    def test_sensitivity_index_pii_is_highest(self):
        assert sensitivity_index(DataClassification.PII) == 4

    def test_ascending_order(self):
        indices = [sensitivity_index(level) for level in DataClassification]
        assert indices == sorted(indices)

    def test_restricted_more_sensitive_than_internal(self):
        assert sensitivity_index(DataClassification.RESTRICTED) > sensitivity_index(
            DataClassification.INTERNAL
        )

    def test_confidential_between_internal_and_restricted(self):
        assert (
            sensitivity_index(DataClassification.INTERNAL)
            < sensitivity_index(DataClassification.CONFIDENTIAL)
            < sensitivity_index(DataClassification.RESTRICTED)
        )


# =============================================================================
# ClassificationMetadata Tests
# =============================================================================


class TestClassificationMetadata:
    """Test ClassificationMetadata dataclass."""

    def test_creation(self):
        meta = ClassificationMetadata(
            classification=DataClassification.INTERNAL,
            label="internal",
            classified_at="2026-01-01T00:00:00+00:00",
            context="test_context",
        )
        assert meta.classification == DataClassification.INTERNAL
        assert meta.label == "internal"
        assert meta.context == "test_context"
        assert meta.pii_detected is False
        assert meta.pii_types == []

    def test_with_pii(self):
        meta = ClassificationMetadata(
            classification=DataClassification.PII,
            label="pii",
            classified_at="2026-01-01T00:00:00+00:00",
            pii_detected=True,
            pii_types=["email", "phone"],
        )
        assert meta.pii_detected is True
        assert "email" in meta.pii_types

    def test_to_dict(self):
        meta = ClassificationMetadata(
            classification=DataClassification.CONFIDENTIAL,
            label="confidential",
            classified_at="2026-01-01T00:00:00+00:00",
            context="finance",
            pii_detected=False,
            pii_types=[],
        )
        d = meta.to_dict()
        assert d["classification"] == "confidential"
        assert d["label"] == "confidential"
        assert d["context"] == "finance"
        assert d["pii_detected"] is False
        assert d["pii_types"] == []
        assert "classified_at" in d


# =============================================================================
# EnforcementResult Tests
# =============================================================================


class TestEnforcementResult:
    """Test EnforcementResult dataclass."""

    def test_allowed(self):
        result = EnforcementResult(
            allowed=True,
            source_classification=DataClassification.INTERNAL,
            target_classification=DataClassification.INTERNAL,
        )
        assert result.allowed is True

    def test_blocked(self):
        result = EnforcementResult(
            allowed=False,
            source_classification=DataClassification.RESTRICTED,
            target_classification=DataClassification.PUBLIC,
            violations=["Cannot flow to lower context"],
        )
        assert result.allowed is False
        assert len(result.violations) == 1

    def test_to_dict(self):
        result = EnforcementResult(
            allowed=False,
            source_classification=DataClassification.RESTRICTED,
            target_classification=DataClassification.PUBLIC,
            violations=["blocked"],
            recommendations=["upgrade context"],
        )
        d = result.to_dict()
        assert d["allowed"] is False
        assert d["source_classification"] == "restricted"
        assert d["target_classification"] == "public"
        assert d["violations"] == ["blocked"]


# =============================================================================
# DataClassifier.tag() Tests
# =============================================================================


class TestTag:
    """Test DataClassifier.tag() metadata generation."""

    def setup_method(self):
        self.classifier = DataClassifier()

    def test_tag_public_data(self):
        meta = self.classifier.tag({"title": "Hello World"})
        assert meta.classification == DataClassification.PUBLIC
        assert meta.label == "public"
        assert meta.pii_detected is False
        assert meta.classified_at  # non-empty timestamp

    def test_tag_pii_data(self):
        meta = self.classifier.tag({"note": "Email me at test@example.com"})
        assert meta.classification == DataClassification.PII
        assert meta.label == "pii"
        assert meta.pii_detected is True
        assert "email" in meta.pii_types

    def test_tag_with_context(self):
        meta = self.classifier.tag({"data": "report"}, context="financial")
        assert meta.classification == DataClassification.CONFIDENTIAL
        assert meta.context == "financial"

    def test_tag_restricted_data(self):
        meta = self.classifier.tag({"api_key": "sk-secret-xxx"})
        assert meta.classification == DataClassification.RESTRICTED
        assert meta.label == "restricted"

    def test_tag_returns_metadata_instance(self):
        meta = self.classifier.tag({"x": "y"})
        assert isinstance(meta, ClassificationMetadata)

    def test_tag_serializable(self):
        meta = self.classifier.tag({"email": "user@test.com"})
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert "classification" in d
        assert "classified_at" in d


# =============================================================================
# DataClassifier.get_active_policy() Tests
# =============================================================================


class TestGetActivePolicy:
    """Test DataClassifier.get_active_policy() summary."""

    def setup_method(self):
        self.classifier = DataClassifier()

    def test_returns_dict(self):
        policy = self.classifier.get_active_policy()
        assert isinstance(policy, dict)

    def test_has_version(self):
        policy = self.classifier.get_active_policy()
        assert "version" in policy
        assert policy["version"] == "1.0"

    def test_has_name_and_description(self):
        policy = self.classifier.get_active_policy()
        assert "name" in policy
        assert "description" in policy
        assert "Aragora" in policy["name"]

    def test_has_all_levels(self):
        policy = self.classifier.get_active_policy()
        assert "levels" in policy
        for level in DataClassification:
            assert level.value in policy["levels"]

    def test_has_policies_for_all_levels(self):
        policy = self.classifier.get_active_policy()
        assert "policies" in policy
        for level in DataClassification:
            assert level.value in policy["policies"]
            level_policy = policy["policies"][level.value]
            assert "encryption_required" in level_policy
            assert "audit_logging" in level_policy
            assert "retention_days" in level_policy

    def test_has_keywords(self):
        policy = self.classifier.get_active_policy()
        assert "keywords" in policy
        assert "pii" in policy["keywords"]
        assert "email" in policy["keywords"]["pii"]

    def test_has_sensitivity_order(self):
        policy = self.classifier.get_active_policy()
        assert "sensitivity_order" in policy
        assert policy["sensitivity_order"][0] == "public"
        assert policy["sensitivity_order"][-1] == "pii"

    def test_custom_policies_reflected(self):
        custom = DEFAULT_POLICIES.copy()
        custom[DataClassification.PUBLIC] = ClassificationPolicy(
            classification=DataClassification.PUBLIC,
            encryption_required=True,
        )
        classifier = DataClassifier(policies=custom)
        policy = classifier.get_active_policy()
        assert policy["policies"]["public"]["encryption_required"] is True


# =============================================================================
# DataClassifier.policies property Tests
# =============================================================================


class TestPoliciesProperty:
    """Test DataClassifier.policies read-only accessor."""

    def test_returns_dict(self):
        classifier = DataClassifier()
        assert isinstance(classifier.policies, dict)

    def test_has_all_levels(self):
        classifier = DataClassifier()
        for level in DataClassification:
            assert level in classifier.policies

    def test_returns_copy(self):
        classifier = DataClassifier()
        p1 = classifier.policies
        p2 = classifier.policies
        assert p1 is not p2  # new dict each time


# =============================================================================
# PolicyEnforcer.enforce_access() Tests
# =============================================================================


class TestPolicyEnforcer:
    """Test PolicyEnforcer cross-context enforcement."""

    def setup_method(self):
        self.enforcer = PolicyEnforcer()

    def test_same_level_allowed(self):
        result = self.enforcer.enforce_access(
            data={"info": "test"},
            source_classification=DataClassification.INTERNAL,
            target_classification=DataClassification.INTERNAL,
        )
        assert result.allowed is True

    def test_higher_target_allowed(self):
        """Data can flow to a more sensitive context."""
        result = self.enforcer.enforce_access(
            data={"info": "test"},
            source_classification=DataClassification.PUBLIC,
            target_classification=DataClassification.CONFIDENTIAL,
        )
        assert result.allowed is True

    def test_restricted_to_public_blocked(self):
        """RESTRICTED data must not flow to PUBLIC context."""
        result = self.enforcer.enforce_access(
            data={"secret": "value"},
            source_classification=DataClassification.RESTRICTED,
            target_classification=DataClassification.PUBLIC,
            is_encrypted=True,
            has_consent=True,
        )
        assert result.allowed is False
        assert any("cannot be exposed" in v.lower() for v in result.violations)

    def test_restricted_to_internal_blocked(self):
        """RESTRICTED data must not flow to INTERNAL context."""
        result = self.enforcer.enforce_access(
            data={"secret": "value"},
            source_classification=DataClassification.RESTRICTED,
            target_classification=DataClassification.INTERNAL,
            is_encrypted=True,
            has_consent=True,
        )
        assert result.allowed is False

    def test_confidential_to_public_blocked(self):
        """CONFIDENTIAL data must not flow to PUBLIC context."""
        result = self.enforcer.enforce_access(
            data={"salary": "100000"},
            source_classification=DataClassification.CONFIDENTIAL,
            target_classification=DataClassification.PUBLIC,
            is_encrypted=True,
        )
        assert result.allowed is False

    def test_pii_to_internal_blocked(self):
        """PII data must not flow to INTERNAL context."""
        result = self.enforcer.enforce_access(
            data={"email": "user@example.com"},
            source_classification=DataClassification.PII,
            target_classification=DataClassification.INTERNAL,
            is_encrypted=True,
            has_consent=True,
        )
        assert result.allowed is False

    def test_internal_to_public_blocked(self):
        """INTERNAL data must not flow to PUBLIC context."""
        result = self.enforcer.enforce_access(
            data={"project": "roadmap"},
            source_classification=DataClassification.INTERNAL,
            target_classification=DataClassification.PUBLIC,
        )
        assert result.allowed is False

    def test_public_to_public_allowed(self):
        result = self.enforcer.enforce_access(
            data={"title": "Hello"},
            source_classification=DataClassification.PUBLIC,
            target_classification=DataClassification.PUBLIC,
        )
        assert result.allowed is True

    def test_enforcement_with_handling_violations(self):
        """Enforcement also checks standard handling rules."""
        result = self.enforcer.enforce_access(
            data={"secret": "value"},
            source_classification=DataClassification.RESTRICTED,
            target_classification=DataClassification.RESTRICTED,
            is_encrypted=False,
            has_consent=False,
        )
        # Same level so no level violation, but encryption + consent missing
        assert result.allowed is False
        assert any("Encryption" in v for v in result.violations)
        assert any("consent" in v.lower() for v in result.violations)

    def test_enforcement_result_type(self):
        result = self.enforcer.enforce_access(
            data={"x": "y"},
            source_classification=DataClassification.PUBLIC,
            target_classification=DataClassification.PUBLIC,
        )
        assert isinstance(result, EnforcementResult)

    def test_recommendations_included(self):
        result = self.enforcer.enforce_access(
            data={"secret": "value"},
            source_classification=DataClassification.RESTRICTED,
            target_classification=DataClassification.PUBLIC,
            is_encrypted=True,
            has_consent=True,
        )
        assert len(result.recommendations) > 0

    def test_custom_classifier(self):
        """PolicyEnforcer should use the provided classifier."""
        classifier = DataClassifier()
        enforcer = PolicyEnforcer(classifier)
        assert enforcer.classifier is classifier

    def test_default_classifier(self):
        """PolicyEnforcer should create a default classifier if none provided."""
        enforcer = PolicyEnforcer()
        assert enforcer.classifier is not None


# =============================================================================
# PolicyEnforcer.audit_label() Tests
# =============================================================================


class TestAuditLabel:
    """Test PolicyEnforcer.audit_label() for audit log labeling."""

    def setup_method(self):
        self.enforcer = PolicyEnforcer()

    def test_public_audit_label(self):
        label = self.enforcer.audit_label({"title": "Hello"})
        assert label["classification"] == "public"
        assert label["label"] == "public"
        assert label["pii_detected"] is False
        assert label["audit_logging_required"] is False
        assert "timestamp" in label

    def test_pii_audit_label(self):
        label = self.enforcer.audit_label({"note": "Email user@test.com"})
        assert label["classification"] == "pii"
        assert label["pii_detected"] is True
        assert "email" in label["pii_types"]
        assert label["audit_logging_required"] is True
        assert label["encryption_required"] is True

    def test_restricted_audit_label(self):
        label = self.enforcer.audit_label({"api_key": "sk-xxx"})
        assert label["classification"] == "restricted"
        assert label["audit_logging_required"] is True
        assert label["encryption_required"] is True

    def test_context_passed_through(self):
        label = self.enforcer.audit_label({"data": "report"}, context="financial")
        assert label["classification"] == "confidential"


# =============================================================================
# PolicyEnforcer.classify_debate_result() Tests
# =============================================================================


class TestClassifyDebateResult:
    """Test PolicyEnforcer.classify_debate_result()."""

    def setup_method(self):
        self.enforcer = PolicyEnforcer()

    def test_debate_result_enriched(self):
        result = {"outcome": "approved", "summary": "No issues found"}
        enriched = self.enforcer.classify_debate_result(result)
        assert "_classification" in enriched
        assert enriched["_classification"]["classification"] == "public"
        # Original fields preserved
        assert enriched["outcome"] == "approved"

    def test_debate_result_with_sensitive_content(self):
        result = {"outcome": "flagged", "summary": "Contains salary data $120k"}
        enriched = self.enforcer.classify_debate_result(result)
        assert enriched["_classification"]["classification"] == "confidential"

    def test_debate_result_with_pii(self):
        result = {"outcome": "approved", "summary": "User test@example.com approved"}
        enriched = self.enforcer.classify_debate_result(result)
        assert enriched["_classification"]["classification"] == "pii"
        assert enriched["_classification"]["pii_detected"] is True

    def test_debate_result_with_list_arguments(self):
        result = {"arguments": ["Point 1", "Internal roadmap details"]}
        enriched = self.enforcer.classify_debate_result(result)
        assert "_classification" in enriched

    def test_debate_result_with_dict_fields(self):
        result = {"reasoning": {"step1": "secret handling"}}
        enriched = self.enforcer.classify_debate_result(result)
        assert "_classification" in enriched

    def test_empty_debate_result(self):
        result = {"id": "123"}
        enriched = self.enforcer.classify_debate_result(result)
        assert "_classification" in enriched


# =============================================================================
# PolicyEnforcer.classify_knowledge_item() Tests
# =============================================================================


class TestClassifyKnowledgeItem:
    """Test PolicyEnforcer.classify_knowledge_item()."""

    def setup_method(self):
        self.enforcer = PolicyEnforcer()

    def test_knowledge_item_enriched(self):
        item = {"title": "Best practices", "content": "Use encryption"}
        enriched = self.enforcer.classify_knowledge_item(item)
        assert "_classification" in enriched
        assert enriched["title"] == "Best practices"

    def test_knowledge_item_with_tags(self):
        item = {"content": "Budget report", "tags": ["financial", "quarterly"]}
        enriched = self.enforcer.classify_knowledge_item(item)
        assert enriched["_classification"]["classification"] == "confidential"

    def test_knowledge_item_with_pii(self):
        item = {"content": "Contact john@example.com for details"}
        enriched = self.enforcer.classify_knowledge_item(item)
        assert enriched["_classification"]["pii_detected"] is True

    def test_knowledge_item_empty(self):
        item = {"id": "abc"}
        enriched = self.enforcer.classify_knowledge_item(item)
        assert "_classification" in enriched


# =============================================================================
# PolicyEnforcer.classify_api_response() Tests
# =============================================================================


class TestClassifyApiResponse:
    """Test PolicyEnforcer.classify_api_response()."""

    def setup_method(self):
        self.enforcer = PolicyEnforcer()

    def test_api_response_enriched(self):
        data = {"status": "ok", "count": 5}
        enriched = self.enforcer.classify_api_response(data)
        assert "_classification" in enriched
        assert enriched["status"] == "ok"

    def test_api_response_with_context(self):
        data = {"report": "Q1 numbers"}
        enriched = self.enforcer.classify_api_response(data, context="financial")
        assert enriched["_classification"]["classification"] == "confidential"

    def test_api_response_with_pii(self):
        data = {"user_email": "user@example.com"}
        enriched = self.enforcer.classify_api_response(data)
        assert enriched["_classification"]["pii_detected"] is True

    def test_api_response_preserves_original(self):
        data = {"key1": "val1", "key2": "val2"}
        enriched = self.enforcer.classify_api_response(data)
        assert enriched["key1"] == "val1"
        assert enriched["key2"] == "val2"
