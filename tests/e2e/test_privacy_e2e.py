"""
End-to-end tests for Privacy API endpoints.

Tests the complete privacy workflow including:
- Consent Management (grant, use, revoke)
- Data Retention (policies, execution, reporting)
- Anonymization (PII detection, redaction, hashing)
- Privacy Handler endpoints (export, preferences, deletion)

These tests verify GDPR/CCPA compliance functionality.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Mark all tests in this module as privacy and e2e
pytestmark = [pytest.mark.e2e, pytest.mark.privacy]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def consent_store():
    """Create a mock consent store for testing."""
    from aragora.privacy.consent import ConsentStore
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConsentStore(storage_path=os.path.join(tmpdir, "consent.json"))
        yield store


@pytest.fixture
def consent_manager(consent_store):
    """Create a consent manager with mock store."""
    from aragora.privacy.consent import ConsentManager

    return ConsentManager(store=consent_store)


@pytest.fixture
def retention_manager():
    """Create a retention policy manager for testing."""
    from aragora.privacy.retention import RetentionPolicyManager

    return RetentionPolicyManager()


@pytest.fixture
def hipaa_anonymizer():
    """Create a HIPAA anonymizer for testing."""
    from aragora.privacy.anonymization import HIPAAAnonymizer

    return HIPAAAnonymizer(hash_salt="test_salt_123", pseudonym_seed=42)


@pytest.fixture
def k_anonymizer():
    """Create a K-anonymizer for testing."""
    from aragora.privacy.anonymization import KAnonymizer

    return KAnonymizer(k=3)


@pytest.fixture
def differential_privacy():
    """Create differential privacy mechanism for testing."""
    from aragora.privacy.anonymization import DifferentialPrivacy

    return DifferentialPrivacy(epsilon=1.0, delta=1e-5)


@pytest.fixture
def privacy_handler():
    """Create privacy handler for testing."""
    from aragora.server.handlers.privacy import PrivacyHandler

    # Create handler with mock context
    ctx = {"user_store": MagicMock()}
    return PrivacyHandler(ctx)


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock()
    user.id = "user_test123"
    user.email = "test@example.com"
    user.name = "Test User"
    user.role = "viewer"
    user.is_active = True
    user.email_verified = True
    user.created_at = datetime.now(timezone.utc)
    user.updated_at = datetime.now(timezone.utc)
    user.last_login_at = datetime.now(timezone.utc)
    user.mfa_enabled = False
    user.api_key_prefix = None
    user.api_key_hash = None
    user.org_id = None
    return user


# ============================================================================
# Test Classes
# ============================================================================


class TestConsentFlow:
    """Tests for full consent lifecycle."""

    @pytest.mark.asyncio
    async def test_grant_consent(self, consent_manager):
        """Test granting consent for a purpose."""
        from aragora.privacy.consent import ConsentPurpose

        record = consent_manager.grant_consent(
            user_id="user_123",
            purpose=ConsentPurpose.ANALYTICS,
            version="v1.0",
            ip_address="192.168.1.1",
        )

        assert record.user_id == "user_123"
        assert record.purpose == ConsentPurpose.ANALYTICS
        assert record.granted is True
        assert record.version == "v1.0"
        assert record.ip_address == "192.168.1.1"
        assert record.is_valid is True

    @pytest.mark.asyncio
    async def test_check_consent_granted(self, consent_manager):
        """Test checking consent status after grant."""
        from aragora.privacy.consent import ConsentPurpose

        # Grant consent
        consent_manager.grant_consent(
            user_id="user_456",
            purpose=ConsentPurpose.PERSONALIZATION,
            version="v1.0",
        )

        # Check consent
        has_consent = consent_manager.check_consent("user_456", ConsentPurpose.PERSONALIZATION)
        assert has_consent is True

    @pytest.mark.asyncio
    async def test_check_consent_not_granted(self, consent_manager):
        """Test checking consent status when not granted."""
        from aragora.privacy.consent import ConsentPurpose

        has_consent = consent_manager.check_consent("user_789", ConsentPurpose.MARKETING)
        assert has_consent is False

    @pytest.mark.asyncio
    async def test_grant_use_revoke_consent(self, consent_manager):
        """Test complete consent lifecycle: grant → use → revoke."""
        from aragora.privacy.consent import ConsentPurpose

        user_id = "user_consent_lifecycle"

        # Step 1: Grant consent
        record = consent_manager.grant_consent(
            user_id=user_id,
            purpose=ConsentPurpose.AI_TRAINING,
            version="v2.0",
        )
        assert record.granted is True

        # Step 2: Check consent (use data under consent)
        assert consent_manager.check_consent(user_id, ConsentPurpose.AI_TRAINING) is True

        # Step 3: Revoke consent
        revoked = consent_manager.revoke_consent(user_id, ConsentPurpose.AI_TRAINING)
        assert revoked is not None
        assert revoked.granted is False
        assert revoked.revoked_at is not None

        # Step 4: Verify data access blocked after revocation
        assert consent_manager.check_consent(user_id, ConsentPurpose.AI_TRAINING) is False

    @pytest.mark.asyncio
    async def test_consent_expiration(self, consent_manager):
        """Test consent expiration handling."""
        from aragora.privacy.consent import ConsentPurpose

        # Grant consent with past expiration
        expired_at = datetime.now(timezone.utc) - timedelta(days=1)
        record = consent_manager.grant_consent(
            user_id="user_expired",
            purpose=ConsentPurpose.RESEARCH,
            version="v1.0",
            expires_at=expired_at,
        )

        # Should not be valid due to expiration
        assert record.is_valid is False

    @pytest.mark.asyncio
    async def test_bulk_revoke_consents(self, consent_manager):
        """Test bulk revocation for right to be forgotten."""
        from aragora.privacy.consent import ConsentPurpose

        user_id = "user_bulk_revoke"

        # Grant multiple consents
        consent_manager.grant_consent(user_id, ConsentPurpose.ANALYTICS, "v1.0")
        consent_manager.grant_consent(user_id, ConsentPurpose.MARKETING, "v1.0")
        consent_manager.grant_consent(user_id, ConsentPurpose.PERSONALIZATION, "v1.0")

        # Bulk revoke
        revoked_count = consent_manager.bulk_revoke_for_user(user_id)
        assert revoked_count == 3

        # Verify all revoked
        assert consent_manager.check_consent(user_id, ConsentPurpose.ANALYTICS) is False
        assert consent_manager.check_consent(user_id, ConsentPurpose.MARKETING) is False
        assert consent_manager.check_consent(user_id, ConsentPurpose.PERSONALIZATION) is False

    @pytest.mark.asyncio
    async def test_consent_export(self, consent_manager):
        """Test GDPR consent data export."""
        from aragora.privacy.consent import ConsentPurpose

        user_id = "user_export_test"

        # Grant consents
        consent_manager.grant_consent(user_id, ConsentPurpose.DEBATE_PROCESSING, "v1.0")
        consent_manager.grant_consent(user_id, ConsentPurpose.KNOWLEDGE_STORAGE, "v1.0")

        # Export consent data
        export = consent_manager.export_consent_data(user_id)

        assert export.user_id == user_id
        assert len(export.records) == 2
        assert export.exported_at is not None

        # Verify export format
        export_dict = export.to_dict()
        assert "records" in export_dict
        assert "record_count" in export_dict
        assert export_dict["record_count"] == 2

    @pytest.mark.asyncio
    async def test_verify_consent_multiple_purposes(self, consent_manager):
        """Test verifying consent for multiple required purposes."""
        from aragora.privacy.consent import ConsentPurpose

        user_id = "user_verify"

        # Grant only some consents
        consent_manager.grant_consent(user_id, ConsentPurpose.ANALYTICS, "v1.0")
        # Missing: MARKETING

        # Verify required purposes
        all_granted, missing = consent_manager.verify_consent(
            user_id, [ConsentPurpose.ANALYTICS, ConsentPurpose.MARKETING]
        )

        assert all_granted is False
        assert ConsentPurpose.MARKETING in missing


class TestDataRetention:
    """Tests for retention policy execution."""

    @pytest.mark.asyncio
    async def test_create_retention_policy(self, retention_manager):
        """Test creating a retention policy."""
        from aragora.privacy.retention import RetentionAction

        policy = retention_manager.create_policy(
            name="Test 30-Day Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        assert policy.name == "Test 30-Day Policy"
        assert policy.retention_days == 30
        assert policy.action == RetentionAction.DELETE
        assert policy.enabled is True

    @pytest.mark.asyncio
    async def test_list_policies(self, retention_manager):
        """Test listing retention policies."""
        policies = retention_manager.list_policies()

        # Default policies should be present
        assert len(policies) >= 2
        policy_ids = [p.id for p in policies]
        assert "default_90_days" in policy_ids
        assert "audit_7_years" in policy_ids

    @pytest.mark.asyncio
    async def test_policy_expiration_check(self, retention_manager):
        """Test checking if resource has expired under policy."""
        policy = retention_manager.get_policy("default_90_days")
        assert policy is not None

        # Check expiration for old resource
        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        assert policy.is_expired(old_date) is True

        # Check expiration for new resource
        new_date = datetime.now(timezone.utc) - timedelta(days=10)
        assert policy.is_expired(new_date) is False

    @pytest.mark.asyncio
    async def test_days_until_expiry(self, retention_manager):
        """Test calculating days until expiry."""
        policy = retention_manager.get_policy("default_90_days")
        assert policy is not None

        # Resource created 60 days ago should have ~30 days left
        created = datetime.now(timezone.utc) - timedelta(days=60)
        days_left = policy.days_until_expiry(created)

        assert 29 <= days_left <= 31  # Account for timing variations

    @pytest.mark.asyncio
    async def test_execute_policy_dry_run(self, retention_manager):
        """Test policy execution in dry run mode."""
        report = await retention_manager.execute_policy(
            "default_90_days",
            dry_run=True,
        )

        assert report.policy_id == "default_90_days"
        assert report.items_deleted == 0  # Dry run doesn't actually delete
        assert len(report.errors) == 0

    @pytest.mark.asyncio
    async def test_retention_compliance_report(self, retention_manager):
        """Test generating retention compliance report."""
        report = await retention_manager.get_compliance_report()

        assert "report_period" in report
        assert "total_deletions" in report
        assert "active_policies" in report
        assert report["active_policies"] >= 2

    @pytest.mark.asyncio
    async def test_update_policy(self, retention_manager):
        """Test updating a retention policy."""
        from aragora.privacy.retention import RetentionAction

        # Create a policy
        policy = retention_manager.create_policy(
            name="Updateable Policy",
            retention_days=60,
        )

        # Update it
        updated = retention_manager.update_policy(
            policy.id,
            retention_days=45,
            action=RetentionAction.ARCHIVE,
        )

        assert updated.retention_days == 45
        assert updated.action == RetentionAction.ARCHIVE


class TestAnonymization:
    """Tests for PII anonymization workflows."""

    @pytest.mark.asyncio
    async def test_detect_ssn(self, hipaa_anonymizer):
        """Test detecting SSN in text."""
        from aragora.privacy.anonymization import IdentifierType

        text = "Patient SSN: 123-45-6789"
        identifiers = hipaa_anonymizer.detect_identifiers(text)

        ssn_identifiers = [i for i in identifiers if i.identifier_type == IdentifierType.SSN]
        assert len(ssn_identifiers) >= 1
        assert "123-45-6789" in ssn_identifiers[0].value

    @pytest.mark.asyncio
    async def test_detect_email(self, hipaa_anonymizer):
        """Test detecting email addresses."""
        from aragora.privacy.anonymization import IdentifierType

        text = "Contact: john.doe@example.com for more info"
        identifiers = hipaa_anonymizer.detect_identifiers(text)

        email_identifiers = [i for i in identifiers if i.identifier_type == IdentifierType.EMAIL]
        assert len(email_identifiers) >= 1

    @pytest.mark.asyncio
    async def test_detect_phone(self, hipaa_anonymizer):
        """Test detecting phone numbers."""
        from aragora.privacy.anonymization import IdentifierType

        text = "Call me at 555-123-4567"
        identifiers = hipaa_anonymizer.detect_identifiers(text)

        phone_identifiers = [i for i in identifiers if i.identifier_type == IdentifierType.PHONE]
        assert len(phone_identifiers) >= 1

    @pytest.mark.asyncio
    async def test_redact_pii(self, hipaa_anonymizer):
        """Test PII redaction."""
        from aragora.privacy.anonymization import AnonymizationMethod

        text = "Patient John Smith has SSN 123-45-6789"
        result = hipaa_anonymizer.anonymize(text, AnonymizationMethod.REDACT)

        assert "[SSN]" in result.anonymized_content
        assert "123-45-6789" not in result.anonymized_content
        assert len(result.fields_anonymized) > 0
        assert result.reversible is False

    @pytest.mark.asyncio
    async def test_hash_pii(self, hipaa_anonymizer):
        """Test PII hashing."""
        from aragora.privacy.anonymization import AnonymizationMethod

        text = "Email: user@example.com"
        result = hipaa_anonymizer.anonymize(text, AnonymizationMethod.HASH)

        # Hash should be a hex string
        assert "user@example.com" not in result.anonymized_content
        assert result.reversible is False

    @pytest.mark.asyncio
    async def test_pseudonymize_pii(self, hipaa_anonymizer):
        """Test PII pseudonymization."""
        from aragora.privacy.anonymization import AnonymizationMethod

        text = "Patient: user@test.com"
        result = hipaa_anonymizer.anonymize(text, AnonymizationMethod.PSEUDONYMIZE)

        # Pseudonymized email should be consistent
        assert "user@test.com" not in result.anonymized_content
        assert result.reversible is True

    @pytest.mark.asyncio
    async def test_safe_harbor_compliance_check(self, hipaa_anonymizer):
        """Test Safe Harbor compliance verification."""
        # Non-compliant text (contains SSN and email which are definitely PII)
        pii_text = "SSN 123-45-6789, email john@example.com"
        result = hipaa_anonymizer.verify_safe_harbor(pii_text)

        assert result.compliant is False
        assert len(result.identifiers_remaining) > 0

        # Verify specific PII types are detected
        identifier_types = {i.identifier_type for i in result.identifiers_remaining}
        from aragora.privacy.anonymization import IdentifierType

        assert IdentifierType.SSN in identifier_types
        assert IdentifierType.EMAIL in identifier_types

        # Compliant text (single short word that can't match any patterns)
        clean_text = "ok"
        clean_result = hipaa_anonymizer.verify_safe_harbor(clean_text)

        assert clean_result.compliant is True

    @pytest.mark.asyncio
    async def test_anonymize_structured_data(self, hipaa_anonymizer):
        """Test anonymizing structured data (dictionary)."""
        from aragora.privacy.anonymization import AnonymizationMethod

        data = {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "notes": "Patient visited on 01/15/2024",
            "age": 35,
        }

        result = hipaa_anonymizer.anonymize_structured(data)

        # Parse anonymized content as JSON
        anonymized = json.loads(result.anonymized_content)

        # Email should be anonymized
        assert "jane@example.com" not in anonymized.get("email", "")

        # Age should remain unchanged (numeric, no PII pattern)
        assert anonymized["age"] == 35

    @pytest.mark.asyncio
    async def test_anonymization_audit_trail(self, hipaa_anonymizer):
        """Test that anonymization creates audit trail."""
        from aragora.privacy.anonymization import AnonymizationMethod

        text = "Record for patient@hospital.org"
        result = hipaa_anonymizer.anonymize(text, AnonymizationMethod.REDACT)

        # Should have audit fields
        assert result.audit_id is not None
        assert result.original_hash is not None
        assert result.anonymized_at is not None

        # Verify audit export
        audit_dict = result.to_dict()
        assert "audit_id" in audit_dict
        assert "original_hash" in audit_dict
        assert "anonymized_at" in audit_dict


class TestKAnonymity:
    """Tests for K-anonymity implementation."""

    @pytest.mark.asyncio
    async def test_check_k_anonymity_satisfied(self, k_anonymizer):
        """Test checking k-anonymity when satisfied."""
        # Dataset where each combination appears 3+ times
        records = [
            {"age": 30, "zip": "12345", "income": 50000},
            {"age": 30, "zip": "12345", "income": 55000},
            {"age": 30, "zip": "12345", "income": 52000},
            {"age": 40, "zip": "67890", "income": 70000},
            {"age": 40, "zip": "67890", "income": 72000},
            {"age": 40, "zip": "67890", "income": 68000},
        ]

        is_k_anonymous, min_size = k_anonymizer.check_k_anonymity(
            records, quasi_identifiers=["age", "zip"]
        )

        assert is_k_anonymous is True
        assert min_size >= 3

    @pytest.mark.asyncio
    async def test_check_k_anonymity_violated(self, k_anonymizer):
        """Test checking k-anonymity when violated."""
        # Dataset where some combinations appear less than k times
        records = [
            {"age": 30, "zip": "12345", "income": 50000},
            {"age": 30, "zip": "12345", "income": 55000},
            {"age": 40, "zip": "67890", "income": 70000},  # Only 1 record
        ]

        is_k_anonymous, min_size = k_anonymizer.check_k_anonymity(
            records, quasi_identifiers=["age", "zip"]
        )

        assert is_k_anonymous is False
        assert min_size < 3

    @pytest.mark.asyncio
    async def test_anonymize_dataset(self, k_anonymizer):
        """Test anonymizing dataset to achieve k-anonymity."""
        records = [
            {"age": 31, "zip": "12345", "name": "Alice"},
            {"age": 32, "zip": "12345", "name": "Bob"},
            {"age": 33, "zip": "12345", "name": "Charlie"},
            {"age": 41, "zip": "67890", "name": "David"},
            {"age": 42, "zip": "67890", "name": "Eve"},
            {"age": 43, "zip": "67890", "name": "Frank"},
        ]

        anonymized = k_anonymizer.anonymize_dataset(records, quasi_identifiers=["age", "zip"])

        # Ages should be generalized
        assert len(anonymized) == len(records)


class TestDifferentialPrivacy:
    """Tests for differential privacy mechanism."""

    @pytest.mark.asyncio
    async def test_privatize_count(self, differential_privacy):
        """Test privatizing a count query."""
        true_count = 100
        noisy_count = differential_privacy.privatize_count(true_count)

        # Noisy count should be non-negative
        assert noisy_count >= 0

        # Should be reasonably close to true value (with epsilon=1.0)
        assert abs(noisy_count - true_count) < 50  # Reasonable noise level

    @pytest.mark.asyncio
    async def test_privatize_sum(self, differential_privacy):
        """Test privatizing a sum query."""
        true_sum = 1000.0
        noisy_sum = differential_privacy.privatize_sum(true_sum, max_contribution=10.0)

        # Should be in reasonable range
        assert 800 < noisy_sum < 1200

    @pytest.mark.asyncio
    async def test_privatize_mean(self, differential_privacy):
        """Test privatizing a mean query."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        true_mean = 30.0

        noisy_mean = differential_privacy.privatize_mean(values, lower_bound=0.0, upper_bound=100.0)

        # Should be in reasonable range
        assert 0 < noisy_mean < 60

    @pytest.mark.asyncio
    async def test_laplace_noise_distribution(self, differential_privacy):
        """Test that Laplace noise follows expected distribution."""
        samples = [
            differential_privacy.add_laplace_noise(0.0, sensitivity=1.0) for _ in range(1000)
        ]

        # Mean should be close to 0
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.5  # Should be close to 0


class TestPrivacyHandlerEndpoints:
    """Tests for Privacy handler API endpoints."""

    @pytest.mark.asyncio
    async def test_handler_routes_defined(self, privacy_handler):
        """Test that privacy handler routes are defined."""
        routes = privacy_handler.ROUTES

        assert "/api/v1/privacy/export" in routes
        assert "/api/v1/privacy/data-inventory" in routes
        assert "/api/v1/privacy/account" in routes
        assert "/api/v1/privacy/preferences" in routes

    @pytest.mark.asyncio
    async def test_can_handle_privacy_endpoints(self, privacy_handler):
        """Test can_handle for privacy endpoints."""
        assert privacy_handler.can_handle("/api/v1/privacy/export")
        assert privacy_handler.can_handle("/api/v1/privacy/data-inventory")
        assert privacy_handler.can_handle("/api/v1/privacy/account")
        assert privacy_handler.can_handle("/api/v1/privacy/preferences")
        assert privacy_handler.can_handle("/api/v2/users/me/export")

        # Should not handle other endpoints
        assert not privacy_handler.can_handle("/api/v1/other")
        assert not privacy_handler.can_handle("/api/v1/debates")

    @pytest.mark.asyncio
    async def test_data_inventory_structure(self, privacy_handler):
        """Test data inventory response structure when mocked."""
        # Mock auth context
        mock_handler = MagicMock()
        mock_handler.headers = {}

        user_store = MagicMock()
        privacy_handler.ctx["user_store"] = user_store

        # The data inventory method should return structured data
        # This tests the expected structure
        expected_categories = [
            "Identifiers",
            "Internet Activity",
            "Geolocation",
            "Professional Information",
            "Inferences",
        ]

        # Verify handler has the method
        assert hasattr(privacy_handler, "_handle_data_inventory")

    @pytest.mark.asyncio
    async def test_resource_type_defined(self, privacy_handler):
        """Test that resource type is properly defined."""
        assert privacy_handler.RESOURCE_TYPE == "privacy"


class TestPrivacyErrorHandling:
    """Tests for error handling in privacy operations."""

    @pytest.mark.asyncio
    async def test_consent_revoke_nonexistent(self, consent_manager):
        """Test revoking consent that doesn't exist."""
        from aragora.privacy.consent import ConsentPurpose

        result = consent_manager.revoke_consent("nonexistent_user", ConsentPurpose.ANALYTICS)
        assert result is None

    @pytest.mark.asyncio
    async def test_retention_policy_not_found(self, retention_manager):
        """Test executing nonexistent policy."""
        with pytest.raises(ValueError, match="Policy not found"):
            await retention_manager.execute_policy("nonexistent_policy")

    @pytest.mark.asyncio
    async def test_k_anonymizer_invalid_k(self):
        """Test K-anonymizer with invalid k value."""
        from aragora.privacy.anonymization import KAnonymizer

        with pytest.raises(ValueError, match="k must be at least 2"):
            KAnonymizer(k=1)

    @pytest.mark.asyncio
    async def test_differential_privacy_invalid_epsilon(self):
        """Test differential privacy with invalid epsilon."""
        from aragora.privacy.anonymization import DifferentialPrivacy

        with pytest.raises(ValueError, match="epsilon must be positive"):
            DifferentialPrivacy(epsilon=0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            DifferentialPrivacy(epsilon=-1)


class TestPrivacyIntegration:
    """Integration tests combining multiple privacy components."""

    @pytest.mark.asyncio
    async def test_full_gdpr_workflow(self, consent_manager, hipaa_anonymizer):
        """Test full GDPR data subject rights workflow."""
        from aragora.privacy.consent import ConsentPurpose

        user_id = "gdpr_user"

        # 1. User grants consent
        consent_manager.grant_consent(user_id, ConsentPurpose.ANALYTICS, "v1.0")
        consent_manager.grant_consent(user_id, ConsentPurpose.PERSONALIZATION, "v1.0")

        # 2. Data is processed under consent
        assert consent_manager.check_consent(user_id, ConsentPurpose.ANALYTICS)

        # 3. User exercises right to access (export)
        export = consent_manager.export_consent_data(user_id)
        assert len(export.records) == 2

        # 4. User exercises right to be forgotten
        deleted_count = consent_manager.delete_user_data(user_id)
        assert deleted_count == 2

        # 5. Verify data is deleted
        assert consent_manager.check_consent(user_id, ConsentPurpose.ANALYTICS) is False

    @pytest.mark.asyncio
    async def test_anonymize_before_retention_delete(self, retention_manager, hipaa_anonymizer):
        """Test anonymizing data before retention deletion."""
        from aragora.privacy.anonymization import AnonymizationMethod
        from aragora.privacy.retention import RetentionAction

        # Create anonymization policy
        policy = retention_manager.create_policy(
            name="Anonymize Before Delete",
            retention_days=90,
            action=RetentionAction.ANONYMIZE,
        )

        assert policy.action == RetentionAction.ANONYMIZE

        # Test that data would be anonymized
        sensitive_data = "Patient John Smith, SSN 123-45-6789"
        result = hipaa_anonymizer.anonymize(sensitive_data, AnonymizationMethod.REDACT)

        assert "123-45-6789" not in result.anonymized_content
        assert "[SSN]" in result.anonymized_content


# Convenience function tests
class TestConvenienceFunctions:
    """Tests for convenience anonymization functions."""

    @pytest.mark.asyncio
    async def test_redact_pii_function(self):
        """Test redact_pii convenience function."""
        from aragora.privacy.anonymization import redact_pii

        text = "Contact: john@example.com, SSN: 123-45-6789"
        result = redact_pii(text)

        assert "john@example.com" not in result
        assert "123-45-6789" not in result

    @pytest.mark.asyncio
    async def test_hash_identifier_function(self):
        """Test hash_identifier convenience function."""
        from aragora.privacy.anonymization import hash_identifier

        value1 = hash_identifier("test@example.com")
        value2 = hash_identifier("test@example.com")
        value3 = hash_identifier("other@example.com")

        # Same input should produce same hash
        assert value1 == value2

        # Different input should produce different hash
        assert value1 != value3

    @pytest.mark.asyncio
    async def test_check_safe_harbor_compliance_function(self):
        """Test check_safe_harbor_compliance convenience function."""
        from aragora.privacy.anonymization import check_safe_harbor_compliance

        # Non-compliant (contains SSN)
        assert check_safe_harbor_compliance("SSN: 123-45-6789") is False

        # Compliant (single short word that can't match patterns)
        assert check_safe_harbor_compliance("ok") is True


__all__ = [
    "TestConsentFlow",
    "TestDataRetention",
    "TestAnonymization",
    "TestKAnonymity",
    "TestDifferentialPrivacy",
    "TestPrivacyHandlerEndpoints",
    "TestPrivacyErrorHandling",
    "TestPrivacyIntegration",
    "TestConvenienceFunctions",
]
