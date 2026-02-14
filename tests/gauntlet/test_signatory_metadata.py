"""Tests for receipt signatory metadata functionality."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aragora.gauntlet.signing import (
    HMACSigner,
    ReceiptSigner,
    SignatoryInfo,
    SignatureMetadata,
    SignedReceipt,
    sign_receipt,
)


@pytest.fixture(autouse=True)
def _reset_default_signer():
    """Reset the module-level _default_signer singleton before and after each test.

    The signing module caches a ReceiptSigner in _default_signer. Without this
    reset, a signer created by one test (with its own ephemeral HMAC key) leaks
    into subsequent tests, causing sign-then-verify mismatches when tests run
    in different orders during broader test sweeps.
    """
    from aragora.gauntlet import signing

    signing._default_signer = None
    yield
    signing._default_signer = None


class TestSignatoryInfo:
    """Tests for SignatoryInfo dataclass."""

    def test_create_signatory_info_required_fields(self):
        """Test creating SignatoryInfo with required fields."""
        signatory = SignatoryInfo(
            name="John Smith",
            email="john.smith@example.com",
        )

        assert signatory.name == "John Smith"
        assert signatory.email == "john.smith@example.com"
        assert signatory.title is None
        assert signatory.organization is None
        assert signatory.role is None
        assert signatory.department is None

    def test_create_signatory_info_all_fields(self):
        """Test creating SignatoryInfo with all fields."""
        signatory = SignatoryInfo(
            name="Jane Doe",
            email="jane.doe@acme.com",
            title="Chief Security Officer",
            organization="ACME Corp",
            role="Security Lead",
            department="Information Security",
        )

        assert signatory.name == "Jane Doe"
        assert signatory.email == "jane.doe@acme.com"
        assert signatory.title == "Chief Security Officer"
        assert signatory.organization == "ACME Corp"
        assert signatory.role == "Security Lead"
        assert signatory.department == "Information Security"

    def test_signatory_to_dict(self):
        """Test SignatoryInfo serialization to dict."""
        signatory = SignatoryInfo(
            name="John Smith",
            email="john@example.com",
            title="Architect",
            organization="Tech Inc",
            role="Approver",
            department="Engineering",
        )

        data = signatory.to_dict()

        assert data["name"] == "John Smith"
        assert data["email"] == "john@example.com"
        assert data["title"] == "Architect"
        assert data["organization"] == "Tech Inc"
        assert data["role"] == "Approver"
        assert data["department"] == "Engineering"

    def test_signatory_from_dict(self):
        """Test SignatoryInfo deserialization from dict."""
        data = {
            "name": "Alice Wilson",
            "email": "alice@company.com",
            "title": "Senior Engineer",
            "organization": "Company LLC",
            "role": "Technical Reviewer",
            "department": "Platform",
        }

        signatory = SignatoryInfo.from_dict(data)

        assert signatory.name == "Alice Wilson"
        assert signatory.email == "alice@company.com"
        assert signatory.title == "Senior Engineer"
        assert signatory.organization == "Company LLC"
        assert signatory.role == "Technical Reviewer"
        assert signatory.department == "Platform"

    def test_signatory_from_dict_partial(self):
        """Test SignatoryInfo deserialization with only required fields."""
        data = {
            "name": "Bob Jones",
            "email": "bob@example.com",
        }

        signatory = SignatoryInfo.from_dict(data)

        assert signatory.name == "Bob Jones"
        assert signatory.email == "bob@example.com"
        assert signatory.title is None
        assert signatory.organization is None

    def test_signatory_roundtrip(self):
        """Test SignatoryInfo serialization roundtrip."""
        original = SignatoryInfo(
            name="Test User",
            email="test@test.com",
            title="Manager",
            organization="TestCo",
            role="Decision Maker",
            department="Operations",
        )

        data = original.to_dict()
        restored = SignatoryInfo.from_dict(data)

        assert restored.name == original.name
        assert restored.email == original.email
        assert restored.title == original.title
        assert restored.organization == original.organization
        assert restored.role == original.role
        assert restored.department == original.department


class TestSignatureMetadataWithSignatory:
    """Tests for SignatureMetadata with signatory info."""

    def test_metadata_without_signatory(self):
        """Test SignatureMetadata without signatory."""
        metadata = SignatureMetadata(
            algorithm="HMAC-SHA256",
            timestamp=datetime.now(timezone.utc).isoformat(),
            key_id="hmac-test",
        )

        assert metadata.signatory is None
        data = metadata.to_dict()
        assert "signatory" not in data

    def test_metadata_with_signatory(self):
        """Test SignatureMetadata with signatory included."""
        signatory = SignatoryInfo(
            name="John Smith",
            email="john@example.com",
            title="Security Officer",
        )

        metadata = SignatureMetadata(
            algorithm="HMAC-SHA256",
            timestamp=datetime.now(timezone.utc).isoformat(),
            key_id="hmac-test",
            signatory=signatory,
        )

        assert metadata.signatory is not None
        assert metadata.signatory.name == "John Smith"

    def test_metadata_to_dict_with_signatory(self):
        """Test SignatureMetadata serialization includes signatory."""
        signatory = SignatoryInfo(
            name="Jane Doe",
            email="jane@example.com",
            role="Approver",
        )

        metadata = SignatureMetadata(
            algorithm="RSA-SHA256",
            timestamp="2026-01-15T10:30:00Z",
            key_id="rsa-abc123",
            signatory=signatory,
        )

        data = metadata.to_dict()

        assert data["algorithm"] == "RSA-SHA256"
        assert data["signatory"]["name"] == "Jane Doe"
        assert data["signatory"]["email"] == "jane@example.com"
        assert data["signatory"]["role"] == "Approver"

    def test_metadata_from_dict_with_signatory(self):
        """Test SignatureMetadata deserialization with signatory."""
        data = {
            "algorithm": "Ed25519",
            "timestamp": "2026-01-15T10:30:00Z",
            "key_id": "ed25519-xyz",
            "version": "1.0",
            "signatory": {
                "name": "Alice Smith",
                "email": "alice@corp.com",
                "title": "CTO",
                "organization": "Corp Inc",
            },
        }

        metadata = SignatureMetadata.from_dict(data)

        assert metadata.algorithm == "Ed25519"
        assert metadata.signatory is not None
        assert metadata.signatory.name == "Alice Smith"
        assert metadata.signatory.title == "CTO"

    def test_metadata_roundtrip_with_signatory(self):
        """Test SignatureMetadata roundtrip with signatory."""
        signatory = SignatoryInfo(
            name="Test User",
            email="test@test.com",
            title="Lead",
            organization="Test Org",
            role="Reviewer",
            department="QA",
        )

        original = SignatureMetadata(
            algorithm="HMAC-SHA256",
            timestamp="2026-02-01T12:00:00Z",
            key_id="hmac-test123",
            version="1.0",
            signatory=signatory,
        )

        data = original.to_dict()
        restored = SignatureMetadata.from_dict(data)

        assert restored.algorithm == original.algorithm
        assert restored.signatory is not None
        assert restored.signatory.name == signatory.name
        assert restored.signatory.email == signatory.email
        assert restored.signatory.role == signatory.role


class TestReceiptSignerWithSignatory:
    """Tests for ReceiptSigner with signatory info."""

    @pytest.fixture
    def signer(self):
        """Create a test signer with known key."""
        return ReceiptSigner(backend=HMACSigner(key_id="test-key"))

    @pytest.fixture
    def sample_receipt(self):
        """Create a sample receipt for testing."""
        return {
            "decision_id": "dec-123",
            "verdict": "APPROVED",
            "confidence": 0.95,
            "rationale": "Test decision",
            "timestamp": "2026-02-01T10:00:00Z",
        }

    @pytest.fixture
    def sample_signatory(self):
        """Create a sample signatory for testing."""
        return SignatoryInfo(
            name="John Smith",
            email="john.smith@company.com",
            title="Senior Architect",
            organization="Tech Company",
            role="Technical Approver",
            department="Engineering",
        )

    def test_sign_without_signatory(self, signer, sample_receipt):
        """Test signing without signatory produces valid signature."""
        signed = signer.sign(sample_receipt)

        assert signed.signature is not None
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"
        assert signed.signature_metadata.signatory is None
        assert signer.verify(signed) is True

    def test_sign_with_signatory(self, signer, sample_receipt, sample_signatory):
        """Test signing with signatory includes signatory in metadata."""
        signed = signer.sign(sample_receipt, signatory=sample_signatory)

        assert signed.signature is not None
        assert signed.signature_metadata.signatory is not None
        assert signed.signature_metadata.signatory.name == "John Smith"
        assert signed.signature_metadata.signatory.email == "john.smith@company.com"
        assert signed.signature_metadata.signatory.role == "Technical Approver"

    def test_sign_with_signatory_verifies(self, signer, sample_receipt, sample_signatory):
        """Test signed receipt with signatory can be verified."""
        signed = signer.sign(sample_receipt, signatory=sample_signatory)

        # Signatory info should not affect signature verification
        assert signer.verify(signed) is True

    def test_signatory_preserved_in_json(self, signer, sample_receipt, sample_signatory):
        """Test signatory info is preserved in JSON serialization."""
        signed = signer.sign(sample_receipt, signatory=sample_signatory)

        json_str = signed.to_json()
        data = json.loads(json_str)

        assert "signature_metadata" in data
        assert "signatory" in data["signature_metadata"]
        assert data["signature_metadata"]["signatory"]["name"] == "John Smith"
        assert data["signature_metadata"]["signatory"]["email"] == "john.smith@company.com"
        assert data["signature_metadata"]["signatory"]["title"] == "Senior Architect"

    def test_signatory_preserved_in_roundtrip(self, signer, sample_receipt, sample_signatory):
        """Test signatory info survives JSON roundtrip."""
        signed = signer.sign(sample_receipt, signatory=sample_signatory)

        json_str = signed.to_json()
        restored = SignedReceipt.from_json(json_str)

        assert restored.signature_metadata.signatory is not None
        assert restored.signature_metadata.signatory.name == sample_signatory.name
        assert restored.signature_metadata.signatory.email == sample_signatory.email
        assert restored.signature_metadata.signatory.title == sample_signatory.title
        assert restored.signature_metadata.signatory.organization == sample_signatory.organization
        assert restored.signature_metadata.signatory.role == sample_signatory.role
        assert restored.signature_metadata.signatory.department == sample_signatory.department


class TestSignReceiptFunction:
    """Tests for the sign_receipt convenience function."""

    def test_sign_receipt_without_signatory(self):
        """Test sign_receipt function without signatory."""
        receipt_data = {"decision_id": "test-1", "verdict": "APPROVED"}
        signed = sign_receipt(receipt_data)

        assert signed.signature is not None
        assert signed.signature_metadata.signatory is None

    def test_sign_receipt_with_signatory(self):
        """Test sign_receipt function with signatory."""
        signatory = SignatoryInfo(
            name="Test User",
            email="test@example.com",
            role="Approver",
        )

        receipt_data = {"decision_id": "test-2", "verdict": "REJECTED"}
        signed = sign_receipt(receipt_data, signatory=signatory)

        assert signed.signature is not None
        assert signed.signature_metadata.signatory is not None
        assert signed.signature_metadata.signatory.name == "Test User"
        assert signed.signature_metadata.signatory.role == "Approver"
