"""
Tests for Gauntlet Cryptographic Signing.

Tests the signing module that provides:
- HMAC-SHA256 symmetric signing
- RSA-SHA256 asymmetric signing
- Ed25519 modern signing
- Receipt signing and verification
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from unittest.mock import patch
import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def sample_receipt_data():
    """Create sample receipt data for signing tests."""
    return {
        "gauntlet_id": "gauntlet-test-123",
        "verdict": "pass",
        "confidence": 0.95,
        "findings": [{"id": "f1", "severity": "low", "title": "Minor issue"}],
        "timestamp": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def hmac_signer():
    """Create an HMAC signer for testing."""
    from aragora.gauntlet.signing import HMACSigner

    return HMACSigner(secret_key=b"test-secret-key-32-bytes-long!!")


# ===========================================================================
# Tests: SignatureMetadata Dataclass
# ===========================================================================


class TestSignatureMetadata:
    """Tests for SignatureMetadata dataclass."""

    def test_creation(self):
        """Test SignatureMetadata creation."""
        from aragora.gauntlet.signing import SignatureMetadata

        meta = SignatureMetadata(
            algorithm="HMAC-SHA256",
            timestamp="2024-01-15T10:30:00Z",
            key_id="test-key-1",
        )

        assert meta.algorithm == "HMAC-SHA256"
        assert meta.timestamp == "2024-01-15T10:30:00Z"
        assert meta.key_id == "test-key-1"
        assert meta.version == "1.0"

    def test_to_dict(self):
        """Test SignatureMetadata.to_dict."""
        from aragora.gauntlet.signing import SignatureMetadata

        meta = SignatureMetadata(
            algorithm="RSA-SHA256",
            timestamp="2024-01-15T10:30:00Z",
            key_id="rsa-key",
            version="1.1",
        )

        result = meta.to_dict()

        assert result == {
            "algorithm": "RSA-SHA256",
            "timestamp": "2024-01-15T10:30:00Z",
            "key_id": "rsa-key",
            "version": "1.1",
        }

    def test_from_dict(self):
        """Test SignatureMetadata.from_dict."""
        from aragora.gauntlet.signing import SignatureMetadata

        data = {
            "algorithm": "Ed25519",
            "timestamp": "2024-01-15T10:30:00Z",
            "key_id": "ed-key",
            "version": "1.0",
        }

        meta = SignatureMetadata.from_dict(data)

        assert meta.algorithm == "Ed25519"
        assert meta.key_id == "ed-key"

    def test_from_dict_default_version(self):
        """Test SignatureMetadata.from_dict with missing version."""
        from aragora.gauntlet.signing import SignatureMetadata

        data = {
            "algorithm": "HMAC-SHA256",
            "timestamp": "2024-01-15T10:30:00Z",
            "key_id": "test-key",
        }

        meta = SignatureMetadata.from_dict(data)

        assert meta.version == "1.0"


# ===========================================================================
# Tests: SignedReceipt Dataclass
# ===========================================================================


class TestSignedReceipt:
    """Tests for SignedReceipt dataclass."""

    def test_creation(self, sample_receipt_data):
        """Test SignedReceipt creation."""
        from aragora.gauntlet.signing import SignatureMetadata, SignedReceipt

        meta = SignatureMetadata(
            algorithm="HMAC-SHA256",
            timestamp="2024-01-15T10:30:00Z",
            key_id="test-key",
        )

        signed = SignedReceipt(
            receipt_data=sample_receipt_data,
            signature="base64signature==",
            signature_metadata=meta,
        )

        assert signed.receipt_data == sample_receipt_data
        assert signed.signature == "base64signature=="

    def test_to_dict(self, sample_receipt_data):
        """Test SignedReceipt.to_dict."""
        from aragora.gauntlet.signing import SignatureMetadata, SignedReceipt

        meta = SignatureMetadata(
            algorithm="HMAC-SHA256",
            timestamp="2024-01-15T10:30:00Z",
            key_id="test-key",
        )

        signed = SignedReceipt(
            receipt_data=sample_receipt_data,
            signature="sig123",
            signature_metadata=meta,
        )

        result = signed.to_dict()

        assert "receipt" in result
        assert "signature" in result
        assert "signature_metadata" in result
        assert result["receipt"] == sample_receipt_data

    def test_to_json(self, sample_receipt_data):
        """Test SignedReceipt.to_json."""
        from aragora.gauntlet.signing import SignatureMetadata, SignedReceipt

        meta = SignatureMetadata(
            algorithm="HMAC-SHA256",
            timestamp="2024-01-15T10:30:00Z",
            key_id="test-key",
        )

        signed = SignedReceipt(
            receipt_data=sample_receipt_data,
            signature="sig123",
            signature_metadata=meta,
        )

        json_str = signed.to_json()
        parsed = json.loads(json_str)

        assert "receipt" in parsed
        assert parsed["signature"] == "sig123"

    def test_from_dict(self, sample_receipt_data):
        """Test SignedReceipt.from_dict."""
        from aragora.gauntlet.signing import SignedReceipt

        data = {
            "receipt": sample_receipt_data,
            "signature": "testsig",
            "signature_metadata": {
                "algorithm": "HMAC-SHA256",
                "timestamp": "2024-01-15T10:30:00Z",
                "key_id": "test-key",
            },
        }

        signed = SignedReceipt.from_dict(data)

        assert signed.receipt_data == sample_receipt_data
        assert signed.signature == "testsig"
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"

    def test_from_json(self, sample_receipt_data):
        """Test SignedReceipt.from_json."""
        from aragora.gauntlet.signing import SignedReceipt

        data = {
            "receipt": sample_receipt_data,
            "signature": "testsig",
            "signature_metadata": {
                "algorithm": "HMAC-SHA256",
                "timestamp": "2024-01-15T10:30:00Z",
                "key_id": "test-key",
            },
        }

        json_str = json.dumps(data)
        signed = SignedReceipt.from_json(json_str)

        assert signed.signature == "testsig"


# ===========================================================================
# Tests: HMACSigner Backend
# ===========================================================================


class TestHMACSigner:
    """Tests for HMACSigner backend."""

    def test_creation_with_key(self):
        """Test HMACSigner creation with provided key."""
        from aragora.gauntlet.signing import HMACSigner

        key = b"my-secret-32-bytes-key-here!!!!"
        signer = HMACSigner(secret_key=key, key_id="my-key")

        assert signer.algorithm == "HMAC-SHA256"
        assert signer.key_id == "my-key"

    def test_creation_generates_key(self):
        """Test HMACSigner generates key if not provided."""
        from aragora.gauntlet.signing import HMACSigner

        signer = HMACSigner()

        assert signer.algorithm == "HMAC-SHA256"
        assert signer.key_id.startswith("hmac-")

    def test_sign_returns_bytes(self, hmac_signer):
        """Test HMACSigner.sign returns bytes."""
        data = b"test data to sign"

        signature = hmac_signer.sign(data)

        assert isinstance(signature, bytes)
        assert len(signature) == 32  # SHA256 = 32 bytes

    def test_sign_deterministic(self, hmac_signer):
        """Test HMACSigner.sign is deterministic."""
        data = b"test data"

        sig1 = hmac_signer.sign(data)
        sig2 = hmac_signer.sign(data)

        assert sig1 == sig2

    def test_sign_different_data(self, hmac_signer):
        """Test different data produces different signatures."""
        sig1 = hmac_signer.sign(b"data1")
        sig2 = hmac_signer.sign(b"data2")

        assert sig1 != sig2

    def test_verify_valid(self, hmac_signer):
        """Test HMACSigner.verify with valid signature."""
        data = b"test data"
        signature = hmac_signer.sign(data)

        assert hmac_signer.verify(data, signature) is True

    def test_verify_invalid_signature(self, hmac_signer):
        """Test HMACSigner.verify with invalid signature."""
        data = b"test data"

        assert hmac_signer.verify(data, b"invalid") is False

    def test_verify_tampered_data(self, hmac_signer):
        """Test HMACSigner.verify with tampered data."""
        data = b"original data"
        signature = hmac_signer.sign(data)

        assert hmac_signer.verify(b"tampered data", signature) is False

    def test_from_env_with_key(self, monkeypatch):
        """Test HMACSigner.from_env with environment variable."""
        from aragora.gauntlet.signing import HMACSigner

        # 32 bytes in hex = 64 chars
        key_hex = "0" * 64
        monkeypatch.setenv("ARAGORA_RECEIPT_SIGNING_KEY", key_hex)

        signer = HMACSigner.from_env()

        assert signer.algorithm == "HMAC-SHA256"

    def test_from_env_without_key(self, monkeypatch):
        """Test HMACSigner.from_env without environment variable."""
        from aragora.gauntlet.signing import HMACSigner

        monkeypatch.delenv("ARAGORA_RECEIPT_SIGNING_KEY", raising=False)

        signer = HMACSigner.from_env()

        assert signer.algorithm == "HMAC-SHA256"


# ===========================================================================
# Tests: RSASigner Backend
# ===========================================================================


class TestRSASigner:
    """Tests for RSASigner backend."""

    def test_requires_cryptography(self):
        """Test RSASigner requires cryptography package."""
        from aragora.gauntlet.signing import CRYPTO_AVAILABLE

        if CRYPTO_AVAILABLE:
            from aragora.gauntlet.signing import RSASigner

            signer = RSASigner.generate_keypair()
            assert signer.algorithm == "RSA-SHA256"
        else:
            pytest.skip("cryptography package not available")

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_generate_keypair(self):
        """Test RSASigner.generate_keypair."""
        from aragora.gauntlet.signing import RSASigner

        signer = RSASigner.generate_keypair(key_id="test-rsa")

        assert signer.algorithm == "RSA-SHA256"
        assert signer.key_id == "test-rsa"

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_sign_and_verify(self):
        """Test RSASigner sign and verify round-trip."""
        from aragora.gauntlet.signing import RSASigner

        signer = RSASigner.generate_keypair()
        data = b"test data for RSA"

        signature = signer.sign(data)

        assert signer.verify(data, signature) is True

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_verify_invalid(self):
        """Test RSASigner.verify with invalid signature."""
        from aragora.gauntlet.signing import RSASigner

        signer = RSASigner.generate_keypair()
        data = b"test data"

        assert signer.verify(data, b"invalid" * 32) is False

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_export_public_key(self):
        """Test RSASigner.export_public_key."""
        from aragora.gauntlet.signing import RSASigner

        signer = RSASigner.generate_keypair()

        pem = signer.export_public_key()

        assert "BEGIN PUBLIC KEY" in pem
        assert "END PUBLIC KEY" in pem

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_sign_without_private_key(self):
        """Test RSASigner.sign without private key raises error."""
        from aragora.gauntlet.signing import RSASigner

        signer = RSASigner(private_key=None, public_key=None)

        with pytest.raises(ValueError, match="Private key required"):
            signer.sign(b"data")

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_verify_without_public_key(self):
        """Test RSASigner.verify without public key raises error."""
        from aragora.gauntlet.signing import RSASigner

        signer = RSASigner(private_key=None, public_key=None)

        with pytest.raises(ValueError, match="Public key required"):
            signer.verify(b"data", b"sig")


# ===========================================================================
# Tests: Ed25519Signer Backend
# ===========================================================================


class TestEd25519Signer:
    """Tests for Ed25519Signer backend."""

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_generate_keypair(self):
        """Test Ed25519Signer.generate_keypair."""
        from aragora.gauntlet.signing import Ed25519Signer

        signer = Ed25519Signer.generate_keypair(key_id="test-ed25519")

        assert signer.algorithm == "Ed25519"
        assert signer.key_id == "test-ed25519"

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_sign_and_verify(self):
        """Test Ed25519Signer sign and verify round-trip."""
        from aragora.gauntlet.signing import Ed25519Signer

        signer = Ed25519Signer.generate_keypair()
        data = b"test data for Ed25519"

        signature = signer.sign(data)

        assert signer.verify(data, signature) is True

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_verify_invalid(self):
        """Test Ed25519Signer.verify with invalid signature."""
        from aragora.gauntlet.signing import Ed25519Signer

        signer = Ed25519Signer.generate_keypair()
        data = b"test data"

        assert signer.verify(data, b"i" * 64) is False

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_sign_without_private_key(self):
        """Test Ed25519Signer.sign without private key raises error."""
        from aragora.gauntlet.signing import Ed25519Signer

        signer = Ed25519Signer(private_key=None, public_key=None)

        with pytest.raises(ValueError, match="Private key required"):
            signer.sign(b"data")

    @pytest.mark.skipif(
        not __import__("aragora.gauntlet.signing", fromlist=["CRYPTO_AVAILABLE"]).CRYPTO_AVAILABLE,
        reason="cryptography package required",
    )
    def test_verify_without_public_key(self):
        """Test Ed25519Signer.verify without public key raises error."""
        from aragora.gauntlet.signing import Ed25519Signer

        signer = Ed25519Signer(private_key=None, public_key=None)

        with pytest.raises(ValueError, match="Public key required"):
            signer.verify(b"data", b"sig")


# ===========================================================================
# Tests: ReceiptSigner High-Level API
# ===========================================================================


class TestReceiptSigner:
    """Tests for ReceiptSigner high-level API."""

    def test_creation_default_backend(self):
        """Test ReceiptSigner creation with default backend."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner()

        assert signer.algorithm == "HMAC-SHA256"

    def test_creation_custom_backend(self, hmac_signer):
        """Test ReceiptSigner creation with custom backend."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner(backend=hmac_signer)

        assert signer.key_id == hmac_signer.key_id

    def test_sign_receipt(self, hmac_signer, sample_receipt_data):
        """Test ReceiptSigner.sign."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner(backend=hmac_signer)

        signed = signer.sign(sample_receipt_data)

        assert signed.receipt_data == sample_receipt_data
        assert signed.signature is not None
        assert signed.signature_metadata.algorithm == "HMAC-SHA256"

    def test_verify_signed_receipt(self, hmac_signer, sample_receipt_data):
        """Test ReceiptSigner.verify."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner(backend=hmac_signer)
        signed = signer.sign(sample_receipt_data)

        assert signer.verify(signed) is True

    def test_verify_tampered_receipt(self, hmac_signer, sample_receipt_data):
        """Test ReceiptSigner.verify with tampered receipt."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner(backend=hmac_signer)
        signed = signer.sign(sample_receipt_data)

        # Tamper with the receipt
        signed.receipt_data["verdict"] = "fail"

        assert signer.verify(signed) is False

    def test_verify_dict(self, hmac_signer, sample_receipt_data):
        """Test ReceiptSigner.verify_dict."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner(backend=hmac_signer)
        signed = signer.sign(sample_receipt_data)

        assert signer.verify_dict(signed.to_dict()) is True

    def test_canonicalize_deterministic(self, hmac_signer):
        """Test canonicalization is deterministic."""
        from aragora.gauntlet.signing import ReceiptSigner

        signer = ReceiptSigner(backend=hmac_signer)

        data1 = {"b": 2, "a": 1}
        data2 = {"a": 1, "b": 2}

        sig1 = signer.sign(data1)
        sig2 = signer.sign(data2)

        # Signatures should match since canonicalization sorts keys
        assert sig1.signature == sig2.signature


# ===========================================================================
# Tests: Module Helper Functions
# ===========================================================================


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def test_get_default_signer(self):
        """Test get_default_signer returns a signer."""
        from aragora.gauntlet.signing import get_default_signer

        signer = get_default_signer()

        assert signer.algorithm == "HMAC-SHA256"

    def test_sign_receipt_function(self, sample_receipt_data):
        """Test sign_receipt function."""
        from aragora.gauntlet import signing

        # Reset default signer
        signing._default_signer = None

        signed = signing.sign_receipt(sample_receipt_data)

        assert signed.receipt_data == sample_receipt_data
        assert signed.signature is not None

    def test_verify_receipt_function(self, sample_receipt_data):
        """Test verify_receipt function."""
        from aragora.gauntlet import signing

        # Reset default signer
        signing._default_signer = None

        signed = signing.sign_receipt(sample_receipt_data)
        is_valid = signing.verify_receipt(signed)

        assert is_valid is True

    def test_roundtrip_sign_verify(self, sample_receipt_data):
        """Test full roundtrip: sign, serialize, deserialize, verify."""
        from aragora.gauntlet.signing import HMACSigner, ReceiptSigner, SignedReceipt

        signer = ReceiptSigner(
            backend=HMACSigner(
                secret_key=b"test-key-for-roundtrip-testing!",
            )
        )

        # Sign
        signed = signer.sign(sample_receipt_data)

        # Serialize to JSON
        json_str = signed.to_json()

        # Deserialize
        restored = SignedReceipt.from_json(json_str)

        # Verify
        assert signer.verify(restored) is True
