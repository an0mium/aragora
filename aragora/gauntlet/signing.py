"""
Cryptographic Signing for Gauntlet Receipts.

Provides digital signatures for decision receipts to ensure:
- Tamper-evidence: Any modification invalidates the signature
- Non-repudiation: Receipts can be verified as authentic
- Audit compliance: Cryptographic proof for regulatory requirements

Supports multiple signing backends:
- HMAC-SHA256: Fast, symmetric signing for internal use
- RSA-SHA256: Asymmetric signing for external verification
- Ed25519: Modern, high-performance signing

"Trust, but verify with cryptographic signatures."
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

# Try to import cryptography for RSA/Ed25519 support
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
    from cryptography.exceptions import InvalidSignature

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


@dataclass
class SignatureMetadata:
    """Metadata about a signature."""

    algorithm: str
    timestamp: str
    key_id: str
    version: str = "1.0"

    def to_dict(self) -> dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
            "key_id": self.key_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "SignatureMetadata":
        return cls(
            algorithm=data["algorithm"],
            timestamp=data["timestamp"],
            key_id=data["key_id"],
            version=data.get("version", "1.0"),
        )


@dataclass
class SignedReceipt:
    """A receipt with cryptographic signature."""

    receipt_data: dict[str, Any]
    signature: str  # Base64-encoded signature
    signature_metadata: SignatureMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt_data,
            "signature": self.signature,
            "signature_metadata": self.signature_metadata.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignedReceipt":
        return cls(
            receipt_data=data["receipt"],
            signature=data["signature"],
            signature_metadata=SignatureMetadata.from_dict(data["signature_metadata"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SignedReceipt":
        return cls.from_dict(json.loads(json_str))


class SigningBackend(ABC):
    """Abstract base class for signing backends."""

    @property
    @abstractmethod
    def algorithm(self) -> str:
        """Return the algorithm name."""
        pass

    @property
    @abstractmethod
    def key_id(self) -> str:
        """Return the key identifier."""
        pass

    @abstractmethod
    def sign(self, data: bytes) -> bytes:
        """Sign data and return signature bytes."""
        pass

    @abstractmethod
    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify a signature. Returns True if valid."""
        pass


class HMACSigner(SigningBackend):
    """HMAC-SHA256 signing backend for symmetric key signing."""

    def __init__(self, secret_key: Optional[bytes] = None, key_id: Optional[str] = None):
        """
        Initialize HMAC signer.

        Args:
            secret_key: 32-byte secret key. Generated if not provided.
            key_id: Identifier for this key. Generated if not provided.
        """
        self._secret_key = secret_key or secrets.token_bytes(32)
        self._key_id = key_id or f"hmac-{secrets.token_hex(4)}"

    @property
    def algorithm(self) -> str:
        return "HMAC-SHA256"

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, data: bytes) -> bytes:
        return hmac.new(self._secret_key, data, hashlib.sha256).digest()

    def verify(self, data: bytes, signature: bytes) -> bool:
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)

    @classmethod
    def from_env(cls, env_var: str = "ARAGORA_RECEIPT_SIGNING_KEY") -> "HMACSigner":
        """Create signer from environment variable (hex-encoded key)."""
        key_hex = os.environ.get(env_var)
        if key_hex:
            return cls(secret_key=bytes.fromhex(key_hex))
        return cls()


class RSASigner(SigningBackend):
    """RSA-SHA256 signing backend for asymmetric key signing."""

    def __init__(
        self,
        private_key: Optional[Any] = None,
        public_key: Optional[Any] = None,
        key_id: Optional[str] = None,
    ):
        """
        Initialize RSA signer.

        Args:
            private_key: RSA private key for signing.
            public_key: RSA public key for verification.
            key_id: Identifier for this key pair.
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for RSA signing")

        self._private_key = private_key
        self._public_key = public_key
        self._key_id = key_id or f"rsa-{secrets.token_hex(4)}"

    @property
    def algorithm(self) -> str:
        return "RSA-SHA256"

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, data: bytes) -> bytes:
        if self._private_key is None:
            raise ValueError("Private key required for signing")
        return self._private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

    def verify(self, data: bytes, signature: bytes) -> bool:
        if self._public_key is None:
            raise ValueError("Public key required for verification")
        try:
            self._public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False

    @classmethod
    def generate_keypair(cls, key_id: Optional[str] = None) -> "RSASigner":
        """Generate a new RSA key pair."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for RSA signing")

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        public_key = private_key.public_key()

        return cls(
            private_key=private_key,
            public_key=public_key,
            key_id=key_id,
        )

    def export_public_key(self) -> str:
        """Export public key in PEM format."""
        if self._public_key is None:
            raise ValueError("No public key available")
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()


class Ed25519Signer(SigningBackend):
    """Ed25519 signing backend for modern, high-performance signing."""

    def __init__(
        self,
        private_key: Optional[Any] = None,
        public_key: Optional[Any] = None,
        key_id: Optional[str] = None,
    ):
        """
        Initialize Ed25519 signer.

        Args:
            private_key: Ed25519 private key for signing.
            public_key: Ed25519 public key for verification.
            key_id: Identifier for this key pair.
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for Ed25519 signing")

        self._private_key = private_key
        self._public_key = public_key
        self._key_id = key_id or f"ed25519-{secrets.token_hex(4)}"

    @property
    def algorithm(self) -> str:
        return "Ed25519"

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, data: bytes) -> bytes:
        if self._private_key is None:
            raise ValueError("Private key required for signing")
        return self._private_key.sign(data)

    def verify(self, data: bytes, signature: bytes) -> bool:
        if self._public_key is None:
            raise ValueError("Public key required for verification")
        try:
            self._public_key.verify(signature, data)
            return True
        except InvalidSignature:
            return False

    @classmethod
    def generate_keypair(cls, key_id: Optional[str] = None) -> "Ed25519Signer":
        """Generate a new Ed25519 key pair."""
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography package required for Ed25519 signing")

        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        return cls(
            private_key=private_key,
            public_key=public_key,
            key_id=key_id,
        )


class ReceiptSigner:
    """
    High-level receipt signing service.

    Signs receipts using configurable backends and produces
    self-contained signed receipt documents.

    Example:
        signer = ReceiptSigner()  # Uses HMAC by default

        # Sign a receipt
        signed = signer.sign(receipt.to_dict())

        # Verify a signed receipt
        is_valid = signer.verify(signed)

        # Export for external verification
        signed_json = signed.to_json()
    """

    def __init__(self, backend: Optional[SigningBackend] = None):
        """
        Initialize receipt signer.

        Args:
            backend: Signing backend to use. Defaults to HMAC-SHA256.
        """
        self._backend = backend or HMACSigner.from_env()

    @property
    def algorithm(self) -> str:
        """Return the signing algorithm in use."""
        return self._backend.algorithm

    @property
    def key_id(self) -> str:
        """Return the key identifier."""
        return self._backend.key_id

    def _canonicalize(self, receipt_data: dict[str, Any]) -> bytes:
        """
        Canonicalize receipt data for signing.

        Uses JSON with sorted keys for deterministic output.
        """
        canonical = json.dumps(receipt_data, sort_keys=True, default=str)
        return canonical.encode("utf-8")

    def sign(self, receipt_data: dict[str, Any]) -> SignedReceipt:
        """
        Sign a receipt and return a SignedReceipt.

        Args:
            receipt_data: Receipt data dictionary (from DecisionReceipt.to_dict())

        Returns:
            SignedReceipt with signature and metadata
        """
        # Canonicalize receipt data
        canonical_data = self._canonicalize(receipt_data)

        # Sign
        signature_bytes = self._backend.sign(canonical_data)
        signature_b64 = base64.b64encode(signature_bytes).decode("ascii")

        # Create metadata
        metadata = SignatureMetadata(
            algorithm=self._backend.algorithm,
            timestamp=datetime.now(timezone.utc).isoformat(),
            key_id=self._backend.key_id,
        )

        return SignedReceipt(
            receipt_data=receipt_data,
            signature=signature_b64,
            signature_metadata=metadata,
        )

    def verify(self, signed_receipt: SignedReceipt) -> bool:
        """
        Verify a signed receipt.

        Args:
            signed_receipt: The SignedReceipt to verify

        Returns:
            True if signature is valid, False otherwise
        """
        # Canonicalize receipt data
        canonical_data = self._canonicalize(signed_receipt.receipt_data)

        # Decode signature
        signature_bytes = base64.b64decode(signed_receipt.signature)

        # Verify
        return self._backend.verify(canonical_data, signature_bytes)

    def verify_dict(self, signed_receipt_dict: dict[str, Any]) -> bool:
        """Verify a signed receipt from dict format."""
        signed_receipt = SignedReceipt.from_dict(signed_receipt_dict)
        return self.verify(signed_receipt)


# Default signer instance for convenience
_default_signer: Optional[ReceiptSigner] = None


def get_default_signer() -> ReceiptSigner:
    """Get or create the default receipt signer."""
    global _default_signer
    if _default_signer is None:
        _default_signer = ReceiptSigner()
    return _default_signer


def sign_receipt(receipt_data: dict[str, Any]) -> SignedReceipt:
    """
    Sign a receipt using the default signer.

    Args:
        receipt_data: Receipt data dictionary

    Returns:
        SignedReceipt with signature
    """
    return get_default_signer().sign(receipt_data)


def verify_receipt(signed_receipt: SignedReceipt) -> bool:
    """
    Verify a signed receipt using the default signer.

    Args:
        signed_receipt: The SignedReceipt to verify

    Returns:
        True if valid
    """
    return get_default_signer().verify(signed_receipt)
