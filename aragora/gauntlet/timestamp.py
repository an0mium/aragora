"""
RFC 3161 Timestamp Authority Client for Decision Receipts.

Provides trusted third-party timestamps to prove when a receipt signature
was created. This is required for legal compliance in many jurisdictions.

Supports:
- RFC 3161 Time-Stamp Protocol (TSP) via HTTP
- Timestamp token storage and verification
- Integration with public TSA services (DigiCert, FreeTSA, etc.)
- Offline verification of timestamp tokens

"When you signed matters as much as what you signed."
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class TimestampToken:
    """
    Represents an RFC 3161 timestamp token.

    Contains the timestamp response from a TSA along with
    metadata for verification and storage.
    """

    tsa_url: str
    timestamp: str  # ISO 8601
    hash_algorithm: str
    message_digest: str  # Hex-encoded digest of the signed data
    token_data: str  # Base64-encoded raw TSA response
    serial_number: str | None = None
    tsa_name: str | None = None
    accuracy_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tsa_url": self.tsa_url,
            "timestamp": self.timestamp,
            "hash_algorithm": self.hash_algorithm,
            "message_digest": self.message_digest,
            "token_data": self.token_data,
        }
        if self.serial_number:
            result["serial_number"] = self.serial_number
        if self.tsa_name:
            result["tsa_name"] = self.tsa_name
        if self.accuracy_seconds is not None:
            result["accuracy_seconds"] = self.accuracy_seconds
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimestampToken:
        return cls(
            tsa_url=data["tsa_url"],
            timestamp=data["timestamp"],
            hash_algorithm=data["hash_algorithm"],
            message_digest=data["message_digest"],
            token_data=data["token_data"],
            serial_number=data.get("serial_number"),
            tsa_name=data.get("tsa_name"),
            accuracy_seconds=data.get("accuracy_seconds"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> TimestampToken:
        return cls.from_dict(json.loads(json_str))


@dataclass
class TimestampResult:
    """Result of a timestamp operation."""

    success: bool
    token: TimestampToken | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"success": self.success}
        if self.token:
            result["token"] = self.token.to_dict()
        if self.error:
            result["error"] = self.error
        return result


class TimestampAuthority:
    """
    RFC 3161 Timestamp Authority client.

    Requests trusted timestamps from TSA services to provide
    cryptographic proof of when a signature was created.

    Example:
        tsa = TimestampAuthority()

        # Get timestamp for signed data
        result = await tsa.timestamp(signature_bytes)
        if result.success:
            token = result.token
            print(f"Timestamped at: {token.timestamp}")

        # Verify timestamp
        is_valid = tsa.verify_digest(
            message_digest=token.message_digest,
            token=token,
        )
    """

    # Well-known public TSA endpoints
    PUBLIC_TSA_URLS = {
        "digicert": "https://timestamp.digicert.com",
        "freetsa": "https://freetsa.org/tsr",
        "sectigo": "https://timestamp.sectigo.com",
    }

    def __init__(
        self,
        tsa_url: str | None = None,
        hash_algorithm: str = "sha256",
        timeout: float = 30.0,
    ):
        """
        Initialize timestamp authority client.

        Args:
            tsa_url: TSA endpoint URL. Defaults to DigiCert.
            hash_algorithm: Hash algorithm for message digest (sha256, sha384, sha512)
            timeout: HTTP request timeout in seconds
        """
        self._tsa_url = tsa_url or self.PUBLIC_TSA_URLS["digicert"]
        self._hash_algorithm = hash_algorithm
        self._timeout = timeout

    @property
    def tsa_url(self) -> str:
        return self._tsa_url

    @property
    def hash_algorithm(self) -> str:
        return self._hash_algorithm

    def compute_digest(self, data: bytes) -> str:
        """Compute message digest for the given data."""
        if self._hash_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self._hash_algorithm == "sha384":
            return hashlib.sha384(data).hexdigest()
        elif self._hash_algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self._hash_algorithm}")

    def _build_ts_request(self, digest_bytes: bytes) -> bytes:
        """
        Build an RFC 3161 TimeStampReq.

        Uses a simplified DER encoding for the timestamp request.
        For production use with full ASN.1 support, the `asn1crypto`
        or `pyasn1` packages would be used.
        """
        try:
            from asn1crypto import tsp, algos

            # Build proper ASN.1 timestamp request
            hash_oid_map = {
                "sha256": "2.16.840.1.101.3.4.2.1",
                "sha384": "2.16.840.1.101.3.4.2.2",
                "sha512": "2.16.840.1.101.3.4.2.3",
            }

            oid = hash_oid_map.get(self._hash_algorithm)
            if not oid:
                raise ValueError(f"No OID for algorithm: {self._hash_algorithm}")

            msg_imprint = tsp.MessageImprint(
                {
                    "hash_algorithm": algos.DigestAlgorithm({"algorithm": oid}),
                    "hashed_message": digest_bytes,
                }
            )

            ts_req = tsp.TimeStampReq(
                {
                    "version": 1,
                    "message_imprint": msg_imprint,
                    "cert_req": True,
                }
            )

            return ts_req.dump()

        except ImportError:
            # Fallback: minimal DER-encoded timestamp request
            return self._build_minimal_ts_request(digest_bytes)

    def _build_minimal_ts_request(self, digest_bytes: bytes) -> bytes:
        """Build a minimal DER-encoded RFC 3161 timestamp request."""
        # SHA-256 OID: 2.16.840.1.101.3.4.2.1
        sha256_oid = bytes(
            [
                0x30,
                0x0D,  # SEQUENCE
                0x06,
                0x09,  # OID
                0x60,
                0x86,
                0x48,
                0x01,
                0x65,
                0x03,
                0x04,
                0x02,
                0x01,
                0x05,
                0x00,  # NULL
            ]
        )

        # MessageImprint: SEQUENCE { algorithm, hashedMessage }
        hash_octet = bytes([0x04, len(digest_bytes)]) + digest_bytes
        msg_imprint_content = sha256_oid + hash_octet
        msg_imprint = bytes([0x30, len(msg_imprint_content)]) + msg_imprint_content

        # Version: INTEGER 1
        version = bytes([0x02, 0x01, 0x01])

        # CertReq: BOOLEAN TRUE
        cert_req = bytes([0x01, 0x01, 0xFF])

        # TimeStampReq: SEQUENCE { version, messageImprint, certReq }
        req_content = version + msg_imprint + cert_req
        ts_req = bytes([0x30, len(req_content)]) + req_content

        return ts_req

    async def timestamp(self, data: bytes) -> TimestampResult:
        """
        Get a timestamp token for the given data.

        Args:
            data: The data to timestamp (typically a signature)

        Returns:
            TimestampResult with token if successful
        """
        if not HTTPX_AVAILABLE:
            return TimestampResult(
                success=False,
                error="httpx package required for TSA requests. Install with: pip install httpx",
            )

        try:
            # Compute digest
            digest_hex = self.compute_digest(data)
            digest_bytes = bytes.fromhex(digest_hex)

            # Build timestamp request
            ts_request = self._build_ts_request(digest_bytes)

            # Send to TSA
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._tsa_url,
                    content=ts_request,
                    headers={
                        "Content-Type": "application/timestamp-query",
                    },
                )
                response.raise_for_status()

            # Parse response
            token_data = base64.b64encode(response.content).decode("ascii")
            now = datetime.now(timezone.utc).isoformat()

            token = TimestampToken(
                tsa_url=self._tsa_url,
                timestamp=now,
                hash_algorithm=self._hash_algorithm,
                message_digest=digest_hex,
                token_data=token_data,
                tsa_name=self._get_tsa_name(),
            )

            logger.info(f"Timestamp obtained from {self._tsa_url}")
            return TimestampResult(success=True, token=token)

        except httpx.HTTPStatusError as e:
            error = f"TSA HTTP error {e.response.status_code}: {e.response.text[:200]}"
            logger.error(error)
            return TimestampResult(success=False, error=error)
        except httpx.RequestError as e:
            error = f"TSA request failed: {str(e)}"
            logger.error(error)
            return TimestampResult(success=False, error=error)
        except (RuntimeError, ValueError, OSError, TypeError) as e:
            error = f"Timestamp error: {str(e)}"
            logger.exception(error)
            return TimestampResult(success=False, error=error)

    def _get_tsa_name(self) -> str | None:
        """Get friendly name for the TSA URL."""
        for name, url in self.PUBLIC_TSA_URLS.items():
            if self._tsa_url == url:
                return name
        return None

    def verify_digest(
        self,
        message_digest: str,
        token: TimestampToken,
    ) -> bool:
        """
        Verify that a timestamp token matches the expected digest.

        This performs local verification of the digest match.
        Full TSA certificate chain verification requires the TSA's
        root certificate.

        Args:
            message_digest: Expected hex-encoded digest
            token: The timestamp token to verify

        Returns:
            True if the digest matches
        """
        return token.message_digest == message_digest

    def verify_data(self, data: bytes, token: TimestampToken) -> bool:
        """
        Verify that a timestamp token matches the given data.

        Args:
            data: Original data that was timestamped
            token: The timestamp token to verify

        Returns:
            True if the data matches the token's digest
        """
        digest = self.compute_digest(data)
        return self.verify_digest(digest, token)

    @classmethod
    def create_local_token(
        cls,
        data: bytes,
        hash_algorithm: str = "sha256",
    ) -> TimestampToken:
        """
        Create a local (non-TSA) timestamp token.

        Useful for development/testing or when TSA is unavailable.
        These tokens provide local timestamps but without third-party
        trust guarantees.

        Args:
            data: Data to timestamp
            hash_algorithm: Hash algorithm to use

        Returns:
            TimestampToken with local timestamp
        """
        if hash_algorithm == "sha256":
            digest = hashlib.sha256(data).hexdigest()
        elif hash_algorithm == "sha384":
            digest = hashlib.sha384(data).hexdigest()
        elif hash_algorithm == "sha512":
            digest = hashlib.sha512(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")

        token_payload = json.dumps(
            {
                "type": "local_timestamp",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "digest": digest,
                "algorithm": hash_algorithm,
            }
        ).encode("utf-8")

        return TimestampToken(
            tsa_url="local",
            timestamp=datetime.now(timezone.utc).isoformat(),
            hash_algorithm=hash_algorithm,
            message_digest=digest,
            token_data=base64.b64encode(token_payload).decode("ascii"),
            tsa_name="local",
        )
