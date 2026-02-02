"""Tests for RFC 3161 Timestamp Authority client."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gauntlet.timestamp import (
    TimestampAuthority,
    TimestampResult,
    TimestampToken,
)


class TestTimestampToken:
    """Tests for TimestampToken dataclass."""

    def test_create_token(self):
        """Test creating a timestamp token."""
        token = TimestampToken(
            tsa_url="https://timestamp.digicert.com",
            timestamp="2026-02-01T12:00:00Z",
            hash_algorithm="sha256",
            message_digest="abc123",
            token_data="dGVzdA==",
        )

        assert token.tsa_url == "https://timestamp.digicert.com"
        assert token.timestamp == "2026-02-01T12:00:00Z"
        assert token.hash_algorithm == "sha256"
        assert token.message_digest == "abc123"

    def test_token_to_dict(self):
        """Test token serialization."""
        token = TimestampToken(
            tsa_url="https://tsa.example.com",
            timestamp="2026-02-01T12:00:00Z",
            hash_algorithm="sha256",
            message_digest="abc123",
            token_data="dGVzdA==",
            serial_number="12345",
            tsa_name="example",
        )

        data = token.to_dict()
        assert data["tsa_url"] == "https://tsa.example.com"
        assert data["serial_number"] == "12345"
        assert data["tsa_name"] == "example"

    def test_token_from_dict(self):
        """Test token deserialization."""
        data = {
            "tsa_url": "https://tsa.example.com",
            "timestamp": "2026-02-01T12:00:00Z",
            "hash_algorithm": "sha256",
            "message_digest": "abc123",
            "token_data": "dGVzdA==",
            "serial_number": "67890",
            "tsa_name": "test",
            "accuracy_seconds": 1.0,
        }

        token = TimestampToken.from_dict(data)
        assert token.tsa_url == "https://tsa.example.com"
        assert token.serial_number == "67890"
        assert token.accuracy_seconds == 1.0

    def test_token_roundtrip(self):
        """Test token serialization roundtrip."""
        original = TimestampToken(
            tsa_url="https://tsa.test.com",
            timestamp="2026-02-01T12:00:00Z",
            hash_algorithm="sha256",
            message_digest="deadbeef",
            token_data="dGVzdGRhdGE=",
            serial_number="999",
            tsa_name="test",
            accuracy_seconds=0.5,
        )

        json_str = original.to_json()
        restored = TimestampToken.from_json(json_str)

        assert restored.tsa_url == original.tsa_url
        assert restored.message_digest == original.message_digest
        assert restored.serial_number == original.serial_number
        assert restored.accuracy_seconds == original.accuracy_seconds

    def test_token_to_dict_optional_fields(self):
        """Test that optional fields are omitted when None."""
        token = TimestampToken(
            tsa_url="https://tsa.test.com",
            timestamp="2026-02-01T12:00:00Z",
            hash_algorithm="sha256",
            message_digest="abc",
            token_data="dGVzdA==",
        )

        data = token.to_dict()
        assert "serial_number" not in data
        assert "tsa_name" not in data
        assert "accuracy_seconds" not in data


class TestTimestampResult:
    """Tests for TimestampResult."""

    def test_success_result(self):
        """Test successful timestamp result."""
        token = TimestampToken(
            tsa_url="https://tsa.test.com",
            timestamp="2026-02-01T12:00:00Z",
            hash_algorithm="sha256",
            message_digest="abc",
            token_data="dGVzdA==",
        )
        result = TimestampResult(success=True, token=token)

        assert result.success is True
        assert result.token is not None
        assert result.error is None

    def test_failure_result(self):
        """Test failed timestamp result."""
        result = TimestampResult(success=False, error="TSA unreachable")

        assert result.success is False
        assert result.token is None
        assert result.error == "TSA unreachable"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = TimestampResult(success=False, error="Test error")
        data = result.to_dict()

        assert data["success"] is False
        assert data["error"] == "Test error"


class TestTimestampAuthority:
    """Tests for TimestampAuthority client."""

    def test_default_tsa_url(self):
        """Test default TSA URL is DigiCert."""
        tsa = TimestampAuthority()
        assert tsa.tsa_url == "https://timestamp.digicert.com"

    def test_custom_tsa_url(self):
        """Test custom TSA URL."""
        tsa = TimestampAuthority(tsa_url="https://custom.tsa.com")
        assert tsa.tsa_url == "https://custom.tsa.com"

    def test_compute_digest_sha256(self):
        """Test SHA-256 digest computation."""
        tsa = TimestampAuthority(hash_algorithm="sha256")
        data = b"test data"
        digest = tsa.compute_digest(data)

        expected = hashlib.sha256(data).hexdigest()
        assert digest == expected

    def test_compute_digest_sha384(self):
        """Test SHA-384 digest computation."""
        tsa = TimestampAuthority(hash_algorithm="sha384")
        data = b"test data"
        digest = tsa.compute_digest(data)

        expected = hashlib.sha384(data).hexdigest()
        assert digest == expected

    def test_compute_digest_sha512(self):
        """Test SHA-512 digest computation."""
        tsa = TimestampAuthority(hash_algorithm="sha512")
        data = b"test data"
        digest = tsa.compute_digest(data)

        expected = hashlib.sha512(data).hexdigest()
        assert digest == expected

    def test_compute_digest_unsupported(self):
        """Test unsupported hash algorithm raises error."""
        tsa = TimestampAuthority(hash_algorithm="md5")
        with pytest.raises(ValueError, match="Unsupported"):
            tsa.compute_digest(b"test")

    def test_verify_digest_match(self):
        """Test digest verification when matching."""
        tsa = TimestampAuthority()
        data = b"test data"
        digest = tsa.compute_digest(data)

        token = TimestampToken(
            tsa_url="https://tsa.test.com",
            timestamp="2026-02-01T12:00:00Z",
            hash_algorithm="sha256",
            message_digest=digest,
            token_data="dGVzdA==",
        )

        assert tsa.verify_digest(digest, token) is True

    def test_verify_digest_mismatch(self):
        """Test digest verification when not matching."""
        tsa = TimestampAuthority()

        token = TimestampToken(
            tsa_url="https://tsa.test.com",
            timestamp="2026-02-01T12:00:00Z",
            hash_algorithm="sha256",
            message_digest="correctdigest",
            token_data="dGVzdA==",
        )

        assert tsa.verify_digest("wrongdigest", token) is False

    def test_verify_data(self):
        """Test data verification against token."""
        tsa = TimestampAuthority()
        data = b"original data"
        digest = tsa.compute_digest(data)

        token = TimestampToken(
            tsa_url="https://tsa.test.com",
            timestamp="2026-02-01T12:00:00Z",
            hash_algorithm="sha256",
            message_digest=digest,
            token_data="dGVzdA==",
        )

        assert tsa.verify_data(data, token) is True
        assert tsa.verify_data(b"tampered data", token) is False

    def test_create_local_token(self):
        """Test creating a local timestamp token."""
        data = b"test data for local timestamp"
        token = TimestampAuthority.create_local_token(data)

        assert token.tsa_url == "local"
        assert token.tsa_name == "local"
        assert token.hash_algorithm == "sha256"
        assert token.message_digest == hashlib.sha256(data).hexdigest()
        assert token.timestamp is not None

        # Verify token data is valid base64 JSON
        payload = json.loads(base64.b64decode(token.token_data))
        assert payload["type"] == "local_timestamp"
        assert payload["digest"] == token.message_digest

    def test_create_local_token_sha384(self):
        """Test creating local token with SHA-384."""
        data = b"test data"
        token = TimestampAuthority.create_local_token(data, hash_algorithm="sha384")

        assert token.hash_algorithm == "sha384"
        assert token.message_digest == hashlib.sha384(data).hexdigest()

    def test_create_local_token_unsupported(self):
        """Test creating local token with unsupported algorithm."""
        with pytest.raises(ValueError, match="Unsupported"):
            TimestampAuthority.create_local_token(b"test", hash_algorithm="md5")

    def test_get_tsa_name_known(self):
        """Test TSA name resolution for known URLs."""
        tsa = TimestampAuthority(tsa_url="https://timestamp.digicert.com")
        assert tsa._get_tsa_name() == "digicert"

    def test_get_tsa_name_unknown(self):
        """Test TSA name resolution for unknown URLs."""
        tsa = TimestampAuthority(tsa_url="https://custom.tsa.com")
        assert tsa._get_tsa_name() is None

    @pytest.mark.asyncio
    async def test_timestamp_success(self):
        """Test successful timestamp request."""
        tsa = TimestampAuthority()

        mock_response = MagicMock()
        mock_response.content = b"fake-tsa-response"
        mock_response.raise_for_status = MagicMock()

        with patch("aragora.gauntlet.timestamp.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                result = await tsa.timestamp(b"test data")

        assert result.success is True
        assert result.token is not None
        assert result.token.hash_algorithm == "sha256"
        assert result.token.message_digest == hashlib.sha256(b"test data").hexdigest()

    @pytest.mark.asyncio
    async def test_timestamp_without_httpx(self):
        """Test timestamp fails gracefully without httpx."""
        tsa = TimestampAuthority()

        with patch("aragora.gauntlet.timestamp.HTTPX_AVAILABLE", False):
            result = await tsa.timestamp(b"test data")

        assert result.success is False
        assert "httpx" in result.error

    @pytest.mark.asyncio
    async def test_timestamp_http_error(self):
        """Test timestamp handles HTTP errors."""
        tsa = TimestampAuthority()

        with patch("aragora.gauntlet.timestamp.HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()

                import httpx

                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.text = "Internal Server Error"
                mock_client.post = AsyncMock(
                    side_effect=httpx.HTTPStatusError(
                        "Server Error",
                        request=MagicMock(),
                        response=mock_response,
                    )
                )
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_cls.return_value = mock_client

                result = await tsa.timestamp(b"test data")

        assert result.success is False
        assert "HTTP error" in result.error

    def test_build_minimal_ts_request(self):
        """Test building minimal timestamp request."""
        tsa = TimestampAuthority()
        digest = bytes.fromhex(hashlib.sha256(b"test").hexdigest())
        request = tsa._build_minimal_ts_request(digest)

        # Should be valid DER: starts with SEQUENCE tag (0x30)
        assert request[0] == 0x30
        assert len(request) > 10

    def test_public_tsa_urls(self):
        """Test that public TSA URLs are configured."""
        assert "digicert" in TimestampAuthority.PUBLIC_TSA_URLS
        assert "freetsa" in TimestampAuthority.PUBLIC_TSA_URLS
        assert "sectigo" in TimestampAuthority.PUBLIC_TSA_URLS


class TestTimestampIntegrationWithSigning:
    """Tests for timestamp integration with receipt signing."""

    def test_timestamp_signed_receipt(self):
        """Test timestamping a signed receipt."""
        from aragora.gauntlet.signing import HMACSigner, ReceiptSigner

        # Sign a receipt
        signer = ReceiptSigner(backend=HMACSigner(key_id="test"))
        receipt_data = {"decision_id": "test-1", "verdict": "APPROVED"}
        signed = signer.sign(receipt_data)

        # Create local timestamp for the signature
        sig_bytes = base64.b64decode(signed.signature)
        token = TimestampAuthority.create_local_token(sig_bytes)

        # Verify timestamp matches signature
        tsa = TimestampAuthority()
        assert tsa.verify_data(sig_bytes, token) is True

        # Add token to receipt
        receipt_dict = signed.to_dict()
        receipt_dict["timestamp_token"] = token.to_dict()

        # Verify token survives serialization
        restored_token = TimestampToken.from_dict(receipt_dict["timestamp_token"])
        assert restored_token.message_digest == token.message_digest
        assert tsa.verify_data(sig_bytes, restored_token) is True

    def test_timestamp_tamper_detection(self):
        """Test that timestamp detects tampering."""
        from aragora.gauntlet.signing import HMACSigner, ReceiptSigner

        signer = ReceiptSigner(backend=HMACSigner(key_id="test"))
        receipt_data = {"decision_id": "test-2", "verdict": "REJECTED"}
        signed = signer.sign(receipt_data)

        sig_bytes = base64.b64decode(signed.signature)
        token = TimestampAuthority.create_local_token(sig_bytes)

        # Tamper with signature
        tampered = sig_bytes + b"\x00"

        tsa = TimestampAuthority()
        assert tsa.verify_data(tampered, token) is False
