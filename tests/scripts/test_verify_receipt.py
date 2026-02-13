"""Tests for the verify_receipt CLI tool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Path to the script
SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "verify_receipt.py"


class TestVerifyReceiptCLI:
    """Tests for verify_receipt.py CLI tool."""

    @pytest.fixture
    def signed_receipt(self):
        """Create a signed receipt for testing."""
        from aragora.gauntlet.signing import HMACSigner, ReceiptSigner, SignatoryInfo

        # Use a known key for testing
        key = b"test-secret-key-for-verification"
        signer = ReceiptSigner(backend=HMACSigner(secret_key=key, key_id="test-key"))

        receipt_data = {
            "decision_id": "test-decision-123",
            "verdict": "APPROVED",
            "confidence": 0.95,
            "rationale": "Test decision for CLI verification",
        }

        signatory = SignatoryInfo(
            name="Test User",
            email="test@example.com",
            title="Tester",
            role="Approver",
        )

        signed = signer.sign(receipt_data, signatory=signatory)
        return signed.to_dict(), key

    @pytest.fixture
    def receipt_file(self, signed_receipt, tmp_path):
        """Write signed receipt to a temporary file."""
        data, key = signed_receipt
        file_path = tmp_path / "test_receipt.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path, key

    def run_cli(self, *args, env=None):
        """Run the verify_receipt CLI with given arguments."""
        cmd = [sys.executable, str(SCRIPT_PATH)] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        return result

    def test_valid_signature(self, receipt_file):
        """Test verification of valid signature."""
        file_path, key = receipt_file
        result = self.run_cli(str(file_path), "--key", key.hex())

        assert result.returncode == 0
        assert "VALID" in result.stdout

    def test_invalid_signature(self, receipt_file):
        """Test detection of invalid signature."""
        file_path, _ = receipt_file
        wrong_key = b"wrong-key-that-wont-work"
        result = self.run_cli(str(file_path), "--key", wrong_key.hex())

        assert result.returncode == 1
        assert "INVALID" in result.stdout

    def test_quiet_mode(self, receipt_file):
        """Test quiet output mode."""
        file_path, key = receipt_file
        result = self.run_cli(str(file_path), "--key", key.hex(), "-q")

        assert result.returncode == 0
        assert result.stdout.strip() == "VALID"

    def test_json_output(self, receipt_file):
        """Test JSON output mode."""
        file_path, key = receipt_file
        result = self.run_cli(str(file_path), "--key", key.hex(), "--json")

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["valid"] is True
        assert output["algorithm"] == "HMAC-SHA256"
        assert "signatory" in output

    def test_verbose_mode(self, receipt_file):
        """Test verbose output shows full receipt."""
        file_path, key = receipt_file
        result = self.run_cli(str(file_path), "--key", key.hex(), "-v")

        assert result.returncode == 0
        assert "VALID" in result.stdout
        assert "Full Receipt Data" in result.stdout
        assert "decision_id" in result.stdout

    def test_shows_signatory_info(self, receipt_file):
        """Test that signatory info is displayed."""
        file_path, key = receipt_file
        result = self.run_cli(str(file_path), "--key", key.hex())

        assert result.returncode == 0
        assert "Signatory Information" in result.stdout
        assert "Test User" in result.stdout
        assert "test@example.com" in result.stdout

    def test_key_from_env(self, receipt_file):
        """Test loading key from environment variable."""
        file_path, key = receipt_file
        env = {"ARAGORA_TEST_KEY": key.hex()}
        result = self.run_cli(str(file_path), "--key-env", "ARAGORA_TEST_KEY", env=env)

        assert result.returncode == 0
        assert "VALID" in result.stdout

    def test_missing_env_var(self, receipt_file):
        """Test error when env var not set."""
        file_path, _ = receipt_file
        result = self.run_cli(str(file_path), "--key-env", "NONEXISTENT_VAR_12345")

        assert result.returncode == 2
        assert "not set" in result.stderr

    def test_file_not_found(self, tmp_path):
        """Test error when receipt file doesn't exist."""
        result = self.run_cli(str(tmp_path / "nonexistent.json"), "--key", "abc123")

        assert result.returncode == 2
        assert "not found" in result.stderr

    def test_invalid_json(self, tmp_path):
        """Test error when receipt file has invalid JSON."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json {")

        result = self.run_cli(str(file_path), "--key", "abc123")

        assert result.returncode == 2
        assert "Invalid JSON" in result.stderr

    def test_missing_signature_field(self, tmp_path):
        """Test error when signature field is missing."""
        file_path = tmp_path / "incomplete.json"
        file_path.write_text(json.dumps({"receipt": {}, "signature_metadata": {}}))

        result = self.run_cli(str(file_path), "--key", "abc123")

        assert result.returncode == 2
        assert "signature" in result.stderr

    def test_missing_key_for_hmac(self, tmp_path):
        """Test error when no key provided for HMAC verification."""
        data = {
            "receipt": {},
            "signature": "test",
            "signature_metadata": {"algorithm": "HMAC-SHA256"},
        }
        file_path = tmp_path / "receipt.json"
        file_path.write_text(json.dumps(data))

        result = self.run_cli(str(file_path))

        assert result.returncode == 2
        assert "requires a signing key" in result.stderr


class TestVerifyReceiptWithRSA:
    """Tests for RSA signature verification."""

    @pytest.fixture
    def rsa_signed_receipt(self, tmp_path):
        """Create an RSA-signed receipt."""
        from aragora.gauntlet.signing import RSASigner, ReceiptSigner

        signer_backend = RSASigner.generate_keypair(key_id="test-rsa")
        signer = ReceiptSigner(backend=signer_backend)

        receipt_data = {
            "decision_id": "rsa-test-123",
            "verdict": "APPROVED",
        }

        signed = signer.sign(receipt_data)

        # Write receipt
        receipt_path = tmp_path / "rsa_receipt.json"
        with open(receipt_path, "w") as f:
            json.dump(signed.to_dict(), f)

        # Write public key
        key_path = tmp_path / "public_key.pem"
        key_path.write_text(signer_backend.export_public_key())

        return receipt_path, key_path

    def run_cli(self, *args):
        """Run the verify_receipt CLI."""
        cmd = [sys.executable, str(SCRIPT_PATH)] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_rsa_verification(self, rsa_signed_receipt):
        """Test RSA signature verification with public key file."""
        receipt_path, key_path = rsa_signed_receipt
        result = self.run_cli(str(receipt_path), "--key-file", str(key_path))

        assert result.returncode == 0
        assert "VALID" in result.stdout


class TestVerifyReceiptWithEd25519:
    """Tests for Ed25519 signature verification."""

    @pytest.fixture
    def ed25519_signed_receipt(self, tmp_path):
        """Create an Ed25519-signed receipt."""
        from cryptography.hazmat.primitives import serialization

        from aragora.gauntlet.signing import Ed25519Signer, ReceiptSigner

        signer_backend = Ed25519Signer.generate_keypair(key_id="test-ed25519")
        signer = ReceiptSigner(backend=signer_backend)

        receipt_data = {
            "decision_id": "ed25519-test-123",
            "verdict": "REJECTED",
        }

        signed = signer.sign(receipt_data)

        # Write receipt
        receipt_path = tmp_path / "ed25519_receipt.json"
        with open(receipt_path, "w") as f:
            json.dump(signed.to_dict(), f)

        # Write public key
        key_path = tmp_path / "ed25519_public.pem"
        public_key_pem = signer_backend._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        key_path.write_bytes(public_key_pem)

        return receipt_path, key_path

    def run_cli(self, *args):
        """Run the verify_receipt CLI."""
        cmd = [sys.executable, str(SCRIPT_PATH)] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_ed25519_verification(self, ed25519_signed_receipt):
        """Test Ed25519 signature verification with public key file."""
        receipt_path, key_path = ed25519_signed_receipt
        result = self.run_cli(str(receipt_path), "--key-file", str(key_path))

        assert result.returncode == 0
        assert "VALID" in result.stdout
