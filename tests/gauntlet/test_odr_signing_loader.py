"""File custody, producer failures, and metadata compatibility with v0.1."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from aragora_verify.verifier import verify

from aragora.gauntlet.odr_export import load_odr_schema, sign_odr_if_configured
from aragora.gauntlet.odr_signing import (
    OdrSigningError,
    compute_key_id,
    generate_signing_key,
    load_signing_key_from_secrets,
    sign_odr_receipt,
)
from aragora.gauntlet.odr_verify import verify_odr_document
from aragora.gauntlet.receipt_models import DecisionReceipt
from tests.gauntlet.odr_test_keys import odr_test_key

ROOT = Path(__file__).resolve().parents[2]
FILE_ENV = "ARAGORA_ODR_SIGNING_KEY_FILE"


@pytest.fixture(autouse=True)
def isolated_signing(monkeypatch):
    monkeypatch.delenv(FILE_ENV, raising=False)
    monkeypatch.delenv("ARAGORA_ODR_SIGNING_KEY_SECRET", raising=False)
    monkeypatch.setenv("ARAGORA_USE_SECRETS_MANAGER", "false")


@pytest.fixture
def odr():
    doc = json.loads((ROOT / "docs/specs/examples/example-approved-clean.odr.json").read_text())
    jsonschema.validate(doc, load_odr_schema())
    return doc


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "test.pem"
    path.write_bytes(
        odr_test_key().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return path


def test_file_key_signs_and_verifies_with_metadata(odr, key_file, monkeypatch):
    monkeypatch.setenv(FILE_ENV, str(key_file))
    monkeypatch.setenv("ARAGORA_ODR_SIGNING_KEY_SECRET", "must-not-be-read")
    with patch("aragora.gauntlet.odr_signing._load_pem_secret_from_aws") as aws:
        signed = sign_odr_if_configured(odr)
    aws.assert_not_called()
    sig = signed["signatures"][0]
    assert sig["key_id"] == compute_key_id(odr_test_key().public_key())
    assert sig["issuer"] and sig["role"] == "emitter"
    assert datetime.fromisoformat(sig["signed_at"]).tzinfo is not None
    jsonschema.validate(signed, load_odr_schema())
    for verifier in (verify_odr_document, verify):
        assert verifier(signed, public_key=odr_test_key().public_key()).ok
        assert not verifier(signed, public_key=generate_signing_key().public_key()).ok
        tampered = copy.deepcopy(signed)
        tampered["reasoning"]["summary"] += " altered"
        assert not verifier(tampered, public_key=odr_test_key().public_key()).ok
        assert verifier(odr).ok  # v0.1 without metadata remains valid.


@pytest.mark.parametrize("value", [None, ""])
def test_unset_and_empty_are_unsigned_and_preserve_aws_path(odr, value, monkeypatch):
    if value is not None:
        monkeypatch.setenv(FILE_ENV, value)
    assert sign_odr_if_configured(odr)["signatures"] == []
    pem = odr_test_key().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    monkeypatch.setenv("ARAGORA_ODR_SIGNING_KEY_SECRET", "named-secret")
    with patch("aragora.gauntlet.odr_signing._load_pem_secret_from_aws", return_value=pem) as aws:
        key = load_signing_key_from_secrets()
    aws.assert_called_once_with("named-secret", explicitly_named=True)
    assert compute_key_id(key.public_key()) == compute_key_id(odr_test_key().public_key())


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "garbage", "ec", "encrypted", "unreadable"]
)
def test_unusable_file_fails_closed(odr, tmp_path, kind, monkeypatch):
    path = tmp_path / "bad.pem"
    if kind == "directory":
        path.mkdir()
    elif kind == "ec":
        path.write_bytes(
            ec.generate_private_key(ec.SECP256R1()).private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    elif kind == "encrypted":
        path.write_bytes(
            odr_test_key().private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.BestAvailableEncryption(b"public-test-password"),
            )
        )
    elif kind != "missing":
        path.write_text("not a key")
    if kind == "unreadable":
        monkeypatch.setattr(Path, "read_bytes", lambda _: _permission_denied())
    monkeypatch.setenv(FILE_ENV, str(path))
    with patch("aragora.gauntlet.odr_signing._load_pem_secret_from_aws") as aws:
        with pytest.raises(OdrSigningError, match="configured but could not be used"):
            sign_odr_if_configured(odr)
    aws.assert_not_called()


def _permission_denied():
    raise PermissionError("denied")


@pytest.mark.parametrize("role", ["emitter", "reviewer", "attestor", "notary"])
def test_optional_signature_metadata_and_legacy_entries(odr, role):
    signed = sign_odr_receipt(
        odr,
        odr_test_key(),
        issuer="test-issuer",
        role=role,
        signed_at="2026-09-04T00:00:00Z",
        expires_at="2027-09-04T00:00:00Z",
    )
    for verifier in (verify_odr_document, verify):
        assert verifier(signed, public_key=odr_test_key().public_key()).ok
    jsonschema.validate(signed, load_odr_schema())
    legacy = copy.deepcopy(signed)
    for field in ("issuer", "role", "signed_at", "expires_at"):
        legacy["signatures"][0].pop(field)
    assert verify_odr_document(legacy, public_key=odr_test_key().public_key()).ok
    assert verify(legacy, public_key=odr_test_key().public_key()).ok


@pytest.mark.parametrize("field,value", [("issuer", ""), ("role", "admin"), ("expires_at", 3)])
def test_invalid_signature_metadata_rejected(odr, field, value):
    signed = sign_odr_receipt(odr, odr_test_key())
    signed["signatures"][0][field] = value
    for verifier in (verify_odr_document, verify):
        assert not verifier(signed, public_key=odr_test_key().public_key()).ok
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(signed, load_odr_schema())
    with pytest.raises(OdrSigningError):
        sign_odr_receipt(odr, odr_test_key(), **{field: value})


@pytest.mark.parametrize("mode", ["missing", "garbage", "ec", "unset", "empty", "valid"])
def test_cli_export_file_key_never_writes_on_failure(tmp_path, key_file, mode):
    native = DecisionReceipt.from_dict({"receipt_id": "test-file-custody", "verdict": "PASS"})
    source = tmp_path / "native.json"
    source.write_text(native.to_json())
    out = tmp_path / "output.odr.json"
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'aragora-verify/src'}")
    if mode == "missing":
        env[FILE_ENV] = str(tmp_path / "missing.pem")
    elif mode in ("garbage", "ec"):
        data = (
            b"garbage"
            if mode == "garbage"
            else ec.generate_private_key(ec.SECP256R1()).private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        key_file.write_bytes(data)
        env[FILE_ENV] = str(key_file)
    elif mode == "valid":
        env[FILE_ENV] = str(key_file)
    elif mode == "empty":
        env[FILE_ENV] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aragora.cli.main",
            "receipt",
            "export",
            str(source),
            "--format",
            "odr",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if mode in ("missing", "garbage", "ec"):
        assert result.returncode == 1
        assert "configured but could not be used" in result.stderr
        assert not out.exists()
    else:
        assert result.returncode == 0, result.stderr
        doc = json.loads(out.read_text())
        assert bool(doc["signatures"]) == (mode == "valid")
        key = odr_test_key().public_key() if mode == "valid" else None
        assert verify_odr_document(doc, public_key=key).ok
        assert verify(doc, public_key=key).ok
