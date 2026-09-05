"""File custody, producer failures, and published v0.1 compatibility."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
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


def test_file_custody_preserves_published_v01_signature_shape(odr, key_file, monkeypatch):
    monkeypatch.setenv(FILE_ENV, str(key_file))
    signed = sign_odr_if_configured(odr)
    assert signed["odr_version"] == "0.1"
    assert len(signed["signatures"]) == 1
    sig = signed["signatures"][0]
    assert set(sig) == {"alg", "key_id", "signature"}
    assert sig["alg"] == "Ed25519"
    assert sig["key_id"] == compute_key_id(odr_test_key().public_key())
    jsonschema.validate(signed, load_odr_schema())
    for verifier in (verify_odr_document, verify):
        assert verifier(signed, public_key=odr_test_key().public_key()).ok
        assert not verifier(signed, public_key=generate_signing_key().public_key()).ok
        tampered = copy.deepcopy(signed)
        tampered["reasoning"]["summary"] += " altered"
        assert not verifier(tampered, public_key=odr_test_key().public_key()).ok
        assert verifier(odr).ok


def test_explicit_secret_name_ignores_file_environment(key_file, monkeypatch):
    pem = key_file.read_bytes()
    monkeypatch.setenv(FILE_ENV, "/nonexistent/key.pem")
    monkeypatch.setenv("ARAGORA_ODR_SIGNING_KEY_SECRET", "environment-secret")
    with (
        patch("aragora.gauntlet.odr_signing._load_pem_secret_from_aws", return_value=pem) as aws,
        patch.object(Path, "read_bytes", side_effect=AssertionError("file must not be read")),
    ):
        key = load_signing_key_from_secrets("explicit-secret")
    aws.assert_called_once_with("explicit-secret", explicitly_named=True)
    assert compute_key_id(key.public_key()) == compute_key_id(odr_test_key().public_key())


def test_file_environment_precedes_secret_environment_without_argument(key_file, monkeypatch):
    monkeypatch.setenv(FILE_ENV, str(key_file))
    monkeypatch.setenv("ARAGORA_ODR_SIGNING_KEY_SECRET", "must-not-be-read")
    with patch("aragora.gauntlet.odr_signing._load_pem_secret_from_aws") as aws:
        key = load_signing_key_from_secrets()
    aws.assert_not_called()
    assert compute_key_id(key.public_key()) == compute_key_id(odr_test_key().public_key())


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
    with patch("aragora.gauntlet.odr_signing._load_pem_secret_from_aws", return_value=pem):
        signed = sign_odr_if_configured(odr)
    assert set(signed["signatures"][0]) == {"alg", "key_id", "signature"}


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


def test_published_key_matches_documented_custody_record():
    from aragora.gauntlet.odr_verify import load_public_key

    kid = "ed25519-44c316618e9a0f58"
    path = ROOT / f"docs/specs/keys/aragora-odr-signing-{kid}.pub.pem"
    pem = path.read_bytes()
    assert b"PRIVATE" not in pem
    assert compute_key_id(load_public_key(pem)) == kid
    spec = (ROOT / "docs/specs/OPEN_DECISION_RECEIPT.md").read_text()
    assert path.name in spec
    assert "/.well-known/aragora-odr-signing-key" in spec
    assert "not a production trust anchor" in spec
    assert "revoke" in spec and "trusted channel" in spec


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
@pytest.mark.parametrize("mode", [0o664, 0o666])
def test_writable_key_permission_fails_closed(odr, key_file, monkeypatch, mode):
    key_file.chmod(mode)
    monkeypatch.setenv(FILE_ENV, str(key_file))
    with pytest.raises(OdrSigningError, match="writable by group or other") as caught:
        sign_odr_if_configured(odr)
    assert str(caught.value).startswith("ODR signing key file is configured but could not be used;")
    assert f"{mode:04o}" in str(caught.value)
    assert str(key_file) not in str(caught.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
@pytest.mark.parametrize("mode", [0o644, 0o444])
def test_readable_key_permission_warns_and_signs(odr, key_file, monkeypatch, caplog, mode):
    key_file.chmod(mode)
    monkeypatch.setenv(FILE_ENV, str(key_file))
    monkeypatch.delenv("ARAGORA_ODR_SIGNING_KEY_STRICT_MODE", raising=False)
    signed = sign_odr_if_configured(odr)
    assert set(signed["signatures"][0]) == {"alg", "key_id", "signature"}
    assert verify_odr_document(signed, public_key=odr_test_key().public_key()).ok
    assert verify(signed, public_key=odr_test_key().public_key()).ok
    warnings = [r.message for r in caplog.records if "readable by group or other" in r.message]
    assert len(warnings) == 1
    assert f"{mode:04o}" in warnings[0]
    assert "ARAGORA_ODR_SIGNING_KEY_STRICT_MODE" in warnings[0]
    assert str(key_file) not in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
@pytest.mark.parametrize("strict", ["true", "1", "YES", "On"])
def test_strict_readable_key_permission_fails_closed(key_file, monkeypatch, strict):
    key_file.chmod(0o644)
    monkeypatch.setenv(FILE_ENV, str(key_file))
    monkeypatch.setenv("ARAGORA_ODR_SIGNING_KEY_STRICT_MODE", strict)
    with pytest.raises(OdrSigningError, match="strict mode") as caught:
        load_signing_key_from_secrets()
    assert str(caught.value).startswith("ODR signing key file is configured but could not be used;")
    assert "0644" in str(caught.value)
    assert str(key_file) not in str(caught.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_strict_private_key_permission_signs(odr, key_file, monkeypatch, caplog):
    key_file.chmod(0o600)
    monkeypatch.setenv(FILE_ENV, str(key_file))
    monkeypatch.setenv("ARAGORA_ODR_SIGNING_KEY_STRICT_MODE", "true")
    signed = sign_odr_if_configured(odr)
    assert verify(signed, public_key=odr_test_key().public_key()).ok
    assert "readable by group or other" not in caplog.text


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_non_regular_key_permission_fails_before_read(tmp_path, monkeypatch):
    monkeypatch.setenv(FILE_ENV, str(tmp_path))
    with patch.object(Path, "read_bytes", side_effect=AssertionError("must not read directory")):
        with pytest.raises(OdrSigningError, match="not a regular file") as caught:
            load_signing_key_from_secrets()
    assert str(caught.value).startswith("ODR signing key file is configured but could not be used;")
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink permissions")
def test_symlink_permission_checks_target(key_file, tmp_path, monkeypatch):
    link = tmp_path / "link.pem"
    link.symlink_to(key_file)
    key_file.chmod(0o666)
    monkeypatch.setenv(FILE_ENV, str(link))
    with pytest.raises(OdrSigningError, match="writable by group or other"):
        load_signing_key_from_secrets()
    key_file.chmod(0o600)
    assert compute_key_id(load_signing_key_from_secrets().public_key()) == compute_key_id(
        odr_test_key().public_key()
    )


def test_non_posix_skips_permission_check(key_file, monkeypatch):
    from types import SimpleNamespace
    from aragora.gauntlet import odr_signing

    key_file.chmod(0o666)
    monkeypatch.setenv(FILE_ENV, str(key_file))
    monkeypatch.setenv("ARAGORA_ODR_SIGNING_KEY_STRICT_MODE", "true")
    with patch.object(odr_signing, "os", SimpleNamespace(name="nt", environ=os.environ)):
        assert compute_key_id(load_signing_key_from_secrets().public_key()) == compute_key_id(
            odr_test_key().public_key()
        )


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, ValueError, OdrSigningError])
def test_unusable_file_logs_exception_class_only(key_file, monkeypatch, caplog, error):
    key_file.chmod(0o600)
    monkeypatch.setenv(FILE_ENV, str(key_file))
    secret_message = f"sensitive exception message {key_file}"
    with patch.object(Path, "read_bytes", side_effect=error(secret_message)):
        with pytest.raises(OdrSigningError, match="expected a readable PKCS#8") as caught:
            load_signing_key_from_secrets()
    assert error.__name__ in caplog.text
    assert secret_message not in caplog.text
    assert str(key_file) not in caplog.text + str(caught.value)
    assert caught.value.__suppress_context__
