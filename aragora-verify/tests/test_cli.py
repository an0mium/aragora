"""CLI exit codes and rendering."""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization

from aragora_verify.cli import main

from _fixtures import make_keypair, sign_odr, valid_odr


def _write(tmp_path: Path, name: str, doc) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _write_pem(tmp_path: Path, public_key) -> str:
    pem = public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    path = tmp_path / "key.pem"
    path.write_bytes(pem)
    return str(path)


def test_cli_valid_receipt_exits_zero(tmp_path: Path, capsys) -> None:
    rc = main([_write(tmp_path, "r.json", valid_odr())])
    assert rc == 0
    assert "VERIFIED" in capsys.readouterr().out


def test_cli_signed_receipt_with_key_exits_zero(tmp_path: Path) -> None:
    private_key, public_key = make_keypair()
    signed = sign_odr(valid_odr(), private_key)
    rc = main([_write(tmp_path, "r.json", signed), "--pubkey", _write_pem(tmp_path, public_key)])
    assert rc == 0


def test_cli_tampered_receipt_exits_one(tmp_path: Path, capsys) -> None:
    private_key, public_key = make_keypair()
    signed = sign_odr(valid_odr(), private_key)
    signed["claim"]["verdict"] = "FAIL"
    rc = main([_write(tmp_path, "r.json", signed), "--pubkey", _write_pem(tmp_path, public_key)])
    assert rc == 1
    assert "FAILED" in capsys.readouterr().out


def test_cli_schema_failure_exits_one(tmp_path: Path) -> None:
    doc = valid_odr()
    del doc["subject"]
    rc = main([_write(tmp_path, "r.json", doc)])
    assert rc == 1


def test_cli_missing_file_exits_two(capsys) -> None:
    rc = main(["/no/such/receipt.json"])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_cli_bad_json_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert main([str(path)]) == 2


def test_cli_json_output_is_machine_readable(tmp_path: Path, capsys) -> None:
    rc = main([_write(tmp_path, "r.json", valid_odr()), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert len(payload["odr_digest"]) == 64
    assert {c["name"] for c in payload["checks"]} >= {
        "schema_conformance",
        "canonical_digest",
        "signature",
    }
