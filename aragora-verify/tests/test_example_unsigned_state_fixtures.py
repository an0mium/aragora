"""Independent verification of the new unsigned example-state fixtures
(m2-odr-unsigned-state-fixtures): approved-clean, blocked/FAIL, and
abstained/inconclusive. Mirrors test_example_live_receipt.py /
test_example_merge_quorum_receipt.py (dict-level `verify()`) and additionally
drives the packaged `aragora-verify` CLI end to end -- this feature's contract
is stated in exit-code + `--json` terms, so the CLI is exercised directly via
`aragora_verify.cli.main()` (no subprocess needed; same entry point the
console script uses). The committed example JSON files are the only artifacts
crossing the package boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aragora_verify import verify
from aragora_verify.cli import main
from aragora_verify.verifier import FAIL, PASS

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "specs" / "examples"

FIXTURES = [
    "example-approved-clean.odr.json",
    "example-blocked.odr.json",
    "example-abstained.odr.json",
]


def _load(filename: str) -> dict[str, Any]:
    return json.loads((EXAMPLES_DIR / filename).read_text(encoding="utf-8"))


def _write(tmp_path: Path, name: str, doc: dict[str, Any]) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _check(result: Any, name: str) -> Any:
    check = next((c for c in result.checks if c.name == name), None)
    assert check is not None, f"check {name!r} not found in {[c.name for c in result.checks]}"
    return check


@pytest.mark.parametrize("filename", FIXTURES)
def test_new_fixture_verifies_independently(filename: str) -> None:
    doc = _load(filename)
    result = verify(doc)  # unsigned: the "signature" check is WARN, not FAIL
    failed = [c for c in result.checks if c.status == FAIL]
    assert result.ok, failed
    assert not failed
    assert _check(result, "schema_conformance").status == PASS
    assert _check(result, "canonical_digest").status == PASS


@pytest.mark.parametrize("filename", FIXTURES)
def test_new_fixture_cli_exits_zero_with_unsigned_warning(filename: str) -> None:
    rc = main([str(EXAMPLES_DIR / filename)])
    assert rc == 0


@pytest.mark.parametrize("filename", FIXTURES)
def test_new_fixture_single_byte_tamper_cli_exits_one(filename: str, tmp_path: Path) -> None:
    raw_text = (EXAMPLES_DIR / filename).read_text(encoding="utf-8")
    marker = '"odr_version": "0.1"'
    assert marker in raw_text
    tampered_text = raw_text.replace(marker, '"odr_version": "0.2"', 1)
    path = tmp_path / filename
    path.write_text(tampered_text, encoding="utf-8")
    rc = main([str(path)])
    assert rc == 1


@pytest.mark.parametrize("filename", ["example-blocked.odr.json", "example-abstained.odr.json"])
def test_weakening_signals_surface_in_json_warnings(filename: str, capsys) -> None:
    rc = main([str(EXAMPLES_DIR / filename), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert len(payload["warnings"]) >= 1


def test_blocked_fixture_json_warnings_name_expected_categories(capsys) -> None:
    rc = main([str(EXAMPLES_DIR / "example-blocked.odr.json"), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    joined = " ".join(payload["warnings"])
    assert "attestation: autonomous" in joined
    assert "undisclosed" in joined
    assert "uncalibrated" in joined


def test_abstained_fixture_json_warnings_name_expected_categories(capsys) -> None:
    rc = main([str(EXAMPLES_DIR / "example-abstained.odr.json"), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    joined = " ".join(payload["warnings"])
    assert "attestation: autonomous" in joined
    assert "quorum: absent" in joined
    assert "reasoning: absent" in joined


def test_quorum_consistency_tamper_cli_exits_one_with_failing_check(tmp_path: Path, capsys) -> None:
    doc = _load("example-blocked.odr.json")
    doc["quorum"]["dissent"]["dissenting_agents"] = ["ghost-agent"]
    doc["quorum"]["dissent"]["present"] = True
    path = _write(tmp_path, "blocked-quorum-tamper.odr.json", doc)
    rc = main([path, "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    checks_by_name = {c["name"]: c for c in payload["checks"]}
    assert checks_by_name["schema_conformance"]["status"] == "pass"
    assert checks_by_name["quorum_consistency"]["status"] == "fail"
    assert "ghost-agent" in checks_by_name["quorum_consistency"]["detail"]
