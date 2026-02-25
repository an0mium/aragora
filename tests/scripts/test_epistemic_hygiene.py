"""Tests for the epistemic hygiene CI gate script."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from scripts.check_epistemic_hygiene import (
    CheckResult,
    Finding,
    validate_directory,
    validate_file,
    validate_receipt,
    main,
    HIGH_CONFIDENCE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Finding / CheckResult unit tests
# ---------------------------------------------------------------------------


class TestFinding:
    """Tests for the Finding dataclass."""

    def test_str_error(self):
        f = Finding(code="EH-001", severity="error", message="missing field")
        assert "[ERROR]" in str(f)
        assert "EH-001" in str(f)
        assert "missing field" in str(f)

    def test_str_warning(self):
        f = Finding(code="EH-003", severity="warning", message="no falsifiers")
        assert "[WARNING]" in str(f)
        assert "EH-003" in str(f)

    def test_str_with_file_path(self):
        f = Finding(code="EH-001", severity="error", message="bad", file_path="/a/b.json")
        assert "/a/b.json" in str(f)

    def test_str_without_file_path(self):
        f = Finding(code="EH-001", severity="error", message="bad")
        result = str(f)
        assert "()" not in result


class TestCheckResult:
    """Tests for the CheckResult dataclass."""

    def test_passed_no_errors(self):
        r = CheckResult()
        assert r.passed is True

    def test_passed_with_errors(self):
        r = CheckResult(errors=[Finding(code="X", severity="error", message="x")])
        assert r.passed is False

    def test_passed_with_only_warnings(self):
        r = CheckResult(warnings=[Finding(code="X", severity="warning", message="x")])
        assert r.passed is True

    def test_passed_strict_no_findings(self):
        r = CheckResult()
        assert r.passed_strict() is True

    def test_passed_strict_with_warnings(self):
        r = CheckResult(warnings=[Finding(code="X", severity="warning", message="x")])
        assert r.passed_strict() is False

    def test_merge(self):
        r1 = CheckResult(
            errors=[Finding(code="A", severity="error", message="a")],
            files_checked=1,
        )
        r2 = CheckResult(
            warnings=[Finding(code="B", severity="warning", message="b")],
            files_checked=2,
        )
        merged = r1.merge(r2)
        assert len(merged.errors) == 1
        assert len(merged.warnings) == 1
        assert merged.files_checked == 3


# ---------------------------------------------------------------------------
# validate_receipt tests
# ---------------------------------------------------------------------------


class TestValidateReceipt:
    """Tests for validate_receipt()."""

    def _valid_receipt(self, **overrides) -> dict:
        """Build a minimal valid receipt."""
        sm = {
            "debate_id": "debate-123",
            "settled_at": "2026-01-01T00:00:00Z",
            "confidence": 0.75,
            "falsifiers": ["F1"],
            "review_horizon": "2026-06-01",
            **overrides,
        }
        return {"settlement_metadata": sm}

    def test_valid_receipt_no_findings(self):
        result = validate_receipt(self._valid_receipt())
        assert result.passed is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    # EH-001: settlement_metadata missing/null/wrong type
    def test_eh001_missing_settlement_metadata(self):
        result = validate_receipt({})
        assert any(f.code == "EH-001" for f in result.errors)

    def test_eh001_null_settlement_metadata(self):
        result = validate_receipt({"settlement_metadata": None})
        assert any(f.code == "EH-001" for f in result.errors)

    def test_eh001_wrong_type(self):
        result = validate_receipt({"settlement_metadata": "not a dict"})
        assert any(f.code == "EH-001" for f in result.errors)

    # EH-002: Required keys
    def test_eh002_missing_required_keys(self):
        result = validate_receipt({"settlement_metadata": {}})
        assert any(f.code == "EH-002" for f in result.errors)

    def test_eh002_partial_keys(self):
        result = validate_receipt({
            "settlement_metadata": {"debate_id": "x", "settled_at": "y"}
        })
        errors_002 = [f for f in result.errors if f.code == "EH-002"]
        assert len(errors_002) == 1
        assert "confidence" in errors_002[0].message
        assert "falsifiers" in errors_002[0].message

    # EH-003: Falsifiers for high-confidence
    def test_eh003_high_confidence_no_falsifiers(self):
        receipt = self._valid_receipt(confidence=0.95, falsifiers=[])
        result = validate_receipt(receipt)
        assert any(f.code == "EH-003" for f in result.warnings)

    def test_eh003_low_confidence_no_falsifiers_ok(self):
        receipt = self._valid_receipt(confidence=0.5, falsifiers=[])
        result = validate_receipt(receipt)
        assert not any(f.code == "EH-003" for f in result.warnings)

    def test_eh003_high_confidence_with_falsifiers_ok(self):
        receipt = self._valid_receipt(confidence=0.95, falsifiers=["F1"])
        result = validate_receipt(receipt)
        assert not any(f.code == "EH-003" for f in result.warnings)

    # EH-004: review_horizon
    def test_eh004_no_review_horizon(self):
        receipt = self._valid_receipt()
        receipt["settlement_metadata"].pop("review_horizon")
        result = validate_receipt(receipt)
        assert any(f.code == "EH-004" for f in result.warnings)

    def test_eh004_empty_review_horizon(self):
        receipt = self._valid_receipt(review_horizon="")
        result = validate_receipt(receipt)
        assert any(f.code == "EH-004" for f in result.warnings)

    # EH-005: Dissent documentation
    def test_eh005_dissent_without_docs(self):
        receipt = self._valid_receipt()
        receipt["dissenting_views"] = [{"agent": "X", "view": "disagree"}]
        result = validate_receipt(receipt)
        assert any(f.code == "EH-005" for f in result.warnings)

    def test_eh005_dissent_with_cruxes(self):
        receipt = self._valid_receipt(cruxes=["crux1"])
        receipt["dissenting_views"] = [{"agent": "X", "view": "disagree"}]
        result = validate_receipt(receipt)
        assert not any(f.code == "EH-005" for f in result.warnings)

    def test_eh005_no_dissent_no_warning(self):
        receipt = self._valid_receipt()
        result = validate_receipt(receipt)
        assert not any(f.code == "EH-005" for f in result.warnings)

    # EH-006: Confidence validation
    def test_eh006_confidence_out_of_range(self):
        receipt = self._valid_receipt(confidence=1.5)
        result = validate_receipt(receipt)
        assert any(f.code == "EH-006" for f in result.errors)

    def test_eh006_confidence_negative(self):
        receipt = self._valid_receipt(confidence=-0.1)
        result = validate_receipt(receipt)
        assert any(f.code == "EH-006" for f in result.errors)

    def test_eh006_confidence_not_a_number(self):
        receipt = self._valid_receipt(confidence="abc")
        result = validate_receipt(receipt)
        assert any(f.code == "EH-006" for f in result.errors)

    def test_eh006_confidence_at_boundaries(self):
        for val in [0.0, 1.0]:
            receipt = self._valid_receipt(confidence=val)
            result = validate_receipt(receipt)
            assert not any(f.code == "EH-006" for f in result.errors)

    # EH-007: debate_id empty
    def test_eh007_empty_debate_id(self):
        receipt = self._valid_receipt(debate_id="   ")
        result = validate_receipt(receipt)
        assert any(f.code == "EH-007" for f in result.errors)

    # EH-008: settled_at empty
    def test_eh008_empty_settled_at(self):
        receipt = self._valid_receipt(settled_at="  ")
        result = validate_receipt(receipt)
        assert any(f.code == "EH-008" for f in result.errors)

    # Strict mode
    def test_strict_mode_promotes_warnings(self):
        receipt = self._valid_receipt(confidence=0.95, falsifiers=[])
        result = validate_receipt(receipt, strict=True)
        # EH-003 warning should now be an error
        assert any(f.code == "EH-003" for f in result.errors)
        assert not any(f.code == "EH-003" for f in result.warnings)


# ---------------------------------------------------------------------------
# validate_file tests
# ---------------------------------------------------------------------------


class TestValidateFile:
    """Tests for validate_file()."""

    def test_valid_json_file(self, tmp_path):
        f = tmp_path / "receipt.json"
        f.write_text(json.dumps({
            "settlement_metadata": {
                "debate_id": "d1",
                "settled_at": "2026-01-01",
                "confidence": 0.7,
                "falsifiers": ["f1"],
                "review_horizon": "2026-06-01",
            }
        }))
        result = validate_file(f)
        assert result.passed is True
        assert result.files_checked == 1

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        result = validate_file(f)
        assert result.passed is False
        assert any(f_.code == "EH-000" for f_ in result.errors)

    def test_file_not_found(self, tmp_path):
        result = validate_file(tmp_path / "nonexistent.json")
        assert result.passed is False
        assert any(f_.code == "EH-000" for f_ in result.errors)

    def test_non_object_root(self, tmp_path):
        f = tmp_path / "list.json"
        f.write_text("[1, 2, 3]")
        result = validate_file(f)
        assert result.passed is False
        assert any(f_.code == "EH-000" for f_ in result.errors)

    def test_file_path_attached_to_findings(self, tmp_path):
        f = tmp_path / "receipt.json"
        f.write_text("{}")
        result = validate_file(f)
        for finding in result.errors:
            assert str(f) in finding.file_path


# ---------------------------------------------------------------------------
# validate_directory tests
# ---------------------------------------------------------------------------


class TestValidateDirectory:
    """Tests for validate_directory()."""

    def test_multiple_files(self, tmp_path):
        for i in range(3):
            f = tmp_path / f"receipt_{i}.json"
            f.write_text(json.dumps({
                "settlement_metadata": {
                    "debate_id": f"d{i}",
                    "settled_at": "2026-01-01",
                    "confidence": 0.5,
                    "falsifiers": [],
                    "review_horizon": "2026-06-01",
                }
            }))
        result = validate_directory(tmp_path)
        assert result.files_checked == 3
        assert result.passed is True

    def test_empty_directory(self, tmp_path):
        result = validate_directory(tmp_path)
        assert result.files_checked == 0
        assert result.passed is True

    def test_mixed_valid_invalid(self, tmp_path):
        good = tmp_path / "good.json"
        good.write_text(json.dumps({
            "settlement_metadata": {
                "debate_id": "d1",
                "settled_at": "2026-01-01",
                "confidence": 0.5,
                "falsifiers": [],
                "review_horizon": "2026-06-01",
            }
        }))
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid")
        result = validate_directory(tmp_path)
        assert result.files_checked == 2
        assert result.passed is False


# ---------------------------------------------------------------------------
# main() CLI tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_check_file_valid(self, tmp_path):
        f = tmp_path / "receipt.json"
        f.write_text(json.dumps({
            "settlement_metadata": {
                "debate_id": "d1",
                "settled_at": "2026-01-01",
                "confidence": 0.5,
                "falsifiers": [],
                "review_horizon": "2026-06-01",
            }
        }))
        assert main(["--check-file", str(f)]) == 0

    def test_check_file_invalid(self, tmp_path):
        f = tmp_path / "receipt.json"
        f.write_text("{}")
        assert main(["--check-file", str(f)]) == 1

    def test_check_file_not_found(self, tmp_path):
        assert main(["--check-file", str(tmp_path / "missing.json")]) == 1

    def test_check_dir(self, tmp_path):
        f = tmp_path / "r.json"
        f.write_text(json.dumps({
            "settlement_metadata": {
                "debate_id": "d1",
                "settled_at": "2026-01-01",
                "confidence": 0.5,
                "falsifiers": [],
                "review_horizon": "2026-06-01",
            }
        }))
        assert main(["--check-dir", str(tmp_path)]) == 0

    def test_check_dir_not_a_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        assert main(["--check-dir", str(f)]) == 1

    def test_strict_mode(self, tmp_path):
        f = tmp_path / "receipt.json"
        f.write_text(json.dumps({
            "settlement_metadata": {
                "debate_id": "d1",
                "settled_at": "2026-01-01",
                "confidence": 0.95,
                "falsifiers": [],
                "review_horizon": "2026-06-01",
            }
        }))
        # Non-strict: passes (only warning for no falsifiers at high confidence)
        assert main(["--check-file", str(f)]) == 0
        # Strict: fails (warning promoted to error)
        assert main(["--check-file", str(f), "--strict"]) == 1

    def test_json_output(self, tmp_path, capsys):
        f = tmp_path / "receipt.json"
        f.write_text(json.dumps({
            "settlement_metadata": {
                "debate_id": "d1",
                "settled_at": "2026-01-01",
                "confidence": 0.5,
                "falsifiers": [],
                "review_horizon": "2026-06-01",
            }
        }))
        main(["--check-file", str(f), "--json-output"])
        output = json.loads(capsys.readouterr().out)
        assert output["passed"] is True
        assert output["files_checked"] == 1

    def test_quiet_mode(self, tmp_path, capsys):
        f = tmp_path / "receipt.json"
        f.write_text("{}")
        main(["--check-file", str(f), "--quiet"])
        out = capsys.readouterr().out
        # Quiet mode still prints summary
        assert "file(s)" in out

    def test_no_args_exits(self):
        with pytest.raises(SystemExit):
            main([])
