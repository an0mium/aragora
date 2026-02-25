#!/usr/bin/env python3
"""Epistemic hygiene CI gate.

Validates that debate receipts contain proper settlement metadata,
falsifiers, confidence horizons, and dissent documentation.

Usage:
    python scripts/check_epistemic_hygiene.py --check-file receipt.json
    python scripts/check_epistemic_hygiene.py --check-dir receipts/
    python scripts/check_epistemic_hygiene.py --check-file receipt.json --strict

Exit codes:
    0 - All checks passed
    1 - Errors found (or warnings in --strict mode)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HIGH_CONFIDENCE_THRESHOLD = 0.8


@dataclass
class Finding:
    """A single validation finding."""

    code: str
    severity: str  # "error" | "warning"
    message: str
    file_path: str = ""

    def __str__(self) -> str:
        prefix = f"[{self.severity.upper()}]"
        loc = f" ({self.file_path})" if self.file_path else ""
        return f"{prefix} {self.code}: {self.message}{loc}"


@dataclass
class CheckResult:
    """Result of validating one or more receipts."""

    errors: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    files_checked: int = 0

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def passed_strict(self) -> bool:
        return len(self.errors) == 0 and len(self.warnings) == 0

    def merge(self, other: CheckResult) -> CheckResult:
        return CheckResult(
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            files_checked=self.files_checked + other.files_checked,
        )


def validate_receipt(receipt: dict[str, Any], strict: bool = False) -> CheckResult:
    """Validate a single receipt dict for epistemic hygiene.

    Args:
        receipt: The receipt dictionary to validate.
        strict: If True, warnings become errors.

    Returns:
        CheckResult with any findings.
    """
    result = CheckResult(files_checked=1)

    def _add(code: str, severity: str, message: str) -> None:
        effective = "error" if strict and severity == "warning" else severity
        finding = Finding(code=code, severity=effective, message=message)
        if effective == "error":
            result.errors.append(finding)
        else:
            result.warnings.append(finding)

    # EH-001: settlement_metadata must exist and be a dict
    sm = receipt.get("settlement_metadata")
    if sm is None:
        _add("EH-001", "error", "settlement_metadata is missing or null")
        return result
    if not isinstance(sm, dict):
        _add("EH-001", "error", f"settlement_metadata is {type(sm).__name__}, expected dict")
        return result

    # EH-002: Required keys
    required_keys = {"debate_id", "settled_at", "confidence", "falsifiers"}
    missing = required_keys - set(sm.keys())
    if missing:
        _add("EH-002", "error", f"Missing required keys: {', '.join(sorted(missing))}")

    # EH-006: Confidence is valid number in [0, 1]
    confidence = sm.get("confidence")
    if confidence is not None:
        try:
            conf_val = float(confidence)
            if not (0.0 <= conf_val <= 1.0):
                _add("EH-006", "error", f"confidence {conf_val} not in [0, 1]")
        except (ValueError, TypeError):
            _add("EH-006", "error", f"confidence is not a valid number: {confidence!r}")
            conf_val = 0.0
    else:
        conf_val = 0.0

    # EH-007: debate_id non-empty
    debate_id = sm.get("debate_id")
    if debate_id is not None and not str(debate_id).strip():
        _add("EH-007", "error", "debate_id is empty")

    # EH-008: settled_at non-empty
    settled_at = sm.get("settled_at")
    if settled_at is not None and not str(settled_at).strip():
        _add("EH-008", "error", "settled_at is empty")

    # EH-003: Falsifiers non-empty for high-confidence verdicts
    falsifiers = sm.get("falsifiers", [])
    if conf_val >= HIGH_CONFIDENCE_THRESHOLD and isinstance(falsifiers, list) and len(falsifiers) == 0:
        _add("EH-003", "warning", f"No falsifiers for high-confidence verdict ({conf_val:.0%})")

    # EH-004: review_horizon (confidence horizon) is set
    review_horizon = sm.get("review_horizon")
    if not review_horizon:
        _add("EH-004", "warning", "review_horizon (confidence horizon) is not set")

    # EH-005: Dissent documentation when dissenting_views present
    dissenting = receipt.get("dissenting_views") or receipt.get("dissent") or []
    if dissenting and isinstance(dissenting, list) and len(dissenting) > 0:
        has_docs = bool(
            sm.get("cruxes")
            or sm.get("alternatives")
            or sm.get("review_notes")
        )
        if not has_docs:
            _add(
                "EH-005",
                "warning",
                f"Dissenting views present ({len(dissenting)}) but no cruxes/alternatives/review_notes",
            )

    return result


def validate_file(path: Path, strict: bool = False) -> CheckResult:
    """Validate a single JSON file."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return CheckResult(
            errors=[Finding(code="EH-000", severity="error", message=f"Invalid JSON: {e}", file_path=str(path))],
            files_checked=1,
        )
    except FileNotFoundError:
        return CheckResult(
            errors=[Finding(code="EH-000", severity="error", message="File not found", file_path=str(path))],
            files_checked=1,
        )

    if not isinstance(data, dict):
        return CheckResult(
            errors=[Finding(code="EH-000", severity="error", message="Root is not an object", file_path=str(path))],
            files_checked=1,
        )

    result = validate_receipt(data, strict=strict)
    for f in result.errors + result.warnings:
        f.file_path = str(path)
    return result


def validate_directory(dir_path: Path, strict: bool = False) -> CheckResult:
    """Validate all JSON files in a directory."""
    combined = CheckResult()
    json_files = sorted(dir_path.glob("*.json"))
    for f in json_files:
        combined = combined.merge(validate_file(f, strict=strict))
    return combined


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Epistemic hygiene CI gate")
    parser.add_argument("--check-file", type=Path, help="Validate a single receipt JSON")
    parser.add_argument("--check-dir", type=Path, help="Validate all JSONs in a directory")
    parser.add_argument("--strict", action="store_true", help="Warnings become errors")
    parser.add_argument("--quiet", action="store_true", help="Only print summary")
    parser.add_argument("--json-output", action="store_true", help="Emit JSON result")

    args = parser.parse_args(argv)

    if not args.check_file and not args.check_dir:
        parser.error("Must specify --check-file or --check-dir")

    result = CheckResult()

    if args.check_file:
        if not args.check_file.exists():
            print(f"Error: {args.check_file} not found", file=sys.stderr)
            return 1
        result = result.merge(validate_file(args.check_file, strict=args.strict))

    if args.check_dir:
        if not args.check_dir.is_dir():
            print(f"Error: {args.check_dir} is not a directory", file=sys.stderr)
            return 1
        result = result.merge(validate_directory(args.check_dir, strict=args.strict))

    if args.json_output:
        output = {
            "passed": result.passed_strict() if args.strict else result.passed,
            "files_checked": result.files_checked,
            "errors": [str(f) for f in result.errors],
            "warnings": [str(f) for f in result.warnings],
        }
        print(json.dumps(output, indent=2))
    elif not args.quiet:
        for f in result.errors:
            print(f)
        for f in result.warnings:
            print(f)

    # Summary
    if not args.json_output:
        total_findings = len(result.errors) + len(result.warnings)
        if total_findings == 0:
            print(f"Epistemic hygiene: {result.files_checked} file(s) checked, all clean")
        else:
            print(
                f"Epistemic hygiene: {result.files_checked} file(s), "
                f"{len(result.errors)} error(s), {len(result.warnings)} warning(s)"
            )

    passed = result.passed_strict() if args.strict else result.passed
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
