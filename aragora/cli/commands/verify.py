"""
CLI command: aragora verify -- validate decision receipt integrity.

Verifies that a decision receipt JSON file has not been tampered with by:
- Recomputing the SHA-256 checksum and comparing it to the stored value
- Checking that required fields (schema_version, verdict, timestamp) are present
- Validating the verdict against the canonical Verdict enum
- Validating the timestamp is valid ISO 8601 format
- If the receipt is signed, verifying the cryptographic signature chain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_verify_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'verify' subcommand."""
    parser = subparsers.add_parser(
        "verify",
        help="Verify a decision receipt's integrity",
        description=(
            "Validate that a decision receipt JSON file has not been tampered with. "
            "Recomputes the SHA-256 checksum, checks schema version, validates the "
            "verdict against the canonical enum, and verifies timestamp format. "
            "For signed receipts, also verifies the cryptographic signature chain."
        ),
    )
    parser.add_argument(
        "receipt_path",
        help="Path to the decision receipt JSON file",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full hash chain details",
    )
    parser.set_defaults(func=cmd_verify)


# ---------------------------------------------------------------------------
# Verdict validation
# ---------------------------------------------------------------------------

# Canonical verdict values from aragora.core_types.Verdict
_VALID_VERDICTS = frozenset(
    {
        "approved",
        "approved_with_conditions",
        "needs_review",
        "rejected",
        # Legacy / gauntlet aliases (case-insensitive matching applied separately)
        "pass",
        "fail",
        "conditional",
    }
)


def _is_valid_verdict(value: str) -> bool:
    """Check whether *value* is a recognised Verdict string."""
    return value.lower() in _VALID_VERDICTS


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------


def _is_valid_iso_timestamp(value: str) -> bool:
    """Return True if *value* can be parsed as an ISO 8601 timestamp."""
    try:
        datetime.fromisoformat(value)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------


def _recompute_checksum(data: dict[str, Any]) -> str:
    """Recompute the receipt checksum the same way DecisionReceipt does."""
    content = json.dumps(
        {
            "receipt_id": data.get("receipt_id", ""),
            "verdict": data.get("verdict", ""),
            "confidence": data.get("confidence", 0.0),
            "findings_count": len(data.get("findings", [])),
            "critical_count": data.get("critical_count", 0),
            "timestamp": data.get("timestamp", ""),
            "audit_trail_id": data.get("audit_trail_id"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def _verify_receipt(data: dict[str, Any], *, verbose: bool = False) -> dict[str, Any]:
    """Run all verification checks on *data* and return a result dict.

    Returns a dict with keys:
        valid (bool): overall pass/fail
        checks (list[dict]): individual check results
        receipt_id (str): the receipt_id if present
        signed (bool): whether the receipt has a signature
    """
    checks: list[dict[str, Any]] = []
    overall_valid = True

    # -- 1. schema_version present ----------------------------------------
    schema_version = data.get("schema_version")
    if schema_version:
        checks.append(
            {
                "name": "schema_version",
                "passed": True,
                "detail": f"schema_version={schema_version}",
            }
        )
    else:
        checks.append(
            {
                "name": "schema_version",
                "passed": False,
                "detail": "schema_version is missing",
            }
        )
        overall_valid = False

    # -- 2. verdict is valid Verdict enum value ---------------------------
    verdict = data.get("verdict")
    if verdict and _is_valid_verdict(verdict):
        checks.append(
            {
                "name": "verdict",
                "passed": True,
                "detail": f"verdict={verdict}",
            }
        )
    elif verdict:
        checks.append(
            {
                "name": "verdict",
                "passed": False,
                "detail": f"verdict '{verdict}' is not a recognised Verdict value",
            }
        )
        overall_valid = False
    else:
        checks.append(
            {
                "name": "verdict",
                "passed": False,
                "detail": "verdict is missing",
            }
        )
        overall_valid = False

    # -- 3. timestamp is valid ISO format ---------------------------------
    timestamp = data.get("timestamp")
    if timestamp and _is_valid_iso_timestamp(timestamp):
        checks.append(
            {
                "name": "timestamp",
                "passed": True,
                "detail": f"timestamp={timestamp}",
            }
        )
    elif timestamp:
        checks.append(
            {
                "name": "timestamp",
                "passed": False,
                "detail": f"timestamp '{timestamp}' is not valid ISO 8601",
            }
        )
        overall_valid = False
    else:
        checks.append(
            {
                "name": "timestamp",
                "passed": False,
                "detail": "timestamp is missing",
            }
        )
        overall_valid = False

    # -- 4. checksum integrity --------------------------------------------
    stored_checksum = data.get("checksum")
    if stored_checksum:
        expected = _recompute_checksum(data)
        if stored_checksum == expected:
            detail = f"checksum={stored_checksum}"
            if verbose:
                detail += f" (recomputed={expected})"
            checks.append(
                {
                    "name": "checksum",
                    "passed": True,
                    "detail": detail,
                }
            )
        else:
            checks.append(
                {
                    "name": "checksum",
                    "passed": False,
                    "detail": (
                        f"checksum mismatch: stored={stored_checksum}, "
                        f"recomputed={expected}"
                    ),
                }
            )
            overall_valid = False
    else:
        checks.append(
            {
                "name": "checksum",
                "passed": False,
                "detail": "checksum is missing",
            }
        )
        overall_valid = False

    # -- 5. signature chain (optional, only for signed receipts) ----------
    is_signed = False
    signature_data = data.get("signature")
    signature_metadata = data.get("signature_metadata")

    if signature_data and signature_metadata:
        is_signed = True
        try:
            from aragora.export.decision_receipt import SignedDecisionReceipt

            signed_receipt = SignedDecisionReceipt.from_dict(data)
            sig_valid = signed_receipt.verify()
            checks.append(
                {
                    "name": "signature",
                    "passed": sig_valid,
                    "detail": (
                        "signature verified"
                        if sig_valid
                        else "signature verification failed"
                    ),
                }
            )
            if not sig_valid:
                overall_valid = False
        except ImportError:
            checks.append(
                {
                    "name": "signature",
                    "passed": False,
                    "detail": "signing backend not available for verification",
                }
            )
            overall_valid = False
        except Exception as exc:
            checks.append(
                {
                    "name": "signature",
                    "passed": False,
                    "detail": f"signature verification error: {exc}",
                }
            )
            overall_valid = False

    return {
        "valid": overall_valid,
        "checks": checks,
        "receipt_id": data.get("receipt_id", "unknown"),
        "signed": is_signed,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a decision receipt's integrity.

    Returns 0 if the receipt is valid, 1 otherwise.
    """
    receipt_path = Path(args.receipt_path)
    output_format: str = getattr(args, "output_format", "text")
    verbose: bool = getattr(args, "verbose", False)

    # -- Load the file ----------------------------------------------------
    if not receipt_path.exists():
        _report_error(
            f"File not found: {receipt_path}",
            output_format=output_format,
        )
        return 1

    try:
        raw = receipt_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _report_error(
            f"Invalid JSON: {exc}",
            output_format=output_format,
        )
        return 1

    if not isinstance(data, dict):
        _report_error(
            "Receipt JSON must be an object (dict), not a list or scalar",
            output_format=output_format,
        )
        return 1

    # -- Run checks -------------------------------------------------------
    result = _verify_receipt(data, verbose=verbose)

    # -- Output -----------------------------------------------------------
    if output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        _print_text_report(result, verbose=verbose)

    return 0 if result["valid"] else 1


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_text_report(result: dict[str, Any], *, verbose: bool = False) -> None:
    """Pretty-print verification results to stdout."""
    receipt_id = result.get("receipt_id", "unknown")
    print(f"\nReceipt Verification: {receipt_id}")
    print("=" * 60)

    for check in result["checks"]:
        icon = "PASS" if check["passed"] else "FAIL"
        print(f"  [{icon}] {check['name']}: {check['detail']}")

    print("")
    if result["valid"]:
        print("Result: VALID -- receipt integrity verified")
    else:
        print("Result: INVALID -- receipt integrity check failed")

    if result.get("signed"):
        print("  (receipt is cryptographically signed)")
    print("")


def _report_error(message: str, *, output_format: str = "text") -> None:
    """Report an error in the requested format."""
    if output_format == "json":
        print(
            json.dumps(
                {"valid": False, "error": message, "checks": []},
                indent=2,
            )
        )
    else:
        print(f"Error: {message}", file=sys.stderr)
