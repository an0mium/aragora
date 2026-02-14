"""
Receipt CLI commands: view, verify, and export decision receipts.

Commands for managing decision receipts:
- view: Open receipt in browser (converts JSON to HTML automatically)
- verify: Verify a receipt's artifact hash and cryptographic signature
- inspect: Display receipt details in terminal
- export: Export receipt to different formats (html, md, json, sarif, pdf, csv)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def add_receipt_parser(subparsers: Any) -> None:
    """Register the 'receipt' subcommand with view/verify/inspect/export actions."""
    receipt_parser = subparsers.add_parser(
        "receipt",
        help="View, verify, and export decision receipts",
        description="""
Manage decision receipt files produced by debates, gauntlets, and reviews.

Subcommands:
  view    <file>             Open receipt in browser (JSON auto-converts to HTML)
  verify  <file>             Check artifact hash and signature integrity
  inspect <file>             Display receipt details in terminal
  export  <file> --format X  Convert between html, md, json, sarif, pdf, csv

Examples:
  aragora receipt view receipt.json
  aragora receipt verify receipt.json
  aragora receipt inspect receipt.json
  aragora receipt export receipt.json --format html --output receipt.html
  aragora receipt view receipt.json --no-browser  # Print HTML to stdout
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    receipt_sub = receipt_parser.add_subparsers(dest="receipt_command")

    # --- view ---
    view_parser = receipt_sub.add_parser(
        "view",
        help="Open a receipt in the browser",
    )
    view_parser.add_argument("receipt", help="Path to receipt file (.json or .html)")
    view_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print HTML to stdout instead of opening browser",
    )
    view_parser.set_defaults(func=_cmd_view)

    # --- verify ---
    verify_parser = receipt_sub.add_parser(
        "verify",
        help="Verify receipt artifact hash and signature integrity",
    )
    verify_parser.add_argument("receipt", help="Path to receipt JSON file")
    verify_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed hash comparison"
    )
    verify_parser.set_defaults(func=cmd_receipt_verify)

    # --- inspect ---
    inspect_parser = receipt_sub.add_parser(
        "inspect",
        help="Display detailed receipt information",
    )
    inspect_parser.add_argument("receipt", help="Path to receipt JSON file")
    inspect_parser.set_defaults(func=cmd_receipt_inspect)

    # --- export ---
    export_parser = receipt_sub.add_parser(
        "export",
        help="Export receipt to different formats",
    )
    export_parser.add_argument("receipt", help="Path to receipt JSON file")
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "html", "md", "markdown", "sarif", "pdf", "csv"],
        default="html",
        help="Output format (default: html)",
    )
    export_parser.add_argument(
        "--output", "-o", help="Output file path (default: prints to stdout for text formats)"
    )
    export_parser.set_defaults(func=cmd_receipt_export)

    # Default when just 'aragora receipt' is called
    receipt_parser.set_defaults(func=cmd_receipt, _parser=receipt_parser)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_receipt_json(path: Path) -> dict[str, Any] | None:
    """Load and parse a receipt JSON file.

    Returns the parsed dict, or None on error (with message printed).
    """
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return None

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"Error: Cannot read file: {e}", file=sys.stderr)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        print("Error: Receipt JSON must be an object, not a list or scalar", file=sys.stderr)
        return None

    return data


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_receipt(args: argparse.Namespace) -> None:
    """Handle 'receipt' command - route to subcommands or show help."""
    subcommand = getattr(args, "receipt_command", None)

    if subcommand == "view":
        _cmd_view(args)
    elif subcommand == "verify":
        cmd_receipt_verify(args)
    elif subcommand == "inspect":
        cmd_receipt_inspect(args)
    elif subcommand == "export":
        cmd_receipt_export(args)
    else:
        parser = getattr(args, "_parser", None)
        if parser:
            parser.print_help()
        else:
            print("Usage: aragora receipt {view,verify,inspect,export} <file>")
            print("Run 'aragora receipt --help' for details.")


def _cmd_view(args: argparse.Namespace) -> None:
    """Open a receipt in the browser."""
    from aragora.cli.receipt_formatter import receipt_to_html

    receipt_path = getattr(args, "receipt", None)
    if not receipt_path:
        print("Error: Receipt file path required", file=sys.stderr)
        sys.exit(1)

    file_path = Path(receipt_path)
    no_browser = getattr(args, "no_browser", False)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # If already HTML, open directly
    if file_path.suffix.lower() in (".html", ".htm"):
        if no_browser:
            print(file_path.read_text(encoding="utf-8"))
        else:
            webbrowser.open(f"file://{file_path.resolve()}")
            print(f"Opened {file_path} in browser.")
        return

    # Load JSON and convert to HTML
    data = _load_receipt_json(file_path)
    if data is None:
        sys.exit(1)

    # Try the full DecisionReceipt.to_html() for richer output, fallback to formatter
    try:
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt.from_dict(data)
        html = receipt.to_html()
    except (ImportError, AttributeError, KeyError, ValueError, TypeError):
        html = receipt_to_html(data)

    if no_browser:
        print(html)
    else:
        fd, tmp_path = tempfile.mkstemp(suffix=".html", prefix="aragora-receipt-")
        with os.fdopen(fd, "w") as f:
            f.write(html)
        webbrowser.open(f"file://{tmp_path}")
        print(f"Receipt opened in browser. Saved to {tmp_path}")


def cmd_receipt_verify(args: argparse.Namespace) -> None:
    """Verify a receipt's artifact hash and signature integrity."""
    receipt_path = getattr(args, "receipt", None)
    verbose = getattr(args, "verbose", False)

    if not receipt_path:
        print("Error: Receipt file path required", file=sys.stderr)
        sys.exit(1)

    path = Path(receipt_path)
    data = _load_receipt_json(path)
    if data is None:
        sys.exit(1)

    receipt_id = data.get("receipt_id", "unknown")
    stored_hash = data.get("artifact_hash", "")

    print(f"\nReceipt Verification: {receipt_id}")
    print("=" * 60)

    checks_passed = 0
    checks_total = 0

    # Check 1: artifact_hash present
    checks_total += 1
    if stored_hash:
        print(f"  [PASS] artifact_hash present: {stored_hash[:16]}...")
        checks_passed += 1
    else:
        print("  [FAIL] artifact_hash is missing")

    # Check 2: Recompute hash using DecisionReceipt logic
    checks_total += 1
    try:
        from aragora.gauntlet.receipt_models import DecisionReceipt

        receipt = DecisionReceipt.from_dict(data)
        if receipt.verify_integrity():
            detail = "integrity verified"
            if verbose:
                detail += f" (stored={stored_hash[:16]}..., recomputed={receipt._calculate_hash()[:16]}...)"
            print(f"  [PASS] {detail}")
            checks_passed += 1
        else:
            expected = receipt._calculate_hash()
            print(
                f"  [FAIL] hash mismatch: stored={stored_hash[:16]}..., expected={expected[:16]}..."
            )
    except ImportError:
        # Fallback: manual hash check
        import hashlib

        content = json.dumps(
            {
                "receipt_id": data.get("receipt_id", ""),
                "gauntlet_id": data.get("gauntlet_id", ""),
                "input_hash": data.get("input_hash", ""),
                "risk_summary": data.get("risk_summary", {}),
                "verdict": data.get("verdict", ""),
                "confidence": data.get("confidence", 0),
            },
            sort_keys=True,
        )
        expected = hashlib.sha256(content.encode()).hexdigest()
        if expected == stored_hash:
            print("  [PASS] hash recomputed and matches")
            checks_passed += 1
        else:
            print(
                f"  [FAIL] hash mismatch: stored={stored_hash[:16]}..., expected={expected[:16]}..."
            )

    # Check 3: Required fields present
    checks_total += 1
    required = ["receipt_id", "verdict", "timestamp", "confidence"]
    missing = [f for f in required if f not in data or data[f] in (None, "")]
    if not missing:
        print(f"  [PASS] required fields present ({', '.join(required)})")
        checks_passed += 1
    else:
        print(f"  [FAIL] missing required fields: {', '.join(missing)}")

    # Check 4: Signature (optional)
    if data.get("signature"):
        checks_total += 1
        try:
            from aragora.gauntlet.receipt_models import DecisionReceipt

            receipt_obj = DecisionReceipt.from_dict(data)
            if receipt_obj.verify_signature():
                print("  [PASS] cryptographic signature verified")
                checks_passed += 1
            else:
                print("  [FAIL] cryptographic signature invalid")
        except Exception as e:
            print(f"  [FAIL] signature verification error: {e}")

    print("")
    if checks_passed == checks_total:
        print(f"Result: VALID ({checks_passed}/{checks_total} checks passed)")
    else:
        print(f"Result: INVALID ({checks_passed}/{checks_total} checks passed)")
    print("")

    sys.exit(0 if checks_passed == checks_total else 1)


def cmd_receipt_inspect(args: argparse.Namespace) -> None:
    """Display detailed receipt information."""
    receipt_path = getattr(args, "receipt", None)

    if not receipt_path:
        print("Error: Receipt file path required", file=sys.stderr)
        sys.exit(1)

    path = Path(receipt_path)
    data = _load_receipt_json(path)
    if data is None:
        sys.exit(1)

    print("\nDecision Receipt")
    print("=" * 60)

    # Basic info
    print("\n--- Basic Information ---")
    print(f"Receipt ID:    {data.get('receipt_id', 'N/A')}")
    print(f"Gauntlet ID:   {data.get('gauntlet_id', 'N/A')}")
    print(f"Debate ID:     {data.get('debate_id', 'N/A')}")
    print(f"Timestamp:     {data.get('timestamp', 'N/A')}")

    # Verdict
    print("\n--- Verdict ---")
    verdict = data.get("verdict", "UNKNOWN")
    confidence = data.get("confidence", 0)
    robustness = data.get("robustness_score", 0)

    verdict_icon = {"PASS": "\u2713", "FAIL": "\u2717", "CONDITIONAL": "\u26a0"}.get(
        verdict.upper(), "?"
    )
    print(f"Verdict:       {verdict_icon} {verdict}")
    print(f"Confidence:    {confidence:.1%}")
    print(f"Robustness:    {robustness:.1%}")

    # Risk summary
    risk_summary = data.get("risk_summary", {})
    if risk_summary:
        print("\n--- Risk Summary ---")
        print(f"Critical:      {risk_summary.get('critical', 0)}")
        print(f"High:          {risk_summary.get('high', 0)}")
        print(f"Medium:        {risk_summary.get('medium', 0)}")
        print(f"Low:           {risk_summary.get('low', 0)}")
        print(f"Total:         {risk_summary.get('total', 0)}")

    # Consensus
    consensus = data.get("consensus_proof", {})
    if consensus:
        print("\n--- Consensus ---")
        print(f"Reached:       {'Yes' if consensus.get('reached') else 'No'}")
        print(f"Method:        {consensus.get('method', 'N/A')}")
        supporting = consensus.get("supporting_agents", [])
        dissenting = consensus.get("dissenting_agents", [])
        print(f"Supporting:    {', '.join(supporting) if supporting else 'None'}")
        print(f"Dissenting:    {', '.join(dissenting) if dissenting else 'None'}")

    # Signature
    print("\n--- Cryptographic ---")
    if data.get("signature"):
        print("Signed:        Yes")
        print(f"Algorithm:     {data.get('signature_algorithm', 'unknown')}")
        print(f"Key ID:        {data.get('signature_key_id', 'N/A')}")
    else:
        print("Signed:        No")

    if data.get("artifact_hash"):
        print(f"Artifact Hash: {data['artifact_hash'][:40]}...")

    if data.get("input_hash"):
        print(f"Input Hash:    {data['input_hash'][:40]}...")

    print("\n" + "=" * 60)


def cmd_receipt_export(args: argparse.Namespace) -> None:
    """Export receipt to different formats."""
    from aragora.cli.receipt_formatter import receipt_to_html, receipt_to_markdown

    receipt_path = getattr(args, "receipt", None)
    output_format = getattr(args, "format", "html")
    output_path = getattr(args, "output", None)

    if not receipt_path:
        print("Error: Receipt file path required", file=sys.stderr)
        sys.exit(1)

    path = Path(receipt_path)
    data = _load_receipt_json(path)
    if data is None:
        sys.exit(1)

    content: str | bytes

    if output_format in ("json",):
        content = json.dumps(data, indent=2, default=str)
    else:
        # Try the full DecisionReceipt for richer output
        try:
            from aragora.gauntlet.receipt_models import DecisionReceipt

            receipt = DecisionReceipt.from_dict(data)

            if output_format == "html":
                content = receipt.to_html()
            elif output_format in ("md", "markdown"):
                content = receipt.to_markdown()
            elif output_format == "sarif":
                content = receipt.to_sarif_json()
            elif output_format == "pdf":
                content = receipt.to_pdf()
            elif output_format == "csv":
                content = receipt.to_csv()
            else:
                content = receipt.to_json()
        except (ImportError, AttributeError, KeyError, ValueError, TypeError):
            # Fallback to simple formatter
            if output_format == "html":
                content = receipt_to_html(data)
            elif output_format in ("md", "markdown"):
                content = receipt_to_markdown(data)
            else:
                content = json.dumps(data, indent=2, default=str)

    if output_path:
        if isinstance(content, bytes):
            Path(output_path).write_bytes(content)
        else:
            Path(output_path).write_text(content)
        print(f"Exported to {output_path}")
    else:
        if isinstance(content, bytes):
            sys.stdout.buffer.write(content)
        else:
            print(content)


# Keep backward-compatible aliases
setup_receipt_parser = add_receipt_parser

__all__ = [
    "add_receipt_parser",
    "cmd_receipt",
    "cmd_receipt_verify",
    "cmd_receipt_inspect",
    "cmd_receipt_export",
    "setup_receipt_parser",
]
