"""
Receipt verification CLI commands.

Commands for verifying and inspecting decision receipts:
- verify: Verify a receipt's cryptographic signature
- inspect: Display receipt details
- export: Export receipt to different formats
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_receipt(args: argparse.Namespace) -> None:
    """Handle 'receipt' command - route to subcommands."""
    subcommand = getattr(args, "receipt_command", None)

    if subcommand == "verify":
        cmd_receipt_verify(args)
    elif subcommand == "inspect":
        cmd_receipt_inspect(args)
    elif subcommand == "export":
        cmd_receipt_export(args)
    else:
        print("Usage: aragora receipt <verify|inspect|export> [options]")
        print("\nSubcommands:")
        print("  verify   Verify a receipt's cryptographic signature")
        print("  inspect  Display receipt details")
        print("  export   Export receipt to different formats")
        sys.exit(1)


def cmd_receipt_verify(args: argparse.Namespace) -> None:
    """Verify a receipt's cryptographic signature."""
    receipt_path = getattr(args, "receipt", None)

    if not receipt_path:
        print("Error: Receipt file path required")
        print("Usage: aragora receipt verify <receipt.json>")
        sys.exit(1)

    path = Path(receipt_path)
    if not path.exists():
        print(f"Error: Receipt file not found: {receipt_path}")
        sys.exit(1)

    try:
        with open(path) as f:
            receipt_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in receipt file: {e}")
        sys.exit(1)

    print("\nReceipt Verification")
    print("=" * 60)

    # Display receipt info
    receipt_id = receipt_data.get("receipt_id", "unknown")
    verdict = receipt_data.get("verdict", "UNKNOWN")
    timestamp = receipt_data.get("timestamp", "unknown")

    print(f"\nReceipt ID: {receipt_id}")
    print(f"Verdict: {verdict}")
    print(f"Timestamp: {timestamp}")

    # Check signature
    signature = receipt_data.get("signature")
    if not signature:
        print("\n\u26a0 Warning: Receipt is unsigned")
        print("  This receipt has no cryptographic signature.")
        print("  It may have been generated without signing enabled.")
        sys.exit(0)

    print(f"\nSignature: {signature[:20]}...")

    # Verify signature
    try:
        from aragora.gauntlet.receipt import DecisionReceipt
        from aragora.gauntlet.signing import verify_receipt_signature

        # Reconstruct receipt for verification
        receipt = DecisionReceipt.from_dict(receipt_data)

        # Get signature algorithm
        algorithm = receipt_data.get("signature_algorithm", "hmac-sha256")
        print(f"Algorithm: {algorithm}")

        # Verify
        is_valid = verify_receipt_signature(receipt, signature, algorithm)

        if is_valid:
            print("\n\u2713 Signature VALID")
            print("  The receipt signature is cryptographically valid.")
            print("  The receipt has not been tampered with.")
        else:
            print("\n\u2717 Signature INVALID")
            print("  WARNING: The receipt signature does not match!")
            print("  The receipt may have been modified or the signing key is incorrect.")
            sys.exit(1)

    except ImportError as e:
        print(f"\n\u26a0 Warning: Could not verify signature: {e}")
        print("  Install required dependencies: pip install aragora[signing]")
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"\n\u2717 Verification failed: {e}")
        sys.exit(1)

    # Verify artifact hash if present
    artifact_hash = receipt_data.get("artifact_hash")
    if artifact_hash:
        print(f"\nArtifact Hash: {artifact_hash[:20]}...")

        # Recompute hash
        try:
            computed_hash = receipt.compute_artifact_hash()
            if computed_hash == artifact_hash:
                print("\u2713 Artifact hash verified")
            else:
                print("\u2717 Artifact hash mismatch!")
                print("  The receipt content may have been modified.")
                sys.exit(1)
        except (AttributeError, ValueError) as e:
            print(f"\u26a0 Could not verify artifact hash: {e}")

    # Check RFC 3161 timestamp if present
    timestamp_token = receipt_data.get("rfc3161_token")
    if timestamp_token:
        print("\n\u2713 RFC 3161 timestamp present")
        print("  This receipt has a trusted timestamp from a Time Stamp Authority.")
        tsa_url = receipt_data.get("tsa_url", "unknown")
        print(f"  TSA: {tsa_url}")

    print("\n" + "=" * 60)
    print("Verification complete.")


def cmd_receipt_inspect(args: argparse.Namespace) -> None:
    """Display detailed receipt information."""
    receipt_path = getattr(args, "receipt", None)

    if not receipt_path:
        print("Error: Receipt file path required")
        print("Usage: aragora receipt inspect <receipt.json>")
        sys.exit(1)

    path = Path(receipt_path)
    if not path.exists():
        print(f"Error: Receipt file not found: {receipt_path}")
        sys.exit(1)

    try:
        with open(path) as f:
            receipt_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in receipt file: {e}")
        sys.exit(1)

    print("\nDecision Receipt")
    print("=" * 60)

    # Basic info
    print("\n--- Basic Information ---")
    print(f"Receipt ID:    {receipt_data.get('receipt_id', 'N/A')}")
    print(f"Gauntlet ID:   {receipt_data.get('gauntlet_id', 'N/A')}")
    print(f"Debate ID:     {receipt_data.get('debate_id', 'N/A')}")
    print(f"Timestamp:     {receipt_data.get('timestamp', 'N/A')}")

    # Verdict
    print("\n--- Verdict ---")
    verdict = receipt_data.get("verdict", "UNKNOWN")
    confidence = receipt_data.get("confidence", 0)
    robustness = receipt_data.get("robustness_score", 0)

    verdict_icon = {"PASS": "\u2713", "FAIL": "\u2717", "CONDITIONAL": "\u26a0"}.get(
        verdict.upper(), "?"
    )
    print(f"Verdict:       {verdict_icon} {verdict}")
    print(f"Confidence:    {confidence:.1%}")
    print(f"Robustness:    {robustness:.1%}")

    # Risk summary
    risk_summary = receipt_data.get("risk_summary", {})
    if risk_summary:
        print("\n--- Risk Summary ---")
        print(f"Critical:      {risk_summary.get('critical', 0)}")
        print(f"High:          {risk_summary.get('high', 0)}")
        print(f"Medium:        {risk_summary.get('medium', 0)}")
        print(f"Low:           {risk_summary.get('low', 0)}")
        print(f"Total:         {risk_summary.get('total', 0)}")

    # Consensus
    consensus = receipt_data.get("consensus_proof", {})
    if consensus:
        print("\n--- Consensus ---")
        print(f"Reached:       {'Yes' if consensus.get('consensus_reached') else 'No'}")
        print(f"Method:        {consensus.get('consensus_method', 'N/A')}")
        supporting = consensus.get("supporting_agents", [])
        dissenting = consensus.get("dissenting_agents", [])
        print(f"Supporting:    {', '.join(supporting) if supporting else 'None'}")
        print(f"Dissenting:    {', '.join(dissenting) if dissenting else 'None'}")

    # Signature
    print("\n--- Cryptographic ---")
    if receipt_data.get("signature"):
        print("Signed:        Yes")
        print(f"Algorithm:     {receipt_data.get('signature_algorithm', 'unknown')}")
        print(f"Key ID:        {receipt_data.get('key_id', 'N/A')}")
    else:
        print("Signed:        No")

    if receipt_data.get("artifact_hash"):
        print(f"Artifact Hash: {receipt_data['artifact_hash'][:40]}...")

    if receipt_data.get("input_hash"):
        print(f"Input Hash:    {receipt_data['input_hash'][:40]}...")

    # Legal hold
    legal_hold = receipt_data.get("legal_hold")
    if legal_hold:
        print("\n--- Legal Hold ---")
        print("Active:        Yes")
        print(f"Matter ID:     {legal_hold.get('matter_id', 'N/A')}")
        print(f"Placed By:     {legal_hold.get('placed_by', 'N/A')}")
        print(f"Placed At:     {legal_hold.get('placed_at', 'N/A')}")

    print("\n" + "=" * 60)


def cmd_receipt_export(args: argparse.Namespace) -> None:
    """Export receipt to different formats."""
    receipt_path = getattr(args, "receipt", None)
    output_format = getattr(args, "format", "json")
    output_path = getattr(args, "output", None)

    if not receipt_path:
        print("Error: Receipt file path required")
        print("Usage: aragora receipt export <receipt.json> --format <format> --output <file>")
        sys.exit(1)

    path = Path(receipt_path)
    if not path.exists():
        print(f"Error: Receipt file not found: {receipt_path}")
        sys.exit(1)

    try:
        with open(path) as f:
            receipt_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in receipt file: {e}")
        sys.exit(1)

    try:
        from aragora.gauntlet.receipt import DecisionReceipt

        receipt = DecisionReceipt.from_dict(receipt_data)

        # Export based on format
        content: str | bytes
        if output_format == "json":
            content = receipt.to_json()
            extension = ".json"
        elif output_format == "html":
            content = receipt.to_html()
            extension = ".html"
        elif output_format == "md" or output_format == "markdown":
            content = receipt.to_markdown()
            extension = ".md"
        elif output_format == "sarif":
            content = str(receipt.to_sarif())
            extension = ".sarif"
        elif output_format == "pdf":
            content = receipt.to_pdf()
            extension = ".pdf"
        elif output_format == "csv":
            content = receipt.to_csv()
            extension = ".csv"
        else:
            print(f"Error: Unknown format: {output_format}")
            print("Supported formats: json, html, md, sarif, pdf, csv")
            sys.exit(1)

        # Determine output path
        if not output_path:
            output_path = f"receipt-{receipt.receipt_id[:12]}{extension}"

        # Write output
        if output_format == "pdf":
            with open(output_path, "wb") as fb:
                fb.write(content if isinstance(content, bytes) else content.encode())
        else:
            with open(output_path, "w") as ft:
                ft.write(content if isinstance(content, str) else content.decode())

        print(f"Receipt exported to: {output_path}")

    except ImportError as e:
        print(f"Error: Could not export receipt: {e}")
        print("Some export formats require additional dependencies.")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        print(f"Error: Export failed: {e}")
        sys.exit(1)


def setup_receipt_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up argument parser for receipt commands."""
    receipt_parser = subparsers.add_parser(
        "receipt",
        help="Receipt verification and inspection commands",
        description="Commands for verifying and inspecting decision receipts.",
    )

    receipt_subparsers = receipt_parser.add_subparsers(dest="receipt_command")

    # verify subcommand
    verify_parser = receipt_subparsers.add_parser(
        "verify",
        help="Verify a receipt's cryptographic signature",
    )
    verify_parser.add_argument(
        "receipt",
        help="Path to receipt JSON file",
    )

    # inspect subcommand
    inspect_parser = receipt_subparsers.add_parser(
        "inspect",
        help="Display detailed receipt information",
    )
    inspect_parser.add_argument(
        "receipt",
        help="Path to receipt JSON file",
    )

    # export subcommand
    export_parser = receipt_subparsers.add_parser(
        "export",
        help="Export receipt to different formats",
    )
    export_parser.add_argument(
        "receipt",
        help="Path to receipt JSON file",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "html", "md", "markdown", "sarif", "pdf", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: receipt-<id>.<ext>)",
    )

    receipt_parser.set_defaults(func=cmd_receipt)


__all__ = [
    "cmd_receipt",
    "cmd_receipt_verify",
    "cmd_receipt_inspect",
    "cmd_receipt_export",
    "setup_receipt_parser",
]
