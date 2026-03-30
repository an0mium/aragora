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
from datetime import datetime
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

    # --- list ---
    list_p = receipt_sub.add_parser("list", help="List recent decision receipts from the database")
    list_p.add_argument("--limit", "-n", type=int, default=20, help="Maximum results (default: 20)")
    list_p.add_argument(
        "--verdict", choices=["pass", "fail", "conditional"], help="Filter by verdict"
    )
    list_p.add_argument(
        "--kind",
        choices=["inbox", "decision", "other"],
        help="Filter by receipt kind",
    )
    list_p.add_argument("--org-id", help="Filter by organization ID")
    list_p.set_defaults(func=cmd_receipt_list)

    # --- show ---
    show_p = receipt_sub.add_parser("show", help="Show a specific receipt by ID")
    show_p.add_argument("id", help="Gauntlet/receipt ID to look up")
    show_p.add_argument(
        "--format",
        "-f",
        choices=["json", "md", "html"],
        default=None,
        help="Output format (default: terminal inspect view)",
    )
    show_p.add_argument("--org-id", help="Organization ID for ownership check")
    show_p.set_defaults(func=cmd_receipt_show)

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


def _format_receipt_created_at(value: Any) -> str:
    """Format storage/legacy receipt timestamps for CLI display."""
    if value is None:
        return "N/A"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M")
        except (OverflowError, OSError, TypeError, ValueError):
            return "N/A"
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return value[:16]
    return "N/A"


def _receipt_row_id(meta: Any) -> str:
    """Return the most useful receipt identifier for list output."""
    for field in ("receipt_id", "gauntlet_id"):
        value = getattr(meta, field, None)
        if value:
            return str(value)
    if isinstance(meta, dict):
        return str(meta.get("receipt_id") or meta.get("gauntlet_id") or "")
    return ""


def _receipt_findings_count(meta: Any) -> int:
    """Extract a best-effort findings count from storage or legacy metadata."""
    total_findings = getattr(meta, "total_findings", None)
    if total_findings is not None:
        return int(total_findings)

    data = getattr(meta, "data", None)
    if not isinstance(data, dict) and isinstance(meta, dict):
        data = meta
    if not isinstance(data, dict):
        return 0

    risk_summary = data.get("risk_summary")
    if isinstance(risk_summary, dict) and risk_summary.get("total") is not None:
        return int(risk_summary["total"])

    findings = data.get("findings")
    if isinstance(findings, list):
        return len(findings)

    vulnerabilities_found = data.get("vulnerabilities_found")
    if vulnerabilities_found is not None:
        return int(vulnerabilities_found)

    return 0


def _receipt_payload_dict(meta: Any) -> dict[str, Any]:
    """Return the richest receipt payload available for display normalization."""
    data = getattr(meta, "data", None)
    if isinstance(data, dict):
        return data
    if isinstance(meta, dict):
        return meta
    return {}


def _normalize_receipt_verdict_and_confidence(
    data: dict[str, Any],
    *,
    verdict: str | None,
    confidence: float | None,
) -> tuple[str, float]:
    """Fill missing verdict/confidence for trust-wedge receipts from nested metadata."""
    normalized_verdict = str(verdict or "").strip()
    normalized_confidence = float(confidence or 0.0)

    triage = data.get("triage_decision")
    if not isinstance(triage, dict):
        return normalized_verdict or "UNKNOWN", normalized_confidence

    triage_confidence = triage.get("confidence")
    if triage_confidence is not None:
        try:
            triage_confidence_value = float(triage_confidence)
        except (TypeError, ValueError):
            triage_confidence_value = None
        else:
            if normalized_confidence == 0.0 and triage_confidence_value != 0.0:
                normalized_confidence = triage_confidence_value

    verdict_missing = not normalized_verdict or normalized_verdict.upper() == "UNKNOWN"
    if verdict_missing:
        if bool(triage.get("blocked_by_policy")):
            normalized_verdict = "BLOCKED"
        else:
            state = str(data.get("state") or triage.get("receipt_state") or "").strip().lower()
            if state in {"approved", "executed"}:
                normalized_verdict = "PASS"
            elif state == "expired":
                normalized_verdict = "FAIL"
            elif state:
                normalized_verdict = "CONDITIONAL"

    return normalized_verdict or "UNKNOWN", normalized_confidence


def _normalize_receipt_payload_for_display(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay inferred verdict/confidence for legacy trust-wedge receipts."""
    normalized = dict(data)
    verdict, confidence = _normalize_receipt_verdict_and_confidence(
        normalized,
        verdict=normalized.get("verdict"),
        confidence=normalized.get("confidence"),
    )
    normalized["verdict"] = verdict
    normalized["confidence"] = confidence
    return normalized


def _receipt_kind(meta: Any) -> str:
    """Classify receipts for operator-facing list output."""
    data = _receipt_payload_dict(meta)
    if "action_intent" in data or "triage_decision" in data:
        return "inbox"
    if any(
        key in data
        for key in (
            "consensus_proof",
            "risk_summary",
            "agent_responses",
            "verdict_reasoning",
        )
    ):
        return "decision"
    return "other"


def _load_storage_receipt_list(limit: int, verdict: str | None) -> list[Any]:
    """Read receipt rows from the durable receipt store."""
    from aragora.storage.receipt_store import get_receipt_store

    store = get_receipt_store()
    return store.list(limit=limit, verdict=verdict.upper() if verdict else None)


def _load_legacy_receipt_list(limit: int, verdict: str | None, org_id: str | None) -> list[Any]:
    """Read receipt rows from the legacy gauntlet store."""
    from aragora.gauntlet.storage import get_storage

    storage = get_storage()
    return storage.list_recent(
        limit=limit,
        verdict=verdict.upper() if verdict else None,
        org_id=org_id,
    )


def _load_storage_receipt(receipt_id: str) -> dict[str, Any] | None:
    """Fetch a receipt from the durable store by receipt or gauntlet ID."""
    from aragora.storage.receipt_store import get_receipt_store

    store = get_receipt_store()
    stored = store.get(receipt_id)
    if stored is None:
        stored = store.get_by_gauntlet(receipt_id)
    return stored.to_full_dict() if stored is not None else None


def _load_legacy_receipt(receipt_id: str, org_id: str | None) -> dict[str, Any] | None:
    """Fetch a receipt from the legacy gauntlet store."""
    from aragora.gauntlet.storage import get_storage

    storage = get_storage()
    return storage.get(receipt_id, org_id=org_id)


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
    elif subcommand == "list":
        cmd_receipt_list(args)
    elif subcommand == "show":
        cmd_receipt_show(args)
    else:
        parser = getattr(args, "_parser", None)
        if parser:
            parser.print_help()
        else:
            print("Usage: aragora receipt {list,show,view,verify,inspect,export} ...")
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
        except (OSError, RuntimeError, ValueError) as e:
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

    # Verdict reasoning
    reasoning = data.get("verdict_reasoning", "")
    if reasoning:
        print("\n--- Verdict Reasoning ---")
        print(f"  {reasoning[:500]}")

    # Agent responses
    agent_responses = data.get("agent_responses", [])
    if agent_responses:
        print(f"\n--- Agent Responses ({len(agent_responses)}) ---")
        for resp in agent_responses[:10]:
            name = resp.get("agent_name", "unknown")
            role = resp.get("role", "")
            model = resp.get("llm_label", "")
            content = resp.get("content", "")
            length = len(content)
            label = f"{name}"
            if model:
                label += f" ({model})"
            if role:
                label += f" [{role}]"
            print(f"  {label}: {length} chars")

    # Cost summary
    cost = data.get("cost_summary")
    if cost and isinstance(cost, dict):
        total = cost.get("total_cost", cost.get("total", 0))
        if total:
            print("\n--- Cost ---")
            print(f"  Total: ${float(total):.4f}")

    # Critique summaries (from config_used)
    config = data.get("config_used", {})
    critiques = config.get("critique_summaries", [])
    if critiques:
        print(f"\n--- Critique Summaries ({len(critiques)}) ---")
        for c in critiques[:5]:
            critic = c.get("critic", "unknown")
            target = c.get("target", "")
            severity = c.get("severity", 0.0)
            issues = c.get("issues", [])
            print(f"  {critic} → {target} (severity: {severity:.1f})")
            for issue in issues[:3]:
                print(f"    - {str(issue)[:100]}")

    # Dissenting views
    dissent = data.get("dissenting_views", [])
    if dissent:
        print(f"\n--- Dissenting Views ({len(dissent)}) ---")
        for view in dissent[:3]:
            print(f"  - {str(view)[:200]}")

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


def cmd_receipt_list(args: argparse.Namespace) -> None:
    """List recent decision receipts from the database."""
    limit = getattr(args, "limit", 20)
    verdict = getattr(args, "verdict", None)
    kind = getattr(args, "kind", None)
    org_id = getattr(args, "org_id", None)

    results: list[Any] = []
    storage_error: Exception | None = None
    legacy_error: Exception | None = None

    if org_id:
        try:
            results = _load_legacy_receipt_list(limit, verdict, org_id)
        except ImportError:
            print("Error: Gauntlet storage module not available", file=sys.stderr)
            sys.exit(1)
        except (OSError, RuntimeError, ValueError) as e:
            print(f"Error: Could not access receipt database: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            results = _load_storage_receipt_list(limit, verdict)
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            storage_error = e

        if not results:
            try:
                results = _load_legacy_receipt_list(limit, verdict, org_id=None)
            except (ImportError, OSError, RuntimeError, ValueError) as e:
                legacy_error = e

        if not results and storage_error is not None and legacy_error is not None:
            print(f"Error: Could not access receipt database: {storage_error}", file=sys.stderr)
            sys.exit(1)

    if kind:
        results = [meta for meta in results if _receipt_kind(meta) == kind]

    if not results:
        print("No receipts found.")
        return

    print(f"{'ID':<14} {'TYPE':<10} {'VERDICT':<12} {'CONF':>6} {'FINDINGS':>8} {'CREATED':<20}")
    print("-" * 75)
    for meta in results:
        payload = _receipt_payload_dict(meta)
        row_id = _receipt_row_id(meta)
        short_id = row_id[:12] + ".." if len(row_id) > 14 else row_id
        receipt_kind = _receipt_kind(meta)
        created = _format_receipt_created_at(getattr(meta, "created_at", None))
        findings = _receipt_findings_count(meta)
        verdict_value, confidence = _normalize_receipt_verdict_and_confidence(
            payload,
            verdict=getattr(meta, "verdict", None),
            confidence=getattr(meta, "confidence", None),
        )
        print(
            f"{short_id:<14} {receipt_kind:<10} {verdict_value:<12} {confidence:>5.0%} {findings:>8} {created:<20}"
        )
    print(f"\n{len(results)} receipt(s) shown.")


def cmd_receipt_show(args: argparse.Namespace) -> None:
    """Show a specific receipt by ID."""
    receipt_id = getattr(args, "id", None)
    output_format = getattr(args, "format", None)
    org_id = getattr(args, "org_id", None)
    if not receipt_id:
        print("Error: Receipt ID required", file=sys.stderr)
        sys.exit(1)

    data: dict[str, Any] | None = None
    storage_error: Exception | None = None
    legacy_error: Exception | None = None

    if org_id:
        try:
            data = _load_legacy_receipt(receipt_id, org_id)
        except ImportError:
            print("Error: Gauntlet storage module not available", file=sys.stderr)
            sys.exit(1)
        except (OSError, RuntimeError, ValueError) as e:
            print(f"Error: Could not access receipt database: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            data = _load_storage_receipt(receipt_id)
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            storage_error = e

        if data is None:
            try:
                data = _load_legacy_receipt(receipt_id, org_id=None)
            except (ImportError, OSError, RuntimeError, ValueError) as e:
                legacy_error = e

        if data is None and storage_error is not None and legacy_error is not None:
            print(f"Error: Could not access receipt database: {storage_error}", file=sys.stderr)
            sys.exit(1)

    if data is None:
        print(f"Error: Receipt not found: {receipt_id}", file=sys.stderr)
        sys.exit(1)
    data = _normalize_receipt_payload_for_display(data)
    if output_format == "json":
        print(json.dumps(data, indent=2, default=str))
    elif output_format == "md":
        try:
            from aragora.gauntlet.receipt_models import DecisionReceipt

            receipt = DecisionReceipt.from_dict(data)
            print(receipt.to_markdown())
        except (ImportError, AttributeError, KeyError, ValueError, TypeError):
            print(json.dumps(data, indent=2, default=str))
    elif output_format == "html":
        try:
            from aragora.gauntlet.receipt_models import DecisionReceipt

            receipt = DecisionReceipt.from_dict(data)
            print(receipt.to_html())
        except (ImportError, AttributeError, KeyError, ValueError, TypeError):
            from aragora.cli.receipt_formatter import receipt_to_html

            print(receipt_to_html(data))
    else:
        _inspect_receipt_data(data)


def _inspect_receipt_data(data: dict[str, Any]) -> None:
    """Display receipt data in terminal."""
    print("\nDecision Receipt")
    print("=" * 60)
    print(f"\nReceipt ID:    {data.get('receipt_id', 'N/A')}")
    print(f"Gauntlet ID:   {data.get('gauntlet_id', 'N/A')}")
    print(f"Type:          {_receipt_kind(data)}")
    print(f"Verdict:       {data.get('verdict', 'UNKNOWN')}")
    confidence = data.get("confidence", 0)
    print(f"Confidence:    {confidence:.1%}")
    state = data.get("state")
    if state:
        print(f"State:         {state}")

    action_intent = data.get("action_intent")
    triage_decision = data.get("triage_decision")
    if isinstance(action_intent, dict) or isinstance(triage_decision, dict):
        action_intent = action_intent if isinstance(action_intent, dict) else {}
        triage_decision = triage_decision if isinstance(triage_decision, dict) else {}
        print("\n--- Inbox Trust Wedge ---")
        print(
            f"Action:        {triage_decision.get('final_action') or action_intent.get('action') or 'N/A'}"
        )
        print(f"Provider:      {action_intent.get('provider', 'N/A')}")
        print(f"Message ID:    {action_intent.get('message_id', 'N/A')}")
        print(
            f"Route:         {triage_decision.get('provider_route') or action_intent.get('provider_route') or 'N/A'}"
        )
        print(
            f"Receipt State: {triage_decision.get('receipt_state') or data.get('state') or 'N/A'}"
        )
        print(f"Blocked:       {'yes' if triage_decision.get('blocked_by_policy') else 'no'}")
        if action_intent.get("label_id") or triage_decision.get("label_id"):
            print(
                f"Label ID:      {triage_decision.get('label_id') or action_intent.get('label_id')}"
            )
        rationale = str(action_intent.get("synthesized_rationale") or "").strip()
        if rationale:
            print(f"Rationale:     {rationale[:280]}")

    risk_summary = data.get("risk_summary", {})
    if risk_summary:
        print(
            f"Findings:      {risk_summary.get('total', 0)} "
            f"(critical: {risk_summary.get('critical', 0)}, "
            f"high: {risk_summary.get('high', 0)})"
        )
    print("=" * 60)


# Keep backward-compatible aliases
setup_receipt_parser = add_receipt_parser

__all__ = [
    "add_receipt_parser",
    "cmd_receipt",
    "cmd_receipt_list",
    "cmd_receipt_show",
    "cmd_receipt_verify",
    "cmd_receipt_inspect",
    "cmd_receipt_export",
    "setup_receipt_parser",
]
