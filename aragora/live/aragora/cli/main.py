"""
Main CLI entry point for Aragora.

Usage:
    python -m aragora [command] [options]
"""

import argparse
import sys
from typing import Optional


def main(args: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="aragora",
        description="Aragora - Multi-agent debate and document audit platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Documents command
    doc_parser = subparsers.add_parser("documents", help="Document management")
    doc_sub = doc_parser.add_subparsers(dest="doc_command")

    # documents upload
    upload_parser = doc_sub.add_parser("upload", help="Upload documents")
    upload_parser.add_argument("files", nargs="+", help="Files to upload")
    upload_parser.add_argument("--batch-size", type=int, default=10, help="Batch size")

    # documents list
    doc_sub.add_parser("list", help="List documents")

    # documents show
    show_parser = doc_sub.add_parser("show", help="Show document details")
    show_parser.add_argument("doc_id", help="Document ID")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Document auditing")
    audit_sub = audit_parser.add_subparsers(dest="audit_command")

    # audit create
    create_parser = audit_sub.add_parser("create", help="Create audit session")
    create_parser.add_argument(
        "--documents", "-d", required=True, help="Comma-separated document IDs"
    )
    create_parser.add_argument(
        "--types",
        "-t",
        default="all",
        help="Audit types: security,compliance,consistency,quality,all",
    )
    create_parser.add_argument("--model", "-m", default="gemini-3-pro", help="Primary model")
    create_parser.add_argument("--name", "-n", help="Session name")

    # audit start
    start_parser = audit_sub.add_parser("start", help="Start audit session")
    start_parser.add_argument("session_id", help="Session ID")

    # audit status
    status_parser = audit_sub.add_parser("status", help="Get audit status")
    status_parser.add_argument("session_id", help="Session ID")

    # audit findings
    findings_parser = audit_sub.add_parser("findings", help="Get audit findings")
    findings_parser.add_argument("session_id", help="Session ID")
    findings_parser.add_argument("--severity", "-s", help="Filter by severity")
    findings_parser.add_argument(
        "--format", "-f", default="table", choices=["table", "json"], help="Output format"
    )

    # audit export
    export_parser = audit_sub.add_parser("export", help="Export audit report")
    export_parser.add_argument("session_id", help="Session ID")
    export_parser.add_argument(
        "--format", "-f", default="json", choices=["json", "pdf", "html"], help="Export format"
    )
    export_parser.add_argument("--output", "-o", required=True, help="Output file")

    # Doctor command
    subparsers.add_parser("doctor", help="Run health checks")

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 0

    if parsed.command == "documents":
        from aragora.cli.documents import documents_cli

        return documents_cli(parsed)

    if parsed.command == "audit":
        from aragora.cli.audit import audit_cli

        return audit_cli(parsed)

    if parsed.command == "doctor":
        from aragora.cli.doctor import main as doctor_main

        return doctor_main()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
