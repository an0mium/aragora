"""
Audit CLI commands for Aragora.
"""

import asyncio
import json
from datetime import datetime
from typing import Any


def audit_cli(args: Any) -> int:
    """Handle audit subcommands."""
    if args.audit_command == "create":
        return asyncio.run(create_audit(args))
    elif args.audit_command == "start":
        return asyncio.run(start_audit(args))
    elif args.audit_command == "status":
        return asyncio.run(audit_status(args))
    elif args.audit_command == "findings":
        return asyncio.run(audit_findings(args))
    elif args.audit_command == "export":
        return asyncio.run(export_audit(args))
    else:
        print("Unknown audit command")
        return 1


async def create_audit(args: Any) -> int:
    """Create a new audit session."""
    from aragora.audit import get_document_auditor

    document_ids = [d.strip() for d in args.documents.split(",")]
    audit_types = [t.strip() for t in args.types.split(",")] if args.types != "all" else None

    print(f"Creating audit session with {len(document_ids)} documents...")

    try:
        auditor = get_document_auditor()
        session = await auditor.create_session(
            document_ids=document_ids,
            audit_types=audit_types,
            name=args.name,
            model=args.model,
        )
        print(f"Session created: {session.id}")
        print(f"Run: aragora audit start {session.id}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def start_audit(args: Any) -> int:
    """Start an audit session."""
    from aragora.audit import get_document_auditor

    print(f"Starting audit: {args.session_id}")

    try:
        auditor = get_document_auditor()

        def on_progress(sid, progress, phase):
            print(f"  [{phase}] {progress*100:.0f}%")

        auditor.on_progress = on_progress
        result = await auditor.run_audit(args.session_id)

        print(f"Completed: {len(result.findings)} findings")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def audit_status(args: Any) -> int:
    """Get audit session status."""
    from aragora.audit import get_document_auditor

    try:
        auditor = get_document_auditor()
        session = auditor.get_session(args.session_id)

        if not session:
            print(f"Session not found: {args.session_id}")
            return 1

        print(f"Session: {session.id}")
        print(f"Status: {session.status.value}")
        print(f"Progress: {session.progress*100:.0f}%")
        print(f"Findings: {len(session.findings)}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def audit_findings(args: Any) -> int:
    """Get audit findings."""
    from aragora.audit import get_document_auditor, FindingSeverity

    try:
        auditor = get_document_auditor()
        severity = FindingSeverity(args.severity.lower()) if args.severity else None
        findings = auditor.get_findings(args.session_id, severity=severity)

        if args.format == "json":
            print(json.dumps([f.to_dict() for f in findings], indent=2, default=str))
        else:
            for f in findings:
                print(f"[{f.severity.value}] {f.title}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def export_audit(args: Any) -> int:
    """Export audit report."""
    from aragora.audit import get_document_auditor

    try:
        auditor = get_document_auditor()
        session = auditor.get_session(args.session_id)

        if not session:
            print("Session not found")
            return 1

        report = {
            "session": session.to_dict(),
            "findings": [f.to_dict() for f in session.findings],
            "exported_at": datetime.utcnow().isoformat(),
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Exported to: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
