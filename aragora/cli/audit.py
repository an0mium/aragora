"""
Audit CLI commands for Aragora.

Commands:
    presets     - List available audit presets (industry-specific configurations)
    preset      - Show details of a specific preset
    types       - List registered audit types and their capabilities
    create      - Create a new audit session (with optional preset)
    start       - Start an audit session
    status      - Get audit session status
    findings    - Get audit findings
    export      - Export audit report
"""

import argparse
import asyncio
import json
from datetime import datetime
from typing import Any


def create_audit_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the audit subparser."""
    audit_parser = subparsers.add_parser(
        "audit",
        help="Document compliance and audit commands",
        description="Create and manage document audit sessions for compliance checking.",
    )

    audit_subparsers = audit_parser.add_subparsers(
        dest="audit_command",
        title="audit commands",
        description="Available audit commands",
    )

    # Presets command
    presets_parser = audit_subparsers.add_parser("presets", help="List available audit presets")
    presets_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Preset detail command
    preset_parser = audit_subparsers.add_parser("preset", help="Show details of a specific preset")
    preset_parser.add_argument("name", help="Preset name (e.g., 'Legal Due Diligence')")
    preset_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Types command
    types_parser = audit_subparsers.add_parser("types", help="List registered audit types")
    types_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Create command
    create_parser = audit_subparsers.add_parser("create", help="Create a new audit session")
    create_parser.add_argument("documents", help="Comma-separated list of document IDs to audit")
    create_parser.add_argument(
        "--types",
        default=None,
        help="Comma-separated audit types (security,quality,consistency,compliance)",
    )
    create_parser.add_argument(
        "--preset",
        default=None,
        help="Use a preset configuration (overrides --types if specified)",
    )
    create_parser.add_argument("--name", default=None, help="Session name")
    create_parser.add_argument("--model", default="gemini-1.5-flash", help="Model for analysis")

    # Start command
    start_parser = audit_subparsers.add_parser("start", help="Start an audit session")
    start_parser.add_argument("session_id", help="Session ID to start")

    # Status command
    status_parser = audit_subparsers.add_parser("status", help="Get audit session status")
    status_parser.add_argument("session_id", help="Session ID")

    # Findings command
    findings_parser = audit_subparsers.add_parser("findings", help="Get audit findings")
    findings_parser.add_argument("session_id", help="Session ID")
    findings_parser.add_argument(
        "--severity",
        choices=["critical", "high", "medium", "low"],
        help="Filter by severity",
    )
    findings_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Export command (legacy JSON export)
    export_parser = audit_subparsers.add_parser("export", help="Export audit data as JSON")
    export_parser.add_argument("session_id", help="Session ID")
    export_parser.add_argument("--output", "-o", required=True, help="Output file path")

    # Report command (new formatted report generation)
    report_parser = audit_subparsers.add_parser("report", help="Generate formatted audit report")
    report_parser.add_argument("session_id", help="Session ID")
    report_parser.add_argument(
        "--format",
        "-f",
        choices=["pdf", "markdown", "html", "json"],
        default="markdown",
        help="Report format (default: markdown)",
    )
    report_parser.add_argument(
        "--template",
        "-t",
        choices=[
            "executive_summary",
            "detailed_findings",
            "compliance_attestation",
            "security_assessment",
        ],
        default="detailed_findings",
        help="Report template (default: detailed_findings)",
    )
    report_parser.add_argument(
        "--output", "-o", help="Output file path (auto-generated if not specified)"
    )
    report_parser.add_argument(
        "--min-severity",
        choices=["critical", "high", "medium", "low", "info"],
        default="low",
        help="Minimum severity to include (default: low)",
    )
    report_parser.add_argument(
        "--include-resolved", action="store_true", help="Include resolved findings"
    )
    report_parser.add_argument("--author", help="Report author name")
    report_parser.add_argument("--company", help="Company name for branding")

    audit_parser.set_defaults(func=audit_cli)


def audit_cli(args: Any) -> int:
    """Handle audit subcommands."""
    if args.audit_command == "presets":
        return asyncio.run(list_presets(args))
    elif args.audit_command == "preset":
        return asyncio.run(show_preset(args))
    elif args.audit_command == "types":
        return asyncio.run(list_types(args))
    elif args.audit_command == "create":
        return asyncio.run(create_audit(args))
    elif args.audit_command == "start":
        return asyncio.run(start_audit(args))
    elif args.audit_command == "status":
        return asyncio.run(audit_status(args))
    elif args.audit_command == "findings":
        return asyncio.run(audit_findings(args))
    elif args.audit_command == "export":
        return asyncio.run(export_audit(args))
    elif args.audit_command == "report":
        return asyncio.run(generate_report(args))
    else:
        print(
            "Unknown audit command. Use: presets, preset, types, create, start, status, findings, export, report"
        )
        return 1


async def list_presets(args: Any) -> int:
    """List available audit presets."""
    from aragora.audit.registry import audit_registry

    try:
        audit_registry.auto_discover()
        presets = audit_registry.list_presets()

        if args.format == "json":
            data = [
                {
                    "name": p.name,
                    "description": p.description,
                    "audit_types": p.audit_types,
                    "custom_rules_count": len(p.custom_rules),
                    "consensus_threshold": p.consensus_threshold,
                }
                for p in presets
            ]
            print(json.dumps(data, indent=2))
        else:
            if not presets:
                print("No presets available.")
                print("Presets are loaded from aragora/audit/presets/ YAML files.")
                return 0

            print("Available Audit Presets:\n")
            for p in presets:
                print(f"  {p.name}")
                print(f"    {p.description}")
                print(f"    Types: {', '.join(p.audit_types)}")
                print(
                    f"    Rules: {len(p.custom_rules)}, Threshold: {p.consensus_threshold*100:.0f}%"
                )
                print()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def show_preset(args: Any) -> int:
    """Show details of a specific preset."""
    from aragora.audit.registry import audit_registry

    try:
        audit_registry.auto_discover()
        preset = audit_registry.get_preset(args.name)

        if not preset:
            print(f"Preset not found: {args.name}")
            print("\nAvailable presets:")
            for p in audit_registry.list_presets():
                print(f"  - {p.name}")
            return 1

        if args.format == "json":
            data = {
                "name": preset.name,
                "description": preset.description,
                "audit_types": preset.audit_types,
                "custom_rules": preset.custom_rules,
                "consensus_threshold": preset.consensus_threshold,
                "agents": preset.agents,
                "parameters": preset.parameters,
            }
            print(json.dumps(data, indent=2))
        else:
            print(f"Preset: {preset.name}")
            print(f"Description: {preset.description}")
            print(f"\nAudit Types: {', '.join(preset.audit_types)}")
            print(f"Consensus Threshold: {preset.consensus_threshold*100:.0f}%")
            if preset.agents:
                print(f"Agents: {', '.join(preset.agents)}")
            if preset.parameters:
                print(f"Parameters: {preset.parameters}")

            if preset.custom_rules:
                print(f"\nCustom Rules ({len(preset.custom_rules)}):")
                for rule in preset.custom_rules:
                    severity = rule.get("severity", "medium")
                    title = rule.get("title", rule.get("pattern", "Unknown"))
                    category = rule.get("category", "general")
                    print(f"  [{severity}] {title} ({category})")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def list_types(args: Any) -> int:
    """List registered audit types."""
    from aragora.audit.registry import audit_registry

    try:
        audit_registry.auto_discover()
        types = audit_registry.list_audit_types()

        if args.format == "json":
            data = [
                {
                    "id": t.id,
                    "display_name": t.display_name,
                    "description": t.description,
                    "version": t.version,
                    "capabilities": t.capabilities,
                }
                for t in types
            ]
            print(json.dumps(data, indent=2))
        else:
            if not types:
                print("No audit types registered.")
                return 0

            print("Registered Audit Types:\n")
            for t in types:
                print(f"  {t.id} ({t.display_name}) v{t.version}")
                print(f"    {t.description}")
                caps = []
                if t.capabilities.get("supports_chunk_analysis", True):
                    caps.append("chunk")
                if t.capabilities.get("supports_cross_document"):
                    caps.append("cross-doc")
                if t.capabilities.get("requires_llm"):
                    caps.append("llm")
                print(f"    Capabilities: {', '.join(caps) or 'none'}")
                print()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def create_audit(args: Any) -> int:
    """Create a new audit session."""
    from aragora.audit import get_document_auditor
    from aragora.audit.registry import audit_registry

    document_ids = [d.strip() for d in args.documents.split(",")]

    # Determine audit types from preset or explicit flag
    audit_types = None
    preset_name = None

    if args.preset:
        # Load preset configuration
        audit_registry.auto_discover()
        preset = audit_registry.get_preset(args.preset)
        if not preset:
            print(f"Preset not found: {args.preset}")
            print("Use 'aragora audit presets' to list available presets.")
            return 1
        audit_types = preset.audit_types
        preset_name = preset.name
        print(f"Using preset: {preset.name}")
    elif args.types:
        audit_types = [t.strip() for t in args.types.split(",")]

    print(f"Creating audit session with {len(document_ids)} documents...")
    if audit_types:
        print(f"Audit types: {', '.join(audit_types)}")

    try:
        auditor = get_document_auditor()
        session = await auditor.create_session(
            document_ids=document_ids,
            audit_types=audit_types,
            name=args.name or (f"Audit-{preset_name}" if preset_name else None),
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


async def generate_report(args: Any) -> int:
    """Generate formatted audit report."""
    from pathlib import Path

    from aragora.audit import get_document_auditor
    from aragora.reports import (
        AuditReportGenerator,
        ReportConfig,
        ReportFormat,
        ReportTemplate,
    )

    try:
        auditor = get_document_auditor()
        session = auditor.get_session(args.session_id)

        if not session:
            print(f"Session not found: {args.session_id}")
            return 1

        # Map format string to enum
        format_map = {
            "pdf": ReportFormat.PDF,
            "markdown": ReportFormat.MARKDOWN,
            "html": ReportFormat.HTML,
            "json": ReportFormat.JSON,
        }
        report_format = format_map[args.format]

        # Map template string to enum
        template_map = {
            "executive_summary": ReportTemplate.EXECUTIVE_SUMMARY,
            "detailed_findings": ReportTemplate.DETAILED_FINDINGS,
            "compliance_attestation": ReportTemplate.COMPLIANCE_ATTESTATION,
            "security_assessment": ReportTemplate.SECURITY_ASSESSMENT,
        }
        template = template_map[args.template]

        # Create config
        config = ReportConfig(
            min_severity=args.min_severity,
            include_resolved=args.include_resolved,
            author=args.author or "",
            company_name=args.company or "Aragora",
        )

        # Generate report
        print(f"Generating {args.format} report with {args.template} template...")
        generator = AuditReportGenerator(config)
        report = await generator.generate(
            session=session,
            format=report_format,
            template=template,
        )

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(report.filename)

        # Save report
        report.save(output_path)

        print("Report generated successfully!")
        print(f"  File: {output_path}")
        print(f"  Format: {report.format.value}")
        print(f"  Size: {report.size_bytes:,} bytes")
        print(f"  Findings: {report.findings_count}")

        return 0

    except ImportError as e:
        print(f"Error: Report module not available: {e}")
        if "weasyprint" in str(e).lower():
            print("For PDF support, install weasyprint: pip install weasyprint")
        return 1
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1
