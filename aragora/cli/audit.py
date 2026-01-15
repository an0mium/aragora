"""
Aragora Audit CLI.

Commands for querying and exporting audit logs for compliance.

Usage:
    aragora audit stats
    aragora audit export --format soc2 --start 2026-01-01 --end 2026-01-31
    aragora audit verify
    aragora audit query --category auth --actor user_123
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path


def cmd_stats(args: argparse.Namespace) -> int:
    """Show audit log statistics."""
    from aragora.audit import AuditLog

    audit = AuditLog()
    stats = audit.get_stats()

    print("\n" + "=" * 60)
    print("ARAGORA AUDIT LOG STATISTICS")
    print("=" * 60)

    print(f"\nTotal events: {stats['total_events']:,}")
    print(f"Retention: {stats['retention_days']} days")

    if stats.get("oldest_event"):
        print(f"Oldest event: {stats['oldest_event'][:10]}")
    if stats.get("newest_event"):
        print(f"Newest event: {stats['newest_event'][:10]}")

    if stats.get("by_category"):
        print("\nEvents by category:")
        for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count:,}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export audit log for compliance."""
    from aragora.audit import AuditLog

    audit = AuditLog()

    # Parse dates
    try:
        start_date = datetime.fromisoformat(args.start)
    except ValueError:
        print(f"Error: Invalid start date format: {args.start}")
        return 1

    try:
        end_date = datetime.fromisoformat(args.end)
    except ValueError:
        print(f"Error: Invalid end date format: {args.end}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = ".json" if args.format in ("json", "soc2") else ".csv"
        output_path = Path(f"audit_export_{start_date.date()}_{end_date.date()}{suffix}")

    print("\n" + "=" * 60)
    print("ARAGORA AUDIT EXPORT")
    print("=" * 60)
    print(f"\nFormat: {args.format}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Output: {output_path}")

    # Export
    if args.format == "json":
        count = audit.export_json(output_path, start_date, end_date, args.org)
    elif args.format == "csv":
        count = audit.export_csv(output_path, start_date, end_date, args.org)
    elif args.format == "soc2":
        result = audit.export_soc2(output_path, start_date, end_date, args.org)
        count = result["events_exported"]
        print(f"\nIntegrity verified: {'Yes' if result['integrity_verified'] else 'NO'}")
        if result["integrity_errors"]:
            print(f"Integrity errors: {result['integrity_errors']}")
    else:
        print(f"Error: Unknown format: {args.format}")
        return 1

    print(f"\nExported {count:,} events to {output_path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify audit log integrity."""
    from aragora.audit import AuditLog

    audit = AuditLog()

    print("\n" + "=" * 60)
    print("ARAGORA AUDIT LOG INTEGRITY VERIFICATION")
    print("=" * 60)

    # Parse dates if provided
    start_date = None
    end_date = None

    if args.start:
        try:
            start_date = datetime.fromisoformat(args.start)
        except ValueError:
            print(f"Error: Invalid start date: {args.start}")
            return 1

    if args.end:
        try:
            end_date = datetime.fromisoformat(args.end)
        except ValueError:
            print(f"Error: Invalid end date: {args.end}")
            return 1

    print("\nVerifying hash chain integrity...")
    is_valid, errors = audit.verify_integrity(start_date, end_date)

    if is_valid:
        print("\n  PASSED: Audit log integrity verified")
        return 0
    else:
        print(f"\n  FAILED: {len(errors)} integrity error(s) detected")
        if args.verbose:
            print("\nErrors:")
            for error in errors[:20]:
                print(f"  - {error}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
        return 1


def cmd_query(args: argparse.Namespace) -> int:
    """Query audit events."""
    from aragora.audit import AuditCategory, AuditLog, AuditOutcome, AuditQuery

    audit = AuditLog()

    # Build query
    query = AuditQuery(limit=args.limit)

    if args.start:
        try:
            query.start_date = datetime.fromisoformat(args.start)
        except ValueError:
            print(f"Error: Invalid start date: {args.start}")
            return 1

    if args.end:
        try:
            query.end_date = datetime.fromisoformat(args.end)
        except ValueError:
            print(f"Error: Invalid end date: {args.end}")
            return 1

    if args.category:
        try:
            query.category = AuditCategory(args.category)
        except ValueError:
            print(f"Error: Invalid category: {args.category}")
            print(f"Valid categories: {', '.join(c.value for c in AuditCategory)}")
            return 1

    if args.actor:
        query.actor_id = args.actor

    if args.outcome:
        try:
            query.outcome = AuditOutcome(args.outcome)
        except ValueError:
            print(f"Error: Invalid outcome: {args.outcome}")
            return 1

    if args.search:
        query.search_text = args.search

    # Execute query
    events = audit.query(query)

    print("\n" + "=" * 60)
    print("ARAGORA AUDIT EVENTS")
    print("=" * 60)
    print(f"\nFound {len(events)} events\n")

    if args.json:
        import json

        print(json.dumps([e.to_dict() for e in events], indent=2))
    else:
        for event in events:
            ts = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            outcome = "+" if event.outcome.value == "success" else "-"
            print(
                f"[{ts}] {outcome} {event.category.value}:{event.action} " f"actor={event.actor_id}"
            )
            if event.resource_type:
                print(f"           resource={event.resource_type}:{event.resource_id}")
            if event.reason and args.verbose:
                print(f"           reason={event.reason}")

    return 0


def create_audit_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create audit subcommand parser."""
    audit_parser = subparsers.add_parser(
        "audit",
        help="Query and export audit logs for compliance",
        description="""
Manage audit logs for enterprise compliance (SOC 2, HIPAA, GDPR, SOX).

Examples:
    aragora audit stats                    # Show audit log statistics
    aragora audit export --format soc2     # Export for SOC 2 audit
    aragora audit verify                   # Verify log integrity
    aragora audit query --category auth    # Query auth events
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    audit_subparsers = audit_parser.add_subparsers(dest="audit_action")

    # Stats
    stats_parser = audit_subparsers.add_parser("stats", help="Show audit log statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Export
    export_parser = audit_subparsers.add_parser("export", help="Export audit log")
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "soc2"],
        default="json",
        help="Export format",
    )
    export_parser.add_argument(
        "--start",
        "-s",
        default=(datetime.utcnow() - timedelta(days=30)).isoformat()[:10],
        help="Start date (YYYY-MM-DD)",
    )
    export_parser.add_argument(
        "--end",
        "-e",
        default=datetime.utcnow().isoformat()[:10],
        help="End date (YYYY-MM-DD)",
    )
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.add_argument("--org", help="Filter by organization ID")
    export_parser.set_defaults(func=cmd_export)

    # Verify
    verify_parser = audit_subparsers.add_parser("verify", help="Verify log integrity")
    verify_parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    verify_parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    verify_parser.add_argument("--verbose", "-v", action="store_true", help="Show errors")
    verify_parser.set_defaults(func=cmd_verify)

    # Query
    query_parser = audit_subparsers.add_parser("query", help="Query audit events")
    query_parser.add_argument("--start", "-s", help="Start date")
    query_parser.add_argument("--end", "-e", help="End date")
    query_parser.add_argument(
        "--category",
        "-c",
        choices=[
            "auth",
            "access",
            "data",
            "admin",
            "billing",
            "debate",
            "api",
            "security",
            "system",
        ],
        help="Filter by category",
    )
    query_parser.add_argument("--actor", "-a", help="Filter by actor ID")
    query_parser.add_argument(
        "--outcome",
        "-o",
        choices=["success", "failure", "denied", "error"],
        help="Filter by outcome",
    )
    query_parser.add_argument("--search", help="Full-text search")
    query_parser.add_argument("--limit", "-n", type=int, default=50, help="Max results")
    query_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    query_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    query_parser.set_defaults(func=cmd_query)

    # Default to stats
    audit_parser.set_defaults(func=cmd_audit_default)


def cmd_audit_default(args: argparse.Namespace) -> int:
    """Default audit command."""
    if not hasattr(args, "audit_action") or args.audit_action is None:
        return cmd_stats(args)
    return 0


def main(args: argparse.Namespace) -> int:
    """Main entry point for audit commands."""
    if hasattr(args, "func"):
        ret: int = args.func(args)
        return ret
    return cmd_audit_default(args)


__all__ = ["create_audit_parser", "main"]
