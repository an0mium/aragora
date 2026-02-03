#!/usr/bin/env python3
"""
Automated Security Audit Script.

Runs comprehensive security scans and optionally triggers multi-agent
debates on critical findings. Designed for CI/CD integration.

Usage:
    # Basic scan
    python scripts/security_audit.py

    # Scan with debate on critical findings
    python scripts/security_audit.py --debate-on-critical

    # CI mode (exit code indicates severity)
    python scripts/security_audit.py --ci --fail-on-high

    # Full scan with report
    python scripts/security_audit.py --output report.json --include-low
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated security audit with multi-agent debate support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic scan of aragora/ directory
  python scripts/security_audit.py

  # Scan specific path
  python scripts/security_audit.py --path ./src

  # CI mode: fail on high severity findings
  python scripts/security_audit.py --ci --fail-on-high

  # Run multi-agent debate on critical findings
  python scripts/security_audit.py --debate-on-critical

  # Output JSON report
  python scripts/security_audit.py --output audit-report.json
        """,
    )

    parser.add_argument(
        "--path",
        default="aragora/",
        help="Path to scan (default: aragora/)",
    )
    parser.add_argument(
        "--include-low",
        action="store_true",
        help="Include low severity findings in report",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: minimal output, exit codes indicate severity",
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 1 if critical findings exist",
    )
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        help="Exit with code 1 if high or critical findings exist",
    )
    parser.add_argument(
        "--debate-on-critical",
        action="store_true",
        help="Run multi-agent debate on critical findings",
    )
    parser.add_argument(
        "--debate-on-high",
        action="store_true",
        help="Run multi-agent debate on high or critical findings",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


async def run_security_scan(path: str, include_low: bool = False) -> dict[str, Any]:
    """Run security scanner on the specified path."""
    from aragora.audit.security_scanner import SecurityScanner, SecuritySeverity

    logger.info(f"[SCAN] Running security scan on {path}...")
    scanner = SecurityScanner(include_low_severity=include_low)
    report = scanner.scan_directory(path)

    # Count by severity
    findings_by_severity: dict[str, list[Any]] = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
        "info": [],
    }

    for finding in report.findings:
        sev = finding.severity.value
        if sev in findings_by_severity:
            findings_by_severity[sev].append(finding)

    return {
        "scanner": "security_scanner",
        "path": path,
        "total_findings": len(report.findings),
        "critical_count": len(findings_by_severity["critical"]),
        "high_count": len(findings_by_severity["high"]),
        "medium_count": len(findings_by_severity["medium"]),
        "low_count": len(findings_by_severity["low"]),
        "findings_by_severity": findings_by_severity,
        "findings": [
            {
                "id": f.id,
                "title": f.title,
                "severity": f.severity.value,
                "category": f.category.value,
                "file": f.file_path,
                "line": f.line_number,
                "description": f.description,
                "recommendation": f.recommendation,
                "cwe_id": f.cwe_id,
            }
            for f in report.findings
        ],
    }


async def run_bug_detection(path: str, include_low: bool = False) -> dict[str, Any]:
    """Run bug detector on the specified path."""
    from aragora.audit.bug_detector import BugDetector

    logger.info(f"[SCAN] Running bug detection on {path}...")
    detector = BugDetector(include_low_severity=include_low)
    report = detector.detect_in_directory(path)

    # Count by severity
    bugs_by_severity: dict[str, list[Any]] = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
    }

    for bug in report.bugs:
        sev = bug.severity.value
        if sev in bugs_by_severity:
            bugs_by_severity[sev].append(bug)

    return {
        "scanner": "bug_detector",
        "path": path,
        "total_bugs": len(report.bugs),
        "critical_count": len(bugs_by_severity["critical"]),
        "high_count": len(bugs_by_severity["high"]),
        "medium_count": len(bugs_by_severity["medium"]),
        "low_count": len(bugs_by_severity["low"]),
        "bugs": [
            {
                "id": b.id,
                "title": b.title,
                "severity": b.severity.value,
                "category": b.category.value,
                "file": b.file_path,
                "line": b.line_number,
                "description": b.description,
                "recommendation": getattr(b, "recommendation", "Review and fix the issue"),
            }
            for b in report.bugs
        ],
    }


async def run_security_debate(
    findings: list[dict[str, Any]],
    severity_filter: str = "critical",
) -> dict[str, Any] | None:
    """Run multi-agent debate on security findings."""
    from aragora.debate.security_debate import run_security_debate
    from aragora.events.security_events import (
        SecurityEvent,
        SecurityEventType,
        SecuritySeverity,
        SecurityFinding as EventFinding,
    )

    # Filter findings by severity
    if severity_filter == "critical":
        filtered = [f for f in findings if f.get("severity") == "critical"]
    else:  # high
        filtered = [f for f in findings if f.get("severity") in ("critical", "high")]

    if not filtered:
        logger.info("[DEBATE] No findings match severity filter, skipping debate")
        return None

    logger.info(f"[DEBATE] Running multi-agent debate on {len(filtered)} findings...")

    # Convert to SecurityFinding objects
    event_findings = []
    for f in filtered:
        # Infer finding_type from category
        category = f.get("category", "vulnerability")
        if "secret" in category.lower():
            finding_type = "secret"
        elif "config" in category.lower():
            finding_type = "misconfiguration"
        else:
            finding_type = "vulnerability"

        event_findings.append(
            EventFinding(
                id=f.get("id", "unknown"),
                finding_type=finding_type,
                title=f.get("title", "Unknown finding"),
                severity=SecuritySeverity(f.get("severity", "medium")),
                description=f.get("description", ""),
                file_path=f.get("file", ""),
                line_number=f.get("line", 0),
                recommendation=f.get("recommendation", ""),
            )
        )

    # Create security event
    event = SecurityEvent(
        event_type=SecurityEventType.SAST_CRITICAL,
        severity=SecuritySeverity.CRITICAL
        if severity_filter == "critical"
        else SecuritySeverity.HIGH,
        repository="aragora",
        findings=event_findings,
        source="security_audit_script",
    )

    # Run debate
    result = await run_security_debate(
        event=event,
        confidence_threshold=0.7,
        timeout_seconds=300,
    )

    return {
        "debate_id": result.debate_id,
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "rounds_used": result.rounds_used,
        "final_answer": result.final_answer,
        "findings_debated": len(filtered),
    }


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = datetime.now(timezone.utc)

    # Run security scan
    try:
        security_results = await run_security_scan(args.path, args.include_low)
    except ImportError as e:
        logger.error(f"Failed to import security scanner: {e}")
        return 2
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        return 2

    # Run bug detection
    try:
        bug_results = await run_bug_detection(args.path, args.include_low)
    except ImportError as e:
        logger.warning(f"Bug detector not available: {e}")
        bug_results = {"total_bugs": 0, "critical_count": 0, "high_count": 0}
    except Exception as e:
        logger.warning(f"Bug detection failed: {e}")
        bug_results = {"total_bugs": 0, "critical_count": 0, "high_count": 0}

    # Aggregate counts
    total_critical = security_results["critical_count"] + bug_results.get("critical_count", 0)
    total_high = security_results["high_count"] + bug_results.get("high_count", 0)
    total_findings = security_results["total_findings"] + bug_results.get("total_bugs", 0)

    # Run debate if requested
    debate_results = None
    if args.debate_on_critical and total_critical > 0:
        try:
            all_findings = security_results.get("findings", [])
            debate_results = await run_security_debate(all_findings, "critical")
        except Exception as e:
            logger.error(f"Security debate failed: {e}")

    elif args.debate_on_high and (total_critical > 0 or total_high > 0):
        try:
            all_findings = security_results.get("findings", [])
            debate_results = await run_security_debate(all_findings, "high")
        except Exception as e:
            logger.error(f"Security debate failed: {e}")

    end_time = datetime.now(timezone.utc)
    duration_ms = (end_time - start_time).total_seconds() * 1000

    # Build report
    report = {
        "timestamp": start_time.isoformat(),
        "duration_ms": round(duration_ms, 2),
        "path": args.path,
        "summary": {
            "total_findings": total_findings,
            "critical": total_critical,
            "high": total_high,
            "medium": security_results["medium_count"] + bug_results.get("medium_count", 0),
            "low": security_results["low_count"] + bug_results.get("low_count", 0),
        },
        "security_scan": security_results,
        "bug_detection": bug_results,
        "debate": debate_results,
    }

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"[REPORT] Written to {args.output}")

    if args.ci:
        # CI mode: minimal output
        print(
            f"Security Audit: {total_critical} critical, {total_high} high, {total_findings} total"
        )
    else:
        # Normal output
        print("\n" + "=" * 60)
        print("SECURITY AUDIT REPORT")
        print("=" * 60)
        print(f"Path: {args.path}")
        print(f"Duration: {duration_ms:.0f}ms")
        print()
        print(f"  Critical: {total_critical}")
        print(f"  High:     {total_high}")
        print(f"  Medium:   {report['summary']['medium']}")
        print(f"  Low:      {report['summary']['low']}")
        print(f"  Total:    {total_findings}")

        if total_critical > 0:
            print("\n[CRITICAL FINDINGS]")
            for f in security_results.get("findings", []):
                if f.get("severity") == "critical":
                    print(f"  - {f['file']}:{f['line']}: {f['title']}")

        if debate_results:
            print("\n[MULTI-AGENT DEBATE RESULTS]")
            print(f"  Consensus: {debate_results['consensus_reached']}")
            print(f"  Confidence: {debate_results['confidence']:.2f}")
            print(f"  Recommendation: {debate_results['final_answer'][:200]}...")

        print()

    # Determine exit code
    if args.fail_on_critical and total_critical > 0:
        return 1
    if args.fail_on_high and (total_critical > 0 or total_high > 0):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
