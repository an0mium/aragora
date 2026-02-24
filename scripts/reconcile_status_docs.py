#!/usr/bin/env python3
"""Status Document Reconciliation Report.

Cross-checks capability matrix, GA checklist, roadmap, and status docs
for contradictions and drift. Generates a report as artifact and optionally
fails on critical mismatches.

Usage:
    python scripts/reconcile_status_docs.py             # Report only
    python scripts/reconcile_status_docs.py --strict     # Fail on critical drift
    python scripts/reconcile_status_docs.py --json       # JSON output
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files to reconcile
CAPABILITY_MATRIX = REPO_ROOT / "docs" / "CAPABILITY_MATRIX.md"
CAPABILITY_YAML = REPO_ROOT / "aragora" / "capability_surfaces.yaml"
GA_CHECKLIST = REPO_ROOT / "docs" / "GA_CHECKLIST.md"
STATUS_DOC = REPO_ROOT / "docs" / "STATUS.md"
STATUS_DIR = REPO_ROOT / "docs" / "status" / "STATUS.md"
ROADMAP = REPO_ROOT / "ROADMAP.md"
CONNECTOR_STATUS = REPO_ROOT / "docs" / "connectors" / "STATUS.md"
EXECUTION_PROGRAM = REPO_ROOT / "docs" / "status" / "EXECUTION_PROGRAM_2026Q2_Q4.md"


def _file_age_days(path: Path) -> int | None:
    """Get age of file in days based on content date markers or mtime."""
    if not path.exists():
        return None
    # Try to find last_updated or Generated date in content
    content = path.read_text(encoding="utf-8", errors="replace")
    for pattern in [
        r'last_updated:\s*"?(\d{4}-\d{2}-\d{2})"?',
        r"Last updated:\s*(\d{4}-\d{2}-\d{2})",
        r"Generated:\s*(\d{4}-\d{2}-\d{2})",
        r"Updated:\s*(\d{4}-\d{2}-\d{2})",
    ]:
        m = re.search(pattern, content)
        if m:
            try:
                d = date.fromisoformat(m.group(1))
                return (date.today() - d).days
            except ValueError:
                pass
    # Fall back to file mtime
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).days


def _count_pattern(path: Path, pattern: str) -> int:
    """Count occurrences of a pattern in a file."""
    if not path.exists():
        return 0
    content = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(pattern, content))


def _extract_checklist_stats(path: Path) -> dict:
    """Extract completed/total from a markdown checklist."""
    if not path.exists():
        return {"complete": 0, "total": 0}
    content = path.read_text(encoding="utf-8", errors="replace")
    checked = len(re.findall(r"- \[x\]", content, re.IGNORECASE))
    unchecked = len(re.findall(r"- \[ \]", content))
    return {"complete": checked, "total": checked + unchecked}


def _check_capability_matrix_freshness() -> list[dict]:
    """Check if capability matrix is up to date with YAML source."""
    findings = []

    if not CAPABILITY_YAML.exists():
        findings.append(
            {
                "severity": "critical",
                "source": "capability_surfaces.yaml",
                "message": "Capability surfaces YAML not found",
            }
        )
        return findings

    yaml_age = _file_age_days(CAPABILITY_YAML)
    matrix_age = _file_age_days(CAPABILITY_MATRIX)

    if yaml_age is not None and matrix_age is not None and matrix_age > yaml_age + 7:
        findings.append(
            {
                "severity": "warning",
                "source": "CAPABILITY_MATRIX.md",
                "message": f"Matrix ({matrix_age}d old) is significantly older than YAML source ({yaml_age}d old). Run: python scripts/generate_capability_matrix.py",
            }
        )

    # Check if generated matrix matches
    try:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "check_capability_matrix_sync.py")],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            findings.append(
                {
                    "severity": "critical",
                    "source": "CAPABILITY_MATRIX.md",
                    "message": "Matrix is out of sync with YAML. Run: python scripts/generate_capability_matrix.py",
                }
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        findings.append(
            {
                "severity": "warning",
                "source": "check_capability_matrix_sync.py",
                "message": "Could not run matrix sync check",
            }
        )

    return findings


def _check_ga_checklist() -> list[dict]:
    """Check GA checklist for completion and blockers."""
    findings = []

    if not GA_CHECKLIST.exists():
        findings.append(
            {
                "severity": "warning",
                "source": "GA_CHECKLIST.md",
                "message": "GA checklist not found",
            }
        )
        return findings

    stats = _extract_checklist_stats(GA_CHECKLIST)
    age = _file_age_days(GA_CHECKLIST)

    if stats["total"] > 0:
        completion = stats["complete"] / stats["total"] * 100
        if completion < 100 and age and age > 14:
            findings.append(
                {
                    "severity": "warning",
                    "source": "GA_CHECKLIST.md",
                    "message": f"GA checklist {completion:.0f}% complete ({stats['complete']}/{stats['total']}) and {age}d since last update",
                }
            )

    # Check for explicit blockers
    content = GA_CHECKLIST.read_text(encoding="utf-8", errors="replace")
    blocker_count = len(re.findall(r"(?i)blocker|blocked|blocking", content))
    if blocker_count > 0:
        findings.append(
            {
                "severity": "info",
                "source": "GA_CHECKLIST.md",
                "message": f"GA checklist references {blocker_count} blocker mentions",
            }
        )

    return findings


def _check_connector_status() -> list[dict]:
    """Check connector status for stubs and beta counts."""
    findings = []

    if not CONNECTOR_STATUS.exists():
        return findings

    content = CONNECTOR_STATUS.read_text(encoding="utf-8", errors="replace")
    # Prefer explicit summary counts when present to avoid false positives from
    # status-definition prose ("Stub | Definition", etc.).
    prod_match = re.search(r"(?im)^\s*-\s*\*\*Production\*\*:\s*(\d+)\s+connectors", content)
    beta_match = re.search(r"(?im)^\s*-\s*\*\*Beta\*\*:\s*(\d+)\s+connectors", content)
    stub_match = re.search(r"(?im)^\s*-\s*\*\*Stub\*\*:\s*(\d+)\s+connectors", content)

    if prod_match and beta_match and stub_match:
        prod_count = int(prod_match.group(1))
        beta_count = int(beta_match.group(1))
        stub_count = int(stub_match.group(1))
    else:
        # Fallback heuristic for non-standard connector status files.
        stub_count = len(re.findall(r"(?i)\bstub\b", content))
        beta_count = len(re.findall(r"(?i)\bbeta\b", content))
        prod_count = len(re.findall(r"(?i)\bproduction\b", content))

    if stub_count > 0:
        findings.append(
            {
                "severity": "warning",
                "source": "connectors/STATUS.md",
                "message": f"Connector status has {stub_count} stub references (target: 0)",
            }
        )

    findings.append(
        {
            "severity": "info",
            "source": "connectors/STATUS.md",
            "message": f"Connectors: ~{prod_count} production, ~{beta_count} beta, ~{stub_count} stub mentions",
        }
    )

    return findings


def _check_staleness() -> list[dict]:
    """Check all status docs for staleness."""
    findings = []
    STALE_THRESHOLD_DAYS = 30

    docs_to_check = [
        (CAPABILITY_MATRIX, "CAPABILITY_MATRIX.md"),
        (GA_CHECKLIST, "GA_CHECKLIST.md"),
        (STATUS_DOC, "STATUS.md"),
        (ROADMAP, "ROADMAP.md"),
        (CONNECTOR_STATUS, "connectors/STATUS.md"),
    ]

    for path, label in docs_to_check:
        if not path.exists():
            continue
        age = _file_age_days(path)
        if age is not None and age > STALE_THRESHOLD_DAYS:
            findings.append(
                {
                    "severity": "warning",
                    "source": label,
                    "message": f"Document is {age} days old (threshold: {STALE_THRESHOLD_DAYS}d)",
                }
            )

    return findings


def reconcile(strict: bool = False) -> dict:
    """Run all reconciliation checks and return report."""
    findings = []
    findings.extend(_check_capability_matrix_freshness())
    findings.extend(_check_ga_checklist())
    findings.extend(_check_connector_status())
    findings.extend(_check_staleness())

    critical = [f for f in findings if f["severity"] == "critical"]
    warnings = [f for f in findings if f["severity"] == "warning"]
    info = [f for f in findings if f["severity"] == "info"]

    report = {
        "generated": datetime.now().isoformat(),
        "findings": findings,
        "summary": {
            "critical": len(critical),
            "warning": len(warnings),
            "info": len(info),
            "total": len(findings),
        },
        "pass": len(critical) == 0 if strict else True,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile status docs for drift")
    parser.add_argument("--strict", action="store_true", help="Fail on critical findings")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--output", type=str, help="Write report to file")
    args = parser.parse_args()

    report = reconcile(strict=args.strict)

    if args.json:
        output = json.dumps(report, indent=2)
    else:
        lines = [
            "# Status Document Reconciliation Report",
            f"Generated: {report['generated'][:10]}",
            "",
            f"## Summary: {report['summary']['critical']} critical, "
            f"{report['summary']['warning']} warnings, {report['summary']['info']} info",
            "",
        ]

        for severity in ["critical", "warning", "info"]:
            items = [f for f in report["findings"] if f["severity"] == severity]
            if items:
                lines.append(f"### {severity.upper()} ({len(items)})")
                lines.append("")
                for f in items:
                    marker = {"critical": "!!", "warning": "!", "info": "-"}.get(severity, "-")
                    lines.append(f"  {marker} [{f['source']}] {f['message']}")
                lines.append("")

        if report["pass"]:
            lines.append("Result: PASS")
        else:
            lines.append("Result: FAIL (critical findings detected)")

        output = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
