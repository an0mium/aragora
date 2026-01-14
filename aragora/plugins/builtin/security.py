"""
Security Scan Plugin - Scan code for security vulnerabilities.

Uses bandit to analyze Python code for common security issues.

Example usage via API:
    POST /api/plugins/security-scan/run
    {
        "input": {"files": ["aragora/server"]},
        "config": {"severity": "medium"}
    }
"""

import asyncio
import json
import shutil
from typing import Any

from aragora.plugins.runner import PluginContext


async def run(context: PluginContext) -> dict[str, Any]:
    """
    Run security scanning on specified files or directories.

    Input:
        files: List of files/directories to scan (default: ["."])
        exclude: Patterns to exclude (default: ["tests", "*.test.py"])

    Config:
        severity: Minimum severity level: "low", "medium", "high" (default: "medium")
        confidence: Minimum confidence: "low", "medium", "high" (default: "medium")
        skip_tests: Skip test files (default: true)

    Output:
        vulnerabilities: List of security issues found
        summary: Summary by severity and confidence
        metrics: Code metrics (lines analyzed, etc.)
    """
    # Check if bandit is available
    if not shutil.which("bandit"):
        context.error("Bandit is not installed. Run: pip install bandit")
        return {"vulnerabilities": [], "error": "Bandit not installed"}

    # Extract input
    files = context.input_data.get("files", ["."])
    exclude = context.input_data.get("exclude", ["tests", "*.test.py"])

    # Extract config
    severity = context.config.get("severity", "medium")
    confidence = context.config.get("confidence", "medium")
    skip_tests = context.config.get("skip_tests", True)

    context.log(f"Running security scan with severity={severity}, confidence={confidence}")

    # Build bandit command
    cmd = ["bandit", "-r", "-f", "json"]

    # Severity filter
    severity_map = {"low": "l", "medium": "m", "high": "h"}
    if severity in severity_map:
        cmd.extend(["-l", severity_map[severity]])

    # Confidence filter
    if confidence in severity_map:
        cmd.extend(["-i", severity_map[confidence]])

    # Exclude patterns
    if exclude:
        cmd.extend(["--exclude", ",".join(exclude)])

    # Skip tests
    if skip_tests:
        cmd.append("-s")
        cmd.append("B101")  # Skip assert_used (common in tests)

    cmd.extend(files)

    # Run bandit
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=context.working_dir,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
    except asyncio.TimeoutError:
        context.error("Security scan timed out after 60 seconds")
        return {"vulnerabilities": [], "error": "Timeout"}
    except Exception as e:
        context.error(f"Failed to run bandit: {e}")
        return {"vulnerabilities": [], "error": str(e)}

    # Parse JSON output
    output_text = stdout.decode("utf-8", errors="replace")
    try:
        data = json.loads(output_text) if output_text.strip() else {}
    except json.JSONDecodeError:
        context.error("Failed to parse bandit output")
        return {"vulnerabilities": [], "error": "Parse error", "raw": output_text[:1000]}

    # Extract vulnerabilities
    vulnerabilities = []
    for result in data.get("results", []):
        vulnerabilities.append(
            {
                "file": result.get("filename", ""),
                "line": result.get("line_number", 0),
                "test_id": result.get("test_id", ""),
                "test_name": result.get("test_name", ""),
                "severity": result.get("issue_severity", "").lower(),
                "confidence": result.get("issue_confidence", "").lower(),
                "message": result.get("issue_text", ""),
                "cwe": result.get("issue_cwe", {}).get("id", ""),
                "more_info": result.get("more_info", ""),
            }
        )

    # Build summary
    summary = _build_summary(vulnerabilities)

    # Extract metrics
    metrics = data.get("metrics", {})
    total_metrics = metrics.get("_totals", {})

    context.log(f"Found {len(vulnerabilities)} security issues")

    return {
        "vulnerabilities": vulnerabilities,
        "summary": summary,
        "metrics": {
            "files_scanned": len(metrics) - 1 if "_totals" in metrics else len(metrics),
            "lines_of_code": total_metrics.get("loc", 0),
            "lines_skipped": total_metrics.get("nosec", 0),
        },
    }


def _build_summary(vulnerabilities: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary of vulnerabilities by severity and confidence."""
    by_severity: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    by_confidence: dict[str, int] = {"high": 0, "medium": 0, "low": 0}

    for vuln in vulnerabilities:
        sev = vuln.get("severity", "low")
        conf = vuln.get("confidence", "low")
        if sev in by_severity:
            by_severity[sev] += 1
        if conf in by_confidence:
            by_confidence[conf] += 1

    return {
        "total": len(vulnerabilities),
        "by_severity": by_severity,
        "by_confidence": by_confidence,
    }
