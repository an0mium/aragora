"""
Lint Plugin - Check code for style and quality issues.

Uses ruff (preferred) or flake8 as fallback to analyze Python code.

Example usage via API:
    POST /api/plugins/lint/run
    {
        "input": {"files": ["aragora/core.py"]},
        "config": {"max_line_length": 120}
    }
"""

import asyncio
import json
import shutil
from typing import Any, Optional

from aragora.plugins.runner import PluginContext


async def run(context: PluginContext) -> dict[str, Any]:
    """
    Run linting on specified files or directories.

    Input:
        files: List of files/directories to lint (default: ["."])
        exclude: Patterns to exclude (default: [])

    Config:
        max_line_length: Maximum line length (default: 120)
        select: List of rules to enable (default: [])
        ignore: List of rules to ignore (default: [])
        tool: "ruff" or "flake8" (default: auto-detect)

    Output:
        issues: List of linting issues found
        summary: Summary of issue counts by severity
        tool_used: Which linter was used
        files_checked: Number of files checked
    """
    # Extract input
    files = context.input_data.get("files", ["."])
    exclude = context.input_data.get("exclude", [])

    # Extract config
    max_line_length = context.config.get("max_line_length", 120)
    select_rules = context.config.get("select", [])
    ignore_rules = context.config.get("ignore", [])
    preferred_tool = context.config.get("tool", "auto")

    # Detect available linter
    tool = _detect_linter(preferred_tool)
    if not tool:
        context.error("No linter available. Install ruff or flake8.")
        return {"issues": [], "error": "No linter available"}

    context.log(f"Using {tool} for linting")

    # Build command
    if tool == "ruff":
        cmd = _build_ruff_command(files, exclude, max_line_length, select_rules, ignore_rules)
    else:
        cmd = _build_flake8_command(files, exclude, max_line_length, select_rules, ignore_rules)

    # Run linter
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=context.working_dir,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        context.error("Linting timed out after 30 seconds")
        return {"issues": [], "error": "Timeout"}
    except Exception as e:
        context.error(f"Failed to run linter: {e}")
        return {"issues": [], "error": str(e)}

    # Parse output
    output_text = stdout.decode("utf-8", errors="replace")
    issues = _parse_output(output_text, tool)

    # Build summary
    summary = _build_summary(issues)

    context.log(f"Found {len(issues)} issues")

    return {
        "issues": issues,
        "summary": summary,
        "tool_used": tool,
        "files_checked": len(files),
        "raw_output": output_text[:5000] if len(output_text) > 5000 else output_text,
    }


def _detect_linter(preferred: str) -> Optional[str]:
    """Detect which linter is available."""
    if preferred == "ruff" and shutil.which("ruff"):
        return "ruff"
    if preferred == "flake8" and shutil.which("flake8"):
        return "flake8"

    # Auto-detect
    if preferred == "auto":
        if shutil.which("ruff"):
            return "ruff"
        if shutil.which("flake8"):
            return "flake8"

    return None


def _build_ruff_command(
    files: list[str],
    exclude: list[str],
    max_line_length: int,
    select: list[str],
    ignore: list[str],
) -> list[str]:
    """Build ruff check command."""
    cmd = ["ruff", "check", "--output-format=json"]

    cmd.append(f"--line-length={max_line_length}")

    for pattern in exclude:
        cmd.append(f"--exclude={pattern}")

    if select:
        cmd.append(f"--select={','.join(select)}")

    if ignore:
        cmd.append(f"--ignore={','.join(ignore)}")

    cmd.extend(files)
    return cmd


def _build_flake8_command(
    files: list[str],
    exclude: list[str],
    max_line_length: int,
    select: list[str],
    ignore: list[str],
) -> list[str]:
    """Build flake8 command."""
    cmd = ["flake8", "--format=json"]

    cmd.append(f"--max-line-length={max_line_length}")

    if exclude:
        cmd.append(f"--exclude={','.join(exclude)}")

    if select:
        cmd.append(f"--select={','.join(select)}")

    if ignore:
        cmd.append(f"--ignore={','.join(ignore)}")

    cmd.extend(files)
    return cmd


def _parse_output(output: str, tool: str) -> list[dict[str, Any]]:
    """Parse linter output into structured issues."""
    issues = []

    if tool == "ruff":
        try:
            data = json.loads(output) if output.strip() else []
            for item in data:
                issues.append(
                    {
                        "file": item.get("filename", ""),
                        "line": item.get("location", {}).get("row", 0),
                        "column": item.get("location", {}).get("column", 0),
                        "code": item.get("code", ""),
                        "message": item.get("message", ""),
                        "severity": _code_to_severity(item.get("code", "")),
                    }
                )
        except json.JSONDecodeError:
            # Fall back to text parsing
            for line in output.strip().split("\n"):
                if ":" in line and line.strip():
                    issues.append({"raw": line, "severity": "warning"})

    elif tool == "flake8":
        # flake8 default format: file:line:col: CODE message
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(":", 3)
            if len(parts) >= 4:
                issues.append(
                    {
                        "file": parts[0],
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "column": int(parts[2]) if parts[2].isdigit() else 0,
                        "message": parts[3].strip(),
                        "severity": "warning",
                    }
                )

    return issues


def _code_to_severity(code: str) -> str:
    """Map lint code to severity level."""
    if code.startswith("E"):  # Error
        return "error"
    if code.startswith("W"):  # Warning
        return "warning"
    if code.startswith("F"):  # PyFlakes (usually errors)
        return "error"
    if code.startswith("C"):  # Convention
        return "info"
    if code.startswith("I"):  # Import order
        return "info"
    return "warning"


def _build_summary(issues: list[dict[str, Any]]) -> dict[str, int]:
    """Build summary of issues by severity."""
    summary = {"error": 0, "warning": 0, "info": 0, "total": len(issues)}
    for issue in issues:
        severity = issue.get("severity", "warning")
        if severity in summary:
            summary[severity] += 1
    return summary
