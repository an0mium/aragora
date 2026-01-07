"""
Test Runner Plugin - Run pytest test suites.

Executes pytest and returns structured test results.

Example usage via API:
    POST /api/plugins/test-runner/run
    {
        "input": {"paths": ["tests/"]},
        "config": {"verbose": true, "markers": "not slow"}
    }
"""

import asyncio
import json
import re
import shutil

from aragora.plugins.runner import PluginContext


async def run(context: PluginContext) -> dict:
    """
    Run pytest on specified test paths.

    Input:
        paths: List of test paths (default: ["tests/"])
        pattern: Test file pattern (default: "test_*.py")
        keyword: Keyword expression for test selection (default: None)

    Config:
        verbose: Enable verbose output (default: false)
        markers: pytest marker expression (default: None)
        maxfail: Stop after N failures (default: 0 = no limit)
        timeout: Per-test timeout in seconds (default: 30)

    Output:
        passed: Number of passed tests
        failed: Number of failed tests
        skipped: Number of skipped tests
        errors: Number of errors
        duration: Total duration in seconds
        failures: Details of failed tests
    """
    # Check if pytest is available
    if not shutil.which("pytest"):
        context.error("pytest is not installed. Run: pip install pytest")
        return {"passed": 0, "failed": 0, "error": "pytest not installed"}

    # Extract input
    paths = context.input_data.get("paths", ["tests/"])
    pattern = context.input_data.get("pattern", "test_*.py")
    keyword = context.input_data.get("keyword")

    # Extract config
    verbose = context.config.get("verbose", False)
    markers = context.config.get("markers")
    maxfail = context.config.get("maxfail", 0)
    per_test_timeout = context.config.get("timeout", 30)

    context.log(f"Running tests in: {', '.join(paths)}")

    # Build pytest command
    cmd = ["pytest", "--tb=short", "-q"]

    # Output format for parsing
    cmd.append("--json-report")
    cmd.append("--json-report-file=-")  # Output to stdout

    if verbose:
        cmd.append("-v")

    if markers:
        cmd.extend(["-m", markers])

    if keyword:
        cmd.extend(["-k", keyword])

    if maxfail > 0:
        cmd.append(f"--maxfail={maxfail}")

    if per_test_timeout:
        cmd.append(f"--timeout={per_test_timeout}")

    cmd.extend(paths)

    # Run pytest
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=context.working_dir,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=300.0  # 5 minute overall timeout
        )
        exit_code = process.returncode
    except asyncio.TimeoutError:
        context.error("Test run timed out after 5 minutes")
        return {"passed": 0, "failed": 0, "error": "Timeout"}
    except Exception as e:
        context.error(f"Failed to run pytest: {e}")
        return {"passed": 0, "failed": 0, "error": str(e)}

    # Parse output
    output_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    # Try to parse JSON report
    result = _parse_json_report(output_text)

    if not result:
        # Fall back to text parsing
        result = _parse_text_output(output_text)

    # Add stderr if there were errors
    if stderr_text.strip():
        result["stderr"] = stderr_text[:2000]

    result["exit_code"] = exit_code
    result["success"] = exit_code == 0

    context.log(
        f"Tests complete: {result.get('passed', 0)} passed, "
        f"{result.get('failed', 0)} failed, {result.get('skipped', 0)} skipped"
    )

    return result


def _parse_json_report(output: str) -> dict:
    """Parse pytest-json-report output."""
    # Find JSON in output
    try:
        # Look for JSON object in output
        match = re.search(r'\{[^{}]*"summary"[^{}]*\}', output, re.DOTALL)
        if match:
            data = json.loads(match.group())
            summary = data.get("summary", {})
            return {
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "skipped": summary.get("skipped", 0),
                "errors": summary.get("error", 0),
                "duration": data.get("duration", 0),
                "total": summary.get("total", 0),
            }
    except (json.JSONDecodeError, AttributeError):
        pass
    return {}


def _parse_text_output(output: str) -> dict:
    """Parse pytest text output as fallback."""
    result = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "duration": 0,
    }

    # Look for summary line like "5 passed, 2 failed in 1.23s"
    summary_match = re.search(
        r'(\d+) passed.*?(?:(\d+) failed)?.*?(?:(\d+) skipped)?.*?in ([\d.]+)s',
        output, re.IGNORECASE
    )
    if summary_match:
        result["passed"] = int(summary_match.group(1) or 0)
        result["failed"] = int(summary_match.group(2) or 0)
        result["skipped"] = int(summary_match.group(3) or 0)
        result["duration"] = float(summary_match.group(4) or 0)

    # Also try other patterns
    if not summary_match:
        for pattern, key in [
            (r'(\d+) passed', 'passed'),
            (r'(\d+) failed', 'failed'),
            (r'(\d+) skipped', 'skipped'),
            (r'(\d+) error', 'errors'),
        ]:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                result[key] = int(match.group(1))

    # Extract failure details
    failures = []
    failure_blocks = re.findall(
        r'FAILED ([^\n]+)\n(.*?)(?=FAILED|$)',
        output, re.DOTALL
    )
    for test_name, details in failure_blocks[:10]:  # Limit to 10
        failures.append({
            "test": test_name.strip(),
            "details": details.strip()[:500],
        })

    if failures:
        result["failures"] = failures

    return result
