#!/usr/bin/env python3
"""Run the Aragora locked dependency pip-audit gate.

The security gate installs audit tools in CI before scanning dependencies. Audit
the exported project lockfile instead of the tool environment so transient tool
dependencies do not block unrelated PRs.

The export intentionally includes all extras and dependency groups. That keeps
runtime, dev, test, docs, and CI-executed project dependencies in one gate while
still excluding dependencies that belong only to the security-tool environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = PROJECT_ROOT / "scripts/security/pip_audit_ignored_vulns.txt"
EXPIRY_WARNING_WINDOW_DAYS = 14
VULN_ID_RE = re.compile(r"^(CVE-\d{4}-\d+|GHSA-[a-z0-9-]+|PYSEC-\d{4}-\d+)$")


def load_ignored_vulns(
    path: Path = DEFAULT_ALLOWLIST,
    *,
    today: dt.date | None = None,
) -> list[str]:
    """Load non-expired vulnerability IDs from ``VULN-ID YYYY-MM-DD`` entries."""
    today = today or dt.date.today()
    ignored: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f"{path}:{line_number}: expected 'VULN-ID YYYY-MM-DD # rationale'")

        vuln_id, expiry_raw = parts
        if not VULN_ID_RE.fullmatch(vuln_id):
            raise SystemExit(f"{path}:{line_number}: invalid vulnerability ID '{vuln_id}'")

        try:
            expires_at = dt.date.fromisoformat(expiry_raw)
        except ValueError as exc:
            raise SystemExit(
                f"{path}:{line_number}: invalid expiry date '{expiry_raw}', expected YYYY-MM-DD"
            ) from exc

        days_until_expiry = (expires_at - today).days
        if days_until_expiry <= 0:
            print(
                f"{path}:{line_number}: allowlist entry {vuln_id} expired on {expires_at}; "
                "not passing it to pip-audit",
                file=sys.stderr,
            )
            continue
        if days_until_expiry <= EXPIRY_WARNING_WINDOW_DAYS:
            print(
                f"{path}:{line_number}: allowlist entry {vuln_id} expires in "
                f"{days_until_expiry} day(s) on {expires_at}",
                file=sys.stderr,
            )
        ignored.append(vuln_id)
    return ignored


def build_pip_audit_command(
    requirements_path: Path,
    ignored_vulns: list[str],
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the pip-audit command for the exported project requirements."""
    cmd = [
        python_executable,
        "-m",
        "pip_audit",
        "--strict",
        "--vulnerability-service",
        "osv",
        "--requirement",
        str(requirements_path),
        "--no-deps",
        "--disable-pip",
    ]
    for vuln_id in ignored_vulns:
        cmd.extend(["--ignore-vuln", vuln_id])
    return cmd


def export_requirements(output_path: Path) -> None:
    """Export all locked project dependency groups for auditing."""
    cmd = [
        "uv",
        "export",
        "--frozen",
        "--all-extras",
        "--all-groups",
        "--no-emit-project",
        "--no-hashes",
        "--output-file",
        str(output_path),
    ]
    try:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "uv export failed: uv is not installed or is not on PATH. Install uv and retry."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = f"\n{stderr}" if stderr else ""
        raise SystemExit(f"uv export failed with exit code {exc.returncode}:{detail}") from exc


def run_gate(requirements_path: Path | None, allowlist_path: Path) -> int:
    ignored_vulns = load_ignored_vulns(allowlist_path)
    if requirements_path is not None:
        cmd = build_pip_audit_command(requirements_path, ignored_vulns)
        return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode

    with tempfile.TemporaryDirectory(prefix="aragora-pip-audit-") as tmp_dir:
        exported = Path(tmp_dir) / "requirements.txt"
        export_requirements(exported)
        cmd = build_pip_audit_command(exported, ignored_vulns)
        return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        help="Audit an existing requirements file instead of exporting uv.lock.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help="Path to newline-delimited pip-audit vulnerability IDs to ignore.",
    )
    args = parser.parse_args(argv)
    return run_gate(args.requirements, args.allowlist)


if __name__ == "__main__":
    raise SystemExit(main())
