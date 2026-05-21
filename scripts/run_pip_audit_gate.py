#!/usr/bin/env python3
"""Run the Aragora product dependency pip-audit gate.

The security gate installs audit tools in CI before scanning dependencies. Audit
the exported product lockfile instead of the tool environment so transient tool
dependencies do not block unrelated PRs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = PROJECT_ROOT / "scripts/security/pip_audit_ignored_vulns.txt"


def load_ignored_vulns(path: Path = DEFAULT_ALLOWLIST) -> list[str]:
    """Load vulnerability IDs, allowing blank lines and shell-style comments."""
    ignored: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        ignored.append(line.split()[0])
    return ignored


def build_pip_audit_command(
    requirements_path: Path,
    ignored_vulns: list[str],
    *,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the pip-audit command for the exported product requirements."""
    cmd = [
        python_executable,
        "-m",
        "pip_audit",
        "--strict",
        "--vulnerability-service",
        "osv",
        "--requirement",
        str(requirements_path),
    ]
    for vuln_id in ignored_vulns:
        cmd.extend(["--ignore-vuln", vuln_id])
    return cmd


def export_requirements(output_path: Path) -> None:
    """Export locked product dependencies for auditing."""
    subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--all-extras",
            "--all-groups",
            "--no-emit-project",
            "--no-hashes",
            "--output-file",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


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
