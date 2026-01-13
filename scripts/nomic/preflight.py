"""
Pre-flight health checks for the Nomic Loop.

Verifies system readiness before starting the self-improvement cycle:
- API key validation
- Database connectivity
- Disk space availability
- Protected file integrity
- Git repository state
"""

import os
import shutil
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PreflightCheck:
    """Result of a single pre-flight check."""

    def __init__(self, name: str, passed: bool, message: str, critical: bool = True):
        self.name = name
        self.passed = passed
        self.message = message
        self.critical = critical

    def __repr__(self) -> str:
        status = "PASS" if self.passed else ("FAIL" if self.critical else "WARN")
        return f"[{status}] {self.name}: {self.message}"


class PreflightReport:
    """Collection of pre-flight check results."""

    def __init__(self):
        self.checks: List[PreflightCheck] = []

    def add(self, check: PreflightCheck) -> None:
        self.checks.append(check)

    @property
    def all_passed(self) -> bool:
        return all(c.passed or not c.critical for c in self.checks)

    @property
    def critical_failures(self) -> List[PreflightCheck]:
        return [c for c in self.checks if not c.passed and c.critical]

    @property
    def warnings(self) -> List[PreflightCheck]:
        return [c for c in self.checks if not c.passed and not c.critical]

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print("PRE-FLIGHT HEALTH CHECK REPORT")
        print("=" * 70 + "\n")

        for check in self.checks:
            if check.passed:
                print(f"  [PASS] {check.name}")
                print(f"         {check.message}")
            elif check.critical:
                print(f"  [FAIL] {check.name}")
                print(f"         {check.message}")
            else:
                print(f"  [WARN] {check.name}")
                print(f"         {check.message}")
            print()

        print("=" * 70)
        if self.all_passed:
            print("STATUS: All critical checks passed. Ready to proceed.")
        else:
            failures = len(self.critical_failures)
            print(f"STATUS: {failures} critical check(s) failed. Cannot proceed.")
        print("=" * 70 + "\n")


def check_api_keys() -> List[PreflightCheck]:
    """Check that required API keys are configured."""
    checks = []

    # At least one primary API key required
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    has_primary = bool(anthropic_key) or bool(openai_key)
    checks.append(
        PreflightCheck(
            name="Primary API Key",
            passed=has_primary,
            message=(
                "Anthropic or OpenAI API key configured"
                if has_primary
                else "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set"
            ),
            critical=True,
        )
    )

    # Check individual keys
    if anthropic_key:
        # Basic format validation
        valid = anthropic_key.startswith("sk-ant-")
        checks.append(
            PreflightCheck(
                name="Anthropic API Key Format",
                passed=valid,
                message="Key format looks valid" if valid else "Key doesn't start with 'sk-ant-'",
                critical=False,
            )
        )

    if openai_key:
        valid = openai_key.startswith("sk-")
        checks.append(
            PreflightCheck(
                name="OpenAI API Key Format",
                passed=valid,
                message="Key format looks valid" if valid else "Key doesn't start with 'sk-'",
                critical=False,
            )
        )

    # OpenRouter fallback (recommended but not required)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    checks.append(
        PreflightCheck(
            name="OpenRouter Fallback",
            passed=bool(openrouter_key),
            message=(
                "Fallback configured"
                if openrouter_key
                else "Not configured - rate limit errors will not fall back"
            ),
            critical=False,
        )
    )

    return checks


def check_disk_space(path: Path, min_gb: float = 1.0) -> PreflightCheck:
    """Check available disk space for backups."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        passed = free_gb >= min_gb

        return PreflightCheck(
            name="Disk Space",
            passed=passed,
            message=(
                f"{free_gb:.1f}GB available"
                if passed
                else f"Only {free_gb:.1f}GB available (need {min_gb}GB for backups)"
            ),
            critical=True,
        )
    except Exception as e:
        return PreflightCheck(
            name="Disk Space",
            passed=False,
            message=f"Could not check disk space: {e}",
            critical=True,
        )


def check_database_connectivity(db_path: Path) -> PreflightCheck:
    """Check database file accessibility."""
    try:
        if not db_path.exists():
            return PreflightCheck(
                name=f"Database ({db_path.name})",
                passed=True,
                message="Will be created on first use",
                critical=False,
            )

        # Try to connect and run a simple query
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute("SELECT 1")
        conn.close()

        return PreflightCheck(
            name=f"Database ({db_path.name})",
            passed=True,
            message="Connected successfully",
            critical=True,
        )
    except Exception as e:
        return PreflightCheck(
            name=f"Database ({db_path.name})",
            passed=False,
            message=f"Connection failed: {e}",
            critical=True,
        )


def check_protected_files(aragora_path: Path, protected_files: List[str]) -> PreflightCheck:
    """Check that protected files exist and are readable."""
    missing = []
    for pf in protected_files:
        full_path = aragora_path / pf
        if not full_path.exists():
            missing.append(pf)

    if missing:
        return PreflightCheck(
            name="Protected Files",
            passed=False,
            message=f"Missing: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}",
            critical=True,
        )

    return PreflightCheck(
        name="Protected Files",
        passed=True,
        message=f"All {len(protected_files)} protected files present",
        critical=True,
    )


def check_git_repository(aragora_path: Path) -> List[PreflightCheck]:
    """Check git repository state."""
    import subprocess

    checks = []

    # Check if it's a git repository
    git_dir = aragora_path / ".git"
    if not git_dir.exists():
        checks.append(
            PreflightCheck(
                name="Git Repository", passed=False, message="Not a git repository", critical=True
            )
        )
        return checks

    checks.append(
        PreflightCheck(
            name="Git Repository", passed=True, message="Valid git repository", critical=True
        )
    )

    # Check for uncommitted changes
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=aragora_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        has_changes = bool(result.stdout.strip())
        checks.append(
            PreflightCheck(
                name="Working Directory",
                passed=not has_changes,
                message=(
                    "Clean" if not has_changes else "Has uncommitted changes (will be backed up)"
                ),
                critical=False,
            )
        )
    except Exception as e:
        checks.append(
            PreflightCheck(
                name="Working Directory",
                passed=False,
                message=f"Could not check: {e}",
                critical=False,
            )
        )

    return checks


def check_backup_directory(aragora_path: Path) -> PreflightCheck:
    """Check backup directory is writable."""
    backup_dir = aragora_path / ".nomic" / "backups"

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        test_file = backup_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()

        return PreflightCheck(
            name="Backup Directory", passed=True, message=f"Writable at {backup_dir}", critical=True
        )
    except Exception as e:
        return PreflightCheck(
            name="Backup Directory",
            passed=False,
            message=f"Cannot write to backup directory: {e}",
            critical=True,
        )


def run_preflight_checks(
    aragora_path: Optional[Path] = None, protected_files: Optional[List[str]] = None
) -> PreflightReport:
    """
    Run all pre-flight health checks.

    Args:
        aragora_path: Path to aragora repository (defaults to parent of scripts/)
        protected_files: List of protected file paths (relative to aragora_path)

    Returns:
        PreflightReport with all check results
    """
    if aragora_path is None:
        aragora_path = Path(__file__).parent.parent.parent

    if protected_files is None:
        protected_files = [
            "CLAUDE.md",
            "aragora/__init__.py",
            "aragora/core.py",
            "scripts/nomic_loop.py",
        ]

    report = PreflightReport()

    # API Keys
    for check in check_api_keys():
        report.add(check)

    # Disk Space
    report.add(check_disk_space(aragora_path, min_gb=1.0))

    # Database Connectivity
    db_paths = [
        aragora_path / "agent_elo.db",
        aragora_path / "continuum.db",
    ]
    for db_path in db_paths:
        if db_path.exists():
            report.add(check_database_connectivity(db_path))

    # Protected Files
    report.add(check_protected_files(aragora_path, protected_files))

    # Git Repository
    for check in check_git_repository(aragora_path):
        report.add(check)

    # Backup Directory
    report.add(check_backup_directory(aragora_path))

    return report


def preflight_cli(aragora_path: Optional[Path] = None) -> bool:
    """
    Run pre-flight checks and print report.

    Returns:
        True if all critical checks passed, False otherwise
    """
    report = run_preflight_checks(aragora_path)
    report.print_report()
    return report.all_passed


if __name__ == "__main__":
    import sys

    success = preflight_cli()
    sys.exit(0 if success else 1)
