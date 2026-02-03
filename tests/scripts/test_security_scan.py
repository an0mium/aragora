"""Tests for the security_scan CLI tool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the script
SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "security_scan.py"


class TestSecurityScanCLI:
    """Tests for security_scan.py CLI tool."""

    def run_cli(self, *args, env=None, cwd=None):
        """Run the security_scan CLI with given arguments."""
        cmd = [sys.executable, str(SCRIPT_PATH)] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
            cwd=cwd,
        )
        return result

    def test_help_flag(self):
        """Test that --help shows usage information."""
        result = self.run_cli("--help")

        assert result.returncode == 0
        stdout_lower = result.stdout.lower()
        assert "usage" in stdout_lower or "Aragora Security Scanner" in result.stdout

    def test_basic_scan_runs(self):
        """Test that a basic scan completes without crashing."""
        result = self.run_cli("--path", "aragora/audit/", "--quiet")

        # Exit code 0 (no critical findings) or 1 (critical findings found) are both valid
        assert result.returncode in (0, 1)
        assert "Scan Results" in result.stdout

    def test_quiet_mode_suppresses_findings(self):
        """Test that quiet mode suppresses detailed finding output."""
        result = self.run_cli("--path", "aragora/audit/", "--quiet")

        # Quiet mode should suppress the detailed "CRITICAL Findings:" section
        assert "CRITICAL Findings:" not in result.stdout

    def test_json_output(self, tmp_path, monkeypatch):
        """Test that --json writes a valid JSON report file."""
        monkeypatch.chdir(tmp_path)
        result = self.run_cli("--path", "aragora/audit/", "--json", "--quiet", cwd=str(tmp_path))

        report_path = tmp_path / "security-report.json"
        assert report_path.exists(), "security-report.json should be created"

        with open(report_path) as f:
            data = json.load(f)

        assert "files_scanned" in data

    def test_invalid_path_handled(self):
        """Test that scanning a nonexistent path does not crash with a traceback."""
        result = self.run_cli("--path", "nonexistent_dir_xyz/")

        # The script should handle the error gracefully; it should not produce
        # an unhandled Python traceback.  A non-zero exit code is acceptable.
        assert "Traceback" not in result.stderr
