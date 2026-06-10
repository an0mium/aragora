"""The fleet sentinel launchd installer must render a usable PATH.

Regression guard: launchd agents inherit a minimal PATH that lacks
/opt/homebrew/bin, so the sentinel's gh_auth check died with
FileNotFoundError('gh') -> status unknown -> exit 2. The rendered plist
must carry an EnvironmentVariables dict whose PATH includes Homebrew and
the standard system bins (mirrors scripts/install_boss_loop_launchd.sh,
which exports PATH in its command string).
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "scripts" / "install_fleet_sentinel_launchd.sh"

EXPECTED_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _render_dry_run() -> bytes:
    proc = subprocess.run(
        ["/bin/bash", str(INSTALLER), "--dry-run"],
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, f"--dry-run failed: {proc.stderr.decode()}"
    return proc.stdout


def test_dry_run_renders_environment_path() -> None:
    data = plistlib.loads(_render_dry_run())
    env = data.get("EnvironmentVariables")
    assert env is not None, "plist is missing the EnvironmentVariables dict"
    assert env.get("PATH") == EXPECTED_PATH


def test_dry_run_plist_keeps_core_keys() -> None:
    data = plistlib.loads(_render_dry_run())
    assert data["Label"] == "com.aragora.fleet-sentinel"
    assert data["ProgramArguments"][-1] == "--json"
    assert data["RunAtLoad"] is True
