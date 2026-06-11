"""Tests for ``scripts/install_fleet_sentinel_launchd.sh``.

Plan v2 Phase 0.1 (Pillar 6) follow-up. The installer renders the launchd
unit for the fleet sentinel; everything here is exercised via ``--dry-run``
so no live LaunchAgents state is touched.

Live incidents this file guards against regressing:

* 2026-06-10: under launchd's minimal default environment the sentinel's
  ``gh_auth`` check raised ``FileNotFoundError('gh')`` -> exit 2, because
  the rendered plist carried no PATH. A reinstall from this script must not
  regress the hand-patched live plist, so the render bakes
  ``EnvironmentVariables.PATH`` in.
* Breaches were silent: the unit had no notification channel. The render now
  wires a default macOS ``osascript`` notification through the sentinel's
  ``--notify-cmd`` ``{summary}`` template (disable with ``--notify-cmd ''``).
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "scripts" / "install_fleet_sentinel_launchd.sh"

EXPECTED_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _dry_run(*extra_args: str) -> str:
    proc = subprocess.run(
        ["/bin/bash", str(INSTALLER), "--dry-run", *extra_args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_installer_bash_syntax_clean() -> None:
    proc = subprocess.run(["/bin/bash", "-n", str(INSTALLER)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_installer_dry_run_renders_plist() -> None:
    out = _dry_run()
    assert "com.aragora.fleet-sentinel" in out
    assert "<integer>600</integer>" in out
    assert "fleet_sentinel.py" in out
    assert ".aragora/overnight/fleet-sentinel.log" in out


def test_installer_dry_run_output_is_parseable_plist() -> None:
    payload = plistlib.loads(_dry_run().encode())
    assert payload["Label"] == "com.aragora.fleet-sentinel"
    assert payload["StartInterval"] == 600
    assert payload["RunAtLoad"] is True
    assert payload["WorkingDirectory"] == str(REPO_ROOT)
    assert "--json" in payload["ProgramArguments"]


def test_installer_renders_path_environment_variable() -> None:
    """Regression guard for the 2026-06-10 launchd FileNotFoundError('gh')."""
    out = _dry_run()
    payload = plistlib.loads(out.encode())
    assert payload["EnvironmentVariables"]["PATH"] == EXPECTED_PATH
    assert EXPECTED_PATH in out


def test_installer_default_notify_cmd_wired() -> None:
    """Default render wires the macOS notification channel via osascript."""
    payload = plistlib.loads(_dry_run().encode())
    args = payload["ProgramArguments"]
    assert "--notify-cmd" in args
    notify_cmd = args[args.index("--notify-cmd") + 1]
    assert notify_cmd.startswith("osascript -e ")
    # Conforms to fleet_sentinel.py's template contract: {summary} placeholder
    # embedded in the AppleScript string literal.
    assert '\\"{summary}\\"' in notify_cmd
    assert "Aragora Fleet Sentinel" in notify_cmd


def test_installer_notify_cmd_empty_disables_channel() -> None:
    out = _dry_run("--notify-cmd", "")
    assert "--notify-cmd" not in out
    assert "osascript" not in out
    payload = plistlib.loads(out.encode())
    assert "--notify-cmd" not in payload["ProgramArguments"]


def test_installer_notify_cmd_custom_override() -> None:
    out = _dry_run("--notify-cmd", "my-notifier --msg {summary}")
    payload = plistlib.loads(out.encode())
    args = payload["ProgramArguments"]
    assert args[args.index("--notify-cmd") + 1] == "my-notifier --msg {summary}"
    assert "osascript" not in out


def test_installer_notify_cmd_xml_escaped() -> None:
    """Custom notify commands with XML-significant chars keep plist output valid."""
    out = _dry_run("--notify-cmd", "notifier --and=a&b --lt=<x> {summary}")
    payload = plistlib.loads(out.encode())
    args = payload["ProgramArguments"]
    assert args[args.index("--notify-cmd") + 1] == "notifier --and=a&b --lt=<x> {summary}"


def test_installer_rejects_unknown_args() -> None:
    proc = subprocess.run(
        ["/bin/bash", str(INSTALLER), "--bogus"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 64
    assert "usage" in proc.stderr


def test_installer_requires_a_mode() -> None:
    proc = subprocess.run(
        ["/bin/bash", str(INSTALLER), "--notify-cmd", "x"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 64
