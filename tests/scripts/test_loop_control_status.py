"""Tests for ``scripts/loop_control_status.py`` and the IO collectors.

The central assertion is the **read-only proof**: driving the CLI with subprocess
faked records every command issued and asserts they are all read-only (no merge /
comment / rerun / push / --apply), and that no filesystem write occurs.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import pathlib
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import aragora.swarm.loop_control_io as io_mod  # noqa: E402


def _load_cli() -> Any:
    script_path = REPO_ROOT / "scripts" / "loop_control_status.py"
    spec = importlib.util.spec_from_file_location("loop_control_status_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()

# Mutating arguments that must never appear as an exact arg in any command the
# plane issues. Scanned as exact args (not substrings) so a benign launchd label
# such as ``com.aragora.swarm-merge-arbiter`` is not a false positive.
FORBIDDEN_TOKENS = (
    "merge",
    "comment",
    "rerun",
    "--apply",
    "push",
    "approve",
    "commit",
    "reset",
    "checkout",
    "clean",
    "-X",
    "POST",
    "PATCH",
    "DELETE",
)
ALLOWED_PY_SCRIPTS = ("publisher_freshness_check.py", "agent_bridge.py")


class _FakeProc:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _healthy_run(recorded: list[list[str]]):
    def run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        recorded.append(list(cmd))
        joined = " ".join(cmd)
        if "publisher_freshness_check.py" in joined:
            return _FakeProc(
                json.dumps({"verdict": "ready", "launchd_loaded": True, "blockers": []})
            )
        if "agent_bridge.py" in joined:
            return _FakeProc(
                json.dumps(
                    {
                        "boss_loop_alive": True,
                        "boss_loop_status": {"owner": "swarm-boss-loop"},
                        "agent_heartbeats": {"count": 2, "fresh_count": 2},
                        "health": {"ok": True},
                        "queue_depth": 0,
                        "timestamp": "2026-06-09T00:00:00Z",
                    }
                )
            )
        if cmd[:2] == ["launchctl", "print"]:
            return _FakeProc("loaded", 0)
        if cmd[:3] == ["git", "worktree", "list"]:
            return _FakeProc("worktree /a\nworktree /b\n", 0)
        return _FakeProc("", 0)

    return run


def test_cli_json_offline(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr(io_mod.subprocess, "run", _healthy_run(recorded))
    rc = cli.main(["--repo", str(tmp_path), "--no-network", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"], "expected at least one record"
    for record in payload["records"]:
        assert record["schema_version"] == "loop-control/v1"
    assert "fleet_safe_to_continue" in payload["summary"]
    # --no-network must not invoke the operator snapshot.
    assert all("agent_bridge.py" not in " ".join(cmd) for cmd in recorded)


def test_cli_with_network_invokes_operator_snapshot(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path
) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr(io_mod.subprocess, "run", _healthy_run(recorded))
    rc = cli.main(["--repo", str(tmp_path), "--json"])
    assert rc == 0
    assert any("agent_bridge.py" in " ".join(cmd) for cmd in recorded)
    assert any("operator-snapshot" in cmd for cmd in recorded)


def test_only_read_only_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr(io_mod.subprocess, "run", _healthy_run(recorded))
    cli.main(["--repo", str(tmp_path), "--json"])
    assert recorded, "expected commands to be issued"
    for cmd in recorded:
        # Exact-arg scan (not substring) so labels like swarm-merge-arbiter pass.
        for bad in FORBIDDEN_TOKENS:
            assert bad not in cmd, f"mutating arg {bad!r} found in {cmd}"
        head = cmd[0]
        is_python = "python" in head.lower() or head == sys.executable
        assert head in ("launchctl", "git") or is_python, f"unexpected command head: {cmd}"
        if head == "git":
            assert cmd[:3] == ["git", "worktree", "list"], f"non-read-only git command: {cmd}"
        elif head == "launchctl":
            assert cmd[1] == "print", f"non-read-only launchctl command: {cmd}"
        else:
            script = next((arg for arg in cmd if arg.endswith(".py")), "")
            assert any(script.endswith(name) for name in ALLOWED_PY_SCRIPTS), f"unexpected: {cmd}"
            if script.endswith("agent_bridge.py"):
                assert "operator-snapshot" in cmd, f"agent_bridge used non-read-only: {cmd}"


def test_no_filesystem_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr(io_mod.subprocess, "run", _healthy_run(recorded))

    def _boom_write_text(self: pathlib.Path, *args: Any, **kwargs: Any) -> int:
        raise AssertionError(f"unexpected write_text to {self}")

    monkeypatch.setattr(pathlib.Path, "write_text", _boom_write_text)

    real_open = builtins.open
    write_opens: list[tuple[str, str]] = []

    def _guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            write_opens.append((str(file), mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded_open)

    rc = cli.main(["--repo", str(tmp_path), "--json"])
    assert rc == 0
    assert write_opens == [], f"unexpected filesystem writes: {write_opens}"


def test_exit_nonzero_on_halt(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path) -> None:
    def degraded_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        joined = " ".join(cmd)
        if "publisher_freshness_check.py" in joined:
            return _FakeProc(
                json.dumps(
                    {
                        "verdict": "degraded",
                        "launchd_loaded": False,
                        "blockers": ["launchd: not-loaded"],
                    }
                )
            )
        return _FakeProc("", 0)

    monkeypatch.setattr(io_mod.subprocess, "run", degraded_run)
    rc = cli.main(
        ["--repo", str(tmp_path), "--loop", "publisher", "--no-network", "--exit-nonzero-on-halt"]
    )
    assert rc == 1
    # And the same fleet is "safe" when nothing is degraded.
    recorded: list[list[str]] = []
    monkeypatch.setattr(io_mod.subprocess, "run", _healthy_run(recorded))
    rc_ok = cli.main(
        ["--repo", str(tmp_path), "--loop", "publisher", "--no-network", "--exit-nonzero-on-halt"]
    )
    assert rc_ok == 0


def test_collect_all_skips_network_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr(io_mod.subprocess, "run", _healthy_run(recorded))
    raw = io_mod.collect_all(tmp_path, allow_network=False)
    boss = raw[io_mod.LoopKind.BOSS_LOOP]
    assert boss["source_status"] == "unavailable"
    assert "no-network" in boss.get("error", "")


def test_boss_loop_owner_stale_ignores_terminal_heartbeat_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def terminal_only_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        assert "agent_bridge.py" in " ".join(cmd)
        return _FakeProc(
            json.dumps(
                {
                    "boss_loop_alive": False,
                    "boss_loop_status": {"owner": "swarm-boss-loop"},
                    "agent_heartbeats": {
                        "count": 2,
                        "fresh_count": 0,
                        "stale_count": 0,
                        "terminal_count": 2,
                    },
                    "health": {"ok": True},
                    "queue_depth": 7,
                    "timestamp": "2026-06-09T00:00:00Z",
                }
            )
        )

    monkeypatch.setattr(io_mod, "_run", terminal_only_run)

    raw = io_mod.collect_boss_loop(tmp_path)

    assert raw["owner_stale"] is False
