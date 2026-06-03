"""Tests for the `aragora elo` CLI command (cmd_elo).

Covers two correctness/trust defects:

1. ``elo agent --agent <typo>`` must report the agent is *not found* rather than
   fabricating a confident-looking default 1500 ELO record for an agent that has
   never participated.
2. ``elo`` error paths (missing required ``--agent``, agent-not-found) must exit
   non-zero so scripts can detect failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from aragora.cli.commands.stats import cmd_elo
from aragora.ranking.elo import EloSystem

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_elo_cli(db_path: Path, *cli_args: str) -> subprocess.CompletedProcess[str]:
    """Spawn the *real* ``aragora elo`` CLI process and return its result.

    This drives the full dispatcher chain (``python -m aragora.cli.main`` ->
    ``main()`` -> ``args.func(args)`` -> ``raise SystemExit(main())``) so the
    assertion is on the actual OS-level process exit code, not just the value
    ``cmd_elo`` returns in-process. A unit test that only inspects ``cmd_elo``'s
    return value would pass even if the dispatcher dropped the int on the floor.
    """
    return subprocess.run(
        [sys.executable, "-m", "aragora.cli.main", "elo", *cli_args, "--db", str(db_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _args(db_path: Path, **kwargs) -> argparse.Namespace:
    base = {"db": str(db_path), "action": "leaderboard", "agent": None, "limit": 10}
    base.update(kwargs)
    return argparse.Namespace(**base)


class TestEloAgentNotFound:
    def test_unknown_agent_reports_not_found_not_fabricated_rating(
        self, tmp_path: Path, capsys
    ) -> None:
        """A typo'd / unknown agent must NOT yield a fabricated 1500 record."""
        db_path = tmp_path / "elo.db"
        # Touch the system so the schema exists but no agent is registered.
        EloSystem(db_path=str(db_path))

        result = cmd_elo(_args(db_path, action="agent", agent="totally_made_up_agent_xyz"))

        out = capsys.readouterr().out
        assert "not found" in out.lower()
        assert "totally_made_up_agent_xyz" in out
        # Must not present a fabricated rating as real data.
        assert "1500" not in out
        assert "ELO Rating" not in out
        # Error path -> non-zero exit.
        assert isinstance(result, int)
        assert result != 0

    def test_known_agent_shows_real_record(self, tmp_path: Path, capsys) -> None:
        """An agent with a recorded match still shows its real record and exits 0."""
        db_path = tmp_path / "elo.db"
        elo = EloSystem(db_path=str(db_path))
        # record_match persists rating rows for both participants.
        elo.record_match(winner="real_agent", loser="other_agent")

        result = cmd_elo(_args(db_path, action="agent", agent="real_agent"))

        out = capsys.readouterr().out
        assert "real_agent" in out
        assert "ELO Rating" in out
        assert "not found" not in out.lower()
        # Success -> exit 0 (None or 0 both treated as success by dispatcher).
        assert result in (None, 0)


class TestEloErrorExitCodes:
    def test_agent_action_missing_agent_arg_exits_nonzero(self, tmp_path: Path, capsys) -> None:
        db_path = tmp_path / "elo.db"
        result = cmd_elo(_args(db_path, action="agent", agent=None))

        out = capsys.readouterr().out
        assert "--agent is required" in out
        assert isinstance(result, int)
        assert result != 0

    def test_history_action_missing_agent_arg_exits_nonzero(self, tmp_path: Path, capsys) -> None:
        db_path = tmp_path / "elo.db"
        result = cmd_elo(_args(db_path, action="history", agent=None))

        out = capsys.readouterr().out
        assert "--agent is required" in out
        assert isinstance(result, int)
        assert result != 0

    def test_leaderboard_success_returns_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "elo.db"
        EloSystem(db_path=str(db_path))
        result = cmd_elo(_args(db_path, action="leaderboard"))
        assert result in (None, 0)


class TestEloSystemHasRating:
    def test_has_rating_false_for_unknown(self, tmp_path: Path) -> None:
        elo = EloSystem(db_path=str(tmp_path / "elo.db"))
        assert elo.has_rating("nobody") is False

    def test_has_rating_true_after_recorded_match(self, tmp_path: Path) -> None:
        elo = EloSystem(db_path=str(tmp_path / "elo.db"))
        elo.record_match(winner="somebody", loser="rival")
        assert elo.has_rating("somebody") is True

    def test_has_rating_does_not_create_row(self, tmp_path: Path) -> None:
        """Existence check must not implicitly register the agent."""
        elo = EloSystem(db_path=str(tmp_path / "elo.db"))
        elo.has_rating("ghost")
        assert "ghost" not in elo.list_agents()


class TestEloCliProcessExitCode:
    """Process-level regression tests for ``aragora elo`` exit codes.

    The unit tests above call ``cmd_elo`` directly and assert on its *return
    value*. That alone does not prove a script running ``aragora elo ...`` sees a
    non-zero exit: the dispatcher must propagate the int return into the process
    exit code. These tests spawn the real CLI subprocess and assert on
    ``returncode`` to lock that end-to-end contract.
    """

    def test_agent_unknown_process_exits_nonzero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "elo.db"
        EloSystem(db_path=str(db_path))  # create schema, no agents

        result = _run_elo_cli(db_path, "agent", "--agent", "totally_made_up_xyz")

        assert result.returncode != 0, result.stdout + result.stderr
        assert result.returncode == 1
        assert "not found" in result.stdout.lower()
        # Must not leak a fabricated default record.
        assert "1500" not in result.stdout
        assert "ELO Rating" not in result.stdout

    def test_agent_missing_required_arg_process_exits_nonzero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "elo.db"
        EloSystem(db_path=str(db_path))

        result = _run_elo_cli(db_path, "agent")

        assert result.returncode != 0, result.stdout + result.stderr
        assert result.returncode == 2
        assert "--agent is required" in result.stdout

    def test_history_missing_required_arg_process_exits_nonzero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "elo.db"
        EloSystem(db_path=str(db_path))

        result = _run_elo_cli(db_path, "history")

        assert result.returncode != 0, result.stdout + result.stderr
        assert result.returncode == 2
        assert "--agent is required" in result.stdout

    def test_leaderboard_success_process_exits_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "elo.db"
        EloSystem(db_path=str(db_path))

        result = _run_elo_cli(db_path)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Leaderboard" in result.stdout

    def test_known_agent_process_exits_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "elo.db"
        elo = EloSystem(db_path=str(db_path))
        elo.record_match(winner="real_agent", loser="other_agent")

        result = _run_elo_cli(db_path, "agent", "--agent", "real_agent")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "real_agent" in result.stdout
        assert "ELO Rating" in result.stdout
        assert "not found" not in result.stdout.lower()
