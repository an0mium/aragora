"""Tests for the lane supervisor CLI launch seam (``scripts/lane_supervisor.py``).

The drainer state machine is unit-tested in ``test_lane_supervisor.py`` with a
fake launch_fn. These tests cover the CLI's *real* ``_worker_launcher_launch``
wiring -- specifically that it provisions a worktree on the work order's branch
when none is pre-set and hands that path to ``WorkerLauncher.launch``. Git and
the launcher are mocked, so no worktree is created and no process is spawned.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location(
        "lane_supervisor_cli", _REPO_ROOT / "scripts" / "lane_supervisor.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_launch_seam_provisions_worktree_when_absent() -> None:
    cli = _load_cli()
    fake_launcher = MagicMock()
    fake_launcher.launch = AsyncMock()

    with (
        patch.object(cli, "_provision_lane_worktree", return_value="/tmp/wt-lane-x") as prov,
        patch("aragora.swarm.worker_launcher.WorkerLauncher", return_value=fake_launcher) as cls,
    ):
        cli._worker_launcher_launch(
            {
                "work_order_id": "lane-pr8426-x",
                "branch": "claude/swarm-graphql-routing",
                "prompt": "CLAIM-OR-YIELD #8426",
                "target_agent": "codex",
            }
        )

    # Provisioned for the order's branch...
    prov.assert_called_once()
    assert prov.call_args.args[0] == "claude/swarm-graphql-routing"
    assert cls.call_args.args[0].detach is True
    # ...and the provisioned path is what launch() receives.
    fake_launcher.launch.assert_awaited_once()
    assert fake_launcher.launch.await_args.kwargs["worktree_path"] == "/tmp/wt-lane-x"
    assert fake_launcher.launch.await_args.kwargs["branch"] == "claude/swarm-graphql-routing"


def test_launch_seam_reuses_preset_worktree_without_provisioning() -> None:
    cli = _load_cli()
    fake_launcher = MagicMock()
    fake_launcher.launch = AsyncMock()

    with (
        patch.object(cli, "_provision_lane_worktree") as prov,
        patch("aragora.swarm.worker_launcher.WorkerLauncher", return_value=fake_launcher) as cls,
    ):
        cli._worker_launcher_launch(
            {"work_order_id": "lane-y", "branch": "main", "worktree": "/existing/wt"}
        )

    prov.assert_not_called()
    assert cls.call_args.args[0].detach is True
    assert fake_launcher.launch.await_args.kwargs["worktree_path"] == "/existing/wt"


def test_provision_refuses_when_branch_checked_out_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Safety: never provision a second checkout of a branch already live in
    # another worktree -- that risks clobbering an interactive session's
    # unpushed work. Must raise rather than --force/-B reset the ref.
    cli = _load_cli()
    monkeypatch.setattr(cli, "_branch_checked_out_elsewhere", lambda *a, **k: True)
    with pytest.raises(RuntimeError, match="already checked out"):
        cli._provision_lane_worktree("claude/some-branch", "lane-w", repo_root=tmp_path)


def test_launch_seam_releases_lane_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # If launch fails after the conductor has claimed the lane, the seam must
    # release the claim (so the PR is re-dispatchable) and then re-raise so the
    # drainer records the order in failed/.
    cli = _load_cli()
    released: list[dict] = []
    monkeypatch.setattr(
        cli, "_release_lane_claim", lambda wo, *, repo_root=None: released.append(wo)
    )
    monkeypatch.setattr(cli, "_provision_lane_worktree", lambda *a, **k: "/tmp/wt")
    fake_launcher = MagicMock()
    fake_launcher.launch = AsyncMock(side_effect=RuntimeError("spawn exploded"))
    with patch("aragora.swarm.worker_launcher.WorkerLauncher", return_value=fake_launcher):
        with pytest.raises(RuntimeError, match="spawn exploded"):
            cli._worker_launcher_launch(
                {"work_order_id": "lane-x", "branch": "b", "owner_session_id": "sess-x"}
            )
    assert released and released[0]["owner_session_id"] == "sess-x"


def test_provision_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    # A git provisioning failure must raise so the drainer records the order in
    # failed/ rather than launching into a missing tree.
    import subprocess

    cli = _load_cli()

    def boom(*_a: Any, **_k: Any) -> None:
        raise subprocess.CalledProcessError(1, ["git", "fetch"])

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        cli._provision_lane_worktree("some-branch", "lane-z")
