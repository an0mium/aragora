"""Orchestration tests for scripts/settle_pr.py (the I/O wrapper around the pure
`settle_plan` brain). These exercise the collect->plan->apply wiring and the
exit-code contract that automation wrappers depend on, with `_collect`/`_auto_merge`
stubbed (no real `gh`/subprocess). The routing logic itself is covered by
test_settle_plan.py; this guards the glue.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SETTLE_PR = _REPO_ROOT / "scripts" / "settle_pr.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("settle_pr_cli_under_test", _SETTLE_PR)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cli(monkeypatch):
    mod = _load_module()
    # Never touch the network: --repo is always passed so _resolve_repo is a no-op,
    # but stub it defensively anyway.
    monkeypatch.setattr(mod, "_resolve_repo", lambda repo: repo or "owner/repo")
    return mod


def _payload(**over):
    base = {
        "tier": 2,
        "head_sha": "abc123",
        "has_supportive_quorum": True,
        "action": "post",
        "supportive_families": ["claude", "grok"],
        "dissenting_families": [],
        "items": [],
        "failures": [],
    }
    base.update(over)
    return base


def test_tier2_apply_auto_merge_success_exits_0(cli, monkeypatch):
    monkeypatch.setattr(cli, "_collect", lambda *a, **k: _payload())
    monkeypatch.setattr(
        cli,
        "_auto_merge",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="merged", stderr=""),
    )
    rc = cli.main(["--repo", "owner/repo", "--pr", "8512", "--apply"])
    assert rc == 0


def test_tier2_apply_auto_merge_failure_exits_1(cli, monkeypatch):
    monkeypatch.setattr(cli, "_collect", lambda *a, **k: _payload())
    monkeypatch.setattr(
        cli, "_auto_merge", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom")
    )
    rc = cli.main(["--repo", "owner/repo", "--pr", "8512", "--apply"])
    assert rc == 1


def test_quorum_unsatisfied_exits_1(cli, monkeypatch):
    monkeypatch.setattr(
        cli,
        "_collect",
        lambda *a, **k: _payload(has_supportive_quorum=False, supportive_families=["claude"]),
    )
    rc = cli.main(["--repo", "owner/repo", "--pr", "8512", "--apply"])
    assert rc == 1


def test_tier4_apply_surfaces_commands_and_exits_3(cli, monkeypatch):
    # Tier 3-4 --apply: commands are SURFACED, not run -> exit 3 (operator action
    # still required), never 0.
    monkeypatch.setattr(cli, "_collect", lambda *a, **k: _payload(tier=4))
    calls = {"auto_merge": 0}
    monkeypatch.setattr(
        cli, "_auto_merge", lambda *a, **k: calls.__setitem__("auto_merge", calls["auto_merge"] + 1)
    )
    rc = cli.main(["--repo", "owner/repo", "--pr", "8382", "--apply", "--operator-login", "alice"])
    assert rc == 3
    assert calls["auto_merge"] == 0  # never auto-merges a Tier-4 PR


def test_prepare_only_reroutes_to_operator_not_dead_end(cli, monkeypatch):
    # collect refused to post (action="prepare") with a stale tier=2: must re-route
    # to the operator path (exit 3 under --apply with a login), never silently
    # withhold auto-merge with no surfaced settle path.
    monkeypatch.setattr(
        cli,
        "_collect",
        lambda *a, **k: _payload(action="prepare", action_reason="tier promoted on recheck"),
    )
    auto = {"n": 0}
    monkeypatch.setattr(cli, "_auto_merge", lambda *a, **k: auto.__setitem__("n", auto["n"] + 1))
    rc = cli.main(["--repo", "owner/repo", "--pr", "8511", "--apply", "--operator-login", "alice"])
    assert rc == 3
    assert auto["n"] == 0  # prepare-only never auto-merges


def test_dry_run_ready_exits_0_without_mutating(cli, monkeypatch):
    monkeypatch.setattr(cli, "_collect", lambda *a, **k: _payload())
    auto = {"n": 0}
    monkeypatch.setattr(cli, "_auto_merge", lambda *a, **k: auto.__setitem__("n", auto["n"] + 1))
    rc = cli.main(["--repo", "owner/repo", "--pr", "8512"])  # no --apply
    assert rc == 0
    assert auto["n"] == 0  # dry-run never calls auto-merge
