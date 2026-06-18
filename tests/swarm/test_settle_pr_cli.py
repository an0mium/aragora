"""Orchestration tests for scripts/settle_pr.py (the I/O wrapper around the pure
`settle_plan` brain). These exercise the collect->plan->apply wiring and the
exit-code contract that automation wrappers depend on, with `_collect`/`_auto_merge`
stubbed (no real `gh`/subprocess). The routing logic itself is covered by
test_settle_plan.py; this guards the glue.
"""

from __future__ import annotations

import importlib.util
import json
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


def test_apply_prepare_clean_quorum_blocks_and_recollects(cli, monkeypatch, capsys):
    # collect refused to post under --apply despite a clean quorum (head moved /
    # recheck pending). "prepare" is NOT a tier signal: the PR must NOT be re-routed
    # to operator settlement (no Tier-4 commands bound to a superseded head); it
    # stays on auto_merge_green, BLOCKED with a re-collect instruction, exit 1, and
    # is never auto-merged over the refusal.
    monkeypatch.setattr(
        cli,
        "_collect",
        lambda *a, **k: _payload(action="prepare", action_reason="head moved since classification"),
    )
    auto = {"n": 0}
    monkeypatch.setattr(cli, "_auto_merge", lambda *a, **k: auto.__setitem__("n", auto["n"] + 1))
    rc = cli.main(
        ["--repo", "owner/repo", "--pr", "8511", "--apply", "--operator-login", "alice", "--json"]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert auto["n"] == 0  # never auto-merges over a refusal-to-post
    assert out["route"] == "auto_merge_green"  # NOT re-routed to operator
    assert out["next_steps"] == []  # no Tier-4 commands surfaced
    assert any("refused to post" in b for b in out["blockers"])


def test_dry_run_ready_exits_0_without_mutating(cli, monkeypatch):
    # Real collect emits action="prepare" on a dry-run (it never posts). That must
    # NOT be read as a tier-authority override -- an eligible Tier 0-2 dry-run must
    # stay on the auto-merge route and exit 0, not misroute to operator settlement.
    monkeypatch.setattr(cli, "_collect", lambda *a, **k: _payload(action="prepare"))
    auto = {"n": 0}
    monkeypatch.setattr(cli, "_auto_merge", lambda *a, **k: auto.__setitem__("n", auto["n"] + 1))
    rc = cli.main(["--repo", "owner/repo", "--pr", "8512"])  # no --apply
    assert rc == 0
    assert auto["n"] == 0  # dry-run never calls auto-merge


def test_apply_prepare_with_unsatisfied_quorum_stays_auto_merge_route(cli, monkeypatch, capsys):
    # Under --apply, action="prepare" caused by an INCOMPLETE quorum (not a tier
    # promotion) must not be mislabeled as operator settlement: the route stays
    # auto_merge_green (it is genuinely Tier 2), blocked on the quorum.
    monkeypatch.setattr(
        cli,
        "_collect",
        lambda *a, **k: _payload(
            action="prepare", has_supportive_quorum=False, supportive_families=["claude"]
        ),
    )
    rc = cli.main(["--repo", "owner/repo", "--pr", "8512", "--apply", "--json"])
    out = capsys.readouterr().out
    assert rc == 1
    assert json.loads(out)["route"] == "auto_merge_green"


def test_apply_prepare_with_dissent_stays_auto_merge_route(cli, monkeypatch, capsys):
    # Same for a dissent-caused prepare: not a tier override.
    monkeypatch.setattr(
        cli,
        "_collect",
        lambda *a, **k: _payload(action="prepare", dissenting_families=["grok"]),
    )
    rc = cli.main(["--repo", "owner/repo", "--pr", "8512", "--apply", "--json"])
    out = capsys.readouterr().out
    assert rc == 1
    assert json.loads(out)["route"] == "auto_merge_green"


def test_tier4_dry_run_previews_runbook_without_login(cli, monkeypatch, capsys):
    # A Tier-4 dry-run with no --operator-login is blocked (exit 1) but still
    # PREVIEWS the settle runbook so the operator sees it before resolving blockers.
    monkeypatch.setattr(cli, "_collect", lambda *a, **k: _payload(tier=4))
    rc = cli.main(["--repo", "owner/repo", "--pr", "8382"])  # dry-run, no login
    out = capsys.readouterr().out
    assert rc == 1  # blocked (no login) -> not ready
    assert "PREVIEW" in out
    assert "settle_tier4_pr.py" in out
    assert "<gh-login>" in out  # placeholder for the missing login


def test_collect_error_is_surfaced_in_text_output(cli, monkeypatch, capsys):
    # collect's root-cause error must appear in the default text path, not only --json.
    monkeypatch.setattr(
        cli, "_collect", lambda *a, **k: {"mode": "collect_evidence", "error": "boom: no JSON"}
    )
    rc = cli.main(["--repo", "owner/repo", "--pr", "8512"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "collect error" in out
    assert "boom: no JSON" in out
