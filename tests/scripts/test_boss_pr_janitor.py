"""Tests for ``scripts/boss_pr_janitor.py`` (boss PR dedupe janitor).

The janitor must never hit the network in tests: the ``gh`` boundary is
mocked via ``monkeypatch`` on the module's ``subprocess.run`` and/or its
``fetch_open_prs`` function, mirroring the style of
``tests/scripts/test_settle_tier4_pr.py``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


janitor = _load_module("boss_pr_janitor.py")


def _pr(
    number: int,
    head: str,
    *,
    draft: bool = True,
    created: str = "2026-06-09T00:00:00Z",
    checks: list[dict[str, str]] | None = None,
    title: str = "boss: automated change",
) -> dict[str, Any]:
    return {
        "number": number,
        "headRefName": head,
        "isDraft": draft,
        "createdAt": created,
        "title": title,
        "statusCheckRollup": checks if checks is not None else [],
    }


def _failing_check() -> list[dict[str, str]]:
    return [{"state": "FAILURE", "conclusion": "FAILURE"}]


def _passing_check() -> list[dict[str, str]]:
    return [{"state": "SUCCESS", "conclusion": "SUCCESS"}]


# ---------------------------------------------------------------------------
# (a) duplicate drafts → one keeper, rest closed with comment naming keeper
# ---------------------------------------------------------------------------


def test_two_drafts_same_issue_keeps_one_closes_other() -> None:
    prs = [
        _pr(100, "aragora/boss-harvest/issue-8061-boss-aaaa", created="2026-06-08T00:00:00Z"),
        _pr(101, "aragora/boss-harvest/issue-8061-boss-bbbb", created="2026-06-09T00:00:00Z"),
    ]
    plan = janitor.build_plan(prs)

    assert len(plan) == 1
    action = plan[0]
    assert action["action"] == "close"
    assert action["pr"] == 100
    assert action["keeper"] == 101
    assert action["issue"] == 8061
    assert "#101" in action["comment"]
    assert "Superseded by #101" in action["comment"]


def test_keeper_prefers_most_recent_with_passing_or_pending_checks() -> None:
    # Newest PR has failing checks; keeper must be the most recent among
    # passing/pending ones instead.
    prs = [
        _pr(
            200,
            "aragora/boss-harvest/issue-8002-boss-aaaa",
            created="2026-06-07T00:00:00Z",
            checks=_passing_check(),
        ),
        _pr(
            201,
            "aragora/boss-harvest/issue-8002-boss-bbbb",
            created="2026-06-08T00:00:00Z",
            checks=[],  # pending / no checks yet → eligible
        ),
        _pr(
            202,
            "aragora/boss-harvest/issue-8002-boss-cccc",
            created="2026-06-09T00:00:00Z",
            checks=_failing_check(),
        ),
    ]
    plan = janitor.build_plan(prs)

    keepers = {a["keeper"] for a in plan}
    closed = sorted(a["pr"] for a in plan)
    assert keepers == {201}
    assert closed == [200, 202]


def test_keeper_falls_back_to_most_recent_when_all_failing() -> None:
    prs = [
        _pr(
            300,
            "aragora/boss/issue-7818-fix-old",
            created="2026-06-07T00:00:00Z",
            checks=_failing_check(),
        ),
        _pr(
            301,
            "aragora/boss/issue-7818-fix-new",
            created="2026-06-09T00:00:00Z",
            checks=_failing_check(),
        ),
    ]
    plan = janitor.build_plan(prs)

    assert len(plan) == 1
    assert plan[0]["pr"] == 300
    assert plan[0]["keeper"] == 301


# ---------------------------------------------------------------------------
# (b) single PR per issue → untouched
# ---------------------------------------------------------------------------


def test_single_pr_per_issue_untouched() -> None:
    prs = [
        _pr(400, "aragora/boss-harvest/issue-1111-boss-aaaa"),
        _pr(401, "aragora/boss/issue-2222-something"),
    ]
    assert janitor.build_plan(prs) == []


# ---------------------------------------------------------------------------
# (c) non-draft (ready) PRs are NEVER closed even if duplicated
# ---------------------------------------------------------------------------


def test_ready_prs_never_closed() -> None:
    prs = [
        _pr(500, "aragora/boss-harvest/issue-3333-boss-aaaa", draft=False),
        _pr(501, "aragora/boss-harvest/issue-3333-boss-bbbb", draft=False),
    ]
    assert janitor.build_plan(prs) == []


def test_ready_pr_in_mixed_group_untouched_drafts_deduped() -> None:
    prs = [
        _pr(600, "aragora/boss-harvest/issue-4444-boss-aaaa", draft=False),
        _pr(601, "aragora/boss-harvest/issue-4444-boss-bbbb", created="2026-06-08T00:00:00Z"),
        _pr(602, "aragora/boss-harvest/issue-4444-boss-cccc", created="2026-06-09T00:00:00Z"),
    ]
    plan = janitor.build_plan(prs)

    closed = {a["pr"] for a in plan}
    assert 600 not in closed, "ready PR must never be closed"
    assert closed == {601}
    assert plan[0]["keeper"] == 602


def test_single_draft_duplicating_ready_pr_not_closed() -> None:
    # Only one draft in the group → nothing to dedupe among drafts.
    prs = [
        _pr(700, "aragora/boss-harvest/issue-5555-boss-aaaa", draft=False),
        _pr(701, "aragora/boss-harvest/issue-5555-boss-bbbb"),
    ]
    assert janitor.build_plan(prs) == []


# ---------------------------------------------------------------------------
# (d) branches outside boss prefixes are never considered
# ---------------------------------------------------------------------------


def test_non_boss_branches_never_considered() -> None:
    prs = [
        _pr(800, "feature/issue-6666-cool-thing"),
        _pr(801, "codex/issue-6666-other-thing"),
        _pr(802, "elves/run-20260610-issue-6666"),
        _pr(803, "aragora/bossy/issue-6666-trap"),
        _pr(804, "prefix/aragora/boss-harvest/issue-6666-nested"),
    ]
    assert janitor.build_plan(prs) == []


def test_extract_issue_number_prefix_matching() -> None:
    assert janitor.extract_issue_number("aragora/boss-harvest/issue-8061-boss-ab12") == 8061
    assert janitor.extract_issue_number("aragora/boss/issue-7818-fix-thing") == 7818
    assert janitor.extract_issue_number("aragora/boss-harvest/issue-") is None
    assert janitor.extract_issue_number("feature/issue-123-x") is None
    assert janitor.extract_issue_number("aragora/boss-harvestissue-123") is None


# ---------------------------------------------------------------------------
# (e) dry-run is the default: no mutating gh command without --apply
# ---------------------------------------------------------------------------


def _duplicate_fixture() -> list[dict[str, Any]]:
    return [
        _pr(900, "aragora/boss-harvest/issue-9001-boss-aaaa", created="2026-06-08T00:00:00Z"),
        _pr(901, "aragora/boss-harvest/issue-9001-boss-bbbb", created="2026-06-09T00:00:00Z"),
    ]


def test_dry_run_default_invokes_no_mutating_gh_command(monkeypatch: Any, capsys: Any) -> None:
    calls: list[list[str]] = []

    def record_run(command: list[str], **kwargs: Any) -> Any:
        calls.append(list(command))
        raise AssertionError(f"subprocess must not run in dry-run with mocked fetch: {command}")

    monkeypatch.setattr(janitor, "fetch_open_prs", lambda repo: _duplicate_fixture())
    monkeypatch.setattr(janitor.subprocess, "run", record_run)

    exit_code = janitor.main(["--repo", "synaptent/aragora"])

    assert exit_code == 0
    assert calls == []
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    plans = [json.loads(line) for line in out_lines]
    close_plans = [p for p in plans if p.get("action") == "close"]
    assert len(close_plans) == 1
    assert close_plans[0]["pr"] == 900
    assert close_plans[0]["keeper"] == 901
    assert close_plans[0]["dry_run"] is True


def test_apply_invokes_close_and_fails_closed_on_error(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(janitor, "fetch_open_prs", lambda repo: _duplicate_fixture())
    monkeypatch.setattr(janitor.subprocess, "run", fake_run)

    exit_code = janitor.main(["--repo", "synaptent/aragora", "--apply"])

    assert exit_code == 1, "any failed apply mutation must fail closed"
    close_commands = [c for c in commands if "close" in c]
    assert len(close_commands) == 1
    cmd = close_commands[0]
    assert cmd[:3] == ["gh", "pr", "close"]
    assert "900" in cmd
    assert any("Superseded by #901" in part for part in cmd)


def test_apply_success_exits_zero(monkeypatch: Any) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(janitor, "fetch_open_prs", lambda repo: _duplicate_fixture())
    monkeypatch.setattr(janitor.subprocess, "run", fake_run)

    assert janitor.main(["--repo", "synaptent/aragora", "--apply"]) == 0


def test_apply_never_closes_ready_prs(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    ready_dupes = [
        _pr(950, "aragora/boss-harvest/issue-9050-boss-aaaa", draft=False),
        _pr(951, "aragora/boss-harvest/issue-9050-boss-bbbb", draft=False),
    ]
    monkeypatch.setattr(janitor, "fetch_open_prs", lambda repo: ready_dupes)
    monkeypatch.setattr(janitor.subprocess, "run", fake_run)

    assert janitor.main(["--repo", "synaptent/aragora", "--apply"]) == 0
    assert all("close" not in c for c in commands)


# ---------------------------------------------------------------------------
# Lazy check-rollup enrichment (read-only, duplicate draft groups only)
# ---------------------------------------------------------------------------


def test_enrich_fetches_rollup_only_for_drafts_in_duplicate_groups(
    monkeypatch: Any,
) -> None:
    fetched: list[int] = []

    def fake_rollup(repo: str, number: int) -> list[dict[str, str]]:
        fetched.append(number)
        return []

    prs = [
        # duplicate draft group → both fetched (rollup key removed below)
        _pr(910, "aragora/boss-harvest/issue-9100-boss-aaaa"),
        _pr(911, "aragora/boss-harvest/issue-9100-boss-bbbb"),
        # ready duplicate in same group → never fetched
        _pr(912, "aragora/boss-harvest/issue-9100-boss-cccc", draft=False),
        # singleton draft → never fetched
        _pr(913, "aragora/boss-harvest/issue-9200-boss-aaaa"),
        # non-boss branch → never fetched
        _pr(914, "feature/issue-9100-unrelated"),
    ]
    for pr in prs:
        pr.pop("statusCheckRollup", None)

    monkeypatch.setattr(janitor, "fetch_status_rollup", fake_rollup)
    janitor.enrich_duplicate_drafts("synaptent/aragora", prs)

    assert sorted(fetched) == [910, 911]


def test_dry_run_with_lazy_enrichment_invokes_only_read_commands(
    monkeypatch: Any, capsys: Any
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(list(command))
        stdout = "{}"
        if command[:3] == ["gh", "pr", "view"]:
            stdout = json.dumps({"statusCheckRollup": []})
        return subprocess.CompletedProcess(command, returncode=0, stdout=stdout, stderr="")

    fixture = _duplicate_fixture()
    for pr in fixture:
        pr.pop("statusCheckRollup", None)

    monkeypatch.setattr(janitor, "fetch_open_prs", lambda repo: fixture)
    monkeypatch.setattr(janitor.subprocess, "run", fake_run)

    assert janitor.main(["--repo", "synaptent/aragora"]) == 0
    assert all(c[:3] == ["gh", "pr", "view"] for c in commands), commands
    assert all("close" not in c for c in commands)


# ---------------------------------------------------------------------------
# (f) --max-closes cap (default 10) respected
# ---------------------------------------------------------------------------


def _many_duplicate_groups(groups: int) -> list[dict[str, Any]]:
    prs: list[dict[str, Any]] = []
    number = 1000
    for issue in range(1, groups + 1):
        for suffix in ("aaaa", "bbbb"):
            prs.append(
                _pr(
                    number,
                    f"aragora/boss-harvest/issue-{issue}-boss-{suffix}",
                    created=f"2026-06-0{1 + (number % 2)}T00:00:00Z",
                )
            )
            number += 1
    return prs


def test_max_closes_default_cap_is_ten() -> None:
    plan = janitor.build_plan(_many_duplicate_groups(15))
    assert len(plan) == 10


def test_max_closes_explicit_cap() -> None:
    plan = janitor.build_plan(_many_duplicate_groups(15), max_closes=3)
    assert len(plan) == 3


def test_max_closes_cap_respected_in_apply(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(janitor, "fetch_open_prs", lambda repo: _many_duplicate_groups(15))
    monkeypatch.setattr(janitor.subprocess, "run", fake_run)

    exit_code = janitor.main(["--repo", "synaptent/aragora", "--apply", "--max-closes", "2"])

    assert exit_code == 0
    close_commands = [c for c in commands if "close" in c]
    assert len(close_commands) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
