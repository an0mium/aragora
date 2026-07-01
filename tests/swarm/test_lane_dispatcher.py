"""Tests for ``aragora.swarm.lane_dispatcher`` -- claim-first lane assignment.

Pure-core: every case feeds explicit candidate and live-claim lists, so the
dispatch decision is exercised without GitHub, the lane registry, or any
spawned worker.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm import lane_dispatcher as ld


def _load_cli() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "lane_dispatcher.py"
    spec = importlib.util.spec_from_file_location("lane_dispatcher_cli_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _cand(number: int, branch: str = "", head: str = "") -> dict[str, Any]:
    return {"number": number, "branch": branch or f"codex/pr{number}", "head": head}


def _seq_session(pr: int) -> str:
    # Deterministic session id for assertion stability.
    return f"sess-{pr}"


# ---------------------------------------------------------------------------
# Assignment decision
# ---------------------------------------------------------------------------


def test_assigns_unclaimed_candidates_up_to_cap() -> None:
    plan = ld.select_assignments(
        candidates=[_cand(1), _cand(2), _cand(3)],
        live_claims_by_pr={},
        max_workers=2,
        session_id_for=_seq_session,
    )
    assert [a.pr for a in plan.assignments] == [1, 2]
    assert plan.deferred == [3]
    assert plan.owned == {}


def test_never_reassigns_a_live_owned_lane() -> None:
    plan = ld.select_assignments(
        candidates=[_cand(1), _cand(2)],
        live_claims_by_pr={1: "other-session"},
        max_workers=5,
        session_id_for=_seq_session,
    )
    assert [a.pr for a in plan.assignments] == [2]
    assert plan.owned == {1: "other-session"}
    assert plan.deferred == []


def test_assignment_carries_branch_and_session() -> None:
    plan = ld.select_assignments(
        candidates=[_cand(7, branch="codex/feature-x", head="abc123")],
        live_claims_by_pr={},
        max_workers=1,
        session_id_for=_seq_session,
    )
    (assignment,) = plan.assignments
    assert assignment.branch == "codex/feature-x"
    assert assignment.head == "abc123"
    assert assignment.owner_session == "sess-7"


def test_order_is_preserved_priority_first() -> None:
    plan = ld.select_assignments(
        candidates=[_cand(30), _cand(10), _cand(20)],
        live_claims_by_pr={},
        max_workers=3,
        session_id_for=_seq_session,
    )
    assert [a.pr for a in plan.assignments] == [30, 10, 20]


def test_zero_workers_assigns_nothing_but_records_owned() -> None:
    plan = ld.select_assignments(
        candidates=[_cand(1), _cand(2)],
        live_claims_by_pr={2: "owner-b"},
        max_workers=0,
        session_id_for=_seq_session,
    )
    assert plan.assignments == []
    assert plan.deferred == [1]
    assert plan.owned == {2: "owner-b"}


def test_non_integer_pr_numbers_are_ignored() -> None:
    plan = ld.select_assignments(
        candidates=[{"number": "oops", "branch": "x"}, _cand(5)],
        live_claims_by_pr={},
        max_workers=5,
        session_id_for=_seq_session,
    )
    assert [a.pr for a in plan.assignments] == [5]


def test_live_claims_with_empty_owner_do_not_block() -> None:
    # An owner string that is empty must not be treated as a live claim.
    plan = ld.select_assignments(
        candidates=[_cand(1)],
        live_claims_by_pr={1: ""},
        max_workers=1,
        session_id_for=_seq_session,
    )
    assert [a.pr for a in plan.assignments] == [1]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def test_worker_prompt_is_claim_first_and_scoped() -> None:
    prompt = ld.build_worker_prompt(
        pr=42, branch="codex/lane-42", session_id="sess-42", repo="synaptent/aragora"
    )
    assert "claim_active_agent_lane.py" in prompt
    assert "--lane-id lane-42-sess-42" in prompt
    assert "--release-stale" not in prompt
    assert "claim command exits non-zero" in prompt
    assert "yielding: claim blocked" in prompt
    assert "CLAIM-OR-YIELD" in prompt
    assert "#42" in prompt
    assert "codex/lane-42" in prompt
    assert "sess-42" in prompt
    # The anti-contention invariant must be explicit.
    assert "Do NOT scout the queue" in prompt
    # And it must forbid the dangerous merge-authority actions.
    assert "Never merge" in prompt
    assert "settle Tier-4" in prompt


def test_worker_prompt_is_short_and_constant() -> None:
    # The whole point is a compact, fixed prompt -- guard against bloat creeping
    # back in (the 200-line recursive prompts were the problem).
    prompt = ld.build_worker_prompt(pr=1, branch="b", session_id="s", repo="synaptent/aragora")
    assert len(prompt.splitlines()) < 30


def test_worker_prompt_source_reflects_target_agent() -> None:
    # Lane attribution must match the agent actually dispatched, not a hardcoded
    # "codex" -- otherwise a claude worker's claim is mislabeled as codex.
    prompt = ld.build_worker_prompt(
        pr=7, branch="b", session_id="s", repo="o/r", target_agent="claude"
    )
    assert "--source claude" in prompt
    assert "--source codex" not in prompt


def test_worker_prompt_rejects_unsafe_branch() -> None:
    # A hostile/brace-bearing ref must neither inject into the shell-out nor
    # break str.format -- it is replaced with a safe placeholder.
    hostile = "b`rm -rf /`/{evil}/$x"
    prompt = ld.build_worker_prompt(pr=9, branch=hostile, session_id="s", repo="o/r")
    assert hostile not in prompt
    assert "(branch for #9)" in prompt
    # A normal branch with slashes/dots is still passed through verbatim.
    ok = ld.build_worker_prompt(pr=9, branch="codex/lane.v2-9", session_id="s", repo="o/r")
    assert "codex/lane.v2-9" in ok


def test_worker_prompt_rejects_unsafe_session_id() -> None:
    # session_id feeds lane_id and the --lane-id/--owner-session shell-out;
    # session_id_for is a public seam, so a non-conforming value must fail fast.
    with pytest.raises(ValueError, match="unsafe session_id"):
        ld.build_worker_prompt(pr=1, branch="b", session_id="bad id; rm -rf /", repo="o/r")


def test_worker_prompt_rejects_unsafe_target_agent() -> None:
    # target_agent is interpolated unquoted into the claim --source shell-out.
    with pytest.raises(ValueError, match="unsafe target_agent"):
        ld.build_worker_prompt(
            pr=1, branch="b", session_id="s", repo="o/r", target_agent="x; rm -rf /"
        )


def test_worker_prompt_rejects_unsafe_repo() -> None:
    # repo is interpolated into the gh shell-out the worker is told to run.
    with pytest.raises(ValueError, match="unsafe repo"):
        ld.build_worker_prompt(pr=1, branch="b", session_id="s", repo="o/r; rm -rf /")
    # A normal owner/name repo is accepted.
    ok = ld.build_worker_prompt(pr=1, branch="b", session_id="s", repo="synaptent/aragora")
    assert "synaptent/aragora" in ok


def test_default_session_id_is_collision_resistant(monkeypatch: Any) -> None:
    monkeypatch.setattr(ld.time, "time", lambda: 1_765_760_000)
    first = ld.default_session_id(42)
    second = ld.default_session_id(42)
    assert first.startswith("codex-lane-pr42-1765760000-")
    assert second.startswith("codex-lane-pr42-1765760000-")
    assert first != second


# ---------------------------------------------------------------------------
# Live-claim parsing accepts both shapes
# ---------------------------------------------------------------------------


def test_live_claims_from_dict_shape() -> None:
    assert ld.live_claims_from_arg({"8405": "a", "8406": "b"}) == {8405: "a", 8406: "b"}


def test_live_claims_from_list_shape() -> None:
    rows = [
        {"pr_number": 8405, "owner_session": "a"},
        {"pr": 8406, "owner": "b"},
        {"number": 8407, "owner_session": ""},  # empty owner dropped
    ]
    assert ld.live_claims_from_arg(rows) == {8405: "a", 8406: "b"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_emits_json_plan(capsys: Any) -> None:
    candidates = json.dumps([_cand(1), _cand(2)])
    claims = json.dumps({"1": "owner-a"})
    code = cli.main(
        [
            "--candidates-json",
            candidates,
            "--live-claims-json",
            claims,
            "--max-workers",
            "5",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [a["pr"] for a in payload["assignments"]] == [2]
    assert payload["owned"] == {"1": "owner-a"}


def test_cli_print_prompt(capsys: Any) -> None:
    code = cli.main(
        ["--print-prompt", "--pr", "99", "--branch", "codex/x", "--session-id", "sess-99"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "PR #99" in out
    assert "sess-99" in out


def test_cli_print_prompt_requires_pr(capsys: Any) -> None:
    assert cli.main(["--print-prompt"]) == 1
