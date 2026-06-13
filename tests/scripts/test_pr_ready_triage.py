"""Tests for ``scripts/pr_ready_triage.py`` (bounded draft→ready promoter).

All boundaries (draft listing, merge-packet probe, ``gh pr ready``) are
injected; no test touches the network or spawns a subprocess. Style mirrors
``tests/scripts/test_boss_pr_janitor.py`` and
``tests/scripts/test_auto_evidence_cycle.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
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


triage = _load_module("pr_ready_triage.py")

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)


def _pr(
    number: int,
    *,
    head: str = "codex/feature-branch",
    title: str = "codex: automated change",
    body: str = "Routine automated change.",
    labels: list[str] | None = None,
    checks: list[dict[str, Any]] | None = None,
    mergeable: str = "MERGEABLE",
    created: str = "2026-06-12T10:00:00Z",  # 2h before NOW: old enough
) -> dict[str, Any]:
    return {
        "number": number,
        "headRefName": head,
        "title": title,
        "body": body,
        "labels": [{"name": name} for name in (labels or [])],
        "statusCheckRollup": checks if checks is not None else [],
        "mergeable": mergeable,
        "createdAt": created,
        "updatedAt": created,
    }


def _entry(
    pr: int,
    *,
    tier: Any = 1,
    human: bool = False,
    dissent: bool = False,
) -> dict[str, Any]:
    return {
        "pr_number": pr,
        "tier": tier,
        "requires_human_risk_settlement": human,
        "unresolved_dissent": dissent,
    }


class ReadyRecorder:
    """Injected ``gh pr ready`` runner that records calls."""

    def __init__(self, results: dict[int, tuple[bool, str]] | None = None) -> None:
        self.calls: list[int] = []
        self.results = results or {}

    def __call__(self, pr: int) -> tuple[bool, str]:
        self.calls.append(pr)
        return self.results.get(pr, (True, ""))


def _run(
    prs: list[dict[str, Any]],
    packets: dict[int, dict[str, Any]],
    *,
    apply: bool = False,
    max_scan: int = 15,
    max_promotions: int = 5,
    min_age_minutes: int = 30,
    branch_prefixes: tuple[str, ...] = ("codex/",),
    breaker_threshold: int = 3,
    mark_ready: ReadyRecorder | None = None,
) -> tuple[dict[str, Any], ReadyRecorder, list[int]]:
    mark_ready = mark_ready or ReadyRecorder()
    packet_calls: list[int] = []

    def fetch_packet(pr: int) -> dict[str, Any]:
        packet_calls.append(pr)
        return packets.get(pr, {})

    summary = triage.run_triage(
        list_prs=lambda: prs,
        fetch_packet=fetch_packet,
        mark_ready=mark_ready,
        apply=apply,
        branch_prefixes=branch_prefixes,
        min_age_minutes=min_age_minutes,
        max_scan=max_scan,
        max_promotions=max_promotions,
        breaker_threshold=breaker_threshold,
        now=NOW,
        log=lambda line: None,
    )
    return summary, mark_ready, packet_calls


def _planned(summary: dict[str, Any]) -> list[int]:
    return [item["pr"] for item in summary["plan"]]


# ---------------------------------------------------------------------------
# Draft-by-design governance PRs: hard never-promote
# ---------------------------------------------------------------------------


def test_tier4_marker_in_body_never_promoted() -> None:
    summary, ready, packet_calls = _run(
        [_pr(1, body="This is a Tier 4 settlement record."), _pr(2)],
        {1: _entry(1), 2: _entry(2)},
        apply=True,
    )
    assert _planned(summary) == [2]
    assert 1 not in packet_calls, "never-promote PRs must be pruned before any packet probe"
    assert ready.calls == [2]


def test_tier4_marker_with_whitespace_variants_blocked() -> None:
    for text in ("tier 4", "Tier4", "TIER   4", "tier\t4"):
        summary, _, _ = _run([_pr(1, body=f"context: {text} change")], {1: _entry(1)})
        assert summary["plan"] == [], f"marker {text!r} must block promotion"


def test_scarmani_mention_never_promoted() -> None:
    summary, ready, packet_calls = _run(
        [_pr(1, body="Escalated via the Scarmani track.")],
        {1: _entry(1)},
        apply=True,
    )
    assert summary["plan"] == []
    assert packet_calls == []
    assert ready.calls == []


def test_draft_by_design_in_title_never_promoted() -> None:
    summary, _, _ = _run(
        [_pr(1, title="governance: draft by design settlement")],
        {1: _entry(1)},
    )
    assert summary["plan"] == []


def test_do_not_mark_ready_marker_blocked() -> None:
    summary, _, _ = _run(
        [_pr(1, body="Reviewers: do NOT mark ready until ops signs off.")],
        {1: _entry(1)},
    )
    assert summary["plan"] == []


def test_operator_settlement_required_marker_blocked() -> None:
    summary, _, _ = _run(
        [_pr(1, body="Operator settlement required before merge.")],
        {1: _entry(1)},
    )
    assert summary["plan"] == []


@pytest.mark.parametrize("label", ["merge:manual", "do-not-promote", "held"])
def test_blocking_labels_each_individually_block(label: str) -> None:
    summary, ready, packet_calls = _run(
        [_pr(1, labels=[label])],
        {1: _entry(1)},
        apply=True,
    )
    assert summary["plan"] == []
    assert packet_calls == []
    assert ready.calls == []


def test_never_promote_survives_perfect_packet() -> None:
    # Even a flawless Tier-0 packet must not rescue a draft-by-design PR.
    summary, ready, _ = _run(
        [_pr(1, body="draft by design", labels=[])],
        {1: _entry(1, tier=0)},
        apply=True,
    )
    assert summary["plan"] == []
    assert ready.calls == []


# ---------------------------------------------------------------------------
# Unicode evasion of the never-promote markers (NFKC + zero-width stripping)
# ---------------------------------------------------------------------------


def test_fullwidth_tier4_marker_blocked() -> None:
    # NFKC folds fullwidth compatibility forms back to ASCII "Tier 4".
    summary, ready, packet_calls = _run(
        [_pr(1, body="Ｔｉｅｒ ４ settlement record.")],
        {1: _entry(1)},
        apply=True,
    )
    assert summary["plan"] == []
    assert packet_calls == []
    assert ready.calls == []


def test_zero_width_split_tier4_marker_blocked() -> None:
    summary, _, packet_calls = _run(
        [_pr(1, body="context: tier\N{ZERO WIDTH SPACE}4 change")],
        {1: _entry(1)},
    )
    assert summary["plan"] == []
    assert packet_calls == []


def test_zero_width_mixed_case_scarmani_blocked() -> None:
    for zw in ("\N{ZERO WIDTH NON-JOINER}", "\N{ZERO WIDTH JOINER}", "\N{WORD JOINER}"):
        summary, _, _ = _run(
            [_pr(1, body=f"Escalated via SCAR{zw}MANI track.")],
            {1: _entry(1)},
        )
        assert summary["plan"] == [], f"zero-width {zw!r} must not split the marker"


def test_separator_and_format_char_evasions_blocked() -> None:
    # Round-2 skeptical-review gaps: soft hyphen (Cf, not zero-width), OGHAM
    # SPACE MARK (the one Zs char NFKC does not fold), and multi-space runs
    # against the literal-space markers.
    bodies = {
        "soft-hyphen": "scar\u00admani settlement",
        "ogham-space": "draft\u1680by\u1680design",
        "nbsp": "draft\u00a0by\u00a0design",
        "multi-space": "draft  by   design",
    }
    for name, body in bodies.items():
        summary, _, packet_calls = _run([_pr(1, body=body)], {1: _entry(1)})
        assert summary["plan"] == [], f"{name} evasion must be blocked"
        assert packet_calls == []


def test_benign_text_passes_marker_normalization() -> None:
    summary, _, _ = _run(
        [_pr(1, title="codex: tidy cache tiers", body="Routine 4-step refactor of tiering.")],
        {1: _entry(1)},
    )
    assert _planned(summary) == [1]


# ---------------------------------------------------------------------------
# Stage-1 pruning: checks, mergeability, prefix, age
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["FAILURE", "ERROR", "CANCELLED", "TIMED_OUT"])
def test_failing_or_errored_checks_pruned(state: str) -> None:
    summary, _, packet_calls = _run(
        [_pr(1, checks=[{"name": "ci", "state": state, "conclusion": state}]), _pr(2)],
        {1: _entry(1), 2: _entry(2)},
    )
    assert _planned(summary) == [2]
    assert packet_calls == [2]


def test_pending_checks_do_not_prune() -> None:
    summary, _, _ = _run(
        [_pr(1, checks=[{"name": "ci", "status": "IN_PROGRESS"}])],
        {1: _entry(1)},
    )
    assert _planned(summary) == [1]


@pytest.mark.parametrize("mergeable", ["CONFLICTING", "UNKNOWN", ""])
def test_not_mergeable_pruned(mergeable: str) -> None:
    summary, _, _ = _run([_pr(1, mergeable=mergeable)], {1: _entry(1)})
    assert summary["plan"] == []


def test_wrong_branch_prefix_pruned() -> None:
    summary, _, _ = _run(
        [_pr(1, head="feature/manual-work"), _pr(2, head="codex/auto")],
        {1: _entry(1), 2: _entry(2)},
    )
    assert _planned(summary) == [2]


def test_repeatable_branch_prefixes() -> None:
    summary, _, _ = _run(
        [_pr(1, head="codex/auto"), _pr(2, head="elves/run-1"), _pr(3, head="feature/x")],
        {1: _entry(1), 2: _entry(2), 3: _entry(3)},
        branch_prefixes=("codex/", "elves/"),
    )
    assert _planned(summary) == [1, 2]


def test_young_pr_pruned_until_min_age() -> None:
    young = _pr(1, created="2026-06-12T11:45:00Z")  # 15m before NOW
    summary, _, packet_calls = _run([young], {1: _entry(1)}, min_age_minutes=30)
    assert summary["plan"] == []
    assert packet_calls == []


def test_unparseable_created_at_fails_closed_as_too_young() -> None:
    summary, _, _ = _run([_pr(1, created="not-a-date")], {1: _entry(1)})
    assert summary["plan"] == []


# ---------------------------------------------------------------------------
# Stage-2 merge-packet gate (fail closed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", [3, 4, None, "garbage"])
def test_tier3_plus_and_unknown_tier_rejected(tier: Any) -> None:
    summary, ready, _ = _run([_pr(1)], {1: _entry(1, tier=tier)}, apply=True)
    assert summary["plan"] == []
    assert [item["pr"] for item in summary["rejected"]] == [1]
    assert ready.calls == []


def test_missing_tier_key_rejected() -> None:
    summary, _, _ = _run([_pr(1)], {1: {"pr_number": 1}})
    assert summary["plan"] == []


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_tier_0_1_2_accepted(tier: int) -> None:
    summary, _, _ = _run([_pr(1)], {1: _entry(1, tier=tier)})
    assert _planned(summary) == [1]


def test_human_settlement_flag_rejected() -> None:
    summary, _, _ = _run([_pr(1)], {1: _entry(1, human=True)})
    assert summary["plan"] == []
    assert "requires_human_risk_settlement" in summary["rejected"][0]["reason"]


def test_unresolved_dissent_rejected() -> None:
    summary, _, _ = _run([_pr(1)], {1: _entry(1, dissent=True)})
    assert summary["plan"] == []
    assert "unresolved_dissent" in summary["rejected"][0]["reason"]


def test_empty_packet_rejected_fail_closed() -> None:
    summary, _, _ = _run([_pr(1)], {1: {}})
    assert summary["plan"] == []
    assert [item["pr"] for item in summary["rejected"]] == [1]


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


def test_max_promotions_caps_plan_and_probing() -> None:
    prs = [_pr(n) for n in (1, 2, 3, 4, 5, 6)]
    packets = {n: _entry(n) for n in (1, 2, 3, 4, 5, 6)}
    summary, _, packet_calls = _run(prs, packets, max_promotions=2)
    assert _planned(summary) == [1, 2]
    assert packet_calls == [1, 2], "probing must stop once the plan is full"


def test_max_scan_caps_packet_probes() -> None:
    prs = [_pr(n) for n in range(1, 11)]
    packets = {n: _entry(n, tier=3) for n in range(1, 11)}  # all rejected at stage 2
    summary, _, packet_calls = _run(prs, packets, max_scan=4)
    assert len(packet_calls) == 4
    assert summary["plan"] == []


def test_candidates_scanned_oldest_pr_first() -> None:
    summary, _, packet_calls = _run(
        [_pr(9), _pr(4), _pr(7)],
        {4: _entry(4), 7: _entry(7), 9: _entry(9)},
    )
    assert packet_calls == [4, 7, 9]


# ---------------------------------------------------------------------------
# Dry-run purity
# ---------------------------------------------------------------------------


def test_dry_run_performs_zero_mutations_and_exits_zero() -> None:
    summary, ready, _ = _run([_pr(1), _pr(2)], {1: _entry(1), 2: _entry(2)}, apply=False)
    assert ready.calls == []
    assert summary["promoted"] == []
    assert summary["exit_code"] == 0
    assert summary["mode"] == "dry-run"
    assert _planned(summary) == [1, 2]


def test_main_dry_run_invokes_no_subprocess_with_mocked_listing(
    monkeypatch: Any, capsys: Any
) -> None:
    def forbidden_run(command: list[str], **kwargs: Any) -> Any:
        raise AssertionError(f"subprocess must not run in dry-run with mocked I/O: {command}")

    monkeypatch.setattr(triage, "default_list_draft_prs", lambda repo: [_pr(1)])
    monkeypatch.setattr(triage, "default_fetch_packet", lambda repo, pr: _entry(pr))
    monkeypatch.setattr(triage.subprocess, "run", forbidden_run)

    exit_code = triage.main(["--repo", "synaptent/aragora"])

    assert exit_code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    promotes = [line for line in lines if line.get("action") == "promote"]
    assert [p["pr"] for p in promotes] == [1]
    assert all(p["dry_run"] is True for p in promotes)


# ---------------------------------------------------------------------------
# Apply path: promotion, fail-closed, breaker
# ---------------------------------------------------------------------------


def test_apply_promotes_planned_prs_and_exits_zero() -> None:
    summary, ready, _ = _run([_pr(1), _pr(2)], {1: _entry(1), 2: _entry(2)}, apply=True)
    assert ready.calls == [1, 2]
    assert summary["promoted"] == [1, 2]
    assert summary["exit_code"] == 0


def test_apply_respects_max_promotions_cap() -> None:
    prs = [_pr(n) for n in (1, 2, 3, 4)]
    summary, ready, _ = _run(
        prs, {n: _entry(n) for n in (1, 2, 3, 4)}, apply=True, max_promotions=2
    )
    assert ready.calls == [1, 2]
    assert summary["promoted"] == [1, 2]


def test_failed_promotion_fails_closed_exit_one() -> None:
    ready = ReadyRecorder(results={1: (False, "boom")})
    summary, _, _ = _run(
        [_pr(1), _pr(2)], {1: _entry(1), 2: _entry(2)}, apply=True, mark_ready=ready
    )
    assert summary["failed"] == [1]
    assert summary["promoted"] == [2]
    assert summary["exit_code"] == 1


def test_breaker_trips_on_three_identical_errors() -> None:
    err = (False, "gh: HTTP 401 bad credentials")
    ready = ReadyRecorder(results={1: err, 2: err, 3: err, 4: err})
    summary, recorder, _ = _run(
        [_pr(n) for n in (1, 2, 3, 4)],
        {n: _entry(n) for n in (1, 2, 3, 4)},
        apply=True,
        mark_ready=ready,
    )
    assert recorder.calls == [1, 2, 3], "PR 4 must remain untouched after the breaker"
    assert summary["breaker_tripped"] is True
    assert summary["exit_code"] == 2


def test_distinct_errors_do_not_trip_breaker() -> None:
    ready = ReadyRecorder(
        results={1: (False, "error A"), 2: (False, "error B"), 3: (False, "error A")}
    )
    summary, recorder, _ = _run(
        [_pr(n) for n in (1, 2, 3)],
        {n: _entry(n) for n in (1, 2, 3)},
        apply=True,
        mark_ready=ready,
    )
    assert len(recorder.calls) == 3
    assert summary["breaker_tripped"] is False
    assert summary["exit_code"] == 1


def test_success_resets_breaker_counter() -> None:
    err = (False, "same error")
    ready = ReadyRecorder(results={1: err, 2: err})  # 3 and 4 succeed by default
    summary, recorder, _ = _run(
        [_pr(n) for n in (1, 2, 3, 4)],
        {n: _entry(n) for n in (1, 2, 3, 4)},
        apply=True,
        mark_ready=ready,
    )
    assert recorder.calls == [1, 2, 3, 4]
    assert summary["breaker_tripped"] is False


# ---------------------------------------------------------------------------
# Summary / downstream hand-off
# ---------------------------------------------------------------------------


def test_summary_names_auto_evidence_cycle_as_downstream() -> None:
    lines: list[str] = []
    triage.run_triage(
        list_prs=lambda: [_pr(1)],
        fetch_packet=lambda pr: _entry(pr),
        mark_ready=ReadyRecorder(),
        apply=True,
        now=NOW,
        log=lines.append,
    )
    summary_lines = [json.loads(line) for line in lines if '"summary"' in line]
    assert summary_lines, "a summary line must always be printed"
    assert "auto_evidence_cycle" in summary_lines[-1]["downstream"]


def test_main_apply_runs_gh_pr_ready(monkeypatch: Any) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        commands.append(list(command))

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr(triage, "default_list_draft_prs", lambda repo: [_pr(7)])
    monkeypatch.setattr(triage, "default_fetch_packet", lambda repo, pr: _entry(pr))
    monkeypatch.setattr(triage.subprocess, "run", fake_run)

    assert triage.main(["--repo", "synaptent/aragora", "--apply"]) == 0
    ready_commands = [c for c in commands if c[:3] == ["gh", "pr", "ready"]]
    assert len(ready_commands) == 1
    assert "7" in ready_commands[0]


def test_main_listing_failure_fails_closed(monkeypatch: Any) -> None:
    def boom(repo: str) -> list[dict[str, Any]]:
        raise RuntimeError("gh pr list failed")

    monkeypatch.setattr(triage, "default_list_draft_prs", boom)
    assert triage.main(["--repo", "synaptent/aragora"]) == 1


@pytest.mark.parametrize("repo", ["not-a-repo", "owner/name/extra", "owner/", "owner/na me"])
def test_malformed_repo_rejected_at_parse_time(repo: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        triage.main(["--repo", repo])
    assert excinfo.value.code == 2, "argparse must reject before any gh call"


def test_well_formed_repo_accepted_by_validator() -> None:
    assert triage.repo_arg("synaptent/aragora") == "synaptent/aragora"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
