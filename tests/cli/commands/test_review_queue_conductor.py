"""Tests for the owner-aware review queue conductor."""

from __future__ import annotations

import argparse

from aragora.cli.commands.review_queue import add_review_queue_parser
from aragora.cli.commands.review_queue_conductor import (
    OWNER_TIMEOUT_CLASSIFICATION,
    READY_BOUNDARY_MARK_READY_CLASSIFICATION,
    TIER3_OR_TIER4_EVIDENCE_CLASSIFICATION,
    ConductorProviders,
    build_queue_conductor_packet,
)


def _view(
    number: int,
    *,
    head: str,
    title: str = "queue helper",
    branch: str | None = None,
    draft: bool = False,
    mergeable: str = "MERGEABLE",
    merge_state: str = "CLEAN",
    files: list[str] | None = None,
) -> dict[str, object]:
    return {
        "number": number,
        "title": title,
        "url": f"https://github.com/synaptent/aragora/pull/{number}",
        "headRefName": branch or f"branch-{number}",
        "headRefOid": head,
        "isDraft": draft,
        "state": "OPEN",
        "mergeable": mergeable,
        "mergeStateStatus": merge_state,
        "updatedAt": "2026-06-06T00:00:00Z",
        "files": [{"path": path} for path in (files or ["scripts/foo.py"])],
        "statusCheckRollup": [
            {"name": "required", "state": "SUCCESS", "bucket": "pass"},
        ],
    }


def _required_green(_pr_number: int, _repo: str | None) -> dict[str, object]:
    return {
        "available": True,
        "error": "",
        "checks": [{"name": "required", "state": "SUCCESS", "bucket": "pass"}],
    }


def _owner_unowned(_branch: str, _timeout: float) -> dict[str, object]:
    return {
        "lookup_state": "ok",
        "state": "unowned",
        "active_owner": False,
        "preserve_no_mutate": False,
    }


def _steering_empty(_branch: str, _timeout: float) -> dict[str, object]:
    return {"lookup_state": "ok", "message_count": 0, "has_pending": False}


def _packet(
    pr_number: int,
    *,
    head: str,
    tier: int = 2,
    not_ready: list[int] | None = None,
    verdict: str = "model_quorum_incomplete",
) -> dict[str, object]:
    return {
        "entries": [
            {
                "pr_number": pr_number,
                "head_sha": head,
                "model_review_quorum": {
                    "tier": tier,
                    "verdict": verdict,
                    "checks_summary": "1/1 green",
                    "counted_model_families": [],
                    "focused_dogfood_present": False,
                    "human_preapproval_recorded": False,
                },
            }
        ],
        "admin_squash_allowed": not bool(not_ready),
        "admin_squash_order": [] if not_ready else [pr_number],
        "not_ready": not_ready or [],
    }


def _flattened_packet(
    pr_number: int,
    *,
    head: str,
    not_ready: list[int] | None = None,
    counted_model_families: list[str] | None = None,
    focused_dogfood_present: bool = True,
    reasons: list[str] | None = None,
) -> dict[str, object]:
    families = ["claude", "openai"] if counted_model_families is None else counted_model_families
    return {
        "entries": [
            {
                "pr_number": pr_number,
                "head_sha": head,
                "tier": 4,
                "verdict": "not_ready_for_settlement",
                "checks_summary": "21/21 green",
                "counted_model_families": families,
                "focused_dogfood_present": focused_dogfood_present,
                "human_preapproval_recorded": False,
                "admin_squash_allowed": False,
                "reasons": reasons or [],
            }
        ],
        "admin_squash_allowed": False,
        "admin_squash_order": [],
        "not_ready": not_ready or [],
    }


def test_conductor_owner_lookup_timeout_preserves_no_mutate() -> None:
    view = _view(7843, head="head-a", draft=True)

    def gh_json(_args: list[str]) -> object:
        return view

    def owner_timeout(_branch: str, _timeout: float) -> dict[str, object]:
        return {"lookup_state": "timeout", "preserve_no_mutate": True}

    packet = build_queue_conductor_packet(
        pr_refs=["7843"],
        providers=ConductorProviders(
            gh_json=gh_json,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _packet(7843, head="head-a", not_ready=[7843]),
            owner_lookup=owner_timeout,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    candidate = packet["candidates"][0]
    assert candidate["classification"] == OWNER_TIMEOUT_CLASSIFICATION
    assert candidate["mutate_allowed"] is False
    assert candidate["owner"]["preserve_no_mutate"] is True


def test_conductor_detects_head_change_and_preserves() -> None:
    view = _view(7843, head="old-head", draft=True)

    packet = build_queue_conductor_packet(
        pr_refs=["7843"],
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _packet(7843, head="new-head", not_ready=[7843]),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    candidate = packet["candidates"][0]
    assert candidate["head_changed"] is True
    assert candidate["classification"] == "head_changed_preserve"
    assert candidate["mutate_allowed"] is False


def test_conductor_selects_unowned_tier2_evidence_candidate_prompt() -> None:
    view = _view(7843, head="exact-head", draft=True)

    packet = build_queue_conductor_packet(
        pr_refs=["7843"],
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _packet(7843, head="exact-head", not_ready=[7843]),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    candidate = packet["candidates"][0]
    assert candidate["classification"] == "unowned_evidence_candidate"
    assert candidate["mutate_allowed"] is True
    assert "PR #7843" in packet["next_prompt"]
    assert "exact-head" in packet["next_prompt"]


def test_conductor_treats_completed_skipped_rollup_as_non_actionable() -> None:
    view = _view(7885, head="exact-head", draft=True)
    view["statusCheckRollup"] = [
        {
            "name": "Core Suites",
            "workflowName": "Core Suites",
            "status": "COMPLETED",
            "conclusion": "SKIPPED",
        },
        {
            "name": "aragora-merge-quorum",
            "workflowName": "Aragora Merge Quorum",
            "status": "COMPLETED",
            "conclusion": "SUCCESS",
        },
    ]

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _flattened_packet(
                7885, head="exact-head", not_ready=[7885]
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    candidate = packet["candidates"][0]
    assert candidate["rollup"]["actionable_non_green"] is False
    assert candidate["rollup"]["pending"] == []
    assert candidate["classification"] == "ready_but_human_gated"


def test_conductor_tier4_missing_evidence_routes_to_evidence_collection() -> None:
    view = _view(7885, head="exact-head", draft=True)

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _flattened_packet(
                7885,
                head="exact-head",
                not_ready=[7885],
                counted_model_families=[],
                focused_dogfood_present=False,
                reasons=[
                    "draft PR",
                    "model quorum incomplete: 0/2 signal(s)",
                    "focused adversarial dogfood evidence is required",
                ],
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    candidate = packet["candidates"][0]
    assert candidate["classification"] == TIER3_OR_TIER4_EVIDENCE_CLASSIFICATION
    assert candidate["mutate_allowed"] is False
    assert "collect exact-head Tier 4 model/dogfood evidence" in candidate["next_action"]
    assert (
        "collect fresh exact-head Tier 4 structured model/dogfood evidence" in packet["next_prompt"]
    )
    assert "recording settlement" in packet["next_prompt"]


def test_conductor_reads_flattened_merge_packet_fields() -> None:
    view = _view(7885, head="exact-head", draft=True)
    view["statusCheckRollup"] = [
        {
            "name": "Core Suites",
            "workflowName": "Core Suites",
            "status": "COMPLETED",
            "conclusion": "SKIPPED",
        }
    ]

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _flattened_packet(
                7885, head="exact-head", not_ready=[7885]
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    merge_summary = packet["candidates"][0]["merge_packet"]
    assert merge_summary["tier"] == 4
    assert merge_summary["checks_summary"] == "21/21 green"
    assert merge_summary["counted_model_families"] == ["claude", "openai"]
    assert merge_summary["focused_dogfood_present"] is True


def test_ready_boundary_mode_emits_mark_ready_authorization_prompt() -> None:
    view = _view(7885, head="exact-head", draft=True)

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        mode="ready-boundary",
        repo_override="synaptent/aragora",
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _flattened_packet(
                7885,
                head="exact-head",
                not_ready=[7885],
                reasons=["workflow/deploy/destructive surface touched", "draft PR"],
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    candidate = packet["candidates"][0]
    ready_boundary = candidate["ready_boundary"]
    assert packet["mode"] == "ready-boundary"
    assert ready_boundary["classification"] == READY_BOUNDARY_MARK_READY_CLASSIFICATION
    assert ready_boundary["eligible_for_mark_ready_authorization"] is True
    assert ready_boundary["blockers"] == []
    assert ready_boundary["post_ready_blockers"] == ["Tier 4 human preapproval/settlement"]
    assert ready_boundary["actionable_rollup_rows"] == []
    assert ready_boundary["evidence_status"]["counted_model_families"] == ["claude", "openai"]
    assert ready_boundary["evidence_status"]["focused_dogfood_present"] is True
    assert ready_boundary["mark_ready_command"] == "gh pr ready 7885 --repo synaptent/aragora"
    assert "I explicitly authorize marking PR #7885 ready" in packet["next_prompt"]
    assert "exact head exact-head" in packet["next_prompt"]
    assert "Do not merge or record settlement" in packet["next_prompt"]


def test_ready_boundary_mode_blocks_missing_evidence() -> None:
    view = _view(7885, head="exact-head", draft=True)

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        mode="ready-boundary",
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _flattened_packet(
                7885,
                head="exact-head",
                not_ready=[7885],
                counted_model_families=["openai"],
                focused_dogfood_present=False,
                reasons=[
                    "draft PR",
                    "model quorum incomplete: 1/2 signal(s)",
                    "focused adversarial dogfood evidence is required",
                ],
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    ready_boundary = packet["candidates"][0]["ready_boundary"]
    assert ready_boundary["classification"] == "ready_boundary_blocked"
    assert ready_boundary["eligible_for_mark_ready_authorization"] is False
    assert "model quorum is incomplete" in ready_boundary["blockers"]
    assert "focused dogfood evidence is missing" in ready_boundary["blockers"]
    assert "re-run review-queue conductor in ready-boundary mode" in packet["next_prompt"]


def test_steering_no_lane_match_is_empty_for_boundary_checks() -> None:
    view = _view(7885, head="exact-head", draft=True)

    def no_lane(_branch: str, _timeout: float) -> dict[str, object]:
        return {
            "lookup_state": "no_lane_match",
            "message_count": 0,
            "has_pending": False,
            "error": "ERROR: no lane matched the requested selector",
        }

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        mode="ready-boundary",
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _flattened_packet(
                7885, head="exact-head", not_ready=[7885], reasons=["draft PR"]
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=no_lane,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    ready_boundary = packet["candidates"][0]["ready_boundary"]
    assert ready_boundary["eligible_for_mark_ready_authorization"] is True
    assert ready_boundary["steering_lookup_state"] == "no_lane_match"


def test_conductor_supersession_hint_blocks_conflict_repair() -> None:
    views = {
        "7821": _view(
            7821,
            head="old",
            title="queue helper conflict repair",
            mergeable="CONFLICTING",
            merge_state="DIRTY",
            files=["scripts/foo.py"],
        ),
        "7831": _view(
            7831,
            head="new",
            title="queue helper replacement",
            files=["scripts/foo.py"],
        ),
    }

    def gh_json(args: list[str]) -> object:
        return views[str(args[2])]

    packet = build_queue_conductor_packet(
        pr_refs=["7821", "7831"],
        providers=ConductorProviders(
            gh_json=gh_json,
            required_surface=_required_green,
            merge_packet=lambda pr_refs, **_kwargs: _packet(
                int(pr_refs[0]), head=views[pr_refs[0]]["headRefOid"]
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    older = packet["candidates"][0]
    assert older["classification"] == "superseded_or_stale"
    assert older["mutate_allowed"] is False
    assert older["supersession_hints"][0]["pr_number"] == 7831


def test_review_queue_parser_accepts_conductor_subcommand() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_review_queue_parser(subparsers)

    args = parser.parse_args(
        [
            "review-queue",
            "conductor",
            "--pr",
            "7843",
            "--mode",
            "ready-boundary",
            "--owner-timeout-seconds",
            "0.5",
            "--json",
        ]
    )

    assert args.review_queue_command == "conductor"
    assert args.pr == ["7843"]
    assert args.mode == "ready-boundary"
    assert args.owner_timeout_seconds == 0.5
