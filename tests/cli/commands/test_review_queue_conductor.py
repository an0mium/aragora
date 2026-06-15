"""Tests for the owner-aware review queue conductor."""

from __future__ import annotations

import argparse

from aragora.cli.commands.review_queue import _GhError, add_review_queue_parser
from aragora.cli.commands.review_queue_conductor import (
    OWNER_TIMEOUT_CLASSIFICATION,
    READY_BOUNDARY_MARK_READY_CLASSIFICATION,
    TIER3_OR_TIER4_EVIDENCE_CLASSIFICATION,
    ConductorProviders,
    build_queue_conductor_packet,
    render_queue_conductor_packet,
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


def test_conductor_treats_cancelled_advisory_rollup_by_workflow_as_non_actionable() -> None:
    view = _view(7885, head="exact-head", draft=True)
    view["statusCheckRollup"] = [
        {
            "name": "check",
            "workflowName": "Metrics Drift",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "check",
            "workflowName": "Module Tier Drift",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "portability",
            "workflowName": "Portability Lint",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "Shadow Scope",
            "workflowName": "Self-Hosted Shadow CI",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
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

    rollup = packet["candidates"][0]["rollup"]
    assert rollup["actionable_non_green"] is False
    assert rollup["actionable_rows"] == []
    assert [row["workflow"] for row in rollup["non_actionable_cancelled"]] == [
        "Metrics Drift",
        "Module Tier Drift",
        "Portability Lint",
        "Self-Hosted Shadow CI",
    ]


def test_conductor_ignores_superseded_cancelled_wrappers_when_successors_pass() -> None:
    view = _view(7885, head="exact-head", draft=True)
    view["statusCheckRollup"] = [
        {
            "name": "aragora-merge-quorum",
            "workflowName": "Aragora Merge Quorum",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "PR Admission Signal (Advisory)",
            "workflowName": "PR Admission Controller",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "lint-run",
            "workflowName": "Lint",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "typecheck-run",
            "workflowName": "Lint",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "lint",
            "workflowName": "Lint",
            "status": "COMPLETED",
            "conclusion": "SUCCESS",
        },
        {
            "name": "typecheck",
            "workflowName": "Lint",
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

    rollup = packet["candidates"][0]["rollup"]
    assert [row["name"] for row in rollup["actionable_rows"]] == ["aragora-merge-quorum"]
    assert [row["name"] for row in rollup["non_actionable_cancelled"]] == [
        "PR Admission Signal (Advisory)",
        "lint-run",
        "typecheck-run",
    ]


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


def test_ready_boundary_mode_emits_watched_rollup_summary() -> None:
    view = _view(7885, head="exact-head", draft=False, merge_state="BLOCKED")
    view["statusCheckRollup"] = [
        {
            "name": "test-fast (infra, tests/nomic tests/control_plane)",
            "workflowName": "Tests",
            "status": "IN_PROGRESS",
            "conclusion": "",
        },
        {
            "name": "Mac TypeScript SDK Shadow",
            "workflowName": "Self-Hosted Shadow CI",
            "status": "QUEUED",
            "conclusion": "",
        },
        {
            "name": "Generate Capability Gap Report",
            "workflowName": "Capability Surface Report",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "Docs Consistency",
            "workflowName": "Docs Consistency",
            "status": "COMPLETED",
            "conclusion": "CANCELLED",
        },
        {
            "name": "aragora-merge-quorum",
            "workflowName": "Aragora Merge Quorum",
            "status": "COMPLETED",
            "conclusion": "FAILURE",
        },
    ]

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        mode="ready-boundary",
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=lambda _pr_number, _repo: {
                "available": True,
                "error": "",
                "checks": [
                    {
                        "name": "aragora-merge-quorum",
                        "state": "FAILURE",
                        "bucket": "fail",
                    }
                ],
            },
            merge_packet=lambda **_kwargs: _flattened_packet(
                7885,
                head="exact-head",
                not_ready=[7885],
                counted_model_families=[],
                focused_dogfood_present=False,
                reasons=[
                    "checks are failing; repair before settlement",
                    "model quorum incomplete: 0/2 signal(s)",
                    "focused adversarial dogfood evidence is required",
                ],
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    watched = packet["candidates"][0]["ready_boundary"]["watched_rollup"]
    assert watched["actionable_non_green"] is True
    assert watched["summary"] == "Tests: 1 pending"
    assert watched["counts_by_workflow"] == {"Tests": {"pending": 1, "fail": 0, "cancel": 0}}
    assert [row["workflow"] for row in watched["actionable_rows"]] == ["Tests"]


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


def test_conductor_graphql_timeout_falls_back_to_rest_metadata_and_check_runs() -> None:
    def gh_json(args: list[str]) -> object:
        if args[:2] == ["pr", "view"]:
            raise _GhError("gh pr view 7885 failed: net/http: TLS handshake timeout")
        raise AssertionError(f"unexpected gh call: {args}")

    def rest_json(args: list[str]) -> object:
        if args == ["api", "repos/synaptent/aragora/pulls/7885"]:
            return {
                "number": 7885,
                "title": "queue conductor",
                "html_url": "https://github.com/synaptent/aragora/pull/7885",
                "head": {"ref": "codex/queue-conductor-command-20260606", "sha": "rest-head"},
                "base": {"ref": "main", "sha": "base-head"},
                "draft": True,
                "state": "open",
                "mergeable": True,
                "mergeable_state": "clean",
                "updated_at": "2026-06-09T00:00:00Z",
                "user": {"login": "codex"},
                "labels": [],
                "changed_files": 1,
            }
        if args == ["api", "repos/synaptent/aragora/pulls/7885/files?per_page=100"]:
            return [{"filename": "aragora/cli/commands/review_queue_conductor.py"}]
        if args == [
            "api",
            "repos/synaptent/aragora/commits/rest-head/check-runs?per_page=100",
        ]:
            return {
                "check_runs": [
                    {
                        "name": "lint",
                        "status": "completed",
                        "conclusion": "success",
                        "html_url": "https://example.test/checks/1",
                        "check_suite": {"app": {"name": "GitHub Actions"}},
                    }
                ]
            }
        raise AssertionError(f"unexpected REST call: {args}")

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        repo_override="synaptent/aragora",
        providers=ConductorProviders(
            gh_json=gh_json,
            rest_json=rest_json,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _flattened_packet(
                7885, head="rest-head", not_ready=[7885]
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    candidate = packet["candidates"][0]
    assert candidate["head_sha"] == "rest-head"
    assert candidate["files"] == ["aragora/cli/commands/review_queue_conductor.py"]
    assert candidate["rollup"]["total"] == 1
    assert candidate["rollup"]["actionable_non_green"] is False
    assert candidate["transport_fallback"]["source"] == "rest"
    assert candidate["transport_fallback"]["check_runs_available"] is True


def test_conductor_transport_blocked_summaries_include_rest_fallback_metadata() -> None:
    def gh_json(args: list[str]) -> object:
        if args[:2] == ["pr", "view"]:
            raise _GhError("gh pr view 8313 failed: GraphQL: API rate limit already exceeded")
        raise AssertionError(f"unexpected gh call: {args}")

    def rest_json(args: list[str]) -> object:
        if args == ["api", "repos/synaptent/aragora/pulls/8313"]:
            return {
                "number": 8313,
                "title": "proof matrix failure reporting",
                "html_url": "https://github.com/synaptent/aragora/pull/8313",
                "head": {"ref": "codex/proof-matrix-failures", "sha": "rest-head"},
                "base": {"ref": "main", "sha": "base-head"},
                "draft": True,
                "state": "open",
                "mergeable": True,
                "mergeable_state": "clean",
                "updated_at": "2026-06-12T00:00:00Z",
                "user": {"login": "codex"},
                "labels": [],
                "changed_files": 2,
            }
        if args == ["api", "repos/synaptent/aragora/pulls/8313/files?per_page=100"]:
            return [{"filename": "scripts/generate_capability_matrix.py"}]
        if args == [
            "api",
            "repos/synaptent/aragora/commits/rest-head/check-runs?per_page=100",
        ]:
            return {
                "check_runs": [
                    {
                        "name": "required",
                        "status": "completed",
                        "conclusion": "success",
                        "html_url": "https://example.test/checks/1",
                        "check_suite": {"app": {"name": "GitHub Actions"}},
                    }
                ]
            }
        raise AssertionError(f"unexpected REST call: {args}")

    def required_transport(_pr_number: int, _repo: str | None) -> dict[str, object]:
        return {
            "available": False,
            "checks": [],
            "error": "gh pr checks failed: GraphQL: API rate limit already exceeded",
            "error_kind": "github_transport",
            "transport_blocked": True,
            "preserve_no_mutate": True,
        }

    packet = build_queue_conductor_packet(
        pr_refs=["8313"],
        repo_override="synaptent/aragora",
        providers=ConductorProviders(
            gh_json=gh_json,
            rest_json=rest_json,
            required_surface=required_transport,
            merge_packet=lambda **_kwargs: (_ for _ in ()).throw(
                _GhError("gh pr view 8313 failed: GraphQL: API rate limit already exceeded")
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
            merge_tree_conflicts=lambda _base, _head: {
                "available": False,
                "conflict": False,
                "conflict_files": [],
            },
        ),
    )

    candidate = packet["candidates"][0]
    assert candidate["classification"] == "transport_blocked_preserve"
    assert candidate["mutate_allowed"] is False
    assert candidate["transport_fallback"]["source"] == "rest"
    assert candidate["required_checks"]["rest_fallback"]["pr"]["head_sha"] == "rest-head"
    assert candidate["merge_packet"]["rest_fallback"]["pr"]["head_sha"] == "rest-head"
    assert candidate["merge_packet"]["rest_fallback"]["mutation_forbidden"] is True


def test_ready_boundary_reports_merge_tree_conflict_files() -> None:
    view = _view(
        7821,
        head="conflict-head",
        draft=False,
        mergeable="CONFLICTING",
        merge_state="DIRTY",
        files=["aragora/cli/commands/review_queue.py"],
    )

    packet = build_queue_conductor_packet(
        pr_refs=["7821"],
        mode="ready-boundary",
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: _flattened_packet(
                7821, head="conflict-head", not_ready=[7821]
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "base-sha",
            merge_tree_conflicts=lambda base, head: {
                "available": True,
                "base_sha": base,
                "head_sha": head,
                "conflict": True,
                "conflict_files": [
                    "aragora/cli/commands/review_queue.py",
                    "docs/reference/CLI_REFERENCE.md",
                ],
            },
        ),
    )

    candidate = packet["candidates"][0]
    merge_tree = candidate["ready_boundary"]["merge_tree"]
    assert candidate["merge_tree"]["conflict"] is True
    assert merge_tree["base_sha"] == "base-sha"
    assert merge_tree["head_sha"] == "conflict-head"
    assert merge_tree["conflict_files"] == [
        "aragora/cli/commands/review_queue.py",
        "docs/reference/CLI_REFERENCE.md",
    ]
    assert "mergeable is CONFLICTING" in candidate["ready_boundary"]["blockers"]
    rendered = render_queue_conductor_packet(packet)
    assert "merge-tree-conflicts: aragora/cli/commands/review_queue.py" in rendered


def test_conductor_transport_failure_classifies_preserve_no_mutate() -> None:
    view = _view(7885, head="exact-head", draft=True)

    packet = build_queue_conductor_packet(
        pr_refs=["7885"],
        providers=ConductorProviders(
            gh_json=lambda _args: view,
            required_surface=_required_green,
            merge_packet=lambda **_kwargs: (_ for _ in ()).throw(
                _GhError("gh pr view 7885 failed: read: connection reset by peer")
            ),
            owner_lookup=_owner_unowned,
            steering_lookup=_steering_empty,
            origin_main_sha=lambda: "main-sha",
        ),
    )

    candidate = packet["candidates"][0]
    assert candidate["classification"] == "transport_blocked_preserve"
    assert candidate["mutate_allowed"] is False
    assert candidate["merge_packet"]["transport_blocked"] is True
    assert candidate["merge_packet"]["preserve_no_mutate"] is True
    assert candidate["merge_packet"]["error_kind"] == "github_transport"


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
