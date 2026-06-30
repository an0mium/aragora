from __future__ import annotations

from datetime import datetime, timezone

from aragora.swarm.queue_disposition import (
    DISPOSITION_CLOSE_OR_DELETE,
    DISPOSITION_HARVEST_NOW,
    DISPOSITION_HUMAN_PACKET,
    DISPOSITION_PARK_PRESERVE,
    build_manifest,
    classify_inventory_candidate,
    classify_pr_disposition,
)

NOW = datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc)


def _pr(
    number: int,
    title: str,
    *,
    draft: bool = False,
    mergeable: str = "MERGEABLE",
    additions: int = 100,
    deletions: int = 0,
    changed_files: int = 2,
    created: str = "2026-06-20T12:00:00Z",
    updated: str = "2026-06-20T12:00:00Z",
) -> dict[str, object]:
    return {
        "number": number,
        "title": title,
        "isDraft": draft,
        "mergeable": mergeable,
        "headRefName": f"codex/pr-{number}",
        "headRefOid": f"head-{number}",
        "createdAt": created,
        "updatedAt": updated,
        "additions": additions,
        "deletions": deletions,
        "changedFiles": changed_files,
        "labels": [],
    }


def test_product_pr_routes_to_harvest_now() -> None:
    item = classify_pr_disposition(
        _pr(8389, "feat(odr): in-package ODR verification engine"),
        merge_packet_entry={"tier": 1, "unresolved_dissent": False},
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_HARVEST_NOW
    assert item["operator_required"] is False
    assert item["open_pr"] == 8389


def test_tier_three_product_routes_to_human_packet() -> None:
    item = classify_pr_disposition(
        _pr(8541, "feat(api): expose crux finder"),
        merge_packet_entry={
            "tier": 3,
            "requires_human_risk_settlement": True,
            "unresolved_dissent": False,
        },
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_HUMAN_PACKET
    assert item["operator_required"] is True


def test_parked_epic_is_preserved_even_when_draft() -> None:
    item = classify_pr_disposition(
        _pr(8714, "[DIC-21] quarantine-eval CLI", draft=True),
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert "high_value_signal=true" in item["evidence"]


def test_model_dissent_on_value_work_never_closes_by_itself() -> None:
    item = classify_pr_disposition(
        _pr(8393, "feat(routing): decision-stakes router"),
        merge_packet_entry={"tier": 1, "unresolved_dissent": True},
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert "model_dissent_present_but_not_value_proof" in item["evidence"]
    assert "never close solely for dissent" in item["next_action"]


def test_string_false_does_not_create_dissent() -> None:
    item = classify_pr_disposition(
        _pr(8393, "feat(routing): decision-stakes router"),
        merge_packet_entry={"tier": "1", "unresolved_dissent": "false"},
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_HARVEST_NOW
    assert "model_dissent_present_but_not_value_proof" not in item["evidence"]


def test_unknown_mergeability_parks_high_value_pr() -> None:
    item = classify_pr_disposition(
        _pr(8393, "feat(routing): decision-stakes router", mergeable="UNKNOWN"),
        merge_packet_entry={"tier": 1, "unresolved_dissent": False},
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert "preserve" in item["next_action"]


def test_tier_three_draft_with_dissent_requires_operator() -> None:
    item = classify_pr_disposition(
        _pr(8541, "feat(api): expose crux finder", draft=True),
        merge_packet_entry={"tier": 3, "unresolved_dissent": "true"},
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert item["operator_required"] is True


def test_merge_packet_failure_parks_pr_until_tier_known() -> None:
    item = classify_pr_disposition(
        {
            **_pr(8541, "feat(api): expose crux finder"),
            "_merge_packet_error": "merge_packet_failed:#8541:transport down",
        },
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert item["operator_required"] is True
    assert "rerun merge-packet" in item["next_action"]


def test_invalid_pr_number_does_not_emit_zero_open_pr() -> None:
    item = classify_pr_disposition(
        {
            **_pr(0, "feat(odr): verify core"),
            "number": None,
            "headRefName": "codex/value",
        },
        now=NOW,
    )

    assert item["id"] == "codex/value"
    assert item["open_pr"] is None
    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert item["operator_required"] is True
    assert "invalid_pr_identity=true" in item["evidence"]


def test_invalid_pr_number_cannot_route_to_harvest_now() -> None:
    item = classify_pr_disposition(
        {
            **_pr(0, "feat(odr): verify core", mergeable="MERGEABLE"),
            "number": "not-a-number",
        },
        merge_packet_entry={"tier": 1, "unresolved_dissent": False},
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert item["operator_required"] is True
    assert "merge-packet safety cannot be proven" in item["next_action"]


def test_stale_maintenance_pr_can_close_only_after_manifest_checks() -> None:
    item = classify_pr_disposition(
        _pr(
            8156,
            "chore(ci): refresh cancelled lint snapshots",
            draft=True,
            mergeable="CONFLICTING",
            created="2026-06-01T12:00:00Z",
            updated="2026-06-01T12:00:00Z",
        ),
        now=NOW,
        stale_days=14,
    )

    assert item["disposition"] == DISPOSITION_CLOSE_OR_DELETE
    assert item["operator_required"] is True
    assert "owner/steering" in item["next_action"]


def test_recently_updated_old_pr_does_not_close_as_stale() -> None:
    item = classify_pr_disposition(
        _pr(
            8156,
            "chore(ci): refresh cancelled lint snapshots",
            draft=False,
            mergeable="MERGEABLE",
            created="2026-05-01T12:00:00Z",
            updated="2026-06-29T12:00:00Z",
        ),
        now=NOW,
        stale_days=14,
    )

    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert "stale_days>=14" not in item["evidence"]


def test_unknown_string_booleans_fail_closed_without_truthy_fallthrough() -> None:
    item = classify_pr_disposition(
        _pr(8393, "feat(routing): decision-stakes router"),
        merge_packet_entry={
            "tier": "1",
            "unresolved_dissent": "definitely-not",
            "requires_human_preapproval": "unknown",
        },
        now=NOW,
    )

    assert item["disposition"] == DISPOSITION_HUMAN_PACKET
    assert item["operator_required"] is True
    assert "model_dissent_present_but_not_value_proof" in item["evidence"]
    assert "merge_packet.requires_human_preapproval=unknown" in item["evidence"]


def test_unique_worktree_routes_to_preserve_operator_packet() -> None:
    item = classify_inventory_candidate(
        {
            "candidate_id": "abc",
            "classification": "unique_unharvested",
            "decision": "harvest_candidate",
            "git": {"branch": "codex/value", "head": "abc123"},
            "links": {"open_prs": []},
            "proof": ["branch has unique commits or diff ahead of base"],
        }
    )

    assert item["item_type"] == "worktree"
    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert item["operator_required"] is True
    assert item["branch"] == "codex/value"


def test_worktree_open_pr_dict_link_is_preserved() -> None:
    item = classify_inventory_candidate(
        {
            "candidate_id": "abc",
            "classification": "open_pr_or_outbox",
            "decision": "preserve",
            "git": {"branch": "codex/value", "head": "abc123"},
            "links": {"open_prs": [{"number": 8718, "title": "manifest"}]},
        }
    )

    assert item["disposition"] == DISPOSITION_PARK_PRESERVE
    assert item["open_pr"] == 8718


def test_patch_equivalent_worktree_requires_manifest_before_delete() -> None:
    item = classify_inventory_candidate(
        {
            "candidate_id": "def",
            "classification": "patch_equivalent_or_merged",
            "decision": "cleanup_candidate",
            "git": {"branch": "codex/old", "head": "def456"},
            "links": {"open_prs": []},
            "proof": ["patch-equivalent to base"],
        }
    )

    assert item["disposition"] == DISPOSITION_CLOSE_OR_DELETE
    assert item["operator_required"] is True
    assert "SHA/path manifest" in item["next_action"]


def test_build_manifest_counts_dispositions() -> None:
    manifest = build_manifest(
        prs=[
            _pr(1, "feat(odr): verify core"),
            _pr(2, "chore(ci): refresh cancelled lint snapshots", draft=True),
        ],
        merge_packet_entries={1: {"tier": 1, "unresolved_dissent": False}},
        inventory_candidates=[
            {
                "candidate_id": "wt",
                "classification": "active_or_dirty",
                "decision": "preserve",
                "git": {"branch": "codex/dirty", "head": "h"},
            }
        ],
        now=NOW,
    )

    assert manifest["schema_version"] == "aragora.queue_disposition_manifest.v1"
    assert manifest["summary"]["total_items"] == 3
    assert manifest["summary"]["by_disposition"][DISPOSITION_HARVEST_NOW] == 1
    assert manifest["summary"]["by_disposition"][DISPOSITION_CLOSE_OR_DELETE] == 1
    assert manifest["summary"]["by_disposition"][DISPOSITION_PARK_PRESERVE] == 1


def test_build_manifest_without_merge_packets_parks_prs() -> None:
    manifest = build_manifest(
        prs=[_pr(1, "feat(odr): verify core")],
        now=NOW,
    )

    assert manifest["summary"]["by_disposition"][DISPOSITION_HARVEST_NOW] == 0
    assert manifest["summary"]["by_disposition"][DISPOSITION_PARK_PRESERVE] == 1
    assert "merge_packet.error=merge_packet_not_collected" in manifest["items"][0]["evidence"]


def test_build_manifest_skips_inventory_duplicate_for_open_pr() -> None:
    manifest = build_manifest(
        prs=[_pr(8718, "feat(odr): verify core")],
        merge_packet_entries={8718: {"tier": 1, "unresolved_dissent": False}},
        inventory_candidates=[
            {
                "candidate_id": "wt",
                "classification": "open_pr_or_outbox",
                "decision": "preserve",
                "git": {"branch": "codex/pr-8718", "head": "head-8718"},
                "links": {"open_prs": [{"number": 8718}]},
            }
        ],
        now=NOW,
    )

    assert manifest["summary"]["total_items"] == 1
    assert manifest["annotations"] == ["inventory_duplicate_skipped:worktree:wt"]
