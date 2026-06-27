from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import scripts.handoff_state as mod
import scripts.classify_handoff_state as cli


HEAD = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OTHER_HEAD = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
RECONCILE_LOCAL_WORK_MARKER_KEYS = (
    "uncommitted_changes",
    "has_uncommitted_changes",
    "uncommitted",
    "unpushed_commits",
    "local_changes",
    "local_work",
    "dirty",
)


class FakeGitHub:
    disabled = False

    def __init__(
        self,
        *,
        open_prs: dict[str, list[dict[str, Any]]] | None = None,
        pr_by_number: dict[int, dict[str, Any]] | None = None,
        refs: dict[str, dict[str, Any]] | None = None,
        errors: dict[str, str] | None = None,
    ) -> None:
        self.open_prs = open_prs or {}
        self.pr_by_number = pr_by_number or {}
        self.refs = refs or {}
        self.errors = errors or {}
        self.pr_calls = 0
        self.pr_number_calls = 0
        self.ref_calls = 0

    def open_prs_for_branch(self, branch: str) -> tuple[list[dict[str, Any]] | None, str | None]:
        self.pr_calls += 1
        error = self.errors.get(f"pr:{branch}")
        if error:
            return None, error
        return self.open_prs.get(branch, []), None

    def open_pr_by_number(self, pr_number: int) -> tuple[dict[str, Any] | None, str | None]:
        self.pr_number_calls += 1
        error = self.errors.get(f"pr-number:{pr_number}")
        if error:
            return None, error
        return self.pr_by_number.get(pr_number), None

    def remote_ref(self, branch: str) -> tuple[dict[str, Any] | None, str | None]:
        self.ref_calls += 1
        error = self.errors.get(f"ref:{branch}")
        if error:
            return None, error
        return self.refs.get(branch), None


class FakeOwnerProbe:
    def __init__(self, payloads: dict[str, dict[str, Any]] | None = None) -> None:
        self.payloads = payloads or {}

    def probe(self, branch: str) -> mod.OwnerEvidence:
        payload = self.payloads.get(branch)
        if payload is None:
            return mod.OwnerEvidence(available=True, matched=False, error="no lane matched")
        return mod.owner_evidence_from_payload(payload)


def _write_outbox(
    state_root: Path,
    *,
    key: str = "open-pr-codex-example-aaaaaaaa",
    branch: str = "codex/example",
    head: str = HEAD,
    base: str | None = None,
    action_type: str = "open_or_update_pr",
    local_evidence: list[dict[str, Any]] | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> Path:
    outbox = state_root / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    path = outbox / f"{key}.json"
    requested_action = {
        "type": action_type,
        "branch": branch,
        "desired_head_sha": head,
        "head_sha": head,
    }
    payload = {
        "idempotency_key": key,
        "requested_action": requested_action,
        "branch": branch,
        "desired_head_sha": head,
        "head_sha": head,
        "repo": "synaptent/aragora",
        "task": f"Open PR for {branch}",
    }
    if base is not None:
        requested_action["base"] = base
        payload["base"] = base
    if local_evidence is not None:
        payload["local_evidence"] = local_evidence
    if extra_payload is not None:
        payload.update(extra_payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_receipt(state_root: Path, key: str, payload: dict[str, Any]) -> None:
    receipts = state_root / ".aragora" / "automation-receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    body = {"idempotency_key": key, **payload}
    (receipts / f"{key}.json").write_text(json.dumps(body), encoding="utf-8")


def _write_status_cache(
    state_root: Path,
    *,
    open_pr_cap_reached: bool = False,
    degraded: bool = False,
    available: bool | None = True,
    generated_at: str | None = None,
    open_pr_heads: list[str] | None = None,
) -> None:
    status = state_root / ".aragora" / "automation-github-status"
    status.mkdir(parents=True, exist_ok=True)
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (status / "latest.json").write_text(
        json.dumps(
            {
                "generated_at": timestamp,
                "github_queue": {
                    "available": available,
                    "degraded": degraded,
                    "degraded_reason": "heavy_open_pr_query_failed:HTTP 504" if degraded else None,
                    "open_pr_heads": open_pr_heads if open_pr_heads is not None else [],
                    "open_pr_heads_cached_at": timestamp,
                    "open_codex_pr_count": len(open_pr_heads) if open_pr_heads is not None else 144,
                    "pressure": {"open_pr_cap_reached": open_pr_cap_reached},
                },
                "limits": {"max_open_prs": 120},
            }
        ),
        encoding="utf-8",
    )


def _write_steering_message(
    state_root: Path,
    *,
    owner_session: str = "engineering-autopilot-Q1",
    branch: str = "codex/example",
    filename: str = "2026-06-24T00-00-00-000Z-fixture.json",
    priority: str = "blocking",
) -> Path:
    inbox = state_root / ".aragora" / "operator-steering" / owner_session
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / filename
    payload = {
        "schema_version": "aragora-operator-steering/1.0",
        "to_session": owner_session,
        "from": "operator",
        "sent_at_utc": "2026-06-24T00:00:00.000Z",
        "lane_id_hint": "Q1",
        "priority": priority,
        "subject": f"Block {branch}",
        "body": f"Please resolve {branch} before non-owner movement.",
        "message_sha256": "fixture-sha",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_steering_receipt(
    message_path: Path,
    *,
    outcome: str,
    read_at_utc: str = "2026-06-24T00:05:00.000Z",
) -> Path:
    receipt_dir = message_path.parent / "_read_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"{read_at_utc.replace(':', '-')}-fixture.json"
    receipt = {
        "schema_version": "aragora-operator-steering-read-receipt/1.0",
        "owner_session": message_path.parent.name,
        "read_by_session": "reader",
        "read_at_utc": read_at_utc,
        "message_filename": message_path.name,
        "message_sha256": "fixture-sha",
        "outcome": outcome,
        "outcome_note": f"fixture {outcome}",
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _classify_one(
    tmp_path: Path,
    *,
    branch: str = "codex/example",
    outbox_file: str = "open-pr-codex-example-aaaaaaaa.json",
    github: FakeGitHub | None = None,
    owner: FakeOwnerProbe | None = None,
) -> dict[str, Any]:
    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file=outbox_file,
        github_client=github or FakeGitHub(),
        owner_probe=owner or FakeOwnerProbe(),
    )
    assert payload["outbox_count"] == 1
    return payload["items"][0]


def test_exact_open_pr_representation_does_not_hide_owner_noise(tmp_path: Path) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                    "head": {"ref": "codex/example", "sha": HEAD},
                }
            ]
        }
    )
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "active",
                "owner_blocking_state": "live_owner",
            }
        }
    )

    item = _classify_one(tmp_path, github=github, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["github"]["exact_open_pr"]["number"] == 8570
    assert item["next_mutation_candidate"] == "owner_followup"
    assert item["safe_to_mutate"] is False


def test_exact_open_pr_representation_is_safe_without_owner_blockers(tmp_path: Path) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                    "head": {"ref": "codex/example", "sha": HEAD},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github, owner=FakeOwnerProbe())

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["safe_to_mutate"] is True


def test_registry_only_owner_probe_representation_is_not_safe_to_mutate(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                    "head": {"ref": "codex/example", "sha": HEAD},
                }
            ]
        }
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=github,
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["safe_to_mutate"] is False
    assert item["next_mutation_candidate"] == "none"
    assert "owner liveness helper was not used" in item["reason"]


def test_exact_draft_open_pr_representation_is_not_safe_to_mutate(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/example", "sha": HEAD},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github, owner=FakeOwnerProbe())

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["next_mutation_candidate"] == "none"
    assert item["safe_to_mutate"] is False


def test_open_pr_without_desired_head_fails_closed(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    handoff = _write_outbox(tmp_path, branch="codex/example")
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    for key in ("desired_head_sha", "target_head_sha", "head_sha", "head", "commit"):
        payload.pop(key, None)
        payload["requested_action"].pop(key, None)
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/example", "sha": OTHER_HEAD},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github, owner=FakeOwnerProbe())

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert "no desired head" in item["reason"]


def test_exact_open_pr_representation_requires_requested_base(tmp_path: Path) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(tmp_path, branch="codex/example", base="main")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "release"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["github"]["exact_open_pr"] is None


def test_missing_base_defaults_to_main_for_exact_open_pr_representation(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "release"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["github"]["exact_open_pr"] is None


def test_non_pr_exact_open_pr_representation_requires_default_base(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(
        tmp_path,
        key="preserve-branch-codex-example-aaaaaaaa",
        branch="codex/example",
        action_type="preserve_branch",
    )
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "release"},
                }
            ]
        }
    )

    item = _classify_one(
        tmp_path,
        outbox_file="preserve-branch-codex-example-aaaaaaaa.json",
        github=github,
    )

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["safe_to_mutate"] is False
    assert item["next_mutation_candidate"] == "none"
    assert item["evidence"]["github"]["exact_open_pr"] is None


def test_exact_open_pr_representation_honors_base_ref_name_alias(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    handoff = _write_outbox(tmp_path, branch="codex/example")
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload.pop("base", None)
    payload["base_ref_name"] = "release"
    payload["requested_action"].pop("base", None)
    payload["requested_action"]["baseRefName"] = "release"
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "main"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["github"]["exact_open_pr"] is None


def test_exact_open_pr_representation_honors_target_head_sha_alias(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    handoff = _write_outbox(tmp_path, branch="codex/example")
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload.pop("desired_head_sha", None)
    payload.pop("head_sha", None)
    payload["target_head_sha"] = HEAD
    payload["requested_action"].pop("desired_head_sha", None)
    payload["requested_action"].pop("head_sha", None)
    handoff.write_text(json.dumps(payload), encoding="utf-8")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "main"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["evidence"]["github"]["exact_open_pr"]["number"] == 8570


def test_exact_open_pr_representation_accepts_reconcile_sha_prefix(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(tmp_path, branch="codex/example", head=HEAD[:12])
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "main"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["evidence"]["github"]["exact_open_pr"]["number"] == 8570


def test_open_pr_sha_prefix_representation_is_not_safe_to_mutate(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example", head=HEAD[:12])
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "main"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github, owner=FakeOwnerProbe())

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["safe_to_mutate"] is False
    assert item["next_mutation_candidate"] == "none"
    assert "full desired-head SHA does not exactly match the PR head" in item["reason"]


def test_exact_open_pr_representation_blocks_when_owner_probe_unavailable(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                    "head": {"ref": "codex/example", "sha": HEAD},
                }
            ]
        }
    )

    class UnavailableOwnerProbe:
        def probe(self, branch: str) -> mod.OwnerEvidence:
            return mod.OwnerEvidence(available=False, error="owner probe failed")

    item = _classify_one(tmp_path, github=github, owner=UnavailableOwnerProbe())

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["safe_to_mutate"] is False


def test_issue_only_receipt_limbo_stays_publication_requested(tmp_path: Path) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, key=key, branch="codex/example")
    _write_receipt(
        tmp_path,
        key,
        {
            "status": "already_satisfied",
            "reason": "existing_issue",
            "existing_issue_url": "https://github.com/synaptent/aragora/issues/123",
            "existing_pr_url": None,
        },
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["receipt"]["issue_only_pr_receipt"] is True
    assert "issue-only receipt" in item["reason"]


def test_issue_only_receipt_limbo_is_cap_blocked_when_open_pr_cap_reached(
    tmp_path: Path,
) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, key=key, branch="codex/example")
    _write_receipt(
        tmp_path,
        key,
        {
            "status": "already_satisfied",
            "reason": "existing_issue",
            "existing_issue_url": "https://github.com/synaptent/aragora/issues/123",
        },
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["evidence"]["receipt"]["issue_only_pr_receipt"] is True
    assert item["next_mutation_candidate"] == "queue_drain"


def test_conflicting_local_evidence_records_fail_closed(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        local_evidence=[
            {"branch": "codex/example", "desired_head_sha": HEAD},
            {"branch": "codex/example", "desired_head_sha": OTHER_HEAD},
        ],
    )
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "main"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert "local_evidence head conflicts" in item["reason"]


def test_conflicting_local_evidence_alias_records_fail_closed(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        local_evidence=[
            {"branch": "codex/example", "target_head_sha": HEAD, "base_ref_name": "main"},
            {"branch": "codex/example", "headRefOid": OTHER_HEAD, "baseRefName": "release"},
        ],
    )
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "main"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert "local_evidence head conflicts" in item["reason"]


def test_local_evidence_equivalent_sha_prefix_records_do_not_conflict(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        head=HEAD,
        local_evidence=[
            {"branch": "codex/example", "desired_head_sha": HEAD},
            {"branch": "codex/example", "target_head_sha": HEAD[:12]},
        ],
    )

    item = _classify_one(tmp_path, github=FakeGitHub())

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert "local_evidence" not in item["evidence"]


def test_local_evidence_prefix_does_not_hide_multiple_full_heads(
    tmp_path: Path,
) -> None:
    first = "abcdef1111111111111111111111111111111111"
    second = "abcdef1222222222222222222222222222222222"
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        head="abcdef1",
        local_evidence=[
            {"branch": "codex/example", "desired_head_sha": "abcdef1"},
            {"branch": "codex/example", "target_head_sha": first},
            {"branch": "codex/example", "headRefOid": second},
        ],
    )

    item = _classify_one(tmp_path, github=FakeGitHub())

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert "multiple local_evidence records disagree" in item["reason"]


def test_single_local_evidence_conflict_with_top_level_fails_closed(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        head=OTHER_HEAD,
        local_evidence=[
            {"branch": "codex/example", "target_head_sha": HEAD, "baseRefName": "main"},
        ],
    )
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "main"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert "local_evidence head conflicts" in item["reason"]


def test_local_evidence_origin_main_base_alias_does_not_conflict(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        base="main",
        local_evidence=[
            {"branch": "codex/example", "target_head_sha": HEAD, "base": "origin/main"},
        ],
    )
    github = FakeGitHub(
        refs={
            "codex/example": {
                "ref": "refs/heads/codex/example",
                "object": {"sha": HEAD},
            }
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert "local_evidence" not in item["evidence"]


def test_requested_action_branch_conflict_without_local_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    handoff = _write_outbox(tmp_path, branch="codex/example")
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["requested_action"]["branch"] = "codex/other"
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    item = _classify_one(tmp_path, github=FakeGitHub())

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["reason"] == "top-level branch conflicts with requested_action branch"


def test_requested_action_head_conflict_without_local_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    handoff = _write_outbox(tmp_path, branch="codex/example")
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["requested_action"]["desired_head_sha"] = OTHER_HEAD
    payload["requested_action"]["head_sha"] = OTHER_HEAD
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    item = _classify_one(tmp_path, github=FakeGitHub())

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["reason"] == "top-level desired head conflicts with requested_action desired head"


def test_live_open_pr_base_ref_is_not_collapsed_as_local_alias(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example", base="main")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "codex/example", "sha": HEAD},
                    "base": {"ref": "origin/main"},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["github"]["exact_open_pr"] is None


def test_terminal_receipts_choose_newest_by_timestamp_not_filename(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "automation-receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    key = "open-pr-codex-example-aaaaaaaa"
    (receipts / "zz-old.json").write_text(
        json.dumps(
            {
                "idempotency_key": key,
                "status": "published",
                "created_pr_url": "https://github.com/synaptent/aragora/pull/1",
                "generated_at": "2026-06-24T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (receipts / "aa-new.json").write_text(
        json.dumps(
            {
                "idempotency_key": key,
                "status": "published",
                "created_pr_url": "https://github.com/synaptent/aragora/pull/2",
                "generated_at": "2026-06-24T01:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    loaded = mod.load_terminal_receipts(receipts)

    assert loaded[key]["created_pr_url"].endswith("/2")


def test_terminal_pr_receipt_without_live_proof_does_not_publish(tmp_path: Path) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, key=key, branch="codex/example")
    _write_receipt(
        tmp_path,
        key,
        {
            "status": "published",
            "reason": "created_pr",
            "created_pr_url": "https://github.com/synaptent/aragora/pull/8570",
        },
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["next_mutation_candidate"] == "none"
    assert "terminal receipt exists" in item["reason"]


def test_terminal_pr_receipt_with_owner_noise_stays_unknown_without_live_proof(
    tmp_path: Path,
) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, key=key, branch="codex/example")
    _write_receipt(
        tmp_path,
        key,
        {
            "status": "published",
            "reason": "created_pr",
            "target_pr": 8570,
        },
    )
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "active",
                "owner_blocking_state": "live_owner",
            }
        }
    )

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["next_mutation_candidate"] == "none"
    assert "terminal receipt exists" in item["reason"]


def test_target_pr_reference_can_prove_exact_open_pr_representation(
    tmp_path: Path,
) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(
        tmp_path,
        key=key,
        branch="codex/original-branch",
        extra_payload={"target_open_pr": 8570},
    )
    github = FakeGitHub(
        pr_by_number={
            8570: {
                "number": 8570,
                "state": "open",
                "draft": False,
                "head": {"ref": "codex/original-branch", "sha": HEAD},
                "base": {"ref": "main"},
                "html_url": "https://github.com/synaptent/aragora/pull/8570",
            }
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["safe_to_mutate"] is True
    assert item["evidence"]["github"]["exact_open_pr"]["number"] == 8570
    assert item["evidence"]["github"]["exact_open_pr"]["head"] == "codex/original-branch"
    assert github.pr_number_calls == 1


def test_target_pr_reference_accepts_matching_branch_pr_lookup(
    tmp_path: Path,
) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(
        tmp_path,
        key=key,
        branch="codex/original-branch",
        extra_payload={"target_open_pr": 8570},
    )
    github = FakeGitHub(
        open_prs={
            "codex/original-branch": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "codex/original-branch", "sha": HEAD},
                    "base": {"ref": "main"},
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["safe_to_mutate"] is True
    assert item["evidence"]["github"]["exact_open_pr"]["number"] == 8570
    assert item["evidence"]["github"]["exact_open_pr"]["head"] == "codex/original-branch"
    assert github.pr_number_calls == 0


def test_target_pr_representation_is_not_safe_when_branch_pr_lookup_degrades(
    tmp_path: Path,
) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(
        tmp_path,
        key=key,
        branch="codex/original-branch",
        extra_payload={"target_open_pr": 8570},
    )
    github = FakeGitHub(
        pr_by_number={
            8570: {
                "number": 8570,
                "state": "open",
                "draft": False,
                "head": {"ref": "codex/original-branch", "sha": HEAD},
                "base": {"ref": "main"},
                "html_url": "https://github.com/synaptent/aragora/pull/8570",
            }
        },
        errors={"pr:codex/original-branch": "gh api failed (HTTP 504)"},
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["safe_to_mutate"] is False
    assert item["next_mutation_candidate"] == "none"
    assert item["evidence"]["github"]["mode"] == "degraded"
    assert item["evidence"]["github"]["exact_open_pr"]["number"] == 8570
    assert "GitHub PR evidence is degraded" in item["reason"]
    assert github.pr_number_calls == 1


def test_target_pr_branch_lookup_must_match_handoff_branch(
    tmp_path: Path,
) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(
        tmp_path,
        key=key,
        branch="codex/original-branch",
        extra_payload={"target_open_pr": 8570},
    )
    github = FakeGitHub(
        open_prs={
            "codex/original-branch": [
                {
                    "number": 8570,
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "codex/represented-elsewhere", "sha": HEAD},
                    "base": {"ref": "main"},
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["safe_to_mutate"] is False
    assert item["next_mutation_candidate"] == "queue_drain"
    assert item["evidence"]["github"]["exact_open_pr"] is None
    assert github.pr_number_calls == 1


def test_target_pr_reference_must_match_handoff_branch(
    tmp_path: Path,
) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(
        tmp_path,
        key=key,
        branch="codex/original-branch",
        extra_payload={"target_open_pr": 8570},
    )
    github = FakeGitHub(
        pr_by_number={
            8570: {
                "number": 8570,
                "state": "open",
                "draft": False,
                "head": {"ref": "codex/represented-elsewhere", "sha": HEAD},
                "base": {"ref": "main"},
                "html_url": "https://github.com/synaptent/aragora/pull/8570",
            }
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["safe_to_mutate"] is False
    assert item["next_mutation_candidate"] == "queue_drain"
    assert item["evidence"]["github"]["exact_open_pr"] is None
    assert github.pr_number_calls == 1


def test_terminal_completed_receipt_without_pr_does_not_suppress_publication(
    tmp_path: Path,
) -> None:
    key = "open-pr-codex-example-aaaaaaaa"
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, key=key, branch="codex/example")
    _write_receipt(
        tmp_path,
        key,
        {
            "status": "completed",
            "reason": "completed_without_pr",
        },
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["next_mutation_candidate"] == "publish_or_represent_pr"


def test_unique_branch_without_pr_is_cap_blocked_when_cache_says_cap_reached(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["evidence"]["queue_cap"]["open_pr_cap_reached"] is True
    assert item["evidence"]["queue_cap"]["raw_open_pr_cap_reached"] is True
    assert item["evidence"]["queue_cap"]["cache_stale"] is False


def test_unique_branch_without_pr_is_publication_requested_when_cap_clear(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value


def test_stale_queue_cap_cache_preserves_raw_cap_block(tmp_path: Path) -> None:
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    _write_status_cache(tmp_path, open_pr_cap_reached=True, generated_at=stale)
    _write_outbox(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["evidence"]["queue_cap"]["raw_open_pr_cap_reached"] is True
    assert item["evidence"]["queue_cap"]["open_pr_cap_reached"] is True
    assert item["evidence"]["queue_cap"]["cache_stale"] is True
    assert item["evidence"]["queue_cap"]["decision_source"] == "expired_cache"


def test_stale_queue_cap_cache_with_raw_clear_fails_closed(tmp_path: Path) -> None:
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    _write_status_cache(tmp_path, open_pr_cap_reached=False, generated_at=stale)
    _write_outbox(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["reason"] == "publication requested but queue-cap evidence is stale or unavailable"
    assert item["evidence"]["queue_cap"]["raw_open_pr_cap_reached"] is False
    assert item["evidence"]["queue_cap"]["open_pr_cap_reached"] is None
    assert item["evidence"]["queue_cap"]["cache_stale"] is True


def test_unavailable_queue_cap_cache_preserves_raw_cap_block(tmp_path: Path) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True, available=False)
    _write_outbox(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["evidence"]["queue_cap"]["github_queue_available"] is False
    assert item["evidence"]["queue_cap"]["raw_open_pr_cap_reached"] is True
    assert item["evidence"]["queue_cap"]["open_pr_cap_reached"] is True
    assert item["evidence"]["queue_cap"]["decision_source"] == "github_queue_unavailable"


def test_unavailable_queue_cap_cache_with_raw_clear_fails_closed(tmp_path: Path) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False, available=False)
    _write_outbox(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["reason"] == "publication requested but queue-cap evidence is stale or unavailable"
    assert item["evidence"]["queue_cap"]["github_queue_available"] is False
    assert item["evidence"]["queue_cap"]["raw_open_pr_cap_reached"] is False
    assert item["evidence"]["queue_cap"]["open_pr_cap_reached"] is None


def test_fresh_degraded_queue_cap_cache_blocks_with_rest_fallback_evidence(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True, degraded=True)
    _write_outbox(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["evidence"]["queue_cap"]["degraded"] is True
    assert item["evidence"]["queue_cap"]["open_pr_cap_reached"] is True
    assert item["evidence"]["queue_cap"]["decision_source"] == "fresh_degraded_cache_rest_fallback"


def test_stale_owner_remote_exact_head_blocks_like_open_pr_path(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}}
    )
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "released",
                "owner_state": "stale",
                "owner_blocking_state": "stale_owner",
                "stale_claim_advisory": {"available": True},
            }
        }
    )

    item = _classify_one(tmp_path, github=github, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["github"]["remote_ref"]["sha"] == HEAD


def test_remote_exact_head_respects_open_pr_cap(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}}
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["evidence"]["github"]["remote_ref"]["sha"] == HEAD
    assert item["next_mutation_candidate"] == "queue_drain"


def test_remote_exact_head_representation_survives_pr_lookup_degradation(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}},
        errors={"pr:codex/example": "gh api failed (TimeoutExpired)"},
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["evidence"]["github"]["mode"] == "degraded"
    assert item["evidence"]["github"]["remote_ref"]["sha"] == HEAD


def test_non_pr_remote_exact_head_does_not_hide_pr_lookup_degradation(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(
        tmp_path,
        key="preserve-branch-codex-example-aaaaaaaa",
        branch="codex/example",
        action_type="preserve_branch",
    )
    github = FakeGitHub(
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}},
        errors={"pr:codex/example": "gh api failed (TimeoutExpired)"},
    )

    item = _classify_one(
        tmp_path,
        outbox_file="preserve-branch-codex-example-aaaaaaaa.json",
        github=github,
    )

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["evidence"]["github"]["mode"] == "degraded"
    assert item["safe_to_mutate"] is False


def test_local_evidence_identical_sha_prefix_does_not_conflict(tmp_path: Path) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        head=HEAD[:12],
        local_evidence=[{"branch": "codex/example", "head_sha": HEAD[:12]}],
    )

    item = _classify_one(tmp_path, github=FakeGitHub())

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert "local_evidence" not in item["evidence"]


def test_remote_exact_head_with_mismatched_open_pr_stays_publication_requested(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 9000,
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "codex/example", "sha": OTHER_HEAD},
                }
            ]
        },
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}},
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["github"]["exact_open_pr"] is None
    assert item["evidence"]["github"]["remote_ref"]["sha"] == HEAD


def test_remote_sha_prefix_representation_is_not_safe_to_mutate(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(
        tmp_path,
        key="preserve-branch-codex-example-aaaaaaaa",
        branch="codex/example",
        head=HEAD[:12],
        action_type="preserve_branch",
    )
    github = FakeGitHub(
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}}
    )

    item = _classify_one(
        tmp_path,
        outbox_file="preserve-branch-codex-example-aaaaaaaa.json",
        github=github,
        owner=FakeOwnerProbe(),
    )

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_REMOTE_BRANCH.value
    assert item["safe_to_mutate"] is False
    assert item["next_mutation_candidate"] == "none"
    assert "full desired-head SHA does not exactly match the remote branch head" in item["reason"]


def test_non_pr_remote_exact_head_with_mismatched_open_pr_is_not_mutable(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        key="preserve-branch-codex-example-aaaaaaaa",
        branch="codex/example",
        action_type="preserve_branch",
    )
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 9000,
                    "state": "open",
                    "draft": False,
                    "head": {"ref": "codex/example", "sha": OTHER_HEAD},
                    "base": {"ref": "main"},
                }
            ]
        },
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}},
    )

    item = _classify_one(
        tmp_path,
        outbox_file="preserve-branch-codex-example-aaaaaaaa.json",
        github=github,
    )

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert item["safe_to_mutate"] is False
    assert item["next_mutation_candidate"] == "none"
    assert item["evidence"]["github"]["exact_open_pr"] is None
    assert item["evidence"]["github"]["remote_ref"]["sha"] == HEAD
    assert "existing open PR head does not match desired head" in item["reason"]


def test_open_pr_evidence_is_compact_when_pr_is_not_exact(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 9000,
                    "state": "open",
                    "draft": True,
                    "html_url": "https://github.com/synaptent/aragora/pull/9000",
                    "head": {"ref": "codex/example", "sha": OTHER_HEAD},
                    "base": {"ref": "main"},
                    "_links": {"self": {"href": "https://api.github.com/noisy"}},
                    "repo": {"full_name": "synaptent/aragora"},
                    "user": {"login": "scarmani"},
                }
            ]
        },
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}},
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["github"]["open_prs"] == [
        {
            "number": 9000,
            "state": "open",
            "draft": True,
            "head": "codex/example",
            "head_sha": OTHER_HEAD,
            "base": "main",
            "html_url": "https://github.com/synaptent/aragora/pull/9000",
        }
    ]


def test_remote_exact_head_does_not_hide_live_owner_gate(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        refs={"codex/example": {"ref": "refs/heads/codex/example", "object": {"sha": HEAD}}}
    )
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "active",
                "owner_blocking_state": "live_owner",
            }
        }
    )

    item = _classify_one(tmp_path, github=github, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert "remote branch" in item["reason"]
    assert item["next_mutation_candidate"] == "owner_followup"


def test_possible_unpushed_work_blocks_non_owner_movement(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "blocked",
                "owner_state": "stale",
                "owner_blocking_state": "stale_owner",
                "advisory_withheld": "possible_unpushed_work",
            }
        }
    )

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_POSSIBLE_UNPUSHED_WORK.value
    assert item["evidence"]["owner"]["advisory_withheld"] == "possible_unpushed_work"


def test_lane_registry_possible_unpushed_marker_blocks_default_classifier(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "released",
                    "advisory_withheld": "possible_unpushed_work",
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.BLOCKED_BY_POSSIBLE_UNPUSHED_WORK.value
    assert item["evidence"]["owner"]["advisory_withheld"] == "possible_unpushed_work"


def test_local_work_marker_keys_match_reconcile_fail_closed_keys() -> None:
    assert mod.LOCAL_WORK_MARKER_KEYS == RECONCILE_LOCAL_WORK_MARKER_KEYS


def test_top_level_local_work_marker_blocks_publication(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        extra_payload={"unpushed_commits": "1"},
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_POSSIBLE_UNPUSHED_WORK.value
    assert item["evidence"]["owner"]["advisory_withheld"] == "possible_unpushed_work"


@pytest.mark.parametrize("marker_value", ["false", "0", "none", "clean", "verified-clean"])
def test_top_level_negative_local_work_marker_does_not_block_publication(
    tmp_path: Path,
    marker_value: str,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        extra_payload={"local_work": marker_value},
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["owner"]["advisory_withheld"] is None


def test_local_evidence_local_work_marker_blocks_publication(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        local_evidence=[
            {
                "branch": "codex/example",
                "desired_head_sha": HEAD,
                "local_work": "yes",
            }
        ],
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_POSSIBLE_UNPUSHED_WORK.value
    assert item["evidence"]["owner"]["advisory_withheld"] == "possible_unpushed_work"


@pytest.mark.parametrize("marker_value", ["false", "0", "none", "clean", "verified-clean"])
def test_local_evidence_negative_local_work_marker_does_not_block_publication(
    tmp_path: Path,
    marker_value: str,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(
        tmp_path,
        branch="codex/example",
        local_evidence=[
            {
                "branch": "codex/example",
                "desired_head_sha": HEAD,
                "local_work": marker_value,
            }
        ],
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["owner"]["advisory_withheld"] is None


def test_lane_registry_terminal_owner_does_not_block_by_default(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "released",
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["owner"]["owner_blocking_state"] is None


def test_lane_registry_active_without_fresh_liveness_blocks_default_classifier(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "active",
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["owner"]["owner_blocking_state"] == "unknown_owner"


def test_lane_registry_active_with_fresh_timestamp_blocks_default_classifier(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "active",
                    "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["owner"]["owner_blocking_state"] == "unknown_owner"


def test_lane_registry_blocked_status_blocks_default_classifier(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "blocked",
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["owner"]["owner_blocking_state"] == "unknown_owner"


def test_lane_registry_active_status_synonyms_block_default_classifier(
    tmp_path: Path,
) -> None:
    for status in (
        "running",
        "pending",
        "queued",
        "acknowledged",
        "working",
    ):
        state_root = tmp_path / status
        _write_status_cache(state_root)
        _write_outbox(state_root, branch="codex/example")
        lanes_path = state_root / ".aragora" / "agent-bridge" / "lanes.json"
        lanes_path.parent.mkdir(parents=True, exist_ok=True)
        lanes_path.write_text(
            json.dumps(
                [
                    {
                        "lane_id": "Q1",
                        "owner_session": "engineering-autopilot-Q1",
                        "branch": "codex/example",
                        "status": status,
                    }
                ]
            ),
            encoding="utf-8",
        )

        payload = mod.classify_handoffs(
            repo_root=state_root,
            state_root=state_root,
            github_repo="synaptent/aragora",
            outbox_file="open-pr-codex-example-aaaaaaaa.json",
            github_client=FakeGitHub(),
        )
        item = payload["items"][0]

        assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
        assert item["evidence"]["owner"]["owner_blocking_state"] == "unknown_owner"


def test_lane_registry_released_worktree_hint_does_not_imply_unpushed_work(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "released",
                    "worktree": "/tmp/aragora-q1",
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["owner"]["advisory_withheld"] is None


def test_lane_registry_terminal_owner_with_available_advisory_does_not_block(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "released",
                    "stale_claim_advisory": {"available": True},
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["owner"]["owner_blocking_state"] is None


def test_lane_registry_possible_unpushed_marker_ignores_unrelated_text(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "released",
                    "note": "documentation mentions possible_unpushed_work",
                    "stale_claim_advisory": {"available": True},
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["owner"]["advisory_withheld"] is None


def test_lane_registry_dirty_signal_overrides_available_advisory(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "released",
                    "stale_claim_advisory": {"available": True},
                    "dirty_worktree": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.BLOCKED_BY_POSSIBLE_UNPUSHED_WORK.value
    assert item["evidence"]["owner"]["advisory_withheld"] == "possible_unpushed_work"


@pytest.mark.parametrize("marker_value", ["false", "0", "none", "clean", "verified-clean"])
def test_lane_registry_negative_dirty_signal_keeps_available_advisory(
    tmp_path: Path,
    marker_value: str,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "released",
                    "stale_claim_advisory": {"available": True},
                    "dirty_worktree": marker_value,
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    item = payload["items"][0]

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["owner"]["advisory_withheld"] is None


def test_owner_payload_evidence_redacts_local_session_metadata(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/example",
                    "status": "active",
                    "worktree": "/private/tmp/sensitive-worktree",
                    "cwd": "/private/tmp/sensitive-worktree",
                    "pid": 12345,
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=FakeGitHub(),
    )
    owner_payload = payload["items"][0]["evidence"]["owner"]["payload"]

    assert owner_payload["lane_id"] == "Q1"
    assert "worktree" not in owner_payload
    assert "cwd" not in owner_payload
    assert "pid" not in owner_payload


def test_human_owner_blocks_handoff(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "human-pr8570",
                "owner_session": "human-operator-pr8570",
                "source": "human",
                "status": "blocked",
                "owner_blocking_state": "live_owner",
            }
        }
    )

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_HUMAN.value


def test_terminal_steering_receipt_consumes_blocking_effect(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    message_path = _write_steering_message(tmp_path, branch="codex/example")
    _write_steering_receipt(message_path, outcome="completed")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["steering"]["pending_message_count"] == 1
    assert item["evidence"]["steering"]["blocking_message_count"] == 1
    assert item["evidence"]["steering"]["resolved_read_receipt_count"] == 1
    assert item["evidence"]["steering"]["ack_protocol"] == "top_level_message_remains_pending"
    assert item["evidence"]["steering"]["latest_read_receipt"]["outcome"] == "completed"
    assert "resolved_message_count" not in item["evidence"]["steering"]
    assert "resolved_by_read_receipt" not in item["evidence"]["steering"]["latest_message"]
    assert "subject" not in item["evidence"]["steering"]["latest_message"]
    assert "body" not in item["evidence"]["steering"]["latest_message"]
    assert item["evidence"]["steering"]["latest_message"]["subject_present"] is True
    assert item["evidence"]["steering"]["latest_message"]["body_present"] is True


def test_nonterminal_steering_receipt_still_blocks(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    message_path = _write_steering_message(tmp_path, branch="codex/example")
    _write_steering_receipt(message_path, outcome="blocked")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["steering"]["pending_message_count"] == 1
    assert item["evidence"]["steering"]["blocking_message_count"] == 1
    assert item["evidence"]["steering"]["human_message_count"] == 0
    assert item["evidence"]["steering"]["latest_read_receipt"]["outcome"] == "blocked"


def test_operator_steering_without_human_keyword_is_owner_block(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    _write_steering_message(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["steering"]["blocking_message_count"] == 1
    assert item["evidence"]["steering"]["human_message_count"] == 0


def test_steering_branch_match_does_not_use_substring(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    _write_steering_message(
        tmp_path,
        branch="codex/example-long",
        owner_session="engineering-autopilot-Q1",
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["steering"]["pending_message_count"] == 0


def test_steering_branch_match_survives_trailing_punctuation(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    inbox = tmp_path / ".aragora" / "operator-steering" / "engineering-autopilot-Q1"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "2026-06-24T00-00-00-000Z-fixture.json").write_text(
        json.dumps(
            {
                "schema_version": "aragora-operator-steering/1.0",
                "to_session": "engineering-autopilot-Q1",
                "priority": "blocking",
                "subject": "Block codex/example.",
                "body": "Please resolve codex/example.",
                "message_sha256": "fixture-sha",
            }
        ),
        encoding="utf-8",
    )

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["steering"]["blocking_message_count"] == 1


def test_steering_lane_hint_match_survives_github_pr_url_without_branch(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    inbox = tmp_path / ".aragora" / "operator-steering" / "engineering-autopilot-Q1"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "2026-06-24T00-00-00-000Z-pr-url.json").write_text(
        json.dumps(
            {
                "schema_version": "aragora-operator-steering/1.0",
                "to_session": "engineering-autopilot-Q1",
                "lane_id_hint": "Q1",
                "priority": "blocking",
                "subject": "Resolve https://github.com/synaptent/aragora/pull/8570",
                "body": "Do not move non-owner state until that PR is represented.",
                "message_sha256": "fixture-sha",
            }
        ),
        encoding="utf-8",
    )
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "released",
                "stale_claim_advisory": {"available": True},
            }
        }
    )

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["steering"]["pending_message_count"] == 1
    assert item["evidence"]["steering"]["blocking_message_count"] == 1


def test_steering_to_session_requires_branch_or_lane_correlation(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    _write_steering_message(
        tmp_path,
        branch="codex/other",
        owner_session="engineering-autopilot-Q1",
    )
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "released",
                "stale_claim_advisory": {"available": True},
            }
        }
    )

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["steering"]["pending_message_count"] == 0


def test_steering_unrelated_branch_message_with_pr_url_stays_excluded(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    inbox = tmp_path / ".aragora" / "operator-steering" / "engineering-autopilot-Q1"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "2026-06-24T00-00-00-000Z-other-pr-url.json").write_text(
        json.dumps(
            {
                "schema_version": "aragora-operator-steering/1.0",
                "to_session": "engineering-autopilot-Q1",
                "priority": "blocking",
                "subject": "Block codex/other",
                "body": "Evidence: https://github.com/synaptent/aragora/pull/8570",
                "message_sha256": "fixture-sha",
            }
        ),
        encoding="utf-8",
    )
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "released",
                "stale_claim_advisory": {"available": True},
            }
        }
    )

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["steering"]["pending_message_count"] == 0


def test_steering_session_wide_message_without_branch_or_lane_does_not_match(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    inbox = tmp_path / ".aragora" / "operator-steering" / "engineering-autopilot-Q1"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "2026-06-24T00-00-00-000Z-generic.json").write_text(
        json.dumps(
            {
                "schema_version": "aragora-operator-steering/1.0",
                "to_session": "engineering-autopilot-Q1",
                "priority": "blocking",
                "subject": "Please wait",
                "body": "Pause until the operator reviews this lane.",
                "message_sha256": "fixture-sha",
            }
        ),
        encoding="utf-8",
    )
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "status": "released",
                "stale_claim_advisory": {"available": True},
            }
        }
    )

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["steering"]["pending_message_count"] == 0


def test_human_detection_ignores_operator_steering_path() -> None:
    assert (
        mod._looks_human(  # noqa: SLF001 - regression coverage for classifier helper
            {
                "path": "/repo/.aragora/operator-steering/worker/message.json",
                "from": "automation",
                "subject": "Block codex/example",
                "body": "Machine-generated lane advisory",
            }
        )
        is False
    )
    assert mod._looks_human({"body": "non-human automated advisory"}) is False  # noqa: SLF001
    assert mod._looks_human({"from": "operator"}) is False  # noqa: SLF001
    assert mod._looks_human({"body": "operator approval required"}) is True  # noqa: SLF001


def test_steering_lane_hint_requires_exact_token(tmp_path: Path) -> None:
    inbox = tmp_path / ".aragora" / "operator-steering" / "engineering-autopilot-Q100"
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "2026-06-24T00-00-00-000Z-q100.json").write_text(
        json.dumps(
            {
                "schema_version": "aragora-operator-steering/1.0",
                "to_session": "engineering-autopilot-Q100",
                "lane_id_hint": "Q100-read-steering",
                "priority": "blocking",
                "subject": "Block codex/other",
                "body": "This message is for codex/other.",
                "message_sha256": "fixture-sha",
            }
        ),
        encoding="utf-8",
    )

    evidence = mod.steering_evidence_for_branch(
        state_root=tmp_path,
        branch="codex/example",
        owner_session=None,
        lane_id="Q1",
    )

    assert evidence.pending_message_count == 0


def test_heads_match_accepts_reconcile_compatible_sha_prefix() -> None:
    full_sha = "abcdef1234567890abcdef1234567890abcdef12"

    assert mod.heads_match(full_sha, full_sha) is True
    assert mod.heads_match("abcdef1", "abcdef1") is True
    assert mod.heads_match("abcdef1", full_sha) is True
    assert mod.heads_match(full_sha, "abcdef1") is True
    assert mod.heads_match("abcdef", full_sha) is False
    assert mod.heads_match("abcdeff", full_sha) is False


def test_narrow_rest_queries_when_open_pr_cache_lacks_branch(
    tmp_path: Path,
) -> None:
    client = mod.NarrowGitHubClient(
        repo_root=tmp_path,
        github_repo="synaptent/aragora",
    )
    seen: list[str] = []

    def capture_api(endpoint: str) -> tuple[Any | None, str | None]:
        seen.append(endpoint)
        return [], None

    client._api = capture_api  # type: ignore[method-assign]

    open_prs, error = client.open_prs_for_branch("elves/run-example")

    assert error is None
    assert open_prs == []
    assert seen == [
        "repos/synaptent/aragora/pulls?state=open&head=synaptent:elves%2Frun-example&per_page=100&page=1"
    ]


def test_narrow_rest_open_pr_query_preserves_owner_separator(tmp_path: Path) -> None:
    client = mod.NarrowGitHubClient(
        repo_root=tmp_path,
        github_repo="synaptent/aragora",
    )
    seen: list[str] = []

    def capture_api(endpoint: str) -> tuple[Any | None, str | None]:
        seen.append(endpoint)
        return [], None

    client._api = capture_api  # type: ignore[method-assign]

    open_prs, error = client.open_prs_for_branch("codex/example")

    assert error is None
    assert open_prs == []
    assert seen == [
        "repos/synaptent/aragora/pulls?state=open&head=synaptent:codex%2Fexample&per_page=100&page=1"
    ]


def test_narrow_rest_open_pr_query_paginates_exact_branch_results(tmp_path: Path) -> None:
    client = mod.NarrowGitHubClient(
        repo_root=tmp_path,
        github_repo="synaptent/aragora",
    )
    page_one = [{"number": index} for index in range(100)]
    page_two = [{"number": 200}]
    seen: list[str] = []

    def capture_api(endpoint: str) -> tuple[Any | None, str | None]:
        seen.append(endpoint)
        if endpoint.endswith("page=1"):
            return page_one, None
        return page_two, None

    client._api = capture_api  # type: ignore[method-assign]

    open_prs, error = client.open_prs_for_branch("codex/example")

    assert error is None
    assert open_prs == [*page_one, *page_two]
    assert seen == [
        "repos/synaptent/aragora/pulls?state=open&head=synaptent:codex%2Fexample&per_page=100&page=1",
        "repos/synaptent/aragora/pulls?state=open&head=synaptent:codex%2Fexample&per_page=100&page=2",
    ]


def test_narrow_rest_open_pr_query_fails_closed_when_page_cap_reached(tmp_path: Path) -> None:
    client = mod.NarrowGitHubClient(
        repo_root=tmp_path,
        github_repo="synaptent/aragora",
    )

    def capture_api(endpoint: str) -> tuple[Any | None, str | None]:
        return [{"number": index} for index in range(100)], None

    client._api = capture_api  # type: ignore[method-assign]

    open_prs, error = client.open_prs_for_branch("codex/example")

    assert open_prs is None
    assert "page cap" in (error or "")


def test_narrow_rest_open_pr_query_fails_closed_for_malformed_github_repo(
    tmp_path: Path,
) -> None:
    client = mod.NarrowGitHubClient(
        repo_root=tmp_path,
        github_repo="synaptent",
    )

    open_prs, error = client.open_prs_for_branch("codex/example")
    ref, ref_error = client.remote_ref("codex/example")

    assert open_prs is None
    assert "owner/name" in (error or "")
    assert ref is None
    assert "owner/name" in (ref_error or "")


def test_remote_ref_treats_structured_not_found_as_absent(tmp_path: Path) -> None:
    client = mod.NarrowGitHubClient(
        repo_root=tmp_path,
        github_repo="synaptent/aragora",
    )

    def capture_api(endpoint: str) -> tuple[Any | None, str | None]:
        return None, "github_not_found: gh api exited 1: HTTP 404: Not Found"

    client._api = capture_api  # type: ignore[method-assign]

    ref, error = client.remote_ref("codex/example")

    assert ref is None
    assert error is None


def test_github_degraded_pr_lookup_fails_closed_for_publication(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(errors={"pr:codex/example": "gh api failed (TimeoutExpired)"})

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.UNKNOWN.value
    assert "GitHub evidence is unavailable" in item["reason"]
    assert item["next_mutation_candidate"] == "none"


def test_github_degraded_pr_lookup_still_reports_local_owner_blocker(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=False)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(errors={"pr:codex/example": "gh api failed (TimeoutExpired)"})
    owner = FakeOwnerProbe(
        {
            "codex/example": {
                "available": True,
                "matched": True,
                "lane_id": "Q1",
                "owner_session": "engineering-autopilot-Q1",
                "owner_blocking_state": "live_owner",
                "status": "active",
            }
        }
    )

    item = _classify_one(tmp_path, github=github, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["github"]["mode"] == "degraded"
    assert item["next_mutation_candidate"] == "owner_followup"


def test_github_degraded_pr_lookup_still_reports_live_queue_cap(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(errors={"pr:codex/example": "gh api failed (TimeoutExpired)"})

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_LIVE_QUEUE_CAP.value
    assert item["evidence"]["github"]["mode"] == "degraded"
    assert item["next_mutation_candidate"] == "queue_drain"


def test_missing_origin_disables_github_instead_of_defaulting_repo(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
    )

    assert payload["github_repo"] is None
    assert payload["github"]["mode"] == "disabled"
    assert payload["items"][0]["evidence"]["github"]["mode"] == "disabled"
    assert payload["items"][0]["state"] == mod.HandoffState.UNKNOWN.value


def test_default_state_root_uses_automation_state_root_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_repo = tmp_path / "shared"
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_status_cache(state_repo)
    _write_outbox(state_repo, branch="codex/example")
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_repo / ".aragora"))

    payload = mod.classify_handoffs(
        repo_root=repo,
        github_repo="synaptent/aragora",
        github_client=FakeGitHub(),
    )

    assert payload["outbox_count"] == 1
    assert payload["state_root"] == str(state_repo / ".aragora")


def test_text_summary_only_omits_items(
    tmp_path: Path,
    capsys,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")

    code = cli.main(
        [
            "--repo",
            str(tmp_path),
            "--state-root",
            str(tmp_path),
            "--outbox-file",
            "open-pr-codex-example-aaaaaaaa.json",
            "--summary-only",
        ]
    )
    output = capsys.readouterr().out

    assert code == 2
    assert "counts:" in output
    assert "items:" not in output


def test_outbox_file_outside_outbox_fails_loudly(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="outbox file must be inside"):
        mod.classify_handoffs(
            repo_root=tmp_path,
            state_root=tmp_path,
            github_repo="synaptent/aragora",
            outbox_file=outside,
            github_client=FakeGitHub(),
        )


def test_cli_fail_on_unsafe_state_returns_nonzero_for_disabled_github(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")

    code = cli.main(
        [
            "--repo",
            str(tmp_path),
            "--state-root",
            str(tmp_path),
            "--outbox-file",
            "open-pr-codex-example-aaaaaaaa.json",
            "--json",
            "--fail-on-unsafe-state",
        ]
    )

    assert code == 2


def test_cli_returns_nonzero_for_unsafe_state_by_default(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")

    code = cli.main(
        [
            "--repo",
            str(tmp_path),
            "--state-root",
            str(tmp_path),
            "--outbox-file",
            "open-pr-codex-example-aaaaaaaa.json",
            "--json",
        ]
    )

    assert code == 2


def test_cli_fail_on_unsafe_state_rejects_partial_github_item_errors() -> None:
    payload = {
        "github": {"mode": "partial", "partial_degradation": True, "item_error_count": 1},
        "items": [
            {
                "state": "blocked_by_owner",
                "next_mutation_candidate": "owner_followup",
                "safe_to_mutate": False,
                "evidence": {"github": {"error": "gh api failed (TimeoutExpired)"}},
            }
        ],
    }

    assert cli._has_unsafe_state(payload) is True  # noqa: SLF001


def test_cli_fail_on_unsafe_state_rejects_partial_github_even_when_items_look_safe() -> None:
    payload = {
        "github": {"mode": "partial", "partial_degradation": True, "item_error_count": 1},
        "items": [
            {
                "state": "represented_by_exact_open_pr",
                "next_mutation_candidate": "write_representation_receipt_then_archive",
                "safe_to_mutate": True,
            }
        ],
    }

    assert cli._has_unsafe_state(payload) is True  # noqa: SLF001


def test_cli_fail_on_unsafe_state_rejects_preserved_not_actionable() -> None:
    payload = {
        "github": {"mode": "ready"},
        "items": [
            {
                "state": "preserved_not_actionable",
                "next_mutation_candidate": "none",
                "safe_to_mutate": False,
            }
        ],
    }

    assert cli._has_unsafe_state(payload) is True  # noqa: SLF001


def test_cli_fail_on_unsafe_state_rejects_non_mutatable_exact_draft_pr() -> None:
    payload = {
        "github": {"mode": "ready"},
        "items": [
            {
                "state": "represented_by_exact_open_pr",
                "next_mutation_candidate": "none",
                "safe_to_mutate": False,
            }
        ],
    }

    assert cli._has_unsafe_state(payload) is True  # noqa: SLF001


def test_classify_reports_partial_github_errors_without_global_blindness(
    tmp_path: Path,
) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, key="open-pr-codex-safe-aaaaaaaa", branch="codex/safe")
    _write_outbox(tmp_path, key="open-pr-codex-blocked-aaaaaaaa", branch="codex/blocked")
    lanes_path = tmp_path / ".aragora" / "agent-bridge" / "lanes.json"
    lanes_path.parent.mkdir(parents=True, exist_ok=True)
    lanes_path.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q1",
                    "owner_session": "engineering-autopilot-Q1",
                    "branch": "codex/blocked",
                    "status": "active",
                }
            ]
        ),
        encoding="utf-8",
    )
    github = FakeGitHub(
        open_prs={
            "codex/safe": [
                {
                    "number": 8599,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/safe", "sha": HEAD},
                }
            ]
        },
        errors={"pr:codex/blocked": "gh api failed (TimeoutExpired)"},
    )

    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        github_client=github,
    )

    assert payload["github"]["mode"] == "partial"
    assert payload["github"]["item_error_count"] == 1
    assert payload["counts"][mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value] == 1
    assert payload["counts"][mod.HandoffState.BLOCKED_BY_OWNER.value] == 1


def test_owner_probe_failure_fails_closed(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "identify_lane_owner.py").write_text(
        "import sys\nprint('database locked', file=sys.stderr)\nsys.exit(2)\n",
        encoding="utf-8",
    )
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    owner = mod.OwnerProbe(repo_root=tmp_path, state_root=tmp_path, timeout_seconds=2)

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_OWNER.value
    assert item["evidence"]["owner"]["available"] is False
    assert item["evidence"]["owner"]["matched"] is False


def test_owner_probe_no_matching_lane_does_not_block(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "identify_lane_owner.py").write_text(
        "import sys\nprint('ERROR: no matching lane criteria', file=sys.stderr)\nsys.exit(1)\n",
        encoding="utf-8",
    )
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    owner = mod.OwnerProbe(repo_root=tmp_path, state_root=tmp_path, timeout_seconds=2)

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["owner"]["available"] is True
    assert item["evidence"]["owner"]["matched"] is False


def test_owner_probe_liveness_dirty_signal_blocks(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "identify_lane_owner.py").write_text(
        "import json\n"
        "print(json.dumps({"
        "'branch': 'codex/example', "
        "'status': 'released', "
        "'dirty_worktree': True, "
        "'stale_claim_advisory': {'available': True}"
        "}))\n",
        encoding="utf-8",
    )
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    owner = mod.OwnerProbe(repo_root=tmp_path, state_root=tmp_path, timeout_seconds=2)

    item = _classify_one(tmp_path, owner=owner)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_POSSIBLE_UNPUSHED_WORK.value
    assert item["evidence"]["owner"]["advisory_withheld"] == "possible_unpushed_work"


def test_selected_outbox_file_cannot_escape_outbox_dir(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    outside = tmp_path / ".aragora" / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="outbox file must be inside"):
        mod._selected_outbox_files(outbox, "../outside.json")  # noqa: SLF001


def test_update_pr_idempotency_key_is_pr_publication_request() -> None:
    assert mod.is_pr_publication_request({"idempotency_key": "update-pr-codex-example-abc123"})


def test_requested_action_python_repr_is_pr_publication_request() -> None:
    payload = {
        "requested_action": "{'type': 'open_or_update_pr', 'branch': 'codex/example'}",
    }

    assert mod.is_pr_publication_request(payload)


def test_graphql_degraded_cache_still_uses_rest_open_pr_fallback(tmp_path: Path) -> None:
    _write_status_cache(tmp_path, open_pr_cap_reached=True, degraded=True)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 8589,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/example", "sha": HEAD},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["evidence"]["queue_cap"]["degraded"] is True
    assert item["evidence"]["github"]["exact_open_pr"]["number"] == 8589


def test_open_pr_head_mismatch_does_not_count_as_exact_representation(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    github = FakeGitHub(
        open_prs={
            "codex/example": [
                {
                    "number": 9000,
                    "state": "open",
                    "draft": True,
                    "head": {"ref": "codex/example", "sha": OTHER_HEAD},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["github"]["exact_open_pr"] is None
