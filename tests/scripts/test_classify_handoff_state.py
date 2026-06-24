from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import scripts.handoff_state as mod


HEAD = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
OTHER_HEAD = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


class FakeGitHub:
    disabled = False

    def __init__(
        self,
        *,
        open_prs: dict[str, list[dict[str, Any]]] | None = None,
        refs: dict[str, dict[str, Any]] | None = None,
        errors: dict[str, str] | None = None,
    ) -> None:
        self.open_prs = open_prs or {}
        self.refs = refs or {}
        self.errors = errors or {}
        self.pr_calls = 0
        self.ref_calls = 0

    def open_prs_for_branch(self, branch: str) -> tuple[list[dict[str, Any]] | None, str | None]:
        self.pr_calls += 1
        error = self.errors.get(f"pr:{branch}")
        if error:
            return None, error
        return self.open_prs.get(branch, []), None

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
) -> Path:
    outbox = state_root / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    path = outbox / f"{key}.json"
    path.write_text(
        json.dumps(
            {
                "idempotency_key": key,
                "requested_action": {
                    "type": "open_or_update_pr",
                    "branch": branch,
                    "desired_head_sha": head,
                    "head_sha": head,
                },
                "branch": branch,
                "desired_head_sha": head,
                "head_sha": head,
                "repo": "synaptent/aragora",
                "task": f"Open PR for {branch}",
            }
        ),
        encoding="utf-8",
    )
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
    github: FakeGitHub | None = None,
    owner: FakeOwnerProbe | None = None,
) -> dict[str, Any]:
    payload = mod.classify_handoffs(
        repo_root=tmp_path,
        state_root=tmp_path,
        github_repo="synaptent/aragora",
        outbox_file="open-pr-codex-example-aaaaaaaa.json",
        github_client=github or FakeGitHub(),
        owner_probe=owner or FakeOwnerProbe(),
    )
    assert payload["outbox_count"] == 1
    return payload["items"][0]


def test_exact_open_pr_representation_wins_over_owner_noise(tmp_path: Path) -> None:
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

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["evidence"]["github"]["exact_open_pr"]["number"] == 8570
    assert item["next_mutation_candidate"] == "write_representation_receipt_then_archive"
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
                    "draft": True,
                    "html_url": "https://github.com/synaptent/aragora/pull/8570",
                    "head": {"ref": "codex/example", "sha": HEAD},
                }
            ]
        }
    )

    item = _classify_one(tmp_path, github=github)

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value
    assert item["safe_to_mutate"] is True


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


def test_stale_queue_cap_cache_does_not_block_publication(tmp_path: Path) -> None:
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    _write_status_cache(tmp_path, open_pr_cap_reached=True, generated_at=stale)
    _write_outbox(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["queue_cap"]["raw_open_pr_cap_reached"] is True
    assert item["evidence"]["queue_cap"]["open_pr_cap_reached"] is None
    assert item["evidence"]["queue_cap"]["cache_stale"] is True
    assert item["evidence"]["queue_cap"]["decision_source"] == "expired_cache"


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


def test_stale_owner_remote_exact_head_is_represented_by_remote_branch(
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

    assert item["state"] == mod.HandoffState.REPRESENTED_BY_EXACT_REMOTE_BRANCH.value
    assert item["evidence"]["github"]["remote_ref"]["sha"] == HEAD


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

    assert item["state"] == mod.HandoffState.PUBLICATION_REQUESTED.value
    assert item["evidence"]["steering"]["pending_message_count"] == 1
    assert item["evidence"]["steering"]["resolved_message_count"] == 1
    assert item["evidence"]["steering"]["blocking_message_count"] == 0
    assert item["evidence"]["steering"]["latest_read_receipt"]["outcome"] == "completed"
    assert item["evidence"]["steering"]["latest_message"]["resolved_by_read_receipt"] is True


def test_nonterminal_steering_receipt_still_blocks(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    message_path = _write_steering_message(tmp_path, branch="codex/example")
    _write_steering_receipt(message_path, outcome="blocked")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_HUMAN.value
    assert item["evidence"]["steering"]["pending_message_count"] == 1
    assert item["evidence"]["steering"]["resolved_message_count"] == 0
    assert item["evidence"]["steering"]["blocking_message_count"] == 1
    assert item["evidence"]["steering"]["human_message_count"] == 1
    assert item["evidence"]["steering"]["latest_read_receipt"]["outcome"] == "blocked"


def test_operator_steering_without_human_keyword_is_human_gate(tmp_path: Path) -> None:
    _write_status_cache(tmp_path)
    _write_outbox(tmp_path, branch="codex/example")
    _write_steering_message(tmp_path, branch="codex/example")

    item = _classify_one(tmp_path)

    assert item["state"] == mod.HandoffState.BLOCKED_BY_HUMAN.value
    assert item["evidence"]["steering"]["human_message_count"] == 1


def test_narrow_rest_queries_when_open_pr_cache_lacks_branch(
    tmp_path: Path,
) -> None:
    client = mod.NarrowGitHubClient(
        repo_root=tmp_path,
        github_repo="synaptent/aragora",
        known_open_pr_heads=set(),
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
        "repos/synaptent/aragora/pulls?state=open&head=synaptent:elves%2Frun-example&per_page=5"
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
        "repos/synaptent/aragora/pulls?state=open&head=synaptent:codex%2Fexample&per_page=5"
    ]


def test_open_pr_head_cache_is_trusted_only_when_complete(tmp_path: Path) -> None:
    _write_status_cache(tmp_path, open_pr_heads=["codex/example"])
    status_path = tmp_path / ".aragora" / "automation-github-status" / "latest.json"

    assert mod.load_fresh_open_pr_head_cache(status_path) == {"codex/example"}

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["github_queue"]["open_codex_pr_count"] = 2
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    assert mod.load_fresh_open_pr_head_cache(status_path) is None


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
