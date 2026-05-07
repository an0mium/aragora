from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.cache_codex_automation_github_status as mod
from scripts.github_cli_health import GitHubCLIHealth


def test_build_status_uses_local_queue_when_github_unavailable(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    (outbox / "open-pr-example.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="sandboxed",
            repo=str(repo_root),
        ),
    )

    payload = mod.build_status(
        repo_root=tmp_path,
        github_repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_prs=12,
        max_open_issues=16,
    )

    assert payload["github_queue"] == {
        "available": False,
        "reason": "connectivity_failed",
    }
    assert payload["local_queue"]["outbox_count"] == 1
    assert payload["local_queue"]["receipt_count"] == 0
    assert payload["local_queue"]["terminal_receipt_count"] == 0
    assert payload["local_queue"]["nonterminal_receipt_count"] == 0
    assert payload["local_queue"]["nonterminal_receipts"] == []
    assert payload["local_queue"]["terminal_receipted_outbox_count"] == 0
    assert payload["local_queue"]["unreceipted_outbox_count"] == 1


def test_local_queue_state_matches_receipts_by_idempotency_key(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-example-abc123"
    (outbox / "handoff-file-name.json").write_text(
        f'{{"idempotency_key": "{key}"}}',
        encoding="utf-8",
    )
    (receipts / "different-receipt-file.json").write_text(
        f'{{"idempotency_key": "{key}", "status": "published"}}',
        encoding="utf-8",
    )

    payload = mod._local_queue_state(
        repo_root=tmp_path,
        outbox_dir=None,
        receipt_dir=None,
    )

    assert payload["outbox_count"] == 1
    assert payload["receipt_count"] == 1
    assert payload["terminal_receipt_count"] == 1
    assert payload["nonterminal_receipt_count"] == 0
    assert payload["nonterminal_receipts"] == []
    assert payload["terminal_receipted_outbox_count"] == 1
    assert payload["unreceipted_outbox_count"] == 0


def test_local_queue_state_reports_duplicate_outbox_handoffs_by_branch(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    first_key = "open-pr-codex-example-abc123"
    second_key = "open-pr-codex-example-def456"
    payloads = {
        "first.json": {
            "idempotency_key": first_key,
            "requested_action": {"type": "open_pr", "branch": "codex/example"},
        },
        "second.json": {
            "idempotency_key": second_key,
            "requested_action": json.dumps({"type": "open_pr", "branch": "codex/example"}),
        },
        "third.json": {
            "idempotency_key": second_key,
            "local_evidence": {"branch": "codex/example"},
        },
    }
    for name, payload in payloads.items():
        (outbox / name).write_text(json.dumps(payload), encoding="utf-8")

    payload = mod._local_queue_state(
        repo_root=tmp_path,
        outbox_dir=None,
        receipt_dir=None,
    )

    assert payload["outbox_count"] == 3
    assert payload["outbox_unique_idempotency_count"] == 2
    assert payload["outbox_duplicate_idempotency_count"] == 1
    assert payload["outbox_duplicate_idempotency_keys"] == [
        {"idempotency_key": second_key, "count": 2}
    ]
    assert payload["outbox_branch_count"] == 3
    assert payload["outbox_unique_branch_count"] == 1
    assert payload["outbox_duplicate_branch_count"] == 2
    assert payload["outbox_duplicate_branches"] == [
        {
            "branch": "codex/example",
            "count": 3,
            "files": ["first.json", "second.json", "third.json"],
            "idempotency_keys": [first_key, second_key, second_key],
        }
    ]


def test_local_queue_state_ignores_nonterminal_receipts(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-example-abc123"
    (outbox / "handoff.json").write_text(
        f'{{"idempotency_key": "{key}"}}',
        encoding="utf-8",
    )
    (receipts / "failed-receipt.json").write_text(
        f'{{"idempotency_key": "{key}", "status": "failed"}}',
        encoding="utf-8",
    )

    payload = mod._local_queue_state(
        repo_root=tmp_path,
        outbox_dir=None,
        receipt_dir=None,
    )

    assert payload["outbox_count"] == 1
    assert payload["receipt_count"] == 1
    assert payload["terminal_receipt_count"] == 0
    assert payload["nonterminal_receipt_count"] == 1
    assert payload["nonterminal_receipts"] == [
        {
            "file": "failed-receipt.json",
            "idempotency_key": key,
            "status": "failed",
        }
    ]
    assert payload["terminal_receipted_outbox_count"] == 0
    assert payload["unreceipted_outbox_count"] == 1


def test_local_queue_state_reports_missing_receipt_status(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "automation-receipts"
    receipts.mkdir(parents=True)
    key = "open-pr-codex-example-abc123"
    (receipts / "legacy-receipt.json").write_text(
        f'{{"idempotency_key": "{key}"}}',
        encoding="utf-8",
    )

    payload = mod._local_queue_state(
        repo_root=tmp_path,
        outbox_dir=None,
        receipt_dir=None,
    )

    assert payload["receipt_count"] == 1
    assert payload["terminal_receipt_count"] == 0
    assert payload["nonterminal_receipt_count"] == 1
    assert payload["nonterminal_receipts"] == [
        {
            "file": "legacy-receipt.json",
            "idempotency_key": key,
            "status": "missing",
        }
    ]


def test_local_queue_state_accepts_direct_dot_aragora_state_root(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "disposable-worktree"
    state_root = tmp_path / "shared" / ".aragora"
    outbox = state_root / "automation-outbox"
    receipts = state_root / "automation-receipts"
    repo.mkdir()
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    (outbox / "handoff.json").write_text(
        '{"idempotency_key": "open-pr-codex-example-abc123"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_root))

    payload = mod._local_queue_state(
        repo_root=repo,
        outbox_dir=None,
        receipt_dir=None,
    )

    assert payload["outbox_dir"] == str(outbox)
    assert payload["receipt_dir"] == str(receipts)
    assert payload["outbox_count"] == 1
    assert payload["receipt_count"] == 0
    assert payload["unreceipted_outbox_count"] == 1


def test_build_status_records_remote_pressure_when_github_available(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=True,
            auth_ok=True,
            api_ok=True,
            mode="ready",
            error="",
            repo=str(repo_root),
        ),
    )
    monkeypatch.setattr(
        mod,
        "_open_codex_prs",
        lambda repo_root, repo: [
            {"headRefName": "codex/a", "mergeStateStatus": "DIRTY"},
            {
                "headRefName": "codex/b",
                "mergeStateStatus": "BLOCKED",
                "reviewDecision": "REVIEW_REQUIRED",
                "statusCheckRollup": [{"conclusion": "SUCCESS"}],
            },
        ],
    )
    monkeypatch.setattr(mod, "_open_boss_ready_count", lambda repo_root, repo, labels: 16)

    payload = mod.build_status(
        repo_root=tmp_path,
        github_repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_prs=2,
        max_open_issues=16,
    )

    queue = payload["github_queue"]
    assert queue["available"] is True
    assert queue["open_codex_pr_count"] == 2
    assert queue["unhealthy_open_pr_count"] == 1
    assert queue["all_open_prs_unhealthy"] is False
    assert queue["pressure"] == {
        "open_pr_cap_reached": True,
        "open_issue_cap_reached": True,
    }
