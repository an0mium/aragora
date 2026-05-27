from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import scripts.codex_automation_harvest_report as mod


def _git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "checkout", "-b", "main"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "codex@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Codex"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    (path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"], cwd=path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "base"], cwd=path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "checkout", "-b", "codex/provider-proof"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    (path / "README.md").write_text("base\nproof\n", encoding="utf-8")
    subprocess.run(
        ["git", "commit", "-am", "proof"], cwd=path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "checkout", "main"], cwd=path, check=True, capture_output=True, text=True
    )


def _outbox_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "Provider dogfood proof receipt",
        "requires_github": True,
        "requested_action": "open_pr",
        "repo": "synaptent/aragora",
        "local_evidence": {"branch": "codex/provider-proof", "head": "abc123"},
        "validation": ["pytest tests/provider.py -q"],
        "idempotency_key": "open-pr-codex-provider-proof-abc123",
        "created_at": "2026-05-27T12:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def test_build_report_classifies_and_ranks_unharvested_handoffs(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    state_root = tmp_path / ".aragora"
    outbox = state_root / "automation-outbox"
    receipts = state_root / "automation-receipts"
    cache = state_root / "automation-github-status"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    cache.mkdir(parents=True)
    (outbox / "proof.json").write_text(json.dumps(_outbox_payload()), encoding="utf-8")
    (outbox / "issue-only.json").write_text(
        json.dumps(
            _outbox_payload(
                task="Investigate stale coordination note",
                local_evidence={},
                idempotency_key="issue-codex-stale-coordination-abc123",
            )
        ),
        encoding="utf-8",
    )
    (receipts / "completed.json").write_text(
        json.dumps({"idempotency_key": "completed", "status": "completed"}),
        encoding="utf-8",
    )
    (cache / "latest.json").write_text(
        json.dumps({"local_queue": {"outbox_count": 2}, "github_health": {"mode": "ready"}}),
        encoding="utf-8",
    )

    report = mod.build_report(
        repo_root=tmp_path,
        state_root=state_root,
        operator_snapshot={
            "pending_steering_messages": {"count": 2, "unread_message_count": 1},
            "agent_heartbeats": {"fresh_count": 0, "stale_count": 4},
        },
        worktree_harvest={"generated_at": "2026-05-27T12:00:00Z", "summary": {"x": 1}},
    )

    assert report["automation_harvest_schema_version"] == 1
    assert report["counts"]["outbox_total_count"] == 2
    assert report["counts"]["terminal_receipt_count"] == 1
    assert report["counts"]["pending_steering_messages"] == 2
    assert report["counts"]["fresh_agent_heartbeats"] == 0
    assert report["classification_counts"] == {
        "issue_only": 1,
        "product_proof_candidate": 1,
    }
    assert report["top_next_handoffs"][0]["classification"] == "product_proof_candidate"
    assert report["recommended_next_target"]["task"] == "Provider dogfood proof receipt"


def test_write_latest_report_uses_stable_schema_path(tmp_path: Path) -> None:
    report = {
        "automation_harvest_schema_version": 1,
        "generated_at": "2026-05-27T12:00:00Z",
        "counts": {},
    }

    path = mod.write_latest_report(report, state_root=tmp_path / ".aragora")

    assert path == tmp_path / ".aragora" / "automation-harvest" / "latest.json"
    assert json.loads(path.read_text(encoding="utf-8"))["automation_harvest_schema_version"] == 1
