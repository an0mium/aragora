from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.github_cli_health import GitHubCLIHealth
import scripts.publish_automation_handoffs as mod
from scripts.publish_automation_handoffs import Handoff, PublishDecision


def _outbox_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "Publish validated repair branch",
        "requires_github": True,
        "requested_action": "open_pr",
        "repo": "synaptent/aragora",
        "local_evidence": {},
        "validation": [],
        "idempotency_key": "open-pr-codex-example-abc123",
        "created_at": "2026-04-24T16:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _memory(root: Path, automation_id: str, text: str) -> Path:
    path = root / "automations" / automation_id / "memory.md"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


def _handoff(title: str = "Fix tmux readiness detection for named Claude lanes") -> str:
    return f"""
# 2026-04-16

Handoff Source: Founder review automation
Priority: MEDIUM
Task Title: {title}
Why Now: Neutral proof-first tmux lanes stay booting even when Claude markers are present.
Repo Evidence:
- session_mux.py falls back to agent name marker inference.
- runbook uses neutral lane names.
Acceptance Criteria:
- Neutral Claude lanes become ready when Claude startup markers are present.
- Codex readiness remains unchanged.
Validation:
- python3 -m pytest tests/swarm/test_session_mux.py -q
Expiration Hours: 72
Backup Task: NONE
""".strip()


def _repo_with_merged_codex_branch(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()

    git("init")
    git("checkout", "-b", "main")
    git("config", "user.email", "codex@example.com")
    git("config", "user.name", "Codex")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "base")
    git("checkout", "-b", "codex/example")
    (repo / "README.md").write_text("base\nchange\n", encoding="utf-8")
    git("commit", "-am", "change")
    head = git("rev-parse", "HEAD")
    git("checkout", "main")
    git("merge", "--ff-only", "codex/example")
    return repo, head


def _repo_with_patch_equivalent_codex_branch(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()

    git("init")
    git("checkout", "-b", "main")
    git("config", "user.email", "codex@example.com")
    git("config", "user.name", "Codex")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "base")
    git("checkout", "-b", "codex/example")
    (repo / "README.md").write_text("base\nbranch change\n", encoding="utf-8")
    git("commit", "-am", "change from branch")
    head = git("rev-parse", "HEAD")
    git("checkout", "main")
    (repo / "README.md").write_text("base\nbranch change\n", encoding="utf-8")
    git("commit", "-am", "same change from main")
    return repo, head


def test_load_handoffs_parses_structured_memory(tmp_path: Path) -> None:
    _memory(tmp_path, "founder-review", _handoff())

    handoffs = mod.load_handoffs(tmp_path, now=datetime(2026, 4, 16, tzinfo=timezone.utc))

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Fix tmux readiness detection for named Claude lanes"
    assert handoffs[0].priority == "MEDIUM"
    assert "Acceptance Criteria:" in handoffs[0].body
    assert "Published from automation memory" in handoffs[0].body


def test_load_outbox_handoffs_parses_structured_json(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    source = outbox / "repair-branch.json"
    source.write_text(
        json.dumps(
            _outbox_payload(
                repo=str(tmp_path),
                local_evidence={
                    "branch": "codex/example",
                    "head": "abc123",
                },
                validation=["pytest tests/example.py -q"],
            )
        ),
        encoding="utf-8",
    )

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Publish validated repair branch"
    assert handoffs[0].source_kind == "outbox"
    assert handoffs[0].idempotency_key == "open-pr-codex-example-abc123"
    assert "Requested Action:" in handoffs[0].body
    assert "Published from automation outbox" in handoffs[0].body


def test_load_outbox_handoffs_extracts_branch_from_list_local_evidence(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    source = outbox / "repair-branch.json"
    source.write_text(
        json.dumps(
            _outbox_payload(
                repo=str(tmp_path),
                local_evidence=[
                    "legacy note",
                    {"branch": "codex/example", "head": "abc123"},
                ],
                validation=["pytest tests/example.py -q"],
            )
        ),
        encoding="utf-8",
    )

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].branch == "codex/example"
    assert (
        mod._outbox_branch_fingerprint(json.loads(source.read_text(encoding="utf-8")))
        == f"open_pr\0{tmp_path}\0codex/example"
    )


def test_load_outbox_handoffs_uses_automation_state_root_for_default_dirs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo_root = tmp_path / "worktree"
    state_root = tmp_path / "state-root"
    repo_root.mkdir()
    outbox = state_root / ".aragora" / "automation-outbox"
    receipts = state_root / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    source = outbox / "repair-branch.json"
    source.write_text(
        json.dumps(
            _outbox_payload(
                repo="synaptent/aragora",
                local_evidence={
                    "branch": "codex/example",
                    "head": "abc123",
                },
                validation=["pytest tests/example.py -q"],
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_root))

    handoffs = mod.load_outbox_handoffs(repo_root)

    assert len(handoffs) == 1
    assert handoffs[0].source_file == str(source.resolve())
    assert handoffs[0].task_title == "Publish validated repair branch"


def test_load_outbox_handoffs_skips_terminal_receipt(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-example-abc123"
    (outbox / "repair-branch.json").write_text(
        json.dumps(_outbox_payload(repo=str(tmp_path), idempotency_key=key)),
        encoding="utf-8",
    )
    (receipts / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(tmp_path) == []


def test_load_outbox_handoffs_skips_terminal_receipt_named_by_file(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-example-abc123"
    (outbox / "repair-branch.json").write_text(
        json.dumps(_outbox_payload(repo=str(tmp_path), idempotency_key=key)),
        encoding="utf-8",
    )
    (receipts / f"{key}.json").write_text(
        json.dumps({"status": "published"}),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(tmp_path) == []


@pytest.mark.parametrize(
    ("reason", "url_key", "url"),
    [
        ("published", "created_issue_url", "https://github.com/synaptent/aragora/issues/7151"),
        (
            "existing_issue",
            "existing_issue_url",
            "https://github.com/synaptent/aragora/issues/6992",
        ),
    ],
)
def test_load_outbox_handoffs_keeps_pr_handoff_after_issue_receipt(
    tmp_path: Path,
    reason: str,
    url_key: str,
    url: str,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-example-abc123"
    (outbox / "repair-branch.json").write_text(
        json.dumps(
            _outbox_payload(
                repo=str(tmp_path),
                idempotency_key=key,
                local_evidence={"branch": "codex/example", "head": "abc123"},
            )
        ),
        encoding="utf-8",
    )
    (receipts / f"{key}.json").write_text(
        json.dumps(
            {
                "idempotency_key": key,
                "status": "published" if reason == "published" else "already_satisfied",
                "reason": reason,
                url_key: url,
            }
        ),
        encoding="utf-8",
    )

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].idempotency_key == key


def test_load_outbox_handoffs_keeps_stale_target_pr_receipt(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()

    git("init")
    git("checkout", "-b", "main")
    git("config", "user.email", "codex@example.com")
    git("config", "user.name", "Codex")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "base")
    git("checkout", "-b", "codex/example")
    (repo / "README.md").write_text("base\nold\n", encoding="utf-8")
    git("commit", "-am", "old")
    old_head = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/codex/example", old_head)
    (repo / "README.md").write_text("base\nold\nnew\n", encoding="utf-8")
    git("commit", "-am", "new")
    desired_head = git("rev-parse", "HEAD")

    outbox = repo / ".aragora" / "automation-outbox"
    receipts = repo / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-example-abc123"
    (outbox / "repair-branch.json").write_text(
        json.dumps(
            _outbox_payload(
                repo="synaptent/aragora",
                idempotency_key=key,
                requested_action={
                    "type": "push_branch_and_open_or_update_pr",
                    "branch": "codex/example",
                    "base": "main",
                    "desired_head_sha": desired_head,
                },
                local_evidence={"branch": "codex/example"},
            )
        ),
        encoding="utf-8",
    )
    (receipts / f"{key}.json").write_text(
        json.dumps(
            {
                "idempotency_key": key,
                "status": "already_satisfied",
                "reason": "target_open_pr",
            }
        ),
        encoding="utf-8",
    )

    handoffs = mod.load_outbox_handoffs(repo)

    assert len(handoffs) == 1
    assert handoffs[0].desired_head == desired_head

    git("update-ref", "-d", "refs/remotes/origin/codex/example")

    handoffs = mod.load_outbox_handoffs(repo)

    assert len(handoffs) == 1
    assert handoffs[0].desired_head == desired_head

    git("update-ref", "refs/remotes/origin/codex/example", desired_head)

    assert mod.load_outbox_handoffs(repo) == []


def test_load_outbox_handoffs_skips_completed_and_skipped_receipts(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    for status in ("completed", "skipped"):
        key = f"open-pr-codex-{status}-abc123"
        (outbox / f"{status}.json").write_text(
            json.dumps(_outbox_payload(repo=str(tmp_path), idempotency_key=key)),
            encoding="utf-8",
        )
        (receipts / f"{key}.json").write_text(
            json.dumps({"idempotency_key": key, "status": status}),
            encoding="utf-8",
        )

    assert mod.load_outbox_handoffs(tmp_path) == []


def test_load_outbox_handoffs_deduplicates_unresolved_idempotency_keys(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    key = "open-pr-codex-example-abc123"
    older = outbox / "older.json"
    newer = outbox / "newer.json"
    older.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish older branch snapshot",
                repo=str(tmp_path),
                idempotency_key=key,
            )
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish newer branch snapshot",
                repo=str(tmp_path),
                idempotency_key=key,
            )
        ),
        encoding="utf-8",
    )
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Publish newer branch snapshot"
    assert handoffs[0].source_file == str(newer)


def test_load_outbox_handoffs_deduplicates_unresolved_branch_handoffs(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    older = outbox / "older.json"
    newer = outbox / "newer.json"
    older.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish older branch snapshot",
                idempotency_key="open-pr-codex-example-old",
                local_evidence={"branch": "codex/example", "head": "abc123"},
            )
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish newer branch snapshot",
                idempotency_key="open-pr-codex-example-new",
                local_evidence={"branch": "codex/example", "head": "def456"},
            )
        ),
        encoding="utf-8",
    )
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Publish newer branch snapshot"
    assert handoffs[0].idempotency_key == "open-pr-codex-example-new"
    assert handoffs[0].source_file == str(newer)


def test_load_outbox_handoffs_deduplicates_pr_action_aliases_for_same_branch(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    older = outbox / "older.json"
    newer = outbox / "newer.json"
    older.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish older branch snapshot",
                requested_action="open_pr",
                idempotency_key="open-pr-codex-example-old",
                local_evidence={"branch": "codex/example", "head": "abc123"},
            )
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish newer branch snapshot",
                requested_action="push_branch_and_open_or_update_pr",
                idempotency_key="open-pr-codex-example-new",
                local_evidence={"branch": "codex/example", "head": "def456"},
            )
        ),
        encoding="utf-8",
    )
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Publish newer branch snapshot"
    assert handoffs[0].idempotency_key == "open-pr-codex-example-new"
    assert handoffs[0].source_file == str(newer)


def test_load_outbox_handoffs_deduplicates_top_level_branch_handoffs(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    older = outbox / "older.json"
    newer = outbox / "newer.json"
    older.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish older branch snapshot",
                branch="codex/example",
                head_sha="abc123",
                idempotency_key="open-pr-codex-example-old",
            )
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish newer branch snapshot",
                idempotency_key="open-pr-codex-example-new",
                local_evidence={"branch": "codex/example", "head": "def456"},
            )
        ),
        encoding="utf-8",
    )
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Publish newer branch snapshot"
    assert handoffs[0].idempotency_key == "open-pr-codex-example-new"
    assert handoffs[0].source_file == str(newer)


def test_load_outbox_handoffs_deduplicates_structured_action_branch_handoffs(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    older = outbox / "older.json"
    newer = outbox / "newer.json"
    older.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish older branch snapshot",
                requested_action={
                    "type": "open_pull_request",
                    "branch": "codex/example",
                    "base": "main",
                },
                idempotency_key="open-pr-codex-example-old",
            )
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish newer branch snapshot",
                requested_action={
                    "type": "open_pull_request",
                    "branch": "codex/example",
                    "base": "main",
                },
                idempotency_key="open-pr-codex-example-new",
            )
        ),
        encoding="utf-8",
    )
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Publish newer branch snapshot"
    assert handoffs[0].idempotency_key == "open-pr-codex-example-new"
    assert handoffs[0].source_file == str(newer)


def test_load_outbox_handoffs_deduplicates_json_string_action_branch_handoffs(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    older = outbox / "older.json"
    newer = outbox / "newer.json"
    requested_action = json.dumps(
        {
            "type": "open_pull_request",
            "branch": "codex/example",
            "base": "main",
            "draft": True,
        }
    )
    older.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish older branch snapshot",
                requested_action=requested_action,
                idempotency_key="open-pr-codex-example-old",
            )
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            _outbox_payload(
                task="Publish newer branch snapshot",
                requested_action=requested_action,
                idempotency_key="open-pr-codex-example-new",
            )
        ),
        encoding="utf-8",
    )
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    handoffs = mod.load_outbox_handoffs(tmp_path)

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Publish newer branch snapshot"
    assert handoffs[0].idempotency_key == "open-pr-codex-example-new"
    assert handoffs[0].source_file == str(newer)
    assert handoffs[0].branch == "codex/example"


def test_load_outbox_handoffs_skips_terminal_receipt_for_same_branch(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    old_key = "open-pr-codex-example-old"
    (outbox / "old.json").write_text(
        json.dumps(
            _outbox_payload(
                idempotency_key=old_key,
                local_evidence={"branch": "codex/example", "head": "abc123"},
            )
        ),
        encoding="utf-8",
    )
    (outbox / "restacked.json").write_text(
        json.dumps(
            _outbox_payload(
                idempotency_key="open-pr-codex-example-new",
                local_evidence={"branch": "codex/example", "head": "def456"},
            )
        ),
        encoding="utf-8",
    )
    (receipts / f"{old_key}.json").write_text(
        json.dumps({"idempotency_key": old_key, "status": "published"}),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(tmp_path) == []


def test_load_outbox_handoffs_skips_already_merged_branch_head(tmp_path: Path) -> None:
    repo, head = _repo_with_merged_codex_branch(tmp_path)
    outbox = repo / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "merged.json").write_text(
        json.dumps(
            _outbox_payload(
                repo="synaptent/aragora",
                idempotency_key="open-pr-codex-example-merged",
                local_evidence={
                    "branch": "codex/example",
                    "head_sha": head,
                    "base": "main",
                },
            )
        ),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(repo) == []


def test_load_outbox_handoffs_skips_already_merged_top_level_head(tmp_path: Path) -> None:
    repo, head = _repo_with_merged_codex_branch(tmp_path)
    outbox = repo / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "merged.json").write_text(
        json.dumps(
            _outbox_payload(
                repo="synaptent/aragora",
                idempotency_key="open-pr-codex-example-merged",
                branch="codex/example",
                head_sha=head,
                base="main",
            )
        ),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(repo) == []


def test_load_outbox_handoffs_skips_patch_equivalent_branch(tmp_path: Path) -> None:
    repo, head = _repo_with_patch_equivalent_codex_branch(tmp_path)
    outbox = repo / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "patch-equivalent.json").write_text(
        json.dumps(
            _outbox_payload(
                repo="synaptent/aragora",
                idempotency_key="open-pr-codex-example-patch-equivalent",
                local_evidence={
                    "branch": "codex/example",
                    "head_sha": head,
                    "base": "main",
                },
            )
        ),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(repo) == []


def test_load_outbox_handoffs_skips_merged_push_branch_request(tmp_path: Path) -> None:
    repo, head = _repo_with_merged_codex_branch(tmp_path)
    outbox = repo / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "merged-push.json").write_text(
        json.dumps(
            _outbox_payload(
                repo="synaptent/aragora",
                requested_action="push_branch_and_open_pr",
                idempotency_key="open-pr-codex-example-merged-push",
                local_evidence={
                    "branch": "codex/example",
                    "head_sha": head,
                    "base": "main",
                },
            )
        ),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(repo) == []


def test_load_outbox_handoffs_skips_merged_open_or_update_pull_request(
    tmp_path: Path,
) -> None:
    repo, head = _repo_with_merged_codex_branch(tmp_path)
    outbox = repo / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "merged-open-or-update.json").write_text(
        json.dumps(
            _outbox_payload(
                repo="synaptent/aragora",
                requested_action="open_or_update_pull_request",
                idempotency_key="open-pr-codex-example-open-or-update",
                local_evidence={
                    "branch": "codex/example",
                    "head_sha": head,
                    "base": "main",
                },
            )
        ),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(repo) == []


def test_load_outbox_handoffs_skips_merged_structured_pr_action_string(
    tmp_path: Path,
) -> None:
    repo, head = _repo_with_merged_codex_branch(tmp_path)
    outbox = repo / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "merged-structured-action.json").write_text(
        json.dumps(
            _outbox_payload(
                repo="synaptent/aragora",
                requested_action=str(
                    {
                        "type": "open_pull_request",
                        "branch": "codex/example",
                        "base": "main",
                    }
                ),
                idempotency_key="open-pr-codex-example-structured-action",
                local_evidence={
                    "branch": "codex/example",
                    "head_sha": head,
                    "base": "main",
                },
            )
        ),
        encoding="utf-8",
    )

    assert mod.load_outbox_handoffs(repo) == []


def test_load_outbox_handoffs_skips_non_github_and_expired(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    common = {
        "task": "Publish validated repair branch",
        "requested_action": "open_pr",
        "repo": str(tmp_path),
        "local_evidence": {},
        "validation": [],
        "created_at": "2026-04-24T16:00:00+00:00",
    }
    (outbox / "local-only.json").write_text(
        json.dumps(
            {
                **common,
                "requires_github": False,
                "idempotency_key": "local-only",
            }
        ),
        encoding="utf-8",
    )
    (outbox / "expired.json").write_text(
        json.dumps(
            {
                **common,
                "requires_github": True,
                "idempotency_key": "expired",
                "expires_at": "2026-04-24T15:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (outbox / "bad-expiration.json").write_text(
        json.dumps(
            _outbox_payload(
                repo=str(tmp_path),
                idempotency_key="bad-expiration",
                expires_at="not-a-date",
            )
        ),
        encoding="utf-8",
    )

    assert (
        mod.load_outbox_handoffs(
            tmp_path,
            now=datetime(2026, 4, 24, 16, 0, tzinfo=timezone.utc),
        )
        == []
    )


def test_load_outbox_handoffs_skips_incomplete_contract_payloads(tmp_path: Path) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)

    for field in mod.REQUIRED_OUTBOX_KEYS:
        payload = _outbox_payload(idempotency_key=f"missing-{field}")
        del payload[field]
        (outbox / f"missing-{field}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    assert mod.load_outbox_handoffs(tmp_path) == []


def test_load_handoffs_skips_expired_and_none_tasks(tmp_path: Path) -> None:
    expired = _memory(tmp_path, "expired", _handoff("Old task"))
    none = _memory(tmp_path, "none", _handoff("NONE").replace("Priority: MEDIUM", "Priority: NONE"))
    old_time = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(expired, (old_time, old_time))
    os.utime(none, (old_time, old_time))

    handoffs = mod.load_handoffs(tmp_path, now=datetime(2026, 4, 16, tzinfo=timezone.utc))

    assert handoffs == []


def test_load_handoffs_uses_latest_structured_block_per_memory(tmp_path: Path) -> None:
    _memory(
        tmp_path,
        "engineering-automation-2",
        _handoff("Old completed OpenAPI task") + "\n\n" + _handoff("Fresh decision task"),
    )

    handoffs = mod.load_handoffs(tmp_path, now=datetime(2026, 4, 16, tzinfo=timezone.utc))

    assert [handoff.task_title for handoff in handoffs] == ["Fresh decision task"]


def test_load_handoffs_uses_newest_timestamp_when_memory_is_out_of_order(
    tmp_path: Path,
) -> None:
    _memory(
        tmp_path,
        "founder-review",
        "\n\n".join(
            [
                "2026-04-16T08:14:42-05:00 - Founder review\n\n"
                + _handoff("Fresh modular dispatch task"),
                "2026-04-16T06:11:29-05:00 - Founder review\n\n"
                + _handoff("Older sqlite lock task"),
            ]
        ),
    )

    handoffs = mod.load_handoffs(tmp_path, now=datetime(2026, 4, 16, 14, 0, tzinfo=timezone.utc))

    assert [handoff.task_title for handoff in handoffs] == ["Fresh modular dispatch task"]


def test_load_handoffs_expires_from_block_timestamp_not_file_mtime(tmp_path: Path) -> None:
    memory = _memory(
        tmp_path,
        "founder-review",
        "2026-04-15T08:14:42-05:00 - Founder review\n\n" + _handoff("Expired task"),
    )
    text = memory.read_text(encoding="utf-8").replace("Expiration Hours: 72", "Expiration Hours: 1")
    memory.write_text(text, encoding="utf-8")
    fresh_time = datetime(2026, 4, 16, 13, 59, tzinfo=timezone.utc).timestamp()
    os.utime(memory, (fresh_time, fresh_time))

    handoffs = mod.load_handoffs(tmp_path, now=datetime(2026, 4, 16, 14, 20, tzinfo=timezone.utc))

    assert handoffs == []


def test_looks_duplicate_does_not_conflate_distinct_handlers() -> None:
    assert not mod._looks_duplicate(
        "Restore PromptEngineHandler OpenAPI and SDK contract",
        "Restore TaskQueueHandler OpenAPI and SDK contract",
    )


def test_decide_handoffs_marks_duplicate_issue(monkeypatch: Any, tmp_path: Path) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "memory.md"),
        task_title="Fix tmux readiness detection for named Claude lanes",
        priority="MEDIUM",
        body="body",
        labels={},
        expires_at=None,
    )
    issue_payload = json.dumps(
        [
            {
                "number": 5889,
                "title": "fix(tmux): detect readiness for neutral Claude lane names",
                "url": "https://github.com/synaptent/aragora/issues/5889",
                "state": "OPEN",
            }
        ]
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "list"] and "--label" in args:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        return subprocess.CompletedProcess(args, 0, issue_payload, "")

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = mod.decide_handoffs(
        [handoff],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_issues=12,
    )

    assert decisions == [
        PublishDecision(
            task_title=handoff.task_title,
            source_file=handoff.source_file,
            eligible=False,
            reason="existing_issue",
            existing_issue_url="https://github.com/synaptent/aragora/issues/5889",
        )
    ]


def test_run_uses_user_auth_for_issue_create(monkeypatch: Any, tmp_path: Path) -> None:
    recorded: dict[str, Any] = {}

    def fake_gh_run(
        args: list[str],
        *,
        timeout: float,
        prefer_app: bool,
        write_op: bool,
        env: dict[str, str],
        max_retries: int,
    ) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded["prefer_app"] = prefer_app
        recorded["write_op"] = write_op
        recorded["env"] = env
        recorded["max_retries"] = max_retries
        return subprocess.CompletedProcess(args=["gh", *args], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "gh_subprocess_run", fake_gh_run)

    result = mod._run(["gh", "issue", "create", "--title", "Example"], cwd=tmp_path)

    assert result.returncode == 0
    assert recorded["args"][:2] == ["issue", "create"]
    assert recorded["prefer_app"] is True
    assert recorded["write_op"] is True
    assert recorded["max_retries"] == 0


def test_decide_handoffs_marks_duplicate_pr(monkeypatch: Any, tmp_path: Path) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "memory.md"),
        task_title="Restore OpenAPI export coverage for decision analytics routes",
        priority="HIGH",
        body="body",
        labels={},
        expires_at=None,
    )
    pr_payload = json.dumps(
        [
            {
                "number": 5891,
                "title": "fix(openapi): restore decision analytics export coverage",
                "url": "https://github.com/synaptent/aragora/pull/5891",
                "state": "MERGED",
            }
        ]
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "list"] and "--label" in args:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        return subprocess.CompletedProcess(args, 0, pr_payload, "")

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = mod.decide_handoffs(
        [handoff],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_issues=12,
    )

    assert decisions == [
        PublishDecision(
            task_title=handoff.task_title,
            source_file=handoff.source_file,
            eligible=False,
            reason="existing_pr",
            existing_pr_url="https://github.com/synaptent/aragora/pull/5891",
        )
    ]


def test_decide_handoffs_routes_explicit_pr_followup_before_issue_cap(
    monkeypatch: Any, tmp_path: Path
) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "memory.md"),
        task_title="Prevent draft approval packets in PR #6288",
        priority="HIGH",
        body=(
            "Why Now: PR #6288 already carries the active review-queue implementation.\n"
            "Repo Evidence:\n- gh pr view 6288 --json number,title,headRefName,url,isDraft\n"
        ),
        labels={},
        expires_at=None,
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "list"] and "--label" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps([{"number": 1}]), "")
        if args[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "number": 6288,
                        "title": "feat(review): add read-only queue packet builder",
                        "url": "https://github.com/synaptent/aragora/pull/6288",
                        "state": "OPEN",
                    }
                ),
                "",
            )
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = mod.decide_handoffs(
        [handoff],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_issues=1,
    )

    assert decisions == [
        PublishDecision(
            task_title=handoff.task_title,
            source_file=handoff.source_file,
            eligible=False,
            reason="target_open_pr",
            existing_pr_url="https://github.com/synaptent/aragora/pull/6288",
        )
    ]


def test_decide_handoffs_routes_branch_handoff_to_open_pr_before_issue_cap(
    monkeypatch: Any, tmp_path: Path
) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "outbox.json"),
        task_title="Open PR for branch publisher receipt-dir CLI compatibility",
        priority="HIGH",
        body="body",
        labels={},
        expires_at=None,
        source_kind="outbox",
        branch="codex/branch-publisher-receipt-dir-compat",
        desired_head="abc1234",
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "list"] and "--label" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps([{"number": 1}]), "")
        if args[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "pr", "list"] and "--head" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    [
                        {
                            "number": 6741,
                            "title": "fix(automation): accept branch publisher receipt dir flag",
                            "url": "https://github.com/synaptent/aragora/pull/6741",
                            "state": "OPEN",
                            "headRefName": "codex/branch-publisher-receipt-dir-compat",
                            "headRefOid": "abc1234deadbeef",
                        }
                    ]
                ),
                "",
            )
        if args[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = mod.decide_handoffs(
        [handoff],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_issues=1,
    )

    assert decisions == [
        PublishDecision(
            task_title=handoff.task_title,
            source_file=handoff.source_file,
            eligible=False,
            reason="target_open_pr",
            existing_pr_url="https://github.com/synaptent/aragora/pull/6741",
        )
    ]


def test_decide_handoffs_keeps_branch_update_actionable_when_pr_head_is_stale(
    monkeypatch: Any, tmp_path: Path
) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "outbox.json"),
        task_title="Refresh PR for backlog audit handoff-protected patch-skip repair",
        priority="MEDIUM",
        body="body",
        labels={},
        expires_at=None,
        source_kind="outbox",
        branch="codex/audit-skip-handoff-protected-patch-checks-20260512",
        desired_head="5091193dfe68d40ead6ac775cd43c507360fa0fe",
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "list"] and "--label" in args:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "pr", "list"] and "--head" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    [
                        {
                            "number": 7105,
                            "title": "fix(audit): skip handoff-protected patch checks",
                            "url": "https://github.com/synaptent/aragora/pull/7105",
                            "state": "OPEN",
                            "headRefName": (
                                "codex/audit-skip-handoff-protected-patch-checks-20260512"
                            ),
                            "headRefOid": "e4d00097d6c44bcdd699973499a0935c0a92f808",
                        }
                    ]
                ),
                "",
            )
        if args[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = mod.decide_handoffs(
        [handoff],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_issues=12,
    )

    assert decisions == [
        PublishDecision(
            task_title=handoff.task_title,
            source_file=handoff.source_file,
            eligible=True,
            reason="eligible",
        )
    ]


def test_decide_handoffs_prefers_branch_pr_over_duplicate_issue(
    monkeypatch: Any, tmp_path: Path
) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "outbox.json"),
        task_title="Open PR for frontend E2E scope coverage of test workflow changes",
        priority="HIGH",
        body="body",
        labels={},
        expires_at=None,
        source_kind="outbox",
        branch="codex/frontend-e2e-test-workflow-scope",
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "list"] and "--label" in args:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "pr", "list"] and "--head" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    [
                        {
                            "number": 7024,
                            "title": "fix(ci): scope test workflow changes to frontend e2e",
                            "url": "https://github.com/synaptent/aragora/pull/7024",
                            "state": "OPEN",
                            "headRefName": "codex/frontend-e2e-test-workflow-scope",
                        }
                    ]
                ),
                "",
            )
        if args[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    [
                        {
                            "number": 6495,
                            "title": (
                                "Open PR for frontend E2E scope coverage of test workflow changes"
                            ),
                            "url": "https://github.com/synaptent/aragora/issues/6495",
                            "state": "OPEN",
                        }
                    ]
                ),
                "",
            )
        if args[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = mod.decide_handoffs(
        [handoff],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_issues=12,
    )

    assert decisions == [
        PublishDecision(
            task_title=handoff.task_title,
            source_file=handoff.source_file,
            eligible=False,
            reason="target_open_pr",
            existing_pr_url="https://github.com/synaptent/aragora/pull/7024",
        )
    ]


def test_decide_handoffs_respects_open_issue_cap(monkeypatch: Any, tmp_path: Path) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "memory.md"),
        task_title="Restore OpenAPI export coverage for decision analytics routes",
        priority="HIGH",
        body="body",
        labels={},
        expires_at=None,
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if "--label" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps([{"number": 1}]), "")
        return subprocess.CompletedProcess(args, 0, "[]", "")

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = mod.decide_handoffs(
        [handoff],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_issues=1,
    )

    assert decisions[0].eligible is False
    assert decisions[0].reason == "open_issue_cap"


def test_summarize_decisions_counts_eligibility_and_reasons(tmp_path: Path) -> None:
    decisions = [
        PublishDecision(
            task_title="Publish ready handoff",
            source_file=str(tmp_path / "ready.json"),
            eligible=True,
            reason="eligible",
        ),
        PublishDecision(
            task_title="Existing issue handoff",
            source_file=str(tmp_path / "existing.json"),
            eligible=False,
            reason="existing_issue",
        ),
        PublishDecision(
            task_title="Second existing issue handoff",
            source_file=str(tmp_path / "existing-2.json"),
            eligible=False,
            reason="existing_issue",
        ),
    ]

    assert mod.summarize_decisions(decisions) == {
        "total": 3,
        "eligible_count": 1,
        "ineligible_count": 2,
        "reason_counts": {
            "eligible": 1,
            "existing_issue": 2,
        },
    }


def test_referenced_pr_numbers_deduplicates_multiple_mentions(tmp_path: Path) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "memory.md"),
        task_title="Amend PR #6288 packet recommendation logic",
        priority="HIGH",
        body="Repo Evidence:\n- pull request #6288 still marks drafts as approve_candidate.\n",
        labels={},
        expires_at=None,
    )

    assert mod._referenced_pr_numbers(handoff) == [6288]


def test_referenced_pr_numbers_extracts_review_branch_slug(tmp_path: Path) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "outbox.json"),
        task_title="Open or update PR for AGT-04 markets predict CLI",
        priority="HIGH",
        body=(
            "Branch: codex/review-pr6808\n"
            "Worktree: /private/tmp/aragora-pr6808\n"
            "Compare: https://github.com/synaptent/aragora/compare/main...codex/review-pr6808\n"
        ),
        labels={},
        expires_at=None,
    )

    assert mod._referenced_pr_numbers(handoff) == [6808]


def test_decide_handoffs_skips_merged_referenced_pr_slug(monkeypatch: Any, tmp_path: Path) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "outbox.json"),
        task_title="Open or update PR for AGT-04 markets predict CLI",
        priority="HIGH",
        body="Requested branch: codex/review-pr6808\n",
        labels={},
        expires_at=None,
        source_kind="outbox",
        branch="codex/review-pr6808",
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "list"] and "--label" in args:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(args, 0, "[]", "")
        if args[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "number": 6808,
                        "title": "[AGT-04] aragora markets predict",
                        "url": "https://github.com/synaptent/aragora/pull/6808",
                        "state": "MERGED",
                    }
                ),
                "",
            )
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = mod.decide_handoffs(
        [handoff],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        max_open_issues=12,
    )

    assert decisions == [
        PublishDecision(
            task_title=handoff.task_title,
            source_file=handoff.source_file,
            eligible=False,
            reason="existing_pr",
            existing_pr_url="https://github.com/synaptent/aragora/pull/6808",
        )
    ]


def test_publish_handoffs_creates_issue_with_labels(monkeypatch: Any, tmp_path: Path) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "memory.md"),
        task_title="Restore OpenAPI export coverage for decision analytics routes",
        priority="HIGH",
        body="body",
        labels={},
        expires_at=None,
    )
    created: list[list[str]] = []

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        created.append(args)
        return subprocess.CompletedProcess(
            args, 0, "https://github.com/synaptent/aragora/issues/5890\n", ""
        )

    monkeypatch.setattr(mod, "_run", fake_run)

    decisions = [
        PublishDecision(
            task_title=handoff.task_title,
            source_file=handoff.source_file,
            eligible=True,
            reason="eligible",
        )
    ]
    published = mod.publish_handoffs(
        [handoff],
        decisions,
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready", "autonomous"],
        limit=1,
    )

    assert published[0].reason == "published"
    assert published[0].created_issue_url == "https://github.com/synaptent/aragora/issues/5890"
    assert created[0][:3] == ["gh", "issue", "create"]
    assert created[0].count("--label") == 2
    assert created[1] == [
        "gh",
        "issue",
        "edit",
        "5890",
        "--repo",
        "synaptent/aragora",
        "--add-label",
        "boss-ready,autonomous",
    ]


def test_publish_handoffs_writes_outbox_receipt(monkeypatch: Any, tmp_path: Path) -> None:
    source = tmp_path / ".aragora" / "automation-outbox" / "example.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    handoff = Handoff(
        source_file=str(source),
        task_title="Publish validated repair branch",
        priority="MEDIUM",
        body="body",
        labels={},
        expires_at=None,
        idempotency_key="open-pr-codex-example-abc123",
        source_kind="outbox",
    )

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "create"]:
            return subprocess.CompletedProcess(
                args, 0, "https://github.com/synaptent/aragora/issues/7000\n", ""
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(mod, "_run", fake_run)

    published = mod.publish_handoffs(
        [handoff],
        [
            PublishDecision(
                task_title=handoff.task_title,
                source_file=handoff.source_file,
                eligible=True,
                reason="eligible",
            )
        ],
        repo_root=tmp_path,
        repo="synaptent/aragora",
        labels=["boss-ready"],
        limit=1,
        receipt_dir=tmp_path / ".aragora" / "automation-receipts",
    )

    receipt_path = (
        tmp_path / ".aragora" / "automation-receipts" / "open-pr-codex-example-abc123.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert published[0].reason == "published"
    assert receipt["status"] == "published"
    assert receipt["created_issue_url"] == "https://github.com/synaptent/aragora/issues/7000"


def test_main_preview_does_not_write_outbox_receipt(
    monkeypatch: Any, tmp_path: Path, capsys
) -> None:
    receipts = tmp_path / ".aragora" / "automation-receipts"
    handoff = Handoff(
        source_file=str(tmp_path / ".aragora" / "automation-outbox" / "example.json"),
        task_title="Publish validated repair branch",
        priority="MEDIUM",
        body="body",
        labels={},
        expires_at=None,
        idempotency_key="open-pr-codex-example-abc123",
        source_kind="outbox",
    )
    monkeypatch.setattr(mod, "_repo_root", lambda path: tmp_path)
    monkeypatch.setattr(mod, "load_handoffs", lambda codex_home, automation_ids=None: [])
    monkeypatch.setattr(
        mod,
        "load_outbox_handoffs",
        lambda repo_root, outbox_dir=None, receipt_dir=None: [handoff],
    )
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=True,
            auth_ok=True,
            api_ok=True,
            mode="ok",
            error=None,
            repo=str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        mod,
        "decide_handoffs",
        lambda *args, **kwargs: [
            PublishDecision(
                task_title=handoff.task_title,
                source_file=handoff.source_file,
                eligible=False,
                reason="existing_issue",
                existing_issue_url="https://github.com/synaptent/aragora/issues/1",
            )
        ],
    )

    exit_code = mod.main(
        [
            "--repo",
            str(tmp_path),
            "--codex-home",
            str(tmp_path),
            "--receipt-dir",
            str(receipts),
            "--json",
        ]
    )

    assert exit_code == 0
    assert not receipts.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["decisions"][0]["reason"] == "existing_issue"
    assert payload["decision_summary"] == {
        "total": 1,
        "eligible_count": 0,
        "ineligible_count": 1,
        "reason_counts": {
            "existing_issue": 1,
        },
    }


def test_main_accepts_explicit_dry_run_alias(monkeypatch: Any, tmp_path: Path, capsys) -> None:
    receipts = tmp_path / ".aragora" / "automation-receipts"
    handoff = Handoff(
        source_file=str(tmp_path / ".aragora" / "automation-outbox" / "example.json"),
        task_title="Publish validated repair branch",
        priority="MEDIUM",
        body="body",
        labels={},
        expires_at=None,
        idempotency_key="open-pr-codex-example-abc123",
        source_kind="outbox",
    )
    monkeypatch.setattr(mod, "_repo_root", lambda path: tmp_path)
    monkeypatch.setattr(mod, "load_handoffs", lambda codex_home, automation_ids=None: [])
    monkeypatch.setattr(
        mod,
        "load_outbox_handoffs",
        lambda repo_root, outbox_dir=None, receipt_dir=None: [handoff],
    )
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=True,
            auth_ok=True,
            api_ok=True,
            mode="ok",
            error=None,
            repo=str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        mod,
        "decide_handoffs",
        lambda *args, **kwargs: [
            PublishDecision(
                task_title=handoff.task_title,
                source_file=handoff.source_file,
                eligible=False,
                reason="existing_issue",
                existing_issue_url="https://github.com/synaptent/aragora/issues/1",
            )
        ],
    )

    exit_code = mod.main(
        [
            "--repo",
            str(tmp_path),
            "--codex-home",
            str(tmp_path),
            "--receipt-dir",
            str(receipts),
            "--json",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not receipts.exists()
    assert '"reason": "existing_issue"' in capsys.readouterr().out


def test_main_rejects_apply_with_dry_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--repo", str(tmp_path), "--apply", "--dry-run"])

    assert excinfo.value.code == 2


def test_main_derives_outbox_dirs_from_explicit_aragora_state_root(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    state_root = tmp_path / ".aragora"
    captured: dict[str, Any] = {}
    monkeypatch.setattr(mod, "_repo_root", lambda _path: tmp_path / "worktree")
    monkeypatch.setattr(mod, "load_handoffs", lambda codex_home, automation_ids=None: [])

    def fake_load_outbox_handoffs(
        repo_root: Path,
        outbox_dir: Path | None = None,
        receipt_dir: Path | None = None,
    ) -> list[Handoff]:
        captured["repo_root"] = repo_root
        captured["outbox_dir"] = outbox_dir
        captured["receipt_dir"] = receipt_dir
        return []

    monkeypatch.setattr(mod, "load_outbox_handoffs", fake_load_outbox_handoffs)
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(repo_root),
        ),
    )

    exit_code = mod.main(
        [
            "--repo",
            str(tmp_path),
            "--codex-home",
            str(tmp_path),
            "--state-root",
            str(state_root),
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured["outbox_dir"] == (state_root / "automation-outbox").resolve()
    assert captured["receipt_dir"] == (state_root / "automation-receipts").resolve()
    assert str((state_root / "automation-outbox").resolve()) in capsys.readouterr().out


def test_main_treats_empty_handoff_queue_as_noop_when_github_unavailable(
    monkeypatch: Any, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(mod, "_repo_root", lambda path: tmp_path)
    monkeypatch.setattr(mod, "load_handoffs", lambda codex_home, automation_ids=None: [])
    monkeypatch.setattr(mod, "load_outbox_handoffs", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="error connecting to api.github.com",
            repo=str(tmp_path),
        ),
    )

    exit_code = mod.main(["--repo", str(tmp_path), "--codex-home", str(tmp_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["handoff_count"] == 0
    assert payload["decisions"] == []
    assert payload["github_health"]["mode"] == "connectivity_failed"


def test_main_reports_github_health_when_unavailable(
    monkeypatch: Any, tmp_path: Path, capsys
) -> None:
    handoff = Handoff(
        source_file=str(tmp_path / "memory.md"),
        task_title="Fix tmux readiness detection for named Claude lanes",
        priority="MEDIUM",
        body="body",
        labels={},
        expires_at=None,
    )
    monkeypatch.setattr(mod, "_repo_root", lambda path: tmp_path)
    monkeypatch.setattr(mod, "load_handoffs", lambda codex_home, automation_ids=None: [handoff])
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=False,
            auth_ok=True,
            api_ok=False,
            mode="connectivity_failed",
            error="error connecting to api.github.com",
            repo=str(tmp_path),
        ),
    )

    exit_code = mod.main(["--repo", str(tmp_path), "--codex-home", str(tmp_path), "--json"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["github_health"]["mode"] == "connectivity_failed"
    assert payload["decisions"][0]["reason"] == "github_unavailable"
    assert payload["decision_summary"] == {
        "total": 1,
        "eligible_count": 0,
        "ineligible_count": 1,
        "reason_counts": {
            "github_unavailable": 1,
        },
    }


def test_main_limits_github_unavailable_decision_preview(
    monkeypatch: Any, tmp_path: Path, capsys
) -> None:
    handoffs = [
        Handoff(
            source_file=str(tmp_path / "first.md"),
            task_title="First offline handoff",
            priority="MEDIUM",
            body="body",
            labels={},
            expires_at=None,
        ),
        Handoff(
            source_file=str(tmp_path / "second.md"),
            task_title="Second offline handoff",
            priority="MEDIUM",
            body="body",
            labels={},
            expires_at=None,
        ),
    ]
    monkeypatch.setattr(mod, "_repo_root", lambda path: tmp_path)
    monkeypatch.setattr(mod, "load_handoffs", lambda codex_home, automation_ids=None: handoffs)
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root: GitHubCLIHealth(
            ready=False,
            auth_ok=True,
            api_ok=False,
            mode="connectivity_failed",
            error="error connecting to api.github.com",
            repo=str(tmp_path),
        ),
    )

    exit_code = mod.main(
        ["--repo", str(tmp_path), "--codex-home", str(tmp_path), "--json", "--limit", "1"]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["handoff_count"] == 2
    assert len(payload["decisions"]) == 1
    assert payload["decisions"][0]["task_title"] == "First offline handoff"
    assert payload["decisions"][0]["reason"] == "github_unavailable"
    assert payload["decision_summary"] == {
        "total": 1,
        "eligible_count": 0,
        "ineligible_count": 1,
        "reason_counts": {
            "github_unavailable": 1,
        },
    }


def test_create_issue_truncates_oversized_body(monkeypatch: Any, tmp_path: Path) -> None:
    bodies: list[str] = []

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "create"]:
            bodies.append(args[args.index("--body") + 1])
            return subprocess.CompletedProcess(
                args, 0, "https://github.com/synaptent/aragora/issues/6000\n", ""
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(mod, "_run", fake_run)

    mod._create_issue(
        tmp_path,
        "synaptent/aragora",
        Handoff(
            source_file=str(tmp_path / "memory.md"),
            task_title="Long issue body",
            priority="HIGH",
            body="x" * (mod.MAX_ISSUE_BODY_CHARS + 1000),
            labels={},
            expires_at=None,
        ),
        labels=["boss-ready"],
    )

    assert len(bodies[0]) <= mod.MAX_ISSUE_BODY_CHARS
    assert "truncated this issue body" in bodies[0]


def test_create_issue_preserves_boundary_sized_body(monkeypatch: Any, tmp_path: Path) -> None:
    bodies: list[str] = []

    def fake_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "issue", "create"]:
            bodies.append(args[args.index("--body") + 1])
            return subprocess.CompletedProcess(
                args, 0, "https://github.com/synaptent/aragora/issues/6001\n", ""
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(mod, "_run", fake_run)
    body = "x" * mod.MAX_ISSUE_BODY_CHARS

    mod._create_issue(
        tmp_path,
        "synaptent/aragora",
        Handoff(
            source_file=str(tmp_path / "memory.md"),
            task_title="Boundary sized issue body",
            priority="HIGH",
            body=body,
            labels={},
            expires_at=None,
        ),
        labels=["boss-ready"],
    )

    assert bodies == [body]
