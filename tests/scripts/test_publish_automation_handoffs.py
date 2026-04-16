from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import scripts.publish_automation_handoffs as mod
from scripts.publish_automation_handoffs import Handoff, PublishDecision


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


def test_load_handoffs_parses_structured_memory(tmp_path: Path) -> None:
    _memory(tmp_path, "founder-review", _handoff())

    handoffs = mod.load_handoffs(tmp_path, now=datetime(2026, 4, 16, tzinfo=timezone.utc))

    assert len(handoffs) == 1
    assert handoffs[0].task_title == "Fix tmux readiness detection for named Claude lanes"
    assert handoffs[0].priority == "MEDIUM"
    assert "Acceptance Criteria:" in handoffs[0].body
    assert "Published from automation memory" in handoffs[0].body


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
