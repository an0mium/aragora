from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    assert '"reason": "existing_issue"' in capsys.readouterr().out


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
    out = capsys.readouterr().out
    assert '"mode": "connectivity_failed"' in out
    assert '"reason": "github_unavailable"' in out


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
