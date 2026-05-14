from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import scripts.reconcile_automation_outbox as mod


def _unhealthy_github() -> SimpleNamespace:
    return SimpleNamespace(
        ready=False,
        mode="connectivity_failed",
        error="error connecting to api.github.com",
    )


def _ready_github() -> SimpleNamespace:
    return SimpleNamespace(ready=True, mode="ready", error="")


def _write_outbox_handoff(
    outbox_dir: Path,
    *,
    branch: str,
    key: str,
    local_evidence: dict[str, Any] | None = None,
) -> Path:
    outbox_dir.mkdir(parents=True, exist_ok=True)
    path = outbox_dir / f"{key}.json"
    payload = {
        "task": f"Publish {branch}",
        "requires_github": True,
        "requested_action": {"type": "open_pr", "branch": branch},
        "repo": "synaptent/aragora",
        "idempotency_key": key,
    }
    if local_evidence is not None:
        payload["local_evidence"] = local_evidence
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_terminal_receipt_keys_falls_back_to_receipt_filename(tmp_path: Path) -> None:
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    key = "open-pr-codex-example-abc123"
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"status": "published"}),
        encoding="utf-8",
    )

    assert mod._terminal_receipt_keys(receipt_dir) == {key}


def test_dry_run_does_not_write_report_by_default(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    rc = mod.main(["--repo", str(tmp_path), "--base", "origin/main"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "report: not written in dry-run" in out
    assert not (tmp_path / ".aragora" / "cleanup-state").exists()


def test_explicit_dry_run_flag_keeps_read_only_default(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    rc = mod.main(["--repo", str(tmp_path), "--dry-run"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "mode: DRY-RUN" in out
    assert "DRY-RUN" in out
    assert not (tmp_path / ".aragora" / "cleanup-state").exists()


def test_json_output_reports_reconciliation_without_human_preamble(
    tmp_path: Path,
    capsys: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    receipt_dir = tmp_path / ".aragora" / "automation-receipts"
    key = "open-pr-codex-json-output-abc123"
    _write_outbox_handoff(outbox_dir, branch="codex/json-output", key=key)
    receipt_dir.mkdir(parents=True)
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )

    rc = mod.main(["--repo", str(tmp_path), "--json"])

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert not out.startswith("outbox_dir:")
    assert payload["dry_run"] is True
    assert payload["applied"] is False
    assert payload["report"] is None
    assert payload["outbox_count"] == 1
    assert payload["terminal_receipt_count"] == 1
    assert payload["counts"]["satisfied_by_existing_receipt"] == 1
    assert payload["archived"] == 1
    assert payload["kept"] == 0
    assert payload["actions"][0]["branch"] == "codex/json-output"


def test_state_root_can_point_at_direct_dot_aragora(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = tmp_path / "disposable-worktree"
    repo.mkdir()
    state_root = tmp_path / "shared-checkout" / ".aragora"
    outbox_dir = state_root / "automation-outbox"
    receipt_dir = state_root / "automation-receipts"
    key = "open-pr-codex-shared-state-abc123"
    _write_outbox_handoff(outbox_dir, branch="codex/shared-state", key=key)
    receipt_dir.mkdir(parents=True)
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )

    rc = mod.main(["--repo", str(repo), "--state-root", str(state_root), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["state_root"] == str(state_root.resolve())
    assert payload["outbox_dir"] == str(outbox_dir.resolve())
    assert payload["receipt_dir"] == str(receipt_dir.resolve())
    assert payload["archive_dir"] == str((state_root / "automation-outbox-archive").resolve())
    assert payload["counts"]["satisfied_by_existing_receipt"] == 1


def test_apply_uses_explicit_shared_state_dirs(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = tmp_path / "disposable-worktree"
    repo.mkdir()
    shared_state = tmp_path / "shared-state"
    outbox_dir = shared_state / "outbox"
    receipt_dir = shared_state / "receipts"
    archive_dir = shared_state / "archive"
    key = "open-pr-codex-explicit-state-abc123"
    handoff = _write_outbox_handoff(outbox_dir, branch="codex/explicit-state", key=key)
    receipt_dir.mkdir(parents=True)
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--repo",
            str(repo),
            "--outbox-dir",
            str(outbox_dir),
            "--receipt-dir",
            str(receipt_dir),
            "--archive-dir",
            str(archive_dir),
            "--apply",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["outbox_dir"] == str(outbox_dir.resolve())
    assert payload["receipt_dir"] == str(receipt_dir.resolve())
    assert payload["archive_dir"] == str(archive_dir.resolve())
    assert payload["archived"] == 1
    assert handoff.exists() is False
    assert (archive_dir / handoff.name).exists()


def test_explicit_outbox_dir_defaults_archive_beside_outbox(
    tmp_path: Path,
    capsys: Any,
) -> None:
    repo = tmp_path / "disposable-worktree"
    repo.mkdir()
    shared_state = tmp_path / "shared-state"
    outbox_dir = shared_state / "automation-outbox"
    receipt_dir = shared_state / "automation-receipts"
    key = "open-pr-codex-explicit-outbox-abc123"
    _write_outbox_handoff(outbox_dir, branch="codex/explicit-outbox", key=key)
    receipt_dir.mkdir(parents=True)
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--repo",
            str(repo),
            "--outbox-dir",
            str(outbox_dir),
            "--receipt-dir",
            str(receipt_dir),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["archive_dir"] == str((shared_state / "automation-outbox-archive").resolve())


def test_apply_archives_outbox_handoff_superseded_by_active_handoff(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    old_key = "open-pr-codex-example-oldaaaa"
    new_key = "open-pr-codex-example-restack-newbbbb"
    old_path = _write_outbox_handoff(
        outbox_dir,
        branch="codex/example",
        key=old_key,
        local_evidence={
            "branch": "codex/example",
            "head_sha": "oldaaaa1111",
        },
    )
    new_path = _write_outbox_handoff(
        outbox_dir,
        branch="codex/example-restack",
        key=new_key,
        local_evidence={
            "branch": "codex/example-restack",
            "head_sha": "newbbbb2222",
            "supersedes_branch": "codex/example",
            "supersedes_head_sha": "oldaaaa1111",
        },
    )

    monkeypatch.setattr(mod, "check_github_cli_health", lambda _root: _ready_github())
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    def fake_run_git(args: list[str], *_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess:
        if args[:2] == ["rev-parse", "--verify"]:
            return subprocess.CompletedProcess(args=["git"], returncode=0, stdout="head\n")
        if args and args[0] == "merge-base":
            return subprocess.CompletedProcess(args=["git"], returncode=1, stdout="")
        if args and args[0] == "cherry":
            return subprocess.CompletedProcess(args=["git"], returncode=0, stdout="+ newbbbb\n")
        return subprocess.CompletedProcess(args=["git"], returncode=0, stdout="")

    monkeypatch.setattr(mod, "run_git", fake_run_git)

    assert mod.main(["--repo", str(tmp_path), "--apply", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["satisfied_by_superseded_handoff"] == 1
    assert payload["archived"] == 1
    assert old_path.exists() is False
    assert new_path.exists() is True
    archived = tmp_path / ".aragora" / "automation-outbox-archive" / old_path.name
    receipt = tmp_path / ".aragora" / "automation-receipts" / f"{old_key}.json"
    assert archived.exists()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["status"] == "already_satisfied"
    assert receipt_payload["synthetic_reason"] == f"superseded by active handoff {new_key}"


def test_dry_run_can_write_report_when_requested(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    rc = mod.main(["--repo", str(tmp_path), "--base", "origin/main", "--write-report"])

    out = capsys.readouterr().out
    reports = list((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    assert rc == 0
    assert "report:" in out
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["applied"] is False


def test_dry_run_out_writes_explicit_report_path(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})
    report_path = tmp_path / "artifacts" / "reconcile-report.json"

    rc = mod.main(["--repo", str(tmp_path), "--json", "--out", str(report_path)])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["report"] == str(report_path)
    assert report_path.exists()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["applied"] is False
    assert not (tmp_path / ".aragora" / "cleanup-state").exists()


def test_branch_from_payload_tolerates_list_local_evidence() -> None:
    payload = {
        "branch": "codex/openrouter-kimi-fallback-haiku",
        "local_evidence": [
            "older handoffs sometimes stored local evidence as bullet text",
        ],
    }

    assert mod._branch_from_payload(payload) == "codex/openrouter-kimi-fallback-haiku"


def test_branch_from_payload_uses_list_local_evidence_mapping() -> None:
    payload = {
        "requested_action": "open_pr",
        "local_evidence": [
            "older handoffs sometimes stored local evidence as bullet text",
            {"branch": "codex/list-evidence"},
        ],
    }

    assert mod._branch_from_payload(payload) == "codex/list-evidence"


def test_branch_from_payload_prefers_structured_local_evidence() -> None:
    payload = {
        "branch": "codex/stale-top-level",
        "local_evidence": {"branch": "codex/structured"},
    }

    assert mod._branch_from_payload(payload) == "codex/structured"


def test_reconcile_existing_receipt_uses_structured_requested_action_branch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    receipt_dir = tmp_path / ".aragora" / "automation-receipts"
    outbox_dir.mkdir(parents=True)
    receipt_dir.mkdir(parents=True)
    key = "open-pr-codex-structured-action-abc123"
    (outbox_dir / "structured-action.json").write_text(
        json.dumps(
            {
                "task": "Publish structured-action branch",
                "requires_github": True,
                "requested_action": {
                    "type": "open_pr",
                    "branch": "codex/structured-action",
                },
                "repo": "synaptent/aragora",
                "local_evidence": {},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": key,
                "created_at": "2026-04-27T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    assert mod.main(["--repo", str(tmp_path), "--write-report"]) == 0

    reports = sorted((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["counts"]["satisfied_by_existing_receipt"] == 1
    assert payload["counts"]["skipped_unparseable"] == 0
    assert payload["actions"][0]["branch"] == "codex/structured-action"


def test_reconcile_keeps_target_pr_receipt_when_desired_head_not_published(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    receipt_dir = tmp_path / ".aragora" / "automation-receipts"
    key = "open-pr-codex-target-pr-refresh-newhead"
    desired_head = "abcdef1234567890abcdef1234567890abcdef12"
    stale_remote_head = "1111111234567890abcdef1234567890abcdef12"
    handoff = _write_outbox_handoff(
        outbox_dir,
        branch="codex/target-pr-refresh",
        key=key,
        local_evidence={
            "branch": "codex/target-pr-refresh",
            "desired_head_sha": desired_head,
        },
    )
    receipt_dir.mkdir(parents=True)
    (receipt_dir / f"{key}.json").write_text(
        json.dumps(
            {
                "idempotency_key": key,
                "status": "already_satisfied",
                "reason": "target_open_pr",
                "existing_pr_url": "https://github.com/synaptent/aragora/pull/7105",
            }
        ),
        encoding="utf-8",
    )

    def fake_run_git(
        args: list[str],
        _root: Path,
        *,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        if args == ["rev-parse", "--verify", "refs/remotes/origin/codex/target-pr-refresh"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=stale_remote_head)
        if args == ["rev-parse", "--verify", "codex/target-pr-refresh"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=desired_head)
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(mod, "run_git", fake_run_git)
    monkeypatch.setattr(
        mod,
        "open_pr_heads",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("open PR fetch should not run for mismatched target PR receipts")
        ),
    )

    assert mod.main(["--repo", str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["blocked_receipt_pr_head_mismatch"] == 1
    assert payload["counts"]["satisfied_by_existing_receipt"] == 0
    assert payload["counts"]["still_protecting_active_work"] == 1
    assert payload["actions"][0]["decision"] == "keep"
    assert "not desired head" in payload["actions"][0]["reason"]
    assert handoff.exists()
    assert not (tmp_path / ".aragora" / "automation-outbox-archive").exists()


def test_apply_preserves_missing_branch_when_open_pr_state_unavailable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    receipt_dir = tmp_path / ".aragora" / "automation-receipts"
    key = "open-pr-codex-missing-abc123"
    handoff = _write_outbox_handoff(outbox_dir, branch="codex/missing", key=key)
    receipt_dir.mkdir(parents=True)

    monkeypatch.setattr(mod, "check_github_cli_health", lambda _root: _unhealthy_github())
    monkeypatch.setattr(
        mod,
        "open_pr_heads",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("open PR fetch should be skipped when GitHub is unhealthy")
        ),
    )
    monkeypatch.setattr(
        mod,
        "run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="missing ref"
        ),
    )

    assert mod.main(["--repo", str(tmp_path), "--apply"]) == 0

    reports = sorted((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["counts"]["blocked_missing_branch_open_pr_unknown"] == 1
    assert payload["counts"]["missing_branch"] == 0
    assert payload["actions"][0]["decision"] == "keep"
    assert "open PR state is unavailable" in payload["actions"][0]["reason"]
    assert handoff.exists()
    assert not (receipt_dir / f"{key}.json").exists()
    assert not list((tmp_path / ".aragora" / "automation-outbox-archive").glob("*.json"))


def test_missing_branch_archives_when_open_pr_state_is_available(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    key = "open-pr-codex-missing-abc123"
    _write_outbox_handoff(outbox_dir, branch="codex/missing", key=key)

    monkeypatch.setattr(mod, "check_github_cli_health", lambda _root: _ready_github())
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})
    monkeypatch.setattr(
        mod,
        "run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="missing ref"
        ),
    )

    assert mod.main(["--repo", str(tmp_path), "--write-report"]) == 0

    reports = sorted((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["counts"]["missing_branch"] == 1
    assert payload["counts"]["blocked_missing_branch_open_pr_unknown"] == 0
    assert payload["actions"][0]["decision"] == "archive"
    assert payload["actions"][0]["reason"] == "branch no longer exists"


def test_landed_branch_archives_without_github_lookup(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    key = "open-pr-codex-landed-abc123"
    _write_outbox_handoff(outbox_dir, branch="codex/landed", key=key)

    def fake_run_git(
        args: list[str],
        _root: Path,
        *,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["rev-parse", "--verify"]:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="abc123\n", stderr=""
            )
        if args[:2] == ["merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(mod, "run_git", fake_run_git)
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("GitHub should not be queried for locally landed work")
        ),
    )
    monkeypatch.setattr(
        mod,
        "open_pr_heads",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("open PR fetch should not run for locally landed work")
        ),
    )

    assert mod.main(["--repo", str(tmp_path), "--write-report"]) == 0

    reports = sorted((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["counts"]["satisfied_by_landed_on_main"] == 1
    assert payload["counts"]["still_protecting_active_work"] == 0
    assert payload["actions"][0]["decision"] == "archive"
    assert (
        payload["actions"][0]["reason"] == "branch work landed on main (merge or patch-equivalent)"
    )
