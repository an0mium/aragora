from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.swarm.credential_envelope import CredentialEnvelope
from aragora.swarm.mission import GateType, GateVerdict
from aragora.swarm import preflight as mod
from aragora.swarm.worker_contract import checksum_contract_payload
from aragora.swarm.worker_process import WorkerProcess


def _worker(*, branch: str, checksum: str | None = None) -> WorkerProcess:
    contract = {
        "runner_type": "codex-cli",
        "agent": "codex",
        "model": "gpt-5.4",
        "profile": "default",
        "permissions": {"allow_full_auto": True},
        "execution_mode": "autonomous",
        "git_auth_mode": "https",
        "gh_api_auth_mode": "none",
        "budget": {"max_wall_time_seconds": 900.0},
        "env_checksum": "env123",
        "mission_id": "mission-rs-worker-contract-preflight",
        "stage_id": "stage-dispatch-ready-preflight",
        "assertion_ids": ["RS-PREFLIGHT-ASSERT-1"],
        "evidence_expectations": ["worker_contract", "worker_contract_checksum", "receipt"],
        "mission_context_policy": {
            "role": "worker",
            "allowed_artifact_classes": ["mission_stage", "file_scope", "validation_command"],
            "max_source_count": 4,
            "max_chars": 12000,
            "freshness_ttl_seconds": 3600,
            "transcript_allowance": "none",
            "required_sources": ["scratch/preflight_worker_check.txt"],
            "forbidden_sources": ["raw_worker_transcript"],
        },
        "contract_version": "1",
    }
    return WorkerProcess(
        work_order_id="preflight-1",
        agent="codex",
        worktree_path="/tmp/preflight",
        branch=branch,
        commit_shas=["deadbeef"],
        worker_contract=contract,
        worker_contract_checksum=checksum or checksum_contract_payload(contract),
    )


def _envelope() -> CredentialEnvelope:
    return CredentialEnvelope.from_environment(
        {
            "ARAGORA_CLAUDE_PROFILE": "claude",
            "GITHUB_TOKEN": "token",
            "OPENAI_API_KEY": "key",
            "ARAGORA_PROVIDER": "openai",
            "PYTEST_AVAILABLE": "true",
            "RUFF_AVAILABLE": "true",
            "SSH_AUTH_SOCK": "/tmp/agent.sock",
        }
    )


def test_run_preflight_returns_structured_result(monkeypatch, tmp_path: Path) -> None:
    branch = "preflight/20260411-test"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    commands: list[list[str]] = []
    cleanup_commands: list[list[str]] = []

    async def fake_run_worker(**_: object) -> WorkerProcess:
        return _worker(branch=branch)

    def fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        commands.append(list(cmd))

    def fake_subprocess_run(cmd: list[str], **_: object) -> SimpleNamespace:
        cleanup_commands.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_branch_name", lambda: branch)
    monkeypatch.setattr(mod, "_run_worker", fake_run_worker)
    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    result = mod.run_preflight(
        repo_root=repo_root,
        agent="codex",
        base_ref="origin/main",
        skip_publication=False,
    )

    expected_worktree = repo_root / ".worktrees" / "preflight-preflight-20260411-test"
    assert commands[0] == [
        "git",
        "worktree",
        "add",
        "-b",
        branch,
        str(expected_worktree),
        "origin/main",
    ]
    assert commands[1] == ["git", "push", "origin", "HEAD"]
    assert "--base" in commands[2]
    assert commands[2][commands[2].index("--base") + 1] == "origin/main"
    assert commands[3][:4] == ["gh", "pr", "close", "--repo"]
    assert result.published is True
    assert result.pull_request_created is True
    assert result.pull_request_closed is True
    assert result.cleanup_worktree_removed is True
    assert result.cleanup_branch_removed is True
    assert result.dispatch_gate["gate_type"] == GateType.DISPATCH_READY.value
    assert result.dispatch_gate["verdict"] == GateVerdict.PASS.value
    assert result.worker["worker_contract_checksum"] == checksum_contract_payload(
        result.worker["worker_contract"]
    )
    assert cleanup_commands[0] == ["git", "worktree", "remove", "--force", str(expected_worktree)]
    assert cleanup_commands[1] == ["git", "branch", "-D", branch]


def test_run_preflight_fails_closed_on_contract_checksum_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    branch = "preflight/20260411-bad-checksum"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    commands: list[list[str]] = []
    cleanup_commands: list[list[str]] = []

    async def fake_run_worker(**_: object) -> WorkerProcess:
        return _worker(branch=branch, checksum="bad-checksum")

    def fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        commands.append(list(cmd))

    def fake_subprocess_run(cmd: list[str], **_: object) -> SimpleNamespace:
        cleanup_commands.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_branch_name", lambda: branch)
    monkeypatch.setattr(mod, "_run_worker", fake_run_worker)
    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    with pytest.raises(RuntimeError, match="checksum"):
        mod.run_preflight(
            repo_root=repo_root,
            agent="codex",
            base_ref="main",
            skip_publication=False,
        )

    assert commands == [
        [
            "git",
            "worktree",
            "add",
            "-b",
            branch,
            str(repo_root / ".worktrees" / "preflight-preflight-20260411-bad-checksum"),
            "main",
        ]
    ]
    assert cleanup_commands[0][:4] == ["git", "worktree", "remove", "--force"]
    assert cleanup_commands[1] == ["git", "branch", "-D", branch]


def test_evaluate_preflight_dispatch_gate_blocks_missing_context_policy() -> None:
    worker = _worker(branch="preflight/20260411-context-gap")
    worker.worker_contract.pop("mission_context_policy", None)
    worker.worker_contract_checksum = checksum_contract_payload(worker.worker_contract)

    gate = mod.evaluate_preflight_dispatch_gate(worker)

    assert gate["gate_type"] == GateType.DISPATCH_READY.value
    assert gate["verdict"] == GateVerdict.BLOCKED.value
    assert "context_policy_unresolved" in gate["failure_classes"]


def _ok_run(cmd: list[str], **_: object) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_run_preflight_checks_success(monkeypatch, tmp_path: Path) -> None:
    env = {
        "ARAGORA_CLAUDE_PROFILE": "claude",
        "GITHUB_TOKEN": "token",
        "OPENAI_API_KEY": "key",
        "ARAGORA_PROVIDER": "openai",
        "PYTEST_AVAILABLE": "true",
        "RUFF_AVAILABLE": "true",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
    }
    envelope = CredentialEnvelope.from_environment(env)
    monkeypatch.setattr(mod.subprocess, "run", _ok_run)

    result = mod.run_preflight(envelope=envelope, repo_root=tmp_path)

    assert result.passed is True
    assert len(result.checks) == 6
    assert result.envelope is envelope
    assert result.duration_seconds >= 0.0


def test_run_preflight_checks_detects_git_dirty(monkeypatch, tmp_path: Path) -> None:
    env = {
        "ARAGORA_CLAUDE_PROFILE": "claude",
        "GITHUB_TOKEN": "token",
        "OPENAI_API_KEY": "key",
        "ARAGORA_PROVIDER": "openai",
        "PYTEST_AVAILABLE": "true",
        "RUFF_AVAILABLE": "true",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
    }
    envelope = CredentialEnvelope.from_environment(env)

    def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
        if cmd[:2] == ["git", "status"]:
            return SimpleNamespace(returncode=0, stdout=" M file.py\n", stderr="")
        return _ok_run(cmd)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = mod.run_preflight(envelope=envelope, repo_root=tmp_path)
    assert result.passed is False
    assert result.checks[0]["name"] == "git_status_clean"
    assert result.checks[0]["passed"] is False


def test_preflight_result_serializes_envelope(monkeypatch, tmp_path: Path) -> None:
    env = {
        "ARAGORA_CLAUDE_PROFILE": "claude",
        "GITHUB_TOKEN": "token",
        "OPENAI_API_KEY": "key",
        "ARAGORA_PROVIDER": "openai",
        "PYTEST_AVAILABLE": "true",
        "RUFF_AVAILABLE": "true",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
    }
    envelope = CredentialEnvelope.from_environment(env)
    monkeypatch.setattr(mod.subprocess, "run", _ok_run)

    result = mod.run_preflight(envelope=envelope, repo_root=tmp_path)
    payload = result.to_dict()
    assert payload["envelope"]["provider"]["provider_name"] == "openai"


def test_run_preflight_uses_contract_file_and_enforces_expected_contract(
    monkeypatch, tmp_path: Path
) -> None:
    branch = "preflight/20260412-contract"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    commands: list[list[str]] = []
    cleanup_commands: list[list[str]] = []
    contract_payload = _worker(branch=branch).worker_contract
    contract_path = tmp_path / "worker_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "worker_contract": contract_payload,
                "worker_contract_checksum": checksum_contract_payload(contract_payload),
            }
        ),
        encoding="utf-8",
    )

    async def fake_run_worker(**kwargs: object) -> WorkerProcess:
        assert kwargs["agent"] == "codex"
        return _worker(branch=branch)

    def fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        commands.append(list(cmd))

    def fake_subprocess_run(cmd: list[str], **_: object) -> SimpleNamespace:
        cleanup_commands.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_branch_name", lambda: branch)
    monkeypatch.setattr(mod, "_run_worker", fake_run_worker)
    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    result = mod.run_preflight(
        repo_root=repo_root,
        skip_publication=True,
        contract_path=contract_path,
    )

    assert result.passed is True
    assert result.agent == "codex"
    assert result.worker["worker_contract"] == contract_payload
    assert commands[0][:4] == ["git", "worktree", "add", "-b"]
    assert cleanup_commands[0][:4] == ["git", "worktree", "remove", "--force"]


def test_run_preflight_rejects_contract_file_checksum_mismatch(tmp_path: Path) -> None:
    contract_payload = _worker(branch="preflight/20260412-contract-bad").worker_contract
    contract_path = tmp_path / "worker_contract_bad.json"
    contract_path.write_text(
        json.dumps(
            {
                "worker_contract": contract_payload,
                "worker_contract_checksum": "bad-checksum",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="checksum"):
        mod.run_preflight(
            repo_root=tmp_path,
            contract_path=contract_path,
        )


def test_run_preflight_rejects_emitted_contract_drift(monkeypatch, tmp_path: Path) -> None:
    branch = "preflight/20260412-contract-drift"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    cleanup_commands: list[list[str]] = []
    contract_payload = _worker(branch=branch).worker_contract
    contract_path = tmp_path / "worker_contract_drift.json"
    contract_path.write_text(json.dumps(contract_payload), encoding="utf-8")

    async def fake_run_worker(**_: object) -> WorkerProcess:
        worker = _worker(branch=branch)
        worker.worker_contract["model"] = "tampered-model"
        worker.worker_contract_checksum = checksum_contract_payload(worker.worker_contract)
        return worker

    def fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        return None

    def fake_subprocess_run(cmd: list[str], **_: object) -> SimpleNamespace:
        cleanup_commands.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_branch_name", lambda: branch)
    monkeypatch.setattr(mod, "_run_worker", fake_run_worker)
    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    with pytest.raises(RuntimeError, match="drifted from the expected contract"):
        mod.run_preflight(
            repo_root=repo_root,
            skip_publication=True,
            contract_path=contract_path,
        )

    assert cleanup_commands[0][:4] == ["git", "worktree", "remove", "--force"]


def test_run_scratch_validation_receipt_persists_success(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 19, 45, 0, tzinfo=timezone.utc)
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(mod, "_receipt_token", lambda: "ab12cd34")

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(list(cmd))
        if cmd[:3] == ["git", "worktree", "add"]:
            Path(cmd[5]).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    receipt = mod.run_scratch_validation_receipt(repo_root=repo_root, envelope=envelope)

    assert receipt.check_type == "scratch"
    assert receipt.passed is True
    assert receipt.ttl_seconds == 86400
    assert receipt.artifacts["draft_pr_number"] is None
    assert receipt.artifacts["draft_pr_url"] == ""
    assert any(check["name"] == "git_commit" for check in receipt.checks)
    receipt_path = (
        repo_root / ".aragora" / "receipts" / "preflight" / f"scratch-{receipt.cache_key}.json"
    )
    assert receipt_path.exists()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["receipt_id"] == receipt.receipt_id
    assert commands[0][:3] == ["git", "worktree", "add"]


def test_run_remote_publish_validation_receipt_records_pr_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 19, 50, 0, tzinfo=timezone.utc)
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(mod, "_receipt_token", lambda: "ef56aa11")

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(list(cmd))
        if cmd[:3] == ["git", "worktree", "add"]:
            Path(cmd[5]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["gh", "pr", "create"]:
            return SimpleNamespace(
                returncode=0,
                stdout="https://github.com/synaptent/aragora/pull/5123\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    receipt = mod.run_remote_publish_validation_receipt(
        repo_root=repo_root,
        envelope=envelope,
    )

    assert receipt.check_type == "remote_publish"
    assert receipt.passed is True
    assert receipt.ttl_seconds == 3600
    assert receipt.artifacts["draft_pr_number"] == 5123
    assert receipt.artifacts["draft_pr_url"] == "https://github.com/synaptent/aragora/pull/5123"
    assert any(check["name"] == "gh_pr_capture" for check in receipt.checks)
    assert ["git", "push", "origin", "HEAD"] in commands
    assert any(cmd[:3] == ["gh", "pr", "close"] for cmd in commands)


def test_load_cached_preflight_receipt_reuses_fresh_successful_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 20, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "_utc_now", lambda: now)

    cache_key = mod._preflight_cache_key(repo_root, envelope, "scratch")
    receipt = mod.PreflightReceipt(
        schema_version=1,
        receipt_id="preflight-scratch-20260412T200000Z-cache1234",
        envelope_seal=envelope.preflight_cache_seal(),
        repo_root=str(repo_root.resolve()),
        check_type="scratch",
        started_at="2026-04-12T20:00:00Z",
        finished_at="2026-04-12T20:00:05Z",
        passed=True,
        checks=[{"name": "git_commit", "passed": True, "detail": "ok"}],
        cache_key=cache_key,
        ttl_seconds=86400,
        expires_at="2026-04-13T20:00:05Z",
        artifacts={"branch": "preflight/scratch/test", "worktree_path": "/tmp/worktree"},
    )
    mod._save_preflight_receipt(repo_root, receipt)

    loaded = mod._load_cached_preflight_receipt(repo_root, envelope, "scratch", now=now)

    assert loaded is not None
    assert loaded.receipt_id == receipt.receipt_id


def test_load_cached_preflight_receipt_misses_when_ttl_expired(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 20, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "_utc_now", lambda: now)

    cache_key = mod._preflight_cache_key(repo_root, envelope, "remote_publish")
    receipt = mod.PreflightReceipt(
        schema_version=1,
        receipt_id="preflight-remote_publish-20260412T190000Z-expired1",
        envelope_seal=envelope.preflight_cache_seal(),
        repo_root=str(repo_root.resolve()),
        check_type="remote_publish",
        started_at="2026-04-12T19:00:00Z",
        finished_at="2026-04-12T19:00:10Z",
        passed=True,
        checks=[{"name": "git_push", "passed": True, "detail": "ok"}],
        cache_key=cache_key,
        ttl_seconds=3600,
        expires_at="2026-04-12T19:30:00Z",
        artifacts={"branch": "preflight/remote/test", "worktree_path": "/tmp/worktree"},
    )
    mod._save_preflight_receipt(repo_root, receipt)

    loaded = mod._load_cached_preflight_receipt(
        repo_root,
        envelope,
        "remote_publish",
        now=now,
    )

    assert loaded is None


def test_load_cached_preflight_receipt_misses_when_envelope_changes(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    changed_envelope = CredentialEnvelope.from_environment(
        {
            "ARAGORA_CLAUDE_PROFILE": "claude",
            "GITHUB_TOKEN": "token",
            "OPENAI_API_KEY": "key",
            "ARAGORA_PROVIDER": "openai",
            "PYTEST_AVAILABLE": "true",
            "RUFF_AVAILABLE": "true",
        }
    )

    cache_key = mod._preflight_cache_key(repo_root, envelope, "scratch")
    receipt = mod.PreflightReceipt(
        schema_version=1,
        receipt_id="preflight-scratch-20260412T200000Z-envelope1",
        envelope_seal=envelope.preflight_cache_seal(),
        repo_root=str(repo_root.resolve()),
        check_type="scratch",
        started_at="2026-04-12T20:00:00Z",
        finished_at="2026-04-12T20:00:05Z",
        passed=True,
        checks=[{"name": "git_commit", "passed": True, "detail": "ok"}],
        cache_key=cache_key,
        ttl_seconds=86400,
        expires_at="2026-04-13T20:00:05Z",
        artifacts={"branch": "preflight/scratch/test", "worktree_path": "/tmp/worktree"},
    )
    mod._save_preflight_receipt(repo_root, receipt)

    loaded = mod._load_cached_preflight_receipt(repo_root, changed_envelope, "scratch")

    assert loaded is None


def test_failed_receipts_are_not_reused(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()

    cache_key = mod._preflight_cache_key(repo_root, envelope, "scratch")
    receipt = mod.PreflightReceipt(
        schema_version=1,
        receipt_id="preflight-scratch-20260412T200000Z-failed001",
        envelope_seal=envelope.preflight_cache_seal(),
        repo_root=str(repo_root.resolve()),
        check_type="scratch",
        started_at="2026-04-12T20:00:00Z",
        finished_at="2026-04-12T20:00:05Z",
        passed=False,
        checks=[{"name": "git_commit", "passed": False, "detail": "failed"}],
        cache_key=cache_key,
        ttl_seconds=86400,
        expires_at="2026-04-13T20:00:05Z",
        artifacts={"branch": "preflight/scratch/test", "worktree_path": "/tmp/worktree"},
    )
    mod._save_preflight_receipt(repo_root, receipt)

    loaded = mod._load_cached_preflight_receipt(repo_root, envelope, "scratch")

    assert loaded is None


def test_cleanup_failure_marks_receipt_not_cacheable(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 19, 55, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(mod, "_receipt_token", lambda: "deadbeef")

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        if cmd[:3] == ["git", "worktree", "add"]:
            Path(cmd[5]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["git", "branch", "-D"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="branch delete failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    receipt = mod.run_scratch_validation_receipt(
        repo_root=repo_root,
        envelope=envelope,
        force_refresh=True,
    )

    assert receipt.passed is False
    assert any(
        check["name"] == "cleanup_branch_delete" and check["passed"] is False
        for check in receipt.checks
    )
    assert mod._load_cached_preflight_receipt(repo_root, envelope, "scratch", now=now) is None
