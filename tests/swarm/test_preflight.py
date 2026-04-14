from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.swarm.credential_envelope import CredentialEnvelope
from aragora.swarm.mission import GateType, GateVerdict
from aragora.swarm import preflight as mod
from aragora.swarm.terminal_truth import TerminalClass
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


def test_run_contract_preflight_receipt_persists_and_returns_receipt(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    contract_payload = _worker(branch="preflight/20260413-contract").worker_contract
    contract_path = tmp_path / "worker_contract.json"
    contract_path.write_text(json.dumps(contract_payload), encoding="utf-8")
    fake_result = mod.PreflightResult(
        repo_root=str(repo_root),
        base_ref="main",
        branch="preflight/20260413-contract",
        worktree_path=str(repo_root / ".worktrees" / "preflight-contract"),
        agent="codex",
        published=False,
        pull_request_created=False,
        pull_request_closed=False,
        cleanup_worktree_removed=True,
        cleanup_branch_removed=True,
        dispatch_gate={
            "gate_type": GateType.DISPATCH_READY.value,
            "verdict": GateVerdict.PASS.value,
            "failure_classes": [],
            "notes": "dispatch ready",
        },
        worker=_worker(branch="preflight/20260413-contract").to_dict(),
        passed=True,
    )

    monkeypatch.setattr(mod, "run_preflight", lambda **_: fake_result)
    receipt = mod.run_contract_preflight_receipt(
        repo_root=repo_root,
        agent="codex",
        base_ref="main",
        skip_publication=True,
        contract_path=contract_path,
        envelope=envelope,
    )

    expected_checksum = checksum_contract_payload(contract_payload)
    assert receipt.passed is True
    assert receipt.check_type == "scratch"
    assert receipt.artifacts["expected_contract_checksum"] == expected_checksum
    assert receipt.artifacts["worker_contract_checksum"] == expected_checksum
    assert any(check["name"] == "dispatch_gate" for check in receipt.checks)
    receipt_path = (
        repo_root / ".aragora" / "receipts" / "preflight" / f"scratch-{receipt.cache_key}.json"
    )
    assert receipt_path.exists()


def test_run_contract_preflight_receipt_persists_failure_on_preflight_error(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    contract_payload = _worker(branch="preflight/20260413-contract-bad").worker_contract
    contract_path = tmp_path / "worker_contract_bad.json"
    contract_path.write_text(json.dumps(contract_payload), encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "run_preflight",
        lambda **_: (_ for _ in ()).throw(RuntimeError("worker contract checksum mismatch")),
    )

    receipt = mod.run_contract_preflight_receipt(
        repo_root=repo_root,
        agent="codex",
        base_ref="main",
        skip_publication=True,
        contract_path=contract_path,
        envelope=envelope,
    )

    assert receipt.passed is False
    assert "checksum mismatch" in receipt.checks[0]["detail"]
    assert receipt.failure_terminal_class == TerminalClass.BLOCKED_NOT_DISPATCH_BOUNDED


def test_evaluate_preflight_receipt_gate_blocks_missing_receipt(tmp_path: Path) -> None:
    gate = mod.evaluate_preflight_receipt_gate(
        None,
        repo_root=tmp_path,
        envelope=_envelope(),
        check_type="scratch",
    )

    assert gate.verdict == GateVerdict.BLOCKED.value
    assert gate.failure_classes == ["receipt_missing"]


def test_evaluate_preflight_receipt_gate_blocks_expired_and_failed_receipts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 13, 2, 0, 0, tzinfo=timezone.utc)
    expired_receipt = mod.PreflightReceipt(
        receipt_id="preflight-scratch-expired",
        envelope_seal=envelope.preflight_cache_seal(),
        repo_root=str(repo_root),
        check_type="scratch",
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:05:00Z",
        passed=True,
        checks=[{"name": "dispatch_gate", "passed": True, "detail": "ok"}],
        cache_key="cache-1",
        ttl_seconds=60,
        expires_at="2026-04-13T00:06:00Z",
        artifacts={},
    )
    expired_gate = mod.evaluate_preflight_receipt_gate(
        expired_receipt,
        repo_root=repo_root,
        envelope=envelope,
        check_type="scratch",
        now=now,
    )
    assert expired_gate.failure_classes == ["receipt_expired"]

    failed_receipt = mod.PreflightReceipt(
        receipt_id="preflight-scratch-failed",
        envelope_seal=envelope.preflight_cache_seal(),
        repo_root=str(repo_root),
        check_type="scratch",
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:05:00Z",
        passed=False,
        checks=[{"name": "dispatch_gate", "passed": False, "detail": "bounded dispatch blocked"}],
        cache_key="cache-2",
        ttl_seconds=3600,
        expires_at="2026-04-13T03:00:00Z",
        artifacts={},
    )
    failed_gate = mod.evaluate_preflight_receipt_gate(
        failed_receipt,
        repo_root=repo_root,
        envelope=envelope,
        check_type="scratch",
        now=now,
    )
    assert failed_gate.verdict == GateVerdict.BLOCKED.value
    assert failed_gate.failure_classes == [TerminalClass.BLOCKED_NOT_DISPATCH_BOUNDED.value]


def test_evaluate_preflight_receipt_gate_blocks_envelope_and_contract_mismatch(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    receipt = mod.PreflightReceipt(
        receipt_id="preflight-remote-valid",
        envelope_seal=envelope.preflight_cache_seal(),
        repo_root=str(repo_root),
        check_type="remote_publish",
        started_at="2026-04-13T00:00:00Z",
        finished_at="2026-04-13T00:05:00Z",
        passed=True,
        checks=[{"name": "dispatch_gate", "passed": True, "detail": "ok"}],
        cache_key="cache-3",
        ttl_seconds=3600,
        expires_at="2026-04-13T03:00:00Z",
        artifacts={
            "target_ref": "main",
            "expected_contract_checksum": "expected-1",
        },
    )

    changed_envelope = CredentialEnvelope.from_environment(
        {
            "ARAGORA_CLAUDE_PROFILE": "different",
            "GITHUB_TOKEN": "token",
            "OPENAI_API_KEY": "key",
        }
    )
    envelope_gate = mod.evaluate_preflight_receipt_gate(
        receipt,
        repo_root=repo_root,
        envelope=changed_envelope,
        check_type="remote_publish",
        base_ref="main",
        expected_contract_checksum="expected-1",
        now=datetime(2026, 4, 13, 2, 0, 0, tzinfo=timezone.utc),
    )
    assert envelope_gate.failure_classes == ["receipt_envelope_mismatch"]

    checksum_gate = mod.evaluate_preflight_receipt_gate(
        receipt,
        repo_root=repo_root,
        envelope=envelope,
        check_type="remote_publish",
        base_ref="main",
        expected_contract_checksum="expected-2",
        now=datetime(2026, 4, 13, 2, 0, 0, tzinfo=timezone.utc),
    )
    assert checksum_gate.failure_classes == ["receipt_contract_mismatch"]


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
    assert receipt.artifacts["target_ref"] == "main"
    assert receipt.artifacts["draft_pr_number"] is None
    assert receipt.artifacts["draft_pr_url"] == ""
    assert any(check["name"] == "git_commit" for check in receipt.checks)
    receipt_path = (
        repo_root / ".aragora" / "receipts" / "preflight" / f"scratch-{receipt.cache_key}.json"
    )
    assert receipt_path.exists()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["receipt_id"] == receipt.receipt_id
    assert commands[0] == [
        "git",
        "worktree",
        "add",
        "-b",
        receipt.artifacts["branch"],
        str(
            repo_root / ".worktrees" / f"preflight-{receipt.artifacts['branch'].replace('/', '-')}"
        ),
        "main",
    ]


def test_preflight_result_failure_terminal_class_maps_runner_auth_failure() -> None:
    result = mod.PreflightResult(
        repo_root="/tmp/repo",
        base_ref="main",
        branch="",
        worktree_path="/tmp/repo",
        agent="codex",
        published=False,
        pull_request_created=False,
        pull_request_closed=False,
        cleanup_worktree_removed=False,
        cleanup_branch_removed=False,
        passed=False,
        checks=[
            {
                "name": "runner_cli",
                "passed": False,
                "detail": "authentication required for codex runner",
            }
        ],
    )

    assert result.failure_terminal_class == TerminalClass.BLOCKED_AUTH_FAILURE


def test_preflight_result_failure_terminal_class_maps_no_runner() -> None:
    result = mod.PreflightResult(
        repo_root="/tmp/repo",
        base_ref="main",
        branch="",
        worktree_path="/tmp/repo",
        agent="codex",
        published=False,
        pull_request_created=False,
        pull_request_closed=False,
        cleanup_worktree_removed=False,
        cleanup_branch_removed=False,
        passed=False,
        checks=[
            {
                "name": "runner_cli",
                "passed": False,
                "detail": "runner command not configured",
            }
        ],
    )

    assert result.failure_terminal_class == TerminalClass.BLOCKED_NO_RUNNER


def test_preflight_result_failure_terminal_class_maps_scope_conflict() -> None:
    result = mod.PreflightResult(
        repo_root="/tmp/repo",
        base_ref="main",
        branch="",
        worktree_path="/tmp/repo",
        agent="codex",
        published=False,
        pull_request_created=False,
        pull_request_closed=False,
        cleanup_worktree_removed=False,
        cleanup_branch_removed=False,
        passed=False,
        checks=[
            {
                "name": "git_worktree_add",
                "passed": False,
                "detail": "fatal: worktree path already exists and scope conflict was detected",
            }
        ],
    )

    assert result.failure_terminal_class == TerminalClass.BLOCKED_NOT_DISPATCH_BOUNDED


def test_preflight_result_failure_terminal_class_maps_dispatch_gate_block() -> None:
    result = mod.PreflightResult(
        repo_root="/tmp/repo",
        base_ref="main",
        branch="",
        worktree_path="/tmp/repo",
        agent="codex",
        published=False,
        pull_request_created=False,
        pull_request_closed=False,
        cleanup_worktree_removed=False,
        cleanup_branch_removed=False,
        passed=False,
        checks=[],
        dispatch_gate={
            "verdict": GateVerdict.BLOCKED.value,
            "failure_classes": ["context_policy_unresolved"],
            "notes": "Dispatch gate failed because the worker is not dispatch bounded.",
        },
    )

    assert result.failure_terminal_class == TerminalClass.BLOCKED_NOT_DISPATCH_BOUNDED


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
    assert receipt.artifacts["target_ref"] == "main"
    assert any(check["name"] == "gh_pr_capture" for check in receipt.checks)
    assert ["git", "push", "origin", "HEAD"] in commands
    assert any(cmd[:3] == ["gh", "pr", "close"] for cmd in commands)


def test_run_remote_publish_validation_receipt_closes_draft_when_create_output_unparseable(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 19, 52, 0, tzinfo=timezone.utc)
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(mod, "_receipt_token", lambda: "ca11ab1e")

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(list(cmd))
        if cmd[:3] == ["git", "worktree", "add"]:
            Path(cmd[5]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["gh", "pr", "create"]:
            return SimpleNamespace(
                returncode=0,
                stdout="draft created successfully\n",
                stderr="",
            )
        if cmd[:3] == ["gh", "pr", "list"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    '[{"number": 6123, "url": '
                    '"https://github.com/synaptent/aragora/pull/6123", '
                    '"isDraft": true, "baseRefName": "release/2026-04-13"}]'
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    receipt = mod.run_remote_publish_validation_receipt(
        repo_root=repo_root,
        envelope=envelope,
        base_ref="release/2026-04-13",
    )

    assert receipt.passed is True
    assert receipt.artifacts["draft_pr_number"] == 6123
    assert receipt.artifacts["draft_pr_url"] == "https://github.com/synaptent/aragora/pull/6123"
    assert receipt.artifacts["target_ref"] == "release/2026-04-13"
    assert any(
        check["name"] == "gh_pr_capture"
        and check["passed"] is True
        and check["detail"] == "https://github.com/synaptent/aragora/pull/6123"
        for check in receipt.checks
    )
    assert any(
        cmd[:5] == ["gh", "pr", "list", "--head", receipt.artifacts["branch"]] for cmd in commands
    )
    assert any(cmd[:4] == ["gh", "pr", "close", "6123"] for cmd in commands)
    assert any(
        cmd == ["git", "push", "origin", "--delete", receipt.artifacts["branch"]]
        for cmd in commands
    )


def test_run_remote_publish_validation_receipt_retains_branch_when_draft_state_unresolved(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 20, 7, 0, tzinfo=timezone.utc)
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(mod, "_receipt_token", lambda: "ca11ab1e")

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(list(cmd))
        if cmd[:3] == ["git", "worktree", "add"]:
            Path(cmd[5]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["gh", "pr", "create"]:
            return SimpleNamespace(
                returncode=0,
                stdout="draft created successfully\n",
                stderr="",
            )
        if cmd[:3] == ["gh", "pr", "list"]:
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    receipt = mod.run_remote_publish_validation_receipt(
        repo_root=repo_root,
        envelope=envelope,
        base_ref="main",
    )

    assert receipt.passed is False
    assert receipt.artifacts["draft_pr_number"] is None
    assert receipt.artifacts["draft_pr_url"] == ""
    assert any(
        check["name"] == "gh_pr_close"
        and check["passed"] is False
        and "remote branch retained for manual cleanup" in check["detail"]
        for check in receipt.checks
    )
    assert any(
        check["name"] == "cleanup_remote_branch_delete"
        and check["passed"] is False
        and "draft PR state could not be confirmed" in check["detail"]
        for check in receipt.checks
    )
    assert sum(1 for cmd in commands if cmd[:3] == ["gh", "pr", "list"]) == 2
    assert not any(cmd[:3] == ["gh", "pr", "close"] for cmd in commands)
    assert not any(
        cmd == ["git", "push", "origin", "--delete", receipt.artifacts["branch"]]
        for cmd in commands
    )


def test_run_remote_publish_validation_receipt_retains_branch_when_draft_close_fails(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 20, 8, 0, tzinfo=timezone.utc)
    commands: list[list[str]] = []

    monkeypatch.setattr(mod, "_utc_now", lambda: now)
    monkeypatch.setattr(mod, "_receipt_token", lambda: "deadbeef")

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(list(cmd))
        if cmd[:3] == ["git", "worktree", "add"]:
            Path(cmd[5]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["gh", "pr", "create"]:
            return SimpleNamespace(
                returncode=0,
                stdout="https://github.com/synaptent/aragora/pull/7777\n",
                stderr="",
            )
        if cmd[:3] == ["gh", "pr", "close"]:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="close failed",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    receipt = mod.run_remote_publish_validation_receipt(
        repo_root=repo_root,
        envelope=envelope,
        base_ref="main",
    )

    assert receipt.passed is False
    assert receipt.artifacts["draft_pr_number"] == 7777
    assert receipt.artifacts["draft_pr_url"] == "https://github.com/synaptent/aragora/pull/7777"
    assert any(
        check["name"] == "gh_pr_close"
        and check["passed"] is False
        and "close failed" in check["detail"]
        for check in receipt.checks
    )
    assert any(
        check["name"] == "cleanup_remote_branch_delete"
        and check["passed"] is False
        and "draft PR close failed" in check["detail"]
        for check in receipt.checks
    )
    assert any(cmd[:4] == ["gh", "pr", "close", "7777"] for cmd in commands)
    assert not any(
        cmd == ["git", "push", "origin", "--delete", receipt.artifacts["branch"]]
        for cmd in commands
    )


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


def test_load_cached_preflight_receipt_does_not_reuse_remote_publish_receipt_for_different_base_ref(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 20, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "_utc_now", lambda: now)

    main_cache_key = mod._preflight_cache_key(
        repo_root,
        envelope,
        "remote_publish",
        base_ref="main",
    )
    receipt = mod.PreflightReceipt(
        schema_version=1,
        receipt_id="preflight-remote_publish-20260412T200000Z-base1111",
        envelope_seal=envelope.preflight_cache_seal(),
        repo_root=str(repo_root.resolve()),
        check_type="remote_publish",
        started_at="2026-04-12T20:00:00Z",
        finished_at="2026-04-12T20:00:05Z",
        passed=True,
        checks=[{"name": "git_push", "passed": True, "detail": "ok"}],
        cache_key=main_cache_key,
        ttl_seconds=3600,
        expires_at="2026-04-12T21:00:05Z",
        artifacts={
            "branch": "preflight/remote/test",
            "worktree_path": "/tmp/worktree",
            "target_ref": "main",
        },
    )
    mod._save_preflight_receipt(repo_root, receipt)

    loaded_same_base = mod._load_cached_preflight_receipt(
        repo_root,
        envelope,
        "remote_publish",
        base_ref="main",
        now=now,
    )
    loaded_different_base = mod._load_cached_preflight_receipt(
        repo_root,
        envelope,
        "remote_publish",
        base_ref="release/2026-04-13",
        now=now,
    )

    assert loaded_same_base is not None
    assert loaded_same_base.receipt_id == receipt.receipt_id
    assert (
        mod._preflight_cache_key(
            repo_root,
            envelope,
            "remote_publish",
            base_ref="release/2026-04-13",
        )
        != main_cache_key
    )
    assert loaded_different_base is None


def test_load_cached_preflight_receipt_misses_when_ttl_expired(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    envelope = _envelope()
    now = datetime(2026, 4, 12, 20, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "_utc_now", lambda: now)

    cache_key = mod._preflight_cache_key(repo_root, envelope, "remote_publish", base_ref="main")
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
        artifacts={
            "branch": "preflight/remote/test",
            "worktree_path": "/tmp/worktree",
            "target_ref": "main",
        },
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
