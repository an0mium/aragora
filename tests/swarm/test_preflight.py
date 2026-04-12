from __future__ import annotations

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
