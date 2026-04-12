from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from aragora.swarm.credential_envelope import CredentialEnvelope
from aragora.swarm.env_utils import git_safe_env
from aragora.swarm.mission import GateEvaluation, GateType, GateVerdict, MissionContextPolicy
from aragora.swarm.worker_contract import checksum_contract_payload
from aragora.swarm.worker_launcher import LaunchConfig, WorkerLauncher, WorkerProcess


@dataclass(slots=True)
class PreflightResult:
    repo_root: str
    base_ref: str
    branch: str
    worktree_path: str
    agent: str
    published: bool
    pull_request_created: bool
    pull_request_closed: bool
    cleanup_worktree_removed: bool
    cleanup_branch_removed: bool
    dispatch_gate: dict[str, Any] = field(default_factory=dict)
    worker: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    checks: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    envelope: CredentialEnvelope | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "base_ref": self.base_ref,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "agent": self.agent,
            "passed": self.passed,
            "checks": list(self.checks),
            "duration_seconds": self.duration_seconds,
            "envelope": self.envelope.to_dict() if self.envelope else None,
            "published": self.published,
            "pull_request_created": self.pull_request_created,
            "pull_request_closed": self.pull_request_closed,
            "cleanup_worktree_removed": self.cleanup_worktree_removed,
            "cleanup_branch_removed": self.cleanup_branch_removed,
            "dispatch_gate": dict(self.dispatch_gate),
            "worker": dict(self.worker),
        }


def _write_stdout_line(text: str) -> None:
    sys.stdout.write(f"{text}\n")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"Command failed: {' '.join(cmd)}")


def _check_git_clean(repo_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        env=git_safe_env(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return {
            "name": "git_status_clean",
            "passed": False,
            "detail": detail or "git status failed",
        }
    output = (result.stdout or "").strip()
    return {
        "name": "git_status_clean",
        "passed": output == "",
        "detail": "clean" if output == "" else "worktree has uncommitted changes",
    }


def _check_can_create_branch(repo_root: Path) -> dict[str, Any]:
    branch = f"preflight/check-{int(time.time())}"
    result = subprocess.run(
        ["git", "branch", branch],
        cwd=str(repo_root),
        env=git_safe_env(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return {
            "name": "git_can_create_branch",
            "passed": False,
            "detail": detail or "branch create failed",
        }
    subprocess.run(
        ["git", "branch", "-D", branch],
        cwd=str(repo_root),
        env=git_safe_env(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return {"name": "git_can_create_branch", "passed": True, "detail": "ok"}


def _check_can_commit(repo_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "commit", "--allow-empty", "--dry-run", "-m", "preflight check"],
        cwd=str(repo_root),
        env=git_safe_env(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return {
            "name": "git_can_commit",
            "passed": False,
            "detail": detail or "commit dry-run failed",
        }
    return {"name": "git_can_commit", "passed": True, "detail": "ok"}


def _check_tool_available(name: str, cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return {"name": name, "passed": False, "detail": detail or "command failed"}
    return {"name": name, "passed": True, "detail": (result.stdout or "").strip() or "ok"}


def _runner_command(envelope: CredentialEnvelope) -> list[str] | None:
    command_path = str(envelope.runner.command_path or "").strip()
    if command_path:
        return [command_path, "--version"]
    profile = str(envelope.runner.profile or "").lower()
    if "codex" in profile:
        return ["codex", "--version"]
    if "claude" in profile:
        return ["claude", "--version"]
    return None


def run_preflight_checks(
    envelope: CredentialEnvelope,
    *,
    repo_root: Path,
) -> PreflightResult:
    start = time.monotonic()
    checks: list[dict[str, Any]] = [
        _check_git_clean(repo_root),
        _check_can_create_branch(repo_root),
        _check_can_commit(repo_root),
        _check_tool_available("ruff_available", ["python3", "-m", "ruff", "--version"]),
        _check_tool_available("pytest_available", ["python3", "-m", "pytest", "--version"]),
    ]
    runner_cmd = _runner_command(envelope)
    if runner_cmd:
        checks.append(_check_tool_available("runner_cli", runner_cmd))
    else:
        checks.append(
            {
                "name": "runner_cli",
                "passed": False,
                "detail": "runner command not configured",
            }
        )
    passed = all(check["passed"] for check in checks)
    duration = time.monotonic() - start
    return PreflightResult(
        passed=passed,
        checks=checks,
        duration_seconds=duration,
        envelope=envelope,
        repo_root=str(repo_root),
        base_ref="",
        branch="",
        worktree_path=str(repo_root),
        agent=envelope.runner.profile or "unknown",
        published=False,
        pull_request_created=False,
        pull_request_closed=False,
        cleanup_worktree_removed=False,
        cleanup_branch_removed=False,
        dispatch_gate={},
        worker={},
    )


def _branch_name() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"preflight/{stamp}"


def _worktree_path(repo_root: Path, branch: str) -> Path:
    return repo_root / ".worktrees" / f"preflight-{branch.replace('/', '-')}"


def _work_order(agent: str) -> dict[str, object]:
    filename = "scratch/preflight_worker_check.txt"
    return {
        "work_order_id": f"preflight-{int(time.time())}",
        "target_agent": agent,
        "mission_id": "mission-rs-worker-contract-preflight",
        "stage_id": "stage-dispatch-ready-preflight",
        "assertion_ids": ["RS-PREFLIGHT-ASSERT-1"],
        "evidence_expectations": [
            "worker_contract",
            "worker_contract_checksum",
            "receipt",
        ],
        "title": "Preflight worker check",
        "description": (
            "Create a file named `scratch/preflight_worker_check.txt` with a single line "
            "timestamp. Commit it with message `chore: preflight worker check`. "
            "Do not modify any other files."
        ),
        "file_scope": [filename],
        "expected_tests": [],
        "metadata": {"admin_approved": True},
    }


async def _run_worker(
    *,
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    agent: str,
) -> WorkerProcess:
    config = LaunchConfig(
        allow_claude_dangerously_skip_permissions=True,
        allow_codex_full_auto=True,
    )
    launcher = WorkerLauncher(config=config)
    work_order = _work_order(agent)
    return await launcher.launch_and_wait(
        work_order,
        worktree_path=str(worktree_path),
        branch=branch,
        timeout=900.0,
    )


def evaluate_preflight_dispatch_gate(worker: WorkerProcess) -> dict[str, Any]:
    contract = dict(worker.worker_contract or {})
    mission_id = str(contract.get("mission_id", "") or "").strip()
    stage_id = str(contract.get("stage_id", "") or "").strip()
    assertion_ids = [
        str(item).strip() for item in contract.get("assertion_ids", []) if str(item).strip()
    ]
    failure_classes: list[str] = []
    notes: list[str] = []

    if not contract:
        failure_classes.append("contract_missing")
        notes.append("Preflight worker did not emit a worker contract.")
    if not worker.worker_contract_checksum:
        failure_classes.append("contract_missing")
        notes.append("Preflight worker did not emit a worker contract checksum.")
    if contract and worker.worker_contract_checksum:
        checksum = checksum_contract_payload(contract)
        if checksum != worker.worker_contract_checksum:
            failure_classes.append("contract_missing")
            notes.append("Worker contract checksum does not match the contract payload.")

    policy = MissionContextPolicy.from_dict(contract.get("mission_context_policy"))
    if not policy.is_resolvable():
        failure_classes.append("context_policy_unresolved")
        notes.append("Worker mission context policy is missing or not enforceable.")

    verdict = GateVerdict.PASS.value if not failure_classes else GateVerdict.BLOCKED.value
    gate = GateEvaluation(
        gate_type=GateType.DISPATCH_READY.value,
        verdict=verdict,
        mission_id=mission_id,
        stage_id=stage_id,
        assertion_ids=assertion_ids,
        failure_classes=failure_classes,
        repair_eligible=any(
            failure in {"contract_missing", "context_policy_unresolved"}
            for failure in failure_classes
        ),
        required_evidence=[
            "worker_contract",
            "worker_contract_checksum",
            "mission_context_policy",
        ],
        notes=" ".join(notes).strip(),
    )
    return gate.to_dict()


def _validate_worker_contract(worker: WorkerProcess) -> dict[str, Any]:
    if not worker.worker_contract:
        raise RuntimeError("Preflight worker did not emit a worker contract.")
    if not worker.worker_contract_checksum:
        raise RuntimeError("Preflight worker did not emit a worker contract checksum.")
    checksum = checksum_contract_payload(worker.worker_contract)
    if checksum != worker.worker_contract_checksum:
        raise RuntimeError(
            "Preflight worker emitted a worker contract checksum that does not match the contract payload."
        )
    gate = evaluate_preflight_dispatch_gate(worker)
    if str(gate.get("verdict", "")).strip() != GateVerdict.PASS.value:
        raise RuntimeError(str(gate.get("notes") or "Preflight dispatch gate failed."))
    return gate


def _create_pr(repo_root: Path, branch: str, base_ref: str) -> None:
    _run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            "synaptent/aragora",
            "--head",
            branch,
            "--base",
            base_ref,
            "--title",
            "[preflight] worker check",
            "--body",
            "Preflight validation of worker read/write/commit/push.",
            "--draft",
        ],
        cwd=repo_root,
    )


def _close_pr(repo_root: Path, branch: str) -> None:
    _run(
        [
            "gh",
            "pr",
            "close",
            "--repo",
            "synaptent/aragora",
            branch,
            "--delete-branch",
            "--comment",
            "Preflight complete - closing.",
        ],
        cwd=repo_root,
    )


def run_preflight(
    *,
    repo_root: Path,
    agent: str | None = None,
    base_ref: str = "main",
    skip_publication: bool = False,
    envelope: CredentialEnvelope | None = None,
) -> PreflightResult:
    if envelope is not None:
        return run_preflight_checks(envelope, repo_root=repo_root)

    start = time.monotonic()
    resolved_repo_root = repo_root.resolve()
    normalized_agent = str(agent or "").strip() or "claude"
    normalized_base_ref = str(base_ref or "main").strip() or "main"
    branch = _branch_name()
    worktree_path = _worktree_path(resolved_repo_root, branch)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    worktree_created = False
    worker: WorkerProcess | None = None
    published = False
    pull_request_created = False
    pull_request_closed = False
    cleanup_worktree_removed = False
    cleanup_branch_removed = False
    dispatch_gate: dict[str, Any] = {}

    try:
        _run(
            ["git", "worktree", "add", "-b", branch, str(worktree_path), normalized_base_ref],
            cwd=resolved_repo_root,
        )
        worktree_created = True

        worker = asyncio.run(
            _run_worker(
                repo_root=resolved_repo_root,
                worktree_path=worktree_path,
                branch=branch,
                agent=normalized_agent,
            )
        )
        dispatch_gate = _validate_worker_contract(worker)
        if not worker.commit_shas:
            raise RuntimeError("Preflight worker did not produce a commit.")

        if not skip_publication:
            _run(["git", "push", "origin", "HEAD"], cwd=worktree_path, env=git_safe_env())
            published = True
            _create_pr(resolved_repo_root, branch, normalized_base_ref)
            pull_request_created = True
            _close_pr(resolved_repo_root, branch)
            pull_request_closed = True
    finally:
        if worktree_created:
            worktree_remove = subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=str(resolved_repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            cleanup_worktree_removed = worktree_remove.returncode == 0
            branch_remove = subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=str(resolved_repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            cleanup_branch_removed = branch_remove.returncode == 0

    passed = False
    if dispatch_gate:
        passed = str(dispatch_gate.get("verdict", "")).strip() == GateVerdict.PASS.value
    duration = time.monotonic() - start
    return PreflightResult(
        passed=passed,
        checks=[],
        duration_seconds=duration,
        envelope=None,
        repo_root=str(resolved_repo_root),
        base_ref=normalized_base_ref,
        branch=branch,
        worktree_path=str(worktree_path),
        agent=normalized_agent,
        published=published,
        pull_request_created=pull_request_created,
        pull_request_closed=pull_request_closed,
        cleanup_worktree_removed=cleanup_worktree_removed,
        cleanup_branch_removed=cleanup_branch_removed,
        dispatch_gate=dispatch_gate,
        worker=worker.to_dict() if worker is not None else {},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run swarm worker preflight.")
    parser.add_argument(
        "--repo-root",
        default=str(Path.cwd()),
        help="Repository root",
    )
    parser.add_argument(
        "--agent",
        default=os.environ.get("WORKER_MODEL", "claude"),
        help="Target agent (claude or codex)",
    )
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Base ref to branch from and target for the temporary PR (default: main).",
    )
    parser.add_argument(
        "--skip-publication",
        action="store_true",
        help="Skip push/PR steps (debug only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a structured result payload.",
    )
    args = parser.parse_args()

    result = run_preflight(
        repo_root=Path(args.repo_root),
        agent=str(args.agent),
        base_ref=str(args.base_ref),
        skip_publication=bool(args.skip_publication),
    )
    if args.json:
        _write_stdout_line(json.dumps(result.to_dict(), indent=2))
    else:
        _write_stdout_line("preflight=ok")
        _write_stdout_line(f"agent={result.agent}")
        _write_stdout_line(f"base_ref={result.base_ref}")
        _write_stdout_line(f"branch={result.branch}")
        checksum = str(result.worker.get("worker_contract_checksum", "")).strip()
        if checksum:
            _write_stdout_line(f"worker_contract_checksum={checksum}")
        commit_shas = [
            str(item) for item in result.worker.get("commit_shas", []) if str(item).strip()
        ]
        if commit_shas:
            _write_stdout_line(f"commit_shas={','.join(commit_shas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
