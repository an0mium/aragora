from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from aragora.swarm.boss_feed import GitHubIssue
from aragora.swarm.credential_envelope import CredentialEnvelope
from aragora.swarm.env_utils import git_safe_env
from aragora.swarm.worker_contract import build_worker_contract
from aragora.swarm.worker_process import LaunchConfig


def _target_agent(
    loop: Any,
    *,
    selected_runner: dict[str, Any] | None,
    requested_target_agent: str | None,
) -> str:
    requested = str(requested_target_agent or "").strip().lower()
    if requested:
        return requested
    if isinstance(selected_runner, dict):
        runner_type = str(selected_runner.get("runner_type", "")).strip().lower()
        if runner_type:
            return runner_type
    default_target = str(loop.config.default_target_agent or "").strip().lower()
    if default_target:
        return default_target
    return "codex"


def _preview_env(
    loop: Any,
    *,
    selected_runner: dict[str, Any] | None,
    requested_target_agent: str | None,
    worker_env: Mapping[str, str] | None,
) -> tuple[str, dict[str, str]]:
    env = dict(os.environ)
    if isinstance(loop._env, dict):
        env.update({str(key): str(value) for key, value in loop._env.items() if str(key).strip()})
    if worker_env:
        env.update({str(key): str(value) for key, value in worker_env.items() if str(key).strip()})

    target_agent = _target_agent(
        loop,
        selected_runner=selected_runner,
        requested_target_agent=requested_target_agent,
    )
    runner_type = (
        str((selected_runner or {}).get("runner_type", "")).strip().lower() or target_agent
    )
    runner_command = str((selected_runner or {}).get("command_path", "")).strip()
    runner_profile = str((selected_runner or {}).get("profile", "")).strip()

    if runner_type == "claude":
        env.setdefault("ARAGORA_CLAUDE_PROFILE", runner_profile or "default")
        env.setdefault("CLAUDE_COMMAND", runner_command or shutil.which("claude") or "claude")
        env.setdefault("ARAGORA_RUNNER_AUTH_MODE", "profile")
    elif runner_type == "codex":
        env.setdefault("CODEX_COMMAND", runner_command or shutil.which("codex") or "codex")
        env.setdefault("ARAGORA_RUNNER_AUTH_MODE", "command")
    elif runner_command:
        env.setdefault("ARAGORA_RUNNER_COMMAND", runner_command)
        env.setdefault("ARAGORA_RUNNER_AUTH_MODE", "command")

    pytest_path = shutil.which("pytest")
    if pytest_path and not (
        env.get("ARAGORA_CAN_RUN_PYTEST") or env.get("PYTEST_AVAILABLE") or env.get("PYTEST_PATH")
    ):
        env["PYTEST_PATH"] = pytest_path
    ruff_path = shutil.which("ruff")
    if ruff_path and not (
        env.get("ARAGORA_CAN_RUN_RUFF") or env.get("RUFF_AVAILABLE") or env.get("RUFF_PATH")
    ):
        env["RUFF_PATH"] = ruff_path
    return target_agent, env


def _preview_work_orders(spec: Any) -> list[dict[str, Any]]:
    preview_orders: list[dict[str, Any]] = []
    raw_orders = getattr(spec, "work_orders", None)
    if isinstance(raw_orders, list):
        for item in raw_orders:
            if not isinstance(item, dict):
                continue
            preview_orders.append(
                {
                    "file_scope": [
                        str(path).strip()
                        for path in item.get("file_scope", [])
                        if str(path).strip()
                    ],
                    "expected_tests": [
                        str(test).strip()
                        for test in item.get("expected_tests", [])
                        if str(test).strip()
                    ],
                    "mission_id": str(item.get("mission_id", "") or "").strip(),
                    "stage_id": str(item.get("stage_id", "") or "").strip(),
                    "assertion_ids": [
                        str(entry).strip()
                        for entry in item.get("assertion_ids", [])
                        if str(entry).strip()
                    ],
                    "evidence_expectations": [
                        str(entry).strip()
                        for entry in item.get("evidence_expectations", [])
                        if str(entry).strip()
                    ],
                    "mission_context_policies": dict(item.get("mission_context_policies") or {}),
                }
            )
    if preview_orders:
        return preview_orders
    return [
        {
            "file_scope": [
                str(path).strip()
                for path in getattr(spec, "file_scope_hints", [])
                if str(path).strip()
            ],
            "expected_tests": [],
            "mission_context_policies": dict(getattr(spec, "mission_context_policies", {}) or {}),
        }
    ]


def _github_cli_authenticated(loop: Any) -> bool:
    cached = getattr(loop, "_dispatch_contract_github_cli_auth_available", None)
    if cached is not None:
        return bool(cached)
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            cwd=Path.cwd(),
            env=git_safe_env(loop._env),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        setattr(loop, "_dispatch_contract_github_cli_auth_available", False)
        return False
    authenticated = result.returncode == 0
    setattr(loop, "_dispatch_contract_github_cli_auth_available", authenticated)
    return authenticated


def _missing_slices(loop: Any, *, envelope: CredentialEnvelope, target_agent: str) -> list[str]:
    missing = list(envelope.missing_slices())
    requires_git = bool(loop.config.auto_publish_deliverables)
    requires_github_api = bool(
        loop.config.auto_publish_deliverables or loop.config.auto_close_already_done_issues
    )
    if not requires_git and "git" in missing:
        missing.remove("git")
    if target_agent in {"claude", "codex"} and "provider" in missing:
        missing.remove("provider")
    if "github_api" in missing and (not requires_github_api or _github_cli_authenticated(loop)):
        missing.remove("github_api")
    allowed = {"runner", "git", "github_api", "verification"}
    if target_agent not in {"claude", "codex"}:
        allowed.add("provider")
    return [item for item in missing if item in allowed]


def _gate_reason(
    *,
    issue_number: int,
    target_agent: str,
    missing_slices: list[str],
    contract_valid: bool,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    next_actions: list[str] = []
    if not contract_valid:
        reasons.append(f"Issue #{issue_number} failed worker contract admission before dispatch.")
        next_actions.append(
            "Refresh the dispatch contract inputs so runner, scope, and mission context policy resolve cleanly before redispatch."
        )
    slice_messages = {
        "runner": f"Issue #{issue_number} is missing a runner slice for `{target_agent}` dispatch.",
        "git": f"Issue #{issue_number} is missing git publish credentials for branch-scoped worktree dispatch.",
        "github_api": f"Issue #{issue_number} is missing GitHub API authentication required for full-auto publication.",
        "provider": f"Issue #{issue_number} is missing provider credentials for `{target_agent}` dispatch.",
        "verification": f"Issue #{issue_number} is missing verification tool capability (pytest and/or ruff).",
    }
    slice_actions = {
        "runner": "Select a runner with a resolvable command/profile before redispatch.",
        "git": "Restore SSH agent or HTTPS git credentials before redispatch.",
        "github_api": "Restore `gh` authentication or a GitHub token before redispatch.",
        "provider": "Restore the provider API key or choose a CLI-backed target agent before redispatch.",
        "verification": "Ensure pytest and ruff are installed and discoverable before redispatch.",
    }
    for missing in missing_slices:
        reasons.append(slice_messages[missing])
        next_actions.append(slice_actions[missing])
    return reasons, list(dict.fromkeys(next_actions))


def dispatch_contract_gate(
    loop: Any,
    issue: GitHubIssue,
    spec: Any,
    selected_runner: dict[str, Any] | None,
    requested_target_agent: str | None,
    worker_env: Mapping[str, str] | None,
    claimed_runner_id: str | None,
) -> dict[str, Any] | None:
    target_agent, preview_env = _preview_env(
        loop,
        selected_runner=selected_runner,
        requested_target_agent=requested_target_agent,
        worker_env=worker_env,
    )
    envelope = CredentialEnvelope.from_environment(preview_env)
    missing_slices = _missing_slices(loop, envelope=envelope, target_agent=target_agent)
    launch_config = LaunchConfig(
        base_branch=loop.config.target_branch,
        execution_mode=loop.config.execution_mode,
        claude_profile=(
            str((selected_runner or {}).get("profile", "")).strip() or None
            if target_agent == "claude"
            else None
        ),
        allow_claude_dangerously_skip_permissions=(
            loop.config.allow_claude_dangerously_skip_permissions
        ),
        allow_codex_full_auto=loop.config.allow_codex_full_auto,
    )
    contract_valid = True
    for work_order in _preview_work_orders(spec):
        try:
            contract = build_worker_contract(
                agent=target_agent,
                config=launch_config,
                worktree_path=str(Path.cwd()),
                env=preview_env,
                work_order=work_order,
            )
            contract.validate()
            if not contract.admission_check():
                contract_valid = False
                break
        except ValueError:
            contract_valid = False
            break

    if contract_valid and not missing_slices:
        return None

    reasons, next_actions = _gate_reason(
        issue_number=issue.number,
        target_agent=target_agent,
        missing_slices=missing_slices,
        contract_valid=contract_valid,
    )
    if claimed_runner_id:
        loop._release_runner_claim(claimed_runner_id)
    return {
        "status": "needs_human",
        "outcome": "blocked_auth_failure" if missing_slices else "blocked",
        "reasons": reasons,
        "next_actions": next_actions,
        "dispatch_contract": {
            "target_agent": target_agent,
            "contract_valid": contract_valid,
            "missing_slices": missing_slices,
            "credential_envelope": envelope.preflight_cache_payload(),
        },
    }
