from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from aragora.swarm.boss_feed import GitHubIssue
from aragora.swarm.credential_envelope import CredentialEnvelope
from aragora.swarm.env_utils import git_safe_env
from aragora.swarm.preflight import (
    PreflightReceipt,
    run_contract_preflight_receipt,
)
from aragora.swarm.terminal_truth import TerminalClass, classify_preflight_failure
from aragora.swarm.worker_contract import WorkerContract, build_worker_contract
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


def _required_preflight_receipts(loop: Any) -> list[str]:
    required = ["scratch"]
    if bool(loop.config.auto_publish_deliverables):
        required.append("remote_publish")
    return required


def _persist_preview_contract(
    *,
    repo_root: Path,
    issue_number: int,
    contract: WorkerContract,
) -> Path:
    checksum = contract.checksum()
    contract_dir = repo_root / ".aragora" / "dispatch_contracts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / f"issue-{issue_number}-{checksum[:12]}.json"
    contract_path.write_text(
        json.dumps(
            {
                "worker_contract": contract.to_dict(),
                "worker_contract_checksum": checksum,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return contract_path


def _preflight_receipt_summary(receipt: PreflightReceipt) -> dict[str, Any]:
    terminal_class = receipt.failure_terminal_class
    return {
        "check_type": receipt.check_type,
        "receipt_id": receipt.receipt_id,
        "passed": receipt.passed,
        "expires_at": receipt.expires_at,
        "failure_terminal_class": terminal_class.value if terminal_class else None,
        "failed_checks": [
            {
                "name": str(item.get("name", "")).strip(),
                "detail": str(item.get("detail", "")).strip(),
            }
            for item in receipt.checks
            if isinstance(item, dict) and not bool(item.get("passed", False))
        ],
    }


def _synthetic_preflight_summary(check_type: str, detail: str) -> dict[str, Any]:
    terminal_class = classify_preflight_failure(
        passed=False,
        checks=[{"name": "preflight_receipt", "passed": False, "detail": detail}],
        dispatch_gate=None,
    )
    return {
        "check_type": check_type,
        "receipt_id": "",
        "passed": False,
        "expires_at": "",
        "failure_terminal_class": (
            terminal_class.value
            if terminal_class
            else TerminalClass.BLOCKED_NOT_DISPATCH_BOUNDED.value
        ),
        "failed_checks": [{"name": "preflight_receipt", "detail": detail}],
    }


def _preflight_failure_outcome(summary: Mapping[str, Any]) -> str:
    terminal_class = str(summary.get("failure_terminal_class", "") or "").strip()
    if terminal_class == TerminalClass.BLOCKED_AUTH_FAILURE.value:
        return "blocked_auth_failure"
    if terminal_class == TerminalClass.BLOCKED_NO_RUNNER.value:
        return "blocked_no_runner"
    return "blocked"


def _preflight_failure_reason(issue_number: int, summary: Mapping[str, Any]) -> str:
    check_type = str(summary.get("check_type", "") or "").strip() or "scratch"
    terminal_class = str(summary.get("failure_terminal_class", "") or "").strip() or "blocked"
    failed_checks = [
        item for item in list(summary.get("failed_checks", []) or []) if isinstance(item, Mapping)
    ]
    detail = "; ".join(
        f"{str(item.get('name', '') or 'check').strip()}: "
        f"{str(item.get('detail', '') or 'failed').strip()}"
        for item in failed_checks[:2]
    ).strip("; ")
    if detail:
        return (
            f"Issue #{issue_number} failed `{check_type}` preflight receipt admission "
            f"({terminal_class}): {detail}"
        )
    return f"Issue #{issue_number} failed `{check_type}` preflight receipt admission ({terminal_class})."


def _preflight_next_action(summary: Mapping[str, Any]) -> str:
    check_type = str(summary.get("check_type", "") or "").strip() or "scratch"
    terminal_class = str(summary.get("failure_terminal_class", "") or "").strip()
    if terminal_class == TerminalClass.BLOCKED_AUTH_FAILURE.value:
        if check_type == "remote_publish":
            return (
                "Restore git/GitHub publish authentication and refresh the remote publish "
                "preflight receipt before redispatch."
            )
        return "Restore the required runner/git credentials and refresh the scratch preflight receipt before redispatch."
    if terminal_class == TerminalClass.BLOCKED_NO_RUNNER.value:
        return "Ensure the selected runner CLI is installed and discoverable, then refresh the preflight receipt before redispatch."
    if check_type == "remote_publish":
        return "Fix the branch push or draft-PR path and refresh the remote publish preflight receipt before redispatch."
    return "Fix the repo write/commit path and refresh the scratch preflight receipt before redispatch."


def _run_dispatch_preflight_receipts(
    loop: Any,
    *,
    repo_root: Path,
    envelope: CredentialEnvelope,
    target_agent: str,
    contract_path: Path,
) -> list[PreflightReceipt]:
    receipts = [
        run_contract_preflight_receipt(
            repo_root=repo_root,
            agent=target_agent,
            base_ref=str(loop.config.target_branch or "main"),
            skip_publication=True,
            contract_path=contract_path,
            envelope=envelope,
        )
    ]
    if bool(loop.config.auto_publish_deliverables):
        receipts.append(
            run_contract_preflight_receipt(
                repo_root=repo_root,
                agent=target_agent,
                base_ref=str(loop.config.target_branch or "main"),
                skip_publication=False,
                contract_path=contract_path,
                envelope=envelope,
            )
        )
    return receipts


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
    required_receipts: list[str] = []
    preflight_receipts: list[dict[str, Any]] = []
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
    preview_contract: WorkerContract | None = None
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
            if preview_contract is None:
                preview_contract = contract
        except ValueError:
            contract_valid = False
            break

    if contract_valid and not missing_slices:
        required_receipts = _required_preflight_receipts(loop)
        try:
            if preview_contract is None:
                raise RuntimeError("dispatch preview contract missing")
            contract_path = _persist_preview_contract(
                repo_root=Path.cwd(),
                issue_number=issue.number,
                contract=preview_contract,
            )
            receipt_payloads = _run_dispatch_preflight_receipts(
                loop,
                repo_root=Path.cwd(),
                envelope=envelope,
                target_agent=target_agent,
                contract_path=contract_path,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed on receipt routing issues
            detail = str(exc or "").strip() or "preflight receipt generation failed"
            preflight_receipts = [_synthetic_preflight_summary(required_receipts[0], detail)]
        else:
            preflight_receipts = [
                _preflight_receipt_summary(receipt) for receipt in receipt_payloads
            ]
            if all(bool(item.get("passed", False)) for item in preflight_receipts):
                return None

    reasons, next_actions = _gate_reason(
        issue_number=issue.number,
        target_agent=target_agent,
        missing_slices=missing_slices,
        contract_valid=contract_valid,
    )
    failed_receipts = [
        summary for summary in preflight_receipts if not bool(summary.get("passed", False))
    ]
    if failed_receipts:
        reasons.extend(
            _preflight_failure_reason(issue.number, summary) for summary in failed_receipts
        )
        next_actions.extend(_preflight_next_action(summary) for summary in failed_receipts)
    outcome = "blocked_auth_failure" if missing_slices else "blocked"
    if failed_receipts:
        outcome = _preflight_failure_outcome(failed_receipts[0])
    if claimed_runner_id:
        loop._release_runner_claim(claimed_runner_id)
    return {
        "status": "needs_human",
        "outcome": outcome,
        "reasons": reasons,
        "next_actions": list(dict.fromkeys(next_actions)),
        "dispatch_contract": {
            "target_agent": target_agent,
            "contract_valid": contract_valid,
            "missing_slices": missing_slices,
            "credential_envelope": envelope.preflight_cache_payload(),
            "required_receipts": required_receipts,
            "preflight_receipts": preflight_receipts,
        },
    }
