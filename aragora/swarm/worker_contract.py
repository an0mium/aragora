from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Mapping

from aragora.pipeline.execution_mode import ExecutionMode
from aragora.swarm.mission import normalize_context_policies
from aragora.swarm.worker_process import LaunchConfig


_CONTRACT_VERSION = "1"
_ENV_CHECKSUM_KEYS = (
    "ARAGORA_",
    "CLAUDE_",
    "CODEX_",
    "GH_",
    "GITHUB_",
)


def checksum_contract_payload(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


@dataclass(slots=True)
class WorkerContract:
    runner_type: str
    agent: str
    model: str
    profile: str
    permissions: dict[str, Any]
    execution_mode: str
    git_auth_mode: str
    gh_api_auth_mode: str
    budget: dict[str, Any]
    env_checksum: str
    mission_id: str = ""
    stage_id: str = ""
    assertion_ids: list[str] | None = None
    evidence_expectations: list[str] | None = None
    mission_context_policy: dict[str, Any] | None = None
    contract_version: str = _CONTRACT_VERSION
    _expected_checksum: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self._expected_checksum = self.checksum()

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_type": self.runner_type,
            "agent": self.agent,
            "model": self.model,
            "profile": self.profile,
            "permissions": dict(self.permissions),
            "execution_mode": self.execution_mode,
            "git_auth_mode": self.git_auth_mode,
            "gh_api_auth_mode": self.gh_api_auth_mode,
            "budget": dict(self.budget),
            "env_checksum": self.env_checksum,
            "mission_id": str(self.mission_id or "").strip(),
            "stage_id": str(self.stage_id or "").strip(),
            "assertion_ids": [
                str(item).strip() for item in list(self.assertion_ids or []) if str(item).strip()
            ],
            "evidence_expectations": [
                str(item).strip()
                for item in list(self.evidence_expectations or [])
                if str(item).strip()
            ],
            "mission_context_policy": dict(self.mission_context_policy or {}),
            "contract_version": self.contract_version,
        }

    def checksum(self) -> str:
        return checksum_contract_payload(self.to_dict())

    def validate(self) -> None:
        required = {
            "runner_type": self.runner_type,
            "agent": self.agent,
            "model": self.model,
            "profile": self.profile,
            "permissions": self.permissions,
            "execution_mode": self.execution_mode,
            "git_auth_mode": self.git_auth_mode,
            "gh_api_auth_mode": self.gh_api_auth_mode,
            "budget": self.budget,
            "env_checksum": self.env_checksum,
            "mission_context_policy": self.mission_context_policy,
            "contract_version": self.contract_version,
        }
        missing = [key for key, value in required.items() if value is None or value == ""]
        if missing:
            raise ValueError(f"Worker contract missing required fields: {', '.join(missing)}")
        if not dict(self.mission_context_policy or {}):
            raise ValueError("Worker contract missing required field: mission_context_policy")

    def admission_check(self) -> bool:
        """Check contract completeness and integrity for dispatch admission.

        Unlike ``validate()`` which raises ``ValueError`` on missing fields,
        this method returns a bool and additionally performs **checksum drift
        detection** — comparing the current contract checksum against the
        checksum captured at construction time.

        Fail-closed: returns ``False`` on any error.  Pure: no I/O, no
        network, no subprocess, no state mutation.
        """
        try:
            # 1. Verify all required fields are non-empty and non-None.
            required_fields = (
                self.runner_type,
                self.agent,
                self.model,
                self.profile,
                self.execution_mode,
                self.git_auth_mode,
                self.gh_api_auth_mode,
                self.budget,
                self.env_checksum,
                self.contract_version,
            )
            for value in required_fields:
                if value is None or value == "":
                    return False
            if not dict(self.mission_context_policy or {}):
                return False

            # 2. Detect checksum drift.
            current_checksum = checksum_contract_payload(self.to_dict())
            if current_checksum != self._expected_checksum:
                return False

            return True
        except Exception:  # noqa: BLE001 — fail-closed by design
            return False


def _env_checksum(env: Mapping[str, str] | None) -> str:
    snapshot: dict[str, str] = {}
    for key, value in dict(env or os.environ).items():
        if any(key.startswith(prefix) for prefix in _ENV_CHECKSUM_KEYS):
            snapshot[key] = value
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_auth_mode(repo_path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "--push", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "unknown"
    remote = (result.stdout or "").strip()
    if remote.startswith("git@"):
        return "ssh"
    if remote.startswith("https://"):
        return "https"
    if remote:
        return "unknown"
    return "unset"


def _gh_api_auth_mode(env: Mapping[str, str] | None) -> str:
    snapshot = dict(env or os.environ)
    token = snapshot.get("GH_TOKEN") or snapshot.get("GITHUB_TOKEN") or ""
    if not token:
        return "none"
    if snapshot.get("GITHUB_APP_ID") or snapshot.get("GITHUB_APP_INSTALLATION_ID"):
        return "app"
    return "user"


def build_worker_contract(
    *,
    agent: str,
    config: LaunchConfig,
    worktree_path: str,
    env: Mapping[str, str] | None = None,
    work_order: Mapping[str, Any] | None = None,
) -> WorkerContract:
    normalized_agent = str(agent or "").strip() or "unknown"
    if normalized_agent == "claude":
        model = str(config.claude_model or "default").strip() or "default"
        profile = str(config.claude_profile or "default").strip() or "default"
        permissions = {
            "allow_dangerous_permissions": bool(config.allow_claude_dangerously_skip_permissions),
        }
        runner_type = "claude-cli"
    elif normalized_agent == "codex":
        model = str(config.codex_model or "default").strip() or "default"
        profile = "default"
        permissions = {"allow_full_auto": bool(config.allow_codex_full_auto)}
        runner_type = "codex-cli"
    else:
        model = "default"
        profile = "default"
        permissions = {}
        runner_type = "cli"

    budget = {
        "max_wall_time_seconds": float(config.timeout_seconds),
        "no_progress_timeout_seconds": float(config.no_progress_timeout_seconds),
        "max_turns": None,
        "max_tokens": None,
        "max_retries": None,
    }
    work_order_data = dict(work_order or {})
    file_scope = [
        str(item).strip() for item in work_order_data.get("file_scope", []) if str(item).strip()
    ]
    evidence_expectations = [
        str(item).strip()
        for item in work_order_data.get("evidence_expectations", [])
        if str(item).strip()
    ]
    if not evidence_expectations:
        evidence_expectations = [
            "worker_contract",
            "worker_contract_checksum",
            *[
                str(item).strip()
                for item in work_order_data.get("expected_tests", [])
                if str(item).strip()
            ],
        ]
    context_policy = normalize_context_policies(
        work_order_data.get("mission_context_policies"),
        file_scope=file_scope,
        evidence_expectations=evidence_expectations,
    )["worker"]

    execution_mode = (
        config.execution_mode.value
        if isinstance(config.execution_mode, ExecutionMode)
        else str(config.execution_mode)
    )

    return WorkerContract(
        runner_type=runner_type,
        agent=normalized_agent,
        model=model,
        profile=profile,
        permissions=permissions,
        execution_mode=execution_mode,
        git_auth_mode=_git_auth_mode(worktree_path),
        gh_api_auth_mode=_gh_api_auth_mode(env),
        budget=budget,
        env_checksum=_env_checksum(env),
        mission_id=str(work_order_data.get("mission_id", "") or "").strip(),
        stage_id=str(work_order_data.get("stage_id", "") or "").strip(),
        assertion_ids=[
            str(item).strip()
            for item in work_order_data.get("assertion_ids", [])
            if str(item).strip()
        ],
        evidence_expectations=evidence_expectations,
        mission_context_policy=context_policy,
    )
