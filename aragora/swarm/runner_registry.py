"""Generic local runner registration and Boss routing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

from aragora.rbac.models import AuthorizationContext
from aragora.swarm.worker_launcher import LaunchConfig

UTC = timezone.utc
VERIFIED_AUTH_MODES = {"chatgpt_login", "api_key", "subscription"}
DEFAULT_RUNNER_STALE_AFTER_SECONDS = 3600
DEFAULT_RUNNER_ROTATION_INTERVAL_SECONDS = 1800.0


@dataclass(frozen=True, slots=True)
class RunnerTypeSpec:
    runner_type: str
    cli_name: str
    path_attr: str | None
    auth_probe: tuple[str, ...] | None
    env_auth_keys: tuple[str, ...]
    default_cost_class: str
    priority_weight: int
    supports_review: bool = True
    supports_login_status: bool = True


RUNNER_TYPE_SPECS: dict[str, RunnerTypeSpec] = {
    "claude": RunnerTypeSpec(
        runner_type="claude",
        cli_name="claude",
        path_attr="claude_path",
        auth_probe=("login", "status"),
        env_auth_keys=("ANTHROPIC_API_KEY",),
        default_cost_class="subscription",
        priority_weight=100,
    ),
    "codex": RunnerTypeSpec(
        runner_type="codex",
        cli_name="codex",
        path_attr="codex_path",
        auth_probe=("login", "status"),
        env_auth_keys=("OPENAI_API_KEY",),
        default_cost_class="subscription",
        priority_weight=80,
    ),
    "gemini-cli": RunnerTypeSpec(
        runner_type="gemini-cli",
        cli_name="gemini",
        path_attr=None,
        auth_probe=None,
        env_auth_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        default_cost_class="api",
        priority_weight=50,
        supports_review=False,
        supports_login_status=False,
    ),
    "openai": RunnerTypeSpec(
        runner_type="openai",
        cli_name="openai",
        path_attr=None,
        auth_probe=None,
        env_auth_keys=("OPENAI_API_KEY",),
        default_cost_class="api",
        priority_weight=45,
        supports_review=False,
        supports_login_status=False,
    ),
    "grok-cli": RunnerTypeSpec(
        runner_type="grok-cli",
        cli_name="grok",
        path_attr=None,
        auth_probe=None,
        env_auth_keys=("XAI_API_KEY", "GROK_API_KEY"),
        default_cost_class="api",
        priority_weight=40,
        supports_review=False,
        supports_login_status=False,
    ),
}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _env_flag_int(env: dict[str, str], key: str, default: int) -> int:
    raw = str(env.get(key, "")).strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _optional_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalized_runner_type(value: str | None) -> str:
    runner_type = _text(value).lower()
    return runner_type or "codex"


def _runner_type_spec(value: str | None) -> RunnerTypeSpec:
    runner_type = _normalized_runner_type(value)
    spec = RUNNER_TYPE_SPECS.get(runner_type)
    if spec is None:
        raise ValueError(f"Unsupported runner type: {runner_type!r}")
    return spec


@dataclass(slots=True)
class RunnerInspection:
    runner_id: str
    runner_type: str
    availability: str
    available: bool
    auth_mode: str
    command_path: str | None = None
    version: str | None = None
    status_summary: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    owner_binding: dict[str, Any] = field(default_factory=dict)
    registered: bool = False
    registry_path: str | None = None
    registered_at: str | None = None
    heartbeat_at: str | None = None
    freshness_status: str = "unknown"
    stale_after_seconds: int = DEFAULT_RUNNER_STALE_AFTER_SECONDS
    next_action: str | None = None
    cost_class: str = "local"
    priority_weight: int = 0
    codex_path: str | None = None  # legacy compatibility for older callers/tests
    profile: str | None = None

    def __post_init__(self) -> None:
        if self.command_path is None and self.codex_path is not None:
            self.command_path = self.codex_path
        if (
            self.codex_path is None
            and self.runner_type == "codex"
            and self.command_path is not None
        ):
            self.codex_path = self.command_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner_id": self.runner_id,
            "runner_type": self.runner_type,
            "availability": self.availability,
            "available": self.available,
            "auth_mode": self.auth_mode,
            "command_path": self.command_path,
            "codex_path": self.codex_path,
            "version": self.version,
            "status_summary": self.status_summary,
            "capabilities": dict(self.capabilities),
            "owner_binding": dict(self.owner_binding),
            "registered": self.registered,
            "registry_path": self.registry_path,
            "registered_at": self.registered_at,
            "heartbeat_at": self.heartbeat_at,
            "freshness_status": self.freshness_status,
            "stale_after_seconds": self.stale_after_seconds,
            "next_action": self.next_action,
            "cost_class": self.cost_class,
            "priority_weight": self.priority_weight,
            "profile": self.profile,
        }


CodexRunnerInspection = RunnerInspection


@dataclass(slots=True)
class BossRoutingDecision:
    owner_binding: dict[str, Any]
    selected_runner_ids: list[str] = field(default_factory=list)
    selected_runners: list[dict[str, Any]] = field(default_factory=list)
    selection_basis: str = ""
    blocked_reason: str | None = None
    rejected_runner_ids: list[str] = field(default_factory=list)
    next_action: str | None = None
    requested_runner_type: str | None = None
    fallback_reason: str | None = None

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_binding": dict(self.owner_binding),
            "selected_runner_ids": list(self.selected_runner_ids),
            "selected_runners": [dict(item) for item in self.selected_runners],
            "selection_basis": self.selection_basis,
            "blocked_reason": self.blocked_reason,
            "rejected_runner_ids": list(self.rejected_runner_ids),
            "next_action": self.next_action,
            "requested_runner_type": self.requested_runner_type,
            "fallback_reason": self.fallback_reason,
        }


class CLIRunnerInspector:
    """Inspect local CLI runner availability and auth state truthfully."""

    def __init__(
        self,
        *,
        runner_type: str = "codex",
        config: LaunchConfig | None = None,
        env: dict[str, str] | None = None,
        profile: str | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config = config or LaunchConfig()
        self.env = dict(os.environ if env is None else env)
        self.spec = _runner_type_spec(runner_type)
        self.profile = _text(profile) or None
        self.repo_root = Path(repo_root or Path.cwd()).resolve()

    def inspect(self) -> RunnerInspection:
        command_name = self._command_name()
        base_command_path = shutil.which(command_name)
        command_path = self._launch_command_path(base_command_path)
        runner_id = self._runner_id(command_path, command_name=command_name)
        owner = owner_binding_from_env(self.env)
        stale_after_seconds = self._stale_after_seconds()

        if not base_command_path or not command_path:
            return RunnerInspection(
                runner_id=runner_id,
                runner_type=self.spec.runner_type,
                availability="unavailable",
                available=False,
                auth_mode="unavailable",
                command_path=None,
                version=None,
                status_summary=None,
                capabilities=self._capabilities(False, None),
                owner_binding=owner,
                freshness_status="unavailable",
                stale_after_seconds=stale_after_seconds,
                next_action=self._next_action(
                    available=False,
                    auth_mode="unavailable",
                    owner_binding=owner,
                ),
                cost_class=self.spec.default_cost_class,
                priority_weight=self.spec.priority_weight,
                profile=self.profile,
            )

        version_result = self._version_result(base_command_path)
        help_result = self._help_result(base_command_path)
        auth_result = self._auth_result(base_command_path)

        help_text = "\n".join(
            part for part in (help_result.get("stdout", ""), help_result.get("stderr", "")) if part
        ).strip()
        auth_text = "\n".join(
            part for part in (auth_result.get("stdout", ""), auth_result.get("stderr", "")) if part
        ).strip()

        available = bool(
            help_result.get("returncode") == 0
            or version_result.get("returncode") == 0
            or help_text
            or version_result.get("stdout")
            or version_result.get("stderr")
        )
        if not available:
            return RunnerInspection(
                runner_id=runner_id,
                runner_type=self.spec.runner_type,
                availability="unavailable",
                available=False,
                auth_mode="unavailable",
                command_path=command_path,
                version=None,
                status_summary=None,
                capabilities=self._capabilities(False, None),
                owner_binding=owner,
                freshness_status="unavailable",
                stale_after_seconds=stale_after_seconds,
                next_action=self._next_action(
                    available=False,
                    auth_mode="unavailable",
                    owner_binding=owner,
                ),
                cost_class=self.spec.default_cost_class,
                priority_weight=self.spec.priority_weight,
                profile=self.profile,
            )

        auth_mode = self._classify_auth_mode(auth_text)
        version = self._first_line(
            version_result.get("stdout") or version_result.get("stderr") or ""
        )
        return RunnerInspection(
            runner_id=runner_id,
            runner_type=self.spec.runner_type,
            availability="available",
            available=True,
            auth_mode=auth_mode,
            command_path=command_path,
            version=version or None,
            status_summary=self._status_summary(auth_text) or None,
            capabilities=self._capabilities(True, help_text),
            owner_binding=owner,
            freshness_status=self._inspection_freshness(available=True, auth_mode=auth_mode),
            stale_after_seconds=stale_after_seconds,
            next_action=self._next_action(
                available=True,
                auth_mode=auth_mode,
                owner_binding=owner,
            ),
            cost_class=self._cost_class(auth_mode),
            priority_weight=self.spec.priority_weight,
            profile=self.profile,
        )

    def _command_name(self) -> str:
        attr = self.spec.path_attr
        if attr and hasattr(self.config, attr):
            return str(getattr(self.config, attr))
        return self.spec.cli_name

    def _auth_result(self, command_path: str) -> dict[str, Any]:
        if self.spec.runner_type == "claude" and self.profile:
            script = self._claude_profile_script()
            if script is None:
                return {"returncode": 1, "stdout": "", "stderr": "claude_profile.sh not found"}
            return self._run_command([script, "status", self.profile])
        if self.spec.auth_probe:
            return self._run_command([command_path, *self.spec.auth_probe])
        return {
            "returncode": 0
            if any(_text(self.env.get(key)) for key in self.spec.env_auth_keys)
            else 1,
            "stdout": "",
            "stderr": "",
        }

    def _capabilities(self, available: bool, help_text: str | None) -> dict[str, Any]:
        text = (help_text or "").lower()
        supports_exec = available and (
            "exec" in text or self.spec.runner_type in {"claude", "codex"}
        )
        supports_review = available and ("review" in text or self.spec.supports_review)
        max_parallel_key = (
            f"ARAGORA_{self.spec.runner_type.upper().replace('-', '_')}_RUNNER_MAX_CONCURRENCY"
        )
        return {
            "supports_exec": supports_exec,
            "supports_review": supports_review,
            "supports_login_status": available and self.spec.supports_login_status,
            "max_parallel_lanes": _env_flag_int(self.env, max_parallel_key, 1),
            "active_lanes": _env_flag_int(self.env, f"{max_parallel_key}_ACTIVE", 0),
        }

    def _classify_auth_mode(self, auth_text: str) -> str:
        lowered = auth_text.lower()
        if self.spec.runner_type == "codex":
            if "chatgpt" in lowered and "logged in" in lowered:
                return "chatgpt_login"
            if "api key" in lowered and "logged in" in lowered:
                return "api_key"
            return "unknown"
        if self.spec.runner_type == "claude":
            if "api key" in lowered and ("logged in" in lowered or "authenticated" in lowered):
                return "api_key"
            if (
                "logged in" in lowered
                or "subscription" in lowered
                or "max" in lowered
                or "pro" in lowered
            ):
                return "subscription"
            return "unknown"
        if any(_text(self.env.get(key)) for key in self.spec.env_auth_keys):
            return "api_key"
        return "unknown"

    def _cost_class(self, auth_mode: str) -> str:
        if auth_mode == "api_key":
            return "api"
        if auth_mode in {"chatgpt_login", "subscription"}:
            return "subscription"
        return self.spec.default_cost_class

    def _stale_after_seconds(self) -> int:
        return _env_flag_int(
            self.env,
            "ARAGORA_RUNNER_STALE_AFTER_SECONDS",
            DEFAULT_RUNNER_STALE_AFTER_SECONDS,
        )

    @staticmethod
    def _inspection_freshness(*, available: bool, auth_mode: str) -> str:
        if not available:
            return "unavailable"
        if auth_mode not in VERIFIED_AUTH_MODES:
            return "unknown"
        return "fresh"

    @staticmethod
    def _first_line(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    @classmethod
    def _status_summary(cls, text: str) -> str:
        first = cls._first_line(text)
        if not first.startswith("{"):
            return first
        try:
            payload = json.loads(first if len(text.splitlines()) == 1 else text)
        except ValueError:
            return first
        if not isinstance(payload, dict):
            return first
        parts = []
        if "loggedIn" in payload:
            parts.append(f"loggedIn={payload.get('loggedIn')}")
        auth_method = _text(payload.get("authMethod"))
        if auth_method:
            parts.append(f"authMethod={auth_method}")
        subscription = _text(payload.get("subscriptionType"))
        if subscription:
            parts.append(f"subscriptionType={subscription}")
        return " ".join(parts) or first

    def _runner_id(self, command_path: str | None, *, command_name: str) -> str:
        identity = f"{self.spec.runner_type}:{platform.node()}:{command_path or command_name}"
        if self.profile:
            identity = f"{identity}:{self.profile}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return f"{self.spec.runner_type}-runner-{digest}"

    def _launch_command_path(self, base_command_path: str | None) -> str | None:
        if self.spec.runner_type == "claude" and self.profile:
            return self._claude_profile_script()
        return base_command_path

    def _version_result(self, base_command_path: str) -> dict[str, Any]:
        return self._run_command([base_command_path, "--version"])

    def _help_result(self, base_command_path: str) -> dict[str, Any]:
        return self._run_command([base_command_path, "--help"])

    def _claude_profile_script(self) -> str | None:
        script = (self.repo_root / "scripts" / "claude_profile.sh").resolve()
        if not script.exists():
            return None
        return str(script)

    @staticmethod
    def _run_command(command: list[str]) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return {"returncode": 1, "stdout": "", "stderr": ""}
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    def _next_action(
        self, *, available: bool, auth_mode: str, owner_binding: dict[str, Any]
    ) -> str:
        runner_label = self.spec.runner_type
        if self.profile:
            runner_label = f"{runner_label}:{self.profile}"
        if not available:
            return f"Install the {runner_label} CLI or add `{self._command_name()}` to PATH before registering this runner."
        if auth_mode == "unknown":
            return f"Confirm the local {runner_label} CLI login state before relying on this runner for routed work."
        if not owner_binding.get("user_id"):
            return (
                "Set `ARAGORA_USER_ID` and optional `ARAGORA_WORKSPACE_ID` before registering this "
                "runner to an Aragora owner context."
            )
        return (
            "Runner is eligible for future Boss-mode routing under the bound Aragora owner context."
        )


class CodexRunnerInspector(CLIRunnerInspector):
    def __init__(
        self, *, config: LaunchConfig | None = None, env: dict[str, str] | None = None
    ) -> None:
        super().__init__(runner_type="codex", config=config, env=env)


class ClaudeRunnerInspector(CLIRunnerInspector):
    def __init__(
        self,
        *,
        config: LaunchConfig | None = None,
        env: dict[str, str] | None = None,
        profile: str | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        super().__init__(
            runner_type="claude",
            config=config,
            env=env,
            profile=profile,
            repo_root=repo_root,
        )


def make_runner_inspector(
    runner_type: str,
    *,
    config: LaunchConfig | None = None,
    env: dict[str, str] | None = None,
    profile: str | None = None,
    repo_root: str | Path | None = None,
) -> CLIRunnerInspector:
    normalized = _normalized_runner_type(runner_type)
    if normalized == "codex":
        return CodexRunnerInspector(config=config, env=env)
    if normalized == "claude":
        return ClaudeRunnerInspector(config=config, env=env, profile=profile, repo_root=repo_root)
    return CLIRunnerInspector(
        runner_type=normalized,
        config=config,
        env=env,
        profile=profile,
        repo_root=repo_root,
    )


def configured_claude_runner_profiles(env: dict[str, str] | None = None) -> list[str]:
    values = dict(os.environ if env is None else env)
    raw = _text(values.get("ARAGORA_CLAUDE_RUNNER_PROFILES"))
    if not raw:
        raw = _text(values.get("ARAGORA_CLAUDE_REVIEW_PROFILES"))
    if not raw:
        return []
    profiles: list[str] = []
    for item in raw.split(","):
        normalized = _text(item)
        if normalized and normalized not in profiles:
            profiles.append(normalized)
    return profiles


def discover_runner_inspections(
    runner_type: str,
    *,
    config: LaunchConfig | None = None,
    env: dict[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> list[RunnerInspection]:
    normalized = _normalized_runner_type(runner_type)
    if normalized == "claude":
        profiles = configured_claude_runner_profiles(env)
        if profiles:
            return [
                make_runner_inspector(
                    normalized,
                    config=config,
                    env=env,
                    profile=profile,
                    repo_root=repo_root,
                ).inspect()
                for profile in profiles
            ]
    return [
        make_runner_inspector(
            normalized,
            config=config,
            env=env,
            repo_root=repo_root,
        ).inspect()
    ]


class LocalRunnerRegistry:
    """Minimal local registry for user-owned CLI runners."""

    def __init__(self, *, path: str | Path | None = None) -> None:
        raw = (
            path
            or os.environ.get("ARAGORA_RUNNER_REGISTRY_PATH")
            or "~/.aragora/swarm_runners.json"
        )
        self.path = Path(raw).expanduser()

    def register(
        self,
        inspection: RunnerInspection,
        *,
        owner_context: AuthorizationContext | None,
    ) -> RunnerInspection:
        registry_path = str(self.path)
        runner_label = inspection.runner_type
        if not inspection.available:
            inspection.registered = False
            inspection.registry_path = registry_path
            inspection.next_action = inspection.next_action or (
                f"Restore local {runner_label} runner availability before registering this runner."
            )
            return inspection
        if inspection.auth_mode == "unknown":
            inspection.registered = False
            inspection.registry_path = registry_path
            inspection.next_action = (
                f"Registration blocked: {runner_label} auth mode is unknown. Confirm the local "
                f"{runner_label} login state before registering this runner for Boss-mode routing."
            )
            return inspection
        if owner_context is None:
            inspection.registered = False
            inspection.registry_path = registry_path
            inspection.next_action = (
                "Set `ARAGORA_USER_ID` and optional `ARAGORA_WORKSPACE_ID` before registering this "
                "runner to an Aragora owner context."
            )
            return inspection

        records = self._load()
        now = _utcnow()
        freshness_status = self._freshness_for_inspection(inspection)
        entry = {
            **inspection.to_dict(),
            "owner_binding": owner_binding_from_context(owner_context),
            "registered": True,
            "registered_at": now,
            "heartbeat_at": now,
            "freshness_status": freshness_status,
            "updated_at": now,
        }
        records["registrations"] = [
            item
            for item in records.get("registrations", [])
            if isinstance(item, dict) and item.get("runner_id") != inspection.runner_id
        ]
        records["registrations"].append(entry)
        self._save(records)

        inspection.owner_binding = owner_binding_from_context(owner_context)
        inspection.registered = True
        inspection.registry_path = registry_path
        inspection.registered_at = now
        inspection.heartbeat_at = now
        inspection.freshness_status = freshness_status
        inspection.next_action = (
            "Runner registered. Future Boss-mode routing can target this owner-bound runner."
        )
        return inspection

    def heartbeat(
        self,
        inspection: RunnerInspection,
        *,
        owner_context: AuthorizationContext | None,
    ) -> RunnerInspection:
        registry_path = str(self.path)
        inspection.registry_path = registry_path
        if owner_context is None:
            inspection.registered = False
            inspection.next_action = (
                "Set `ARAGORA_USER_ID` and `ARAGORA_WORKSPACE_ID` before refreshing a runner "
                "heartbeat for Boss-mode routing."
            )
            return inspection

        records = self._load()
        registrations = [
            dict(item) for item in records.get("registrations", []) if isinstance(item, dict)
        ]
        existing = next(
            (
                item
                for item in registrations
                if item.get("runner_id") == inspection.runner_id
                and self._is_owner_compatible(item, owner_context=owner_context)
            ),
            None,
        )
        if existing is None:
            inspection.registered = False
            inspection.next_action = (
                "Register this runner for the current Aragora user/workspace context before "
                "refreshing its heartbeat."
            )
            return inspection

        now = _utcnow()
        freshness_status = self._freshness_for_inspection(inspection)
        owner_binding = owner_binding_from_context(owner_context)
        updated_entry = {
            **existing,
            **inspection.to_dict(),
            "registered": True,
            "owner_binding": owner_binding,
            "registered_at": existing.get("registered_at"),
            "heartbeat_at": now,
            "freshness_status": freshness_status,
            "updated_at": now,
        }
        records["registrations"] = [
            updated_entry if item.get("runner_id") == inspection.runner_id else item
            for item in registrations
        ]
        self._save(records)

        inspection.owner_binding = owner_binding
        inspection.registered = True
        inspection.registered_at = _text(existing.get("registered_at")) or None
        inspection.heartbeat_at = now
        inspection.freshness_status = freshness_status
        inspection.next_action = self._heartbeat_next_action(
            freshness_status, runner_type=inspection.runner_type
        )
        return inspection

    def list_registrations(self) -> list[dict[str, Any]]:
        return [
            dict(item) for item in self._load().get("registrations", []) if isinstance(item, dict)
        ]

    def resolve_boss_routing(
        self,
        *,
        owner_context: AuthorizationContext | None,
        requested_runner_type: str | None = None,
        allowed_profiles: set[str] | None = None,
        rotation_interval_seconds: float = DEFAULT_RUNNER_ROTATION_INTERVAL_SECONDS,
    ) -> BossRoutingDecision:
        owner_binding = owner_binding_from_context(owner_context)
        requested = (
            _normalized_runner_type(requested_runner_type) if requested_runner_type else None
        )
        allowed_profile_set = {item for item in (allowed_profiles or set()) if _text(item)}
        selection_basis = (
            "registered=true, freshness_status=fresh, availability=available, auth_mode verified, "
            "owner_binding compatible, capacity available, ordered by requested runner type, "
            "priority_weight, cost preference, and rotation-aware profile balancing"
        )
        if owner_context is None or not _text(owner_context.user_id):
            return BossRoutingDecision(
                owner_binding=owner_binding,
                selection_basis=selection_basis,
                blocked_reason="missing_owner_context",
                next_action=(
                    "Set `ARAGORA_USER_ID` and `ARAGORA_WORKSPACE_ID` before running Boss mode so "
                    "Aragora can route only onto authorized registered runners."
                ),
                requested_runner_type=requested,
            )

        eligible: list[dict[str, Any]] = []
        rejected_runner_ids: list[str] = []
        saw_compatible_nonfresh = False
        saw_requested_type = False
        now = datetime.now(UTC)
        for runner in self.list_registrations():
            runner_type = _text(runner.get("runner_type"))
            if allowed_profile_set and runner_type == "claude":
                runner_profile = _text(runner.get("profile"))
                if runner_profile not in allowed_profile_set:
                    runner_id = _text(runner.get("runner_id"))
                    if runner_id:
                        rejected_runner_ids.append(runner_id)
                    continue
            if self._is_owner_compatible(runner, owner_context=owner_context):
                if bool(runner.get("registered")) and self._freshness_status(runner) != "fresh":
                    saw_compatible_nonfresh = True
                if requested and runner_type == requested:
                    saw_requested_type = True
            if not self._is_runner_eligible(runner, owner_context=owner_context):
                runner_id = _text(runner.get("runner_id"))
                if runner_id:
                    rejected_runner_ids.append(runner_id)
                continue
            eligible.append(self._runner_summary(runner))

        if not eligible:
            return BossRoutingDecision(
                owner_binding=owner_binding,
                selection_basis=selection_basis,
                blocked_reason=(
                    "no_fresh_registered_runners"
                    if saw_compatible_nonfresh
                    else "no_eligible_registered_runners"
                ),
                rejected_runner_ids=sorted(set(rejected_runner_ids)),
                next_action=(
                    "Refresh the heartbeat for an available registered runner in this exact Aragora "
                    "user/workspace context before running Boss mode."
                    if saw_compatible_nonfresh
                    else "Register an available runner for this exact Aragora user/workspace context "
                    "before running Boss mode."
                ),
                requested_runner_type=requested,
            )

        eligible.sort(
            key=lambda item: (
                0 if requested and item["runner_type"] == requested else 1,
                -int(item.get("priority_weight", 0)),
                self._cost_rank(_text(item.get("cost_class"))),
                self._rotation_rank(
                    item,
                    now=now,
                    rotation_interval_seconds=rotation_interval_seconds,
                ),
                int(item.get("claimed_lanes", 0)),
                int(item.get("selection_count", 0)),
                self._last_selected_sort_key(item),
                -int(item.get("available_capacity", 0)),
                _text(item.get("runner_id")),
            )
        )

        fallback_reason = None
        if requested and eligible and eligible[0]["runner_type"] != requested:
            fallback_reason = (
                "requested_runner_type_unavailable"
                if saw_requested_type
                else "requested_runner_type_not_registered"
            )

        return BossRoutingDecision(
            owner_binding=owner_binding,
            selected_runner_ids=[item["runner_id"] for item in eligible],
            selected_runners=eligible,
            selection_basis=selection_basis,
            rejected_runner_ids=sorted(set(rejected_runner_ids)),
            next_action="Boss mode will route only through the selected registered runner set.",
            requested_runner_type=requested,
            fallback_reason=fallback_reason,
        )

    def claim_runner(
        self,
        runner_id: str,
        *,
        owner_context: AuthorizationContext | None,
    ) -> dict[str, Any] | None:
        normalized_runner_id = _text(runner_id)
        if not normalized_runner_id:
            return None
        records = self._load()
        registrations = [
            dict(item) for item in records.get("registrations", []) if isinstance(item, dict)
        ]
        updated_entry: dict[str, Any] | None = None
        for item in registrations:
            if _text(item.get("runner_id")) != normalized_runner_id:
                continue
            if owner_context is not None and not self._is_owner_compatible(
                item,
                owner_context=owner_context,
            ):
                return None
            claimed_lanes = _optional_int(item.get("claimed_lanes"))
            max_parallel = int(dict(item.get("capabilities") or {}).get("max_parallel_lanes") or 1)
            if self._effective_active_lanes(item) >= max_parallel:
                return None
            updated_entry = {
                **item,
                "claimed_lanes": claimed_lanes + 1,
                "last_selected_at": _utcnow(),
                "selection_count": _optional_int(item.get("selection_count")) + 1,
            }
            break
        if updated_entry is None:
            return None
        records["registrations"] = [
            updated_entry if _text(item.get("runner_id")) == normalized_runner_id else item
            for item in registrations
        ]
        self._save(records)
        return self._runner_summary(updated_entry)

    def release_runner_claim(
        self,
        runner_id: str,
        *,
        owner_context: AuthorizationContext | None = None,
    ) -> dict[str, Any] | None:
        normalized_runner_id = _text(runner_id)
        if not normalized_runner_id:
            return None
        records = self._load()
        registrations = [
            dict(item) for item in records.get("registrations", []) if isinstance(item, dict)
        ]
        updated_entry: dict[str, Any] | None = None
        for item in registrations:
            if _text(item.get("runner_id")) != normalized_runner_id:
                continue
            if owner_context is not None and not self._is_owner_compatible(
                item,
                owner_context=owner_context,
            ):
                return None
            updated_entry = {
                **item,
                "claimed_lanes": max(0, _optional_int(item.get("claimed_lanes")) - 1),
            }
            break
        if updated_entry is None:
            return None
        records["registrations"] = [
            updated_entry if _text(item.get("runner_id")) == normalized_runner_id else item
            for item in registrations
        ]
        self._save(records)
        return self._runner_summary(updated_entry)

    def _runner_summary(self, runner: dict[str, Any]) -> dict[str, Any]:
        capabilities = dict(runner.get("capabilities") or {})
        max_parallel = int(capabilities.get("max_parallel_lanes") or 1)
        claimed_lanes = _optional_int(runner.get("claimed_lanes"))
        active_lanes = self._effective_active_lanes(runner)
        return {
            "runner_id": _text(runner.get("runner_id")),
            "runner_type": _text(runner.get("runner_type")),
            "profile": _text(runner.get("profile")) or None,
            "auth_mode": _text(runner.get("auth_mode")),
            "cost_class": _text(runner.get("cost_class")) or "local",
            "priority_weight": int(runner.get("priority_weight") or 0),
            "freshness_status": self._freshness_status(runner),
            "heartbeat_at": _text(runner.get("heartbeat_at")) or None,
            "stale_after_seconds": int(
                runner.get("stale_after_seconds") or DEFAULT_RUNNER_STALE_AFTER_SECONDS
            ),
            "owner_binding": dict(runner.get("owner_binding") or {}),
            "capabilities": capabilities,
            "claimed_lanes": claimed_lanes,
            "selection_count": _optional_int(runner.get("selection_count")),
            "last_selected_at": _text(runner.get("last_selected_at")) or None,
            "available_capacity": max(0, max_parallel - active_lanes),
            "active_lanes": active_lanes,
            "command_path": _text(runner.get("command_path") or runner.get("codex_path")) or None,
        }

    def _is_runner_eligible(
        self,
        runner: dict[str, Any],
        *,
        owner_context: AuthorizationContext,
    ) -> bool:
        if not bool(runner.get("registered")):
            return False
        if _text(runner.get("availability")) != "available" or not bool(
            runner.get("available", True)
        ):
            return False
        if _text(runner.get("auth_mode")) not in VERIFIED_AUTH_MODES:
            return False
        if self._freshness_status(runner) != "fresh":
            return False
        if not self._is_owner_compatible(runner, owner_context=owner_context):
            return False
        capabilities = dict(runner.get("capabilities") or {})
        max_parallel = int(capabilities.get("max_parallel_lanes") or 1)
        active_lanes = self._effective_active_lanes(runner)
        if active_lanes >= max_parallel:
            return False
        return True

    @staticmethod
    def _effective_active_lanes(runner: dict[str, Any]) -> int:
        capabilities = dict(runner.get("capabilities") or {})
        base_active = _optional_int(capabilities.get("active_lanes") or runner.get("active_lanes"))
        return base_active + _optional_int(runner.get("claimed_lanes"))

    @staticmethod
    def _last_selected_sort_key(runner: dict[str, Any]) -> float:
        value = _text(runner.get("last_selected_at"))
        if not value:
            return 0.0
        parsed = LocalRunnerRegistry._parse_timestamp(value)
        if parsed is None:
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()

    @staticmethod
    def _rotation_rank(
        runner: dict[str, Any],
        *,
        now: datetime,
        rotation_interval_seconds: float,
    ) -> tuple[int, float]:
        value = _text(runner.get("last_selected_at"))
        if not value:
            return (0, 0.0)
        parsed = LocalRunnerRegistry._parse_timestamp(value)
        if parsed is None:
            return (0, 0.0)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        age_seconds = max(0.0, (now - parsed).total_seconds())
        hot = 1 if age_seconds < rotation_interval_seconds else 0
        return (hot, -age_seconds)

    @staticmethod
    def _is_owner_compatible(
        runner: dict[str, Any],
        *,
        owner_context: AuthorizationContext,
    ) -> bool:
        owner_binding = dict(runner.get("owner_binding") or {})
        if _text(owner_binding.get("user_id")) != _text(owner_context.user_id):
            return False

        runner_workspace = _text(owner_binding.get("workspace_id"))
        context_workspace = _text(owner_context.workspace_id)
        if runner_workspace != context_workspace:
            return False

        runner_org = _text(owner_binding.get("org_id"))
        context_org = _text(owner_context.org_id)
        if runner_org and context_org and runner_org != context_org:
            return False

        return True

    @staticmethod
    def _freshness_for_inspection(inspection: RunnerInspection) -> str:
        if not inspection.available:
            return "unavailable"
        if inspection.auth_mode not in VERIFIED_AUTH_MODES:
            return "unknown"
        return "fresh"

    def _freshness_status(self, runner: dict[str, Any]) -> str:
        if _text(runner.get("availability")) != "available" or not bool(
            runner.get("available", True)
        ):
            return "unavailable"
        if _text(runner.get("auth_mode")) not in VERIFIED_AUTH_MODES:
            return "unknown"

        heartbeat_at = _text(runner.get("heartbeat_at"))
        if not heartbeat_at:
            return "stale"
        heartbeat_dt = self._parse_timestamp(heartbeat_at)
        if heartbeat_dt is None:
            return "unknown"

        stale_after_seconds = int(
            runner.get("stale_after_seconds") or DEFAULT_RUNNER_STALE_AFTER_SECONDS
        )
        age_seconds = max(0.0, (datetime.now(UTC) - heartbeat_dt).total_seconds())
        if age_seconds > stale_after_seconds:
            return "stale"
        return "fresh"

    @staticmethod
    def _heartbeat_next_action(freshness_status: str, *, runner_type: str) -> str:
        if freshness_status == "fresh":
            return (
                "Runner heartbeat refreshed. Boss mode can route work to this runner while the "
                "heartbeat remains fresh."
            )
        if freshness_status == "unavailable":
            return (
                f"Runner heartbeat recorded an unavailable {runner_type} runner. Restore the local "
                "CLI before relying on Boss-mode routing."
            )
        return (
            "Runner heartbeat recorded a non-fresh state. Confirm local availability and auth "
            "before relying on Boss-mode routing."
        )

    @staticmethod
    def _cost_rank(cost_class: str) -> int:
        ranks = {"subscription": 0, "local": 1, "api": 2}
        return ranks.get(cost_class, 3)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"registrations": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"registrations": []}
        registrations = data.get("registrations", [])
        if not isinstance(registrations, list):
            registrations = []
        return {"registrations": registrations}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.path)


def authorization_context_from_env(
    env: dict[str, str] | None = None,
) -> AuthorizationContext | None:
    values = dict(os.environ if env is None else env)
    user_id = _text(values.get("ARAGORA_USER_ID") or values.get("ARAGORA_ACTOR_ID"))
    if not user_id:
        return None
    workspace_id = (
        _text(values.get("ARAGORA_WORKSPACE_ID") or values.get("ARAGORA_WORKSPACE")) or None
    )
    org_id = _text(values.get("ARAGORA_ORG_ID")) or None
    email = _text(values.get("ARAGORA_USER_EMAIL")) or None
    role = _text(values.get("ARAGORA_ROLE"))
    roles = {role} if role else set()
    return AuthorizationContext(
        user_id=user_id,
        user_email=email,
        org_id=org_id,
        workspace_id=workspace_id,
        roles=roles,
    )


def owner_binding_from_context(context: AuthorizationContext | None) -> dict[str, Any]:
    if context is None:
        return {"user_id": None, "workspace_id": None, "org_id": None}
    return {
        "user_id": context.user_id,
        "workspace_id": context.workspace_id,
        "org_id": context.org_id,
    }


def owner_binding_from_env(env: dict[str, str] | None = None) -> dict[str, Any]:
    return owner_binding_from_context(authorization_context_from_env(env))
