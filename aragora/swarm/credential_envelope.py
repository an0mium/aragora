"""Credential envelope slices for reliability preflight checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from aragora.swarm.env_utils import git_safe_env


def _bool_from_env(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _first_present(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(env.get(key, "")).strip()
        if value:
            return value
    return ""


@dataclass
class RunnerCredential:
    profile: str
    command_path: str
    auth_mode: str

    def is_complete(self) -> bool:
        if not self.auth_mode or self.auth_mode == "none":
            return False
        return bool(self.profile or self.command_path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunnerCredential":
        return cls(
            profile=str(data.get("profile", "") or ""),
            command_path=str(data.get("command_path", "") or ""),
            auth_mode=str(data.get("auth_mode", "") or ""),
        )


@dataclass
class GitCredential:
    ssh_key_available: bool
    https_token_available: bool
    safe_env: dict[str, str] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return bool(self.ssh_key_available or self.https_token_available)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["safe_env"] = dict(self.safe_env)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GitCredential":
        return cls(
            ssh_key_available=bool(data.get("ssh_key_available", False)),
            https_token_available=bool(data.get("https_token_available", False)),
            safe_env=dict(data.get("safe_env", {}) or {}),
        )


@dataclass
class GitHubApiCredential:
    token_source: str
    rate_limit_remaining: int | None

    def is_complete(self) -> bool:
        return self.token_source in {"user", "app"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GitHubApiCredential":
        remaining = data.get("rate_limit_remaining", None)
        return cls(
            token_source=str(data.get("token_source", "") or "none"),
            rate_limit_remaining=int(remaining) if isinstance(remaining, int) else None,
        )


@dataclass
class ProviderCredential:
    api_key_present: bool
    provider_name: str

    def is_complete(self) -> bool:
        return bool(self.api_key_present and self.provider_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProviderCredential":
        return cls(
            api_key_present=bool(data.get("api_key_present", False)),
            provider_name=str(data.get("provider_name", "") or ""),
        )


@dataclass
class VerificationCredential:
    can_run_pytest: bool
    can_run_ruff: bool

    def is_complete(self) -> bool:
        return bool(self.can_run_pytest and self.can_run_ruff)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationCredential":
        return cls(
            can_run_pytest=bool(data.get("can_run_pytest", False)),
            can_run_ruff=bool(data.get("can_run_ruff", False)),
        )


@dataclass
class CredentialEnvelope:
    runner: RunnerCredential
    git: GitCredential
    github_api: GitHubApiCredential
    provider: ProviderCredential
    verification: VerificationCredential

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> "CredentialEnvelope":
        snapshot = dict(env)
        runner_profile = _first_present(
            snapshot,
            (
                "ARAGORA_RUNNER_PROFILE",
                "ARAGORA_CLAUDE_PROFILE",
                "ARAGORA_CODEX_PROFILE",
            ),
        )
        command_path = _first_present(
            snapshot,
            (
                "ARAGORA_RUNNER_COMMAND",
                "CLAUDE_COMMAND",
                "CODEX_COMMAND",
            ),
        )
        auth_mode = _first_present(snapshot, ("ARAGORA_RUNNER_AUTH_MODE",))
        if not auth_mode:
            if runner_profile:
                auth_mode = "profile"
            elif command_path:
                auth_mode = "command"
            else:
                auth_mode = "none"

        ssh_key_available = bool(
            snapshot.get("SSH_AUTH_SOCK")
            or snapshot.get("GIT_SSH_COMMAND")
            or _bool_from_env(snapshot.get("ARAGORA_SSH_KEY_AVAILABLE"))
        )
        https_token_available = bool(
            snapshot.get("GIT_HTTPS_TOKEN")
            or snapshot.get("GH_TOKEN")
            or snapshot.get("GITHUB_TOKEN")
            or snapshot.get("ARAGORA_GIT_HTTPS_TOKEN")
        )

        token_source = "none"
        if snapshot.get("GITHUB_APP_ID") or snapshot.get("GITHUB_APP_INSTALLATION_ID"):
            token_source = "app"
        elif snapshot.get("GH_TOKEN") or snapshot.get("GITHUB_TOKEN"):
            token_source = "user"
        remaining = snapshot.get("GITHUB_RATE_LIMIT_REMAINING") or snapshot.get(
            "GH_RATE_LIMIT_REMAINING"
        )
        rate_limit_remaining = None
        if remaining is not None:
            try:
                rate_limit_remaining = int(remaining)
            except (TypeError, ValueError):
                rate_limit_remaining = None

        provider_name = (
            _first_present(
                snapshot,
                (
                    "ARAGORA_PROVIDER",
                    "ARAGORA_PROVIDER_NAME",
                    "OPENAI_PROVIDER",
                    "PROVIDER_NAME",
                ),
            )
            or "unknown"
        )
        api_key_present = any(
            bool(snapshot.get(key))
            for key in (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "XAI_API_KEY",
                "GROK_API_KEY",
                "OPENROUTER_API_KEY",
                "MISTRAL_API_KEY",
                "DEEPSEEK_API_KEY",
                "TINKER_API_KEY",
                "ARAGORA_PROVIDER_API_KEY",
            )
        )

        can_run_pytest = bool(
            _bool_from_env(snapshot.get("ARAGORA_CAN_RUN_PYTEST"))
            or _bool_from_env(snapshot.get("PYTEST_AVAILABLE"))
            or snapshot.get("PYTEST_PATH")
        )
        can_run_ruff = bool(
            _bool_from_env(snapshot.get("ARAGORA_CAN_RUN_RUFF"))
            or _bool_from_env(snapshot.get("RUFF_AVAILABLE"))
            or snapshot.get("RUFF_PATH")
        )

        return cls(
            runner=RunnerCredential(
                profile=runner_profile,
                command_path=command_path,
                auth_mode=auth_mode,
            ),
            git=GitCredential(
                ssh_key_available=ssh_key_available,
                https_token_available=https_token_available,
                safe_env=git_safe_env(snapshot),
            ),
            github_api=GitHubApiCredential(
                token_source=token_source,
                rate_limit_remaining=rate_limit_remaining,
            ),
            provider=ProviderCredential(
                api_key_present=api_key_present,
                provider_name=provider_name,
            ),
            verification=VerificationCredential(
                can_run_pytest=can_run_pytest,
                can_run_ruff=can_run_ruff,
            ),
        )

    def is_complete(self) -> bool:
        return all(
            (
                self.runner.is_complete(),
                self.git.is_complete(),
                self.github_api.is_complete(),
                self.provider.is_complete(),
                self.verification.is_complete(),
            )
        )

    def missing_slices(self) -> list[str]:
        missing: list[str] = []
        if not self.runner.is_complete():
            missing.append("runner")
        if not self.git.is_complete():
            missing.append("git")
        if not self.github_api.is_complete():
            missing.append("github_api")
        if not self.provider.is_complete():
            missing.append("provider")
        if not self.verification.is_complete():
            missing.append("verification")
        return missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": self.runner.to_dict(),
            "git": self.git.to_dict(),
            "github_api": self.github_api.to_dict(),
            "provider": self.provider.to_dict(),
            "verification": self.verification.to_dict(),
        }

    def preflight_cache_payload(self) -> dict[str, Any]:
        """Return a stable capability-only payload for preflight receipt caching.

        This intentionally excludes volatile fields such as ``git.safe_env`` and
        ``github_api.rate_limit_remaining`` so cached receipts do not churn when
        ambient environment details change without affecting execution capability.
        """

        return {
            "runner": {
                "profile": self.runner.profile,
                "command_path": self.runner.command_path,
                "auth_mode": self.runner.auth_mode,
            },
            "git": {
                "ssh_key_available": self.git.ssh_key_available,
                "https_token_available": self.git.https_token_available,
            },
            "github_api": {
                "token_source": self.github_api.token_source,
            },
            "provider": {
                "api_key_present": self.provider.api_key_present,
                "provider_name": self.provider.provider_name,
            },
            "verification": {
                "can_run_pytest": self.verification.can_run_pytest,
                "can_run_ruff": self.verification.can_run_ruff,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CredentialEnvelope":
        return cls(
            runner=RunnerCredential.from_dict(data.get("runner", {}) or {}),
            git=GitCredential.from_dict(data.get("git", {}) or {}),
            github_api=GitHubApiCredential.from_dict(data.get("github_api", {}) or {}),
            provider=ProviderCredential.from_dict(data.get("provider", {}) or {}),
            verification=VerificationCredential.from_dict(data.get("verification", {}) or {}),
        )

    def seal(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def preflight_cache_seal(self) -> str:
        payload = json.dumps(
            self.preflight_cache_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()
