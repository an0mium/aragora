from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import ssl
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aragora.agents.base import create_agent
from aragora.agents.errors.exceptions import CLISubprocessError

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_PROVIDER_ORDER = ("codex", "claude", "openrouter")
DEFAULT_CLAUDE_REVIEW_PROFILES = tuple(f"max-{index:02d}" for index in range(1, 13))
_BILLING_MARKERS = ("credit balance", "billing", "payment required", "purchase credits")
_MODEL_FAMILY_OVERRIDES = {
    "anthropic-api": "claude",
    "claude": "claude",
    "codex": "codex",
    "openai": "codex",
    "openai-api": "codex",
    "openrouter": "openrouter",
}


@dataclass(slots=True)
class ReviewCandidate:
    provider: str
    label: str
    profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "provider": self.provider,
            "label": self.label,
        }
        if self.profile:
            payload["profile"] = self.profile
        return payload


class ReviewRoutingError(RuntimeError):
    def __init__(
        self,
        attempts: list[dict[str, Any]],
        *,
        category: str = "unavailable",
        public_message: str | None = None,
    ) -> None:
        self.attempts = attempts
        self.category = str(category or "unavailable").strip() or "unavailable"
        self.public_message = str(public_message or "").strip() or _review_routing_public_message(
            self.category
        )
        super().__init__(self.public_message)


def resolve_review_candidates(
    *,
    worker_model: str,
    preferred_review_model: str,
) -> list[ReviewCandidate]:
    worker_family = _model_family(worker_model)
    preferred_family = _model_family(preferred_review_model)
    configured_order = _review_provider_order()
    families: list[str] = []

    if (
        preferred_family
        and preferred_family in configured_order
        and preferred_family != worker_family
    ):
        families.append(preferred_family)
    for provider in configured_order:
        if provider == worker_family and provider != "openrouter":
            continue
        if provider not in families:
            families.append(provider)

    candidates: list[ReviewCandidate] = []
    for provider in families:
        if provider == "claude":
            for profile in _claude_review_profiles():
                candidates.append(
                    ReviewCandidate(
                        provider="claude",
                        label=f"claude:{profile}",
                        profile=profile,
                    )
                )
            continue
        candidates.append(ReviewCandidate(provider=provider, label=provider))
    return candidates


def preflight_review_candidate(
    candidate: ReviewCandidate,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if candidate.provider == "codex":
        return _cli_preflight("codex")
    if candidate.provider == "claude":
        return _claude_profile_preflight(candidate, repo_root=repo_root)
    if candidate.provider == "openrouter":
        return _openrouter_preflight()
    return {
        "ok": False,
        "detail": f"Unsupported review provider: {candidate.provider}",
    }


async def generate_review_response(
    prompt: str,
    *,
    worker_model: str,
    preferred_review_model: str,
    repo_root: Path,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for candidate in resolve_review_candidates(
        worker_model=worker_model,
        preferred_review_model=preferred_review_model,
    ):
        preflight = preflight_review_candidate(candidate, repo_root=repo_root)
        if not preflight.get("ok", False):
            attempts.append(
                {
                    "candidate": candidate.label,
                    "stage": "preflight",
                    "detail": str(preflight.get("detail", "unavailable")).strip() or "unavailable",
                }
            )
            continue
        try:
            response = await _run_review_candidate(candidate, prompt, repo_root=repo_root)
        except CLISubprocessError as exc:
            logger.warning("review candidate %s failed: %s", candidate.label, exc)
            attempts.append(
                _failure_attempt(
                    candidate.label,
                    stage="generate",
                    exc=exc,
                )
            )
            continue
        except Exception as exc:
            logger.warning("review candidate %s failed: %s", candidate.label, exc)
            attempts.append(
                _failure_attempt(
                    candidate.label,
                    stage="generate",
                    exc=exc,
                )
            )
            continue
        attempts.append(
            {
                "candidate": candidate.label,
                "stage": "generate",
                "detail": "ok",
            }
        )
        return {
            "candidate": candidate.to_dict(),
            "response": response,
            "attempts": attempts,
        }
    raise ReviewRoutingError(
        attempts,
        category=_review_routing_category(attempts),
    )


async def _run_review_candidate(
    candidate: ReviewCandidate,
    prompt: str,
    *,
    repo_root: Path,
) -> str:
    if candidate.provider == "claude":
        return await _run_claude_profile_candidate(candidate, prompt, repo_root=repo_root)
    if candidate.provider == "codex":
        agent = create_agent(
            "codex",
            name="campaign-review",
            role="critic",
            enable_fallback=False,
        )
        return await agent.generate(prompt)
    if candidate.provider == "openrouter":
        agent = create_agent(
            "openrouter",
            name="campaign-review",
            role="critic",
            enable_fallback=False,
        )
        return await agent.generate(prompt)
    raise RuntimeError(f"Unsupported review provider: {candidate.provider}")


def _cli_preflight(command_name: str) -> dict[str, Any]:
    if shutil.which(command_name):
        return {"ok": True, "detail": f"{command_name} is available"}
    return {"ok": False, "detail": f"{command_name} CLI not found on PATH"}


def _claude_profile_preflight(candidate: ReviewCandidate, *, repo_root: Path) -> dict[str, Any]:
    script = _claude_profile_script(repo_root)
    if script is None:
        return {"ok": False, "detail": "claude_profile.sh not found"}
    if not candidate.profile:
        return {"ok": False, "detail": "Claude review profile is missing"}
    if not shutil.which("claude"):
        return {"ok": False, "detail": "claude CLI not found on PATH"}
    result = subprocess.run(
        [str(script), "status", candidate.profile],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode == 0:
        return {"ok": True, "detail": f"{candidate.label} authenticated"}
    detail = (result.stderr or result.stdout or "").strip()
    return {"ok": False, "detail": detail or f"{candidate.label} is unavailable"}


def _openrouter_preflight() -> dict[str, Any]:
    if not os.environ.get("OPENROUTER_API_KEY"):
        return {"ok": False, "detail": "OPENROUTER_API_KEY is not configured"}
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection(("openrouter.ai", 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname="openrouter.ai"):
                pass
    except OSError as exc:
        return {"ok": False, "detail": f"OpenRouter TLS check failed: {exc}"}
    return {"ok": True, "detail": "OpenRouter API key and TLS look healthy"}


def _claude_profile_script(repo_root: Path) -> Path | None:
    script = (repo_root / "scripts" / "claude_profile.sh").resolve()
    return script if script.exists() else None


async def _run_claude_profile_candidate(
    candidate: ReviewCandidate,
    prompt: str,
    *,
    repo_root: Path,
) -> str:
    script = _claude_profile_script(repo_root)
    if script is None:
        raise CLISubprocessError("claude_profile.sh not found", agent_name=candidate.label)
    if not candidate.profile:
        raise CLISubprocessError("Claude review profile is missing", agent_name=candidate.label)
    proc = await asyncio.create_subprocess_exec(
        str(script),
        "exec",
        candidate.profile,
        "--",
        "claude",
        "--print",
        "-p",
        "-",
        cwd=str(repo_root),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout=300)
    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    if proc.returncode != 0:
        raise CLISubprocessError(
            message=f"Claude profile command failed for {candidate.label}",
            agent_name=candidate.label,
            returncode=proc.returncode,
            stderr=(stderr_text or stdout_text).strip()[:500] or None,
        )
    response = _strip_claude_profile_wrapper(stdout_text)
    if not response:
        raise CLISubprocessError(
            message=f"Claude profile command returned empty output for {candidate.label}",
            agent_name=candidate.label,
            returncode=proc.returncode,
            stderr=stderr_text.strip()[:500] or None,
        )
    return response


def _strip_claude_profile_wrapper(output: str) -> str:
    lines = []
    for line in output.splitlines():
        if line.startswith("Using profile home:"):
            continue
        if line.startswith("Command:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _candidate_failure_detail(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, CLISubprocessError):
        raw = str(exc.stderr or exc).strip().lower()
        if any(marker in raw for marker in _BILLING_MARKERS):
            return ("billing_exhausted", "Reviewer credits are exhausted.")
        return ("cli_failure", "Reviewer CLI command failed.")
    return (exc.__class__.__name__, exc.__class__.__name__)


def _failure_attempt(candidate: str, *, stage: str, exc: Exception) -> dict[str, Any]:
    kind, detail = _candidate_failure_detail(exc)
    return {
        "candidate": candidate,
        "stage": stage,
        "kind": kind,
        "detail": detail,
    }


def _review_routing_category(attempts: list[dict[str, Any]]) -> str:
    if any(str(item.get("kind", "")).strip() == "billing_exhausted" for item in attempts):
        return "billing_exhausted"
    return "unavailable"


def _review_routing_public_message(category: str) -> str:
    if category == "billing_exhausted":
        return "Reviewer capacity is exhausted. Check the active reviewer account and available credits."
    return "No configured review candidate succeeded. Check logs for detail."


def _review_provider_order() -> list[str]:
    raw = str(os.environ.get("ARAGORA_REVIEW_PROVIDER_ORDER", "")).strip()
    if not raw:
        return list(DEFAULT_REVIEW_PROVIDER_ORDER)
    result: list[str] = []
    for item in raw.split(","):
        normalized = str(item).strip().lower()
        if normalized and normalized not in result:
            result.append(normalized)
    return result or list(DEFAULT_REVIEW_PROVIDER_ORDER)


def _claude_review_profiles() -> list[str]:
    raw = str(os.environ.get("ARAGORA_CLAUDE_REVIEW_PROFILES", "")).strip()
    if not raw:
        return list(DEFAULT_CLAUDE_REVIEW_PROFILES)
    result: list[str] = []
    for item in raw.split(","):
        normalized = str(item).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result or list(DEFAULT_CLAUDE_REVIEW_PROFILES)


def _model_family(model_type: str) -> str:
    normalized = str(model_type or "").strip().lower()
    if normalized in _MODEL_FAMILY_OVERRIDES:
        return _MODEL_FAMILY_OVERRIDES[normalized]
    if normalized.startswith("claude"):
        return "claude"
    if normalized.startswith("gpt-"):
        return "codex"
    return normalized
