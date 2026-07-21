"""Bounded Claude text generation through the explicit VibeProxy policy."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from aragora.agents.transports.vibeproxy import (
    ModelTransportPolicy,
    TransportMode,
    VibeProxyConfigurationError,
    VibeProxyTimeoutError,
    VibeProxyUnavailableError,
)

DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
VIBEPROXY_HARNESS = "local VibeProxy Anthropic Messages transport"
VIBEPROXY_TIMEOUT_ENV = "ARAGORA_COLLECT_EVIDENCE_VIBEPROXY_TIMEOUT_SECONDS"
DEFAULT_VIBEPROXY_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class ClaudeVibeProxyAttempt:
    """Result of the optional VibeProxy leg before a direct Claude fallback."""

    attempted: bool
    required: bool
    ok: bool
    text: str = ""
    error: str = ""
    harness: str = ""
    timeout_seconds: float = 0.0


def _attempt_timeout(reviewer_timeout: float, mode: TransportMode) -> float:
    raw = os.environ.get(VIBEPROXY_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            configured = float(raw)
        except ValueError as exc:
            raise VibeProxyConfigurationError(
                f"{VIBEPROXY_TIMEOUT_ENV} must be a positive finite number"
            ) from exc
        if not math.isfinite(configured) or configured <= 0:
            raise VibeProxyConfigurationError(
                f"{VIBEPROXY_TIMEOUT_ENV} must be a positive finite number"
            )
    else:
        configured = DEFAULT_VIBEPROXY_TIMEOUT_SECONDS

    # Prefer mode must leave time for the direct reviewer path. Required mode
    # can consume the full reviewer budget because fallback is prohibited.
    budget = reviewer_timeout if mode is TransportMode.REQUIRED else reviewer_timeout / 2
    return min(configured, budget)


def run_claude_vibeproxy(
    prompt: str,
    *,
    reviewer_timeout: float,
    model: str = DEFAULT_CLAUDE_MODEL,
    policy: ModelTransportPolicy | None = None,
) -> ClaudeVibeProxyAttempt:
    """Run one exact-model Claude attempt when VibeProxy is explicitly selected.

    Direct mode returns ``attempted=False`` without touching the proxy. Invalid
    configuration fails closed. In prefer mode an unavailable proxy returns a
    non-required failure so the caller may use its existing direct path.
    """

    try:
        resolved_policy = policy or ModelTransportPolicy.from_env()
    except VibeProxyConfigurationError as exc:
        return ClaudeVibeProxyAttempt(
            attempted=True,
            required=True,
            ok=False,
            error=f"VibeProxy configuration error: {exc}",
        )

    if resolved_policy.mode is TransportMode.DIRECT:
        return ClaudeVibeProxyAttempt(attempted=False, required=False, ok=False)

    required = resolved_policy.mode is TransportMode.REQUIRED
    timeout = 0.0
    try:
        timeout = _attempt_timeout(reviewer_timeout, resolved_policy.mode)
        route = resolved_policy.resolve("anthropic", model, capabilities=("chat",))
        if route.transport != "vibeproxy" or resolved_policy.client is None:
            return ClaudeVibeProxyAttempt(
                attempted=True,
                required=required,
                ok=False,
                error=route.fallback_reason or "VibeProxy route unavailable",
                timeout_seconds=timeout,
            )
        text = resolved_policy.client.anthropic_message(
            model=route.resolved_model,
            prompt=prompt,
            timeout=timeout,
        )
    except VibeProxyTimeoutError as exc:
        return ClaudeVibeProxyAttempt(
            attempted=True,
            required=required,
            ok=False,
            error=str(exc),
            timeout_seconds=timeout,
        )
    except VibeProxyConfigurationError as exc:
        return ClaudeVibeProxyAttempt(
            attempted=True,
            required=True,
            ok=False,
            error=str(exc),
            timeout_seconds=timeout,
        )
    except VibeProxyUnavailableError as exc:
        return ClaudeVibeProxyAttempt(
            attempted=True,
            required=required,
            ok=False,
            error=str(exc),
            timeout_seconds=timeout,
        )
    return ClaudeVibeProxyAttempt(
        attempted=True,
        required=required,
        ok=True,
        text=text,
        harness=f"{VIBEPROXY_HARNESS} (model: {route.resolved_model})",
        timeout_seconds=timeout,
    )


__all__ = [
    "ClaudeVibeProxyAttempt",
    "DEFAULT_CLAUDE_MODEL",
    "VIBEPROXY_HARNESS",
    "VIBEPROXY_TIMEOUT_ENV",
    "run_claude_vibeproxy",
]
