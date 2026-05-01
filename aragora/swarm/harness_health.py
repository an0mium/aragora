"""Session-scoped health registry for harnesses.

Mirrors the design of :mod:`aragora.routing.session_circuit_breaker` but at
the *harness* layer (Claude Code, Codex, Aider, agent-bridge, etc.) rather
than the *provider* layer (Anthropic, OpenAI, etc.).

Why this exists
---------------

The Round 30f plan calls out harness reliability as Phase 2. The boss-loop
and supervisor today launch external CLIs without a single source of truth
for "is harness X currently healthy?". Each spawn site decides on its own
whether to retry, fall back, or give up. This module gives us:

  - One in-memory registry, thread-safe, that any caller can consult.
  - Three failure categories with separate pinning rules:
      * AUTH failures (401/403/missing API key) — permanent for the
        session; do not retry, escalate to fallback.
      * QUOTA failures (429/rate-limit/budget-exhausted) — permanent for
        the session.
      * TRANSIENT failures (timeouts, 5xx, exit-code != 0 on a single
        run) — sliding window; only pin after a threshold of failures
        within a window.
  - A snapshot API (:meth:`HarnessHealthRegistry.snapshot`) suitable for
    rendering by ``aragora swarm harness-status``.

Pinning is *opt-in advice*. Callers must consult
:meth:`HarnessHealthRegistry.is_available` before invoking and treat a
``False`` result as "use the fallback ladder". This module does not block
calls; it just records and reports.

Scope (Phase 2)
---------------

This is the **registry + snapshot** layer. The fallback ladder that
*consumes* this registry lives in :mod:`aragora.swarm.harness_fallback`.
The CLI surface lives in :mod:`aragora.cli.commands.harness_status`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


__all__ = [
    "AUTH_FAILURE_CODES",
    "FailureCategory",
    "HarnessAttempt",
    "HarnessFailure",
    "HarnessHealthRegistry",
    "HarnessHealthSnapshot",
    "PERMANENT_FAILURE_CODES",
    "QUOTA_FAILURE_CODES",
    "TRANSIENT_FAILURE_CODES",
    "TRANSIENT_FAILURE_THRESHOLD",
    "TRANSIENT_WINDOW_SECONDS",
    "classify_failure",
    "get_harness_health_registry",
    "record_harness_result",
    "reset_harness_health_registry",
]


# Failure-classification thresholds. Pinned to match
# session_circuit_breaker so harness-level and provider-level operators see
# consistent behavior.
AUTH_FAILURE_CODES: frozenset[int] = frozenset({401, 403})
QUOTA_FAILURE_CODES: frozenset[int] = frozenset({429})
PERMANENT_FAILURE_CODES: frozenset[int] = AUTH_FAILURE_CODES | QUOTA_FAILURE_CODES
TRANSIENT_FAILURE_CODES: frozenset[int] = frozenset({500, 502, 503, 504})

TRANSIENT_FAILURE_THRESHOLD: int = 3
TRANSIENT_WINDOW_SECONDS: float = 300.0


class FailureCategory(Enum):
    """Classification of a harness failure."""

    AUTH = "auth"
    QUOTA = "quota"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class HarnessFailure:
    """One failure event for a harness."""

    harness: str
    category: FailureCategory
    reason: str
    status_code: int | None = None
    at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class HarnessAttempt:
    """One attempt event (success or failure) for a harness.

    Used for the snapshot's ``last_attempt_at`` / ``last_outcome`` fields
    so operators can tell at a glance whether a harness is *idle* or
    *actively failing*.
    """

    harness: str
    success: bool
    at: float = field(default_factory=time.monotonic)
    detail: str = ""


@dataclass(frozen=True, slots=True)
class HarnessHealthSnapshot:
    """Read-only view of a single harness's health.

    Rendered by ``aragora swarm harness-status``.
    """

    harness: str
    available: bool
    permanent_pin_reason: str | None
    transient_failure_count_in_window: int
    last_attempt_at_monotonic: float | None
    last_outcome: str | None  # "success" | "failure" | None
    last_failure_reason: str | None
    last_failure_category: str | None  # FailureCategory.value or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "available": self.available,
            "permanent_pin_reason": self.permanent_pin_reason,
            "transient_failure_count_in_window": self.transient_failure_count_in_window,
            "last_attempt_at_monotonic": self.last_attempt_at_monotonic,
            "last_outcome": self.last_outcome,
            "last_failure_reason": self.last_failure_reason,
            "last_failure_category": self.last_failure_category,
        }


def classify_failure(
    *,
    status_code: int | None,
    reason: str,
) -> FailureCategory:
    """Classify a failure into one of three categories.

    Heuristics:
      - status_code in :data:`AUTH_FAILURE_CODES` -> AUTH.
      - status_code in :data:`QUOTA_FAILURE_CODES` -> QUOTA.
      - reason mentions ``api key``/``unauthorized``/``forbidden`` -> AUTH.
      - reason mentions ``rate limit``/``quota``/``budget`` -> QUOTA.
      - everything else -> TRANSIENT.

    The string heuristics let CLI harnesses (which don't return HTTP
    codes) signal auth/quota pinning by including the right phrase in
    their stderr.
    """
    if status_code is not None:
        if status_code in AUTH_FAILURE_CODES:
            return FailureCategory.AUTH
        if status_code in QUOTA_FAILURE_CODES:
            return FailureCategory.QUOTA
    lower = reason.lower()
    if any(
        marker in lower for marker in ("unauthorized", "forbidden", "api key", "missing credential")
    ):
        return FailureCategory.AUTH
    if any(
        marker in lower for marker in ("rate limit", "quota", "budget", "429", "too many request")
    ):
        return FailureCategory.QUOTA
    return FailureCategory.TRANSIENT


class HarnessHealthRegistry:
    """In-memory, session-scoped registry of harness health.

    Thread-safe; resets only on process restart (or via
    :func:`reset_harness_health_registry`, intended for tests).
    """

    def __init__(
        self,
        *,
        transient_threshold: int = TRANSIENT_FAILURE_THRESHOLD,
        transient_window_seconds: float = TRANSIENT_WINDOW_SECONDS,
    ) -> None:
        if transient_threshold < 1:
            raise ValueError("transient_threshold must be >= 1")
        if transient_window_seconds <= 0:
            raise ValueError("transient_window_seconds must be > 0")
        self._lock = threading.RLock()
        self._transient_threshold = transient_threshold
        self._transient_window = transient_window_seconds
        # harness -> list[HarnessFailure] (rolling)
        self._failures: dict[str, list[HarnessFailure]] = {}
        # harness -> permanent pin reason (None == not pinned)
        self._permanent_pins: dict[str, str] = {}
        # harness -> last attempt
        self._last_attempts: dict[str, HarnessAttempt] = {}

    # -- recording API -------------------------------------------------

    def record_attempt(self, harness: str) -> None:
        """Record that an invocation of ``harness`` is being attempted.

        Lets the snapshot show "tried recently, no outcome yet". Most
        callers won't need this; record_success / record_failure imply
        an attempt.
        """
        with self._lock:
            self._last_attempts[harness] = HarnessAttempt(
                harness=harness, success=False, detail="in_progress"
            )

    def record_success(self, harness: str, *, detail: str = "") -> None:
        """Record a successful invocation. Resets the transient-failure window."""
        with self._lock:
            self._failures.pop(harness, None)
            self._last_attempts[harness] = HarnessAttempt(
                harness=harness, success=True, detail=detail
            )

    def record_failure(
        self,
        harness: str,
        *,
        reason: str,
        status_code: int | None = None,
    ) -> FailureCategory:
        """Record a failed invocation. Returns the classified category.

        AUTH and QUOTA failures pin the harness permanently for the
        session. TRANSIENT failures accumulate; once
        ``transient_threshold`` is reached within
        ``transient_window_seconds``, the harness is pinned with the
        reason ``"too many transient failures"``.
        """
        category = classify_failure(status_code=status_code, reason=reason)
        failure = HarnessFailure(
            harness=harness,
            category=category,
            reason=reason,
            status_code=status_code,
        )
        with self._lock:
            self._last_attempts[harness] = HarnessAttempt(
                harness=harness, success=False, detail=reason
            )
            if category in (FailureCategory.AUTH, FailureCategory.QUOTA):
                self._permanent_pins[harness] = (
                    f"{category.value}: {reason}" if reason else f"{category.value} failure"
                )
                logger.warning(
                    "harness_health: pinning %s permanently (%s): %s",
                    harness,
                    category.value,
                    reason,
                )
                return category
            # Transient: prune old, append, check threshold.
            now = time.monotonic()
            window = self._failures.setdefault(harness, [])
            cutoff = now - self._transient_window
            window[:] = [f for f in window if f.at >= cutoff]
            window.append(failure)
            if len(window) >= self._transient_threshold:
                self._permanent_pins[harness] = (
                    f"transient threshold ({self._transient_threshold}) reached "
                    f"in {int(self._transient_window)}s window"
                )
                logger.warning(
                    "harness_health: pinning %s after %d transient failures",
                    harness,
                    len(window),
                )
            return category

    # -- query API -----------------------------------------------------

    def is_available(self, harness: str) -> bool:
        """Return True iff the harness is not permanently pinned."""
        with self._lock:
            return harness not in self._permanent_pins

    def permanent_pin_reason(self, harness: str) -> str | None:
        with self._lock:
            return self._permanent_pins.get(harness)

    def transient_failures_in_window(self, harness: str) -> int:
        """Number of transient failures within the rolling window."""
        with self._lock:
            window = self._failures.get(harness, [])
            cutoff = time.monotonic() - self._transient_window
            return sum(1 for f in window if f.at >= cutoff)

    def snapshot(self, harness: str) -> HarnessHealthSnapshot:
        """Read-only snapshot for one harness."""
        with self._lock:
            pin = self._permanent_pins.get(harness)
            last = self._last_attempts.get(harness)
            window = self._failures.get(harness, [])
            cutoff = time.monotonic() - self._transient_window
            transient_count = sum(1 for f in window if f.at >= cutoff)
            last_failure = window[-1] if window else None
            last_outcome: str | None
            if last is None:
                last_outcome = None
            elif last.detail == "in_progress":
                last_outcome = "in_progress"
            elif last.success:
                last_outcome = "success"
            else:
                last_outcome = "failure"
            return HarnessHealthSnapshot(
                harness=harness,
                available=pin is None,
                permanent_pin_reason=pin,
                transient_failure_count_in_window=transient_count,
                last_attempt_at_monotonic=last.at if last else None,
                last_outcome=last_outcome,
                last_failure_reason=last_failure.reason if last_failure else None,
                last_failure_category=(last_failure.category.value if last_failure else None),
            )

    def snapshot_all(self, *, harnesses: list[str] | None = None) -> list[HarnessHealthSnapshot]:
        """Snapshots for all known harnesses, plus any explicitly requested.

        ``harnesses`` lets callers force a snapshot for harnesses that
        haven't been touched yet (so the CLI can show "claude-code:
        never invoked" rather than omitting it).
        """
        with self._lock:
            seen: set[str] = set()
            seen.update(self._failures.keys())
            seen.update(self._permanent_pins.keys())
            seen.update(self._last_attempts.keys())
            if harnesses:
                seen.update(harnesses)
        return [self.snapshot(name) for name in sorted(seen)]


_SINGLETON_LOCK = threading.Lock()
_SINGLETON: HarnessHealthRegistry | None = None


def get_harness_health_registry() -> HarnessHealthRegistry:
    """Return the process-wide harness health registry, lazily created."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = HarnessHealthRegistry()
        return _SINGLETON


def reset_harness_health_registry() -> None:
    """Reset the singleton. Intended for tests only."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = None


def record_harness_result(
    *,
    harness: str,
    success: bool,
    error_message: str | None = None,
    error_output: str | None = None,
    status_code: int | None = None,
) -> None:
    """Convenience: record a harness call outcome on the singleton.

    This is the single helper that real harness call sites
    (:class:`aragora.harnesses.claude_code.ClaudeCodeHarness`,
    :class:`aragora.harnesses.codex.CodexHarness`) should use after
    each call. It does the right thing regardless of whether the
    harness has been touched before in this process.

    Failure reason is composed from ``error_message`` (preferred) and
    falls back to a tail of ``error_output`` so stderr-only signals
    (e.g., a CLI's ``Unauthorized: invalid API key`` printed without
    an HTTP code) feed cleanly into :func:`classify_failure`.
    """
    registry = get_harness_health_registry()
    if success:
        registry.record_success(harness)
        return
    reason = (error_message or "").strip()
    if not reason and error_output:
        # Take the tail (last 240 chars) so we keep the most diagnostic
        # bit of stderr without flooding the registry.
        reason = error_output.strip()[-240:]
    if not reason:
        reason = "harness call failed (no error message captured)"
    registry.record_failure(harness, reason=reason, status_code=status_code)
