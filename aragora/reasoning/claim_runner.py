"""DIC-14 — executable claim runner.

A *claim* in Aragora is a structured assertion ("the rate-limiter
allows ≤100 req/s per IP") that a debate produces. Today claims are
text-only — the system can debate them, score them, and persist them,
but it cannot **mechanically check** whether they hold against a body
of evidence.

This module adds the missing piece: an `ExecutableClaim` is a name +
predicate function. A `ClaimRunner` evaluates a list of executable
claims against a shared `ClaimContext` (typed dict / object), captures
each verdict + per-claim evidence string + elapsed time, and returns
a structured `ClaimReport` that downstream code (decision-receipt
emitter, Knowledge-Mound writer, Arena post-debate enrichment) can
consume.

This is intentionally **lightweight** — it does NOT introduce a new
DSL, a new persistence schema, or a new debate phase. It is the
minimum surface area that turns Aragora's text claims into
audit-grade verifiable artifacts.

Design constraints:
    - Pure Python; no new dependencies.
    - Synchronous AND async predicates supported; runner detects.
    - Unhandled exceptions in a checker do not crash the runner; they
      become a ``CheckerError`` verdict so peer claims still run.
    - Every checker has a hard timeout; runaway predicates are killed
      via ``asyncio.wait_for``.
    - Reports are JSON-serializable so they can flow into receipts.

**Sync-predicate timeout caveat (round 30e Phase H, codex review):**
Sync predicates run on a thread via :func:`asyncio.to_thread`. Python
threads cannot be force-cancelled, so a sync predicate that ignores
external cancellation will keep running in the background after we
return a ``ClaimVerdict.TIMEOUT`` verdict. For long-running sync
checks, prefer an ``async def`` predicate that periodically yields
to the event loop, or use a checker that internally honours a
deadline. The runner's timeout protects the *caller's* wall-clock
budget but does not guarantee the worker thread terminates.

Example::

    from aragora.reasoning.claim_runner import (
        ExecutableClaim, ClaimRunner, ClaimContext,
    )

    def lat_p99_under_100ms(ctx):
        return (ctx["latency_p99_ms"] <= 100,
                f"observed p99={ctx['latency_p99_ms']}ms")

    claims = [ExecutableClaim("perf.p99", lat_p99_under_100ms)]
    runner = ClaimRunner(default_timeout_seconds=5)
    report = await runner.run(claims, {"latency_p99_ms": 87})
    assert report.all_passed
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

ClaimContext = dict[str, Any]
"""Shared bag of evidence/inputs passed to every claim predicate."""

# A predicate may return either a bool, or a (bool, evidence_str) tuple.
# Async predicates may return an awaitable yielding the same.
PredicateResult = bool | tuple[bool, str]
SyncPredicate = Callable[[ClaimContext], PredicateResult]
AsyncPredicate = Callable[[ClaimContext], Awaitable[PredicateResult]]
Predicate = SyncPredicate | AsyncPredicate


class ClaimVerdict(str, Enum):
    """Outcome of evaluating a single claim."""

    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    ERROR = "error"  # checker raised an unexpected exception


# Default per-claim timeout (seconds). Aligned with the dialog harness'
# 60s default so a slow check fails fast rather than blocking a round.
DEFAULT_TIMEOUT_SECONDS: float = 60.0

# Hard cap on the captured evidence string so a malicious or buggy
# checker cannot blow up downstream consumers.
MAX_EVIDENCE_CHARS: int = 4_000


@dataclass(frozen=True)
class ExecutableClaim:
    """A named claim plus a predicate that decides whether it holds.

    Attributes:
        name: Stable identifier (e.g. ``"perf.p99_under_100ms"``).
            Used in the report and in receipts. Should be unique
            within a single ``ClaimRunner.run()`` call.
        predicate: Callable taking a ``ClaimContext`` and returning
            either ``bool`` or ``(bool, evidence_str)``.
        timeout_seconds: Hard cap. ``None`` uses runner default.
        description: Human-readable explanation surfaced in reports.
    """

    name: str
    predicate: Predicate
    timeout_seconds: float | None = None
    description: str = ""


@dataclass
class ClaimResult:
    """Outcome of evaluating a single claim."""

    name: str
    verdict: ClaimVerdict
    evidence: str
    elapsed_seconds: float
    started_at: str
    finished_at: str
    description: str = ""
    error: str | None = None

    def passed(self) -> bool:
        return self.verdict == ClaimVerdict.PASS

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


@dataclass
class ClaimReport:
    """Aggregate over a batch run."""

    results: list[ClaimResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.verdict == ClaimVerdict.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.verdict == ClaimVerdict.FAIL)

    @property
    def errored(self) -> int:
        return sum(1 for r in self.results if r.verdict == ClaimVerdict.ERROR)

    @property
    def timed_out(self) -> int:
        return sum(1 for r in self.results if r.verdict == ClaimVerdict.TIMEOUT)

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def to_json(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "timed_out": self.timed_out,
            "all_passed": self.all_passed,
            "results": [r.to_json() for r in self.results],
        }


def _truncate_evidence(s: str, *, max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    """Cap evidence at ``max_chars`` with a sentinel suffix."""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"... [TRUNCATED: original {len(s)} chars]"


def _normalise_predicate_result(result: PredicateResult) -> tuple[bool, str]:
    """Coerce a predicate's return value into ``(passed, evidence_str)``."""
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                f"predicate must return bool or (bool, str), got tuple of length {len(result)}"
            )
        passed, evidence = result
        if not isinstance(passed, bool):
            raise TypeError(
                f"predicate tuple's first element must be bool, got {type(passed).__name__}"
            )
        if not isinstance(evidence, str):
            raise TypeError(
                f"predicate tuple's second element must be str, got {type(evidence).__name__}"
            )
        return passed, _truncate_evidence(evidence)
    if isinstance(result, bool):
        return result, ""
    raise TypeError(f"predicate must return bool or (bool, str), got {type(result).__name__}")


async def _run_one(
    claim: ExecutableClaim,
    context: ClaimContext,
    default_timeout: float,
) -> ClaimResult:
    """Evaluate a single claim, capturing all errors as structured verdicts."""
    started_at = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    timeout = claim.timeout_seconds if claim.timeout_seconds is not None else default_timeout

    verdict = ClaimVerdict.ERROR
    evidence = ""
    error_msg: str | None = None

    try:
        # Detect async vs sync predicate. `inspect.iscoroutinefunction`
        # works for both `async def` and decorated coroutines.
        coro: Awaitable[Any]
        if inspect.iscoroutinefunction(claim.predicate):
            coro = claim.predicate(context)
        else:
            # Run sync predicates in a thread so blocking I/O doesn't
            # stall the event loop.
            coro = asyncio.to_thread(claim.predicate, context)

        raw = await asyncio.wait_for(coro, timeout=timeout)
        passed, evidence = _normalise_predicate_result(raw)
        verdict = ClaimVerdict.PASS if passed else ClaimVerdict.FAIL
    except asyncio.TimeoutError:
        verdict = ClaimVerdict.TIMEOUT
        error_msg = f"checker exceeded {timeout:.1f}s timeout"
        logger.warning("claim_runner: %s timed out after %.1fs", claim.name, timeout)
    except Exception as exc:  # noqa: BLE001 — capture any predicate error
        verdict = ClaimVerdict.ERROR
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.warning("claim_runner: %s raised %s", claim.name, error_msg)

    finished_at = datetime.now(timezone.utc)
    elapsed = time.monotonic() - monotonic_start

    return ClaimResult(
        name=claim.name,
        verdict=verdict,
        evidence=evidence,
        elapsed_seconds=round(elapsed, 3),
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        description=claim.description,
        error=error_msg,
    )


@dataclass
class ClaimRunner:
    """Runs a batch of executable claims concurrently and reports verdicts."""

    default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_concurrency: int = 0
    """0 means unbounded; >0 caps parallel evaluations."""

    async def run(
        self,
        claims: Sequence[ExecutableClaim],
        context: ClaimContext,
    ) -> ClaimReport:
        """Evaluate every claim against ``context``; return structured report."""
        self._validate_claims(claims)
        started_at = datetime.now(timezone.utc)
        monotonic_start = time.monotonic()

        if not claims:
            finished_at = datetime.now(timezone.utc)
            return ClaimReport(
                results=[],
                started_at=started_at.isoformat(),
                finished_at=finished_at.isoformat(),
                elapsed_seconds=0.0,
            )

        if self.max_concurrency > 0:
            sem = asyncio.Semaphore(self.max_concurrency)

            async def bounded(c: ExecutableClaim) -> ClaimResult:
                async with sem:
                    return await _run_one(c, context, self.default_timeout_seconds)

            tasks = [bounded(c) for c in claims]
        else:
            tasks = [_run_one(c, context, self.default_timeout_seconds) for c in claims]

        # ``return_exceptions=True`` defends against bugs in the runner
        # itself; predicate errors are already captured inside
        # ``_run_one``.
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[ClaimResult] = []
        for c, r in zip(claims, raw):
            if isinstance(r, BaseException):
                now = datetime.now(timezone.utc).isoformat()
                results.append(
                    ClaimResult(
                        name=c.name,
                        verdict=ClaimVerdict.ERROR,
                        evidence="",
                        elapsed_seconds=0.0,
                        started_at=now,
                        finished_at=now,
                        description=c.description,
                        error=f"runner-level error: {type(r).__name__}: {r}",
                    )
                )
            else:
                results.append(r)

        finished_at = datetime.now(timezone.utc)
        elapsed = time.monotonic() - monotonic_start
        return ClaimReport(
            results=results,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _validate_claims(claims: Iterable[ExecutableClaim]) -> None:
        """Reject duplicate claim names and obvious type errors."""
        seen: set[str] = set()
        for c in claims:
            if not isinstance(c, ExecutableClaim):
                raise TypeError(f"expected ExecutableClaim, got {type(c).__name__}")
            if not c.name or not isinstance(c.name, str):
                raise ValueError(f"claim must have non-empty string name, got {c.name!r}")
            if c.name in seen:
                raise ValueError(f"duplicate claim name: {c.name!r}")
            seen.add(c.name)
            if not callable(c.predicate):
                raise TypeError(
                    f"claim {c.name!r} predicate must be callable, got {type(c.predicate).__name__}"
                )


__all__ = [
    "ClaimContext",
    "ClaimReport",
    "ClaimResult",
    "ClaimRunner",
    "ClaimVerdict",
    "DEFAULT_TIMEOUT_SECONDS",
    "ExecutableClaim",
    "MAX_EVIDENCE_CHARS",
]
