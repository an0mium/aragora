"""Declarative fallback ladders for harness selection.

A :class:`FallbackLadder` is an ordered list of harness names. Calling
:meth:`FallbackLadder.next_available` consults
:class:`aragora.swarm.harness_health.HarnessHealthRegistry` and returns
the first non-pinned harness in order, or ``None`` if every step in the
ladder has been pinned.

Why a separate module
---------------------

The health registry must stay free of *policy* (which harness to prefer
when several are healthy). The ladder owns that policy. Spawn sites that
just want "give me whichever harness is alive" call into the ladder; the
registry just records and reports.

Usage
-----

>>> from aragora.swarm.harness_fallback import FallbackLadder
>>> ladder = FallbackLadder.default_implementation_ladder()
>>> chosen = ladder.next_available()
>>> if chosen is None:
...     raise RuntimeError("no implementation harness available")

Default ladders
---------------

Two ladders are provided:

  - ``default_implementation_ladder`` — for "implement this change"
    style work. Order: ``claude-code -> codex``.
  - ``default_review_ladder`` — for "review/critique this change"
    style work. Order: ``claude-code -> codex``.

Note: Earlier drafts included ``aider`` as a third implementation
step, but no real ``AiderHarness`` is wired into this codebase
(``aragora.harnesses`` only ships ``ClaudeCodeHarness`` and
``CodexHarness``). Round 30g removes the dangling reference rather
than ship a stub: a ladder must only name harnesses that can actually
run. Add ``aider`` back when an honest harness implementation lands.

These mirror today's call sites; new ladders can be constructed by
callers or registered as named ladders later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from aragora.swarm.harness_health import (
    HarnessHealthRegistry,
    get_harness_health_registry,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FallbackLadder",
    "FallbackResolution",
    "default_implementation_ladder",
    "default_review_ladder",
]


@dataclass(frozen=True, slots=True)
class FallbackResolution:
    """Result of resolving a ladder against the registry."""

    chosen: str | None
    skipped: tuple[str, ...]
    reasons: dict[str, str] = field(default_factory=dict)

    def is_resolved(self) -> bool:
        return self.chosen is not None


@dataclass(frozen=True, slots=True)
class FallbackLadder:
    """An ordered list of harnesses to try in turn."""

    name: str
    steps: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"FallbackLadder({self.name!r}) requires at least one step")
        seen: set[str] = set()
        for step in self.steps:
            if not isinstance(step, str) or not step:
                raise ValueError(
                    f"FallbackLadder({self.name!r}) steps must be non-empty strings; got {step!r}"
                )
            if step in seen:
                raise ValueError(f"FallbackLadder({self.name!r}) has duplicate step {step!r}")
            seen.add(step)

    def next_available(
        self,
        *,
        registry: HarnessHealthRegistry | None = None,
    ) -> FallbackResolution:
        """Return the first available harness in the ladder.

        ``registry`` defaults to the process-wide singleton. Pass an
        explicit instance in tests.
        """
        reg = registry or get_harness_health_registry()
        skipped: list[str] = []
        reasons: dict[str, str] = {}
        for step in self.steps:
            if reg.is_available(step):
                return FallbackResolution(
                    chosen=step,
                    skipped=tuple(skipped),
                    reasons=reasons,
                )
            skipped.append(step)
            reason = reg.permanent_pin_reason(step) or "pinned"
            reasons[step] = reason
            logger.info(
                "harness_fallback: %s skipping %s (%s)",
                self.name,
                step,
                reason,
            )
        return FallbackResolution(chosen=None, skipped=tuple(skipped), reasons=reasons)

    @classmethod
    def from_steps(cls, name: str, steps: Sequence[str]) -> "FallbackLadder":
        """Construct from a sequence (validated)."""
        return cls(name=name, steps=tuple(steps))


def default_implementation_ladder() -> FallbackLadder:
    """Order tuned for code-implementation work today.

    Rationale:
      - claude-code is preferred for repo-wide reasoning + edit volume.
      - codex is the second-best fallback (matches existing
        cli_agents.py preference order).

    ``aider`` was removed in Round 30g: there is no real
    ``AiderHarness`` shipped in :mod:`aragora.harnesses`, and the
    contract is that a ladder must name only harnesses that can
    actually run. Re-add when a real harness lands.
    """
    return FallbackLadder(
        name="implementation",
        steps=("claude-code", "codex"),
    )


def default_review_ladder() -> FallbackLadder:
    """Order tuned for code-review work today."""
    return FallbackLadder(
        name="review",
        steps=("claude-code", "codex"),
    )
