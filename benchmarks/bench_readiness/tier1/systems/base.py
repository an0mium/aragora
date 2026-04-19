"""Common dataclass for system-under-test outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SystemOutput:
    """Canonical output from a system run on a single task.

    Attributes
    ----------
    system:
        Display name of the system, e.g. ``solo_opus_4_7`` or
        ``aragora_debate_3x_opus``.
    answer:
        The final textual answer handed to the judge.
    latency_sec:
        Wall-clock seconds from run start to answer return.
    tokens_in / tokens_out:
        Best-effort token counts. May be 0 for systems that don't surface them.
    cost_usd:
        Best-effort dollar cost. May be 0 if not tracked.
    error:
        Non-empty if the run failed. Callers should skip judging failed runs.
    raw:
        Free-form per-system internals (number of rounds, consensus method,
        raw model responses, etc.) — not persisted to CSV.
    """

    system: str
    answer: str
    latency_sec: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
