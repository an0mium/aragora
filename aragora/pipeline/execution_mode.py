"""Execution mode for the Aragora pipeline.

AUTONOMOUS: Pre-approved by config. Used by boss loop, swarm, nomic loop.
    Safety comes from scope limits, merge gates, and explicit launch config.
INTERACTIVE: Per-action approval required. Used by API handlers, attended CLI.
    Safety comes from capability gates and the backbone ledger.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    AUTONOMOUS = "autonomous"
    INTERACTIVE = "interactive"


def resolve_safety_mode(
    mode: ExecutionMode | str | None,
    *,
    auth_context: Any | None = None,
    default: ExecutionMode = ExecutionMode.AUTONOMOUS,
) -> ExecutionMode:
    """Resolve safety mode from an explicit value or execution context."""
    if isinstance(mode, ExecutionMode):
        return mode

    normalized = str(mode or "").strip().lower()
    if normalized == ExecutionMode.INTERACTIVE.value:
        return ExecutionMode.INTERACTIVE
    if normalized == ExecutionMode.AUTONOMOUS.value:
        return ExecutionMode.AUTONOMOUS

    if auth_context is not None:
        return ExecutionMode.INTERACTIVE
    return default
