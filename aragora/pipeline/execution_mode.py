"""Execution mode for the Aragora pipeline.

AUTONOMOUS: Pre-approved by config. Used by boss loop, swarm, nomic loop.
    Safety comes from scope limits, merge gates, and explicit launch config.
INTERACTIVE: Per-action approval required. Used by API handlers, attended CLI.
    Safety comes from capability gates and the backbone ledger.
"""

from __future__ import annotations
from enum import Enum


class ExecutionMode(str, Enum):
    AUTONOMOUS = "autonomous"
    INTERACTIVE = "interactive"
