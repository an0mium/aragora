"""Backward-compatibility shim for agent channel helpers."""

from __future__ import annotations

from aragora.debate.orchestrator_setup import setup_agent_channels, teardown_agent_channels

__all__ = [
    "setup_agent_channels",
    "teardown_agent_channels",
]
