"""Backward-compatibility shim for debate strategy helpers."""

from __future__ import annotations

from aragora.debate.orchestrator_setup import (
    init_debate_strategy,
    init_fabric_integration,
    init_post_debate_workflow,
)

__all__ = [
    "init_debate_strategy",
    "init_fabric_integration",
    "init_post_debate_workflow",
]
