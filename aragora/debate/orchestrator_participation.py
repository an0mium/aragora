"""Backward-compatibility shim - consolidated into orchestrator_init.py."""

from aragora.debate.orchestrator_init import (  # noqa: F401
    init_event_bus,
    init_user_participation,
)

__all__ = ["init_user_participation", "init_event_bus"]
