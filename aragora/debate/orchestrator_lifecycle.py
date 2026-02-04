"""Backward-compatibility shim - consolidated into orchestrator_setup.py."""

from aragora.debate.orchestrator_setup import (  # noqa: F401
    init_caches,
    init_checkpoint_ops,
    init_event_emitter,
    init_lifecycle_manager,
)

__all__ = [
    "init_caches",
    "init_lifecycle_manager",
    "init_event_emitter",
    "init_checkpoint_ops",
]
