"""Backward-compatibility shim - consolidated into orchestrator_init.py."""

from aragora.debate.orchestrator_init import (  # noqa: F401
    init_context_delegator,
    init_prompt_context_builder,
)

__all__ = ["init_prompt_context_builder", "init_context_delegator"]
