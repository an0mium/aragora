"""Backward-compatibility shim - consolidated into orchestrator_init.py."""

from aragora.debate.orchestrator_init import (  # noqa: F401
    cleanup_convergence,
    init_convergence,
    reinit_convergence_for_debate,
)

__all__ = ["init_convergence", "reinit_convergence_for_debate", "cleanup_convergence"]
