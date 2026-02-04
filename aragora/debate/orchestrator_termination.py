"""Backward-compatibility shim - consolidated into orchestrator_init.py."""

from aragora.debate.orchestrator_init import init_termination_checker  # noqa: F401

__all__ = ["init_termination_checker"]
