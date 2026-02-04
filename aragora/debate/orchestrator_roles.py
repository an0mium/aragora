"""Backward-compatibility shim - consolidated into orchestrator_init.py."""

from aragora.debate.orchestrator_init import init_roles_and_stances  # noqa: F401

__all__ = ["init_roles_and_stances"]
