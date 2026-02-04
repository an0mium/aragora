"""Backward-compatibility shim - consolidated into orchestrator_setup.py."""

from aragora.debate.orchestrator_setup import (  # noqa: F401
    format_conclusion,
    translate_conclusions,
)

__all__ = ["format_conclusion", "translate_conclusions"]
