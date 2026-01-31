"""
Debate configuration package.

Provides centralized configuration for debate module defaults.
"""

from aragora.debate.config.defaults import (
    DEBATE_DEFAULTS,
    DebateDefaults,
    get_debate_defaults,
)

__all__ = [
    "DEBATE_DEFAULTS",
    "DebateDefaults",
    "get_debate_defaults",
]
