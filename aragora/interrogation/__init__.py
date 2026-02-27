"""Interrogation Engine: Debate-driven prompt clarification.

The unique Aragora differentiator: before executing on any prompt, the system
runs an adversarial debate among agents to determine WHICH clarifying questions
matter most, then asks the user only the highest-value questions.

This is not single-LLM question generation — it's multi-agent deliberation
about what's worth asking, with cross-verification of assumptions.

Usage:
    from aragora.interrogation import InterrogationEngine, InterrogationConfig

    engine = InterrogationEngine()
    result = await engine.interrogate("Make our product better")
    # result.prioritized_questions — debate-ranked questions
    # result.crystallized_spec — MoSCoW specification
    # result.research_context — gathered evidence
"""

from aragora.interrogation.engine import (
    InterrogationConfig,
    InterrogationEngine,
    InterrogationResult,
)
from aragora.interrogation.crystallizer import Crystallizer, CrystallizedSpec

__all__ = [
    "Crystallizer",
    "CrystallizedSpec",
    "InterrogationConfig",
    "InterrogationEngine",
    "InterrogationResult",
]
