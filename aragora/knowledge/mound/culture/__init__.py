"""
Culture Accumulator - Organizational learning from debates.

Implements the "culture" concept inspired by Agno's shared memory system,
where agents accumulate and share reasoning patterns, decision heuristics,
and organizational wisdom over time.

Key concepts:
- ReasoningPattern: Learned patterns of how to reason about problems
- DecisionHeuristic: Rules of thumb for decision making
- StigmergicSignal: Indirect agent communication through environment

Usage:
    from aragora.knowledge.mound.culture import (
        CultureAccumulator,
        ReasoningPattern,
        DecisionHeuristic,
        StigmergyManager,
    )

    # Initialize culture accumulator
    culture = CultureAccumulator()
    await culture.initialize()

    # Extract patterns from a debate
    patterns = await culture.extract_patterns_from_debate(debate_result)

    # Query relevant patterns for a new task
    relevant = await culture.get_relevant_patterns(context="security review")
"""

from aragora.knowledge.mound.culture.patterns import (
    DecisionHeuristic,
    PatternType,
    ReasoningPattern,
)
from aragora.knowledge.mound.culture.accumulator import CultureAccumulator, DebateObservation
from aragora.knowledge.mound.culture.stigmergy import (
    PheromoneTrail,
    SignalType,
    StigmergicSignal,
    StigmergyManager,
)

__all__ = [
    "CultureAccumulator",
    "DebateObservation",
    "DecisionHeuristic",
    "PatternType",
    "PheromoneTrail",
    "ReasoningPattern",
    "SignalType",
    "StigmergicSignal",
    "StigmergyManager",
]
