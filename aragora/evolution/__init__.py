"""
Prompt evolution module.

Enables agents to improve their prompts based on successful debate patterns.
"""

from aragora.evolution.ab_testing import (
    ABTest,
    ABTestManager,
    ABTestResult,
    ABTestStatus,
)
from aragora.evolution.evolver import EvolutionStrategy, PromptEvolver
from aragora.evolution.pattern_extractor import (
    Pattern,
    PatternExtractor,
    Strategy,
    StrategyIdentifier,
    extract_patterns,
    identify_strategies,
)
from aragora.evolution.tracker import EvolutionTracker

__all__ = [
    "PromptEvolver",
    "EvolutionStrategy",
    "PatternExtractor",
    "StrategyIdentifier",
    "Pattern",
    "Strategy",
    "extract_patterns",
    "identify_strategies",
    "ABTest",
    "ABTestResult",
    "ABTestStatus",
    "ABTestManager",
    "EvolutionTracker",
]
