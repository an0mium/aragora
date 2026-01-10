"""
Prompt evolution module.

Enables agents to improve their prompts based on successful debate patterns.
"""

from aragora.evolution.evolver import PromptEvolver, EvolutionStrategy
from aragora.evolution.pattern_extractor import (
    PatternExtractor,
    StrategyIdentifier,
    Pattern,
    Strategy,
    extract_patterns,
    identify_strategies,
)
from aragora.evolution.ab_testing import (
    ABTest,
    ABTestResult,
    ABTestStatus,
    ABTestManager,
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
