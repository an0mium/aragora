"""
Context Gathering Strategies.

Each strategy is responsible for gathering a specific type of context
for debates. Strategies are designed to be:
- Independently testable
- Have clear timeout handling
- Return None on failure (never raise in gather methods)

Available Strategies:
- ClaudeSearchStrategy: Web research using Claude's built-in search
- EvidenceStrategy: Evidence from web, GitHub, and local docs
- TrendingStrategy: Trending topics from social platforms (Pulse)

Usage:
    from aragora.debate.context_strategies import ClaudeSearchStrategy

    strategy = ClaudeSearchStrategy()
    result = await strategy.gather_with_timeout(task, timeout=60.0)
"""

from .base import CachingStrategy, ContextStrategy
from .claude_search import ClaudeSearchStrategy
from .evidence import EvidenceStrategy
from .trending import TrendingStrategy

__all__ = [
    # Base classes
    "ContextStrategy",
    "CachingStrategy",
    # Strategies
    "ClaudeSearchStrategy",
    "EvidenceStrategy",
    "TrendingStrategy",
]
