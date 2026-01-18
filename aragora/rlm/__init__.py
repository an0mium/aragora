"""
Recursive Language Models (RLM) integration for Aragora.

Based on the paper: "Recursive Language Models" (arXiv:2512.24601)
by Alex L. Zhang, Tim Kraska, and Omar Khattab.

RLMs enable LLMs to process inputs far exceeding their context windows by
treating long prompts as an external environment that can be programmatically
examined, decomposed, and recursively processed.

Integration points in Aragora:
1. Debate context compression - Hierarchical summary trees
2. Knowledge Mound queries - Recursive retrieval with abstraction
3. Repository understanding - Hierarchical codebase models

Usage:
    from aragora.rlm import RLMContext, HierarchicalCompressor

    # Create hierarchical context from debate history
    compressor = HierarchicalCompressor(model="claude")
    hierarchical_ctx = await compressor.compress(debate_history)

    # Query with recursive decomposition
    result = await hierarchical_ctx.query("What were the key disagreements?")
"""

from .types import (
    RLMConfig,
    RLMContext,
    AbstractionLevel,
    CompressionResult,
    DecompositionStrategy,
    RLMQuery,
    RLMResult,
)
from .compressor import HierarchicalCompressor
from .repl import RLMEnvironment
from .strategies import (
    PeekStrategy,
    GrepStrategy,
    PartitionMapStrategy,
    SummarizeStrategy,
    HierarchicalStrategy,
    AutoStrategy,
    get_strategy,
)
from .bridge import (
    AragoraRLM,
    DebateContextAdapter,
    KnowledgeMoundAdapter,
    RLMBackendConfig,
    create_aragora_rlm,
    HAS_OFFICIAL_RLM,
)

__all__ = [
    # Types
    "RLMConfig",
    "RLMContext",
    "AbstractionLevel",
    "CompressionResult",
    "DecompositionStrategy",
    "RLMQuery",
    "RLMResult",
    # Core
    "HierarchicalCompressor",
    "RLMEnvironment",
    # Strategies
    "PeekStrategy",
    "GrepStrategy",
    "PartitionMapStrategy",
    "SummarizeStrategy",
    "HierarchicalStrategy",
    "AutoStrategy",
    "get_strategy",
    # Bridge (official RLM integration)
    "AragoraRLM",
    "DebateContextAdapter",
    "KnowledgeMoundAdapter",
    "RLMBackendConfig",
    "create_aragora_rlm",
    "HAS_OFFICIAL_RLM",
]
