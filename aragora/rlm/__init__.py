"""
Recursive Language Models (RLM) integration for Aragora.

Based on the paper: "Recursive Language Models" (arXiv:2512.24601)
by Alex L. Zhang, Tim Kraska, and Omar Khattab.
MIT Licensed: https://github.com/alexzhang13/rlm

How Real RLM Works (from the paper):
1. Context is stored as a Python variable in a REPL environment (NOT in prompt)
2. LLM writes code to programmatically examine/grep/partition the context
3. LLM can recursively call itself on context subsets
4. LLM dynamically decides decomposition strategy (grep, map-reduce, peek, etc.)
5. Full context remains accessible - no information loss from truncation

The key insight is that "long prompts should not be fed into the neural network
directly but should instead be treated as part of the environment that the LLM
can symbolically interact with."

Installation:
    # Install with real RLM support
    pip install aragora[rlm]

    # Or install the official library directly
    pip install rlm

Usage with Real RLM:
    from aragora.rlm import AragoraRLM, DebateContextAdapter, HAS_OFFICIAL_RLM

    if HAS_OFFICIAL_RLM:
        # Real RLM - context in REPL, LLM writes code to query
        rlm = AragoraRLM(backend="openai", model="gpt-4o")
        adapter = DebateContextAdapter(rlm)
        answer = await adapter.query_debate(
            "What were the main disagreements?",
            debate_result
        )
    else:
        # Fallback - hierarchical summarization (still useful, but not RLM)
        from aragora.rlm import HierarchicalCompressor
        compressor = HierarchicalCompressor()
        result = await compressor.compress(content, source_type="debate")

Fallback Mode (when rlm package not installed):
- Uses HierarchicalCompressor for LLM-based summarization
- Creates abstraction levels (FULL, DETAILED, SUMMARY, ABSTRACT, METADATA)
- Preserves semantics but doesn't provide true RLM capabilities
- No REPL environment, no recursive self-calls

Integration points in Aragora:
1. Debate context queries - Ask questions about long debate histories
2. Knowledge Mound queries - Recursive retrieval with abstraction
3. Repository understanding - Hierarchical codebase models
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
from .batch import (
    BatchConfig,
    BatchItemStatus,
    BatchItemResult,
    BatchResult,
    llm_batch,
    llm_batch_detailed,
    llm_batch_with_progress,
    batch_map,
    batch_filter,
    batch_first,
    batch_race,
)
from .adapter import (
    RLMContextAdapter,
    RegisteredContent,
    get_adapter,
)
from .factory import (
    get_rlm,
    get_compressor,
    compress_and_query,
    reset_singleton,
    get_factory_metrics,
    reset_metrics,
    log_metrics_summary,
    RLMFactoryMetrics,
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
    # Batch parallelism
    "BatchConfig",
    "BatchItemStatus",
    "BatchItemResult",
    "BatchResult",
    "llm_batch",
    "llm_batch_detailed",
    "llm_batch_with_progress",
    "batch_map",
    "batch_filter",
    "batch_first",
    "batch_race",
    # Context adapter (external environment pattern)
    "RLMContextAdapter",
    "RegisteredContent",
    "get_adapter",
    # Factory (preferred entry point)
    "get_rlm",
    "get_compressor",
    "compress_and_query",
    "reset_singleton",
    # Factory metrics (observability)
    "get_factory_metrics",
    "reset_metrics",
    "log_metrics_summary",
    "RLMFactoryMetrics",
]
