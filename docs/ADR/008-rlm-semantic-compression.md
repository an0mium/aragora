# ADR-008: Recursive Language Model (RLM) Semantic Compression

## Status
Accepted

## Context
Multi-agent debates generate extensive context that exceeds LLM token limits. The system needed a strategy to:
- Compress long conversation histories without losing critical information
- Maintain semantic coherence across abstraction levels
- Enable efficient context retrieval for subsequent rounds
- Support hierarchical reasoning over compressed representations

Traditional truncation approaches lose important early context and disrupt chain-of-thought reasoning.

## Decision
We implemented a **Recursive Language Model (RLM)** approach based on arXiv:2512.24601, providing hierarchical semantic compression for debate context.

### Core Components

**AbstractionLevel** (`aragora/rlm/types.py`):
```python
class AbstractionLevel(Enum):
    FULL = 0        # Original content
    DETAILED = 1    # ~50% compression
    SUMMARY = 2     # ~80% compression
    ABSTRACT = 3    # ~95% compression
    METADATA = 4    # Tags and routing only
```

**DecompositionStrategy**:
- `PEEK` - Inspect initial sections for structure
- `GREP` - Keyword-based relevance filtering
- `PARTITION_MAP` - Recursive chunking with parallel processing
- `SUMMARIZE` - Extract key points from subsets
- `HIERARCHICAL` - Navigate pre-built abstraction hierarchy
- `AUTO` - Strategy selection by the RLM itself

**RLMConfig**:
```python
@dataclass
class RLMConfig:
    root_model: str = "claude"     # Model for root LM
    sub_model: str = "gpt-4o-mini" # Model for sub-LM calls
    max_depth: int = 2             # Maximum recursion depth
    max_sub_calls: int = 10        # Maximum sub-LM calls per level
    target_tokens: int = 4000      # Target context per level
    compression_ratio: float = 0.3 # Target compression per level
```

### Integration Points

**Debate Context** (`aragora/debate/context.py`):
- RLM compression applied to conversation history
- Abstraction hierarchy maintained across rounds
- Citations preserve source references

**Memory System** (`aragora/memory/cross_debate_rlm.py`):
- Cross-debate learning with RLM compression
- Pattern extraction from compressed histories
- Semantic similarity across abstraction levels

**Context Gatherer** (`aragora/debate/context_gatherer.py`):
- Hybrid retrieval: dense vectors + RLM navigation
- Abstraction-aware relevance scoring

### Compression Pipeline

1. **Chunking**: Split context into semantic units
2. **Parallel Compression**: Run sub-LM calls on chunks
3. **Hierarchical Merging**: Build abstraction tree
4. **Citation Preservation**: Track source locations
5. **Cache Invalidation**: TTL-based cache for compressions

## Consequences

**Positive:**
- 5-10x context expansion capability
- Semantic coherence preserved across abstraction levels
- Parallel sub-calls reduce latency
- Citations enable source tracing
- Cache reduces repeated compression costs

**Negative:**
- Additional API calls increase cost (~20-30%)
- Compression latency for initial processing
- Potential information loss at high abstraction levels
- Complexity in debugging compression decisions

**Mitigations:**
- Use cheaper models (gpt-4o-mini) for sub-calls
- Cache aggressively with 1-hour TTL
- Preserve FULL level for critical operations
- Include citation format for traceability

## References
- `aragora/rlm/types.py` - Type definitions
- `aragora/rlm/compressor.py` - Compression implementation
- `aragora/rlm/bridge.py` - Integration bridge
- `aragora/memory/cross_debate_rlm.py` - Memory integration
- `aragora/debate/context_gatherer.py` - Context retrieval
- arXiv:2512.24601 - RLM paper reference
