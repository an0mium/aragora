# ADR-008: Recursive Language Model (RLM) Semantic Compression

## Status
Accepted (Updated Jan 2026 - Prime Intellect Alignment)

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

## Update: Prime Intellect Alignment (Jan 2026)

### Iterative Refinement Protocol

The RLM now supports iterative refinement following the Prime Intellect approach:

**RLMResult Schema** (`aragora/rlm/types.py`):
```python
@dataclass
class RLMResult:
    answer: str                          # The answer/response
    ready: bool = True                   # Whether answer is complete
    iteration: int = 0                   # Current refinement iteration
    refinement_history: list[str] = []   # History of refinements
    confidence: float = 0.0              # Answer confidence score
    tokens_processed: int = 0            # Tokens consumed
    nodes_examined: list[str] = []       # Context nodes examined
    sub_calls_made: int = 0              # Sub-LM calls made
```

**Refinement Loop** (`aragora/rlm/bridge.py`):
```python
async def query_with_refinement(
    self,
    query: str,
    context: RLMContext,
    max_iterations: int = 3,
    strategy: str = "auto",
    feedback_generator: Optional[Callable] = None,
) -> RLMResult:
    """
    Execute query with iterative refinement.

    The loop continues until:
    - result.ready == True (LLM signals completion)
    - max_iterations is reached

    Between iterations, feedback is injected to guide improvement.
    """
```

### REPL Primitives

New REPL primitives for refinement control:

| Primitive | Usage | Description |
|-----------|-------|-------------|
| `FINAL(answer, ready)` | `FINAL("result", ready=False)` | Return answer with readiness flag |
| `SET_READY(bool)` | `SET_READY(True)` | Explicitly set readiness status |
| `FEEDBACK()` | `feedback = FEEDBACK()` | Get feedback from previous iteration |

### REST API Endpoints

New API endpoints for RLM operations:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/debates/{id}/query-rlm` | Query debate with refinement |
| POST | `/api/debates/{id}/compress` | Compress debate context |
| GET | `/api/debates/{id}/context/{level}` | Get abstraction level |
| GET | `/api/debates/{id}/refinement-status` | Check refinement progress |
| POST | `/api/knowledge/query-rlm` | Query knowledge mound |

**Example: Query with Refinement**
```bash
curl -X POST /api/debates/debate-123/query-rlm \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": "What was the consensus on pricing?",
    "strategy": "auto",
    "max_iterations": 3,
    "start_level": "SUMMARY"
  }'

# Response:
{
  "answer": "The debate reached consensus that...",
  "ready": true,
  "iteration": 1,
  "refinement_history": ["Initial attempt..."],
  "confidence": 0.85,
  "nodes_examined": ["node-1", "node-2"],
  "tokens_processed": 5000,
  "sub_calls_made": 3
}
```

### RL Training Infrastructure

New training module (`aragora/rlm/training/`) for reinforcement learning:

**Components:**
- `buffer.py` - Experience replay buffer with trajectory storage
- `reward.py` - Reward models for debate outcomes
- `policy.py` - Policy interfaces for strategy/refinement decisions

**Reward Model:**
```python
class DebateOutcomeReward(RewardModel):
    consensus_weight: float = 0.4    # Reached consensus?
    efficiency_weight: float = 0.2   # Fewer sub-calls is better
    confidence_weight: float = 0.2   # Higher confidence is better
    iteration_penalty: float = 0.1   # Fewer refinements is better
```

### Prometheus Metrics

New metrics for observability (`aragora/rlm/metrics.py`):

| Metric | Type | Description |
|--------|------|-------------|
| `rlm_refinement_iterations` | Histogram | Iterations per query |
| `rlm_refinement_success` | Counter | Successful refinements |
| `rlm_refinement_duration` | Histogram | Time per refinement |
| `rlm_ready_false_rate` | Counter | Not-ready responses by iteration |

### Backward Compatibility

All changes are backward compatible:
- `ready=True` default preserves existing behavior
- New methods are additive (don't replace existing)
- Increased output limit (8192 chars) is more permissive

## References
- `aragora/rlm/types.py` - Type definitions
- `aragora/rlm/compressor.py` - Compression implementation
- `aragora/rlm/bridge.py` - Integration bridge (query_with_refinement)
- `aragora/rlm/repl.py` - REPL with FINAL/SET_READY/FEEDBACK
- `aragora/rlm/metrics.py` - Prometheus metrics
- `aragora/rlm/training/` - RL training infrastructure
- `aragora/server/handlers/features/rlm.py` - REST API handler
- `aragora/memory/cross_debate_rlm.py` - Memory integration
- `aragora/debate/context_gatherer.py` - Context retrieval
- arXiv:2512.24601 - RLM paper reference
