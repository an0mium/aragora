# RLM Integration Guide

> **Note:** This guide covers the `aragora/rlm/` module — programmatic context access via REPL.
> For RLM-pattern debate early-termination (ready signals, vote collection), see
> [RLM_GUIDE.md](./RLM_GUIDE.md). These are distinct systems that share a name.

## What RLM Does in Aragora

The `aragora.rlm` module integrates Recursive Language Models (arXiv:2512.24601 by Zhang, Kraska,
and Khattab). The core idea: long context must not be fed directly into the neural network.
Instead, context lives in a REPL environment as Python variables, and the LLM writes code to
examine, search, and partition it.

This is not compression. The model retains full access to the original context and decides
dynamically how to navigate it (grep, map-reduce, peek, hierarchical summary).

When the official `rlm` package is not installed, the module falls back to
`HierarchicalCompressor`, which pre-processes content into five abstraction levels (FULL,
DETAILED, SUMMARY, ABSTRACT, METADATA). Functional, but not true RLM.

## Installation

```bash
# With true RLM support
pip install aragora[rlm]

# Verify
python -c "from aragora.rlm import HAS_OFFICIAL_RLM; print(HAS_OFFICIAL_RLM)"
```

## Key Exports

| Export | Location | Purpose |
|--------|----------|---------|
| `AragoraRLM` | `bridge.py` | Main interface wrapping the official `rlm` library |
| `DebateContextAdapter` | `bridge.py` | Formats debate histories for RLM queries |
| `KnowledgeMoundAdapter` | `bridge.py` | Connects RLM to Knowledge Mound retrieval |
| `HierarchicalCompressor` | `compressor.py` | Fallback when `rlm` package is absent |
| `RLMEnvironment` | `repl.py` | Manages the REPL sandbox |
| `get_rlm` / `get_compressor` | `factory.py` | Preferred singleton entry points |
| `llm_batch` | `batch.py` | Parallel sub-LLM dispatch with early stopping |
| `RLMContextAdapter` | `adapter.py` | Register external content for REPL access |
| `HAS_OFFICIAL_RLM` | `bridge.py` | Runtime flag for conditional logic |

## Basic Usage

Use the factory functions — they handle singleton lifecycle and mode detection automatically:

```python
from aragora.rlm import get_rlm, compress_and_query, HAS_OFFICIAL_RLM

# Preferred: singleton instance, auto-selects true RLM or compression fallback
rlm = get_rlm()
if rlm:
    result = await rlm.compress_and_query(
        query="What consensus was reached on the rate limit approach?",
        content=debate_transcript,
        source_type="debate",
    )
    print(result.answer)

# Convenience wrapper (same as above, one call)
result = await compress_and_query(query, content, source_type="debate")
```

To require true RLM (raise if unavailable):

```python
from aragora.rlm import get_rlm, RLMMode

rlm = get_rlm(mode=RLMMode.TRUE_RLM)  # Raises RLMProviderError if rlm not installed
```

## Environment Configuration

```bash
# Auto-detect (default): prefer true RLM, fall back to compression
export ARAGORA_RLM_MODE=auto

# Force true RLM (fail fast if unavailable)
export ARAGORA_RLM_MODE=true_rlm
export ARAGORA_RLM_BACKEND=anthropic   # or openai

# Force compression-only
export ARAGORA_RLM_MODE=compression
```

## Batch Parallelism

`llm_batch` dispatches work to multiple sub-LLMs in parallel, each with a fresh context window:

```python
from aragora.rlm import llm_batch, BatchConfig

config = BatchConfig(
    max_concurrent=5,
    timeout_per_item=30.0,
    retry_on_error=True,
    max_retries=2,
)

results = await llm_batch(
    items=[proposal1, proposal2, proposal3],
    process_fn=generate_critique,
    config=config,
    # Optional: stop as soon as a condition is met
    early_stop=lambda partial: has_clear_majority(partial),
)
```

Convenience combinators: `batch_map`, `batch_filter`, `batch_first` (return first successful
result), `batch_race` (return fastest).

## KnowledgeMound Integration

Two KM adapters bridge RLM with the Knowledge Mound:

- `aragora/knowledge/mound/adapters/rlm_adapter.py` — stores RLM query results and compression
  hierarchies as KM entries
- `aragora/knowledge/mound/adapters/rlm_context_adapter.py` — makes KM content available to the
  RLM REPL for recursive retrieval

The `KnowledgeMoundAdapter` in `bridge.py` wraps these for direct use:

```python
from aragora.rlm import KnowledgeMoundAdapter, AragoraRLM

rlm = AragoraRLM(backend="anthropic", model="claude-opus-4-7")
km_adapter = KnowledgeMoundAdapter(rlm, mound=knowledge_mound)
result = await km_adapter.query("Summarize findings on auth module security")
```

## Arena Context Compression

When RLM compression is enabled on a debate, the context init phase automatically:

1. Compresses long context via `AragoraRLM.compress_and_query()`
2. Builds an `RLMContext` with `AbstractionLevel.SUMMARY` node from the result
3. Sets it on the prompt builder via `set_rlm_context()` so agents can drill down
4. Emits Prometheus metrics (`record_rlm_compression`) for success/failure tracking

This means agents in the debate can access both the compressed summary and, via the
prompt builder, navigate to detailed sections on demand.

## Trajectory Learning

`DebateTrajectoryCollector` (in `debate_integration.py`) records RLM query trajectories from
debate sessions and feeds them back as training signal via the Nomic Loop:

```python
from aragora.rlm import create_training_hook

# Attach to Arena for automatic trajectory collection
hook = create_training_hook(collector=get_debate_trajectory_collector())
arena = Arena(env, agents, protocol, hooks={"post_debate": hook})
```

### Per-Round Recording

For richer training data, the debate rounds phase records proposals and critiques each round:

```python
from aragora.rlm.debate_integration import get_debate_trajectory_collector

collector = get_debate_trajectory_collector()

# Called automatically each round when RLM compression is enabled
collector.record_round(
    debate_id="debate_001",
    round_num=0,
    proposals=[{"agent": "alice", "content": "My proposal..."}],
    critiques=[{"agent": "bob", "content": "I disagree because..."}],
    convergence_similarity=0.3,
)

# Finalize merges per-round steps with outcome data
trajectory = collector.record_debate_outcome(
    debate_id="debate_001",
    task="Design caching system",
    consensus_reached=True,
    confidence=0.85,
)
# trajectory.steps contains per-round data followed by outcome steps
```

## Observability

RLM compression emits Prometheus metrics automatically when `prometheus_client` is installed.
The metrics module is at `aragora/server/prometheus_rlm.py`.

**Prometheus metrics emitted:**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `aragora_rlm_compressions_total` | Counter | `source_type`, `status` | Total compression calls (success/failure) |
| `aragora_rlm_compression_ratio` | Histogram | `source_type` | Ratio of compressed/original tokens |
| `aragora_rlm_tokens_saved_total` | Counter | `source_type` | Tokens saved by compression |
| `aragora_rlm_compression_duration_seconds` | Histogram | `source_type`, `levels` | Time per compression |
| `aragora_rlm_queries_total` | Counter | `query_type`, `level` | RLM context queries |
| `aragora_rlm_cache_hits_total` | Counter | — | Cache hits |
| `aragora_rlm_cache_misses_total` | Counter | — | Cache misses |
| `aragora_rlm_refinement_iterations` | Histogram | `strategy` | Iterations until ready=True |

**Programmatic access:**

```python
from aragora.rlm import get_factory_metrics

metrics = get_factory_metrics()
print(f"True RLM calls: {metrics['true_rlm_calls']}")
print(f"Compression fallback calls: {metrics['compression_fallback_calls']}")
```

## Error Handling

```python
from aragora.rlm import (
    RLMError,
    RLMTimeoutError,
    RLMContextOverflowError,
    RLMProviderError,
    RLMCircuitOpenError,
)

try:
    result = await rlm.query(context, question)
except RLMCircuitOpenError:
    # Circuit breaker open — RLM provider unreachable
    result = fallback_summary(context)
except RLMTimeoutError:
    result = truncated_response(context)
```

## Further Reading

- Paper: "Recursive Language Models" (arXiv:2512.24601) — Zhang, Kraska, Khattab
- [RLM Developer Guide](./RLM_DEVELOPER_GUIDE.md) — in-depth factory and adapter patterns
- [RLM User Guide](./RLM_USER_GUIDE.md) — end-user perspective and quick start
- [RLM Guide](./RLM_GUIDE.md) — debate early-termination patterns (separate concept)
