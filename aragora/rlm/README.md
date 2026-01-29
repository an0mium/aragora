# RLM - Recursive Language Models

Context management enabling LLMs to handle long context windows by treating context as an external environment queried programmatically.

Based on *"Recursive Language Models"* (arXiv:2512.24601) by Prime Intellect.

## Quick Start

```python
from aragora.rlm import get_rlm, is_true_rlm_available

# Get RLM instance (prefers TRUE RLM, auto-fallback to compression)
rlm = get_rlm()

# Query long context
result = await rlm.compress_and_query(
    query="What is the main consensus?",
    content=long_debate_history,
    source_type="debate"
)

# Check which approach was used
if result.used_true_rlm:
    print("Used TRUE RLM (REPL-based)")
elif result.used_compression_fallback:
    print("Used compression fallback")
```

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `get_rlm()` | `factory.py` | Main entry point (singleton) |
| `AragoraRLM` | `bridge.py` | Official RLM library wrapper |
| `RLMEnvironment` | `repl.py` | Sandboxed Python execution |
| `RLMContextAdapter` | `adapter.py` | External environment pattern |
| `HierarchicalCompressor` | `compressor.py` | Compression fallback |

## Architecture

```
rlm/
├── factory.py           # Singleton factory with metrics
├── types.py             # RLMConfig, RLMContext, RLMResult
├── bridge.py            # Official RLM integration
├── repl.py              # Sandboxed REPL environment
├── adapter.py           # External environment pattern
├── strategies.py        # Decomposition strategies
├── batch.py             # Parallel processing
├── compressor.py        # Hierarchical compression
├── debate_helpers.py    # Debate context navigation
├── knowledge_helpers.py # Knowledge Mound helpers
├── exceptions.py        # RLM-specific exceptions
└── training/            # RL training pipeline
    ├── buffer.py        # Experience replay
    ├── policy.py        # Strategy selection
    ├── reward.py        # Reward from debate outcomes
    └── trainer.py       # Training orchestration
```

## Two Operational Modes

### TRUE RLM (Preferred)
REPL-based: Model writes Python code to examine context stored as variables.

```python
# Context stored externally, not in prompts
context = DebateREPLContext(messages=[...], agents=[...])

# Model generates code like:
# relevant = [m for m in messages if m.agent == "claude"]
# summary = summarize(relevant[:5])
```

### Compression (Fallback)
Hierarchical summarization when official RLM library unavailable.

```python
# 5 abstraction levels
FULL → DETAILED → SUMMARY → ABSTRACT → METADATA
```

## Decomposition Strategies

| Strategy | Purpose |
|----------|---------|
| `PEEK` | Examine initial sections |
| `GREP` | Keyword/regex search |
| `PARTITION_MAP` | Chunk and process |
| `SUMMARIZE` | Recursive summarization |
| `HIERARCHICAL` | Navigate abstraction trees |
| `AUTO` | LLM selects best strategy |

## Batch Processing

```python
from aragora.rlm.batch import llm_batch, batch_map

# Process multiple items concurrently
results = await batch_map(
    items=documents,
    func=analyze_document,
    max_concurrent=5
)
```

## Security Features

- **REPL Sandboxing**: AST validation, blocked imports/builtins
- **Regex Protection**: ReDoS prevention, timeout enforcement
- **Memory Limits**: Per-value caps (10MB default)
- **Circuit Breaker**: Resilience patterns for provider failures

## Configuration

```python
from aragora.rlm import RLMConfig

config = RLMConfig(
    max_content_bytes=40_000_000,  # 40MB
    max_recursion_depth=2,
    target_tokens_per_level=8000,
    compression_ratio=0.3,
    cache_ttl=3600
)
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARAGORA_RLM_MODE` | `auto` | Mode (true_rlm, compression, auto) |
| `ARAGORA_RLM_REQUIRE_TRUE` | `false` | Enforce TRUE RLM |
| `ARAGORA_RLM_MAX_CONTENT_BYTES` | `40000000` | Max content size |
| `ARAGORA_RLM_TARGET_TOKENS` | `8000` | Target tokens per level |

## Metrics

```python
from aragora.rlm import get_factory_metrics

metrics = get_factory_metrics()
print(f"TRUE RLM calls: {metrics['true_rlm_calls']}")
print(f"Compression calls: {metrics['compression_calls']}")
print(f"Fallbacks: {metrics['fallback_to_compression']}")
```

## Related

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [Debate](../debate/README.md) - Debate orchestration
- [Knowledge Mound](../knowledge/mound/README.md) - Knowledge storage
