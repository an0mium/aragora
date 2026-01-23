---
title: RLM Developer Guide
description: RLM Developer Guide
---

# RLM Developer Guide

This guide covers practical integration of the RLM (Recursive Language Models) module in Aragora applications. It complements the [RLM User Guide](./rlm-user) and [RLM Integration Guide](./rlm-integration).

## Two Concepts Named "RLM"

Aragora uses "RLM" for two related but distinct concepts:

1. **RLM Module** (arXiv:2512.24601) - The `aragora/rlm/` package that treats long context as an external environment with REPL-based navigation.

2. **RLM Patterns** (Prime Intellect) - Debate termination patterns using `ready` signals and iterative refinement (covered in [RLM_GUIDE.md](./rlm)).

This guide focuses on the **RLM Module** for context compression and navigation.

## Quick Start

### Using the Factory (Recommended)

```python
from aragora.rlm import get_rlm, get_compressor

# Preferred: get singleton RLM instance
rlm = get_rlm()
if rlm:
    result = await rlm.query(context, question, strategy="auto")
    print(result.answer)

# Alternative: get compressor directly
compressor = get_compressor()
if compressor:
    hierarchy = await compressor.compress(long_text)
    print(f"Compression levels: {len(hierarchy.levels)}")
```

### Environment Configuration

```bash
# Enable TRUE RLM (requires `pip install rlm`)
export ARAGORA_RLM_MODE=true_rlm
export ARAGORA_RLM_BACKEND=anthropic  # or openai

# Use compression fallback only
export ARAGORA_RLM_MODE=compression

# Auto-detect (default)
export ARAGORA_RLM_MODE=auto
```

## When to Use RLM

| Scenario | Use RLM? | Why |
|----------|----------|-----|
| Debate history > 50K tokens | Yes | Prevents context window overflow |
| Cross-debate knowledge retrieval | Yes | Efficient compressed storage |
| Single-turn Q&A | No | Direct API call is simpler |
| Real-time streaming | Partial | Adds latency for compression |
| Document summarization | Yes | Hierarchical abstraction |

## RLM Modes

### TRUE_RLM Mode

When the official `rlm` package is installed, the LLM can:
- Write code to examine context programmatically
- Call itself recursively on context subsets
- Use REPL primitives: `grep()`, `peek()`, `partition()`

```python
from aragora.rlm import AragoraRLM, RLMMode

rlm = AragoraRLM(mode=RLMMode.TRUE_RLM)
result = await rlm.query(
    context=debate_history,
    question="What were the main disagreements?",
    strategy="grep",  # LLM writes grep code to find disagreements
)
```

### COMPRESSION Mode (Fallback)

When TRUE RLM is unavailable, uses `HierarchicalCompressor`:

```python
from aragora.rlm import HierarchicalCompressor, AbstractionLevel

compressor = HierarchicalCompressor(model="claude-sonnet-4-20250514")
hierarchy = await compressor.compress(
    content=long_document,
    target_levels=[
        AbstractionLevel.DETAILED,  # 50% compression
        AbstractionLevel.SUMMARY,   # 80% compression
    ]
)

# Navigate by level
detailed = hierarchy.get_level(AbstractionLevel.DETAILED)
summary = hierarchy.get_level(AbstractionLevel.SUMMARY)
```

## Compression Levels

| Level | Compression | Use Case |
|-------|------------|----------|
| FULL | 0% | Complete original context |
| DETAILED | 50% | Recent debate rounds, key arguments |
| SUMMARY | 80% | Overview with major points |
| ABSTRACT | 95% | Executive summary only |
| METADATA | 99% | Tags, topics, participants only |

## Integration Patterns

### Pattern 1: Debate Context Compression

```python
from aragora.rlm import get_rlm
from aragora.debate.context_gatherer import ContextGatherer

async def prepare_debate_context(debate_id: str, round_num: int):
    gatherer = ContextGatherer()
    raw_context = await gatherer.gather(debate_id)

    # Compress after round 3 to manage token limits
    if round_num > 3:
        rlm = get_rlm()
        if rlm:
            compressed = await rlm.compress(
                raw_context,
                preserve_recent=2,  # Keep last 2 rounds full
            )
            return compressed.get_context_for_prompt()

    return raw_context
```

### Pattern 2: Knowledge Mound Query

```python
from aragora.rlm import get_rlm

async def query_knowledge_with_rlm(query: str, workspace_id: str):
    rlm = get_rlm()
    if not rlm:
        # Fallback to direct query
        return await mound.query(query)

    # RLM-enhanced query with recursive navigation
    result = await rlm.query(
        context=await mound.get_workspace_knowledge(workspace_id),
        question=query,
        strategy="hierarchical",
    )
    return result.answer, result.sources
```

### Pattern 3: Cross-Debate Memory

```python
from aragora.memory.cross_debate_rlm import CrossDebateMemory

memory = CrossDebateMemory()

# Store with RLM compression
await memory.store_debate(
    debate_id="debate_123",
    topic="API design patterns",
    consensus="REST with OpenAPI spec",
    key_points=["Consistency", "Documentation"],
)

# Retrieve with RLM navigation
context = await memory.get_relevant_context(
    task="How should we design our new API?",
    max_tokens=2000,
)
```

## Error Handling

```python
from aragora.rlm import get_rlm, RLMUnavailableError

async def safe_rlm_query(context: str, question: str):
    rlm = get_rlm()

    if rlm is None:
        # RLM not available - use fallback
        return await direct_llm_query(context, question)

    try:
        result = await rlm.query(context, question)
        return result.answer
    except RLMUnavailableError:
        # TRUE RLM failed, compressor available
        return await rlm.query(
            context,
            question,
            mode=RLMMode.COMPRESSION,  # Force compression
        )
    except Exception as e:
        logger.warning(f"RLM query failed: \{e\}")
        return await direct_llm_query(context, question)
```

## Strategies

| Strategy | When to Use | Performance |
|----------|-------------|-------------|
| `auto` | Unknown query type | Good balance |
| `grep` | Finding specific patterns | Fast |
| `peek` | Quick overview | Very fast |
| `hierarchical` | Complex analysis | Slower but thorough |
| `partition_map` | Large documents | Parallelizable |
| `summarize` | Simple compression | Fast |

```python
# Strategy selection
result = await rlm.query(
    context=document,
    question="Find all security vulnerabilities",
    strategy="grep",  # Best for pattern matching
)

result = await rlm.query(
    context=debate_history,
    question="What is the overall consensus?",
    strategy="hierarchical",  # Best for synthesis
)
```

## Metrics & Monitoring

### Factory Metrics

```python
from aragora.rlm.factory import get_rlm_stats

stats = get_rlm_stats()
print(f"Mode: {stats['mode']}")
print(f"Compression calls: {stats['compression_count']}")
print(f"Avg compression time: {stats['avg_compression_ms']}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']}")
```

### Prometheus Metrics

Available at `/metrics` endpoint:

- `aragora_rlm_queries_total` - Total RLM queries
- `aragora_rlm_compression_seconds` - Compression latency histogram
- `aragora_rlm_tokens_saved` - Token reduction from compression
- `aragora_rlm_cache_hits` - Cache effectiveness

## Cache Management

```python
from aragora.rlm.compressor import get_compression_cache

cache = get_compression_cache()

# Cache stats
print(f"Size: {cache.size}/{cache.max_size}")
print(f"Hit rate: {cache.hit_rate:.1%}")

# Clear cache
cache.clear()

# Configure cache size
from aragora.rlm import RLMConfig

config = RLMConfig(
    cache_max_size=500,      # Max entries
    cache_ttl_seconds=3600,  # 1 hour TTL
)
```

## Testing

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_rlm():
    """Mock RLM for testing."""
    with patch("aragora.rlm.factory.get_rlm") as mock:
        rlm = AsyncMock()
        rlm.query.return_value.answer = "Test answer"
        mock.return_value = rlm
        yield rlm

async def test_my_function(mock_rlm):
    result = await my_rlm_function("question")
    mock_rlm.query.assert_called_once()
    assert result == "Test answer"
```

## Best Practices

1. **Use the factory** - Always use `get_rlm()` instead of instantiating directly
2. **Handle unavailability** - RLM may not be available; have fallbacks
3. **Choose appropriate strategy** - Match strategy to query type
4. **Monitor cache hit rates** - Low rates indicate potential optimization opportunities
5. **Preserve recent context** - Keep last 1-2 debate rounds uncompressed
6. **Log compression ratios** - Track token savings for cost analysis

## Common Issues

### RLM Returns None

```python
rlm = get_rlm()
if rlm is None:
    # Check: Is `pip install aragora[rlm]` installed?
    # Check: Is ARAGORA_RLM_MODE set correctly?
    # Check: Are required API keys present?
    pass
```

### High Latency

- Use `strategy="peek"` for quick checks
- Increase cache TTL for repeated queries
- Consider `mode=RLMMode.COMPRESSION` if TRUE_RLM is slow

### Token Limits Still Exceeded

- Reduce `preserve_recent` count
- Use more aggressive compression levels
- Split context into smaller chunks

## API Reference

See the main API reference in [RLM_USER_GUIDE.md](./rlm-user#api-reference).

### Factory Functions

| Function | Description |
|----------|-------------|
| `get_rlm()` | Get singleton RLM instance |
| `get_compressor()` | Get direct compressor access |
| `compress_and_query()` | Convenience for compress+query |
| `get_rlm_stats()` | Get usage metrics |
| `reset_rlm()` | Reset singleton (testing only) |

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `mode` | `auto` | TRUE_RLM, COMPRESSION, or AUTO |
| `backend` | `anthropic` | LLM backend for queries |
| `cache_max_size` | 100 | Max cached compressions |
| `cache_ttl_seconds` | 1800 | Cache entry lifetime |
| `default_strategy` | `auto` | Default query strategy |
| `preserve_recent` | 2 | Rounds to keep uncompressed |

## Related Documentation

- [RLM User Guide](./rlm-user) - Concepts and quick start
- [RLM Integration Guide](./rlm-integration) - Architecture and HTTP API
- [RLM Patterns Guide](./rlm) - Prime Intellect debate termination
- [ADR-008](../analysis/adr/008-rlm-semantic-compression) - Architecture decision record
