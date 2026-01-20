# RLM (Recursive Language Models) Integration Guide

This guide explains how Aragora integrates with RLM for context compression and hierarchical abstraction.

## Overview

RLM provides **context compression** to enable longer debates without hitting token limits. When debates exceed a configurable threshold (default: 3 rounds), RLM automatically compresses older messages while preserving recent context.

**Token Efficiency Gains**: 15-20% reduction in context size for extended debates.

## Quick Start

RLM is **enabled by default** in Aragora. No configuration needed for basic usage:

```python
from aragora import Arena, Environment, DebateProtocol

# RLM auto-enables after round 3
arena = Arena(
    environment=Environment(task="Complex analysis task"),
    agents=agents,
    protocol=DebateProtocol(rounds=10),  # Extended debate
)
result = await arena.run()
```

## Configuration

### ArenaConfig Options

```python
from aragora.debate import ArenaConfig

config = ArenaConfig(
    # RLM Settings
    use_rlm_limiter=True,              # Enable RLM compression (default: True)
    rlm_compression_threshold=3000,     # Chars before compression triggers
    rlm_max_recent_messages=5,          # Keep N recent messages at full detail
    rlm_summary_level="SUMMARY",        # Abstraction level: SUMMARY, DETAIL, FULL
    rlm_compression_round_threshold=3,  # Auto-enable after N rounds
)
```

### Abstraction Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `SUMMARY` | High-level key points only | Long debates (>10 rounds) |
| `DETAIL` | Moderate compression | Medium debates (5-10 rounds) |
| `FULL` | No compression | Short debates (<5 rounds) |

## Architecture

```
Debate Flow with RLM
────────────────────────────────────────────────────────
Round 1-3: Full context (no compression)
    │
    ▼
Round 4+: RLM compression auto-enables
    │
    ├── Recent messages (N=5): Full detail
    │
    └── Older messages: Hierarchical compression
            │
            ├── Level 1: Key arguments preserved
            ├── Level 2: Evidence summarized
            └── Level 3: Meta-summary
────────────────────────────────────────────────────────
```

## Core Components

### RLMCognitiveLoadLimiter

Main class for debate context compression:

```python
from aragora.debate.cognitive_limiter_rlm import (
    RLMCognitiveBudget,
    RLMCognitiveLoadLimiter,
)

# Configure budget
budget = RLMCognitiveBudget(
    enable_rlm_compression=True,
    compression_threshold=3000,
    max_recent_full_messages=5,
    summary_level="SUMMARY",
)

# Create limiter
limiter = RLMCognitiveLoadLimiter(budget=budget)

# Compress context
result = await limiter.compress_context_async(
    messages=debate_messages,
    critiques=critique_list,
)

print(f"Compression ratio: {result.original_chars / result.compressed_chars:.2f}x")
```

### RLM Factory

Singleton factory for RLM instances:

```python
from aragora.rlm import get_rlm, get_compressor, compress_and_query

# Get RLM instance (uses TRUE RLM if installed, fallback otherwise)
rlm = get_rlm()

# Get compressor directly
compressor = get_compressor()

# Convenience function for compress + query
result = await compress_and_query(
    query="What are the key arguments?",
    content=long_debate_transcript,
    source_type="debate",
)
```

## HTTP API Endpoints

### Generic RLM API (`/api/rlm/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rlm/compress` | POST | Compress content with hierarchical abstraction |
| `/api/rlm/query` | POST | Query compressed contexts |
| `/api/rlm/stats` | GET | Cache statistics and system status |
| `/api/rlm/strategies` | GET | List available compression strategies |
| `/api/rlm/contexts` | GET | List stored compressed contexts |
| `/api/rlm/context/{id}` | GET/DELETE | Get or delete specific context |

### Debate-Specific RLM API (`/api/debates/{id}/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/debates/{id}/query-rlm` | POST | Query debate with RLM refinement |
| `/api/debates/{id}/compress` | POST | Compress debate context |
| `/api/debates/{id}/context/{level}` | GET | Get at specific abstraction level |
| `/api/debates/{id}/refinement-status` | GET | Check iterative refinement progress |

### Example: Compress and Query

```bash
# Compress content
curl -X POST http://localhost:8080/api/rlm/compress \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Long debate transcript...",
    "source_type": "debate",
    "compression_levels": 3
  }'

# Query compressed context
curl -X POST http://localhost:8080/api/rlm/query \
  -H "Content-Type: application/json" \
  -d '{
    "context_id": "ctx_abc123",
    "query": "What were the main disagreements?"
  }'
```

## Compression Strategies

Available strategies for content decomposition:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `auto` | Automatic selection based on content | General use (recommended) |
| `peek` | Quick summary extraction | Fast overview |
| `grep` | Pattern-based extraction | Finding specific info |
| `partition_map` | Divide and conquer | Large documents |
| `summarize` | Comprehensive summary | Final synthesis |
| `hierarchical` | Multi-level abstraction | Extended debates |

## TRUE RLM vs Fallback

Aragora supports two modes:

### TRUE RLM (REPL-based)

When the official `rlm` package is installed:
- Full REPL-based execution
- Interactive query refinement
- More sophisticated compression

```bash
pip install rlm  # Optional: enables TRUE RLM
```

### Fallback (HierarchicalCompressor)

Without the `rlm` package:
- Compression-only mode
- Hierarchical abstraction levels
- No REPL execution

**Check mode:**
```python
from aragora.rlm import HAS_OFFICIAL_RLM
print(f"TRUE RLM available: {HAS_OFFICIAL_RLM}")
```

## Integration with Memory

RLM integrates with Aragora's memory systems:

### Cross-Debate Memory

```python
from aragora.memory.cross_debate_rlm import CrossDebateRLMMemory

memory = CrossDebateRLMMemory()
# Automatically uses RLM for context compression across debates
```

### Knowledge Mound

```python
# RLM-enhanced knowledge queries
from aragora.knowledge.mound.api.rlm import query_with_rlm

result = await query_with_rlm(
    mound=knowledge_mound,
    query="Find relevant precedents",
)
```

## Metrics & Monitoring

### Factory Metrics

```python
from aragora.rlm import get_factory_metrics

metrics = get_factory_metrics()
print(metrics)
# {
#     "get_rlm_calls": 15,
#     "successful_queries": 12,
#     "compression_fallback_calls": 3,
#     "singleton_hits": 14,
#     ...
# }
```

### Prometheus Export

RLM metrics are exported to Prometheus via `/metrics`:

- `aragora_rlm_compressions_total`
- `aragora_rlm_queries_total`
- `aragora_rlm_compression_ratio`
- `aragora_rlm_query_latency_seconds`

## Troubleshooting

### RLM Not Compressing

1. Check if compression threshold is reached:
   ```python
   # Default threshold is 3000 chars
   print(f"Context length: {len(context)}")
   print(f"Threshold: {config.rlm_compression_threshold}")
   ```

2. Verify RLM is enabled:
   ```python
   print(f"RLM enabled: {arena.use_rlm_limiter}")
   ```

### Low Compression Ratio

- Increase `compression_levels` (default: 3)
- Use `hierarchical` strategy for better compression
- Check if content is already concise

### Query Performance

- Enable caching: `cache_size=100, cache_ttl_seconds=3600`
- Use batch queries for multiple questions
- Consider TRUE RLM for complex queries

## Related Documentation

- [Arena Configuration](ARCHITECTURE.md#arena-config)
- [Memory Systems](MEMORY.md)
- [Knowledge Mound](KNOWLEDGE_MOUND.md)
- [API Reference](API_REFERENCE.md#rlm-endpoints)
