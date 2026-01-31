# RLM (Recursive Language Model) Concepts

REPL-based programmatic context access for handling large content.

## Overview

RLM provides models with a REPL (Read-Eval-Print Loop) where they can write code to examine and query large contexts programmatically, rather than relying on compression or truncation.

**Key distinction**: RLM is NOT compression - it's programmatic context access.

## Modes

| Mode | Description | When Used |
|------|-------------|-----------|
| **TRUE_RLM** | REPL-based (preferred) | When `rlm` package installed |
| **COMPRESSION** | Hierarchical summarization | Fallback when no RLM |
| **AUTO** | TRUE_RLM if available, else compression | Default |

## Usage

### Factory Pattern

```python
from aragora.rlm import get_rlm, RLMMode

# AUTO mode - prefers TRUE RLM (recommended)
rlm = get_rlm()
result = await rlm.compress_and_query(
    query="What is the main topic?",
    content=long_document,
    source_type="document"
)

# Check which mode was used
if result.used_true_rlm:
    print("Used TRUE RLM (REPL-based)")
elif result.used_compression_fallback:
    print("Used compression fallback")
```

### Requiring TRUE RLM

```python
# Strict mode - raises if TRUE RLM unavailable
rlm = get_rlm(mode=RLMMode.TRUE_RLM)

# Or via parameter
rlm = get_rlm(require_true_rlm=True)

# Or via decorator
from aragora.rlm import require_true_rlm_decorator

@require_true_rlm_decorator()
async def analyze_debate(debate_result):
    rlm = get_rlm()
    return await rlm.compress_and_query(...)
```

### Convenience Function

```python
from aragora.rlm import compress_and_query

result = await compress_and_query(
    query="Summarize the key points",
    content=long_document,
    source_type="document"
)
```

## Configuration

### RLMConfig

```python
from aragora.rlm import RLMConfig, RLMMode

config = RLMConfig(
    mode=RLMMode.AUTO,                    # Mode selection
    require_true_rlm=False,               # Strict mode
    warn_on_compression_fallback=True,    # Log warning on fallback
    max_content_bytes=10_000_000,         # 10MB limit
    max_repl_memory_mb=512,               # REPL memory limit
    target_tokens=4000,                   # Compression target
    overlap_tokens=200,                   # Chunk overlap
)

rlm = get_rlm(config=config)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ARAGORA_RLM_MODE` | `true_rlm`, `compression`, or `auto` |
| `ARAGORA_RLM_REQUIRE_TRUE` | `true` to require TRUE RLM |
| `ARAGORA_RLM_WARN_FALLBACK` | `false` to suppress fallback warnings |
| `ARAGORA_RLM_MAX_CONTENT_BYTES` | Content size limit |
| `ARAGORA_RLM_MAX_REPL_MEMORY_MB` | REPL memory limit |
| `ARAGORA_RLM_TARGET_TOKENS` | Compression target tokens |
| `ARAGORA_RLM_OVERLAP_TOKENS` | Chunk overlap tokens |

## TRUE RLM Architecture

When the official `rlm` package is installed:

```
User Query + Large Content
         ↓
    RLM Bridge (AragoraRLM)
         ↓
    Official RLM Library
         ↓
    Model REPL Session
         ↓
    Model writes code to examine content
         ↓
    Code execution → results
         ↓
    Model formulates answer
         ↓
    RLMResult
```

The model can:
- Search through content
- Extract specific sections
- Run analysis code
- Build structured understanding
- Query incrementally

## Compression Fallback

When TRUE RLM is unavailable, hierarchical compression is used:

```
Large Content
     ↓
Chunking (with overlap)
     ↓
Per-chunk summarization
     ↓
Summary aggregation
     ↓
Final compression
     ↓
Compressed context + Query → LLM → Answer
```

### HierarchicalCompressor

```python
from aragora.rlm import get_compressor

compressor = get_compressor()
compressed = await compressor.compress(
    content=large_document,
    target_tokens=4000
)
```

## RLMResult

```python
@dataclass
class RLMResult:
    answer: str                      # The response
    used_true_rlm: bool             # TRUE RLM was used
    used_compression_fallback: bool  # Compression was used
    original_tokens: int            # Input size
    compressed_tokens: int          # After compression (if used)
    compression_ratio: float        # Reduction ratio
    processing_time_ms: float       # Total time
    source_type: str                # document, debate, email, etc.
```

## Observability

### Factory Metrics

```python
from aragora.rlm import get_factory_metrics, log_metrics_summary

metrics = get_factory_metrics()
print(f"TRUE RLM calls: {metrics['true_rlm_calls']}")
print(f"Compression fallback: {metrics['compression_fallback_calls']}")
print(f"Success rate: {metrics['successful_queries'] / (metrics['successful_queries'] + metrics['failed_queries']):.1%}")

# Or log summary
log_metrics_summary()
```

### Mode Info

```python
from aragora.rlm import get_rlm_mode_info

info = get_rlm_mode_info()
# {
#     "true_rlm_available": True,
#     "effective_mode": "auto",
#     "env_mode": None,
#     "env_require": False,
#     "warning": None
# }
```

## Singleton Pattern

The factory provides a singleton for efficiency:

```python
rlm1 = get_rlm()  # Creates new instance
rlm2 = get_rlm()  # Returns cached singleton

# Force new instance
rlm3 = get_rlm(force_new=True)

# Reset singleton
from aragora.rlm import reset_singleton
reset_singleton()
```

## Best Practices

1. **Always use factory**: `get_rlm()` handles detection and fallback
2. **Check result flags**: Know which mode was actually used
3. **Use AUTO mode**: Let the system pick the best available
4. **Monitor metrics**: Track TRUE RLM vs fallback usage
5. **Set appropriate limits**: Configure memory/size limits for your use case

## Installation

```bash
# Basic (compression fallback only)
pip install aragora

# With TRUE RLM support (recommended)
pip install aragora[rlm]
# or
pip install rlm
```

## Implementation Files

| Component | Source |
|-----------|--------|
| Factory | `aragora/rlm/factory.py` |
| Bridge | `aragora/rlm/bridge.py` |
| Compressor | `aragora/rlm/compressor.py` |
| Types | `aragora/rlm/types.py` |
| Handler | `aragora/rlm/handler.py` |

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - RLM usage in main documentation
- [docs/ENVIRONMENT.md](../ENVIRONMENT.md) - Environment configuration
