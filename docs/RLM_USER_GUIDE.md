# Recursive Language Models (RLM) User Guide

## Overview

Aragora integrates Recursive Language Models (RLM) based on the paper ["Recursive Language Models" (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601) by Zhang, Kraska, and Khattab.

**Key Insight**: Long context should not be fed directly into neural networks. Instead, context is stored in a REPL environment where the LLM can programmatically examine, search, and recursively query it.

## Installation

```bash
# Install Aragora with RLM support
pip install aragora[rlm]

# Or install the official RLM library directly
pip install rlm
```

## Quick Start

### Basic Usage

```python
from aragora.rlm import AragoraRLM, DebateContextAdapter, HAS_OFFICIAL_RLM

# Check if official RLM is available
if HAS_OFFICIAL_RLM:
    # Create RLM instance with official library
    rlm = AragoraRLM(backend="openai", model="gpt-4o")
    adapter = DebateContextAdapter(rlm)

    # Query a debate
    answer = await adapter.query_debate(
        "What were the main disagreements?",
        debate_result
    )
else:
    # Fallback to hierarchical compression
    from aragora.rlm import HierarchicalCompressor
    compressor = HierarchicalCompressor()
    result = await compressor.compress(content, source_type="debate")
```

### With Debate Context

```python
from aragora.rlm import DebateContextAdapter, create_aragora_rlm

# Create RLM for debate analysis
rlm = create_aragora_rlm(backend="anthropic", model="claude-3-5-sonnet-20241022")
adapter = DebateContextAdapter(rlm)

# Load a debate result
debate = await arena.run()

# Query the debate history
answer = await adapter.query_debate(
    "What arguments changed during the debate?",
    debate
)

# Get specific information
claims = await adapter.query_debate(
    "List all verifiable claims made by claude agent",
    debate
)
```

## Concepts

### Abstraction Levels

RLM maintains content at multiple abstraction levels:

| Level | Description | Compression |
|-------|-------------|-------------|
| `FULL` | Original full content | 0% |
| `DETAILED` | Detailed summary | ~50% |
| `SUMMARY` | Key points | ~80% |
| `ABSTRACT` | High-level overview | ~95% |
| `METADATA` | Tags and routing info | ~99% |

### Decomposition Strategies

The LLM can use different strategies to examine context:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `PEEK` | Inspect initial sections | Understanding structure |
| `GREP` | Keyword/regex search | Finding specific mentions |
| `PARTITION_MAP` | Chunk and recurse | Large documents |
| `SUMMARIZE` | Extract key points | Getting overview |
| `HIERARCHICAL` | Navigate abstraction tree | Multi-level queries |
| `AUTO` | Let RLM decide | General use |

## Configuration

### RLMConfig Options

```python
from aragora.rlm import RLMConfig, DecompositionStrategy

config = RLMConfig(
    # Model configuration
    root_model="claude",          # Primary model
    sub_model="gpt-4o-mini",      # Model for sub-calls (cheaper)

    # Recursion limits
    max_depth=2,                  # Maximum recursion depth
    max_sub_calls=10,             # Max sub-LM calls per level

    # Context management
    target_tokens=4000,           # Target size per level
    overlap_tokens=200,           # Overlap between chunks

    # Compression settings
    compression_ratio=0.3,        # Target compression per level
    preserve_structure=True,      # Maintain document structure

    # Strategy selection
    default_strategy=DecompositionStrategy.AUTO,

    # Performance
    parallel_sub_calls=True,      # Run sub-calls in parallel
    cache_compressions=True,      # Cache results
    cache_ttl_seconds=3600,       # Cache TTL

    # Output format
    include_citations=True,       # Include source references
    citation_format="[L{level}:{chunk}]",
)
```

## Integration with Debates

### Cognitive Load Limiter

RLM powers the cognitive load limiter to manage debate context:

```python
from aragora.debate import create_rlm_limiter, RLMCognitiveBudget

# Create RLM-based limiter
limiter = create_rlm_limiter(
    max_input_tokens=8000,
    compression_threshold=6000,
    summary_level="detailed",
)

# Use with Arena
arena = Arena(
    env, agents, protocol,
    use_rlm_limiter=True,
    rlm_limiter=limiter,
)
```

### Cross-Debate Memory

RLM enables efficient cross-debate memory retrieval:

```python
from aragora.memory.cross_debate_rlm import CrossDebateRLM

memory = CrossDebateRLM()

# Store debate outcome
await memory.store(debate_result)

# Query across debates
relevant = await memory.query(
    "Previous discussions about caching strategies",
    limit=5,
)
```

## Knowledge Mound Integration

Query the Knowledge Mound with RLM for recursive retrieval:

```python
from aragora.rlm import KnowledgeMoundAdapter

adapter = KnowledgeMoundAdapter(rlm, knowledge_mound)

# Recursive query with abstraction
result = await adapter.query(
    "What are the security best practices mentioned?",
    start_level=AbstractionLevel.SUMMARY,
    max_depth=2,
)
```

## API Reference

### Core Classes

#### `AragoraRLM`
Main RLM interface for Aragora integration.

```python
rlm = AragoraRLM(
    backend="openai",           # "openai", "anthropic", "local"
    model="gpt-4o",             # Model identifier
    config=RLMConfig(),         # Optional configuration
)

# Execute a query
result = await rlm.query(query, context)
```

#### `HierarchicalCompressor`
Creates multi-level abstractions of content.

```python
compressor = HierarchicalCompressor(config=RLMConfig())

# Compress content
result = await compressor.compress(
    content="...",
    source_type="debate",  # "debate", "code", "document", "text"
)

# Access levels
summary = result.context.get_at_level(AbstractionLevel.SUMMARY)
```

#### `DebateContextAdapter`
Specialized adapter for debate histories.

```python
adapter = DebateContextAdapter(rlm)

# Query debate
answer = await adapter.query_debate(query, debate_result)

# Get structured analysis
analysis = await adapter.analyze_positions(debate_result)
```

### Types

#### `RLMContext`
Hierarchical context representation.

```python
context = RLMContext(
    original_content="...",
    original_tokens=10000,
)

# Navigate hierarchy
summary = context.get_at_level(AbstractionLevel.SUMMARY)
node = context.get_node("node_id")
children = context.drill_down("parent_id")
```

#### `RLMResult`
Query result with provenance.

```python
result = RLMResult(
    answer="...",
    ready=True,                 # Complete answer
    nodes_examined=["n1", "n2"],
    citations=[...],
    confidence=0.85,
)
```

## Best Practices

### 1. Choose the Right Strategy

```python
# For finding specific information
query = RLMQuery(
    query="What did Claude say about security?",
    preferred_strategy=DecompositionStrategy.GREP,
)

# For understanding overall structure
query = RLMQuery(
    query="What are the main themes?",
    preferred_strategy=DecompositionStrategy.HIERARCHICAL,
)
```

### 2. Set Appropriate Limits

```python
config = RLMConfig(
    max_depth=2,              # Don't recurse too deep
    max_sub_calls=10,         # Limit API costs
    target_tokens=4000,       # Match model context
)
```

### 3. Use Caching

```python
config = RLMConfig(
    cache_compressions=True,
    cache_ttl_seconds=3600,   # 1 hour
)
```

### 4. Handle Fallback

```python
from aragora.rlm import HAS_OFFICIAL_RLM

if HAS_OFFICIAL_RLM:
    # Use full RLM capabilities
    result = await rlm.query(query, context)
else:
    # Graceful fallback
    compressor = HierarchicalCompressor()
    compressed = await compressor.compress(content)
    # Use compressed content with standard LLM call
```

## Troubleshooting

### "RLM library not installed"

Install the official library:
```bash
pip install rlm
```

### High Latency

- Reduce `max_depth` and `max_sub_calls`
- Enable `parallel_sub_calls`
- Use caching with `cache_compressions=True`

### Poor Quality Results

- Start with higher abstraction level
- Increase `target_tokens` for more context
- Use `preserve_structure=True` for documents

### Memory Issues

- Reduce `target_tokens`
- Limit `max_sub_calls`
- Clear cache periodically

## Further Reading

- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
- [Official RLM Repository](https://github.com/alexzhang13/rlm)
- [Aragora Memory Documentation](./MEMORY_STRATEGY.md)
