"""
Prometheus metrics for RLM (Recursive Language Models) compression.

Provides metrics for monitoring RLM compression efficiency, token savings,
and hierarchical context management performance.

Usage:
    from aragora.rlm.metrics import record_compression, record_query

    # Record a compression operation
    record_compression(
        source_type="debate",
        original_tokens=10000,
        compressed_tokens=3000,
        levels=3,
        duration_seconds=0.5,
    )

    # Record a query operation
    record_query(
        query_type="drill_down",
        level="SUMMARY",
        duration_seconds=0.02,
    )

Requirements:
    pip install prometheus-client

Environment Variables:
    METRICS_ENABLED: Set to "true" to enable metrics (default: true)

See docs/OBSERVABILITY.md for configuration guide.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# Prometheus metrics - initialized lazily
_initialized = False

# Metric instances (will be set during initialization)
RLM_COMPRESSIONS: Any = None
RLM_COMPRESSION_RATIO: Any = None
RLM_TOKENS_SAVED: Any = None
RLM_COMPRESSION_DURATION: Any = None
RLM_QUERIES: Any = None
RLM_QUERY_DURATION: Any = None
RLM_CACHE_HITS: Any = None
RLM_CACHE_MISSES: Any = None
RLM_CONTEXT_LEVELS: Any = None
RLM_MEMORY_USAGE: Any = None


def _init_metrics() -> bool:
    """Initialize Prometheus metrics lazily."""
    global _initialized
    global RLM_COMPRESSIONS, RLM_COMPRESSION_RATIO, RLM_TOKENS_SAVED
    global RLM_COMPRESSION_DURATION, RLM_QUERIES, RLM_QUERY_DURATION
    global RLM_CACHE_HITS, RLM_CACHE_MISSES, RLM_CONTEXT_LEVELS, RLM_MEMORY_USAGE

    if _initialized:
        return True

    try:
        from aragora.observability.config import get_metrics_config

        config = get_metrics_config()
        if not config.enabled:
            _init_noop_metrics()
            _initialized = True
            return False
    except ImportError:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram, Summary

        # Compression metrics
        RLM_COMPRESSIONS = Counter(
            "aragora_rlm_compressions_total",
            "Total RLM compression operations",
            ["source_type", "status"],
        )

        RLM_COMPRESSION_RATIO = Histogram(
            "aragora_rlm_compression_ratio",
            "Compression ratio (compressed/original tokens)",
            ["source_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

        RLM_TOKENS_SAVED = Counter(
            "aragora_rlm_tokens_saved_total",
            "Total tokens saved through compression",
            ["source_type"],
        )

        RLM_COMPRESSION_DURATION = Histogram(
            "aragora_rlm_compression_duration_seconds",
            "Time taken for compression operations",
            ["source_type", "levels"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # Query metrics
        RLM_QUERIES = Counter(
            "aragora_rlm_queries_total",
            "Total RLM context queries",
            ["query_type", "level"],
        )

        RLM_QUERY_DURATION = Histogram(
            "aragora_rlm_query_duration_seconds",
            "Time taken for context queries",
            ["query_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
        )

        # Cache metrics
        RLM_CACHE_HITS = Counter(
            "aragora_rlm_cache_hits_total",
            "RLM compression cache hits",
        )

        RLM_CACHE_MISSES = Counter(
            "aragora_rlm_cache_misses_total",
            "RLM compression cache misses",
        )

        # Context hierarchy metrics
        RLM_CONTEXT_LEVELS = Histogram(
            "aragora_rlm_context_levels",
            "Number of abstraction levels created",
            ["source_type"],
            buckets=[1, 2, 3, 4, 5],
        )

        RLM_MEMORY_USAGE = Gauge(
            "aragora_rlm_memory_bytes",
            "Memory used by RLM context cache",
        )

        _initialized = True
        logger.info("RLM Prometheus metrics initialized")
        return True

    except ImportError:
        logger.warning("prometheus_client not installed, RLM metrics disabled")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global RLM_COMPRESSIONS, RLM_COMPRESSION_RATIO, RLM_TOKENS_SAVED
    global RLM_COMPRESSION_DURATION, RLM_QUERIES, RLM_QUERY_DURATION
    global RLM_CACHE_HITS, RLM_CACHE_MISSES, RLM_CONTEXT_LEVELS, RLM_MEMORY_USAGE

    class NoopMetric:
        """No-op metric that accepts any method call."""

        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None

    noop = NoopMetric()
    RLM_COMPRESSIONS = noop
    RLM_COMPRESSION_RATIO = noop
    RLM_TOKENS_SAVED = noop
    RLM_COMPRESSION_DURATION = noop
    RLM_QUERIES = noop
    RLM_QUERY_DURATION = noop
    RLM_CACHE_HITS = noop
    RLM_CACHE_MISSES = noop
    RLM_CONTEXT_LEVELS = noop
    RLM_MEMORY_USAGE = noop


def record_compression(
    source_type: str,
    original_tokens: int,
    compressed_tokens: int,
    levels: int = 1,
    duration_seconds: float = 0.0,
    success: bool = True,
) -> None:
    """
    Record a compression operation.

    Args:
        source_type: Type of content compressed (debate, document, etc.)
        original_tokens: Token count before compression
        compressed_tokens: Token count after compression
        levels: Number of abstraction levels created
        duration_seconds: Time taken for compression
        success: Whether compression succeeded
    """
    _init_metrics()

    status = "success" if success else "failure"
    RLM_COMPRESSIONS.labels(source_type=source_type, status=status).inc()

    if success and original_tokens > 0:
        ratio = compressed_tokens / original_tokens
        RLM_COMPRESSION_RATIO.labels(source_type=source_type).observe(ratio)

        tokens_saved = original_tokens - compressed_tokens
        if tokens_saved > 0:
            RLM_TOKENS_SAVED.labels(source_type=source_type).inc(tokens_saved)

        RLM_CONTEXT_LEVELS.labels(source_type=source_type).observe(levels)

    if duration_seconds > 0:
        RLM_COMPRESSION_DURATION.labels(
            source_type=source_type,
            levels=str(levels),
        ).observe(duration_seconds)


def record_query(
    query_type: str,
    level: str = "SUMMARY",
    duration_seconds: float = 0.0,
) -> None:
    """
    Record a context query operation.

    Args:
        query_type: Type of query (drill_down, roll_up, search, etc.)
        level: Abstraction level queried (ABSTRACT, SUMMARY, DETAILED, FULL)
        duration_seconds: Time taken for query
    """
    _init_metrics()

    RLM_QUERIES.labels(query_type=query_type, level=level).inc()

    if duration_seconds > 0:
        RLM_QUERY_DURATION.labels(query_type=query_type).observe(duration_seconds)


def record_cache_hit() -> None:
    """Record a compression cache hit."""
    _init_metrics()
    RLM_CACHE_HITS.inc()


def record_cache_miss() -> None:
    """Record a compression cache miss."""
    _init_metrics()
    RLM_CACHE_MISSES.inc()


def set_memory_usage(bytes_used: int) -> None:
    """Set current memory usage for RLM context cache."""
    _init_metrics()
    RLM_MEMORY_USAGE.set(bytes_used)


@contextmanager
def measure_compression(
    source_type: str,
    original_tokens: int,
) -> Generator[dict, None, None]:
    """
    Context manager for measuring compression operations.

    Usage:
        with measure_compression("debate", 10000) as ctx:
            result = await compressor.compress(content)
            ctx["compressed_tokens"] = result.compressed_tokens
            ctx["levels"] = len(result.levels)

    Args:
        source_type: Type of content being compressed
        original_tokens: Token count before compression

    Yields:
        Dictionary to populate with compression results
    """
    start_time = time.perf_counter()
    context: dict = {
        "compressed_tokens": 0,
        "levels": 1,
        "success": True,
    }

    try:
        yield context
    except Exception:
        context["success"] = False
        raise
    finally:
        duration = time.perf_counter() - start_time
        record_compression(
            source_type=source_type,
            original_tokens=original_tokens,
            compressed_tokens=context.get("compressed_tokens", 0),
            levels=context.get("levels", 1),
            duration_seconds=duration,
            success=context.get("success", True),
        )


@contextmanager
def measure_query(query_type: str, level: str = "SUMMARY") -> Generator[None, None, None]:
    """
    Context manager for measuring query operations.

    Usage:
        with measure_query("drill_down", "DETAILED"):
            result = context.get_at_level(AbstractionLevel.DETAILED)

    Args:
        query_type: Type of query being performed
        level: Abstraction level being queried
    """
    start_time = time.perf_counter()

    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        record_query(
            query_type=query_type,
            level=level,
            duration_seconds=duration,
        )


# Convenience functions for common operations
def record_debate_compression(
    original_tokens: int,
    compressed_tokens: int,
    levels: int,
    duration_seconds: float,
) -> None:
    """Record compression of debate context."""
    record_compression(
        source_type="debate",
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        levels=levels,
        duration_seconds=duration_seconds,
    )


def record_document_compression(
    original_tokens: int,
    compressed_tokens: int,
    levels: int,
    duration_seconds: float,
) -> None:
    """Record compression of document content."""
    record_compression(
        source_type="document",
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        levels=levels,
        duration_seconds=duration_seconds,
    )


def record_knowledge_compression(
    original_tokens: int,
    compressed_tokens: int,
    levels: int,
    duration_seconds: float,
) -> None:
    """Record compression of knowledge base content."""
    record_compression(
        source_type="knowledge",
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        levels=levels,
        duration_seconds=duration_seconds,
    )
