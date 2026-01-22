"""
RLM Factory - standardized initialization for AragoraRLM.

Provides consistent patterns for obtaining RLM instances across the codebase.
Hides HAS_OFFICIAL_RLM checks and provides clear logging.

The factory ensures all consumers get the correct RLM behavior:
- TRUE RLM (REPL-based) when official `rlm` package is installed (PREFERRED)
- Compression fallback when official package is not available

Phase 12 TRUE RLM Prioritization:
- AUTO mode (default): Prefer TRUE RLM, gracefully fall back to compression
- TRUE_RLM mode: Require TRUE RLM, raise error if not available
- COMPRESSION mode: Force compression (for testing/specific use cases)

Usage:
    from aragora.rlm import get_rlm, RLMMode

    # Get singleton instance - prefers TRUE RLM (recommended)
    rlm = get_rlm()
    result = await rlm.compress_and_query(query, content, source_type)

    # Require TRUE RLM (raises if not available)
    rlm = get_rlm(mode=RLMMode.TRUE_RLM)

    # Or use convenience function
    from aragora.rlm import compress_and_query
    result = await compress_and_query(query, content, source_type)

    # Access metrics
    from aragora.rlm import get_factory_metrics
    metrics = get_factory_metrics()
    print(f"TRUE RLM calls: {metrics['true_rlm_calls']}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

from .types import RLMMode

if TYPE_CHECKING:
    from .bridge import AragoraRLM
    from .compressor import HierarchicalCompressor
    from .types import RLMConfig, RLMResult

logger = logging.getLogger(__name__)


@dataclass
class RLMFactoryMetrics:
    """Metrics for RLM factory observability.

    Tracks usage patterns to help understand which RLM approach is being used.
    """

    # Factory call counts
    get_rlm_calls: int = 0
    get_compressor_calls: int = 0
    compress_and_query_calls: int = 0

    # Instance creation counts
    rlm_instances_created: int = 0
    compressor_instances_created: int = 0

    # RLM type tracking (from results)
    true_rlm_calls: int = 0
    compression_fallback_calls: int = 0

    # Success/failure tracking
    successful_queries: int = 0
    failed_queries: int = 0

    # Singleton behavior tracking
    singleton_hits: int = 0
    singleton_misses: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert metrics to dictionary for logging/export."""
        return {
            "get_rlm_calls": self.get_rlm_calls,
            "get_compressor_calls": self.get_compressor_calls,
            "compress_and_query_calls": self.compress_and_query_calls,
            "rlm_instances_created": self.rlm_instances_created,
            "compressor_instances_created": self.compressor_instances_created,
            "true_rlm_calls": self.true_rlm_calls,
            "compression_fallback_calls": self.compression_fallback_calls,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "singleton_hits": self.singleton_hits,
            "singleton_misses": self.singleton_misses,
        }

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.get_rlm_calls = 0
        self.get_compressor_calls = 0
        self.compress_and_query_calls = 0
        self.rlm_instances_created = 0
        self.compressor_instances_created = 0
        self.true_rlm_calls = 0
        self.compression_fallback_calls = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.singleton_hits = 0
        self.singleton_misses = 0


# Global metrics instance
_metrics = RLMFactoryMetrics()

# Singleton instance for reuse
_rlm_instance: Optional["AragoraRLM"] = None


def get_rlm(
    config: Optional["RLMConfig"] = None,
    force_new: bool = False,
    mode: Optional[RLMMode] = None,
    require_true_rlm: bool = False,
) -> "AragoraRLM":
    """
    Get AragoraRLM instance - PREFERS TRUE RLM when available.

    This is the preferred way to obtain RLM functionality across the codebase.
    It will use TRUE RLM (REPL-based) when official library is installed,
    otherwise falls back to compression-based approach.

    Phase 12 TRUE RLM Prioritization:
    - By default (AUTO mode), prefers TRUE RLM over compression
    - Logs warning when falling back to compression
    - Can require TRUE RLM and raise error if not available

    Args:
        config: Optional RLMConfig for customization
        force_new: If True, create new instance instead of reusing singleton
        mode: RLMMode to use (defaults to AUTO which prefers TRUE RLM)
        require_true_rlm: If True, raise RuntimeError if TRUE RLM not available

    Returns:
        AragoraRLM instance

    Raises:
        RuntimeError: If require_true_rlm=True and TRUE RLM not available

    Example:
        >>> from aragora.rlm import get_rlm, RLMMode
        >>> rlm = get_rlm()  # AUTO mode - prefers TRUE RLM
        >>> result = await rlm.compress_and_query(
        ...     query="What is the main topic?",
        ...     content=long_document,
        ...     source_type="document",
        ... )
        >>> if result.used_true_rlm:
        ...     print("Used TRUE RLM (REPL-based)")
        >>> elif result.used_compression_fallback:
        ...     print("Used compression fallback")
        >>>
        >>> # Require TRUE RLM (raises if not available)
        >>> rlm_strict = get_rlm(mode=RLMMode.TRUE_RLM)
    """
    global _rlm_instance
    _metrics.get_rlm_calls += 1

    from .bridge import AragoraRLM, HAS_OFFICIAL_RLM

    # Determine effective mode from config, parameter, or environment
    effective_mode = mode
    if effective_mode is None and config is not None:
        effective_mode = config.mode
    if effective_mode is None:
        # Check environment variable
        env_mode = os.environ.get("ARAGORA_RLM_MODE", "").lower()
        if env_mode == "true_rlm":
            effective_mode = RLMMode.TRUE_RLM
        elif env_mode == "compression":
            effective_mode = RLMMode.COMPRESSION
        else:
            effective_mode = RLMMode.AUTO  # Default: prefer TRUE RLM

    # Check require_true_rlm from config or parameter
    effective_require = require_true_rlm
    if not effective_require and config is not None:
        effective_require = config.require_true_rlm
    if not effective_require:
        effective_require = os.environ.get("ARAGORA_RLM_REQUIRE_TRUE", "").lower() == "true"

    # Validate TRUE_RLM mode requirements
    if effective_mode == RLMMode.TRUE_RLM or effective_require:
        if not HAS_OFFICIAL_RLM:
            error_msg = (
                "TRUE RLM required but official RLM library not installed. "
                "Install with: pip install aragora[rlm] or pip install rlm"
            )
            logger.error(f"[RLM Factory] {error_msg}")
            raise RuntimeError(error_msg)

    # Return cached instance if available and appropriate
    if _rlm_instance is not None and not force_new and config is None and mode is None:
        _metrics.singleton_hits += 1
        return _rlm_instance

    _metrics.singleton_misses += 1

    rlm = AragoraRLM(aragora_config=config)
    _metrics.rlm_instances_created += 1

    # Determine warn_on_fallback setting
    warn_on_fallback = True
    if config is not None:
        warn_on_fallback = config.warn_on_compression_fallback
    if os.environ.get("ARAGORA_RLM_WARN_FALLBACK", "").lower() == "false":
        warn_on_fallback = False

    if HAS_OFFICIAL_RLM:
        logger.info(
            "[RLM Factory] Created AragoraRLM with TRUE RLM support "
            "(REPL-based, model writes code to examine context - PREFERRED)"
        )
    else:
        # Compression fallback - warn if configured
        if warn_on_fallback and effective_mode in (RLMMode.AUTO, RLMMode.TRUE_RLM):
            logger.warning(
                "[RLM Factory] TRUE RLM not available, using compression fallback. "
                "For better performance, install: pip install aragora[rlm]"
            )
        else:
            logger.info(
                "[RLM Factory] Created AragoraRLM with compression fallback "
                "(official RLM not installed - pip install aragora[rlm] for TRUE RLM)"
            )

    # Cache if using default config and no specific mode
    if config is None and not force_new and mode is None:
        _rlm_instance = rlm

    return rlm


def get_compressor(config: Optional["RLMConfig"] = None) -> "HierarchicalCompressor":
    """
    Get HierarchicalCompressor directly (for specific compression-only use cases).

    Prefer get_rlm() in most cases. Use this only when you specifically
    need compression without the TRUE RLM option, such as:
    - Pre-processing before other operations
    - Testing compression behavior specifically
    - Legacy code that requires compressor API

    Args:
        config: Optional RLMConfig for customization

    Returns:
        HierarchicalCompressor instance
    """
    from .compressor import HierarchicalCompressor

    _metrics.get_compressor_calls += 1
    _metrics.compressor_instances_created += 1
    logger.debug("[RLM Factory] Creating direct HierarchicalCompressor (compression-only)")
    return HierarchicalCompressor(config=config)


async def compress_and_query(
    query: str,
    content: str,
    source_type: str = "general",
    config: Optional["RLMConfig"] = None,
    mode: Optional[RLMMode] = None,
    require_true_rlm: bool = False,
) -> "RLMResult":
    """
    Convenience function for common compress+query pattern.

    PREFERS TRUE RLM when available, compression fallback otherwise.
    This is a shortcut for:
        rlm = get_rlm()
        result = await rlm.compress_and_query(...)

    Args:
        query: The question to answer about the content
        content: The content to analyze
        source_type: Type of content (debate, document, email, etc.)
        config: Optional RLMConfig for customization
        mode: RLMMode to use (defaults to AUTO which prefers TRUE RLM)
        require_true_rlm: If True, raise RuntimeError if TRUE RLM not available

    Returns:
        RLMResult with answer and tracking flags (used_true_rlm, used_compression_fallback)

    Example:
        >>> from aragora.rlm import compress_and_query
        >>> result = await compress_and_query(
        ...     query="Summarize the key points",
        ...     content=long_document,
        ...     source_type="document",
        ... )
        >>> print(result.answer)
        >>> if result.used_true_rlm:
        ...     print("Used TRUE RLM (preferred)")
    """
    _metrics.compress_and_query_calls += 1

    rlm = get_rlm(config=config, mode=mode, require_true_rlm=require_true_rlm)
    try:
        result = await rlm.compress_and_query(
            query=query,
            content=content,
            source_type=source_type,
        )
        _metrics.successful_queries += 1

        # Track which approach was used
        if result.used_true_rlm:
            _metrics.true_rlm_calls += 1
        elif result.used_compression_fallback:
            _metrics.compression_fallback_calls += 1

        return result
    except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as e:
        _metrics.failed_queries += 1
        logger.warning(f"RLM compress_and_query failed with expected error: {e}")
        raise
    except Exception as e:
        _metrics.failed_queries += 1
        logger.exception(f"RLM compress_and_query failed with unexpected error: {e}")
        raise


def reset_singleton() -> None:
    """
    Reset the singleton RLM instance.

    Useful for testing or when configuration needs to change.
    """
    global _rlm_instance
    _rlm_instance = None
    logger.debug("[RLM Factory] Singleton instance reset")


def get_factory_metrics() -> Dict[str, int]:
    """
    Get current factory metrics.

    Returns a dictionary with metrics tracking RLM usage patterns:
    - get_rlm_calls: Number of get_rlm() calls
    - get_compressor_calls: Number of get_compressor() calls
    - compress_and_query_calls: Number of compress_and_query() calls
    - rlm_instances_created: Total RLM instances created
    - compressor_instances_created: Total compressor instances created
    - true_rlm_calls: Queries that used TRUE RLM (REPL-based)
    - compression_fallback_calls: Queries that used compression fallback
    - successful_queries: Successful query count
    - failed_queries: Failed query count
    - singleton_hits: Times singleton was reused
    - singleton_misses: Times new instance was created

    Example:
        >>> from aragora.rlm import get_factory_metrics
        >>> metrics = get_factory_metrics()
        >>> print(f"TRUE RLM usage: {metrics['true_rlm_calls']}")
        >>> print(f"Compression fallback: {metrics['compression_fallback_calls']}")
    """
    return _metrics.to_dict()


def reset_metrics() -> None:
    """
    Reset all factory metrics to zero.

    Useful for testing or starting fresh metric collection.
    """
    _metrics.reset()
    logger.debug("[RLM Factory] Metrics reset")


def log_metrics_summary() -> None:
    """
    Log a summary of factory metrics.

    Useful for debugging or periodic monitoring.
    """
    metrics = _metrics.to_dict()
    logger.info(
        "[RLM Factory Metrics Summary]\n"
        f"  API Calls: get_rlm={metrics['get_rlm_calls']}, "
        f"get_compressor={metrics['get_compressor_calls']}, "
        f"compress_and_query={metrics['compress_and_query_calls']}\n"
        f"  Instances: rlm={metrics['rlm_instances_created']}, "
        f"compressor={metrics['compressor_instances_created']}\n"
        f"  RLM Type: true_rlm={metrics['true_rlm_calls']}, "
        f"compression_fallback={metrics['compression_fallback_calls']}\n"
        f"  Results: success={metrics['successful_queries']}, "
        f"failed={metrics['failed_queries']}\n"
        f"  Singleton: hits={metrics['singleton_hits']}, "
        f"misses={metrics['singleton_misses']}"
    )


def require_true_rlm_decorator():
    """
    Decorator factory that enforces TRUE RLM usage at function level.

    Apply this to async functions that should only use TRUE RLM (not compression).
    Raises RuntimeError if TRUE RLM is not available when the function is called.

    Example:
        >>> from aragora.rlm import require_true_rlm_decorator
        >>>
        >>> @require_true_rlm_decorator()
        ... async def analyze_debate(debate_result):
        ...     rlm = get_rlm()
        ...     return await rlm.compress_and_query(
        ...         "Summarize the debate",
        ...         str(debate_result),
        ...         "debate",
        ...     )
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            from .bridge import HAS_OFFICIAL_RLM

            if not HAS_OFFICIAL_RLM:
                raise RuntimeError(
                    f"Function {func.__name__} requires TRUE RLM but official library "
                    "is not installed. Install with: pip install aragora[rlm]"
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def is_true_rlm_available() -> bool:
    """
    Check if TRUE RLM (official library) is available.

    Use this to conditionally enable features that require TRUE RLM.

    Example:
        >>> from aragora.rlm import is_true_rlm_available
        >>> if is_true_rlm_available():
        ...     result = await rlm.query(q, ctx)  # TRUE RLM
        ... else:
        ...     result = await fallback_analysis(q, ctx)  # Alternative
    """
    from .bridge import HAS_OFFICIAL_RLM

    return HAS_OFFICIAL_RLM


def get_rlm_mode_info() -> dict:
    """
    Get information about the current RLM mode and availability.

    Returns a dictionary with:
    - true_rlm_available: Whether official RLM library is installed
    - effective_mode: Current mode (AUTO, TRUE_RLM, or COMPRESSION)
    - env_mode: Mode from environment variable (if set)
    - env_require: Whether ARAGORA_RLM_REQUIRE_TRUE is set

    Example:
        >>> from aragora.rlm import get_rlm_mode_info
        >>> info = get_rlm_mode_info()
        >>> if info['true_rlm_available']:
        ...     print("TRUE RLM is available!")
        >>> print(f"Effective mode: {info['effective_mode']}")
    """
    from .bridge import HAS_OFFICIAL_RLM

    env_mode = os.environ.get("ARAGORA_RLM_MODE", "").lower()
    env_require = os.environ.get("ARAGORA_RLM_REQUIRE_TRUE", "").lower() == "true"

    # Determine effective mode
    if env_mode == "true_rlm":
        effective = RLMMode.TRUE_RLM
    elif env_mode == "compression":
        effective = RLMMode.COMPRESSION
    else:
        effective = RLMMode.AUTO

    return {
        "true_rlm_available": HAS_OFFICIAL_RLM,
        "effective_mode": effective.value,
        "env_mode": env_mode or None,
        "env_require": env_require,
        "warning": (
            "TRUE RLM not available, using compression fallback"
            if not HAS_OFFICIAL_RLM and effective != RLMMode.COMPRESSION
            else None
        ),
    }


__all__ = [
    "get_rlm",
    "get_compressor",
    "compress_and_query",
    "reset_singleton",
    "get_factory_metrics",
    "reset_metrics",
    "log_metrics_summary",
    "RLMFactoryMetrics",
    "RLMMode",
    "require_true_rlm_decorator",
    "is_true_rlm_available",
    "get_rlm_mode_info",
]
