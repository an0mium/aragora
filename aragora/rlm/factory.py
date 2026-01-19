"""
RLM Factory - standardized initialization for AragoraRLM.

Provides consistent patterns for obtaining RLM instances across the codebase.
Hides HAS_OFFICIAL_RLM checks and provides clear logging.

The factory ensures all consumers get the correct RLM behavior:
- TRUE RLM (REPL-based) when official `rlm` package is installed
- Compression fallback when official package is not available

Usage:
    from aragora.rlm import get_rlm

    # Get singleton instance (recommended for most cases)
    rlm = get_rlm()
    result = await rlm.compress_and_query(query, content, source_type)

    # Or use convenience function
    from aragora.rlm import compress_and_query
    result = await compress_and_query(query, content, source_type)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .bridge import AragoraRLM
    from .compressor import HierarchicalCompressor
    from .types import RLMConfig, RLMResult

logger = logging.getLogger(__name__)

# Singleton instance for reuse
_rlm_instance: Optional["AragoraRLM"] = None


def get_rlm(
    config: Optional["RLMConfig"] = None,
    force_new: bool = False,
) -> "AragoraRLM":
    """
    Get AragoraRLM instance (routes to TRUE RLM when available).

    This is the preferred way to obtain RLM functionality across the codebase.
    It will use TRUE RLM (REPL-based) when official library is installed,
    otherwise falls back to compression-based approach.

    Args:
        config: Optional RLMConfig for customization
        force_new: If True, create new instance instead of reusing singleton

    Returns:
        AragoraRLM instance

    Example:
        >>> from aragora.rlm import get_rlm
        >>> rlm = get_rlm()
        >>> result = await rlm.compress_and_query(
        ...     query="What is the main topic?",
        ...     content=long_document,
        ...     source_type="document",
        ... )
        >>> if result.used_true_rlm:
        ...     print("Used TRUE RLM (REPL-based)")
        >>> elif result.used_compression_fallback:
        ...     print("Used compression fallback")
    """
    global _rlm_instance

    # Return cached instance if available and appropriate
    if _rlm_instance is not None and not force_new and config is None:
        return _rlm_instance

    from .bridge import AragoraRLM, HAS_OFFICIAL_RLM

    rlm = AragoraRLM(aragora_config=config)

    if HAS_OFFICIAL_RLM:
        logger.info(
            "[RLM Factory] Created AragoraRLM with TRUE RLM support "
            "(REPL-based, model writes code to examine context)"
        )
    else:
        logger.info(
            "[RLM Factory] Created AragoraRLM with compression fallback "
            "(official RLM not installed - pip install rlm for TRUE RLM)"
        )

    # Cache if using default config
    if config is None and not force_new:
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

    logger.debug("[RLM Factory] Creating direct HierarchicalCompressor (compression-only)")
    return HierarchicalCompressor(config=config)


async def compress_and_query(
    query: str,
    content: str,
    source_type: str = "general",
    config: Optional["RLMConfig"] = None,
) -> "RLMResult":
    """
    Convenience function for common compress+query pattern.

    Uses TRUE RLM when available, compression fallback otherwise.
    This is a shortcut for:
        rlm = get_rlm()
        result = await rlm.compress_and_query(...)

    Args:
        query: The question to answer about the content
        content: The content to analyze
        source_type: Type of content (debate, document, email, etc.)
        config: Optional RLMConfig for customization

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
    """
    rlm = get_rlm(config=config)
    return await rlm.compress_and_query(
        query=query,
        content=content,
        source_type=source_type,
    )


def reset_singleton() -> None:
    """
    Reset the singleton RLM instance.

    Useful for testing or when configuration needs to change.
    """
    global _rlm_instance
    _rlm_instance = None
    logger.debug("[RLM Factory] Singleton instance reset")


__all__ = [
    "get_rlm",
    "get_compressor",
    "compress_and_query",
    "reset_singleton",
]
