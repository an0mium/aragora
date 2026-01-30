"""
TTS (Text-to-Speech) synthesis metrics.

Provides Prometheus metrics for tracking TTS synthesis operations
including synthesis count, latency, and platform usage.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
TTS_SYNTHESIS_TOTAL: Any = None
TTS_SYNTHESIS_LATENCY: Any = None

_initialized = False


def init_tts_metrics() -> None:
    """Initialize TTS synthesis metrics."""
    global _initialized
    global TTS_SYNTHESIS_TOTAL, TTS_SYNTHESIS_LATENCY

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

        TTS_SYNTHESIS_TOTAL = Counter(
            "aragora_tts_synthesis_total",
            "Total TTS synthesis operations",
            ["voice", "platform"],
        )

        TTS_SYNTHESIS_LATENCY = Histogram(
            "aragora_tts_synthesis_latency_seconds",
            "TTS synthesis latency in seconds",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 20],
        )

        _initialized = True
        logger.debug("TTS metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize TTS metrics: {e}")
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global TTS_SYNTHESIS_TOTAL, TTS_SYNTHESIS_LATENCY

    TTS_SYNTHESIS_TOTAL = NoOpMetric()
    TTS_SYNTHESIS_LATENCY = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_tts_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_tts_synthesis(voice: str, platform: str = "unknown") -> None:
    """Record a TTS synthesis operation.

    Args:
        voice: Voice identifier used for synthesis
        platform: Platform where TTS was triggered (e.g., telegram, slack, web)
    """
    _ensure_init()
    TTS_SYNTHESIS_TOTAL.labels(voice=voice, platform=platform).inc()


def record_tts_latency(latency_seconds: float) -> None:
    """Record TTS synthesis latency.

    Args:
        latency_seconds: Time taken for TTS synthesis in seconds
    """
    _ensure_init()
    TTS_SYNTHESIS_LATENCY.observe(latency_seconds)


@contextmanager
def track_tts_synthesis(voice: str, platform: str = "unknown") -> Generator[None, None, None]:
    """Context manager to track TTS synthesis operations.

    Records both the synthesis count and latency automatically.

    Args:
        voice: Voice identifier used for synthesis
        platform: Platform where TTS was triggered

    Example:
        with track_tts_synthesis("en-US-Neural", "telegram"):
            audio = await synthesize_speech(text)
    """
    _ensure_init()
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_tts_synthesis(voice, platform)
        record_tts_latency(latency)


__all__ = [
    # Metrics
    "TTS_SYNTHESIS_TOTAL",
    "TTS_SYNTHESIS_LATENCY",
    # Functions
    "init_tts_metrics",
    "record_tts_synthesis",
    "record_tts_latency",
    "track_tts_synthesis",
]
