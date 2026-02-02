"""
Core metrics initialization and coordination.

Provides centralized initialization for all metrics submodules
and manages the global initialization state.
"""

from __future__ import annotations

import logging
import re

from aragora.observability.config import get_metrics_config
from aragora.observability.metrics.base import get_metrics_enabled

logger = logging.getLogger(__name__)

# Global initialization state
_initialized = False

# Endpoint normalization regex
_ENDPOINT_PATTERN = re.compile(
    r"(/api/v\d+)?/(debates|agents|workspaces|tenants|users|knowledge|receipts|memory)"
    r"/[a-zA-Z0-9_-]+(?:/|$)"
)


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint paths to reduce cardinality.

    Args:
        endpoint: Raw endpoint path

    Returns:
        Normalized endpoint path with dynamic segments replaced
    """
    match = _ENDPOINT_PATTERN.match(endpoint)
    if match:
        prefix = match.group(1) or ""
        resource = match.group(2)
        return f"{prefix}/{resource}/:id"
    return endpoint


def init_core_metrics() -> bool:
    """Initialize all core metrics.

    This is the main entry point for initializing the metrics system.
    It initializes all submodules in the correct order.

    Returns:
        True if metrics were successfully initialized, False otherwise
    """
    global _initialized

    if _initialized:
        return get_metrics_enabled()

    config = get_metrics_config()
    if not config.enabled:
        logger.info("Metrics disabled by configuration")
        _initialized = True
        return False

    try:
        # Import and initialize all submodules
        from aragora.observability.metrics.agent import init_agent_metrics
        from aragora.observability.metrics.cache import init_cache_metrics
        from aragora.observability.metrics.checkpoint import init_checkpoint_metrics
        from aragora.observability.metrics.consensus import (
            init_consensus_metrics,
            init_enhanced_consensus_metrics,
        )
        from aragora.observability.metrics.convergence import init_convergence_metrics
        from aragora.observability.metrics.custom import init_custom_metrics
        from aragora.observability.metrics.debate import init_debate_metrics
        from aragora.observability.metrics.evidence import init_evidence_metrics
        from aragora.observability.metrics.explainability import init_explainability_metrics
        from aragora.observability.metrics.governance import init_governance_metrics
        from aragora.observability.metrics.km import init_km_metrics
        from aragora.observability.metrics.marketplace import init_marketplace_metrics
        from aragora.observability.metrics.memory import init_memory_metrics
        from aragora.observability.metrics.notification import init_notification_metrics
        from aragora.observability.metrics.ranking import init_ranking_metrics
        from aragora.observability.metrics.request import init_request_metrics
        from aragora.observability.metrics.security import init_security_metrics
        from aragora.observability.metrics.system import init_system_metrics
        from aragora.observability.metrics.task_queue import init_task_queue_metrics
        from aragora.observability.metrics.tts import init_tts_metrics
        from aragora.observability.metrics.user_mapping import init_user_mapping_metrics
        from aragora.observability.metrics.workflow import init_workflow_metrics

        # Initialize all submodules
        init_request_metrics()
        init_agent_metrics()
        init_debate_metrics()
        init_system_metrics()
        init_custom_metrics()

        # Cache and memory
        init_cache_metrics()
        init_memory_metrics()

        # Debate-related
        init_tts_metrics()
        init_convergence_metrics()
        init_consensus_metrics()
        init_enhanced_consensus_metrics()
        init_evidence_metrics()

        # Workflow and governance
        init_workflow_metrics()
        init_governance_metrics()
        init_checkpoint_metrics()

        # Knowledge and ranking
        init_km_metrics()
        init_ranking_metrics()

        # Infrastructure
        init_task_queue_metrics()
        init_notification_metrics()
        init_marketplace_metrics()
        init_explainability_metrics()
        init_user_mapping_metrics()
        init_security_metrics()

        _initialized = True
        logger.info("All Prometheus metrics initialized")
        return True

    except ImportError as e:
        logger.warning(
            "prometheus-client not installed, metrics disabled: %s. "
            "Install with: pip install prometheus-client",
            e,
        )
        _initialized = True
        return False
    except (TypeError, ValueError, RuntimeError) as e:
        # Configuration or initialization errors - recoverable
        logger.error(
            "Failed to initialize metrics due to configuration error",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        _initialized = True
        return False
    except OSError as e:
        # Resource errors (e.g., port in use, file access) - recoverable
        logger.error(
            "Failed to initialize metrics due to resource error",
            extra={"error_type": type(e).__name__, "error": str(e)},
        )
        _initialized = True
        return False


def is_initialized() -> bool:
    """Check if metrics have been initialized.

    Returns:
        True if metrics have been initialized
    """
    return _initialized


def reset_initialization() -> None:
    """Reset the initialization state.

    This is primarily useful for testing purposes.
    """
    global _initialized
    _initialized = False


__all__ = [
    "init_core_metrics",
    "is_initialized",
    "reset_initialization",
    "_normalize_endpoint",
]
