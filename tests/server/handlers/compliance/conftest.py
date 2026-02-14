"""
Pytest configuration for compliance handler tests.

Provides fixtures and configuration specific to compliance handler testing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def reset_prometheus_registry():
    """Reset Prometheus registry to avoid metric collision errors.

    The compliance handler methods use @track_handler which registers
    Prometheus metrics. When running multiple test files, the same
    metrics get registered multiple times causing ValueError.

    This fixture clears the registry before each test and resets
    the metrics module initialization state.
    """
    try:
        from prometheus_client import REGISTRY
    except ImportError:
        yield
        return

    # Collect metric names to unregister
    collectors_to_remove = []
    for collector in list(REGISTRY._collector_to_names.keys()):
        # Only remove aragora metrics
        names = REGISTRY._collector_to_names.get(collector, [])
        if any(name.startswith("aragora_") for name in names):
            collectors_to_remove.append(collector)

    # Unregister aragora collectors
    for collector in collectors_to_remove:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass  # Ignore if already unregistered

    # Reset metrics module initialization flags
    import aragora.observability.metrics.request as request_metrics
    import aragora.observability.metrics.custom as custom_metrics

    request_metrics._initialized = False
    custom_metrics._initialized = False

    yield

    # Clean up after test
    collectors_to_remove = []
    for collector in list(REGISTRY._collector_to_names.keys()):
        names = REGISTRY._collector_to_names.get(collector, [])
        if any(name.startswith("aragora_") for name in names):
            collectors_to_remove.append(collector)

    for collector in collectors_to_remove:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
