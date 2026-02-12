"""
Metrics initialization helpers.

Extracted from metrics/__init__.py for maintainability.
Provides internal helpers for initializing and refreshing metric state
across all submodules.
"""

from __future__ import annotations

import logging

from aragora.observability.metrics.base import (
    ensure_metrics_initialized,
    get_metrics_enabled,
)
from aragora.observability.metrics.core import init_core_metrics

logger = logging.getLogger(__name__)


def _metrics_modules():
    """Collect all metrics submodule references."""
    from aragora.observability.metrics import agent as agent_module
    from aragora.observability.metrics import agents as agents_module
    from aragora.observability.metrics import bridge as bridge_module
    from aragora.observability.metrics import cache as cache_module
    from aragora.observability.metrics import checkpoint as checkpoint_module
    from aragora.observability.metrics import consensus as consensus_module
    from aragora.observability.metrics import convergence as convergence_module
    from aragora.observability.metrics import custom as custom_module
    from aragora.observability.metrics import debate as debate_module
    from aragora.observability.metrics import evidence as evidence_module
    from aragora.observability.metrics import explainability as explainability_module
    from aragora.observability.metrics import fabric as fabric_module
    from aragora.observability.metrics import governance as governance_module
    from aragora.observability.metrics import km as km_module
    from aragora.observability.metrics import marketplace as marketplace_module
    from aragora.observability.metrics import memory as memory_module
    from aragora.observability.metrics import notification as notification_module
    from aragora.observability.metrics import ranking as ranking_module
    from aragora.observability.metrics import request as request_module
    from aragora.observability.metrics import security as security_module
    from aragora.observability.metrics import slo as slo_module
    from aragora.observability.metrics import system as system_module
    from aragora.observability.metrics import task_queue as task_queue_module
    from aragora.observability.metrics import tts as tts_module
    from aragora.observability.metrics import user_mapping as user_mapping_module
    from aragora.observability.metrics import webhook as webhook_module
    from aragora.observability.metrics import workflow as workflow_module

    return [
        request_module,
        agent_module,
        agents_module,
        debate_module,
        system_module,
        custom_module,
        cache_module,
        memory_module,
        convergence_module,
        tts_module,
        workflow_module,
        evidence_module,
        ranking_module,
        bridge_module,
        km_module,
        consensus_module,
        task_queue_module,
        governance_module,
        user_mapping_module,
        checkpoint_module,
        notification_module,
        marketplace_module,
        explainability_module,
        security_module,
        webhook_module,
        slo_module,
        fabric_module,
    ]


def _refresh_exports() -> None:
    """Refresh exported symbols to point at the latest module state."""
    import aragora.observability.metrics as _pkg

    export_names = set(_pkg.__all__)
    modules = _metrics_modules()
    for module in modules:
        for name in export_names:
            if hasattr(module, name):
                _pkg.__dict__[name] = getattr(module, name)


def _init_noop_metrics_all() -> None:
    """Initialize NoOp metrics across all submodules."""
    import aragora.observability.metrics as _pkg

    modules = _metrics_modules()
    for module in modules:
        init_noop = getattr(module, "_init_noop_metrics", None)
        if callable(init_noop):
            init_noop()
        if hasattr(module, "_initialized"):
            module._initialized = True
    _pkg._initialized = True
    _refresh_exports()


def _init_metrics_all() -> bool:
    """Initialize metrics with Prometheus if available."""
    import aragora.observability.metrics as _pkg

    if _pkg._initialized:
        return get_metrics_enabled()
    enabled = ensure_metrics_initialized()
    _pkg._initialized = True
    _refresh_exports()
    return enabled
