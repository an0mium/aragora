"""
Metrics package for Aragora observability.

This package provides Prometheus metrics for monitoring request rates, latencies,
agent performance, and debate statistics.

For backward compatibility, all metrics and functions are re-exported from
the parent metrics module (aragora.observability.metrics).
"""

# Import the parent metrics module using importlib to avoid circular imports
import importlib.util
import os
import sys

# Get the path to the parent metrics.py file
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_metrics_path = os.path.join(_parent_dir, "metrics.py")

# Load the parent metrics module
_spec = importlib.util.spec_from_file_location("_aragora_metrics_impl", _metrics_path)
_metrics_module = importlib.util.module_from_spec(_spec)
sys.modules["_aragora_metrics_impl"] = _metrics_module
_spec.loader.exec_module(_metrics_module)

# Re-export everything from the parent metrics module
from _aragora_metrics_impl import *  # noqa: F401, F403, E402

# Explicitly re-export private functions used by tests (not included in * import)
from _aragora_metrics_impl import _init_metrics, _init_noop_metrics  # noqa: F401, E402

# Also import from submodules for explicit access
from aragora.observability.metrics.base import (  # noqa: F401, E402
    NoOpMetric,
    get_metrics_enabled,
    ensure_metrics_initialized,
)

from aragora.observability.metrics.bridge import (  # noqa: F401, E402
    init_bridge_metrics,
    record_bridge_sync,
    record_bridge_sync_latency,
    record_bridge_error,
)

from aragora.observability.metrics.km import (  # noqa: F401, E402
    init_km_metrics,
)

from aragora.observability.metrics.notification import (  # noqa: F401, E402
    init_notification_metrics,
    record_notification_sent,
    record_notification_error,
    set_notification_queue_size,
)

from aragora.observability.metrics.slo import (  # noqa: F401, E402
    init_slo_metrics,
    record_slo_check,
    record_slo_violation,
    record_operation_latency,
    check_and_record_slo,
    track_operation_slo,
    get_slo_metrics_summary,
)

from aragora.observability.metrics.webhook import (  # noqa: F401, E402
    record_webhook_delivery,
    record_webhook_retry,
    set_queue_size as set_webhook_queue_size,
    set_active_endpoints as set_webhook_active_endpoints,
    WebhookDeliveryTimer,
)

# Explicit re-exports for mypy compatibility (dynamic imports aren't tracked)
from _aragora_metrics_impl import (  # noqa: F401, E402
    start_metrics_server,
    record_rbac_check,
    record_tts_synthesis,
    record_tts_latency,
)
