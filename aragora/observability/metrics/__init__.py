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

from aragora.observability.metrics.debate import (  # noqa: F401, E402
    init_debate_metrics,
    record_debate_completion as record_debate_completion_v2,
    record_phase_duration as record_phase_duration_v2,
    record_agent_participation as record_agent_participation_v2,
    record_slow_debate,
    record_slow_round as record_slow_round_v2,
    record_round_latency as record_round_latency_v2,
    set_active_debates,
    increment_active_debates,
    decrement_active_debates,
    set_consensus_rate as set_consensus_rate_v2,
    track_debate as track_debate_v2,
    track_phase as track_phase_v2,
)

from aragora.observability.metrics.request import (  # noqa: F401, E402
    init_request_metrics,
    record_request as record_request_v2,
    record_latency,
    measure_latency as measure_latency_v2,
    measure_async_latency as measure_async_latency_v2,
)

from aragora.observability.metrics.agent import (  # noqa: F401, E402
    init_agent_metrics,
    record_agent_call as record_agent_call_v2,
    record_agent_latency,
    record_agent_error,
    record_token_usage,
    track_agent_call,
)

# Explicit re-exports for mypy compatibility (dynamic imports aren't tracked)
from _aragora_metrics_impl import (  # noqa: F401, E402
    # Server/startup
    start_metrics_server,
    # Core recording functions
    measure_async_latency,
    measure_latency,
    record_request,
    record_agent_call,
    record_agent_participation,
    record_memory_operation,
    record_cache_hit,
    record_cache_miss,
    set_consensus_rate,
    # Debate metrics
    record_debate_completion,
    record_phase_duration,
    record_round_latency,
    record_slow_round,
    record_adaptive_round_change,
    record_calibration_adjustment,
    record_evidence_citation_bonus,
    record_process_evaluation_bonus,
    # Checkpoint metrics
    record_checkpoint_operation,
    track_checkpoint_operation,
    record_checkpoint_restore_result,
    # Ranking/ELO metrics
    record_learning_bonus,
    record_voting_accuracy_update,
    # RBAC/Security metrics
    record_rbac_check,
    # TTS metrics
    record_tts_synthesis,
    record_tts_latency,
    # Migration metrics
    record_migration_record,
    record_migration_error,
    # User mapping metrics
    record_user_mapping_operation,
    record_user_mapping_cache_hit,
    record_user_mapping_cache_miss,
    # Governance metrics
    record_governance_verification,
    record_governance_decision,
    record_governance_approval,
    # RLM metrics
    record_rlm_cache_hit,
    record_rlm_cache_miss,
    # Knowledge Mound metrics
    record_km_event_emitted,
    record_km_operation,
    record_km_cache_access,
    record_km_adapter_sync,
    # Task queue metrics
    record_task_queue_operation,
    record_task_queue_recovery,
    record_task_queue_cleanup,
    # Context managers
    track_debate,
    track_phase,
    track_websocket_connection,
)
