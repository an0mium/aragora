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
from _aragora_metrics_impl import _init_metrics as _impl_init_metrics  # noqa: F401, E402
from _aragora_metrics_impl import _init_noop_metrics as _impl_init_noop_metrics  # noqa: F401, E402
from _aragora_metrics_impl import _normalize_endpoint as _impl_normalize_endpoint  # noqa: F401, E402

# Keep package-level attributes in sync with the implementation module.
_SYNC_NAMES = set(getattr(_metrics_module, "__all__", []))
_SYNC_NAMES.update({"_initialized", "_metrics_server", "_normalize_endpoint"})


def _sync_public_attrs() -> None:
    for name in _SYNC_NAMES:
        if hasattr(_metrics_module, name):
            globals()[name] = getattr(_metrics_module, name)


def _init_metrics() -> bool:
    result = _impl_init_metrics()
    _sync_public_attrs()
    return result


def _init_noop_metrics() -> None:
    _impl_init_noop_metrics()
    _sync_public_attrs()


def init_core_metrics() -> bool:
    result = _metrics_module.init_core_metrics()
    _sync_public_attrs()
    return result


_normalize_endpoint = _impl_normalize_endpoint

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

from aragora.observability.metrics.marketplace import (  # noqa: F401, E402
    init_marketplace_metrics,
    set_marketplace_templates_count,
    record_marketplace_download,
    record_marketplace_rating,
    record_marketplace_review,
    record_marketplace_operation_latency,
    track_marketplace_operation,
)

from aragora.observability.metrics.explainability import (  # noqa: F401, E402
    init_explainability_metrics,
    set_batch_explainability_jobs_active,
    record_batch_explainability_job,
    record_batch_explainability_debate,
    record_batch_explainability_error,
    track_batch_explainability_debate,
)

from aragora.observability.metrics.fabric import (  # noqa: F401, E402
    init_fabric_metrics,
    set_agents_active as set_fabric_agents_active,
    set_agents_health as set_fabric_agents_health,
    record_agent_spawned as record_fabric_agent_spawned,
    record_agent_terminated as record_fabric_agent_terminated,
    record_task_queued as record_fabric_task_queued,
    record_task_completed as record_fabric_task_completed,
    record_task_cancelled as record_fabric_task_cancelled,
    set_task_queue_depth as set_fabric_task_queue_depth,
    record_policy_decision as record_fabric_policy_decision,
    set_pending_approvals as set_fabric_pending_approvals,
    set_budget_usage as set_fabric_budget_usage,
    record_budget_alert as record_fabric_budget_alert,
    record_fabric_stats,
    track_fabric_task,
)

from aragora.observability.metrics.task_queue import (  # noqa: F401, E402
    init_task_queue_metrics,
    record_task_queue_operation as record_task_queue_operation_v2,
    set_task_queue_size,
    record_task_queue_recovery as record_task_queue_recovery_v2,
    record_task_queue_cleanup as record_task_queue_cleanup_v2,
    track_task_queue_operation,
)

from aragora.observability.metrics.governance import (  # noqa: F401, E402
    init_governance_metrics,
    record_governance_decision as record_governance_decision_v2,
    record_governance_verification as record_governance_verification_v2,
    record_governance_approval as record_governance_approval_v2,
    record_governance_store_latency,
    set_governance_artifacts_active,
    track_governance_store_operation,
)

from aragora.observability.metrics.user_mapping import (  # noqa: F401, E402
    init_user_mapping_metrics,
    record_user_mapping_operation as record_user_mapping_operation_v2,
    record_user_mapping_cache_hit as record_user_mapping_cache_hit_v2,
    record_user_mapping_cache_miss as record_user_mapping_cache_miss_v2,
    set_user_mappings_active,
)

from aragora.observability.metrics.checkpoint import (  # noqa: F401, E402
    init_checkpoint_metrics,
    record_checkpoint_operation as record_checkpoint_operation_v2,
    record_checkpoint_restore_result as record_checkpoint_restore_result_v2,
    track_checkpoint_operation as track_checkpoint_operation_v2,
)

from aragora.observability.metrics.consensus import (  # noqa: F401, E402
    init_consensus_metrics,
    init_enhanced_consensus_metrics,
    record_consensus_ingestion,
    record_consensus_dissent,
    record_consensus_evolution,
    record_consensus_evidence_linked,
    record_consensus_agreement_ratio,
)

# TTS metrics (new Phase 1 submodule)
from aragora.observability.metrics.tts import (  # noqa: F401, E402
    init_tts_metrics,
    record_tts_synthesis as record_tts_synthesis_v2,
    record_tts_latency as record_tts_latency_v2,
    track_tts_synthesis,
    TTS_SYNTHESIS_TOTAL,
    TTS_SYNTHESIS_LATENCY,
)

# Cache metrics (new Phase 1 submodule)
from aragora.observability.metrics.cache import (  # noqa: F401, E402
    init_cache_metrics,
    record_cache_hit as record_cache_hit_v2,
    record_cache_miss as record_cache_miss_v2,
    record_knowledge_cache_hit as record_knowledge_cache_hit_v2,
    record_knowledge_cache_miss as record_knowledge_cache_miss_v2,
    record_rlm_cache_hit as record_rlm_cache_hit_v2,
    record_rlm_cache_miss as record_rlm_cache_miss_v2,
    CACHE_HITS,
    CACHE_MISSES,
    KNOWLEDGE_CACHE_HITS,
    KNOWLEDGE_CACHE_MISSES,
    RLM_CACHE_HITS,
    RLM_CACHE_MISSES,
)

# Convergence metrics (new Phase 1 submodule)
from aragora.observability.metrics.convergence import (  # noqa: F401, E402
    init_convergence_metrics,
    record_convergence_check as record_convergence_check_v2,
    record_process_evaluation_bonus as record_process_evaluation_bonus_v2,
    record_rlm_ready_quorum as record_rlm_ready_quorum_v2,
    CONVERGENCE_CHECKS_TOTAL,
    PROCESS_EVALUATION_BONUSES,
    RLM_READY_QUORUM_EVENTS,
)

# Workflow metrics (new Phase 2 submodule)
from aragora.observability.metrics.workflow import (  # noqa: F401, E402
    init_workflow_metrics,
    record_workflow_trigger as record_workflow_trigger_v2,
    record_workflow_template_created as record_workflow_template_created_v2,
    record_workflow_template_execution as record_workflow_template_execution_v2,
    track_workflow_template_execution as track_workflow_template_execution_v2,
    WORKFLOW_TRIGGERS,
    WORKFLOW_TEMPLATES_CREATED,
    WORKFLOW_TEMPLATE_EXECUTIONS,
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY,
)

# Memory metrics (new Phase 2 submodule)
from aragora.observability.metrics.memory import (  # noqa: F401, E402
    init_memory_metrics,
    record_memory_operation as record_memory_operation_v2,
    record_memory_coordinator_write as record_memory_coordinator_write_v2,
    record_adaptive_round_change as record_adaptive_round_change_v2,
    MEMORY_OPERATIONS,
    MEMORY_COORDINATOR_WRITES,
    ADAPTIVE_ROUND_CHANGES,
)

# Evidence metrics (new Phase 2 submodule)
from aragora.observability.metrics.evidence import (  # noqa: F401, E402
    init_evidence_metrics,
    record_evidence_stored as record_evidence_stored_v2,
    record_evidence_citation_bonus as record_evidence_citation_bonus_v2,
    record_culture_patterns as record_culture_patterns_v2,
    EVIDENCE_STORED,
    EVIDENCE_CITATION_BONUSES,
    CULTURE_PATTERNS,
)

# Ranking metrics (new Phase 3 submodule)
from aragora.observability.metrics.ranking import (  # noqa: F401, E402
    init_ranking_metrics,
    record_calibration_adjustment as record_calibration_adjustment_v2,
    record_learning_bonus as record_learning_bonus_v2,
    record_voting_accuracy_update as record_voting_accuracy_update_v2,
    record_selection_feedback_adjustment as record_selection_feedback_adjustment_v2,
    record_performance_routing_decision,
    record_performance_routing_latency,
    track_performance_routing,
    record_novelty_score_calculation,
    record_novelty_penalty,
    record_echo_chamber_detection,
    record_relationship_bias_adjustment,
    record_rlm_selection_recommendation,
    record_calibration_cost_calculation,
    record_budget_filtering_event,
    record_outcome_complexity_adjustment,
    record_analytics_selection_recommendation,
    CALIBRATION_ADJUSTMENTS,
    LEARNING_BONUSES,
    VOTING_ACCURACY_UPDATES,
    SELECTION_FEEDBACK_ADJUSTMENTS,
    PERFORMANCE_ROUTING_DECISIONS,
    PERFORMANCE_ROUTING_LATENCY,
    NOVELTY_SCORE_CALCULATIONS,
    NOVELTY_PENALTIES,
    ECHO_CHAMBER_DETECTIONS,
    RELATIONSHIP_BIAS_ADJUSTMENTS,
    RLM_SELECTION_RECOMMENDATIONS,
    CALIBRATION_COST_CALCULATIONS,
    BUDGET_FILTERING_EVENTS,
    OUTCOME_COMPLEXITY_ADJUSTMENTS,
    ANALYTICS_SELECTION_RECOMMENDATIONS,
)

# Explicit re-exports for mypy compatibility (dynamic imports aren't tracked)
from _aragora_metrics_impl import (  # noqa: F401, E402
    # Server/startup
    start_metrics_server,
    # NOTE: init_core_metrics is wrapped above (line 56) to sync public attrs
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
    # Cache metrics
    record_knowledge_cache_hit,
    record_knowledge_cache_miss,
    # Convergence metrics
    record_convergence_check,
    record_rlm_ready_quorum,
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
    record_km_inbound_event,
    # Task queue metrics
    record_task_queue_operation,
    record_task_queue_recovery,
    record_task_queue_cleanup,
    # Context managers
    track_debate,
    track_phase,
    track_websocket_connection,
    # Handler instrumentation
    track_handler,
    # Metrics server control
    stop_metrics_server,
)
