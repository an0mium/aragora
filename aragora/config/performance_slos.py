"""
Performance Service Level Objectives (SLOs).

Defines baseline performance thresholds for key operations. These SLOs
are used for monitoring, alerting, and capacity planning.

SLO Tiers:
- P50: 50th percentile (median) - acceptable performance
- P90: 90th percentile - good performance
- P99: 99th percentile - maximum acceptable latency

Usage:
    from aragora.config.performance_slos import get_slo_config, SLOConfig

    config = get_slo_config()
    if latency_ms > config.km_query.p99_ms:
        logger.warning("KM query exceeded P99 SLO")
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LatencySLO:
    """Latency SLO for a specific operation.

    All values are in milliseconds.
    """

    p50_ms: float  # 50th percentile (median)
    p90_ms: float  # 90th percentile
    p99_ms: float  # 99th percentile
    timeout_ms: float  # Hard timeout

    def is_within_slo(self, latency_ms: float, percentile: str = "p99") -> bool:
        """Check if latency is within the specified percentile SLO."""
        threshold = getattr(self, f"{percentile}_ms", self.p99_ms)
        return latency_ms <= threshold


@dataclass
class ThroughputSLO:
    """Throughput SLO for a specific operation."""

    min_rps: float  # Minimum requests per second
    target_rps: float  # Target requests per second
    max_rps: float  # Maximum sustainable RPS


@dataclass
class AvailabilitySLO:
    """Availability SLO for a service or component."""

    target_percentage: float  # e.g., 99.9 for "three nines"
    max_consecutive_failures: int  # Circuit breaker threshold
    recovery_timeout_seconds: float  # Time before retry after failure


# ============================================================================
# Knowledge Mound SLOs
# ============================================================================


@dataclass
class KMQuerySLO(LatencySLO):
    """SLO for Knowledge Mound query operations."""

    p50_ms: float = 50.0
    p90_ms: float = 150.0
    p99_ms: float = 500.0
    timeout_ms: float = 5000.0
    max_results: int = 100  # Default result limit


@dataclass
class KMIngestionSLO(LatencySLO):
    """SLO for Knowledge Mound ingestion operations."""

    p50_ms: float = 100.0
    p90_ms: float = 300.0
    p99_ms: float = 1000.0
    timeout_ms: float = 10000.0
    batch_size: int = 50  # Max nodes per batch


@dataclass
class KMCheckpointSLO(LatencySLO):
    """SLO for Knowledge Mound checkpoint operations."""

    p50_ms: float = 500.0
    p90_ms: float = 2000.0
    p99_ms: float = 5000.0
    timeout_ms: float = 30000.0
    max_size_mb: int = 100  # Max checkpoint size


# ============================================================================
# Consensus Ingestion SLOs
# ============================================================================


@dataclass
class ConsensusIngestionSLO(LatencySLO):
    """SLO for consensus ingestion to Knowledge Mound."""

    p50_ms: float = 200.0
    p90_ms: float = 500.0
    p99_ms: float = 1500.0
    timeout_ms: float = 10000.0
    max_claims_per_consensus: int = 10
    max_dissents_per_consensus: int = 10


# ============================================================================
# Adapter Sync SLOs
# ============================================================================


@dataclass
class AdapterSyncSLO(LatencySLO):
    """SLO for adapter sync operations."""

    p50_ms: float = 300.0
    p90_ms: float = 800.0
    p99_ms: float = 2000.0
    timeout_ms: float = 15000.0


@dataclass
class AdapterForwardSyncSLO(LatencySLO):
    """SLO for adapter forward sync (source → KM) operations."""

    p50_ms: float = 100.0
    p90_ms: float = 300.0
    p99_ms: float = 800.0
    timeout_ms: float = 5000.0


@dataclass
class AdapterReverseSLO(LatencySLO):
    """SLO for adapter reverse query (KM → source) operations."""

    p50_ms: float = 50.0
    p90_ms: float = 150.0
    p99_ms: float = 500.0
    timeout_ms: float = 3000.0


@dataclass
class AdapterSemanticSearchSLO(LatencySLO):
    """SLO for adapter semantic search operations."""

    p50_ms: float = 100.0
    p90_ms: float = 300.0
    p99_ms: float = 1000.0
    timeout_ms: float = 5000.0


@dataclass
class AdapterValidationSLO(LatencySLO):
    """SLO for adapter validation feedback operations."""

    p50_ms: float = 200.0
    p90_ms: float = 500.0
    p99_ms: float = 1500.0
    timeout_ms: float = 10000.0


# ============================================================================
# Cross-Subscriber SLOs
# ============================================================================


@dataclass
class EventDispatchSLO(LatencySLO):
    """SLO for event dispatch to handlers."""

    p50_ms: float = 10.0
    p90_ms: float = 50.0
    p99_ms: float = 200.0
    timeout_ms: float = 5000.0


@dataclass
class HandlerExecutionSLO(LatencySLO):
    """SLO for individual handler execution."""

    p50_ms: float = 50.0
    p90_ms: float = 200.0
    p99_ms: float = 1000.0
    timeout_ms: float = 10000.0


# ============================================================================
# Memory Operations SLOs
# ============================================================================


@dataclass
class MemoryStoreSLO(LatencySLO):
    """SLO for memory store operations."""

    p50_ms: float = 20.0
    p90_ms: float = 80.0
    p99_ms: float = 300.0
    timeout_ms: float = 2000.0


@dataclass
class MemoryRecallSLO(LatencySLO):
    """SLO for memory recall operations."""

    p50_ms: float = 30.0
    p90_ms: float = 100.0
    p99_ms: float = 400.0
    timeout_ms: float = 3000.0


# ============================================================================
# Debate Operations SLOs
# ============================================================================


@dataclass
class DebateRoundSLO(LatencySLO):
    """SLO for debate round execution."""

    p50_ms: float = 5000.0
    p90_ms: float = 15000.0
    p99_ms: float = 30000.0
    timeout_ms: float = 120000.0  # 2 minutes per round


@dataclass
class ConsensusDetectionSLO(LatencySLO):
    """SLO for consensus detection."""

    p50_ms: float = 100.0
    p90_ms: float = 300.0
    p99_ms: float = 1000.0
    timeout_ms: float = 5000.0


# ============================================================================
# RLM (Recursive Language Model) SLOs
# ============================================================================


@dataclass
class RLMCompressionSLO(LatencySLO):
    """SLO for RLM context compression."""

    p50_ms: float = 200.0
    p90_ms: float = 500.0
    p99_ms: float = 1500.0
    timeout_ms: float = 10000.0
    max_input_tokens: int = 100000


@dataclass
class RLMStreamingSLO(LatencySLO):
    """SLO for RLM streaming operations (time-to-first-token)."""

    p50_ms: float = 100.0
    p90_ms: float = 300.0
    p99_ms: float = 800.0
    timeout_ms: float = 5000.0


@dataclass
class RLMQuerySLO(LatencySLO):
    """SLO for RLM query operations."""

    p50_ms: float = 150.0
    p90_ms: float = 400.0
    p99_ms: float = 1200.0
    timeout_ms: float = 10000.0


# ============================================================================
# Workflow Engine SLOs
# ============================================================================


@dataclass
class WorkflowExecutionSLO(LatencySLO):
    """SLO for workflow step execution (per node)."""

    p50_ms: float = 500.0
    p90_ms: float = 2000.0
    p99_ms: float = 5000.0
    timeout_ms: float = 60000.0


@dataclass
class WorkflowCheckpointSLO(LatencySLO):
    """SLO for workflow checkpoint persistence."""

    p50_ms: float = 50.0
    p90_ms: float = 150.0
    p99_ms: float = 500.0
    timeout_ms: float = 5000.0


@dataclass
class WorkflowRecoverySLO(LatencySLO):
    """SLO for workflow recovery from checkpoint."""

    p50_ms: float = 200.0
    p90_ms: float = 500.0
    p99_ms: float = 1500.0
    timeout_ms: float = 10000.0


# ============================================================================
# Control Plane SLOs
# ============================================================================


@dataclass
class ControlPlaneLeaderElectionSLO(LatencySLO):
    """SLO for leader election operations."""

    p50_ms: float = 100.0
    p90_ms: float = 300.0
    p99_ms: float = 1000.0
    timeout_ms: float = 10000.0


@dataclass
class ControlPlaneConfigSyncSLO(LatencySLO):
    """SLO for configuration sync across instances."""

    p50_ms: float = 50.0
    p90_ms: float = 150.0
    p99_ms: float = 500.0
    timeout_ms: float = 5000.0


@dataclass
class ControlPlaneHealthCheckSLO(LatencySLO):
    """SLO for control plane health checks."""

    p50_ms: float = 10.0
    p90_ms: float = 30.0
    p99_ms: float = 100.0
    timeout_ms: float = 1000.0


# ============================================================================
# WebSocket SLOs
# ============================================================================


@dataclass
class WebSocketConnectionSLO(LatencySLO):
    """SLO for WebSocket connection establishment."""

    p50_ms: float = 50.0
    p90_ms: float = 150.0
    p99_ms: float = 500.0
    timeout_ms: float = 5000.0


@dataclass
class WebSocketMessageSLO(LatencySLO):
    """SLO for WebSocket message delivery."""

    p50_ms: float = 10.0
    p90_ms: float = 30.0
    p99_ms: float = 100.0
    timeout_ms: float = 1000.0


@dataclass
class WebSocketBroadcastSLO(LatencySLO):
    """SLO for WebSocket broadcast to all connections."""

    p50_ms: float = 50.0
    p90_ms: float = 150.0
    p99_ms: float = 500.0
    timeout_ms: float = 5000.0


# ============================================================================
# Bot Platform SLOs
# ============================================================================


@dataclass
class BotResponseSLO(LatencySLO):
    """SLO for bot platform response times (Slack, Discord, etc.)."""

    p50_ms: float = 500.0
    p90_ms: float = 1500.0
    p99_ms: float = 3000.0
    timeout_ms: float = 30000.0


@dataclass
class BotWebhookSLO(LatencySLO):
    """SLO for bot webhook acknowledgment (return 200 within platform limits)."""

    p50_ms: float = 100.0
    p90_ms: float = 500.0
    p99_ms: float = 2500.0  # Slack requires < 3s
    timeout_ms: float = 3000.0


# ============================================================================
# API Endpoint SLOs
# ============================================================================


@dataclass
class APIEndpointSLO(LatencySLO):
    """SLO for API endpoint response times."""

    p50_ms: float = 100.0
    p90_ms: float = 500.0
    p99_ms: float = 2000.0
    timeout_ms: float = 30000.0


# ============================================================================
# Aggregated SLO Config
# ============================================================================


@dataclass
class SLOConfig:
    """Aggregated SLO configuration for all operations."""

    # Knowledge Mound
    km_query: KMQuerySLO = field(default_factory=KMQuerySLO)
    km_ingestion: KMIngestionSLO = field(default_factory=KMIngestionSLO)
    km_checkpoint: KMCheckpointSLO = field(default_factory=KMCheckpointSLO)

    # Consensus
    consensus_ingestion: ConsensusIngestionSLO = field(default_factory=ConsensusIngestionSLO)
    consensus_detection: ConsensusDetectionSLO = field(default_factory=ConsensusDetectionSLO)

    # Adapters
    adapter_sync: AdapterSyncSLO = field(default_factory=AdapterSyncSLO)
    adapter_forward_sync: AdapterForwardSyncSLO = field(default_factory=AdapterForwardSyncSLO)
    adapter_reverse: AdapterReverseSLO = field(default_factory=AdapterReverseSLO)
    adapter_semantic_search: AdapterSemanticSearchSLO = field(
        default_factory=AdapterSemanticSearchSLO
    )
    adapter_validation: AdapterValidationSLO = field(default_factory=AdapterValidationSLO)

    # Events
    event_dispatch: EventDispatchSLO = field(default_factory=EventDispatchSLO)
    handler_execution: HandlerExecutionSLO = field(default_factory=HandlerExecutionSLO)

    # Memory
    memory_store: MemoryStoreSLO = field(default_factory=MemoryStoreSLO)
    memory_recall: MemoryRecallSLO = field(default_factory=MemoryRecallSLO)

    # Debate
    debate_round: DebateRoundSLO = field(default_factory=DebateRoundSLO)

    # RLM
    rlm_compression: RLMCompressionSLO = field(default_factory=RLMCompressionSLO)
    rlm_streaming: RLMStreamingSLO = field(default_factory=RLMStreamingSLO)
    rlm_query: RLMQuerySLO = field(default_factory=RLMQuerySLO)

    # Workflow
    workflow_execution: WorkflowExecutionSLO = field(default_factory=WorkflowExecutionSLO)
    workflow_checkpoint: WorkflowCheckpointSLO = field(default_factory=WorkflowCheckpointSLO)
    workflow_recovery: WorkflowRecoverySLO = field(default_factory=WorkflowRecoverySLO)

    # Control Plane
    control_plane_leader_election: ControlPlaneLeaderElectionSLO = field(
        default_factory=ControlPlaneLeaderElectionSLO
    )
    control_plane_config_sync: ControlPlaneConfigSyncSLO = field(
        default_factory=ControlPlaneConfigSyncSLO
    )
    control_plane_health_check: ControlPlaneHealthCheckSLO = field(
        default_factory=ControlPlaneHealthCheckSLO
    )

    # WebSocket
    websocket_connection: WebSocketConnectionSLO = field(default_factory=WebSocketConnectionSLO)
    websocket_message: WebSocketMessageSLO = field(default_factory=WebSocketMessageSLO)
    websocket_broadcast: WebSocketBroadcastSLO = field(default_factory=WebSocketBroadcastSLO)

    # Bot Platforms
    bot_response: BotResponseSLO = field(default_factory=BotResponseSLO)
    bot_webhook: BotWebhookSLO = field(default_factory=BotWebhookSLO)

    # API
    api_endpoint: APIEndpointSLO = field(default_factory=APIEndpointSLO)

    # Global availability
    availability: AvailabilitySLO = field(
        default_factory=lambda: AvailabilitySLO(
            target_percentage=99.9,
            max_consecutive_failures=5,
            recovery_timeout_seconds=30.0,
        )
    )


# Singleton instance
_slo_config: Optional[SLOConfig] = None


def get_slo_config() -> SLOConfig:
    """Get the global SLO configuration.

    Returns:
        SLOConfig with default SLO values
    """
    global _slo_config
    if _slo_config is None:
        _slo_config = SLOConfig()
    return _slo_config


def reset_slo_config() -> None:
    """Reset the SLO configuration (for testing)."""
    global _slo_config
    _slo_config = None


def check_latency_slo(
    operation: str,
    latency_ms: float,
    percentile: str = "p99",
) -> tuple[bool, str]:
    """Check if a latency measurement is within SLO.

    Args:
        operation: Operation name (e.g., "km_query", "consensus_ingestion")
        latency_ms: Measured latency in milliseconds
        percentile: SLO percentile to check against ("p50", "p90", "p99")

    Returns:
        Tuple of (is_within_slo: bool, message: str)
    """
    config = get_slo_config()

    slo = getattr(config, operation, None)
    if slo is None:
        return True, f"No SLO defined for {operation}"

    threshold = getattr(slo, f"{percentile}_ms", slo.p99_ms)
    is_within = latency_ms <= threshold

    if is_within:
        return (
            True,
            f"{operation} latency {latency_ms:.1f}ms within {percentile} SLO ({threshold}ms)",
        )
    else:
        return (
            False,
            f"{operation} latency {latency_ms:.1f}ms EXCEEDS {percentile} SLO ({threshold}ms)",
        )


# ============================================================================
# SLO Summary for Documentation
# ============================================================================

SLO_SUMMARY = """
Performance SLO Summary
=======================

Knowledge Mound Operations:
  - Query:       P50=50ms,  P90=150ms, P99=500ms,  Timeout=5s
  - Ingestion:   P50=100ms, P90=300ms, P99=1000ms, Timeout=10s
  - Checkpoint:  P50=500ms, P90=2s,    P99=5s,     Timeout=30s

Consensus Operations:
  - Ingestion:   P50=200ms, P90=500ms, P99=1500ms, Timeout=10s
  - Detection:   P50=100ms, P90=300ms, P99=1000ms, Timeout=5s

Adapter Operations:
  - Sync:        P50=300ms, P90=800ms, P99=2000ms, Timeout=15s

Event Operations:
  - Dispatch:    P50=10ms,  P90=50ms,  P99=200ms,  Timeout=5s
  - Handler:     P50=50ms,  P90=200ms, P99=1000ms, Timeout=10s

Memory Operations:
  - Store:       P50=20ms,  P90=80ms,  P99=300ms,  Timeout=2s
  - Recall:      P50=30ms,  P90=100ms, P99=400ms,  Timeout=3s

Debate Operations:
  - Round:       P50=5s,    P90=15s,   P99=30s,    Timeout=120s

RLM Operations:
  - Compression: P50=200ms, P90=500ms, P99=1500ms, Timeout=10s
  - Streaming:   P50=100ms, P90=300ms, P99=800ms,  Timeout=5s (TTFT)
  - Query:       P50=150ms, P90=400ms, P99=1200ms, Timeout=10s

Workflow Operations:
  - Execution:   P50=500ms, P90=2s,    P99=5s,     Timeout=60s
  - Checkpoint:  P50=50ms,  P90=150ms, P99=500ms,  Timeout=5s
  - Recovery:    P50=200ms, P90=500ms, P99=1500ms, Timeout=10s

Control Plane Operations:
  - Leader:      P50=100ms, P90=300ms, P99=1000ms, Timeout=10s
  - Config Sync: P50=50ms,  P90=150ms, P99=500ms,  Timeout=5s
  - Health:      P50=10ms,  P90=30ms,  P99=100ms,  Timeout=1s

WebSocket Operations:
  - Connection:  P50=50ms,  P90=150ms, P99=500ms,  Timeout=5s
  - Message:     P50=10ms,  P90=30ms,  P99=100ms,  Timeout=1s
  - Broadcast:   P50=50ms,  P90=150ms, P99=500ms,  Timeout=5s

Bot Platform Operations:
  - Response:    P50=500ms, P90=1.5s,  P99=3s,     Timeout=30s
  - Webhook:     P50=100ms, P90=500ms, P99=2500ms, Timeout=3s

API Endpoints:
  - Response:    P50=100ms, P90=500ms, P99=2000ms, Timeout=30s

Availability:
  - Target: 99.9% uptime
  - Max consecutive failures: 5
  - Recovery timeout: 30s
"""
