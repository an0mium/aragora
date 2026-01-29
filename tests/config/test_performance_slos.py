"""Tests for aragora.config.performance_slos module.

Covers all SLO dataclasses, default values, threshold validation,
factory functions, and the aggregated SLOConfig.
"""

import dataclasses

import pytest

from aragora.config.performance_slos import (
    APIEndpointSLO,
    AdapterForwardSyncSLO,
    AdapterReverseSLO,
    AdapterSemanticSearchSLO,
    AdapterSyncSLO,
    AdapterValidationSLO,
    AvailabilitySLO,
    BotResponseSLO,
    BotWebhookSLO,
    ConsensusDetectionSLO,
    ConsensusIngestionSLO,
    ControlPlaneConfigSyncSLO,
    ControlPlaneHealthCheckSLO,
    ControlPlaneLeaderElectionSLO,
    DebateRoundSLO,
    EventDispatchSLO,
    HandlerExecutionSLO,
    KMCheckpointSLO,
    KMIngestionSLO,
    KMQuerySLO,
    LatencySLO,
    MemoryRecallSLO,
    MemoryStoreSLO,
    RLMCompressionSLO,
    RLMQuerySLO,
    RLMStreamingSLO,
    SLOConfig,
    ThroughputSLO,
    WebSocketBroadcastSLO,
    WebSocketConnectionSLO,
    WebSocketMessageSLO,
    WorkflowCheckpointSLO,
    WorkflowExecutionSLO,
    WorkflowRecoverySLO,
    check_latency_slo,
    get_slo_config,
    reset_slo_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global SLO singleton before and after every test."""
    reset_slo_config()
    yield
    reset_slo_config()


# ---------------------------------------------------------------------------
# 1. Base LatencySLO
# ---------------------------------------------------------------------------


class TestLatencySLO:
    def test_construction(self):
        slo = LatencySLO(p50_ms=10, p90_ms=50, p99_ms=200, timeout_ms=1000)
        assert slo.p50_ms == 10
        assert slo.p90_ms == 50
        assert slo.p99_ms == 200
        assert slo.timeout_ms == 1000

    def test_is_within_slo_p99(self):
        slo = LatencySLO(p50_ms=10, p90_ms=50, p99_ms=200, timeout_ms=1000)
        assert slo.is_within_slo(200) is True
        assert slo.is_within_slo(201) is False

    def test_is_within_slo_p50(self):
        slo = LatencySLO(p50_ms=10, p90_ms=50, p99_ms=200, timeout_ms=1000)
        assert slo.is_within_slo(10, percentile="p50") is True
        assert slo.is_within_slo(11, percentile="p50") is False

    def test_is_within_slo_p90(self):
        slo = LatencySLO(p50_ms=10, p90_ms=50, p99_ms=200, timeout_ms=1000)
        assert slo.is_within_slo(50, percentile="p90") is True
        assert slo.is_within_slo(51, percentile="p90") is False

    def test_is_within_slo_unknown_percentile_falls_back_to_p99(self):
        slo = LatencySLO(p50_ms=10, p90_ms=50, p99_ms=200, timeout_ms=1000)
        # Unknown percentile should fallback to p99_ms
        assert slo.is_within_slo(200, percentile="p100") is True
        assert slo.is_within_slo(201, percentile="p100") is False


# ---------------------------------------------------------------------------
# 2. ThroughputSLO
# ---------------------------------------------------------------------------


class TestThroughputSLO:
    def test_construction(self):
        slo = ThroughputSLO(min_rps=10, target_rps=100, max_rps=500)
        assert slo.min_rps == 10
        assert slo.target_rps == 100
        assert slo.max_rps == 500


# ---------------------------------------------------------------------------
# 3. AvailabilitySLO
# ---------------------------------------------------------------------------


class TestAvailabilitySLO:
    def test_construction(self):
        slo = AvailabilitySLO(
            target_percentage=99.9,
            max_consecutive_failures=5,
            recovery_timeout_seconds=30.0,
        )
        assert slo.target_percentage == 99.9
        assert slo.max_consecutive_failures == 5
        assert slo.recovery_timeout_seconds == 30.0


# ---------------------------------------------------------------------------
# 4-12. Knowledge Mound / Adapter / Event / Memory / Debate / RLM / Workflow /
#        Control Plane / WebSocket / Bot / API SLO subclasses
# ---------------------------------------------------------------------------

# Each LatencySLO subclass is tested for default values and inheritance.

_LATENCY_SLO_SUBCLASSES = [
    (KMQuerySLO, 50.0, 150.0, 500.0, 5000.0),
    (KMIngestionSLO, 100.0, 300.0, 1000.0, 10000.0),
    (KMCheckpointSLO, 500.0, 2000.0, 5000.0, 30000.0),
    (ConsensusIngestionSLO, 200.0, 500.0, 1500.0, 10000.0),
    (ConsensusDetectionSLO, 100.0, 300.0, 1000.0, 5000.0),
    (AdapterSyncSLO, 300.0, 800.0, 2000.0, 15000.0),
    (AdapterForwardSyncSLO, 100.0, 300.0, 800.0, 5000.0),
    (AdapterReverseSLO, 50.0, 150.0, 500.0, 3000.0),
    (AdapterSemanticSearchSLO, 100.0, 300.0, 1000.0, 5000.0),
    (AdapterValidationSLO, 200.0, 500.0, 1500.0, 10000.0),
    (EventDispatchSLO, 10.0, 50.0, 200.0, 5000.0),
    (HandlerExecutionSLO, 50.0, 200.0, 1000.0, 10000.0),
    (MemoryStoreSLO, 20.0, 80.0, 300.0, 2000.0),
    (MemoryRecallSLO, 30.0, 100.0, 400.0, 3000.0),
    (DebateRoundSLO, 5000.0, 15000.0, 30000.0, 120000.0),
    (RLMCompressionSLO, 200.0, 500.0, 1500.0, 10000.0),
    (RLMStreamingSLO, 100.0, 300.0, 800.0, 5000.0),
    (RLMQuerySLO, 150.0, 400.0, 1200.0, 10000.0),
    (WorkflowExecutionSLO, 500.0, 2000.0, 5000.0, 60000.0),
    (WorkflowCheckpointSLO, 50.0, 150.0, 500.0, 5000.0),
    (WorkflowRecoverySLO, 200.0, 500.0, 1500.0, 10000.0),
    (ControlPlaneLeaderElectionSLO, 100.0, 300.0, 1000.0, 10000.0),
    (ControlPlaneConfigSyncSLO, 50.0, 150.0, 500.0, 5000.0),
    (ControlPlaneHealthCheckSLO, 10.0, 30.0, 100.0, 1000.0),
    (WebSocketConnectionSLO, 50.0, 150.0, 500.0, 5000.0),
    (WebSocketMessageSLO, 10.0, 30.0, 100.0, 1000.0),
    (WebSocketBroadcastSLO, 50.0, 150.0, 500.0, 5000.0),
    (BotResponseSLO, 500.0, 1500.0, 3000.0, 30000.0),
    (BotWebhookSLO, 100.0, 500.0, 2500.0, 3000.0),
    (APIEndpointSLO, 100.0, 500.0, 2000.0, 30000.0),
]


@pytest.mark.parametrize(
    "cls,p50,p90,p99,timeout",
    _LATENCY_SLO_SUBCLASSES,
    ids=[c[0].__name__ for c in _LATENCY_SLO_SUBCLASSES],
)
class TestLatencySLOSubclassDefaults:
    def test_default_values(self, cls, p50, p90, p99, timeout):
        slo = cls()
        assert slo.p50_ms == p50
        assert slo.p90_ms == p90
        assert slo.p99_ms == p99
        assert slo.timeout_ms == timeout

    def test_is_subclass_of_latency_slo(self, cls, p50, p90, p99, timeout):
        assert issubclass(cls, LatencySLO)

    def test_is_within_slo_inherited(self, cls, p50, p90, p99, timeout):
        slo = cls()
        assert slo.is_within_slo(p99) is True
        assert slo.is_within_slo(p99 + 1) is False


# ---------------------------------------------------------------------------
# Extra fields on specific subclasses
# ---------------------------------------------------------------------------


class TestSubclassExtraFields:
    def test_km_query_max_results(self):
        slo = KMQuerySLO()
        assert slo.max_results == 100

    def test_km_ingestion_batch_size(self):
        slo = KMIngestionSLO()
        assert slo.batch_size == 50

    def test_km_checkpoint_max_size_mb(self):
        slo = KMCheckpointSLO()
        assert slo.max_size_mb == 100

    def test_consensus_ingestion_limits(self):
        slo = ConsensusIngestionSLO()
        assert slo.max_claims_per_consensus == 10
        assert slo.max_dissents_per_consensus == 10

    def test_rlm_compression_max_input_tokens(self):
        slo = RLMCompressionSLO()
        assert slo.max_input_tokens == 100000


# ---------------------------------------------------------------------------
# SLOConfig aggregated dataclass
# ---------------------------------------------------------------------------


class TestSLOConfig:
    def test_default_construction(self):
        config = SLOConfig()
        assert isinstance(config.km_query, KMQuerySLO)
        assert isinstance(config.debate_round, DebateRoundSLO)
        assert isinstance(config.availability, AvailabilitySLO)

    def test_all_fields_populated(self):
        config = SLOConfig()
        for f in dataclasses.fields(config):
            value = getattr(config, f.name)
            assert value is not None, f"Field {f.name} is None"

    def test_availability_defaults(self):
        config = SLOConfig()
        assert config.availability.target_percentage == 99.9
        assert config.availability.max_consecutive_failures == 5
        assert config.availability.recovery_timeout_seconds == 30.0

    def test_field_count(self):
        """SLOConfig should have 31 fields (30 operation SLOs + availability)."""
        config = SLOConfig()
        assert len(dataclasses.fields(config)) == 31

    def test_dataclass_asdict_roundtrip(self):
        """Verify serialization via dataclasses.asdict and reconstruction."""
        config = SLOConfig()
        d = dataclasses.asdict(config)
        assert isinstance(d, dict)
        # Spot-check nested structure
        assert d["km_query"]["p50_ms"] == 50.0
        assert d["availability"]["target_percentage"] == 99.9


# ---------------------------------------------------------------------------
# Singleton / factory functions
# ---------------------------------------------------------------------------


class TestGetSLOConfig:
    def test_returns_slo_config(self):
        config = get_slo_config()
        assert isinstance(config, SLOConfig)

    def test_singleton_identity(self):
        a = get_slo_config()
        b = get_slo_config()
        assert a is b

    def test_reset_clears_singleton(self):
        a = get_slo_config()
        reset_slo_config()
        b = get_slo_config()
        assert a is not b


# ---------------------------------------------------------------------------
# check_latency_slo helper
# ---------------------------------------------------------------------------


class TestCheckLatencySLO:
    def test_within_slo(self):
        ok, msg = check_latency_slo("km_query", 100.0)
        assert ok is True
        assert "within" in msg.lower()

    def test_exceeds_slo(self):
        ok, msg = check_latency_slo("km_query", 600.0)
        assert ok is False
        assert "exceeds" in msg.lower()

    def test_exact_threshold(self):
        ok, _ = check_latency_slo("km_query", 500.0, percentile="p99")
        assert ok is True

    def test_custom_percentile(self):
        ok, _ = check_latency_slo("km_query", 50.0, percentile="p50")
        assert ok is True
        ok2, _ = check_latency_slo("km_query", 51.0, percentile="p50")
        assert ok2 is False

    def test_unknown_operation(self):
        ok, msg = check_latency_slo("nonexistent_operation", 99999.0)
        assert ok is True
        assert "No SLO defined" in msg

    def test_message_contains_operation_name(self):
        _, msg = check_latency_slo("event_dispatch", 5.0)
        assert "event_dispatch" in msg

    def test_message_contains_latency_value(self):
        _, msg = check_latency_slo("memory_store", 15.0)
        assert "15.0" in msg


# ---------------------------------------------------------------------------
# Percentile ordering invariant
# ---------------------------------------------------------------------------


class TestPercentileOrdering:
    """All LatencySLO subclasses must satisfy p50 <= p90 <= p99 <= timeout."""

    @pytest.mark.parametrize(
        "cls",
        [c[0] for c in _LATENCY_SLO_SUBCLASSES],
        ids=[c[0].__name__ for c in _LATENCY_SLO_SUBCLASSES],
    )
    def test_ordering(self, cls):
        slo = cls()
        assert slo.p50_ms <= slo.p90_ms <= slo.p99_ms <= slo.timeout_ms
