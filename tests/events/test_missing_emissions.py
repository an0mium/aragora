"""Tests for wiring missing event emissions across subsystems.

Validates that CRUX_DETECTED and MEMORY_COORDINATION events are emitted
at the correct points in crux_detector.py and coordinator.py respectively,
and that both handle import/runtime failures gracefully.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / minimal stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeBeliefDistribution:
    """Minimal belief distribution for testing."""

    p_true: float = 0.5
    p_false: float = 0.5

    @property
    def entropy(self) -> float:
        if self.p_true <= 0 or self.p_true >= 1:
            return 0.0
        return -(self.p_true * math.log2(self.p_true) + self.p_false * math.log2(self.p_false))

    @property
    def confidence(self) -> float:
        return max(self.p_true, self.p_false)

    def to_dict(self) -> dict:
        return {"p_true": self.p_true, "p_false": self.p_false}


@dataclass
class _FakeNode:
    """Minimal belief-network node for testing."""

    claim_id: str
    claim_statement: str
    author: str
    posterior: _FakeBeliefDistribution = field(
        default_factory=_FakeBeliefDistribution,
    )
    centrality: float = 0.5
    child_ids: list[str] = field(default_factory=list)


class _FakeNetwork:
    """Minimal belief network stub for CruxDetector."""

    def __init__(self, nodes: dict[str, _FakeNode] | None = None):
        self.nodes: dict[str, _FakeNode] = nodes or {}
        self.factors: dict = {}
        self.node_factors: dict = {}

    def propagate(self) -> None:
        pass  # no-op for testing

    def get_node_by_claim(self, claim_id: str) -> _FakeNode | None:
        for node in self.nodes.values():
            if node.claim_id == claim_id:
                return node
        return None

    def get_contested_claims(self) -> list:
        return []


def _build_network_with_cruxes() -> _FakeNetwork:
    """Build a network that will produce at least one crux above min_score."""
    nodes = {
        "n1": _FakeNode(
            claim_id="c1",
            claim_statement="Claim one",
            author="agent-a",
            posterior=_FakeBeliefDistribution(p_true=0.6, p_false=0.4),
            centrality=0.8,
        ),
        "n2": _FakeNode(
            claim_id="c2",
            claim_statement="Claim two",
            author="agent-b",
            posterior=_FakeBeliefDistribution(p_true=0.4, p_false=0.6),
            centrality=0.3,
        ),
    }
    return _FakeNetwork(nodes)


def _build_empty_network() -> _FakeNetwork:
    """Build a network with no nodes (no cruxes possible)."""
    return _FakeNetwork()


# ---------------------------------------------------------------------------
# CRUX_DETECTED tests
# ---------------------------------------------------------------------------


class TestCruxDetectedEmission:
    """Tests for CRUX_DETECTED event emission in crux_detector.detect_cruxes."""

    def test_crux_detected_emitted_when_cruxes_found(self):
        """dispatch_event should be called for each crux detected."""
        from aragora.reasoning.crux_detector import CruxDetector

        network = _build_network_with_cruxes()
        detector = CruxDetector(network)

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            result = detector.detect_cruxes(top_k=5, min_score=0.0)

        # Should have emitted one event per crux
        assert len(result.cruxes) > 0
        assert mock_dispatch.call_count == len(result.cruxes)

        # Verify event type and data shape for first call
        event_type, data = mock_dispatch.call_args_list[0][0]
        assert event_type == "crux_detected"
        assert "claim_id" in data
        assert "crux_score" in data
        assert "influence_score" in data
        assert "disagreement_score" in data
        assert "contesting_agents" in data
        assert "resolution_impact" in data
        assert "statement" in data

    def test_crux_detected_not_emitted_when_no_cruxes(self):
        """No events emitted when the network is empty."""
        from aragora.reasoning.crux_detector import CruxDetector

        network = _build_empty_network()
        detector = CruxDetector(network)

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            result = detector.detect_cruxes()

        assert len(result.cruxes) == 0
        mock_dispatch.assert_not_called()

    def test_crux_detected_not_emitted_when_all_below_min_score(self):
        """No events emitted when all cruxes are below min_score."""
        from aragora.reasoning.crux_detector import CruxDetector

        network = _build_network_with_cruxes()
        detector = CruxDetector(network)

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            # Use a very high min_score so all cruxes are filtered out
            result = detector.detect_cruxes(min_score=999.0)

        assert len(result.cruxes) == 0
        mock_dispatch.assert_not_called()

    def test_crux_detected_graceful_on_import_error(self):
        """Emission failure due to ImportError should not break detect_cruxes."""
        from aragora.reasoning.crux_detector import CruxDetector

        network = _build_network_with_cruxes()
        detector = CruxDetector(network)

        with patch.dict("sys.modules", {"aragora.events.dispatcher": None}):
            # Should not raise
            result = detector.detect_cruxes(top_k=5, min_score=0.0)

        assert len(result.cruxes) > 0

    def test_crux_detected_graceful_on_runtime_error(self):
        """Emission failure due to RuntimeError should not break detect_cruxes."""
        from aragora.reasoning.crux_detector import CruxDetector

        network = _build_network_with_cruxes()
        detector = CruxDetector(network)

        mock_dispatch = MagicMock(side_effect=RuntimeError("dispatcher down"))
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            # Should not raise
            result = detector.detect_cruxes(top_k=5, min_score=0.0)

        assert len(result.cruxes) > 0

    def test_crux_detected_statement_truncated(self):
        """Long statements should be truncated to 200 chars in the event."""
        from aragora.reasoning.crux_detector import CruxDetector

        network = _FakeNetwork(
            {
                "n1": _FakeNode(
                    claim_id="c1",
                    claim_statement="X" * 500,
                    author="agent-a",
                    posterior=_FakeBeliefDistribution(p_true=0.6, p_false=0.4),
                    centrality=0.8,
                ),
            }
        )
        detector = CruxDetector(network)

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            result = detector.detect_cruxes(top_k=5, min_score=0.0)

        assert len(result.cruxes) > 0
        _event_type, data = mock_dispatch.call_args_list[0][0]
        assert len(data["statement"]) <= 200

    def test_crux_detected_scores_are_rounded(self):
        """Scores in the event data should be rounded to 4 decimal places."""
        from aragora.reasoning.crux_detector import CruxDetector

        network = _build_network_with_cruxes()
        detector = CruxDetector(network)

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            result = detector.detect_cruxes(top_k=5, min_score=0.0)

        assert len(result.cruxes) > 0
        _event_type, data = mock_dispatch.call_args_list[0][0]
        # Check that scores are rounded floats
        for key in ("crux_score", "influence_score", "disagreement_score", "resolution_impact"):
            val = data[key]
            assert isinstance(val, float)
            # Rounded to 4 decimals means no more than 4 digits after point
            assert val == round(val, 4)


# ---------------------------------------------------------------------------
# MEMORY_COORDINATION tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeDebateResult:
    """Minimal debate result for coordinator tests."""

    final_answer: str = "Test conclusion"
    confidence: float = 0.85
    consensus_reached: bool = True
    winner: str | None = "agent-a"
    rounds_used: int = 3
    key_claims: list[str] = field(default_factory=list)


@dataclass
class _FakeEnvironment:
    """Minimal environment for context."""

    task: str = "Test debate task"


@dataclass
class _FakeDebateContext:
    """Minimal debate context for coordinator tests."""

    debate_id: str = "debate-123"
    domain: str = "general"
    env: _FakeEnvironment = field(default_factory=_FakeEnvironment)
    result: _FakeDebateResult | None = field(default_factory=_FakeDebateResult)
    agents: list = field(default_factory=list)


class _FakeAgent:
    def __init__(self, name: str):
        self.name = name


class TestMemoryCoordinationEmission:
    """Tests for MEMORY_COORDINATION event emission in coordinator."""

    @pytest.fixture
    def mock_continuum(self):
        cm = MagicMock()
        cm.store_pattern = MagicMock(return_value="entry-1")
        cm.delete = MagicMock(return_value={"deleted": True})
        return cm

    @pytest.fixture
    def coordinator_with_continuum(self, mock_continuum):
        from aragora.memory.coordinator import CoordinatorOptions, MemoryCoordinator

        return MemoryCoordinator(
            continuum_memory=mock_continuum,
            options=CoordinatorOptions(
                write_continuum=True,
                write_consensus=False,
                write_critique=False,
                write_mound=False,
                write_supermemory=False,
            ),
        )

    @pytest.mark.asyncio
    async def test_memory_coordination_emitted_on_success(self, coordinator_with_continuum):
        """dispatch_event should be called when transaction succeeds."""
        ctx = _FakeDebateContext(agents=[_FakeAgent("a1")])

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            transaction = await coordinator_with_continuum.commit_debate_outcome(ctx)

        assert transaction.success
        mock_dispatch.assert_called_once()

        event_type, data = mock_dispatch.call_args[0]
        assert event_type == "memory_coordination"
        assert data["transaction_id"] == transaction.id
        assert data["debate_id"] == "debate-123"
        assert data["success"] is True
        assert data["operations_count"] == len(transaction.operations)
        assert "skipped_count" in data

    @pytest.mark.asyncio
    async def test_memory_coordination_not_emitted_on_failure(self):
        """No event should be emitted when all operations fail."""
        from aragora.memory.coordinator import CoordinatorOptions, MemoryCoordinator

        # Create a coordinator with a broken continuum that will fail
        broken_cm = MagicMock()
        broken_cm.store_pattern = MagicMock(side_effect=ValueError("write failed"))

        coordinator = MemoryCoordinator(
            continuum_memory=broken_cm,
            options=CoordinatorOptions(
                write_continuum=True,
                write_consensus=False,
                write_critique=False,
                write_mound=False,
                write_supermemory=False,
                rollback_on_failure=False,
                max_retries=0,
            ),
        )
        ctx = _FakeDebateContext(agents=[_FakeAgent("a1")])

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            transaction = await coordinator.commit_debate_outcome(ctx)

        assert not transaction.success
        mock_dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_coordination_graceful_on_import_error(self, coordinator_with_continuum):
        """ImportError during event emission should not break commit_debate_outcome."""
        ctx = _FakeDebateContext(agents=[_FakeAgent("a1")])

        with patch.dict("sys.modules", {"aragora.events.dispatcher": None}):
            # Should not raise
            transaction = await coordinator_with_continuum.commit_debate_outcome(ctx)

        assert transaction.success

    @pytest.mark.asyncio
    async def test_memory_coordination_graceful_on_runtime_error(self, coordinator_with_continuum):
        """RuntimeError during event emission should not break commit_debate_outcome."""
        ctx = _FakeDebateContext(agents=[_FakeAgent("a1")])

        mock_dispatch = MagicMock(side_effect=RuntimeError("dispatcher down"))
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            # Should not raise
            transaction = await coordinator_with_continuum.commit_debate_outcome(ctx)

        assert transaction.success

    @pytest.mark.asyncio
    async def test_memory_coordination_skipped_count_reported(self):
        """The skipped_count should reflect operations skipped due to thresholds."""
        from aragora.memory.coordinator import CoordinatorOptions, MemoryCoordinator

        mock_cm = MagicMock()
        mock_cm.store_pattern = MagicMock(return_value="entry-1")

        mock_km = MagicMock()

        coordinator = MemoryCoordinator(
            continuum_memory=mock_cm,
            knowledge_mound=mock_km,
            options=CoordinatorOptions(
                write_continuum=True,
                write_consensus=False,
                write_critique=False,
                write_mound=True,
                write_supermemory=False,
                min_confidence_for_mound=0.99,  # Higher than result's 0.85
            ),
        )
        ctx = _FakeDebateContext(agents=[_FakeAgent("a1")])

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            transaction = await coordinator.commit_debate_outcome(ctx)

        assert transaction.success
        mock_dispatch.assert_called_once()
        _event_type, data = mock_dispatch.call_args[0]
        # Mound write should be skipped (confidence 0.85 < 0.99 threshold)
        assert data["skipped_count"] == 1

    @pytest.mark.asyncio
    async def test_memory_coordination_no_result_no_emission(self, coordinator_with_continuum):
        """When context has no result, no operations or events should occur."""
        ctx = _FakeDebateContext(result=None)

        mock_dispatch = MagicMock()
        with patch(
            "aragora.events.dispatcher.dispatch_event",
            mock_dispatch,
        ):
            transaction = await coordinator_with_continuum.commit_debate_outcome(ctx)

        # No operations means no success branch (transaction has 0 ops, so success is True vacuously)
        # But the method returns early before executing anything
        assert len(transaction.operations) == 0
        mock_dispatch.assert_not_called()
