"""
Integration tests for Knowledge Bridges.

Tests the bridge classes that connect various Aragora systems to KnowledgeMound:
- MetaLearnerBridge: Hyperparameter adjustments → pattern nodes
- EvidenceBridge: External evidence → evidence nodes
- PatternBridge: Extracted patterns → pattern nodes
- KnowledgeBridgeHub: Unified access to all bridges

Run with:
    pytest tests/knowledge/test_bridges.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock KnowledgeMound for Testing
# =============================================================================


class MockKnowledgeMound:
    """Mock KnowledgeMound for testing bridges.

    Stores actual KnowledgeNode objects created by the bridges.
    """

    def __init__(self, workspace_id: str = "test-workspace"):
        self._workspace_id = workspace_id
        self.nodes: Dict[str, Any] = {}
        self._node_counter = 0

    async def add_node(self, node: Any) -> str:
        """Add a node and return its ID."""
        self._node_counter += 1
        node_id = f"node-{self._node_counter}"
        self.nodes[node_id] = node
        return node_id

    async def get_node(self, node_id: str) -> Optional[Any]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    async def query(self, query: str, limit: int = 10) -> List[Any]:
        """Query nodes by content."""
        results = []
        for node in self.nodes.values():
            if hasattr(node, "content") and query.lower() in node.content.lower():
                results.append(node)
                if len(results) >= limit:
                    break
        return results


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    return MockKnowledgeMound()


# =============================================================================
# MetaLearnerBridge Tests
# =============================================================================


class TestMetaLearnerBridge:
    """Tests for MetaLearnerBridge."""

    @pytest.fixture
    def bridge(self, mock_mound):
        """Create a MetaLearnerBridge with mock mound."""
        from aragora.knowledge.bridges import MetaLearnerBridge

        return MetaLearnerBridge(mock_mound)

    @pytest.mark.asyncio
    async def test_init(self, bridge, mock_mound):
        """Test bridge initialization."""
        assert bridge.mound is mock_mound

    @pytest.mark.asyncio
    async def test_capture_adjustment_empty(self, bridge):
        """Test capture_adjustment with empty adjustments returns None."""
        # Mock metrics
        metrics = MagicMock()
        metrics.pattern_retention_rate = 0.8
        metrics.forgetting_rate = 0.1
        hyperparams = MagicMock()

        result = await bridge.capture_adjustment(
            metrics=metrics,
            adjustments={},  # Empty
            hyperparams=hyperparams,
            cycle_number=1,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_capture_adjustment_creates_node(self, bridge, mock_mound):
        """Test capture_adjustment creates a pattern node."""
        # Mock metrics
        metrics = MagicMock()
        metrics.pattern_retention_rate = 0.85
        metrics.forgetting_rate = 0.05
        metrics.prediction_accuracy = 0.92

        hyperparams = MagicMock()
        hyperparams.learning_rate = 0.01
        hyperparams.consolidation_threshold = 0.7

        adjustments = {
            "learning_rate": {"old": 0.01, "new": 0.015},
            "consolidation_threshold": {"old": 0.7, "new": 0.75},
        }

        result = await bridge.capture_adjustment(
            metrics=metrics,
            adjustments=adjustments,
            hyperparams=hyperparams,
            cycle_number=5,
        )

        assert result is not None
        assert result.startswith("node-")
        assert len(mock_mound.nodes) == 1

        node = mock_mound.nodes[result]
        assert node.node_type == "pattern"
        assert "cycle 5" in node.content.lower()


# =============================================================================
# EvidenceBridge Tests
# =============================================================================


class TestEvidenceBridge:
    """Tests for EvidenceBridge."""

    @pytest.fixture
    def bridge(self, mock_mound):
        """Create an EvidenceBridge with mock mound."""
        from aragora.knowledge.bridges import EvidenceBridge

        return EvidenceBridge(mock_mound)

    @pytest.mark.asyncio
    async def test_init(self, bridge, mock_mound):
        """Test bridge initialization."""
        assert bridge.mound is mock_mound

    @pytest.mark.asyncio
    async def test_store_evidence_basic(self, bridge, mock_mound):
        """Test storing basic evidence."""
        node_id = await bridge.store_evidence(
            content="Climate change is accelerating according to IPCC report",
            source="https://ipcc.ch/report/2024",
            evidence_type="citation",
            supports_claim=True,
            strength=0.9,
        )

        assert node_id is not None
        assert node_id.startswith("node-")
        assert len(mock_mound.nodes) == 1

        node = mock_mound.nodes[node_id]
        assert node.node_type == "evidence"
        assert "IPCC" in node.content
        assert node.confidence == 0.9

    @pytest.mark.asyncio
    async def test_store_evidence_with_metadata(self, bridge, mock_mound):
        """Test storing evidence with metadata."""
        node_id = await bridge.store_evidence(
            content="Statistical analysis shows correlation",
            source="study-123",
            evidence_type="data",
            supports_claim=False,  # Refutes
            strength=0.7,
            metadata={"study_year": 2024, "sample_size": 10000},
        )

        assert node_id is not None
        node = mock_mound.nodes[node_id]
        assert node.confidence == 0.7

    @pytest.mark.asyncio
    async def test_store_multiple_evidence(self, bridge, mock_mound):
        """Test storing multiple evidence items."""
        for i in range(5):
            await bridge.store_evidence(
                content=f"Evidence {i}",
                source=f"source-{i}",
                strength=0.5 + i * 0.1,
            )

        assert len(mock_mound.nodes) == 5

    @pytest.mark.asyncio
    async def test_store_from_collector_evidence(self, bridge, mock_mound):
        """Test storing evidence from collector object."""
        # Mock evidence object
        evidence = MagicMock()
        evidence.evidence_id = "ev-123"
        evidence.content = "Tool output shows X"
        evidence.source = "tool:calculator"
        evidence.evidence_type = "tool_output"
        evidence.supports_claim = True
        evidence.strength = 0.95
        evidence.metadata = {"tool": "calculator"}

        node_id = await bridge.store_from_collector_evidence(
            evidence=evidence,
            claim_node_id="claim-456",
        )

        assert node_id is not None
        node = mock_mound.nodes[node_id]
        assert node.node_type == "evidence"
        assert "claim-456" in node.supports


# =============================================================================
# PatternBridge Tests
# =============================================================================


class TestPatternBridge:
    """Tests for PatternBridge."""

    @pytest.fixture
    def bridge(self, mock_mound):
        """Create a PatternBridge with mock mound."""
        from aragora.knowledge.bridges import PatternBridge

        return PatternBridge(mock_mound)

    @pytest.mark.asyncio
    async def test_init(self, bridge, mock_mound):
        """Test bridge initialization."""
        assert bridge.mound is mock_mound

    @pytest.mark.asyncio
    async def test_store_pattern_basic(self, bridge, mock_mound):
        """Test storing a basic pattern."""
        node_id = await bridge.store_pattern(
            description="Agents tend to reach consensus faster on technical topics",
            frequency=0.75,
            example_debates=["debate-1", "debate-2", "debate-3"],
        )

        assert node_id is not None
        node = mock_mound.nodes[node_id]
        assert node.node_type == "pattern"
        assert "consensus" in node.content.lower()
        assert node.confidence == 0.75

    @pytest.mark.asyncio
    async def test_store_pattern_with_category(self, bridge, mock_mound):
        """Test storing pattern with category."""
        node_id = await bridge.store_pattern(
            description="Expert agents contribute more in specialized domains",
            frequency=0.8,
            category="agent_behavior",
        )

        assert node_id is not None

    @pytest.mark.asyncio
    async def test_store_debate_patterns(self, bridge, mock_mound):
        """Test storing patterns from debate analysis."""
        patterns = [
            {"description": "Round 1 proposals are often revised", "frequency": 0.9},
            {"description": "Critiques improve proposal quality", "frequency": 0.85},
            {"description": "Consensus emerges by round 3", "frequency": 0.7},
        ]

        node_ids = []
        for p in patterns:
            node_id = await bridge.store_pattern(
                description=p["description"],
                frequency=p["frequency"],
            )
            node_ids.append(node_id)

        assert len(node_ids) == 3
        assert len(mock_mound.nodes) == 3


# =============================================================================
# KnowledgeBridgeHub Tests
# =============================================================================


class TestKnowledgeBridgeHub:
    """Tests for KnowledgeBridgeHub."""

    @pytest.fixture
    def hub(self, mock_mound):
        """Create a KnowledgeBridgeHub with mock mound."""
        from aragora.knowledge.bridges import KnowledgeBridgeHub

        return KnowledgeBridgeHub(mock_mound)

    def test_init(self, hub, mock_mound):
        """Test hub initialization."""
        assert hub.mound is mock_mound
        # Bridges not yet created (lazy)
        assert hub._meta_learner is None
        assert hub._evidence is None
        assert hub._patterns is None

    def test_lazy_meta_learner_bridge(self, hub):
        """Test lazy initialization of MetaLearner bridge."""
        from aragora.knowledge.bridges import MetaLearnerBridge

        bridge = hub.meta_learner
        assert isinstance(bridge, MetaLearnerBridge)
        assert hub._meta_learner is bridge

        # Second access returns same instance
        assert hub.meta_learner is bridge

    def test_lazy_evidence_bridge(self, hub):
        """Test lazy initialization of Evidence bridge."""
        from aragora.knowledge.bridges import EvidenceBridge

        bridge = hub.evidence
        assert isinstance(bridge, EvidenceBridge)
        assert hub._evidence is bridge

        # Second access returns same instance
        assert hub.evidence is bridge

    def test_lazy_patterns_bridge(self, hub):
        """Test lazy initialization of Pattern bridge."""
        from aragora.knowledge.bridges import PatternBridge

        bridge = hub.patterns
        assert isinstance(bridge, PatternBridge)
        assert hub._patterns is bridge

        # Second access returns same instance
        assert hub.patterns is bridge

    @pytest.mark.asyncio
    async def test_hub_usage_pattern(self, hub, mock_mound):
        """Test typical hub usage pattern."""
        # Store evidence via hub
        evidence_id = await hub.evidence.store_evidence(
            content="Evidence content",
            source="test-source",
        )

        # Store pattern via hub
        pattern_id = await hub.patterns.store_pattern(
            description="Test pattern",
            frequency=0.8,
        )

        assert evidence_id is not None
        assert pattern_id is not None
        assert len(mock_mound.nodes) == 2

    @pytest.mark.asyncio
    async def test_hub_bridges_share_mound(self, hub, mock_mound):
        """Test that all bridges share the same mound."""
        # Use all bridges
        await hub.evidence.store_evidence("E1", "s1")
        await hub.patterns.store_pattern("P1", 0.5)

        # All nodes in same mound
        assert len(mock_mound.nodes) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestBridgeIntegration:
    """Integration tests for bridge interactions."""

    @pytest.fixture
    def hub(self, mock_mound):
        """Create hub for integration tests."""
        from aragora.knowledge.bridges import KnowledgeBridgeHub

        return KnowledgeBridgeHub(mock_mound)

    @pytest.mark.asyncio
    async def test_evidence_supports_pattern(self, hub, mock_mound):
        """Test evidence supporting a pattern."""
        # Store pattern first
        pattern_id = await hub.patterns.store_pattern(
            description="Debates with more rounds have higher quality outcomes",
            frequency=0.8,
        )

        # Store evidence supporting the pattern
        evidence_id = await hub.evidence.store_evidence(
            content="Analysis of 100 debates shows positive correlation",
            source="internal-study",
            supports_claim=True,
            strength=0.9,
        )

        assert pattern_id is not None
        assert evidence_id is not None

    @pytest.mark.asyncio
    async def test_multiple_bridges_concurrent(self, hub, mock_mound):
        """Test concurrent use of multiple bridges."""

        async def store_evidence():
            for i in range(10):
                await hub.evidence.store_evidence(f"Evidence {i}", f"source-{i}")

        async def store_patterns():
            for i in range(10):
                await hub.patterns.store_pattern(f"Pattern {i}", 0.5)

        await asyncio.gather(store_evidence(), store_patterns())

        assert len(mock_mound.nodes) == 20

    @pytest.mark.asyncio
    async def test_bridge_error_handling(self, hub):
        """Test error handling in bridges."""
        # Mock mound to raise error
        hub.mound.add_node = AsyncMock(side_effect=RuntimeError("Storage error"))

        with pytest.raises(RuntimeError):
            await hub.evidence.store_evidence("content", "source")
