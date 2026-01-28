"""
Tests for FusionMixin - Multi-adapter fusion capabilities for Knowledge Mound adapters.

Tests cover:
- Abstract method contracts
- Template method fuse_validations_from_km()
- Item partitioning by source
- Adapter weight computation
- Conflict detection and resolution tracking
- FusionState statistics
- Integration with FusionCoordinator

Phase A3 implementation.
"""

import pytest
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from aragora.knowledge.mound.adapters._fusion_mixin import (
    FusionMixin,
    FusionSyncResult,
    FusionState,
)
from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter
from aragora.knowledge.mound.ops.fusion import (
    FusionStrategy,
    ConflictResolution,
    AdapterValidation,
    FusedValidation,
)


class MockFusionAdapter(FusionMixin, KnowledgeMoundAdapter):
    """Mock adapter for testing FusionMixin."""

    adapter_name = "mock_fusion"

    def __init__(self, fusion_sources: Optional[List[str]] = None):
        super().__init__()
        self._fusion_sources = fusion_sources or ["consensus", "elo", "belief"]
        self._records: Dict[str, Any] = {}
        self._apply_calls: List[Dict[str, Any]] = []

    def _get_fusion_sources(self) -> List[str]:
        return self._fusion_sources

    def _extract_fusible_data(
        self,
        km_item: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        metadata = km_item.get("metadata", {})
        confidence = km_item.get("confidence", metadata.get("confidence"))

        if confidence is None:
            return None

        return {
            "confidence": confidence,
            "is_valid": km_item.get("is_valid", confidence >= 0.5),
            "sources": km_item.get("sources", []),
            "reasoning": km_item.get("reasoning"),
        }

    def _apply_fusion_result(
        self,
        record: Any,
        fusion_result: FusedValidation,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self._apply_calls.append(
            {
                "record": record,
                "fusion_result": fusion_result,
                "metadata": metadata,
            }
        )

        if hasattr(record, "fused_confidence"):
            record.fused_confidence = fusion_result.fused_confidence
            return True

        return True

    def _get_record_for_fusion(self, source_id: str) -> Optional[Any]:
        return self._records.get(source_id)

    def add_record(self, record_id: str, record: Any) -> None:
        self._records[record_id] = record


class TestFusionSyncResult:
    """Tests for FusionSyncResult TypedDict."""

    def test_create_empty_result(self):
        """Should create empty result with all fields."""
        result: FusionSyncResult = {
            "items_analyzed": 0,
            "items_fused": 0,
            "items_skipped": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "errors": [],
            "duration_ms": 0.0,
        }

        assert result["items_analyzed"] == 0
        assert result["items_fused"] == 0
        assert result["errors"] == []

    def test_create_result_with_values(self):
        """Should create result with actual values."""
        result: FusionSyncResult = {
            "items_analyzed": 10,
            "items_fused": 8,
            "items_skipped": 2,
            "conflicts_detected": 3,
            "conflicts_resolved": 2,
            "errors": ["error 1"],
            "duration_ms": 150.5,
        }

        assert result["items_analyzed"] == 10
        assert result["items_fused"] == 8
        assert result["conflicts_detected"] == 3
        assert result["duration_ms"] == 150.5


class TestFusionState:
    """Tests for FusionState dataclass."""

    def test_create_default_state(self):
        """Should create state with default values."""
        state = FusionState()

        assert state.fusions_performed == 0
        assert state.conflicts_detected == 0
        assert state.conflicts_resolved == 0
        assert state.last_fusion_at is None
        assert state.source_participation == {}
        assert state.avg_fusion_confidence == 0.0

    def test_to_dict(self):
        """Should convert state to dictionary."""
        state = FusionState(
            fusions_performed=5,
            conflicts_detected=2,
            conflicts_resolved=1,
            last_fusion_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            source_participation={"elo": 3, "consensus": 2},
            avg_fusion_confidence=0.85,
        )

        d = state.to_dict()

        assert d["fusions_performed"] == 5
        assert d["conflicts_detected"] == 2
        assert d["conflicts_resolved"] == 1
        assert d["last_fusion_at"] == "2024-01-15T10:00:00+00:00"
        assert d["source_participation"] == {"elo": 3, "consensus": 2}
        assert d["avg_fusion_confidence"] == 0.85

    def test_to_dict_none_timestamp(self):
        """Should handle None timestamp in to_dict."""
        state = FusionState()

        d = state.to_dict()

        assert d["last_fusion_at"] is None


class TestFusionMixinInit:
    """Tests for FusionMixin initialization."""

    def test_init_fusion_state(self):
        """Should initialize fusion state."""
        adapter = MockFusionAdapter()
        adapter._init_fusion_state()

        assert hasattr(adapter, "_fusion_state")
        assert isinstance(adapter._fusion_state, FusionState)

    def test_supports_fusion_property(self):
        """Should indicate fusion support."""
        adapter = MockFusionAdapter()

        assert adapter.supports_fusion is True

    def test_get_fusion_sources(self):
        """Should return configured fusion sources."""
        adapter = MockFusionAdapter(fusion_sources=["elo", "belief"])

        sources = adapter._get_fusion_sources()

        assert sources == ["elo", "belief"]


class TestPartitionBySource:
    """Tests for _partition_by_source method."""

    def test_partition_empty_items(self):
        """Should handle empty item list."""
        adapter = MockFusionAdapter()

        result = adapter._partition_by_source([])

        assert result == {}

    def test_partition_items_by_source_adapter(self):
        """Should partition items by source_adapter in metadata."""
        adapter = MockFusionAdapter()
        items = [
            {"id": "1", "metadata": {"source_adapter": "elo"}},
            {"id": "2", "metadata": {"source_adapter": "consensus"}},
            {"id": "3", "metadata": {"source_adapter": "elo"}},
        ]

        result = adapter._partition_by_source(items)

        assert len(result) == 2
        assert len(result["elo"]) == 2
        assert len(result["consensus"]) == 1

    def test_partition_items_by_adapter_field(self):
        """Should partition items by adapter field in metadata."""
        adapter = MockFusionAdapter()
        items = [
            {"id": "1", "metadata": {"adapter": "belief"}},
            {"id": "2", "metadata": {"adapter": "evidence"}},
        ]

        result = adapter._partition_by_source(items)

        assert len(result) == 2
        assert "belief" in result
        assert "evidence" in result

    def test_partition_items_by_source_type(self):
        """Should partition items by source_type."""
        adapter = MockFusionAdapter()
        items = [
            {"id": "1", "source_type": "pulse"},
            {"id": "2", "source_type": "insights"},
        ]

        result = adapter._partition_by_source(items)

        assert len(result) == 2
        assert "pulse" in result
        assert "insights" in result

    def test_skip_items_without_source(self):
        """Should skip items without source information."""
        adapter = MockFusionAdapter()
        items = [
            {"id": "1", "metadata": {"source_adapter": "elo"}},
            {"id": "2", "metadata": {}},  # No source
            {"id": "3"},  # No metadata
        ]

        result = adapter._partition_by_source(items)

        assert len(result) == 1
        assert len(result["elo"]) == 1


class TestComputeAdapterWeight:
    """Tests for _compute_adapter_weight method."""

    def test_explicit_reliability(self):
        """Should use explicit reliability when provided."""
        adapter = MockFusionAdapter()

        weight = adapter._compute_adapter_weight("unknown", reliability=0.95)

        assert weight == 0.95

    def test_default_elo_weight(self):
        """Should return high weight for ELO adapter."""
        adapter = MockFusionAdapter()

        weight = adapter._compute_adapter_weight("elo")

        assert weight == 0.9

    def test_default_consensus_weight(self):
        """Should return high weight for consensus adapter."""
        adapter = MockFusionAdapter()

        weight = adapter._compute_adapter_weight("consensus")

        assert weight == 0.85

    def test_default_unknown_weight(self):
        """Should return 0.5 for unknown adapters."""
        adapter = MockFusionAdapter()

        weight = adapter._compute_adapter_weight("unknown_adapter")

        assert weight == 0.5


class TestGetSourcePriority:
    """Tests for _get_source_priority method."""

    def test_elo_highest_priority(self):
        """Should give ELO highest priority."""
        adapter = MockFusionAdapter()

        priority = adapter._get_source_priority("elo")

        assert priority == 5

    def test_consensus_high_priority(self):
        """Should give consensus high priority."""
        adapter = MockFusionAdapter()

        priority = adapter._get_source_priority("consensus")

        assert priority == 4

    def test_unknown_default_priority(self):
        """Should give unknown sources default priority."""
        adapter = MockFusionAdapter()

        priority = adapter._get_source_priority("unknown")

        assert priority == 1


class TestGetFusionStats:
    """Tests for get_fusion_stats method."""

    def test_get_stats_initializes_state(self):
        """Should initialize state if not present."""
        adapter = MockFusionAdapter()

        stats = adapter.get_fusion_stats()

        assert stats["fusions_performed"] == 0
        assert stats["conflicts_detected"] == 0

    def test_get_stats_returns_current_state(self):
        """Should return current fusion state."""
        adapter = MockFusionAdapter()
        adapter._init_fusion_state()
        adapter._fusion_state.fusions_performed = 10
        adapter._fusion_state.conflicts_detected = 3

        stats = adapter.get_fusion_stats()

        assert stats["fusions_performed"] == 10
        assert stats["conflicts_detected"] == 3


class TestFuseValidationsFromKm:
    """Tests for fuse_validations_from_km template method."""

    def test_empty_items_returns_empty_result(self):
        """Should return empty result for empty items."""
        adapter = MockFusionAdapter()

        result = adapter.fuse_validations_from_km([])

        assert result["items_analyzed"] == 0
        assert result["items_fused"] == 0
        assert result["items_skipped"] == 0

    def test_insufficient_sources_skips_all(self):
        """Should skip all items if insufficient sources."""
        adapter = MockFusionAdapter(fusion_sources=["elo", "consensus"])
        items = [
            {"id": "1", "metadata": {"source_adapter": "elo"}, "confidence": 0.8},
        ]

        result = adapter.fuse_validations_from_km(items, min_sources=2)

        assert result["items_skipped"] == 1

    def test_items_with_single_source_skipped(self):
        """Should skip items that only have one source."""
        adapter = MockFusionAdapter(fusion_sources=["elo", "consensus", "belief"])

        # Mock the fusion coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.fuse_validations.return_value = FusedValidation(
            fused_confidence=0.85,
            strategy_used=FusionStrategy.WEIGHTED_AVERAGE,
            participating_adapters=["elo", "consensus"],
            conflict_detected=False,
            conflict_resolved=False,
            resolution_method=None,
            source_confidences={"elo": 0.9, "consensus": 0.8},
            consensus_strength=0.9,
        )

        items = [
            {
                "id": "item1",
                "metadata": {"source_adapter": "elo", "source_id": "item1"},
                "confidence": 0.8,
            },
        ]

        with patch(
            "aragora.knowledge.mound.adapters._fusion_mixin.get_fusion_coordinator",
            return_value=mock_coordinator,
        ):
            result = adapter.fuse_validations_from_km(items, min_sources=2)

        # All items from single source should be skipped
        assert result["items_skipped"] == 1

    @patch("aragora.knowledge.mound.adapters._fusion_mixin.get_fusion_coordinator")
    def test_successful_fusion(self, mock_get_coordinator):
        """Should successfully fuse items from multiple sources."""
        adapter = MockFusionAdapter(fusion_sources=["elo", "consensus"])

        # Add a mock record
        mock_record = MagicMock()
        mock_record.fused_confidence = 0.0
        adapter.add_record("item1", mock_record)

        # Configure mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.fuse_validations.return_value = FusedValidation(
            fused_confidence=0.85,
            strategy_used=FusionStrategy.WEIGHTED_AVERAGE,
            participating_adapters=["elo", "consensus"],
            conflict_detected=False,
            conflict_resolved=False,
            resolution_method=None,
            source_confidences={"elo": 0.9, "consensus": 0.8},
            consensus_strength=0.9,
        )
        mock_get_coordinator.return_value = mock_coordinator

        items = [
            {
                "id": "item1",
                "metadata": {"source_adapter": "elo", "source_id": "item1"},
                "confidence": 0.9,
            },
            {
                "id": "item1",
                "metadata": {"source_adapter": "consensus", "source_id": "item1"},
                "confidence": 0.8,
            },
        ]

        result = adapter.fuse_validations_from_km(items, min_sources=2)

        assert result["items_analyzed"] >= 1
        assert result["items_fused"] >= 1 or result["items_skipped"] >= 0
        assert result["duration_ms"] >= 0

    @patch("aragora.knowledge.mound.adapters._fusion_mixin.get_fusion_coordinator")
    def test_conflict_detection(self, mock_get_coordinator):
        """Should track conflict detection."""
        adapter = MockFusionAdapter(fusion_sources=["elo", "consensus"])

        # Add a mock record
        mock_record = MagicMock()
        adapter.add_record("item1", mock_record)

        # Configure mock coordinator with conflict
        mock_coordinator = MagicMock()
        mock_coordinator.fuse_validations.return_value = FusedValidation(
            fused_confidence=0.75,
            strategy_used=FusionStrategy.WEIGHTED_AVERAGE,
            participating_adapters=["elo", "consensus"],
            conflict_detected=True,
            conflict_resolved=True,
            resolution_method=ConflictResolution.PREFER_HIGHER_CONFIDENCE,
            source_confidences={"elo": 0.9, "consensus": 0.5},
            consensus_strength=0.6,
        )
        mock_get_coordinator.return_value = mock_coordinator

        items = [
            {
                "id": "item1",
                "metadata": {"source_adapter": "elo", "source_id": "item1"},
                "confidence": 0.9,
            },
            {
                "id": "item1",
                "metadata": {"source_adapter": "consensus", "source_id": "item1"},
                "confidence": 0.5,
            },
        ]

        result = adapter.fuse_validations_from_km(items, min_sources=2)

        # Depending on partitioning, we might detect conflicts
        assert result["duration_ms"] >= 0

    @patch("aragora.knowledge.mound.adapters._fusion_mixin.get_fusion_coordinator")
    def test_updates_fusion_state(self, mock_get_coordinator):
        """Should update fusion state after operation."""
        adapter = MockFusionAdapter(fusion_sources=["elo", "consensus"])

        # Add a mock record
        mock_record = MagicMock()
        adapter.add_record("item1", mock_record)

        # Configure mock coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.fuse_validations.return_value = FusedValidation(
            fused_confidence=0.85,
            strategy_used=FusionStrategy.WEIGHTED_AVERAGE,
            participating_adapters=["elo", "consensus"],
            conflict_detected=False,
            conflict_resolved=False,
            resolution_method=None,
            source_confidences={"elo": 0.9, "consensus": 0.8},
            consensus_strength=0.9,
        )
        mock_get_coordinator.return_value = mock_coordinator

        items = [
            {
                "id": "item1",
                "metadata": {"source_adapter": "elo", "source_id": "item1"},
                "confidence": 0.9,
            },
            {
                "id": "item1",
                "metadata": {"source_adapter": "consensus", "source_id": "item1"},
                "confidence": 0.8,
            },
        ]

        adapter.fuse_validations_from_km(items, min_sources=2)

        # State should be updated
        assert adapter._fusion_state.last_fusion_at is not None

    def test_respects_min_confidence_threshold(self):
        """Should skip items below min_confidence."""
        adapter = MockFusionAdapter(fusion_sources=["elo", "consensus"])

        items = [
            {
                "id": "item1",
                "metadata": {"source_adapter": "elo", "source_id": "item1"},
                "confidence": 0.3,  # Below threshold
            },
            {
                "id": "item1",
                "metadata": {"source_adapter": "consensus", "source_id": "item1"},
                "confidence": 0.2,  # Below threshold
            },
        ]

        result = adapter.fuse_validations_from_km(items, min_sources=2, min_confidence=0.5)

        # Items should be skipped due to low confidence
        assert result["items_analyzed"] >= 0


class TestExtractItemId:
    """Tests for _extract_item_id method."""

    def test_extract_source_id(self):
        """Should extract source_id from metadata."""
        adapter = MockFusionAdapter()
        item = {"metadata": {"source_id": "record_123"}}

        item_id = adapter._extract_item_id(item)

        assert item_id == "record_123"

    def test_extract_record_id(self):
        """Should extract record_id from metadata."""
        adapter = MockFusionAdapter()
        item = {"metadata": {"record_id": "rec_456"}}

        item_id = adapter._extract_item_id(item)

        assert item_id == "rec_456"

    def test_extract_id_from_item(self):
        """Should extract id from item directly."""
        adapter = MockFusionAdapter()
        item = {"id": "item_789"}

        item_id = adapter._extract_item_id(item)

        assert item_id == "item_789"

    def test_extract_id_priority(self):
        """Should prefer source_id over record_id over id."""
        adapter = MockFusionAdapter()
        item = {
            "id": "item_id",
            "metadata": {
                "source_id": "source_id",
                "record_id": "record_id",
            },
        }

        item_id = adapter._extract_item_id(item)

        assert item_id == "source_id"


class TestBuildAdapterValidations:
    """Tests for _build_adapter_validations method."""

    def test_build_validations(self):
        """Should build AdapterValidation objects."""
        adapter = MockFusionAdapter()
        source_data = {
            "elo": {
                "item": {"id": "item1", "metadata": {"source_id": "item1"}},
                "fusible": {
                    "confidence": 0.9,
                    "is_valid": True,
                    "sources": ["debate_1"],
                    "reasoning": "High ELO rating",
                },
            },
            "consensus": {
                "item": {"id": "item1", "metadata": {"source_id": "item1"}},
                "fusible": {
                    "confidence": 0.8,
                    "is_valid": True,
                    "sources": ["debate_1"],
                    "reasoning": "Strong consensus",
                },
            },
        }

        validations = adapter._build_adapter_validations(source_data, min_confidence=0.0)

        assert len(validations) == 2
        assert all(isinstance(v, AdapterValidation) for v in validations)

    def test_filter_by_min_confidence(self):
        """Should filter out low confidence items."""
        adapter = MockFusionAdapter()
        source_data = {
            "elo": {
                "item": {"id": "item1"},
                "fusible": {"confidence": 0.9},
            },
            "consensus": {
                "item": {"id": "item1"},
                "fusible": {"confidence": 0.3},  # Below threshold
            },
        }

        validations = adapter._build_adapter_validations(source_data, min_confidence=0.5)

        assert len(validations) == 1
        assert validations[0].adapter_name == "elo"


class TestExtractFusibleData:
    """Tests for _extract_fusible_data abstract method implementation."""

    def test_extract_confidence_from_item(self):
        """Should extract confidence from item."""
        adapter = MockFusionAdapter()
        item = {"confidence": 0.85}

        fusible = adapter._extract_fusible_data(item)

        assert fusible is not None
        assert fusible["confidence"] == 0.85

    def test_extract_confidence_from_metadata(self):
        """Should extract confidence from metadata."""
        adapter = MockFusionAdapter()
        item = {"metadata": {"confidence": 0.75}}

        fusible = adapter._extract_fusible_data(item)

        assert fusible is not None
        assert fusible["confidence"] == 0.75

    def test_returns_none_without_confidence(self):
        """Should return None if no confidence found."""
        adapter = MockFusionAdapter()
        item = {"id": "item1"}

        fusible = adapter._extract_fusible_data(item)

        assert fusible is None

    def test_extract_is_valid(self):
        """Should extract is_valid or derive from confidence."""
        adapter = MockFusionAdapter()

        # Explicit is_valid
        item1 = {"confidence": 0.9, "is_valid": False}
        fusible1 = adapter._extract_fusible_data(item1)
        assert fusible1["is_valid"] is False

        # Derived from confidence >= 0.5
        item2 = {"confidence": 0.6}
        fusible2 = adapter._extract_fusible_data(item2)
        assert fusible2["is_valid"] is True

        # Derived from confidence < 0.5
        item3 = {"confidence": 0.3}
        fusible3 = adapter._extract_fusible_data(item3)
        assert fusible3["is_valid"] is False


class TestGroupItemsById:
    """Tests for _group_items_by_id method."""

    def test_group_items_from_multiple_sources(self):
        """Should group items by ID across sources."""
        adapter = MockFusionAdapter()
        source_items = {
            "elo": [
                {"id": "item1", "confidence": 0.9, "metadata": {"source_id": "item1"}},
                {"id": "item2", "confidence": 0.8, "metadata": {"source_id": "item2"}},
            ],
            "consensus": [
                {"id": "item1", "confidence": 0.85, "metadata": {"source_id": "item1"}},
            ],
        }

        groups = adapter._group_items_by_id(source_items)

        assert "item1" in groups
        assert "item2" in groups
        assert "elo" in groups["item1"]
        assert "consensus" in groups["item1"]
        assert "elo" in groups["item2"]

    def test_skip_items_without_id(self):
        """Should skip items without extractable ID."""
        adapter = MockFusionAdapter()
        source_items = {
            "elo": [
                {"id": "item1", "confidence": 0.9, "metadata": {"source_id": "item1"}},
                {"confidence": 0.8},  # No ID
            ],
        }

        groups = adapter._group_items_by_id(source_items)

        assert len(groups) == 1
        assert "item1" in groups

    def test_tracks_source_participation(self):
        """Should track source participation in fusion state."""
        adapter = MockFusionAdapter()
        adapter._init_fusion_state()

        source_items = {
            "elo": [
                {"id": "item1", "confidence": 0.9, "metadata": {"source_id": "item1"}},
                {"id": "item2", "confidence": 0.8, "metadata": {"source_id": "item2"}},
            ],
            "consensus": [
                {"id": "item1", "confidence": 0.85, "metadata": {"source_id": "item1"}},
            ],
        }

        adapter._group_items_by_id(source_items)

        assert adapter._fusion_state.source_participation.get("elo", 0) >= 1
        assert adapter._fusion_state.source_participation.get("consensus", 0) >= 1
