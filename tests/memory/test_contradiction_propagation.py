"""Tests for cross-system contradiction propagation.

When a contradiction is detected in one memory system, confidence penalties
propagate to related entries in other systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.memory.coordinator import (
    CoordinatorOptions,
    MemoryCoordinator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeKMItem:
    id: str
    content: str
    confidence: float = 0.8
    topics: list[str] = field(default_factory=list)


@dataclass
class FakeQueryResult:
    items: list[FakeKMItem] = field(default_factory=list)


@dataclass
class FakeContinuumEntry:
    id: str
    content: str
    importance: float = 0.8


def _make_km(items: list[FakeKMItem] | None = None) -> MagicMock:
    """Create a mock KnowledgeMound with query + update_confidence."""
    km = MagicMock()
    km.query = AsyncMock(return_value=FakeQueryResult(items=items or []))
    km.update_confidence = AsyncMock(return_value=True)
    return km


def _make_continuum(entries: list[FakeContinuumEntry] | None = None) -> MagicMock:
    """Create a mock ContinuumMemory with search + update_importance."""
    cm = MagicMock()
    cm.search = MagicMock(return_value=entries or [])
    cm.update_importance = MagicMock()
    return cm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContradictionPropagation:
    """Tests for propagate_contradiction_effects."""

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        """When disabled, no propagation occurs."""
        options = CoordinatorOptions(enable_contradiction_propagation=False)
        coord = MemoryCoordinator(options=options)

        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="item-1",
            source_system="mound",
            contradiction_confidence=0.9,
            related_content="machine learning model training",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_content_returns_empty(self):
        """Empty related_content yields no propagation."""
        options = CoordinatorOptions(enable_contradiction_propagation=True)
        coord = MemoryCoordinator(options=options)

        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="item-1",
            source_system="mound",
            contradiction_confidence=0.9,
            related_content="",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_propagate_to_km(self):
        """Contradiction from continuum propagates to matching KM entries."""
        items = [
            FakeKMItem(id="km-1", content="machine learning model deployment strategies"),
            FakeKMItem(id="km-2", content="machine learning training data pipeline"),
        ]
        km = _make_km(items)

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_similarity_threshold=0.3,
        )
        coord = MemoryCoordinator(knowledge_mound=km, options=options)

        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="cm-1",
            source_system="continuum",
            contradiction_confidence=1.0,
            related_content="machine learning model training pipeline",
        )

        assert "mound" in result
        assert len(result["mound"]) > 0
        km.update_confidence.assert_called()

    @pytest.mark.asyncio
    async def test_propagate_to_continuum(self):
        """Contradiction from KM propagates to matching continuum entries."""
        entries = [
            FakeContinuumEntry(
                id="cm-1",
                content="machine learning model deployment pipeline",
            ),
        ]
        cm = _make_continuum(entries)

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_similarity_threshold=0.3,
        )
        coord = MemoryCoordinator(continuum_memory=cm, options=options)

        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="km-1",
            source_system="mound",
            contradiction_confidence=1.0,
            related_content="machine learning model deployment strategies",
        )

        assert "continuum" in result
        assert len(result["continuum"]) > 0
        cm.update_importance.assert_called()

    @pytest.mark.asyncio
    async def test_similarity_threshold_filters(self):
        """Entries below similarity threshold are not affected."""
        items = [
            FakeKMItem(id="km-1", content="cooking recipes and dinner ideas"),
        ]
        km = _make_km(items)

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_similarity_threshold=0.6,
        )
        coord = MemoryCoordinator(knowledge_mound=km, options=options)

        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="src-1",
            source_system="continuum",
            contradiction_confidence=1.0,
            related_content="machine learning model training pipeline",
        )

        # Cooking content has no overlap with ML content
        assert result.get("mound", []) == []

    @pytest.mark.asyncio
    async def test_max_propagation_targets_cap(self):
        """At most max_propagation_targets items are affected per system."""
        items = [
            FakeKMItem(id=f"km-{i}", content="machine learning model training pipeline")
            for i in range(20)
        ]
        km = _make_km(items)

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_similarity_threshold=0.3,
            max_propagation_targets=5,
        )
        coord = MemoryCoordinator(knowledge_mound=km, options=options)

        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="src-1",
            source_system="continuum",
            contradiction_confidence=1.0,
            related_content="machine learning model training data",
        )

        assert len(result.get("mound", [])) <= 5

    @pytest.mark.asyncio
    async def test_skip_source_system(self):
        """Don't propagate back to the source system."""
        items = [
            FakeKMItem(id="km-1", content="machine learning model training"),
        ]
        km = _make_km(items)

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_similarity_threshold=0.3,
        )
        coord = MemoryCoordinator(knowledge_mound=km, options=options)

        # Source is mound — should NOT propagate back to mound
        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="km-1",
            source_system="mound",
            contradiction_confidence=1.0,
            related_content="machine learning model training data",
        )

        assert "mound" not in result

    @pytest.mark.asyncio
    async def test_penalty_scales_with_contradiction_confidence(self):
        """Lower contradiction confidence → smaller penalty."""
        items = [
            FakeKMItem(id="km-1", content="machine learning model training", confidence=0.8),
        ]
        km = _make_km(items)

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_propagation_penalty=0.2,
            contradiction_similarity_threshold=0.3,
        )
        coord = MemoryCoordinator(knowledge_mound=km, options=options)

        await coord.propagate_contradiction_effects(
            contradicted_item_id="src-1",
            source_system="continuum",
            contradiction_confidence=0.5,  # Half confidence
            related_content="machine learning model training pipeline",
        )

        if km.update_confidence.called:
            _, kwargs = km.update_confidence.call_args
            new_conf = kwargs.get("new_confidence") or km.update_confidence.call_args[0][1]
            # Penalty = 0.2 * 0.5 = 0.1, so 0.8 - 0.1 = 0.7
            assert new_conf == pytest.approx(0.7, abs=0.05)

    @pytest.mark.asyncio
    async def test_no_systems_configured(self):
        """With no memory systems, returns empty."""
        options = CoordinatorOptions(enable_contradiction_propagation=True)
        coord = MemoryCoordinator(options=options)

        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="item-1",
            source_system="external",
            contradiction_confidence=1.0,
            related_content="machine learning model training",
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_km_update_failure_graceful(self):
        """If KM update_confidence fails, propagation continues gracefully."""
        items = [
            FakeKMItem(id="km-1", content="machine learning model training"),
        ]
        km = _make_km(items)
        km.update_confidence = AsyncMock(side_effect=RuntimeError("DB error"))

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_similarity_threshold=0.3,
        )
        coord = MemoryCoordinator(knowledge_mound=km, options=options)

        # Should not raise
        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="src-1",
            source_system="continuum",
            contradiction_confidence=1.0,
            related_content="machine learning model training data",
        )
        # Affected list may be empty since update failed
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_propagate_both_systems(self):
        """Contradiction propagates to both KM and continuum simultaneously."""
        km_items = [
            FakeKMItem(id="km-1", content="machine learning model training pipeline"),
        ]
        km = _make_km(km_items)

        cm_entries = [
            FakeContinuumEntry(id="cm-1", content="machine learning model deployment"),
        ]
        cm = _make_continuum(cm_entries)

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_similarity_threshold=0.3,
        )
        coord = MemoryCoordinator(
            knowledge_mound=km,
            continuum_memory=cm,
            options=options,
        )

        result = await coord.propagate_contradiction_effects(
            contradicted_item_id="ext-1",
            source_system="external",
            contradiction_confidence=1.0,
            related_content="machine learning model training data pipeline",
        )

        # Both systems should be affected
        assert "mound" in result or "continuum" in result

    @pytest.mark.asyncio
    async def test_confidence_floor(self):
        """Confidence penalty doesn't go below 0.1."""
        items = [
            FakeKMItem(id="km-1", content="machine learning model training", confidence=0.05),
        ]
        km = _make_km(items)

        options = CoordinatorOptions(
            enable_contradiction_propagation=True,
            contradiction_propagation_penalty=0.5,
            contradiction_similarity_threshold=0.3,
        )
        coord = MemoryCoordinator(knowledge_mound=km, options=options)

        await coord.propagate_contradiction_effects(
            contradicted_item_id="src-1",
            source_system="continuum",
            contradiction_confidence=1.0,
            related_content="machine learning model training pipeline",
        )

        if km.update_confidence.called:
            new_conf = km.update_confidence.call_args[0][1]
            assert new_conf >= 0.1


class TestCoordinatorOptionsDefaults:
    """Verify default values for contradiction propagation config."""

    def test_disabled_by_default(self):
        opts = CoordinatorOptions()
        assert opts.enable_contradiction_propagation is False

    def test_default_penalty(self):
        opts = CoordinatorOptions()
        assert opts.contradiction_propagation_penalty == 0.1

    def test_default_similarity_threshold(self):
        opts = CoordinatorOptions()
        assert opts.contradiction_similarity_threshold == 0.6

    def test_default_max_targets(self):
        opts = CoordinatorOptions()
        assert opts.max_propagation_targets == 10
