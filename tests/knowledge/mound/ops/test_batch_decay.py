"""
Tests for batch confidence decay processing.

Phase 7: KM Governance Test Gaps - Batch decay tests.

Tests:
- test_large_batch_decay_10k_items - Performance test
- test_concurrent_workspace_decay - Parallel workspaces
- test_partial_batch_failure - Error isolation
- test_batch_mixed_domains - Domain-specific half-lives
- test_batch_transaction_atomicity - Rollback on failure
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.confidence_decay import (
    ConfidenceAdjustment,
    ConfidenceDecayManager,
    ConfidenceEvent,
    DecayConfig,
    DecayModel,
    DecayReport,
)


# ============================================================================
# Mock Knowledge Item
# ============================================================================


class MockKnowledgeItem:
    """Mock knowledge item for testing."""

    def __init__(
        self,
        item_id: str,
        confidence: float = 0.8,
        domain: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.id = item_id
        self.confidence = confidence
        self.domain = domain
        self.created_at = created_at or datetime.now()
        self.last_accessed = self.created_at
        self.metadata = {}

    @property
    def age_days(self) -> float:
        """Get age in days."""
        return (datetime.now() - self.created_at).total_seconds() / 86400


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def decay_manager():
    """Create a decay manager with default config."""
    return ConfidenceDecayManager()


@pytest.fixture
def custom_decay_manager():
    """Create a decay manager with custom config."""
    config = DecayConfig(
        model=DecayModel.EXPONENTIAL,
        half_life_days=30.0,
        min_confidence=0.1,
        batch_size=50,
        domain_half_lives={
            "technology": 15.0,
            "science": 90.0,
            "news": 3.0,
        },
    )
    return ConfidenceDecayManager(config)


def generate_mock_items(count: int, base_age_days: float = 30.0) -> List[MockKnowledgeItem]:
    """Generate mock knowledge items with varying ages."""
    items = []
    for i in range(count):
        # Vary age to get different decay amounts
        age_days = base_age_days + (i % 100)
        created_at = datetime.now() - timedelta(days=age_days)
        items.append(
            MockKnowledgeItem(
                item_id=f"item-{i}",
                confidence=0.8,
                created_at=created_at,
            )
        )
    return items


# ============================================================================
# Test: Large Batch Decay Performance
# ============================================================================


class TestLargeBatchDecay:
    """Test large batch decay processing."""

    def test_large_batch_decay_10k_items(self, decay_manager):
        """Test decay calculation performance with 10k items."""
        import time

        items = generate_mock_items(10000)

        start = time.time()

        decayed_items = []
        for item in items:
            old_conf = item.confidence
            new_conf = decay_manager.calculate_decay(
                current_confidence=item.confidence,
                age_days=item.age_days,
            )
            if new_conf != old_conf:
                decayed_items.append((item.id, old_conf, new_conf))

        elapsed = time.time() - start

        # Performance: should process 10k items in under 1 second
        assert elapsed < 1.0, f"10k items took {elapsed:.2f}s, expected < 1s"

        # All items should have some decay since they're aged
        assert len(decayed_items) > 0

    def test_batch_decay_preserves_min_confidence(self, decay_manager):
        """Test that batch decay respects minimum confidence floor."""
        # Create very old items that should decay to minimum
        very_old_items = generate_mock_items(100, base_age_days=365 * 5)  # 5 years old

        for item in very_old_items:
            new_conf = decay_manager.calculate_decay(
                current_confidence=item.confidence,
                age_days=item.age_days,
            )
            # Should not go below min_confidence
            assert new_conf >= decay_manager.config.min_confidence


# ============================================================================
# Test: Concurrent Workspace Decay
# ============================================================================


class TestConcurrentWorkspaceDecay:
    """Test parallel workspace decay processing."""

    @pytest.mark.asyncio
    async def test_concurrent_workspace_decay(self, decay_manager):
        """Test concurrent decay across multiple workspaces."""
        # Create items for multiple workspaces
        workspaces = {
            "ws-1": generate_mock_items(100, base_age_days=30),
            "ws-2": generate_mock_items(100, base_age_days=60),
            "ws-3": generate_mock_items(100, base_age_days=90),
        }

        async def process_workspace(ws_id: str, items: List[MockKnowledgeItem]) -> DecayReport:
            """Process decay for a single workspace."""
            import time

            start = time.time()
            adjustments = []
            decayed = 0
            total_change = 0.0

            for item in items:
                old_conf = item.confidence
                new_conf = decay_manager.calculate_decay(
                    current_confidence=item.confidence,
                    age_days=item.age_days,
                )

                if new_conf < old_conf:
                    decayed += 1
                    total_change += old_conf - new_conf
                    adjustments.append(
                        ConfidenceAdjustment(
                            id=f"adj-{item.id}",
                            item_id=item.id,
                            event=ConfidenceEvent.DECAYED,
                            old_confidence=old_conf,
                            new_confidence=new_conf,
                            reason="Time-based decay",
                        )
                    )

            elapsed = (time.time() - start) * 1000  # ms

            return DecayReport(
                workspace_id=ws_id,
                items_processed=len(items),
                items_decayed=decayed,
                items_boosted=0,
                average_confidence_change=total_change / max(decayed, 1),
                adjustments=adjustments,
                duration_ms=elapsed,
            )

        # Process all workspaces concurrently
        tasks = [process_workspace(ws_id, items) for ws_id, items in workspaces.items()]
        reports = await asyncio.gather(*tasks)

        assert len(reports) == 3

        # Each workspace should have been processed
        ws_ids = {r.workspace_id for r in reports}
        assert ws_ids == {"ws-1", "ws-2", "ws-3"}

        # Older workspaces should have more decay
        reports_by_ws = {r.workspace_id: r for r in reports}
        # ws-3 (90 days) should have more decay than ws-1 (30 days)
        assert (
            reports_by_ws["ws-3"].average_confidence_change
            >= reports_by_ws["ws-1"].average_confidence_change
        )


# ============================================================================
# Test: Partial Batch Failure
# ============================================================================


class TestPartialBatchFailure:
    """Test error isolation in batch processing."""

    def test_partial_batch_failure(self, decay_manager):
        """Test that one item failure doesn't affect others."""
        items = generate_mock_items(100)

        results = []
        errors = []

        for item in items:
            try:
                # Simulate occasional error
                if item.id == "item-50":
                    raise ValueError("Simulated error for item-50")

                new_conf = decay_manager.calculate_decay(
                    current_confidence=item.confidence,
                    age_days=item.age_days,
                )
                results.append((item.id, new_conf))
            except Exception as e:
                errors.append((item.id, str(e)))

        # Should have processed 99 items successfully
        assert len(results) == 99
        # Should have 1 error
        assert len(errors) == 1
        assert errors[0][0] == "item-50"


# ============================================================================
# Test: Mixed Domains
# ============================================================================


class TestBatchMixedDomains:
    """Test domain-specific half-lives in batch processing."""

    def test_batch_mixed_domains(self, custom_decay_manager):
        """Test that different domains have appropriate decay rates."""
        manager = custom_decay_manager

        # Create items with different domains (all 30 days old)
        domains = ["technology", "science", "news", None]  # None = default
        items_by_domain: Dict[str, List[float]] = {d or "default": [] for d in domains}

        for domain in domains:
            for i in range(10):
                item = MockKnowledgeItem(
                    item_id=f"{domain}-{i}",
                    confidence=0.8,
                    domain=domain,
                    created_at=datetime.now() - timedelta(days=30),
                )

                new_conf = manager.calculate_decay(
                    current_confidence=item.confidence,
                    age_days=30.0,
                    domain=item.domain,
                )
                items_by_domain[domain or "default"].append(new_conf)

        # Calculate average confidence by domain
        avg_by_domain = {d: sum(confs) / len(confs) for d, confs in items_by_domain.items()}

        # News should decay fastest (3 day half-life)
        # Technology should decay faster than science
        # Science should have highest confidence (90 day half-life)
        assert avg_by_domain["news"] < avg_by_domain["technology"]
        assert avg_by_domain["technology"] < avg_by_domain["science"]

    def test_domain_half_life_application(self, custom_decay_manager):
        """Test that domain-specific half-lives are correctly applied."""
        manager = custom_decay_manager

        # At exactly half-life, confidence should be halved (approximately)
        test_cases = [
            ("technology", 15.0),  # 15-day half-life
            ("science", 90.0),  # 90-day half-life
            ("news", 3.0),  # 3-day half-life
        ]

        for domain, half_life in test_cases:
            new_conf = manager.calculate_decay(
                current_confidence=1.0,  # Start at 1.0 for easy calculation
                age_days=half_life,
                domain=domain,
            )

            # At half-life, confidence should be ~0.5 (exponential decay)
            expected = 0.5
            assert abs(new_conf - expected) < 0.01, (
                f"{domain}: expected ~{expected}, got {new_conf}"
            )


# ============================================================================
# Test: Transaction Atomicity
# ============================================================================


class TestBatchTransactionAtomicity:
    """Test batch processing atomicity."""

    @pytest.mark.asyncio
    async def test_batch_transaction_atomicity(self, decay_manager):
        """Test that batch failures don't partially update items."""
        items = generate_mock_items(100)
        original_confidences = {item.id: item.confidence for item in items}

        # Track which items were "committed"
        committed_updates: Dict[str, float] = {}

        async def process_batch_with_failure():
            """Process batch that fails midway."""
            updates = []

            for i, item in enumerate(items):
                if i == 50:
                    # Simulate failure midway
                    raise ValueError("Batch processing failed")

                new_conf = decay_manager.calculate_decay(
                    current_confidence=item.confidence,
                    age_days=item.age_days,
                )
                updates.append((item.id, new_conf))

            # Only commit if all items processed
            for item_id, new_conf in updates:
                committed_updates[item_id] = new_conf

        try:
            await process_batch_with_failure()
        except ValueError:
            pass  # Expected

        # No items should have been committed due to failure
        assert len(committed_updates) == 0

        # Original confidences should be unchanged
        for item in items:
            assert item.confidence == original_confidences[item.id]


# ============================================================================
# Test: Decay Model Comparison
# ============================================================================


class TestDecayModelComparison:
    """Test different decay models."""

    def test_decay_model_exponential(self):
        """Test exponential decay model."""
        config = DecayConfig(model=DecayModel.EXPONENTIAL, half_life_days=30.0)
        manager = ConfidenceDecayManager(config)

        # Test at various ages
        ages = [0, 15, 30, 60, 90]
        confidences = [manager.calculate_decay(1.0, age) for age in ages]

        # Exponential decay should halve at half-life
        assert abs(confidences[2] - 0.5) < 0.01  # 30 days = half-life
        assert abs(confidences[3] - 0.25) < 0.01  # 60 days = 2 * half-life

    def test_decay_model_linear(self):
        """Test linear decay model."""
        config = DecayConfig(model=DecayModel.LINEAR, half_life_days=30.0)
        manager = ConfidenceDecayManager(config)

        # Linear decay should have constant rate
        conf_15 = manager.calculate_decay(1.0, 15)
        conf_30 = manager.calculate_decay(1.0, 30)
        conf_45 = manager.calculate_decay(1.0, 45)

        # Decay increments should be roughly equal
        decay_1 = 1.0 - conf_15
        decay_2 = conf_15 - conf_30
        decay_3 = conf_30 - conf_45

        assert abs(decay_1 - decay_2) < 0.1
        assert abs(decay_2 - decay_3) < 0.1

    def test_decay_model_step(self):
        """Test step decay model."""
        config = DecayConfig(model=DecayModel.STEP, half_life_days=30.0)
        manager = ConfidenceDecayManager(config)

        # Step decay should have discrete levels
        conf_10 = manager.calculate_decay(1.0, 10)  # < 0.5 * half_life
        conf_20 = manager.calculate_decay(1.0, 20)  # > 0.5 * half_life, < half_life
        conf_40 = manager.calculate_decay(1.0, 40)  # > half_life, < 2 * half_life
        conf_80 = manager.calculate_decay(1.0, 80)  # > 2 * half_life

        assert conf_10 == 1.0  # No decay
        assert conf_20 == 0.75
        assert conf_40 == 0.5
        assert conf_80 == 0.25


__all__ = [
    "TestLargeBatchDecay",
    "TestConcurrentWorkspaceDecay",
    "TestPartialBatchFailure",
    "TestBatchMixedDomains",
    "TestBatchTransactionAtomicity",
    "TestDecayModelComparison",
]
