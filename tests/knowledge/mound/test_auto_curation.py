"""
Tests for Knowledge Mound Auto-Curation (Phase 4).

Tests cover:
- Quality scoring calculation
- Curation policy management
- Tier promotion/demotion logic
- Integration with dedup/pruning
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.ops.auto_curation import (
    AutoCurationMixin,
    CurationPolicy,
    CurationCandidate,
    CurationResult,
    CurationAction,
    QualityScore,
    TierLevel,
    TIER_ORDER,
)


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_quality_score_recommendation_promote(self):
        """High scores recommend promotion."""
        score = QualityScore(
            node_id="test_1",
            overall_score=0.9,
            freshness_score=0.9,
            confidence_score=0.85,
            usage_score=0.95,
            relevance_score=0.8,
            relationship_score=0.7,
        )
        assert score.recommendation == CurationAction.PROMOTE

    def test_quality_score_recommendation_archive(self):
        """Very low scores recommend archive."""
        score = QualityScore(
            node_id="test_2",
            overall_score=0.2,
            freshness_score=0.1,
            confidence_score=0.2,
            usage_score=0.1,
            relevance_score=0.3,
            relationship_score=0.2,
        )
        assert score.recommendation == CurationAction.ARCHIVE

    def test_quality_score_recommendation_demote(self):
        """Low scores recommend demotion."""
        score = QualityScore(
            node_id="test_3",
            overall_score=0.4,
            freshness_score=0.5,
            confidence_score=0.4,
            usage_score=0.3,
            relevance_score=0.4,
            relationship_score=0.3,
        )
        assert score.recommendation == CurationAction.DEMOTE

    def test_quality_score_recommendation_refresh(self):
        """Low freshness recommends refresh."""
        score = QualityScore(
            node_id="test_4",
            overall_score=0.6,
            freshness_score=0.2,  # Very stale
            confidence_score=0.7,
            usage_score=0.6,
            relevance_score=0.6,
            relationship_score=0.5,
        )
        assert score.recommendation == CurationAction.REFRESH


class TestCurationPolicy:
    """Tests for CurationPolicy configuration."""

    def test_default_policy(self):
        """Default policy has valid weights."""
        policy = CurationPolicy(workspace_id="test")
        assert policy.validate()
        assert policy.enabled

    def test_invalid_policy_weights(self):
        """Invalid weights fail validation."""
        policy = CurationPolicy(
            workspace_id="test",
            freshness_weight=0.5,
            confidence_weight=0.5,
            usage_weight=0.5,  # Sum > 1.0
            relevance_weight=0.2,
            relationship_weight=0.1,
        )
        assert not policy.validate()

    def test_policy_thresholds(self):
        """Policy thresholds can be customized."""
        policy = CurationPolicy(
            workspace_id="test",
            quality_threshold=0.6,
            promotion_threshold=0.9,
            demotion_threshold=0.4,
        )
        assert policy.quality_threshold == 0.6
        assert policy.promotion_threshold == 0.9
        assert policy.demotion_threshold == 0.4


class TestTierLogic:
    """Tests for tier level ordering."""

    def test_tier_order(self):
        """Tiers are ordered hot -> glacial."""
        assert TIER_ORDER == [
            TierLevel.HOT,
            TierLevel.WARM,
            TierLevel.COLD,
            TierLevel.GLACIAL,
        ]

    def test_tier_values(self):
        """Tier enum values are strings."""
        assert TierLevel.HOT.value == "hot"
        assert TierLevel.GLACIAL.value == "glacial"


class TestCurationResult:
    """Tests for CurationResult dataclass."""

    def test_total_actions(self):
        """Total actions calculated correctly."""
        result = CurationResult(
            workspace_id="test",
            executed_at=datetime.now(timezone.utc),
            policy_id="policy_1",
            duration_ms=100.0,
            items_analyzed=50,
            avg_quality_score=0.6,
            promoted_count=5,
            demoted_count=10,
            archived_count=3,
            merged_count=2,
            refreshed_count=1,
            flagged_count=4,
        )
        assert result.total_actions == 25  # 5+10+3+2+1+4

    def test_empty_result(self):
        """Empty result has zero actions."""
        result = CurationResult(
            workspace_id="test",
            executed_at=datetime.now(timezone.utc),
            policy_id="policy_1",
            duration_ms=50.0,
            items_analyzed=0,
            avg_quality_score=0.0,
        )
        assert result.total_actions == 0


class TestAutoCurationMixin:
    """Tests for AutoCurationMixin methods."""

    @pytest.fixture
    def mock_mound(self):
        """Create mock mound with auto-curation mixin."""

        class MockMound(AutoCurationMixin):
            """Mock mound for testing."""

            pass

        mound = MockMound()
        mound._curation_policies = {}
        mound._curation_history = []
        return mound

    @pytest.mark.asyncio
    async def test_set_and_get_policy(self, mock_mound):
        """Can set and retrieve curation policy."""
        policy = CurationPolicy(
            workspace_id="ws_1",
            name="test_policy",
            quality_threshold=0.7,
        )
        await mock_mound.set_curation_policy(policy)
        retrieved = await mock_mound.get_curation_policy("ws_1")
        assert retrieved is not None
        assert retrieved.name == "test_policy"
        assert retrieved.quality_threshold == 0.7

    @pytest.mark.asyncio
    async def test_set_invalid_policy_raises(self, mock_mound):
        """Setting invalid policy raises ValueError."""
        policy = CurationPolicy(
            workspace_id="ws_1",
            freshness_weight=0.9,
            confidence_weight=0.9,  # Invalid: sum > 1
            usage_weight=0.0,
            relevance_weight=0.0,
            relationship_weight=0.0,
        )
        with pytest.raises(ValueError):
            await mock_mound.set_curation_policy(policy)

    @pytest.mark.asyncio
    async def test_get_nonexistent_policy_returns_none(self, mock_mound):
        """Getting nonexistent policy returns None."""
        policy = await mock_mound.get_curation_policy("nonexistent")
        assert policy is None

    def test_get_higher_tier(self, mock_mound):
        """Tier promotion logic works."""
        assert mock_mound._get_higher_tier("warm") == "hot"
        assert mock_mound._get_higher_tier("cold") == "warm"
        assert mock_mound._get_higher_tier("hot") is None  # Already highest

    def test_get_lower_tier(self, mock_mound):
        """Tier demotion logic works."""
        assert mock_mound._get_lower_tier("warm") == "cold"
        assert mock_mound._get_lower_tier("hot") == "warm"
        assert mock_mound._get_lower_tier("glacial") is None  # Already lowest


class TestCurationActions:
    """Tests for CurationAction enum."""

    def test_action_values(self):
        """Action enum has expected values."""
        assert CurationAction.PROMOTE.value == "promote"
        assert CurationAction.DEMOTE.value == "demote"
        assert CurationAction.MERGE.value == "merge"
        assert CurationAction.ARCHIVE.value == "archive"
        assert CurationAction.REFRESH.value == "refresh"
        assert CurationAction.FLAG.value == "flag"


# Integration tests would require more setup
# These are covered in the e2e test suite
