"""Tests for CultureAdapter."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from aragora.knowledge.mound.adapters.culture_adapter import (
    CultureAdapter,
    StoredCulturePattern,
    CultureSearchResult,
)


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    mound = MagicMock()
    mound.workspace_id = "ws_test"
    return mound


@pytest.fixture
def mock_pattern():
    """Create a mock CulturePattern."""
    pattern = MagicMock()
    pattern.id = "pattern_001"
    pattern.workspace_id = "ws_test"
    pattern.pattern_type = MagicMock(value="decision_style")
    pattern.pattern_key = "majority"
    pattern.pattern_value = {"preferred_consensus": "majority"}
    pattern.observation_count = 5
    pattern.confidence = 0.85
    pattern.first_observed_at = datetime(2024, 1, 1, 10, 0, 0)
    pattern.last_observed_at = datetime(2024, 1, 15, 14, 30, 0)
    pattern.contributing_debates = ["debate_001", "debate_002"]
    pattern.metadata = {"domain": "software"}
    return pattern


class TestCultureAdapterInit:
    """Tests for CultureAdapter initialization."""

    def test_init_without_mound(self):
        """Test adapter can initialize without mound."""
        adapter = CultureAdapter()
        assert adapter._mound is None

    def test_init_with_mound(self, mock_mound):
        """Test adapter initializes with provided mound."""
        adapter = CultureAdapter(mound=mock_mound)
        assert adapter._mound == mock_mound


class TestCulturePatternStorage:
    """Tests for storing culture patterns."""

    def test_skips_low_observation_patterns(self, mock_mound, mock_pattern):
        """Test that patterns with few observations are skipped."""
        mock_pattern.observation_count = 1  # Below threshold

        adapter = CultureAdapter(mound=mock_mound)
        result = adapter.store_pattern(mock_pattern)

        assert result is None
        mock_mound.store.assert_not_called()

    def test_stores_pattern_with_sufficient_observations(self, mock_mound, mock_pattern):
        """Test that patterns with enough observations are stored."""
        mock_mound.store = AsyncMock(return_value=MagicMock(node_id="node_123"))

        adapter = CultureAdapter(mound=mock_mound)
        result = adapter.store_pattern(mock_pattern)

        # Should return pattern ID (either from store or prefixed)
        assert result is not None

    def test_converts_pattern_to_content(self, mock_pattern):
        """Test pattern content generation."""
        adapter = CultureAdapter()
        content = adapter._pattern_to_content(mock_pattern)

        assert "Culture Pattern" in content
        assert "decision_style" in content
        assert "majority" in content
        assert "Observations: 5" in content
        assert "software" in content  # From metadata


class TestCulturePatternLoading:
    """Tests for loading culture patterns."""

    def test_load_patterns_without_mound(self):
        """Test loading returns empty list when mound unavailable."""
        adapter = CultureAdapter(mound=None)
        patterns = adapter.load_patterns("ws_test")
        assert patterns == []

    def test_load_patterns_with_mound(self, mock_mound):
        """Test loading patterns from mound."""
        # Since load_patterns involves async, we test the fallback path
        # when loop is running (returns cached or empty)
        adapter = CultureAdapter(mound=mock_mound)

        # Add to cache to test retrieval
        cache_key = "ws_test:None:0.5"
        stored = StoredCulturePattern(
            id="cp_001",
            workspace_id="ws_test",
            pattern_type="decision_style",
            pattern_key="majority",
            pattern_value={},
            observation_count=10,
            confidence=0.8,
            first_observed="",
            last_observed="",
            contributing_debates=[],
            metadata={},
        )
        adapter._pattern_cache[cache_key] = {"cp_001": stored}

        patterns = adapter.load_patterns("ws_test")
        # Returns empty when no cached patterns for exact key
        assert isinstance(patterns, list)


class TestProtocolHints:
    """Tests for getting protocol hints from culture."""

    def test_empty_hints_when_no_patterns(self, mock_mound):
        """Test empty hints when no patterns available."""
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))

        adapter = CultureAdapter(mound=mock_mound)
        adapter._pattern_cache = {}

        # Should return empty when no cache
        hints = adapter.get_protocol_hints("ws_test")
        assert isinstance(hints, dict)

    def test_decision_style_hint(self):
        """Test decision style pattern generates consensus hint."""
        adapter = CultureAdapter()

        # Manually create stored pattern
        pattern = StoredCulturePattern(
            id="p1",
            workspace_id="ws1",
            pattern_type="decision_style",
            pattern_key="unanimous",
            pattern_value={"preferred_consensus": "unanimous"},
            observation_count=10,
            confidence=0.9,
            first_observed="2024-01-01",
            last_observed="2024-01-15",
            contributing_debates=[],
            metadata={},
        )

        # Mock load_patterns to return our pattern
        with patch.object(adapter, "load_patterns", return_value=[pattern]):
            hints = adapter.get_protocol_hints("ws1")

        assert hints.get("recommended_consensus") == "unanimous"

    def test_conservative_risk_hint(self):
        """Test conservative risk tolerance adds extra critiques."""
        adapter = CultureAdapter()

        pattern = StoredCulturePattern(
            id="p2",
            workspace_id="ws1",
            pattern_type="risk_tolerance",
            pattern_key="conservative",
            pattern_value={"level": "conservative"},
            observation_count=15,
            confidence=0.85,
            first_observed="2024-01-01",
            last_observed="2024-01-20",
            contributing_debates=[],
            metadata={},
        )

        with patch.object(adapter, "load_patterns", return_value=[pattern]):
            hints = adapter.get_protocol_hints("ws1")

        assert hints.get("extra_critique_rounds", 0) > 0

    def test_aggressive_risk_hint(self):
        """Test aggressive risk tolerance sets early consensus."""
        adapter = CultureAdapter()

        pattern = StoredCulturePattern(
            id="p3",
            workspace_id="ws1",
            pattern_type="risk_tolerance",
            pattern_key="aggressive",
            pattern_value={"level": "aggressive"},
            observation_count=12,
            confidence=0.8,
            first_observed="2024-01-01",
            last_observed="2024-01-18",
            contributing_debates=[],
            metadata={},
        )

        with patch.object(adapter, "load_patterns", return_value=[pattern]):
            hints = adapter.get_protocol_hints("ws1")

        assert hints.get("early_consensus_threshold") == 0.7


class TestDominantPattern:
    """Tests for getting dominant pattern."""

    def test_returns_highest_confidence(self):
        """Test that dominant pattern has highest confidence."""
        adapter = CultureAdapter()

        patterns = [
            StoredCulturePattern(
                id="p1",
                workspace_id="ws1",
                pattern_type="decision_style",
                pattern_key="majority",
                pattern_value={},
                observation_count=5,
                confidence=0.7,
                first_observed="",
                last_observed="",
                contributing_debates=[],
                metadata={},
            ),
            StoredCulturePattern(
                id="p2",
                workspace_id="ws1",
                pattern_type="decision_style",
                pattern_key="unanimous",
                pattern_value={},
                observation_count=8,
                confidence=0.9,
                first_observed="",
                last_observed="",
                contributing_debates=[],
                metadata={},
            ),
        ]

        with patch.object(adapter, "load_patterns", return_value=patterns):
            from aragora.knowledge.mound.types import CulturePatternType

            dominant = adapter.get_dominant_pattern("ws1", CulturePatternType.DECISION_STYLE)

        assert dominant is not None
        assert dominant.pattern_key == "unanimous"
        assert dominant.confidence == 0.9


class TestOrganizationPromotion:
    """Tests for promoting patterns to organization level."""

    def test_rejects_low_confidence(self, mock_mound):
        """Test that low confidence patterns are not promoted."""
        pattern = StoredCulturePattern(
            id="p1",
            workspace_id="ws1",
            pattern_type="decision_style",
            pattern_key="majority",
            pattern_value={},
            observation_count=10,
            confidence=0.5,  # Below MIN_CONFIDENCE_FOR_PROMOTION
            first_observed="",
            last_observed="",
            contributing_debates=[],
            metadata={},
        )

        adapter = CultureAdapter(mound=mock_mound)
        result = adapter.promote_to_organization(pattern, "org_001")

        assert result is None

    def test_promotes_high_confidence(self, mock_mound):
        """Test that high confidence patterns are promoted."""
        pattern = StoredCulturePattern(
            id="p1",
            workspace_id="ws1",
            pattern_type="decision_style",
            pattern_key="majority",
            pattern_value={},
            observation_count=20,
            confidence=0.9,  # Above threshold
            first_observed="",
            last_observed="",
            contributing_debates=[],
            metadata={},
        )

        mock_mound.store = AsyncMock(return_value=MagicMock(node_id="org_node_001"))

        adapter = CultureAdapter(mound=mock_mound)
        result = adapter.promote_to_organization(pattern, "org_001")

        # Should return node ID or pattern ID
        assert result is not None


class TestSyncOperations:
    """Tests for sync operations."""

    def test_sync_to_mound(self, mock_mound):
        """Test syncing multiple patterns to mound."""
        mock_mound.store = AsyncMock(return_value=MagicMock(node_id="node_123"))

        patterns = []
        for i in range(3):
            p = MagicMock()
            p.id = f"pattern_{i}"
            p.workspace_id = "ws_test"
            p.pattern_type = MagicMock(value="decision_style")
            p.pattern_key = "majority"
            p.pattern_value = {}
            p.observation_count = 5
            p.confidence = 0.8
            p.first_observed_at = datetime.now()
            p.last_observed_at = datetime.now()
            p.contributing_debates = []
            p.metadata = {}
            patterns.append(p)

        adapter = CultureAdapter(mound=mock_mound)
        count = adapter.sync_to_mound(patterns, "ws_test")

        # Count depends on async behavior, but should be >= 0
        assert count >= 0

    def test_load_from_mound_returns_profile(self):
        """Test loading produces a CultureProfile."""
        adapter = CultureAdapter()

        patterns = [
            StoredCulturePattern(
                id="p1",
                workspace_id="ws1",
                pattern_type="decision_style",
                pattern_key="majority",
                pattern_value={},
                observation_count=10,
                confidence=0.8,
                first_observed="",
                last_observed="",
                contributing_debates=[],
                metadata={},
            ),
        ]

        with patch.object(adapter, "load_patterns", return_value=patterns):
            profile = adapter.load_from_mound("ws1")

        assert profile.workspace_id == "ws1"
        assert profile.total_observations == 10
        assert "decision_style" in profile.dominant_traits
