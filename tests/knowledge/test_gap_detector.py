"""
Tests for Knowledge Gap Detection and Recommendations.

Tests cover:
- KnowledgeGapDetector class
- Coverage gap detection
- Staleness detection
- Contradiction detection
- Recommendation generation
- Coverage score calculation
- Dataclass serialization
- Handler endpoint routing
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.gap_detector import (
    Contradiction,
    KnowledgeGap,
    KnowledgeGapDetector,
    Priority,
    Recommendation,
    RecommendedAction,
    StaleKnowledge,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    mound = MagicMock()
    mound.query = AsyncMock()
    mound.get = AsyncMock()
    mound.get_stale_knowledge = AsyncMock(return_value=[])
    mound.detect_contradictions = AsyncMock(return_value=[])
    return mound


@pytest.fixture
def detector(mock_mound):
    """Create a KnowledgeGapDetector with a mock mound."""
    return KnowledgeGapDetector(mound=mock_mound, workspace_id="test_workspace")


def _make_item(
    item_id: str = "item-1",
    content: str = "Test content",
    confidence: float = 0.8,
    updated_at: datetime | None = None,
    domain: str = "legal",
    source_type: str = "debate",
):
    """Create a mock knowledge item."""
    item = MagicMock()
    item.id = item_id
    item.content = content
    item.confidence = confidence
    item.updated_at = updated_at or datetime.now()
    item.created_at = updated_at or datetime.now()
    item.domain = domain
    item.source_type = source_type
    return item


def _make_query_result(items=None):
    """Create a mock query result."""
    result = MagicMock()
    result.items = items or []
    return result


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestDataclasses:
    """Tests for gap detector dataclasses."""

    def test_knowledge_gap_to_dict(self):
        """Should serialize KnowledgeGap to dictionary."""
        gap = KnowledgeGap(
            domain="legal",
            topic="contracts",
            description="Low coverage in contracts",
            severity=0.75,
            expected_items=10,
            actual_items=3,
        )
        d = gap.to_dict()
        assert d["domain"] == "legal"
        assert d["topic"] == "contracts"
        assert d["severity"] == 0.75
        assert d["expected_items"] == 10
        assert d["actual_items"] == 3
        assert "detected_at" in d

    def test_stale_knowledge_to_dict(self):
        """Should serialize StaleKnowledge to dictionary."""
        stale = StaleKnowledge(
            item_id="item-1",
            content_preview="Old content about legal matters",
            domain="legal",
            age_days=120.5,
            confidence=0.6,
            last_updated=datetime(2025, 10, 1),
            staleness_score=0.8,
        )
        d = stale.to_dict()
        assert d["item_id"] == "item-1"
        assert d["age_days"] == 120.5
        assert d["confidence"] == 0.6
        assert d["staleness_score"] == 0.8
        assert d["last_updated"] == "2025-10-01T00:00:00"

    def test_stale_knowledge_truncates_preview(self):
        """Should truncate content preview to 200 chars."""
        stale = StaleKnowledge(
            item_id="item-1",
            content_preview="x" * 300,
            domain="legal",
            age_days=100,
            confidence=0.5,
            last_updated=None,
            staleness_score=0.7,
        )
        d = stale.to_dict()
        assert len(d["content_preview"]) == 200

    def test_contradiction_to_dict(self):
        """Should serialize Contradiction to dictionary."""
        c = Contradiction(
            item_a_id="item-1",
            item_b_id="item-2",
            item_a_preview="Contracts require 30-day notice",
            item_b_preview="Contracts require 90-day notice",
            domain="legal/contracts",
            conflict_score=0.85,
        )
        d = c.to_dict()
        assert d["item_a_id"] == "item-1"
        assert d["item_b_id"] == "item-2"
        assert d["conflict_score"] == 0.85
        assert d["domain"] == "legal/contracts"

    def test_recommendation_to_dict(self):
        """Should serialize Recommendation to dictionary."""
        rec = Recommendation(
            priority=Priority.HIGH,
            action=RecommendedAction.CREATE,
            description="Add more legal knowledge",
            domain="legal",
            impact_score=0.9,
            metadata={"gap_type": "coverage"},
        )
        d = rec.to_dict()
        assert d["priority"] == "high"
        assert d["action"] == "create"
        assert d["impact_score"] == 0.9
        assert d["metadata"] == {"gap_type": "coverage"}

    def test_priority_enum_values(self):
        """Should have correct priority values."""
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"

    def test_action_enum_values(self):
        """Should have correct action values."""
        assert RecommendedAction.CREATE.value == "create"
        assert RecommendedAction.UPDATE.value == "update"
        assert RecommendedAction.REVIEW.value == "review"
        assert RecommendedAction.ARCHIVE.value == "archive"


# =============================================================================
# Coverage Gap Detection Tests
# =============================================================================


class TestDetectCoverageGaps:
    """Tests for detect_coverage_gaps method."""

    @pytest.mark.asyncio
    async def test_detects_sparse_domain(self, detector, mock_mound):
        """Should detect when a domain has fewer items than expected."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 3)

        gaps = await detector.detect_coverage_gaps("legal")

        assert len(gaps) >= 1
        # The first gap should be for the overall domain
        domain_gap = [g for g in gaps if g.domain == "legal"]
        assert len(domain_gap) == 1
        assert domain_gap[0].actual_items == 3
        assert domain_gap[0].expected_items == 20
        assert domain_gap[0].severity > 0.5

    @pytest.mark.asyncio
    async def test_no_gap_when_sufficient_items(self, detector, mock_mound):
        """Should not report gap when domain has enough items."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 25)

        gaps = await detector.detect_coverage_gaps("legal")

        domain_gaps = [g for g in gaps if g.domain == "legal" and g.topic == "legal"]
        assert len(domain_gaps) == 0

    @pytest.mark.asyncio
    async def test_custom_min_expected(self, detector, mock_mound):
        """Should use custom min_expected when provided."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 3)

        gaps = await detector.detect_coverage_gaps("legal", min_expected=5)

        domain_gaps = [g for g in gaps if g.domain == "legal" and g.topic == "legal"]
        assert len(domain_gaps) == 1
        assert domain_gaps[0].expected_items == 5

    @pytest.mark.asyncio
    async def test_detects_subdomain_gaps(self, detector, mock_mound):
        """Should detect gaps in subdomains."""
        # Domain query returns plenty, subdomain queries return sparse
        mock_mound.query.return_value = _make_query_result([_make_item()] * 25)

        # Override subdomain queries to return sparse results
        call_count = 0
        original_query = mock_mound.query

        async def side_effect_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Domain query - plenty of items
                return _make_query_result([_make_item()] * 25)
            else:
                # Subdomain queries - sparse
                return _make_query_result([_make_item()] * 2)

        mock_mound.query = AsyncMock(side_effect=side_effect_query)

        gaps = await detector.detect_coverage_gaps("legal")

        # Should have subdomain gaps
        subdomain_gaps = [g for g in gaps if "/" in g.domain]
        assert len(subdomain_gaps) > 0

    @pytest.mark.asyncio
    async def test_sorted_by_severity(self, detector, mock_mound):
        """Should return gaps sorted by severity descending."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 2)

        gaps = await detector.detect_coverage_gaps("legal")

        if len(gaps) > 1:
            for i in range(len(gaps) - 1):
                assert gaps[i].severity >= gaps[i + 1].severity

    @pytest.mark.asyncio
    async def test_handles_query_error(self, detector, mock_mound):
        """Should return empty list on query failure."""
        mock_mound.query = AsyncMock(side_effect=RuntimeError("Connection failed"))

        gaps = await detector.detect_coverage_gaps("legal")

        # Should still return a gap indicating empty domain
        assert isinstance(gaps, list)

    @pytest.mark.asyncio
    async def test_empty_domain(self, detector, mock_mound):
        """Should report maximum severity for empty domain."""
        mock_mound.query.return_value = _make_query_result([])

        gaps = await detector.detect_coverage_gaps("legal")

        domain_gaps = [g for g in gaps if g.domain == "legal" and g.topic == "legal"]
        assert len(domain_gaps) == 1
        assert domain_gaps[0].severity == 1.0
        assert domain_gaps[0].actual_items == 0


# =============================================================================
# Staleness Detection Tests
# =============================================================================


class TestDetectStaleness:
    """Tests for detect_staleness method."""

    @pytest.mark.asyncio
    async def test_detects_stale_entries(self, detector, mock_mound):
        """Should detect entries older than max_age_days."""
        old_date = datetime.now() - timedelta(days=120)
        stale_item = _make_item(item_id="old-1", updated_at=old_date)
        mock_mound.get_stale_knowledge.return_value = [stale_item]

        stale = await detector.detect_staleness(max_age_days=90)

        assert len(stale) == 1
        assert stale[0].item_id == "old-1"
        assert stale[0].age_days > 90

    @pytest.mark.asyncio
    async def test_no_stale_when_fresh(self, detector, mock_mound):
        """Should return empty list when all items are fresh."""
        fresh_item = _make_item(updated_at=datetime.now())
        mock_mound.get_stale_knowledge.return_value = [fresh_item]

        stale = await detector.detect_staleness(max_age_days=90)

        # Fresh item should not be in stale list
        assert len(stale) == 0

    @pytest.mark.asyncio
    async def test_staleness_score_increases_with_age(self, detector, mock_mound):
        """Should assign higher staleness score to older entries."""
        old_120 = _make_item(item_id="old-120", updated_at=datetime.now() - timedelta(days=120))
        old_200 = _make_item(item_id="old-200", updated_at=datetime.now() - timedelta(days=200))
        mock_mound.get_stale_knowledge.return_value = [old_120, old_200]

        stale = await detector.detect_staleness(max_age_days=90)

        assert len(stale) == 2
        scores = {s.item_id: s.staleness_score for s in stale}
        assert scores["old-200"] > scores["old-120"]

    @pytest.mark.asyncio
    async def test_low_confidence_increases_staleness(self, detector, mock_mound):
        """Should penalize low-confidence stale entries more."""
        old_date = datetime.now() - timedelta(days=120)
        high_conf = _make_item(item_id="high", updated_at=old_date, confidence=0.9)
        low_conf = _make_item(item_id="low", updated_at=old_date, confidence=0.2)
        mock_mound.get_stale_knowledge.return_value = [high_conf, low_conf]

        stale = await detector.detect_staleness(max_age_days=90)

        scores = {s.item_id: s.staleness_score for s in stale}
        assert scores["low"] > scores["high"]

    @pytest.mark.asyncio
    async def test_sorted_by_staleness_score(self, detector, mock_mound):
        """Should return entries sorted by staleness score descending."""
        items = [
            _make_item(item_id=f"item-{i}", updated_at=datetime.now() - timedelta(days=100 + i * 30))
            for i in range(5)
        ]
        mock_mound.get_stale_knowledge.return_value = items

        stale = await detector.detect_staleness(max_age_days=90)

        if len(stale) > 1:
            for i in range(len(stale) - 1):
                assert stale[i].staleness_score >= stale[i + 1].staleness_score

    @pytest.mark.asyncio
    async def test_handles_missing_updated_at(self, detector, mock_mound):
        """Should skip items without updated_at."""
        item_no_date = MagicMock()
        item_no_date.updated_at = None
        item_no_date.created_at = None
        mock_mound.get_stale_knowledge.return_value = [item_no_date]

        stale = await detector.detect_staleness()

        assert len(stale) == 0

    @pytest.mark.asyncio
    async def test_handles_string_datetime(self, detector, mock_mound):
        """Should parse ISO format datetime strings."""
        item = MagicMock()
        old_date = datetime.now() - timedelta(days=120)
        item.updated_at = old_date.isoformat()
        item.created_at = old_date.isoformat()
        item.id = "str-date"
        item.content = "test"
        item.confidence = 0.5
        item.domain = "legal"
        mock_mound.get_stale_knowledge.return_value = [item]

        stale = await detector.detect_staleness(max_age_days=90)

        assert len(stale) == 1

    @pytest.mark.asyncio
    async def test_fallback_to_query_when_no_get_stale(self, detector, mock_mound):
        """Should use query fallback when get_stale_knowledge is unavailable."""
        del mock_mound.get_stale_knowledge

        old_item = _make_item(updated_at=datetime.now() - timedelta(days=120))
        mock_mound.query.return_value = _make_query_result([old_item])

        stale = await detector.detect_staleness(max_age_days=90)

        assert len(stale) == 1

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self, detector, mock_mound):
        """Should return empty list on error."""
        mock_mound.get_stale_knowledge = AsyncMock(side_effect=RuntimeError("DB error"))

        stale = await detector.detect_staleness()

        assert stale == []


# =============================================================================
# Contradiction Detection Tests
# =============================================================================


class TestDetectContradictions:
    """Tests for detect_contradictions method."""

    @pytest.mark.asyncio
    async def test_detects_contradictions(self, detector, mock_mound):
        """Should return contradictions from the mound."""
        raw_contradiction = MagicMock()
        raw_contradiction.item_a_id = "item-1"
        raw_contradiction.item_b_id = "item-2"
        raw_contradiction.conflict_score = 0.85
        raw_contradiction.domain = "legal/contracts"
        raw_contradiction.detected_at = datetime.now()

        mock_mound.detect_contradictions.return_value = [raw_contradiction]
        mock_mound.get.return_value = _make_item(content="Contract notice content")

        contradictions = await detector.detect_contradictions()

        assert len(contradictions) == 1
        assert contradictions[0].item_a_id == "item-1"
        assert contradictions[0].item_b_id == "item-2"
        assert contradictions[0].conflict_score == 0.85

    @pytest.mark.asyncio
    async def test_empty_when_no_contradictions(self, detector, mock_mound):
        """Should return empty list when no contradictions exist."""
        mock_mound.detect_contradictions.return_value = []

        contradictions = await detector.detect_contradictions()

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_sorted_by_conflict_score(self, detector, mock_mound):
        """Should sort contradictions by conflict score descending."""
        contras = []
        for score in [0.3, 0.9, 0.6]:
            c = MagicMock()
            c.item_a_id = f"a-{score}"
            c.item_b_id = f"b-{score}"
            c.conflict_score = score
            c.domain = "legal"
            c.detected_at = datetime.now()
            contras.append(c)

        mock_mound.detect_contradictions.return_value = contras

        result = await detector.detect_contradictions()

        assert result[0].conflict_score == 0.9
        assert result[1].conflict_score == 0.6
        assert result[2].conflict_score == 0.3

    @pytest.mark.asyncio
    async def test_falls_back_to_get_contradictions(self, detector, mock_mound):
        """Should try get_contradictions if detect_contradictions is unavailable."""
        del mock_mound.detect_contradictions
        mock_mound.get_contradictions = AsyncMock(return_value=[])

        result = await detector.detect_contradictions()

        assert result == []
        mock_mound.get_contradictions.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_no_contradiction_support(self, detector, mock_mound):
        """Should return empty list when mound has no contradiction methods."""
        del mock_mound.detect_contradictions
        # Ensure get_contradictions doesn't exist either
        if hasattr(mock_mound, "get_contradictions"):
            del mock_mound.get_contradictions

        result = await detector.detect_contradictions()

        assert result == []

    @pytest.mark.asyncio
    async def test_handles_error(self, detector, mock_mound):
        """Should return empty list on error."""
        mock_mound.detect_contradictions = AsyncMock(side_effect=RuntimeError("fail"))

        result = await detector.detect_contradictions()

        assert result == []


# =============================================================================
# Recommendations Tests
# =============================================================================


class TestGetRecommendations:
    """Tests for get_recommendations method."""

    @pytest.mark.asyncio
    async def test_generates_coverage_recommendations(self, detector, mock_mound):
        """Should generate create recommendations for coverage gaps."""
        mock_mound.query.return_value = _make_query_result([])
        mock_mound.get_stale_knowledge.return_value = []
        mock_mound.detect_contradictions.return_value = []

        recs = await detector.get_recommendations(domain="legal")

        create_recs = [r for r in recs if r.action == RecommendedAction.CREATE]
        assert len(create_recs) > 0
        assert all(r.priority in (Priority.HIGH, Priority.MEDIUM, Priority.LOW) for r in create_recs)

    @pytest.mark.asyncio
    async def test_generates_staleness_recommendations(self, detector, mock_mound):
        """Should generate update recommendations for stale entries."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 25)
        old_item = _make_item(updated_at=datetime.now() - timedelta(days=120))
        mock_mound.get_stale_knowledge.return_value = [old_item]
        mock_mound.detect_contradictions.return_value = []

        recs = await detector.get_recommendations(domain="legal")

        update_recs = [r for r in recs if r.action == RecommendedAction.UPDATE]
        assert len(update_recs) >= 1

    @pytest.mark.asyncio
    async def test_generates_contradiction_recommendations(self, detector, mock_mound):
        """Should generate review recommendations for contradictions."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 25)
        mock_mound.get_stale_knowledge.return_value = []

        raw_c = MagicMock()
        raw_c.item_a_id = "item-1"
        raw_c.item_b_id = "item-2"
        raw_c.conflict_score = 0.8
        raw_c.domain = "legal"
        raw_c.detected_at = datetime.now()
        mock_mound.detect_contradictions.return_value = [raw_c]

        recs = await detector.get_recommendations(domain="legal")

        review_recs = [r for r in recs if r.action == RecommendedAction.REVIEW]
        assert len(review_recs) >= 1

    @pytest.mark.asyncio
    async def test_sorted_by_priority_and_impact(self, detector, mock_mound):
        """Should sort recommendations by priority then impact."""
        mock_mound.query.return_value = _make_query_result([])
        mock_mound.get_stale_knowledge.return_value = []
        mock_mound.detect_contradictions.return_value = []

        recs = await detector.get_recommendations()

        if len(recs) > 1:
            priority_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(recs) - 1):
                p1 = priority_order.get(recs[i].priority.value, 2)
                p2 = priority_order.get(recs[i + 1].priority.value, 2)
                if p1 == p2:
                    assert recs[i].impact_score >= recs[i + 1].impact_score
                else:
                    assert p1 <= p2

    @pytest.mark.asyncio
    async def test_respects_limit(self, detector, mock_mound):
        """Should respect the limit parameter."""
        mock_mound.query.return_value = _make_query_result([])
        mock_mound.get_stale_knowledge.return_value = []
        mock_mound.detect_contradictions.return_value = []

        recs = await detector.get_recommendations(limit=3)

        assert len(recs) <= 3

    @pytest.mark.asyncio
    async def test_analyzes_all_domains_when_none(self, detector, mock_mound):
        """Should analyze all known domains when domain is None."""
        mock_mound.query.return_value = _make_query_result([])
        mock_mound.get_stale_knowledge.return_value = []
        mock_mound.detect_contradictions.return_value = []

        recs = await detector.get_recommendations(domain=None)

        domains = {r.domain for r in recs}
        # Should have recommendations from multiple domains
        assert len(domains) >= 1

    @pytest.mark.asyncio
    async def test_archive_recommendation_for_very_stale_low_confidence(
        self, detector, mock_mound
    ):
        """Should recommend archive for very stale, low-confidence entries."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 25)
        very_old = _make_item(
            updated_at=datetime.now() - timedelta(days=500),
            confidence=0.1,
        )
        mock_mound.get_stale_knowledge.return_value = [very_old]
        mock_mound.detect_contradictions.return_value = []

        recs = await detector.get_recommendations(domain="legal")

        archive_recs = [r for r in recs if r.action == RecommendedAction.ARCHIVE]
        assert len(archive_recs) >= 1


# =============================================================================
# Coverage Score Tests
# =============================================================================


class TestGetCoverageScore:
    """Tests for get_coverage_score method."""

    @pytest.mark.asyncio
    async def test_empty_domain_scores_zero(self, detector, mock_mound):
        """Should return 0.0 for empty domain."""
        mock_mound.query.return_value = _make_query_result([])

        score = await detector.get_coverage_score("legal")

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_full_domain_scores_high(self, detector, mock_mound):
        """Should return high score for well-populated domain."""
        items = [
            _make_item(
                item_id=f"item-{i}",
                confidence=0.9,
                updated_at=datetime.now() - timedelta(days=5),
            )
            for i in range(25)
        ]
        mock_mound.query.return_value = _make_query_result(items)

        score = await detector.get_coverage_score("legal")

        assert score > 0.7

    @pytest.mark.asyncio
    async def test_score_between_zero_and_one(self, detector, mock_mound):
        """Should always return score in [0, 1] range."""
        items = [_make_item(item_id=f"item-{i}") for i in range(10)]
        mock_mound.query.return_value = _make_query_result(items)

        score = await detector.get_coverage_score("legal")

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_low_confidence_lowers_score(self, detector, mock_mound):
        """Should return lower score when items have low confidence."""
        low_conf_items = [
            _make_item(item_id=f"item-{i}", confidence=0.2)
            for i in range(20)
        ]
        high_conf_items = [
            _make_item(item_id=f"item-{i}", confidence=0.95)
            for i in range(20)
        ]

        mock_mound.query.return_value = _make_query_result(low_conf_items)
        low_score = await detector.get_coverage_score("legal")

        mock_mound.query.return_value = _make_query_result(high_conf_items)
        high_score = await detector.get_coverage_score("legal")

        assert high_score > low_score

    @pytest.mark.asyncio
    async def test_old_items_lower_score(self, detector, mock_mound):
        """Should return lower score when items are old."""
        old_items = [
            _make_item(item_id=f"item-{i}", updated_at=datetime.now() - timedelta(days=300))
            for i in range(20)
        ]
        fresh_items = [
            _make_item(item_id=f"item-{i}", updated_at=datetime.now() - timedelta(days=5))
            for i in range(20)
        ]

        mock_mound.query.return_value = _make_query_result(old_items)
        old_score = await detector.get_coverage_score("legal")

        mock_mound.query.return_value = _make_query_result(fresh_items)
        fresh_score = await detector.get_coverage_score("legal")

        assert fresh_score > old_score

    @pytest.mark.asyncio
    async def test_handles_query_error(self, detector, mock_mound):
        """Should return 0.0 on query error."""
        mock_mound.query = AsyncMock(side_effect=RuntimeError("Connection failed"))

        score = await detector.get_coverage_score("legal")

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_unknown_domain_uses_default_expected(self, detector, mock_mound):
        """Should use default expected count for unknown domains."""
        items = [_make_item(item_id=f"item-{i}") for i in range(12)]
        mock_mound.query.return_value = _make_query_result(items)

        score = await detector.get_coverage_score("unknown_domain")

        # With 12 items vs default expected of 10, depth_factor should be 1.0
        assert score > 0.0


# =============================================================================
# Handler Tests
# =============================================================================


class TestKnowledgeGapHandler:
    """Tests for the HTTP handler."""

    def test_handler_import(self):
        """Should be importable from the handler module."""
        from aragora.server.handlers.knowledge.gaps import KnowledgeGapHandler

        handler = KnowledgeGapHandler()
        assert handler is not None

    def test_can_handle_gaps_path(self):
        """Should handle /api/v1/knowledge/gaps paths."""
        from aragora.server.handlers.knowledge.gaps import KnowledgeGapHandler

        handler = KnowledgeGapHandler()
        assert handler.can_handle("/api/v1/knowledge/gaps") is True
        assert handler.can_handle("/api/v1/knowledge/gaps/recommendations") is True
        assert handler.can_handle("/api/v1/knowledge/gaps/score") is True
        assert handler.can_handle("/api/v1/other/path") is False

    def test_handler_in_knowledge_init(self):
        """Should be exported from knowledge handlers __init__."""
        from aragora.server.handlers.knowledge import KnowledgeGapHandler

        assert KnowledgeGapHandler is not None
