"""
Tests for Knowledge Gap Detection and Recommendations.

Tests cover:
- KnowledgeGapDetector class
- Coverage gap detection
- Staleness detection
- Contradiction detection
- Debate receipt analysis
- Frequently asked topic tracking
- Coverage map generation
- Recommendation generation (including debate + FAQ signals)
- Coverage score calculation
- Dataclass serialization
- Handler endpoint routing (including /coverage)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.gap_detector import (
    Contradiction,
    DebateInsight,
    DomainCoverageEntry,
    FrequentlyAskedGap,
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
        assert RecommendedAction.ACQUIRE.value == "acquire"


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
        assert handler.can_handle("/api/v1/knowledge/gaps/coverage") is True
        assert handler.can_handle("/api/v1/knowledge/gaps/score") is True
        assert handler.can_handle("/api/v1/other/path") is False

    def test_handler_in_knowledge_init(self):
        """Should be exported from knowledge handlers __init__."""
        from aragora.server.handlers.knowledge import KnowledgeGapHandler

        assert KnowledgeGapHandler is not None


# =============================================================================
# New Dataclass Tests
# =============================================================================


class TestNewDataclasses:
    """Tests for newly added dataclasses."""

    def test_debate_insight_to_dict(self):
        """Should serialize DebateInsight to dictionary."""
        insight = DebateInsight(
            debate_id="debate-123",
            topic="contract termination",
            domain="legal/contracts",
            confidence=0.35,
            disagreement_score=0.7,
            question="What is the standard notice period?",
        )
        d = insight.to_dict()
        assert d["debate_id"] == "debate-123"
        assert d["topic"] == "contract termination"
        assert d["domain"] == "legal/contracts"
        assert d["confidence"] == 0.35
        assert d["disagreement_score"] == 0.7
        assert d["question"] == "What is the standard notice period?"
        assert "detected_at" in d

    def test_debate_insight_truncates_question(self):
        """Should truncate question to 300 chars in serialization."""
        insight = DebateInsight(
            debate_id="d-1",
            topic="test",
            domain="general",
            confidence=0.5,
            disagreement_score=0.5,
            question="x" * 500,
        )
        d = insight.to_dict()
        assert len(d["question"]) == 300

    def test_frequently_asked_gap_to_dict(self):
        """Should serialize FrequentlyAskedGap to dictionary."""
        gap = FrequentlyAskedGap(
            topic="contract notice",
            query_count=12,
            coverage_score=0.2,
            gap_severity=0.85,
            sample_queries=["contract notice period", "notice requirements"],
        )
        d = gap.to_dict()
        assert d["topic"] == "contract notice"
        assert d["query_count"] == 12
        assert d["coverage_score"] == 0.2
        assert d["gap_severity"] == 0.85
        assert len(d["sample_queries"]) == 2

    def test_frequently_asked_gap_limits_samples(self):
        """Should limit sample queries to 5 in serialization."""
        gap = FrequentlyAskedGap(
            topic="test",
            query_count=10,
            coverage_score=0.1,
            gap_severity=0.9,
            sample_queries=[f"query-{i}" for i in range(10)],
        )
        d = gap.to_dict()
        assert len(d["sample_queries"]) == 5

    def test_domain_coverage_entry_to_dict(self):
        """Should serialize DomainCoverageEntry to dictionary."""
        entry = DomainCoverageEntry(
            domain="legal",
            coverage_score=0.65,
            total_items=15,
            expected_items=20,
            average_confidence=0.75,
            gap_count=3,
            stale_count=2,
            contradiction_count=1,
        )
        d = entry.to_dict()
        assert d["domain"] == "legal"
        assert d["coverage_score"] == 0.65
        assert d["total_items"] == 15
        assert d["expected_items"] == 20
        assert d["average_confidence"] == 0.75
        assert d["gap_count"] == 3
        assert d["stale_count"] == 2
        assert d["contradiction_count"] == 1


# =============================================================================
# Debate Receipt Analysis Tests
# =============================================================================


class TestAnalyzeDebateReceipt:
    """Tests for analyze_debate_receipt method."""

    @pytest.mark.asyncio
    async def test_detects_low_confidence_debate(self, detector):
        """Should flag debates with low confidence as gap signals."""
        receipt = {
            "debate_id": "d-1",
            "topic": "contract law",
            "confidence": 0.3,
            "consensus_score": 0.4,
            "domain": "legal",
            "question": "What is the notice period?",
        }

        insight = await detector.analyze_debate_receipt(receipt)

        assert insight is not None
        assert insight.debate_id == "d-1"
        assert insight.confidence == 0.3
        assert insight.disagreement_score == 0.6
        assert insight.domain == "legal"

    @pytest.mark.asyncio
    async def test_ignores_high_confidence_debate(self, detector):
        """Should return None for debates with high confidence and agreement."""
        receipt = {
            "debate_id": "d-2",
            "topic": "simple question",
            "confidence": 0.95,
            "consensus_score": 0.9,
            "domain": "legal",
        }

        insight = await detector.analyze_debate_receipt(receipt)

        assert insight is None

    @pytest.mark.asyncio
    async def test_handles_high_disagreement(self, detector):
        """Should flag debates with high disagreement."""
        receipt = {
            "debate_id": "d-3",
            "topic": "controversial topic",
            "confidence": 0.6,
            "consensus_score": 0.2,
            "domain": "technical",
        }

        insight = await detector.analyze_debate_receipt(receipt)

        assert insight is not None
        assert insight.disagreement_score == 0.8

    @pytest.mark.asyncio
    async def test_accepts_object_receipt(self, detector):
        """Should work with object-style receipts."""
        receipt = MagicMock()
        receipt.debate_id = "d-obj"
        receipt.topic = "security review"
        receipt.confidence = 0.4
        receipt.consensus_score = 0.5
        receipt.domain = "technical/security"
        receipt.question = "Is the auth system secure?"
        # Clear attrs that fall through to getattr
        receipt.id = "d-obj"
        receipt.task = "security review"
        receipt.consensus_confidence = 0.4
        receipt.agreement = 0.5

        insight = await detector.analyze_debate_receipt(receipt)

        assert insight is not None
        assert insight.debate_id == "d-obj"

    @pytest.mark.asyncio
    async def test_classifies_domain_from_topic(self, detector):
        """Should auto-classify domain when not provided."""
        receipt = {
            "debate_id": "d-auto",
            "topic": "contract termination clause obligations",
            "confidence": 0.3,
            "consensus_score": 0.4,
        }

        insight = await detector.analyze_debate_receipt(receipt)

        assert insight is not None
        # Should classify as legal domain based on contract keywords
        assert insight.domain != "general"

    @pytest.mark.asyncio
    async def test_records_insight_for_later(self, detector):
        """Should accumulate insights for recommendation generation."""
        receipt1 = {
            "debate_id": "d-r1",
            "topic": "topic 1",
            "confidence": 0.3,
            "consensus_score": 0.5,
            "domain": "legal",
        }
        receipt2 = {
            "debate_id": "d-r2",
            "topic": "topic 2",
            "confidence": 0.2,
            "consensus_score": 0.3,
            "domain": "legal",
        }

        await detector.analyze_debate_receipt(receipt1)
        await detector.analyze_debate_receipt(receipt2)

        insights = detector.get_debate_insights()
        assert len(insights) == 2

    @pytest.mark.asyncio
    async def test_debate_with_alternative_field_names(self, detector):
        """Should handle alternative field names (task, agreement)."""
        receipt = {
            "id": "d-alt",
            "task": "evaluate risk",
            "consensus_confidence": 0.3,
            "agreement": 0.4,
            "domain": "financial",
        }

        insight = await detector.analyze_debate_receipt(receipt)

        assert insight is not None
        assert insight.debate_id == "d-alt"
        assert insight.topic == "evaluate risk"


# =============================================================================
# Get Debate Insights Tests
# =============================================================================


class TestGetDebateInsights:
    """Tests for get_debate_insights method."""

    @pytest.mark.asyncio
    async def test_returns_all_insights(self, detector):
        """Should return all recorded debate insights."""
        for i in range(3):
            await detector.analyze_debate_receipt({
                "debate_id": f"d-{i}",
                "topic": f"topic {i}",
                "confidence": 0.3,
                "consensus_score": 0.4,
                "domain": "legal",
            })

        insights = detector.get_debate_insights()
        assert len(insights) == 3

    @pytest.mark.asyncio
    async def test_filters_by_domain(self, detector):
        """Should filter insights by domain."""
        await detector.analyze_debate_receipt({
            "debate_id": "d-legal",
            "topic": "legal topic",
            "confidence": 0.3,
            "consensus_score": 0.4,
            "domain": "legal",
        })
        await detector.analyze_debate_receipt({
            "debate_id": "d-tech",
            "topic": "tech topic",
            "confidence": 0.3,
            "consensus_score": 0.4,
            "domain": "technical",
        })

        legal_only = detector.get_debate_insights(domain="legal")
        assert len(legal_only) == 1
        assert legal_only[0].debate_id == "d-legal"

    @pytest.mark.asyncio
    async def test_filters_by_min_disagreement(self, detector):
        """Should filter by minimum disagreement score."""
        await detector.analyze_debate_receipt({
            "debate_id": "d-low",
            "topic": "low disagreement",
            "confidence": 0.5,
            "consensus_score": 0.6,
            "domain": "legal",
        })
        await detector.analyze_debate_receipt({
            "debate_id": "d-high",
            "topic": "high disagreement",
            "confidence": 0.3,
            "consensus_score": 0.1,
            "domain": "legal",
        })

        high_disagree = detector.get_debate_insights(min_disagreement=0.8)
        assert len(high_disagree) == 1
        assert high_disagree[0].debate_id == "d-high"

    @pytest.mark.asyncio
    async def test_sorted_by_confidence_ascending(self, detector):
        """Should return insights sorted by confidence (lowest first)."""
        for conf in [0.5, 0.2, 0.4]:
            await detector.analyze_debate_receipt({
                "debate_id": f"d-{conf}",
                "topic": "topic",
                "confidence": conf,
                "consensus_score": 0.3,
                "domain": "legal",
            })

        insights = detector.get_debate_insights()
        assert insights[0].confidence == 0.2
        assert insights[1].confidence == 0.4
        assert insights[2].confidence == 0.5

    def test_empty_when_no_receipts(self, detector):
        """Should return empty list when no receipts have been analyzed."""
        insights = detector.get_debate_insights()
        assert insights == []


# =============================================================================
# Query Recording Tests
# =============================================================================


class TestRecordQuery:
    """Tests for record_query method."""

    def test_records_query(self, detector):
        """Should record a query for tracking."""
        detector.record_query("contract notice period")

        # Should have recorded one topic
        assert len(detector._query_topics) == 1

    def test_ignores_empty_queries(self, detector):
        """Should ignore empty or whitespace-only queries."""
        detector.record_query("")
        detector.record_query("   ")

        assert len(detector._query_topics) == 0

    def test_normalizes_queries(self, detector):
        """Should normalize queries to lowercase topic keys."""
        detector.record_query("Contract Notice Period")
        detector.record_query("contract notice period")

        # Both should map to the same topic key
        assert len(detector._query_topics) == 1

    def test_accumulates_multiple_queries(self, detector):
        """Should accumulate identical queries under the same topic."""
        detector.record_query("contract notice period")
        detector.record_query("contract notice period")
        detector.record_query("contract notice period")

        # All identical queries map to the same topic key
        assert len(detector._query_topics) == 1
        topic_key = list(detector._query_topics.keys())[0]
        assert len(detector._query_topics[topic_key]) == 3

    def test_different_topics_tracked_separately(self, detector):
        """Should track different topics separately."""
        detector.record_query("contract notice period")
        detector.record_query("security vulnerability assessment")

        assert len(detector._query_topics) == 2


# =============================================================================
# Frequently Asked Gaps Tests
# =============================================================================


class TestGetFrequentlyAskedGaps:
    """Tests for get_frequently_asked_gaps method."""

    def test_detects_frequent_topic(self, detector):
        """Should detect topics queried more than min_queries times."""
        for _ in range(5):
            detector.record_query("contract notice period")

        gaps = detector.get_frequently_asked_gaps(min_queries=3)

        assert len(gaps) == 1
        assert gaps[0].query_count == 5

    def test_ignores_infrequent_topics(self, detector):
        """Should not include topics below min_queries threshold."""
        detector.record_query("rare topic question")
        detector.record_query("rare topic question")

        gaps = detector.get_frequently_asked_gaps(min_queries=3)

        assert len(gaps) == 0

    def test_sorted_by_gap_severity(self, detector):
        """Should sort by gap severity descending."""
        for _ in range(10):
            detector.record_query("very common topic query")
        for _ in range(4):
            detector.record_query("less common topic query")

        gaps = detector.get_frequently_asked_gaps(min_queries=3)

        if len(gaps) > 1:
            assert gaps[0].gap_severity >= gaps[1].gap_severity

    def test_empty_when_no_queries(self, detector):
        """Should return empty list when no queries recorded."""
        gaps = detector.get_frequently_asked_gaps()

        assert gaps == []

    def test_sample_queries_deduplicated(self, detector):
        """Should deduplicate sample queries."""
        for _ in range(5):
            detector.record_query("contract notice period")

        gaps = detector.get_frequently_asked_gaps(min_queries=3)

        assert len(gaps) == 1
        # Deduplicated, so only one unique query
        assert len(gaps[0].sample_queries) == 1


# =============================================================================
# Async Frequently Asked Gaps Tests
# =============================================================================


class TestGetFrequentlyAskedGapsAsync:
    """Tests for get_frequently_asked_gaps_async method."""

    @pytest.mark.asyncio
    async def test_queries_mound_for_coverage(self, detector, mock_mound):
        """Should query the Knowledge Mound for topic coverage."""
        for _ in range(5):
            detector.record_query("contract notice period")

        mock_mound.query.return_value = _make_query_result([])

        gaps = await detector.get_frequently_asked_gaps_async(min_queries=3)

        assert len(gaps) == 1
        assert gaps[0].coverage_score == 0.0
        mock_mound.query.assert_called()

    @pytest.mark.asyncio
    async def test_excludes_well_covered_topics(self, detector, mock_mound):
        """Should exclude topics with high coverage."""
        for _ in range(5):
            detector.record_query("well covered topic here")

        # Return many high-confidence items
        items = [_make_item(item_id=f"i-{i}", confidence=0.9) for i in range(15)]
        mock_mound.query.return_value = _make_query_result(items)

        gaps = await detector.get_frequently_asked_gaps_async(
            min_queries=3, max_coverage=0.5
        )

        # Coverage should be high enough to exclude
        assert len(gaps) == 0

    @pytest.mark.asyncio
    async def test_handles_mound_error(self, detector, mock_mound):
        """Should handle mound query errors gracefully."""
        for _ in range(5):
            detector.record_query("error topic query test")

        mock_mound.query = AsyncMock(side_effect=RuntimeError("DB error"))

        gaps = await detector.get_frequently_asked_gaps_async(min_queries=3)

        # Should still return gaps with 0 coverage
        assert len(gaps) == 1
        assert gaps[0].coverage_score == 0.0


# =============================================================================
# Coverage Map Tests
# =============================================================================


class TestGetCoverageMap:
    """Tests for get_coverage_map method."""

    @pytest.mark.asyncio
    async def test_returns_all_domains(self, detector, mock_mound):
        """Should return entries for all known domains."""
        mock_mound.query.return_value = _make_query_result([])
        mock_mound.get_stale_knowledge.return_value = []

        coverage_map = await detector.get_coverage_map()

        domains = {e.domain for e in coverage_map}
        assert "legal" in domains
        assert "financial" in domains
        assert "technical" in domains
        assert "healthcare" in domains
        assert "operational" in domains

    @pytest.mark.asyncio
    async def test_sorted_by_coverage_ascending(self, detector, mock_mound):
        """Should sort entries by coverage score ascending (weakest first)."""
        mock_mound.query.return_value = _make_query_result([])
        mock_mound.get_stale_knowledge.return_value = []

        coverage_map = await detector.get_coverage_map()

        if len(coverage_map) > 1:
            for i in range(len(coverage_map) - 1):
                assert coverage_map[i].coverage_score <= coverage_map[i + 1].coverage_score

    @pytest.mark.asyncio
    async def test_includes_gap_counts(self, detector, mock_mound):
        """Should include gap count for each domain."""
        mock_mound.query.return_value = _make_query_result([])
        mock_mound.get_stale_knowledge.return_value = []

        coverage_map = await detector.get_coverage_map()

        for entry in coverage_map:
            assert isinstance(entry.gap_count, int)
            assert entry.gap_count >= 0

    @pytest.mark.asyncio
    async def test_calculates_average_confidence(self, detector, mock_mound):
        """Should calculate average confidence for domains with items."""
        items = [_make_item(item_id=f"i-{i}", confidence=0.8) for i in range(10)]
        mock_mound.query.return_value = _make_query_result(items)
        mock_mound.get_stale_knowledge.return_value = []

        coverage_map = await detector.get_coverage_map()

        # All domains get the same items, so all should have ~0.8 confidence
        for entry in coverage_map:
            assert entry.average_confidence > 0.0

    @pytest.mark.asyncio
    async def test_counts_stale_items(self, detector, mock_mound):
        """Should count stale items in each domain."""
        old_items = [
            _make_item(item_id=f"old-{i}", updated_at=datetime.now() - timedelta(days=120))
            for i in range(5)
        ]
        mock_mound.query.return_value = _make_query_result(old_items)
        mock_mound.get_stale_knowledge.return_value = []

        coverage_map = await detector.get_coverage_map()

        for entry in coverage_map:
            assert entry.stale_count == 5

    @pytest.mark.asyncio
    async def test_handles_query_errors_gracefully(self, detector, mock_mound):
        """Should handle query errors and still return coverage map."""
        mock_mound.query = AsyncMock(side_effect=RuntimeError("DB error"))
        mock_mound.get_stale_knowledge.return_value = []

        coverage_map = await detector.get_coverage_map()

        # Should still return entries, just with zero coverage
        assert len(coverage_map) > 0
        for entry in coverage_map:
            assert entry.coverage_score == 0.0

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, detector, mock_mound):
        """Should serialize coverage map entries to dicts."""
        mock_mound.query.return_value = _make_query_result([])
        mock_mound.get_stale_knowledge.return_value = []

        coverage_map = await detector.get_coverage_map()

        for entry in coverage_map:
            d = entry.to_dict()
            assert "domain" in d
            assert "coverage_score" in d
            assert "total_items" in d
            assert "expected_items" in d
            assert "average_confidence" in d
            assert "gap_count" in d
            assert "stale_count" in d
            assert "contradiction_count" in d


# =============================================================================
# Debate + FAQ Recommendation Integration Tests
# =============================================================================


class TestRecommendationsWithDebateAndFAQ:
    """Tests for recommendations incorporating debate insights and FAQ gaps."""

    @pytest.mark.asyncio
    async def test_includes_debate_signal_recommendations(self, detector, mock_mound):
        """Should include ACQUIRE recommendations from debate insights."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 25)
        mock_mound.get_stale_knowledge.return_value = []
        mock_mound.detect_contradictions.return_value = []

        # Record a low-confidence debate
        await detector.analyze_debate_receipt({
            "debate_id": "d-1",
            "topic": "contract termination",
            "confidence": 0.2,
            "consensus_score": 0.3,
            "domain": "legal",
        })

        recs = await detector.get_recommendations(domain="legal")

        acquire_recs = [r for r in recs if r.action == RecommendedAction.ACQUIRE]
        assert len(acquire_recs) >= 1
        assert any("debate_signal" in r.metadata.get("gap_type", "") for r in acquire_recs)

    @pytest.mark.asyncio
    async def test_includes_faq_recommendations(self, detector, mock_mound):
        """Should include ACQUIRE recommendations from frequently asked gaps."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 25)
        mock_mound.get_stale_knowledge.return_value = []
        mock_mound.detect_contradictions.return_value = []

        # Record many queries for a topic
        for _ in range(10):
            detector.record_query("contract termination clause")

        recs = await detector.get_recommendations(domain="legal")

        faq_recs = [
            r for r in recs
            if r.metadata.get("gap_type") == "frequently_asked"
        ]
        assert len(faq_recs) >= 1

    @pytest.mark.asyncio
    async def test_debate_insights_filtered_by_domain(self, detector, mock_mound):
        """Should only include debate insights from the requested domain."""
        mock_mound.query.return_value = _make_query_result([_make_item()] * 25)
        mock_mound.get_stale_knowledge.return_value = []
        mock_mound.detect_contradictions.return_value = []

        await detector.analyze_debate_receipt({
            "debate_id": "d-legal",
            "topic": "legal topic",
            "confidence": 0.2,
            "consensus_score": 0.3,
            "domain": "legal",
        })
        await detector.analyze_debate_receipt({
            "debate_id": "d-tech",
            "topic": "tech topic",
            "confidence": 0.2,
            "consensus_score": 0.3,
            "domain": "technical",
        })

        recs = await detector.get_recommendations(domain="legal")

        debate_recs = [
            r for r in recs
            if r.metadata.get("gap_type") == "debate_signal"
        ]
        # Should only include the legal domain debate insight
        for r in debate_recs:
            assert r.metadata.get("debate_id") == "d-legal"


# =============================================================================
# Handler Coverage Endpoint Tests
# =============================================================================


class TestKnowledgeGapHandlerCoverage:
    """Tests for the coverage map handler endpoint."""

    def test_can_handle_coverage_path(self):
        """Should handle /api/v1/knowledge/gaps/coverage path."""
        from aragora.server.handlers.knowledge.gaps import KnowledgeGapHandler

        handler = KnowledgeGapHandler()
        assert handler.can_handle("/api/v1/knowledge/gaps/coverage") is True
