"""
Tests for evidence quality scoring.

Tests the QualityScorer, QualityScores, QualityContext, QualityFilter,
and QualityTier components of the evidence system.
"""

import pytest
from datetime import datetime, timedelta

from aragora.evidence.quality import (
    QualityContext,
    QualityFilter,
    QualityScorer,
    QualityScores,
    QualityTier,
    score_evidence_snippet,
)
from aragora.evidence.metadata import (
    EnrichedMetadata,
    Provenance,
    SourceType,
)


# =============================================================================
# QualityTier Tests
# =============================================================================


class TestQualityTier:
    """Tests for QualityTier enum."""

    def test_quality_tier_values(self):
        """Test all quality tier values."""
        assert QualityTier.EXCELLENT.value == "excellent"
        assert QualityTier.GOOD.value == "good"
        assert QualityTier.FAIR.value == "fair"
        assert QualityTier.POOR.value == "poor"
        assert QualityTier.UNRELIABLE.value == "unreliable"

    def test_quality_tier_from_score_excellent(self):
        """Test excellent tier classification."""
        assert QualityTier.from_score(0.90) == QualityTier.EXCELLENT
        assert QualityTier.from_score(0.85) == QualityTier.EXCELLENT
        assert QualityTier.from_score(1.0) == QualityTier.EXCELLENT

    def test_quality_tier_from_score_good(self):
        """Test good tier classification."""
        assert QualityTier.from_score(0.70) == QualityTier.GOOD
        assert QualityTier.from_score(0.80) == QualityTier.GOOD
        assert QualityTier.from_score(0.84) == QualityTier.GOOD

    def test_quality_tier_from_score_fair(self):
        """Test fair tier classification."""
        assert QualityTier.from_score(0.50) == QualityTier.FAIR
        assert QualityTier.from_score(0.60) == QualityTier.FAIR
        assert QualityTier.from_score(0.69) == QualityTier.FAIR

    def test_quality_tier_from_score_poor(self):
        """Test poor tier classification."""
        assert QualityTier.from_score(0.30) == QualityTier.POOR
        assert QualityTier.from_score(0.40) == QualityTier.POOR
        assert QualityTier.from_score(0.49) == QualityTier.POOR

    def test_quality_tier_from_score_unreliable(self):
        """Test unreliable tier classification."""
        assert QualityTier.from_score(0.0) == QualityTier.UNRELIABLE
        assert QualityTier.from_score(0.20) == QualityTier.UNRELIABLE
        assert QualityTier.from_score(0.29) == QualityTier.UNRELIABLE


# =============================================================================
# QualityScores Tests
# =============================================================================


class TestQualityScores:
    """Tests for QualityScores dataclass."""

    def test_quality_scores_defaults(self):
        """Test QualityScores default values."""
        scores = QualityScores()
        assert scores.relevance_score == 0.5
        assert scores.freshness_score == 0.5
        assert scores.authority_score == 0.5
        assert scores.completeness_score == 0.5
        assert scores.consistency_score == 0.5

    def test_quality_scores_weights_defaults(self):
        """Test default weight values."""
        scores = QualityScores()
        assert scores.relevance_weight == 0.35
        assert scores.freshness_weight == 0.15
        assert scores.authority_weight == 0.30
        assert scores.completeness_weight == 0.10
        assert scores.consistency_weight == 0.10

    def test_quality_scores_weights_sum_to_one(self):
        """Test weights sum to 1.0."""
        scores = QualityScores()
        total = (
            scores.relevance_weight +
            scores.freshness_weight +
            scores.authority_weight +
            scores.completeness_weight +
            scores.consistency_weight
        )
        assert abs(total - 1.0) < 0.001

    def test_quality_scores_overall_score(self):
        """Test overall score calculation."""
        scores = QualityScores(
            relevance_score=1.0,
            freshness_score=1.0,
            authority_score=1.0,
            completeness_score=1.0,
            consistency_score=1.0,
        )
        assert abs(scores.overall_score - 1.0) < 0.001

    def test_quality_scores_overall_score_zero(self):
        """Test overall score with all zeros."""
        scores = QualityScores(
            relevance_score=0.0,
            freshness_score=0.0,
            authority_score=0.0,
            completeness_score=0.0,
            consistency_score=0.0,
        )
        assert scores.overall_score == 0.0

    def test_quality_scores_overall_score_mixed(self):
        """Test overall score with mixed values."""
        scores = QualityScores(
            relevance_score=0.8,
            freshness_score=0.6,
            authority_score=0.9,
            completeness_score=0.7,
            consistency_score=0.5,
        )
        # Verify calculation
        expected = (
            0.8 * 0.35 +  # relevance
            0.6 * 0.15 +  # freshness
            0.9 * 0.30 +  # authority
            0.7 * 0.10 +  # completeness
            0.5 * 0.10    # consistency
        )
        assert abs(scores.overall_score - expected) < 0.001

    def test_quality_scores_quality_tier(self):
        """Test quality tier property."""
        excellent = QualityScores(
            relevance_score=0.9,
            freshness_score=0.9,
            authority_score=0.9,
            completeness_score=0.9,
            consistency_score=0.9,
        )
        assert excellent.quality_tier == QualityTier.EXCELLENT

        poor = QualityScores(
            relevance_score=0.3,
            freshness_score=0.3,
            authority_score=0.3,
            completeness_score=0.3,
            consistency_score=0.3,
        )
        assert poor.quality_tier == QualityTier.POOR

    def test_quality_scores_to_dict(self):
        """Test QualityScores serialization."""
        scores = QualityScores(
            relevance_score=0.8,
            authority_score=0.7,
        )
        data = scores.to_dict()
        assert data["relevance_score"] == 0.8
        assert data["authority_score"] == 0.7
        assert "overall_score" in data
        assert "quality_tier" in data
        assert "weights" in data

    def test_quality_scores_from_dict(self):
        """Test QualityScores deserialization."""
        data = {
            "relevance_score": 0.75,
            "freshness_score": 0.65,
            "authority_score": 0.85,
            "completeness_score": 0.55,
            "consistency_score": 0.60,
            "weights": {
                "relevance": 0.40,
                "freshness": 0.10,
                "authority": 0.30,
                "completeness": 0.10,
                "consistency": 0.10,
            },
        }
        scores = QualityScores.from_dict(data)
        assert scores.relevance_score == 0.75
        assert scores.freshness_score == 0.65
        assert scores.relevance_weight == 0.40


# =============================================================================
# QualityContext Tests
# =============================================================================


class TestQualityContext:
    """Tests for QualityContext dataclass."""

    def test_quality_context_defaults(self):
        """Test QualityContext default values."""
        ctx = QualityContext()
        assert ctx.query == ""
        assert ctx.keywords == []
        assert ctx.required_topics == set()
        assert ctx.preferred_sources == set()
        assert ctx.blocked_sources == set()
        assert ctx.max_age_days == 365
        assert ctx.min_word_count == 50
        assert ctx.require_citations is False

    def test_quality_context_with_values(self):
        """Test QualityContext with custom values."""
        ctx = QualityContext(
            query="machine learning",
            keywords=["AI", "neural", "network"],
            required_topics={"deep learning"},
            preferred_sources={"arxiv.org", "nature.com"},
            blocked_sources={"spam.com"},
            max_age_days=30,
            min_word_count=100,
            require_citations=True,
        )
        assert ctx.query == "machine learning"
        assert len(ctx.keywords) == 3
        assert "deep learning" in ctx.required_topics
        assert ctx.require_citations is True


# =============================================================================
# QualityScorer Tests
# =============================================================================


class TestQualityScorer:
    """Tests for QualityScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a QualityScorer instance."""
        return QualityScorer()

    def test_score_basic_content(self, scorer):
        """Test scoring basic content."""
        scores = scorer.score(content="This is test content for quality scoring.")
        assert isinstance(scores, QualityScores)
        assert 0 <= scores.relevance_score <= 1
        assert 0 <= scores.overall_score <= 1

    def test_score_with_context_query(self, scorer):
        """Test scoring with query context."""
        ctx = QualityContext(query="machine learning algorithms")
        content = "Machine learning algorithms are powerful tools for data analysis."
        scores = scorer.score(content=content, context=ctx)
        assert scores.relevance_score > 0.3  # Should match some of the query

    def test_score_with_context_keywords(self, scorer):
        """Test scoring with keyword context."""
        ctx = QualityContext(keywords=["python", "testing", "pytest"])
        content = "Python testing with pytest is essential for quality code."
        scores = scorer.score(content=content, context=ctx)
        assert scores.relevance_score > 0.5

    def test_score_no_relevance_match(self, scorer):
        """Test scoring with no relevance match."""
        ctx = QualityContext(
            query="quantum computing",
            keywords=["quantum", "qubit", "superposition"],
        )
        content = "This content is about cooking recipes and food preparation."
        scores = scorer.score(content=content, context=ctx)
        assert scores.relevance_score < 0.5

    def test_score_freshness_very_fresh(self, scorer):
        """Test freshness scoring for very fresh content."""
        now = datetime.now()
        scores = scorer.score(
            content="Test content",
            fetched_at=now,
        )
        assert scores.freshness_score >= 0.9

    def test_score_freshness_day_old(self, scorer):
        """Test freshness scoring for day-old content."""
        yesterday = datetime.now() - timedelta(days=1)
        scores = scorer.score(
            content="Test content",
            fetched_at=yesterday,
        )
        assert 0.8 <= scores.freshness_score <= 0.95

    def test_score_freshness_week_old(self, scorer):
        """Test freshness scoring for week-old content."""
        week_ago = datetime.now() - timedelta(days=7)
        scores = scorer.score(
            content="Test content",
            fetched_at=week_ago,
        )
        assert 0.7 <= scores.freshness_score <= 0.9

    def test_score_freshness_month_old(self, scorer):
        """Test freshness scoring for month-old content."""
        month_ago = datetime.now() - timedelta(days=30)
        scores = scorer.score(
            content="Test content",
            fetched_at=month_ago,
        )
        assert 0.6 <= scores.freshness_score <= 0.85

    def test_score_freshness_year_old(self, scorer):
        """Test freshness scoring for year-old content."""
        year_ago = datetime.now() - timedelta(days=365)
        scores = scorer.score(
            content="Test content",
            fetched_at=year_ago,
        )
        assert scores.freshness_score < 0.6

    def test_score_freshness_stale(self, scorer):
        """Test freshness scoring for stale content."""
        old = datetime.now() - timedelta(days=730)  # 2 years
        scores = scorer.score(
            content="Test content",
            fetched_at=old,
        )
        assert scores.freshness_score <= 0.4

    def test_score_freshness_no_date(self, scorer):
        """Test freshness scoring without date."""
        scores = scorer.score(content="Test content")
        assert scores.freshness_score == 0.5  # Neutral

    def test_score_authority_academic(self, scorer):
        """Test authority scoring for academic source."""
        meta = EnrichedMetadata(source_type=SourceType.ACADEMIC)
        scores = scorer.score(
            content="Academic research content.",
            metadata=meta,
        )
        assert scores.authority_score > 0.8

    def test_score_authority_documentation(self, scorer):
        """Test authority scoring for documentation source."""
        meta = EnrichedMetadata(source_type=SourceType.DOCUMENTATION)
        scores = scorer.score(
            content="Technical documentation.",
            metadata=meta,
        )
        assert scores.authority_score > 0.7

    def test_score_authority_social(self, scorer):
        """Test authority scoring for social source."""
        meta = EnrichedMetadata(source_type=SourceType.SOCIAL)
        scores = scorer.score(
            content="Social media post.",
            metadata=meta,
        )
        assert scores.authority_score < 0.5

    def test_score_authority_unknown(self, scorer):
        """Test authority scoring for unknown source."""
        meta = EnrichedMetadata(source_type=SourceType.UNKNOWN)
        scores = scorer.score(
            content="Unknown source content.",
            metadata=meta,
        )
        assert scores.authority_score < 0.5

    def test_score_authority_from_url(self, scorer):
        """Test authority scoring from URL domain."""
        scores = scorer.score(
            content="Content from authoritative source.",
            url="https://arxiv.org/abs/1234.5678",
        )
        assert scores.authority_score > 0.7

    def test_score_authority_preferred_source(self, scorer):
        """Test authority boost for preferred source."""
        ctx = QualityContext(preferred_sources={"trusted_source"})
        scores = scorer.score(
            content="Content from trusted source.",
            source="trusted_source",
            context=ctx,
        )
        base_scores = scorer.score(
            content="Content from trusted source.",
            source="other_source",
            context=ctx,
        )
        assert scores.authority_score >= base_scores.authority_score

    def test_score_authority_blocked_source(self, scorer):
        """Test authority penalty for blocked source."""
        ctx = QualityContext(blocked_sources={"spam_source"})
        scores = scorer.score(
            content="Content from spam source.",
            source="spam_source",
            context=ctx,
        )
        base_scores = scorer.score(
            content="Content from spam source.",
            source="other_source",
            context=ctx,
        )
        assert scores.authority_score < base_scores.authority_score

    def test_score_authority_peer_reviewed(self, scorer):
        """Test authority boost for peer-reviewed content."""
        meta = EnrichedMetadata(
            source_type=SourceType.ACADEMIC,
            provenance=Provenance(peer_reviewed=True),
        )
        scores = scorer.score(
            content="Peer-reviewed research.",
            metadata=meta,
        )
        assert scores.authority_score > 0.9

    def test_score_authority_doi(self, scorer):
        """Test authority boost for DOI."""
        meta_with_doi = EnrichedMetadata(
            source_type=SourceType.ACADEMIC,
            provenance=Provenance(doi="10.1234/test"),
        )
        meta_without_doi = EnrichedMetadata(
            source_type=SourceType.ACADEMIC,
        )
        scores_with = scorer.score(
            content="Content with DOI.",
            metadata=meta_with_doi,
        )
        scores_without = scorer.score(
            content="Content without DOI.",
            metadata=meta_without_doi,
        )
        # DOI should boost authority for same source type
        assert scores_with.authority_score >= scores_without.authority_score

    def test_score_completeness_short(self, scorer):
        """Test completeness scoring for short content."""
        ctx = QualityContext(min_word_count=50)
        scores = scorer.score(
            content="Very short.",
            context=ctx,
        )
        assert scores.completeness_score < 0.5

    def test_score_completeness_medium(self, scorer):
        """Test completeness scoring for medium content."""
        content = " ".join(["word"] * 200)  # 200 words
        scores = scorer.score(content=content)
        assert scores.completeness_score > 0.5

    def test_score_completeness_long(self, scorer):
        """Test completeness scoring for long content."""
        content = " ".join(["word"] * 1000)  # 1000 words
        scores = scorer.score(content=content)
        assert scores.completeness_score > 0.8

    def test_score_completeness_with_citations(self, scorer):
        """Test completeness boost with citations."""
        meta = EnrichedMetadata(has_citations=True)
        scores = scorer.score(
            content=" ".join(["word"] * 100),
            metadata=meta,
        )
        base_scores = scorer.score(content=" ".join(["word"] * 100))
        assert scores.completeness_score >= base_scores.completeness_score

    def test_score_completeness_with_data(self, scorer):
        """Test completeness boost with data."""
        meta = EnrichedMetadata(has_data=True)
        scores = scorer.score(
            content=" ".join(["word"] * 100),
            metadata=meta,
        )
        base_scores = scorer.score(content=" ".join(["word"] * 100))
        assert scores.completeness_score >= base_scores.completeness_score

    def test_score_completeness_citation_required(self, scorer):
        """Test completeness penalty when citations required but missing."""
        ctx = QualityContext(require_citations=True)
        meta = EnrichedMetadata(has_citations=False)
        scores = scorer.score(
            content=" ".join(["word"] * 200),
            metadata=meta,
            context=ctx,
        )
        # Without required citations, score should be penalized
        base_scores = scorer.score(
            content=" ".join(["word"] * 200),
            context=QualityContext(require_citations=False),
        )
        assert scores.completeness_score < base_scores.completeness_score

    def test_score_consistency_base(self, scorer):
        """Test base consistency scoring."""
        scores = scorer.score(content="Clear and consistent content.")
        assert scores.consistency_score >= 0.5

    def test_score_consistency_definitive(self, scorer):
        """Test consistency boost for definitive statements."""
        content = "This approach is definitely the best. It has been proven to work."
        scores = scorer.score(content=content)
        assert scores.consistency_score > 0.7

    def test_score_consistency_uncertain(self, scorer):
        """Test consistency penalty for uncertain language."""
        content = "Maybe this works. Perhaps it could be better. It's possibly correct. Unknown if it helps."
        scores = scorer.score(content=content)
        assert scores.consistency_score < 0.7

    def test_score_with_default_context(self):
        """Test scoring with default context."""
        ctx = QualityContext(query="test query")
        scorer = QualityScorer(default_context=ctx)
        scores = scorer.score(content="Content about test query.")
        assert scores.relevance_score > 0.3


# =============================================================================
# QualityFilter Tests
# =============================================================================


class TestQualityFilter:
    """Tests for QualityFilter class."""

    @pytest.fixture
    def filter(self):
        """Create a QualityFilter instance."""
        return QualityFilter(
            min_overall_score=0.5,
            min_relevance_score=0.3,
            min_authority_score=0.3,
        )

    @pytest.fixture
    def mock_evidence_list(self):
        """Create mock evidence snippets."""
        class MockEvidence:
            def __init__(self, snippet, url="", source="web", reliability=0.5):
                self.snippet = snippet
                self.url = url
                self.source = source
                self.reliability_score = reliability
                self.fetched_at = datetime.now()

        return [
            MockEvidence(
                "High quality academic content with citations [1].",
                url="https://arxiv.org/abs/123",
                source="academic",
                reliability=0.9,
            ),
            MockEvidence(
                "Medium quality web content.",
                url="https://example.com",
                source="web",
                reliability=0.6,
            ),
            MockEvidence(
                "Low.",  # Very short
                source="unknown",
                reliability=0.3,
            ),
        ]

    def test_filter_basic(self, filter, mock_evidence_list):
        """Test basic filtering."""
        passed = filter.filter(mock_evidence_list)
        assert isinstance(passed, list)
        # High quality should pass, very short may not
        assert len(passed) <= len(mock_evidence_list)

    def test_filter_with_context(self, filter, mock_evidence_list):
        """Test filtering with context."""
        ctx = QualityContext(query="academic content")
        passed = filter.filter(mock_evidence_list, context=ctx)
        assert isinstance(passed, list)

    def test_filter_all_pass(self, mock_evidence_list):
        """Test filter with low thresholds."""
        filter = QualityFilter(
            min_overall_score=0.0,
            min_relevance_score=0.0,
            min_authority_score=0.0,
        )
        passed = filter.filter(mock_evidence_list)
        assert len(passed) == len(mock_evidence_list)

    def test_filter_none_pass(self, mock_evidence_list):
        """Test filter with very high thresholds."""
        filter = QualityFilter(
            min_overall_score=0.99,
            min_relevance_score=0.99,
            min_authority_score=0.99,
        )
        passed = filter.filter(mock_evidence_list)
        assert len(passed) == 0

    def test_filter_empty_list(self, filter):
        """Test filtering empty list."""
        passed = filter.filter([])
        assert passed == []

    def test_rank_basic(self, filter, mock_evidence_list):
        """Test basic ranking."""
        ranked = filter.rank(mock_evidence_list)
        assert isinstance(ranked, list)
        assert len(ranked) == len(mock_evidence_list)
        # Each item is (evidence, scores) tuple
        assert all(len(item) == 2 for item in ranked)

    def test_rank_order(self, filter, mock_evidence_list):
        """Test ranking order (descending by overall score)."""
        ranked = filter.rank(mock_evidence_list)
        scores = [item[1].overall_score for item in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_with_top_k(self, filter, mock_evidence_list):
        """Test ranking with limit."""
        ranked = filter.rank(mock_evidence_list, top_k=1)
        assert len(ranked) == 1

    def test_rank_with_context(self, filter, mock_evidence_list):
        """Test ranking with context."""
        ctx = QualityContext(query="academic")
        ranked = filter.rank(mock_evidence_list, context=ctx)
        assert len(ranked) > 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestScoreEvidenceSnippet:
    """Tests for score_evidence_snippet convenience function."""

    def test_score_evidence_snippet_basic(self):
        """Test scoring a mock evidence snippet."""
        class MockSnippet:
            snippet = "Test content about machine learning."
            url = "https://docs.python.org/test"
            source = "documentation"
            fetched_at = datetime.now()

        scores = score_evidence_snippet(MockSnippet())
        assert isinstance(scores, QualityScores)
        assert 0 <= scores.overall_score <= 1

    def test_score_evidence_snippet_with_query(self):
        """Test scoring with query."""
        class MockSnippet:
            snippet = "Machine learning algorithms for data analysis."
            url = ""
            source = "web"
            fetched_at = datetime.now()

        scores = score_evidence_snippet(
            MockSnippet(),
            query="machine learning",
        )
        assert scores.relevance_score > 0.3

    def test_score_evidence_snippet_with_keywords(self):
        """Test scoring with keywords."""
        class MockSnippet:
            snippet = "Python testing with pytest and unittest."
            url = ""
            source = "web"
            fetched_at = datetime.now()

        scores = score_evidence_snippet(
            MockSnippet(),
            keywords=["python", "testing", "pytest"],
        )
        assert scores.relevance_score > 0.5

    def test_score_evidence_snippet_with_scorer(self):
        """Test scoring with custom scorer."""
        class MockSnippet:
            snippet = "Custom scorer test content."
            url = ""
            source = "web"
            fetched_at = datetime.now()

        scorer = QualityScorer()
        scores = score_evidence_snippet(MockSnippet(), scorer=scorer)
        assert isinstance(scores, QualityScores)
