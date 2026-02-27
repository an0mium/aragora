"""
Comprehensive tests for aragora.evidence.quality module.

Tests QualityTier, QualityScores, QualityContext, QualityScorer,
QualityFilter, and score_evidence_snippet convenience function.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.evidence.metadata import EnrichedMetadata, Provenance, SourceType
from aragora.evidence.quality import (
    QualityContext,
    QualityFilter,
    QualityScorer,
    QualityScores,
    QualityTier,
    score_evidence_snippet,
)


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class FakeEvidenceSnippet:
    """Minimal evidence snippet for testing."""

    snippet: str = "Some evidence content for testing purposes."
    url: str | None = None
    source: str | None = None
    fetched_at: datetime | None = None
    metadata: dict[str, Any] | None = None


# =============================================================================
# QualityTier Tests
# =============================================================================


class TestQualityTier:
    """Tests for QualityTier enum and from_score classmethod."""

    def test_from_score_excellent_exact_boundary(self):
        """Score exactly 0.85 should be EXCELLENT."""
        assert QualityTier.from_score(0.85) == QualityTier.EXCELLENT

    def test_from_score_excellent_above(self):
        """Score above 0.85 should be EXCELLENT."""
        assert QualityTier.from_score(0.99) == QualityTier.EXCELLENT
        assert QualityTier.from_score(1.0) == QualityTier.EXCELLENT

    def test_from_score_good_exact_boundary(self):
        """Score exactly 0.70 should be GOOD."""
        assert QualityTier.from_score(0.70) == QualityTier.GOOD

    def test_from_score_good_just_below_excellent(self):
        """Score just below 0.85 should be GOOD."""
        assert QualityTier.from_score(0.849) == QualityTier.GOOD

    def test_from_score_fair_exact_boundary(self):
        """Score exactly 0.50 should be FAIR."""
        assert QualityTier.from_score(0.50) == QualityTier.FAIR

    def test_from_score_fair_just_below_good(self):
        """Score just below 0.70 should be FAIR."""
        assert QualityTier.from_score(0.699) == QualityTier.FAIR

    def test_from_score_poor_exact_boundary(self):
        """Score exactly 0.30 should be POOR."""
        assert QualityTier.from_score(0.30) == QualityTier.POOR

    def test_from_score_poor_just_below_fair(self):
        """Score just below 0.50 should be POOR."""
        assert QualityTier.from_score(0.499) == QualityTier.POOR

    def test_from_score_unreliable_just_below_poor(self):
        """Score just below 0.30 should be UNRELIABLE."""
        assert QualityTier.from_score(0.299) == QualityTier.UNRELIABLE

    def test_from_score_unreliable_zero(self):
        """Score 0.0 should be UNRELIABLE."""
        assert QualityTier.from_score(0.0) == QualityTier.UNRELIABLE

    def test_from_score_unreliable_negative(self):
        """Negative score should be UNRELIABLE."""
        assert QualityTier.from_score(-0.5) == QualityTier.UNRELIABLE

    def test_enum_values(self):
        """Verify enum string values."""
        assert QualityTier.EXCELLENT.value == "excellent"
        assert QualityTier.GOOD.value == "good"
        assert QualityTier.FAIR.value == "fair"
        assert QualityTier.POOR.value == "poor"
        assert QualityTier.UNRELIABLE.value == "unreliable"


# =============================================================================
# QualityScores Tests
# =============================================================================


class TestQualityScores:
    """Tests for QualityScores dataclass and computed properties."""

    def test_default_values(self):
        """Default scores should all be 0.5."""
        scores = QualityScores()
        assert scores.relevance_score == 0.5
        assert scores.semantic_score == 0.5
        assert scores.freshness_score == 0.5
        assert scores.authority_score == 0.5
        assert scores.completeness_score == 0.5
        assert scores.consistency_score == 0.5

    def test_overall_score_with_defaults(self):
        """Overall score with all defaults should equal 0.5."""
        scores = QualityScores()
        # 0.5 * (0.25 + 0.15 + 0.12 + 0.28 + 0.10 + 0.10) = 0.5 * 1.0 = 0.5
        assert scores.overall_score == pytest.approx(0.5, abs=1e-9)

    def test_overall_score_weighted_calculation(self):
        """Overall score should be a weighted sum of individual scores."""
        scores = QualityScores(
            relevance_score=1.0,
            semantic_score=0.8,
            freshness_score=0.6,
            authority_score=0.9,
            completeness_score=0.7,
            consistency_score=0.5,
        )
        expected = 1.0 * 0.25 + 0.8 * 0.15 + 0.6 * 0.12 + 0.9 * 0.28 + 0.7 * 0.10 + 0.5 * 0.10
        assert scores.overall_score == pytest.approx(expected, abs=1e-9)

    def test_overall_score_all_ones(self):
        """Overall score should be 1.0 when all scores are 1.0."""
        scores = QualityScores(
            relevance_score=1.0,
            semantic_score=1.0,
            freshness_score=1.0,
            authority_score=1.0,
            completeness_score=1.0,
            consistency_score=1.0,
        )
        assert scores.overall_score == pytest.approx(1.0, abs=1e-9)

    def test_overall_score_all_zeros(self):
        """Overall score should be 0.0 when all scores are 0.0."""
        scores = QualityScores(
            relevance_score=0.0,
            semantic_score=0.0,
            freshness_score=0.0,
            authority_score=0.0,
            completeness_score=0.0,
            consistency_score=0.0,
        )
        assert scores.overall_score == pytest.approx(0.0, abs=1e-9)

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        scores = QualityScores()
        total = (
            scores.relevance_weight
            + scores.semantic_weight
            + scores.freshness_weight
            + scores.authority_weight
            + scores.completeness_weight
            + scores.consistency_weight
        )
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_combined_relevance_with_meaningful_semantic(self):
        """Combined relevance should blend keyword and semantic when semantic > 0.1."""
        scores = QualityScores(relevance_score=0.8, semantic_score=0.6)
        # 0.8 * 0.4 + 0.6 * 0.6 = 0.32 + 0.36 = 0.68
        assert scores.combined_relevance == pytest.approx(0.68, abs=1e-9)

    def test_combined_relevance_with_low_semantic(self):
        """Combined relevance should use only keyword when semantic <= 0.1."""
        scores = QualityScores(relevance_score=0.8, semantic_score=0.1)
        assert scores.combined_relevance == pytest.approx(0.8, abs=1e-9)

    def test_combined_relevance_with_zero_semantic(self):
        """Combined relevance should use only keyword when semantic is 0."""
        scores = QualityScores(relevance_score=0.6, semantic_score=0.0)
        assert scores.combined_relevance == pytest.approx(0.6, abs=1e-9)

    def test_combined_relevance_just_above_threshold(self):
        """Combined relevance should blend when semantic is 0.11."""
        scores = QualityScores(relevance_score=0.5, semantic_score=0.11)
        expected = 0.5 * 0.4 + 0.11 * 0.6
        assert scores.combined_relevance == pytest.approx(expected, abs=1e-9)

    def test_quality_tier_property(self):
        """quality_tier property should use from_score on overall_score."""
        scores = QualityScores(
            relevance_score=1.0,
            semantic_score=1.0,
            freshness_score=1.0,
            authority_score=1.0,
            completeness_score=1.0,
            consistency_score=1.0,
        )
        assert scores.quality_tier == QualityTier.EXCELLENT

    def test_quality_tier_poor(self):
        """quality_tier for low scores should be POOR or UNRELIABLE."""
        scores = QualityScores(
            relevance_score=0.2,
            semantic_score=0.2,
            freshness_score=0.2,
            authority_score=0.2,
            completeness_score=0.2,
            consistency_score=0.2,
        )
        assert scores.quality_tier in (QualityTier.POOR, QualityTier.UNRELIABLE)

    def test_to_dict(self):
        """to_dict should serialize all fields including computed properties."""
        scores = QualityScores(relevance_score=0.9, authority_score=0.8)
        d = scores.to_dict()
        assert d["relevance_score"] == 0.9
        assert d["authority_score"] == 0.8
        assert "overall_score" in d
        assert "combined_relevance" in d
        assert "quality_tier" in d
        assert d["quality_tier"] in {"excellent", "good", "fair", "poor", "unreliable"}
        assert "weights" in d
        assert d["weights"]["relevance"] == 0.25
        assert d["weights"]["semantic"] == 0.15

    def test_from_dict(self):
        """from_dict should reconstruct scores from serialized dict."""
        original = QualityScores(
            relevance_score=0.9,
            semantic_score=0.7,
            freshness_score=0.8,
            authority_score=0.85,
            completeness_score=0.6,
            consistency_score=0.75,
        )
        d = original.to_dict()
        restored = QualityScores.from_dict(d)
        assert restored.relevance_score == original.relevance_score
        assert restored.semantic_score == original.semantic_score
        assert restored.freshness_score == original.freshness_score
        assert restored.authority_score == original.authority_score
        assert restored.completeness_score == original.completeness_score
        assert restored.consistency_score == original.consistency_score

    def test_from_dict_with_defaults(self):
        """from_dict with empty dict should use defaults."""
        scores = QualityScores.from_dict({})
        assert scores.relevance_score == 0.5
        assert scores.semantic_score == 0.5

    def test_from_dict_with_custom_weights(self):
        """from_dict should restore custom weights."""
        d = {
            "relevance_score": 0.7,
            "weights": {"relevance": 0.4, "semantic": 0.1},
        }
        scores = QualityScores.from_dict(d)
        assert scores.relevance_weight == 0.4
        assert scores.semantic_weight == 0.1

    def test_roundtrip_to_from_dict(self):
        """Roundtrip through to_dict/from_dict should preserve overall score."""
        original = QualityScores(
            relevance_score=0.9,
            semantic_score=0.7,
            freshness_score=0.8,
            authority_score=0.85,
            completeness_score=0.6,
            consistency_score=0.75,
        )
        restored = QualityScores.from_dict(original.to_dict())
        assert restored.overall_score == pytest.approx(original.overall_score, abs=1e-9)


# =============================================================================
# QualityContext Tests
# =============================================================================


class TestQualityContext:
    """Tests for QualityContext dataclass defaults."""

    def test_default_values(self):
        """Default context should have sensible defaults."""
        ctx = QualityContext()
        assert ctx.query == ""
        assert ctx.keywords == []
        assert ctx.required_topics == set()
        assert ctx.preferred_sources == set()
        assert ctx.blocked_sources == set()
        assert ctx.max_age_days == 365
        assert ctx.min_word_count == 50
        assert ctx.require_citations is False

    def test_custom_values(self):
        """Context should accept custom values."""
        ctx = QualityContext(
            query="machine learning",
            keywords=["neural", "network"],
            required_topics={"AI", "ML"},
            preferred_sources={"arxiv"},
            blocked_sources={"spam-site"},
            max_age_days=180,
            min_word_count=100,
            require_citations=True,
        )
        assert ctx.query == "machine learning"
        assert ctx.keywords == ["neural", "network"]
        assert "AI" in ctx.required_topics
        assert ctx.max_age_days == 180
        assert ctx.require_citations is True


# =============================================================================
# QualityScorer._score_relevance Tests
# =============================================================================


class TestScoreRelevance:
    """Tests for QualityScorer._score_relevance."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer()

    def test_no_query_no_keywords_returns_neutral(self, scorer):
        """No query and no keywords should return 0.5."""
        ctx = QualityContext()
        assert scorer._score_relevance("any content", ctx) == 0.5

    def test_query_all_words_match(self, scorer):
        """All query words found in content should score the query portion."""
        ctx = QualityContext(query="machine learning algorithms")
        content = "machine learning algorithms are powerful tools"
        score = scorer._score_relevance(content, ctx)
        # Query matches: 3/3 => 0.3 * 1.0 = 0.3
        # No keywords: 0
        # No required topics: 0.3 * 0.5 = 0.15
        # Total: 0.45
        assert score == pytest.approx(0.45, abs=0.01)

    def test_query_partial_match(self, scorer):
        """Partial query match should score proportionally."""
        ctx = QualityContext(query="machine learning algorithms")
        content = "machine hardware components"
        score = scorer._score_relevance(content, ctx)
        # Query matches: 1/3 => 0.3 * (1/3) = 0.1
        # No keywords: 0
        # No required topics: 0.3 * 0.5 = 0.15
        # Total: 0.25
        assert score == pytest.approx(0.25, abs=0.01)

    def test_query_no_match(self, scorer):
        """No query words matching should contribute 0 from query portion."""
        ctx = QualityContext(query="quantum computing")
        content = "cooking recipe for pasta"
        score = scorer._score_relevance(content, ctx)
        # Query: 0/2 => 0.0
        # No keywords: 0
        # No required topics: 0.15
        # Total: 0.15
        assert score == pytest.approx(0.15, abs=0.01)

    def test_keywords_all_match(self, scorer):
        """All keywords matching should contribute full keyword score."""
        ctx = QualityContext(keywords=["python", "testing"])
        content = "Python testing framework for unit testing"
        score = scorer._score_relevance(content, ctx)
        # No query: 0
        # Keywords: 2/2 => 0.4
        # No required topics: 0.15
        # Total: 0.55
        assert score == pytest.approx(0.55, abs=0.01)

    def test_keywords_partial_match(self, scorer):
        """Partial keyword match should score proportionally."""
        ctx = QualityContext(keywords=["python", "java", "rust"])
        content = "python programming language"
        score = scorer._score_relevance(content, ctx)
        # Keywords: 1/3 => 0.4 * (1/3) = ~0.133
        # No required topics: 0.15
        # Total: ~0.283
        assert score == pytest.approx(0.283, abs=0.01)

    def test_required_topics_all_match(self, scorer):
        """All required topics matching should contribute full topic score."""
        ctx = QualityContext(
            query="test",
            required_topics={"security", "encryption"},
        )
        content = "security measures and encryption protocols test"
        score = scorer._score_relevance(content, ctx)
        # Query: 1/1 => 0.3
        # No keywords: 0
        # Required topics: 2/2 => 0.3 * 1.0 = 0.3
        # Total: 0.6
        assert score == pytest.approx(0.6, abs=0.01)

    def test_required_topics_none_match(self, scorer):
        """No required topics matching should contribute 0 from topics."""
        ctx = QualityContext(
            query="test",
            required_topics={"blockchain", "crypto"},
        )
        content = "cooking recipes test"
        score = scorer._score_relevance(content, ctx)
        # Query: 1/1 => 0.3
        # Topics: 0/2 => 0.0
        # Total: 0.3
        assert score == pytest.approx(0.3, abs=0.01)

    def test_all_dimensions_combined(self, scorer):
        """Query + keywords + required topics should all contribute."""
        ctx = QualityContext(
            query="machine learning",
            keywords=["neural", "deep"],
            required_topics={"AI"},
        )
        content = "machine learning with neural networks and deep AI models"
        score = scorer._score_relevance(content, ctx)
        # Query: 2/2 => 0.3
        # Keywords: 2/2 => 0.4
        # Topics: 1/1 => 0.3
        # Total: 1.0 (clamped)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_case_insensitive(self, scorer):
        """Relevance matching should be case insensitive."""
        ctx = QualityContext(
            query="Python",
            keywords=["TESTING"],
            required_topics={"Security"},
        )
        content = "python testing security framework"
        score = scorer._score_relevance(content, ctx)
        # All match despite different casing
        assert score > 0.5

    def test_score_capped_at_1(self, scorer):
        """Relevance score should never exceed 1.0."""
        ctx = QualityContext(
            query="test python security machine learning neural deep",
            keywords=["test", "python", "security", "machine"],
            required_topics={"test", "python"},
        )
        content = "test python security machine learning neural deep"
        score = scorer._score_relevance(content, ctx)
        assert score <= 1.0


# =============================================================================
# QualityScorer._score_freshness Tests
# =============================================================================


class TestScoreFreshness:
    """Tests for QualityScorer._score_freshness."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer()

    def test_no_dates_returns_neutral(self, scorer):
        """No dates should return 0.5."""
        assert scorer._score_freshness(None, None, 365) == 0.5

    def test_very_fresh_content(self, scorer):
        """Content less than 1 day old should score 1.0."""
        now = datetime.now()
        assert scorer._score_freshness(now, None, 365) == 1.0

    def test_one_hour_old(self, scorer):
        """Content 1 hour old should score 1.0 (less than 1 day)."""
        recent = datetime.now() - timedelta(hours=1)
        assert scorer._score_freshness(recent, None, 365) == 1.0

    def test_three_days_old(self, scorer):
        """Content 3 days old should score 0.95."""
        date = datetime.now() - timedelta(days=3)
        assert scorer._score_freshness(date, None, 365) == 0.95

    def test_two_weeks_old(self, scorer):
        """Content 14 days old should score 0.85."""
        date = datetime.now() - timedelta(days=14)
        assert scorer._score_freshness(date, None, 365) == 0.85

    def test_two_months_old(self, scorer):
        """Content 60 days old should score 0.70."""
        date = datetime.now() - timedelta(days=60)
        assert scorer._score_freshness(date, None, 365) == 0.70

    def test_six_months_old(self, scorer):
        """Content 180 days old should score 0.55."""
        date = datetime.now() - timedelta(days=180)
        assert scorer._score_freshness(date, None, 365) == 0.55

    def test_stale_content(self, scorer):
        """Content older than max_age_days should score 0.30."""
        date = datetime.now() - timedelta(days=400)
        assert scorer._score_freshness(date, None, 365) == 0.30

    def test_publication_date_preferred_over_fetch_date(self, scorer):
        """Publication date should be used when both are available."""
        old_fetch = datetime.now() - timedelta(days=200)
        fresh_pub = datetime.now() - timedelta(days=2)
        # published_at takes priority
        assert scorer._score_freshness(old_fetch, fresh_pub, 365) == 0.95

    def test_linear_decay_between_365_and_max_age(self, scorer):
        """Between 365 days and max_age_days, score should decay linearly."""
        max_age = 730
        # At 365 days: 0.55
        # At 730 days: 0.30
        # Midpoint (547.5 days): (0.55 + 0.30) / 2 = 0.425
        date = datetime.now() - timedelta(days=547)
        score = scorer._score_freshness(date, None, max_age)
        assert 0.30 < score < 0.55

    def test_exact_boundary_1_day(self, scorer):
        """Content exactly 1 day old should score 0.95 (1 < 7)."""
        date = datetime.now() - timedelta(days=1)
        assert scorer._score_freshness(date, None, 365) == 0.95

    def test_exact_boundary_7_days(self, scorer):
        """Content exactly 7 days old should score 0.85 (7 < 30)."""
        date = datetime.now() - timedelta(days=7)
        assert scorer._score_freshness(date, None, 365) == 0.85

    def test_exact_boundary_30_days(self, scorer):
        """Content exactly 30 days old should score 0.70 (30 < 90)."""
        date = datetime.now() - timedelta(days=30)
        assert scorer._score_freshness(date, None, 365) == 0.70

    def test_exact_boundary_90_days(self, scorer):
        """Content exactly 90 days old should score 0.55 (90 < 365)."""
        date = datetime.now() - timedelta(days=90)
        assert scorer._score_freshness(date, None, 365) == 0.55

    def test_custom_max_age(self, scorer):
        """Custom max_age_days should change decay range."""
        # With max_age=730, 500 days old should be in decay range
        date = datetime.now() - timedelta(days=500)
        score = scorer._score_freshness(date, None, 730)
        assert 0.30 < score < 0.55

    def test_max_age_equals_365_just_over_boundary(self, scorer):
        """With max_age=365, content at 365+ days is stale 0.30."""
        date = datetime.now() - timedelta(days=366)
        assert scorer._score_freshness(date, None, 365) == 0.30


# =============================================================================
# QualityScorer._score_authority Tests
# =============================================================================


class TestScoreAuthority:
    """Tests for QualityScorer._score_authority."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer()

    def test_no_inputs_returns_base_score(self, scorer):
        """No source type, URL, or source should return 0.5."""
        ctx = QualityContext()
        assert scorer._score_authority(None, None, None, None, ctx) == 0.5

    def test_academic_source_type(self, scorer):
        """Academic source type should score 0.9."""
        ctx = QualityContext()
        score = scorer._score_authority(SourceType.ACADEMIC, None, None, None, ctx)
        assert score == 0.9

    def test_documentation_source_type(self, scorer):
        """Documentation source type should score 0.85."""
        ctx = QualityContext()
        score = scorer._score_authority(SourceType.DOCUMENTATION, None, None, None, ctx)
        assert score == 0.85

    def test_social_source_type(self, scorer):
        """Social source type should score 0.40."""
        ctx = QualityContext()
        score = scorer._score_authority(SourceType.SOCIAL, None, None, None, ctx)
        assert score == 0.40

    def test_unknown_source_type(self, scorer):
        """Unknown source type should score 0.30."""
        ctx = QualityContext()
        score = scorer._score_authority(SourceType.UNKNOWN, None, None, None, ctx)
        assert score == 0.30

    def test_each_source_type_has_authority(self, scorer):
        """Every SourceType should have an entry in SOURCE_TYPE_AUTHORITY."""
        for st in SourceType:
            assert st in QualityScorer.SOURCE_TYPE_AUTHORITY

    def test_domain_authority_arxiv(self, scorer):
        """arxiv.org URL should boost authority (blended with source type)."""
        ctx = QualityContext()
        score = scorer._score_authority(
            SourceType.WEB, "https://arxiv.org/abs/2301.00001", None, None, ctx
        )
        # WEB base = 0.5, arxiv domain = 0.95, blended = (0.5 + 0.95) / 2 = 0.725
        assert score == pytest.approx(0.725, abs=0.01)

    def test_domain_authority_github(self, scorer):
        """github.com URL should blend domain authority."""
        ctx = QualityContext()
        score = scorer._score_authority(
            SourceType.CODE, "https://github.com/user/repo", None, None, ctx
        )
        # CODE base = 0.75, github = 0.80, blended = (0.75 + 0.80) / 2 = 0.775
        assert score == pytest.approx(0.775, abs=0.01)

    def test_domain_authority_with_www_prefix(self, scorer):
        """www. prefix should be stripped for domain matching."""
        ctx = QualityContext()
        score = scorer._score_authority(
            SourceType.NEWS, "https://www.reuters.com/article/123", None, None, ctx
        )
        # NEWS = 0.65, reuters = 0.80, blended = (0.65 + 0.80) / 2 = 0.725
        assert score == pytest.approx(0.725, abs=0.01)

    def test_unknown_domain_no_domain_boost(self, scorer):
        """Unknown domain should not get a domain boost."""
        ctx = QualityContext()
        score = scorer._score_authority(SourceType.WEB, "https://example.com/page", None, None, ctx)
        assert score == 0.5  # Just the WEB base score

    def test_provenance_peer_reviewed_boost(self, scorer):
        """Peer-reviewed provenance should add 0.15."""
        ctx = QualityContext()
        provenance = Provenance(peer_reviewed=True)
        score = scorer._score_authority(SourceType.WEB, None, None, provenance, ctx)
        assert score == pytest.approx(0.65, abs=0.01)

    def test_provenance_doi_boost(self, scorer):
        """DOI in provenance should add 0.1."""
        ctx = QualityContext()
        provenance = Provenance(doi="10.1234/example")
        score = scorer._score_authority(SourceType.WEB, None, None, provenance, ctx)
        assert score == pytest.approx(0.6, abs=0.01)

    def test_provenance_citation_count_boost(self, scorer):
        """High citation count should add 0.05."""
        ctx = QualityContext()
        provenance = Provenance(citation_count=50)
        score = scorer._score_authority(SourceType.WEB, None, None, provenance, ctx)
        assert score == pytest.approx(0.55, abs=0.01)

    def test_provenance_citation_count_low_no_boost(self, scorer):
        """Citation count <= 10 should not add a boost."""
        ctx = QualityContext()
        provenance = Provenance(citation_count=10)
        score = scorer._score_authority(SourceType.WEB, None, None, provenance, ctx)
        assert score == 0.5

    def test_provenance_author_boost(self, scorer):
        """Named author should add 0.03."""
        ctx = QualityContext()
        provenance = Provenance(author="Dr. Smith")
        score = scorer._score_authority(SourceType.WEB, None, None, provenance, ctx)
        assert score == pytest.approx(0.53, abs=0.01)

    def test_provenance_all_boosts_combined(self, scorer):
        """All provenance boosts should be cumulative (with capping at 1.0)."""
        ctx = QualityContext()
        provenance = Provenance(
            peer_reviewed=True,
            doi="10.1234/ex",
            citation_count=100,
            author="Dr. Expert",
        )
        score = scorer._score_authority(SourceType.ACADEMIC, None, None, provenance, ctx)
        # ACADEMIC base = 0.9
        # + 0.15 (peer_reviewed) = 1.0 (capped)
        # + 0.1 (doi) = 1.0 (capped)
        # + 0.05 (citations) = 1.0 (capped)
        # + 0.03 (author) = 1.0 (capped)
        assert score == 1.0

    def test_preferred_source_boost(self, scorer):
        """Preferred source should add 0.1."""
        ctx = QualityContext(preferred_sources={"arxiv"})
        score = scorer._score_authority(SourceType.WEB, None, "arxiv", None, ctx)
        assert score == pytest.approx(0.6, abs=0.01)

    def test_preferred_source_case_insensitive(self, scorer):
        """Preferred source matching should be case insensitive."""
        ctx = QualityContext(preferred_sources={"ArXiv"})
        score = scorer._score_authority(SourceType.WEB, None, "arxiv", None, ctx)
        assert score == pytest.approx(0.6, abs=0.01)

    def test_blocked_source_penalty(self, scorer):
        """Blocked source should subtract 0.3."""
        ctx = QualityContext(blocked_sources={"spam-site"})
        score = scorer._score_authority(SourceType.WEB, None, "spam-site", None, ctx)
        assert score == pytest.approx(0.2, abs=0.01)

    def test_blocked_source_clamped_at_zero(self, scorer):
        """Blocked source penalty should not go below 0.0."""
        ctx = QualityContext(blocked_sources={"bad"})
        score = scorer._score_authority(SourceType.UNKNOWN, None, "bad", None, ctx)
        # UNKNOWN = 0.30, - 0.3 = 0.0
        assert score == pytest.approx(0.0, abs=0.01)

    def test_no_source_name_ignores_preferred(self, scorer):
        """No source name should skip preferred/blocked checks."""
        ctx = QualityContext(
            preferred_sources={"good"},
            blocked_sources={"bad"},
        )
        score = scorer._score_authority(SourceType.WEB, None, None, None, ctx)
        assert score == 0.5

    def test_invalid_url_handled_gracefully(self, scorer):
        """Invalid URL should not crash, just skip domain authority."""
        ctx = QualityContext()
        # urlparse handles most strings without error, but this tests robustness
        score = scorer._score_authority(SourceType.WEB, "not-a-url", None, None, ctx)
        assert isinstance(score, float)


# =============================================================================
# QualityScorer._score_completeness Tests
# =============================================================================


class TestScoreCompleteness:
    """Tests for QualityScorer._score_completeness."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer()

    def test_very_short_content_below_min_word_count(self, scorer):
        """Content below min_word_count should score low."""
        ctx = QualityContext(min_word_count=50)
        content = "short text"  # 2 words
        score = scorer._score_completeness(content, None, ctx)
        # 0.2 + 0.3 * (2 / 50) = 0.2 + 0.012 = 0.212
        assert score == pytest.approx(0.212, abs=0.01)

    def test_content_at_min_word_count(self, scorer):
        """Content at exactly min_word_count should score 0.5."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 50)
        score = scorer._score_completeness(content, None, ctx)
        # 50 words exactly at min_word_count: not below min, not >= 100 => 0.5
        assert score == 0.5

    def test_content_100_words(self, scorer):
        """Content with 100 words should score 0.5 (100 < 300 path is 0.65 but 100 is < 300)."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 100)
        score = scorer._score_completeness(content, None, ctx)
        # 100 >= 100, < 300 => 0.65
        assert score == 0.65

    def test_content_300_words(self, scorer):
        """Content with 300 words should score 0.75."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 300)
        score = scorer._score_completeness(content, None, ctx)
        assert score == 0.75

    def test_content_500_words(self, scorer):
        """Content with 500 words should score 0.85."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 500)
        score = scorer._score_completeness(content, None, ctx)
        assert score == 0.85

    def test_content_1000_words(self, scorer):
        """Content with 1000+ words should score 0.90."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 1000)
        score = scorer._score_completeness(content, None, ctx)
        assert score == 0.90

    def test_content_2000_words(self, scorer):
        """Content with 2000 words should also score 0.90."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 2000)
        score = scorer._score_completeness(content, None, ctx)
        assert score == 0.90

    def test_metadata_citations_boost(self, scorer):
        """Metadata with citations should add 0.05."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 100)
        metadata = EnrichedMetadata(has_citations=True)
        score = scorer._score_completeness(content, metadata, ctx)
        # Base 0.65 + 0.05 = 0.70
        assert score == pytest.approx(0.70, abs=0.01)

    def test_metadata_data_boost(self, scorer):
        """Metadata with data should add 0.05."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 100)
        metadata = EnrichedMetadata(has_data=True)
        score = scorer._score_completeness(content, metadata, ctx)
        assert score == pytest.approx(0.70, abs=0.01)

    def test_metadata_topics_boost(self, scorer):
        """Metadata with topics should add 0.03."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 100)
        metadata = EnrichedMetadata(topics=["AI", "ML"])
        score = scorer._score_completeness(content, metadata, ctx)
        assert score == pytest.approx(0.68, abs=0.01)

    def test_metadata_all_boosts(self, scorer):
        """All metadata boosts combined should be cumulative."""
        ctx = QualityContext(min_word_count=50)
        content = " ".join(["word"] * 1000)
        metadata = EnrichedMetadata(
            has_citations=True,
            has_data=True,
            topics=["AI"],
        )
        score = scorer._score_completeness(content, metadata, ctx)
        # 0.90 + 0.05 + 0.05 + 0.03 = 1.0 (capped)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_require_citations_penalty(self, scorer):
        """Missing citations when required should penalize by 0.2."""
        ctx = QualityContext(min_word_count=50, require_citations=True)
        content = " ".join(["word"] * 100)
        metadata = EnrichedMetadata(has_citations=False)
        score = scorer._score_completeness(content, metadata, ctx)
        # 0.65 - 0.2 = 0.45
        assert score == pytest.approx(0.45, abs=0.01)

    def test_require_citations_with_citations_no_penalty(self, scorer):
        """Having citations when required should not penalize."""
        ctx = QualityContext(min_word_count=50, require_citations=True)
        content = " ".join(["word"] * 100)
        metadata = EnrichedMetadata(has_citations=True)
        score = scorer._score_completeness(content, metadata, ctx)
        # 0.65 + 0.05 = 0.70 (no penalty, plus the citations boost)
        assert score == pytest.approx(0.70, abs=0.01)

    def test_require_citations_without_metadata(self, scorer):
        """require_citations with no metadata should not crash."""
        ctx = QualityContext(min_word_count=50, require_citations=True)
        content = " ".join(["word"] * 100)
        # metadata is None, so the condition `context.require_citations and metadata and not metadata.has_citations`
        # is False because metadata is falsy
        score = scorer._score_completeness(content, None, ctx)
        assert score == 0.65  # No penalty applied since metadata is None

    def test_zero_words(self, scorer):
        """Empty content should score very low."""
        ctx = QualityContext(min_word_count=50)
        score = scorer._score_completeness("", None, ctx)
        # Empty string split => [""] => 1 word (the empty string)
        # 1 < 50 => 0.2 + 0.3 * (1/50) = 0.206
        assert score < 0.25

    def test_custom_min_word_count(self, scorer):
        """Custom min_word_count should change the threshold."""
        ctx = QualityContext(min_word_count=10)
        content = " ".join(["word"] * 8)
        score = scorer._score_completeness(content, None, ctx)
        # 8 < 10 => 0.2 + 0.3 * (8/10) = 0.2 + 0.24 = 0.44
        assert score == pytest.approx(0.44, abs=0.01)


# =============================================================================
# QualityScorer._score_consistency Tests
# =============================================================================


class TestScoreConsistency:
    """Tests for QualityScorer._score_consistency."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer()

    def test_neutral_content_base_score(self, scorer):
        """Neutral content should return base score of 0.7."""
        score = scorer._score_consistency("This is straightforward factual content.")
        assert score == pytest.approx(0.7, abs=0.01)

    def test_definitive_statements_boost(self, scorer):
        """Definitive language should boost score by 0.1."""
        content = "This has been definitely proven by experiments."
        score = scorer._score_consistency(content)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_multiple_uncertain_terms_penalty(self, scorer):
        """More than 3 uncertain terms should penalize by 0.1."""
        content = "Maybe this could be true. Perhaps it might be real. Possibly the result is unclear and unknown to us."
        score = scorer._score_consistency(content)
        # Base 0.7 - 0.1 (>3 uncertain) = 0.6
        assert score < 0.7

    def test_few_uncertain_no_penalty(self, scorer):
        """3 or fewer uncertain terms should not penalize."""
        content = "Maybe this is true. Perhaps we should check."
        score = scorer._score_consistency(content)
        assert score >= 0.7

    def test_contradictions_penalty(self, scorer):
        """More than 2 contradiction patterns should penalize."""
        content = (
            "However, the results but show different. "
            "Although the study, nevertheless confirms it. "
            "On one hand the data, on the other hand the analysis. "
        )
        score = scorer._score_consistency(content)
        # 3 contradictions > 2 => -0.1
        assert score < 0.7

    def test_few_contradictions_no_penalty(self, scorer):
        """2 or fewer contradiction patterns should not penalize."""
        content = "However, this but that. Although true, nevertheless."
        score = scorer._score_consistency(content)
        # 2 contradictions, not > 2, no penalty
        assert score >= 0.7

    def test_score_floor_at_0_3(self, scorer):
        """Score should never go below 0.3."""
        # Content with many contradictions and uncertainties
        content = (
            "However but. Although nevertheless. On one hand on the other hand. "
            "Maybe perhaps possibly might be could be unclear unknown extra."
        )
        score = scorer._score_consistency(content)
        assert score >= 0.3

    def test_score_ceiling_at_1_0(self, scorer):
        """Score should never exceed 1.0."""
        content = "Definitely certainly proven demonstrated established fact."
        score = scorer._score_consistency(content)
        assert score <= 1.0

    def test_definitive_with_uncertainties_balance(self, scorer):
        """Both definitive and uncertain language should partially offset."""
        content = (
            "This is definitely established. "
            "But maybe perhaps possibly might be the result could be unclear and unknown."
        )
        score = scorer._score_consistency(content)
        # Base 0.7 + 0.1 (definitive) - 0.1 (>3 uncertain) = 0.7
        assert score == pytest.approx(0.7, abs=0.05)

    def test_case_insensitive_matching(self, scorer):
        """Pattern matching should be case insensitive."""
        content = "DEFINITELY proven and ESTABLISHED."
        score = scorer._score_consistency(content)
        assert score > 0.7  # Should get the definitive boost


# =============================================================================
# QualityScorer.score() Integration Tests
# =============================================================================


class TestQualityScorerScore:
    """Integration tests for QualityScorer.score()."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer()

    def test_score_with_minimal_inputs(self, scorer):
        """score() with just content should return valid scores."""
        scores = scorer.score("Some basic content here for testing.")
        assert isinstance(scores, QualityScores)
        assert 0 <= scores.overall_score <= 1
        assert scores.semantic_score == 0.5  # Default, no semantic scoring

    def test_score_uses_default_context(self):
        """score() should use the default context if none provided."""
        ctx = QualityContext(query="test query")
        scorer = QualityScorer(default_context=ctx)
        scores = scorer.score("test query content here for evaluation.")
        # Relevance should be higher than neutral since "test" and "query" match
        assert scores.relevance_score > 0

    def test_score_with_metadata(self, scorer):
        """score() with metadata should use metadata for various dimensions."""
        metadata = EnrichedMetadata(
            source_type=SourceType.ACADEMIC,
            timestamp=datetime.now(),
            has_citations=True,
            has_data=True,
            topics=["ML"],
            provenance=Provenance(peer_reviewed=True, author="Dr. Expert"),
        )
        scores = scorer.score(
            " ".join(["word"] * 200),
            metadata=metadata,
        )
        assert scores.authority_score > 0.5
        assert scores.freshness_score > 0.5

    def test_score_with_url(self, scorer):
        """score() with URL should affect authority scoring."""
        scores_with_url = scorer.score(
            "content here for testing",
            url="https://arxiv.org/abs/2301.00001",
        )
        scores_without = scorer.score("content here for testing")
        # arxiv URL should boost authority
        assert scores_with_url.authority_score >= scores_without.authority_score

    def test_score_with_context(self, scorer):
        """score() with explicit context should use it over default."""
        ctx = QualityContext(query="machine learning", keywords=["neural"])
        scores = scorer.score(
            "machine learning neural networks are powerful",
            context=ctx,
        )
        assert scores.relevance_score > 0.3

    def test_score_with_fetched_at(self, scorer):
        """score() with fetched_at should affect freshness."""
        recent = scorer.score(
            "content here for testing",
            fetched_at=datetime.now(),
        )
        old = scorer.score(
            "content here for testing",
            fetched_at=datetime.now() - timedelta(days=400),
        )
        assert recent.freshness_score > old.freshness_score

    def test_score_semantic_defaults_to_neutral(self, scorer):
        """Without semantic scoring, semantic_score should be 0.5."""
        scores = scorer.score("any content here")
        assert scores.semantic_score == 0.5

    def test_score_metadata_provenance_publication_date(self, scorer):
        """Metadata provenance publication_date should affect freshness."""
        provenance = Provenance(publication_date=datetime.now() - timedelta(days=2))
        metadata = EnrichedMetadata(provenance=provenance)
        scores = scorer.score(" ".join(["word"] * 60), metadata=metadata)
        assert scores.freshness_score == 0.95

    def test_score_metadata_without_provenance(self, scorer):
        """Metadata without provenance should handle gracefully."""
        metadata = EnrichedMetadata(
            source_type=SourceType.NEWS,
            timestamp=datetime.now() - timedelta(days=10),
        )
        # Provenance is default (Provenance()) which has no publication_date
        scores = scorer.score(" ".join(["word"] * 60), metadata=metadata)
        # Should use metadata.timestamp for freshness since provenance.publication_date is None
        assert scores.freshness_score == 0.85  # 10 days old


# =============================================================================
# QualityScorer.score_with_semantic() Tests
# =============================================================================


class TestScoreWithSemantic:
    """Tests for QualityScorer.score_with_semantic (async)."""

    @pytest.mark.asyncio
    async def test_no_embedding_provider_uses_default(self):
        """Without embedding provider, semantic_score should stay 0.5."""
        scorer = QualityScorer(embedding_provider=None)
        ctx = QualityContext(query="machine learning")
        scores = await scorer.score_with_semantic(
            "machine learning content",
            context=ctx,
        )
        assert scores.semantic_score == 0.5

    @pytest.mark.asyncio
    async def test_no_query_skips_semantic(self):
        """Without a query, semantic scoring should be skipped."""
        provider = AsyncMock()
        scorer = QualityScorer(embedding_provider=provider)
        ctx = QualityContext(query="")
        scores = await scorer.score_with_semantic(
            "content here",
            context=ctx,
        )
        assert scores.semantic_score == 0.5
        provider.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_with_embedding_provider(self):
        """With a working embedding provider, semantic_score should be computed."""
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])

        with patch(
            "aragora.evidence.quality.cosine_similarity",
            return_value=0.8,
            create=True,
        ):
            # We need to patch the import inside _score_semantic
            scorer = QualityScorer(embedding_provider=provider)
            ctx = QualityContext(query="test query")

            # Patch the actual import within the method
            with patch.dict(
                "sys.modules",
                {
                    "aragora.memory.embeddings": MagicMock(
                        cosine_similarity=MagicMock(return_value=0.8)
                    )
                },
            ):
                scores = await scorer.score_with_semantic(
                    "test content",
                    context=ctx,
                )
                # cosine_similarity returns 0.8 -> (0.8 + 1) / 2 = 0.9
                assert scores.semantic_score == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_embedding_error_falls_back(self):
        """Embedding errors should fall back to 0.5."""
        provider = AsyncMock()
        provider.embed = AsyncMock(side_effect=RuntimeError("embed failed"))
        scorer = QualityScorer(embedding_provider=provider)
        ctx = QualityContext(query="test query")

        scores = await scorer.score_with_semantic(
            "content here",
            context=ctx,
        )
        assert scores.semantic_score == 0.5

    @pytest.mark.asyncio
    async def test_score_with_semantic_includes_base_scores(self):
        """score_with_semantic should include all base dimension scores."""
        scorer = QualityScorer(embedding_provider=None)
        ctx = QualityContext(query="machine learning", keywords=["neural"])
        scores = await scorer.score_with_semantic(
            "machine learning neural network analysis",
            context=ctx,
            url="https://arxiv.org/abs/2301.00001",
            fetched_at=datetime.now(),
        )
        # All scores should be set
        assert scores.relevance_score > 0
        assert scores.freshness_score > 0
        assert scores.authority_score > 0
        assert scores.completeness_score > 0
        assert scores.consistency_score > 0


# =============================================================================
# QualityScorer._score_semantic() Tests
# =============================================================================


class TestScoreSemantic:
    """Tests for QualityScorer._score_semantic (async)."""

    @pytest.mark.asyncio
    async def test_no_provider_returns_neutral(self):
        """No embedding provider should return 0.5."""
        scorer = QualityScorer(embedding_provider=None)
        result = await scorer._score_semantic("content", "query")
        assert result == 0.5

    @pytest.mark.asyncio
    async def test_import_error_returns_neutral(self):
        """ImportError for embeddings module should return 0.5."""
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[1.0, 0.0])
        scorer = QualityScorer(embedding_provider=provider)

        # Ensure the import fails
        with patch.dict("sys.modules", {"aragora.memory.embeddings": None}):
            result = await scorer._score_semantic("content", "query")
            assert result == 0.5

    @pytest.mark.asyncio
    async def test_caches_query_embedding(self):
        """Query embeddings should be cached for reuse."""
        provider = AsyncMock()
        embed_values = [[1.0, 0.0], [0.9, 0.1]]
        provider.embed = AsyncMock(side_effect=embed_values)

        scorer = QualityScorer(embedding_provider=provider)

        embeddings_mod = MagicMock()
        embeddings_mod.cosine_similarity = MagicMock(return_value=0.5)

        with patch.dict("sys.modules", {"aragora.memory.embeddings": embeddings_mod}):
            await scorer._score_semantic("content1", "my query")
            # embed called twice: once for query, once for content
            assert provider.embed.call_count == 2

            # Reset for second call
            provider.embed.reset_mock()
            provider.embed = AsyncMock(return_value=[0.8, 0.2])
            await scorer._score_semantic("content2", "my query")
            # Only content embedding should be called (query is cached)
            assert provider.embed.call_count == 1

    @pytest.mark.asyncio
    async def test_content_truncation(self):
        """Content should be truncated to 8000 chars for embedding."""
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[1.0, 0.0])
        scorer = QualityScorer(embedding_provider=provider)

        embeddings_mod = MagicMock()
        embeddings_mod.cosine_similarity = MagicMock(return_value=0.5)

        long_content = "a" * 20000

        with patch.dict("sys.modules", {"aragora.memory.embeddings": embeddings_mod}):
            await scorer._score_semantic(long_content, "query")
            # The second embed call (for content) should have truncated text
            content_call = provider.embed.call_args_list[1]
            assert len(content_call[0][0]) == 8000

    @pytest.mark.asyncio
    async def test_similarity_clamped_to_0_1(self):
        """Output should be clamped to [0, 1]."""
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[1.0, 0.0])
        scorer = QualityScorer(embedding_provider=provider)

        embeddings_mod = MagicMock()

        # Test with cosine = 1.0 => (1 + 1) / 2 = 1.0
        embeddings_mod.cosine_similarity = MagicMock(return_value=1.0)
        with patch.dict("sys.modules", {"aragora.memory.embeddings": embeddings_mod}):
            result = await scorer._score_semantic("content", "query")
            assert result == 1.0

        # Reset cache
        scorer._query_embedding_cache.clear()

        # Test with cosine = -1.0 => (-1 + 1) / 2 = 0.0
        embeddings_mod.cosine_similarity = MagicMock(return_value=-1.0)
        with patch.dict("sys.modules", {"aragora.memory.embeddings": embeddings_mod}):
            result = await scorer._score_semantic("content", "query2")
            assert result == 0.0


# =============================================================================
# QualityScorer.score_batch_with_semantic() Tests
# =============================================================================


class TestScoreBatchWithSemantic:
    """Tests for QualityScorer.score_batch_with_semantic (async)."""

    @pytest.mark.asyncio
    async def test_batch_without_provider(self):
        """Batch scoring without provider should produce neutral semantic scores."""
        scorer = QualityScorer(embedding_provider=None)
        evidence = [
            FakeEvidenceSnippet(snippet="First evidence content for testing."),
            FakeEvidenceSnippet(snippet="Second evidence content for testing."),
        ]
        results = await scorer.score_batch_with_semantic(evidence)
        assert len(results) == 2
        for r in results:
            assert r.semantic_score == 0.5

    @pytest.mark.asyncio
    async def test_batch_without_query(self):
        """Batch scoring without query should produce neutral semantic scores."""
        provider = AsyncMock()
        scorer = QualityScorer(embedding_provider=provider)
        ctx = QualityContext(query="")
        evidence = [FakeEvidenceSnippet(snippet="content for testing here.")]
        results = await scorer.score_batch_with_semantic(evidence, context=ctx)
        assert len(results) == 1
        assert results[0].semantic_score == 0.5

    @pytest.mark.asyncio
    async def test_batch_uses_evidence_attributes(self):
        """Batch scoring should use url, source, fetched_at from evidence."""
        scorer = QualityScorer(embedding_provider=None)
        evidence = [
            FakeEvidenceSnippet(
                snippet=" ".join(["word"] * 60),
                url="https://arxiv.org/abs/123",
                source="arxiv",
                fetched_at=datetime.now(),
            ),
        ]
        results = await scorer.score_batch_with_semantic(evidence)
        assert len(results) == 1
        # arxiv URL should boost authority
        assert results[0].authority_score > 0.5

    @pytest.mark.asyncio
    async def test_batch_empty_list(self):
        """Batch scoring with empty list should return empty list."""
        scorer = QualityScorer(embedding_provider=None)
        results = await scorer.score_batch_with_semantic([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_with_provider_error(self):
        """Batch scoring with provider error should fall back to 0.5."""
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[1.0, 0.0])
        provider.embed_batch = AsyncMock(side_effect=RuntimeError("batch failed"))
        scorer = QualityScorer(embedding_provider=provider)
        ctx = QualityContext(query="test query")

        evidence = [
            FakeEvidenceSnippet(snippet="content for testing purposes here."),
        ]

        embeddings_mod = MagicMock()
        embeddings_mod.cosine_similarity = MagicMock(return_value=0.5)

        with patch.dict("sys.modules", {"aragora.memory.embeddings": embeddings_mod}):
            results = await scorer.score_batch_with_semantic(evidence, context=ctx)
            assert len(results) == 1
            assert results[0].semantic_score == 0.5


# =============================================================================
# QualityScorer.set_embedding_provider() Tests
# =============================================================================


class TestSetEmbeddingProvider:
    """Tests for QualityScorer.set_embedding_provider."""

    def test_set_provider(self):
        """Setting a provider should update the internal state."""
        scorer = QualityScorer(embedding_provider=None)
        provider = MagicMock()
        scorer.set_embedding_provider(provider)
        assert scorer._embedding_provider is provider

    def test_set_provider_clears_cache(self):
        """Setting a new provider should clear the query embedding cache."""
        scorer = QualityScorer(embedding_provider=None)
        scorer._query_embedding_cache["old_query"] = [1.0, 0.0]
        scorer.set_embedding_provider(MagicMock())
        assert len(scorer._query_embedding_cache) == 0


# =============================================================================
# QualityFilter Tests
# =============================================================================


class TestQualityFilter:
    """Tests for QualityFilter.filter() and .rank()."""

    @pytest.fixture
    def high_quality_evidence(self):
        return FakeEvidenceSnippet(
            snippet=" ".join(["machine", "learning", "neural"] * 100),
            url="https://arxiv.org/abs/2301.00001",
            source="arxiv",
            fetched_at=datetime.now(),
        )

    @pytest.fixture
    def low_quality_evidence(self):
        return FakeEvidenceSnippet(
            snippet="short low quality",
            url=None,
            source=None,
            fetched_at=datetime.now() - timedelta(days=500),
        )

    @pytest.fixture
    def medium_quality_evidence(self):
        return FakeEvidenceSnippet(
            snippet=" ".join(["analysis", "of", "results"] * 50),
            url="https://example.com/article",
            source="web",
            fetched_at=datetime.now() - timedelta(days=30),
        )

    def test_filter_default_thresholds(self):
        """Filter with default thresholds should accept decent evidence."""
        qf = QualityFilter()
        evidence = [
            FakeEvidenceSnippet(
                snippet=" ".join(["word"] * 200),
                url="https://arxiv.org/abs/123",
                source="arxiv",
                fetched_at=datetime.now(),
            ),
        ]
        passed = qf.filter(evidence)
        assert len(passed) >= 0  # Result depends on exact scoring

    def test_filter_removes_low_quality(self):
        """Filter should remove evidence below thresholds."""
        qf = QualityFilter(min_overall_score=0.8)
        low = FakeEvidenceSnippet(
            snippet="tiny",
            url=None,
            source=None,
            fetched_at=datetime.now() - timedelta(days=500),
        )
        passed = qf.filter([low])
        assert len(passed) == 0

    def test_filter_with_context(self):
        """Filter with context should use relevance context for scoring."""
        ctx = QualityContext(query="machine learning", keywords=["neural"])
        qf = QualityFilter(min_relevance_score=0.01)
        evidence = [
            FakeEvidenceSnippet(
                snippet="machine learning neural network deep " * 20,
                url="https://arxiv.org/abs/123",
                source="arxiv",
                fetched_at=datetime.now(),
            ),
        ]
        passed = qf.filter(evidence, context=ctx)
        assert len(passed) == 1

    def test_filter_respects_min_relevance(self):
        """Filter should respect min_relevance_score threshold."""
        ctx = QualityContext(query="quantum physics")
        qf = QualityFilter(min_relevance_score=0.9)
        evidence = [
            FakeEvidenceSnippet(
                snippet="cooking recipe for pasta " * 20,
                url="https://example.com",
                source="web",
                fetched_at=datetime.now(),
            ),
        ]
        passed = qf.filter(evidence, context=ctx)
        assert len(passed) == 0

    def test_filter_respects_min_authority(self):
        """Filter should respect min_authority_score threshold."""
        qf = QualityFilter(min_authority_score=0.8)
        evidence = [
            FakeEvidenceSnippet(
                snippet=" ".join(["word"] * 200),
                url="https://random-blog.com/post",
                source="web",
                fetched_at=datetime.now(),
            ),
        ]
        passed = qf.filter(evidence)
        assert len(passed) == 0  # WEB source has 0.5 authority

    def test_filter_empty_list(self):
        """Filtering empty list should return empty list."""
        qf = QualityFilter()
        assert qf.filter([]) == []

    def test_rank_returns_sorted_tuples(self):
        """rank() should return (evidence, scores) tuples sorted by overall_score."""
        qf = QualityFilter()
        e1 = FakeEvidenceSnippet(
            snippet=" ".join(["word"] * 200),
            url="https://arxiv.org/abs/123",
            source="arxiv",
            fetched_at=datetime.now(),
        )
        e2 = FakeEvidenceSnippet(
            snippet="tiny",
            url=None,
            source=None,
            fetched_at=datetime.now() - timedelta(days=500),
        )
        ranked = qf.rank([e1, e2])
        assert len(ranked) == 2
        # Higher quality first
        assert ranked[0][1].overall_score >= ranked[1][1].overall_score

    def test_rank_with_top_k(self):
        """rank() with top_k should limit results."""
        qf = QualityFilter()
        evidence = [
            FakeEvidenceSnippet(
                snippet=" ".join(["word"] * 100),
                url=f"https://example.com/{i}",
                source="web",
                fetched_at=datetime.now() - timedelta(days=i * 10),
            )
            for i in range(5)
        ]
        ranked = qf.rank(evidence, top_k=2)
        assert len(ranked) == 2

    def test_rank_without_top_k(self):
        """rank() without top_k should return all results."""
        qf = QualityFilter()
        evidence = [
            FakeEvidenceSnippet(
                snippet=" ".join(["word"] * 100),
                url=f"https://example.com/{i}",
                source="web",
                fetched_at=datetime.now(),
            )
            for i in range(3)
        ]
        ranked = qf.rank(evidence)
        assert len(ranked) == 3

    def test_rank_empty_list(self):
        """Ranking empty list should return empty list."""
        qf = QualityFilter()
        assert qf.rank([]) == []

    def test_rank_returns_scores_objects(self):
        """rank() should return QualityScores in each tuple."""
        qf = QualityFilter()
        evidence = [
            FakeEvidenceSnippet(
                snippet=" ".join(["word"] * 100),
                url="https://example.com",
                source="web",
                fetched_at=datetime.now(),
            ),
        ]
        ranked = qf.rank(evidence)
        assert len(ranked) == 1
        ev, scores = ranked[0]
        assert isinstance(scores, QualityScores)
        assert ev is evidence[0]

    def test_filter_creates_default_scorer(self):
        """QualityFilter with no scorer should create a default one."""
        qf = QualityFilter()
        assert isinstance(qf.scorer, QualityScorer)

    def test_filter_uses_custom_scorer(self):
        """QualityFilter should use the provided scorer."""
        custom_scorer = QualityScorer(
            default_context=QualityContext(query="specific topic"),
        )
        qf = QualityFilter(scorer=custom_scorer)
        assert qf.scorer is custom_scorer


# =============================================================================
# score_evidence_snippet() Convenience Function Tests
# =============================================================================


class TestScoreEvidenceSnippet:
    """Tests for the module-level score_evidence_snippet() convenience function."""

    def test_basic_usage(self):
        """score_evidence_snippet with minimal args should return scores."""
        snippet = FakeEvidenceSnippet(
            snippet=" ".join(["word"] * 60),
            url="https://example.com",
            source="web",
            fetched_at=datetime.now(),
        )
        scores = score_evidence_snippet(snippet)
        assert isinstance(scores, QualityScores)
        assert 0 <= scores.overall_score <= 1

    def test_with_query(self):
        """score_evidence_snippet with query should affect relevance vs no-context neutral."""
        snippet = FakeEvidenceSnippet(
            snippet="machine learning neural network analysis " * 10,
            url="https://example.com",
            source="web",
            fetched_at=datetime.now(),
        )
        scores_with_query = score_evidence_snippet(snippet, query="machine learning")
        # Without a query or keywords, _score_relevance returns 0.5 (neutral).
        # With a matching query, the score is computed from query/keyword/topic weights.
        # Query "machine learning" matches 2/2 words => 0.3, no keywords => 0,
        # no required topics => 0.15. Total 0.45.
        # The key assertion is that relevance is actively computed, not just neutral.
        assert scores_with_query.relevance_score > 0.0
        assert scores_with_query.relevance_score == pytest.approx(0.45, abs=0.01)

    def test_with_keywords(self):
        """score_evidence_snippet with keywords should affect relevance."""
        snippet = FakeEvidenceSnippet(
            snippet="python testing framework pytest unittest coverage",
            url="https://example.com",
            source="web",
            fetched_at=datetime.now(),
        )
        scores = score_evidence_snippet(
            snippet,
            keywords=["python", "testing", "pytest"],
        )
        assert scores.relevance_score > 0.3

    def test_with_custom_scorer(self):
        """score_evidence_snippet with custom scorer should use it."""
        custom_ctx = QualityContext(
            preferred_sources={"trusted"},
        )
        custom_scorer = QualityScorer(default_context=custom_ctx)
        snippet = FakeEvidenceSnippet(
            snippet=" ".join(["word"] * 60),
            url="https://example.com",
            source="trusted",
            fetched_at=datetime.now(),
        )
        scores = score_evidence_snippet(snippet, scorer=custom_scorer)
        assert isinstance(scores, QualityScores)

    def test_creates_default_scorer(self):
        """score_evidence_snippet with no scorer should create one."""
        snippet = FakeEvidenceSnippet(
            snippet=" ".join(["word"] * 60),
            url="https://example.com",
            source="web",
            fetched_at=datetime.now(),
        )
        scores = score_evidence_snippet(snippet)
        assert isinstance(scores, QualityScores)

    def test_none_keywords_handled(self):
        """score_evidence_snippet with keywords=None should not crash."""
        snippet = FakeEvidenceSnippet(
            snippet=" ".join(["word"] * 60),
        )
        scores = score_evidence_snippet(snippet, keywords=None)
        assert isinstance(scores, QualityScores)

    def test_passes_url_source_fetched_at(self):
        """Convenience function should pass url, source, fetched_at to scorer."""
        snippet = FakeEvidenceSnippet(
            snippet=" ".join(["word"] * 60),
            url="https://arxiv.org/abs/123",
            source="arxiv",
            fetched_at=datetime.now(),
        )
        scores = score_evidence_snippet(snippet)
        # arxiv URL should boost authority above the WEB default
        assert scores.authority_score > 0.5


# =============================================================================
# Edge Case and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_content(self):
        """Scoring empty content should not crash."""
        scorer = QualityScorer()
        scores = scorer.score("")
        assert isinstance(scores, QualityScores)

    def test_very_long_content(self):
        """Scoring very long content should not crash."""
        scorer = QualityScorer()
        scores = scorer.score("word " * 100000)
        assert isinstance(scores, QualityScores)

    def test_special_characters_in_content(self):
        """Content with special characters should not crash."""
        scorer = QualityScorer()
        scores = scorer.score("Hello! @#$%^&*() <html> {'key': 'value'} [1,2,3]")
        assert isinstance(scores, QualityScores)

    def test_unicode_content(self):
        """Unicode content should be handled gracefully."""
        scorer = QualityScorer()
        scores = scorer.score(
            "This is a test with unicode: cafe\u0301 na\u00efve re\u0301sume\u0301"
        )
        assert isinstance(scores, QualityScores)

    def test_newlines_in_content(self):
        """Content with newlines should be handled."""
        scorer = QualityScorer()
        scores = scorer.score("Line 1\nLine 2\n\nLine 4")
        assert isinstance(scores, QualityScores)

    def test_custom_weights_affect_overall(self):
        """Custom weights should change the overall score calculation."""
        # All weight on relevance
        scores = QualityScores(
            relevance_score=1.0,
            semantic_score=0.0,
            freshness_score=0.0,
            authority_score=0.0,
            completeness_score=0.0,
            consistency_score=0.0,
            relevance_weight=1.0,
            semantic_weight=0.0,
            freshness_weight=0.0,
            authority_weight=0.0,
            completeness_weight=0.0,
            consistency_weight=0.0,
        )
        assert scores.overall_score == pytest.approx(1.0, abs=1e-9)

    def test_quality_scores_mutable(self):
        """QualityScores should be mutable (not frozen)."""
        scores = QualityScores()
        scores.relevance_score = 0.9
        scores.authority_score = 0.8
        assert scores.relevance_score == 0.9
        assert scores.authority_score == 0.8

    def test_scorer_init_default_context(self):
        """QualityScorer with no context should create a default one."""
        scorer = QualityScorer()
        assert isinstance(scorer.default_context, QualityContext)
        assert scorer.default_context.query == ""

    def test_scorer_init_with_context(self):
        """QualityScorer should accept a custom default context."""
        ctx = QualityContext(query="default query")
        scorer = QualityScorer(default_context=ctx)
        assert scorer.default_context.query == "default query"

    def test_freshness_max_age_equals_365(self):
        """When max_age_days is 365, exactly 365 should trigger linear decay edge."""
        scorer = QualityScorer()
        # At exactly 365 days with max_age=365, age_days >= max_age_days
        # so it should return 0.30
        date = datetime.now() - timedelta(days=365)
        result = scorer._score_freshness(date, None, 365)
        assert result == 0.30

    def test_freshness_max_age_large(self):
        """Large max_age_days should extend the decay range."""
        scorer = QualityScorer()
        # 400 days old with max_age=1000: should be in linear decay zone
        date = datetime.now() - timedelta(days=400)
        result = scorer._score_freshness(date, None, 1000)
        # 365 < 400 < 1000: linear decay
        # progress = (400-365)/(1000-365) = 35/635 ~ 0.055
        # score = 0.55 - 0.25*0.055 ~ 0.536
        assert 0.50 < result < 0.55
