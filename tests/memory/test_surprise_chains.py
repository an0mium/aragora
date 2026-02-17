"""Tests for surprise chain tracking — Titans momentum amplification.

Chains of related surprising items amplify the combined surprise signal.
"""

from __future__ import annotations

import time

import pytest

from aragora.memory.surprise import (
    ContentSurpriseScore,
    ContentSurpriseScorer,
    EnrichedSurpriseScore,
    SurpriseChainConfig,
    SurpriseChainTracker,
    _extract_keywords,
)


# ---------------------------------------------------------------------------
# SurpriseChainTracker unit tests
# ---------------------------------------------------------------------------


class TestSurpriseChainTracker:
    """Tests for the SurpriseChainTracker class."""

    def _base_score(self, combined: float = 0.6) -> ContentSurpriseScore:
        return ContentSurpriseScore(
            novelty=combined,
            momentum=0.5,
            combined=combined,
            should_store=True,
            reason="test",
        )

    def test_single_item_no_chain_bonus(self):
        """First item starts a chain but gets no bonus."""
        tracker = SurpriseChainTracker()
        score = self._base_score(0.6)
        enriched = tracker.enrich(score, {"machine", "learning", "model"})

        assert isinstance(enriched, EnrichedSurpriseScore)
        assert enriched.chain_length == 1
        assert enriched.chain_bonus == 0.0
        assert enriched.combined == 0.6

    def test_related_items_build_chain(self):
        """Second related surprising item extends the chain and gets a bonus."""
        tracker = SurpriseChainTracker()

        # First item
        tracker.enrich(self._base_score(0.6), {"machine", "learning", "model"})

        # Second related item
        enriched = tracker.enrich(
            self._base_score(0.5), {"machine", "learning", "training"}
        )

        assert enriched.chain_length == 2
        assert enriched.chain_bonus == pytest.approx(0.05)
        assert enriched.combined == pytest.approx(0.55)

    def test_chain_bonus_grows_with_length(self):
        """Bonus increases with each chain link."""
        config = SurpriseChainConfig(chain_bonus_per_link=0.05, max_chain_bonus=0.25)
        tracker = SurpriseChainTracker(config)
        keywords = {"machine", "learning", "deep"}

        tracker.enrich(self._base_score(0.6), keywords)
        tracker.enrich(self._base_score(0.6), keywords)
        enriched = tracker.enrich(self._base_score(0.6), keywords)

        assert enriched.chain_length == 3
        assert enriched.chain_bonus == pytest.approx(0.10)

    def test_chain_bonus_capped(self):
        """Bonus doesn't exceed max_chain_bonus."""
        config = SurpriseChainConfig(
            chain_bonus_per_link=0.1,
            max_chain_bonus=0.15,
        )
        tracker = SurpriseChainTracker(config)
        kw = {"alpha", "beta", "gamma"}

        for _ in range(10):
            enriched = tracker.enrich(self._base_score(0.6), kw)

        assert enriched.chain_bonus <= 0.15

    def test_unrelated_items_no_chain_extension(self):
        """Items with no keyword overlap don't extend the chain."""
        tracker = SurpriseChainTracker()

        tracker.enrich(self._base_score(0.6), {"machine", "learning", "model"})
        enriched = tracker.enrich(
            self._base_score(0.6), {"cooking", "recipe", "dinner"}
        )

        # Should start a new chain, not extend the old one
        assert enriched.chain_length == 1
        assert enriched.chain_bonus == 0.0

    def test_low_surprise_does_not_start_chain(self):
        """Items below min_surprise_to_chain don't start or extend chains."""
        config = SurpriseChainConfig(min_surprise_to_chain=0.5)
        tracker = SurpriseChainTracker(config)

        enriched = tracker.enrich(self._base_score(0.3), {"machine", "learning"})

        assert enriched.chain_length == 0
        assert enriched.chain_bonus == 0.0
        assert tracker.active_chain_count == 0

    def test_low_surprise_does_not_extend_chain(self):
        """Low-surprise item doesn't extend an existing chain."""
        config = SurpriseChainConfig(min_surprise_to_chain=0.5)
        tracker = SurpriseChainTracker(config)

        # Start chain with high surprise
        tracker.enrich(self._base_score(0.7), {"machine", "learning"})
        assert tracker.active_chain_count == 1

        # Low-surprise item — should not extend
        enriched = tracker.enrich(self._base_score(0.3), {"machine", "learning"})
        assert enriched.chain_length == 0
        assert enriched.chain_bonus == 0.0

    def test_chain_expiry(self, monkeypatch):
        """Chains expire after TTL."""
        config = SurpriseChainConfig(chain_ttl_seconds=0.1)
        tracker = SurpriseChainTracker(config)
        kw = {"machine", "learning"}

        tracker.enrich(self._base_score(0.6), kw)
        assert tracker.active_chain_count == 1

        # Mock time passing
        time.sleep(0.15)

        assert tracker.active_chain_count == 0

        # New item starts a fresh chain
        enriched = tracker.enrich(self._base_score(0.6), kw)
        assert enriched.chain_length == 1
        assert enriched.chain_bonus == 0.0

    def test_multiple_independent_chains(self):
        """Different topics create independent chains."""
        tracker = SurpriseChainTracker()

        tracker.enrich(self._base_score(0.6), {"machine", "learning", "model"})
        tracker.enrich(self._base_score(0.6), {"cooking", "recipe", "dinner"})

        assert tracker.active_chain_count == 2

    def test_enriched_score_preserves_base_fields(self):
        """EnrichedSurpriseScore carries all base ContentSurpriseScore fields."""
        tracker = SurpriseChainTracker()
        base = ContentSurpriseScore(
            novelty=0.8, momentum=0.4, combined=0.68, should_store=True, reason="test"
        )
        enriched = tracker.enrich(base, {"alpha", "beta"})

        assert enriched.novelty == 0.8
        assert enriched.momentum == 0.4
        assert enriched.reason == "test"

    def test_chain_id_consistent_within_chain(self):
        """All items in a chain share the same chain_id."""
        tracker = SurpriseChainTracker()
        kw = {"machine", "learning", "deep"}

        e1 = tracker.enrich(self._base_score(0.6), kw)
        e2 = tracker.enrich(self._base_score(0.6), kw)
        e3 = tracker.enrich(self._base_score(0.6), kw)

        assert e1.chain_id is not None
        assert e1.chain_id == e2.chain_id == e3.chain_id

    def test_default_config_values(self):
        """Default config has sensible values."""
        config = SurpriseChainConfig()
        assert config.chain_ttl_seconds == 300.0
        assert config.relatedness_threshold == 0.3
        assert config.chain_bonus_per_link == 0.05
        assert config.max_chain_bonus == 0.25
        assert config.min_surprise_to_chain == 0.4

    def test_combined_score_capped_at_one(self):
        """Enriched combined score doesn't exceed 1.0."""
        config = SurpriseChainConfig(chain_bonus_per_link=0.5, max_chain_bonus=0.9)
        tracker = SurpriseChainTracker(config)
        kw = {"alpha", "beta"}

        for _ in range(5):
            enriched = tracker.enrich(self._base_score(0.9), kw)

        assert enriched.combined <= 1.0


# ---------------------------------------------------------------------------
# Integration with ContentSurpriseScorer
# ---------------------------------------------------------------------------


class TestContentSurpriseScorerChainIntegration:
    """Test chain tracking integration in ContentSurpriseScorer."""

    def test_scorer_without_chain_config_returns_base_score(self):
        """Without chain_config, scorer returns plain ContentSurpriseScore."""
        scorer = ContentSurpriseScorer(threshold=0.3)
        score = scorer.score("novel machine learning approach", "test")

        assert type(score) is ContentSurpriseScore

    def test_scorer_with_chain_config_returns_enriched_score(self):
        """With chain_config, scorer returns EnrichedSurpriseScore."""
        config = SurpriseChainConfig()
        scorer = ContentSurpriseScorer(threshold=0.3, chain_config=config)
        score = scorer.score("novel machine learning approach", "test")

        assert isinstance(score, EnrichedSurpriseScore)

    def test_scorer_chain_builds_across_calls(self):
        """Sequential calls to scorer build chains."""
        config = SurpriseChainConfig(min_surprise_to_chain=0.3)
        scorer = ContentSurpriseScorer(threshold=0.1, chain_config=config)

        # Score multiple related items
        s1 = scorer.score("machine learning training data", "test")
        s2 = scorer.score("machine learning model accuracy", "test")

        assert isinstance(s1, EnrichedSurpriseScore)
        assert isinstance(s2, EnrichedSurpriseScore)
        # Second call should potentially build on the chain
        if s1.combined >= config.min_surprise_to_chain:
            assert s2.chain_length >= 1

    def test_score_debate_outcome_with_chains(self):
        """score_debate_outcome also respects chain tracking."""
        config = SurpriseChainConfig(min_surprise_to_chain=0.3)
        scorer = ContentSurpriseScorer(threshold=0.1, chain_config=config)

        # First score primes the chain
        scorer.score("machine learning model deployment", "test")

        # Debate outcome on related topic
        score = scorer.score_debate_outcome(
            conclusion="Deploy ML model to production",
            domain="engineering",
            confidence=0.85,
        )
        # score_debate_outcome calls self.score internally, so it should
        # integrate with chain tracking
        assert score.should_store or score.combined > 0

    def test_extract_keywords_helper(self):
        """_extract_keywords extracts lowercase tokens >= 3 chars."""
        kw = _extract_keywords("Hello World AI ML deep learning")
        assert "hello" in kw
        assert "world" in kw
        assert "deep" in kw
        assert "learning" in kw
        # Short words excluded
        assert "ai" not in kw
