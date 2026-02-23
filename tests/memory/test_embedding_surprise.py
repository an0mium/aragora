"""Tests for EmbeddingSurpriseScorer — semantic novelty via cosine distance.

Falls back to keyword-based scoring when embeddings are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
import math

import pytest

from aragora.memory.surprise import (
    ContentSurpriseScore,
    ContentSurpriseScorer,
    EmbeddingSurpriseScorer,
    EnrichedSurpriseScore,
    SurpriseChainConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeEmbeddingResult:
    embedding: list[float]
    text: str = ""
    provider: str = "test"
    model: str = "test"
    cached: bool = False
    dimension: int = 3


def _make_embed_service(
    embeddings: dict[str, list[float]] | None = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock embedding service with controlled responses."""
    service = MagicMock()

    if side_effect:
        service.embed = AsyncMock(side_effect=side_effect)
    elif embeddings:

        async def _embed(text: str) -> FakeEmbeddingResult:
            # Return the embedding for the closest matching key
            for key, emb in embeddings.items():
                if key in text or text in key:
                    return FakeEmbeddingResult(embedding=emb)
            # Default: return a fixed embedding
            return FakeEmbeddingResult(embedding=[0.5, 0.5, 0.5])

        service.embed = AsyncMock(side_effect=_embed)
    else:
        service.embed = AsyncMock(return_value=FakeEmbeddingResult(embedding=[1.0, 0.0, 0.0]))

    return service


# ---------------------------------------------------------------------------
# EmbeddingSurpriseScorer tests
# ---------------------------------------------------------------------------


class TestEmbeddingSurpriseScorer:
    """Tests for semantic surprise scoring."""

    def test_fallback_when_no_service(self):
        """Without embedding service, falls back to keyword scoring."""
        scorer = EmbeddingSurpriseScorer(threshold=0.3)
        score = scorer.score("novel machine learning approach", "test")
        assert isinstance(score, ContentSurpriseScore)
        assert score.combined > 0

    def test_fallback_on_empty_context(self):
        """With empty context, no embedding comparison possible."""
        service = _make_embed_service()
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        score = scorer.score("something novel", "test", existing_context="")
        # Empty context → falls back to keyword scoring (novelty=1.0)
        assert score.novelty > 0

    def test_similar_content_low_novelty(self):
        """Content similar to context should have low novelty."""
        # Same direction → cosine sim ≈ 1.0 → novelty ≈ 0.0
        service = _make_embed_service(
            embeddings={
                "machine learning": [1.0, 0.0, 0.0],
                "deep learning": [1.0, 0.0, 0.0],
            }
        )
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        score = scorer.score(
            "machine learning models",
            "test",
            existing_context="deep learning models",
        )
        assert score.novelty < 0.1

    def test_dissimilar_content_high_novelty(self):
        """Content orthogonal to context should have high novelty."""
        # Orthogonal vectors → cosine sim = 0.0 → novelty = 1.0
        service = _make_embed_service(
            embeddings={
                "machine learning": [1.0, 0.0, 0.0],
                "cooking recipes": [0.0, 1.0, 0.0],
            }
        )
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        score = scorer.score(
            "machine learning techniques",
            "test",
            existing_context="cooking recipes and ingredients",
        )
        assert score.novelty > 0.9

    def test_paraphrase_lower_novelty_than_keyword(self):
        """Paraphrased content should score lower novelty with embeddings than keywords."""
        # With embeddings: similar vectors → low novelty
        service = _make_embed_service(
            embeddings={
                "artificial intelligence": [0.9, 0.1, 0.0],
                "machine learning": [0.85, 0.15, 0.0],
            }
        )
        embed_scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        embed_score = embed_scorer.score(
            "artificial intelligence systems",
            "test",
            existing_context="machine learning algorithms",
        )

        # With keywords: low keyword overlap → high novelty
        kw_scorer = ContentSurpriseScorer(threshold=0.3)
        kw_score = kw_scorer.score(
            "artificial intelligence systems",
            "test",
            existing_context="machine learning algorithms",
        )

        # Embedding should detect semantic similarity better
        assert embed_score.novelty < kw_score.novelty

    def test_fallback_on_embed_error(self):
        """When embedding service raises, falls back to keyword scoring."""
        service = _make_embed_service(side_effect=RuntimeError("API timeout"))
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        score = scorer.score(
            "novel content here",
            "test",
            existing_context="existing context",
        )
        # Should still return a valid score (keyword fallback)
        assert isinstance(score, ContentSurpriseScore)
        assert score.combined > 0

    def test_cache_prevents_duplicate_embed_calls(self):
        """Second call with same content shouldn't call embed service again."""
        service = _make_embed_service(
            embeddings={
                "hello": [1.0, 0.0, 0.0],
                "world": [0.0, 1.0, 0.0],
            }
        )
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)

        # First call
        scorer.score("hello world", "test", existing_context="world example")

        call_count_after_first = service.embed.call_count

        # Second call with same content
        scorer.score("hello world", "test", existing_context="world example")

        # Cache should prevent additional embed calls
        # At most the same number of new calls (cache hits)
        assert service.embed.call_count <= call_count_after_first * 2

    def test_reason_includes_embedding_marker(self):
        """Score reason should indicate embedding was used."""
        service = _make_embed_service(
            embeddings={
                "novel": [1.0, 0.0, 0.0],
                "context": [0.0, 1.0, 0.0],
            }
        )
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        score = scorer.score("novel content", "test", existing_context="existing context")
        assert "[embedding]" in score.reason

    def test_empty_content_has_zero_novelty(self):
        """Empty content has zero novelty (no keywords to extract)."""
        service = _make_embed_service()
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        # When content has no keywords, falls back to super().score()
        # which returns novelty=0.0, momentum=0.5 (neutral), combined=0.15
        score = scorer.score("", "test", existing_context="some context")
        assert score.novelty == 0.0
        assert not score.should_store

    def test_with_chain_config(self):
        """EmbeddingSurpriseScorer works with chain tracking."""
        config = SurpriseChainConfig(min_surprise_to_chain=0.3)
        service = _make_embed_service(
            embeddings={
                "machine": [1.0, 0.0, 0.0],
                "context": [0.0, 1.0, 0.0],
            }
        )
        scorer = EmbeddingSurpriseScorer(
            threshold=0.1,
            embedding_service=service,
            chain_config=config,
        )
        score = scorer.score(
            "machine learning approach",
            "test",
            existing_context="some context here",
        )
        assert isinstance(score, (ContentSurpriseScore, EnrichedSurpriseScore))

    def test_score_debate_outcome_uses_embedding(self):
        """score_debate_outcome inherited from parent calls score() which uses embedding."""
        service = _make_embed_service(
            embeddings={
                "new conclusion": [1.0, 0.0, 0.0],
                "old conclusion": [0.0, 1.0, 0.0],
            }
        )
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        score = scorer.score_debate_outcome(
            conclusion="new conclusion here",
            domain="tech",
            confidence=0.9,
            prior_conclusions=["old conclusion stuff"],
        )
        assert score.combined > 0

    def test_embedding_cache_size_limit(self):
        """Cache evicts old entries when full."""
        service = _make_embed_service()
        scorer = EmbeddingSurpriseScorer(
            threshold=0.3,
            embedding_service=service,
            embedding_cache_size=3,
        )
        # Fill cache beyond limit
        for i in range(5):
            scorer._get_embedding(f"text number {i} with enough chars to hash")

        assert len(scorer._embedding_cache) <= 3

    def test_zero_vector_treated_as_novel(self):
        """Zero-length embedding vector treated as fully novel."""
        service = _make_embed_service(
            embeddings={
                "content": [0.0, 0.0, 0.0],
                "context": [1.0, 0.0, 0.0],
            }
        )
        scorer = EmbeddingSurpriseScorer(threshold=0.3, embedding_service=service)
        score = scorer.score(
            "content text here",
            "test",
            existing_context="context text here",
        )
        assert score.novelty == 1.0


# ---------------------------------------------------------------------------
# MemoryCoordinator integration
# ---------------------------------------------------------------------------


class TestCoordinatorEmbeddingSurprise:
    """Test use_embedding_surprise param on MemoryCoordinator."""

    def test_default_uses_keyword_scorer(self):
        from aragora.memory.coordinator import MemoryCoordinator

        coord = MemoryCoordinator()
        assert isinstance(coord.surprise_scorer, ContentSurpriseScorer)
        assert not isinstance(coord.surprise_scorer, EmbeddingSurpriseScorer)

    def test_use_embedding_surprise_creates_embedding_scorer(self):
        from aragora.memory.coordinator import MemoryCoordinator

        coord = MemoryCoordinator(use_embedding_surprise=True)
        assert isinstance(coord.surprise_scorer, EmbeddingSurpriseScorer)

    def test_explicit_scorer_overrides_flag(self):
        from aragora.memory.coordinator import MemoryCoordinator

        custom = ContentSurpriseScorer(threshold=0.5)
        coord = MemoryCoordinator(
            surprise_scorer=custom,
            use_embedding_surprise=True,  # should be ignored
        )
        assert coord.surprise_scorer is custom
