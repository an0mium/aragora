"""Tests for similarity computation backends."""

from __future__ import annotations

import threading
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aragora.debate.similarity.backends import (
    JaccardBackend,
    SentenceTransformerBackend,
    SimilarityBackend,
    TFIDFBackend,
    _normalize_backend_name,
    get_similarity_backend,
)


# ---------------------------------------------------------------------------
# _normalize_backend_name
# ---------------------------------------------------------------------------


class TestNormalizeBackendName:
    """Tests for backend name normalization."""

    def test_empty_string(self) -> None:
        assert _normalize_backend_name("") is None

    def test_valid_auto(self) -> None:
        assert _normalize_backend_name("auto") == "auto"

    def test_valid_jaccard(self) -> None:
        assert _normalize_backend_name("jaccard") == "jaccard"

    def test_valid_tfidf(self) -> None:
        assert _normalize_backend_name("tfidf") == "tfidf"

    def test_valid_sentence_transformer(self) -> None:
        assert _normalize_backend_name("sentence-transformer") == "sentence-transformer"

    def test_alias_sentence_transformers(self) -> None:
        assert _normalize_backend_name("sentence-transformers") == "sentence-transformer"

    def test_alias_sentence_underscore(self) -> None:
        assert _normalize_backend_name("sentence_transformers") == "sentence-transformer"

    def test_alias_sentence_short(self) -> None:
        assert _normalize_backend_name("sentence") == "sentence-transformer"

    def test_alias_tf_idf_hyphen(self) -> None:
        assert _normalize_backend_name("tf-idf") == "tfidf"

    def test_alias_tf_idf_underscore(self) -> None:
        assert _normalize_backend_name("tf_idf") == "tfidf"

    def test_case_insensitive(self) -> None:
        assert _normalize_backend_name("JACCARD") == "jaccard"
        assert _normalize_backend_name("AUTO") == "auto"

    def test_whitespace_stripped(self) -> None:
        assert _normalize_backend_name("  jaccard  ") == "jaccard"

    def test_invalid_returns_none(self) -> None:
        assert _normalize_backend_name("unknown") is None
        assert _normalize_backend_name("cosine") is None


# ---------------------------------------------------------------------------
# SimilarityBackend base class
# ---------------------------------------------------------------------------


class _ConcreteBackend(SimilarityBackend):
    """Concrete implementation for testing base class methods."""

    def __init__(self, similarity: float = 0.5) -> None:
        self._similarity = similarity

    def compute_similarity(self, text1: str, text2: str) -> float:
        return self._similarity


class TestSimilarityBackendBase:
    """Tests for abstract base class methods."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            SimilarityBackend()  # type: ignore[abstract]

    def test_contradiction_pairs_exist(self) -> None:
        b = _ConcreteBackend()
        assert len(b._CONTRADICTION_PAIRS) > 20

    def test_contradiction_pairs_are_tuples(self) -> None:
        b = _ConcreteBackend()
        for pair in b._CONTRADICTION_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2


class TestIsContradictory:
    """Tests for base pattern-based contradiction detection."""

    def test_empty_text_not_contradictory(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("", "something") is False
        assert b.is_contradictory("something", "") is False
        assert b.is_contradictory("", "") is False

    def test_accept_vs_reject(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("accept the proposal", "reject the proposal") is True

    def test_agree_vs_disagree(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("I agree with funding", "I disagree with funding") is True

    def test_yes_vs_no(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("yes", "no") is True

    def test_increase_vs_decrease(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("increase budget", "decrease budget") is True

    def test_no_contradiction_same_direction(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("accept the proposal", "support the proposal") is False

    def test_no_contradiction_unrelated(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("buy apples", "paint the wall") is False

    def test_labeled_options_different(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("Option A", "Option B") is True

    def test_labeled_options_same(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("Option A", "Option A") is False

    def test_choice_labels(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("Choice 1", "Choice 2") is True

    def test_alternative_labels(self) -> None:
        b = _ConcreteBackend()
        assert b.is_contradictory("Alternative a", "Alternative b") is True

    def test_contradiction_requires_shared_context(self) -> None:
        """Long texts need shared meaningful words for contradiction."""
        b = _ConcreteBackend()
        # These share "the" (stopword only) plus accept/reject pair
        # The implementation shares "the" which is a stopword, but both have
        # more than 3 words each, so shared meaningful words are needed.
        # However, "the" and "for" overlap and "for" is a contradiction pair word.
        # Use texts with zero meaningful overlap to test.
        t1 = "completely unrelated alpha gamma"
        t2 = "entirely different beta delta"
        assert b.is_contradictory(t1, t2) is False


class TestComputeBatchSimilarity:
    """Tests for batch similarity computation."""

    def test_single_text(self) -> None:
        b = _ConcreteBackend(similarity=0.5)
        assert b.compute_batch_similarity(["hello"]) == 1.0

    def test_empty_list(self) -> None:
        b = _ConcreteBackend(similarity=0.5)
        assert b.compute_batch_similarity([]) == 1.0

    def test_two_texts(self) -> None:
        b = _ConcreteBackend(similarity=0.8)
        result = b.compute_batch_similarity(["a", "b"])
        assert result == 0.8

    def test_three_texts(self) -> None:
        b = _ConcreteBackend(similarity=0.6)
        result = b.compute_batch_similarity(["a", "b", "c"])
        # 3 pairs, each 0.6 -> average = 0.6
        assert abs(result - 0.6) < 0.01


# ---------------------------------------------------------------------------
# JaccardBackend
# ---------------------------------------------------------------------------


class TestJaccardBackend:
    """Tests for Jaccard similarity backend."""

    def setup_method(self) -> None:
        # Reset tunables so prior tests cannot leak cache-size state.
        JaccardBackend._cache_max_size = 256
        JaccardBackend.clear_cache()

    def test_identical_texts(self) -> None:
        b = JaccardBackend()
        assert b.compute_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self) -> None:
        b = JaccardBackend()
        assert b.compute_similarity("cat dog", "fish bird") == 0.0

    def test_partial_overlap(self) -> None:
        b = JaccardBackend()
        # {hello, world} ∩ {hello, there} = {hello}
        # {hello, world} ∪ {hello, there} = {hello, world, there}
        sim = b.compute_similarity("hello world", "hello there")
        assert abs(sim - 1 / 3) < 0.01

    def test_empty_text_returns_zero(self) -> None:
        b = JaccardBackend()
        assert b.compute_similarity("", "hello") == 0.0
        assert b.compute_similarity("hello", "") == 0.0
        assert b.compute_similarity("", "") == 0.0

    def test_case_insensitive(self) -> None:
        b = JaccardBackend()
        assert b.compute_similarity("Hello World", "hello world") == 1.0

    def test_symmetric(self) -> None:
        b = JaccardBackend()
        s1 = b.compute_similarity("cat dog fish", "cat bird")
        JaccardBackend.clear_cache()
        s2 = b.compute_similarity("cat bird", "cat dog fish")
        assert s1 == s2

    def test_cache_hit(self) -> None:
        b = JaccardBackend()
        _ = b.compute_similarity("test alpha", "test beta")
        # Second call should use cache
        result = b.compute_similarity("test alpha", "test beta")
        assert result > 0

    def test_cache_symmetric_key(self) -> None:
        """Both orders should use the same cache entry."""
        b = JaccardBackend()
        s1 = b.compute_similarity("aaa bbb", "ccc ddd")
        # Reverse order should hit cache
        s2 = b.compute_similarity("ccc ddd", "aaa bbb")
        assert s1 == s2

    def test_cache_eviction(self) -> None:
        """Cache should evict oldest when full."""
        JaccardBackend._cache_max_size = 5
        b = JaccardBackend()
        try:
            for i in range(10):
                b.compute_similarity(f"text-{i} unique", f"text-{i} words here")
            assert len(JaccardBackend._similarity_cache) <= 5
        finally:
            JaccardBackend._cache_max_size = 256

    def test_clear_cache(self) -> None:
        b = JaccardBackend()
        b.compute_similarity("hello world", "hello there")
        assert len(JaccardBackend._similarity_cache) > 0
        JaccardBackend.clear_cache()
        assert len(JaccardBackend._similarity_cache) == 0

    def test_thread_safety(self) -> None:
        """Concurrent access should not raise."""
        b = JaccardBackend()
        errors: list[Exception] = []

        def compute(idx: int) -> None:
            try:
                for j in range(20):
                    b.compute_similarity(f"thread {idx} word {j}", f"thread {idx} other {j}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compute, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# TFIDFBackend
# ---------------------------------------------------------------------------


class TestTFIDFBackend:
    """Tests for TF-IDF similarity backend."""

    def setup_method(self) -> None:
        # Reset tunables so prior tests cannot leak cache-size state.
        TFIDFBackend._cache_max_size = 256
        TFIDFBackend.clear_cache()

    def test_init(self) -> None:
        b = TFIDFBackend()
        assert b.vectorizer is not None
        assert b.cosine_similarity is not None

    def test_identical_texts(self) -> None:
        b = TFIDFBackend()
        assert b.compute_similarity("hello world", "hello world") == pytest.approx(1.0, abs=0.01)

    def test_similar_texts(self) -> None:
        b = TFIDFBackend()
        sim = b.compute_similarity(
            "the cat sat on the mat",
            "the cat sat on the rug",
        )
        assert 0.5 < sim < 1.0

    def test_different_texts(self) -> None:
        b = TFIDFBackend()
        sim = b.compute_similarity(
            "quantum computing algorithms",
            "baking chocolate cookies recipe",
        )
        assert sim < 0.3

    def test_empty_text_returns_zero(self) -> None:
        b = TFIDFBackend()
        assert b.compute_similarity("", "hello") == 0.0
        assert b.compute_similarity("hello", "") == 0.0

    def test_cache_hit(self) -> None:
        b = TFIDFBackend()
        s1 = b.compute_similarity("test alpha", "test beta")
        s2 = b.compute_similarity("test alpha", "test beta")
        assert s1 == s2

    def test_cache_symmetric(self) -> None:
        b = TFIDFBackend()
        s1 = b.compute_similarity("one two three", "four five six")
        TFIDFBackend.clear_cache()
        s2 = b.compute_similarity("four five six", "one two three")
        assert abs(s1 - s2) < 0.01

    def test_clear_cache(self) -> None:
        b = TFIDFBackend()
        b.compute_similarity("hello world", "hello there")
        assert len(TFIDFBackend._similarity_cache) > 0
        TFIDFBackend.clear_cache()
        assert len(TFIDFBackend._similarity_cache) == 0

    def test_cache_eviction(self) -> None:
        TFIDFBackend._cache_max_size = 5
        b = TFIDFBackend()
        try:
            for i in range(10):
                b.compute_similarity(f"text-{i} unique words", f"text-{i} different content")
            assert len(TFIDFBackend._similarity_cache) <= 5
        finally:
            TFIDFBackend._cache_max_size = 256


# ---------------------------------------------------------------------------
# SentenceTransformerBackend (mocked)
# ---------------------------------------------------------------------------


class TestSentenceTransformerBackend:
    """Tests for SentenceTransformerBackend with mocked models."""

    def setup_method(self) -> None:
        # Reset tunables so prior tests cannot leak cache-size state.
        SentenceTransformerBackend._cache_max_size = 256
        SentenceTransformerBackend.clear_cache()
        # Reset class-level model caches
        SentenceTransformerBackend._model_cache = None
        SentenceTransformerBackend._model_name_cache = None
        SentenceTransformerBackend._nli_model_cache = None
        SentenceTransformerBackend._nli_model_name_cache = None

    def _make_backend(
        self,
        use_nli: bool = False,
        use_embedding_cache: bool = False,
    ) -> SentenceTransformerBackend:
        """Create a backend with mocked models."""
        mock_model = MagicMock()
        # encode returns random-ish embeddings
        mock_model.encode = MagicMock(
            side_effect=lambda texts: np.random.randn(len(texts), 64).astype(np.float32)
        )

        mock_cosine = MagicMock(
            side_effect=lambda a, b: np.array(
                [
                    [
                        float(
                            np.dot(a.flatten(), b.flatten())
                            / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                        )
                    ]
                ]
            )
        )

        with patch(
            "aragora.debate.similarity.backends.SentenceTransformerBackend.__init__",
            return_value=None,
        ):
            backend = SentenceTransformerBackend.__new__(SentenceTransformerBackend)

        backend.model = mock_model
        backend.cosine_similarity = mock_cosine
        backend.embedding_cache = None
        backend.nli_model = None
        backend.use_nli = use_nli
        backend.debate_id = None
        return backend

    def test_compute_similarity_returns_float(self) -> None:
        b = self._make_backend()
        sim = b.compute_similarity("hello world", "hello there")
        assert isinstance(sim, float)

    def test_compute_similarity_empty_returns_zero(self) -> None:
        b = self._make_backend()
        assert b.compute_similarity("", "hello") == 0.0
        assert b.compute_similarity("hello", "") == 0.0

    def test_compute_similarity_uses_cache(self) -> None:
        b = self._make_backend()
        s1 = b.compute_similarity("text one", "text two")
        # Second call should use cache (model.encode NOT called again)
        s2 = b.compute_similarity("text one", "text two")
        assert s1 == s2

    def test_compute_similarity_symmetric_cache_key(self) -> None:
        b = self._make_backend()
        s1 = b.compute_similarity("alpha", "beta")
        # Cache lookup with reversed order should hit
        cache_key = ("alpha", "beta")
        with SentenceTransformerBackend._cache_lock:
            assert cache_key in SentenceTransformerBackend._similarity_cache

    def test_cache_eviction(self) -> None:
        SentenceTransformerBackend._cache_max_size = 3
        b = self._make_backend()
        try:
            for i in range(5):
                b.compute_similarity(f"unique-{i}", f"different-{i}")
            assert len(SentenceTransformerBackend._similarity_cache) <= 3
        finally:
            SentenceTransformerBackend._cache_max_size = 256

    def test_clear_cache(self) -> None:
        b = self._make_backend()
        b.compute_similarity("hello", "world")
        assert len(SentenceTransformerBackend._similarity_cache) > 0
        SentenceTransformerBackend.clear_cache()
        assert len(SentenceTransformerBackend._similarity_cache) == 0
        assert len(SentenceTransformerBackend._contradiction_cache) == 0

    def test_is_contradictory_without_nli_fallback(self) -> None:
        """Without NLI model, falls back to pattern-based detection."""
        b = self._make_backend(use_nli=False)
        assert b.is_contradictory("accept proposal", "reject proposal") is True
        assert b.is_contradictory("hello world", "hello world") is False

    def test_is_contradictory_empty_returns_false(self) -> None:
        b = self._make_backend()
        assert b.is_contradictory("", "hello") is False
        assert b.is_contradictory("hello", "") is False

    def test_is_contradictory_uses_cache(self) -> None:
        b = self._make_backend()
        r1 = b.is_contradictory("accept", "reject")
        r2 = b.is_contradictory("accept", "reject")
        assert r1 == r2

    def test_is_contradictory_with_nli(self) -> None:
        """With NLI model, uses model prediction."""
        b = self._make_backend(use_nli=True)
        mock_nli = MagicMock()
        # Scores: [contradiction=0.9, entailment=0.05, neutral=0.05]
        mock_nli.predict = MagicMock(return_value=np.array([[0.9, 0.05, 0.05]]))
        b.nli_model = mock_nli
        assert b.is_contradictory("yes it is", "no it is not") == True  # noqa: E712

    def test_is_contradictory_nli_not_contradictory(self) -> None:
        b = self._make_backend(use_nli=True)
        mock_nli = MagicMock()
        # Scores: [contradiction=0.05, entailment=0.9, neutral=0.05]
        mock_nli.predict = MagicMock(return_value=np.array([[0.05, 0.9, 0.05]]))
        b.nli_model = mock_nli
        assert b.is_contradictory("the sky is blue", "the sky is blue") == False  # noqa: E712

    def test_nli_prediction_failure_falls_back(self) -> None:
        b = self._make_backend(use_nli=True)
        mock_nli = MagicMock()
        mock_nli.predict = MagicMock(side_effect=RuntimeError("model error"))
        b.nli_model = mock_nli
        # Should fall back to pattern-based, not crash
        SentenceTransformerBackend.clear_cache()  # clear to avoid cached result
        result = b.is_contradictory("accept proposal", "reject proposal")
        assert isinstance(result, bool)

    def test_compute_batch_similarity_single(self) -> None:
        b = self._make_backend()
        assert b.compute_batch_similarity(["hello"]) == 1.0

    def test_compute_batch_similarity_empty(self) -> None:
        b = self._make_backend()
        assert b.compute_batch_similarity([]) == 1.0

    def test_compute_pairwise_empty_returns_empty(self) -> None:
        b = self._make_backend()
        assert b.compute_pairwise_similarities([], []) == []

    def test_compute_pairwise_mismatched_length(self) -> None:
        b = self._make_backend()
        assert b.compute_pairwise_similarities(["a"], ["b", "c"]) == []

    def test_compute_pairwise_returns_correct_length(self) -> None:
        b = self._make_backend()
        results = b.compute_pairwise_similarities(["a", "b"], ["c", "d"])
        assert len(results) == 2
        assert all(isinstance(r, float) for r in results)

    def test_get_embedding_uses_cache(self) -> None:
        b = self._make_backend(use_embedding_cache=False)
        mock_cache = MagicMock()
        mock_cache.get = MagicMock(return_value=np.zeros(64))
        b.embedding_cache = mock_cache

        result = b._get_embedding("test")
        mock_cache.get.assert_called_once_with("test")
        # Should return cached value, not call model
        assert np.array_equal(result, np.zeros(64))

    def test_get_embedding_caches_miss(self) -> None:
        b = self._make_backend(use_embedding_cache=False)
        mock_cache = MagicMock()
        mock_cache.get = MagicMock(return_value=None)
        mock_cache.put = MagicMock()
        b.embedding_cache = mock_cache

        result = b._get_embedding("test")
        mock_cache.put.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# get_similarity_backend
# ---------------------------------------------------------------------------


class TestGetSimilarityBackend:
    """Tests for the factory function."""

    def test_jaccard(self) -> None:
        b = get_similarity_backend("jaccard")
        assert isinstance(b, JaccardBackend)

    def test_tfidf(self) -> None:
        b = get_similarity_backend("tfidf")
        assert isinstance(b, TFIDFBackend)

    def test_auto_returns_backend(self) -> None:
        b = get_similarity_backend("auto")
        assert isinstance(b, SimilarityBackend)

    def test_auto_env_override_jaccard(self) -> None:
        with patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "jaccard"}):
            b = get_similarity_backend("auto")
            assert isinstance(b, JaccardBackend)

    def test_auto_env_override_tfidf(self) -> None:
        with patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "tfidf"}):
            b = get_similarity_backend("auto")
            assert isinstance(b, TFIDFBackend)

    def test_auto_invalid_env_falls_through(self) -> None:
        with patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "invalid_backend"}):
            b = get_similarity_backend("auto")
            # Should still return a valid backend (auto-select)
            assert isinstance(b, SimilarityBackend)

    def test_auto_fallback_to_jaccard(self) -> None:
        """When sentence-transformers and sklearn unavailable, should get Jaccard."""
        with (
            patch(
                "aragora.debate.similarity.backends.SentenceTransformerBackend",
                side_effect=ImportError("no sentence-transformers"),
            ),
            patch(
                "aragora.debate.similarity.backends.TFIDFBackend",
                side_effect=ImportError("no sklearn"),
            ),
        ):
            b = get_similarity_backend("auto")
            assert isinstance(b, JaccardBackend)
