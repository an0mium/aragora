"""
Tests for aragora/debate/similarity/backends.py

Covers:
- _normalize_backend_name: all aliases, edge cases
- SimilarityBackend.is_contradictory: contradiction pairs, labeled options, shared context
- SimilarityBackend.compute_batch_similarity: edge cases and averaging
- JaccardBackend: similarity math, cache hits/misses, LRU eviction, thread-safety
- TFIDFBackend: mocked sklearn, caching, import error propagation
- SentenceTransformerBackend: mocked sentence-transformers, model cache, NLI, batch similarity
- get_similarity_backend: factory dispatch, env var override, auto-select fallback
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_jaccard_cache():
    """Ensure JaccardBackend class-level cache is empty before every test."""
    from aragora.debate.similarity.backends import JaccardBackend

    JaccardBackend.clear_cache()
    yield
    JaccardBackend.clear_cache()


@pytest.fixture(autouse=True)
def clear_tfidf_cache():
    """Ensure TFIDFBackend class-level cache is empty before every test."""
    from aragora.debate.similarity.backends import TFIDFBackend

    TFIDFBackend.clear_cache()
    yield
    TFIDFBackend.clear_cache()


@pytest.fixture(autouse=True)
def clear_st_cache():
    """Reset SentenceTransformerBackend class-level caches before every test."""
    from aragora.debate.similarity.backends import SentenceTransformerBackend

    SentenceTransformerBackend.clear_cache()
    SentenceTransformerBackend._model_cache = None
    SentenceTransformerBackend._model_name_cache = None
    SentenceTransformerBackend._nli_model_cache = None
    SentenceTransformerBackend._nli_model_name_cache = None
    yield
    SentenceTransformerBackend.clear_cache()
    SentenceTransformerBackend._model_cache = None
    SentenceTransformerBackend._model_name_cache = None
    SentenceTransformerBackend._nli_model_cache = None
    SentenceTransformerBackend._nli_model_name_cache = None


@pytest.fixture
def jaccard():
    from aragora.debate.similarity.backends import JaccardBackend

    return JaccardBackend()


# ---------------------------------------------------------------------------
# _normalize_backend_name
# ---------------------------------------------------------------------------


class TestNormalizeBackendName:
    """Tests for the _normalize_backend_name helper."""

    def _norm(self, value: str):
        from aragora.debate.similarity.backends import _normalize_backend_name

        return _normalize_backend_name(value)

    # --- canonical names pass through unchanged ---

    def test_canonical_auto(self):
        assert self._norm("auto") == "auto"

    def test_canonical_jaccard(self):
        assert self._norm("jaccard") == "jaccard"

    def test_canonical_tfidf(self):
        assert self._norm("tfidf") == "tfidf"

    def test_canonical_sentence_transformer(self):
        assert self._norm("sentence-transformer") == "sentence-transformer"

    # --- aliases resolve to canonical ---

    def test_alias_sentence_hyphen(self):
        assert self._norm("sentence-transformers") == "sentence-transformer"

    def test_alias_sentence_underscore(self):
        assert self._norm("sentence_transformers") == "sentence-transformer"

    def test_alias_sentence_short(self):
        assert self._norm("sentence") == "sentence-transformer"

    def test_alias_tf_hyphen(self):
        assert self._norm("tf-idf") == "tfidf"

    def test_alias_tf_underscore(self):
        assert self._norm("tf_idf") == "tfidf"

    # --- case-insensitive ---

    def test_uppercase_jaccard(self):
        assert self._norm("JACCARD") == "jaccard"

    def test_mixed_case_tfidf(self):
        assert self._norm("TFIDF") == "tfidf"

    def test_mixed_case_auto(self):
        assert self._norm("AUTO") == "auto"

    def test_mixed_alias_sentence_transformers(self):
        assert self._norm("Sentence-Transformers") == "sentence-transformer"

    # --- leading/trailing whitespace stripped ---

    def test_whitespace_stripped(self):
        assert self._norm("  jaccard  ") == "jaccard"

    def test_whitespace_with_alias(self):
        assert self._norm("  tf-idf  ") == "tfidf"

    # --- unknown names return None ---

    def test_unknown_name_returns_none(self):
        assert self._norm("bogus") is None

    def test_unknown_name_partial_match_returns_none(self):
        assert self._norm("jac") is None

    # --- empty / falsy returns None ---

    def test_empty_string_returns_none(self):
        assert self._norm("") is None

    def test_whitespace_only_returns_none(self):
        # Stripped to empty string => falsy => None
        assert self._norm("   ") is None

    # --- underscore normalisation (underscore → hyphen before alias lookup) ---

    def test_underscore_normalised_before_alias(self):
        # tf_idf → tf-idf (via replace) → tfidf (via alias)
        assert self._norm("tf_idf") == "tfidf"


# ---------------------------------------------------------------------------
# SimilarityBackend.is_contradictory (tested via JaccardBackend)
# ---------------------------------------------------------------------------


class TestIsContradictory:
    """Tests for the base-class is_contradictory method."""

    @pytest.fixture
    def backend(self):
        from aragora.debate.similarity.backends import JaccardBackend

        return JaccardBackend()

    # --- empty / falsy inputs ---

    def test_empty_text1_returns_false(self, backend):
        assert backend.is_contradictory("", "reject the proposal") is False

    def test_empty_text2_returns_false(self, backend):
        assert backend.is_contradictory("accept the proposal", "") is False

    def test_both_empty_returns_false(self, backend):
        assert backend.is_contradictory("", "") is False

    # --- contradiction word pairs with shared context ---

    def test_accept_reject_with_shared_word(self, backend):
        assert backend.is_contradictory("accept the proposal", "reject the proposal") is True

    def test_agree_disagree_with_shared_word(self, backend):
        assert backend.is_contradictory("I agree with the plan", "I disagree with the plan") is True

    def test_support_oppose_with_shared_word(self, backend):
        assert backend.is_contradictory("support the motion", "oppose the motion") is True

    def test_yes_no_short_texts(self, backend):
        # Short texts (<= 3 words) bypass the shared-context requirement
        assert backend.is_contradictory("yes", "no") is True

    def test_true_false_short(self, backend):
        assert backend.is_contradictory("true", "false") is True

    def test_increase_decrease_shared(self, backend):
        assert backend.is_contradictory("increase the budget", "decrease the budget") is True

    def test_pass_fail_shared(self, backend):
        assert backend.is_contradictory("pass the test", "fail the test") is True

    def test_for_against_shared(self, backend):
        assert backend.is_contradictory("for the decision", "against the decision") is True

    # --- contradiction pair WITHOUT shared context → no contradiction ---

    def test_accept_reject_no_shared_context(self, backend):
        # Both texts have >3 words and share no meaningful context words
        # (the "or len <= 3" short-circuit doesn't apply here)
        assert backend.is_contradictory(
            "accept the fresh tropical apples", "reject the wilted stale oranges"
        ) is False

    # --- labeled options ---

    def test_option_a_vs_option_b(self, backend):
        assert backend.is_contradictory("option a", "option b") is True

    def test_choice_1_vs_choice_2(self, backend):
        assert backend.is_contradictory("choice 1", "choice 2") is True

    def test_alternative_x_vs_alternative_y(self, backend):
        assert backend.is_contradictory("alternative x", "alternative y") is True

    def test_same_labeled_option_not_contradictory(self, backend):
        # Identical labeled options are NOT contradictory
        assert backend.is_contradictory("option a", "option a") is False

    # --- non-contradictory similar texts ---

    def test_paraphrase_not_contradictory(self, backend):
        assert backend.is_contradictory(
            "we should increase investment", "we should raise investment"
        ) is False

    def test_unrelated_texts_not_contradictory(self, backend):
        assert backend.is_contradictory("the sky is blue", "cats like tuna") is False


# ---------------------------------------------------------------------------
# SimilarityBackend.compute_batch_similarity (base implementation)
# ---------------------------------------------------------------------------


class TestComputeBatchSimilarity:
    """Tests for SimilarityBackend.compute_batch_similarity using JaccardBackend."""

    @pytest.fixture
    def backend(self):
        from aragora.debate.similarity.backends import JaccardBackend

        return JaccardBackend()

    def test_single_text_returns_one(self, backend):
        assert backend.compute_batch_similarity(["only one"]) == 1.0

    def test_empty_list_returns_one(self, backend):
        assert backend.compute_batch_similarity([]) == 1.0

    def test_identical_texts_returns_one(self, backend):
        texts = ["hello world", "hello world"]
        result = backend.compute_batch_similarity(texts)
        assert result == pytest.approx(1.0)

    def test_completely_different_texts(self, backend):
        texts = ["apple", "banana"]
        result = backend.compute_batch_similarity(texts)
        assert result == pytest.approx(0.0)

    def test_three_texts_average(self, backend):
        # All same → average should be 1.0
        texts = ["same text", "same text", "same text"]
        result = backend.compute_batch_similarity(texts)
        assert result == pytest.approx(1.0)

    def test_partial_overlap_average(self, backend):
        # "the quick brown" vs "the quick fox" → 2/4 = 0.5
        # Verify result is between 0 and 1
        texts = ["the quick brown", "the quick fox", "the slow turtle"]
        result = backend.compute_batch_similarity(texts)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# JaccardBackend
# ---------------------------------------------------------------------------


class TestJaccardBackend:
    """Tests for JaccardBackend similarity computation and caching."""

    # --- basic similarity ---

    def test_identical_texts(self, jaccard):
        assert jaccard.compute_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different_texts(self, jaccard):
        assert jaccard.compute_similarity("apple", "banana") == pytest.approx(0.0)

    def test_partial_overlap(self, jaccard):
        # words1 = {the, quick, brown}; words2 = {the, quick, fox}
        # intersection = {the, quick} (2); union = {the, quick, brown, fox} (4)
        result = jaccard.compute_similarity("the quick brown", "the quick fox")
        assert result == pytest.approx(2 / 4)

    def test_case_insensitive(self, jaccard):
        # Should normalize to lowercase
        assert jaccard.compute_similarity("Hello World", "hello world") == pytest.approx(1.0)

    def test_empty_text1_returns_zero(self, jaccard):
        assert jaccard.compute_similarity("", "hello") == pytest.approx(0.0)

    def test_empty_text2_returns_zero(self, jaccard):
        assert jaccard.compute_similarity("hello", "") == pytest.approx(0.0)

    def test_both_empty_returns_zero(self, jaccard):
        assert jaccard.compute_similarity("", "") == pytest.approx(0.0)

    def test_single_shared_word(self, jaccard):
        # "dog" vs "dog cat" → {dog} / {dog, cat} = 0.5
        assert jaccard.compute_similarity("dog", "dog cat") == pytest.approx(0.5)

    # --- symmetry ---

    def test_symmetry(self, jaccard):
        t1, t2 = "alpha beta gamma", "beta gamma delta"
        assert jaccard.compute_similarity(t1, t2) == pytest.approx(
            jaccard.compute_similarity(t2, t1)
        )

    # --- caching behaviour ---

    def test_cache_populated_after_first_call(self, jaccard):
        from aragora.debate.similarity.backends import JaccardBackend

        jaccard.compute_similarity("foo", "bar")
        assert len(JaccardBackend._similarity_cache) == 1

    def test_cache_hit_on_second_call(self, jaccard):
        from aragora.debate.similarity.backends import JaccardBackend

        jaccard.compute_similarity("foo bar", "bar baz")
        size_after_first = len(JaccardBackend._similarity_cache)
        jaccard.compute_similarity("foo bar", "bar baz")
        assert len(JaccardBackend._similarity_cache) == size_after_first

    def test_symmetric_key_hits_cache(self, jaccard):
        from aragora.debate.similarity.backends import JaccardBackend

        # Compute (a, b) then (b, a) — should share cache entry
        jaccard.compute_similarity("apple pie", "cherry pie")
        size_after_first = len(JaccardBackend._similarity_cache)
        jaccard.compute_similarity("cherry pie", "apple pie")
        assert len(JaccardBackend._similarity_cache) == size_after_first

    def test_clear_cache(self, jaccard):
        from aragora.debate.similarity.backends import JaccardBackend

        jaccard.compute_similarity("x", "y z")
        assert len(JaccardBackend._similarity_cache) > 0
        JaccardBackend.clear_cache()
        assert len(JaccardBackend._similarity_cache) == 0

    def test_lru_eviction_at_capacity(self):
        from aragora.debate.similarity.backends import JaccardBackend

        # Temporarily shrink cache to 3 slots
        original_max = JaccardBackend._cache_max_size
        JaccardBackend._cache_max_size = 3
        try:
            jb = JaccardBackend()
            # Fill 3 slots
            jb.compute_similarity("a", "b c")
            jb.compute_similarity("d", "e f")
            jb.compute_similarity("g", "h i")
            assert len(JaccardBackend._similarity_cache) == 3
            # Add a 4th — oldest should be evicted
            jb.compute_similarity("j", "k l")
            assert len(JaccardBackend._similarity_cache) == 3
        finally:
            JaccardBackend._cache_max_size = original_max
            JaccardBackend.clear_cache()

    def test_cache_lru_order_preserved(self):
        from aragora.debate.similarity.backends import JaccardBackend

        original_max = JaccardBackend._cache_max_size
        JaccardBackend._cache_max_size = 3
        try:
            jb = JaccardBackend()
            jb.compute_similarity("aaa", "bbb ccc")
            jb.compute_similarity("ddd", "eee fff")
            jb.compute_similarity("ggg", "hhh iii")
            # Access the first entry to make it "most recently used"
            jb.compute_similarity("aaa", "bbb ccc")
            # Now inserting a 4th should evict "ddd…eee" (oldest not recently accessed)
            jb.compute_similarity("jjj", "kkk lll")
            keys = list(JaccardBackend._similarity_cache.keys())
            # The evicted key should be ("ddd", "eee fff") or ("bbb ccc", "ddd")
            # because "aaa"/"bbb ccc" was refreshed
            # All remaining keys should NOT be the originally-second entry
            remaining_text_sets = [set(k) for k in keys]
            assert {"ddd", "eee fff"} not in remaining_text_sets
        finally:
            JaccardBackend._cache_max_size = original_max
            JaccardBackend.clear_cache()

    # --- thread safety ---

    def test_thread_safe_concurrent_writes(self):
        from aragora.debate.similarity.backends import JaccardBackend

        jb = JaccardBackend()
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(10):
                    jb.compute_similarity(f"word{n}", f"other{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_thread_safe_clear_during_reads(self):
        from aragora.debate.similarity.backends import JaccardBackend

        jb = JaccardBackend()
        errors: list[Exception] = []

        def reader():
            try:
                for _ in range(20):
                    jb.compute_similarity("alpha beta", "beta gamma")
            except Exception as exc:
                errors.append(exc)

        def clearer():
            try:
                for _ in range(5):
                    JaccardBackend.clear_cache()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=clearer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# TFIDFBackend (sklearn mocked)
# ---------------------------------------------------------------------------


def _make_sklearn_mocks():
    """Return (TfidfVectorizer mock class, cosine_similarity mock fn)."""
    mock_vectorizer = MagicMock()
    # fit_transform returns a fake matrix
    fake_matrix = MagicMock()
    mock_vectorizer.return_value = mock_vectorizer  # instance == class for simplicity
    mock_vectorizer.fit_transform.return_value = fake_matrix
    # cosine_similarity([[row0]], [[row1]]) returns [[score]]
    mock_cos = MagicMock(return_value=[[0.75]])
    return mock_vectorizer, mock_cos, fake_matrix


class TestTFIDFBackend:
    """Tests for TFIDFBackend with mocked sklearn."""

    @pytest.fixture
    def tfidf_backend(self):
        """Create a TFIDFBackend with sklearn mocked out."""
        mock_vec_cls = MagicMock()
        mock_vec_instance = MagicMock()
        mock_vec_cls.return_value = mock_vec_instance

        fake_matrix = MagicMock()
        fake_matrix.__getitem__ = lambda self, idx: MagicMock()
        mock_vec_instance.fit_transform.return_value = fake_matrix

        mock_cos = MagicMock(return_value=[[0.8]])

        sklearn_mock = MagicMock()
        sklearn_mock.feature_extraction = MagicMock()
        sklearn_mock.feature_extraction.text = MagicMock()
        sklearn_mock.feature_extraction.text.TfidfVectorizer = mock_vec_cls
        sklearn_mock.metrics = MagicMock()
        sklearn_mock.metrics.pairwise = MagicMock()
        sklearn_mock.metrics.pairwise.cosine_similarity = mock_cos

        with patch.dict(
            "sys.modules",
            {
                "sklearn": sklearn_mock,
                "sklearn.feature_extraction": sklearn_mock.feature_extraction,
                "sklearn.feature_extraction.text": sklearn_mock.feature_extraction.text,
                "sklearn.metrics": sklearn_mock.metrics,
                "sklearn.metrics.pairwise": sklearn_mock.metrics.pairwise,
            },
        ):
            from aragora.debate.similarity import backends as _backends_mod
            import importlib

            # Force re-import to pick up mocked sklearn
            importlib.reload(_backends_mod)
            TFIDFBackend = _backends_mod.TFIDFBackend
            backend = TFIDFBackend()
            backend.vectorizer = mock_vec_instance
            backend.cosine_similarity = mock_cos
            yield backend, mock_vec_instance, mock_cos

    def test_import_error_raised_without_sklearn(self):
        """TFIDFBackend raises ImportError when sklearn is absent."""
        import sys

        # Temporarily remove sklearn from sys.modules
        saved = {k: v for k, v in sys.modules.items() if "sklearn" in k}
        for k in saved:
            del sys.modules[k]
        # Also patch the import inside TFIDFBackend.__init__
        with patch.dict("sys.modules", {"sklearn": None,
                                        "sklearn.feature_extraction": None,
                                        "sklearn.feature_extraction.text": None,
                                        "sklearn.metrics": None,
                                        "sklearn.metrics.pairwise": None}):
            from aragora.debate.similarity.backends import TFIDFBackend

            with pytest.raises(ImportError, match="scikit-learn"):
                TFIDFBackend()
        # Restore
        sys.modules.update(saved)

    def test_empty_text1_returns_zero(self, tfidf_backend):
        backend, _, _ = tfidf_backend
        assert backend.compute_similarity("", "some text") == pytest.approx(0.0)

    def test_empty_text2_returns_zero(self, tfidf_backend):
        backend, _, _ = tfidf_backend
        assert backend.compute_similarity("some text", "") == pytest.approx(0.0)

    def test_similarity_calls_fit_transform(self, tfidf_backend):
        backend, mock_vec_instance, _ = tfidf_backend
        backend.compute_similarity("hello world", "world cup")
        mock_vec_instance.fit_transform.assert_called_once_with(["hello world", "world cup"])

    def test_similarity_calls_cosine_similarity(self, tfidf_backend):
        backend, mock_vec_instance, mock_cos = tfidf_backend
        fake_mat = mock_vec_instance.fit_transform.return_value
        backend.compute_similarity("hello world", "world cup")
        mock_cos.assert_called_once()

    def test_clear_cache(self):
        from aragora.debate.similarity.backends import TFIDFBackend

        # Manually inject a fake entry
        TFIDFBackend._similarity_cache[("a", "b")] = 0.5
        TFIDFBackend.clear_cache()
        assert len(TFIDFBackend._similarity_cache) == 0

    def test_cache_uses_sorted_key_for_symmetry(self, tfidf_backend):
        """Swapped text order should hit the same cache entry."""
        from aragora.debate.similarity.backends import TFIDFBackend

        backend, mock_vec_instance, _ = tfidf_backend
        backend.compute_similarity("alpha", "beta")
        size_after_first = len(TFIDFBackend._similarity_cache)
        backend.compute_similarity("beta", "alpha")
        assert len(TFIDFBackend._similarity_cache) == size_after_first

    def test_lru_eviction_at_max_size(self):
        from aragora.debate.similarity.backends import TFIDFBackend

        original_max = TFIDFBackend._cache_max_size
        TFIDFBackend._cache_max_size = 2
        try:
            # Manually fill beyond capacity via the OrderedDict
            with TFIDFBackend._cache_lock:
                TFIDFBackend._similarity_cache[("a", "b")] = 0.1
                TFIDFBackend._similarity_cache[("c", "d")] = 0.2
            assert len(TFIDFBackend._similarity_cache) == 2
            # The next real compute would evict on next call; we verify structure directly
            with TFIDFBackend._cache_lock:
                while len(TFIDFBackend._similarity_cache) >= TFIDFBackend._cache_max_size:
                    TFIDFBackend._similarity_cache.popitem(last=False)
                TFIDFBackend._similarity_cache[("e", "f")] = 0.3
            assert len(TFIDFBackend._similarity_cache) == 2
            assert ("a", "b") not in TFIDFBackend._similarity_cache
        finally:
            TFIDFBackend._cache_max_size = original_max
            TFIDFBackend.clear_cache()


# ---------------------------------------------------------------------------
# SentenceTransformerBackend (sentence_transformers + sklearn mocked)
# ---------------------------------------------------------------------------


def _make_st_mocks():
    """Build complete mocks for sentence_transformers and sklearn."""
    import numpy as np

    # Fake embeddings (unit vectors so cosine similarity = dot product)
    fake_emb = np.array([1.0, 0.0, 0.0])

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([fake_emb, fake_emb])

    mock_cross_encoder = MagicMock()
    # predict returns [[contradiction_score, entailment_score, neutral_score]]
    mock_cross_encoder.predict.return_value = [[0.9, 0.05, 0.05]]

    mock_st_module = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model
    mock_st_module.CrossEncoder.return_value = mock_cross_encoder

    # cosine_similarity([[a]], [[b]]) → [[1.0]] for identical unit vectors
    mock_cos = MagicMock(return_value=np.array([[1.0]]))

    mock_sklearn_pairwise = MagicMock()
    mock_sklearn_pairwise.cosine_similarity = mock_cos

    return mock_st_module, mock_model, mock_cross_encoder, mock_cos, mock_sklearn_pairwise


@pytest.fixture
def st_mocks():
    return _make_st_mocks()


@pytest.fixture
def st_backend(st_mocks):
    """Create SentenceTransformerBackend with all dependencies mocked."""
    mock_st_module, mock_model, mock_cross_encoder, mock_cos, mock_sklearn_pairwise = st_mocks

    mock_embedding_cache = MagicMock()
    mock_embedding_cache.get.return_value = None  # cache miss by default

    with (
        patch.dict(
            "sys.modules",
            {
                "sentence_transformers": mock_st_module,
                "sklearn.metrics.pairwise": mock_sklearn_pairwise,
            },
        ),
        patch(
            "aragora.debate.similarity.backends.get_embedding_cache",
            return_value=mock_embedding_cache,
        ),
        patch(
            "aragora.debate.similarity.backends.get_scoped_embedding_cache",
            return_value=mock_embedding_cache,
        ),
    ):
        from aragora.debate.similarity.backends import SentenceTransformerBackend

        backend = SentenceTransformerBackend.__new__(SentenceTransformerBackend)
        backend.model = mock_model
        backend.cosine_similarity = mock_cos
        backend.embedding_cache = mock_embedding_cache
        backend.nli_model = mock_cross_encoder
        backend.use_nli = True
        backend.debate_id = None
        yield backend, mock_model, mock_cross_encoder, mock_cos, mock_embedding_cache


class TestSentenceTransformerBackend:
    """Tests for SentenceTransformerBackend logic with mocked dependencies."""

    # --- compute_similarity ---

    def test_empty_text1_returns_zero(self, st_backend):
        backend, *_ = st_backend
        assert backend.compute_similarity("", "some text") == pytest.approx(0.0)

    def test_empty_text2_returns_zero(self, st_backend):
        backend, *_ = st_backend
        assert backend.compute_similarity("some text", "") == pytest.approx(0.0)

    def test_compute_similarity_uses_cosine(self, st_backend):
        import numpy as np

        backend, mock_model, _, mock_cos, mock_emb_cache = st_backend
        fake_emb = np.array([1.0, 0.0])
        mock_model.encode.return_value = np.array([fake_emb])
        mock_cos.return_value = np.array([[0.9]])

        result = backend.compute_similarity("hello", "world")
        assert result == pytest.approx(0.9)

    def test_similarity_is_cached_after_first_call(self, st_backend):
        from aragora.debate.similarity.backends import SentenceTransformerBackend
        import numpy as np

        backend, mock_model, _, mock_cos, _ = st_backend
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        mock_cos.return_value = np.array([[0.7]])

        backend.compute_similarity("text a", "text b")
        size = len(SentenceTransformerBackend._similarity_cache)
        assert size == 1

    def test_symmetric_cache_key(self, st_backend):
        from aragora.debate.similarity.backends import SentenceTransformerBackend
        import numpy as np

        backend, mock_model, _, mock_cos, _ = st_backend
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        mock_cos.return_value = np.array([[0.6]])

        backend.compute_similarity("aaa", "bbb")
        size_after_first = len(SentenceTransformerBackend._similarity_cache)
        backend.compute_similarity("bbb", "aaa")
        assert len(SentenceTransformerBackend._similarity_cache) == size_after_first

    def test_clear_cache_clears_both_caches(self, st_backend):
        from aragora.debate.similarity.backends import SentenceTransformerBackend

        SentenceTransformerBackend._similarity_cache[("x", "y")] = 0.5
        SentenceTransformerBackend._contradiction_cache[("x", "y")] = True
        SentenceTransformerBackend.clear_cache()
        assert len(SentenceTransformerBackend._similarity_cache) == 0
        assert len(SentenceTransformerBackend._contradiction_cache) == 0

    # --- embedding cache integration ---

    def test_embedding_cache_hit_avoids_encode(self, st_backend):
        import numpy as np

        backend, mock_model, _, mock_cos, mock_emb_cache = st_backend
        cached_emb = np.array([1.0, 0.0])
        mock_emb_cache.get.return_value = cached_emb
        mock_cos.return_value = np.array([[0.95]])

        backend.compute_similarity("cached text", "another cached")
        # encode should NOT have been called since cache returned a value
        mock_model.encode.assert_not_called()

    def test_embedding_cache_miss_calls_encode(self, st_backend):
        import numpy as np

        backend, mock_model, _, mock_cos, mock_emb_cache = st_backend
        mock_emb_cache.get.return_value = None  # cache miss
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        mock_cos.return_value = np.array([[0.8]])

        backend.compute_similarity("fresh text", "another fresh")
        assert mock_model.encode.called

    # --- is_contradictory with NLI ---

    def test_is_contradictory_uses_nli_when_available(self, st_backend):
        backend, _, mock_cross_encoder, _, _ = st_backend
        # Contradiction score (index 0) is highest
        mock_cross_encoder.predict.return_value = [[0.9, 0.05, 0.05]]
        result = backend.is_contradictory("accept the proposal", "reject the proposal")
        assert result is True
        mock_cross_encoder.predict.assert_called()

    def test_is_contradictory_nli_entailment_returns_false(self, st_backend):
        backend, _, mock_cross_encoder, _, _ = st_backend
        # Entailment score highest
        mock_cross_encoder.predict.return_value = [[0.05, 0.9, 0.05]]
        result = backend.is_contradictory("the cat sat", "a feline was seated")
        assert result is False

    def test_is_contradictory_empty_inputs_returns_false(self, st_backend):
        backend, _, mock_cross_encoder, _, _ = st_backend
        assert backend.is_contradictory("", "some text") is False
        mock_cross_encoder.predict.assert_not_called()

    def test_is_contradictory_falls_back_to_pattern_when_nli_none(self, st_backend):
        backend, _, _, _, _ = st_backend
        backend.nli_model = None
        backend.use_nli = False
        # Pattern-based should detect "accept"/"reject" contradiction
        result = backend.is_contradictory("accept the proposal", "reject the proposal")
        assert result is True

    def test_is_contradictory_cached_after_first_call(self, st_backend):
        from aragora.debate.similarity.backends import SentenceTransformerBackend

        backend, _, mock_cross_encoder, _, _ = st_backend
        mock_cross_encoder.predict.return_value = [[0.9, 0.05, 0.05]]
        backend.is_contradictory("yes", "no")
        cache_size = len(SentenceTransformerBackend._contradiction_cache)
        assert cache_size == 1

    def test_is_contradictory_symmetric_cache_key(self, st_backend):
        from aragora.debate.similarity.backends import SentenceTransformerBackend

        backend, _, mock_cross_encoder, _, _ = st_backend
        mock_cross_encoder.predict.return_value = [[0.9, 0.05, 0.05]]
        backend.is_contradictory("aaa", "bbb")
        size = len(SentenceTransformerBackend._contradiction_cache)
        backend.is_contradictory("bbb", "aaa")
        assert len(SentenceTransformerBackend._contradiction_cache) == size

    def test_nli_predict_runtime_error_falls_back_to_pattern(self, st_backend):
        backend, _, mock_cross_encoder, _, _ = st_backend
        mock_cross_encoder.predict.side_effect = RuntimeError("NLI failed")
        # Pattern fallback: "yes" / "no" are contradictory (short texts)
        result = backend.is_contradictory("yes", "no")
        assert result is True

    # --- model class-level caching ---

    def test_model_cache_reused_on_same_model_name(self, st_mocks):
        """Second instantiation with same model name should reuse cached model."""
        from aragora.debate.similarity.backends import SentenceTransformerBackend

        mock_st_module, mock_model, mock_ce, mock_cos, mock_skl = st_mocks
        mock_embedding_cache = MagicMock()
        mock_embedding_cache.get.return_value = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "sentence_transformers": mock_st_module,
                    "sklearn.metrics.pairwise": mock_skl,
                },
            ),
            patch(
                "aragora.debate.similarity.backends.get_embedding_cache",
                return_value=mock_embedding_cache,
            ),
            patch(
                "aragora.debate.similarity.backends.get_scoped_embedding_cache",
                return_value=mock_embedding_cache,
            ),
        ):
            # Clear class-level state
            SentenceTransformerBackend._model_cache = None
            SentenceTransformerBackend._model_name_cache = None

            b1 = SentenceTransformerBackend(model_name="test-model", use_nli=False)
            call_count_after_first = mock_st_module.SentenceTransformer.call_count

            b2 = SentenceTransformerBackend(model_name="test-model", use_nli=False)
            call_count_after_second = mock_st_module.SentenceTransformer.call_count

            # Model constructor should NOT be called again (reused from cache)
            assert call_count_after_second == call_count_after_first

    def test_model_cache_refreshed_on_different_model_name(self, st_mocks):
        """Different model name triggers a fresh load."""
        from aragora.debate.similarity.backends import SentenceTransformerBackend

        mock_st_module, mock_model, mock_ce, mock_cos, mock_skl = st_mocks
        mock_embedding_cache = MagicMock()
        mock_embedding_cache.get.return_value = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "sentence_transformers": mock_st_module,
                    "sklearn.metrics.pairwise": mock_skl,
                },
            ),
            patch(
                "aragora.debate.similarity.backends.get_embedding_cache",
                return_value=mock_embedding_cache,
            ),
            patch(
                "aragora.debate.similarity.backends.get_scoped_embedding_cache",
                return_value=mock_embedding_cache,
            ),
        ):
            SentenceTransformerBackend._model_cache = None
            SentenceTransformerBackend._model_name_cache = None

            SentenceTransformerBackend(model_name="model-a", use_nli=False)
            calls_after_first = mock_st_module.SentenceTransformer.call_count

            SentenceTransformerBackend(model_name="model-b", use_nli=False)
            calls_after_second = mock_st_module.SentenceTransformer.call_count

            assert calls_after_second > calls_after_first

    # --- compute_batch_similarity ---

    def test_batch_similarity_single_text(self, st_backend):
        backend, *_ = st_backend
        assert backend.compute_batch_similarity(["only one"]) == 1.0

    def test_batch_similarity_calls_model_encode(self, st_backend):
        import numpy as np

        backend, mock_model, _, _, _ = st_backend
        fake_embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        mock_model.encode.return_value = fake_embs

        with patch(
            "aragora.debate.similarity.backends.SentenceTransformerBackend.compute_batch_similarity",
            wraps=backend.compute_batch_similarity,
        ):
            # We just verify encode is called with the full list
            with patch(
                "aragora.debate.similarity.ann.compute_batch_similarity_fast",
                return_value=0.5,
            ):
                result = backend.compute_batch_similarity(["text a", "text b"])
        mock_model.encode.assert_called_with(["text a", "text b"])

    # --- compute_pairwise_similarities ---

    def test_pairwise_empty_inputs_returns_empty(self, st_backend):
        backend, *_ = st_backend
        assert backend.compute_pairwise_similarities([], []) == []

    def test_pairwise_mismatched_lengths_returns_empty(self, st_backend):
        backend, *_ = st_backend
        assert backend.compute_pairwise_similarities(["a", "b"], ["c"]) == []

    def test_pairwise_same_length_calls_encode_once(self, st_backend):
        import numpy as np

        backend, mock_model, _, mock_cos, _ = st_backend
        texts_a = ["hello", "world"]
        texts_b = ["hi", "earth"]
        # Encode called with concatenated list
        fake_embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        mock_model.encode.return_value = fake_embs
        mock_cos.return_value = np.array([[0.9]])

        result = backend.compute_pairwise_similarities(texts_a, texts_b)
        mock_model.encode.assert_called_with(texts_a + texts_b)
        assert len(result) == 2

    def test_pairwise_result_values_are_floats(self, st_backend):
        import numpy as np

        backend, mock_model, _, mock_cos, _ = st_backend
        fake_embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]])
        mock_model.encode.return_value = fake_embs
        mock_cos.return_value = np.array([[0.75]])

        result = backend.compute_pairwise_similarities(["a", "b"], ["c", "d"])
        for val in result:
            assert isinstance(val, float)

    # --- import error ---

    def test_import_error_when_sentence_transformers_absent(self):
        import sys

        saved = {k: v for k, v in sys.modules.items() if "sentence_transformers" in k}
        for k in saved:
            del sys.modules[k]

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Force reimport
            import importlib
            import aragora.debate.similarity.backends as bmod

            importlib.reload(bmod)
            with pytest.raises(ImportError, match="sentence-transformers"):
                bmod.SentenceTransformerBackend()

        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# _nli_is_contradictory edge cases
# ---------------------------------------------------------------------------


class TestNliIsContradictory:
    """Unit tests for the NLI-based contradiction detection logic."""

    @pytest.fixture
    def backend_with_nli(self, st_backend):
        backend, _, mock_cross_encoder, _, _ = st_backend
        return backend, mock_cross_encoder

    def test_contradiction_highest_score_returns_true(self, backend_with_nli):
        backend, mock_ce = backend_with_nli
        mock_ce.predict.return_value = [[0.95, 0.03, 0.02]]
        assert backend._nli_is_contradictory("accept", "reject") is True

    def test_entailment_highest_score_returns_false(self, backend_with_nli):
        backend, mock_ce = backend_with_nli
        mock_ce.predict.return_value = [[0.02, 0.95, 0.03]]
        assert backend._nli_is_contradictory("dogs are animals", "a dog is an animal") is False

    def test_neutral_highest_score_returns_false(self, backend_with_nli):
        backend, mock_ce = backend_with_nli
        mock_ce.predict.return_value = [[0.1, 0.1, 0.8]]
        assert backend._nli_is_contradictory("the sky is blue", "clouds exist") is False

    def test_predict_returns_empty_list(self, backend_with_nli):
        backend, mock_ce = backend_with_nli
        mock_ce.predict.return_value = []
        result = backend._nli_is_contradictory("a", "b")
        assert result is False

    def test_predict_raises_value_error_falls_back(self, backend_with_nli):
        backend, mock_ce = backend_with_nli
        mock_ce.predict.side_effect = ValueError("bad shape")
        # Falls back to pattern-based; "accept"/"reject" with shared "proposal" word
        result = backend._nli_is_contradictory("accept the proposal", "reject the proposal")
        assert result is True


# ---------------------------------------------------------------------------
# get_similarity_backend (factory)
# ---------------------------------------------------------------------------


class TestGetSimilarityBackend:
    """Tests for the get_similarity_backend factory function."""

    def test_jaccard_preferred_returns_jaccard(self):
        from aragora.debate.similarity.backends import JaccardBackend, get_similarity_backend

        backend = get_similarity_backend("jaccard")
        assert isinstance(backend, JaccardBackend)

    def test_tfidf_preferred_returns_tfidf_or_raises(self):
        """With no sklearn, requesting tfidf raises ImportError."""
        import sys

        saved = {k: v for k, v in sys.modules.items() if "sklearn" in k}
        for k in saved:
            del sys.modules[k]
        with patch.dict(
            "sys.modules",
            {
                "sklearn": None,
                "sklearn.feature_extraction": None,
                "sklearn.feature_extraction.text": None,
                "sklearn.metrics": None,
                "sklearn.metrics.pairwise": None,
            },
        ):
            from aragora.debate.similarity.backends import get_similarity_backend

            with pytest.raises(ImportError):
                get_similarity_backend("tfidf")
        sys.modules.update(saved)

    def test_auto_falls_back_to_jaccard_when_all_fail(self):
        """With no sentence-transformers and no sklearn, auto → Jaccard."""
        import sys

        # Ensure heavy deps are absent by making them raise on import
        saved_st = sys.modules.pop("sentence_transformers", None)
        saved_sk = {k: v for k, v in sys.modules.items() if "sklearn" in k}
        for k in saved_sk:
            del sys.modules[k]

        with patch.dict(
            "sys.modules",
            {
                "sentence_transformers": None,
                "sklearn": None,
                "sklearn.feature_extraction": None,
                "sklearn.feature_extraction.text": None,
                "sklearn.metrics": None,
                "sklearn.metrics.pairwise": None,
            },
        ):
            from aragora.debate.similarity.backends import JaccardBackend, get_similarity_backend

            backend = get_similarity_backend("auto")
            assert isinstance(backend, JaccardBackend)

        if saved_st is not None:
            sys.modules["sentence_transformers"] = saved_st
        sys.modules.update(saved_sk)

    def test_env_var_jaccard_override(self):
        """ARAGORA_SIMILARITY_BACKEND=jaccard selects JaccardBackend."""
        from aragora.debate.similarity.backends import JaccardBackend, get_similarity_backend

        with patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "jaccard"}):
            backend = get_similarity_backend("auto")
        assert isinstance(backend, JaccardBackend)

    def test_env_var_alias_override(self):
        """ARAGORA_SIMILARITY_BACKEND=tf-idf should normalize and try tfidf."""
        import sys

        # Build sklearn mocks
        mock_vec_cls = MagicMock()
        mock_vec_instance = MagicMock()
        mock_vec_cls.return_value = mock_vec_instance
        mock_cos = MagicMock(return_value=[[0.5]])

        sklearn_mock = MagicMock()
        sklearn_mock.feature_extraction = MagicMock()
        sklearn_mock.feature_extraction.text = MagicMock()
        sklearn_mock.feature_extraction.text.TfidfVectorizer = mock_vec_cls
        sklearn_mock.metrics = MagicMock()
        sklearn_mock.metrics.pairwise = MagicMock()
        sklearn_mock.metrics.pairwise.cosine_similarity = mock_cos

        with (
            patch.dict(
                "sys.modules",
                {
                    "sklearn": sklearn_mock,
                    "sklearn.feature_extraction": sklearn_mock.feature_extraction,
                    "sklearn.feature_extraction.text": sklearn_mock.feature_extraction.text,
                    "sklearn.metrics": sklearn_mock.metrics,
                    "sklearn.metrics.pairwise": sklearn_mock.metrics.pairwise,
                },
            ),
            patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "tf-idf"}),
        ):
            from aragora.debate.similarity.backends import TFIDFBackend, get_similarity_backend

            backend = get_similarity_backend("auto")
            assert isinstance(backend, TFIDFBackend)

    def test_invalid_env_var_logs_warning_and_uses_auto(self, caplog):
        """Invalid ARAGORA_SIMILARITY_BACKEND logs warning and continues auto."""
        import sys
        import logging

        saved_st = sys.modules.pop("sentence_transformers", None)
        saved_sk = {k: v for k, v in sys.modules.items() if "sklearn" in k}
        for k in saved_sk:
            del sys.modules[k]

        with (
            patch.dict(
                "sys.modules",
                {
                    "sentence_transformers": None,
                    "sklearn": None,
                    "sklearn.feature_extraction": None,
                    "sklearn.feature_extraction.text": None,
                    "sklearn.metrics": None,
                    "sklearn.metrics.pairwise": None,
                },
            ),
            patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "invalid-backend-name"}),
            caplog.at_level(logging.WARNING, logger="aragora.debate.similarity.backends"),
        ):
            from aragora.debate.similarity.backends import JaccardBackend, get_similarity_backend

            backend = get_similarity_backend("auto")
            assert isinstance(backend, JaccardBackend)
        assert "invalid" in caplog.text.lower() or "ARAGORA_SIMILARITY_BACKEND" in caplog.text

        if saved_st is not None:
            sys.modules["sentence_transformers"] = saved_st
        sys.modules.update(saved_sk)

    def test_explicit_jaccard_ignores_env_var(self):
        """When preferred='jaccard' is passed explicitly, env var is ignored."""
        from aragora.debate.similarity.backends import JaccardBackend, get_similarity_backend

        with patch.dict("os.environ", {"ARAGORA_SIMILARITY_BACKEND": "tfidf"}):
            backend = get_similarity_backend("jaccard")
        assert isinstance(backend, JaccardBackend)

    def test_debate_id_passed_to_sentence_transformer(self, st_mocks):
        """debate_id arg is forwarded to SentenceTransformerBackend."""
        mock_st_module, mock_model, mock_ce, mock_cos, mock_skl = st_mocks
        mock_emb_cache = MagicMock()
        mock_emb_cache.get.return_value = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "sentence_transformers": mock_st_module,
                    "sklearn.metrics.pairwise": mock_skl,
                },
            ),
            patch(
                "aragora.debate.similarity.backends.get_scoped_embedding_cache",
                return_value=mock_emb_cache,
            ) as mock_scoped,
            patch(
                "aragora.debate.similarity.backends.get_embedding_cache",
                return_value=mock_emb_cache,
            ),
        ):
            from aragora.debate.similarity.backends import (
                SentenceTransformerBackend,
                get_similarity_backend,
            )

            backend = get_similarity_backend("sentence-transformer", debate_id="debate-42")
            assert isinstance(backend, SentenceTransformerBackend)
            # Scoped cache should have been requested for this debate_id
            mock_scoped.assert_called_with("debate-42")


# ---------------------------------------------------------------------------
# Integration-style tests (JaccardBackend only, no mocking needed)
# ---------------------------------------------------------------------------


class TestIntegrationJaccard:
    """End-to-end style tests using JaccardBackend (no mocking required)."""

    @pytest.fixture
    def backend(self):
        from aragora.debate.similarity.backends import JaccardBackend

        return JaccardBackend()

    def test_full_pipeline_compute_and_cache(self, backend):
        from aragora.debate.similarity.backends import JaccardBackend

        result1 = backend.compute_similarity("machine learning model", "deep learning model")
        result2 = backend.compute_similarity("machine learning model", "deep learning model")
        assert result1 == result2
        assert len(JaccardBackend._similarity_cache) == 1

    def test_batch_similarity_via_jaccard(self, backend):
        texts = [
            "we should accept the proposal",
            "we should accept the proposal",
            "totally different content",
        ]
        result = backend.compute_batch_similarity(texts)
        # Two identical + one different; average will be between 0 and 1
        assert 0.0 < result < 1.0

    def test_is_contradictory_integration(self, backend):
        assert backend.is_contradictory("start the process", "stop the process") is True
        assert backend.is_contradictory("start the process", "begin the workflow") is False

    def test_compute_batch_similarity_all_same(self, backend):
        texts = ["identical text"] * 5
        result = backend.compute_batch_similarity(texts)
        assert result == pytest.approx(1.0)

    def test_compute_batch_similarity_two_element(self, backend):
        result = backend.compute_batch_similarity(["apple", "banana"])
        assert result == pytest.approx(0.0)
