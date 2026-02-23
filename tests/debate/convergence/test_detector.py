"""
Tests for aragora/debate/convergence/detector.py

Covers:
- ConvergenceDetector.__init__: default and custom params, backend selection
- _select_backend: factory path, env override path, fallback to JaccardBackend
- check_convergence: early-return conditions, per-agent similarity, status classification,
  consecutive stable count accumulation, batch vs individual compute paths
- check_within_round_convergence: single-agent early return, Jaccard fallback path,
  early termination on below-threshold pair
- check_convergence_fast: vectorized path (when _get_embedding available), fallback
  to check_convergence
- record_convergence_metrics: normal store path, no-store path, ImportError, runtime error
- cleanup: calls cache helpers when debate_id set, noop when not set
- reset: resets consecutive_stable_count
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATCH_FACTORY = "aragora.debate.similarity.factory.get_backend"
PATCH_GET_SIM_BACKEND = "aragora.debate.similarity.backends.get_similarity_backend"
PATCH_FACTORY_IN_MODULE = "aragora.debate.convergence.detector.get_backend"  # imported locally


def _make_jaccard():
    """Return a real JaccardBackend instance."""
    from aragora.debate.similarity.backends import JaccardBackend

    return JaccardBackend()


def _make_detector(**kwargs):
    """
    Create a ConvergenceDetector with all heavy imports mocked out.

    Patches `aragora.debate.similarity.factory.get_backend` so that
    _select_backend() never touches real ML libraries.
    """
    fake_backend = _make_jaccard()
    # The factory is imported lazily inside _select_backend, so patch the
    # name as it appears in the factory module (which is then re-looked-up).
    with patch("aragora.debate.similarity.factory.get_backend", return_value=fake_backend):
        from aragora.debate.convergence.detector import ConvergenceDetector

        detector = ConvergenceDetector(**kwargs)
    return detector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """Ensure ARAGORA_CONVERGENCE_BACKEND is unset for all tests."""
    monkeypatch.delenv("ARAGORA_CONVERGENCE_BACKEND", raising=False)


@pytest.fixture()
def detector():
    return _make_detector()


@pytest.fixture()
def two_agent_responses():
    return {
        "alice": "the quick brown fox jumps over the lazy dog",
        "bob": "the quick brown fox leaps over the sleeping cat",
    }


@pytest.fixture()
def identical_responses():
    return {
        "alice": "artificial intelligence will transform industry",
        "bob": "artificial intelligence will transform industry",
    }


@pytest.fixture()
def diverged_responses():
    return {
        "alice": "cats are the best pets ever",
        "bob": "rockets propulsion system hydrogen oxygen",
    }


# ===========================================================================
# 1. Initialization
# ===========================================================================


class TestInit:
    def test_default_thresholds(self, detector):
        assert detector.convergence_threshold == 0.85
        assert detector.divergence_threshold == 0.40

    def test_default_min_rounds(self, detector):
        assert detector.min_rounds_before_check == 1

    def test_default_consecutive_rounds_needed(self, detector):
        assert detector.consecutive_rounds_needed == 1

    def test_consecutive_stable_count_starts_at_zero(self, detector):
        assert detector.consecutive_stable_count == 0

    def test_debate_id_stored(self):
        d = _make_detector(debate_id="debate-42")
        assert d.debate_id == "debate-42"

    def test_debate_id_none_by_default(self, detector):
        assert detector.debate_id is None

    def test_custom_thresholds(self):
        d = _make_detector(convergence_threshold=0.90, divergence_threshold=0.30)
        assert d.convergence_threshold == 0.90
        assert d.divergence_threshold == 0.30

    def test_backend_is_set(self, detector):
        from aragora.debate.similarity.backends import SimilarityBackend

        assert isinstance(detector.backend, SimilarityBackend)


# ===========================================================================
# 2. _select_backend
# ===========================================================================


class TestSelectBackend:
    def test_uses_factory_by_default(self):
        fake = _make_jaccard()
        with patch("aragora.debate.similarity.factory.get_backend", return_value=fake) as mock_gb:
            from aragora.debate.convergence.detector import ConvergenceDetector

            d = ConvergenceDetector()
        mock_gb.assert_called_once()
        assert d.backend is fake

    def test_falls_back_to_jaccard_on_factory_error(self):
        from aragora.debate.convergence.detector import ConvergenceDetector

        with patch(
            "aragora.debate.similarity.factory.get_backend", side_effect=ImportError("no torch")
        ):
            d = ConvergenceDetector()
        # Use class name check to avoid test pollution from sys.modules patching
        assert type(d.backend).__name__ == "JaccardBackend"

    def test_env_override_uses_get_similarity_backend(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_CONVERGENCE_BACKEND", "jaccard")
        fake = _make_jaccard()
        # get_similarity_backend is imported at module level into detector, so patch it there
        with patch(
            "aragora.debate.convergence.detector.get_similarity_backend", return_value=fake
        ) as mock_gsb:
            with patch("aragora.debate.similarity.factory.get_backend", return_value=fake):
                from aragora.debate.convergence.detector import ConvergenceDetector

                d = ConvergenceDetector()
        mock_gsb.assert_called_once_with("jaccard", debate_id=None)
        assert d.backend is fake

    def test_env_override_fallback_on_import_error(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_CONVERGENCE_BACKEND", "jaccard")
        fake = _make_jaccard()
        from aragora.debate.convergence.detector import ConvergenceDetector

        # get_similarity_backend is imported at module level into detector, so patch it there
        with patch(
            "aragora.debate.convergence.detector.get_similarity_backend",
            side_effect=ImportError("missing"),
        ):
            with patch("aragora.debate.similarity.factory.get_backend", return_value=fake):
                d = ConvergenceDetector()
        # Falls back to factory, which returns fake
        assert d.backend is fake

    def test_invalid_env_override_ignored(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_CONVERGENCE_BACKEND", "nonexistent_backend")
        fake = _make_jaccard()
        with patch("aragora.debate.similarity.factory.get_backend", return_value=fake) as mock_gb:
            from aragora.debate.convergence.detector import ConvergenceDetector

            d = ConvergenceDetector()
        # Factory is still called (env override was invalid, skipped)
        mock_gb.assert_called_once()


# ===========================================================================
# 3. check_convergence — early-return conditions
# ===========================================================================


class TestCheckConvergenceEarlyReturn:
    def test_returns_none_at_round_1_with_min_rounds_1(self, detector, two_agent_responses):
        result = detector.check_convergence(
            two_agent_responses, two_agent_responses, round_number=1
        )
        assert result is None

    def test_returns_none_when_round_equals_min_rounds(self):
        d = _make_detector(min_rounds_before_check=3)
        r = {"a": "hello world"}
        assert d.check_convergence(r, r, round_number=3) is None

    def test_proceeds_when_round_exceeds_min_rounds(self):
        d = _make_detector(min_rounds_before_check=1)
        r = {"a": "hello world"}
        result = d.check_convergence(r, r, round_number=2)
        assert result is not None

    def test_returns_none_when_no_common_agents(self, detector):
        current = {"alice": "hello"}
        previous = {"bob": "hello"}
        result = detector.check_convergence(current, previous, round_number=2)
        assert result is None

    def test_uses_only_common_agents(self, detector):
        current = {"alice": "hello world", "carol": "extra text here"}
        previous = {"alice": "hello world", "dave": "other text"}
        result = detector.check_convergence(current, previous, round_number=2)
        # carol/dave are not common — only alice is used
        assert result is not None
        assert "alice" in result.per_agent_similarity
        assert "carol" not in result.per_agent_similarity
        assert "dave" not in result.per_agent_similarity


# ===========================================================================
# 4. check_convergence — status classification
# ===========================================================================


class TestCheckConvergenceStatus:
    def test_converged_status_on_identical_texts(self, identical_responses):
        d = _make_detector(convergence_threshold=0.85, consecutive_rounds_needed=1)
        result = d.check_convergence(identical_responses, identical_responses, round_number=2)
        assert result is not None
        assert result.converged is True
        assert result.status == "converged"
        assert result.min_similarity == pytest.approx(1.0)
        assert result.avg_similarity == pytest.approx(1.0)

    def test_diverging_status_on_completely_different_texts(self, diverged_responses):
        d = _make_detector(divergence_threshold=0.40)
        result = d.check_convergence(diverged_responses, diverged_responses, round_number=2)
        # Identical texts always score 1.0; we need truly different current vs previous
        current = {"alice": "cats are great pets", "bob": "rockets fly high"}
        previous = {"alice": "hydrogen oxygen propulsion", "bob": "dogs fetch sticks ball"}
        result = d.check_convergence(current, previous, round_number=2)
        assert result is not None
        assert result.status == "diverging"
        assert result.converged is False

    def test_refining_status_in_middle_range(self):
        # Use a divergence_threshold low enough that moderate-overlap texts don't hit "diverging"
        d = _make_detector(convergence_threshold=0.85, divergence_threshold=0.20)
        # alice pair: Jaccard = 5/9 ~0.56 (above 0.20, below 0.85)
        # bob pair: Jaccard = 1/4 = 0.25 (above 0.20, below 0.85)
        current = {"alice": "the quick brown fox jumps over fence", "bob": "a lazy dog runs fast"}
        previous = {"alice": "the quick brown cat jumps over wall", "bob": "a slow cat runs away"}
        result = d.check_convergence(current, previous, round_number=2)
        assert result is not None
        assert result.status == "refining"
        assert result.converged is False

    def test_converged_requires_consecutive_rounds(self):
        d = _make_detector(convergence_threshold=0.0, consecutive_rounds_needed=2)
        r = {"alice": "hello world"}
        # First high-similarity check: not yet converged (count = 1, need 2)
        r1 = d.check_convergence(r, r, round_number=2)
        assert r1 is not None
        assert r1.converged is False
        assert r1.consecutive_stable_rounds == 1
        # Second high-similarity check: now converged (count = 2)
        r2 = d.check_convergence(r, r, round_number=3)
        assert r2 is not None
        assert r2.converged is True
        assert r2.consecutive_stable_rounds == 2

    def test_diverging_resets_consecutive_stable_count(self):
        # Use a high convergence_threshold so we can first satisfy it (identical text),
        # then use completely unrelated texts that fall below divergence_threshold=0.40.
        d = _make_detector(
            convergence_threshold=0.85, divergence_threshold=0.40, consecutive_rounds_needed=3
        )
        same = {"alice": "hello world"}
        d.check_convergence(same, same, round_number=2)  # Jaccard=1.0 >= 0.85 → count -> 1
        # Now use totally unrelated texts so Jaccard similarity is ~0 (below divergence_threshold)
        current = {"alice": "quantum entanglement physics experiment"}
        previous = {"alice": "cooking pasta tomato sauce recipe"}
        result = d.check_convergence(current, previous, round_number=3)
        assert result is not None
        assert result.status == "diverging"
        assert d.consecutive_stable_count == 0

    def test_refining_resets_consecutive_stable_count(self):
        d = _make_detector(convergence_threshold=0.85, divergence_threshold=0.10)
        same = {"alice": "hello world"}
        d.check_convergence(same, same, round_number=2)  # count -> 1
        # Moderate similarity texts — in refining range
        current = {"alice": "the cat sat on the mat in the house"}
        previous = {"alice": "the dog ran on the street near school"}
        d.check_convergence(current, previous, round_number=3)
        assert d.consecutive_stable_count == 0

    def test_per_agent_similarity_in_result(self, identical_responses):
        d = _make_detector()
        result = d.check_convergence(identical_responses, identical_responses, round_number=2)
        assert result is not None
        assert set(result.per_agent_similarity.keys()) == {"alice", "bob"}
        for sim in result.per_agent_similarity.values():
            assert 0.0 <= sim <= 1.0


# ===========================================================================
# 5. check_convergence — batch vs individual compute paths
# ===========================================================================


class TestCheckConvergenceBatchVsIndividual:
    def test_uses_compute_pairwise_when_available(self):
        """If backend has compute_pairwise_similarities, it must be used."""
        fake_backend = MagicMock()
        fake_backend.compute_pairwise_similarities.return_value = [0.9, 0.9]
        # Remove compute_similarity so individual path can't be used
        del fake_backend.compute_similarity

        d = _make_detector()
        d.backend = fake_backend

        current = {"a": "hello", "b": "world"}
        previous = {"a": "hello", "b": "world"}
        result = d.check_convergence(current, previous, round_number=2)
        assert result is not None
        fake_backend.compute_pairwise_similarities.assert_called_once()
        # Values returned by batch method should appear in result
        assert list(result.per_agent_similarity.values()) == pytest.approx([0.9, 0.9])

    def test_uses_individual_compute_when_no_batch(self):
        """When backend lacks compute_pairwise_similarities, use compute_similarity."""
        d = _make_detector()
        # JaccardBackend doesn't have compute_pairwise_similarities
        assert not hasattr(d.backend, "compute_pairwise_similarities")

        current = {"alice": "hello world foo"}
        previous = {"alice": "hello world bar"}
        result = d.check_convergence(current, previous, round_number=2)
        assert result is not None
        assert "alice" in result.per_agent_similarity
        assert result.per_agent_similarity["alice"] > 0


# ===========================================================================
# 6. ConvergenceResult fields
# ===========================================================================


class TestConvergenceResultFields:
    def test_result_has_all_fields(self, identical_responses):
        d = _make_detector()
        result = d.check_convergence(identical_responses, identical_responses, round_number=2)
        assert result is not None
        assert isinstance(result.converged, bool)
        assert isinstance(result.status, str)
        assert isinstance(result.min_similarity, float)
        assert isinstance(result.avg_similarity, float)
        assert isinstance(result.per_agent_similarity, dict)
        assert isinstance(result.consecutive_stable_rounds, int)

    def test_min_leq_avg_similarity(self, two_agent_responses):
        d = _make_detector()
        result = d.check_convergence(two_agent_responses, two_agent_responses, round_number=2)
        assert result is not None
        assert result.min_similarity <= result.avg_similarity


# ===========================================================================
# 7. check_within_round_convergence
# ===========================================================================


class TestCheckWithinRoundConvergence:
    def test_single_agent_returns_true_1_1(self, detector):
        responses = {"alice": "only one agent here"}
        converged, min_sim, avg_sim = detector.check_within_round_convergence(responses)
        assert converged is True
        assert min_sim == pytest.approx(1.0)
        assert avg_sim == pytest.approx(1.0)

    def test_jaccard_path_identical_texts(self, detector):
        responses = {
            "alice": "hello world test",
            "bob": "hello world test",
        }
        converged, min_sim, avg_sim = detector.check_within_round_convergence(
            responses, threshold=0.8
        )
        assert converged is True
        assert min_sim == pytest.approx(1.0)

    def test_jaccard_path_diverged_texts_early_termination(self, detector):
        responses = {
            "alice": "cats are the best pets everyone loves them",
            "bob": "rockets propulsion hydrogen oxygen combustion",
        }
        converged, min_sim, avg_sim = detector.check_within_round_convergence(
            responses, threshold=0.5
        )
        assert converged is False
        assert min_sim < 0.5

    def test_custom_threshold_applied(self, detector):
        # Texts with Jaccard ~0.33
        responses = {
            "alice": "hello world test",
            "bob": "hello mars exam",
        }
        # With a very low threshold, should converge
        converged_low, min_sim, _ = detector.check_within_round_convergence(
            responses, threshold=0.1
        )
        assert converged_low is True

        # With a very high threshold, should not converge
        converged_high, _, _ = detector.check_within_round_convergence(responses, threshold=0.99)
        assert converged_high is False

    def test_defaults_to_convergence_threshold(self):
        d = _make_detector(convergence_threshold=0.99)
        responses = {
            "alice": "hello world test",
            "bob": "goodbye earth exam",
        }
        # Jaccard will be low; with threshold 0.99 it definitely won't converge
        converged, _, _ = d.check_within_round_convergence(responses)
        assert converged is False

    def test_empty_responses_dict(self, detector):
        # Less than 2 texts → trivially True
        converged, min_sim, avg_sim = detector.check_within_round_convergence({})
        assert converged is True


# ===========================================================================
# 8. check_convergence_fast
# ===========================================================================


class TestCheckConvergenceFast:
    def test_returns_none_before_min_rounds(self, detector, two_agent_responses):
        result = detector.check_convergence_fast(
            two_agent_responses, two_agent_responses, round_number=1
        )
        assert result is None

    def test_returns_none_no_common_agents(self, detector):
        result = detector.check_convergence_fast({"alice": "hi"}, {"bob": "hi"}, round_number=2)
        assert result is None

    def test_falls_back_to_check_convergence_without_embeddings(self, detector):
        """JaccardBackend has no _get_embedding, so fast path uses standard check."""
        current = {"alice": "hello world foo bar"}
        previous = {"alice": "hello world baz qux"}
        result_fast = detector.check_convergence_fast(current, previous, round_number=2)
        result_std = detector.check_convergence(current, previous, round_number=2)
        assert result_fast is not None
        assert result_std is not None
        # Both should produce same status
        assert result_fast.status == result_std.status

    def test_vectorized_path_with_embedding_backend(self):
        """When backend has _get_embedding, the numpy path should be used."""
        import numpy as np

        fake_backend = MagicMock()
        # Simulate identical embeddings (cosine sim = 1.0)
        embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        fake_backend._get_embedding.return_value = embedding

        d = _make_detector()
        d.backend = fake_backend

        current = {"alice": "hello", "bob": "world"}
        previous = {"alice": "hello", "bob": "world"}
        result = d.check_convergence_fast(current, previous, round_number=2)
        assert result is not None
        assert result.min_similarity == pytest.approx(1.0)
        assert result.avg_similarity == pytest.approx(1.0)
        assert result.converged is True

    def test_fast_status_converged_on_identical(self, identical_responses):
        d = _make_detector(convergence_threshold=0.85)
        result = d.check_convergence_fast(identical_responses, identical_responses, round_number=2)
        assert result is not None
        assert result.status in ("converged", "refining")  # depends on consecutive count


# ===========================================================================
# 9. record_convergence_metrics
# ===========================================================================


class TestRecordConvergenceMetrics:
    def test_calls_store_when_available(self, detector):
        mock_store = MagicMock()
        mock_store.store = MagicMock()

        with patch(
            "aragora.debate.convergence.history.get_convergence_history_store",
            return_value=mock_store,
        ):
            detector.record_convergence_metrics(
                topic="AI governance",
                convergence_round=3,
                total_rounds=5,
                final_similarity=0.92,
                per_round_similarity=[0.5, 0.7, 0.92],
            )

        mock_store.store.assert_called_once_with(
            topic="AI governance",
            convergence_round=3,
            total_rounds=5,
            final_similarity=0.92,
            per_round_similarity=[0.5, 0.7, 0.92],
            debate_id="",
        )

    def test_noop_when_store_returns_none(self, detector):
        with patch(
            "aragora.debate.convergence.history.get_convergence_history_store",
            return_value=None,
        ):
            # Should not raise
            detector.record_convergence_metrics("topic", 0, 3, 0.5)

    def test_handles_import_error_silently(self, detector):
        with patch(
            "aragora.debate.convergence.detector.ConvergenceDetector.record_convergence_metrics",
            wraps=detector.record_convergence_metrics,
        ):
            # Patch the import inside the method
            import builtins

            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "aragora.debate.convergence.history":
                    raise ImportError("not available")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # Should not raise
                detector.record_convergence_metrics("topic", 1, 2, 0.8)

    def test_handles_runtime_error_silently(self, detector):
        mock_store = MagicMock()
        mock_store.store.side_effect = RuntimeError("DB down")
        with patch(
            "aragora.debate.convergence.history.get_convergence_history_store",
            return_value=mock_store,
        ):
            # Should not raise
            detector.record_convergence_metrics("topic", 1, 2, 0.8)

    def test_debate_id_used_in_store_call(self):
        d = _make_detector(debate_id="debate-123")
        mock_store = MagicMock()
        with patch(
            "aragora.debate.convergence.history.get_convergence_history_store",
            return_value=mock_store,
        ):
            d.record_convergence_metrics("topic", 2, 4, 0.88)

        _, kwargs = mock_store.store.call_args
        assert kwargs.get("debate_id") == "debate-123"

    def test_per_round_similarity_defaults_to_none(self, detector):
        mock_store = MagicMock()
        with patch(
            "aragora.debate.convergence.history.get_convergence_history_store",
            return_value=mock_store,
        ):
            detector.record_convergence_metrics("topic", 0, 2, 0.6)

        _, kwargs = mock_store.store.call_args
        assert kwargs.get("per_round_similarity") is None


# ===========================================================================
# 10. cleanup
# ===========================================================================


class TestCleanup:
    def test_cleanup_calls_cache_helpers_when_debate_id_set(self):
        d = _make_detector(debate_id="debate-xyz")
        with (
            patch("aragora.debate.convergence.detector.cleanup_embedding_cache") as mock_emb,
            patch("aragora.debate.convergence.detector.cleanup_similarity_cache") as mock_sim,
        ):
            d.cleanup()

        mock_emb.assert_called_once_with("debate-xyz")
        mock_sim.assert_called_once_with("debate-xyz")

    def test_cleanup_noop_when_no_debate_id(self):
        d = _make_detector(debate_id=None)
        with (
            patch("aragora.debate.convergence.detector.cleanup_embedding_cache") as mock_emb,
            patch("aragora.debate.convergence.detector.cleanup_similarity_cache") as mock_sim,
        ):
            d.cleanup()

        mock_emb.assert_not_called()
        mock_sim.assert_not_called()


# ===========================================================================
# 11. reset
# ===========================================================================


class TestReset:
    def test_reset_sets_count_to_zero(self, detector):
        detector.consecutive_stable_count = 5
        detector.reset()
        assert detector.consecutive_stable_count == 0

    def test_reset_after_partial_convergence(self):
        d = _make_detector(convergence_threshold=0.0, consecutive_rounds_needed=3)
        same = {"alice": "hello world"}
        d.check_convergence(same, same, round_number=2)  # count -> 1
        d.check_convergence(same, same, round_number=3)  # count -> 2
        assert d.consecutive_stable_count == 2
        d.reset()
        assert d.consecutive_stable_count == 0


# ===========================================================================
# 12. Integration-style multi-round scenario
# ===========================================================================


class TestMultiRoundScenario:
    def test_full_debate_convergence_sequence(self):
        """Simulate a 4-round debate that converges in round 3."""
        d = _make_detector(
            convergence_threshold=0.85,
            divergence_threshold=0.10,
            min_rounds_before_check=1,
            consecutive_rounds_needed=1,
        )

        # Round 1: skip (round_number == min_rounds_before_check)
        r1_current = {"alice": "AI is transforming many industries rapidly today"}
        r1_prev = {"alice": "AI may change industries but it is unclear"}
        assert d.check_convergence(r1_current, r1_prev, round_number=1) is None

        # Round 2: refining
        r2_current = {"alice": "AI is transforming industries in healthcare finance education"}
        r2_prev = {"alice": "AI is transforming many industries rapidly today"}
        result2 = d.check_convergence(r2_current, r2_prev, round_number=2)
        assert result2 is not None

        # Round 3: converged (identical to previous)
        same = {"alice": "AI is transforming industries in healthcare finance education"}
        result3 = d.check_convergence(same, same, round_number=3)
        assert result3 is not None
        assert result3.converged is True
        assert result3.status == "converged"

    def test_consecutive_rounds_accumulate_correctly(self):
        d = _make_detector(convergence_threshold=0.0, consecutive_rounds_needed=3)
        same = {"alice": "hello world"}
        for round_num in range(2, 6):
            result = d.check_convergence(same, same, round_number=round_num)
            assert result is not None
            expected_count = round_num - 1  # rounds 2,3,4,5 → count 1,2,3,4
            assert result.consecutive_stable_rounds == expected_count
            if round_num >= 4:
                assert result.converged is True
            else:
                assert result.converged is False
