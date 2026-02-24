"""Tests for convergence detection module."""

from __future__ import annotations

from aragora_debate.convergence import ConvergenceDetector, ConvergenceResult


class TestConvergenceResult:
    def test_creation(self):
        result = ConvergenceResult(converged=True, similarity=0.9, round_num=2)
        assert result.converged is True
        assert result.similarity == 0.9
        assert result.round_num == 2
        assert result.pair_similarities == {}

    def test_with_pairs(self):
        result = ConvergenceResult(
            converged=False,
            similarity=0.5,
            round_num=1,
            pair_similarities={"a:b": 0.5, "a:c": 0.4},
        )
        assert len(result.pair_similarities) == 2


class TestConvergenceDetector:
    def setup_method(self):
        self.detector = ConvergenceDetector(threshold=0.85)

    def test_identical_proposals_converge(self):
        proposals = {
            "agent1": "We should use caching.",
            "agent2": "We should use caching.",
        }
        result = self.detector.check(proposals, round_num=1)
        assert result.converged is True
        assert result.similarity == 1.0

    def test_different_proposals_dont_converge(self):
        proposals = {
            "agent1": "We should use microservices with Kubernetes and Docker.",
            "agent2": "A monolithic architecture with PostgreSQL is better.",
        }
        result = self.detector.check(proposals, round_num=1)
        assert result.converged is False
        assert result.similarity < 0.85

    def test_similar_proposals(self):
        proposals = {
            "agent1": "Caching with Redis improves performance significantly. Use TTL of 300s.",
            "agent2": "Caching with Redis improves performance significantly. Use TTL of 600s.",
        }
        result = self.detector.check(proposals, round_num=1)
        assert result.similarity > 0.7

    def test_single_agent(self):
        result = self.detector.check({"agent1": "Some text"}, round_num=1)
        assert result.converged is False
        assert result.similarity == 0.0

    def test_empty_proposals(self):
        result = self.detector.check({}, round_num=1)
        assert result.converged is False

    def test_three_agents_pairwise(self):
        proposals = {
            "a": "Use caching for speed.",
            "b": "Use caching for speed.",
            "c": "Use caching for speed.",
        }
        result = self.detector.check(proposals)
        assert result.converged is True
        assert len(result.pair_similarities) == 3  # a:b, a:c, b:c

    def test_custom_threshold(self):
        detector = ConvergenceDetector(threshold=0.5)
        proposals = {
            "a": "The approach works well with this configuration setup.",
            "b": "The approach works nicely with this configuration setup.",
        }
        result = detector.check(proposals)
        assert result.converged is True

    def test_check_trend_first_round(self):
        proposals = {"a": "proposal A", "b": "proposal B"}
        result = self.detector.check_trend(proposals, round_num=1)
        assert isinstance(result, ConvergenceResult)
        # No prev round comparison possible
        assert not any(":prev" in k for k in result.pair_similarities)

    def test_check_trend_second_round(self):
        self.detector.check_trend(
            {"a": "initial proposal A", "b": "initial proposal B"},
            round_num=1,
        )
        result = self.detector.check_trend(
            {"a": "initial proposal A revised", "b": "initial proposal B revised"},
            round_num=2,
        )
        # Should have prev comparisons
        prev_keys = [k for k in result.pair_similarities if ":prev" in k]
        assert len(prev_keys) == 2  # a:prev, b:prev

    def test_history_tracking(self):
        self.detector.check({"a": "round 1"}, round_num=1)
        self.detector.check({"a": "round 2"}, round_num=2)
        assert len(self.detector.history) == 2

    def test_reset(self):
        self.detector.check({"a": "text"})
        self.detector.reset()
        assert len(self.detector.history) == 0

    def test_empty_string_similarity(self):
        proposals = {"a": "", "b": ""}
        result = self.detector.check(proposals)
        assert result.similarity == 1.0
