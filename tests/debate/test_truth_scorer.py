"""Tests for the Persuasion vs Truth Scorer."""

from __future__ import annotations

import pytest

from aragora.debate.truth_scorer import (
    EvidenceType,
    RhetoricType,
    TruthScorer,
)


@pytest.fixture
def scorer():
    return TruthScorer()


# ── Evidence Detection Tests ──────────────────────────────────────


class TestEvidenceDetection:
    def test_detect_data(self, scorer):
        text = "Response time improved by 40%, from 200ms to 120ms."
        items = scorer._detect_evidence(text)
        types = {i.evidence_type for i in items}
        assert EvidenceType.DATA in types

    def test_detect_citation(self, scorer):
        text = "According to the 2024 study by Smith et al., the results show improvement."
        items = scorer._detect_evidence(text)
        types = {i.evidence_type for i in items}
        assert EvidenceType.CITATION in types

    def test_detect_experiment(self, scorer):
        text = "We benchmarked the new algorithm and measured a 2x speedup."
        items = scorer._detect_evidence(text)
        types = {i.evidence_type for i in items}
        assert EvidenceType.EXPERIMENT in types

    def test_detect_logical_proof(self, scorer):
        text = "Since X implies Y, and Y implies Z, therefore X implies Z."
        items = scorer._detect_evidence(text)
        types = {i.evidence_type for i in items}
        assert EvidenceType.LOGICAL_PROOF in types

    def test_detect_example(self, scorer):
        text = "For example, consider the case where the cache is full."
        items = scorer._detect_evidence(text)
        types = {i.evidence_type for i in items}
        assert EvidenceType.EXAMPLE in types

    def test_no_evidence_in_rhetoric(self, scorer):
        text = "Everyone knows this is obviously the best approach."
        items = scorer._detect_evidence(text)
        # May detect some false positives, but should be minimal
        data_items = [i for i in items if i.evidence_type == EvidenceType.DATA]
        assert len(data_items) == 0


# ── Rhetoric Detection Tests ─────────────────────────────────────


class TestRhetoricDetection:
    def test_detect_emotional_appeal(self, scorer):
        text = "This devastating failure is a catastrophic threat to our system."
        items = scorer._detect_rhetoric(text)
        types = {i.rhetoric_type for i in items}
        assert RhetoricType.EMOTIONAL_APPEAL in types

    def test_detect_authority_without_data(self, scorer):
        text = "It is well known and obvious that experts say this is correct."
        items = scorer._detect_rhetoric(text)
        types = {i.rhetoric_type for i in items}
        assert RhetoricType.AUTHORITY_WITHOUT_DATA in types

    def test_detect_social_proof(self, scorer):
        text = "The majority of developers agree this is the standard approach."
        items = scorer._detect_rhetoric(text)
        types = {i.rhetoric_type for i in items}
        assert RhetoricType.SOCIAL_PROOF in types

    def test_detect_confidence_display(self, scorer):
        text = "The fact is this is absolutely, definitely the right choice."
        items = scorer._detect_rhetoric(text)
        types = {i.rhetoric_type for i in items}
        assert RhetoricType.CONFIDENCE_DISPLAY in types

    def test_no_rhetoric_in_evidence(self, scorer):
        text = "The benchmark results show 150ms p99 latency with n=1000 samples."
        items = scorer._detect_rhetoric(text)
        # Should detect minimal rhetoric
        emotional = [i for i in items if i.rhetoric_type == RhetoricType.EMOTIONAL_APPEAL]
        assert len(emotional) == 0


# ── Repetition Detection Tests ────────────────────────────────────


class TestRepetitionDetection:
    def test_detect_repetition(self, scorer):
        text = (
            "We need to act now. We need to act now. We need to act now. "
            "The situation requires immediate action."
        )
        items = scorer._detect_repetition(text)
        assert len(items) >= 1
        assert items[0].rhetoric_type == RhetoricType.REPETITION

    def test_no_repetition_in_normal_text(self, scorer):
        text = "The system uses a cache layer for performance. Database queries are optimized."
        items = scorer._detect_repetition(text)
        assert len(items) == 0


# ── Full Scoring Tests ────────────────────────────────────────────


class TestFullScoring:
    def test_evidence_heavy_argument(self, scorer):
        text = (
            "The benchmark results show a 45% improvement in response time, "
            "decreasing from 200ms to 110ms. According to the 2024 performance study, "
            "this correlates with a p<0.01 significance level. We tested with n=500 "
            "requests and measured consistent results across 3 replications."
        )
        score = scorer.score(text)
        assert score.truth_ratio > 0.5
        assert len(score.evidence_items) >= 3
        assert "evidence" in score.overall_assessment.lower()

    def test_rhetoric_heavy_argument(self, scorer):
        text = (
            "Everyone knows this is obviously the best approach. The majority agrees. "
            "It is absolutely, undeniably clear that this is the right choice. "
            "Imagine the devastating consequences if we don't act now! "
            "The fact is, no one would deny this is true."
        )
        score = scorer.score(text)
        assert score.truth_ratio < 0.5
        assert len(score.rhetoric_items) >= 3
        assert "rhetoric" in score.overall_assessment.lower()

    def test_balanced_argument(self, scorer):
        text = (
            "The data shows a 30% improvement. However, the majority of experts "
            "agree this approach is superior, which is obviously true. "
            "According to the 2024 study, performance increased significantly."
        )
        score = scorer.score(text)
        assert 0.2 < score.truth_ratio < 0.8
        assert len(score.evidence_items) > 0
        assert len(score.rhetoric_items) > 0

    def test_empty_text(self, scorer):
        score = scorer.score("")
        assert score.truth_ratio == 0.5  # neutral default
        assert len(score.evidence_items) == 0
        assert len(score.rhetoric_items) == 0

    def test_to_dict(self, scorer):
        score = scorer.score("Testing 45% improvement with p<0.05.")
        d = score.to_dict()
        assert "evidence_score" in d
        assert "rhetoric_score" in d
        assert "truth_ratio" in d
        assert "evidence_count" in d
        assert "rhetoric_count" in d


# ── Debate Round Scoring Tests ────────────────────────────────────


class TestDebateRoundScoring:
    def test_score_debate_round(self, scorer):
        contributions = [
            {
                "agent": "claude",
                "text": "The benchmark shows 50ms latency reduction, measured over 1000 requests.",
            },
            {
                "agent": "gpt4",
                "text": "Everyone obviously knows this is the best approach.",
            },
        ]
        scores = scorer.score_debate_round(contributions)
        assert "claude" in scores
        assert "gpt4" in scores
        # Claude's evidence-based argument should score higher
        assert scores["claude"].truth_ratio > scores["gpt4"].truth_ratio


# ── Assessment Text Tests ─────────────────────────────────────────


class TestAssessment:
    def test_high_truth_ratio_assessment(self, scorer):
        score = scorer.score(
            "According to study X (2024), the results showed 40% improvement "
            "with p<0.01 across n=500 samples. We replicated this 3 times."
        )
        assert "evidence" in score.overall_assessment.lower()

    def test_low_truth_ratio_assessment(self, scorer):
        score = scorer.score(
            "Obviously everyone knows this is absolutely the best. "
            "It would be devastating not to choose this incredible approach."
        )
        assert "rhetoric" in score.overall_assessment.lower()
