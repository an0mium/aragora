"""Tests for aragora.debate.disagreement — DisagreementReporter."""

from __future__ import annotations

import pytest

from aragora.core import Critique, DisagreementReport, Vote
from aragora.debate.disagreement import DisagreementReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vote(agent: str, choice: str, confidence: float = 0.8) -> Vote:
    return Vote(agent=agent, choice=choice, reasoning="", confidence=confidence)


def _critique(
    agent: str,
    target: str,
    issues: list[str] | None = None,
    severity: float = 0.5,
) -> Critique:
    return Critique(
        agent=agent,
        target_agent=target,
        target_content="",
        issues=issues or [],
        suggestions=[],
        severity=severity,
        reasoning="test reasoning",
    )


# ---------------------------------------------------------------------------
# DisagreementReporter — init
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        r = DisagreementReporter()
        assert r.low_confidence_threshold == 0.6
        assert r.high_severity_threshold == 0.7
        assert r.max_risk_areas == 5
        assert r.max_severe_critiques == 3

    def test_custom(self):
        r = DisagreementReporter(
            low_confidence_threshold=0.5,
            high_severity_threshold=0.8,
            max_risk_areas=3,
            max_severe_critiques=2,
        )
        assert r.low_confidence_threshold == 0.5
        assert r.high_severity_threshold == 0.8


# ---------------------------------------------------------------------------
# generate_report — empty inputs
# ---------------------------------------------------------------------------


class TestGenerateReportEmpty:
    def test_no_votes_no_critiques(self):
        r = DisagreementReporter()
        report = r.generate_report([], [])
        assert isinstance(report, DisagreementReport)

    def test_only_votes(self):
        r = DisagreementReporter()
        report = r.generate_report([_vote("a", "x")], [])
        assert isinstance(report, DisagreementReport)

    def test_only_critiques(self):
        r = DisagreementReporter()
        report = r.generate_report([], [_critique("a", "b", ["issue1"])])
        assert isinstance(report, DisagreementReport)


# ---------------------------------------------------------------------------
# agreement_score
# ---------------------------------------------------------------------------


class TestAgreementScore:
    def test_unanimous_votes(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "X"), _vote("c", "X")]
        report = r.generate_report(votes, [])
        assert report.agreement_score == 1.0

    def test_split_votes(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "Y")]
        report = r.generate_report(votes, [])
        assert report.agreement_score == 0.5

    def test_majority_votes(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "X"), _vote("c", "Y")]
        report = r.generate_report(votes, [])
        assert report.agreement_score == pytest.approx(2 / 3)

    def test_single_vote_no_score(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X")]
        report = r.generate_report(votes, [])
        # Only 1 voter, len > 1 check fails, score stays at default
        assert hasattr(report, "agreement_score")


# ---------------------------------------------------------------------------
# agent_alignment
# ---------------------------------------------------------------------------


class TestAgentAlignment:
    def test_with_winner(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "Y")]
        report = r.generate_report(votes, [], winner="X")
        assert report.agent_alignment["a"] == 1.0
        assert report.agent_alignment["b"] == 0.0

    def test_without_winner(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "Y")]
        report = r.generate_report(votes, [])
        assert report.agent_alignment == {}

    def test_all_aligned(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "X")]
        report = r.generate_report(votes, [], winner="X")
        assert all(v == 1.0 for v in report.agent_alignment.values())


# ---------------------------------------------------------------------------
# unanimous_critiques
# ---------------------------------------------------------------------------


class TestUnanimousCritiques:
    def test_unanimous_issue(self):
        r = DisagreementReporter()
        critiques = [
            _critique("critic1", "target", ["security flaw"]),
            _critique("critic2", "target", ["security flaw"]),
        ]
        report = r.generate_report([], critiques)
        assert len(report.unanimous_critiques) == 1
        assert "security flaw" in report.unanimous_critiques[0]

    def test_no_unanimous(self):
        r = DisagreementReporter()
        critiques = [
            _critique("critic1", "target", ["issue A"]),
            _critique("critic2", "target", ["issue B"]),
        ]
        report = r.generate_report([], critiques)
        assert report.unanimous_critiques == []

    def test_single_critic(self):
        r = DisagreementReporter()
        critiques = [_critique("critic1", "target", ["issue A"])]
        report = r.generate_report([], critiques)
        # Need >1 critic for unanimous
        assert report.unanimous_critiques == []

    def test_case_insensitive_matching(self):
        r = DisagreementReporter()
        critiques = [
            _critique("c1", "t", ["Security Flaw"]),
            _critique("c2", "t", ["security flaw"]),
        ]
        report = r.generate_report([], critiques)
        assert len(report.unanimous_critiques) == 1


# ---------------------------------------------------------------------------
# split_opinions
# ---------------------------------------------------------------------------


class TestSplitOpinions:
    def test_split_vote(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "X"), _vote("c", "Y")]
        report = r.generate_report(votes, [])
        assert len(report.split_opinions) == 1
        desc, majority, minority = report.split_opinions[0]
        assert "Vote split" in desc
        assert "a" in majority or "b" in majority
        assert "c" in minority

    def test_no_split(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "X")]
        report = r.generate_report(votes, [])
        assert report.split_opinions == []

    def test_single_voter(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X")]
        report = r.generate_report(votes, [])
        assert report.split_opinions == []

    def test_three_way_split(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X"), _vote("b", "Y"), _vote("c", "Z")]
        report = r.generate_report(votes, [])
        # Multiple minority splits
        assert len(report.split_opinions) >= 2


# ---------------------------------------------------------------------------
# risk_areas
# ---------------------------------------------------------------------------


class TestRiskAreas:
    def test_low_confidence_vote(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X", confidence=0.3)]
        report = r.generate_report(votes, [])
        assert len(report.risk_areas) >= 1
        assert "low confidence" in report.risk_areas[0]
        assert "a" in report.risk_areas[0]

    def test_high_confidence_no_risk(self):
        r = DisagreementReporter()
        votes = [_vote("a", "X", confidence=0.9)]
        report = r.generate_report(votes, [])
        # No low-confidence risk areas
        low_conf_risks = [r for r in report.risk_areas if "low confidence" in r]
        assert len(low_conf_risks) == 0

    def test_severe_critique_of_winner(self):
        r = DisagreementReporter()
        critiques = [_critique("c", "winner_agent", ["big problem"], severity=0.9)]
        report = r.generate_report([], critiques, winner="winner_agent")
        severe_risks = [r for r in report.risk_areas if "High-severity" in r]
        assert len(severe_risks) >= 1

    def test_severe_critique_not_of_winner(self):
        r = DisagreementReporter()
        critiques = [_critique("c", "other_agent", ["big problem"], severity=0.9)]
        report = r.generate_report([], critiques, winner="winner_agent")
        severe_risks = [r for r in report.risk_areas if "High-severity" in r]
        assert len(severe_risks) == 0

    def test_max_risk_areas_capped(self):
        r = DisagreementReporter(max_risk_areas=2)
        votes = [
            _vote("a", "X", confidence=0.1),
            _vote("b", "X", confidence=0.2),
            _vote("c", "X", confidence=0.3),
        ]
        report = r.generate_report(votes, [])
        low_conf = [r for r in report.risk_areas if "low confidence" in r]
        assert len(low_conf) <= 2

    def test_max_severe_critiques_capped(self):
        r = DisagreementReporter(max_severe_critiques=1)
        critiques = [
            _critique("c1", "w", ["issue1"], severity=0.9),
            _critique("c2", "w", ["issue2"], severity=0.8),
        ]
        report = r.generate_report([], critiques, winner="w")
        severe_risks = [r for r in report.risk_areas if "High-severity" in r]
        assert len(severe_risks) <= 1

    def test_critique_without_issues(self):
        r = DisagreementReporter()
        critiques = [_critique("c", "w", [], severity=0.9)]
        report = r.generate_report([], critiques, winner="w")
        severe_risks = [r for r in report.risk_areas if "High-severity" in r]
        # Should still report with "various issues" fallback
        assert len(severe_risks) >= 1
        assert "various issues" in severe_risks[0]


# ---------------------------------------------------------------------------
# Full integration
# ---------------------------------------------------------------------------


class TestFullReport:
    def test_comprehensive(self):
        r = DisagreementReporter()
        votes = [
            _vote("alice", "X", confidence=0.9),
            _vote("bob", "X", confidence=0.4),
            _vote("carol", "Y", confidence=0.7),
        ]
        critiques = [
            _critique("bob", "alice", ["lacks detail"], severity=0.8),
            _critique("carol", "alice", ["lacks detail"], severity=0.6),
        ]
        report = r.generate_report(votes, critiques, winner="X")

        # Agreement score: 2/3
        assert report.agreement_score == pytest.approx(2 / 3)
        # Alice and bob aligned with winner
        assert report.agent_alignment["alice"] == 1.0
        assert report.agent_alignment["carol"] == 0.0
        # Unanimous critique
        assert len(report.unanimous_critiques) >= 1
        # Split opinion
        assert len(report.split_opinions) >= 1
        # Bob has low confidence
        assert any("bob" in r and "low confidence" in r for r in report.risk_areas)
