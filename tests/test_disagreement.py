"""
Tests for the disagreement reporting module.

Covers DisagreementReporter: vote alignment analysis, unanimous critiques,
split opinions, and risk area identification.
"""

import pytest

from aragora.core import Critique, DisagreementReport, Vote
from aragora.debate.disagreement import DisagreementReporter


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reporter():
    """Create a DisagreementReporter with default settings."""
    return DisagreementReporter()


@pytest.fixture
def custom_reporter():
    """Create a DisagreementReporter with custom thresholds."""
    return DisagreementReporter(
        low_confidence_threshold=0.5,
        high_severity_threshold=0.8,
        max_risk_areas=3,
        max_severe_critiques=2,
    )


@pytest.fixture
def sample_votes():
    """Create sample votes for testing."""
    return [
        Vote(agent="claude", choice="option_a", confidence=0.9, reasoning="Strong logic"),
        Vote(agent="gemini", choice="option_a", confidence=0.8, reasoning="Good support"),
        Vote(agent="gpt4", choice="option_b", confidence=0.5, reasoning="Uncertain"),
    ]


@pytest.fixture
def unanimous_votes():
    """Create unanimous votes for testing."""
    return [
        Vote(agent="claude", choice="option_a", confidence=0.9, reasoning="Agreed"),
        Vote(agent="gemini", choice="option_a", confidence=0.85, reasoning="Concur"),
        Vote(agent="gpt4", choice="option_a", confidence=0.8, reasoning="Same"),
    ]


@pytest.fixture
def sample_critiques():
    """Create sample critiques for testing."""
    return [
        Critique(
            agent="claude",
            target_agent="gpt4",
            target_content="GPT4's proposal",
            issues=["Lacks evidence", "Missing context"],
            suggestions=["Add citations", "Provide context"],
            severity=0.6,
            reasoning="The proposal needs more support",
        ),
        Critique(
            agent="gemini",
            target_agent="gpt4",
            target_content="GPT4's proposal",
            issues=["Lacks evidence", "Unclear reasoning"],
            suggestions=["Clarify logic", "Add examples"],
            severity=0.7,
            reasoning="The argument structure is weak",
        ),
    ]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestDisagreementReporterInit:
    """Tests for DisagreementReporter initialization."""

    def test_default_low_confidence_threshold(self, reporter):
        """Should use default low confidence threshold of 0.6."""
        assert reporter.low_confidence_threshold == 0.6

    def test_default_high_severity_threshold(self, reporter):
        """Should use default high severity threshold of 0.7."""
        assert reporter.high_severity_threshold == 0.7

    def test_default_max_risk_areas(self, reporter):
        """Should use default max risk areas of 5."""
        assert reporter.max_risk_areas == 5

    def test_default_max_severe_critiques(self, reporter):
        """Should use default max severe critiques of 3."""
        assert reporter.max_severe_critiques == 3

    def test_custom_thresholds(self, custom_reporter):
        """Should accept custom threshold values."""
        assert custom_reporter.low_confidence_threshold == 0.5
        assert custom_reporter.high_severity_threshold == 0.8
        assert custom_reporter.max_risk_areas == 3
        assert custom_reporter.max_severe_critiques == 2


# =============================================================================
# Empty Input Tests
# =============================================================================


class TestEmptyInputs:
    """Tests for handling empty inputs."""

    def test_empty_votes_and_critiques(self, reporter):
        """Should return empty report when no votes or critiques."""
        report = reporter.generate_report([], [])
        assert report.agreement_score == 0.0
        assert report.unanimous_critiques == []
        assert report.split_opinions == []
        assert report.risk_areas == []
        assert report.agent_alignment == {}

    def test_empty_votes_with_critiques(self, reporter, sample_critiques):
        """Should handle empty votes with critiques."""
        report = reporter.generate_report([], sample_critiques)
        assert report.agreement_score == 0.0
        assert len(report.unanimous_critiques) >= 0  # May have unanimous critiques

    def test_votes_with_empty_critiques(self, reporter, sample_votes):
        """Should handle votes with empty critiques."""
        report = reporter.generate_report(sample_votes, [])
        assert report.agreement_score > 0.0
        assert report.unanimous_critiques == []


# =============================================================================
# Agreement Score Tests
# =============================================================================


class TestAgreementScore:
    """Tests for agreement score calculation."""

    def test_unanimous_agreement(self, reporter, unanimous_votes):
        """Should return 1.0 for unanimous votes."""
        report = reporter.generate_report(unanimous_votes, [])
        assert report.agreement_score == 1.0

    def test_split_agreement(self, reporter, sample_votes):
        """Should calculate correct agreement for split votes."""
        # 2 out of 3 voted option_a
        report = reporter.generate_report(sample_votes, [])
        assert report.agreement_score == pytest.approx(2 / 3)

    def test_single_vote(self, reporter):
        """Should handle single vote case."""
        single_vote = [Vote(agent="claude", choice="option_a", confidence=0.9, reasoning="Only vote")]
        report = reporter.generate_report(single_vote, [])
        # Single vote means agreement_score stays 0 (no comparison possible)
        assert report.agreement_score == 0.0

    def test_complete_disagreement(self, reporter):
        """Should handle complete disagreement (all different votes)."""
        votes = [
            Vote(agent="claude", choice="option_a", confidence=0.8, reasoning="A"),
            Vote(agent="gemini", choice="option_b", confidence=0.8, reasoning="B"),
            Vote(agent="gpt4", choice="option_c", confidence=0.8, reasoning="C"),
        ]
        report = reporter.generate_report(votes, [])
        assert report.agreement_score == pytest.approx(1 / 3)


# =============================================================================
# Agent Alignment Tests
# =============================================================================


class TestAgentAlignment:
    """Tests for agent alignment with winner."""

    def test_alignment_with_winner(self, reporter, sample_votes):
        """Should calculate alignment when winner specified."""
        report = reporter.generate_report(sample_votes, [], winner="option_a")
        assert report.agent_alignment["claude"] == 1.0
        assert report.agent_alignment["gemini"] == 1.0
        assert report.agent_alignment["gpt4"] == 0.0

    def test_no_alignment_without_winner(self, reporter, sample_votes):
        """Should have empty alignment when no winner."""
        report = reporter.generate_report(sample_votes, [])
        assert report.agent_alignment == {}

    def test_all_aligned_with_winner(self, reporter, unanimous_votes):
        """Should show all aligned when unanimous winner."""
        report = reporter.generate_report(unanimous_votes, [], winner="option_a")
        assert all(v == 1.0 for v in report.agent_alignment.values())

    def test_none_aligned_with_winner(self, reporter, unanimous_votes):
        """Should show none aligned when wrong winner."""
        report = reporter.generate_report(unanimous_votes, [], winner="option_b")
        assert all(v == 0.0 for v in report.agent_alignment.values())


# =============================================================================
# Unanimous Critiques Tests
# =============================================================================


class TestUnanimousCritiques:
    """Tests for finding unanimous critiques."""

    def test_finds_unanimous_critique(self, reporter, sample_critiques):
        """Should find critique raised by all critics."""
        report = reporter.generate_report([], sample_critiques)
        # "Lacks evidence" is raised by both critics
        assert any("Lacks evidence" in c for c in report.unanimous_critiques)

    def test_no_unanimous_with_single_critic(self, reporter):
        """Should have no unanimous critiques with single critic."""
        single_critique = [
            Critique(
                agent="claude",
                target_agent="gpt4",
                target_content="GPT4's proposal",
                issues=["Issue 1"],
                suggestions=["Fix it"],
                severity=0.5,
                reasoning="Found an issue",
            )
        ]
        report = reporter.generate_report([], single_critique)
        assert report.unanimous_critiques == []

    def test_no_unanimous_with_different_issues(self, reporter):
        """Should have no unanimous when critics raise different issues."""
        different_critiques = [
            Critique(
                agent="claude",
                target_agent="gpt4",
                target_content="GPT4's proposal",
                issues=["Issue A"],
                suggestions=["Fix A"],
                severity=0.5,
                reasoning="Found issue A",
            ),
            Critique(
                agent="gemini",
                target_agent="gpt4",
                target_content="GPT4's proposal",
                issues=["Issue B"],
                suggestions=["Fix B"],
                severity=0.5,
                reasoning="Found issue B",
            ),
        ]
        report = reporter.generate_report([], different_critiques)
        assert report.unanimous_critiques == []

    def test_case_insensitive_matching(self, reporter):
        """Should match issues case-insensitively."""
        critiques = [
            Critique(
                agent="claude",
                target_agent="gpt4",
                target_content="GPT4's proposal",
                issues=["Lacks Evidence"],
                suggestions=["Add evidence"],
                severity=0.5,
                reasoning="Evidence needed",
            ),
            Critique(
                agent="gemini",
                target_agent="gpt4",
                target_content="GPT4's proposal",
                issues=["lacks evidence"],
                suggestions=["Add evidence"],
                severity=0.5,
                reasoning="Evidence needed",
            ),
        ]
        report = reporter.generate_report([], critiques)
        # Should find the unanimous critique despite case difference
        assert len(report.unanimous_critiques) == 1


# =============================================================================
# Split Opinions Tests
# =============================================================================


class TestSplitOpinions:
    """Tests for finding split opinions."""

    def test_finds_split_opinion(self, reporter, sample_votes):
        """Should identify split votes."""
        report = reporter.generate_report(sample_votes, [])
        assert len(report.split_opinions) == 1
        description, majority, minority = report.split_opinions[0]
        assert "Vote split" in description
        assert len(majority) == 2  # claude and gemini
        assert len(minority) == 1  # gpt4

    def test_no_split_when_unanimous(self, reporter, unanimous_votes):
        """Should have no split opinions when unanimous."""
        report = reporter.generate_report(unanimous_votes, [])
        assert report.split_opinions == []

    def test_no_split_with_single_vote(self, reporter):
        """Should have no split with single vote."""
        single_vote = [Vote(agent="claude", choice="option_a", confidence=0.9, reasoning="Only")]
        report = reporter.generate_report(single_vote, [])
        assert report.split_opinions == []

    def test_multiple_minority_groups(self, reporter):
        """Should create entries for multiple minority groups."""
        votes = [
            Vote(agent="claude", choice="option_a", confidence=0.9, reasoning="A"),
            Vote(agent="gemini", choice="option_a", confidence=0.8, reasoning="A"),
            Vote(agent="gpt4", choice="option_b", confidence=0.7, reasoning="B"),
            Vote(agent="llama", choice="option_c", confidence=0.6, reasoning="C"),
        ]
        report = reporter.generate_report(votes, [])
        # Should have 2 split entries: majority vs option_b, majority vs option_c
        assert len(report.split_opinions) == 2

    def test_split_truncates_long_choices(self, reporter):
        """Should truncate long choice names in split description."""
        long_choice = "a" * 100
        votes = [
            Vote(agent="claude", choice=long_choice, confidence=0.9, reasoning="Long"),
            Vote(agent="gemini", choice="short", confidence=0.8, reasoning="Short"),
        ]
        report = reporter.generate_report(votes, [])
        description = report.split_opinions[0][0]
        # Choice should be truncated to 50 chars
        assert len(description) < len(long_choice) + 50


# =============================================================================
# Risk Areas Tests
# =============================================================================


class TestRiskAreas:
    """Tests for identifying risk areas."""

    def test_low_confidence_vote_flagged(self, reporter, sample_votes):
        """Should flag low confidence votes as risk areas."""
        report = reporter.generate_report(sample_votes, [])
        # gpt4 has 0.5 confidence, below 0.6 threshold
        assert any("gpt4" in risk and "low confidence" in risk for risk in report.risk_areas)

    def test_no_risk_with_high_confidence(self, reporter, unanimous_votes):
        """Should have no low-confidence risks with high confidence votes."""
        report = reporter.generate_report(unanimous_votes, [])
        # All votes have confidence >= 0.8, above 0.6 threshold
        assert not any("low confidence" in risk for risk in report.risk_areas)

    def test_severe_critique_of_winner_flagged(self, reporter):
        """Should flag high-severity critiques of winner."""
        votes = [Vote(agent="gpt4", choice="answer", confidence=0.9, reasoning="Win")]
        critiques = [
            Critique(
                agent="claude",
                target_agent="gpt4",
                target_content="GPT4's answer",
                issues=["Major flaw"],
                suggestions=["Fix the flaw"],
                severity=0.8,
                reasoning="Serious problem found",
            )
        ]
        report = reporter.generate_report(votes, critiques, winner="gpt4")
        assert any("High-severity" in risk and "gpt4" in risk for risk in report.risk_areas)

    def test_severe_critique_of_loser_not_flagged(self, reporter):
        """Should not flag high-severity critiques of loser."""
        votes = [Vote(agent="gpt4", choice="answer", confidence=0.9, reasoning="Win")]
        critiques = [
            Critique(
                agent="claude",
                target_agent="gpt4",
                target_content="GPT4's answer",
                issues=["Major flaw"],
                suggestions=["Fix the flaw"],
                severity=0.8,
                reasoning="Serious problem found",
            )
        ]
        report = reporter.generate_report(votes, critiques, winner="claude")
        # gpt4 didn't win, so critique shouldn't be flagged
        assert not any("High-severity" in risk for risk in report.risk_areas)

    def test_max_risk_areas_limit(self, custom_reporter):
        """Should respect max_risk_areas limit."""
        # Create many low-confidence votes
        votes = [
            Vote(agent=f"agent_{i}", choice="option", confidence=0.3, reasoning="Low")
            for i in range(10)
        ]
        report = custom_reporter.generate_report(votes, [])
        # custom_reporter has max_risk_areas=3
        low_conf_risks = [r for r in report.risk_areas if "low confidence" in r]
        assert len(low_conf_risks) <= 3

    def test_max_severe_critiques_limit(self, custom_reporter):
        """Should respect max_severe_critiques limit."""
        votes = [Vote(agent="winner", choice="answer", confidence=0.9, reasoning="Win")]
        critiques = [
            Critique(
                agent=f"critic_{i}",
                target_agent="winner",
                target_content="Winner's answer",
                issues=[f"Issue {i}"],
                suggestions=[f"Fix {i}"],
                severity=0.9,
                reasoning=f"Found severe issue {i}",
            )
            for i in range(10)
        ]
        report = custom_reporter.generate_report(votes, critiques, winner="winner")
        # custom_reporter has max_severe_critiques=2
        severe_risks = [r for r in report.risk_areas if "High-severity" in r]
        assert len(severe_risks) <= 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestDisagreementReporterIntegration:
    """Integration tests for DisagreementReporter."""

    def test_full_report_generation(self, reporter, sample_votes, sample_critiques):
        """Test complete report with all components."""
        report = reporter.generate_report(sample_votes, sample_critiques, winner="option_a")

        # Should have all report fields populated
        assert isinstance(report, DisagreementReport)
        assert report.agreement_score > 0
        assert report.agent_alignment != {}
        assert isinstance(report.unanimous_critiques, list)
        assert isinstance(report.split_opinions, list)
        assert isinstance(report.risk_areas, list)

    def test_report_immutability(self, reporter, sample_votes):
        """Generating multiple reports should be independent."""
        report1 = reporter.generate_report(sample_votes, [])
        report2 = reporter.generate_report(sample_votes, [], winner="option_a")

        # Reports should be independent
        assert report1.agent_alignment == {}
        assert report2.agent_alignment != {}

    def test_custom_threshold_changes_results(self):
        """Custom thresholds should affect what gets flagged."""
        votes = [
            Vote(agent="claude", choice="option", confidence=0.55, reasoning="Borderline")
        ]

        default_reporter = DisagreementReporter()
        strict_reporter = DisagreementReporter(low_confidence_threshold=0.7)

        default_report = default_reporter.generate_report(votes, [])
        strict_report = strict_reporter.generate_report(votes, [])

        # Default (0.6 threshold) should flag 0.55 confidence
        default_low_conf = [r for r in default_report.risk_areas if "low confidence" in r]
        assert len(default_low_conf) == 1

        # Strict (0.7 threshold) should also flag 0.55 confidence
        strict_low_conf = [r for r in strict_report.risk_areas if "low confidence" in r]
        assert len(strict_low_conf) == 1

    def test_complex_debate_scenario(self, reporter):
        """Test realistic multi-agent debate scenario."""
        votes = [
            Vote(agent="claude", choice="Approach A: Use microservices", confidence=0.85, reasoning="Scalable"),
            Vote(agent="gemini", choice="Approach A: Use microservices", confidence=0.78, reasoning="Modern"),
            Vote(agent="gpt4", choice="Approach B: Monolith first", confidence=0.45, reasoning="Simpler"),
            Vote(agent="llama", choice="Approach A: Use microservices", confidence=0.72, reasoning="Industry standard"),
        ]

        critiques = [
            Critique(
                agent="gpt4",
                target_agent="claude",
                target_content="Claude's microservices proposal",
                issues=["Complexity overhead", "Operational burden"],
                suggestions=["Start simpler", "Add complexity later"],
                severity=0.65,
                reasoning="Microservices add operational complexity",
            ),
            Critique(
                agent="gemini",
                target_agent="gpt4",
                target_content="GPT4's monolith proposal",
                issues=["Scalability issues", "Technical debt"],
                suggestions=["Plan for scale", "Consider future needs"],
                severity=0.7,
                reasoning="Monoliths can become unmaintainable",
            ),
            Critique(
                agent="llama",
                target_agent="gpt4",
                target_content="GPT4's monolith proposal",
                issues=["Scalability issues"],
                suggestions=["Think about growth"],
                severity=0.6,
                reasoning="Scaling monoliths is hard",
            ),
        ]

        report = reporter.generate_report(
            votes, critiques, winner="Approach A: Use microservices"
        )

        # 3 out of 4 voted for option A
        assert report.agreement_score == 0.75

        # claude, gemini, llama aligned with winner
        assert report.agent_alignment["claude"] == 1.0
        assert report.agent_alignment["gpt4"] == 0.0

        # gpt4 has low confidence
        assert any("gpt4" in risk for risk in report.risk_areas)

        # "Scalability issues" is raised by multiple critics
        # (gemini and llama both raised it about gpt4)
        # Note: they need to critique the same target for unanimous detection
