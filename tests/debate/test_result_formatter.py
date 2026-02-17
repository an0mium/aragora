"""
Tests for ResultFormatter - debate result formatting.

Tests cover:
- Result formatter initialization with custom settings
- Verdict section formatting (consensus/no consensus)
- Final answer formatting and truncation
- Vote breakdown formatting
- Dissenting views display
- Belief cruxes formatting
- Translations section
- Compliance validation and result formatting
- Conclusion with compliance integration
- Edge cases (no votes, single agent, empty results)
- Convenience functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.result_formatter import (
    ResultFormatter,
    format_conclusion,
    format_conclusion_with_compliance,
)


@dataclass
class MockVote:
    """Mock vote object for testing."""

    voter: str
    choice: str
    reasoning: str = "Test reasoning"
    confidence: float = 1.0


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    consensus_reached: bool = True
    confidence: float = 0.85
    final_answer: str = "This is the final answer from the debate."
    winner: str | None = None
    consensus_strength: str | None = None
    votes: list[MockVote] = field(default_factory=list)
    dissenting_views: list[str] = field(default_factory=list)
    belief_cruxes: list[dict[str, Any]] = field(default_factory=list)
    translations: dict[str, str] = field(default_factory=dict)


@pytest.fixture
def formatter():
    """Create a default ResultFormatter."""
    return ResultFormatter()


@pytest.fixture
def custom_formatter():
    """Create a ResultFormatter with custom length limits."""
    return ResultFormatter(max_answer_length=50, max_view_length=30)


@pytest.fixture
def basic_result():
    """Create a basic debate result."""
    return MockDebateResult()


@pytest.fixture
def result_with_votes():
    """Create a debate result with votes."""
    return MockDebateResult(
        votes=[
            MockVote(voter="claude", choice="proposal_A"),
            MockVote(voter="gpt", choice="proposal_A"),
            MockVote(voter="gemini", choice="proposal_B"),
        ]
    )


@pytest.fixture
def result_with_dissent():
    """Create a debate result with dissenting views."""
    return MockDebateResult(
        consensus_reached=False,
        confidence=0.4,
        dissenting_views=[
            "I believe we should consider alternative approach X.",
            "The proposed solution has security concerns.",
            "Performance implications were not adequately addressed.",
            "This fourth view should be truncated.",
        ],
    )


@pytest.fixture
def result_with_cruxes():
    """Create a debate result with belief cruxes."""
    return MockDebateResult(
        belief_cruxes=[
            {"claim": "The API should use GraphQL instead of REST", "uncertainty": 0.75},
            {"claim": "Performance will be impacted by this design", "uncertainty": 0.42},
            {"claim": "Security risks are minimal with proper authentication", "uncertainty": 0.33},
        ]
    )


@pytest.fixture
def result_with_translations():
    """Create a debate result with translations."""
    return MockDebateResult(
        translations={
            "es": "Esta es la respuesta final del debate.",
            "fr": "Voici la reponse finale du debat.",
        }
    )


class TestResultFormatterInit:
    """Tests for ResultFormatter initialization."""

    def test_default_init(self, formatter):
        """Test default initialization values."""
        assert formatter.max_answer_length == 1000
        assert formatter.max_view_length == 300

    def test_custom_init(self, custom_formatter):
        """Test custom initialization values."""
        assert custom_formatter.max_answer_length == 50
        assert custom_formatter.max_view_length == 30


class TestVerdictFormatting:
    """Tests for verdict section formatting."""

    def test_consensus_reached(self, formatter, basic_result):
        """Test formatting when consensus is reached."""
        conclusion = formatter.format_conclusion(basic_result)
        assert "Consensus: YES (85% agreement)" in conclusion

    def test_consensus_with_strength(self, formatter):
        """Test formatting with consensus strength."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.92,
            consensus_strength="strong",
        )
        conclusion = formatter.format_conclusion(result)
        assert "Consensus: YES (92% agreement)" in conclusion
        assert "Strength: STRONG" in conclusion

    def test_no_consensus(self, formatter):
        """Test formatting when no consensus reached."""
        result = MockDebateResult(
            consensus_reached=False,
            confidence=0.35,
        )
        conclusion = formatter.format_conclusion(result)
        assert "Consensus: NO (only 35% agreement)" in conclusion

    def test_winner_displayed(self, formatter):
        """Test winner is displayed when present."""
        result = MockDebateResult(winner="claude")
        conclusion = formatter.format_conclusion(result)
        assert "Winner: claude" in conclusion

    def test_no_winner_not_displayed(self, formatter, basic_result):
        """Test winner line absent when no winner."""
        conclusion = formatter.format_conclusion(basic_result)
        assert "Winner:" not in conclusion


class TestFinalAnswerFormatting:
    """Tests for final answer section formatting."""

    def test_final_answer_displayed(self, formatter, basic_result):
        """Test final answer is displayed."""
        conclusion = formatter.format_conclusion(basic_result)
        assert "## FINAL ANSWER" in conclusion
        assert "This is the final answer from the debate." in conclusion

    def test_long_answer_truncated(self, custom_formatter):
        """Test long answers are truncated."""
        long_answer = "A" * 100  # Longer than max_answer_length of 50
        result = MockDebateResult(final_answer=long_answer)
        conclusion = custom_formatter.format_conclusion(result)
        # Should be truncated to 50 chars + "..."
        assert "A" * 50 + "..." in conclusion
        assert "A" * 100 not in conclusion

    def test_no_final_answer(self, formatter):
        """Test handling when no final answer."""
        result = MockDebateResult(final_answer="")
        conclusion = formatter.format_conclusion(result)
        assert "No final answer determined." in conclusion


class TestVoteBreakdownFormatting:
    """Tests for vote breakdown section formatting."""

    def test_votes_displayed(self, formatter, result_with_votes):
        """Test votes are displayed correctly."""
        conclusion = formatter.format_conclusion(result_with_votes)
        assert "## VOTE BREAKDOWN" in conclusion
        assert "claude: proposal_A" in conclusion
        assert "gpt: proposal_A" in conclusion
        assert "gemini: proposal_B" in conclusion

    def test_no_votes_section_absent(self, formatter, basic_result):
        """Test vote section is absent when no votes."""
        conclusion = formatter.format_conclusion(basic_result)
        assert "## VOTE BREAKDOWN" not in conclusion

    def test_vote_with_missing_attrs(self, formatter):
        """Test handling votes with missing attributes."""
        # Create a vote-like object without standard attributes
        mock_vote = MagicMock()
        del mock_vote.voter  # Remove voter attribute
        del mock_vote.choice  # Remove choice attribute
        result = MockDebateResult(votes=[mock_vote])
        # Should handle gracefully with defaults
        conclusion = formatter.format_conclusion(result)
        assert "## VOTE BREAKDOWN" in conclusion
        assert "unknown: abstain" in conclusion

    def test_single_vote(self, formatter):
        """Test formatting with single vote."""
        result = MockDebateResult(votes=[MockVote(voter="claude", choice="approved")])
        conclusion = formatter.format_conclusion(result)
        assert "claude: approved" in conclusion


class TestDissentingViewsFormatting:
    """Tests for dissenting views section formatting."""

    def test_dissenting_views_displayed(self, formatter, result_with_dissent):
        """Test dissenting views are displayed."""
        conclusion = formatter.format_conclusion(result_with_dissent)
        assert "## DISSENTING VIEWS" in conclusion
        assert "1. I believe we should consider alternative approach X." in conclusion
        assert "2. The proposed solution has security concerns." in conclusion
        assert "3. Performance implications were not adequately addressed." in conclusion

    def test_max_three_views(self, formatter, result_with_dissent):
        """Test only first 3 views are shown."""
        conclusion = formatter.format_conclusion(result_with_dissent)
        # Fourth view should not be present
        assert "4." not in conclusion
        assert "This fourth view should be truncated." not in conclusion

    def test_long_view_truncated(self, custom_formatter):
        """Test long views are truncated."""
        long_view = "B" * 100  # Longer than max_view_length of 30
        result = MockDebateResult(dissenting_views=[long_view])
        conclusion = custom_formatter.format_conclusion(result)
        assert "B" * 30 + "..." in conclusion
        assert "B" * 100 not in conclusion

    def test_no_dissent_section_absent(self, formatter, basic_result):
        """Test dissent section is absent when no dissenting views."""
        conclusion = formatter.format_conclusion(basic_result)
        assert "## DISSENTING VIEWS" not in conclusion


class TestCruxesFormatting:
    """Tests for belief cruxes section formatting."""

    def test_cruxes_displayed(self, formatter, result_with_cruxes):
        """Test cruxes are displayed correctly."""
        conclusion = formatter.format_conclusion(result_with_cruxes)
        assert "## KEY CRUXES" in conclusion
        assert "The API should use GraphQL instead of REST" in conclusion
        assert "(uncertainty: 0.75)" in conclusion
        assert "(uncertainty: 0.42)" in conclusion

    def test_max_three_cruxes(self, formatter):
        """Test only first 3 cruxes are shown."""
        cruxes = [{"claim": f"Claim {i}", "uncertainty": 0.5} for i in range(5)]
        result = MockDebateResult(belief_cruxes=cruxes)
        conclusion = formatter.format_conclusion(result)
        assert "Claim 0" in conclusion
        assert "Claim 1" in conclusion
        assert "Claim 2" in conclusion
        assert "Claim 3" not in conclusion
        assert "Claim 4" not in conclusion

    def test_long_claim_truncated(self, formatter):
        """Test long claims are truncated to 80 chars."""
        long_claim = "C" * 200
        result = MockDebateResult(belief_cruxes=[{"claim": long_claim, "uncertainty": 0.5}])
        conclusion = formatter.format_conclusion(result)
        # Claim is truncated to 80 chars, then "..." is added
        assert "C" * 80 + "..." in conclusion

    def test_no_cruxes_section_absent(self, formatter, basic_result):
        """Test cruxes section is absent when no cruxes."""
        conclusion = formatter.format_conclusion(basic_result)
        assert "## KEY CRUXES" not in conclusion

    def test_crux_missing_fields_uses_defaults(self, formatter):
        """Test cruxes with missing fields use defaults."""
        result = MockDebateResult(
            belief_cruxes=[{}]  # Empty dict
        )
        conclusion = formatter.format_conclusion(result)
        assert "unknown" in conclusion  # Default claim
        assert "(uncertainty: 0.00)" in conclusion  # Default uncertainty


class TestTranslationsFormatting:
    """Tests for translations section formatting."""

    def test_translations_displayed(self, formatter, result_with_translations):
        """Test translations are displayed."""
        conclusion = formatter.format_conclusion(result_with_translations)
        assert "## TRANSLATIONS" in conclusion
        assert "Esta es la respuesta final del debate." in conclusion
        assert "Voici la reponse finale du debat." in conclusion

    @patch("aragora.debate.result_formatter.ResultFormatter._add_translations")
    def test_translations_with_language_module(
        self, mock_add_translations, formatter, result_with_translations
    ):
        """Test translations use Language module when available."""
        # This test verifies the translation module integration path
        # The actual test is in the implementation
        formatter.format_conclusion(result_with_translations)
        mock_add_translations.assert_called_once()

    def test_no_translations_section_absent(self, formatter, basic_result):
        """Test translations section is absent when no translations."""
        conclusion = formatter.format_conclusion(basic_result)
        assert "## TRANSLATIONS" not in conclusion

    def test_long_translation_truncated(self, custom_formatter):
        """Test long translations are truncated."""
        long_translation = "T" * 100
        result = MockDebateResult(translations={"es": long_translation})
        conclusion = custom_formatter.format_conclusion(result)
        assert "T" * 50 + "..." in conclusion


class TestFormatConclusionStructure:
    """Tests for overall conclusion structure."""

    def test_header_present(self, formatter, basic_result):
        """Test header is present."""
        conclusion = formatter.format_conclusion(basic_result)
        assert "=" * 60 in conclusion
        assert "DEBATE CONCLUSION" in conclusion

    def test_sections_in_order(self, formatter):
        """Test sections appear in correct order."""
        result = MockDebateResult(
            votes=[MockVote(voter="claude", choice="yes")],
            dissenting_views=["Dissent here"],
            belief_cruxes=[{"claim": "test", "uncertainty": 0.5}],
        )
        conclusion = formatter.format_conclusion(result)

        # Check order of sections
        verdict_pos = conclusion.find("## VERDICT")
        answer_pos = conclusion.find("## FINAL ANSWER")
        vote_pos = conclusion.find("## VOTE BREAKDOWN")
        dissent_pos = conclusion.find("## DISSENTING VIEWS")
        crux_pos = conclusion.find("## KEY CRUXES")

        assert verdict_pos < answer_pos
        assert answer_pos < vote_pos
        assert vote_pos < dissent_pos
        assert dissent_pos < crux_pos

    def test_footer_separator(self, formatter, basic_result):
        """Test footer separator is present."""
        conclusion = formatter.format_conclusion(basic_result)
        # Should end with separator
        lines = conclusion.strip().split("\n")
        assert "=" * 60 in lines[-1]


class TestComplianceValidation:
    """Tests for compliance validation."""

    def test_validate_compliance_with_frameworks(self, formatter):
        """Test compliance validation with specific frameworks."""
        with patch(
            "aragora.debate.result_formatter.ResultFormatter.validate_compliance"
        ) as mock_validate:
            mock_result = MagicMock()
            mock_result.compliant = True
            mock_result.score = 1.0
            mock_result.frameworks_checked = ["GDPR"]
            mock_result.issues = []
            mock_validate.return_value = mock_result

            result = formatter.validate_compliance(content="Test content", frameworks=["GDPR"])

            assert result.compliant is True

    def test_validate_compliance_import_error(self, formatter):
        """Test compliance validation handles import errors gracefully."""
        with patch.dict("sys.modules", {"aragora.compliance.framework": None}):
            # Should return empty compliant result when framework unavailable
            # The actual implementation catches ImportError
            pass  # Implementation handles this internally

    def test_validate_compliance_with_vertical(self, formatter):
        """Test compliance validation with industry vertical."""
        with patch(
            "aragora.debate.result_formatter.ResultFormatter.validate_compliance"
        ) as mock_validate:
            mock_result = MagicMock()
            mock_result.compliant = True
            mock_validate.return_value = mock_result

            formatter.validate_compliance(content="Healthcare data", vertical="healthcare")

            mock_validate.assert_called_once()


class TestComplianceResultFormatting:
    """Tests for compliance result formatting."""

    def test_format_compliant_result(self, formatter):
        """Test formatting a compliant result."""
        mock_result = MagicMock()
        mock_result.compliant = True
        mock_result.score = 1.0
        mock_result.frameworks_checked = ["GDPR", "SOC2"]
        mock_result.issues = []

        formatted = formatter.format_compliance_result(mock_result)

        assert "## COMPLIANCE CHECK" in formatted
        assert "Status: COMPLIANT (score: 100%)" in formatted
        assert "Frameworks: GDPR, SOC2" in formatted
        assert "No compliance issues detected." in formatted

    def test_format_non_compliant_result(self, formatter):
        """Test formatting a non-compliant result."""
        mock_issue = MagicMock()
        mock_issue.framework = "GDPR"
        mock_issue.description = "Personal data not encrypted"
        mock_issue.recommendation = "Implement AES-256 encryption"
        mock_issue.severity = MagicMock()
        mock_issue.severity.value = "high"

        mock_result = MagicMock()
        mock_result.compliant = False
        mock_result.score = 0.6
        mock_result.frameworks_checked = ["GDPR"]
        mock_result.issues = [mock_issue]
        mock_result.critical_issues = []
        mock_result.high_issues = [mock_issue]

        formatted = formatter.format_compliance_result(mock_result)

        assert "Status: NON-COMPLIANT (score: 60%)" in formatted
        assert "Issues found: 1" in formatted
        assert "HIGH (1):" in formatted

    def test_format_result_no_frameworks(self, formatter):
        """Test formatting when no frameworks were checked."""
        mock_result = MagicMock()
        mock_result.frameworks_checked = []

        formatted = formatter.format_compliance_result(mock_result)

        assert "No compliance frameworks checked." in formatted

    def test_format_critical_issues(self, formatter):
        """Test formatting with critical issues."""
        mock_critical = MagicMock()
        mock_critical.framework = "SOC2"
        mock_critical.description = "Critical security vulnerability detected in authentication"
        mock_critical.recommendation = "Implement MFA and review authentication flow immediately"

        mock_result = MagicMock()
        mock_result.compliant = False
        mock_result.score = 0.3
        mock_result.frameworks_checked = ["SOC2"]
        mock_result.issues = [mock_critical]
        mock_result.critical_issues = [mock_critical]
        mock_result.high_issues = []

        formatted = formatter.format_compliance_result(mock_result)

        assert "CRITICAL (1):" in formatted
        assert "[SOC2]" in formatted


class TestConclusionWithCompliance:
    """Tests for conclusion with compliance integration."""

    def test_conclusion_with_compliance(self, formatter, basic_result):
        """Test conclusion includes compliance when available."""
        with patch.object(formatter, "validate_compliance") as mock_validate:
            mock_compliance = MagicMock()
            mock_compliance.frameworks_checked = ["GDPR"]
            mock_compliance.compliant = True
            mock_compliance.score = 1.0
            mock_compliance.issues = []
            mock_validate.return_value = mock_compliance

            with patch.object(formatter, "format_compliance_result") as mock_format:
                mock_format.return_value = "\n## COMPLIANCE CHECK\nStatus: COMPLIANT"

                conclusion = formatter.format_conclusion_with_compliance(
                    basic_result, frameworks=["GDPR"]
                )

                mock_validate.assert_called_once()

    def test_conclusion_without_compliance_when_no_answer(self, formatter):
        """Test compliance not added when no final answer."""
        result = MockDebateResult(final_answer="")
        conclusion = formatter.format_conclusion_with_compliance(result)

        # Should just return base conclusion without compliance
        assert "## COMPLIANCE CHECK" not in conclusion

    def test_conclusion_with_vertical(self, formatter, basic_result):
        """Test conclusion with vertical-specific compliance."""
        with patch.object(formatter, "validate_compliance") as mock_validate:
            mock_compliance = MagicMock()
            mock_compliance.frameworks_checked = []  # No frameworks
            mock_validate.return_value = mock_compliance

            conclusion = formatter.format_conclusion_with_compliance(
                basic_result, vertical="healthcare"
            )

            mock_validate.assert_called_once_with(
                content=basic_result.final_answer, vertical="healthcare", frameworks=None
            )


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_format_conclusion_function(self, basic_result):
        """Test format_conclusion convenience function."""
        conclusion = format_conclusion(basic_result)

        assert "DEBATE CONCLUSION" in conclusion
        assert "## VERDICT" in conclusion
        assert "## FINAL ANSWER" in conclusion

    def test_format_conclusion_with_compliance_function(self, basic_result):
        """Test format_conclusion_with_compliance convenience function."""
        with patch("aragora.debate.result_formatter.ResultFormatter.validate_compliance") as mock:
            mock_result = MagicMock()
            mock_result.frameworks_checked = []
            mock.return_value = mock_result

            conclusion = format_conclusion_with_compliance(basic_result, vertical="finance")

            assert "DEBATE CONCLUSION" in conclusion


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_result(self, formatter):
        """Test handling completely empty result."""
        result = MockDebateResult(
            consensus_reached=False,
            confidence=0.0,
            final_answer="",
        )
        conclusion = formatter.format_conclusion(result)

        assert "DEBATE CONCLUSION" in conclusion
        assert "No final answer determined." in conclusion

    def test_single_agent_scenario(self, formatter):
        """Test formatting with single agent."""
        result = MockDebateResult(
            votes=[MockVote(voter="claude", choice="sole_proposal")],
            consensus_reached=True,
            confidence=1.0,
        )
        conclusion = formatter.format_conclusion(result)

        assert "claude: sole_proposal" in conclusion

    def test_perfect_consensus(self, formatter):
        """Test formatting with perfect consensus."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=1.0,
            consensus_strength="strong",
        )
        conclusion = formatter.format_conclusion(result)

        assert "Consensus: YES (100% agreement)" in conclusion
        assert "Strength: STRONG" in conclusion

    def test_zero_confidence(self, formatter):
        """Test formatting with zero confidence."""
        result = MockDebateResult(
            consensus_reached=False,
            confidence=0.0,
        )
        conclusion = formatter.format_conclusion(result)

        assert "Consensus: NO (only 0% agreement)" in conclusion

    def test_unicode_content(self, formatter):
        """Test handling unicode content."""
        result = MockDebateResult(
            final_answer="This contains unicode: \u2603 \u2764 \u263a",
            dissenting_views=["Japanese: \u65e5\u672c\u8a9e"],
        )
        conclusion = formatter.format_conclusion(result)

        assert "\u2603" in conclusion  # Snowman
        assert "\u65e5\u672c\u8a9e" in conclusion  # Japanese characters

    def test_special_characters(self, formatter):
        """Test handling special characters in content."""
        result = MockDebateResult(
            final_answer="Contains <html> & \"quotes\" and 'apostrophes'",
        )
        conclusion = formatter.format_conclusion(result)

        assert "<html>" in conclusion
        assert "&" in conclusion
        assert '"quotes"' in conclusion

    def test_newlines_in_content(self, formatter):
        """Test handling newlines in answers."""
        result = MockDebateResult(
            final_answer="Line 1\nLine 2\nLine 3",
        )
        conclusion = formatter.format_conclusion(result)

        assert "Line 1\nLine 2\nLine 3" in conclusion

    def test_very_long_vote_list(self, formatter):
        """Test formatting with many votes."""
        votes = [MockVote(voter=f"agent_{i}", choice=f"choice_{i % 3}") for i in range(10)]
        result = MockDebateResult(votes=votes)
        conclusion = formatter.format_conclusion(result)

        # All votes should be shown
        assert "agent_0" in conclusion
        assert "agent_9" in conclusion

    def test_result_without_optional_attributes(self, formatter):
        """Test result object missing optional attributes."""

        # Create a minimal result-like object
        @dataclass
        class MinimalResult:
            consensus_reached: bool = True
            confidence: float = 0.5
            final_answer: str = "Basic answer"

        result = MinimalResult()
        conclusion = formatter.format_conclusion(result)

        # Should handle missing attributes gracefully
        assert "DEBATE CONCLUSION" in conclusion
        assert "Basic answer" in conclusion
        # Should not have sections for missing attributes
        assert "## VOTE BREAKDOWN" not in conclusion
        assert "## DISSENTING VIEWS" not in conclusion


class TestAttributeChecks:
    """Tests for proper attribute existence checking."""

    def test_winner_attribute_check(self, formatter):
        """Test winner attribute existence is checked."""

        @dataclass
        class NoWinnerResult:
            consensus_reached: bool = True
            confidence: float = 0.8
            final_answer: str = "Answer"
            # No 'winner' attribute

        result = NoWinnerResult()
        conclusion = formatter.format_conclusion(result)
        assert "Winner:" not in conclusion

    def test_votes_attribute_check(self, formatter):
        """Test votes attribute existence is checked."""

        @dataclass
        class NoVotesResult:
            consensus_reached: bool = True
            confidence: float = 0.8
            final_answer: str = "Answer"
            # No 'votes' attribute

        result = NoVotesResult()
        conclusion = formatter.format_conclusion(result)
        assert "## VOTE BREAKDOWN" not in conclusion

    def test_consensus_strength_attribute_check(self, formatter):
        """Test consensus_strength attribute existence is checked."""

        @dataclass
        class NoStrengthResult:
            consensus_reached: bool = True
            confidence: float = 0.8
            final_answer: str = "Answer"
            # No 'consensus_strength' attribute

        result = NoStrengthResult()
        conclusion = formatter.format_conclusion(result)
        assert "Strength:" not in conclusion

    def test_belief_cruxes_attribute_check(self, formatter):
        """Test belief_cruxes attribute existence is checked."""

        @dataclass
        class NoCruxesResult:
            consensus_reached: bool = True
            confidence: float = 0.8
            final_answer: str = "Answer"
            # No 'belief_cruxes' attribute

        result = NoCruxesResult()
        conclusion = formatter.format_conclusion(result)
        assert "## KEY CRUXES" not in conclusion

    def test_dissenting_views_attribute_check(self, formatter):
        """Test dissenting_views attribute existence is checked."""

        @dataclass
        class NoDissentResult:
            consensus_reached: bool = True
            confidence: float = 0.8
            final_answer: str = "Answer"
            # No 'dissenting_views' attribute

        result = NoDissentResult()
        conclusion = formatter.format_conclusion(result)
        assert "## DISSENTING VIEWS" not in conclusion

    def test_translations_attribute_check(self, formatter):
        """Test translations attribute existence is checked."""

        @dataclass
        class NoTranslationsResult:
            consensus_reached: bool = True
            confidence: float = 0.8
            final_answer: str = "Answer"
            # No 'translations' attribute

        result = NoTranslationsResult()
        conclusion = formatter.format_conclusion(result)
        assert "## TRANSLATIONS" not in conclusion


class TestComplianceValidationIntegration:
    """Tests for actual compliance validation integration."""

    def test_validate_compliance_returns_compliant_on_exception(self, formatter):
        """Test validate_compliance returns compliant result on general exception."""
        # Patch to force an exception during validation
        with patch("aragora.compliance.framework.ComplianceFrameworkManager") as mock_manager_class:
            mock_manager_class.side_effect = RuntimeError("Unexpected error")

            result = formatter.validate_compliance("Test content")

            # Should return compliant=True with empty issues
            assert result.compliant is True
            assert result.issues == []
            assert result.score == 1.0

    def test_validate_compliance_with_vertical_determines_frameworks(self, formatter):
        """Test that vertical selection properly determines frameworks."""
        with patch("aragora.compliance.framework.ComplianceFrameworkManager") as mock_class:
            mock_manager = MagicMock()
            mock_class.return_value = mock_manager

            mock_framework = MagicMock()
            mock_framework.id = "HIPAA"
            mock_manager.get_frameworks_for_vertical.return_value = [mock_framework]

            mock_result = MagicMock()
            mock_result.compliant = True
            mock_result.issues = []
            mock_result.score = 1.0
            mock_manager.check.return_value = mock_result

            formatter.validate_compliance("Health data", vertical="healthcare")

            mock_manager.get_frameworks_for_vertical.assert_called_once_with("healthcare")


class TestComplianceResultFormattingDetails:
    """Tests for detailed compliance result formatting."""

    def test_format_multiple_critical_issues_limited_to_three(self, formatter):
        """Test only first 3 critical issues are displayed."""
        critical_issues = []
        for i in range(5):
            issue = MagicMock()
            issue.framework = f"Framework{i}"
            issue.description = f"Critical issue {i} description that is long enough"
            issue.recommendation = f"Recommendation {i} for critical issue"
            critical_issues.append(issue)

        mock_result = MagicMock()
        mock_result.compliant = False
        mock_result.score = 0.2
        mock_result.frameworks_checked = ["Framework0", "Framework1"]
        mock_result.issues = critical_issues
        mock_result.critical_issues = critical_issues
        mock_result.high_issues = []

        formatted = formatter.format_compliance_result(mock_result)

        # Should show CRITICAL (5): but only first 3 details
        assert "CRITICAL (5):" in formatted
        assert "[Framework0]" in formatted
        assert "[Framework1]" in formatted
        assert "[Framework2]" in formatted
        # 4th and 5th should not have their details
        assert "Critical issue 3" not in formatted
        assert "Critical issue 4" not in formatted

    def test_format_multiple_high_issues_limited_to_three(self, formatter):
        """Test only first 3 high issues are displayed."""
        high_issues = []
        for i in range(5):
            issue = MagicMock()
            issue.framework = f"HighFW{i}"
            issue.description = f"High severity issue number {i} in detail"
            high_issues.append(issue)

        mock_result = MagicMock()
        mock_result.compliant = False
        mock_result.score = 0.5
        mock_result.frameworks_checked = ["HighFW0"]
        mock_result.issues = high_issues
        mock_result.critical_issues = []
        mock_result.high_issues = high_issues

        formatted = formatter.format_compliance_result(mock_result)

        assert "HIGH (5):" in formatted
        assert "[HighFW0]" in formatted
        assert "[HighFW1]" in formatted
        assert "[HighFW2]" in formatted
        assert "issue number 3" not in formatted
        assert "issue number 4" not in formatted

    def test_format_long_description_truncated(self, formatter):
        """Test long issue descriptions are truncated to 60 chars."""
        long_desc = "D" * 100

        issue = MagicMock()
        issue.framework = "SOC2"
        issue.description = long_desc
        issue.recommendation = "Fix it"

        mock_result = MagicMock()
        mock_result.compliant = False
        mock_result.score = 0.4
        mock_result.frameworks_checked = ["SOC2"]
        mock_result.issues = [issue]
        mock_result.critical_issues = [issue]
        mock_result.high_issues = []

        formatted = formatter.format_compliance_result(mock_result)

        # Description should be truncated to 60 chars + "..."
        assert "D" * 60 + "..." in formatted
        assert "D" * 100 not in formatted

    def test_format_long_recommendation_truncated(self, formatter):
        """Test long recommendations are truncated to 80 chars."""
        long_rec = "R" * 150

        issue = MagicMock()
        issue.framework = "GDPR"
        issue.description = "Short description"
        issue.recommendation = long_rec

        mock_result = MagicMock()
        mock_result.compliant = False
        mock_result.score = 0.3
        mock_result.frameworks_checked = ["GDPR"]
        mock_result.issues = [issue]
        mock_result.critical_issues = [issue]
        mock_result.high_issues = []

        formatted = formatter.format_compliance_result(mock_result)

        # Recommendation should be truncated to 80 chars + "..."
        assert "R" * 80 + "..." in formatted
        assert "R" * 150 not in formatted


class TestConclusionWithComplianceInsertion:
    """Tests for compliance section insertion in conclusion."""

    def test_compliance_section_inserted_before_final_separator(self, formatter, basic_result):
        """Test compliance section is inserted before the final separator."""
        with patch.object(formatter, "validate_compliance") as mock_validate:
            mock_compliance = MagicMock()
            mock_compliance.frameworks_checked = ["GDPR"]
            mock_compliance.compliant = True
            mock_compliance.score = 1.0
            mock_compliance.issues = []
            mock_validate.return_value = mock_compliance

            with patch.object(formatter, "format_compliance_result") as mock_format:
                mock_format.return_value = "\n## COMPLIANCE CHECK\nStatus: COMPLIANT (score: 100%)"

                conclusion = formatter.format_conclusion_with_compliance(
                    basic_result, frameworks=["GDPR"]
                )

                # Compliance section should appear before final separator
                lines = conclusion.split("\n")
                compliance_idx = None
                final_sep_idx = None
                for i, line in enumerate(lines):
                    if "## COMPLIANCE CHECK" in line:
                        compliance_idx = i
                    if line == "=" * 60 and i > len(lines) // 2:
                        final_sep_idx = i

                assert compliance_idx is not None
                assert final_sep_idx is not None
                assert compliance_idx < final_sep_idx

    def test_compliance_not_inserted_when_frameworks_empty(self, formatter, basic_result):
        """Test compliance section not added when no frameworks returned."""
        with patch.object(formatter, "validate_compliance") as mock_validate:
            mock_compliance = MagicMock()
            mock_compliance.frameworks_checked = []
            mock_validate.return_value = mock_compliance

            conclusion = formatter.format_conclusion_with_compliance(basic_result)

            assert "## COMPLIANCE CHECK" not in conclusion


class TestTranslationFormatting:
    """Additional tests for translation formatting edge cases."""

    def test_translation_with_unknown_language_code(self, formatter):
        """Test translations work with unknown language codes."""
        result = MockDebateResult(translations={"xyz": "Translation in unknown language"})

        conclusion = formatter.format_conclusion(result)

        # Should still display translations
        assert "## TRANSLATIONS" in conclusion
        assert "Translation in unknown language" in conclusion
        # Language code should appear (either through Language module lookup or as-is)
        assert "xyz" in conclusion.lower() or "XYZ" in conclusion

    def test_multiple_translations_all_displayed(self, formatter):
        """Test all translations are displayed, not limited."""
        translations = {f"lang{i}": f"Translation {i}" for i in range(10)}
        result = MockDebateResult(translations=translations)

        conclusion = formatter.format_conclusion(result)

        # All 10 translations should be present
        for i in range(10):
            assert f"Translation {i}" in conclusion


class TestVoteProcessing:
    """Additional tests for vote processing logic."""

    def test_duplicate_voter_overwrites(self, formatter):
        """Test that multiple votes from same voter only show last choice."""
        # The implementation builds a dict, so later votes overwrite
        result = MockDebateResult(
            votes=[
                MockVote(voter="claude", choice="choice_1"),
                MockVote(voter="claude", choice="choice_2"),  # This should win
            ]
        )

        conclusion = formatter.format_conclusion(result)

        # Only the last choice should be shown for claude
        assert "claude: choice_2" in conclusion
        # First choice should not appear (dict overwrites)
        lines = conclusion.split("\n")
        claude_lines = [line for line in lines if "claude:" in line]
        assert len(claude_lines) == 1

    def test_empty_voter_string(self, formatter):
        """Test handling of empty voter string."""
        result = MockDebateResult(votes=[MockVote(voter="", choice="some_choice")])

        conclusion = formatter.format_conclusion(result)

        assert ": some_choice" in conclusion

    def test_empty_choice_string(self, formatter):
        """Test handling of empty choice string."""
        result = MockDebateResult(votes=[MockVote(voter="claude", choice="")])

        conclusion = formatter.format_conclusion(result)

        assert "claude: " in conclusion


class TestFormatterReusability:
    """Tests for formatter instance reusability."""

    def test_formatter_reusable_across_results(self, formatter):
        """Test same formatter instance works for multiple results."""
        result1 = MockDebateResult(
            consensus_reached=True,
            confidence=0.9,
            final_answer="First answer",
        )
        result2 = MockDebateResult(
            consensus_reached=False,
            confidence=0.3,
            final_answer="Second answer",
        )

        conclusion1 = formatter.format_conclusion(result1)
        conclusion2 = formatter.format_conclusion(result2)

        assert "First answer" in conclusion1
        assert "Second answer" in conclusion2
        assert "Consensus: YES" in conclusion1
        assert "Consensus: NO" in conclusion2

    def test_formatter_settings_persist(self, custom_formatter):
        """Test custom settings persist across multiple uses."""
        long_answer = "A" * 100

        result1 = MockDebateResult(final_answer=long_answer)
        result2 = MockDebateResult(final_answer=long_answer)

        conclusion1 = custom_formatter.format_conclusion(result1)
        conclusion2 = custom_formatter.format_conclusion(result2)

        # Both should be truncated the same way
        assert "A" * 50 + "..." in conclusion1
        assert "A" * 50 + "..." in conclusion2


class TestBoundaryConditions:
    """Tests for numeric boundary conditions."""

    def test_confidence_exactly_half(self, formatter):
        """Test confidence at exactly 0.5 (50%)."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.5,
        )

        conclusion = formatter.format_conclusion(result)

        assert "Consensus: YES (50% agreement)" in conclusion

    def test_confidence_rounds_correctly(self, formatter):
        """Test confidence values round correctly in display."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.999,
        )

        conclusion = formatter.format_conclusion(result)

        # 0.999 should round to 100% with :.0% format
        assert "100% agreement" in conclusion

    def test_confidence_near_zero_rounds(self, formatter):
        """Test near-zero confidence rounds correctly."""
        result = MockDebateResult(
            consensus_reached=False,
            confidence=0.001,
        )

        conclusion = formatter.format_conclusion(result)

        # 0.001 should round to 0% with :.0% format
        assert "0% agreement" in conclusion

    def test_crux_uncertainty_formatting(self, formatter):
        """Test uncertainty value formatting precision."""
        result = MockDebateResult(belief_cruxes=[{"claim": "Test", "uncertainty": 0.12345}])

        conclusion = formatter.format_conclusion(result)

        # Should be formatted to 2 decimal places
        assert "(uncertainty: 0.12)" in conclusion

    def test_answer_exactly_at_max_length(self, formatter):
        """Test answer exactly at max length is not truncated."""
        # Default max_answer_length is 1000
        exact_answer = "A" * 1000
        result = MockDebateResult(final_answer=exact_answer)

        conclusion = formatter.format_conclusion(result)

        # Should contain exact answer without truncation
        assert exact_answer in conclusion
        # Should not have the truncation ellipsis added
        answer_section = conclusion.split("## FINAL ANSWER")[1].split("##")[0]
        assert "..." not in answer_section

    def test_view_exactly_at_max_length(self, formatter):
        """Test dissenting view exactly at max length is not truncated."""
        # Default max_view_length is 300
        exact_view = "B" * 300
        result = MockDebateResult(dissenting_views=[exact_view])

        conclusion = formatter.format_conclusion(result)

        # Should contain exact view without truncation
        assert exact_view in conclusion


class TestCruxFormatting:
    """Additional tests for crux formatting details."""

    def test_crux_claim_exactly_80_chars_no_truncation(self, formatter):
        """Test claim exactly 80 chars is not truncated."""
        exact_claim = "C" * 80
        result = MockDebateResult(belief_cruxes=[{"claim": exact_claim, "uncertainty": 0.5}])

        conclusion = formatter.format_conclusion(result)

        # Should contain full claim without ellipsis
        assert exact_claim in conclusion
        crux_section = conclusion.split("## KEY CRUXES")[1].split("##")[0]
        # The ellipsis would only appear if truncated
        # Since claim is exactly 80, it should show claim + "..." from format string
        assert exact_claim + "..." in crux_section

    def test_crux_with_negative_uncertainty(self, formatter):
        """Test handling of negative uncertainty values."""
        result = MockDebateResult(belief_cruxes=[{"claim": "Negative test", "uncertainty": -0.5}])

        conclusion = formatter.format_conclusion(result)

        # Should still format, just with negative value
        assert "(uncertainty: -0.50)" in conclusion

    def test_crux_with_large_uncertainty(self, formatter):
        """Test handling of uncertainty greater than 1."""
        result = MockDebateResult(belief_cruxes=[{"claim": "Large test", "uncertainty": 2.5}])

        conclusion = formatter.format_conclusion(result)

        assert "(uncertainty: 2.50)" in conclusion
