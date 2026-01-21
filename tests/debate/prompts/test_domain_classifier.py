"""
Tests for the domain classifier module.

Tests keyword-based domain detection for debate questions.
"""

import pytest

from aragora.debate.prompts.domain_classifier import (
    detect_question_domain_keywords,
    word_match,
    get_domain_keywords,
    classify_by_keywords,
    DOMAIN_KEYWORDS,
)


class TestWordMatch:
    """Tests for word_match function."""

    def test_exact_match(self):
        """Test exact word matching."""
        assert word_match("this is about api design", ["api"]) is True

    def test_no_substring_match(self):
        """Test that substrings don't match (e.g., 'api' in 'capitalism')."""
        assert word_match("discuss capitalism", ["api"]) is False

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        assert word_match("what is the meaning of life", ["meaning"]) is True

    def test_multiple_keywords(self):
        """Test matching with multiple keywords."""
        assert word_match("database architecture", ["code", "database"]) is True

    def test_no_match(self):
        """Test when no keywords match."""
        assert word_match("the weather is nice", ["code", "api"]) is False

    def test_multi_word_keyword(self):
        """Test matching multi-word keywords."""
        assert word_match("what is the good life", ["good life"]) is True


class TestDetectQuestionDomainKeywords:
    """Tests for detect_question_domain_keywords function."""

    def test_philosophical_meaning_of_life(self):
        """Test philosophical domain detection."""
        assert detect_question_domain_keywords("What is the meaning of life?") == "philosophical"

    def test_philosophical_consciousness(self):
        """Test consciousness as philosophical."""
        assert (
            detect_question_domain_keywords("Is consciousness fundamental to reality?")
            == "philosophical"
        )

    def test_philosophical_afterlife(self):
        """Test afterlife as philosophical."""
        assert detect_question_domain_keywords("What happens in the afterlife?") == "philosophical"

    def test_ethics_should_question(self):
        """Test ethics domain detection."""
        assert detect_question_domain_keywords("Should we ban autonomous weapons?") == "ethics"

    def test_ethics_moral_question(self):
        """Test moral as ethics domain."""
        assert detect_question_domain_keywords("Is it moral to eat meat?") == "ethics"

    def test_technical_api_design(self):
        """Test technical domain detection."""
        # Note: "should" triggers ethics detection first, so use a question without "should"
        assert (
            detect_question_domain_keywords("What is the best API design pattern?") == "technical"
        )

    def test_technical_database(self):
        """Test database as technical."""
        assert (
            detect_question_domain_keywords("What database architecture works best?") == "technical"
        )

    def test_technical_no_false_positive(self):
        """Test that 'api' doesn't match inside 'capitalism'."""
        # This should be 'ethics' (due to "best" being close to "good or bad") or 'general', not 'technical'
        result = detect_question_domain_keywords("Is capitalism the best economic system?")
        assert result != "technical"

    def test_general_fallback(self):
        """Test general domain for unclassified questions."""
        # Avoid "should" which triggers ethics detection
        assert detect_question_domain_keywords("What color is the sky?") == "general"

    def test_philosophical_priority_over_ethics(self):
        """Test that philosophical keywords take priority."""
        # "meaning" + "should" - philosophical should win
        assert (
            detect_question_domain_keywords("What meaning should we find in life?")
            == "philosophical"
        )


class TestGetDomainKeywords:
    """Tests for get_domain_keywords function."""

    def test_get_philosophical_keywords(self):
        """Test getting philosophical keywords."""
        keywords = get_domain_keywords("philosophical")
        assert "meaning" in keywords
        assert "consciousness" in keywords
        assert "soul" in keywords

    def test_get_technical_keywords(self):
        """Test getting technical keywords."""
        keywords = get_domain_keywords("technical")
        assert "api" in keywords
        assert "database" in keywords

    def test_unknown_domain_returns_empty(self):
        """Test unknown domain returns empty set."""
        keywords = get_domain_keywords("unknown_domain")
        assert keywords == set()


class TestClassifyByKeywords:
    """Tests for classify_by_keywords function."""

    def test_classify_philosophical_question(self):
        """Test classification returns domain matches."""
        result = classify_by_keywords("What is the meaning of consciousness?")
        assert result["philosophical"] is True
        assert result["technical"] is False

    def test_classify_mixed_question(self):
        """Test question that could match multiple domains."""
        result = classify_by_keywords("Should we implement the database security fix?")
        assert result["technical"] is True
        assert result["ethics"] is True  # "should"

    def test_classify_no_matches(self):
        """Test question with no domain matches."""
        result = classify_by_keywords("The weather is nice today")
        assert all(not v for v in result.values())


class TestDomainKeywordsConstant:
    """Tests for DOMAIN_KEYWORDS constant."""

    def test_has_required_domains(self):
        """Test that all required domains are present."""
        assert "philosophical" in DOMAIN_KEYWORDS
        assert "ethics" in DOMAIN_KEYWORDS
        assert "technical" in DOMAIN_KEYWORDS

    def test_keywords_are_lists(self):
        """Test that keyword values are lists."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            assert isinstance(keywords, list), f"{domain} keywords should be a list"

    def test_keywords_are_lowercase(self):
        """Test that all keywords are lowercase."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in {domain} should be lowercase"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
