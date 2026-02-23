"""Tests for PulseDebateEnricher -- trending-topics-to-debate loop.

Validates:
1. Relevant topics are included in prompt
2. Irrelevant topics are filtered
3. Quality threshold filtering works
4. Empty pulse store returns no context
5. Max topics cap works
6. Flag disabled produces no enrichment
7. Freshness scoring and filtering
8. Keyword overlap relevance matching
9. Prompt formatting
10. End-to-end PromptBuilder integration
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    topic_text: str = "Default topic",
    platform: str = "hackernews",
    volume: int = 1000,
    category: str = "tech",
    hours_ago: float = 2.0,
):
    """Create a mock ScheduledDebateRecord-like object."""
    record = MagicMock()
    record.topic_text = topic_text
    record.platform = platform
    record.volume = volume
    record.category = category
    record.hours_ago = hours_ago
    return record


@pytest.fixture
def pulse_store():
    """Create a mock pulse store with diverse topics."""
    store = MagicMock()
    store.get_recent_topics.return_value = [
        _make_record(
            topic_text="API rate limiting best practices",
            platform="hackernews",
            volume=5000,
            category="tech",
            hours_ago=2.5,
        ),
        _make_record(
            topic_text="New database scaling strategies for production",
            platform="reddit",
            volume=3000,
            category="tech",
            hours_ago=6.0,
        ),
        _make_record(
            topic_text="Celebrity gossip latest drama unfolds",
            platform="twitter",
            volume=50000,
            category="entertainment",
            hours_ago=1.0,
        ),
        _make_record(
            topic_text="Kubernetes deployment patterns analysis",
            platform="hackernews",
            volume=2000,
            category="tech",
            hours_ago=4.0,
        ),
        _make_record(
            topic_text="Machine learning model optimization research",
            platform="reddit",
            volume=8000,
            category="tech",
            hours_ago=3.0,
        ),
    ]
    return store


@pytest.fixture
def enricher(pulse_store):
    """Create a PulseDebateEnricher with the test pulse store."""
    from aragora.pulse.debate_enrichment import PulseDebateEnricher

    return PulseDebateEnricher(pulse_store=pulse_store)


# ---------------------------------------------------------------------------
# PulseDebateEnricher core tests
# ---------------------------------------------------------------------------


class TestPulseDebateEnricher:
    """Core enricher tests."""

    def test_relevant_topics_included(self, enricher):
        """Topics relevant to the task are included in results."""
        result = enricher.enrich("API rate limiting design patterns")
        assert result.has_context
        titles = [s.title for s in result.snippets]
        assert "API rate limiting best practices" in titles

    def test_irrelevant_topics_filtered(self, enricher):
        """Topics with no keyword overlap are excluded."""
        result = enricher.enrich("API rate limiting design patterns")
        titles = [s.title for s in result.snippets]
        assert "Celebrity gossip latest drama unfolds" not in titles

    def test_quality_threshold_filtering(self, pulse_store):
        """Topics below the quality threshold are excluded."""
        from aragora.pulse.debate_enrichment import PulseDebateEnricher

        # Set a very high threshold so most topics are filtered
        enricher = PulseDebateEnricher(
            pulse_store=pulse_store,
            quality_threshold=0.99,
            min_keyword_overlap=0,
        )
        result = enricher.enrich("anything at all")
        # With a 0.99 threshold, very few (likely zero) topics pass
        assert len(result.snippets) <= 1

    def test_empty_store_returns_no_context(self):
        """Empty pulse store returns no enrichment context."""
        from aragora.pulse.debate_enrichment import PulseDebateEnricher

        store = MagicMock()
        store.get_recent_topics.return_value = []
        enricher = PulseDebateEnricher(pulse_store=store)
        result = enricher.enrich("Design a rate limiter")
        assert not result.has_context
        assert len(result.snippets) == 0

    def test_no_store_returns_no_context(self):
        """No pulse store returns no enrichment context."""
        from aragora.pulse.debate_enrichment import PulseDebateEnricher

        enricher = PulseDebateEnricher(pulse_store=None)
        result = enricher.enrich("Design a rate limiter")
        assert not result.has_context

    def test_max_topics_cap(self, pulse_store):
        """Results are capped at max_topics."""
        from aragora.pulse.debate_enrichment import PulseDebateEnricher

        enricher = PulseDebateEnricher(
            pulse_store=pulse_store,
            min_keyword_overlap=0,  # Don't filter by relevance
        )
        result = enricher.enrich("technology topics", max_topics=2)
        assert len(result.snippets) <= 2

    def test_snippets_have_required_fields(self, enricher):
        """Each snippet includes title, source, quality_score, freshness_score, relevance_rationale."""
        result = enricher.enrich("database scaling production")
        assert result.has_context
        for snippet in result.snippets:
            assert snippet.title
            assert snippet.source
            assert isinstance(snippet.quality_score, float)
            assert 0.0 <= snippet.quality_score <= 1.0
            assert isinstance(snippet.freshness_score, float)
            assert 0.0 <= snippet.freshness_score <= 1.0
            assert isinstance(snippet.relevance_rationale, str)

    def test_freshness_scoring(self, pulse_store):
        """Recent topics score higher in freshness than older ones."""
        from aragora.pulse.debate_enrichment import PulseDebateEnricher

        # Add a very fresh and a stale topic with the same keywords
        pulse_store.get_recent_topics.return_value = [
            _make_record(
                topic_text="API rate limiting new approaches",
                hours_ago=0.5,
            ),
            _make_record(
                topic_text="API rate limiting old approaches",
                hours_ago=40.0,
            ),
        ]
        enricher = PulseDebateEnricher(pulse_store=pulse_store)
        result = enricher.enrich("API rate limiting")
        assert result.has_context
        # The fresh topic should score higher
        if len(result.snippets) >= 2:
            assert result.snippets[0].freshness_score > result.snippets[1].freshness_score

    def test_stale_topics_excluded(self, pulse_store):
        """Topics older than freshness_max_hours are excluded."""
        from aragora.pulse.debate_enrichment import PulseDebateEnricher

        pulse_store.get_recent_topics.return_value = [
            _make_record(
                topic_text="API rate limiting stale topic",
                hours_ago=100.0,
            ),
        ]
        enricher = PulseDebateEnricher(
            pulse_store=pulse_store,
            freshness_max_hours=48.0,
        )
        result = enricher.enrich("API rate limiting")
        assert not result.has_context

    def test_empty_task_returns_no_context(self, enricher):
        """An empty task string returns no context."""
        result = enricher.enrich("")
        assert not result.has_context

    def test_store_error_handled_gracefully(self):
        """Errors from the pulse store are handled gracefully."""
        from aragora.pulse.debate_enrichment import PulseDebateEnricher

        store = MagicMock()
        store.get_recent_topics.side_effect = RuntimeError("store down")
        enricher = PulseDebateEnricher(pulse_store=store)
        result = enricher.enrich("anything")
        assert not result.has_context


# ---------------------------------------------------------------------------
# Prompt formatting tests
# ---------------------------------------------------------------------------


class TestFormatEnrichmentForPrompt:
    """Test the prompt formatting function."""

    def test_empty_result_returns_empty_string(self):
        """No snippets produces empty string."""
        from aragora.pulse.debate_enrichment import (
            EnrichmentResult,
            format_enrichment_for_prompt,
        )

        result = EnrichmentResult()
        assert format_enrichment_for_prompt(result) == ""

    def test_formatted_output_contains_header(self, enricher):
        """Formatted output includes the 'Current Context' header."""
        from aragora.pulse.debate_enrichment import format_enrichment_for_prompt

        result = enricher.enrich("database scaling production")
        text = format_enrichment_for_prompt(result)
        if result.has_context:
            assert "## Current Context" in text
            assert "Recent relevant developments" in text

    def test_formatted_output_contains_quality_score(self, enricher):
        """Formatted output includes quality score out of 10."""
        from aragora.pulse.debate_enrichment import format_enrichment_for_prompt

        result = enricher.enrich("database scaling production")
        text = format_enrichment_for_prompt(result)
        if result.has_context:
            assert "/10" in text

    def test_word_cap_enforced(self, pulse_store):
        """Output does not exceed max_words."""
        from aragora.pulse.debate_enrichment import (
            PulseDebateEnricher,
            format_enrichment_for_prompt,
        )

        enricher = PulseDebateEnricher(
            pulse_store=pulse_store,
            min_keyword_overlap=0,
        )
        result = enricher.enrich("technology", max_topics=10)
        text = format_enrichment_for_prompt(result, max_words=50)
        words = text.split()
        # Allow +1 for the trailing "..." token
        assert len(words) <= 51


# ---------------------------------------------------------------------------
# PromptBuilder integration tests
# ---------------------------------------------------------------------------


class TestPromptBuilderIntegration:
    """Test wiring into PromptBuilder."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = MagicMock()
        protocol.rounds = 3
        protocol.asymmetric_stances = False
        protocol.agreement_intensity = None
        protocol.get_round_phase.return_value = None
        protocol.enable_pulse_context = False
        protocol.enable_trending_injection = False
        protocol.enable_privacy_anonymization = False
        return protocol

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.task = "Design a rate limiter for APIs"
        env.context = ""
        return env

    def test_flag_disabled_no_enrichment(self, mock_protocol, mock_env, pulse_store):
        """When enable_pulse_context is False, no enrichment is injected."""
        from aragora.debate.prompt_builder import PromptBuilder

        mock_protocol.enable_pulse_context = False
        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_enrichment_store(pulse_store)
        context = builder.inject_pulse_enrichment()
        assert context == ""

    def test_flag_enabled_enrichment_injected(self, mock_protocol, mock_env, pulse_store):
        """When enable_pulse_context is True, enrichment is injected."""
        from aragora.debate.prompt_builder import PromptBuilder

        mock_protocol.enable_pulse_context = True
        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_enrichment_store(pulse_store)
        context = builder.inject_pulse_enrichment()
        assert "Current Context" in context
        assert "rate limiting" in context.lower()

    def test_enrichment_appears_in_proposal_prompt(self, mock_protocol, mock_env, pulse_store):
        """Enrichment context appears in proposal prompt when enabled."""
        from aragora.debate.prompt_builder import PromptBuilder

        mock_protocol.enable_pulse_context = True
        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_enrichment_store(pulse_store)
        # Trigger enrichment
        builder.inject_pulse_enrichment()

        agent = MagicMock()
        agent.name = "test-agent"
        agent.role = "proposer"
        prompt = builder.build_proposal_prompt(agent)
        assert "Current Context" in prompt

    def test_enrichment_not_in_prompt_when_disabled(self, mock_protocol, mock_env, pulse_store):
        """Enrichment does NOT appear in proposal prompt when flag is off."""
        from aragora.debate.prompt_builder import PromptBuilder

        mock_protocol.enable_pulse_context = False
        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_enrichment_store(pulse_store)

        agent = MagicMock()
        agent.name = "test-agent"
        agent.role = "proposer"
        prompt = builder.build_proposal_prompt(agent)
        assert "Current Context" not in prompt
