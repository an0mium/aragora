"""Tests for Pulse trending topics in debate context.

Validates that:
1. ArenaKnowledgeManager.fetch_pulse_topics queries the pulse store
2. PromptBuilder.format_pulse_context formats pulse data with velocity and source
3. Pulse context is injected into proposal prompts when enabled
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# =========================================================================
# ArenaKnowledgeManager.fetch_pulse_topics tests
# =========================================================================


class TestFetchPulseTopics:
    """Test pulse topic fetching in knowledge manager."""

    @pytest.fixture
    def pulse_store(self):
        """Create a mock pulse store."""
        store = MagicMock()
        record1 = MagicMock()
        record1.topic_text = "API rate limiting best practices"
        record1.platform = "hackernews"
        record1.volume = 5000
        record1.category = "tech"
        record1.hours_ago = 2.5

        record2 = MagicMock()
        record2.topic_text = "New database scaling strategies"
        record2.platform = "reddit"
        record2.volume = 3000
        record2.category = "tech"
        record2.hours_ago = 6.0

        record3 = MagicMock()
        record3.topic_text = "Celebrity gossip latest"
        record3.platform = "twitter"
        record3.volume = 50000
        record3.category = "entertainment"
        record3.hours_ago = 1.0

        store.get_recent_topics.return_value = [record1, record2, record3]
        return store

    @pytest.fixture
    def km(self, pulse_store):
        """Create an ArenaKnowledgeManager with pulse store."""
        from aragora.debate.knowledge_manager import ArenaKnowledgeManager

        return ArenaKnowledgeManager(
            pulse_store=pulse_store,
            enable_pulse_context=True,
        )

    def test_fetch_returns_topics(self, km):
        """fetch_pulse_topics returns matching topics."""
        results = km.fetch_pulse_topics("API rate limiting design", limit=5)
        assert len(results) > 0

    def test_fetch_sorts_by_relevance(self, km):
        """Topics matching the task come first."""
        results = km.fetch_pulse_topics("API rate limiting design")
        # The API rate limiting topic should rank first
        assert results[0]["topic"] == "API rate limiting best practices"

    def test_fetch_includes_metadata(self, km):
        """Each result includes platform, volume, category, hours_ago."""
        results = km.fetch_pulse_topics("API rate limiting")
        topic = results[0]
        assert "platform" in topic
        assert "volume" in topic
        assert "category" in topic
        assert "hours_ago" in topic

    def test_fetch_respects_limit(self, km):
        """Results are limited to the requested count."""
        results = km.fetch_pulse_topics("anything", limit=1)
        assert len(results) <= 1

    def test_fetch_disabled(self, pulse_store):
        """No results when enable_pulse_context=False."""
        from aragora.debate.knowledge_manager import ArenaKnowledgeManager

        km = ArenaKnowledgeManager(
            pulse_store=pulse_store,
            enable_pulse_context=False,
        )
        results = km.fetch_pulse_topics("API rate limiting")
        assert results == []

    def test_fetch_no_store(self):
        """No results when pulse_store is None."""
        from aragora.debate.knowledge_manager import ArenaKnowledgeManager

        km = ArenaKnowledgeManager(enable_pulse_context=True)
        results = km.fetch_pulse_topics("API rate limiting")
        assert results == []

    def test_fetch_empty_store(self, pulse_store):
        """No results when store returns empty."""
        from aragora.debate.knowledge_manager import ArenaKnowledgeManager

        pulse_store.get_recent_topics.return_value = []
        km = ArenaKnowledgeManager(
            pulse_store=pulse_store,
            enable_pulse_context=True,
        )
        results = km.fetch_pulse_topics("API rate limiting")
        assert results == []

    def test_fetch_handles_errors(self, pulse_store):
        """Errors in pulse store are handled gracefully."""
        from aragora.debate.knowledge_manager import ArenaKnowledgeManager

        pulse_store.get_recent_topics.side_effect = AttributeError("broken")
        km = ArenaKnowledgeManager(
            pulse_store=pulse_store,
            enable_pulse_context=True,
        )
        results = km.fetch_pulse_topics("API rate limiting")
        assert results == []


# =========================================================================
# PromptBuilder.format_pulse_context tests
# =========================================================================


class TestFormatPulseContext:
    """Test pulse context formatting in PromptBuilder."""

    @pytest.fixture
    def mock_protocol(self):
        """Create a mock debate protocol."""
        protocol = MagicMock()
        protocol.rounds = 3
        protocol.asymmetric_stances = False
        protocol.agreement_intensity = None
        protocol.get_round_phase.return_value = None
        return protocol

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        env = MagicMock()
        env.task = "Design a rate limiter"
        env.context = ""
        return env

    @pytest.fixture
    def pulse_topics(self):
        """Sample pulse topics."""
        return [
            {
                "topic": "API rate limiting",
                "platform": "hackernews",
                "volume": 15000,
                "category": "tech",
                "hours_ago": 2.5,
            },
            {
                "topic": "Redis caching strategies",
                "platform": "reddit",
                "volume": 800,
                "category": "tech",
                "hours_ago": 8.0,
            },
        ]

    def test_empty_when_no_topics(self, mock_protocol, mock_env):
        """Empty string when no pulse topics set."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        assert builder.format_pulse_context() == ""

    def test_formats_topics(self, mock_protocol, mock_env, pulse_topics):
        """Topics are formatted with title, source, volume, recency."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_topics(pulse_topics)
        context = builder.format_pulse_context()
        assert "PULSE" in context
        assert "API rate limiting" in context
        assert "hackernews" in context
        assert "15,000" in context
        assert "2.5h ago" in context

    def test_high_velocity_label(self, mock_protocol, mock_env):
        """High volume topics get a velocity label."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_topics(
            [
                {
                    "topic": "Breaking news",
                    "platform": "twitter",
                    "volume": 50000,
                    "category": "news",
                    "hours_ago": 0.5,
                }
            ]
        )
        context = builder.format_pulse_context()
        assert "HIGH VELOCITY" in context

    def test_rising_velocity_label(self, mock_protocol, mock_env):
        """Medium volume topics get RISING label."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_topics(
            [
                {
                    "topic": "New framework",
                    "platform": "reddit",
                    "volume": 5000,
                    "category": "tech",
                    "hours_ago": 3.0,
                }
            ]
        )
        context = builder.format_pulse_context()
        assert "RISING" in context

    def test_respects_max_topics(self, mock_protocol, mock_env, pulse_topics):
        """Respects max_topics limit."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_topics(pulse_topics)
        context = builder.format_pulse_context(max_topics=1)
        # Only one topic should appear
        assert "API rate limiting" in context
        assert "Redis" not in context

    def test_pulse_in_proposal_prompt(self, mock_protocol, mock_env, pulse_topics):
        """Pulse context appears in proposal prompt when set."""
        from aragora.debate.prompt_builder import PromptBuilder

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)
        builder.set_pulse_topics(pulse_topics)
        agent = MagicMock()
        agent.name = "test-agent"
        agent.role = "proposer"
        prompt = builder.build_proposal_prompt(agent)
        assert "PULSE" in prompt
