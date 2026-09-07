"""Tests for Culture → Debate Protocol integration."""

import asyncio

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from aragora.events.types import StreamEvent, StreamEventType
from aragora.events.cross_subscribers import (
    CrossSubscriberManager,
    reset_cross_subscriber_manager,
)


@pytest.fixture
def fresh_manager():
    """Create a fresh manager for each test, with domain-home subscribers applied.

    ``mound_to_culture`` moved to ``KnowledgeEventSubscriber`` (P4a Batch E2c) and
    is wired only via ``apply_registered_subscribers``, so this bootstraps rather
    than using a bare ``CrossSubscriberManager()``.
    """
    reset_cross_subscriber_manager()
    from aragora.debate.event_subscribers import bootstrap_debate_event_subscribers

    return bootstrap_debate_event_subscribers()


class TestCultureToDebateHandler:
    """Tests for the culture_to_debate handler (pattern updates)."""

    def test_handler_registered(self, fresh_manager):
        """Test that culture_to_debate handler is registered."""
        assert "culture_to_debate" in fresh_manager._stats

    def test_ignores_non_culture_mound_updates(self, fresh_manager):
        """Test that handler ignores MOUND_UPDATED with other types."""
        event = StreamEvent(
            type=StreamEventType.MOUND_UPDATED,
            data={
                "update_type": "knowledge_node",
                "workspace_id": "ws1",
            },
        )
        # Should not raise or process
        fresh_manager.dispatch(event)
        stats = fresh_manager.get_stats()
        assert stats["culture_to_debate"]["events_processed"] == 1

    def test_processes_culture_pattern_updates(self, fresh_manager):
        """Test that handler processes culture_patterns updates."""
        event = StreamEvent(
            type=StreamEventType.MOUND_UPDATED,
            data={
                "update_type": "culture_patterns",
                "patterns_count": 5,
                "workspace_id": "ws1",
            },
        )
        fresh_manager.dispatch(event)
        stats = fresh_manager.get_stats()
        assert stats["culture_to_debate"]["events_processed"] == 1


class TestMoundToCultureHandler:
    """Tests for the mound_to_culture handler (debate start)."""

    def test_handler_registered(self, fresh_manager):
        """Test that mound_to_culture handler is registered."""
        assert "mound_to_culture" in fresh_manager._stats

    def test_triggers_on_debate_start(self, fresh_manager):
        """Test that handler is triggered on DEBATE_START."""
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate_001",
                "domain": "software",
                "question": "How should we implement caching?",
                "protocol": {"rounds": 3, "consensus": "majority"},
            },
        )
        # Handler should be called (even if KM not available)
        fresh_manager.dispatch(event)
        stats = fresh_manager.get_stats()
        assert stats["mound_to_culture"]["events_processed"] == 1

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_retrieves_culture_profile(self, mock_get_mound, fresh_manager):
        """Test that handler retrieves culture profile from KM."""
        # Create mock mound with culture profile
        mock_mound = MagicMock()
        mock_profile = MagicMock()
        mock_profile.dominant_pattern = None
        mock_profile.patterns = []
        mock_mound.get_culture_profile = AsyncMock(return_value=mock_profile)
        mock_get_mound.return_value = mock_mound

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate_002",
                "domain": "legal",
                "question": "What are the contract requirements?",
            },
        )
        fresh_manager.dispatch(event)
        stats = fresh_manager.get_stats()
        assert stats["mound_to_culture"]["events_processed"] == 1

    @pytest.mark.asyncio
    @patch("aragora.knowledge.mound.get_knowledge_mound")
    async def test_stores_profile_when_event_loop_already_running(
        self, mock_get_mound, fresh_manager
    ):
        """The create_task() branch (running loop) must persist the profile too.

        Regression test: previously only the ``asyncio.run`` fallback (no
        running loop) called ``_store_debate_culture``; the common case of a
        handler invoked from within an already-running loop scheduled the
        retrieval via ``create_task`` and silently dropped the result.
        """
        from aragora.knowledge.event_subscribers import get_knowledge_event_subscriber

        mock_mound = MagicMock()
        mock_mound.is_initialized = True
        mock_profile = MagicMock()
        mock_profile.dominant_pattern = None
        mock_profile.patterns = []
        mock_mound.get_culture_profile = AsyncMock(return_value=mock_profile)
        mock_get_mound.return_value = mock_mound

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"debate_id": "debate_async_001", "domain": "software"},
        )
        # dispatch() is synchronous; called from this async test it runs with
        # a loop already active, so the handler takes the create_task path.
        fresh_manager.dispatch(event)

        # Let the scheduled task and its done-callback run to completion.
        await asyncio.sleep(0.05)

        subscriber = get_knowledge_event_subscriber()
        assert "debate_async_001" in subscriber._debate_cultures


class TestKnowledgeSubscriberSurvivesRepeatedBootstrap:
    """Regression: repeated bootstrap calls must not drop culture state.

    Production calls ``bootstrap_debate_event_subscribers()`` from both
    ``Arena.init_context`` (stores culture hints) and
    ``Arena.get_culture_hints`` (reads them back), without resetting the
    cross-subscriber manager in between. ``register()`` previously
    unconditionally replaced the registered ``KnowledgeEventSubscriber``
    instance on every call, so the second bootstrap call silently swapped in
    a fresh, empty subscriber even though
    ``CrossSubscriberManager.apply_registered_subscribers`` had already wired
    the first instance's bound methods into the (still-live) manager.
    """

    def test_hints_survive_second_bootstrap_call(self, fresh_manager):
        from aragora.debate.event_subscribers import bootstrap_debate_event_subscribers
        from aragora.knowledge.event_subscribers import get_knowledge_event_subscriber

        subscriber = get_knowledge_event_subscriber()
        subscriber._debate_cultures["debate_repeat_001"] = {
            "protocol_hints": {"recommended_consensus": "unanimous"},
            "domain": "legal",
        }

        # Simulate the second `bootstrap_debate_event_subscribers()` call that
        # `get_culture_hints()` makes against the same, still-live manager.
        bootstrap_debate_event_subscribers()

        hints = get_knowledge_event_subscriber().get_debate_culture_hints("debate_repeat_001")
        assert hints == {"recommended_consensus": "unanimous"}

    def test_register_reuses_existing_instance(self, fresh_manager):
        from aragora.knowledge import event_subscribers as knowledge_home

        first = knowledge_home.get_knowledge_event_subscriber()
        knowledge_home.register()
        second = knowledge_home.get_knowledge_event_subscriber()
        assert first is second


class TestDebateCultureHints:
    """Tests for debate culture hints storage and retrieval.

    Storage lives on ``KnowledgeEventSubscriber``, not ``CrossSubscriberManager``
    (P4a Batch E2c).
    """

    def test_get_debate_culture_hints_empty(self, fresh_manager):
        """Test getting hints for non-existent debate."""
        from aragora.knowledge.event_subscribers import get_knowledge_event_subscriber

        hints = get_knowledge_event_subscriber().get_debate_culture_hints("nonexistent_debate")
        assert hints == {}

    def test_store_and_retrieve_hints(self, fresh_manager):
        """Test storing and retrieving culture hints."""
        from aragora.knowledge.event_subscribers import get_knowledge_event_subscriber

        subscriber = get_knowledge_event_subscriber()
        # Manually store hints (simulating what _store_debate_culture does)
        subscriber._debate_cultures["debate_001"] = {
            "protocol_hints": {
                "recommended_consensus": "unanimous",
                "extra_critique_rounds": 1,
            },
            "domain": "legal",
        }

        hints = subscriber.get_debate_culture_hints("debate_001")
        assert hints["recommended_consensus"] == "unanimous"
        assert hints["extra_critique_rounds"] == 1


class TestCultureProfileProcessing:
    """Tests for processing CultureProfile into protocol hints.

    ``_store_debate_culture``/``get_debate_culture_hints`` live on
    ``KnowledgeEventSubscriber``, not ``CrossSubscriberManager`` (P4a Batch E2c).
    """

    def test_store_debate_culture_with_decision_style(self, fresh_manager):
        """Test extracting decision style from culture profile."""
        from aragora.knowledge.event_subscribers import get_knowledge_event_subscriber

        subscriber = get_knowledge_event_subscriber()

        # Create mock profile with dominant pattern
        mock_profile = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.pattern_type = "decision_style"
        mock_pattern.value = "unanimous"
        mock_profile.dominant_pattern = mock_pattern
        mock_profile.patterns = []

        subscriber._store_debate_culture("debate_003", mock_profile, "general")

        hints = subscriber.get_debate_culture_hints("debate_003")
        assert hints.get("recommended_consensus") == "unanimous"

    def test_store_debate_culture_with_risk_tolerance_conservative(self, fresh_manager):
        """Test extracting conservative risk tolerance."""
        from aragora.knowledge.event_subscribers import get_knowledge_event_subscriber

        subscriber = get_knowledge_event_subscriber()

        mock_profile = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.pattern_type = "risk_tolerance"
        mock_pattern.value = "conservative"
        mock_profile.dominant_pattern = mock_pattern
        mock_profile.patterns = []

        subscriber._store_debate_culture("debate_004", mock_profile, "finance")

        hints = subscriber.get_debate_culture_hints("debate_004")
        assert hints.get("extra_critique_rounds") == 1

    def test_store_debate_culture_with_risk_tolerance_aggressive(self, fresh_manager):
        """Test extracting aggressive risk tolerance."""
        from aragora.knowledge.event_subscribers import get_knowledge_event_subscriber

        subscriber = get_knowledge_event_subscriber()

        mock_profile = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.pattern_type = "risk_tolerance"
        mock_pattern.value = "aggressive"
        mock_profile.dominant_pattern = mock_pattern
        mock_profile.patterns = []

        subscriber._store_debate_culture("debate_005", mock_profile, "startup")

        hints = subscriber.get_debate_culture_hints("debate_005")
        assert hints.get("early_consensus_threshold") == 0.7

    def test_store_debate_culture_with_domain_patterns(self, fresh_manager):
        """Test extracting domain-specific patterns."""
        from aragora.knowledge.event_subscribers import get_knowledge_event_subscriber

        subscriber = get_knowledge_event_subscriber()

        mock_profile = MagicMock()
        mock_profile.dominant_pattern = None

        # Create domain-specific patterns
        mock_pattern1 = MagicMock()
        mock_pattern1.domain = "legal"
        mock_pattern1.pattern_type = "decision_style"
        mock_pattern1.value = "conservative"
        mock_pattern1.confidence = 0.9

        mock_pattern2 = MagicMock()
        mock_pattern2.domain = "software"
        mock_pattern2.pattern_type = "debate_dynamics"
        mock_pattern2.value = "fast"
        mock_pattern2.confidence = 0.8

        mock_profile.patterns = [mock_pattern1, mock_pattern2]

        subscriber._store_debate_culture("debate_006", mock_profile, "legal")

        hints = subscriber.get_debate_culture_hints("debate_006")
        domain_patterns = hints.get("domain_patterns", [])
        assert len(domain_patterns) == 1
        assert domain_patterns[0]["value"] == "conservative"


class TestOrchestratorCultureIntegration:
    """Tests for orchestrator culture hint application."""

    @patch("aragora.knowledge.event_subscribers.get_knowledge_event_subscriber")
    @patch("aragora.debate.event_subscribers.bootstrap_debate_event_subscribers")
    def test_orchestrator_gets_culture_hints(
        self, mock_bootstrap, mock_get_subscriber, monkeypatch
    ):
        """Test that orchestrator retrieves culture hints."""
        from aragora.debate.orchestrator import Arena
        from aragora.core_types import Environment

        monkeypatch.delenv("ARAGORA_OFFLINE", raising=False)

        mock_subscriber = MagicMock()
        mock_subscriber.get_debate_culture_hints.return_value = {
            "recommended_consensus": "majority"
        }
        mock_get_subscriber.return_value = mock_subscriber

        # Create minimal arena
        environment = Environment(task="Test question")
        agents = [MagicMock(), MagicMock()]
        arena = Arena(environment=environment, agents=agents)

        hints = arena._get_culture_hints("test_debate")
        assert hints == {"recommended_consensus": "majority"}

    def test_apply_culture_hints_empty(self):
        """Test applying empty hints does nothing."""
        from aragora.debate.orchestrator import Arena
        from aragora.core_types import Environment

        environment = Environment(task="Test question")
        agents = [MagicMock(), MagicMock()]
        arena = Arena(environment=environment, agents=agents)

        # Should not raise
        arena._apply_culture_hints({})
        arena._apply_culture_hints(None)

    def test_apply_culture_hints_with_consensus(self):
        """Test applying consensus hint."""
        from aragora.debate.orchestrator import Arena
        from aragora.core_types import Environment

        environment = Environment(task="Test question")
        agents = [MagicMock(), MagicMock()]
        arena = Arena(environment=environment, agents=agents)

        arena._apply_culture_hints({"recommended_consensus": "unanimous"})
        assert arena._culture_consensus_hint == "unanimous"

    def test_apply_culture_hints_with_extra_critiques(self):
        """Test applying extra critique rounds hint."""
        from aragora.debate.orchestrator import Arena
        from aragora.core_types import Environment

        environment = Environment(task="Test question")
        agents = [MagicMock(), MagicMock()]
        arena = Arena(environment=environment, agents=agents)

        arena._apply_culture_hints({"extra_critique_rounds": 2})
        assert arena._culture_extra_critiques == 2

    def test_apply_culture_hints_with_early_consensus(self):
        """Test applying early consensus threshold."""
        from aragora.debate.orchestrator import Arena
        from aragora.core_types import Environment

        environment = Environment(task="Test question")
        agents = [MagicMock(), MagicMock()]
        arena = Arena(environment=environment, agents=agents)

        arena._apply_culture_hints({"early_consensus_threshold": 0.7})
        assert arena._culture_early_consensus == 0.7
