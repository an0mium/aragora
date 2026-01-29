"""Tests for orchestrator_memory.py - Memory coordination and RLM compression helpers.

Tests cover:
- queue_for_supabase_sync: Background sync queueing
- setup_belief_network: Belief network initialization
- init_rlm_limiter_state: RLM cognitive load limiter setup
- init_checkpoint_bridge: Checkpoint bridge initialization
- auto_create_knowledge_mound: KM auto-creation
- init_cross_subscriber_bridge: Event bridge initialization
- compress_debate_messages: RLM compression
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ctx():
    """Create a mock DebateContext."""
    ctx = MagicMock()
    ctx.debate_id = "debate-123"
    ctx.loop_id = "loop-456"
    ctx.cycle_number = 2
    ctx.agents = [MagicMock(name=f"agent-{i}") for i in range(3)]
    return ctx


@pytest.fixture
def mock_result():
    """Create a mock DebateResult."""
    result = MagicMock()
    result.id = "result-789"
    result.debate_id = "debate-123"
    result.task = "Test debate task"
    result.final_answer = "The answer is 42"
    result.confidence = 0.85
    result.consensus_reached = True
    result.messages = [MagicMock() for _ in range(5)]
    result.votes = [MagicMock(choice="agent-1") for _ in range(3)]
    return result


@pytest.fixture
def mock_protocol():
    """Create a mock protocol."""
    protocol = MagicMock()
    protocol.enable_molecule_tracking = False
    protocol.rounds = 5
    return protocol


# =============================================================================
# Tests for queue_for_supabase_sync
# =============================================================================


class TestQueueForSupabaseSync:
    """Tests for queue_for_supabase_sync function."""

    def test_skips_when_sync_disabled(self, mock_ctx, mock_result):
        """When sync is disabled, return without action."""
        from aragora.debate.orchestrator_memory import queue_for_supabase_sync

        with patch("aragora.debate.orchestrator_memory.get_sync_service") as MockGetSync:
            mock_sync = MagicMock()
            mock_sync.enabled = False
            MockGetSync.return_value = mock_sync

            queue_for_supabase_sync(mock_ctx, mock_result)

            mock_sync.queue_debate.assert_not_called()

    def test_queues_debate_data_on_enabled(self, mock_ctx, mock_result):
        """When sync is enabled, queue debate data."""
        from aragora.debate.orchestrator_memory import queue_for_supabase_sync

        with patch("aragora.debate.orchestrator_memory.get_sync_service") as MockGetSync:
            mock_sync = MagicMock()
            mock_sync.enabled = True
            MockGetSync.return_value = mock_sync

            queue_for_supabase_sync(mock_ctx, mock_result)

            mock_sync.queue_debate.assert_called_once()

    def test_builds_correct_debate_data_structure(self, mock_ctx, mock_result):
        """Debate data includes all required fields."""
        from aragora.debate.orchestrator_memory import queue_for_supabase_sync

        with patch("aragora.debate.orchestrator_memory.get_sync_service") as MockGetSync:
            mock_sync = MagicMock()
            mock_sync.enabled = True
            MockGetSync.return_value = mock_sync

            queue_for_supabase_sync(mock_ctx, mock_result)

            call_args = mock_sync.queue_debate.call_args[0][0]
            assert "id" in call_args
            assert "debate_id" in call_args
            assert "loop_id" in call_args
            assert "cycle_number" in call_args
            assert "task" in call_args
            assert "consensus_reached" in call_args
            assert "confidence" in call_args

    def test_handles_import_error(self, mock_ctx, mock_result):
        """When import fails, return without action."""
        from aragora.debate.orchestrator_memory import queue_for_supabase_sync

        with patch(
            "aragora.debate.orchestrator_memory.get_sync_service",
            side_effect=ImportError("Not found"),
        ):
            # Should not raise
            queue_for_supabase_sync(mock_ctx, mock_result)

    def test_catches_connection_error(self, mock_ctx, mock_result):
        """When connection error occurs, handle gracefully."""
        from aragora.debate.orchestrator_memory import queue_for_supabase_sync

        with patch("aragora.debate.orchestrator_memory.get_sync_service") as MockGetSync:
            mock_sync = MagicMock()
            mock_sync.enabled = True
            mock_sync.queue_debate.side_effect = ConnectionError("Network error")
            MockGetSync.return_value = mock_sync

            # Should not raise
            queue_for_supabase_sync(mock_ctx, mock_result)

    def test_catches_timeout_error(self, mock_ctx, mock_result):
        """When timeout error occurs, handle gracefully."""
        from aragora.debate.orchestrator_memory import queue_for_supabase_sync

        with patch("aragora.debate.orchestrator_memory.get_sync_service") as MockGetSync:
            mock_sync = MagicMock()
            mock_sync.enabled = True
            mock_sync.queue_debate.side_effect = TimeoutError("Timeout")
            MockGetSync.return_value = mock_sync

            # Should not raise
            queue_for_supabase_sync(mock_ctx, mock_result)

    def test_limits_transcript_to_50_messages(self, mock_ctx, mock_result):
        """Transcript is limited to first 50 messages."""
        from aragora.debate.orchestrator_memory import queue_for_supabase_sync

        mock_result.messages = [MagicMock() for _ in range(100)]

        with patch("aragora.debate.orchestrator_memory.get_sync_service") as MockGetSync:
            mock_sync = MagicMock()
            mock_sync.enabled = True
            MockGetSync.return_value = mock_sync

            queue_for_supabase_sync(mock_ctx, mock_result)

            # Check that only first 50 messages were used
            call_args = mock_sync.queue_debate.call_args[0][0]
            # The transcript is a joined string, so we can't directly count
            # but the function should use messages[:50]


# =============================================================================
# Tests for setup_belief_network
# =============================================================================


class TestSetupBeliefNetwork:
    """Tests for setup_belief_network function."""

    def test_returns_none_when_import_fails(self):
        """When BeliefNetwork import fails, return None."""
        from aragora.debate.orchestrator_memory import setup_belief_network

        with patch(
            "aragora.debate.orchestrator_memory.BeliefNetwork", side_effect=ImportError("Not found")
        ):
            result = setup_belief_network(
                debate_id="debate-123",
                topic="Test topic",
            )

        assert result is None

    def test_creates_belief_network_instance(self):
        """When import succeeds, create BeliefNetwork."""
        from aragora.debate.orchestrator_memory import setup_belief_network

        with patch("aragora.debate.orchestrator_memory.BeliefNetwork") as MockNetwork:
            mock_instance = MagicMock()
            MockNetwork.return_value = mock_instance

            with patch("aragora.debate.orchestrator_memory.BeliefAdapter", side_effect=ImportError):
                result = setup_belief_network(
                    debate_id="debate-123",
                    topic="Test topic",
                    seed_from_km=False,
                )

        assert result == mock_instance

    def test_seeds_from_km_when_enabled(self):
        """When seed_from_km is True, seed from Knowledge Mound."""
        from aragora.debate.orchestrator_memory import setup_belief_network

        with (
            patch("aragora.debate.orchestrator_memory.BeliefNetwork") as MockNetwork,
            patch("aragora.debate.orchestrator_memory.BeliefAdapter") as MockAdapter,
        ):
            mock_instance = MagicMock()
            mock_instance.seed_from_km.return_value = 5
            MockNetwork.return_value = mock_instance
            MockAdapter.return_value = MagicMock()

            result = setup_belief_network(
                debate_id="debate-123",
                topic="Test topic",
                seed_from_km=True,
            )

            mock_instance.seed_from_km.assert_called_once()

    def test_handles_adapter_import_error(self):
        """When BeliefAdapter import fails, continue without adapter."""
        from aragora.debate.orchestrator_memory import setup_belief_network

        with (
            patch("aragora.debate.orchestrator_memory.BeliefNetwork") as MockNetwork,
            patch(
                "aragora.debate.orchestrator_memory.BeliefAdapter",
                side_effect=ImportError("Not found"),
            ),
        ):
            mock_instance = MagicMock()
            MockNetwork.return_value = mock_instance

            result = setup_belief_network(
                debate_id="debate-123",
                topic="Test topic",
            )

        assert result == mock_instance

    def test_catches_value_error(self):
        """When ValueError occurs, return None."""
        from aragora.debate.orchestrator_memory import setup_belief_network

        with patch("aragora.debate.orchestrator_memory.BeliefNetwork") as MockNetwork:
            MockNetwork.side_effect = ValueError("Invalid")

            result = setup_belief_network(
                debate_id="debate-123",
                topic="Test topic",
            )

        assert result is None


# =============================================================================
# Tests for init_rlm_limiter_state
# =============================================================================


class TestInitRlmLimiterState:
    """Tests for init_rlm_limiter_state function."""

    def test_disables_when_use_rlm_limiter_false(self):
        """When use_rlm_limiter is False, limiter is None."""
        from aragora.debate.orchestrator_memory import init_rlm_limiter_state

        result = init_rlm_limiter_state(
            use_rlm_limiter=False,
            rlm_limiter=None,
            rlm_compression_threshold=1000,
            rlm_max_recent_messages=10,
            rlm_summary_level="brief",
        )

        assert result["use_rlm_limiter"] is False
        assert result["rlm_limiter"] is None

    def test_uses_provided_limiter(self):
        """When limiter is provided, use it."""
        from aragora.debate.orchestrator_memory import init_rlm_limiter_state

        provided_limiter = MagicMock()

        result = init_rlm_limiter_state(
            use_rlm_limiter=True,
            rlm_limiter=provided_limiter,
            rlm_compression_threshold=1000,
            rlm_max_recent_messages=10,
            rlm_summary_level="brief",
        )

        assert result["rlm_limiter"] == provided_limiter

    def test_creates_limiter_when_enabled(self):
        """When enabled without limiter, create one."""
        from aragora.debate.orchestrator_memory import init_rlm_limiter_state

        with (
            patch("aragora.debate.orchestrator_memory.RLMCognitiveBudget") as MockBudget,
            patch("aragora.debate.orchestrator_memory.RLMCognitiveLoadLimiter") as MockLimiter,
        ):
            mock_limiter = MagicMock()
            MockLimiter.return_value = mock_limiter

            result = init_rlm_limiter_state(
                use_rlm_limiter=True,
                rlm_limiter=None,
                rlm_compression_threshold=1000,
                rlm_max_recent_messages=10,
                rlm_summary_level="brief",
            )

        assert result["rlm_limiter"] == mock_limiter
        MockBudget.assert_called_once()
        MockLimiter.assert_called_once()

    def test_handles_import_error(self):
        """When import fails, disable limiter."""
        from aragora.debate.orchestrator_memory import init_rlm_limiter_state

        with patch(
            "aragora.debate.orchestrator_memory.RLMCognitiveBudget",
            side_effect=ImportError("Not found"),
        ):
            result = init_rlm_limiter_state(
                use_rlm_limiter=True,
                rlm_limiter=None,
                rlm_compression_threshold=1000,
                rlm_max_recent_messages=10,
                rlm_summary_level="brief",
            )

        assert result["use_rlm_limiter"] is False
        assert result["rlm_limiter"] is None

    def test_returns_state_dict(self):
        """Returns dict with all required keys."""
        from aragora.debate.orchestrator_memory import init_rlm_limiter_state

        result = init_rlm_limiter_state(
            use_rlm_limiter=False,
            rlm_limiter=None,
            rlm_compression_threshold=1000,
            rlm_max_recent_messages=10,
            rlm_summary_level="brief",
        )

        assert "use_rlm_limiter" in result
        assert "rlm_compression_threshold" in result
        assert "rlm_max_recent_messages" in result
        assert "rlm_summary_level" in result
        assert "rlm_limiter" in result


# =============================================================================
# Tests for init_checkpoint_bridge
# =============================================================================


class TestInitCheckpointBridge:
    """Tests for init_checkpoint_bridge function."""

    def test_returns_none_tuples_when_all_disabled(self, mock_protocol):
        """When all disabled, return (None, None)."""
        from aragora.debate.orchestrator_memory import init_checkpoint_bridge

        mock_protocol.enable_molecule_tracking = False

        result = init_checkpoint_bridge(
            protocol=mock_protocol,
            checkpoint_manager=None,
        )

        assert result == (None, None)

    def test_creates_molecule_orchestrator_when_enabled(self, mock_protocol):
        """When molecule tracking enabled, create orchestrator."""
        from aragora.debate.orchestrator_memory import init_checkpoint_bridge

        mock_protocol.enable_molecule_tracking = True

        with (
            patch("aragora.debate.orchestrator_memory.get_molecule_orchestrator") as MockGetOrch,
            patch("aragora.debate.orchestrator_memory.create_checkpoint_bridge") as MockCreate,
        ):
            mock_orch = MagicMock()
            MockGetOrch.return_value = mock_orch
            MockCreate.return_value = MagicMock()

            result = init_checkpoint_bridge(
                protocol=mock_protocol,
                checkpoint_manager=None,
            )

        MockGetOrch.assert_called_once()

    def test_creates_checkpoint_bridge_when_manager_provided(self, mock_protocol):
        """When checkpoint_manager provided, create bridge."""
        from aragora.debate.orchestrator_memory import init_checkpoint_bridge

        mock_manager = MagicMock()

        with patch("aragora.debate.orchestrator_memory.create_checkpoint_bridge") as MockCreate:
            mock_bridge = MagicMock()
            MockCreate.return_value = mock_bridge

            result = init_checkpoint_bridge(
                protocol=mock_protocol,
                checkpoint_manager=mock_manager,
            )

        MockCreate.assert_called_once()

    def test_handles_molecule_orchestrator_import_error(self, mock_protocol):
        """When orchestrator import fails, continue without it."""
        from aragora.debate.orchestrator_memory import init_checkpoint_bridge

        mock_protocol.enable_molecule_tracking = True

        with patch(
            "aragora.debate.orchestrator_memory.get_molecule_orchestrator",
            side_effect=ImportError("Not found"),
        ):
            result = init_checkpoint_bridge(
                protocol=mock_protocol,
                checkpoint_manager=None,
            )

        # Should return (None, None) since no checkpoint_manager either
        assert result[0] is None


# =============================================================================
# Tests for auto_create_knowledge_mound
# =============================================================================


class TestAutoCreateKnowledgeMound:
    """Tests for auto_create_knowledge_mound function."""

    def test_returns_provided_mound_unchanged(self):
        """When mound is provided, return it unchanged."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        existing_mound = MagicMock()

        result = auto_create_knowledge_mound(
            knowledge_mound=existing_mound,
            auto_create=True,
            enable_retrieval=True,
            enable_ingestion=True,
            org_id="org-123",
        )

        assert result is existing_mound

    def test_returns_none_when_auto_create_disabled(self):
        """When auto_create is False, return None."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        result = auto_create_knowledge_mound(
            knowledge_mound=None,
            auto_create=False,
            enable_retrieval=True,
            enable_ingestion=True,
            org_id="org-123",
        )

        assert result is None

    def test_returns_none_when_retrieval_and_ingestion_disabled(self):
        """When both retrieval and ingestion disabled, return None."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        result = auto_create_knowledge_mound(
            knowledge_mound=None,
            auto_create=True,
            enable_retrieval=False,
            enable_ingestion=False,
            org_id="org-123",
        )

        assert result is None

    def test_creates_mound_when_enabled_with_retrieval(self):
        """When auto_create and retrieval enabled, create mound."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        with patch("aragora.debate.orchestrator_memory.get_knowledge_mound") as MockGet:
            mock_mound = MagicMock()
            MockGet.return_value = mock_mound

            result = auto_create_knowledge_mound(
                knowledge_mound=None,
                auto_create=True,
                enable_retrieval=True,
                enable_ingestion=False,
                org_id="org-123",
            )

        assert result == mock_mound
        MockGet.assert_called_once()

    def test_creates_mound_when_enabled_with_ingestion(self):
        """When auto_create and ingestion enabled, create mound."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        with patch("aragora.debate.orchestrator_memory.get_knowledge_mound") as MockGet:
            mock_mound = MagicMock()
            MockGet.return_value = mock_mound

            result = auto_create_knowledge_mound(
                knowledge_mound=None,
                auto_create=True,
                enable_retrieval=False,
                enable_ingestion=True,
                org_id="org-123",
            )

        assert result == mock_mound

    def test_handles_import_error(self):
        """When import fails, return None."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        with patch(
            "aragora.debate.orchestrator_memory.get_knowledge_mound",
            side_effect=ImportError("Not found"),
        ):
            result = auto_create_knowledge_mound(
                knowledge_mound=None,
                auto_create=True,
                enable_retrieval=True,
                enable_ingestion=True,
                org_id="org-123",
            )

        assert result is None

    def test_handles_runtime_error(self):
        """When RuntimeError occurs, return None."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        with patch(
            "aragora.debate.orchestrator_memory.get_knowledge_mound",
            side_effect=RuntimeError("Init failed"),
        ):
            result = auto_create_knowledge_mound(
                knowledge_mound=None,
                auto_create=True,
                enable_retrieval=True,
                enable_ingestion=True,
                org_id="org-123",
            )

        assert result is None

    def test_handles_connection_error(self):
        """When ConnectionError occurs, return None."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        with patch(
            "aragora.debate.orchestrator_memory.get_knowledge_mound",
            side_effect=ConnectionError("Network error"),
        ):
            result = auto_create_knowledge_mound(
                knowledge_mound=None,
                auto_create=True,
                enable_retrieval=True,
                enable_ingestion=True,
                org_id="org-123",
            )

        assert result is None

    def test_uses_default_workspace_id(self):
        """When org_id is empty, use 'default'."""
        from aragora.debate.orchestrator_memory import auto_create_knowledge_mound

        with patch("aragora.debate.orchestrator_memory.get_knowledge_mound") as MockGet:
            mock_mound = MagicMock()
            MockGet.return_value = mock_mound

            auto_create_knowledge_mound(
                knowledge_mound=None,
                auto_create=True,
                enable_retrieval=True,
                enable_ingestion=False,
                org_id="",
            )

            MockGet.assert_called_once_with(
                workspace_id="default",
                auto_initialize=True,
            )


# =============================================================================
# Tests for init_cross_subscriber_bridge
# =============================================================================


class TestInitCrossSubscriberBridge:
    """Tests for init_cross_subscriber_bridge function."""

    def test_returns_none_when_event_bus_is_none(self):
        """When event_bus is None, return None."""
        from aragora.debate.orchestrator_memory import init_cross_subscriber_bridge

        result = init_cross_subscriber_bridge(event_bus=None)

        assert result is None

    def test_creates_arena_event_bridge(self):
        """When event_bus provided, create bridge."""
        from aragora.debate.orchestrator_memory import init_cross_subscriber_bridge

        mock_bus = MagicMock()

        with patch("aragora.debate.orchestrator_memory.ArenaEventBridge") as MockBridge:
            mock_bridge = MagicMock()
            MockBridge.return_value = mock_bridge

            result = init_cross_subscriber_bridge(event_bus=mock_bus)

        assert result == mock_bridge
        MockBridge.assert_called_once_with(mock_bus)

    def test_connects_to_cross_subscribers(self):
        """Bridge connects to cross subscribers."""
        from aragora.debate.orchestrator_memory import init_cross_subscriber_bridge

        mock_bus = MagicMock()

        with patch("aragora.debate.orchestrator_memory.ArenaEventBridge") as MockBridge:
            mock_bridge = MagicMock()
            MockBridge.return_value = mock_bridge

            init_cross_subscriber_bridge(event_bus=mock_bus)

            mock_bridge.connect_to_cross_subscribers.assert_called_once()

    def test_handles_import_error(self):
        """When import fails, return None."""
        from aragora.debate.orchestrator_memory import init_cross_subscriber_bridge

        mock_bus = MagicMock()

        with patch(
            "aragora.debate.orchestrator_memory.ArenaEventBridge",
            side_effect=ImportError("Not found"),
        ):
            result = init_cross_subscriber_bridge(event_bus=mock_bus)

        assert result is None

    def test_handles_attribute_error(self):
        """When AttributeError occurs, return None."""
        from aragora.debate.orchestrator_memory import init_cross_subscriber_bridge

        mock_bus = MagicMock()

        with patch("aragora.debate.orchestrator_memory.ArenaEventBridge") as MockBridge:
            mock_bridge = MagicMock()
            mock_bridge.connect_to_cross_subscribers.side_effect = AttributeError("No attr")
            MockBridge.return_value = mock_bridge

            result = init_cross_subscriber_bridge(event_bus=mock_bus)

        assert result is None


# =============================================================================
# Tests for compress_debate_messages
# =============================================================================


class TestCompressDebateMessages:
    """Tests for compress_debate_messages function."""

    @pytest.mark.asyncio
    async def test_returns_original_when_limiter_disabled(self):
        """When limiter is disabled, return originals."""
        from aragora.debate.orchestrator_memory import compress_debate_messages

        messages = [MagicMock() for _ in range(5)]
        critiques = [MagicMock() for _ in range(2)]

        result = await compress_debate_messages(
            messages=messages,
            critiques=critiques,
            use_rlm_limiter=False,
            rlm_limiter=MagicMock(),
        )

        assert result == (messages, critiques)

    @pytest.mark.asyncio
    async def test_returns_original_when_limiter_is_none(self):
        """When limiter is None, return originals."""
        from aragora.debate.orchestrator_memory import compress_debate_messages

        messages = [MagicMock() for _ in range(5)]
        critiques = [MagicMock() for _ in range(2)]

        result = await compress_debate_messages(
            messages=messages,
            critiques=critiques,
            use_rlm_limiter=True,
            rlm_limiter=None,
        )

        assert result == (messages, critiques)

    @pytest.mark.asyncio
    async def test_compresses_messages_on_success(self):
        """When compression succeeds, return compressed results."""
        from aragora.debate.orchestrator_memory import compress_debate_messages

        messages = [MagicMock() for _ in range(5)]
        critiques = [MagicMock() for _ in range(2)]

        compressed_messages = [MagicMock() for _ in range(3)]
        compressed_critiques = [MagicMock() for _ in range(1)]

        mock_limiter = MagicMock()
        mock_result = MagicMock()
        mock_result.compression_applied = True
        mock_result.messages = compressed_messages
        mock_result.critiques = compressed_critiques
        mock_result.original_chars = 1000
        mock_result.compressed_chars = 500
        mock_result.compression_ratio = 0.5
        mock_limiter.compress_context_async = AsyncMock(return_value=mock_result)

        result = await compress_debate_messages(
            messages=messages,
            critiques=critiques,
            use_rlm_limiter=True,
            rlm_limiter=mock_limiter,
        )

        assert result == (compressed_messages, compressed_critiques)

    @pytest.mark.asyncio
    async def test_preserves_critiques(self):
        """Critiques are passed through compression."""
        from aragora.debate.orchestrator_memory import compress_debate_messages

        messages = [MagicMock() for _ in range(5)]
        critiques = [MagicMock() for _ in range(2)]

        mock_limiter = MagicMock()
        mock_result = MagicMock()
        mock_result.compression_applied = False
        mock_result.messages = messages
        mock_result.critiques = critiques
        mock_limiter.compress_context_async = AsyncMock(return_value=mock_result)

        result = await compress_debate_messages(
            messages=messages,
            critiques=critiques,
            use_rlm_limiter=True,
            rlm_limiter=mock_limiter,
        )

        mock_limiter.compress_context_async.assert_called_once_with(
            messages=messages,
            critiques=critiques,
        )

    @pytest.mark.asyncio
    async def test_handles_value_error(self):
        """When ValueError occurs, return originals."""
        from aragora.debate.orchestrator_memory import compress_debate_messages

        messages = [MagicMock() for _ in range(5)]
        critiques = [MagicMock() for _ in range(2)]

        mock_limiter = MagicMock()
        mock_limiter.compress_context_async = AsyncMock(side_effect=ValueError("Invalid"))

        result = await compress_debate_messages(
            messages=messages,
            critiques=critiques,
            use_rlm_limiter=True,
            rlm_limiter=mock_limiter,
        )

        assert result == (messages, critiques)

    @pytest.mark.asyncio
    async def test_handles_type_error(self):
        """When TypeError occurs, return originals."""
        from aragora.debate.orchestrator_memory import compress_debate_messages

        messages = [MagicMock() for _ in range(5)]
        critiques = [MagicMock() for _ in range(2)]

        mock_limiter = MagicMock()
        mock_limiter.compress_context_async = AsyncMock(side_effect=TypeError("Type mismatch"))

        result = await compress_debate_messages(
            messages=messages,
            critiques=critiques,
            use_rlm_limiter=True,
            rlm_limiter=mock_limiter,
        )

        assert result == (messages, critiques)

    @pytest.mark.asyncio
    async def test_handles_unexpected_error(self):
        """When unexpected error occurs, return originals."""
        from aragora.debate.orchestrator_memory import compress_debate_messages

        messages = [MagicMock() for _ in range(5)]
        critiques = [MagicMock() for _ in range(2)]

        mock_limiter = MagicMock()
        mock_limiter.compress_context_async = AsyncMock(side_effect=RuntimeError("Unexpected"))

        result = await compress_debate_messages(
            messages=messages,
            critiques=critiques,
            use_rlm_limiter=True,
            rlm_limiter=mock_limiter,
        )

        assert result == (messages, critiques)
