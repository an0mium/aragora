"""Tests for RecoveryNarrator integration with checkpoint system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.recovery_narrator import (
    RecoveryNarrative,
    RecoveryNarrator,
    get_narrator,
    reset_narrator,
    setup_narrator_with_checkpoint_manager,
    integrate_narrator_with_checkpoint_webhook,
)


class TestRecoveryNarratorCheckpointTemplates:
    """Tests for checkpoint-related narrator templates."""

    def setup_method(self):
        """Reset narrator before each test."""
        reset_narrator()

    def test_checkpoint_created_templates_exist(self):
        """Test that checkpoint_created templates are defined."""
        narrator = RecoveryNarrator()
        assert "checkpoint_created" in narrator.TEMPLATES
        templates = narrator.TEMPLATES["checkpoint_created"]
        assert "headlines" in templates
        assert "narratives" in templates
        assert templates["mood"] == "neutral"

    def test_debate_resumed_templates_exist(self):
        """Test that debate_resumed templates are defined."""
        narrator = RecoveryNarrator()
        assert "debate_resumed" in narrator.TEMPLATES
        templates = narrator.TEMPLATES["debate_resumed"]
        assert "headlines" in templates
        assert "narratives" in templates
        assert templates["mood"] == "triumphant"

    def test_debate_paused_templates_exist(self):
        """Test that debate_paused templates are defined."""
        narrator = RecoveryNarrator()
        assert "debate_paused" in narrator.TEMPLATES
        templates = narrator.TEMPLATES["debate_paused"]
        assert "headlines" in templates
        assert "narratives" in templates
        assert templates["mood"] == "neutral"

    def test_checkpoint_restored_templates_exist(self):
        """Test that checkpoint_restored templates are defined."""
        narrator = RecoveryNarrator()
        assert "checkpoint_restored" in narrator.TEMPLATES
        templates = narrator.TEMPLATES["checkpoint_restored"]
        assert "headlines" in templates
        assert "narratives" in templates
        assert templates["mood"] == "triumphant"


class TestRecoveryNarratorCheckpointNarration:
    """Tests for generating checkpoint narratives."""

    def setup_method(self):
        """Reset narrator before each test."""
        reset_narrator()

    def test_narrate_checkpoint_created(self):
        """Test generating checkpoint_created narrative."""
        narrator = RecoveryNarrator()
        narrative = narrator.narrate(
            "checkpoint_created",
            "System",
            {"round": 5, "debate_id": "test-123"},
        )

        assert isinstance(narrative, RecoveryNarrative)
        assert narrative.event_type == "checkpoint_created"
        assert narrative.agent == "System"
        assert narrative.mood == "neutral"
        assert "5" in narrative.narrative or "round" in narrative.narrative.lower()

    def test_narrate_debate_resumed(self):
        """Test generating debate_resumed narrative."""
        narrator = RecoveryNarrator()
        narrative = narrator.narrate(
            "debate_resumed",
            "System",
            {"round": 3, "debate_id": "test-456"},
        )

        assert isinstance(narrative, RecoveryNarrative)
        assert narrative.event_type == "debate_resumed"
        assert narrative.mood == "triumphant"
        # Narrative should indicate resumption (various templates may be used)
        assert len(narrative.narrative) > 0

    def test_narrate_checkpoint_restored(self):
        """Test generating checkpoint_restored narrative."""
        narrator = RecoveryNarrator()
        narrative = narrator.narrate(
            "checkpoint_restored",
            "System",
            {"round": 7, "agent_count": 4},
        )

        assert isinstance(narrative, RecoveryNarrative)
        assert narrative.event_type == "checkpoint_restored"
        assert narrative.mood == "triumphant"

    def test_narrate_debate_paused(self):
        """Test generating debate_paused narrative."""
        narrator = RecoveryNarrator()
        narrative = narrator.narrate(
            "debate_paused",
            "System",
            {"round": 2},
        )

        assert isinstance(narrative, RecoveryNarrative)
        assert narrative.event_type == "debate_paused"
        assert narrative.mood == "neutral"


class TestSetupNarratorWithCheckpointManager:
    """Tests for checkpoint manager integration setup."""

    def setup_method(self):
        """Reset narrator before each test."""
        reset_narrator()

    def test_setup_creates_checkpoint_handlers(self):
        """Test that setup creates checkpoint handlers."""
        narrator = setup_narrator_with_checkpoint_manager()

        assert hasattr(narrator, "_checkpoint_handlers")
        assert "on_checkpoint" in narrator._checkpoint_handlers
        assert "on_resume" in narrator._checkpoint_handlers

    def test_setup_uses_provided_narrator(self):
        """Test that setup uses provided narrator instance."""
        custom_narrator = RecoveryNarrator()
        result = setup_narrator_with_checkpoint_manager(custom_narrator)

        assert result is custom_narrator
        assert hasattr(result, "_checkpoint_handlers")

    def test_on_checkpoint_handler_generates_narrative(self):
        """Test that on_checkpoint handler generates narrative."""
        narrator = setup_narrator_with_checkpoint_manager()
        handler = narrator._checkpoint_handlers["on_checkpoint"]

        # Call handler with checkpoint event
        handler(
            {
                "checkpoint": {
                    "current_round": 5,
                    "debate_id": "test-debate",
                }
            }
        )

        # Check narrative was generated
        recent = narrator.get_recent_narratives(limit=1)
        assert len(recent) == 1
        assert recent[0]["event_type"] == "checkpoint_created"

    def test_on_resume_handler_generates_narrative(self):
        """Test that on_resume handler generates narrative."""
        narrator = setup_narrator_with_checkpoint_manager()
        handler = narrator._checkpoint_handlers["on_resume"]

        # Call handler with resume event
        handler(
            {
                "checkpoint": {
                    "current_round": 3,
                    "debate_id": "test-debate",
                    "agent_states": [{"agent_name": "claude"}, {"agent_name": "gpt-4"}],
                }
            }
        )

        # Check narrative was generated
        recent = narrator.get_recent_narratives(limit=1)
        assert len(recent) == 1
        assert recent[0]["event_type"] == "debate_resumed"

    def test_handler_broadcasts_when_callback_set(self):
        """Test that handlers broadcast when callback is configured."""
        broadcast_calls = []

        def capture_broadcast(event):
            broadcast_calls.append(event)

        narrator = RecoveryNarrator(broadcast_callback=capture_broadcast)
        setup_narrator_with_checkpoint_manager(narrator)
        handler = narrator._checkpoint_handlers["on_checkpoint"]

        handler(
            {
                "checkpoint": {
                    "current_round": 1,
                    "debate_id": "test",
                }
            }
        )

        assert len(broadcast_calls) == 1
        assert broadcast_calls[0]["type"] == "recovery_narrative"
        assert broadcast_calls[0]["data"]["event_type"] == "checkpoint_created"


class TestIntegrateNarratorWithCheckpointWebhook:
    """Tests for webhook integration."""

    def setup_method(self):
        """Reset narrator before each test."""
        reset_narrator()

    def test_registers_handlers_with_webhook(self):
        """Test that handlers are registered with webhook."""
        # Create mock webhook
        webhook = MagicMock()
        webhook.on_checkpoint = MagicMock(return_value=lambda x: x)
        webhook.on_resume = MagicMock(return_value=lambda x: x)

        integrate_narrator_with_checkpoint_webhook(webhook)

        # Verify registration was called
        assert webhook.on_checkpoint.called
        assert webhook.on_resume.called

    def test_uses_provided_narrator(self):
        """Test that integration uses provided narrator."""
        webhook = MagicMock()
        webhook.on_checkpoint = MagicMock(return_value=lambda x: x)
        webhook.on_resume = MagicMock(return_value=lambda x: x)

        custom_narrator = RecoveryNarrator()
        integrate_narrator_with_checkpoint_webhook(webhook, custom_narrator)

        assert hasattr(custom_narrator, "_checkpoint_handlers")


class TestCheckpointManagerWithNarrator:
    """Tests for CheckpointManager narrator integration."""

    @pytest.fixture
    def mock_webhook(self):
        """Create mock webhook for testing."""
        webhook = MagicMock()
        webhook.emit = AsyncMock()
        return webhook

    @pytest.fixture
    def sample_checkpoint_data(self):
        """Sample checkpoint creation data."""
        return {
            "debate_id": "test-debate-123",
            "task": "Test task for debate",
            "current_round": 3,
            "total_rounds": 5,
            "phase": "critique",
            "messages": [],
            "critiques": [],
            "votes": [],
            "agents": [],
        }

    @pytest.mark.asyncio
    async def test_checkpoint_manager_emits_on_checkpoint(
        self, mock_webhook, sample_checkpoint_data
    ):
        """Test that CheckpointManager emits checkpoint event."""
        from aragora.debate.checkpoint import (
            CheckpointManager,
            FileCheckpointStore,
            CheckpointConfig,
        )
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            config = CheckpointConfig()
            manager = CheckpointManager(store=store, config=config, webhook=mock_webhook)

            # Create checkpoint
            checkpoint = await manager.create_checkpoint(
                debate_id=sample_checkpoint_data["debate_id"],
                task=sample_checkpoint_data["task"],
                current_round=sample_checkpoint_data["current_round"],
                total_rounds=sample_checkpoint_data["total_rounds"],
                phase=sample_checkpoint_data["phase"],
                messages=sample_checkpoint_data["messages"],
                critiques=sample_checkpoint_data["critiques"],
                votes=sample_checkpoint_data["votes"],
                agents=sample_checkpoint_data["agents"],
            )

            # Verify emit was called
            mock_webhook.emit.assert_called_once()
            call_args = mock_webhook.emit.call_args
            assert call_args[0][0] == "on_checkpoint"
            assert call_args[0][1]["debate_id"] == "test-debate-123"
            assert call_args[0][1]["round"] == 3

    @pytest.mark.asyncio
    async def test_checkpoint_manager_emits_on_resume(self, mock_webhook):
        """Test that CheckpointManager emits resume event."""
        from aragora.debate.checkpoint import (
            CheckpointManager,
            FileCheckpointStore,
            CheckpointConfig,
            DebateCheckpoint,
            CheckpointStatus,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            config = CheckpointConfig()
            manager = CheckpointManager(store=store, config=config, webhook=mock_webhook)

            # Create and save a checkpoint first
            checkpoint = DebateCheckpoint(
                checkpoint_id="cp-test-001",
                debate_id="test-debate-456",
                task="Resume test task",
                current_round=2,
                total_rounds=5,
                phase="propose",
                messages=[],
                critiques=[],
                votes=[],
                agent_states=[],
                status=CheckpointStatus.COMPLETE,
            )
            await store.save(checkpoint)

            # Resume from checkpoint
            resumed = await manager.resume_from_checkpoint("cp-test-001", "test-user")

            # Verify emit was called for resume
            assert mock_webhook.emit.call_count >= 1
            # Find the on_resume call
            resume_calls = [
                call for call in mock_webhook.emit.call_args_list if call[0][0] == "on_resume"
            ]
            assert len(resume_calls) == 1
            assert resume_calls[0][0][1]["debate_id"] == "test-debate-456"
            assert resume_calls[0][0][1]["resumed_by"] == "test-user"


class TestEndToEndNarratorCheckpointIntegration:
    """End-to-end tests for narrator-checkpoint integration."""

    def setup_method(self):
        """Reset narrator before each test."""
        reset_narrator()

    @pytest.mark.asyncio
    async def test_full_checkpoint_flow_generates_narratives(self):
        """Test that full checkpoint flow generates expected narratives."""
        from aragora.debate.checkpoint import (
            CheckpointManager,
            CheckpointWebhook,
            FileCheckpointStore,
            CheckpointConfig,
        )
        import tempfile

        broadcast_calls = []

        def capture_broadcast(event):
            broadcast_calls.append(event)

        narrator = RecoveryNarrator(broadcast_callback=capture_broadcast)
        webhook = CheckpointWebhook()

        # Integrate narrator with webhook
        integrate_narrator_with_checkpoint_webhook(webhook, narrator)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileCheckpointStore(tmpdir)
            config = CheckpointConfig()
            manager = CheckpointManager(store=store, config=config, webhook=webhook)

            # Create checkpoint
            checkpoint = await manager.create_checkpoint(
                debate_id="e2e-test",
                task="End to end test",
                current_round=1,
                total_rounds=3,
                phase="propose",
                messages=[],
                critiques=[],
                votes=[],
                agents=[],
            )

            # Verify checkpoint narrative was generated
            assert len(broadcast_calls) == 1
            assert broadcast_calls[0]["type"] == "recovery_narrative"
            assert broadcast_calls[0]["data"]["event_type"] == "checkpoint_created"

            # Resume checkpoint
            broadcast_calls.clear()
            await manager.resume_from_checkpoint(checkpoint.checkpoint_id, "user")

            # Verify resume narrative was generated
            assert len(broadcast_calls) == 1
            assert broadcast_calls[0]["type"] == "recovery_narrative"
            assert broadcast_calls[0]["data"]["event_type"] == "debate_resumed"

    def test_narrator_tracks_mood_across_checkpoint_events(self):
        """Test that narrator tracks mood distribution."""
        narrator = RecoveryNarrator()

        # Generate various checkpoint events
        narrator.narrate("checkpoint_created", "System", {"round": 1})
        narrator.narrate("debate_paused", "System", {"round": 2})
        narrator.narrate("debate_resumed", "System", {"round": 2})
        narrator.narrate("checkpoint_restored", "System", {"round": 2, "agent_count": 4})

        mood_summary = narrator.get_mood_summary()

        assert "distribution" in mood_summary
        assert "neutral" in mood_summary["distribution"]
        assert "triumphant" in mood_summary["distribution"]
