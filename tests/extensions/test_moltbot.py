"""
Tests for the Moltbot extension - Consumer/Device interface layer.

Tests inbox management, gateway orchestration, voice processing, and onboarding.
"""

import pytest
from pathlib import Path

from aragora.extensions.moltbot import (
    Channel,
    ChannelConfig,
    ChannelType,
    DeviceNode,
    DeviceNodeConfig,
    InboxMessage,
    InboxMessageStatus,
    OnboardingFlow,
    OnboardingStep,
    VoiceSession,
    VoiceSessionConfig,
    InboxManager,
    LocalGateway,
    VoiceProcessor,
    OnboardingOrchestrator,
)


# =============================================================================
# InboxManager Tests
# =============================================================================


class TestInboxManager:
    """Tests for InboxManager."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager with temp storage."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.mark.asyncio
    async def test_register_channel(self, inbox: InboxManager):
        """Test registering a communication channel."""
        config = ChannelConfig(
            type=ChannelType.SMS,
            name="SMS Channel",
            priority=10,
        )
        channel = await inbox.register_channel(config, user_id="user-1")

        assert channel.id
        assert channel.config.type == ChannelType.SMS
        assert channel.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_list_channels_with_filters(self, inbox: InboxManager):
        """Test listing channels with filters."""
        config1 = ChannelConfig(type=ChannelType.SMS, name="SMS")
        config2 = ChannelConfig(type=ChannelType.EMAIL, name="Email")

        await inbox.register_channel(config1, user_id="user-1")
        await inbox.register_channel(config2, user_id="user-2")

        # Filter by type
        sms_channels = await inbox.list_channels(channel_type=ChannelType.SMS)
        assert len(sms_channels) == 1

        # Filter by user
        user1_channels = await inbox.list_channels(user_id="user-1")
        assert len(user1_channels) == 1

    @pytest.mark.asyncio
    async def test_receive_message(self, inbox: InboxManager):
        """Test receiving an inbound message."""
        config = ChannelConfig(type=ChannelType.SMS, name="SMS")
        channel = await inbox.register_channel(config, user_id="user-1")

        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Hello, I need help!",
        )

        assert message.id
        assert message.direction == "inbound"
        assert message.content == "Hello, I need help!"
        assert message.intent == "help_request"

    @pytest.mark.asyncio
    async def test_send_message(self, inbox: InboxManager):
        """Test sending an outbound message."""
        config = ChannelConfig(type=ChannelType.SMS, name="SMS")
        channel = await inbox.register_channel(config, user_id="user-1")

        message = await inbox.send_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Thank you for contacting us.",
        )

        assert message.direction == "outbound"
        assert message.status == InboxMessageStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_message_threading(self, inbox: InboxManager):
        """Test message threading."""
        config = ChannelConfig(type=ChannelType.SMS, name="SMS")
        channel = await inbox.register_channel(config, user_id="user-1")

        # First message creates thread
        msg1 = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Question?",
        )

        # Reply uses same thread
        msg2 = await inbox.send_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Answer",
            reply_to=msg1.id,
        )

        assert msg2.thread_id == msg1.thread_id

        # Get thread
        thread = await inbox.get_thread(msg1.thread_id)
        assert len(thread) == 2

    @pytest.mark.asyncio
    async def test_mark_read(self, inbox: InboxManager):
        """Test marking a message as read."""
        config = ChannelConfig(type=ChannelType.SMS, name="SMS")
        channel = await inbox.register_channel(config, user_id="user-1")

        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Test",
        )

        updated = await inbox.mark_read(message.id)
        assert updated.status == InboxMessageStatus.READ
        assert updated.read_at is not None

    @pytest.mark.asyncio
    async def test_get_stats(self, inbox: InboxManager):
        """Test getting inbox statistics."""
        config = ChannelConfig(type=ChannelType.SMS, name="SMS")
        channel = await inbox.register_channel(config, user_id="user-1")

        await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Test",
        )

        stats = await inbox.get_stats()
        assert stats["channels_total"] == 1
        assert stats["messages_total"] == 1
        assert stats["messages_inbound"] == 1


# =============================================================================
# LocalGateway Tests
# =============================================================================


class TestLocalGateway:
    """Tests for LocalGateway."""

    @pytest.fixture
    def gateway(self, tmp_path: Path) -> LocalGateway:
        """Create a gateway with temp storage."""
        return LocalGateway(
            gateway_id="test-gateway",
            storage_path=tmp_path / "gateway",
        )

    @pytest.mark.asyncio
    async def test_register_device(self, gateway: LocalGateway):
        """Test registering a device."""
        config = DeviceNodeConfig(
            name="Sensor-1",
            device_type="iot",
            capabilities=["temperature", "humidity"],
        )
        device = await gateway.register_device(config, user_id="user-1")

        assert device.id
        assert device.config.name == "Sensor-1"
        assert device.gateway_id == "test-gateway"
        assert device.status == "offline"

    @pytest.mark.asyncio
    async def test_device_heartbeat(self, gateway: LocalGateway):
        """Test device heartbeat."""
        config = DeviceNodeConfig(name="Sensor", device_type="iot")
        device = await gateway.register_device(config, user_id="user-1")

        # Send heartbeat
        updated = await gateway.heartbeat(
            device_id=device.id,
            state={"temperature": 22.5},
            metrics={"battery_level": 85.0},
        )

        assert updated.status == "online"
        assert updated.state["temperature"] == 22.5
        assert updated.battery_level == 85.0
        assert updated.last_heartbeat is not None

    @pytest.mark.asyncio
    async def test_update_state(self, gateway: LocalGateway):
        """Test updating device state."""
        config = DeviceNodeConfig(name="Sensor", device_type="iot")
        device = await gateway.register_device(config, user_id="user-1")

        updated = await gateway.update_state(
            device_id=device.id,
            state={"mode": "active", "threshold": 30},
        )

        assert updated.state["mode"] == "active"
        assert updated.state["threshold"] == 30

    @pytest.mark.asyncio
    async def test_send_command(self, gateway: LocalGateway):
        """Test sending a command to a device."""
        config = DeviceNodeConfig(
            name="Sensor",
            device_type="iot",
            capabilities=["reboot"],
        )
        device = await gateway.register_device(config, user_id="user-1")

        # Device must be online
        await gateway.heartbeat(device.id)

        result = await gateway.send_command(
            device_id=device.id,
            command="reboot",
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_send_command_offline_device(self, gateway: LocalGateway):
        """Test that commands fail for offline devices."""
        config = DeviceNodeConfig(name="Sensor", device_type="iot", capabilities=["test"])
        device = await gateway.register_device(config, user_id="user-1")

        result = await gateway.send_command(device.id, "test")
        assert result["success"] is False
        assert "offline" in result["error"]

    @pytest.mark.asyncio
    async def test_list_devices_with_filters(self, gateway: LocalGateway):
        """Test listing devices with filters."""
        config1 = DeviceNodeConfig(name="Sensor-1", device_type="iot")
        config2 = DeviceNodeConfig(name="Phone", device_type="mobile")

        device1 = await gateway.register_device(config1, user_id="user-1")
        await gateway.register_device(config2, user_id="user-2")

        # Make device1 online
        await gateway.heartbeat(device1.id)

        # Filter by type
        iot_devices = await gateway.list_devices(device_type="iot")
        assert len(iot_devices) == 1

        # Filter by status
        online_devices = await gateway.list_devices(status="online")
        assert len(online_devices) == 1

    @pytest.mark.asyncio
    async def test_event_subscription(self, gateway: LocalGateway):
        """Test event subscription."""
        events = []

        def on_event(event):
            events.append(event)

        gateway.subscribe(on_event)

        config = DeviceNodeConfig(name="Sensor", device_type="iot")
        await gateway.register_device(config, user_id="user-1")

        assert len(events) == 1
        assert events[0]["type"] == "device_registered"

    @pytest.mark.asyncio
    async def test_get_stats(self, gateway: LocalGateway):
        """Test getting gateway statistics."""
        config = DeviceNodeConfig(name="Sensor", device_type="iot")
        device = await gateway.register_device(config, user_id="user-1")
        await gateway.heartbeat(device.id)

        stats = await gateway.get_stats()
        assert stats["gateway_id"] == "test-gateway"
        assert stats["devices_total"] == 1
        assert stats["devices_online"] == 1


# =============================================================================
# VoiceProcessor Tests
# =============================================================================


class TestVoiceProcessor:
    """Tests for VoiceProcessor."""

    @pytest.fixture
    def voice(self, tmp_path: Path) -> VoiceProcessor:
        """Create a voice processor with temp storage."""
        return VoiceProcessor(storage_path=tmp_path / "voice")

    @pytest.mark.asyncio
    async def test_create_session(self, voice: VoiceProcessor):
        """Test creating a voice session."""
        config = VoiceSessionConfig(language="en-US")
        session = await voice.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        assert session.id
        assert session.config.language == "en-US"
        assert session.status == "active"

    @pytest.mark.asyncio
    async def test_process_audio(self, voice: VoiceProcessor):
        """Test processing audio for transcription."""
        config = VoiceSessionConfig()
        session = await voice.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await voice.process_audio(session.id, b"mock_audio_data")

        assert result["success"] is True
        assert "transcript" in result

        # Check session was updated
        session = await voice.get_session(session.id)
        assert session.turns == 1

    @pytest.mark.asyncio
    async def test_synthesize_speech(self, voice: VoiceProcessor):
        """Test synthesizing speech from text."""
        config = VoiceSessionConfig()
        session = await voice.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await voice.synthesize_speech(session.id, "Hello, how can I help?")

        assert result["success"] is True
        assert "audio" in result

        # Check session was updated
        session = await voice.get_session(session.id)
        assert session.words_spoken > 0

    @pytest.mark.asyncio
    async def test_end_session(self, voice: VoiceProcessor):
        """Test ending a voice session."""
        config = VoiceSessionConfig()
        session = await voice.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        ended = await voice.end_session(session.id, reason="user_hangup")

        assert ended.status == "ended"
        assert ended.ended_at is not None
        assert ended.metadata["end_reason"] == "user_hangup"

    @pytest.mark.asyncio
    async def test_pause_resume_session(self, voice: VoiceProcessor):
        """Test pausing and resuming a session."""
        config = VoiceSessionConfig()
        session = await voice.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        # Pause
        paused = await voice.pause_session(session.id)
        assert paused.status == "paused"

        # Resume
        resumed = await voice.resume_session(session.id)
        assert resumed.status == "active"

    @pytest.mark.asyncio
    async def test_get_transcript(self, voice: VoiceProcessor):
        """Test getting session transcript."""
        config = VoiceSessionConfig()
        session = await voice.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        await voice.process_audio(session.id, b"audio")
        await voice.synthesize_speech(session.id, "Response")

        transcript = await voice.get_transcript(session.id)
        assert len(transcript) == 2
        assert transcript[0]["type"] == "user"
        assert transcript[1]["type"] == "system"

    @pytest.mark.asyncio
    async def test_get_stats(self, voice: VoiceProcessor):
        """Test getting voice processor statistics."""
        config = VoiceSessionConfig()
        await voice.create_session(
            config=config,
            user_id="user-1",
            channel_id="channel-1",
        )

        stats = await voice.get_stats()
        assert stats["sessions_total"] == 1
        assert stats["sessions_active"] == 1
        assert "mock" in stats["stt_providers_available"]


# =============================================================================
# OnboardingOrchestrator Tests
# =============================================================================


class TestOnboardingOrchestrator:
    """Tests for OnboardingOrchestrator."""

    @pytest.fixture
    def onboarding(self, tmp_path: Path) -> OnboardingOrchestrator:
        """Create an onboarding orchestrator with temp storage."""
        return OnboardingOrchestrator(storage_path=tmp_path / "onboarding")

    @pytest.mark.asyncio
    async def test_create_flow(self, onboarding: OnboardingOrchestrator):
        """Test creating an onboarding flow."""
        flow = await onboarding.create_flow(
            name="New User Signup",
            description="Welcome new users",
        )

        assert flow.id
        assert flow.name == "New User Signup"
        assert flow.status == "draft"

    @pytest.mark.asyncio
    async def test_add_steps(self, onboarding: OnboardingOrchestrator):
        """Test adding steps to a flow."""
        flow = await onboarding.create_flow(name="Signup")

        step1 = await onboarding.add_step(
            flow_id=flow.id,
            name="Welcome",
            step_type="info",
            content={"message": "Welcome to the platform!"},
        )

        step2 = await onboarding.add_step(
            flow_id=flow.id,
            name="Email",
            step_type="input",
            validation={"email": ["required", "email"]},
        )

        assert step1.order == 0
        assert step2.order == 1

        flow = await onboarding.get_flow(flow.id)
        assert len(flow.steps) == 2

    @pytest.mark.asyncio
    async def test_activate_flow(self, onboarding: OnboardingOrchestrator):
        """Test activating a flow."""
        flow = await onboarding.create_flow(name="Signup")
        await onboarding.add_step(flow.id, "Step 1", "info")

        activated = await onboarding.activate_flow(flow.id)
        assert activated.status == "active"

    @pytest.mark.asyncio
    async def test_activate_flow_requires_steps(self, onboarding: OnboardingOrchestrator):
        """Test that activation requires steps."""
        flow = await onboarding.create_flow(name="Empty Flow")

        with pytest.raises(ValueError, match="no steps"):
            await onboarding.activate_flow(flow.id)

    @pytest.mark.asyncio
    async def test_start_session(self, onboarding: OnboardingOrchestrator):
        """Test starting an onboarding session."""
        flow = await onboarding.create_flow(name="Signup")
        await onboarding.add_step(flow.id, "Welcome", "info")
        await onboarding.activate_flow(flow.id)

        session = await onboarding.start_session(
            flow_id=flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        assert session.id
        assert session.status == "in_progress"
        assert session.current_step is not None

    @pytest.mark.asyncio
    async def test_submit_step(self, onboarding: OnboardingOrchestrator):
        """Test submitting step data."""
        flow = await onboarding.create_flow(name="Signup")
        await onboarding.add_step(flow.id, "Name", "input")
        await onboarding.add_step(flow.id, "Confirm", "info")
        await onboarding.activate_flow(flow.id)

        session = await onboarding.start_session(
            flow_id=flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await onboarding.submit_step(
            session_id=session.id,
            data={"name": "John Doe"},
        )

        assert result["success"] is True
        assert "next_step" in result
        assert result["next_step"]["name"] == "Confirm"

    @pytest.mark.asyncio
    async def test_validation(self, onboarding: OnboardingOrchestrator):
        """Test step validation."""
        flow = await onboarding.create_flow(name="Signup")
        await onboarding.add_step(
            flow.id,
            "Email",
            "input",
            validation={"email": {"required": True, "type": "email"}},
        )
        await onboarding.activate_flow(flow.id)

        session = await onboarding.start_session(
            flow_id=flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        # Invalid email
        result = await onboarding.submit_step(
            session_id=session.id,
            data={"email": "invalid"},
        )

        assert result["success"] is False
        assert "validation_errors" in result

    @pytest.mark.asyncio
    async def test_flow_completion(self, onboarding: OnboardingOrchestrator):
        """Test completing a flow."""
        flow = await onboarding.create_flow(name="Signup")
        await onboarding.add_step(flow.id, "Only Step", "info")
        await onboarding.activate_flow(flow.id)

        session = await onboarding.start_session(
            flow_id=flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await onboarding.submit_step(session.id, data={})

        assert result["success"] is True
        assert result["completed"] is True

        session = await onboarding.get_session(session.id)
        assert session.status == "completed"

    @pytest.mark.asyncio
    async def test_branch_conditions(self, onboarding: OnboardingOrchestrator):
        """Test conditional branching."""
        flow = await onboarding.create_flow(name="Survey")

        step1 = await onboarding.add_step(flow.id, "Type", "input")
        step_business = await onboarding.add_step(flow.id, "Business", "info")
        await onboarding.add_step(flow.id, "Personal", "info")

        # Update step1 with branch conditions
        flow = await onboarding.get_flow(flow.id)
        flow.steps[0].branch_conditions = {
            "type:business": step_business.id,
        }

        await onboarding.activate_flow(flow.id)

        session = await onboarding.start_session(
            flow_id=flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        result = await onboarding.submit_step(
            session.id,
            data={"type": "business"},
        )

        assert result["next_step"]["name"] == "Business"

    @pytest.mark.asyncio
    async def test_abandon_session(self, onboarding: OnboardingOrchestrator):
        """Test abandoning a session."""
        flow = await onboarding.create_flow(name="Signup")
        await onboarding.add_step(flow.id, "Step", "info")
        await onboarding.activate_flow(flow.id)

        session = await onboarding.start_session(
            flow_id=flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )

        abandoned = await onboarding.abandon_session(session.id, reason="user_left")

        assert abandoned.status == "abandoned"

    @pytest.mark.asyncio
    async def test_get_flow_stats(self, onboarding: OnboardingOrchestrator):
        """Test getting flow statistics."""
        flow = await onboarding.create_flow(name="Signup")
        await onboarding.add_step(flow.id, "Step", "info")
        await onboarding.activate_flow(flow.id)

        session = await onboarding.start_session(
            flow_id=flow.id,
            user_id="user-1",
            channel_id="channel-1",
        )
        await onboarding.submit_step(session.id, data={})

        stats = await onboarding.get_flow_stats(flow.id)

        assert stats["started"] == 1
        assert stats["completed"] == 1
        assert stats["completion_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_get_stats(self, onboarding: OnboardingOrchestrator):
        """Test getting overall statistics."""
        flow = await onboarding.create_flow(name="Signup")
        await onboarding.add_step(flow.id, "Step", "info")
        await onboarding.activate_flow(flow.id)

        stats = await onboarding.get_stats()

        assert stats["flows_total"] == 1
        assert stats["flows_active"] == 1
