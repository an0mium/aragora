"""
Tests for Moltbot Extension Data Models.

Tests core data structures: Channel, InboxMessage, DeviceNode, VoiceSession, OnboardingFlow.
"""

import pytest
from datetime import datetime

from aragora.extensions.moltbot.models import (
    Channel,
    ChannelConfig,
    ChannelType,
    DeviceNode,
    DeviceNodeConfig,
    InboxMessage,
    InboxMessageStatus,
    OnboardingFlow,
    OnboardingSession,
    OnboardingStep,
    VoiceSession,
    VoiceSessionConfig,
)


class TestChannelType:
    """Tests for ChannelType enum."""

    def test_channel_types_exist(self):
        """Test that all expected channel types exist."""
        assert ChannelType.SMS.value == "sms"
        assert ChannelType.EMAIL.value == "email"
        assert ChannelType.WHATSAPP.value == "whatsapp"
        assert ChannelType.TELEGRAM.value == "telegram"
        assert ChannelType.SLACK.value == "slack"
        assert ChannelType.DISCORD.value == "discord"
        assert ChannelType.WEB.value == "web"
        assert ChannelType.VOICE.value == "voice"
        assert ChannelType.PUSH.value == "push"

    def test_channel_type_from_value(self):
        """Test creating ChannelType from string value."""
        assert ChannelType("sms") == ChannelType.SMS
        assert ChannelType("email") == ChannelType.EMAIL
        assert ChannelType("whatsapp") == ChannelType.WHATSAPP


class TestInboxMessageStatus:
    """Tests for InboxMessageStatus enum."""

    def test_message_statuses_exist(self):
        """Test that all expected message statuses exist."""
        assert InboxMessageStatus.PENDING.value == "pending"
        assert InboxMessageStatus.PROCESSING.value == "processing"
        assert InboxMessageStatus.DELIVERED.value == "delivered"
        assert InboxMessageStatus.READ.value == "read"
        assert InboxMessageStatus.RESPONDED.value == "responded"
        assert InboxMessageStatus.FAILED.value == "failed"
        assert InboxMessageStatus.ARCHIVED.value == "archived"


class TestChannelConfig:
    """Tests for ChannelConfig dataclass."""

    def test_create_channel_config(self):
        """Test creating a channel config."""
        config = ChannelConfig(
            type=ChannelType.SMS,
            name="My SMS Channel",
        )

        assert config.type == ChannelType.SMS
        assert config.name == "My SMS Channel"
        assert config.credentials == {}
        assert config.enabled is True
        assert config.priority == 0
        assert config.rate_limit == 100

    def test_channel_config_with_credentials(self):
        """Test channel config with credentials."""
        config = ChannelConfig(
            type=ChannelType.WHATSAPP,
            name="WhatsApp Business",
            credentials={"api_key": "secret", "phone_id": "12345"},
            priority=5,
            rate_limit=50,
        )

        assert config.credentials["api_key"] == "secret"
        assert config.priority == 5
        assert config.rate_limit == 50

    def test_channel_config_disabled(self):
        """Test disabled channel config."""
        config = ChannelConfig(
            type=ChannelType.EMAIL,
            name="Disabled Email",
            enabled=False,
        )

        assert config.enabled is False


class TestChannel:
    """Tests for Channel dataclass."""

    def test_create_channel(self):
        """Test creating a channel."""
        config = ChannelConfig(type=ChannelType.TELEGRAM, name="Telegram")
        channel = Channel(
            id="chan-1",
            config=config,
            user_id="user-1",
        )

        assert channel.id == "chan-1"
        assert channel.config.type == ChannelType.TELEGRAM
        assert channel.user_id == "user-1"
        assert channel.tenant_id is None
        assert channel.status == "active"
        assert channel.message_count == 0

    def test_channel_with_tenant(self):
        """Test channel with tenant ID."""
        config = ChannelConfig(type=ChannelType.SLACK, name="Slack")
        channel = Channel(
            id="chan-2",
            config=config,
            user_id="user-1",
            tenant_id="tenant-1",
        )

        assert channel.tenant_id == "tenant-1"

    def test_channel_metadata(self):
        """Test channel metadata."""
        config = ChannelConfig(type=ChannelType.WEB, name="Web Chat")
        channel = Channel(
            id="chan-3",
            config=config,
            user_id="user-1",
            metadata={"theme": "dark", "language": "en"},
        )

        assert channel.metadata["theme"] == "dark"
        assert channel.metadata["language"] == "en"


class TestInboxMessage:
    """Tests for InboxMessage dataclass."""

    def test_create_inbound_message(self):
        """Test creating an inbound message."""
        message = InboxMessage(
            id="msg-1",
            channel_id="chan-1",
            user_id="user-1",
            direction="inbound",
            content="Hello, world!",
        )

        assert message.id == "msg-1"
        assert message.channel_id == "chan-1"
        assert message.direction == "inbound"
        assert message.content == "Hello, world!"
        assert message.content_type == "text"
        assert message.status == InboxMessageStatus.PENDING

    def test_create_outbound_message(self):
        """Test creating an outbound message."""
        message = InboxMessage(
            id="msg-2",
            channel_id="chan-1",
            user_id="user-1",
            direction="outbound",
            content="Response message",
            status=InboxMessageStatus.DELIVERED,
        )

        assert message.direction == "outbound"
        assert message.status == InboxMessageStatus.DELIVERED

    def test_message_with_thread(self):
        """Test message with threading."""
        message = InboxMessage(
            id="msg-3",
            channel_id="chan-1",
            user_id="user-1",
            direction="inbound",
            content="Reply",
            thread_id="thread-1",
            reply_to="msg-1",
        )

        assert message.thread_id == "thread-1"
        assert message.reply_to == "msg-1"

    def test_message_with_intent(self):
        """Test message with intent detection."""
        message = InboxMessage(
            id="msg-4",
            channel_id="chan-1",
            user_id="user-1",
            direction="inbound",
            content="Help me with billing",
            intent="help_request",
            entities={"topic": "billing"},
            sentiment=0.3,
        )

        assert message.intent == "help_request"
        assert message.entities["topic"] == "billing"
        assert message.sentiment == 0.3

    def test_message_content_types(self):
        """Test different message content types."""
        text_msg = InboxMessage(
            id="msg-5",
            channel_id="chan-1",
            user_id="user-1",
            direction="inbound",
            content="Text message",
            content_type="text",
        )

        image_msg = InboxMessage(
            id="msg-6",
            channel_id="chan-1",
            user_id="user-1",
            direction="inbound",
            content="https://example.com/image.png",
            content_type="image",
        )

        assert text_msg.content_type == "text"
        assert image_msg.content_type == "image"


class TestDeviceNodeConfig:
    """Tests for DeviceNodeConfig dataclass."""

    def test_create_device_config(self):
        """Test creating device node config."""
        config = DeviceNodeConfig(
            name="Living Room Sensor",
            device_type="iot",
        )

        assert config.name == "Living Room Sensor"
        assert config.device_type == "iot"
        assert config.capabilities == []
        assert config.connection_type == "mqtt"
        assert config.heartbeat_interval == 60

    def test_device_config_with_capabilities(self):
        """Test device config with capabilities."""
        config = DeviceNodeConfig(
            name="Smart Hub",
            device_type="embedded",
            capabilities=["temperature", "humidity", "motion"],
            connection_type="websocket",
            heartbeat_interval=30,
        )

        assert "temperature" in config.capabilities
        assert "motion" in config.capabilities
        assert config.connection_type == "websocket"
        assert config.heartbeat_interval == 30


class TestDeviceNode:
    """Tests for DeviceNode dataclass."""

    def test_create_device_node(self):
        """Test creating device node."""
        config = DeviceNodeConfig(name="Test Device", device_type="iot")
        device = DeviceNode(
            id="device-1",
            config=config,
            user_id="user-1",
            gateway_id="gateway-1",
        )

        assert device.id == "device-1"
        assert device.config.name == "Test Device"
        assert device.user_id == "user-1"
        assert device.gateway_id == "gateway-1"
        assert device.status == "offline"
        assert device.messages_sent == 0
        assert device.messages_received == 0
        assert device.errors == 0

    def test_device_node_with_metrics(self):
        """Test device node with metrics."""
        config = DeviceNodeConfig(name="Battery Device", device_type="mobile")
        device = DeviceNode(
            id="device-2",
            config=config,
            user_id="user-1",
            gateway_id="gateway-1",
            status="online",
            battery_level=0.75,
            signal_strength=0.9,
            firmware_version="1.2.3",
        )

        assert device.status == "online"
        assert device.battery_level == 0.75
        assert device.signal_strength == 0.9
        assert device.firmware_version == "1.2.3"

    def test_device_node_state(self):
        """Test device node state tracking."""
        config = DeviceNodeConfig(name="State Device", device_type="iot")
        device = DeviceNode(
            id="device-3",
            config=config,
            user_id="user-1",
            gateway_id="gateway-1",
            state={"temperature": 22.5, "humidity": 45},
        )

        assert device.state["temperature"] == 22.5
        assert device.state["humidity"] == 45


class TestVoiceSessionConfig:
    """Tests for VoiceSessionConfig dataclass."""

    def test_create_voice_config(self):
        """Test creating voice session config."""
        config = VoiceSessionConfig()

        assert config.language == "en-US"
        assert config.voice_id == "default"
        assert config.sample_rate == 16000
        assert config.encoding == "pcm"
        assert config.enable_stt is True
        assert config.enable_tts is True
        assert config.enable_vad is True
        assert config.silence_timeout == 2.0
        assert config.max_duration == 300.0

    def test_custom_voice_config(self):
        """Test custom voice session config."""
        config = VoiceSessionConfig(
            language="es-ES",
            voice_id="spanish-female",
            sample_rate=48000,
            encoding="opus",
            enable_vad=False,
            silence_timeout=3.0,
        )

        assert config.language == "es-ES"
        assert config.voice_id == "spanish-female"
        assert config.sample_rate == 48000
        assert config.enable_vad is False


class TestVoiceSession:
    """Tests for VoiceSession dataclass."""

    def test_create_voice_session(self):
        """Test creating voice session."""
        config = VoiceSessionConfig()
        session = VoiceSession(
            id="voice-1",
            config=config,
            user_id="user-1",
            channel_id="chan-1",
        )

        assert session.id == "voice-1"
        assert session.user_id == "user-1"
        assert session.channel_id == "chan-1"
        assert session.status == "active"
        assert session.transcripts == []
        assert session.turns == 0
        assert session.words_spoken == 0
        assert session.words_heard == 0

    def test_voice_session_with_transcripts(self):
        """Test voice session with transcripts."""
        config = VoiceSessionConfig()
        session = VoiceSession(
            id="voice-2",
            config=config,
            user_id="user-1",
            channel_id="chan-1",
            transcripts=[
                {"type": "user", "text": "Hello"},
                {"type": "system", "text": "Hi there!"},
            ],
            turns=2,
            words_spoken=2,
            words_heard=1,
        )

        assert len(session.transcripts) == 2
        assert session.turns == 2


class TestOnboardingStep:
    """Tests for OnboardingStep dataclass."""

    def test_create_info_step(self):
        """Test creating info step."""
        step = OnboardingStep(
            id="step-1",
            name="Welcome",
            type="info",
            content={"message": "Welcome to our service!"},
        )

        assert step.id == "step-1"
        assert step.name == "Welcome"
        assert step.type == "info"
        assert step.required is True
        assert step.order == 0
        assert step.retry_limit == 3

    def test_create_input_step(self):
        """Test creating input step."""
        step = OnboardingStep(
            id="step-2",
            name="Email",
            type="input",
            content={"field": "email", "label": "Enter your email"},
            validation={"email": ["required", "email"]},
        )

        assert step.type == "input"
        assert step.validation is not None
        assert "email" in step.validation

    def test_create_decision_step(self):
        """Test creating decision step with branching."""
        step = OnboardingStep(
            id="step-3",
            name="Plan Selection",
            type="decision",
            content={"options": ["basic", "premium"]},
            branch_conditions={
                "plan:basic": "basic-setup",
                "plan:premium": "premium-setup",
            },
        )

        assert step.type == "decision"
        assert len(step.branch_conditions) == 2


class TestOnboardingFlow:
    """Tests for OnboardingFlow dataclass."""

    def test_create_flow(self):
        """Test creating onboarding flow."""
        flow = OnboardingFlow(
            id="flow-1",
            name="User Registration",
            description="New user registration flow",
        )

        assert flow.id == "flow-1"
        assert flow.name == "User Registration"
        assert flow.status == "draft"
        assert flow.steps == []
        assert flow.started_count == 0
        assert flow.completed_count == 0
        assert flow.abandoned_count == 0

    def test_flow_with_steps(self):
        """Test flow with steps."""
        steps = [
            OnboardingStep(id="s1", name="Welcome", type="info"),
            OnboardingStep(id="s2", name="Email", type="input"),
            OnboardingStep(id="s3", name="Verify", type="verification"),
        ]

        flow = OnboardingFlow(
            id="flow-2",
            name="Complete Flow",
            steps=steps,
            status="active",
        )

        assert len(flow.steps) == 3
        assert flow.status == "active"

    def test_flow_with_targeting(self):
        """Test flow with channel targeting."""
        flow = OnboardingFlow(
            id="flow-3",
            name="Mobile Onboarding",
            target_segment="mobile_users",
            channels=[ChannelType.PUSH, ChannelType.SMS],
        )

        assert flow.target_segment == "mobile_users"
        assert ChannelType.PUSH in flow.channels
        assert ChannelType.SMS in flow.channels


class TestOnboardingSession:
    """Tests for OnboardingSession dataclass."""

    def test_create_session(self):
        """Test creating onboarding session."""
        session = OnboardingSession(
            id="session-1",
            flow_id="flow-1",
            user_id="user-1",
            channel_id="chan-1",
        )

        assert session.id == "session-1"
        assert session.flow_id == "flow-1"
        assert session.user_id == "user-1"
        assert session.status == "in_progress"
        assert session.completed_steps == []
        assert session.collected_data == {}

    def test_session_with_progress(self):
        """Test session with progress."""
        session = OnboardingSession(
            id="session-2",
            flow_id="flow-1",
            user_id="user-1",
            channel_id="chan-1",
            current_step="step-3",
            completed_steps=["step-1", "step-2"],
            collected_data={"email": "user@example.com", "name": "John"},
            verification_status={"email": True},
        )

        assert session.current_step == "step-3"
        assert len(session.completed_steps) == 2
        assert session.collected_data["email"] == "user@example.com"
        assert session.verification_status["email"] is True

    def test_session_statuses(self):
        """Test different session statuses."""
        completed = OnboardingSession(
            id="s1",
            flow_id="f1",
            user_id="u1",
            channel_id="c1",
            status="completed",
        )

        abandoned = OnboardingSession(
            id="s2",
            flow_id="f1",
            user_id="u2",
            channel_id="c1",
            status="abandoned",
        )

        paused = OnboardingSession(
            id="s3",
            flow_id="f1",
            user_id="u3",
            channel_id="c1",
            status="paused",
        )

        assert completed.status == "completed"
        assert abandoned.status == "abandoned"
        assert paused.status == "paused"
