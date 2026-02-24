"""
Tests for Oracle Voice TTS wiring.

Validates that the TTS event bridge:
- Subscribes to critique events when ``enable_oracle_voice`` is True
- Does NOT subscribe to critique events when the flag is False
- Synthesizes voice for critique events when the flag is on
- Remains silent for critiques when the flag is off
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.stream.tts_event_bridge import TTSEventBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeDebateEvent:
    """Minimal stand-in for DebateEvent."""

    event_type: str = "agent_message"
    debate_id: str = "debate-1"
    data: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    span_id: str | None = None


class FakeEventBus:
    """Tracks subscribe/unsubscribe calls."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Any]] = {}

    def subscribe(self, event_type: str, handler: Any) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Any) -> bool:
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False


def _make_bridge(
    *,
    voice_has_session: bool = True,
    synthesize_return: int = 1,
) -> tuple[TTSEventBridge, AsyncMock, MagicMock]:
    tts = AsyncMock()
    tts.synthesize_for_debate = AsyncMock(return_value=synthesize_return)

    voice_handler = MagicMock()
    voice_handler.has_voice_session.return_value = voice_has_session

    bridge = TTSEventBridge(tts=tts, voice_handler=voice_handler)
    return bridge, tts, voice_handler


# ---------------------------------------------------------------------------
# Feature flag gating
# ---------------------------------------------------------------------------


class TestOracleVoiceFeatureFlag:
    """Verify that critique subscription is gated by enable_oracle_voice."""

    def test_critique_subscribed_when_flag_enabled(self) -> None:
        """Bridge subscribes to 'critique' events when flag is on."""
        with patch(
            "aragora.server.stream.tts_event_bridge._is_oracle_voice_enabled",
            return_value=True,
        ):
            bridge, _, _ = _make_bridge()
            bus = FakeEventBus()
            bridge.connect(bus)

        assert "critique" in bus._handlers
        assert len(bus._handlers["critique"]) == 1
        assert "agent_message" in bus._handlers
        assert len(bus._handlers["agent_message"]) == 1

    def test_critique_not_subscribed_when_flag_disabled(self) -> None:
        """Bridge does NOT subscribe to 'critique' events when flag is off."""
        with patch(
            "aragora.server.stream.tts_event_bridge._is_oracle_voice_enabled",
            return_value=False,
        ):
            bridge, _, _ = _make_bridge()
            bus = FakeEventBus()
            bridge.connect(bus)

        assert "critique" not in bus._handlers or len(bus._handlers.get("critique", [])) == 0
        # agent_message is always subscribed
        assert len(bus._handlers["agent_message"]) == 1


# ---------------------------------------------------------------------------
# End-to-end voice synthesis
# ---------------------------------------------------------------------------


class TestOracleVoiceSynthesis:
    """Test that critique events trigger TTS when oracle voice is enabled."""

    @pytest.mark.asyncio
    async def test_critique_triggers_synthesis_when_enabled(self) -> None:
        with patch(
            "aragora.server.stream.tts_event_bridge._is_oracle_voice_enabled",
            return_value=True,
        ):
            bridge, tts, _ = _make_bridge()
            bus = FakeEventBus()
            bridge.connect(bus)

        # Simulate a critique event arriving via the event bus handler
        critique_event = FakeDebateEvent(
            event_type="critique",
            debate_id="d1",
            data={"content": "I disagree with this argument. ", "agent": "gpt-4"},
        )
        await bridge._on_agent_message(critique_event)
        await asyncio.sleep(0.05)

        tts.synthesize_for_debate.assert_awaited_once()
        call_kwargs = tts.synthesize_for_debate.call_args
        text = call_kwargs.kwargs.get("text", call_kwargs[1].get("text", ""))
        assert "disagree" in text

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_critique_silent_when_disabled(self) -> None:
        """When flag is off, critiques don't even reach the bridge handler."""
        with patch(
            "aragora.server.stream.tts_event_bridge._is_oracle_voice_enabled",
            return_value=False,
        ):
            bridge, tts, _ = _make_bridge()
            bus = FakeEventBus()
            bridge.connect(bus)

        # critique handler was never registered, so even if we manually
        # call the handler, the key behavior is that it's not subscribed
        assert "critique" not in bus._handlers or len(bus._handlers.get("critique", [])) == 0
        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_agent_message_still_works_with_flag_on(self) -> None:
        """Normal agent_message synthesis works regardless of flag state."""
        with patch(
            "aragora.server.stream.tts_event_bridge._is_oracle_voice_enabled",
            return_value=True,
        ):
            bridge, tts, _ = _make_bridge()
            bus = FakeEventBus()
            bridge.connect(bus)

        event = FakeDebateEvent(
            debate_id="d1",
            data={"content": "Regular agent message. ", "agent": "claude"},
        )
        await bridge._on_agent_message(event)
        await asyncio.sleep(0.05)

        tts.synthesize_for_debate.assert_awaited_once()
        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_agent_message_works_with_flag_off(self) -> None:
        """Normal agent_message synthesis works when oracle voice is off."""
        with patch(
            "aragora.server.stream.tts_event_bridge._is_oracle_voice_enabled",
            return_value=False,
        ):
            bridge, tts, _ = _make_bridge()
            bus = FakeEventBus()
            bridge.connect(bus)

        event = FakeDebateEvent(
            debate_id="d1",
            data={"content": "Regular agent message. ", "agent": "claude"},
        )
        await bridge._on_agent_message(event)
        await asyncio.sleep(0.05)

        tts.synthesize_for_debate.assert_awaited_once()
        await bridge.shutdown()


# ---------------------------------------------------------------------------
# Shutdown cleanup
# ---------------------------------------------------------------------------


class TestOracleVoiceShutdown:
    """Verify critique unsubscribe on shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_unsubscribes_critique(self) -> None:
        with patch(
            "aragora.server.stream.tts_event_bridge._is_oracle_voice_enabled",
            return_value=True,
        ):
            bridge, _, _ = _make_bridge()
            bus = FakeEventBus()
            bridge.connect(bus)

        assert len(bus._handlers.get("critique", [])) == 1

        await bridge.shutdown()

        # critique handler should be removed (unsubscribe is best-effort;
        # if it was never there it silently returns False)
        assert len(bus._handlers.get("critique", [])) == 0
        assert len(bus._handlers.get("agent_message", [])) == 0


# ---------------------------------------------------------------------------
# Feature flag integration
# ---------------------------------------------------------------------------


class TestFeatureFlagIntegration:
    """Test the feature flag registration itself."""

    def test_enable_oracle_voice_registered_in_registry(self) -> None:
        from aragora.config.feature_flags import get_flag_registry

        registry = get_flag_registry()
        flag = registry.get_definition("enable_oracle_voice")
        assert flag is not None
        assert flag.default is False
        assert flag.env_var == "ENABLE_ORACLE_VOICE"

    def test_enable_oracle_voice_disabled_by_default(self) -> None:
        from aragora.config.feature_flags import is_enabled

        # Default is False, so unless env var is set, this should be False
        assert is_enabled("enable_oracle_voice") is False

    def test_enable_oracle_voice_respects_env_var(self) -> None:
        import os

        from aragora.config.feature_flags import get_flag_registry, reset_flag_registry

        reset_flag_registry()  # Clear cached registry
        try:
            os.environ["ENABLE_ORACLE_VOICE"] = "true"
            registry = get_flag_registry()
            assert registry.is_enabled("enable_oracle_voice") is True
        finally:
            os.environ.pop("ENABLE_ORACLE_VOICE", None)
            reset_flag_registry()
