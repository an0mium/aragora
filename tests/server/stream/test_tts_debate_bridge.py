"""
Tests for TTS-to-debate streaming bridge integration.

Covers all four integration fixes:
- Fix 1: Arena event_bus property accessor
- Fix 2: TTS bridge wiring during debate execution
- Fix 3: WebSocket connection registration in VoiceStreamHandler
- Fix 4: Bridge cleanup on debate end
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fix 1: Arena event_bus property
# ---------------------------------------------------------------------------


class TestArenaEventBusProperty:
    """Verify the Arena.event_bus property exposes the internal EventBus."""

    def _make_arena(self) -> Any:
        """Create a minimal Arena with mocked dependencies."""
        from aragora.core import Agent, Environment
        from aragora.debate.protocol import DebateProtocol

        env = Environment(task="Test task")
        agents = [MagicMock(spec=Agent, name=f"agent-{i}") for i in range(2)]
        for a in agents:
            a.name = f"agent-{agents.index(a)}"

        with patch("aragora.debate.orchestrator_init.run_init_subsystems"):
            arena = MagicMock()
            # Simulate the real attribute
            arena._event_bus = None
            return arena

    def test_event_bus_property_getter(self):
        """Arena.event_bus returns the stored _event_bus value."""
        from aragora.debate.orchestrator import Arena

        # Verify the property exists on the class
        assert hasattr(Arena, "event_bus")
        prop = getattr(Arena, "event_bus")
        assert isinstance(prop, property), "event_bus should be a property"

    def test_event_bus_property_setter(self):
        """Arena.event_bus can be set via the property setter."""
        from aragora.debate.orchestrator import Arena

        prop = getattr(Arena, "event_bus")
        assert prop.fset is not None, "event_bus property should have a setter"

    def test_event_bus_roundtrip(self):
        """Setting and getting event_bus through the property works correctly."""
        from aragora.debate.orchestrator import Arena

        # Create a mock arena instance that has the property
        mock_bus = MagicMock()

        # Use a real instance trick: create a minimal object with the property
        # We test the property descriptor directly
        class FakeArena:
            _event_bus = None

        # Apply the property from Arena
        FakeArena.event_bus = Arena.event_bus

        instance = FakeArena()
        assert instance.event_bus is None

        instance.event_bus = mock_bus
        assert instance.event_bus is mock_bus
        assert instance._event_bus is mock_bus


# ---------------------------------------------------------------------------
# Fix 2: TTS bridge wiring during debate execution
# ---------------------------------------------------------------------------


class TestTTSBridgeWiring:
    """Verify that on_arena_created/on_arena_finished callbacks are called."""

    def test_execute_debate_thread_calls_on_arena_created(self):
        """on_arena_created callback is invoked after Arena construction."""
        callback = MagicMock()
        created_arenas = []

        def capture_arena(arena):
            created_arenas.append(arena)
            callback(arena)

        with (
            patch(
                "aragora.server.stream.debate_executor.AgentSpec.coerce_list",
                return_value=[
                    MagicMock(provider="anthropic-api", name="a1", persona=None, model=None),
                    MagicMock(provider="openai-api", name="a2", persona=None, model=None),
                ],
            ),
            patch(
                "aragora.server.stream.debate_executor._filter_agent_specs_with_fallback",
                side_effect=lambda specs, emitter, did: (
                    specs,
                    [s.name for s in specs],
                    [],
                ),
            ),
            patch(
                "aragora.server.stream.debate_executor._create_debate_agents",
                return_value=[MagicMock(name="a1"), MagicMock(name="a2")],
            ),
            patch("aragora.server.stream.debate_executor.Arena") as mock_arena_cls,
            patch("aragora.server.stream.debate_executor.Environment"),
            patch("aragora.server.stream.debate_executor.DebateProtocol"),
            patch("aragora.server.stream.debate_executor.create_arena_hooks"),
        ):
            mock_arena = MagicMock()
            mock_arena.protocol.timeout_seconds = 0
            # Make arena.run() a coroutine
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            mock_result.final_answer = "Test answer"

            async def fake_run():
                return mock_result

            mock_arena.run = fake_run
            mock_arena.event_bus = MagicMock()
            mock_arena_cls.return_value = mock_arena

            emitter = MagicMock()
            emitter.set_loop_id = MagicMock()

            from aragora.server.stream.debate_executor import execute_debate_thread

            execute_debate_thread(
                debate_id="test-debate",
                question="What is the meaning of life?",
                agents_str="anthropic-api,openai-api",
                rounds=1,
                consensus="majority",
                trending_topic=None,
                emitter=emitter,
                on_arena_created=capture_arena,
            )

            callback.assert_called_once()
            assert len(created_arenas) == 1

    def test_execute_debate_thread_calls_on_arena_finished(self):
        """on_arena_finished callback is invoked after debate completes."""
        finished_callback = MagicMock()

        with (
            patch(
                "aragora.server.stream.debate_executor.AgentSpec.coerce_list",
                return_value=[
                    MagicMock(provider="anthropic-api", name="a1", persona=None, model=None),
                    MagicMock(provider="openai-api", name="a2", persona=None, model=None),
                ],
            ),
            patch(
                "aragora.server.stream.debate_executor._filter_agent_specs_with_fallback",
                side_effect=lambda specs, emitter, did: (
                    specs,
                    [s.name for s in specs],
                    [],
                ),
            ),
            patch(
                "aragora.server.stream.debate_executor._create_debate_agents",
                return_value=[MagicMock(name="a1"), MagicMock(name="a2")],
            ),
            patch("aragora.server.stream.debate_executor.Arena") as mock_arena_cls,
            patch("aragora.server.stream.debate_executor.Environment"),
            patch("aragora.server.stream.debate_executor.DebateProtocol"),
            patch("aragora.server.stream.debate_executor.create_arena_hooks"),
        ):
            mock_arena = MagicMock()
            mock_arena.protocol.timeout_seconds = 0
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            mock_result.final_answer = "Answer"

            async def fake_run():
                return mock_result

            mock_arena.run = fake_run
            mock_arena_cls.return_value = mock_arena

            emitter = MagicMock()
            emitter.set_loop_id = MagicMock()

            from aragora.server.stream.debate_executor import execute_debate_thread

            execute_debate_thread(
                debate_id="test-debate-2",
                question="Test?",
                agents_str="anthropic-api,openai-api",
                rounds=1,
                consensus="majority",
                trending_topic=None,
                emitter=emitter,
                on_arena_finished=finished_callback,
            )

            finished_callback.assert_called_once_with(mock_arena)

    def test_on_arena_finished_called_even_on_failure(self):
        """on_arena_finished is invoked even when arena.run() fails."""
        finished_callback = MagicMock()

        with (
            patch(
                "aragora.server.stream.debate_executor.AgentSpec.coerce_list",
                return_value=[
                    MagicMock(provider="anthropic-api", name="a1", persona=None, model=None),
                    MagicMock(provider="openai-api", name="a2", persona=None, model=None),
                ],
            ),
            patch(
                "aragora.server.stream.debate_executor._filter_agent_specs_with_fallback",
                side_effect=lambda specs, emitter, did: (
                    specs,
                    [s.name for s in specs],
                    [],
                ),
            ),
            patch(
                "aragora.server.stream.debate_executor._create_debate_agents",
                return_value=[MagicMock(name="a1"), MagicMock(name="a2")],
            ),
            patch("aragora.server.stream.debate_executor.Arena") as mock_arena_cls,
            patch("aragora.server.stream.debate_executor.Environment"),
            patch("aragora.server.stream.debate_executor.DebateProtocol"),
            patch("aragora.server.stream.debate_executor.create_arena_hooks"),
        ):
            mock_arena = MagicMock()
            mock_arena.protocol.timeout_seconds = 0

            async def failing_run():
                raise RuntimeError("Debate exploded")

            mock_arena.run = failing_run
            mock_arena_cls.return_value = mock_arena

            emitter = MagicMock()
            emitter.set_loop_id = MagicMock()

            from aragora.server.stream.debate_executor import execute_debate_thread

            execute_debate_thread(
                debate_id="test-debate-3",
                question="Fail?",
                agents_str="anthropic-api,openai-api",
                rounds=1,
                consensus="majority",
                trending_topic=None,
                emitter=emitter,
                on_arena_finished=finished_callback,
            )

            # Callback should still be called despite failure
            finished_callback.assert_called_once_with(mock_arena)

    def test_on_arena_created_callback_failure_does_not_block_debate(self):
        """A failing on_arena_created callback should not prevent the debate."""
        def bad_callback(arena):
            raise RuntimeError("Callback exploded")

        with (
            patch(
                "aragora.server.stream.debate_executor.AgentSpec.coerce_list",
                return_value=[
                    MagicMock(provider="anthropic-api", name="a1", persona=None, model=None),
                    MagicMock(provider="openai-api", name="a2", persona=None, model=None),
                ],
            ),
            patch(
                "aragora.server.stream.debate_executor._filter_agent_specs_with_fallback",
                side_effect=lambda specs, emitter, did: (
                    specs,
                    [s.name for s in specs],
                    [],
                ),
            ),
            patch(
                "aragora.server.stream.debate_executor._create_debate_agents",
                return_value=[MagicMock(name="a1"), MagicMock(name="a2")],
            ),
            patch("aragora.server.stream.debate_executor.Arena") as mock_arena_cls,
            patch("aragora.server.stream.debate_executor.Environment"),
            patch("aragora.server.stream.debate_executor.DebateProtocol"),
            patch("aragora.server.stream.debate_executor.create_arena_hooks"),
        ):
            mock_arena = MagicMock()
            mock_arena.protocol.timeout_seconds = 0
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            mock_result.final_answer = "Answer"

            async def fake_run():
                return mock_result

            mock_arena.run = fake_run
            mock_arena_cls.return_value = mock_arena

            emitter = MagicMock()
            emitter.set_loop_id = MagicMock()

            from aragora.server.stream.debate_executor import execute_debate_thread

            # Should not raise even though callback fails
            execute_debate_thread(
                debate_id="test-debate-4",
                question="Test?",
                agents_str="anthropic-api,openai-api",
                rounds=1,
                consensus="majority",
                trending_topic=None,
                emitter=emitter,
                on_arena_created=bad_callback,
            )


class TestServerTTSCallbacks:
    """Verify AiohttpUnifiedServer._on_arena_created/_on_arena_finished."""

    def test_on_arena_created_registers_tts(self):
        """_on_arena_created wires TTS integration to event_bus."""
        with (
            patch("aragora.server.stream.servers.VoiceStreamHandler"),
            patch("aragora.server.stream.tts_integration.TTSIntegration") as mock_tts_cls,
            patch(
                "aragora.server.stream.tts_integration.get_tts_integration"
            ) as mock_get_tts,
            patch("aragora.server.stream.tts_integration.set_tts_integration"),
        ):
            mock_tts = MagicMock()
            mock_get_tts.return_value = mock_tts
            mock_tts_cls.return_value = mock_tts

            from aragora.server.stream.servers import AiohttpUnifiedServer

            server = AiohttpUnifiedServer(port=8080)

            mock_arena = MagicMock()
            mock_event_bus = MagicMock()
            mock_arena.event_bus = mock_event_bus

            server._on_arena_created(mock_arena)

            # TTS integration should be registered on the event bus
            mock_tts.register.assert_called_once_with(mock_event_bus)

    def test_on_arena_created_skips_when_no_event_bus(self):
        """_on_arena_created does nothing when arena has no event_bus."""
        with (
            patch("aragora.server.stream.servers.VoiceStreamHandler"),
            patch("aragora.server.stream.tts_integration.TTSIntegration") as mock_tts_cls,
            patch(
                "aragora.server.stream.tts_integration.get_tts_integration"
            ) as mock_get_tts,
            patch("aragora.server.stream.tts_integration.set_tts_integration"),
        ):
            mock_tts = MagicMock()
            mock_get_tts.return_value = mock_tts
            mock_tts_cls.return_value = mock_tts

            from aragora.server.stream.servers import AiohttpUnifiedServer

            server = AiohttpUnifiedServer(port=8080)

            mock_arena = MagicMock()
            mock_arena.event_bus = None

            server._on_arena_created(mock_arena)

            # Should not try to register
            mock_tts.register.assert_not_called()

    def test_on_arena_created_skips_when_no_tts_integration(self):
        """_on_arena_created does nothing when TTS integration is not available."""
        with (
            patch("aragora.server.stream.servers.VoiceStreamHandler"),
            patch("aragora.server.stream.tts_integration.TTSIntegration") as mock_tts_cls,
            patch(
                "aragora.server.stream.tts_integration.get_tts_integration",
                return_value=None,
            ),
            patch("aragora.server.stream.tts_integration.set_tts_integration"),
        ):
            mock_tts_cls.return_value = MagicMock()

            from aragora.server.stream.servers import AiohttpUnifiedServer

            server = AiohttpUnifiedServer(port=8080)

            # Reset mock after constructor (which also calls get_tts_integration)
            mock_arena = MagicMock()
            mock_arena.event_bus = MagicMock()

            with patch(
                "aragora.server.stream.tts_integration.get_tts_integration",
                return_value=None,
            ):
                server._on_arena_created(mock_arena)

            # No crash, no registration attempt


# ---------------------------------------------------------------------------
# Fix 3: WebSocket connection registration in VoiceStreamHandler
# ---------------------------------------------------------------------------


class TestVoiceConnectionRegistration:
    """Verify WebSocket connections are registered/unregistered in VoiceStreamHandler."""

    def test_voice_connections_dict_exists(self):
        """VoiceStreamHandler has a _voice_connections dict."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        server = MagicMock()
        server.emitter = MagicMock()
        handler = VoiceStreamHandler(server)

        assert hasattr(handler, "_voice_connections")
        assert isinstance(handler._voice_connections, dict)
        assert len(handler._voice_connections) == 0

    def test_get_ws_for_session_returns_registered_ws(self):
        """_get_ws_for_session finds WebSockets registered in _voice_connections."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        server = MagicMock()
        server.emitter = MagicMock()
        # Ensure server doesn't have fallback registries
        del server.ws_connections
        del server.voice_connections

        handler = VoiceStreamHandler(server)

        mock_ws = MagicMock()
        handler._voice_connections["session-123"] = mock_ws

        result = handler._get_ws_for_session("session-123")
        assert result is mock_ws

    def test_get_ws_for_session_returns_none_for_unknown(self):
        """_get_ws_for_session returns None for unregistered sessions."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        server = MagicMock()
        server.emitter = MagicMock()
        del server.ws_connections
        del server.voice_connections

        handler = VoiceStreamHandler(server)

        result = handler._get_ws_for_session("nonexistent")
        assert result is None

    def test_get_ws_for_session_falls_back_to_server(self):
        """_get_ws_for_session falls back to server.ws_connections."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        mock_ws = MagicMock()
        server = MagicMock()
        server.emitter = MagicMock()
        server.ws_connections = {"session-456": mock_ws}
        server.voice_connections = {}

        handler = VoiceStreamHandler(server)

        result = handler._get_ws_for_session("session-456")
        assert result is mock_ws

    @pytest.mark.asyncio
    async def test_handle_websocket_registers_and_unregisters_ws(self):
        """handle_websocket registers the WS on connect and unregisters on disconnect."""
        from aragora.server.stream.voice_stream import VoiceStreamHandler

        server = MagicMock()
        server.emitter = MagicMock()
        server.emitter.emit = MagicMock()

        whisper = MagicMock()
        whisper.is_available = True

        handler = VoiceStreamHandler(server, whisper=whisper)

        # Create mock request and WebSocket
        mock_request = MagicMock()
        mock_request.headers = {}
        transport = MagicMock()
        transport.get_extra_info = MagicMock(return_value=("192.168.1.1", 5000))
        mock_request.transport = transport

        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.send_bytes = AsyncMock()
        mock_ws.closed = False

        # Make the WebSocket iteration end immediately (no messages)
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        debate_id = "debate-test"

        await handler.handle_websocket(mock_request, mock_ws, debate_id)

        # After handle_websocket returns (WS disconnected), the connection
        # should have been unregistered
        # The session_id is generated inside handle_websocket, so we check
        # that _voice_connections is empty after cleanup
        assert len(handler._voice_connections) == 0


# ---------------------------------------------------------------------------
# Fix 4: Bridge cleanup on debate end
# ---------------------------------------------------------------------------


class TestBridgeCleanup:
    """Verify TTS bridge is cleaned up when debates end."""

    def test_on_arena_finished_logs_completion(self):
        """_on_arena_finished completes without error."""
        with (
            patch("aragora.server.stream.servers.VoiceStreamHandler"),
            patch("aragora.server.stream.tts_integration.TTSIntegration") as mock_tts_cls,
            patch(
                "aragora.server.stream.tts_integration.get_tts_integration"
            ) as mock_get_tts,
            patch("aragora.server.stream.tts_integration.set_tts_integration"),
        ):
            mock_tts = MagicMock()
            mock_get_tts.return_value = mock_tts
            mock_tts_cls.return_value = mock_tts

            from aragora.server.stream.servers import AiohttpUnifiedServer

            server = AiohttpUnifiedServer(port=8080)

            mock_arena = MagicMock()
            # Should not raise
            server._on_arena_finished(mock_arena)

    def test_debate_stream_server_stop_tts_bridge(self):
        """DebateStreamServer.stop_tts_bridge clears the bridge reference."""
        from aragora.server.stream.debate_stream_server import DebateStreamServer

        server = DebateStreamServer(enable_tts=True)
        mock_bridge = AsyncMock()
        server._tts_bridge = mock_bridge

        # Run stop_tts_bridge
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server.stop_tts_bridge())
        finally:
            loop.close()

        mock_bridge.shutdown.assert_called_once()
        assert server._tts_bridge is None

    def test_debate_stream_server_stop_tts_bridge_noop_when_no_bridge(self):
        """stop_tts_bridge is safe to call when no bridge exists."""
        from aragora.server.stream.debate_stream_server import DebateStreamServer

        server = DebateStreamServer(enable_tts=True)
        assert server._tts_bridge is None

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server.stop_tts_bridge())
        finally:
            loop.close()

        # No crash
        assert server._tts_bridge is None

    def test_graceful_shutdown_stops_tts_bridge(self):
        """DebateStreamServer.graceful_shutdown stops the TTS bridge."""
        from aragora.server.stream.debate_stream_server import DebateStreamServer

        server = DebateStreamServer(enable_tts=True)
        mock_bridge = AsyncMock()
        server._tts_bridge = mock_bridge

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server.graceful_shutdown())
        finally:
            loop.close()

        mock_bridge.shutdown.assert_called_once()
        assert server._tts_bridge is None
