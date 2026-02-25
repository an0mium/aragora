"""Tests for debate result routing to chat origins.

Verifies that Arena.run() routes debate results back to the originating
chat channel (Telegram, Slack, Discord, Teams, WhatsApp, email, webhook)
when enable_result_routing is set on ArenaConfig.

Tests cover:
- Routing is called when enabled and origin metadata exists
- Routing is skipped when disabled (default)
- Routing errors don't propagate to the caller
- Different origin types are dispatched correctly
- Inline debate_origin metadata on Environment triggers routing
- Origin registration from environment metadata
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.orchestrator_runner import cleanup_debate_resources

# Patch targets: lazy imports inside cleanup_debate_resources resolve from
# the source module, so we patch the source.
_ROUTE_RESULT = "aragora.server.result_router.route_result"
_REGISTER_ORIGIN = "aragora.server.debate_origin.register_debate_origin"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arena(
    *,
    enable_result_routing: bool = False,
    env_metadata: dict | None = None,
) -> MagicMock:
    """Create a mock Arena with the minimal attributes needed by cleanup_debate_resources."""
    arena = MagicMock()
    arena.enable_result_routing = enable_result_routing
    arena.enable_auto_execution = False

    # Environment
    env = MagicMock()
    env.task = "Test debate task"
    if env_metadata is not None:
        env.metadata = env_metadata
    else:
        # Simulate Environment without metadata attr (getattr default kicks in)
        del env.metadata
    arena.env = env

    # Protocol
    arena.protocol = MagicMock()
    arena.protocol.checkpoint_cleanup_on_success = False
    arena.protocol.enable_translation = False

    # Internal methods used by cleanup_debate_resources
    arena.autonomic = MagicMock()
    arena.autonomic.set_debate_cost_tracker = MagicMock()
    arena._cleanup_convergence_cache = MagicMock()
    arena._teardown_agent_channels = AsyncMock()
    arena.cleanup_checkpoints = AsyncMock(return_value=0)
    arena._translate_conclusions = AsyncMock()

    return arena


def _make_result(debate_id: str = "test-debate-123") -> MagicMock:
    """Create a mock DebateResult."""
    result = MagicMock()
    result.debate_id = debate_id
    result.consensus_reached = True
    result.final_answer = "The answer is 42"
    result.confidence = 0.9
    result.winner = "agent-1"
    result.to_dict.return_value = {
        "debate_id": debate_id,
        "consensus_reached": True,
        "final_answer": "The answer is 42",
        "confidence": 0.9,
    }
    return result


def _make_state(
    debate_id: str = "test-debate-123",
    result: MagicMock | None = None,
) -> MagicMock:
    """Create a mock _DebateExecutionState."""
    if result is None:
        result = _make_result(debate_id)
    state = MagicMock()
    state.debate_id = debate_id
    state.debate_status = "completed"
    state.ctx.finalize_result.return_value = result
    state.ctx.result = result
    return state


# ---------------------------------------------------------------------------
# Tests: Routing enabled/disabled
# ---------------------------------------------------------------------------


class TestResultRoutingEnabled:
    """Verify routing is called when enable_result_routing=True."""

    @pytest.mark.asyncio
    async def test_route_result_called_when_enabled(self):
        """route_result is invoked when enable_result_routing=True."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            await cleanup_debate_resources(arena, state)
            mock_route.assert_awaited_once_with(
                "test-debate-123",
                state.ctx.finalize_result.return_value.to_dict.return_value,
            )

    @pytest.mark.asyncio
    async def test_route_result_not_called_when_disabled(self):
        """route_result is NOT invoked when enable_result_routing=False (default)."""
        arena = _make_arena(enable_result_routing=False)
        state = _make_state()

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
        ) as mock_route:
            await cleanup_debate_resources(arena, state)
            mock_route.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_result_skipped_when_result_is_none(self):
        """Routing is skipped when finalize_result returns None."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()
        state.ctx.finalize_result.return_value = None

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
        ) as mock_route:
            await cleanup_debate_resources(arena, state)
            mock_route.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Error resilience
# ---------------------------------------------------------------------------


class TestResultRoutingErrorResilience:
    """Routing errors must not propagate to the caller."""

    @pytest.mark.asyncio
    async def test_import_error_does_not_propagate(self):
        """ImportError from route_result import does not crash cleanup."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()

        # Simulate ImportError by making the import fail
        with patch.dict(
            "sys.modules",
            {"aragora.server.result_router": None},
        ):
            # Should not raise
            result = await cleanup_debate_resources(arena, state)
            assert result is not None

    @pytest.mark.asyncio
    async def test_runtime_error_does_not_propagate(self):
        """RuntimeError from route_result does not crash cleanup."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection refused"),
        ):
            result = await cleanup_debate_resources(arena, state)
            assert result is not None

    @pytest.mark.asyncio
    async def test_os_error_does_not_propagate(self):
        """OSError from route_result does not crash cleanup."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
            side_effect=OSError("network unreachable"),
        ):
            result = await cleanup_debate_resources(arena, state)
            assert result is not None

    @pytest.mark.asyncio
    async def test_type_error_does_not_propagate(self):
        """TypeError from route_result does not crash cleanup."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
            side_effect=TypeError("bad arg"),
        ):
            result = await cleanup_debate_resources(arena, state)
            assert result is not None

    @pytest.mark.asyncio
    async def test_value_error_does_not_propagate(self):
        """ValueError from route_result does not crash cleanup."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
            side_effect=ValueError("invalid debate_id"),
        ):
            result = await cleanup_debate_resources(arena, state)
            assert result is not None


# ---------------------------------------------------------------------------
# Tests: Inline debate_origin metadata on Environment
# ---------------------------------------------------------------------------


class TestInlineOriginMetadata:
    """When debate_origin metadata is set on Environment, routing activates."""

    @pytest.mark.asyncio
    async def test_env_metadata_triggers_routing_even_when_flag_disabled(self):
        """debate_origin in env.metadata enables routing even without the flag."""
        arena = _make_arena(
            enable_result_routing=False,
            env_metadata={
                "debate_origin": {
                    "platform": "telegram",
                    "channel_id": "12345678",
                    "user_id": "87654321",
                }
            },
        )
        state = _make_state()

        with (
            patch(
                _ROUTE_RESULT,
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_route,
            patch(
                _REGISTER_ORIGIN,
            ) as mock_register,
        ):
            await cleanup_debate_resources(arena, state)
            mock_route.assert_awaited_once()
            # Origin should be registered from env metadata
            mock_register.assert_called_once_with(
                debate_id="test-debate-123",
                platform="telegram",
                channel_id="12345678",
                user_id="87654321",
                metadata={},
            )

    @pytest.mark.asyncio
    async def test_env_metadata_without_platform_does_not_register(self):
        """debate_origin without platform key enables routing but skips registration."""
        arena = _make_arena(
            enable_result_routing=False,
            env_metadata={
                "debate_origin": {
                    "channel_id": "12345678",
                    # no "platform" key
                }
            },
        )
        state = _make_state()

        with (
            patch(
                _ROUTE_RESULT,
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_route,
            patch(
                _REGISTER_ORIGIN,
            ) as mock_register,
        ):
            await cleanup_debate_resources(arena, state)
            # Routing is attempted (debate_origin is truthy)
            mock_route.assert_awaited_once()
            # But origin was NOT registered (no platform)
            mock_register.assert_not_called()

    @pytest.mark.asyncio
    async def test_origin_registration_failure_does_not_block_routing(self):
        """If register_debate_origin fails, routing still proceeds."""
        arena = _make_arena(
            enable_result_routing=False,
            env_metadata={
                "debate_origin": {
                    "platform": "slack",
                    "channel_id": "C012345",
                    "user_id": "U012345",
                }
            },
        )
        state = _make_state()

        with (
            patch(
                _ROUTE_RESULT,
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_route,
            patch(
                _REGISTER_ORIGIN,
                side_effect=RuntimeError("store unavailable"),
            ),
        ):
            # Should not raise
            result = await cleanup_debate_resources(arena, state)
            assert result is not None
            mock_route.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: Different origin types dispatched correctly
# ---------------------------------------------------------------------------


class TestOriginTypeDispatch:
    """Verify that different origin platforms are dispatched through route_result."""

    @pytest.mark.parametrize(
        "platform",
        ["telegram", "slack", "discord", "teams", "whatsapp", "email", "webhook"],
    )
    @pytest.mark.asyncio
    async def test_platform_origin_dispatched(self, platform: str):
        """Each platform type is dispatched through route_result."""
        arena = _make_arena(
            enable_result_routing=False,
            env_metadata={
                "debate_origin": {
                    "platform": platform,
                    "channel_id": f"{platform}-channel-1",
                    "user_id": f"{platform}-user-1",
                    "metadata": {"webhook_url": "https://example.com/hook"},
                }
            },
        )
        state = _make_state()

        with (
            patch(
                _ROUTE_RESULT,
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_route,
            patch(
                _REGISTER_ORIGIN,
            ) as mock_register,
        ):
            await cleanup_debate_resources(arena, state)

            # route_result is called with the debate_id
            mock_route.assert_awaited_once()
            call_args = mock_route.call_args
            assert call_args[0][0] == "test-debate-123"

            # Origin was registered with the correct platform
            mock_register.assert_called_once()
            reg_kwargs = mock_register.call_args
            assert reg_kwargs.kwargs["platform"] == platform
            assert reg_kwargs.kwargs["channel_id"] == f"{platform}-channel-1"
            assert reg_kwargs.kwargs["user_id"] == f"{platform}-user-1"
            assert reg_kwargs.kwargs["metadata"] == {
                "webhook_url": "https://example.com/hook"
            }


# ---------------------------------------------------------------------------
# Tests: ArenaConfig flag
# ---------------------------------------------------------------------------


class TestArenaConfigFlag:
    """Test that ArenaConfig.enable_result_routing works correctly."""

    def test_arena_config_default_is_false(self):
        """enable_result_routing defaults to False (opt-in)."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.enable_result_routing is False

    def test_arena_config_can_be_enabled(self):
        """enable_result_routing can be set to True."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(enable_result_routing=True)
        assert config.enable_result_routing is True

    def test_arena_config_to_arena_kwargs_includes_flag(self):
        """to_arena_kwargs includes enable_result_routing."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(enable_result_routing=True)
        kwargs = config.to_arena_kwargs()
        assert kwargs["enable_result_routing"] is True

    def test_arena_config_to_arena_kwargs_default(self):
        """to_arena_kwargs includes enable_result_routing=False by default."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        kwargs = config.to_arena_kwargs()
        assert kwargs["enable_result_routing"] is False


# ---------------------------------------------------------------------------
# Tests: Result dict serialization
# ---------------------------------------------------------------------------


class TestResultSerialization:
    """Verify result is properly serialized before routing."""

    @pytest.mark.asyncio
    async def test_result_with_to_dict_uses_to_dict(self):
        """When result has to_dict(), that method is used for serialization."""
        arena = _make_arena(enable_result_routing=True)
        mock_result = _make_result()
        state = _make_state(result=mock_result)

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            await cleanup_debate_resources(arena, state)
            # Verify to_dict was called and its return value was passed
            mock_result.to_dict.assert_called()
            call_args = mock_route.call_args
            assert call_args[0][1] == mock_result.to_dict.return_value

    @pytest.mark.asyncio
    async def test_result_without_to_dict_uses_fallback(self):
        """When result lacks to_dict(), a fallback dict is constructed."""
        arena = _make_arena(enable_result_routing=True)
        mock_result = MagicMock()
        # Remove to_dict to force fallback path
        del mock_result.to_dict
        mock_result.winner = "agent-2"
        mock_result.consensus_reached = True
        mock_result.final_answer = "Fallback answer"
        mock_result.confidence = 0.75
        state = _make_state(result=mock_result)

        with patch(
            _ROUTE_RESULT,
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            await cleanup_debate_resources(arena, state)
            call_args = mock_route.call_args
            result_dict = call_args[0][1]
            assert result_dict["debate_id"] == "test-debate-123"
            assert result_dict["winner"] == "agent-2"
            assert result_dict["consensus_reached"] is True
            assert result_dict["final_answer"] == "Fallback answer"
            assert result_dict["confidence"] == 0.75


# ---------------------------------------------------------------------------
# Tests: route_result return value
# ---------------------------------------------------------------------------


class TestRouteResultReturnValue:
    """Verify behavior based on route_result return value."""

    @pytest.mark.asyncio
    async def test_successful_routing_logs_info(self):
        """When route_result returns True, info-level log is emitted."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()

        with (
            patch(
                _ROUTE_RESULT,
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "aragora.debate.orchestrator_runner.logger",
            ) as mock_logger,
        ):
            await cleanup_debate_resources(arena, state)
            # Check that info-level log was called for successful routing
            info_calls = [
                c for c in mock_logger.info.call_args_list
                if "result_routing" in str(c)
            ]
            assert len(info_calls) >= 1

    @pytest.mark.asyncio
    async def test_failed_routing_logs_debug(self):
        """When route_result returns False, debug-level log is emitted."""
        arena = _make_arena(enable_result_routing=True)
        state = _make_state()

        with (
            patch(
                _ROUTE_RESULT,
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "aragora.debate.orchestrator_runner.logger",
            ) as mock_logger,
        ):
            await cleanup_debate_resources(arena, state)
            debug_calls = [
                c for c in mock_logger.debug.call_args_list
                if "result_routing" in str(c)
            ]
            assert len(debug_calls) >= 1
