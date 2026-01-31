"""
Tests for ToolBridge - MCP-Chat Bridge for tool invocation from chat.

Tests cover:
- Tool invocation and execution
- Result formatting for different platforms
- Progress streaming
- Error handling (timeouts, connection errors, rate limits)
- Retry logic
- Platform-specific formatting
- Tool registration
- Invocation management
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.chat.tool_bridge import (
    ToolBridge,
    ToolStatus,
    Platform,
    ToolInvocation,
    ProgressUpdate,
    ResultFormatter,
    ErrorHandler,
    get_tool_bridge,
    reset_tool_bridges,
)


class TestToolBridgeInit:
    """Tests for ToolBridge initialization."""

    def test_init_with_default_platform(self):
        """Should default to GENERIC platform."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()
            assert bridge.platform == Platform.GENERIC

    def test_init_with_string_platform(self):
        """Should accept string platform names."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(platform="slack")
            assert bridge.platform == Platform.SLACK

    def test_init_with_platform_enum(self):
        """Should accept Platform enum values."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(platform=Platform.DISCORD)
            assert bridge.platform == Platform.DISCORD

    def test_init_with_invalid_platform_string(self):
        """Should fallback to GENERIC for unknown platforms."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(platform="unknown_platform")
            assert bridge.platform == Platform.GENERIC

    def test_init_with_custom_timeout(self):
        """Should accept custom timeout."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(timeout_seconds=600.0)
            assert bridge.timeout_seconds == 600.0

    def test_init_with_custom_max_retries(self):
        """Should accept custom max retries."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(max_retries=5)
            assert bridge.max_retries == 5

    def test_init_creates_formatter_and_error_handler(self):
        """Should create formatter and error handler."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(platform="slack")
            assert isinstance(bridge.formatter, ResultFormatter)
            assert isinstance(bridge.error_handler, ErrorHandler)
            assert bridge.formatter.platform == Platform.SLACK
            assert bridge.error_handler.platform == Platform.SLACK


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_slack_platform(self):
        """Should have slack platform."""
        assert Platform.SLACK.value == "slack"

    def test_discord_platform(self):
        """Should have discord platform."""
        assert Platform.DISCORD.value == "discord"

    def test_telegram_platform(self):
        """Should have telegram platform."""
        assert Platform.TELEGRAM.value == "telegram"

    def test_teams_platform(self):
        """Should have teams platform."""
        assert Platform.TEAMS.value == "teams"

    def test_whatsapp_platform(self):
        """Should have whatsapp platform."""
        assert Platform.WHATSAPP.value == "whatsapp"

    def test_generic_platform(self):
        """Should have generic platform."""
        assert Platform.GENERIC.value == "generic"


class TestToolStatusEnum:
    """Tests for ToolStatus enum."""

    def test_pending_status(self):
        """Should have pending status."""
        assert ToolStatus.PENDING.value == "pending"

    def test_running_status(self):
        """Should have running status."""
        assert ToolStatus.RUNNING.value == "running"

    def test_completed_status(self):
        """Should have completed status."""
        assert ToolStatus.COMPLETED.value == "completed"

    def test_failed_status(self):
        """Should have failed status."""
        assert ToolStatus.FAILED.value == "failed"

    def test_cancelled_status(self):
        """Should have cancelled status."""
        assert ToolStatus.CANCELLED.value == "cancelled"

    def test_timeout_status(self):
        """Should have timeout status."""
        assert ToolStatus.TIMEOUT.value == "timeout"


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_custom_tool(self):
        """Should register custom tool."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            async def custom_tool(**args):
                return {"result": "success"}

            bridge.register_tool("custom_tool", custom_tool)

            assert "custom_tool" in bridge._tools
            assert bridge._tools["custom_tool"] == custom_tool

    def test_get_available_tools(self):
        """Should return list of available tool names."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()
            bridge._tools = {
                "tool_a": lambda: None,
                "tool_b": lambda: None,
            }

            tools = bridge.get_available_tools()

            assert "tool_a" in tools
            assert "tool_b" in tools

    def test_get_tool_help_with_docstring(self):
        """Should return tool docstring as help."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            async def documented_tool():
                """This is the help text for the tool."""
                pass

            bridge._tools = {"documented_tool": documented_tool}

            help_text = bridge.get_tool_help("documented_tool")

            assert help_text == "This is the help text for the tool."

    def test_get_tool_help_no_docstring(self):
        """Should return None when tool has no docstring."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()
            bridge._tools = {"no_doc_tool": lambda: None}

            help_text = bridge.get_tool_help("no_doc_tool")

            assert help_text is None

    def test_get_tool_help_unknown_tool(self):
        """Should return None for unknown tool."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()
            bridge._tools = {}

            help_text = bridge.get_tool_help("unknown")

            assert help_text is None


class TestInvokeTool:
    """Tests for invoke_tool method."""

    @pytest.mark.asyncio
    async def test_invoke_tool_success(self):
        """Should invoke tool and return formatted result."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(platform="generic")

            async def test_tool(**args):
                return {"success": True, "value": args.get("input")}

            bridge._tools = {"test_tool": test_tool}

            formatted, raw = await bridge.invoke_tool(
                tool_name="test_tool",
                args={"input": "test_value"},
                channel_id="channel-123",
            )

            assert raw["success"] is True
            assert raw["value"] == "test_value"
            assert "Operation completed" in formatted

    @pytest.mark.asyncio
    async def test_invoke_tool_unknown_tool(self):
        """Should return error for unknown tool."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()
            bridge._tools = {}

            formatted, raw = await bridge.invoke_tool(
                tool_name="nonexistent_tool",
                args={},
                channel_id="channel-123",
            )

            assert "error" in raw
            assert "Unknown tool" in raw["error"]

    @pytest.mark.asyncio
    async def test_invoke_tool_timeout(self):
        """Should handle timeout errors."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(timeout_seconds=0.01, max_retries=0)

            async def slow_tool(**args):
                await asyncio.sleep(1.0)
                return {"result": "done"}

            bridge._tools = {"slow_tool": slow_tool}

            formatted, raw = await bridge.invoke_tool(
                tool_name="slow_tool",
                args={},
                channel_id="channel-123",
            )

            assert "error" in raw
            assert (
                ToolStatus.FAILED.value
                in bridge._invocations[list(bridge._invocations.keys())[0]].status.value
            )

    @pytest.mark.asyncio
    async def test_invoke_tool_with_retries(self):
        """Should retry on retryable errors."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(max_retries=2)
            call_count = 0

            async def flaky_tool(**args):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Connection failed")
                return {"success": True}

            bridge._tools = {"flaky_tool": flaky_tool}

            formatted, raw = await bridge.invoke_tool(
                tool_name="flaky_tool",
                args={},
                channel_id="channel-123",
            )

            assert raw["success"] is True
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_invoke_tool_non_retryable_error(self):
        """Should not retry non-retryable errors."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(max_retries=2)
            call_count = 0

            async def failing_tool(**args):
                nonlocal call_count
                call_count += 1
                raise ValueError("Invalid input")

            bridge._tools = {"failing_tool": failing_tool}

            formatted, raw = await bridge.invoke_tool(
                tool_name="failing_tool",
                args={},
                channel_id="channel-123",
            )

            assert "error" in raw
            assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_invoke_tool_records_invocation(self):
        """Should record invocation in internal storage."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            async def test_tool(**args):
                return {"done": True}

            bridge._tools = {"test_tool": test_tool}

            await bridge.invoke_tool(
                tool_name="test_tool",
                args={"param": "value"},
                channel_id="channel-123",
                user_id="user-456",
            )

            assert len(bridge._invocations) == 1
            invocation = list(bridge._invocations.values())[0]
            assert invocation.tool_name == "test_tool"
            assert invocation.channel_id == "channel-123"
            assert invocation.user_id == "user-456"
            assert invocation.status == ToolStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_invoke_tool_with_custom_timeout(self):
        """Should use custom timeout when provided."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge(timeout_seconds=300.0)

            async def quick_tool(**args):
                return {"quick": True}

            bridge._tools = {"quick_tool": quick_tool}

            # Pass a short timeout to verify it's used
            formatted, raw = await bridge.invoke_tool(
                tool_name="quick_tool",
                args={},
                channel_id="channel-123",
                timeout=0.5,
            )

            assert raw["quick"] is True


class TestResultFormatter:
    """Tests for ResultFormatter class."""

    def test_format_error_slack(self):
        """Should format errors for Slack."""
        formatter = ResultFormatter(Platform.SLACK)

        result = formatter.format_error("test_tool", "Something went wrong")

        assert ":x:" in result
        assert "*test_tool*" in result
        assert "Something went wrong" in result

    def test_format_error_discord(self):
        """Should format errors for Discord."""
        formatter = ResultFormatter(Platform.DISCORD)

        result = formatter.format_error("test_tool", "Something went wrong")

        assert ":x:" in result
        assert "**test_tool**" in result
        assert "Something went wrong" in result

    def test_format_error_generic(self):
        """Should format errors for generic platforms."""
        formatter = ResultFormatter(Platform.GENERIC)

        result = formatter.format_error("test_tool", "Something went wrong")

        assert "Error in test_tool" in result
        assert "Something went wrong" in result

    def test_format_progress_slack(self):
        """Should format progress for Slack."""
        formatter = ResultFormatter(Platform.SLACK)

        result = formatter.format_progress(50.0, "Processing...")

        assert "50%" in result
        assert "_Processing..._" in result
        assert "[" in result and "]" in result

    def test_format_progress_discord(self):
        """Should format progress for Discord."""
        formatter = ResultFormatter(Platform.DISCORD)

        result = formatter.format_progress(75.0, "Almost done")

        assert "75%" in result
        assert "*Almost done*" in result

    def test_format_debate_result_slack(self):
        """Should format debate results for Slack."""
        formatter = ResultFormatter(Platform.SLACK)

        result = formatter.format_result(
            "run_debate",
            {
                "task": "Should we use microservices?",
                "final_answer": "Yes, for scalability",
                "consensus_reached": True,
                "confidence": 0.85,
            },
        )

        assert ":speech_balloon:" in result
        assert "*Debate Complete*" in result
        assert "microservices" in result
        assert ":white_check_mark:" in result
        assert "85" in result

    def test_format_debate_result_discord(self):
        """Should format debate results for Discord."""
        formatter = ResultFormatter(Platform.DISCORD)

        result = formatter.format_result(
            "run_debate",
            {
                "task": "Should we use microservices?",
                "final_answer": "Yes",
                "consensus_reached": False,
                "confidence": 0.5,
            },
        )

        assert "**Debate Complete**" in result
        assert ":x:" in result  # No consensus

    def test_format_poll_result(self):
        """Should format poll results."""
        formatter = ResultFormatter(Platform.SLACK)

        result = formatter.format_result(
            "create_poll",
            {
                "poll_id": "poll-12345",
            },
        )

        assert "Poll created" in result
        assert "poll-12345" in result

    def test_format_knowledge_result_with_entries(self):
        """Should format knowledge query results."""
        formatter = ResultFormatter(Platform.GENERIC)

        result = formatter.format_result(
            "query_knowledge",
            {
                "entries": [
                    {"title": "Entry 1"},
                    {"title": "Entry 2"},
                ]
            },
        )

        assert "Found 2 entries" in result
        assert "Entry 1" in result
        assert "Entry 2" in result

    def test_format_knowledge_result_empty(self):
        """Should handle empty knowledge results."""
        formatter = ResultFormatter(Platform.GENERIC)

        result = formatter.format_result("query_knowledge", {"entries": []})

        assert "No knowledge entries found" in result

    def test_format_gauntlet_result_slack(self):
        """Should format gauntlet results for Slack."""
        formatter = ResultFormatter(Platform.SLACK)

        result = formatter.format_result(
            "run_gauntlet",
            {
                "score": 0.95,
                "findings": [{"id": "f1"}, {"id": "f2"}],
            },
        )

        assert ":shield:" in result
        assert "Gauntlet Complete" in result
        assert "95" in result
        assert "2" in result

    def test_format_result_with_error_key(self):
        """Should handle results with error key."""
        formatter = ResultFormatter(Platform.GENERIC)

        result = formatter.format_result(
            "any_tool",
            {
                "error": "Something failed",
            },
        )

        assert "Error in any_tool" in result
        assert "Something failed" in result

    def test_format_generic_result_success(self):
        """Should format generic successful results."""
        formatter = ResultFormatter(Platform.GENERIC)

        result = formatter.format_result(
            "unknown_tool",
            {
                "success": True,
                "data": "some data",
            },
        )

        assert "Operation completed successfully" in result

    def test_create_progress_bar(self):
        """Should create text progress bar."""
        formatter = ResultFormatter(Platform.GENERIC)

        bar = formatter._create_progress_bar(50.0, width=10)

        assert bar == "[=====     ]"

    def test_create_progress_bar_full(self):
        """Should create full progress bar."""
        formatter = ResultFormatter(Platform.GENERIC)

        bar = formatter._create_progress_bar(100.0, width=10)

        assert bar == "[==========]"

    def test_create_progress_bar_empty(self):
        """Should create empty progress bar."""
        formatter = ResultFormatter(Platform.GENERIC)

        bar = formatter._create_progress_bar(0.0, width=10)

        assert bar == "[          ]"


class TestErrorHandler:
    """Tests for ErrorHandler class."""

    def test_handle_timeout_error(self):
        """Should handle TimeoutError with friendly message."""
        handler = ErrorHandler(Platform.GENERIC)

        result = handler.handle_error(TimeoutError(), "slow_tool")

        assert "took too long" in result
        assert "slow_tool" in result

    def test_handle_connection_error(self):
        """Should handle ConnectionError with friendly message."""
        handler = ErrorHandler(Platform.GENERIC)

        result = handler.handle_error(ConnectionError(), "network_tool")

        assert "connect to the service" in result
        assert "network_tool" in result

    def test_handle_value_error(self):
        """Should handle ValueError with message."""
        handler = ErrorHandler(Platform.GENERIC)

        result = handler.handle_error(ValueError("bad input"), "input_tool")

        assert "Invalid input" in result
        assert "bad input" in result

    def test_handle_permission_error(self):
        """Should handle PermissionError with friendly message."""
        handler = ErrorHandler(Platform.GENERIC)

        result = handler.handle_error(PermissionError(), "restricted_tool")

        assert "permission" in result

    def test_handle_generic_error(self):
        """Should handle unknown errors."""
        handler = ErrorHandler(Platform.GENERIC)

        result = handler.handle_error(RuntimeError("Unexpected"), "tool")

        assert "error occurred" in result
        assert "Unexpected" in result

    def test_handle_error_slack_format(self):
        """Should format errors for Slack."""
        handler = ErrorHandler(Platform.SLACK)

        result = handler.handle_error(ValueError("test"), "tool")

        assert ":warning:" in result
        assert "*tool*" in result

    def test_handle_error_discord_format(self):
        """Should format errors for Discord."""
        handler = ErrorHandler(Platform.DISCORD)

        result = handler.handle_error(ValueError("test"), "tool")

        assert ":warning:" in result
        assert "**tool**" in result

    def test_should_retry_timeout(self):
        """Should mark TimeoutError as retryable."""
        handler = ErrorHandler(Platform.GENERIC)

        assert handler.should_retry(TimeoutError()) is True

    def test_should_retry_connection(self):
        """Should mark ConnectionError as retryable."""
        handler = ErrorHandler(Platform.GENERIC)

        assert handler.should_retry(ConnectionError()) is True

    def test_should_not_retry_value_error(self):
        """Should not retry ValueError."""
        handler = ErrorHandler(Platform.GENERIC)

        assert handler.should_retry(ValueError()) is False

    def test_should_not_retry_type_error(self):
        """Should not retry TypeError."""
        handler = ErrorHandler(Platform.GENERIC)

        assert handler.should_retry(TypeError()) is False


class TestProgressStreaming:
    """Tests for stream_tool_progress method."""

    @pytest.mark.asyncio
    async def test_stream_progress_unknown_tool(self):
        """Should yield failed status for unknown tool."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()
            bridge._tools = {}

            updates = []
            async for update in bridge.stream_tool_progress(
                tool_name="unknown",
                args={},
                channel_id="channel-123",
            ):
                updates.append(update)

            assert len(updates) == 1
            assert updates[0].status == ToolStatus.FAILED
            assert "Unknown tool" in updates[0].message

    @pytest.mark.asyncio
    async def test_stream_progress_success(self):
        """Should stream progress updates for successful tool."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            async def quick_tool(**args):
                return {"completed": True}

            bridge._tools = {"quick_tool": quick_tool}

            updates = []
            async for update in bridge.stream_tool_progress(
                tool_name="quick_tool",
                args={},
                channel_id="channel-123",
                update_interval=0.01,
            ):
                updates.append(update)

            # Should have at least initial and final updates
            assert len(updates) >= 2
            assert updates[0].status == ToolStatus.RUNNING
            assert updates[0].progress == 0.0
            assert updates[-1].status == ToolStatus.COMPLETED
            assert updates[-1].progress == 100.0
            assert updates[-1].partial_result is not None

    @pytest.mark.asyncio
    async def test_stream_progress_tool_failure(self):
        """Should stream failure status when tool raises."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            async def failing_tool(**args):
                raise RuntimeError("Tool crashed")

            bridge._tools = {"failing_tool": failing_tool}

            updates = []
            async for update in bridge.stream_tool_progress(
                tool_name="failing_tool",
                args={},
                channel_id="channel-123",
                update_interval=0.01,
            ):
                updates.append(update)

            final = updates[-1]
            assert final.status == ToolStatus.FAILED
            assert "Tool crashed" in final.message


class TestInvocationManagement:
    """Tests for invocation management methods."""

    @pytest.mark.asyncio
    async def test_get_invocation(self):
        """Should retrieve invocation by ID."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            async def test_tool(**args):
                return {"done": True}

            bridge._tools = {"test_tool": test_tool}

            await bridge.invoke_tool(
                tool_name="test_tool",
                args={},
                channel_id="channel-123",
            )

            invocation_id = list(bridge._invocations.keys())[0]
            invocation = await bridge.get_invocation(invocation_id)

            assert invocation is not None
            assert invocation.id == invocation_id
            assert invocation.tool_name == "test_tool"

    @pytest.mark.asyncio
    async def test_get_invocation_not_found(self):
        """Should return None for unknown invocation ID."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            invocation = await bridge.get_invocation("nonexistent-id")

            assert invocation is None

    @pytest.mark.asyncio
    async def test_cancel_invocation_running(self):
        """Should cancel running invocation."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            # Manually create a running invocation
            invocation = ToolInvocation(
                id="inv-123",
                tool_name="test",
                args={},
                status=ToolStatus.RUNNING,
                channel_id="channel",
                user_id=None,
                platform=Platform.GENERIC,
                started_at=datetime.now(timezone.utc),
            )
            bridge._invocations["inv-123"] = invocation

            result = await bridge.cancel_invocation("inv-123")

            assert result is True
            assert invocation.status == ToolStatus.CANCELLED
            assert invocation.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_invocation_not_running(self):
        """Should not cancel non-running invocation."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            invocation = ToolInvocation(
                id="inv-123",
                tool_name="test",
                args={},
                status=ToolStatus.COMPLETED,
                channel_id="channel",
                user_id=None,
                platform=Platform.GENERIC,
                started_at=datetime.now(timezone.utc),
            )
            bridge._invocations["inv-123"] = invocation

            result = await bridge.cancel_invocation("inv-123")

            assert result is False
            assert invocation.status == ToolStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_invocation_not_found(self):
        """Should return False for unknown invocation."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            result = await bridge.cancel_invocation("nonexistent")

            assert result is False


class TestToolInvocationDataclass:
    """Tests for ToolInvocation dataclass."""

    def test_create_invocation(self):
        """Should create invocation with all fields."""
        invocation = ToolInvocation(
            id="inv-123",
            tool_name="test_tool",
            args={"key": "value"},
            status=ToolStatus.PENDING,
            channel_id="channel-456",
            user_id="user-789",
            platform=Platform.SLACK,
            started_at=datetime.now(timezone.utc),
        )

        assert invocation.id == "inv-123"
        assert invocation.tool_name == "test_tool"
        assert invocation.args == {"key": "value"}
        assert invocation.status == ToolStatus.PENDING
        assert invocation.channel_id == "channel-456"
        assert invocation.user_id == "user-789"
        assert invocation.platform == Platform.SLACK
        assert invocation.progress == 0.0
        assert invocation.result is None
        assert invocation.error is None

    def test_invocation_default_values(self):
        """Should have sensible defaults."""
        invocation = ToolInvocation(
            id="inv-1",
            tool_name="tool",
            args={},
            status=ToolStatus.RUNNING,
            channel_id="ch",
            user_id=None,
            platform=Platform.GENERIC,
            started_at=datetime.now(timezone.utc),
        )

        assert invocation.completed_at is None
        assert invocation.result is None
        assert invocation.error is None
        assert invocation.progress == 0.0
        assert invocation.progress_message is None
        assert invocation.metadata == {}


class TestProgressUpdateDataclass:
    """Tests for ProgressUpdate dataclass."""

    def test_create_progress_update(self):
        """Should create progress update with all fields."""
        update = ProgressUpdate(
            progress=50.0,
            message="Halfway done",
            status=ToolStatus.RUNNING,
            timestamp=datetime.now(timezone.utc),
            partial_result={"partial": True},
        )

        assert update.progress == 50.0
        assert update.message == "Halfway done"
        assert update.status == ToolStatus.RUNNING
        assert update.partial_result == {"partial": True}

    def test_progress_update_default_partial_result(self):
        """Should default partial_result to None."""
        update = ProgressUpdate(
            progress=0.0,
            message="Starting",
            status=ToolStatus.RUNNING,
            timestamp=datetime.now(timezone.utc),
        )

        assert update.partial_result is None


class TestGetToolBridge:
    """Tests for get_tool_bridge singleton function."""

    def test_get_tool_bridge_creates_new(self):
        """Should create new bridge for platform."""
        reset_tool_bridges()

        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = get_tool_bridge("slack")

            assert bridge.platform == Platform.SLACK

    def test_get_tool_bridge_returns_cached(self):
        """Should return cached bridge for same platform."""
        reset_tool_bridges()

        with patch.object(ToolBridge, "_register_default_tools"):
            bridge1 = get_tool_bridge("discord")
            bridge2 = get_tool_bridge("discord")

            assert bridge1 is bridge2

    def test_get_tool_bridge_different_platforms(self):
        """Should return different bridges for different platforms."""
        reset_tool_bridges()

        with patch.object(ToolBridge, "_register_default_tools"):
            slack = get_tool_bridge("slack")
            discord = get_tool_bridge("discord")

            assert slack is not discord
            assert slack.platform == Platform.SLACK
            assert discord.platform == Platform.DISCORD

    def test_get_tool_bridge_unknown_platform(self):
        """Should fallback to generic for unknown platforms."""
        reset_tool_bridges()

        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = get_tool_bridge("unknown_platform")

            assert bridge.platform == Platform.GENERIC


class TestResetToolBridges:
    """Tests for reset_tool_bridges function."""

    def test_reset_clears_all_bridges(self):
        """Should clear all cached bridges."""
        with patch.object(ToolBridge, "_register_default_tools"):
            # Create some bridges
            _ = get_tool_bridge("slack")
            _ = get_tool_bridge("discord")

            reset_tool_bridges()

            # After reset, should create new bridges
            from aragora.connectors.chat.tool_bridge import _bridges

            assert len(_bridges) == 0


class TestDebateResultFormatting:
    """Additional tests for debate result formatting edge cases."""

    def test_format_debate_missing_fields(self):
        """Should handle missing fields gracefully."""
        formatter = ResultFormatter(Platform.GENERIC)

        result = formatter.format_result("run_debate", {})

        assert "N/A" in result
        assert "Debate Complete" in result

    def test_format_debate_zero_confidence(self):
        """Should handle zero confidence."""
        formatter = ResultFormatter(Platform.SLACK)

        result = formatter.format_result(
            "run_debate",
            {
                "task": "Test",
                "final_answer": "Answer",
                "consensus_reached": True,
                "confidence": 0,
            },
        )

        assert "0" in result


class TestConcurrentInvocations:
    """Tests for concurrent tool invocations."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_invocations(self):
        """Should handle multiple concurrent invocations."""
        with patch.object(ToolBridge, "_register_default_tools"):
            bridge = ToolBridge()

            async def async_tool(**args):
                await asyncio.sleep(0.01)
                return {"id": args.get("id")}

            bridge._tools = {"async_tool": async_tool}

            # Run multiple invocations concurrently
            results = await asyncio.gather(
                bridge.invoke_tool("async_tool", {"id": 1}, "ch-1"),
                bridge.invoke_tool("async_tool", {"id": 2}, "ch-1"),
                bridge.invoke_tool("async_tool", {"id": 3}, "ch-1"),
            )

            assert len(results) == 3
            assert all(r[1]["id"] in [1, 2, 3] for r in results)
            assert len(bridge._invocations) == 3
