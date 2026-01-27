"""
Tool Bridge: MCP-Chat Bridge.

Bridges between MCP tools and chat platforms, providing:
- Tool invocation from chat context
- Result formatting for each platform
- Progress reporting during execution
- Error handling with user-friendly messages

Usage:
    from aragora.connectors.chat.tool_bridge import ToolBridge

    bridge = ToolBridge(platform="slack")

    # Invoke a tool from chat
    result = await bridge.invoke_tool(
        tool_name="run_debate",
        args={"question": "Should we use microservices?"},
        channel_id="C123456",
    )

    # Stream progress to chat
    async for update in bridge.stream_tool_progress(tool_name, args, channel_id):
        await send_update(update)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ToolStatus(str, Enum):
    """Status of tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class Platform(str, Enum):
    """Supported chat platforms."""

    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    TEAMS = "teams"
    WHATSAPP = "whatsapp"
    GENERIC = "generic"


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""

    id: str
    tool_name: str
    args: Dict[str, Any]
    status: ToolStatus
    channel_id: str
    user_id: Optional[str]
    platform: Platform
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0
    progress_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressUpdate:
    """Progress update for streaming."""

    progress: float  # 0-100
    message: str
    status: ToolStatus
    timestamp: datetime
    partial_result: Optional[Dict[str, Any]] = None


class ResultFormatter:
    """Formats tool results for different platforms."""

    def __init__(self, platform: Platform):
        """
        Initialize the formatter.

        Args:
            platform: Target platform
        """
        self.platform = platform

    def format_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """
        Format a tool result for the platform.

        Args:
            tool_name: Name of the tool
            result: Tool result dict

        Returns:
            Formatted string for the platform
        """
        if "error" in result:
            return self.format_error(tool_name, result["error"])

        formatter = self._get_result_formatter(tool_name)
        return formatter(result)

    def format_error(self, tool_name: str, error: str) -> str:
        """Format an error message."""
        if self.platform == Platform.SLACK:
            return f":x: *{tool_name}* failed:\n```{error}```"
        elif self.platform == Platform.DISCORD:
            return f":x: **{tool_name}** failed:\n```{error}```"
        else:
            return f"Error in {tool_name}: {error}"

    def format_progress(self, progress: float, message: str) -> str:
        """Format a progress update."""
        bar = self._create_progress_bar(progress)

        if self.platform == Platform.SLACK:
            return f"{bar} {progress:.0f}%\n_{message}_"
        elif self.platform == Platform.DISCORD:
            return f"{bar} {progress:.0f}%\n*{message}*"
        else:
            return f"{bar} {progress:.0f}% - {message}"

    def _get_result_formatter(self, tool_name: str) -> Callable[[Dict[str, Any]], str]:
        """Get the formatter for a specific tool."""
        formatters = {
            "run_debate": self._format_debate_result,
            "get_debate": self._format_debate_result,
            "create_poll": self._format_poll_result,
            "query_knowledge": self._format_knowledge_result,
            "run_gauntlet": self._format_gauntlet_result,
        }
        return formatters.get(tool_name, self._format_generic_result)

    def _format_debate_result(self, result: Dict[str, Any]) -> str:
        """Format debate result."""
        lines = []

        if self.platform == Platform.SLACK:
            lines.append(":speech_balloon: *Debate Complete*")
            lines.append("")
            lines.append(f"*Question:* {result.get('task', 'N/A')}")
            lines.append(f"*Answer:* {result.get('final_answer', 'N/A')}")
            consensus = ":white_check_mark:" if result.get("consensus_reached") else ":x:"
            lines.append(f"*Consensus:* {consensus}")
            lines.append(f"*Confidence:* {result.get('confidence', 0):.1%}")
        elif self.platform == Platform.DISCORD:
            lines.append(":speech_balloon: **Debate Complete**")
            lines.append("")
            lines.append(f"**Question:** {result.get('task', 'N/A')}")
            lines.append(f"**Answer:** {result.get('final_answer', 'N/A')}")
            consensus = ":white_check_mark:" if result.get("consensus_reached") else ":x:"
            lines.append(f"**Consensus:** {consensus}")
            lines.append(f"**Confidence:** {result.get('confidence', 0):.1%}")
        else:
            lines.append("Debate Complete")
            lines.append(f"Question: {result.get('task', 'N/A')}")
            lines.append(f"Answer: {result.get('final_answer', 'N/A')}")
            lines.append(f"Consensus: {'Yes' if result.get('consensus_reached') else 'No'}")
            lines.append(f"Confidence: {result.get('confidence', 0):.1%}")

        return "\n".join(lines)

    def _format_poll_result(self, result: Dict[str, Any]) -> str:
        """Format poll creation result."""
        if self.platform == Platform.SLACK:
            return f":bar_chart: Poll created: {result.get('poll_id', 'N/A')}"
        return f"Poll created: {result.get('poll_id', 'N/A')}"

    def _format_knowledge_result(self, result: Dict[str, Any]) -> str:
        """Format knowledge query result."""
        entries = result.get("entries", [])
        if not entries:
            return "No knowledge entries found."

        lines = [f"Found {len(entries)} entries:"]
        for entry in entries[:5]:
            lines.append(f"- {entry.get('title', 'Untitled')}")

        return "\n".join(lines)

    def _format_gauntlet_result(self, result: Dict[str, Any]) -> str:
        """Format gauntlet result."""
        findings = result.get("findings", [])
        score = result.get("score", 0)

        if self.platform in (Platform.SLACK, Platform.DISCORD):
            prefix = "*" if self.platform == Platform.SLACK else "**"
            return (
                f":shield: {prefix}Gauntlet Complete{prefix}\n"
                f"Score: {score:.1%}\n"
                f"Findings: {len(findings)}"
            )
        return f"Gauntlet Complete - Score: {score:.1%}, Findings: {len(findings)}"

    def _format_generic_result(self, result: Dict[str, Any]) -> str:
        """Format a generic result."""
        if "success" in result and result["success"]:
            return "Operation completed successfully."
        return str(result)

    def _create_progress_bar(self, progress: float, width: int = 20) -> str:
        """Create a text progress bar."""
        filled = int(width * progress / 100)
        empty = width - filled
        return f"[{'=' * filled}{' ' * empty}]"


class ErrorHandler:
    """Handles errors with user-friendly messages."""

    def __init__(self, platform: Platform):
        """
        Initialize the error handler.

        Args:
            platform: Target platform
        """
        self.platform = platform

    def handle_error(self, error: Exception, tool_name: str) -> str:
        """
        Create a user-friendly error message.

        Args:
            error: The exception
            tool_name: Name of the tool that failed

        Returns:
            User-friendly error message
        """
        error_type = type(error).__name__
        error_msg = str(error)

        # Map common errors to friendly messages
        friendly_messages = {
            "TimeoutError": "The operation took too long. Please try again.",
            "ConnectionError": "Could not connect to the service. Please try again later.",
            "ValueError": f"Invalid input: {error_msg}",
            "PermissionError": "You don't have permission to perform this action.",
            "RateLimitError": "Too many requests. Please wait a moment and try again.",
        }

        message = friendly_messages.get(error_type, f"An error occurred: {error_msg}")

        if self.platform == Platform.SLACK:
            return f":warning: *{tool_name}* encountered an issue:\n{message}"
        elif self.platform == Platform.DISCORD:
            return f":warning: **{tool_name}** encountered an issue:\n{message}"
        else:
            return f"Error in {tool_name}: {message}"

    def should_retry(self, error: Exception) -> bool:
        """Check if the error is retryable."""
        retryable = (
            "TimeoutError",
            "ConnectionError",
            "RateLimitError",
            "ServiceUnavailable",
        )
        return type(error).__name__ in retryable


class ToolBridge:
    """
    Bridges between MCP tools and chat platforms.

    Provides:
    - Tool invocation from chat context
    - Result formatting
    - Progress streaming
    - Error handling
    """

    def __init__(
        self,
        platform: str | Platform = Platform.GENERIC,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
    ):
        """
        Initialize the tool bridge.

        Args:
            platform: Target chat platform
            timeout_seconds: Default timeout for tool execution
            max_retries: Maximum retry attempts for retryable errors
        """
        if isinstance(platform, str):
            try:
                self.platform = Platform(platform.lower())
            except ValueError:
                self.platform = Platform.GENERIC
        else:
            self.platform = platform

        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        self.formatter = ResultFormatter(self.platform)
        self.error_handler = ErrorHandler(self.platform)

        self._invocations: Dict[str, ToolInvocation] = {}
        self._tools: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default MCP tools."""
        from aragora.mcp.tools_module.debate import run_debate_tool, get_debate_tool
        from aragora.mcp.tools_module.chat_actions import (
            send_message_tool,
            create_poll_tool,
            trigger_debate_tool,
            post_receipt_tool,
        )
        from aragora.mcp.tools_module.context_tools import (
            fetch_channel_context_tool,
            fetch_debate_context_tool,
            analyze_conversation_tool,
        )

        self._tools = {
            "run_debate": run_debate_tool,
            "get_debate": get_debate_tool,
            "send_message": send_message_tool,
            "create_poll": create_poll_tool,
            "trigger_debate": trigger_debate_tool,
            "post_receipt": post_receipt_tool,
            "fetch_channel_context": fetch_channel_context_tool,
            "fetch_debate_context": fetch_debate_context_tool,
            "analyze_conversation": analyze_conversation_tool,
        }

    def register_tool(self, name: str, tool_fn: Callable) -> None:
        """Register a custom tool."""
        self._tools[name] = tool_fn

    async def invoke_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        channel_id: str,
        user_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Invoke a tool and return formatted result.

        Args:
            tool_name: Name of the tool to invoke
            args: Tool arguments
            channel_id: Originating channel ID
            user_id: Optional user ID
            timeout: Optional timeout override

        Returns:
            Tuple of (formatted_message, raw_result)
        """
        import uuid

        # Get tool function
        tool_fn = self._tools.get(tool_name)
        if not tool_fn:
            error_msg = f"Unknown tool: {tool_name}"
            return self.error_handler.handle_error(ValueError(error_msg), tool_name), {
                "error": error_msg
            }

        # Create invocation record
        invocation = ToolInvocation(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            args=args,
            status=ToolStatus.PENDING,
            channel_id=channel_id,
            user_id=user_id,
            platform=self.platform,
            started_at=datetime.now(timezone.utc),
        )

        async with self._lock:
            self._invocations[invocation.id] = invocation

        # Execute with retries
        timeout_val = timeout or self.timeout_seconds
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                invocation.status = ToolStatus.RUNNING
                invocation.progress_message = f"Attempt {attempt + 1}/{self.max_retries + 1}"

                result = await asyncio.wait_for(
                    tool_fn(**args),
                    timeout=timeout_val,
                )

                invocation.status = ToolStatus.COMPLETED
                invocation.completed_at = datetime.now(timezone.utc)
                invocation.result = result
                invocation.progress = 100.0

                formatted = self.formatter.format_result(tool_name, result)
                return formatted, result

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Tool timed out after {timeout_val}s")
                if not self.error_handler.should_retry(last_error):
                    break

            except Exception as e:
                last_error = e
                if not self.error_handler.should_retry(e):
                    break

                # Wait before retry
                await asyncio.sleep(1.0 * (attempt + 1))

        # All retries failed
        invocation.status = ToolStatus.FAILED
        invocation.completed_at = datetime.now(timezone.utc)
        invocation.error = str(last_error)

        error_msg = self.error_handler.handle_error(last_error, tool_name)
        return error_msg, {"error": str(last_error)}

    async def stream_tool_progress(
        self,
        tool_name: str,
        args: Dict[str, Any],
        channel_id: str,
        update_interval: float = 2.0,
    ) -> AsyncIterator[ProgressUpdate]:
        """
        Stream tool progress updates.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            channel_id: Originating channel
            update_interval: Seconds between updates

        Yields:
            ProgressUpdate objects
        """

        tool_fn = self._tools.get(tool_name)
        if not tool_fn:
            yield ProgressUpdate(
                progress=0.0,
                message=f"Unknown tool: {tool_name}",
                status=ToolStatus.FAILED,
                timestamp=datetime.now(timezone.utc),
            )
            return

        # Start task
        task = asyncio.create_task(tool_fn(**args))
        start_time = time.time()
        estimated_duration = 30.0  # Default estimate

        yield ProgressUpdate(
            progress=0.0,
            message=f"Starting {tool_name}...",
            status=ToolStatus.RUNNING,
            timestamp=datetime.now(timezone.utc),
        )

        while not task.done():
            elapsed = time.time() - start_time
            # Estimate progress based on elapsed time
            estimated_progress = min(95.0, (elapsed / estimated_duration) * 100)

            yield ProgressUpdate(
                progress=estimated_progress,
                message=f"Running {tool_name}... ({elapsed:.0f}s)",
                status=ToolStatus.RUNNING,
                timestamp=datetime.now(timezone.utc),
            )

            await asyncio.sleep(update_interval)

        # Get result
        try:
            result = await task
            yield ProgressUpdate(
                progress=100.0,
                message="Complete!",
                status=ToolStatus.COMPLETED,
                timestamp=datetime.now(timezone.utc),
                partial_result=result,
            )
        except Exception as e:
            yield ProgressUpdate(
                progress=0.0,
                message=str(e),
                status=ToolStatus.FAILED,
                timestamp=datetime.now(timezone.utc),
            )

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())

    def get_tool_help(self, tool_name: str) -> Optional[str]:
        """Get help text for a tool."""
        tool_fn = self._tools.get(tool_name)
        if tool_fn and tool_fn.__doc__:
            return tool_fn.__doc__.strip()
        return None

    async def get_invocation(self, invocation_id: str) -> Optional[ToolInvocation]:
        """Get an invocation by ID."""
        return self._invocations.get(invocation_id)

    async def cancel_invocation(self, invocation_id: str) -> bool:
        """Cancel a running invocation."""
        invocation = self._invocations.get(invocation_id)
        if invocation and invocation.status == ToolStatus.RUNNING:
            invocation.status = ToolStatus.CANCELLED
            invocation.completed_at = datetime.now(timezone.utc)
            return True
        return False


# Singleton instance per platform
_bridges: Dict[Platform, ToolBridge] = {}


def get_tool_bridge(platform: str = "generic") -> ToolBridge:
    """Get the tool bridge for a platform."""
    try:
        p = Platform(platform.lower())
    except ValueError:
        p = Platform.GENERIC

    if p not in _bridges:
        _bridges[p] = ToolBridge(platform=p)

    return _bridges[p]


def reset_tool_bridges() -> None:
    """Reset all tool bridges (for testing)."""
    global _bridges
    _bridges = {}
