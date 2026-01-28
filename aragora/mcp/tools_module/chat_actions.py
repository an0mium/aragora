"""
MCP Chat Action Tools.

Tool-first chat interactions for sending messages, creating polls,
triggering debates, and posting receipts.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def send_message_tool(
    channel_id: str,
    content: str,
    platform: str = "slack",
    thread_id: Optional[str] = None,
    reply_to: Optional[str] = None,
    format_type: str = "text",
) -> Dict[str, Any]:
    """
    Send a message to a chat channel.

    Args:
        channel_id: ID of the channel to send to
        content: Message content
        platform: Platform type (slack, discord, telegram, teams)
        thread_id: Optional thread to reply in
        reply_to: Optional message ID to reply to
        format_type: Format type (text, markdown, blocks)

    Returns:
        Dict with send result including message ID
    """
    try:
        connector = await _get_chat_connector(platform)
        if not connector:
            return {
                "error": f"No connector available for platform: {platform}",
                "channel_id": channel_id,
            }

        # Format message based on platform
        formatted_content = _format_message(content, platform, format_type)

        # Send message
        message_id = None
        if hasattr(connector, "send_message"):
            result = await connector.send_message(
                channel_id=channel_id,
                content=formatted_content,
                thread_id=thread_id,
                reply_to=reply_to,
            )
            message_id = result.get("message_id") if isinstance(result, dict) else str(result)

        return {
            "success": True,
            "message_id": message_id or f"msg_{uuid.uuid4().hex[:8]}",
            "channel_id": channel_id,
            "platform": platform,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return {
            "error": str(e),
            "channel_id": channel_id,
            "platform": platform,
        }


async def create_poll_tool(
    channel_id: str,
    question: str,
    options: List[str],
    platform: str = "slack",
    duration_minutes: int = 60,
    anonymous: bool = False,
    multiple_choice: bool = False,
) -> Dict[str, Any]:
    """
    Create a voting poll in a chat channel.

    Args:
        channel_id: ID of the channel
        question: Poll question
        options: List of poll options
        platform: Platform type
        duration_minutes: Poll duration in minutes
        anonymous: Whether votes are anonymous
        multiple_choice: Whether multiple options can be selected

    Returns:
        Dict with poll creation result including poll ID
    """
    try:
        if not options or len(options) < 2:
            return {"error": "Poll requires at least 2 options"}

        if len(options) > 10:
            return {"error": "Poll can have at most 10 options"}

        connector = await _get_chat_connector(platform)
        if not connector:
            return {"error": f"No connector for platform: {platform}"}

        poll_id = f"poll_{uuid.uuid4().hex[:8]}"

        # Create poll message
        poll_message = _format_poll(question, options, platform)

        # Send poll
        message_id = None
        if hasattr(connector, "create_poll"):
            result = await connector.create_poll(
                channel_id=channel_id,
                question=question,
                options=options,
                duration_minutes=duration_minutes,
                anonymous=anonymous,
                multiple_choice=multiple_choice,
            )
            message_id = result.get("message_id")
        elif hasattr(connector, "send_message"):
            result = await connector.send_message(
                channel_id=channel_id,
                content=poll_message,
            )
            message_id = result.get("message_id") if isinstance(result, dict) else str(result)

        return {
            "success": True,
            "poll_id": poll_id,
            "message_id": message_id,
            "channel_id": channel_id,
            "question": question,
            "options": options,
            "duration_minutes": duration_minutes,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to create poll: {e}")
        return {"error": str(e), "channel_id": channel_id}


async def trigger_debate_tool(
    channel_id: str,
    question: str,
    platform: str = "slack",
    agents: str = "anthropic-api,openai-api",
    rounds: int = 3,
    stream_progress: bool = True,
    post_receipt: bool = True,
) -> Dict[str, Any]:
    """
    Start a debate from chat context.

    Triggers a debate and optionally streams progress back to the channel.

    Args:
        channel_id: ID of the originating channel
        question: The debate question
        platform: Platform type
        agents: Comma-separated agent IDs
        rounds: Number of debate rounds
        stream_progress: Whether to stream progress to channel
        post_receipt: Whether to post receipt on completion

    Returns:
        Dict with debate trigger result including debate ID
    """
    try:
        from aragora.mcp.tools_module.debate import run_debate_tool

        # Note: Origin tracking for result routing will be implemented when
        # origin-aware debate routing is added. For now, we route results
        # through the debate tool's callback mechanism.

        # Notify channel that debate is starting
        await send_message_tool(
            channel_id=channel_id,
            content=f"Starting debate on: {question}\nAgents: {agents}\nRounds: {rounds}",
            platform=platform,
            format_type="markdown",
        )

        # Run the debate
        result = await run_debate_tool(
            question=question,
            agents=agents,
            rounds=rounds,
        )

        if "error" in result:
            await send_message_tool(
                channel_id=channel_id,
                content=f"Debate failed: {result['error']}",
                platform=platform,
            )
            return result

        # Post receipt if requested
        if post_receipt:
            await post_receipt_tool(
                channel_id=channel_id,
                debate_id=result["debate_id"],
                platform=platform,
                include_summary=True,
            )

        return {
            "success": True,
            "debate_id": result["debate_id"],
            "channel_id": channel_id,
            "question": question,
            "final_answer": result.get("final_answer"),
            "consensus_reached": result.get("consensus_reached"),
            "confidence": result.get("confidence"),
        }

    except Exception as e:
        logger.error(f"Failed to trigger debate: {e}")
        return {"error": str(e), "channel_id": channel_id}


async def post_receipt_tool(
    channel_id: str,
    debate_id: str,
    platform: str = "slack",
    include_summary: bool = True,
    include_hash: bool = True,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Post a debate receipt to a channel.

    Posts a formatted receipt with cryptographic proof of the debate outcome.

    Args:
        channel_id: ID of the channel
        debate_id: ID of the debate
        platform: Platform type
        include_summary: Include debate summary
        include_hash: Include cryptographic hash
        thread_id: Optional thread to post in

    Returns:
        Dict with receipt posting result
    """
    try:
        from aragora.export.decision_receipt import DecisionReceipt
        from aragora.mcp.tools_module.debate import get_debate_tool

        # Get debate data
        debate = await get_debate_tool(debate_id)
        if "error" in debate:
            return debate

        # Generate a lightweight receipt snapshot for chat display
        receipt_obj = DecisionReceipt(
            receipt_id=f"rcpt_{uuid.uuid4().hex[:12]}",
            gauntlet_id=debate_id,
            input_summary=debate.get("task", ""),
            confidence=debate.get("confidence", 0.0),
            verdict="APPROVED" if debate.get("consensus_reached") else "NEEDS_REVIEW",
            agents_involved=debate.get("agents", []),
            rounds_completed=len(debate.get("rounds", []))
            if debate.get("rounds")
            else debate.get("rounds_completed", 0),
        )
        receipt = {
            "task": debate.get("task", ""),
            "final_answer": debate.get("final_answer", ""),
            "consensus_reached": debate.get("consensus_reached", False),
            "confidence": debate.get("confidence", 0.0),
            "hash": receipt_obj.checksum,
            "timestamp": receipt_obj.timestamp,
        }

        # Format receipt for chat
        receipt_content = _format_receipt(receipt, platform, include_summary, include_hash)

        # Post to channel
        result = await send_message_tool(
            channel_id=channel_id,
            content=receipt_content,
            platform=platform,
            thread_id=thread_id,
            format_type="markdown",
        )

        return {
            "success": True,
            "debate_id": debate_id,
            "receipt_hash": receipt.get("hash") if include_hash else None,
            "message_id": result.get("message_id"),
            "channel_id": channel_id,
            "posted_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to post receipt: {e}")
        return {"error": str(e), "debate_id": debate_id}


async def update_message_tool(
    message_id: str,
    channel_id: str,
    content: str,
    platform: str = "slack",
) -> Dict[str, Any]:
    """
    Update an existing message.

    Args:
        message_id: ID of the message to update
        channel_id: ID of the channel
        content: New message content
        platform: Platform type

    Returns:
        Dict with update result
    """
    try:
        connector = await _get_chat_connector(platform)
        if not connector:
            return {"error": f"No connector for platform: {platform}"}

        if hasattr(connector, "update_message"):
            await connector.update_message(
                message_id=message_id,
                channel_id=channel_id,
                content=content,
            )

        return {
            "success": True,
            "message_id": message_id,
            "channel_id": channel_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to update message: {e}")
        return {"error": str(e), "message_id": message_id}


async def add_reaction_tool(
    message_id: str,
    channel_id: str,
    emoji: str,
    platform: str = "slack",
) -> Dict[str, Any]:
    """
    Add a reaction to a message.

    Args:
        message_id: ID of the message
        channel_id: ID of the channel
        emoji: Emoji to add (without colons)
        platform: Platform type

    Returns:
        Dict with reaction result
    """
    try:
        connector = await _get_chat_connector(platform)
        if not connector:
            return {"error": f"No connector for platform: {platform}"}

        if hasattr(connector, "add_reaction"):
            await connector.add_reaction(
                message_id=message_id,
                channel_id=channel_id,
                emoji=emoji,
            )

        return {
            "success": True,
            "message_id": message_id,
            "emoji": emoji,
        }

    except Exception as e:
        logger.error(f"Failed to add reaction: {e}")
        return {"error": str(e), "message_id": message_id}


async def create_thread_tool(
    channel_id: str,
    parent_message_id: str,
    content: str,
    platform: str = "slack",
) -> Dict[str, Any]:
    """
    Create a thread reply.

    Args:
        channel_id: ID of the channel
        parent_message_id: ID of the parent message
        content: Thread message content
        platform: Platform type

    Returns:
        Dict with thread creation result
    """
    return await send_message_tool(
        channel_id=channel_id,
        content=content,
        platform=platform,
        thread_id=parent_message_id,
    )


async def stream_progress_tool(
    channel_id: str,
    message_id: str,
    progress: float,
    status: str,
    platform: str = "slack",
) -> Dict[str, Any]:
    """
    Stream progress update to a message.

    Updates an existing message with progress information.

    Args:
        channel_id: ID of the channel
        message_id: ID of the message to update
        progress: Progress percentage (0-100)
        status: Status message
        platform: Platform type

    Returns:
        Dict with update result
    """
    progress_bar = _create_progress_bar(progress)
    content = f"{progress_bar} {progress:.0f}%\n{status}"

    return await update_message_tool(
        message_id=message_id,
        channel_id=channel_id,
        content=content,
        platform=platform,
    )


# Helper functions


async def _get_chat_connector(platform: str) -> Optional[Any]:
    """Get the chat connector for a platform."""
    try:
        if platform == "slack":
            from aragora.connectors.slack import SlackConnector

            return SlackConnector()
        elif platform == "discord":
            from aragora.connectors.chat.discord import DiscordConnector

            return DiscordConnector()
        elif platform == "telegram":
            from aragora.connectors.chat.telegram import TelegramConnector

            return TelegramConnector()
        elif platform == "teams":
            from aragora.connectors.chat.teams import TeamsConnector

            return TeamsConnector()
        else:
            logger.warning(f"Unknown platform: {platform}")
            return None
    except ImportError as e:
        logger.warning(f"Could not import connector for {platform}: {e}")
        return None


def _format_message(content: str, platform: str, format_type: str) -> str:
    """Format message content for platform."""
    if format_type == "text":
        return content

    if format_type == "markdown":
        # Platform-specific markdown handling
        if platform == "slack":
            # Slack uses different markdown syntax
            content = content.replace("**", "*")  # Bold
            content = content.replace("__", "_")  # Italic
        # Discord and others use standard markdown

    return content


def _format_poll(question: str, options: List[str], platform: str) -> str:
    """Format poll as message."""
    lines = [f"**Poll:** {question}", ""]

    if platform == "slack":
        for i, opt in enumerate(options):
            emoji = f":{['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'keycap_ten'][i]}:"
            lines.append(f"{emoji} {opt}")
    else:
        for i, opt in enumerate(options, 1):
            lines.append(f"{i}. {opt}")

    lines.append("")
    lines.append("_React to vote!_")

    return "\n".join(lines)


def _format_receipt(
    receipt: Dict[str, Any],
    platform: str,
    include_summary: bool,
    include_hash: bool,
) -> str:
    """Format receipt for chat."""
    lines = ["**Debate Receipt**", ""]

    if include_summary:
        lines.append(f"**Question:** {receipt.get('task', 'N/A')}")
        lines.append(f"**Answer:** {receipt.get('final_answer', 'N/A')}")
        lines.append(f"**Consensus:** {'Yes' if receipt.get('consensus_reached') else 'No'}")
        lines.append(f"**Confidence:** {receipt.get('confidence', 0):.1%}")
        lines.append("")

    if include_hash:
        lines.append(f"**Hash:** `{receipt.get('hash', 'N/A')[:16]}...`")
        lines.append(f"**Timestamp:** {receipt.get('timestamp', 'N/A')}")

    return "\n".join(lines)


def _create_progress_bar(progress: float, width: int = 20) -> str:
    """Create a text progress bar."""
    filled = int(width * progress / 100)
    empty = width - filled
    return f"[{'=' * filled}{' ' * empty}]"
