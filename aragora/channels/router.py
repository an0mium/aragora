"""
Channel Router - Dock-based message routing for debate results.

Provides a unified interface for routing debate results, receipts, and
error messages through the Channel Dock system.

This module is the primary entry point for sending debate-related
messages to chat platforms.

Example:
    from aragora.channels.router import ChannelRouter

    router = ChannelRouter()
    await router.route_result(platform, channel_id, result, **kwargs)
    await router.route_receipt(platform, channel_id, receipt, receipt_url)
    await router.route_error(platform, channel_id, error_message)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from aragora.channels.dock import ChannelCapability, MessageType, SendResult
from aragora.channels.normalized import NormalizedMessage, MessageFormat

if TYPE_CHECKING:
    from aragora.channels.dock import ChannelDock
    from aragora.channels.registry import DockRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "ChannelRouter",
    "get_channel_router",
]


class ChannelRouter:
    """
    Routes messages to chat platforms through the dock system.

    Provides high-level methods for common debate messaging needs:
    - Debate results
    - Decision receipts
    - Error messages
    - Voice messages
    """

    # Platform name normalization map
    PLATFORM_ALIASES = {
        "gchat": "google_chat",
        "googlechat": "google_chat",
        "ms_teams": "teams",
        "msteams": "teams",
        "wa": "whatsapp",
    }

    def __init__(self, registry: Optional["DockRegistry"] = None):
        """
        Initialize the router.

        Args:
            registry: DockRegistry to use, or None to use global registry
        """
        self._registry = registry

    @property
    def registry(self) -> "DockRegistry":
        """Get the dock registry (lazily loaded)."""
        if self._registry is None:
            from aragora.channels.registry import get_dock_registry

            self._registry = get_dock_registry()
        return self._registry

    def _normalize_platform(self, platform: str) -> str:
        """Normalize platform name to canonical form."""
        platform = platform.lower().strip()
        return self.PLATFORM_ALIASES.get(platform, platform)

    async def _get_dock(self, platform: str) -> Optional["ChannelDock"]:
        """Get an initialized dock for a platform."""
        platform = self._normalize_platform(platform)
        dock = self.registry.get_dock(platform)
        if dock is None:
            logger.warning(f"No dock available for platform: {platform}")
            return None

        if not dock.is_initialized:
            success = await dock.initialize()
            if not success:
                logger.warning(f"Failed to initialize dock for platform: {platform}")
                return None

        return dock

    async def route_result(
        self,
        platform: str,
        channel_id: str,
        result: dict[str, Any],
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        include_voice: bool = False,
        **kwargs: Any,
    ) -> SendResult:
        """
        Route a debate result to a chat platform.

        Args:
            platform: Target platform (slack, telegram, discord, etc.)
            channel_id: Channel/chat/space identifier
            result: Debate result dict with keys like consensus_reached,
                    final_answer, confidence, task, etc.
            thread_id: Optional thread ID for threaded replies
            message_id: Optional message ID to reply to
            include_voice: Whether to also send TTS voice message
            **kwargs: Additional platform-specific options

        Returns:
            SendResult indicating success or failure
        """
        dock = await self._get_dock(platform)
        if dock is None:
            return SendResult.fail(
                error=f"Platform '{platform}' not available",
                platform=platform,
                channel_id=channel_id,
            )

        # Build normalized message from result
        message = self._build_result_message(result)
        message.thread_id = thread_id
        message.reply_to = message_id

        # Merge kwargs for platform-specific needs
        send_kwargs = {**kwargs}
        if thread_id:
            send_kwargs["thread_ts"] = thread_id  # Slack
            send_kwargs["thread_name"] = thread_id  # Google Chat
        if message_id:
            send_kwargs["reply_to_message_id"] = message_id  # Telegram
            send_kwargs["message_reference"] = message_id  # Discord

        # Try using dock's send_result if available
        if hasattr(dock, "send_result"):
            send_result = await dock.send_result(
                channel_id, result, thread_id=thread_id, **send_kwargs
            )
        else:
            send_result = await dock.send_message(channel_id, message, **send_kwargs)

        # Send voice if requested and supported
        if send_result.success and include_voice:
            await self._send_voice_result(dock, channel_id, result, **send_kwargs)

        return send_result

    async def route_receipt(
        self,
        platform: str,
        channel_id: str,
        receipt: Any,
        receipt_url: str,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """
        Route a decision receipt to a chat platform.

        Args:
            platform: Target platform
            channel_id: Channel identifier
            receipt: DecisionReceipt object
            receipt_url: URL to view full receipt
            thread_id: Optional thread ID
            **kwargs: Additional options

        Returns:
            SendResult indicating success or failure
        """
        dock = await self._get_dock(platform)
        if dock is None:
            return SendResult.fail(
                error=f"Platform '{platform}' not available",
                platform=platform,
                channel_id=channel_id,
            )

        # Build receipt message
        message = self._build_receipt_message(receipt, receipt_url)
        message.thread_id = thread_id

        # Try dock's send_receipt if available
        if hasattr(dock, "send_receipt"):
            summary = self._format_receipt_summary(receipt, receipt_url)
            return await dock.send_receipt(
                channel_id, summary, receipt_url, thread_id=thread_id, **kwargs
            )
        else:
            return await dock.send_message(channel_id, message, **kwargs)

    async def route_error(
        self,
        platform: str,
        channel_id: str,
        error_message: str,
        debate_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """
        Route an error message to a chat platform.

        Args:
            platform: Target platform
            channel_id: Channel identifier
            error_message: The error message (can be technical or friendly)
            debate_id: Optional debate ID for reference
            thread_id: Optional thread ID
            message_id: Optional message ID to reply to
            **kwargs: Additional options

        Returns:
            SendResult indicating success or failure
        """
        dock = await self._get_dock(platform)
        if dock is None:
            return SendResult.fail(
                error=f"Platform '{platform}' not available",
                platform=platform,
                channel_id=channel_id,
            )

        # Format friendly error
        friendly_message = self._format_error_for_chat(error_message, debate_id)

        # Build error message
        message = NormalizedMessage(
            content=friendly_message,
            message_type=MessageType.ERROR,
            format=MessageFormat.MARKDOWN,
            title="Notice",
            thread_id=thread_id,
            reply_to=message_id,
        )

        # Try dock's send_error if available
        if hasattr(dock, "send_error"):
            return await dock.send_error(channel_id, friendly_message, **kwargs)
        else:
            return await dock.send_message(channel_id, message, **kwargs)

    async def route_voice(
        self,
        platform: str,
        channel_id: str,
        audio_data: bytes,
        text: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """
        Route a voice message to a chat platform.

        Args:
            platform: Target platform
            channel_id: Channel identifier
            audio_data: Audio bytes (usually OGG/Opus)
            text: Optional caption text
            **kwargs: Additional options

        Returns:
            SendResult indicating success or failure
        """
        dock = await self._get_dock(platform)
        if dock is None:
            return SendResult.fail(
                error=f"Platform '{platform}' not available",
                platform=platform,
                channel_id=channel_id,
            )

        # Check voice capability
        if not dock.supports(ChannelCapability.VOICE):
            return SendResult.fail(
                error=f"Platform '{platform}' does not support voice messages",
                platform=platform,
                channel_id=channel_id,
            )

        return await dock.send_voice(channel_id, audio_data, text, **kwargs)

    def _build_result_message(self, result: dict[str, Any]) -> NormalizedMessage:
        """Build a NormalizedMessage from debate result."""
        consensus = result.get("consensus_reached", False)
        confidence = result.get("confidence", 0)
        answer = result.get("final_answer", "No conclusion reached.")
        task = result.get("task", "")

        # Truncate long answers
        if len(answer) > 2000:
            answer = answer[:2000] + "..."

        # Build content
        status_emoji = "✅" if consensus else "❌"
        confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))

        content = (
            f"**Status:** {status_emoji} {'Consensus Reached' if consensus else 'No Consensus'}\n"
            f"**Confidence:** {confidence_bar} {confidence:.0%}\n\n"
            f"**Conclusion:**\n{answer}"
        )

        message = NormalizedMessage(
            content=content,
            message_type=MessageType.RESULT,
            format=MessageFormat.MARKDOWN,
            title="Debate Complete",
        )

        # Add metadata
        if task:
            message.metadata["topic"] = task[:200]

        return message

    def _build_receipt_message(self, receipt: Any, url: str) -> NormalizedMessage:
        """Build a NormalizedMessage from a receipt."""
        summary = self._format_receipt_summary(receipt, url)

        message = NormalizedMessage(
            content=summary,
            message_type=MessageType.RECEIPT,
            format=MessageFormat.MARKDOWN,
            title="Decision Receipt",
        )

        # Add view button
        message.with_button("View Full Receipt", url)

        return message

    def _format_receipt_summary(self, receipt: Any, url: str) -> str:
        """Create compact receipt summary for chat platforms."""
        emoji_map = {
            "APPROVED": "✅",
            "APPROVED_WITH_CONDITIONS": "⚠️",
            "NEEDS_REVIEW": "🔍",
            "REJECTED": "❌",
        }

        verdict = getattr(receipt, "verdict", "UNKNOWN")
        emoji = emoji_map.get(verdict, "📋")
        confidence = getattr(receipt, "confidence", 0)
        critical = getattr(receipt, "critical_count", 0)
        high = getattr(receipt, "high_count", 0)

        cost_line = ""
        if hasattr(receipt, "cost_usd") and receipt.cost_usd > 0:
            cost_line = f"\n• Cost: ${receipt.cost_usd:.4f}"
            if hasattr(receipt, "budget_limit_usd") and receipt.budget_limit_usd:
                pct = (receipt.cost_usd / receipt.budget_limit_usd) * 100
                cost_line += f" ({pct:.0f}% of budget)"

        return f"""{emoji} **Decision Receipt**
• Verdict: {verdict}
• Confidence: {confidence:.0%}
• Findings: {critical} critical, {high} high{cost_line}
• [View Full Receipt]({url})"""

    def _format_error_for_chat(
        self,
        error: str,
        debate_id: Optional[str] = None,
    ) -> str:
        """Convert technical errors to user-friendly messages."""
        error_map = {
            "rate limit": "Your request is being processed. Results will arrive shortly.",
            "429": "Our AI agents are experiencing high demand. Your request is queued.",
            "timeout": "There was a delay processing your request. Please wait a moment.",
            "timed out": "The analysis is taking longer than expected. Results will be sent when ready.",
            "not found": "We couldn't find that debate. Please start a new one.",
            "404": "The requested resource wasn't found. Please try again.",
            "unauthorized": "Please reconnect the Aragora app to continue.",
            "401": "Authentication required. Please reconnect the Aragora app.",
            "forbidden": "You don't have permission for this action. Please check with your workspace admin.",
            "403": "Access denied. Please verify your permissions.",
            "connection": "We're experiencing connectivity issues. Please try again in a moment.",
            "service unavailable": "Our service is temporarily busy. Please try again shortly.",
            "503": "Service temporarily unavailable. Please retry in a few moments.",
            "internal": "Something went wrong on our end. We're looking into it.",
            "500": "An unexpected error occurred. Please try again.",
            "budget": "This request would exceed your organization's budget limit.",
            "quota": "You've reached your usage quota for this period.",
            "invalid": "The request couldn't be processed. Please check your input.",
        }

        error_lower = error.lower()
        for pattern, friendly_msg in error_map.items():
            if pattern in error_lower:
                if debate_id:
                    return f"{friendly_msg}\n\n_Debate ID: {debate_id}_"
                return friendly_msg

        # Default fallback
        msg = "We encountered an issue processing your request. Please try again."
        if debate_id:
            return f"{msg}\n\n_Debate ID: {debate_id}_"
        return msg

    async def _send_voice_result(
        self,
        dock: "ChannelDock",
        channel_id: str,
        result: dict[str, Any],
        **kwargs: Any,
    ) -> Optional[SendResult]:
        """Send TTS voice version of result if supported."""
        if not dock.supports(ChannelCapability.VOICE):
            return None

        try:
            from aragora.connectors.chat.tts_bridge import get_tts_bridge

            bridge = get_tts_bridge()

            # Create voice summary
            consensus = "reached" if result.get("consensus_reached", False) else "not reached"
            confidence = result.get("confidence", 0)
            answer = result.get("final_answer", "No conclusion available.")

            if len(answer) > 300:
                answer = answer[:300] + ". See full text for details."

            voice_text = (
                f"Debate complete. Consensus was {consensus} "
                f"with {confidence:.0%} confidence. "
                f"Conclusion: {answer}"
            )

            audio_path = await bridge.synthesize_response(voice_text, voice="consensus")
            if audio_path:
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                return await dock.send_voice(channel_id, audio_data, **kwargs)

        except ImportError:
            logger.debug("TTS bridge not available")
        except Exception as e:
            logger.warning(f"TTS synthesis failed: {e}")

        return None


# Global router singleton
_channel_router: Optional[ChannelRouter] = None


def get_channel_router() -> ChannelRouter:
    """
    Get the global channel router singleton.

    Returns:
        ChannelRouter instance
    """
    global _channel_router
    if _channel_router is None:
        _channel_router = ChannelRouter()
    return _channel_router
