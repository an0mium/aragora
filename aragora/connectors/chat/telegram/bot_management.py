"""
Telegram Bot Connector - Bot Management.

Contains bot command setup, info retrieval, and chat member count.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TelegramBotManagementMixin:
    """Mixin providing bot management operations for TelegramConnector."""

    async def _telegram_api_request(
        self,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        operation: str = "api_call",
        **kwargs: Any,
    ) -> tuple[bool, dict[str, Any] | None, str | None]: ...

    async def set_my_commands(
        self,
        commands: list[dict[str, str]],
        scope: dict[str, Any] | None = None,
        language_code: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Set the list of the bot's commands.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            commands: List of command dicts with 'command' and 'description' keys
            scope: Scope for which commands apply (all users, specific chat, etc.)
            language_code: Language code for commands
            **kwargs: Additional parameters

        Returns:
            True if successful
        """
        payload: dict[str, Any] = {
            "commands": json.dumps(commands),
        }

        if scope:
            payload["scope"] = json.dumps(scope)
        if language_code:
            payload["language_code"] = language_code

        success, data, error = await self._telegram_api_request(
            "setMyCommands",
            payload=payload,
            operation="set_my_commands",
        )

        if not success:
            logger.error(f"Failed to set commands: {error}")

        return success

    async def get_me(self) -> dict[str, Any] | None:
        """Get basic information about the bot.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Returns:
            Bot user object with id, username, first_name, etc.
        """
        success, data, error = await self._telegram_api_request(
            "getMe",
            method="GET",
            operation="get_me",
        )

        if success and data:
            bot_info: dict[str, Any] | None = data.get("result")
            return bot_info

        logger.error(f"Failed to get bot info: {error}")
        return None

    async def get_chat_member_count(self, channel_id: str) -> int | None:
        """Get the number of members in a chat.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            channel_id: Target chat ID

        Returns:
            Number of members or None if failed
        """
        success, data, error = await self._telegram_api_request(
            "getChatMemberCount",
            payload={"chat_id": channel_id},
            operation="get_chat_member_count",
        )

        if success and data:
            count: int | None = data.get("result")
            return count

        logger.error(f"Failed to get chat member count: {error}")
        return None
