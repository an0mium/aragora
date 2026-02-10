"""
Telegram Bot Connector - Inline Query Support.

Contains inline query answering and result building functionality.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TelegramInlineMixin:
    """Mixin providing inline query support for TelegramConnector."""

    async def answer_inline_query(
        self,
        inline_query_id: str,
        results: list[dict[str, Any]],
        cache_time: int = 300,
        is_personal: bool = False,
        next_offset: str | None = None,
        button: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> bool:
        """Answer an inline query with results.

        Uses _telegram_api_request for circuit breaker, retry, and timeout handling.

        Args:
            inline_query_id: Unique identifier for the inline query
            results: List of InlineQueryResult objects. Each should have:
                - type: "article", "photo", "video", etc.
                - id: Unique identifier
                - title: Title of the result
                - input_message_content: Content to send when selected
            cache_time: Time in seconds to cache results (default 300)
            is_personal: Whether results are personalized per user
            next_offset: Offset for pagination
            button: Button to show above results
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        payload: dict[str, Any] = {
            "inline_query_id": inline_query_id,
            "results": json.dumps(results),
            "cache_time": cache_time,
            "is_personal": is_personal,
        }

        if next_offset:
            payload["next_offset"] = next_offset
        if button:
            payload["button"] = json.dumps(button)

        success, data, error = await self._telegram_api_request(
            "answerInlineQuery",
            payload=payload,
            operation="answer_inline_query",
        )

        if not success:
            logger.error(f"Failed to answer inline query: {error}")

        return success

    def build_inline_article_result(
        self,
        result_id: str,
        title: str,
        message_text: str,
        description: str | None = None,
        url: str | None = None,
        thumb_url: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build an InlineQueryResultArticle for use with answer_inline_query.

        Args:
            result_id: Unique identifier for this result
            title: Title of the result
            message_text: Text to send when selected
            description: Short description
            url: URL to associate with the result
            thumb_url: Thumbnail URL
            **kwargs: Additional fields

        Returns:
            Dict formatted as InlineQueryResultArticle
        """
        result: dict[str, Any] = {
            "type": "article",
            "id": result_id,
            "title": title,
            "input_message_content": {
                "message_text": message_text,
                "parse_mode": self.parse_mode,
            },
        }

        if description:
            result["description"] = description
        if url:
            result["url"] = url
        if thumb_url:
            result["thumb_url"] = thumb_url

        result.update(kwargs)
        return result
