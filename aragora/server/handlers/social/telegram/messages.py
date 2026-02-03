"""
Telegram message formatting and API communication methods.

Provides mixin methods for sending messages, answering callbacks,
inline queries, and voice messages via the Telegram Bot API.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ..telemetry import record_api_call, record_api_latency

logger = logging.getLogger(__name__)

# TTS support
TTS_VOICE_ENABLED = os.environ.get("TELEGRAM_TTS_ENABLED", "false").lower() == "true"


def _tg():
    """Lazy import of the telegram package for patchable attribute access."""
    from aragora.server.handlers.social import telegram as telegram_module

    return telegram_module


class TelegramMessagesMixin:
    """Mixin providing Telegram message sending and API communication methods."""

    async def _send_message_async(
        self,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send a message to Telegram chat."""
        import time

        from aragora.server.http_client_pool import get_http_pool

        tg = _tg()
        if not tg.TELEGRAM_BOT_TOKEN:
            logger.warning("Cannot send message: TELEGRAM_BOT_TOKEN not configured")
            return

        start_time = time.time()
        status = "success"
        try:
            url = f"{tg.TELEGRAM_API_BASE}{tg.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if reply_markup:
                payload["reply_markup"] = reply_markup

            pool = get_http_pool()
            async with pool.get_session("telegram") as client:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=30,
                )
                result = response.json()
                if not result.get("ok"):
                    logger.warning("Telegram API error: %s", result.get("description"))
                    status = "error"
        except Exception as e:
            logger.error("Error sending Telegram message: %s", e)
            status = "error"
        finally:
            latency = time.time() - start_time
            record_api_call("telegram", "sendMessage", status)
            record_api_latency("telegram", "sendMessage", latency)

    async def _answer_callback_async(
        self,
        callback_query_id: str,
        text: str,
        show_alert: bool = False,
    ) -> None:
        """Answer a callback query."""
        import time

        from aragora.server.http_client_pool import get_http_pool

        tg = _tg()
        if not tg.TELEGRAM_BOT_TOKEN:
            return

        start_time = time.time()
        status = "success"
        try:
            url = f"{tg.TELEGRAM_API_BASE}{tg.TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
            payload = {
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": show_alert,
            }

            pool = get_http_pool()
            async with pool.get_session("telegram") as client:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=10,
                )
                result = response.json()
                if not result.get("ok"):
                    logger.warning("Telegram callback answer failed: %s", result.get("description"))
                    status = "error"
        except Exception as e:
            logger.error("Error answering Telegram callback: %s", e)
            status = "error"
        finally:
            latency = time.time() - start_time
            record_api_call("telegram", "answerCallbackQuery", status)
            record_api_latency("telegram", "answerCallbackQuery", latency)

    async def _answer_inline_query_async(
        self,
        inline_query_id: str,
        results: list[dict[str, Any]],
    ) -> None:
        """Answer an inline query."""
        import time

        from aragora.server.http_client_pool import get_http_pool

        tg = _tg()
        if not tg.TELEGRAM_BOT_TOKEN:
            return

        start_time = time.time()
        status = "success"
        try:
            url = f"{tg.TELEGRAM_API_BASE}{tg.TELEGRAM_BOT_TOKEN}/answerInlineQuery"
            payload = {
                "inline_query_id": inline_query_id,
                "results": results,
                "cache_time": 10,
            }

            pool = get_http_pool()
            async with pool.get_session("telegram") as client:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=10,
                )
                result = response.json()
                if not result.get("ok"):
                    logger.warning("Telegram inline answer failed: %s", result.get("description"))
                    status = "error"
        except Exception as e:
            logger.error("Error answering Telegram inline query: %s", e)
            status = "error"
        finally:
            latency = time.time() - start_time
            record_api_call("telegram", "answerInlineQuery", status)
            record_api_latency("telegram", "answerInlineQuery", latency)

    async def _send_voice_summary(
        self,
        chat_id: int,
        topic: str,
        final_answer: str | None,
        consensus_reached: bool,
        confidence: float,
        rounds_used: int,
    ) -> None:
        """Send a voice summary of the debate result."""
        try:
            from ..tts_helper import get_tts_helper

            helper = get_tts_helper()
            if not helper.is_available:
                logger.debug("TTS not available for voice summary")
                return

            result = await helper.synthesize_debate_result(
                task=topic,
                final_answer=final_answer,
                consensus_reached=consensus_reached,
                confidence=confidence,
                rounds_used=rounds_used,
            )

            if result:
                await self._send_voice_async(
                    chat_id,
                    result.audio_bytes,
                    result.duration_seconds,
                )
        except Exception as e:
            logger.warning("Failed to send voice summary: %s", e)

    async def _send_voice_async(
        self,
        chat_id: int,
        audio_bytes: bytes,
        duration: float,
    ) -> None:
        """Send a voice message to Telegram chat."""
        from aragora.server.http_client_pool import get_http_pool

        tg = _tg()
        if not tg.TELEGRAM_BOT_TOKEN:
            logger.warning("Cannot send voice: TELEGRAM_BOT_TOKEN not configured")
            return

        try:
            url = f"{tg.TELEGRAM_API_BASE}{tg.TELEGRAM_BOT_TOKEN}/sendVoice"

            # Use httpx multipart file upload
            files = {
                "voice": ("voice.ogg", audio_bytes, "audio/ogg"),
            }
            data = {
                "chat_id": str(chat_id),
                "duration": str(int(duration)),
            }

            pool = get_http_pool()
            async with pool.get_session("telegram_voice") as client:
                response = await client.post(
                    url,
                    files=files,
                    data=data,
                    timeout=60,
                )
                result = response.json()
                if not result.get("ok"):
                    logger.warning("Telegram sendVoice failed: %s", result.get("description"))
                else:
                    logger.info("Voice message sent to chat %s", chat_id)
        except Exception as e:
            logger.error("Error sending Telegram voice: %s", e)
