"""Telegram sender for debate origin result routing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..models import DebateOrigin
from ..formatting import _format_result_message
from ..voice import _synthesize_voice

logger = logging.getLogger(__name__)


async def _send_telegram_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not configured")
        return False

    chat_id = origin.channel_id
    message = _format_result_message(result, origin)

    try:
        import httpx

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        # Reply to original message if we have it
        if origin.message_id:
            data["reply_to_message_id"] = origin.message_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data)
            if response.is_success:
                logger.info(f"Telegram result sent to {chat_id}")
                return True
            else:
                logger.warning(f"Telegram send failed: {response.status_code}")
                return False

    except (OSError, TimeoutError, httpx.HTTPError) as e:
        logger.error(f"Telegram result send error: {e}")
        return False


async def _send_telegram_receipt(origin: DebateOrigin, summary: str) -> bool:
    """Post receipt summary to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return False

    try:
        import httpx

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": origin.channel_id,
            "text": summary,
            "parse_mode": "Markdown",
        }

        if origin.message_id:
            data["reply_to_message_id"] = origin.message_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data)
            if response.is_success:
                logger.info(f"Telegram receipt posted to {origin.channel_id}")
                return True
            return False

    except (OSError, TimeoutError, httpx.HTTPError) as e:
        logger.error(f"Telegram receipt post error: {e}")
        return False


async def _send_telegram_error(origin: DebateOrigin, message: str) -> bool:
    """Send error message to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return False

    try:
        import httpx

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": origin.channel_id,
            "text": message,
            "parse_mode": "Markdown",
        }

        if origin.message_id:
            data["reply_to_message_id"] = origin.message_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data)
            return response.is_success

    except (OSError, TimeoutError, httpx.HTTPError) as e:
        logger.error(f"Telegram error send failed: {e}")
        return False


async def _send_telegram_voice(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send voice message to Telegram."""
    audio_path = await _synthesize_voice(result, origin)
    if not audio_path:
        return False

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return False

    try:
        import httpx

        url = f"https://api.telegram.org/bot{token}/sendVoice"
        chat_id = origin.channel_id

        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(audio_path, "rb") as audio_file:
                files = {"voice": ("voice.ogg", audio_file, "audio/ogg")}
                data = {"chat_id": chat_id}
                if origin.message_id:
                    data["reply_to_message_id"] = origin.message_id

                response = await client.post(url, data=data, files=files)

                if response.is_success:
                    logger.info(f"Telegram voice sent to {chat_id}")
                    return True
                else:
                    logger.warning(f"Telegram voice send failed: {response.status_code}")
                    return False

    except (OSError, TimeoutError, httpx.HTTPError) as e:
        logger.error(f"Telegram voice send error: {e}")
        return False
    finally:
        # Cleanup temp file
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError as e:
            logger.debug(f"Failed to cleanup temp file: {e}")
