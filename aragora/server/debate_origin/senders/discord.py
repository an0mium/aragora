"""Discord sender for debate origin result routing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..models import DebateOrigin
from ..formatting import _format_result_message
from ..voice import _synthesize_voice

logger = logging.getLogger(__name__)


async def _send_discord_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result to Discord."""
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        logger.debug("DISCORD_BOT_TOKEN not configured")
        return False

    channel_id = origin.channel_id
    message = _format_result_message(result, origin, markdown=True)

    try:
        import httpx

        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
        }
        data: dict[str, Any] = {"content": message}

        # Reply to original message if we have it
        if origin.message_id:
            data["message_reference"] = {"message_id": origin.message_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                logger.info("Discord result sent to %s", channel_id)
                return True
            else:
                logger.warning("Discord send failed: %s", response.status_code)
                return False

    except (OSError, TimeoutError, httpx.HTTPError) as e:
        logger.error("Discord result send error: %s", e)
        return False


async def _send_discord_receipt(origin: DebateOrigin, summary: str) -> bool:
    """Post receipt summary to Discord."""
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        return False

    try:
        import httpx

        url = f"https://discord.com/api/v10/channels/{origin.channel_id}/messages"
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
        }
        data: dict[str, Any] = {"content": summary}

        if origin.message_id:
            data["message_reference"] = {"message_id": origin.message_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                logger.info("Discord receipt posted to %s", origin.channel_id)
                return True
            return False

    except (OSError, TimeoutError, httpx.HTTPError) as e:
        logger.error("Discord receipt post error: %s", e)
        return False


async def _send_discord_error(origin: DebateOrigin, message: str) -> bool:
    """Send error message to Discord."""
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        return False

    try:
        import httpx

        url = f"https://discord.com/api/v10/channels/{origin.channel_id}/messages"
        headers = {
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
        }
        data: dict[str, Any] = {"content": message}

        if origin.message_id:
            data["message_reference"] = {"message_id": origin.message_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            return response.is_success

    except (OSError, TimeoutError, httpx.HTTPError) as e:
        logger.error("Discord error send failed: %s", e)
        return False


async def _send_discord_voice(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send voice message to Discord (as audio attachment)."""
    audio_path = await _synthesize_voice(result, origin)
    if not audio_path:
        return False

    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        return False

    try:
        import httpx

        url = f"https://discord.com/api/v10/channels/{origin.channel_id}/messages"
        headers = {"Authorization": f"Bot {token}"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(audio_path, "rb") as audio_file:
                files = {"file": ("debate_result.ogg", audio_file, "audio/ogg")}
                data = {"content": "Voice summary of the debate result:"}

                if origin.message_id:
                    data["message_reference"] = str({"message_id": origin.message_id})

                response = await client.post(url, headers=headers, data=data, files=files)

                if response.is_success:
                    logger.info("Discord voice sent to %s", origin.channel_id)
                    return True
                else:
                    logger.warning("Discord voice send failed: %s", response.status_code)
                    return False

    except (OSError, TimeoutError, httpx.HTTPError) as e:
        logger.error("Discord voice send error: %s", e)
        return False
    finally:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError as e:
            logger.debug("Failed to cleanup temp file: %s", e)
