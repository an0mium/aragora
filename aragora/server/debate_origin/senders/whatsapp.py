"""WhatsApp sender for debate origin result routing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..models import DebateOrigin
from ..formatting import _format_result_message
from ..voice import _synthesize_voice

logger = logging.getLogger(__name__)


async def _send_whatsapp_result(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send result to WhatsApp."""
    token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")

    if not token or not phone_id:
        logger.warning("WhatsApp credentials not configured")
        return False

    to_number = origin.channel_id
    message = _format_result_message(result, origin, markdown=False)

    try:
        import httpx

        url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {"preview_url": False, "body": message},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=data, headers=headers)
            if response.is_success:
                logger.info(f"WhatsApp result sent to {to_number}")
                return True
            else:
                logger.warning(f"WhatsApp send failed: {response.status_code}")
                return False

    except Exception as e:
        logger.error(f"WhatsApp result send error: {e}")
        return False


async def _send_whatsapp_voice(origin: DebateOrigin, result: dict[str, Any]) -> bool:
    """Send voice message to WhatsApp."""
    audio_path = await _synthesize_voice(result, origin)
    if not audio_path:
        return False

    token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
    if not token or not phone_id:
        return False

    try:
        import httpx

        # WhatsApp requires media to be uploaded first
        upload_url = f"https://graph.facebook.com/v18.0/{phone_id}/media"
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Upload media
            with open(audio_path, "rb") as audio_file:
                files = {"file": ("voice.ogg", audio_file, "audio/ogg")}
                data = {"messaging_product": "whatsapp", "type": "audio"}
                upload_response = await client.post(
                    upload_url, headers=headers, data=data, files=files
                )

                if not upload_response.is_success:
                    logger.warning(f"WhatsApp media upload failed: {upload_response.status_code}")
                    return False

                media_id = upload_response.json().get("id")
                if not media_id:
                    logger.warning("WhatsApp media upload returned no ID")
                    return False

            # Send voice message
            send_url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
            send_data = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": origin.channel_id,
                "type": "audio",
                "audio": {"id": media_id},
            }

            send_response = await client.post(
                send_url, headers={**headers, "Content-Type": "application/json"}, json=send_data
            )

            if send_response.is_success:
                logger.info(f"WhatsApp voice sent to {origin.channel_id}")
                return True
            else:
                logger.warning(f"WhatsApp voice send failed: {send_response.status_code}")
                return False

    except Exception as e:
        logger.error(f"WhatsApp voice send error: {e}")
        return False
    finally:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError as e:
            logger.debug(f"Failed to cleanup temp file: {e}")
