"""
WhatsApp Cloud API message sending utilities.

Handles sending:
- Text messages
- Interactive button messages
- Voice/audio messages (with media upload)
"""

from __future__ import annotations

import logging
from typing import Any

from .config import (
    WHATSAPP_ACCESS_TOKEN,
    WHATSAPP_API_BASE,
    WHATSAPP_PHONE_NUMBER_ID,
)
from ..telemetry import (
    record_api_call,
    record_api_latency,
)

logger = logging.getLogger(__name__)


async def send_text_message(to_number: str, text: str) -> None:
    """Send a text message via WhatsApp Cloud API."""
    import time

    from aragora.server.http_client_pool import get_http_pool

    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.warning("Cannot send message: WhatsApp not configured")
        return

    start_time = time.time()
    status = "success"
    try:
        url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }

        pool = get_http_pool()
        async with pool.get_session("whatsapp") as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            if response.status_code != 200:
                result = response.json()
                logger.warning("WhatsApp API error: %s", result)
                status = "error"
    except (OSError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
        logger.error("Error sending WhatsApp message: %s", e)
        status = "error"
    finally:
        latency = time.time() - start_time
        record_api_call("whatsapp", "sendMessage", status)
        record_api_latency("whatsapp", "sendMessage", latency)


async def send_interactive_buttons(
    to_number: str,
    body_text: str,
    buttons: list[dict[str, str]],
    header_text: str | None = None,
) -> None:
    """Send an interactive buttons message.

    buttons: List of dicts with 'id' and 'title' keys (max 3 buttons)
    """
    import time

    from aragora.server.http_client_pool import get_http_pool

    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.warning("Cannot send message: WhatsApp not configured")
        return

    start_time = time.time()
    status = "success"
    try:
        url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"

        # WhatsApp allows max 3 buttons
        button_list = [
            {"type": "reply", "reply": {"id": b["id"], "title": b["title"][:20]}}
            for b in buttons[:3]
        ]

        interactive: dict[str, Any] = {
            "type": "button",
            "body": {"text": body_text[:1024]},  # Max 1024 chars
            "action": {"buttons": button_list},
        }

        if header_text:
            interactive["header"] = {"type": "text", "text": header_text[:60]}

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "interactive",
            "interactive": interactive,
        }

        pool = get_http_pool()
        async with pool.get_session("whatsapp") as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            if response.status_code != 200:
                result = response.json()
                logger.warning("WhatsApp API error: %s", result)
                status = "error"
                # Fall back to plain text if interactive fails
                await send_text_message(to_number, body_text)
    except (OSError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
        logger.error("Error sending WhatsApp interactive message: %s", e)
        status = "error"
        # Fall back to plain text
        await send_text_message(to_number, body_text)
    finally:
        latency = time.time() - start_time
        record_api_call("whatsapp", "sendInteractive", status)
        record_api_latency("whatsapp", "sendInteractive", latency)


async def send_voice_summary(
    to_number: str,
    topic: str,
    final_answer: str | None,
    consensus_reached: bool,
    confidence: float,
    rounds_used: int,
) -> None:
    """Send a voice summary of the debate result.

    Uses TTS to synthesize the result and sends as audio message.
    """
    try:
        from aragora.server.handlers.social.tts_helper import get_tts_helper

        helper = get_tts_helper()
        if not helper.is_available:
            logger.debug("TTS not available for WhatsApp voice summary")
            return

        result = await helper.synthesize_debate_result(
            task=topic,
            final_answer=final_answer,
            consensus_reached=consensus_reached,
            confidence=confidence,
            rounds_used=rounds_used,
        )

        if result:
            await send_voice_message(
                to_number,
                result.audio_bytes,
                result.format,
            )

    except ImportError:
        logger.debug("TTS helper not available")
    except (OSError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
        logger.warning("Failed to send WhatsApp voice summary: %s", e)


async def send_voice_message(
    to_number: str,
    audio_bytes: bytes,
    audio_format: str = "mp3",
) -> None:
    """Send an audio message via WhatsApp Cloud API.

    WhatsApp requires uploading media first, then sending with media ID.
    Note: Voice upload uses httpx's file upload capabilities.
    """
    from aragora.server.http_client_pool import get_http_pool

    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        logger.warning("Cannot send voice: WhatsApp not configured")
        return

    try:
        # Step 1: Upload audio to WhatsApp Media API
        upload_url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/media"

        # Determine MIME type
        mime_types = {
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "wav": "audio/wav",
            "m4a": "audio/mp4",
        }
        mime_type = mime_types.get(audio_format, "audio/mpeg")

        pool = get_http_pool()
        async with pool.get_session("whatsapp_media") as client:
            # Upload the audio file using httpx multipart
            files = {
                "file": (f"voice.{audio_format}", audio_bytes, mime_type),
            }
            data = {
                "messaging_product": "whatsapp",
                "type": mime_type,
            }
            upload_response = await client.post(
                upload_url,
                files=files,
                data=data,
                headers={
                    "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                },
                timeout=60,
            )
            upload_result = upload_response.json()

            if upload_response.status_code != 200:
                logger.warning("WhatsApp media upload failed: %s", upload_result)
                return

            media_id = upload_result.get("id")
            if not media_id:
                logger.warning("No media ID returned from upload")
                return

            # Step 2: Send audio message with media ID
            send_url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to_number,
                "type": "audio",
                "audio": {"id": media_id},
            }

            send_response = await client.post(
                send_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            if send_response.status_code != 200:
                send_result = send_response.json()
                logger.warning("WhatsApp audio send failed: %s", send_result)
            else:
                logger.info("WhatsApp voice message sent to %s", to_number)

    except (OSError, ConnectionError, TimeoutError, ValueError, RuntimeError) as e:
        logger.error("Error sending WhatsApp voice message: %s", e)
