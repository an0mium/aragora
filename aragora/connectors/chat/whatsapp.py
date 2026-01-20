"""
WhatsApp Business API Connector.

Implements ChatPlatformConnector for WhatsApp using the Cloud API.

Environment Variables:
- WHATSAPP_ACCESS_TOKEN: Permanent access token from Meta
- WHATSAPP_PHONE_NUMBER_ID: Phone number ID from Meta
- WHATSAPP_BUSINESS_ACCOUNT_ID: Business account ID
- WHATSAPP_VERIFY_TOKEN: Webhook verification token
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .base import ChatPlatformConnector
from .models import (
    BotCommand,
    ChatChannel,
    ChatEvidence,
    ChatMessage,
    ChatUser,
    FileAttachment,
    InteractionType,
    MessageButton,
    MessageType,
    SendMessageResponse,
    UserInteraction,
    VoiceMessage,
    WebhookEvent,
)

# Environment configuration
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_BUSINESS_ACCOUNT_ID = os.environ.get("WHATSAPP_BUSINESS_ACCOUNT_ID", "")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")

# WhatsApp Cloud API
WHATSAPP_API_BASE = "https://graph.facebook.com/v18.0"


class WhatsAppConnector(ChatPlatformConnector):
    """
    WhatsApp connector using Meta Cloud API.

    Supports:
    - Sending text messages
    - Interactive messages (buttons, lists)
    - Media messages (images, documents, audio)
    - Message templates
    - Reply messages (context)
    - Webhook handling
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        business_account_id: Optional[str] = None,
        verify_token: Optional[str] = None,
        app_secret: Optional[str] = None,
        **config: Any,
    ):
        """
        Initialize WhatsApp connector.

        Args:
            access_token: Meta Cloud API access token
            phone_number_id: WhatsApp Business phone number ID
            business_account_id: WhatsApp Business account ID
            verify_token: Webhook verification token
            app_secret: App secret for webhook signature verification
            **config: Additional configuration
        """
        super().__init__(
            bot_token=access_token or WHATSAPP_ACCESS_TOKEN,
            signing_secret=app_secret or WHATSAPP_APP_SECRET,
            **config,
        )
        self.phone_number_id = phone_number_id or WHATSAPP_PHONE_NUMBER_ID
        self.business_account_id = business_account_id or WHATSAPP_BUSINESS_ACCOUNT_ID
        self.verify_token = verify_token or WHATSAPP_VERIFY_TOKEN

    @property
    def platform_name(self) -> str:
        return "whatsapp"

    @property
    def platform_display_name(self) -> str:
        return "WhatsApp"

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a message to a WhatsApp user."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        # Build message payload
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": channel_id,  # Phone number
        }

        # Add context for replies
        if thread_id:
            payload["context"] = {"message_id": thread_id}

        # Check if interactive message (has buttons)
        if blocks and any(b.get("type") in ("button", "list") for b in blocks):
            payload["type"] = "interactive"
            payload["interactive"] = self._build_interactive(text, blocks)
        else:
            payload["type"] = "text"
            payload["text"] = {"body": text, "preview_url": True}

        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{WHATSAPP_API_BASE}/{self.phone_number_id}/messages",
                json=payload,
                headers=headers,
            )
            data = response.json()

            if "error" in data:
                error = data["error"]
                logger.error(f"WhatsApp send failed: {error.get('message')}")
                return SendMessageResponse(
                    success=False,
                    error=error.get("message", "Unknown error"),
                )

            messages = data.get("messages", [{}])
            return SendMessageResponse(
                success=True,
                message_id=messages[0].get("id") if messages else None,
                channel_id=channel_id,
                timestamp=datetime.now(),
            )

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Update a message.

        Note: WhatsApp doesn't support editing messages.
        This sends a new message instead.
        """
        logger.warning("WhatsApp doesn't support message editing, sending new message")
        return await self.send_message(
            channel_id,
            text,
            blocks,
            thread_id=message_id,  # Reply to original
        )

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a message.

        Note: WhatsApp doesn't support deleting messages via API.
        """
        logger.warning("WhatsApp doesn't support message deletion via API")
        return False

    async def upload_file(
        self,
        channel_id: str,
        file_path: str,
        filename: Optional[str] = None,
        comment: Optional[str] = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload and send a file as a document."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        # First, upload media
        media_id = await self._upload_media(file_path, "document")

        # Then send message with media
        payload = {
            "messaging_product": "whatsapp",
            "to": channel_id,
            "type": "document",
            "document": {
                "id": media_id,
                "filename": filename or file_path.split("/")[-1],
            },
        }

        if comment:
            payload["document"]["caption"] = comment

        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{WHATSAPP_API_BASE}/{self.phone_number_id}/messages",
                json=payload,
                headers=headers,
            )
            data = response.json()

            if "error" in data:
                raise RuntimeError(data["error"].get("message", "Upload failed"))

            return FileAttachment(
                file_id=media_id,
                filename=filename or file_path.split("/")[-1],
            )

    async def _upload_media(self, file_path: str, media_type: str) -> str:
        """Upload media to WhatsApp servers."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(file_path)

        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (file_path.split("/")[-1], f, mime_type)}
                data = {
                    "messaging_product": "whatsapp",
                    "type": mime_type or "application/octet-stream",
                }
                headers = {"Authorization": f"Bearer {self.bot_token}"}

                response = await client.post(
                    f"{WHATSAPP_API_BASE}/{self.phone_number_id}/media",
                    data=data,
                    files=files,
                    headers=headers,
                )
                result = response.json()

                if "error" in result:
                    raise RuntimeError(result["error"].get("message", "Media upload failed"))

                return result.get("id", "")

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> bytes:
        """Download a file by media ID."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        headers = {"Authorization": f"Bearer {self.bot_token}"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Get media URL
            response = await client.get(
                f"{WHATSAPP_API_BASE}/{file_id}",
                headers=headers,
            )
            data = response.json()

            if "error" in data:
                raise RuntimeError(data["error"].get("message", "Failed to get media"))

            media_url = data.get("url")
            if not media_url:
                raise RuntimeError("No media URL returned")

            # Download file
            response = await client.get(media_url, headers=headers)
            return response.content

    async def handle_webhook(
        self,
        payload: dict[str, Any],
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> WebhookEvent:
        """Process incoming WhatsApp webhook."""
        # Verify signature if app secret is set
        if self.signing_secret and headers:
            signature = headers.get("x-hub-signature-256", "")
            if not self._verify_signature(payload, signature):
                logger.warning("Invalid webhook signature")
                return WebhookEvent(
                    event_type="invalid_signature",
                    platform="whatsapp",
                    raw_payload=payload,
                )

        # Parse webhook structure
        entry = payload.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        # Handle message
        messages = value.get("messages", [])
        if messages:
            msg = messages[0]
            contact = value.get("contacts", [{}])[0]
            return WebhookEvent(
                event_type="message",
                platform="whatsapp",
                channel_id=msg.get("from"),
                user_id=msg.get("from"),
                message_id=msg.get("id"),
                timestamp=datetime.fromtimestamp(int(msg.get("timestamp", 0))),
                raw_payload=payload,
            )

        # Handle status updates
        statuses = value.get("statuses", [])
        if statuses:
            status = statuses[0]
            return WebhookEvent(
                event_type=f"status_{status.get('status')}",
                platform="whatsapp",
                message_id=status.get("id"),
                timestamp=datetime.fromtimestamp(int(status.get("timestamp", 0))),
                raw_payload=payload,
            )

        return WebhookEvent(
            event_type="unknown",
            platform="whatsapp",
            raw_payload=payload,
        )

    def _verify_signature(self, payload: dict, signature: str) -> bool:
        """Verify webhook signature."""
        if not signature.startswith("sha256="):
            return False

        expected_sig = signature[7:]
        body = json.dumps(payload, separators=(",", ":"))
        computed_sig = hmac.new(
            self.signing_secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected_sig, computed_sig)

    async def parse_message(
        self,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> ChatMessage:
        """Parse a WhatsApp message into ChatMessage."""
        entry = payload.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        msg = value.get("messages", [{}])[0]
        contact = value.get("contacts", [{}])[0]

        # Determine message type
        msg_type = MessageType.TEXT
        msg_type_str = msg.get("type", "text")
        if msg_type_str == "audio":
            msg_type = MessageType.VOICE
        elif msg_type_str == "document":
            msg_type = MessageType.FILE
        elif msg_type_str == "image":
            msg_type = MessageType.IMAGE

        # Extract text content
        text = ""
        if msg_type_str == "text":
            text = msg.get("text", {}).get("body", "")
        elif msg_type_str == "interactive":
            interactive = msg.get("interactive", {})
            if "button_reply" in interactive:
                text = interactive["button_reply"].get("title", "")
            elif "list_reply" in interactive:
                text = interactive["list_reply"].get("title", "")

        return ChatMessage(
            message_id=msg.get("id", ""),
            channel_id=msg.get("from", ""),
            user=ChatUser(
                user_id=msg.get("from", ""),
                username=contact.get("wa_id", ""),
                display_name=contact.get("profile", {}).get("name", ""),
                platform="whatsapp",
            ),
            text=text,
            timestamp=datetime.fromtimestamp(int(msg.get("timestamp", 0))),
            message_type=msg_type,
            platform="whatsapp",
            raw_data=msg,
            thread_id=msg.get("context", {}).get("id"),
        )

    async def parse_command(
        self,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> Optional[BotCommand]:
        """
        Parse a command from message.

        Note: WhatsApp doesn't have native commands.
        This looks for messages starting with / or !
        """
        message = await self.parse_message(payload)
        text = message.text

        if not text or not (text.startswith("/") or text.startswith("!")):
            return None

        prefix = text[0]
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        return BotCommand(
            command=command,
            args=args,
            user_id=message.user.user_id,
            channel_id=message.channel_id,
            message_id=message.message_id,
            platform="whatsapp",
            raw_data=message.raw_data,
        )

    async def handle_interaction(
        self,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> UserInteraction:
        """Handle button click or list selection."""
        message = await self.parse_message(payload)
        msg = message.raw_data or {}

        interactive = msg.get("interactive", {})
        interaction_type = InteractionType.BUTTON_CLICK

        if "button_reply" in interactive:
            action_id = interactive["button_reply"].get("id", "")
            action_value = interactive["button_reply"].get("title", "")
        elif "list_reply" in interactive:
            action_id = interactive["list_reply"].get("id", "")
            action_value = interactive["list_reply"].get("title", "")
            interaction_type = InteractionType.MENU_SELECT
        else:
            action_id = ""
            action_value = ""

        return UserInteraction(
            interaction_type=interaction_type,
            user_id=message.user.user_id,
            channel_id=message.channel_id,
            message_id=message.message_id,
            action_id=action_id,
            action_value=action_value,
            platform="whatsapp",
            raw_data=msg,
        )

    async def send_voice_message(
        self,
        channel_id: str,
        audio_data: bytes,
        duration: Optional[int] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a voice message."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        # Upload audio first
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": ("voice.ogg", audio_data, "audio/ogg")}
            data = {"messaging_product": "whatsapp", "type": "audio/ogg"}
            headers = {"Authorization": f"Bearer {self.bot_token}"}

            response = await client.post(
                f"{WHATSAPP_API_BASE}/{self.phone_number_id}/media",
                data=data,
                files=files,
                headers=headers,
            )
            result = response.json()

            if "error" in result:
                return SendMessageResponse(
                    success=False,
                    error=result["error"].get("message", "Audio upload failed"),
                )

            media_id = result.get("id")

            # Send audio message
            payload = {
                "messaging_product": "whatsapp",
                "to": channel_id,
                "type": "audio",
                "audio": {"id": media_id},
            }

            response = await client.post(
                f"{WHATSAPP_API_BASE}/{self.phone_number_id}/messages",
                json=payload,
                headers={"Authorization": f"Bearer {self.bot_token}", "Content-Type": "application/json"},
            )
            data = response.json()

            if "error" in data:
                return SendMessageResponse(
                    success=False,
                    error=data["error"].get("message", "Voice send failed"),
                )

            messages = data.get("messages", [{}])
            return SendMessageResponse(
                success=True,
                message_id=messages[0].get("id") if messages else None,
                channel_id=channel_id,
                timestamp=datetime.now(),
            )

    async def download_voice_message(
        self,
        voice_message: VoiceMessage,
        **kwargs: Any,
    ) -> bytes:
        """Download a voice message."""
        return await self.download_file(voice_message.file_id)

    async def get_channel_info(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> ChatChannel:
        """
        Get channel info.

        Note: WhatsApp is 1:1 messaging, so channel = phone number.
        """
        return ChatChannel(
            channel_id=channel_id,
            name=channel_id,  # Phone number
            channel_type="private",
            platform="whatsapp",
        )

    async def get_user_info(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> ChatUser:
        """
        Get user info.

        Note: WhatsApp provides limited user info.
        """
        return ChatUser(
            user_id=user_id,
            username=user_id,  # Phone number
            platform="whatsapp",
        )

    async def extract_evidence(
        self,
        message: ChatMessage,
        **kwargs: Any,
    ) -> ChatEvidence:
        """Extract evidence from a message for debate."""
        return ChatEvidence(
            content=message.text,
            source_url=f"whatsapp:{message.channel_id}/{message.message_id}",
            author=message.user.display_name or message.channel_id,
            timestamp=message.timestamp,
            platform="whatsapp",
            channel_id=message.channel_id,
            message_id=message.message_id,
            metadata={"raw": message.raw_data},
        )

    def _build_interactive(self, text: str, blocks: list[dict]) -> dict:
        """Build interactive message payload."""
        buttons = []
        list_items = []

        for block in blocks:
            if block.get("type") == "button":
                buttons.append({
                    "type": "reply",
                    "reply": {
                        "id": block.get("action_id", block.get("value", "")),
                        "title": block.get("text", "")[:20],  # Max 20 chars
                    },
                })
            elif block.get("type") == "list_item":
                list_items.append({
                    "id": block.get("action_id", block.get("value", "")),
                    "title": block.get("text", "")[:24],  # Max 24 chars
                    "description": block.get("description", "")[:72],  # Max 72 chars
                })

        if list_items:
            return {
                "type": "list",
                "header": {"type": "text", "text": "Options"},
                "body": {"text": text},
                "action": {
                    "button": "Select",
                    "sections": [{"title": "Options", "rows": list_items[:10]}],  # Max 10
                },
            }
        elif buttons:
            return {
                "type": "button",
                "body": {"text": text},
                "action": {"buttons": buttons[:3]},  # Max 3 buttons
            }
        else:
            return {"type": "button", "body": {"text": text}, "action": {"buttons": []}}

    async def send_template(
        self,
        channel_id: str,
        template_name: str,
        language_code: str = "en",
        components: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Send a template message.

        Templates must be pre-approved by WhatsApp.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        payload = {
            "messaging_product": "whatsapp",
            "to": channel_id,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code},
            },
        }

        if components:
            payload["template"]["components"] = components

        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{WHATSAPP_API_BASE}/{self.phone_number_id}/messages",
                json=payload,
                headers=headers,
            )
            data = response.json()

            if "error" in data:
                return SendMessageResponse(
                    success=False,
                    error=data["error"].get("message", "Template send failed"),
                )

            messages = data.get("messages", [{}])
            return SendMessageResponse(
                success=True,
                message_id=messages[0].get("id") if messages else None,
                channel_id=channel_id,
                timestamp=datetime.now(),
            )

    async def verify_webhook(
        self,
        mode: str,
        token: str,
        challenge: str,
    ) -> Optional[str]:
        """
        Verify webhook subscription (GET request).

        Returns challenge if verification succeeds, None otherwise.
        """
        if mode == "subscribe" and token == self.verify_token:
            logger.info("WhatsApp webhook verified")
            return challenge
        logger.warning("WhatsApp webhook verification failed")
        return None


__all__ = ["WhatsAppConnector"]
