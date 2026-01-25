"""
WhatsApp Business API Connector.

Implements ChatPlatformConnector for WhatsApp using the Cloud API.
Includes circuit breaker protection for resilient API interactions.

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

    Includes circuit breaker protection for resilient API interactions.

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
        blocks: Optional[list[dict[str, Any]]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a message to a WhatsApp user with circuit breaker protection."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(
                success=False,
                error=cb_error or "Circuit breaker is open",
            )

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

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.post(
                    f"{WHATSAPP_API_BASE}/{self.phone_number_id}/messages",
                    json=payload,
                    headers=headers,
                )
                data = response.json()

                if "error" in data:
                    error = data["error"]
                    error_code = error.get("code", 0)
                    # Record failure for rate limit errors
                    if error_code in (4, 80007, 130429):  # Rate limit codes
                        self._record_failure(Exception(f"Rate limited: {error.get('message')}"))
                    logger.error(f"WhatsApp send failed: {error.get('message')}")
                    return SendMessageResponse(
                        success=False,
                        error=error.get("message", "Unknown error"),
                    )

                self._record_success()
                messages = data.get("messages", [{}])
                return SendMessageResponse(
                    success=True,
                    message_id=messages[0].get("id") if messages else None,
                    channel_id=channel_id,
                    timestamp=datetime.now().isoformat(),
                )
        except Exception as e:
            self._record_failure(e)
            logger.error(f"WhatsApp send exception: {e}")
            return SendMessageResponse(
                success=False,
                error=str(e),
            )

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Update a message with circuit breaker protection.

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

    async def upload_file(  # type: ignore[override]
        self,
        channel_id: str,
        file_path: str,
        filename: Optional[str] = None,
        comment: Optional[str] = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload and send a file as a document with circuit breaker protection."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            raise RuntimeError(cb_error or "Circuit breaker is open")

        try:
            # First, upload media
            media_id = await self._upload_media(file_path, "document")

            # Then send message with media
            doc_payload: dict[str, Any] = {
                "id": media_id,
                "filename": filename or file_path.split("/")[-1],
            }
            if comment:
                doc_payload["caption"] = comment

            payload: dict[str, Any] = {
                "messaging_product": "whatsapp",
                "to": channel_id,
                "type": "document",
                "document": doc_payload,
            }

            headers = {
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.post(
                    f"{WHATSAPP_API_BASE}/{self.phone_number_id}/messages",
                    json=payload,
                    headers=headers,
                )
                data = response.json()

                if "error" in data:
                    error = data["error"]
                    error_code = error.get("code", 0)
                    if error_code in (4, 80007, 130429):  # Rate limit codes
                        self._record_failure(Exception(f"Rate limited: {error.get('message')}"))
                    raise RuntimeError(error.get("message", "Upload failed"))

                self._record_success()
                return FileAttachment(
                    id=media_id,
                    filename=filename or file_path.split("/")[-1],
                    content_type="application/octet-stream",
                    size=0,  # Size unknown after upload
                )
        except Exception as e:
            self._record_failure(e)
            raise

    async def _upload_media(self, file_path: str, media_type: str) -> str:
        """Upload media to WhatsApp servers with circuit breaker protection."""
        import mimetypes

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            raise RuntimeError(cb_error or "Circuit breaker is open")

        mime_type, _ = mimetypes.guess_type(file_path)

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
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
                        error = result["error"]
                        error_code = error.get("code", 0)
                        if error_code in (4, 80007, 130429):  # Rate limit codes
                            self._record_failure(Exception(f"Rate limited: {error.get('message')}"))
                        raise RuntimeError(error.get("message", "Media upload failed"))

                    self._record_success()
                    media_id: str = result.get("id", "")
                    return media_id
        except Exception as e:
            self._record_failure(e)
            raise

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        """Download a file by media ID with circuit breaker protection.

        Args:
            file_id: WhatsApp media ID to download
            **kwargs: Additional options (url, filename for hints)

        Returns:
            FileAttachment with content populated
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            raise RuntimeError(cb_error or "Circuit breaker is open")

        headers = {"Authorization": f"Bearer {self.bot_token}"}

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                # Get media URL and metadata
                response = await client.get(
                    f"{WHATSAPP_API_BASE}/{file_id}",
                    headers=headers,
                )
                data = response.json()

                if "error" in data:
                    error = data["error"]
                    error_code = error.get("code", 0)
                    if error_code in (4, 80007, 130429):  # Rate limit codes
                        self._record_failure(Exception(f"Rate limited: {error.get('message')}"))
                    raise RuntimeError(error.get("message", "Failed to get media"))

                media_url = data.get("url")
                if not media_url:
                    raise RuntimeError("No media URL returned")

                # Extract metadata from the response
                mime_type = data.get("mime_type", "application/octet-stream")
                file_size = data.get("file_size", 0)

                # Download file
                response = await client.get(media_url, headers=headers)
                content = response.content
                self._record_success()

                # Use filename hint or generate from mime type
                filename = kwargs.get("filename")
                if not filename:
                    ext = ".ogg"  # Default for voice
                    if "audio/ogg" in mime_type:
                        ext = ".ogg"
                    elif "audio/mpeg" in mime_type or "audio/mp3" in mime_type:
                        ext = ".mp3"
                    elif "audio/aac" in mime_type or "audio/mp4" in mime_type:
                        ext = ".m4a"
                    elif "audio/wav" in mime_type:
                        ext = ".wav"
                    filename = f"audio_{file_id[:8]}{ext}"

                return FileAttachment(
                    id=file_id,
                    filename=filename,
                    content_type=mime_type,
                    size=file_size or len(content),
                    url=media_url,
                    content=content,
                    metadata={"whatsapp_mime_type": mime_type},
                )
        except Exception as e:
            self._record_failure(e)
            raise

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
            value.get("contacts", [{}])[0]
            return WebhookEvent(
                event_type="message",
                platform="whatsapp",
                timestamp=datetime.fromtimestamp(int(msg.get("timestamp", 0))),
                raw_payload=payload,
                metadata={
                    "channel_id": msg.get("from"),
                    "user_id": msg.get("from"),
                    "message_id": msg.get("id"),
                },
            )

        # Handle status updates
        statuses = value.get("statuses", [])
        if statuses:
            status = statuses[0]
            return WebhookEvent(
                event_type=f"status_{status.get('status')}",
                platform="whatsapp",
                timestamp=datetime.fromtimestamp(int(status.get("timestamp", 0))),
                raw_payload=payload,
                metadata={"message_id": status.get("id")},
            )

        return WebhookEvent(
            event_type="unknown",
            platform="whatsapp",
            raw_payload=payload,
        )

    def _verify_signature(self, payload: dict[str, Any], signature: str) -> bool:
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
            msg_type = MessageType.FILE  # Images treated as files

        # Extract text content
        content = ""
        if msg_type_str == "text":
            content = msg.get("text", {}).get("body", "")
        elif msg_type_str == "interactive":
            interactive = msg.get("interactive", {})
            if "button_reply" in interactive:
                content = interactive["button_reply"].get("title", "")
            elif "list_reply" in interactive:
                content = interactive["list_reply"].get("title", "")

        # Build proper ChatChannel and ChatUser objects
        sender_id = msg.get("from", "")
        channel = ChatChannel(
            id=sender_id,
            platform="whatsapp",
            name=contact.get("profile", {}).get("name"),
            is_dm=True,
        )
        author = ChatUser(
            id=sender_id,
            platform="whatsapp",
            username=contact.get("wa_id", ""),
            display_name=contact.get("profile", {}).get("name", ""),
        )

        return ChatMessage(
            id=msg.get("id", ""),
            platform="whatsapp",
            channel=channel,
            author=author,
            content=content,
            message_type=msg_type,
            timestamp=datetime.fromtimestamp(int(msg.get("timestamp", 0))),
            thread_id=msg.get("context", {}).get("id"),
            metadata={"raw_data": msg},
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
        text = message.content

        if not text or not (text.startswith("/") or text.startswith("!")):
            return None

        parts = text[1:].split()
        command_name = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        return BotCommand(
            name=command_name,
            text=text,
            args=args,
            user=message.author,
            channel=message.channel,
            platform="whatsapp",
            metadata={"message_id": message.id},
        )

    async def handle_interaction(
        self,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> UserInteraction:
        """Handle button click or list selection."""
        message = await self.parse_message(payload)
        msg = message.metadata.get("raw_data", {})

        interactive = msg.get("interactive", {})
        interaction_type = InteractionType.BUTTON_CLICK

        if "button_reply" in interactive:
            action_id = interactive["button_reply"].get("id", "")
            action_value = interactive["button_reply"].get("title", "")
        elif "list_reply" in interactive:
            action_id = interactive["list_reply"].get("id", "")
            action_value = interactive["list_reply"].get("title", "")
            interaction_type = InteractionType.SELECT_MENU
        else:
            action_id = ""
            action_value = ""

        return UserInteraction(
            id=f"interaction-{message.id}",
            interaction_type=interaction_type,
            action_id=action_id,
            value=action_value,
            user=message.author,
            channel=message.channel,
            message_id=message.id,
            platform="whatsapp",
        )

    async def send_voice_message(  # type: ignore[override]
        self,
        channel_id: str,
        audio_data: bytes,
        duration: Optional[int] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send a voice message with circuit breaker protection."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(
                success=False,
                error=cb_error or "Circuit breaker is open",
            )

        try:
            # Upload audio first
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
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
                    error = result["error"]
                    error_code = error.get("code", 0)
                    if error_code in (4, 80007, 130429):  # Rate limit codes
                        self._record_failure(Exception(f"Rate limited: {error.get('message')}"))
                    return SendMessageResponse(
                        success=False,
                        error=error.get("message", "Audio upload failed"),
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
                    headers={
                        "Authorization": f"Bearer {self.bot_token}",
                        "Content-Type": "application/json",
                    },
                )
                data = response.json()

                if "error" in data:
                    error = data["error"]
                    error_code = error.get("code", 0)
                    if error_code in (4, 80007, 130429):  # Rate limit codes
                        self._record_failure(Exception(f"Rate limited: {error.get('message')}"))
                    return SendMessageResponse(
                        success=False,
                        error=error.get("message", "Voice send failed"),
                    )

                self._record_success()
                messages = data.get("messages", [{}])
                return SendMessageResponse(
                    success=True,
                    message_id=messages[0].get("id") if messages else None,
                    channel_id=channel_id,
                    timestamp=datetime.now().isoformat(),
                )
        except Exception as e:
            self._record_failure(e)
            logger.error(f"WhatsApp voice message exception: {e}")
            return SendMessageResponse(
                success=False,
                error=str(e),
            )

    async def download_voice_message(
        self,
        voice_message: VoiceMessage,
        **kwargs: Any,
    ) -> bytes:
        """Download a voice message with circuit breaker protection."""
        attachment = await self.download_file(voice_message.file.id)
        return attachment.content or b""

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
            id=channel_id,
            platform="whatsapp",
            name=channel_id,  # Phone number
            is_dm=True,
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
            id=user_id,
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
            id=f"evidence-{message.id}",
            source_id=message.id,
            platform="whatsapp",
            channel_id=message.channel.id,
            content=message.content,
            author_id=message.author.id,
            author_name=message.author.display_name or message.channel.id,
            timestamp=message.timestamp,
            source_message=message,
        )

    def _build_interactive(self, text: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        """Build interactive message payload."""
        buttons = []
        list_items = []

        for block in blocks:
            if block.get("type") == "button":
                buttons.append(
                    {
                        "type": "reply",
                        "reply": {
                            "id": block.get("action_id", block.get("value", "")),
                            "title": block.get("text", "")[:20],  # Max 20 chars
                        },
                    }
                )
            elif block.get("type") == "list_item":
                list_items.append(
                    {
                        "id": block.get("action_id", block.get("value", "")),
                        "title": block.get("text", "")[:24],  # Max 24 chars
                        "description": block.get("description", "")[:72],  # Max 72 chars
                    }
                )

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
        components: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """
        Send a template message with circuit breaker protection.

        Templates must be pre-approved by WhatsApp.
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WhatsApp connector")

        # Check circuit breaker before making request
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(
                success=False,
                error=cb_error or "Circuit breaker is open",
            )

        template_data: dict[str, Any] = {
            "name": template_name,
            "language": {"code": language_code},
        }
        if components:
            template_data["components"] = components

        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": channel_id,
            "type": "template",
            "template": template_data,
        }

        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.post(
                    f"{WHATSAPP_API_BASE}/{self.phone_number_id}/messages",
                    json=payload,
                    headers=headers,
                )
                data = response.json()

                if "error" in data:
                    error = data["error"]
                    error_code = error.get("code", 0)
                    if error_code in (4, 80007, 130429):  # Rate limit codes
                        self._record_failure(Exception(f"Rate limited: {error.get('message')}"))
                    return SendMessageResponse(
                        success=False,
                        error=error.get("message", "Template send failed"),
                    )

                self._record_success()
                messages = data.get("messages", [{}])
                return SendMessageResponse(
                    success=True,
                    message_id=messages[0].get("id") if messages else None,
                    channel_id=channel_id,
                    timestamp=datetime.now().isoformat(),
                )
        except Exception as e:
            self._record_failure(e)
            logger.error(f"WhatsApp template send exception: {e}")
            return SendMessageResponse(
                success=False,
                error=str(e),
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

    # ==========================================================================
    # Abstract method implementations
    # ==========================================================================

    def format_blocks(  # type: ignore[override]
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[dict[str, Any]]] = None,
        buttons: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format content as WhatsApp-compatible blocks."""
        blocks: list[dict[str, Any]] = []

        if title:
            blocks.append({"type": "header", "text": title})

        if body:
            blocks.append({"type": "body", "text": body})

        if fields:
            for field in fields:
                blocks.append(
                    {
                        "type": "field",
                        "label": field.get("label", ""),
                        "value": field.get("value", ""),
                    }
                )

        if buttons:
            for btn in buttons:
                blocks.append(
                    {
                        "type": "button",
                        "text": btn.get("text", "")[:20],  # WhatsApp limits to 20 chars
                        "action_id": btn.get("action_id", ""),
                        "value": btn.get("value", ""),
                    }
                )

        return blocks

    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Format a button for WhatsApp interactive message."""
        if url:
            return {"type": "url_button", "text": text, "url": url}
        return {
            "type": "button",
            "text": text[:20],  # WhatsApp limits to 20 chars
            "action_id": action_id,
            "value": value or action_id,
        }

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """Parse a WhatsApp webhook payload into a WebhookEvent."""
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return WebhookEvent(
                platform="whatsapp",
                event_type="error",
                raw_payload={},
            )

        # Verify signature if app secret is set
        if self.signing_secret:
            signature = headers.get("X-Hub-Signature-256", "")
            expected = (
                "sha256="
                + hmac.new(
                    self.signing_secret.encode(),
                    body,
                    hashlib.sha256,
                ).hexdigest()
            )
            if not hmac.compare_digest(signature, expected):
                logger.warning("WhatsApp webhook signature mismatch")
                return WebhookEvent(
                    platform="whatsapp",
                    event_type="error",
                    raw_payload={"error": "signature_mismatch"},
                )

        # Extract event from payload
        entry = payload.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        # Determine event type
        if "messages" in value:
            messages = value.get("messages", [])
            if messages:
                msg = messages[0]
                msg_type = msg.get("type", "text")

                if msg_type == "interactive":
                    return WebhookEvent(
                        platform="whatsapp",
                        event_type="interaction",
                        raw_payload=payload,
                    )

                return WebhookEvent(
                    platform="whatsapp",
                    event_type="message",
                    raw_payload=payload,
                )

        if "statuses" in value:
            return WebhookEvent(
                platform="whatsapp",
                event_type="status",
                raw_payload=payload,
            )

        return WebhookEvent(
            platform="whatsapp",
            event_type="unknown",
            raw_payload=payload,
        )

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        ephemeral: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a bot command."""
        # WhatsApp doesn't have native commands, so just send a message
        if command.channel:
            return await self.send_message(
                channel_id=command.channel.id,
                text=text,
                blocks=blocks,
                **kwargs,
            )
        return SendMessageResponse(success=False, error="No channel for command response")

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a user interaction (button click or list selection)."""
        if interaction.channel:
            return await self.send_message(
                channel_id=interaction.channel.id,
                text=text,
                blocks=blocks,
                **kwargs,
            )
        return SendMessageResponse(success=False, error="No channel for interaction response")


__all__ = ["WhatsAppConnector"]
