"""
Microsoft Teams Chat Connector.

Implements ChatPlatformConnector for Microsoft Teams using
Bot Framework and Adaptive Cards.

Environment Variables:
- TEAMS_APP_ID: Bot application ID
- TEAMS_APP_PASSWORD: Bot application password
- TEAMS_TENANT_ID: Optional tenant ID for single-tenant apps
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import httpx for API calls
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available - Teams connector will have limited functionality")

from .base import ChatPlatformConnector
from .models import (
    BotCommand,
    ChatChannel,
    ChatMessage,
    ChatUser,
    FileAttachment,
    InteractionType,
    MessageButton,
    SendMessageResponse,
    UserInteraction,
    WebhookEvent,
)

# Environment configuration
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID", "")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD", "")
TEAMS_TENANT_ID = os.environ.get("TEAMS_TENANT_ID", "")

# Bot Framework API endpoints
BOT_FRAMEWORK_AUTH_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
BOT_FRAMEWORK_API_BASE = "https://smba.trafficmanager.net"


class TeamsConnector(ChatPlatformConnector):
    """
    Microsoft Teams connector using Bot Framework.

    Supports:
    - Sending messages with Adaptive Cards
    - Responding to commands and interactions
    - File uploads via OneDrive integration
    - Threaded conversations
    """

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_password: Optional[str] = None,
        tenant_id: Optional[str] = None,
        **config: Any,
    ):
        """
        Initialize Teams connector.

        Args:
            app_id: Bot application ID (defaults to TEAMS_APP_ID env var)
            app_password: Bot application password (defaults to TEAMS_APP_PASSWORD)
            tenant_id: Optional tenant ID for single-tenant apps
            **config: Additional configuration
        """
        super().__init__(
            bot_token=app_password or TEAMS_APP_PASSWORD,
            signing_secret=None,  # Teams uses JWT validation
            **config,
        )
        self.app_id = app_id or TEAMS_APP_ID
        self.app_password = app_password or TEAMS_APP_PASSWORD
        self.tenant_id = tenant_id or TEAMS_TENANT_ID
        self._access_token: Optional[str] = None
        self._token_expires: float = 0

    @property
    def platform_name(self) -> str:
        return "teams"

    @property
    def platform_display_name(self) -> str:
        return "Microsoft Teams"

    async def _get_access_token(self) -> str:
        """Get or refresh Bot Framework access token."""
        import time

        if self._access_token and time.time() < self._token_expires - 60:
            return self._access_token

        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx required for Teams API calls")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                BOT_FRAMEWORK_AUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.app_id,
                    "client_secret": self.app_password,
                    "scope": "https://api.botframework.com/.default",
                },
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data["access_token"]
            self._token_expires = time.time() + data.get("expires_in", 3600)

            return self._access_token

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        thread_id: Optional[str] = None,
        service_url: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send message to Teams channel."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(
                success=False,
                error="httpx not available",
            )

        try:
            token = await self._get_access_token()
            base_url = service_url or BOT_FRAMEWORK_API_BASE
            conv_id = conversation_id or channel_id

            # Build activity payload
            activity = {
                "type": "message",
                "text": text,
            }

            # Add Adaptive Card if blocks provided
            if blocks:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            # Handle threaded reply
            if thread_id:
                activity["replyToId"] = thread_id

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/v3/conversations/{conv_id}/activities",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=activity,
                )
                response.raise_for_status()
                data = response.json()

                return SendMessageResponse(
                    success=True,
                    message_id=data.get("id"),
                    channel_id=conv_id,
                )

        except Exception as e:
            logger.error(f"Teams send_message error: {e}")
            return SendMessageResponse(
                success=False,
                error=str(e),
            )

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        service_url: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Update an existing Teams message."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            token = await self._get_access_token()
            base_url = service_url or BOT_FRAMEWORK_API_BASE

            activity = {
                "type": "message",
                "text": text,
            }

            if blocks:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{base_url}/v3/conversations/{channel_id}/activities/{message_id}",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=activity,
                )
                response.raise_for_status()

                return SendMessageResponse(
                    success=True,
                    message_id=message_id,
                    channel_id=channel_id,
                )

        except Exception as e:
            logger.error(f"Teams update_message error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        service_url: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Delete a Teams message."""
        if not HTTPX_AVAILABLE:
            return False

        try:
            token = await self._get_access_token()
            base_url = service_url or BOT_FRAMEWORK_API_BASE

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{base_url}/v3/conversations/{channel_id}/activities/{message_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                return response.status_code in (200, 204)

        except Exception as e:
            logger.error(f"Teams delete_message error: {e}")
            return False

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: Optional[list[dict]] = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Teams command (mention or direct message)."""
        if command.response_url:
            # Use response URL for async response
            return await self._send_to_response_url(
                command.response_url,
                text,
                blocks,
            )

        if command.channel:
            return await self.send_message(
                command.channel.id,
                text,
                blocks,
                service_url=command.metadata.get("service_url"),
                **kwargs,
            )

        return SendMessageResponse(
            success=False,
            error="No channel or response URL available",
        )

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: Optional[list[dict]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Teams Adaptive Card action."""
        if interaction.response_url:
            return await self._send_to_response_url(
                interaction.response_url,
                text,
                blocks,
            )

        if interaction.channel and interaction.message_id and replace_original:
            return await self.update_message(
                interaction.channel.id,
                interaction.message_id,
                text,
                blocks,
                service_url=interaction.metadata.get("service_url"),
            )

        if interaction.channel:
            return await self.send_message(
                interaction.channel.id,
                text,
                blocks,
                service_url=interaction.metadata.get("service_url"),
            )

        return SendMessageResponse(success=False, error="No response target available")

    async def _send_to_response_url(
        self,
        response_url: str,
        text: str,
        blocks: Optional[list[dict]] = None,
    ) -> SendMessageResponse:
        """Send response to a Bot Framework response URL."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            token = await self._get_access_token()

            activity = {
                "type": "message",
                "text": text,
            }

            if blocks:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "body": blocks,
                        },
                    }
                ]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    response_url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=activity,
                )
                response.raise_for_status()

                return SendMessageResponse(success=True)

        except Exception as e:
            logger.error(f"Teams response URL error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload file to Teams (via OneDrive integration)."""
        # Teams file upload requires Graph API - simplified implementation
        logger.warning("Teams file upload requires Microsoft Graph API setup")
        return FileAttachment(
            id="",
            filename=filename,
            content_type=content_type,
            size=len(content),
            content=content,
        )

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        """Download file from Teams."""
        logger.warning("Teams file download requires Microsoft Graph API setup")
        return FileAttachment(
            id=file_id,
            filename="",
            content_type="application/octet-stream",
            size=0,
        )

    def format_blocks(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[tuple[str, str]]] = None,
        actions: Optional[list[MessageButton]] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Format content as Adaptive Card elements."""
        elements: list[dict] = []

        if title:
            elements.append(
                {
                    "type": "TextBlock",
                    "text": title,
                    "size": "Large",
                    "weight": "Bolder",
                }
            )

        if body:
            elements.append(
                {
                    "type": "TextBlock",
                    "text": body,
                    "wrap": True,
                }
            )

        if fields:
            fact_set = {
                "type": "FactSet",
                "facts": [{"title": label, "value": value} for label, value in fields],
            }
            elements.append(fact_set)

        if actions:
            action_set = {
                "type": "ActionSet",
                "actions": [
                    self.format_button(btn.text, btn.action_id, btn.value, btn.style)
                    for btn in actions
                ],
            }
            elements.append(action_set)

        return elements

    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: str = "default",
        url: Optional[str] = None,
    ) -> dict:
        """Format an Adaptive Card action button."""
        if url:
            return {
                "type": "Action.OpenUrl",
                "title": text,
                "url": url,
            }

        action = {
            "type": "Action.Submit",
            "title": text,
            "data": {
                "action": action_id,
                "value": value or action_id,
            },
        }

        if style == "danger":
            action["style"] = "destructive"
        elif style == "primary":
            action["style"] = "positive"

        return action

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """
        Verify Bot Framework JWT token.

        Uses PyJWT to validate the token against Microsoft's public keys.
        Falls back to header presence check if PyJWT is not available.
        """
        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("Missing or invalid Authorization header")
            return False

        # Use JWT verification if available
        try:
            from .jwt_verify import verify_teams_webhook, HAS_JWT

            if HAS_JWT:
                return verify_teams_webhook(auth_header, self.app_id)
            else:
                logger.warning(
                    "PyJWT not available - accepting Teams webhook without full verification. "
                    "Install PyJWT for secure webhook validation: pip install PyJWT"
                )
                return True
        except ImportError:
            logger.warning("JWT verification module not available")
            return True

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """Parse Teams Bot Framework activity into WebhookEvent."""
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return WebhookEvent(
                platform=self.platform_name,
                event_type="error",
                raw_payload={},
            )

        activity_type = payload.get("type", "")
        service_url = payload.get("serviceUrl", "")

        # Parse user
        from_data = payload.get("from", {})
        user = ChatUser(
            id=from_data.get("id", ""),
            platform=self.platform_name,
            display_name=from_data.get("name"),
            metadata={"aadObjectId": from_data.get("aadObjectId")},
        )

        # Parse channel
        conversation = payload.get("conversation", {})
        channel = ChatChannel(
            id=conversation.get("id", ""),
            platform=self.platform_name,
            name=conversation.get("name"),
            is_private=conversation.get("isGroup") is False,
            team_id=conversation.get("tenantId"),
            metadata={"conversationType": conversation.get("conversationType")},
        )

        event = WebhookEvent(
            platform=self.platform_name,
            event_type=activity_type,
            raw_payload=payload,
            metadata={"service_url": service_url},
        )

        if activity_type == "message":
            # Regular message
            text = payload.get("text", "")

            # Check for command (bot mention)
            entities = payload.get("entities", [])
            is_mention = any(e.get("type") == "mention" for e in entities)

            if is_mention and text.strip().startswith("<at>"):
                # Extract command after mention
                import re

                clean_text = re.sub(r"<at>.*?</at>\s*", "", text).strip()
                parts = clean_text.split(maxsplit=1)

                event.command = BotCommand(
                    name=parts[0] if parts else "",
                    text=clean_text,
                    args=parts[1].split() if len(parts) > 1 else [],
                    user=user,
                    channel=channel,
                    platform=self.platform_name,
                    metadata={"service_url": service_url},
                )
            else:
                event.message = ChatMessage(
                    id=payload.get("id", ""),
                    platform=self.platform_name,
                    channel=channel,
                    author=user,
                    content=text,
                    thread_id=payload.get("replyToId"),
                    metadata={"service_url": service_url},
                )

        elif activity_type == "invoke":
            # Adaptive Card action
            action_data = payload.get("value", {})

            event.interaction = UserInteraction(
                id=payload.get("id", ""),
                interaction_type=InteractionType.BUTTON_CLICK,
                action_id=action_data.get("action", ""),
                value=action_data.get("value"),
                user=user,
                channel=channel,
                message_id=payload.get("replyToId"),
                platform=self.platform_name,
                metadata={"service_url": service_url},
            )

        return event
