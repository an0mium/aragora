"""
Teams Bot channel and team operations.

Provides functionality for:
- Team/channel listing and management
- Conversation reference management
- Proactive messaging to channels
- Member operations
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from aragora.server.handlers.bots.teams_utils import (
    _conversation_references,
    get_conversation_reference,
)

if TYPE_CHECKING:
    from aragora.server.handlers.bots.teams.handler import TeamsBot

logger = logging.getLogger(__name__)

# Environment variables
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID") or os.environ.get("MS_APP_ID")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD")


class TeamsChannelManager:
    """Manages Teams channel and conversation operations.

    Provides functionality for:
    - Listing teams and channels
    - Managing conversation references
    - Sending proactive messages
    - Member management
    """

    def __init__(self, bot: TeamsBot | None = None):
        """Initialize the channel manager.

        Args:
            bot: Optional TeamsBot instance for sending messages.
        """
        self.bot = bot
        self._connector: Any | None = None

    async def _get_connector(self) -> Any:
        """Get the Teams connector for API calls."""
        if self.bot:
            return await self.bot._get_connector()

        if self._connector is None:
            try:
                from aragora.connectors.chat.teams import TeamsConnector

                self._connector = TeamsConnector(
                    app_id=TEAMS_APP_ID or "",
                    app_password=TEAMS_APP_PASSWORD or "",
                )
            except ImportError:
                logger.warning("Teams connector not available")
                return None
        return self._connector

    # =========================================================================
    # Conversation Reference Management
    # =========================================================================

    def get_all_conversation_references(self) -> dict[str, dict[str, Any]]:
        """Get all stored conversation references.

        Returns:
            Dict mapping conversation IDs to their references.
        """
        return dict(_conversation_references)

    def get_reference(self, conversation_id: str) -> dict[str, Any] | None:
        """Get a stored conversation reference.

        Args:
            conversation_id: The conversation ID.

        Returns:
            Conversation reference dict or None.
        """
        return get_conversation_reference(conversation_id)

    def remove_reference(self, conversation_id: str) -> bool:
        """Remove a conversation reference.

        Args:
            conversation_id: The conversation ID to remove.

        Returns:
            True if the reference was removed.
        """
        if conversation_id in _conversation_references:
            del _conversation_references[conversation_id]
            logger.info(f"Removed conversation reference for {conversation_id}")
            return True
        return False

    def clear_references(self) -> int:
        """Clear all conversation references.

        Returns:
            Number of references cleared.
        """
        count = len(_conversation_references)
        _conversation_references.clear()
        logger.info(f"Cleared {count} conversation references")
        return count

    # =========================================================================
    # Team/Channel Operations
    # =========================================================================

    async def get_team_details(self, team_id: str, service_url: str) -> dict[str, Any] | None:
        """Get details about a team.

        Args:
            team_id: The team ID.
            service_url: Bot Framework service URL.

        Returns:
            Team details dict or None if not found.
        """
        connector = await self._get_connector()
        if not connector:
            return None

        try:
            token = await connector._get_access_token()

            response = await connector._http_request(
                method="GET",
                url=f"{service_url}/v3/teams/{team_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                },
                operation="get_team_details",
            )

            return response

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get team details: {e}")
            return None

    async def get_team_channels(self, team_id: str, service_url: str) -> list[dict[str, Any]]:
        """Get list of channels in a team.

        Args:
            team_id: The team ID.
            service_url: Bot Framework service URL.

        Returns:
            List of channel info dicts.
        """
        connector = await self._get_connector()
        if not connector:
            return []

        try:
            token = await connector._get_access_token()

            response = await connector._http_request(
                method="GET",
                url=f"{service_url}/v3/teams/{team_id}/conversations",
                headers={
                    "Authorization": f"Bearer {token}",
                },
                operation="get_team_channels",
            )

            return response.get("conversations", [])

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get team channels: {e}")
            return []

    async def get_conversation_members(
        self, conversation_id: str, service_url: str
    ) -> list[dict[str, Any]]:
        """Get members of a conversation.

        Args:
            conversation_id: The conversation ID.
            service_url: Bot Framework service URL.

        Returns:
            List of member info dicts.
        """
        connector = await self._get_connector()
        if not connector:
            return []

        try:
            token = await connector._get_access_token()

            response = await connector._http_request(
                method="GET",
                url=f"{service_url}/v3/conversations/{conversation_id}/members",
                headers={
                    "Authorization": f"Bearer {token}",
                },
                operation="get_conversation_members",
            )

            return response if isinstance(response, list) else []

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get conversation members: {e}")
            return []

    async def get_member(
        self, conversation_id: str, member_id: str, service_url: str
    ) -> dict[str, Any] | None:
        """Get a specific member from a conversation.

        Args:
            conversation_id: The conversation ID.
            member_id: The member's user ID.
            service_url: Bot Framework service URL.

        Returns:
            Member info dict or None.
        """
        connector = await self._get_connector()
        if not connector:
            return None

        try:
            token = await connector._get_access_token()

            response = await connector._http_request(
                method="GET",
                url=f"{service_url}/v3/conversations/{conversation_id}/members/{member_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                },
                operation="get_member",
            )

            return response

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get member: {e}")
            return None

    # =========================================================================
    # Proactive Messaging
    # =========================================================================

    async def send_to_conversation(
        self,
        conversation_id: str,
        text: str | None = None,
        card: dict[str, Any] | None = None,
        fallback_text: str = "",
    ) -> bool:
        """Send a proactive message to a conversation.

        Uses stored conversation references to send messages.

        Args:
            conversation_id: Target conversation ID.
            text: Plain text message.
            card: Adaptive Card to send.
            fallback_text: Fallback text for card messages.

        Returns:
            True if the message was sent.
        """
        if self.bot:
            return await self.bot.send_proactive_message(
                conversation_id=conversation_id,
                text=text,
                card=card,
                fallback_text=fallback_text,
            )

        # Direct connector approach without bot
        ref = get_conversation_reference(conversation_id)
        if not ref:
            logger.warning(f"No conversation reference for {conversation_id}")
            return False

        connector = await self._get_connector()
        if not connector:
            return False

        try:
            service_url = ref.get("service_url", "")
            token = await connector._get_access_token()

            activity: dict[str, Any] = {
                "type": "message",
                "text": text or fallback_text or "",
            }

            if card:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ]

            await connector._http_request(
                method="POST",
                url=f"{service_url}/v3/conversations/{conversation_id}/activities",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=activity,
                operation="proactive_message",
            )

            return True

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to send proactive message: {e}")
            return False

    async def send_to_channel(
        self,
        team_id: str,
        channel_id: str,
        service_url: str,
        text: str | None = None,
        card: dict[str, Any] | None = None,
        fallback_text: str = "",
    ) -> bool:
        """Send a message to a specific channel.

        Args:
            team_id: The team ID.
            channel_id: The channel ID.
            service_url: Bot Framework service URL.
            text: Plain text message.
            card: Adaptive Card to send.
            fallback_text: Fallback text for card messages.

        Returns:
            True if the message was sent.
        """
        connector = await self._get_connector()
        if not connector:
            return False

        try:
            token = await connector._get_access_token()

            activity: dict[str, Any] = {
                "type": "message",
                "text": text or fallback_text or "",
            }

            if card:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ]

            # Create conversation in channel
            conversation_params = {
                "bot": {
                    "id": TEAMS_APP_ID,
                },
                "isGroup": True,
                "tenantId": "",  # Will be filled by service
                "activity": activity,
                "channelData": {
                    "teamsChannelId": channel_id,
                    "teamsTeamId": team_id,
                },
            }

            await connector._http_request(
                method="POST",
                url=f"{service_url}/v3/conversations",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=conversation_params,
                operation="send_to_channel",
            )

            return True

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to send to channel: {e}")
            return False

    async def create_personal_conversation(
        self, user_id: str, tenant_id: str, service_url: str
    ) -> str | None:
        """Create a 1:1 conversation with a user.

        Args:
            user_id: The user's ID.
            tenant_id: The tenant ID.
            service_url: Bot Framework service URL.

        Returns:
            The new conversation ID or None if creation failed.
        """
        connector = await self._get_connector()
        if not connector:
            return None

        try:
            token = await connector._get_access_token()

            conversation_params = {
                "bot": {
                    "id": TEAMS_APP_ID,
                },
                "members": [
                    {
                        "id": user_id,
                    }
                ],
                "channelData": {
                    "tenant": {
                        "id": tenant_id,
                    }
                },
            }

            response = await connector._http_request(
                method="POST",
                url=f"{service_url}/v3/conversations",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=conversation_params,
                operation="create_personal_conversation",
            )

            return response.get("id")

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to create personal conversation: {e}")
            return None

    # =========================================================================
    # Message Management
    # =========================================================================

    async def update_activity(
        self,
        conversation_id: str,
        activity_id: str,
        service_url: str,
        text: str | None = None,
        card: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing activity (message).

        Args:
            conversation_id: The conversation ID.
            activity_id: The activity/message ID to update.
            service_url: Bot Framework service URL.
            text: New text content.
            card: New Adaptive Card content.

        Returns:
            True if the update was successful.
        """
        connector = await self._get_connector()
        if not connector:
            return False

        try:
            token = await connector._get_access_token()

            activity: dict[str, Any] = {
                "type": "message",
                "id": activity_id,
                "text": text or "",
            }

            if card:
                activity["attachments"] = [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": card,
                    }
                ]

            await connector._http_request(
                method="PUT",
                url=f"{service_url}/v3/conversations/{conversation_id}/activities/{activity_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=activity,
                operation="update_activity",
            )

            return True

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to update activity: {e}")
            return False

    async def delete_activity(
        self, conversation_id: str, activity_id: str, service_url: str
    ) -> bool:
        """Delete an activity (message).

        Args:
            conversation_id: The conversation ID.
            activity_id: The activity/message ID to delete.
            service_url: Bot Framework service URL.

        Returns:
            True if the deletion was successful.
        """
        connector = await self._get_connector()
        if not connector:
            return False

        try:
            token = await connector._get_access_token()

            await connector._http_request(
                method="DELETE",
                url=f"{service_url}/v3/conversations/{conversation_id}/activities/{activity_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                },
                operation="delete_activity",
            )

            return True

        except (RuntimeError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to delete activity: {e}")
            return False


__all__ = [
    "TeamsChannelManager",
]
