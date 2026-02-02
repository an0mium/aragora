"""
Social Namespace API

Provides social media integration for publishing debate results:
- YouTube OAuth authentication
- Publishing debates to Twitter/X
- Publishing debates to YouTube
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

Visibility = Literal["public", "unlisted", "private"]


class SocialAPI:
    """
    Synchronous Social API.

    Provides social media integration for publishing debate results.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Check YouTube connection status
        >>> status = client.social.get_youtube_status()
        >>> if status["connected"]:
        ...     result = client.social.publish_to_youtube(
        ...         debate_id="deb_123",
        ...         title="AI Debate: Should we adopt microservices?"
        ...     )
        ...     print(result["url"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_youtube_auth_url(self) -> dict[str, Any]:
        """
        Get YouTube OAuth authorization URL.

        Returns:
            Dictionary with 'auth_url' to redirect user for OAuth
            and 'state' for CSRF protection
        """
        return self._client.request("GET", "/api/youtube/auth")

    def handle_youtube_callback(
        self,
        code: str,
        state: str,
    ) -> dict[str, Any]:
        """
        Handle YouTube OAuth callback.

        Args:
            code: OAuth authorization code from callback
            state: State parameter for CSRF validation

        Returns:
            Dictionary with 'success' boolean
        """
        params = {"code": code, "state": state}
        return self._client.request("GET", "/api/youtube/callback", params=params)

    def get_youtube_status(self) -> dict[str, Any]:
        """
        Get YouTube connector status.

        Returns:
            YouTube connection status including:
            - connected: Whether YouTube is connected
            - channel_id: The connected channel ID
            - channel_name: The connected channel name
            - expires_at: When the OAuth token expires
        """
        return self._client.request("GET", "/api/youtube/status")

    def publish_to_twitter(
        self,
        debate_id: str,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: Visibility | None = None,
    ) -> dict[str, Any]:
        """
        Publish a debate to Twitter/X.

        Args:
            debate_id: The debate to publish
            title: Optional custom title
            description: Optional custom description
            tags: Optional list of tags/hashtags
            visibility: Content visibility (public, unlisted, private)

        Returns:
            Publish result with success status, platform, URL, and post_id
        """
        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        if tags:
            data["tags"] = tags
        if visibility:
            data["visibility"] = visibility

        return self._client.request(
            "POST",
            f"/api/debates/{debate_id}/publish/twitter",
            json=data if data else None,
        )

    def publish_to_youtube(
        self,
        debate_id: str,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: Visibility | None = None,
    ) -> dict[str, Any]:
        """
        Publish a debate to YouTube.

        Args:
            debate_id: The debate to publish
            title: Optional custom title
            description: Optional custom description
            tags: Optional list of tags
            visibility: Video visibility (public, unlisted, private)

        Returns:
            Publish result with success status, platform, URL, and post_id
        """
        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        if tags:
            data["tags"] = tags
        if visibility:
            data["visibility"] = visibility

        return self._client.request(
            "POST",
            f"/api/debates/{debate_id}/publish/youtube",
            json=data if data else None,
        )


class AsyncSocialAPI:
    """
    Asynchronous Social API.

    Provides social media integration for publishing debate results.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.social.get_youtube_status()
        ...     if status["connected"]:
        ...         result = await client.social.publish_to_youtube(
        ...             debate_id="deb_123",
        ...             title="AI Debate: Should we adopt microservices?"
        ...         )
        ...         print(result["url"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_youtube_auth_url(self) -> dict[str, Any]:
        """
        Get YouTube OAuth authorization URL.

        Returns:
            Dictionary with 'auth_url' to redirect user for OAuth
            and 'state' for CSRF protection
        """
        return await self._client.request("GET", "/api/youtube/auth")

    async def handle_youtube_callback(
        self,
        code: str,
        state: str,
    ) -> dict[str, Any]:
        """
        Handle YouTube OAuth callback.

        Args:
            code: OAuth authorization code from callback
            state: State parameter for CSRF validation

        Returns:
            Dictionary with 'success' boolean
        """
        params = {"code": code, "state": state}
        return await self._client.request("GET", "/api/youtube/callback", params=params)

    async def get_youtube_status(self) -> dict[str, Any]:
        """
        Get YouTube connector status.

        Returns:
            YouTube connection status including:
            - connected: Whether YouTube is connected
            - channel_id: The connected channel ID
            - channel_name: The connected channel name
            - expires_at: When the OAuth token expires
        """
        return await self._client.request("GET", "/api/youtube/status")

    async def publish_to_twitter(
        self,
        debate_id: str,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: Visibility | None = None,
    ) -> dict[str, Any]:
        """
        Publish a debate to Twitter/X.

        Args:
            debate_id: The debate to publish
            title: Optional custom title
            description: Optional custom description
            tags: Optional list of tags/hashtags
            visibility: Content visibility (public, unlisted, private)

        Returns:
            Publish result with success status, platform, URL, and post_id
        """
        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        if tags:
            data["tags"] = tags
        if visibility:
            data["visibility"] = visibility

        return await self._client.request(
            "POST",
            f"/api/debates/{debate_id}/publish/twitter",
            json=data if data else None,
        )

    async def publish_to_youtube(
        self,
        debate_id: str,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        visibility: Visibility | None = None,
    ) -> dict[str, Any]:
        """
        Publish a debate to YouTube.

        Args:
            debate_id: The debate to publish
            title: Optional custom title
            description: Optional custom description
            tags: Optional list of tags
            visibility: Video visibility (public, unlisted, private)

        Returns:
            Publish result with success status, platform, URL, and post_id
        """
        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        if tags:
            data["tags"] = tags
        if visibility:
            data["visibility"] = visibility

        return await self._client.request(
            "POST",
            f"/api/debates/{debate_id}/publish/youtube",
            json=data if data else None,
        )
