"""Tests for Social namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# YouTube OAuth Operations
# =========================================================================


class TestSocialYouTubeOAuth:
    """Tests for YouTube OAuth operations."""

    def test_get_youtube_auth_url(self) -> None:
        """Get YouTube OAuth authorization URL."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "auth_url": "https://accounts.google.com/o/oauth2/auth?...",
                "state": "abc123xyz",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.social.get_youtube_auth_url()

            mock_request.assert_called_once_with("GET", "/api/youtube/auth")
            assert "auth_url" in result
            assert result["state"] == "abc123xyz"
            client.close()

    def test_handle_youtube_callback(self) -> None:
        """Handle YouTube OAuth callback."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.social.handle_youtube_callback(code="auth_code_123", state="abc123xyz")

            mock_request.assert_called_once_with(
                "GET",
                "/api/youtube/callback",
                params={"code": "auth_code_123", "state": "abc123xyz"},
            )
            assert result["success"] is True
            client.close()

    def test_get_youtube_status_connected(self) -> None:
        """Get YouTube status when connected."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "connected": True,
                "channel_id": "UCxxxx",
                "channel_name": "My Channel",
                "expires_at": "2025-02-15T10:00:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.social.get_youtube_status()

            mock_request.assert_called_once_with("GET", "/api/youtube/status")
            assert result["connected"] is True
            assert result["channel_name"] == "My Channel"
            client.close()

    def test_get_youtube_status_disconnected(self) -> None:
        """Get YouTube status when not connected."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"connected": False}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.social.get_youtube_status()

            assert result["connected"] is False
            assert "channel_id" not in result
            client.close()


# =========================================================================
# Publish to Twitter Operations
# =========================================================================


class TestSocialTwitterPublish:
    """Tests for Twitter publish operations."""

    def test_publish_to_twitter_basic(self) -> None:
        """Publish to Twitter with minimal options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "success": True,
                "platform": "twitter",
                "url": "https://twitter.com/i/status/12345",
                "post_id": "12345",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.social.publish_to_twitter(debate_id="deb_123")

            mock_request.assert_called_once_with(
                "POST", "/api/debates/deb_123/publish/twitter", json=None
            )
            assert result["success"] is True
            assert result["platform"] == "twitter"
            client.close()

    def test_publish_to_twitter_with_options(self) -> None:
        """Publish to Twitter with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "platform": "twitter"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.social.publish_to_twitter(
                debate_id="deb_456",
                title="AI Debate Summary",
                description="A fascinating debate about AI safety",
                tags=["AI", "safety", "debate"],
                visibility="public",
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["title"] == "AI Debate Summary"
            assert json_data["description"] == "A fascinating debate about AI safety"
            assert json_data["tags"] == ["AI", "safety", "debate"]
            assert json_data["visibility"] == "public"
            client.close()


# =========================================================================
# Publish to YouTube Operations
# =========================================================================


class TestSocialYouTubePublish:
    """Tests for YouTube publish operations."""

    def test_publish_to_youtube_basic(self) -> None:
        """Publish to YouTube with minimal options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "success": True,
                "platform": "youtube",
                "url": "https://youtube.com/watch?v=abc123",
                "post_id": "abc123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.social.publish_to_youtube(debate_id="deb_789")

            mock_request.assert_called_once_with(
                "POST", "/api/debates/deb_789/publish/youtube", json=None
            )
            assert result["success"] is True
            assert result["platform"] == "youtube"
            client.close()

    def test_publish_to_youtube_with_all_options(self) -> None:
        """Publish to YouTube with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "platform": "youtube"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.social.publish_to_youtube(
                debate_id="deb_101",
                title="Multi-Agent Debate: Climate Solutions",
                description="Watch AI agents debate the best approaches to climate change",
                tags=["AI", "climate", "debate", "multi-agent"],
                visibility="unlisted",
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["title"] == "Multi-Agent Debate: Climate Solutions"
            assert "climate change" in json_data["description"]
            assert len(json_data["tags"]) == 4
            assert json_data["visibility"] == "unlisted"
            client.close()

    def test_publish_to_youtube_private(self) -> None:
        """Publish to YouTube as private."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "platform": "youtube"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.social.publish_to_youtube(debate_id="deb_102", visibility="private")

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["visibility"] == "private"
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncSocial:
    """Tests for async Social API."""

    @pytest.mark.asyncio
    async def test_async_get_youtube_auth_url(self) -> None:
        """Get YouTube auth URL asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"auth_url": "https://...", "state": "xyz"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.social.get_youtube_auth_url()

                mock_request.assert_called_once_with("GET", "/api/youtube/auth")
                assert "auth_url" in result

    @pytest.mark.asyncio
    async def test_async_handle_youtube_callback(self) -> None:
        """Handle YouTube callback asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"success": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.social.handle_youtube_callback(
                    code="code_abc", state="state_xyz"
                )

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/youtube/callback",
                    params={"code": "code_abc", "state": "state_xyz"},
                )
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_async_get_youtube_status(self) -> None:
        """Get YouTube status asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"connected": True, "channel_name": "Test"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.social.get_youtube_status()

                mock_request.assert_called_once_with("GET", "/api/youtube/status")
                assert result["connected"] is True

    @pytest.mark.asyncio
    async def test_async_publish_to_twitter(self) -> None:
        """Publish to Twitter asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "platform": "twitter"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.social.publish_to_twitter(
                    debate_id="deb_async", title="Async Test"
                )

                call_args = mock_request.call_args
                assert call_args[0][1] == "/api/debates/deb_async/publish/twitter"
                assert result["platform"] == "twitter"

    @pytest.mark.asyncio
    async def test_async_publish_to_youtube(self) -> None:
        """Publish to YouTube asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"success": True, "platform": "youtube"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.social.publish_to_youtube(
                    debate_id="deb_async2", title="Async YouTube Test"
                )

                call_args = mock_request.call_args
                assert call_args[0][1] == "/api/debates/deb_async2/publish/youtube"
                assert result["platform"] == "youtube"
