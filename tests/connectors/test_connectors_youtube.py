"""
Tests for YouTube uploader connector.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from aragora.connectors.youtube_uploader import (
    YouTubeVideoMetadata,
    UploadResult,
    YouTubeRateLimiter,
    YouTubeUploaderConnector,
    YouTubeAuthError,
    YouTubeAPIError,
    YouTubeQuotaError,
    create_video_metadata_from_debate,
    MAX_TITLE_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_TAGS_LENGTH,
)
from aragora.resilience import CircuitBreaker


class TestYouTubeVideoMetadata:
    """Tests for YouTubeVideoMetadata dataclass."""

    def test_basic_creation(self):
        """Should create metadata with basic fields."""
        meta = YouTubeVideoMetadata(
            title="Test Video",
            description="Test description",
        )
        assert meta.title == "Test Video"
        assert meta.description == "Test description"
        assert meta.category_id == "28"  # Default: Science & Technology
        assert meta.privacy_status == "public"

    def test_title_truncation(self):
        """Should truncate title exceeding max length."""
        long_title = "A" * 150
        meta = YouTubeVideoMetadata(title=long_title, description="desc")
        assert len(meta.title) == MAX_TITLE_LENGTH
        assert meta.title.endswith("...")

    def test_description_truncation(self):
        """Should truncate description exceeding max length."""
        long_desc = "A" * 6000
        meta = YouTubeVideoMetadata(title="Title", description=long_desc)
        assert len(meta.description) == MAX_DESCRIPTION_LENGTH
        assert meta.description.endswith("...")

    def test_tags_truncation(self):
        """Should truncate tags exceeding total length limit."""
        long_tags = ["verylongtag" * 10] * 10  # Each ~110 chars
        meta = YouTubeVideoMetadata(
            title="Title",
            description="Desc",
            tags=long_tags,
        )
        total_len = sum(len(tag) for tag in meta.tags)
        assert total_len <= MAX_TAGS_LENGTH

    def test_to_api_body(self):
        """Should convert to YouTube API request format."""
        meta = YouTubeVideoMetadata(
            title="Test Video",
            description="Test description",
            tags=["tag1", "tag2"],
            category_id="22",
            privacy_status="unlisted",
        )
        body = meta.to_api_body()

        assert body["snippet"]["title"] == "Test Video"
        assert body["snippet"]["description"] == "Test description"
        assert body["snippet"]["tags"] == ["tag1", "tag2"]
        assert body["snippet"]["categoryId"] == "22"
        assert body["status"]["privacyStatus"] == "unlisted"


class TestUploadResult:
    """Tests for UploadResult dataclass."""

    def test_success_result(self):
        """Should create successful upload result."""
        result = UploadResult(
            video_id="abc123",
            title="Test Video",
            url="https://youtube.com/watch?v=abc123",
            success=True,
        )
        assert result.success is True
        assert result.error is None
        assert result.upload_status == "complete"

    def test_failure_result(self):
        """Should create failed upload result."""
        result = UploadResult(
            video_id="",
            title="Test Video",
            url="",
            success=False,
            error="Upload failed",
        )
        assert result.success is False
        assert result.error == "Upload failed"


class TestYouTubeRateLimiter:
    """Tests for YouTubeRateLimiter."""

    def test_initial_state(self):
        """Should start with full quota."""
        limiter = YouTubeRateLimiter(daily_quota=10000)
        assert limiter.remaining_quota == 10000
        assert limiter.can_upload() is True

    def test_record_upload(self):
        """Should deduct 1600 units for upload."""
        limiter = YouTubeRateLimiter(daily_quota=10000)
        limiter.record_upload()
        assert limiter.remaining_quota == 8400

    def test_record_api_call(self):
        """Should deduct custom units for API call."""
        limiter = YouTubeRateLimiter(daily_quota=100)
        limiter.record_api_call(10)
        assert limiter.remaining_quota == 90

    def test_can_upload_blocks_when_insufficient(self):
        """Should block upload when quota insufficient."""
        limiter = YouTubeRateLimiter(daily_quota=1000)
        assert limiter.can_upload() is False  # Need 1600, have 1000

    def test_quota_resets_after_day(self):
        """Should reset quota after reset time passes."""
        limiter = YouTubeRateLimiter(daily_quota=10000)
        limiter.record_upload()  # Use 1600

        # Force reset time to be in the past
        limiter.reset_time = time.time() - 1

        assert limiter.remaining_quota == 10000  # Should reset


class TestCircuitBreaker:
    """Tests for CircuitBreaker (using canonical from aragora.resilience)."""

    def test_initial_state_closed(self):
        """Should start in closed state."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        assert cb.can_proceed() is True
        assert cb.is_open is False

    def test_opens_after_threshold(self):
        """Should open after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False

        cb.record_failure()  # Third failure
        assert cb.is_open is True
        assert cb.can_proceed() is False

    def test_resets_on_success(self):
        """Should reset failures on success."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failures == 0

    def test_allows_recovery_after_timeout(self):
        """Should allow requests after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=1)
        cb.record_failure()
        cb.record_failure()  # Opens circuit

        assert cb.can_proceed() is False

        # Force open time to be in the past
        cb._single_open_at = time.time() - 2

        assert cb.can_proceed() is True


class TestYouTubeUploaderConnector:
    """Tests for YouTubeUploaderConnector."""

    def test_is_configured_false_without_credentials(self):
        """Should report not configured without credentials."""
        connector = YouTubeUploaderConnector(
            client_id="",
            client_secret="",
            refresh_token="",
        )
        assert connector.is_configured is False

    def test_is_configured_true_with_credentials(self):
        """Should report configured with all credentials."""
        connector = YouTubeUploaderConnector(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="test-token",
        )
        assert connector.is_configured is True

    def test_get_auth_url(self):
        """Should generate valid OAuth URL."""
        connector = YouTubeUploaderConnector(
            client_id="test-client-id",
            client_secret="test-secret",
            refresh_token="",
        )
        url = connector.get_auth_url("http://localhost/callback", state="test-state")

        assert "test-client-id" in url
        assert "localhost" in url
        assert "test-state" in url
        assert "youtube.upload" in url

    @pytest.mark.asyncio
    async def test_upload_fails_without_config(self, tmp_path):
        """Should fail upload when not configured."""
        connector = YouTubeUploaderConnector(
            client_id="",
            client_secret="",
            refresh_token="",
        )

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")

        metadata = YouTubeVideoMetadata(title="Test", description="Test")
        result = await connector.upload(video_path, metadata)

        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_upload_fails_when_circuit_open(self, tmp_path):
        """Should fail upload when circuit breaker is open."""
        connector = YouTubeUploaderConnector(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="test-token",
        )

        # Trip the circuit breaker
        connector.circuit_breaker.record_failure()
        connector.circuit_breaker.record_failure()
        connector.circuit_breaker.record_failure()

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")

        metadata = YouTubeVideoMetadata(title="Test", description="Test")
        result = await connector.upload(video_path, metadata)

        assert result.success is False
        assert "Circuit breaker" in result.error

    @pytest.mark.asyncio
    async def test_upload_fails_quota_exceeded(self, tmp_path):
        """Should fail upload when quota exceeded."""
        connector = YouTubeUploaderConnector(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="test-token",
        )

        # Use up all quota
        connector.rate_limiter = YouTubeRateLimiter(daily_quota=100)

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"fake video content")

        metadata = YouTubeVideoMetadata(title="Test", description="Test")
        result = await connector.upload(video_path, metadata)

        assert result.success is False
        assert "quota exceeded" in result.error.lower()

    @pytest.mark.asyncio
    async def test_upload_fails_file_not_found(self, tmp_path):
        """Should fail upload when video file doesn't exist."""
        connector = YouTubeUploaderConnector(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="test-token",
        )

        video_path = tmp_path / "nonexistent.mp4"
        metadata = YouTubeVideoMetadata(title="Test", description="Test")
        result = await connector.upload(video_path, metadata)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_refresh_token_success(self):
        """Should refresh access token successfully."""
        connector = YouTubeUploaderConnector(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="test-refresh-token",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            result = await connector._refresh_access_token()

        assert result is True
        assert connector._access_token == "new-access-token"

    @pytest.mark.asyncio
    async def test_refresh_token_failure(self):
        """Should handle token refresh failure."""
        connector = YouTubeUploaderConnector(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="test-refresh-token",
        )

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            result = await connector._refresh_access_token()

        assert result is False

    @pytest.mark.asyncio
    async def test_exchange_code_success(self):
        """Should exchange auth code for tokens."""
        connector = YouTubeUploaderConnector(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "access-token",
            "refresh_token": "new-refresh-token",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            result = await connector.exchange_code("auth-code", "http://localhost/cb")

        assert result["access_token"] == "access-token"
        assert connector.refresh_token == "new-refresh-token"

    @pytest.mark.asyncio
    async def test_exchange_code_failure(self):
        """Should raise error on failed code exchange."""
        connector = YouTubeUploaderConnector(
            client_id="test-id",
            client_secret="test-secret",
            refresh_token="",
        )

        mock_response = MagicMock()
        mock_response.status_code = 400

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            with pytest.raises(YouTubeAuthError):
                await connector.exchange_code("bad-code", "http://localhost/cb")


class TestCreateVideoMetadataFromDebate:
    """Tests for create_video_metadata_from_debate helper."""

    def test_creates_valid_metadata(self):
        """Should create valid metadata from debate info."""
        metadata = create_video_metadata_from_debate(
            task="Should AI systems be open source?",
            agents=["claude", "gpt-4", "gemini"],
            consensus_reached=True,
            debate_id="debate-123",
        )

        assert "AI Debate" in metadata.title
        assert "Should AI systems" in metadata.title
        assert "claude" in metadata.description
        assert "Consensus reached" in metadata.description
        assert "AI" in metadata.tags
        assert metadata.category_id == "28"

    def test_truncates_long_task(self):
        """Should truncate very long task in title."""
        long_task = "A" * 200
        metadata = create_video_metadata_from_debate(
            task=long_task,
            agents=["agent1"],
            consensus_reached=False,
            debate_id="debate-123",
        )

        assert len(metadata.title) <= MAX_TITLE_LENGTH

    def test_includes_all_agents_in_description(self):
        """Should list all agents in description."""
        agents = ["claude", "gpt-4", "gemini", "llama"]
        metadata = create_video_metadata_from_debate(
            task="Test task",
            agents=agents,
            consensus_reached=False,
            debate_id="debate-123",
        )

        for agent in agents:
            assert agent in metadata.description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
