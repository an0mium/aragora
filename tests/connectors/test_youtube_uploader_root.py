"""
Comprehensive tests for YouTube uploader connector.

Tests cover:
- YouTubeVideoMetadata (4 tests)
- YouTubeRateLimiter (3 tests)
- CircuitBreaker (3 tests)
- OAuth & Authentication (5 tests)
- Upload Validation (3 tests)
- Upload Flow (4 tests)
- Video Status & Factory (4 tests)
"""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.youtube_uploader import (
    MAX_DESCRIPTION_LENGTH,
    MAX_TAGS_LENGTH,
    MAX_TITLE_LENGTH,
    UploadResult,
    YouTubeRateLimiter,
    YouTubeUploaderConnector,
    YouTubeVideoMetadata,
    create_video_metadata_from_debate,
)
from aragora.resilience import CircuitBreaker


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_youtube_credentials():
    """YouTube API credentials for testing."""
    return {
        "client_id": "test_client_id.apps.googleusercontent.com",
        "client_secret": "test_client_secret",
        "refresh_token": "test_refresh_token",
    }


@pytest.fixture
def configured_connector(mock_youtube_credentials):
    """Configured YouTube connector."""
    return YouTubeUploaderConnector(**mock_youtube_credentials)


@pytest.fixture
def unconfigured_connector():
    """Unconfigured YouTube connector."""
    return YouTubeUploaderConnector()


@pytest.fixture
def circuit_breaker():
    """Fresh circuit breaker instance."""
    return CircuitBreaker(failure_threshold=3, cooldown_seconds=300.0)


@pytest.fixture
def rate_limiter():
    """Fresh rate limiter instance."""
    return YouTubeRateLimiter(daily_quota=10000)


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a temporary video file."""
    video_path = tmp_path / "test_video.mp4"
    # Create small test video content
    video_path.write_bytes(b"\x00\x00\x00\x1c\x66\x74\x79\x70" + b"\x00" * 100)
    return video_path


@pytest.fixture
def sample_metadata():
    """Sample video metadata."""
    return YouTubeVideoMetadata(
        title="Test Video",
        description="Test description",
        tags=["test", "video"],
    )


@pytest.fixture
def mock_token_response():
    """Mock token exchange response."""
    return {
        "access_token": "test_access_token",
        "expires_in": 3600,
        "refresh_token": "new_refresh_token",
        "token_type": "Bearer",
    }


@pytest.fixture
def mock_upload_response():
    """Mock successful upload response."""
    return {
        "id": "dQw4w9WgXcQ",
        "status": {"uploadStatus": "uploaded"},
    }


# =============================================================================
# YouTubeVideoMetadata Tests
# =============================================================================


class TestYouTubeVideoMetadata:
    """Tests for YouTubeVideoMetadata dataclass."""

    def test_title_truncation_at_100_chars(self):
        """Test that title is truncated at 100 characters."""
        long_title = "A" * 150
        metadata = YouTubeVideoMetadata(title=long_title, description="Test")

        assert len(metadata.title) <= MAX_TITLE_LENGTH
        assert metadata.title.endswith("...")

    def test_description_truncation_at_5000_chars(self):
        """Test that description is truncated at 5000 characters."""
        long_desc = "B" * 6000
        metadata = YouTubeVideoMetadata(title="Test", description=long_desc)

        assert len(metadata.description) <= MAX_DESCRIPTION_LENGTH
        assert metadata.description.endswith("...")

    def test_tags_truncation_at_500_total_chars(self):
        """Test that tags are truncated when total exceeds 500 chars."""
        # Create tags that exceed 500 chars total
        many_tags = [f"tag{i:05d}" for i in range(100)]  # Each ~8 chars
        metadata = YouTubeVideoMetadata(title="Test", description="Test", tags=many_tags)

        total_len = sum(len(tag) for tag in metadata.tags)
        assert total_len <= MAX_TAGS_LENGTH

    def test_to_api_body_structure_correct(self):
        """Test that to_api_body returns correct structure."""
        metadata = YouTubeVideoMetadata(
            title="Test Title",
            description="Test Description",
            tags=["tag1", "tag2"],
            category_id="22",
            privacy_status="unlisted",
            made_for_kids=False,
            language="en",
        )

        body = metadata.to_api_body()

        assert "snippet" in body
        assert "status" in body
        assert body["snippet"]["title"] == "Test Title"
        assert body["snippet"]["description"] == "Test Description"
        assert body["snippet"]["tags"] == ["tag1", "tag2"]
        assert body["snippet"]["categoryId"] == "22"
        assert body["snippet"]["defaultLanguage"] == "en"
        assert body["status"]["privacyStatus"] == "unlisted"
        assert body["status"]["selfDeclaredMadeForKids"] is False


# =============================================================================
# YouTubeRateLimiter Tests
# =============================================================================


class TestYouTubeRateLimiter:
    """Tests for YouTubeRateLimiter class."""

    def test_can_upload_checks_1600_unit_cost(self, rate_limiter):
        """Test that can_upload checks for 1600 quota units."""
        assert rate_limiter.can_upload() is True

        # Use most of quota
        rate_limiter.used_quota = 9000
        assert rate_limiter.can_upload() is False  # 9000 + 1600 > 10000

    def test_quota_resets_after_24_hours(self, rate_limiter):
        """Test that quota resets after 24 hours."""
        rate_limiter.used_quota = 9000
        rate_limiter.reset_time = time.time() - 1  # In the past

        # Should reset and allow upload
        assert rate_limiter.can_upload() is True
        assert rate_limiter.used_quota == 0

    def test_record_api_calls_with_custom_units(self, rate_limiter):
        """Test recording API calls with custom unit counts."""
        rate_limiter.record_api_call(100)
        assert rate_limiter.used_quota == 100

        rate_limiter.record_api_call(50)
        assert rate_limiter.used_quota == 150

    def test_remaining_quota_property(self, rate_limiter):
        """Test remaining_quota property."""
        assert rate_limiter.remaining_quota == 10000

        rate_limiter.used_quota = 3000
        assert rate_limiter.remaining_quota == 7000

    def test_record_upload_adds_1600_units(self, rate_limiter):
        """Test that record_upload adds 1600 units."""
        rate_limiter.record_upload()
        assert rate_limiter.used_quota == 1600


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_opens_after_3_failures(self, circuit_breaker):
        """Test that circuit opens after 3 failures."""
        assert circuit_breaker.is_open is False

        for i in range(2):
            circuit_breaker.record_failure()
            assert circuit_breaker.is_open is False

        circuit_breaker.record_failure()  # 3rd failure
        assert circuit_breaker.is_open is True

    def test_blocks_when_open(self, circuit_breaker):
        """Test that can_proceed returns False when open."""
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.is_open is True
        assert circuit_breaker.can_proceed() is False

    def test_recovery_attempt_after_300s(self, circuit_breaker):
        """Test that recovery is attempted after 300 seconds."""
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.can_proceed() is False

        # Simulate 300+ seconds passing (use internal property for single-entity mode)
        circuit_breaker._single_open_at = time.time() - 301

        assert circuit_breaker.can_proceed() is True

    def test_success_resets_circuit(self, circuit_breaker):
        """Test that success resets the circuit breaker."""
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.is_open is True

        circuit_breaker.record_success()

        assert circuit_breaker.is_open is False
        assert circuit_breaker.failures == 0


# =============================================================================
# OAuth & Authentication Tests
# =============================================================================


class TestOAuthAuthentication:
    """Tests for OAuth 2.0 authentication."""

    def test_get_auth_url_includes_state_csrf(self, configured_connector):
        """Test that auth URL includes state parameter for CSRF protection."""
        url = configured_connector.get_auth_url(
            redirect_uri="http://localhost/callback",
            state="random_csrf_token",
        )

        assert "state=random_csrf_token" in url
        assert configured_connector.AUTH_URL in url

    def test_get_auth_url_includes_youtube_upload_scope(self, configured_connector):
        """Test that auth URL includes youtube.upload scope."""
        url = configured_connector.get_auth_url(redirect_uri="http://localhost/callback")

        assert "youtube.upload" in url
        assert "scope=" in url

    @pytest.mark.asyncio
    async def test_exchange_code_success_stores_tokens(
        self, configured_connector, mock_token_response
    ):
        """Test that successful code exchange stores tokens."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_token_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await configured_connector.exchange_code(
                code="auth_code_123",
                redirect_uri="http://localhost/callback",
            )

            assert result is not None
            assert result["access_token"] == "test_access_token"
            assert configured_connector.refresh_token == "new_refresh_token"

    @pytest.mark.asyncio
    async def test_exchange_code_failure_raises_error(self, configured_connector):
        """Test that failed code exchange raises YouTubeAuthError."""
        from aragora.connectors.youtube_uploader import YouTubeAuthError

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Invalid code"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            with pytest.raises(YouTubeAuthError):
                await configured_connector.exchange_code(
                    code="bad_code",
                    redirect_uri="http://localhost/callback",
                )

    @pytest.mark.asyncio
    async def test_token_refresh_uses_5_min_buffer(self, configured_connector):
        """Test that token refresh uses 5 minute buffer (expires_in - 300)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "refreshed_token",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            with patch("time.time", return_value=1000000):
                result = await configured_connector._refresh_access_token()

                assert result is True
                # Expiry should be 3600 - 300 = 3300 seconds from now
                assert configured_connector._token_expiry == 1000000 + 3300


# =============================================================================
# Upload Validation Tests
# =============================================================================


class TestUploadValidation:
    """Tests for upload validation."""

    @pytest.mark.asyncio
    async def test_error_when_credentials_missing(
        self, unconfigured_connector, sample_metadata, temp_video_file
    ):
        """Test that upload returns error when credentials missing."""
        result = await unconfigured_connector.upload(temp_video_file, sample_metadata)

        assert result.success is False
        assert "credentials not configured" in result.error

    @pytest.mark.asyncio
    async def test_error_when_video_file_missing(
        self, configured_connector, sample_metadata, tmp_path
    ):
        """Test that upload returns error when video file doesn't exist."""
        missing_file = tmp_path / "nonexistent.mp4"

        # Need to mock token first
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        result = await configured_connector.upload(missing_file, sample_metadata)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_upload(
        self, configured_connector, sample_metadata, temp_video_file
    ):
        """Test that upload is blocked when circuit breaker is open."""
        # Open the circuit breaker
        for _ in range(3):
            configured_connector.circuit_breaker.record_failure()

        result = await configured_connector.upload(temp_video_file, sample_metadata)

        assert result.success is False
        assert "Circuit breaker" in result.error


# =============================================================================
# Upload Flow Tests
# =============================================================================


class TestUploadFlow:
    """Tests for upload flow."""

    @pytest.mark.asyncio
    async def test_upload_success_init_200_transfer_201(
        self, configured_connector, sample_metadata, temp_video_file, mock_upload_response
    ):
        """Test successful upload: init (200 + Location) -> transfer (201)."""
        # Mock access token
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        init_response = MagicMock()
        init_response.status_code = 200
        init_response.headers = {"Location": "https://upload.youtube.com/resumable/123"}

        upload_response = MagicMock()
        upload_response.status_code = 201
        upload_response.json.return_value = mock_upload_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = init_response
            mock_instance.put.return_value = upload_response

            result = await configured_connector.upload(temp_video_file, sample_metadata)

            assert result.success is True
            assert result.video_id == "dQw4w9WgXcQ"

    @pytest.mark.asyncio
    async def test_returns_correct_video_url_format(
        self, configured_connector, sample_metadata, temp_video_file, mock_upload_response
    ):
        """Test that successful upload returns correct video URL."""
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        init_response = MagicMock()
        init_response.status_code = 200
        init_response.headers = {"Location": "https://upload.youtube.com/resumable/123"}

        upload_response = MagicMock()
        upload_response.status_code = 201
        upload_response.json.return_value = mock_upload_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = init_response
            mock_instance.put.return_value = upload_response

            result = await configured_connector.upload(temp_video_file, sample_metadata)

            assert result.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    @pytest.mark.asyncio
    async def test_init_failure_403_records_circuit_failure(
        self, configured_connector, sample_metadata, temp_video_file
    ):
        """Test that init failure (403) records circuit breaker failure."""
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        init_failures = configured_connector.circuit_breaker.failures

        init_response = MagicMock()
        init_response.status_code = 403

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = init_response

            result = await configured_connector.upload(temp_video_file, sample_metadata)

            assert result.success is False
            assert configured_connector.circuit_breaker.failures == init_failures + 1

    @pytest.mark.asyncio
    async def test_missing_location_header_returns_error(
        self, configured_connector, sample_metadata, temp_video_file
    ):
        """Test that missing Location header returns error."""
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        init_response = MagicMock()
        init_response.status_code = 200
        init_response.headers = {}  # No Location header

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = init_response

            result = await configured_connector.upload(temp_video_file, sample_metadata)

            assert result.success is False
            assert "No upload URL" in result.error


# =============================================================================
# Video Status & Factory Tests
# =============================================================================


class TestVideoStatusAndFactory:
    """Tests for video status and metadata factory."""

    @pytest.mark.asyncio
    async def test_get_video_status_success_with_items_array(self, configured_connector):
        """Test get_video_status returns first item from items array."""
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [{"id": "abc123", "status": {"uploadStatus": "processed"}}]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.get.return_value = mock_response

            result = await configured_connector.get_video_status("abc123")

            assert result is not None
            assert result["id"] == "abc123"

    @pytest.mark.asyncio
    async def test_get_video_status_raises_error_when_token_unavailable(self, configured_connector):
        """Test that get_video_status raises error when token unavailable."""
        from aragora.connectors.youtube_uploader import YouTubeAuthError

        # No access token set
        configured_connector._access_token = None
        configured_connector._token_expiry = None
        configured_connector.refresh_token = ""  # Also no refresh token

        with pytest.raises(YouTubeAuthError):
            await configured_connector.get_video_status("abc123")

    def test_create_video_metadata_from_debate_factory_works(self):
        """Test create_video_metadata_from_debate factory function."""
        metadata = create_video_metadata_from_debate(
            task="Should AI be regulated?",
            agents=["Claude", "GPT-4", "Gemini"],
            consensus_reached=True,
            debate_id="debate_123",
        )

        assert isinstance(metadata, YouTubeVideoMetadata)
        assert "Should AI be regulated?" in metadata.title
        assert "Claude" in metadata.description
        assert "Consensus reached" in metadata.description
        assert "AI" in metadata.tags
        assert metadata.category_id == "28"  # Science & Technology

    def test_create_video_metadata_handles_missing_fields_gracefully(self):
        """Test that factory handles minimal input gracefully."""
        metadata = create_video_metadata_from_debate(
            task="Test topic",
            agents=[],
            consensus_reached=False,
            debate_id="test_123",
        )

        assert metadata.title is not None
        assert "No consensus" in metadata.description
        assert len(metadata.tags) > 0


# =============================================================================
# UploadResult Tests
# =============================================================================


class TestUploadResult:
    """Tests for UploadResult dataclass."""

    def test_creates_successful_result(self):
        """Test creating a successful upload result."""
        result = UploadResult(
            video_id="abc123",
            title="Test Video",
            url="https://www.youtube.com/watch?v=abc123",
            success=True,
            upload_status="complete",
        )
        assert result.success is True
        assert result.video_id == "abc123"
        assert result.error is None

    def test_creates_failed_result(self):
        """Test creating a failed upload result."""
        result = UploadResult(
            video_id="",
            title="Test Video",
            url="",
            success=False,
            error="Upload failed",
        )
        assert result.success is False
        assert result.error == "Upload failed"


# =============================================================================
# Integration Tests
# =============================================================================


class TestYouTubeConnectorIntegration:
    """Integration tests for YouTube connector."""

    @pytest.mark.asyncio
    async def test_successful_upload_records_quota_usage(
        self, configured_connector, sample_metadata, temp_video_file, mock_upload_response
    ):
        """Test that successful upload records quota usage."""
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        initial_quota = configured_connector.rate_limiter.used_quota

        init_response = MagicMock()
        init_response.status_code = 200
        init_response.headers = {"Location": "https://upload.youtube.com/resumable/123"}

        upload_response = MagicMock()
        upload_response.status_code = 201
        upload_response.json.return_value = mock_upload_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = init_response
            mock_instance.put.return_value = upload_response

            await configured_connector.upload(temp_video_file, sample_metadata)

            assert configured_connector.rate_limiter.used_quota == initial_quota + 1600

    @pytest.mark.asyncio
    async def test_quota_exceeded_blocks_upload(
        self, configured_connector, sample_metadata, temp_video_file
    ):
        """Test that upload is blocked when quota exceeded."""
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        # Use most of quota - also set reset_time so it doesn't get reset
        configured_connector.rate_limiter.used_quota = 9000
        configured_connector.rate_limiter.reset_time = time.time() + 86400

        result = await configured_connector.upload(temp_video_file, sample_metadata)

        assert result.success is False
        assert "quota exceeded" in result.error.lower()

    @pytest.mark.asyncio
    async def test_successful_upload_resets_circuit_breaker(
        self, configured_connector, sample_metadata, temp_video_file, mock_upload_response
    ):
        """Test that successful upload resets circuit breaker."""
        configured_connector._access_token = "test_token"
        configured_connector._token_expiry = time.time() + 3600

        # Record some failures (but not enough to open)
        configured_connector.circuit_breaker.record_failure()
        configured_connector.circuit_breaker.record_failure()

        init_response = MagicMock()
        init_response.status_code = 200
        init_response.headers = {"Location": "https://upload.youtube.com/resumable/123"}

        upload_response = MagicMock()
        upload_response.status_code = 201
        upload_response.json.return_value = mock_upload_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = init_response
            mock_instance.put.return_value = upload_response

            result = await configured_connector.upload(temp_video_file, sample_metadata)

            assert result.success is True
            assert configured_connector.circuit_breaker.failures == 0
            assert configured_connector.circuit_breaker.is_open is False

    def test_is_configured_with_all_credentials(self, configured_connector):
        """Test is_configured returns True with all credentials."""
        assert configured_connector.is_configured is True

    def test_is_configured_false_when_missing_credentials(self, unconfigured_connector):
        """Test is_configured returns False when missing credentials."""
        assert unconfigured_connector.is_configured is False
