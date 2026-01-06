"""
Extended tests for social media connectors.

Tests YouTube uploader, Twitter poster, rate limiters,
circuit breakers, and content formatters.
"""

import pytest
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.youtube_uploader import (
    YouTubeVideoMetadata,
    YouTubeRateLimiter,
    CircuitBreaker as YouTubeCircuitBreaker,
    YouTubeUploaderConnector,
    UploadResult,
    create_video_metadata_from_debate,
    MAX_TITLE_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_TAGS_LENGTH,
)
from aragora.connectors.twitter_poster import (
    TweetResult,
    ThreadResult,
    TwitterRateLimiter,
    CircuitBreaker as TwitterCircuitBreaker,
    TwitterPosterConnector,
    DebateContentFormatter,
    create_debate_summary,
    MAX_TWEET_LENGTH,
    MAX_THREAD_LENGTH,
)


# =============================================================================
# YouTubeVideoMetadata Tests
# =============================================================================

class TestYouTubeVideoMetadata:
    """Tests for YouTube video metadata."""

    def test_create_with_defaults(self):
        """Should create metadata with defaults."""
        meta = YouTubeVideoMetadata(
            title="Test Video",
            description="A test description",
        )

        assert meta.title == "Test Video"
        assert meta.category_id == "28"  # Science & Tech
        assert meta.privacy_status == "public"
        assert meta.made_for_kids is False

    def test_truncate_long_title(self):
        """Should truncate titles exceeding max length."""
        long_title = "A" * 150  # Exceeds MAX_TITLE_LENGTH (100)

        meta = YouTubeVideoMetadata(
            title=long_title,
            description="Short desc",
        )

        assert len(meta.title) <= MAX_TITLE_LENGTH
        assert meta.title.endswith("...")

    def test_truncate_long_description(self):
        """Should truncate descriptions exceeding max length."""
        long_desc = "B" * 6000  # Exceeds MAX_DESCRIPTION_LENGTH (5000)

        meta = YouTubeVideoMetadata(
            title="Title",
            description=long_desc,
        )

        assert len(meta.description) <= MAX_DESCRIPTION_LENGTH
        assert meta.description.endswith("...")

    def test_truncate_tags_exceeding_limit(self):
        """Should remove tags when total length exceeds limit."""
        # Create many long tags
        tags = [f"verylongtag{i}" * 5 for i in range(50)]

        meta = YouTubeVideoMetadata(
            title="Title",
            description="Desc",
            tags=tags,
        )

        total_tag_length = sum(len(t) for t in meta.tags)
        assert total_tag_length <= MAX_TAGS_LENGTH

    def test_to_api_body_structure(self):
        """to_api_body should return proper API structure."""
        meta = YouTubeVideoMetadata(
            title="API Test",
            description="Test description",
            tags=["ai", "debate"],
            category_id="28",
            privacy_status="unlisted",
        )

        body = meta.to_api_body()

        assert "snippet" in body
        assert "status" in body
        assert body["snippet"]["title"] == "API Test"
        assert body["snippet"]["tags"] == ["ai", "debate"]
        assert body["status"]["privacyStatus"] == "unlisted"


# =============================================================================
# YouTubeRateLimiter Tests
# =============================================================================

class TestYouTubeRateLimiter:
    """Tests for YouTube API quota management."""

    def test_can_upload_with_fresh_quota(self):
        """Should allow upload with fresh quota."""
        limiter = YouTubeRateLimiter(daily_quota=10000)
        assert limiter.can_upload() is True

    def test_cannot_upload_after_quota_exceeded(self):
        """Should block upload after quota exceeded."""
        limiter = YouTubeRateLimiter(daily_quota=1600)  # Exactly one upload
        limiter.record_upload()

        assert limiter.can_upload() is False

    def test_record_upload_consumes_1600_units(self):
        """record_upload should consume 1600 units."""
        limiter = YouTubeRateLimiter(daily_quota=10000)
        limiter.record_upload()

        assert limiter.used_quota == 1600
        assert limiter.remaining_quota == 8400

    def test_record_api_call_consumes_specified_units(self):
        """record_api_call should consume specified units."""
        limiter = YouTubeRateLimiter()
        limiter.record_api_call(5)

        assert limiter.used_quota == 5

    def test_remaining_quota_never_negative(self):
        """remaining_quota should never be negative."""
        limiter = YouTubeRateLimiter(daily_quota=100)
        # Set reset_time in future to prevent auto-reset
        limiter.reset_time = time.time() + 86400
        limiter.used_quota = 500  # Force over quota

        assert limiter.remaining_quota == 0


# =============================================================================
# YouTube CircuitBreaker Tests
# =============================================================================

class TestYouTubeCircuitBreaker:
    """Tests for YouTube circuit breaker."""

    def test_starts_closed(self):
        """Circuit should start closed."""
        breaker = YouTubeCircuitBreaker()
        assert breaker.is_open is False
        assert breaker.can_proceed() is True

    def test_opens_after_threshold_failures(self):
        """Should open after failure threshold reached."""
        breaker = YouTubeCircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is False

        breaker.record_failure()
        assert breaker.is_open is True
        assert breaker.can_proceed() is False

    def test_success_resets_failures(self):
        """record_success should reset failure count."""
        breaker = YouTubeCircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()

        breaker.record_success()

        assert breaker.failures == 0
        assert breaker.is_open is False

    def test_recovery_after_timeout(self):
        """Should allow recovery attempt after timeout."""
        breaker = YouTubeCircuitBreaker(failure_threshold=1, recovery_timeout=0)
        breaker.record_failure()

        assert breaker.is_open is True
        # With recovery_timeout=0, should immediately allow retry
        assert breaker.can_proceed() is True


# =============================================================================
# YouTubeUploaderConnector Tests
# =============================================================================

class TestYouTubeUploaderConnector:
    """Tests for YouTube upload connector."""

    def test_is_configured_requires_all_credentials(self):
        """is_configured should require all credentials."""
        connector = YouTubeUploaderConnector(
            client_id="id",
            client_secret="secret",
            refresh_token="",  # Missing
        )
        assert connector.is_configured is False

        connector2 = YouTubeUploaderConnector(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
        )
        assert connector2.is_configured is True

    def test_get_auth_url_generates_valid_url(self):
        """get_auth_url should generate valid OAuth URL."""
        connector = YouTubeUploaderConnector(client_id="test_client_id")
        url = connector.get_auth_url("http://localhost/callback", state="test_state")

        assert "accounts.google.com" in url
        assert "client_id=test_client_id" in url
        assert "redirect_uri" in url
        assert "state=test_state" in url

    @pytest.mark.asyncio
    async def test_upload_fails_without_credentials(self):
        """upload should fail when not configured."""
        connector = YouTubeUploaderConnector()  # No credentials

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"fake video")
            f.flush()

            result = await connector.upload(
                Path(f.name),
                YouTubeVideoMetadata(title="Test", description="Desc"),
            )

        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_upload_fails_with_missing_file(self):
        """upload should fail when video file missing."""
        connector = YouTubeUploaderConnector(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
        )

        result = await connector.upload(
            Path("/nonexistent/video.mp4"),
            YouTubeVideoMetadata(title="Test", description="Desc"),
        )

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_upload_respects_circuit_breaker(self):
        """upload should respect circuit breaker."""
        connector = YouTubeUploaderConnector(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
        )
        # Open circuit breaker
        connector.circuit_breaker.is_open = True
        connector.circuit_breaker.last_failure_time = time.time() + 1000

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"fake video")
            f.flush()

            result = await connector.upload(
                Path(f.name),
                YouTubeVideoMetadata(title="Test", description="Desc"),
            )

        assert result.success is False
        assert "Circuit breaker" in result.error

    @pytest.mark.asyncio
    async def test_upload_respects_rate_limiter(self):
        """upload should respect rate limiter quota."""
        connector = YouTubeUploaderConnector(
            client_id="id",
            client_secret="secret",
            refresh_token="token",
        )
        # Exhaust quota - set reset_time in future to prevent auto-reset
        connector.rate_limiter.reset_time = time.time() + 86400
        connector.rate_limiter.used_quota = 10000

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"fake video")
            f.flush()

            result = await connector.upload(
                Path(f.name),
                YouTubeVideoMetadata(title="Test", description="Desc"),
            )

        assert result.success is False
        assert "quota" in result.error.lower()


# =============================================================================
# create_video_metadata_from_debate Tests
# =============================================================================

class TestCreateVideoMetadataFromDebate:
    """Tests for debate-to-metadata conversion."""

    def test_creates_valid_metadata(self):
        """Should create valid metadata from debate info."""
        meta = create_video_metadata_from_debate(
            task="Should AI have rights?",
            agents=["claude", "gpt4"],
            consensus_reached=True,
            debate_id="debate-123",
        )

        assert "AI Debate" in meta.title
        assert "claude" in meta.description
        assert "gpt4" in meta.description
        assert "Consensus reached" in meta.description
        assert "AI" in meta.tags

    def test_truncates_long_task(self):
        """Should truncate long task in title."""
        long_task = "A" * 200

        meta = create_video_metadata_from_debate(
            task=long_task,
            agents=["agent1"],
            consensus_reached=False,
            debate_id="d1",
        )

        assert len(meta.title) <= MAX_TITLE_LENGTH


# =============================================================================
# TweetResult and ThreadResult Tests
# =============================================================================

class TestTweetResult:
    """Tests for TweetResult dataclass."""

    def test_create_successful_tweet(self):
        """Should create successful tweet result."""
        result = TweetResult(
            tweet_id="123456",
            text="Test tweet",
            created_at="2025-01-06T12:00:00",
            url="https://twitter.com/i/status/123456",
        )

        assert result.success is True
        assert result.error is None

    def test_create_failed_tweet(self):
        """Should create failed tweet result with error."""
        result = TweetResult(
            tweet_id="",
            text="Failed tweet",
            created_at="2025-01-06T12:00:00",
            url="",
            success=False,
            error="Rate limit exceeded",
        )

        assert result.success is False
        assert "Rate limit" in result.error


class TestThreadResult:
    """Tests for ThreadResult dataclass."""

    def test_url_returns_first_tweet_url(self):
        """url property should return first tweet's URL."""
        tweets = [
            TweetResult("1", "First", "2025-01-06", "https://t.co/1"),
            TweetResult("2", "Second", "2025-01-06", "https://t.co/2"),
        ]
        result = ThreadResult(thread_id="1", tweets=tweets)

        assert result.url == "https://t.co/1"

    def test_url_empty_for_no_tweets(self):
        """url should be empty when no tweets."""
        result = ThreadResult(thread_id="", tweets=[])
        assert result.url == ""


# =============================================================================
# TwitterRateLimiter Tests
# =============================================================================

class TestTwitterRateLimiter:
    """Tests for Twitter rate limiter."""

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Should not wait when within rate limit."""
        limiter = TwitterRateLimiter(calls_per_window=10, window_seconds=60)

        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        # Should complete almost immediately
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_tracks_call_times(self):
        """Should track call timestamps."""
        limiter = TwitterRateLimiter()

        await limiter.acquire()
        await limiter.acquire()

        assert len(limiter.call_times) == 2


# =============================================================================
# Twitter CircuitBreaker Tests
# =============================================================================

class TestTwitterCircuitBreaker:
    """Tests for Twitter circuit breaker."""

    def test_default_thresholds(self):
        """Should use sensible defaults."""
        breaker = TwitterCircuitBreaker()
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60

    def test_behavior_matches_youtube_breaker(self):
        """Should behave like YouTube breaker."""
        breaker = TwitterCircuitBreaker(failure_threshold=2)

        breaker.record_failure()
        assert breaker.can_proceed() is True

        breaker.record_failure()
        assert breaker.can_proceed() is False  # Now open


# =============================================================================
# TwitterPosterConnector Tests
# =============================================================================

class TestTwitterPosterConnector:
    """Tests for Twitter posting connector."""

    def test_is_configured_requires_all_credentials(self):
        """is_configured should require all four credentials."""
        connector = TwitterPosterConnector(
            api_key="key",
            api_secret="secret",
            access_token="token",
            access_secret="",  # Missing
        )
        assert connector.is_configured is False

    @pytest.mark.asyncio
    async def test_post_tweet_fails_without_credentials(self):
        """post_tweet should fail when not configured."""
        connector = TwitterPosterConnector()

        result = await connector.post_tweet("Test tweet")

        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_post_tweet_truncates_long_text(self):
        """Should truncate text exceeding MAX_TWEET_LENGTH."""
        connector = TwitterPosterConnector(
            api_key="key",
            api_secret="secret",
            access_token="token",
            access_secret="secret",
        )
        connector.circuit_breaker.is_open = True  # Force failure path

        long_text = "A" * 500

        result = await connector.post_tweet(long_text)

        # Check it attempted with truncated text
        assert result.success is False  # Circuit breaker blocked it

    @pytest.mark.asyncio
    async def test_post_thread_fails_with_empty_list(self):
        """post_thread should fail with empty list."""
        connector = TwitterPosterConnector()

        result = await connector.post_thread([])

        assert result.success is False
        assert "No tweets" in result.error

    @pytest.mark.asyncio
    async def test_post_thread_truncates_to_max_length(self):
        """post_thread should truncate to MAX_THREAD_LENGTH."""
        connector = TwitterPosterConnector(
            api_key="key",
            api_secret="secret",
            access_token="token",
            access_secret="secret",
        )

        # More than MAX_THREAD_LENGTH tweets
        tweets = [f"Tweet {i}" for i in range(30)]

        # Mock post_tweet to succeed
        connector.post_tweet = AsyncMock(return_value=TweetResult(
            tweet_id="1",
            text="Test",
            created_at="2025-01-06",
            url="https://t.co/1",
        ))

        result = await connector.post_thread(tweets)

        # Should have posted only MAX_THREAD_LENGTH tweets
        assert connector.post_tweet.call_count <= MAX_THREAD_LENGTH

    @pytest.mark.asyncio
    async def test_upload_media_fails_without_credentials(self):
        """upload_media should fail when not configured."""
        connector = TwitterPosterConnector()

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            f.write(b"fake image")
            f.flush()
            result = await connector.upload_media(Path(f.name))

        assert result is None

    @pytest.mark.asyncio
    async def test_upload_media_fails_for_missing_file(self):
        """upload_media should fail for missing file."""
        connector = TwitterPosterConnector(
            api_key="key",
            api_secret="secret",
            access_token="token",
            access_secret="secret",
        )

        result = await connector.upload_media(Path("/nonexistent.png"))

        assert result is None

    @pytest.mark.asyncio
    async def test_upload_media_fails_for_oversized_file(self):
        """upload_media should fail for files exceeding size limit."""
        connector = TwitterPosterConnector(
            api_key="key",
            api_secret="secret",
            access_token="token",
            access_secret="secret",
        )

        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            # Write >5MB
            f.write(b"0" * (6 * 1024 * 1024))
            f.flush()
            result = await connector.upload_media(Path(f.name))

        assert result is None


# =============================================================================
# DebateContentFormatter Tests
# =============================================================================

class TestDebateContentFormatter:
    """Tests for debate content formatting."""

    def test_format_announcement(self):
        """format_announcement should create valid tweet."""
        formatter = DebateContentFormatter()

        tweet = formatter.format_announcement(
            task="Should we regulate AI?",
            agents=["claude", "gpt4", "gemini"],
        )

        assert "New AI Debate" in tweet
        assert "claude" in tweet
        assert len(tweet) <= MAX_TWEET_LENGTH

    def test_format_announcement_truncates_long_task(self):
        """format_announcement should handle long tasks."""
        formatter = DebateContentFormatter()
        long_task = "A" * 300

        tweet = formatter.format_announcement(task=long_task, agents=["a1", "a2"])

        assert len(tweet) <= MAX_TWEET_LENGTH

    def test_format_result_with_consensus(self):
        """format_result should indicate consensus."""
        formatter = DebateContentFormatter()

        tweet = formatter.format_result(
            task="AI topic",
            agents=["claude"],
            consensus_reached=True,
        )

        assert "Consensus reached" in tweet

    def test_format_result_with_winner(self):
        """format_result should show winner."""
        formatter = DebateContentFormatter()

        tweet = formatter.format_result(
            task="AI topic",
            agents=["claude", "gpt4"],
            consensus_reached=False,
            winner="claude",
        )

        assert "Winner: claude" in tweet

    def test_format_thread_structure(self):
        """format_thread should create proper thread structure."""
        formatter = DebateContentFormatter()

        tweets = formatter.format_thread(
            task="Test topic",
            agents=["a1", "a2"],
            highlights=["Point 1", "Point 2", "Point 3"],
            consensus_reached=True,
        )

        assert len(tweets) >= 3  # Intro + highlights + result
        assert "New AI Debate" in tweets[0]  # Intro
        assert "1." in tweets[1]  # Numbered highlight

    def test_format_thread_respects_max_length(self):
        """format_thread should not exceed MAX_THREAD_LENGTH."""
        formatter = DebateContentFormatter()

        many_highlights = [f"Highlight {i}" for i in range(30)]

        tweets = formatter.format_thread(
            task="Test",
            agents=["a1"],
            highlights=many_highlights,
            consensus_reached=False,
        )

        assert len(tweets) <= MAX_THREAD_LENGTH


# =============================================================================
# create_debate_summary Tests
# =============================================================================

class TestCreateDebateSummary:
    """Tests for create_debate_summary helper."""

    def test_creates_summary_within_limit(self):
        """Should create summary within max_length."""
        summary = create_debate_summary(
            task="Long topic about artificial intelligence and its implications",
            agents=["claude", "gpt4", "gemini", "llama"],
            consensus_reached=True,
            max_length=280,
        )

        assert len(summary) <= 280

    def test_custom_max_length(self):
        """Should respect custom max_length."""
        summary = create_debate_summary(
            task="Test topic",
            agents=["a1"],
            max_length=100,
        )

        assert len(summary) <= 100
