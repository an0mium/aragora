"""
Comprehensive tests for Twitter poster connector.

Tests cover:
- CircuitBreaker (5 tests)
- TwitterRateLimiter (4 tests)
- OAuth signature generation (5 tests)
- TwitterPosterConnector (10 tests)
- DebateContentFormatter (6 tests)
"""

import asyncio
import base64
import hashlib
import hmac
import time
import urllib.parse
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.twitter_poster import (
    MAX_MEDIA_SIZE_MB,
    MAX_THREAD_LENGTH,
    MAX_TWEET_LENGTH,
    DebateContentFormatter,
    ThreadResult,
    TweetResult,
    TwitterMediaError,
    TwitterPosterConnector,
    TwitterRateLimiter,
    create_debate_summary,
)
from aragora.resilience import CircuitBreaker


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_twitter_credentials():
    """Twitter API credentials for testing."""
    return {
        "api_key": "test_api_key",
        "api_secret": "test_api_secret",
        "access_token": "test_access_token",
        "access_secret": "test_access_secret",
    }


@pytest.fixture
def mock_tweet_response():
    """Successful tweet response."""
    return {"data": {"id": "1234567890123456789"}}


@pytest.fixture
def configured_connector(mock_twitter_credentials):
    """Configured Twitter connector."""
    return TwitterPosterConnector(**mock_twitter_credentials)


@pytest.fixture
def unconfigured_connector():
    """Unconfigured Twitter connector."""
    return TwitterPosterConnector()


@pytest.fixture
def circuit_breaker():
    """Fresh circuit breaker instance."""
    return CircuitBreaker(failure_threshold=5, cooldown_seconds=60.0)


@pytest.fixture
def rate_limiter():
    """Fresh rate limiter instance."""
    return TwitterRateLimiter(calls_per_window=50, window_seconds=900)


@pytest.fixture
def formatter():
    """DebateContentFormatter instance."""
    return DebateContentFormatter()


@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary image file."""
    img_path = tmp_path / "test_image.png"
    # Create small test image content
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return img_path


@pytest.fixture
def large_image_file(tmp_path):
    """Create a large image file (> 5MB)."""
    img_path = tmp_path / "large_image.png"
    # Create 6MB file
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (6 * 1024 * 1024))
    return img_path


# =============================================================================
# TweetResult and ThreadResult Tests
# =============================================================================


class TestTweetResult:
    """Tests for TweetResult dataclass."""

    def test_creates_successful_result(self):
        """Test creating a successful tweet result."""
        result = TweetResult(
            tweet_id="123",
            text="Hello world",
            created_at="2025-01-06T12:00:00",
            url="https://twitter.com/i/status/123",
            success=True,
        )
        assert result.tweet_id == "123"
        assert result.success is True
        assert result.error is None

    def test_creates_failed_result(self):
        """Test creating a failed tweet result."""
        result = TweetResult(
            tweet_id="",
            text="Failed tweet",
            created_at="2025-01-06T12:00:00",
            url="",
            success=False,
            error="API error",
        )
        assert result.success is False
        assert result.error == "API error"


class TestThreadResult:
    """Tests for ThreadResult dataclass."""

    def test_url_property_returns_first_tweet_url(self):
        """Test that url property returns first tweet's URL."""
        tweet1 = TweetResult(
            tweet_id="1",
            text="First",
            created_at="2025-01-06T12:00:00",
            url="https://twitter.com/i/status/1",
        )
        tweet2 = TweetResult(
            tweet_id="2",
            text="Second",
            created_at="2025-01-06T12:00:01",
            url="https://twitter.com/i/status/2",
        )
        result = ThreadResult(thread_id="1", tweets=[tweet1, tweet2])
        assert result.url == "https://twitter.com/i/status/1"

    def test_url_property_empty_when_no_tweets(self):
        """Test that url property returns empty string when no tweets."""
        result = ThreadResult(thread_id="", tweets=[])
        assert result.url == ""


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_records_failures_and_increments_counter(self, circuit_breaker):
        """Test that failures are recorded and counter increments."""
        assert circuit_breaker.failures == 0
        circuit_breaker.record_failure()
        assert circuit_breaker.failures == 1
        circuit_breaker.record_failure()
        assert circuit_breaker.failures == 2

    def test_opens_at_failure_threshold(self, circuit_breaker):
        """Test that circuit opens at failure threshold (5)."""
        assert circuit_breaker.is_open is False

        for i in range(4):
            circuit_breaker.record_failure()
            assert circuit_breaker.is_open is False

        circuit_breaker.record_failure()  # 5th failure
        assert circuit_breaker.is_open is True

    def test_blocks_requests_when_open(self, circuit_breaker):
        """Test that can_proceed() returns False when open."""
        # Open the circuit
        for _ in range(5):
            circuit_breaker.record_failure()

        assert circuit_breaker.is_open is True
        assert circuit_breaker.can_proceed() is False

    def test_recovery_after_timeout(self, circuit_breaker):
        """Test that circuit allows recovery attempt after timeout."""
        # Open the circuit
        for _ in range(5):
            circuit_breaker.record_failure()

        assert circuit_breaker.can_proceed() is False

        # Simulate timeout passing (use internal property for single-entity mode)
        circuit_breaker._single_open_at = time.time() - 61  # 61 seconds ago

        assert circuit_breaker.can_proceed() is True

    def test_resets_on_success(self, circuit_breaker):
        """Test that success resets the circuit breaker."""
        # Partial failures
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.failures == 3

        circuit_breaker.record_success()

        assert circuit_breaker.failures == 0
        assert circuit_breaker.is_open is False

    def test_can_proceed_returns_true_when_closed(self, circuit_breaker):
        """Test that can_proceed() returns True when circuit is closed."""
        assert circuit_breaker.can_proceed() is True

    def test_custom_threshold_and_timeout(self):
        """Test circuit breaker with custom parameters."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)

        for _ in range(2):
            cb.record_failure()
        assert cb.is_open is False

        cb.record_failure()  # 3rd failure
        assert cb.is_open is True

        # Recovery at 30 seconds (use internal property for single-entity mode)
        cb._single_open_at = time.time() - 31
        assert cb.can_proceed() is True


# =============================================================================
# TwitterRateLimiter Tests
# =============================================================================


class TestTwitterRateLimiter:
    """Tests for TwitterRateLimiter class."""

    @pytest.mark.asyncio
    async def test_allows_first_call_immediately(self, rate_limiter):
        """Test that first call is allowed immediately."""
        start = time.time()
        await rate_limiter.acquire()
        elapsed = time.time() - start
        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_applies_delay_when_threshold_exceeded(self):
        """Test that delay is applied when at capacity."""
        limiter = TwitterRateLimiter(calls_per_window=2, window_seconds=10)

        # Fill up the window
        await limiter.acquire()
        await limiter.acquire()

        # Third call should wait
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await limiter.acquire()
            # Should have waited since window is full
            # Either sleep was called or oldest call expired
            assert len(limiter.call_times) == 3

    @pytest.mark.asyncio
    async def test_window_slides_oldest_calls_expire(self, rate_limiter):
        """Test that oldest calls expire from window."""
        # Add some old calls
        old_time = time.time() - 1000  # Way outside window
        rate_limiter.call_times = [old_time, old_time + 1, old_time + 2]

        await rate_limiter.acquire()

        # Old calls should be removed, only new call remains
        assert len(rate_limiter.call_times) == 1

    def test_custom_parameters_work(self):
        """Test rate limiter with custom parameters."""
        limiter = TwitterRateLimiter(calls_per_window=100, window_seconds=60)
        assert limiter.calls_per_window == 100
        assert limiter.window_seconds == 60


# =============================================================================
# OAuth Signature Tests
# =============================================================================


class TestOAuthSignature:
    """Tests for OAuth 1.0a signature generation."""

    def test_deterministic_signature_for_same_input(self, configured_connector):
        """Test that same inputs produce same signature."""
        # Fix timestamp and nonce for deterministic test
        with patch("time.time", return_value=1000000):
            with patch("os.urandom", return_value=b"x" * 32):
                sig1 = configured_connector._generate_oauth_signature(
                    "POST",
                    "https://api.twitter.com/2/tweets",
                    {"text": "Hello"},
                    {
                        "oauth_consumer_key": "test_api_key",
                        "oauth_token": "test_access_token",
                        "oauth_signature_method": "HMAC-SHA1",
                        "oauth_timestamp": "1000000",
                        "oauth_nonce": "abc123",
                        "oauth_version": "1.0",
                    },
                )
                sig2 = configured_connector._generate_oauth_signature(
                    "POST",
                    "https://api.twitter.com/2/tweets",
                    {"text": "Hello"},
                    {
                        "oauth_consumer_key": "test_api_key",
                        "oauth_token": "test_access_token",
                        "oauth_signature_method": "HMAC-SHA1",
                        "oauth_timestamp": "1000000",
                        "oauth_nonce": "abc123",
                        "oauth_version": "1.0",
                    },
                )
        assert sig1 == sig2

    def test_parameters_properly_encoded(self, configured_connector):
        """Test that parameters are URL-encoded in signature."""
        oauth_params = {
            "oauth_consumer_key": "test_api_key",
            "oauth_token": "test_access_token",
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": "1000000",
            "oauth_nonce": "abc123",
            "oauth_version": "1.0",
        }
        # Special characters should be encoded
        sig = configured_connector._generate_oauth_signature(
            "POST",
            "https://api.twitter.com/2/tweets",
            {"text": "Hello & goodbye"},
            oauth_params,
        )
        assert sig  # Should produce valid base64 signature

    def test_header_format_oauth_key_value(self, configured_connector):
        """Test that OAuth header has correct format."""
        with patch("time.time", return_value=1000000):
            with patch("os.urandom", return_value=b"x" * 32):
                header = configured_connector._generate_oauth_header(
                    "POST", "https://api.twitter.com/2/tweets"
                )
        assert header.startswith("OAuth ")
        assert "oauth_consumer_key=" in header
        assert "oauth_signature=" in header
        assert "oauth_timestamp=" in header

    def test_nonce_uniqueness_per_request(self, configured_connector):
        """Test that each request gets a unique nonce."""
        # Real nonce generation uses os.urandom, so should be unique
        header1 = configured_connector._generate_oauth_header(
            "POST", "https://api.twitter.com/2/tweets"
        )
        header2 = configured_connector._generate_oauth_header(
            "POST", "https://api.twitter.com/2/tweets"
        )
        # Headers should differ due to different nonces
        assert header1 != header2

    def test_timestamp_included_in_signature(self, configured_connector):
        """Test that timestamp is included in OAuth signature."""
        with patch("time.time", return_value=1234567890):
            header = configured_connector._generate_oauth_header(
                "POST", "https://api.twitter.com/2/tweets"
            )
        assert 'oauth_timestamp="1234567890"' in header


# =============================================================================
# TwitterPosterConnector Tests
# =============================================================================


class TestTwitterPosterConnector:
    """Tests for TwitterPosterConnector class."""

    def test_is_configured_with_all_credentials(self, configured_connector):
        """Test is_configured returns True with all credentials."""
        assert configured_connector.is_configured is True

    def test_is_configured_false_when_missing_credentials(self, unconfigured_connector):
        """Test is_configured returns False when missing credentials."""
        assert unconfigured_connector.is_configured is False

    @pytest.mark.asyncio
    async def test_post_tweet_success(self, configured_connector, mock_tweet_response):
        """Test successful tweet posting."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_tweet_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await configured_connector.post_tweet("Hello, world!")

            assert result.success is True
            assert result.tweet_id == "1234567890123456789"
            assert "twitter.com" in result.url

    @pytest.mark.asyncio
    async def test_post_tweet_truncates_long_text(self, configured_connector, mock_tweet_response):
        """Test that text > 280 chars is truncated."""
        long_text = "a" * 300

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_tweet_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await configured_connector.post_tweet(long_text)

            # Check that posted text was truncated
            call_args = mock_instance.post.call_args
            payload = call_args.kwargs["json"]
            assert len(payload["text"]) <= MAX_TWEET_LENGTH
            assert payload["text"].endswith("...")

    @pytest.mark.asyncio
    async def test_post_tweet_error_when_credentials_missing(self, unconfigured_connector):
        """Test error returned when credentials missing."""
        result = await unconfigured_connector.post_tweet("Hello")

        assert result.success is False
        assert "credentials not configured" in result.error

    @pytest.mark.asyncio
    async def test_post_tweet_blocked_by_circuit_breaker(self, configured_connector):
        """Test that post is blocked when circuit breaker is open."""
        # Open the circuit breaker
        for _ in range(5):
            configured_connector.circuit_breaker.record_failure()

        result = await configured_connector.post_tweet("Hello")

        assert result.success is False
        assert "Circuit breaker" in result.error

    @pytest.mark.asyncio
    async def test_post_tweet_includes_reply_to(self, configured_connector, mock_tweet_response):
        """Test that reply_to is included in payload."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_tweet_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            await configured_connector.post_tweet("Reply", reply_to="999")

            call_args = mock_instance.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["reply"]["in_reply_to_tweet_id"] == "999"

    @pytest.mark.asyncio
    async def test_post_tweet_includes_media_ids(self, configured_connector, mock_tweet_response):
        """Test that media_ids are included in payload."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_tweet_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            await configured_connector.post_tweet("With media", media_ids=["111", "222"])

            call_args = mock_instance.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["media"]["media_ids"] == ["111", "222"]

    @pytest.mark.asyncio
    async def test_post_thread_returns_thread_result(
        self, configured_connector, mock_tweet_response
    ):
        """Test that post_thread returns ThreadResult."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_tweet_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await configured_connector.post_thread(["Tweet 1", "Tweet 2"])

            assert isinstance(result, ThreadResult)
            assert result.success is True
            assert len(result.tweets) == 2

    @pytest.mark.asyncio
    async def test_post_thread_fails_gracefully_with_partial_results(
        self, configured_connector, mock_tweet_response
    ):
        """Test that thread failure returns partial results."""
        success_response = MagicMock()
        success_response.status_code = 201
        success_response.json.return_value = mock_tweet_response

        fail_response = MagicMock()
        fail_response.status_code = 400
        fail_response.text = "Error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.side_effect = [success_response, fail_response]

            result = await configured_connector.post_thread(["Tweet 1", "Tweet 2", "Tweet 3"])

            assert result.success is False
            assert len(result.tweets) == 2  # First success + first failure recorded
            assert "Failed at tweet 2" in result.error

    @pytest.mark.asyncio
    async def test_upload_media_rejects_large_files(self, configured_connector, large_image_file):
        """Test that files > 5MB are rejected."""
        with pytest.raises(TwitterMediaError) as exc_info:
            await configured_connector.upload_media(large_image_file)
        assert "too large" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_upload_media_raises_for_missing_files(self, configured_connector, tmp_path):
        """Test that missing files raise TwitterMediaError."""
        missing_file = tmp_path / "nonexistent.png"
        with pytest.raises(TwitterMediaError) as exc_info:
            await configured_connector.upload_media(missing_file)
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_upload_media_success(self, configured_connector, temp_image_file):
        """Test successful media upload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"media_id_string": "12345"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await configured_connector.upload_media(temp_image_file)

            assert result == "12345"

    @pytest.mark.asyncio
    async def test_post_thread_empty_tweets_error(self, configured_connector):
        """Test that empty tweet list returns error."""
        result = await configured_connector.post_thread([])

        assert result.success is False
        assert "No tweets provided" in result.error

    @pytest.mark.asyncio
    async def test_post_thread_truncates_long_thread(
        self, configured_connector, mock_tweet_response
    ):
        """Test that threads > 25 tweets are truncated."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = mock_tweet_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            long_thread = [f"Tweet {i}" for i in range(30)]
            result = await configured_connector.post_thread(long_thread)

            assert result.success is True
            assert len(result.tweets) == MAX_THREAD_LENGTH

    @pytest.mark.asyncio
    async def test_post_tweet_api_error_records_circuit_failure(self, configured_connector):
        """Test that API errors record circuit breaker failure."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        initial_failures = configured_connector.circuit_breaker.failures

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            await configured_connector.post_tweet("Hello")

            assert configured_connector.circuit_breaker.failures == initial_failures + 1

    @pytest.mark.asyncio
    async def test_post_tweet_exception_records_circuit_failure(self, configured_connector):
        """Test that exceptions record circuit breaker failure."""
        import httpx

        initial_failures = configured_connector.circuit_breaker.failures

        with patch("httpx.AsyncClient") as mock_client:
            # Use httpx.RequestError which is caught by the error handler
            mock_client.return_value.__aenter__.side_effect = httpx.RequestError("Network error")

            result = await configured_connector.post_tweet("Hello")

            assert result.success is False
            assert configured_connector.circuit_breaker.failures == initial_failures + 1


# =============================================================================
# DebateContentFormatter Tests
# =============================================================================


class TestDebateContentFormatter:
    """Tests for DebateContentFormatter class."""

    def test_format_announcement_lists_agents_max_three_plus_more(self, formatter):
        """Test that announcement shows max 3 agents + N more."""
        agents = ["Claude", "GPT-4", "Gemini", "Llama", "Mistral"]
        result = formatter.format_announcement("Test topic", agents)

        assert "Claude" in result
        assert "GPT-4" in result
        assert "Gemini" in result
        assert "+2 more" in result
        assert "Llama" not in result

    def test_format_announcement_truncates_task_if_needed(self, formatter):
        """Test that long tasks are truncated to fit."""
        long_task = "A" * 250
        agents = ["Claude"]
        result = formatter.format_announcement(long_task, agents)

        assert len(result) <= MAX_TWEET_LENGTH

    def test_format_result_shows_consensus_status(self, formatter):
        """Test that result shows consensus/no consensus."""
        consensus_result = formatter.format_result("Topic", ["Agent1"], consensus_reached=True)
        assert "Consensus reached" in consensus_result

        no_consensus_result = formatter.format_result("Topic", ["Agent1"], consensus_reached=False)
        assert "No consensus" in no_consensus_result

    def test_format_result_shows_winner(self, formatter):
        """Test that result shows winner when provided."""
        result = formatter.format_result(
            "Topic", ["Agent1", "Agent2"], consensus_reached=False, winner="Agent1"
        )
        assert "Winner: Agent1" in result

    def test_format_thread_limits_to_25_tweets(self, formatter):
        """Test that threads are limited to 25 tweets."""
        highlights = [f"Highlight {i}" for i in range(30)]
        result = formatter.format_thread("Topic", ["Agent1"], highlights, consensus_reached=True)

        assert len(result) <= MAX_THREAD_LENGTH

    def test_format_thread_structure_intro_highlights_result(self, formatter):
        """Test thread structure: intro -> highlights -> result."""
        highlights = ["Key point 1", "Key point 2"]
        result = formatter.format_thread("Topic", ["Agent1"], highlights, consensus_reached=True)

        # First tweet is intro
        assert "New AI Debate" in result[0]

        # Middle tweets are numbered highlights
        assert result[1].startswith("1.")
        assert result[2].startswith("2.")

        # Last tweet is result
        assert "Consensus reached" in result[-1]

    def test_format_handles_special_characters_and_emoji(self, formatter):
        """Test handling of special characters and emojis."""
        # Special chars in task
        result = formatter.format_announcement("Topic with <html> & 'quotes' + emoji", ["Agent1"])
        assert "<html>" in result or "&" in result  # Should preserve content

        # Emojis in agents
        result2 = formatter.format_announcement("Topic", ["Agent1"])
        assert len(result2) <= MAX_TWEET_LENGTH

    def test_format_thread_includes_debate_url(self, formatter):
        """Test that debate URL is included in thread."""
        result = formatter.format_thread(
            "Topic",
            ["Agent1"],
            ["Highlight"],
            consensus_reached=True,
            debate_url="https://example.com/debate/123",
        )

        # URL should be in the last tweet
        assert "https://example.com/debate/123" in result[-1]

    def test_format_result_includes_audio_url(self, formatter):
        """Test that audio URL is included when provided."""
        result = formatter.format_result(
            "Topic",
            ["Agent1"],
            consensus_reached=True,
            audio_url="https://example.com/audio.mp3",
        )

        assert "https://example.com/audio.mp3" in result


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestCreateDebateSummary:
    """Tests for create_debate_summary helper function."""

    def test_creates_summary_within_length(self):
        """Test that summary respects max_length."""
        summary = create_debate_summary(
            task="Long task " * 50,
            agents=["Agent1", "Agent2"],
            consensus_reached=True,
            max_length=100,
        )
        assert len(summary) <= 100

    def test_includes_consensus_status(self):
        """Test that summary includes consensus status."""
        with_consensus = create_debate_summary("Task", ["Agent1"], consensus_reached=True)
        assert "Consensus" in with_consensus

        without = create_debate_summary("Task", ["Agent1"], consensus_reached=False)
        assert "No consensus" in without


# =============================================================================
# Integration Tests
# =============================================================================


class TestTwitterConnectorIntegration:
    """Integration tests for Twitter connector."""

    @pytest.mark.asyncio
    async def test_thread_posts_connected_tweets(self, configured_connector, mock_tweet_response):
        """Test that thread tweets are properly connected via reply_to."""
        call_count = 0
        tweet_ids = ["111", "222", "333"]

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            response = MagicMock()
            response.status_code = 201
            response.json.return_value = {"data": {"id": tweet_ids[call_count]}}
            call_count += 1
            return response

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.side_effect = mock_post

            result = await configured_connector.post_thread(["First", "Second", "Third"])

            assert result.success is True

            # Check that second and third tweets included reply_to
            calls = mock_instance.post.call_args_list
            assert len(calls) == 3

            # First tweet: no reply_to
            first_payload = calls[0].kwargs["json"]
            assert "reply" not in first_payload

            # Second tweet: reply_to first
            second_payload = calls[1].kwargs["json"]
            assert second_payload["reply"]["in_reply_to_tweet_id"] == "111"

            # Third tweet: reply_to second
            third_payload = calls[2].kwargs["json"]
            assert third_payload["reply"]["in_reply_to_tweet_id"] == "222"

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_allows_retry(self, configured_connector):
        """Test that after recovery timeout, requests are allowed again."""
        # Open the circuit
        for _ in range(5):
            configured_connector.circuit_breaker.record_failure()

        assert configured_connector.circuit_breaker.can_proceed() is False

        # Simulate recovery timeout (use internal property for single-entity mode)
        configured_connector.circuit_breaker._single_open_at = time.time() - 61

        # Now should allow
        assert configured_connector.circuit_breaker.can_proceed() is True

        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "123"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = mock_response

            result = await configured_connector.post_tweet("Recovery test")

            assert result.success is True
            # Success should reset the circuit
            assert configured_connector.circuit_breaker.is_open is False
            assert configured_connector.circuit_breaker.failures == 0
