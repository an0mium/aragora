"""
Tests for Twitter/X connector.

Tests the Twitter API v2 integration for searching tweets,
fetching tweet details, and retrieving user timelines.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.twitter import (
    TwitterConnector,
    TWITTER_SEARCH_URL,
    TWITTER_TWEET_URL,
    TWITTER_USER_TWEETS_URL,
    TWEET_URL_TEMPLATE,
)
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# Sample Twitter API v2 responses
SAMPLE_SEARCH_RESPONSE = {
    "data": [
        {
            "id": "1234567890",
            "text": "This is a test tweet about AI and machine learning. Very interesting stuff!",
            "author_id": "111111",
            "created_at": "2026-01-14T10:00:00.000Z",
            "lang": "en",
            "source": "Twitter Web App",
            "public_metrics": {
                "retweet_count": 150,
                "reply_count": 42,
                "like_count": 500,
                "quote_count": 25,
            },
        },
        {
            "id": "1234567891",
            "text": "Another tweet about tech startups and innovation.",
            "author_id": "222222",
            "created_at": "2026-01-13T15:30:00.000Z",
            "lang": "en",
            "source": "Twitter for iPhone",
            "public_metrics": {
                "retweet_count": 50,
                "reply_count": 10,
                "like_count": 200,
                "quote_count": 5,
            },
        },
    ],
    "includes": {
        "users": [
            {
                "id": "111111",
                "username": "techexpert",
                "name": "Tech Expert",
                "verified": True,
                "public_metrics": {
                    "followers_count": 150000,
                    "following_count": 500,
                    "tweet_count": 5000,
                },
            },
            {
                "id": "222222",
                "username": "startupguy",
                "name": "Startup Guy",
                "verified": False,
                "public_metrics": {
                    "followers_count": 5000,
                    "following_count": 200,
                    "tweet_count": 1000,
                },
            },
        ],
    },
    "meta": {
        "newest_id": "1234567890",
        "oldest_id": "1234567891",
        "result_count": 2,
    },
}

SAMPLE_TWEET_RESPONSE = {
    "data": {
        "id": "9876543210",
        "text": "A single tweet about programming and software development.",
        "author_id": "333333",
        "created_at": "2026-01-14T12:00:00.000Z",
        "lang": "en",
        "source": "Twitter Web App",
        "conversation_id": "9876543210",
        "public_metrics": {
            "retweet_count": 100,
            "reply_count": 30,
            "like_count": 300,
            "quote_count": 10,
        },
    },
    "includes": {
        "users": [
            {
                "id": "333333",
                "username": "devpro",
                "name": "Dev Pro",
                "verified": True,
                "public_metrics": {
                    "followers_count": 200000,
                    "following_count": 300,
                    "tweet_count": 8000,
                },
            },
        ],
    },
}

SAMPLE_USER_TWEETS_RESPONSE = {
    "data": [
        {
            "id": "5555555551",
            "text": "My first tweet today about coding.",
            "author_id": "444444",
            "created_at": "2026-01-14T08:00:00.000Z",
            "lang": "en",
            "source": "Twitter Web App",
            "public_metrics": {
                "retweet_count": 20,
                "reply_count": 5,
                "like_count": 80,
                "quote_count": 2,
            },
        },
        {
            "id": "5555555552",
            "text": "My second tweet today about debugging.",
            "author_id": "444444",
            "created_at": "2026-01-14T09:00:00.000Z",
            "lang": "en",
            "source": "Twitter Web App",
            "public_metrics": {
                "retweet_count": 15,
                "reply_count": 3,
                "like_count": 60,
                "quote_count": 1,
            },
        },
    ],
    "meta": {
        "result_count": 2,
    },
}

EMPTY_RESPONSE = {"data": [], "meta": {"result_count": 0}}


class TestTwitterConnector:
    """Tests for TwitterConnector initialization and properties."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    @pytest.fixture
    def unconfigured_connector(self):
        """Create an unconfigured connector."""
        return TwitterConnector(bearer_token="", rate_limit_delay=0.0)

    def test_connector_init_defaults(self, connector):
        """Connector should initialize with default values."""
        assert connector.timeout == 30
        assert connector.rate_limit_delay == 0.0
        assert connector.default_confidence == 0.5

    def test_connector_name(self, connector):
        """Connector should have correct name."""
        assert connector.name == "Twitter"

    def test_source_type(self, connector):
        """Source type should be EXTERNAL_API."""
        assert connector.source_type == SourceType.EXTERNAL_API

    def test_is_available(self, connector):
        """Should check httpx availability."""
        # httpx should be available in test environment
        assert connector.is_available is True

    def test_is_configured_with_token(self, connector):
        """Configured connector should report is_configured=True."""
        assert connector.is_configured is True

    def test_is_configured_without_token(self, unconfigured_connector):
        """Unconfigured connector should report is_configured=False."""
        assert unconfigured_connector.is_configured is False

    def test_custom_config(self):
        """Connector should accept custom configuration."""
        connector = TwitterConnector(
            bearer_token="custom-token",
            timeout=60,
            rate_limit_delay=2.0,
            default_confidence=0.6,
            max_cache_entries=100,
            cache_ttl_seconds=900.0,
        )

        assert connector.timeout == 60
        assert connector.rate_limit_delay == 2.0
        assert connector.default_confidence == 0.6

    def test_get_headers(self, connector):
        """Headers should include Bearer token."""
        headers = connector._get_headers()
        assert headers["Authorization"] == "Bearer test-bearer-token"
        assert headers["Content-Type"] == "application/json"


class TestTwitterSearch:
    """Tests for search functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    @pytest.fixture
    def unconfigured_connector(self):
        """Create an unconfigured connector."""
        return TwitterConnector(bearer_token="", rate_limit_delay=0.0)

    @pytest.mark.asyncio
    async def test_search_returns_evidence_list(self, connector):
        """Search should return a list of Evidence objects."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("AI machine learning")

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, Evidence) for r in results)

    @pytest.mark.asyncio
    async def test_search_parses_tweet_content(self, connector):
        """Search should correctly parse tweet content."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("AI")

        # Check first result (verified user with high followers)
        first = results[0]
        assert first.id == "twitter:1234567890"
        assert "AI and machine learning" in first.content
        assert first.author == "@techexpert"
        assert first.metadata["is_verified"] is True
        assert first.metadata["followers_count"] == 150000
        assert first.metadata["retweet_count"] == 150
        assert first.metadata["like_count"] == 500

        # Check second result (non-verified user)
        second = results[1]
        assert second.id == "twitter:1234567891"
        assert second.author == "@startupguy"
        assert second.metadata["is_verified"] is False

    @pytest.mark.asyncio
    async def test_search_confidence_verified_user(self, connector):
        """Verified users should get confidence bonus."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test")

        # Verified user with high engagement should have higher confidence
        verified_result = results[0]
        non_verified_result = results[1]
        assert verified_result.confidence > non_verified_result.confidence

    @pytest.mark.asyncio
    async def test_search_authority_high_followers(self, connector):
        """High follower count should increase authority."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test")

        # User with 150k followers (verified) should have authority 0.7
        high_followers = results[0]
        assert high_followers.authority == 0.7

        # User with 5k followers should have authority 0.45
        low_followers = results[1]
        assert low_followers.authority == 0.45

    @pytest.mark.asyncio
    async def test_search_handles_empty_results(self, connector):
        """Search should handle empty results gracefully."""
        empty_response = {"data": [], "meta": {"result_count": 0}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = empty_response
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("nonexistent_query_xyz")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_unconfigured_returns_empty(self, unconfigured_connector):
        """Unconfigured connector should return empty results."""
        results = await unconfigured_connector.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, connector):
        """Search should respect the limit parameter."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test", limit=1)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_clamps_limit_to_100(self, connector):
        """Search should clamp limit to API maximum of 100."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            await connector.search("test", limit=200)

            # Check that the API was called with max_results clamped
            call_args = mock_client_instance.get.call_args
            params = call_args.kwargs.get("params", {})
            # max_results should be clamped to 100
            assert params.get("max_results") <= 100

    @pytest.mark.asyncio
    async def test_search_handles_api_error(self, connector):
        """Search should handle API errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = Exception("Server Error")

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test query")

        # Should return empty list on error, not raise
        assert results == []


class TestTwitterFetch:
    """Tests for fetch functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    @pytest.fixture
    def unconfigured_connector(self):
        """Create an unconfigured connector."""
        return TwitterConnector(bearer_token="", rate_limit_delay=0.0)

    @pytest.mark.asyncio
    async def test_fetch_returns_evidence(self, connector):
        """Fetch should return an Evidence object for valid tweet ID."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_TWEET_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            result = await connector.fetch("twitter:9876543210")

        assert result is not None
        assert isinstance(result, Evidence)
        assert result.id == "twitter:9876543210"
        assert "programming and software" in result.content
        assert result.author == "@devpro"

    @pytest.mark.asyncio
    async def test_fetch_strips_prefix(self, connector):
        """Fetch should handle IDs with or without twitter: prefix."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_TWEET_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            # Test with prefix
            result1 = await connector.fetch("twitter:9876543210")
            assert result1 is not None

    @pytest.mark.asyncio
    async def test_fetch_returns_cached(self, connector):
        """Fetch should return cached evidence if available."""
        cached = Evidence(
            id="twitter:cached123",
            source_type=SourceType.EXTERNAL_API,
            source_id="cached123",
            content="Cached tweet content",
            title="@cached: Cached tweet content",
        )
        connector._cache_put("twitter:cached123", cached)

        result = await connector.fetch("twitter:cached123")

        assert result is not None
        assert result.content == "Cached tweet content"

    @pytest.mark.asyncio
    async def test_fetch_unconfigured_returns_none(self, unconfigured_connector):
        """Unconfigured connector should return None."""
        result = await unconfigured_connector.fetch("twitter:123")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_handles_not_found(self, connector):
        """Fetch should handle not found errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("Not Found")

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            result = await connector.fetch("twitter:nonexistent")

        assert result is None


class TestTwitterUserTweets:
    """Tests for get_user_tweets functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    @pytest.fixture
    def unconfigured_connector(self):
        """Create an unconfigured connector."""
        return TwitterConnector(bearer_token="", rate_limit_delay=0.0)

    @pytest.mark.asyncio
    async def test_get_user_tweets_returns_list(self, connector):
        """get_user_tweets should return a list of Evidence objects."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_USER_TWEETS_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.get_user_tweets("444444")

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, Evidence) for r in results)

    @pytest.mark.asyncio
    async def test_get_user_tweets_exclude_params(self, connector):
        """get_user_tweets should pass exclude parameters correctly."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_USER_TWEETS_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            await connector.get_user_tweets("444444", exclude_replies=True, exclude_retweets=True)

            # Check that exclude params were passed
            call_args = mock_client_instance.get.call_args
            params = call_args.kwargs.get("params", {})
            assert "exclude" in params
            assert "replies" in params["exclude"]
            assert "retweets" in params["exclude"]

    @pytest.mark.asyncio
    async def test_get_user_tweets_unconfigured_returns_empty(self, unconfigured_connector):
        """Unconfigured connector should return empty list."""
        results = await unconfigured_connector.get_user_tweets("123")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_user_tweets_handles_error(self, connector):
        """get_user_tweets should handle errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = Exception("Unauthorized")

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.get_user_tweets("444444")

        assert results == []


class TestTwitterHashtagSearch:
    """Tests for hashtag search functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    @pytest.mark.asyncio
    async def test_search_hashtag_adds_prefix(self, connector):
        """search_hashtag should add # prefix if missing."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            await connector.search_hashtag("AI")

            # Check that # was added to query
            call_args = mock_client_instance.get.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("query") == "#AI"

    @pytest.mark.asyncio
    async def test_search_hashtag_preserves_prefix(self, connector):
        """search_hashtag should not double # prefix."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            await connector.search_hashtag("#MachineLearning")

            # Check that # was not doubled
            call_args = mock_client_instance.get.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("query") == "#MachineLearning"


class TestTwitterFromUserSearch:
    """Tests for search_from_user functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    @pytest.mark.asyncio
    async def test_search_from_user_builds_query(self, connector):
        """search_from_user should build correct from: query."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            await connector.search_from_user("elonmusk")

            # Check that from: operator was used
            call_args = mock_client_instance.get.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("query") == "from:elonmusk"

    @pytest.mark.asyncio
    async def test_search_from_user_with_query(self, connector):
        """search_from_user should combine query with from: operator."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            await connector.search_from_user("elonmusk", query="AI")

            # Check that query + from: was used
            call_args = mock_client_instance.get.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("query") == "AI from:elonmusk"


class TestTwitterRateLimiting:
    """Tests for rate limiting."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector with custom rate limit."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.1,
        )

    @pytest.mark.asyncio
    async def test_rate_limiting_delays_requests(self, connector):
        """Rate limiting should delay consecutive requests."""
        import time

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            start = time.time()
            await connector.search("query1")
            await connector.search("query2")
            elapsed = time.time() - start

        # Should have at least one rate limit delay
        assert elapsed >= connector.rate_limit_delay


class TestTwitterCaching:
    """Tests for caching functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector with caching."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
            max_cache_entries=10,
            cache_ttl_seconds=3600.0,
        )

    def test_cache_initialized(self, connector):
        """Cache should be initialized."""
        assert hasattr(connector, "_cache")

    def test_cache_put_and_get(self, connector):
        """Cache should store and retrieve evidence."""
        evidence = Evidence(
            id="twitter:test123",
            source_type=SourceType.EXTERNAL_API,
            source_id="test123",
            content="Test tweet content",
            title="@test: Test tweet content",
        )

        connector._cache_put("twitter:test123", evidence)
        cached = connector._cache_get("twitter:test123")

        assert cached is not None
        assert cached.id == "twitter:test123"


class TestTwitterHealthCheck:
    """Tests for health check functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    @pytest.fixture
    def unconfigured_connector(self):
        """Create an unconfigured connector."""
        return TwitterConnector(bearer_token="", rate_limit_delay=0.0)

    @pytest.mark.asyncio
    async def test_health_check_success(self, connector):
        """Health check should return healthy status on success."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            health = await connector.health_check()

        assert health.is_available is True
        assert health.is_configured is True
        assert health.is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_rate_limited_is_healthy(self, connector):
        """Health check should treat 429 as connected (healthy)."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            health = await connector.health_check()

        # 429 means connected but rate limited
        assert health.is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_unconfigured(self, unconfigured_connector):
        """Health check should report unconfigured status."""
        health = await unconfigured_connector.health_check()

        assert health.is_available is True
        assert health.is_configured is False
        assert health.is_healthy is False


class TestTwitterURLs:
    """Tests for URL constants and generation."""

    def test_search_url(self):
        """Search URL should be correctly defined."""
        assert TWITTER_SEARCH_URL == "https://api.twitter.com/2/tweets/search/recent"

    def test_tweet_url(self):
        """Tweet URL should be correctly defined."""
        assert TWITTER_TWEET_URL == "https://api.twitter.com/2/tweets"

    def test_user_tweets_url(self):
        """User tweets URL should include placeholder."""
        assert "{user_id}" in TWITTER_USER_TWEETS_URL

    def test_tweet_url_template(self):
        """Tweet URL template should generate correct URLs."""
        tweet_id = "1234567890"
        url = TWEET_URL_TEMPLATE.format(tweet_id=tweet_id)
        assert url == "https://twitter.com/i/status/1234567890"


class TestTwitterParseTweet:
    """Tests for _parse_tweet functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    def test_parse_tweet_empty_id(self, connector):
        """_parse_tweet should return None for tweet without ID."""
        tweet = {"text": "Some content but no ID"}
        result = connector._parse_tweet(tweet, {})
        assert result is None

    def test_parse_tweet_empty_text(self, connector):
        """_parse_tweet should return None for tweet without text."""
        tweet = {"id": "123", "text": ""}
        result = connector._parse_tweet(tweet, {})
        assert result is None

    def test_parse_tweet_minimal(self, connector):
        """_parse_tweet should handle minimal tweet data."""
        tweet = {"id": "123", "text": "Minimal tweet content"}
        result = connector._parse_tweet(tweet, {})

        assert result is not None
        assert result.id == "twitter:123"
        assert result.content == "Minimal tweet content"
        assert result.author == "@unknown"

    def test_parse_tweet_title_truncation(self, connector):
        """_parse_tweet should truncate long titles."""
        long_text = "A" * 100  # 100 characters
        tweet = {"id": "123", "text": long_text}
        result = connector._parse_tweet(tweet, {})

        assert result is not None
        # Title should be truncated with ellipsis
        assert len(result.title) < len(f"@unknown: {long_text}")
        assert "..." in result.title

    def test_parse_tweet_confidence_capped(self, connector):
        """_parse_tweet should cap confidence at 0.80."""
        # Create a tweet with very high engagement to test cap
        tweet = {
            "id": "123",
            "text": "Viral tweet",
            "author_id": "999",
            "public_metrics": {
                "retweet_count": 100000,
                "reply_count": 50000,
                "like_count": 500000,
                "quote_count": 20000,
            },
        }
        includes = {
            "users": [
                {
                    "id": "999",
                    "username": "celebrity",
                    "name": "Celebrity",
                    "verified": True,
                    "public_metrics": {"followers_count": 10000000},
                }
            ]
        }

        result = connector._parse_tweet(tweet, includes)

        assert result is not None
        # Confidence should be capped at 0.80
        assert result.confidence <= 0.80


class TestTwitterMetadata:
    """Tests for metadata in Evidence objects."""

    @pytest.fixture
    def connector(self):
        """Create a Twitter connector for testing."""
        return TwitterConnector(
            bearer_token="test-bearer-token",
            rate_limit_delay=0.0,
        )

    @pytest.mark.asyncio
    async def test_metadata_contains_engagement(self, connector):
        """Evidence metadata should contain engagement metrics."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test")

        first = results[0]
        assert "retweet_count" in first.metadata
        assert "reply_count" in first.metadata
        assert "like_count" in first.metadata
        assert "quote_count" in first.metadata
        assert "total_engagement" in first.metadata

    @pytest.mark.asyncio
    async def test_metadata_contains_author_info(self, connector):
        """Evidence metadata should contain author information."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test")

        first = results[0]
        assert "author_id" in first.metadata
        assert "author_name" in first.metadata
        assert "username" in first.metadata
        assert "is_verified" in first.metadata
        assert "followers_count" in first.metadata

    @pytest.mark.asyncio
    async def test_metadata_contains_tweet_info(self, connector):
        """Evidence metadata should contain tweet information."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            results = await connector.search("test")

        first = results[0]
        assert "lang" in first.metadata
        assert "source" in first.metadata
