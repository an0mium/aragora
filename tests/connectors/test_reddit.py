"""
Tests for Reddit connector.

Tests the Reddit JSON API integration for evidence collection.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.reddit import (
    RedditConnector,
    REDDIT_SEARCH_URL,
    REDDIT_SUBREDDIT_URL,
    REDDIT_POST_URL,
    REDDIT_USER_AGENT,
)
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# Sample Reddit API responses
SAMPLE_SEARCH_RESPONSE = {
    "kind": "Listing",
    "data": {
        "after": "t3_nextpage",
        "children": [
            {
                "kind": "t3",
                "data": {
                    "id": "abc123",
                    "title": "Rust vs Go: A Performance Comparison",
                    "author": "rust_enthusiast",
                    "selftext": "I've been comparing Rust and Go performance...",
                    "url": "https://www.reddit.com/r/programming/comments/abc123",
                    "permalink": "/r/programming/comments/abc123/rust_vs_go_comparison",
                    "score": 500,
                    "num_comments": 150,
                    "upvote_ratio": 0.92,
                    "created_utc": 1609459200,
                    "subreddit": "programming",
                    "subreddit_subscribers": 5000000,
                    "is_self": True,
                    "over_18": False,
                    "spoiler": False,
                    "stickied": False,
                    "distinguished": None,
                    "link_flair_text": "Discussion",
                },
            },
            {
                "kind": "t3",
                "data": {
                    "id": "def456",
                    "title": "How I optimized my Go application",
                    "author": "golang_dev",
                    "selftext": "",
                    "url": "https://blog.example.com/go-optimization",
                    "permalink": "/r/golang/comments/def456/how_i_optimized",
                    "score": 200,
                    "num_comments": 45,
                    "upvote_ratio": 0.85,
                    "created_utc": 1609372800,
                    "subreddit": "golang",
                    "subreddit_subscribers": 200000,
                    "is_self": False,
                    "over_18": False,
                    "spoiler": False,
                    "stickied": False,
                    "distinguished": None,
                    "link_flair_text": None,
                },
            },
        ],
    },
}

SAMPLE_EMPTY_RESPONSE = {
    "kind": "Listing",
    "data": {
        "after": None,
        "children": [],
    },
}

SAMPLE_POST_DETAIL_RESPONSE = [
    {
        "kind": "Listing",
        "data": {
            "children": [
                {
                    "kind": "t3",
                    "data": {
                        "id": "abc123",
                        "title": "Detailed Post Title",
                        "author": "post_author",
                        "selftext": "This is the full post content with details.",
                        "url": "https://www.reddit.com/r/test/comments/abc123",
                        "permalink": "/r/test/comments/abc123/detailed_post",
                        "score": 1000,
                        "num_comments": 250,
                        "upvote_ratio": 0.95,
                        "created_utc": 1609459200,
                        "subreddit": "test",
                        "subreddit_subscribers": 1500000,
                        "is_self": True,
                        "over_18": False,
                        "spoiler": False,
                        "stickied": False,
                        "distinguished": None,
                        "link_flair_text": "Verified",
                    },
                }
            ],
        },
    },
    {
        "kind": "Listing",
        "data": {
            "children": [
                {
                    "kind": "t1",
                    "data": {
                        "body": "This is the top comment with insightful analysis.",
                        "author": "commenter1",
                        "score": 500,
                    },
                },
                {
                    "kind": "t1",
                    "data": {
                        "body": "Another great comment.",
                        "author": "commenter2",
                        "score": 200,
                    },
                },
                {
                    "kind": "t1",
                    "data": {
                        "body": "[deleted]",
                        "author": "[deleted]",
                        "score": 0,
                    },
                },
            ],
        },
    },
]

SAMPLE_SUBREDDIT_RESPONSE = {
    "kind": "Listing",
    "data": {
        "after": "t3_nextpage",
        "children": [
            {
                "kind": "t3",
                "data": {
                    "id": "hot123",
                    "title": "Hot Post in Subreddit",
                    "author": "hot_poster",
                    "selftext": "Trending content here",
                    "url": "https://www.reddit.com/r/python/comments/hot123",
                    "permalink": "/r/python/comments/hot123/hot_post",
                    "score": 1500,
                    "num_comments": 300,
                    "upvote_ratio": 0.96,
                    "created_utc": 1609459200,
                    "subreddit": "python",
                    "subreddit_subscribers": 800000,
                    "is_self": True,
                    "over_18": False,
                    "spoiler": False,
                    "stickied": False,
                    "distinguished": None,
                    "link_flair_text": None,
                },
            },
        ],
    },
}


class TestRedditConnector:
    """Tests for RedditConnector initialization and properties."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector instance."""
        return RedditConnector(rate_limit_delay=0.0)

    def test_connector_init(self, connector):
        """Connector should initialize with default values."""
        assert connector.timeout == 30
        assert connector.default_confidence == 0.6

    def test_connector_name(self, connector):
        """Connector should have correct name."""
        assert connector.name == "Reddit"

    def test_source_type(self, connector):
        """Source type should be EXTERNAL_API."""
        assert connector.source_type == SourceType.EXTERNAL_API

    def test_is_available(self, connector):
        """Should check httpx availability."""
        # httpx should be available in test environment
        assert connector.is_available is True

    def test_is_configured(self, connector):
        """Reddit JSON API requires no configuration."""
        assert connector.is_configured is True

    def test_custom_config(self):
        """Connector should accept custom configuration."""
        connector = RedditConnector(
            timeout=60,
            rate_limit_delay=1.0,
            default_confidence=0.7,
            max_cache_entries=100,
            cache_ttl_seconds=7200.0,
        )

        assert connector.timeout == 60
        assert connector.rate_limit_delay == 1.0
        assert connector.default_confidence == 0.7

    def test_repr(self, connector):
        """Test string representation."""
        repr_str = repr(connector)
        assert "Reddit" in repr_str or "external_api" in repr_str


class TestRedditSearch:
    """Tests for search functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector instance with no rate limiting."""
        return RedditConnector(rate_limit_delay=0.0)

    @pytest.mark.asyncio
    async def test_search_returns_evidence_list(self, connector):
        """Search should return a list of Evidence objects."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.search("rust vs go")

            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, Evidence) for r in results)

    @pytest.mark.asyncio
    async def test_search_parses_post_content(self, connector):
        """Search should correctly parse post content."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.search("rust vs go")

            # Check first result
            first = results[0]
            assert first.id == "reddit:abc123"
            assert "Rust vs Go" in first.title
            assert first.author == "rust_enthusiast"
            assert first.metadata["score"] == 500
            assert first.metadata["num_comments"] == 150
            assert first.metadata["subreddit"] == "programming"
            assert first.metadata["upvote_ratio"] == 0.92

    @pytest.mark.asyncio
    async def test_search_handles_empty_results(self, connector):
        """Search should handle empty results gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_EMPTY_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.search("nonexistent_query_xyz")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_api_error(self, connector):
        """Search should handle API errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = Exception("Connection failed")
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.search("test query")

            # Should return empty list on error, not raise
            assert results == []

    @pytest.mark.asyncio
    async def test_search_with_subreddit_filter(self, connector):
        """Search should support subreddit filtering."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.search("rust", subreddit="programming")

            assert len(results) > 0
            # Verify the URL was constructed correctly
            call_args = mock_instance.get.call_args
            assert "programming" in str(call_args)

    @pytest.mark.asyncio
    async def test_search_limit_clamped(self, connector):
        """Search limit should be clamped to max 100."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SEARCH_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            await connector.search("test", limit=200)

            # Verify limit was clamped in params
            call_args = mock_instance.get.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("limit", 0) <= 100


class TestRedditFetch:
    """Tests for fetch functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector instance."""
        return RedditConnector(rate_limit_delay=0.0)

    @pytest.mark.asyncio
    async def test_fetch_by_id(self, connector):
        """Fetch should retrieve a specific post by ID."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_POST_DETAIL_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await connector.fetch("abc123")

            assert result is not None
            assert result.id == "reddit:abc123"
            assert "Detailed Post Title" in result.title

    @pytest.mark.asyncio
    async def test_fetch_with_reddit_prefix(self, connector):
        """Fetch should handle reddit: prefix in ID."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_POST_DETAIL_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await connector.fetch("reddit:abc123")

            assert result is not None
            assert result.id == "reddit:abc123"

    @pytest.mark.asyncio
    async def test_fetch_includes_top_comments(self, connector):
        """Fetch should include top comments in metadata."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_POST_DETAIL_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await connector.fetch("abc123")

            assert result is not None
            assert "top_comments" in result.metadata
            # Should skip deleted comments
            comments = result.metadata["top_comments"]
            assert len(comments) == 2
            assert comments[0]["author"] == "commenter1"

    @pytest.mark.asyncio
    async def test_fetch_returns_cached(self, connector):
        """Fetch should return cached evidence."""
        cached = Evidence(
            id="reddit:cached123",
            source_type=SourceType.EXTERNAL_API,
            source_id="cached123",
            content="Cached post content",
            title="Cached Post",
        )
        connector._cache_put("reddit:cached123", cached)

        result = await connector.fetch("reddit:cached123")

        assert result is not None
        assert result.title == "Cached Post"

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, connector):
        """Fetch should return None for non-existent posts."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = Exception("Not found")
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await connector.fetch("nonexistent123")

            assert result is None


class TestRedditSubreddit:
    """Tests for subreddit access methods."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector instance."""
        return RedditConnector(rate_limit_delay=0.0)

    @pytest.mark.asyncio
    async def test_get_subreddit(self, connector):
        """get_subreddit should return posts from a subreddit."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SUBREDDIT_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.get_subreddit("python", sort="hot")

            assert len(results) == 1
            assert "Hot Post" in results[0].title

    @pytest.mark.asyncio
    async def test_get_hot(self, connector):
        """get_hot should return hot posts."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SUBREDDIT_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.get_hot("python")

            assert len(results) == 1
            call_args = mock_instance.get.call_args
            assert "hot" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_top(self, connector):
        """get_top should return top posts."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SUBREDDIT_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.get_top("python", time_filter="week")

            assert len(results) == 1
            call_args = mock_instance.get.call_args
            assert "top" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_new(self, connector):
        """get_new should return new posts."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = SAMPLE_SUBREDDIT_RESPONSE
            mock_response.raise_for_status = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.get_new("python")

            assert len(results) == 1
            call_args = mock_instance.get.call_args
            assert "new" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_subreddit_handles_error(self, connector):
        """get_subreddit should handle errors gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = Exception("Subreddit not found")
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            results = await connector.get_subreddit("nonexistent_sub")

            assert results == []


class TestRedditRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector with custom rate limit."""
        return RedditConnector(rate_limit_delay=0.1)

    @pytest.mark.asyncio
    async def test_rate_limiting_delays_requests(self, connector):
        """Rate limiting should delay consecutive requests."""
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_EMPTY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            start = time.time()
            await connector.search("query1")
            await connector.search("query2")
            elapsed = time.time() - start

            # Should have at least one rate limit delay
            assert elapsed >= connector.rate_limit_delay


class TestRedditCaching:
    """Tests for caching functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector with caching."""
        return RedditConnector(
            rate_limit_delay=0.0,
            max_cache_entries=10,
            cache_ttl_seconds=3600.0,
        )

    def test_cache_initialized(self, connector):
        """Cache should be initialized."""
        assert hasattr(connector, "_cache")

    def test_cache_put_and_get(self, connector):
        """Cache operations should work correctly."""
        evidence = Evidence(
            id="reddit:cache_test",
            source_type=SourceType.EXTERNAL_API,
            source_id="cache_test",
            content="Cached content",
            title="Cached Title",
        )

        connector._cache_put("reddit:cache_test", evidence)
        cached = connector._cache_get("reddit:cache_test")

        assert cached is not None
        assert cached.id == "reddit:cache_test"


class TestRedditURLs:
    """Tests for URL constants."""

    def test_search_url(self):
        """Search URL should be correctly defined."""
        assert "reddit.com" in REDDIT_SEARCH_URL
        assert "search.json" in REDDIT_SEARCH_URL

    def test_subreddit_url(self):
        """Subreddit URL should have placeholder."""
        assert "{subreddit}" in REDDIT_SUBREDDIT_URL

    def test_post_url(self):
        """Post URL should have placeholder."""
        assert "{post_id}" in REDDIT_POST_URL

    def test_user_agent(self):
        """User agent should be defined."""
        assert "aragora" in REDDIT_USER_AGENT.lower()


class TestRedditPostParsing:
    """Tests for post parsing functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector instance."""
        return RedditConnector(rate_limit_delay=0.0)

    def test_parse_listing(self, connector):
        """_parse_listing should correctly parse listing response."""
        results = connector._parse_listing(SAMPLE_SEARCH_RESPONSE)

        assert len(results) == 2
        assert all(isinstance(r, Evidence) for r in results)

    def test_parse_listing_empty(self, connector):
        """_parse_listing should handle empty listings."""
        results = connector._parse_listing(SAMPLE_EMPTY_RESPONSE)
        assert results == []

    def test_parse_post_self_post(self, connector):
        """_parse_post should correctly parse self posts."""
        post_data = SAMPLE_SEARCH_RESPONSE["data"]["children"][0]["data"]
        evidence = connector._parse_post(post_data)

        assert evidence is not None
        assert evidence.id == "reddit:abc123"
        assert "Rust vs Go" in evidence.title
        assert "comparing Rust and Go" in evidence.content
        assert evidence.metadata["is_self"] is True

    def test_parse_post_link_post(self, connector):
        """_parse_post should correctly parse link posts."""
        post_data = SAMPLE_SEARCH_RESPONSE["data"]["children"][1]["data"]
        evidence = connector._parse_post(post_data)

        assert evidence is not None
        assert evidence.id == "reddit:def456"
        assert "optimized" in evidence.title
        # Link posts should include the external URL
        assert "blog.example.com" in evidence.content
        assert evidence.metadata["is_self"] is False

    def test_parse_post_no_id(self, connector):
        """_parse_post should return None for posts without ID."""
        evidence = connector._parse_post({})
        assert evidence is None

    def test_parse_post_confidence_calculation(self, connector):
        """_parse_post should calculate confidence based on engagement."""
        # High engagement post
        high_engagement = {
            "id": "high123",
            "title": "Popular Post",
            "selftext": "Content",
            "score": 5000,
            "num_comments": 500,
            "upvote_ratio": 0.98,
            "subreddit": "popular",
            "subreddit_subscribers": 2000000,
            "created_utc": 1609459200,
        }
        evidence = connector._parse_post(high_engagement)

        assert evidence is not None
        # High engagement should boost confidence
        assert evidence.confidence > connector.default_confidence

    def test_parse_post_authority_by_subreddit_size(self, connector):
        """_parse_post should calculate authority based on subreddit size."""
        # Large subreddit
        large_sub = {
            "id": "large123",
            "title": "Post",
            "selftext": "Content",
            "score": 100,
            "num_comments": 10,
            "upvote_ratio": 0.8,
            "subreddit": "large",
            "subreddit_subscribers": 5000000,
            "created_utc": 1609459200,
        }
        evidence_large = connector._parse_post(large_sub)
        assert evidence_large.authority == 0.6

        # Medium subreddit
        medium_sub = {
            "id": "medium123",
            "title": "Post",
            "selftext": "Content",
            "score": 100,
            "num_comments": 10,
            "upvote_ratio": 0.8,
            "subreddit": "medium",
            "subreddit_subscribers": 500000,
            "created_utc": 1609459200,
        }
        evidence_medium = connector._parse_post(medium_sub)
        assert evidence_medium.authority == 0.65

        # Small subreddit
        small_sub = {
            "id": "small123",
            "title": "Post",
            "selftext": "Content",
            "score": 100,
            "num_comments": 10,
            "upvote_ratio": 0.8,
            "subreddit": "small",
            "subreddit_subscribers": 5000,
            "created_utc": 1609459200,
        }
        evidence_small = connector._parse_post(small_sub)
        assert evidence_small.authority == 0.5

    def test_parse_post_detail(self, connector):
        """_parse_post_detail should parse full post with comments."""
        evidence = connector._parse_post_detail(SAMPLE_POST_DETAIL_RESPONSE)

        assert evidence is not None
        assert evidence.id == "reddit:abc123"
        assert "top_comments" in evidence.metadata
        assert len(evidence.metadata["top_comments"]) == 2

    def test_parse_post_detail_empty(self, connector):
        """_parse_post_detail should handle empty response."""
        evidence = connector._parse_post_detail([])
        assert evidence is None

        evidence = connector._parse_post_detail(None)
        assert evidence is None


class TestRedditHealthCheck:
    """Tests for health check functionality."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector instance."""
        return RedditConnector(rate_limit_delay=0.0)

    @pytest.mark.asyncio
    async def test_health_check_success(self, connector):
        """Health check should succeed with valid response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await connector._perform_health_check(timeout=5.0)

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, connector):
        """Health check should fail with error response."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await connector._perform_health_check(timeout=5.0)

            assert result is False


class TestRedditHeaders:
    """Tests for HTTP headers."""

    @pytest.fixture
    def connector(self):
        """Create a Reddit connector instance."""
        return RedditConnector(rate_limit_delay=0.0)

    def test_get_headers(self, connector):
        """Headers should include user agent."""
        headers = connector._get_headers()

        assert "User-Agent" in headers
        assert "aragora" in headers["User-Agent"].lower()
        assert "Accept" in headers
        assert headers["Accept"] == "application/json"


class TestRedditHttpxUnavailable:
    """Tests for when httpx is not available."""

    @pytest.fixture
    def connector_no_httpx(self):
        """Create a connector that simulates httpx unavailable."""
        connector = RedditConnector(rate_limit_delay=0.0)
        return connector

    @pytest.mark.asyncio
    async def test_search_without_httpx(self, connector_no_httpx):
        """Search should return empty list when httpx unavailable."""
        with patch("aragora.connectors.reddit.HTTPX_AVAILABLE", False):
            connector = RedditConnector(rate_limit_delay=0.0)
            results = await connector.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_fetch_without_httpx(self, connector_no_httpx):
        """Fetch should return None when httpx unavailable."""
        with patch("aragora.connectors.reddit.HTTPX_AVAILABLE", False):
            connector = RedditConnector(rate_limit_delay=0.0)
            result = await connector.fetch("test123")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_subreddit_without_httpx(self, connector_no_httpx):
        """get_subreddit should return empty list when httpx unavailable."""
        with patch("aragora.connectors.reddit.HTTPX_AVAILABLE", False):
            connector = RedditConnector(rate_limit_delay=0.0)
            results = await connector.get_subreddit("test")
            assert results == []

    def test_is_available_without_httpx(self):
        """is_available should return False when httpx unavailable."""
        with patch("aragora.connectors.reddit.HTTPX_AVAILABLE", False):
            connector = RedditConnector(rate_limit_delay=0.0)
            assert connector.is_available is False
