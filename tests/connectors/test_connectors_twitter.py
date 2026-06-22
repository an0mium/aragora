"""
Tests for Twitter Connector.

Tests the TwitterConnector class for searching and fetching tweets.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time
import os

from aragora.connectors.twitter import TwitterConnector
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


class TestTwitterConnectorBasics:
    """Test basic connector properties."""

    def test_connector_name(self):
        """Test connector name property."""
        connector = TwitterConnector()
        assert connector.name == "Twitter"

    def test_source_type(self):
        """Test source type is EXTERNAL_API."""
        connector = TwitterConnector()
        assert connector.source_type == SourceType.EXTERNAL_API

    def test_default_confidence(self):
        """Test default confidence is lower due to less fact-checking."""
        connector = TwitterConnector()
        assert connector.default_confidence == 0.5

    def test_is_available(self):
        """Test is_available is True when httpx installed."""
        connector = TwitterConnector()
        assert connector.is_available is True

    def test_is_configured_without_token(self):
        """Test is_configured returns False without token."""
        connector = TwitterConnector(bearer_token="")
        assert connector.is_configured is False

    def test_is_configured_with_token(self):
        """Test is_configured returns True with token."""
        connector = TwitterConnector(bearer_token="test_token")
        assert connector.is_configured is True


class TestTwitterConnectorSearch:
    """Test search functionality."""

    @pytest.fixture
    def mock_search_response(self):
        """Mock Twitter search API response."""
        return {
            "data": [
                {
                    "id": "1234567890",
                    "text": "This is a test tweet about AI safety and alignment research.",
                    "author_id": "111",
                    "created_at": "2024-01-15T10:30:00.000Z",
                    "public_metrics": {
                        "retweet_count": 100,
                        "reply_count": 25,
                        "like_count": 500,
                        "quote_count": 10,
                    },
                    "lang": "en",
                    "source": "Twitter Web App",
                },
                {
                    "id": "1234567891",
                    "text": "Another tweet discussing machine learning breakthroughs.",
                    "author_id": "222",
                    "created_at": "2024-01-15T09:00:00.000Z",
                    "public_metrics": {
                        "retweet_count": 50,
                        "reply_count": 10,
                        "like_count": 200,
                        "quote_count": 5,
                    },
                    "lang": "en",
                    "source": "Twitter for iPhone",
                },
            ],
            "includes": {
                "users": [
                    {
                        "id": "111",
                        "username": "airesearcher",
                        "name": "AI Researcher",
                        "verified": True,
                        "public_metrics": {
                            "followers_count": 50000,
                            "following_count": 500,
                        },
                    },
                    {
                        "id": "222",
                        "username": "mldev",
                        "name": "ML Developer",
                        "verified": False,
                        "public_metrics": {
                            "followers_count": 5000,
                            "following_count": 200,
                        },
                    },
                ]
            },
        }

    @pytest.mark.asyncio
    async def test_search_without_token_returns_empty(self):
        """Test search returns empty when not configured."""
        connector = TwitterConnector(bearer_token="")
        results = await connector.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_parses_results(self, mock_search_response):
        """Test that search correctly parses Twitter response."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_search_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("AI safety")

            assert len(results) == 2
            assert all(isinstance(r, Evidence) for r in results)

    @pytest.mark.asyncio
    async def test_search_evidence_properties(self, mock_search_response):
        """Test evidence object has correct properties."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_search_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("AI safety")
            evidence = results[0]

            assert evidence.id == "twitter:1234567890"
            assert "test tweet about AI safety" in evidence.content
            assert evidence.author == "@airesearcher"
            assert evidence.source_type == SourceType.EXTERNAL_API
            assert "twitter.com" in evidence.url

    @pytest.mark.asyncio
    async def test_search_metadata(self, mock_search_response):
        """Test evidence metadata contains Twitter-specific fields."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_search_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("AI safety")
            metadata = results[0].metadata

            assert metadata["retweet_count"] == 100
            assert metadata["like_count"] == 500
            assert metadata["username"] == "airesearcher"
            assert metadata["is_verified"] is True
            assert metadata["followers_count"] == 50000

    @pytest.mark.asyncio
    async def test_search_handles_auth_error(self):
        """Test search handles authentication errors."""
        connector = TwitterConnector(bearer_token="invalid_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 401

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("test query")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_rate_limit(self):
        """Test search handles rate limit errors."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("test query")
            assert results == []


class TestTwitterConnectorFetch:
    """Test fetch functionality."""

    @pytest.fixture
    def mock_tweet_response(self):
        """Mock Twitter single tweet API response."""
        return {
            "data": {
                "id": "9876543210",
                "text": "Detailed tweet content with more information.",
                "author_id": "333",
                "created_at": "2024-01-15T12:00:00.000Z",
                "public_metrics": {
                    "retweet_count": 200,
                    "reply_count": 50,
                    "like_count": 1000,
                    "quote_count": 25,
                },
                "lang": "en",
                "conversation_id": "9876543210",
            },
            "includes": {
                "users": [
                    {
                        "id": "333",
                        "username": "expertuser",
                        "name": "Domain Expert",
                        "verified": True,
                        "public_metrics": {
                            "followers_count": 200000,
                        },
                    }
                ]
            },
        }

    @pytest.mark.asyncio
    async def test_fetch_tweet(self, mock_tweet_response):
        """Test fetching a specific tweet."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_tweet_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            evidence = await connector.fetch("9876543210")

            assert evidence is not None
            assert evidence.id == "twitter:9876543210"
            assert "Detailed tweet content" in evidence.content
            assert evidence.author == "@expertuser"

    @pytest.mark.asyncio
    async def test_fetch_handles_prefix(self, mock_tweet_response):
        """Test fetch handles 'twitter:' prefix in ID."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_tweet_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            evidence = await connector.fetch("twitter:9876543210")

            assert evidence is not None
            assert evidence.id == "twitter:9876543210"

    @pytest.mark.asyncio
    async def test_fetch_caches_result(self, mock_tweet_response):
        """Test fetch caches result."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_tweet_response
            mock_response.raise_for_status = MagicMock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            # First fetch
            await connector.fetch("9876543210")

            # Second fetch should use cache
            await connector.fetch("9876543210")

            # API should only be called once
            assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_not_found(self):
        """Test fetch handles 404."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 404

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            evidence = await connector.fetch("nonexistent")
            assert evidence is None


class TestTwitterConnectorConfidence:
    """Test confidence and authority calculations."""

    def test_verified_user_gets_bonus(self):
        """Test verified users get confidence bonus."""
        connector = TwitterConnector(bearer_token="test_token")

        base_tweet = {
            "id": "123",
            "text": "Test tweet",
            "author_id": "111",
            "created_at": "2024-01-15T10:00:00.000Z",
            "public_metrics": {
                "retweet_count": 10,
                "reply_count": 5,
                "like_count": 50,
                "quote_count": 2,
            },
        }

        verified_user = {
            "111": {
                "id": "111",
                "username": "verified",
                "verified": True,
                "public_metrics": {"followers_count": 10000},
            }
        }

        unverified_user = {
            "111": {
                "id": "111",
                "username": "unverified",
                "verified": False,
                "public_metrics": {"followers_count": 10000},
            }
        }

        verified_evidence = connector._parse_tweet(base_tweet, {}, verified_user)
        unverified_evidence = connector._parse_tweet(base_tweet, {}, unverified_user)

        assert verified_evidence.confidence > unverified_evidence.confidence

    def test_high_engagement_increases_confidence(self):
        """Test higher engagement leads to higher confidence."""
        connector = TwitterConnector(bearer_token="test_token")

        users = {
            "111": {
                "id": "111",
                "username": "user",
                "verified": False,
                "public_metrics": {"followers_count": 1000},
            }
        }

        high_engagement = {
            "id": "123",
            "text": "Viral tweet",
            "author_id": "111",
            "created_at": "2024-01-15T10:00:00.000Z",
            "public_metrics": {
                "retweet_count": 10000,
                "reply_count": 5000,
                "like_count": 50000,
                "quote_count": 2000,
            },
        }

        low_engagement = {
            "id": "456",
            "text": "Regular tweet",
            "author_id": "111",
            "created_at": "2024-01-15T10:00:00.000Z",
            "public_metrics": {
                "retweet_count": 1,
                "reply_count": 0,
                "like_count": 5,
                "quote_count": 0,
            },
        }

        high_evidence = connector._parse_tweet(high_engagement, {}, users)
        low_evidence = connector._parse_tweet(low_engagement, {}, users)

        assert high_evidence.confidence > low_evidence.confidence

    def test_confidence_capped(self):
        """Test confidence is capped at 0.80."""
        connector = TwitterConnector(bearer_token="test_token")

        users = {
            "111": {
                "id": "111",
                "username": "megainfluencer",
                "verified": True,
                "public_metrics": {"followers_count": 50_000_000},
            }
        }

        max_tweet = {
            "id": "123",
            "text": "Maximum engagement tweet",
            "author_id": "111",
            "created_at": "2024-01-15T10:00:00.000Z",
            "public_metrics": {
                "retweet_count": 1_000_000,
                "reply_count": 500_000,
                "like_count": 5_000_000,
                "quote_count": 100_000,
            },
        }

        evidence = connector._parse_tweet(max_tweet, {}, users)
        assert evidence.confidence <= 0.80


class TestTwitterConnectorHelperMethods:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_search_hashtag(self):
        """Test search_hashtag adds # prefix."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch.object(connector, "search", new_callable=AsyncMock) as mock:
            mock.return_value = []

            await connector.search_hashtag("AIResearch")

            mock.assert_called_once_with("#AIResearch", limit=10)

    @pytest.mark.asyncio
    async def test_search_hashtag_preserves_existing_hash(self):
        """Test search_hashtag doesn't double the #."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch.object(connector, "search", new_callable=AsyncMock) as mock:
            mock.return_value = []

            await connector.search_hashtag("#MachineLearning")

            mock.assert_called_once_with("#MachineLearning", limit=10)

    @pytest.mark.asyncio
    async def test_search_from_user(self):
        """Test search_from_user constructs correct query."""
        connector = TwitterConnector(bearer_token="test_token")

        with patch.object(connector, "search", new_callable=AsyncMock) as mock:
            mock.return_value = []

            await connector.search_from_user("elonmusk", query="AI")

            mock.assert_called_once_with("AI from:elonmusk", limit=10)


class TestTwitterConnectorRateLimiting:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_delay(self):
        """Test rate limiting enforces delay."""
        connector = TwitterConnector(bearer_token="test_token", rate_limit_delay=0.1)

        start_time = time.time()

        await connector._rate_limit()
        await connector._rate_limit()

        elapsed = time.time() - start_time
        assert elapsed >= 0.1


class TestTwitterConnectorEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_tweet_missing_id(self):
        """Test parsing tweet without ID returns None."""
        connector = TwitterConnector(bearer_token="test_token")

        tweet = {"text": "No ID", "author_id": "111"}
        evidence = connector._parse_tweet(tweet, {})

        assert evidence is None

    def test_parse_tweet_empty_text(self):
        """Test parsing tweet with empty text returns None."""
        connector = TwitterConnector(bearer_token="test_token")

        tweet = {"id": "123", "text": "", "author_id": "111"}
        evidence = connector._parse_tweet(tweet, {})

        assert evidence is None

    def test_parse_tweet_missing_author(self):
        """Test parsing tweet with missing author info."""
        connector = TwitterConnector(bearer_token="test_token")

        tweet = {
            "id": "123",
            "text": "Tweet without user info",
            "author_id": "999",
            "created_at": "2024-01-15T10:00:00.000Z",
            "public_metrics": {
                "retweet_count": 0,
                "reply_count": 0,
                "like_count": 0,
                "quote_count": 0,
            },
        }

        evidence = connector._parse_tweet(tweet, {})

        assert evidence is not None
        assert evidence.author == "@unknown"

    def test_parse_empty_results(self):
        """Test parsing empty search results."""
        connector = TwitterConnector(bearer_token="test_token")

        data = {"data": [], "includes": {"users": []}}
        results = connector._parse_search_results(data)

        assert results == []


# Integration-style test
class TestTwitterConnectorIntegration:
    """Integration tests for full flow."""

    @pytest.mark.asyncio
    async def test_search_to_evidence_flow(self):
        """Test complete search to evidence flow."""
        connector = TwitterConnector(bearer_token="test_token")

        mock_response = {
            "data": [
                {
                    "id": "integration_test",
                    "text": "Integration test tweet for Twitter connector.",
                    "author_id": "test_user",
                    "created_at": "2024-01-15T10:00:00.000Z",
                    "public_metrics": {
                        "retweet_count": 10,
                        "reply_count": 5,
                        "like_count": 100,
                        "quote_count": 2,
                    },
                    "lang": "en",
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "test_user",
                        "username": "testaccount",
                        "name": "Test Account",
                        "verified": False,
                        "public_metrics": {"followers_count": 1000},
                    }
                ]
            },
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_http_response = MagicMock()
            mock_http_response.status_code = 200
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_http_response
            )

            results = await connector.search("integration test")

            assert len(results) == 1
            evidence = results[0]

            # Verify evidence structure
            assert evidence.id.startswith("twitter:")
            assert evidence.source_type == SourceType.EXTERNAL_API
            assert "Integration test tweet" in evidence.content
            assert evidence.url is not None
            assert evidence.confidence > 0
            assert evidence.freshness > 0
            assert evidence.authority > 0
            assert "retweet_count" in evidence.metadata
            assert "username" in evidence.metadata
