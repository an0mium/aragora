"""Tests for Pulse namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestPulseTrending:
    """Tests for trending topics operations."""

    def test_get_trending_default(self) -> None:
        """Get trending topics with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "topics": [
                    {"id": "topic_1", "title": "AI Safety", "score": 0.95},
                    {"id": "topic_2", "title": "LLM Research", "score": 0.89},
                ],
                "total": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.pulse.get_trending()

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/pulse/trending",
                params={"limit": 20},
            )
            assert len(result["topics"]) == 2
            client.close()

    def test_get_trending_with_source(self) -> None:
        """Get trending topics filtered by source."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"topics": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.get_trending(source="hackernews")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["source"] == "hackernews"
            client.close()

    def test_get_trending_with_category(self) -> None:
        """Get trending topics filtered by category."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"topics": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.get_trending(category="technology")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["category"] == "technology"
            client.close()

    def test_get_trending_with_limit(self) -> None:
        """Get trending topics with custom limit."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"topics": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.get_trending(limit=50)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["limit"] == 50
            client.close()

    def test_get_trending_reddit_source(self) -> None:
        """Get trending topics from Reddit."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"topics": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.get_trending(source="reddit")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["source"] == "reddit"
            client.close()

    def test_get_trending_twitter_source(self) -> None:
        """Get trending topics from Twitter."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"topics": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.get_trending(source="twitter")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["source"] == "twitter"
            client.close()


class TestPulseTopics:
    """Tests for topic detail operations."""

    def test_get_topic(self) -> None:
        """Get details for a specific topic."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "topic_123",
                "title": "AI Safety Research",
                "score": 0.92,
                "sources": ["hackernews", "reddit"],
                "history": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.pulse.get_topic("topic_123")

            mock_request.assert_called_once_with("GET", "/api/v1/pulse/topics/topic_123")
            assert result["id"] == "topic_123"
            client.close()


class TestPulseSearch:
    """Tests for pulse search operations."""

    def test_search_default(self) -> None:
        """Search topics with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "results": [
                    {"id": "topic_1", "title": "Machine Learning", "relevance": 0.9},
                ],
                "total": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.pulse.search("machine learning")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/pulse/search",
                params={"q": "machine learning", "limit": 20},
            )
            assert len(result["results"]) == 1
            client.close()

    def test_search_with_source(self) -> None:
        """Search topics filtered by source."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.search("AI safety", source="hackernews")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["source"] == "hackernews"
            client.close()

    def test_search_with_min_score(self) -> None:
        """Search topics with minimum score threshold."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.search("LLMs", min_score=0.8)

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["min_score"] == 0.8
            client.close()

    def test_search_with_limit(self) -> None:
        """Search topics with custom limit."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.search("transformers", limit=100)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["limit"] == 100
            client.close()


class TestPulseStats:
    """Tests for pulse statistics operations."""

    def test_get_stats(self) -> None:
        """Get pulse system statistics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_topics": 5000,
                "total_sources": 3,
                "ingestion_rate": 100,
                "last_updated": "2024-01-15T12:00:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.pulse.get_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/pulse/stats")
            assert result["total_topics"] == 5000
            client.close()


class TestPulseSources:
    """Tests for pulse sources operations."""

    def test_get_sources(self) -> None:
        """List configured pulse sources."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "sources": [
                    {"name": "hackernews", "enabled": True, "refresh_interval": 300},
                    {"name": "reddit", "enabled": True, "refresh_interval": 600},
                    {"name": "twitter", "enabled": False, "refresh_interval": 120},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.pulse.get_sources()

            mock_request.assert_called_once_with("GET", "/api/v1/pulse/sources")
            assert len(result) == 3
            assert result[0]["name"] == "hackernews"
            client.close()


class TestPulseCategories:
    """Tests for pulse categories operations."""

    def test_get_categories(self) -> None:
        """List available topic categories."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "categories": [
                    {"name": "technology", "topic_count": 1500},
                    {"name": "science", "topic_count": 800},
                    {"name": "finance", "topic_count": 600},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.pulse.get_categories()

            mock_request.assert_called_once_with("GET", "/api/v1/pulse/categories")
            assert len(result) == 3
            client.close()


class TestPulseSuggest:
    """Tests for debate topic suggestion operations."""

    def test_suggest_debate_topic_default(self) -> None:
        """Get a suggested debate topic with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "topic": "Should companies adopt AI coding assistants?",
                "score": 0.88,
                "reasoning": "High trending on HackerNews with diverse opinions",
                "source_topics": ["topic_1", "topic_2"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.pulse.suggest_debate_topic()

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/pulse/suggest",
                params={},
            )
            assert "topic" in result
            assert "reasoning" in result
            client.close()

    def test_suggest_debate_topic_with_source(self) -> None:
        """Get suggested topic from specific source."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"topic": "Reddit discussion topic"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.suggest_debate_topic(source="reddit")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["source"] == "reddit"
            client.close()

    def test_suggest_debate_topic_with_category(self) -> None:
        """Get suggested topic in specific category."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"topic": "Finance topic"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.suggest_debate_topic(category="finance")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["category"] == "finance"
            client.close()


class TestPulseHistory:
    """Tests for topic history operations."""

    def test_get_history_default(self) -> None:
        """Get trending history with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "history": [
                    {"date": "2024-01-15", "score": 0.9},
                    {"date": "2024-01-14", "score": 0.85},
                    {"date": "2024-01-13", "score": 0.78},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.pulse.get_history("topic_123")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/pulse/topics/topic_123/history",
                params={"days": 7},
            )
            assert len(result) == 3
            client.close()

    def test_get_history_custom_days(self) -> None:
        """Get trending history with custom days."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"history": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.pulse.get_history("topic_123", days=30)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["days"] == 30
            client.close()


class TestAsyncPulse:
    """Tests for async pulse API."""

    @pytest.mark.asyncio
    async def test_async_get_trending(self) -> None:
        """Get trending topics asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"topics": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.pulse.get_trending()

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/pulse/trending",
                    params={"limit": 20},
                )
                assert "topics" in result

    @pytest.mark.asyncio
    async def test_async_get_trending_with_source(self) -> None:
        """Get trending topics with source filter asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"topics": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.pulse.get_trending(source="hackernews", limit=10)

                call_args = mock_request.call_args
                params = call_args[1]["params"]
                assert params["source"] == "hackernews"
                assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_async_get_topic(self) -> None:
        """Get topic details asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "topic_123", "title": "AI Topic"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.pulse.get_topic("topic_123")

                mock_request.assert_called_once_with("GET", "/api/v1/pulse/topics/topic_123")
                assert result["id"] == "topic_123"

    @pytest.mark.asyncio
    async def test_async_search(self) -> None:
        """Search topics asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.pulse.search(
                    "machine learning", source="reddit", min_score=0.5
                )

                call_args = mock_request.call_args
                params = call_args[1]["params"]
                assert params["q"] == "machine learning"
                assert params["source"] == "reddit"
                assert params["min_score"] == 0.5
                assert "results" in result

    @pytest.mark.asyncio
    async def test_async_get_stats(self) -> None:
        """Get stats asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"total_topics": 1000}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.pulse.get_stats()

                mock_request.assert_called_once_with("GET", "/api/v1/pulse/stats")
                assert result["total_topics"] == 1000

    @pytest.mark.asyncio
    async def test_async_get_sources(self) -> None:
        """Get sources asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"sources": [{"name": "hackernews"}]}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.pulse.get_sources()

                mock_request.assert_called_once_with("GET", "/api/v1/pulse/sources")
                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_async_suggest_debate_topic(self) -> None:
        """Get suggested debate topic asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"topic": "Async suggested topic"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.pulse.suggest_debate_topic(
                    source="hackernews", category="technology"
                )

                call_args = mock_request.call_args
                params = call_args[1]["params"]
                assert params["source"] == "hackernews"
                assert params["category"] == "technology"
                assert "topic" in result

    @pytest.mark.asyncio
    async def test_async_get_history(self) -> None:
        """Get topic history asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"history": [{"date": "2024-01-15"}]}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.pulse.get_history("topic_123", days=14)

                call_args = mock_request.call_args
                assert call_args[1]["params"]["days"] == 14
                assert len(result) == 1
