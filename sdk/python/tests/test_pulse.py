"""Tests for Pulse namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


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

