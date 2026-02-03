"""Tests for TwitterPosterConnector - init, config, formatting."""

import pytest

from aragora.connectors.twitter_poster import (
    DebateContentFormatter,
    ThreadResult,
    TweetResult,
    TwitterPosterConnector,
    TwitterRateLimiter,
    create_debate_summary,
)


class TestTwitterPosterInit:
    """Initialization and configuration checks."""

    def test_default_init_unconfigured(self, monkeypatch):
        monkeypatch.delenv("TWITTER_API_KEY", raising=False)
        monkeypatch.delenv("TWITTER_API_SECRET", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_SECRET", raising=False)
        connector = TwitterPosterConnector()
        assert not connector.is_configured

    def test_explicit_credentials(self):
        connector = TwitterPosterConnector(
            api_key="k", api_secret="s", access_token="t", access_secret="a"
        )
        assert connector.is_configured

    def test_partial_credentials_not_configured(self):
        connector = TwitterPosterConnector(api_key="k")
        assert not connector.is_configured


class TestPostTweetUnconfigured:
    """Posting without credentials returns graceful failure."""

    @pytest.mark.asyncio
    async def test_post_tweet_unconfigured(self):
        connector = TwitterPosterConnector()
        result = await connector.post_tweet("Hello world")
        assert isinstance(result, TweetResult)
        assert not result.success
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_post_thread_empty(self):
        connector = TwitterPosterConnector(
            api_key="k", api_secret="s", access_token="t", access_secret="a"
        )
        result = await connector.post_thread([])
        assert isinstance(result, ThreadResult)
        assert not result.success
        assert "No tweets" in result.error


class TestTwitterRateLimiter:
    """Rate limiter behavior."""

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self):
        limiter = TwitterRateLimiter(calls_per_window=10, window_seconds=900)
        # Should not block
        await limiter.acquire()
        assert len(limiter.call_times) == 1


class TestDebateContentFormatter:
    """Content formatting for Twitter."""

    def test_format_announcement(self):
        formatter = DebateContentFormatter()
        tweet = formatter.format_announcement(
            task="Should we use microservices?",
            agents=["Claude", "GPT-4", "Gemini"],
        )
        assert "microservices" in tweet
        assert len(tweet) <= 280

    def test_format_announcement_truncation(self):
        formatter = DebateContentFormatter()
        tweet = formatter.format_announcement(
            task="x" * 300,
            agents=["Agent1"],
        )
        assert len(tweet) <= 280

    def test_format_result_consensus(self):
        formatter = DebateContentFormatter()
        tweet = formatter.format_result(
            task="Rate limiting strategy",
            agents=["Claude", "GPT-4"],
            consensus_reached=True,
        )
        assert "Consensus" in tweet

    def test_format_result_winner(self):
        formatter = DebateContentFormatter()
        tweet = formatter.format_result(
            task="Architecture choice",
            agents=["Claude", "GPT-4"],
            consensus_reached=False,
            winner="Claude",
        )
        assert "Claude" in tweet

    def test_format_thread(self):
        formatter = DebateContentFormatter()
        tweets = formatter.format_thread(
            task="API design",
            agents=["Claude", "GPT-4"],
            highlights=["Key point 1", "Key point 2"],
            consensus_reached=True,
        )
        assert len(tweets) >= 3  # intro + highlights + result
        assert all(len(t) <= 280 for t in tweets)

    def test_create_debate_summary(self):
        summary = create_debate_summary(
            task="Test topic",
            agents=["Agent1", "Agent2"],
            consensus_reached=False,
        )
        assert len(summary) <= 280
        assert "Test topic" in summary


class TestTweetResultDataclass:
    """TweetResult and ThreadResult basic behavior."""

    def test_tweet_result_defaults(self):
        r = TweetResult(tweet_id="123", text="hi", created_at="now", url="u")
        assert r.success is True
        assert r.error is None

    def test_thread_result_url(self):
        tweets = [TweetResult(tweet_id="1", text="a", created_at="now", url="https://t.co/1")]
        r = ThreadResult(thread_id="1", tweets=tweets)
        assert r.url == "https://t.co/1"

    def test_thread_result_empty_url(self):
        r = ThreadResult(thread_id="")
        assert r.url == ""
