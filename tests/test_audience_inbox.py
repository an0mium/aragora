"""
Tests for AudienceInbox, TokenBucket rate limiting, and conviction-weighted voting.

Moved from aragora/server/test_audience_inbox.py to proper test location.
"""

import time
import threading

import pytest

from aragora.server.stream import (
    TokenBucket, AudienceInbox, AudienceMessage, normalize_intensity
)
from aragora.debate.protocol import DebateProtocol, user_vote_multiplier


class TestTokenBucket:
    """Tests for TokenBucket rate limiting."""

    def test_token_bucket_basic(self):
        """Test token bucket rate limiting."""
        # 5 tokens per minute = 1 token every 12 seconds
        bucket = TokenBucket(rate_per_minute=5.0, burst_size=10)

        # Should have full tokens initially
        assert bucket.consume(10) == True
        assert bucket.consume(1) == False  # Should be empty

    @pytest.mark.slow
    def test_token_bucket_refill(self):
        """Test token bucket refills over time."""
        bucket = TokenBucket(rate_per_minute=5.0, burst_size=10)

        # Consume all tokens
        assert bucket.consume(10) == True
        assert bucket.consume(1) == False

        # Wait for tokens to refill (at least 12 seconds for 1 token)
        time.sleep(13)

        # Should have at least 1 token now
        assert bucket.consume(1) == True


class TestAudienceInbox:
    """Tests for AudienceInbox message queuing."""

    def test_audience_inbox_basic(self):
        """Test thread-safe audience message queuing."""
        inbox = AudienceInbox()

        # Add some messages
        messages = [
            AudienceMessage(type="vote", loop_id="test-loop", payload={"choice": "option1"}),
            AudienceMessage(type="suggestion", loop_id="test-loop", payload={"suggestion": "Great idea!"}),
            AudienceMessage(type="vote", loop_id="test-loop", payload={"choice": "option2"}),
        ]

        for msg in messages:
            inbox.put(msg)

        # Drain messages
        drained = inbox.get_all()

        assert len(drained) == 3
        assert drained[0].type == "vote"
        assert drained[0].payload["choice"] == "option1"
        assert drained[1].type == "suggestion"
        assert drained[1].payload["suggestion"] == "Great idea!"
        assert drained[2].type == "vote"
        assert drained[2].payload["choice"] == "option2"

        # Should be empty now
        assert len(inbox.get_all()) == 0

    def test_audience_inbox_thread_safety(self):
        """Test that AudienceInbox is thread-safe."""
        inbox = AudienceInbox()
        results = []

        def producer():
            for i in range(100):
                msg = AudienceMessage(
                    type="vote",
                    loop_id="test-loop",
                    payload={"choice": f"option{i}"}
                )
                inbox.put(msg)

        def consumer():
            messages = inbox.get_all()
            results.append(len(messages))

        # Start producer and consumer threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        producer_thread.join()  # Wait for producer to finish

        consumer_thread.start()
        consumer_thread.join()

        # Should have consumed all 100 messages
        assert sum(results) == 100

    def test_get_summary_with_histograms(self):
        """Test get_summary includes conviction histograms."""
        inbox = AudienceInbox()

        # Add votes with different intensities
        votes = [
            AudienceMessage(type="vote", loop_id="test", payload={"choice": "A", "intensity": 8}),
            AudienceMessage(type="vote", loop_id="test", payload={"choice": "A", "intensity": 9}),
            AudienceMessage(type="vote", loop_id="test", payload={"choice": "B", "intensity": 3}),
            AudienceMessage(type="vote", loop_id="test", payload={"choice": "B", "intensity": 4}),
            AudienceMessage(type="suggestion", loop_id="test", payload={"text": "Great!"}),
        ]

        for v in votes:
            inbox.put(v)

        summary = inbox.get_summary()

        # Basic counts
        assert summary["votes"]["A"] == 2
        assert summary["votes"]["B"] == 2
        assert summary["suggestions"] == 1

        # Histograms present
        assert "histograms" in summary
        assert "A" in summary["histograms"]
        assert "B" in summary["histograms"]

        # Check histogram values
        assert summary["histograms"]["A"][8] == 1
        assert summary["histograms"]["A"][9] == 1
        assert summary["histograms"]["B"][3] == 1
        assert summary["histograms"]["B"][4] == 1

        # Conviction distribution
        assert "conviction_distribution" in summary
        assert summary["conviction_distribution"][8] == 1
        assert summary["conviction_distribution"][9] == 1
        assert summary["conviction_distribution"][3] == 1
        assert summary["conviction_distribution"][4] == 1

        # Weighted votes (high intensity votes count more)
        assert summary["weighted_votes"]["A"] > summary["weighted_votes"]["B"]

    def test_get_summary_loop_id_filter(self):
        """Test get_summary filters by loop_id."""
        inbox = AudienceInbox()

        # Add votes for different loops
        inbox.put(AudienceMessage(type="vote", loop_id="loop1", payload={"choice": "X", "intensity": 5}))
        inbox.put(AudienceMessage(type="vote", loop_id="loop1", payload={"choice": "Y", "intensity": 7}))
        inbox.put(AudienceMessage(type="vote", loop_id="loop2", payload={"choice": "X", "intensity": 3}))

        # Filter by loop1
        summary1 = inbox.get_summary(loop_id="loop1")
        assert summary1["votes"]["X"] == 1
        assert summary1["votes"]["Y"] == 1
        assert "loop2" not in str(summary1["votes"])

        # Filter by loop2
        summary2 = inbox.get_summary(loop_id="loop2")
        assert summary2["votes"]["X"] == 1
        assert "Y" not in summary2["votes"]

        # No filter = all
        summary_all = inbox.get_summary()
        assert summary_all["votes"]["X"] == 2
        assert summary_all["votes"]["Y"] == 1


class TestNormalizeIntensity:
    """Tests for intensity normalization."""

    def test_normal_values(self):
        """Test intensity normalization for normal values."""
        assert normalize_intensity(5) == 5
        assert normalize_intensity(1) == 1
        assert normalize_intensity(10) == 10

    def test_clamping(self):
        """Test intensity clamping at boundaries."""
        assert normalize_intensity(0) == 1   # Below min
        assert normalize_intensity(11) == 10  # Above max
        assert normalize_intensity(-5) == 1  # Negative

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        assert normalize_intensity(None) == 5  # None -> default
        assert normalize_intensity("invalid") == 5  # String -> default
        assert normalize_intensity(5.7) == 5  # Float -> truncated int

    def test_edge_cases(self):
        """Test edge cases."""
        assert normalize_intensity("3") == 3  # String number
        assert normalize_intensity(3.9) == 3  # Float truncation


class TestUserVoteMultiplier:
    """Tests for conviction-weighted vote multiplier."""

    def test_neutral_intensity(self):
        """Test neutral intensity = 1.0 multiplier."""
        protocol = DebateProtocol()
        assert user_vote_multiplier(5, protocol) == 1.0

    def test_low_conviction(self):
        """Test low conviction = lower weight."""
        protocol = DebateProtocol()
        low = user_vote_multiplier(1, protocol)
        assert low == 0.5  # Min multiplier

    def test_high_conviction(self):
        """Test high conviction = higher weight."""
        protocol = DebateProtocol()
        high = user_vote_multiplier(10, protocol)
        assert high == 2.0  # Max multiplier

    def test_intermediate_values(self):
        """Test intermediate conviction values."""
        protocol = DebateProtocol()

        mid_low = user_vote_multiplier(3, protocol)
        assert 0.5 < mid_low < 1.0

        mid_high = user_vote_multiplier(7, protocol)
        assert 1.0 < mid_high < 2.0

    def test_clamping_at_edges(self):
        """Test clamping at edges."""
        protocol = DebateProtocol()
        assert user_vote_multiplier(0, protocol) == 0.5  # Below min -> min
        assert user_vote_multiplier(15, protocol) == 2.0  # Above max -> max
