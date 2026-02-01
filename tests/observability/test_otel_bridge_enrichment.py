"""
Tests for OTel bridge span enrichment functions.

Tests the debate-specific and HTTP-specific span enrichment helpers
added to aragora.server.middleware.otel_bridge.
"""

from unittest.mock import MagicMock

import pytest

from aragora.server.middleware.otel_bridge import (
    enrich_span_with_debate_context,
    enrich_span_with_http_context,
)


class TestEnrichSpanWithDebateContext:
    """Tests for enrich_span_with_debate_context."""

    def test_with_none_span(self):
        """Test enrichment with None span does not raise."""
        enrich_span_with_debate_context(None, debate_id="d-1")

    def test_with_all_attributes(self):
        """Test enrichment sets all debate attributes."""
        span = MagicMock()
        enrich_span_with_debate_context(
            span,
            debate_id="debate-123",
            round_number=2,
            agent_name="claude",
            phase="propose",
        )
        span.set_attribute.assert_any_call("debate.id", "debate-123")
        span.set_attribute.assert_any_call("debate.round_number", 2)
        span.set_attribute.assert_any_call("agent.name", "claude")
        span.set_attribute.assert_any_call("debate.phase", "propose")

    def test_with_partial_attributes(self):
        """Test enrichment with only some attributes."""
        span = MagicMock()
        enrich_span_with_debate_context(span, debate_id="d-1")
        span.set_attribute.assert_called_once_with("debate.id", "d-1")

    def test_with_no_attributes(self):
        """Test enrichment with no optional attributes does not call set_attribute."""
        span = MagicMock()
        enrich_span_with_debate_context(span)
        span.set_attribute.assert_not_called()

    def test_with_span_using_set_tag(self):
        """Test enrichment falls back to set_tag if set_attribute is not available."""
        span = MagicMock(spec=[])  # No spec, no set_attribute
        # Create span that only has set_tag
        span_with_tag = MagicMock()
        del span_with_tag.set_attribute  # Remove set_attribute
        span_with_tag.set_tag = MagicMock()

        enrich_span_with_debate_context(span_with_tag, debate_id="d-2")
        span_with_tag.set_tag.assert_called_once_with("debate.id", "d-2")


class TestEnrichSpanWithHttpContext:
    """Tests for enrich_span_with_http_context."""

    def test_with_none_span(self):
        """Test enrichment with None span does not raise."""
        enrich_span_with_http_context(None, method="GET", path="/api/test")

    def test_with_all_attributes(self):
        """Test enrichment sets all HTTP attributes."""
        span = MagicMock()
        enrich_span_with_http_context(
            span,
            method="POST",
            path="/api/debates",
            status_code=201,
            client_ip="192.168.1.1",
            user_agent="TestAgent/1.0",
        )
        span.set_attribute.assert_any_call("http.method", "POST")
        span.set_attribute.assert_any_call("http.target", "/api/debates")
        span.set_attribute.assert_any_call("http.status_code", 201)
        span.set_attribute.assert_any_call("net.peer.ip", "192.168.1.1")
        span.set_attribute.assert_any_call("http.user_agent", "TestAgent/1.0")

    def test_with_partial_attributes(self):
        """Test enrichment with only method and path."""
        span = MagicMock()
        enrich_span_with_http_context(span, method="GET", path="/health")
        assert span.set_attribute.call_count == 2

    def test_user_agent_truncation(self):
        """Test very long user agent strings are truncated."""
        span = MagicMock()
        long_ua = "A" * 500
        enrich_span_with_http_context(span, user_agent=long_ua)
        # The user_agent should be truncated to 200 chars
        call_args = span.set_attribute.call_args_list[0]
        assert call_args[0][0] == "http.user_agent"
        assert len(call_args[0][1]) == 200

    def test_with_no_attributes(self):
        """Test enrichment with no optional attributes."""
        span = MagicMock()
        enrich_span_with_http_context(span)
        span.set_attribute.assert_not_called()

    def test_with_status_code_zero(self):
        """Test status_code=0 is still set (not falsy-skipped)."""
        span = MagicMock()
        enrich_span_with_http_context(span, status_code=0)
        span.set_attribute.assert_called_once_with("http.status_code", 0)
