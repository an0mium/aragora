"""Tests for orchestration data models.

Covers all enums, dataclasses, and parsing methods in
``aragora.server.handlers.orchestration.models``:

- TeamStrategy enum values
- OutputFormat enum values
- KnowledgeContextSource (construction, from_string)
- OutputChannel (construction, from_string with URL/thread handling)
- OrchestrationRequest (construction, from_dict with all parsing branches)
- OrchestrationResult (construction, to_dict serialization)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from aragora.config import MAX_ROUNDS
from aragora.server.handlers.orchestration.models import (
    KnowledgeContextSource,
    OrchestrationRequest,
    OrchestrationResult,
    OutputChannel,
    OutputFormat,
    TeamStrategy,
)


# ============================================================================
# A. TeamStrategy Enum
# ============================================================================


class TestTeamStrategy:
    """Verify every variant of the TeamStrategy enum."""

    def test_specified_value(self):
        assert TeamStrategy.SPECIFIED.value == "specified"

    def test_best_for_domain_value(self):
        assert TeamStrategy.BEST_FOR_DOMAIN.value == "best_for_domain"

    def test_diverse_value(self):
        assert TeamStrategy.DIVERSE.value == "diverse"

    def test_fast_value(self):
        assert TeamStrategy.FAST.value == "fast"

    def test_random_value(self):
        assert TeamStrategy.RANDOM.value == "random"

    def test_total_member_count(self):
        assert len(TeamStrategy) == 5

    def test_construct_from_value(self):
        assert TeamStrategy("specified") is TeamStrategy.SPECIFIED
        assert TeamStrategy("diverse") is TeamStrategy.DIVERSE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TeamStrategy("nonexistent")

    def test_members_are_unique(self):
        values = [m.value for m in TeamStrategy]
        assert len(values) == len(set(values))


# ============================================================================
# B. OutputFormat Enum
# ============================================================================


class TestOutputFormat:
    """Verify every variant of the OutputFormat enum."""

    def test_standard_value(self):
        assert OutputFormat.STANDARD.value == "standard"

    def test_decision_receipt_value(self):
        assert OutputFormat.DECISION_RECEIPT.value == "decision_receipt"

    def test_summary_value(self):
        assert OutputFormat.SUMMARY.value == "summary"

    def test_github_review_value(self):
        assert OutputFormat.GITHUB_REVIEW.value == "github_review"

    def test_slack_message_value(self):
        assert OutputFormat.SLACK_MESSAGE.value == "slack_message"

    def test_total_member_count(self):
        assert len(OutputFormat) == 5

    def test_construct_from_value(self):
        assert OutputFormat("standard") is OutputFormat.STANDARD
        assert OutputFormat("slack_message") is OutputFormat.SLACK_MESSAGE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            OutputFormat("pdf")


# ============================================================================
# C. KnowledgeContextSource
# ============================================================================


class TestKnowledgeContextSource:
    """Test KnowledgeContextSource construction and parsing."""

    # -- Direct construction -------------------------------------------------

    def test_defaults(self):
        src = KnowledgeContextSource(source_type="slack", source_id="C123")
        assert src.source_type == "slack"
        assert src.source_id == "C123"
        assert src.lookback_minutes == 60
        assert src.max_items == 50

    def test_custom_values(self):
        src = KnowledgeContextSource(
            source_type="confluence",
            source_id="page-42",
            lookback_minutes=120,
            max_items=10,
        )
        assert src.lookback_minutes == 120
        assert src.max_items == 10

    # -- from_string ---------------------------------------------------------

    def test_from_string_with_colon(self):
        src = KnowledgeContextSource.from_string("slack:C123")
        assert src.source_type == "slack"
        assert src.source_id == "C123"

    def test_from_string_without_colon_defaults_to_document(self):
        src = KnowledgeContextSource.from_string("my-doc-id")
        assert src.source_type == "document"
        assert src.source_id == "my-doc-id"

    def test_from_string_preserves_colons_in_id(self):
        """Only the first colon separates type from id."""
        src = KnowledgeContextSource.from_string("github:org/repo:main")
        assert src.source_type == "github"
        assert src.source_id == "org/repo:main"

    def test_from_string_empty_id(self):
        src = KnowledgeContextSource.from_string("slack:")
        assert src.source_type == "slack"
        assert src.source_id == ""

    def test_from_string_empty_string(self):
        src = KnowledgeContextSource.from_string("")
        assert src.source_type == "document"
        assert src.source_id == ""

    def test_from_string_defaults_not_overridden(self):
        src = KnowledgeContextSource.from_string("jira:PROJ-123")
        assert src.lookback_minutes == 60
        assert src.max_items == 50


# ============================================================================
# D. OutputChannel
# ============================================================================


class TestOutputChannel:
    """Test OutputChannel construction and from_string parsing."""

    # -- Direct construction -------------------------------------------------

    def test_defaults(self):
        ch = OutputChannel(channel_type="slack", channel_id="C456")
        assert ch.channel_type == "slack"
        assert ch.channel_id == "C456"
        assert ch.thread_id is None

    def test_with_thread_id(self):
        ch = OutputChannel(channel_type="slack", channel_id="C456", thread_id="ts123")
        assert ch.thread_id == "ts123"

    # -- from_string: URL detection ------------------------------------------

    def test_from_string_https_url_becomes_webhook(self):
        ch = OutputChannel.from_string("https://example.com/hook")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "https://example.com/hook"

    def test_from_string_http_url_becomes_webhook(self):
        ch = OutputChannel.from_string("http://example.com/hook")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "http://example.com/hook"

    # -- from_string: no colon -----------------------------------------------

    def test_from_string_no_colon_defaults_to_webhook(self):
        ch = OutputChannel.from_string("plain-channel-id")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "plain-channel-id"

    # -- from_string: webhook prefix -----------------------------------------

    def test_from_string_webhook_prefix_with_url(self):
        ch = OutputChannel.from_string("webhook:https://example.com/hook")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "https://example.com/hook"

    def test_from_string_webhook_prefix_with_http_url(self):
        ch = OutputChannel.from_string("webhook:http://example.com/hook")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "http://example.com/hook"

    # -- from_string: http/https as channel_type prefix ----------------------

    def test_from_string_http_colon_prefix(self):
        """'http://...' parsed as type=http, remainder starts with //."""
        ch = OutputChannel.from_string("http://example.com/hook")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "http://example.com/hook"

    def test_from_string_https_colon_prefix(self):
        ch = OutputChannel.from_string("https://example.com/hook")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "https://example.com/hook"

    # -- from_string: normal type:id -----------------------------------------

    def test_from_string_slack_channel(self):
        ch = OutputChannel.from_string("slack:C789")
        assert ch.channel_type == "slack"
        assert ch.channel_id == "C789"
        assert ch.thread_id is None

    def test_from_string_teams_channel(self):
        ch = OutputChannel.from_string("teams:team-channel-1")
        assert ch.channel_type == "teams"
        assert ch.channel_id == "team-channel-1"

    def test_from_string_discord_channel(self):
        ch = OutputChannel.from_string("discord:123456789")
        assert ch.channel_type == "discord"
        assert ch.channel_id == "123456789"

    def test_from_string_email_channel(self):
        ch = OutputChannel.from_string("email:user@example.com")
        assert ch.channel_type == "email"
        assert ch.channel_id == "user@example.com"

    def test_from_string_telegram_channel(self):
        ch = OutputChannel.from_string("telegram:@mychannel")
        assert ch.channel_type == "telegram"
        assert ch.channel_id == "@mychannel"

    # -- from_string: type:id:thread_id --------------------------------------

    def test_from_string_with_thread_id(self):
        ch = OutputChannel.from_string("slack:C789:ts1234567")
        assert ch.channel_type == "slack"
        assert ch.channel_id == "C789"
        assert ch.thread_id == "ts1234567"

    def test_from_string_thread_id_preserves_rest(self):
        """Everything after second colon is the thread_id."""
        ch = OutputChannel.from_string("teams:chan:thread:extra")
        assert ch.channel_type == "teams"
        assert ch.channel_id == "chan"
        assert ch.thread_id == "thread:extra"

    # -- from_string: case normalization -------------------------------------

    def test_from_string_type_is_lowercased(self):
        ch = OutputChannel.from_string("SLACK:C789")
        assert ch.channel_type == "slack"

    def test_from_string_mixed_case_type(self):
        ch = OutputChannel.from_string("Teams:channel-1")
        assert ch.channel_type == "teams"

    # -- from_string: edge cases ---------------------------------------------

    def test_from_string_empty_id(self):
        ch = OutputChannel.from_string("slack:")
        assert ch.channel_type == "slack"
        assert ch.channel_id == ""

    def test_from_string_empty_string(self):
        ch = OutputChannel.from_string("")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == ""

    def test_from_string_webhook_plain_id(self):
        """webhook:some-id where id does not start with http."""
        ch = OutputChannel.from_string("webhook:hook-abc")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "hook-abc"


# ============================================================================
# E. OrchestrationRequest
# ============================================================================


class TestOrchestrationRequest:
    """Test OrchestrationRequest construction and from_dict parsing."""

    # -- Direct construction defaults ----------------------------------------

    def test_defaults(self):
        req = OrchestrationRequest(question="test?")
        assert req.question == "test?"
        assert req.knowledge_sources == []
        assert req.workspaces == []
        assert req.team_strategy is TeamStrategy.BEST_FOR_DOMAIN
        assert req.agents == []
        assert req.output_channels == []
        assert req.output_format is OutputFormat.STANDARD
        assert req.require_consensus is True
        assert req.priority == "normal"
        assert req.max_rounds == MAX_ROUNDS
        assert req.timeout_seconds == 300.0
        assert req.template is None
        assert req.notify is True
        assert req.dry_run is False
        assert req.metadata == {}
        # request_id is a UUID string
        uuid.UUID(req.request_id)  # validates format

    def test_request_ids_are_unique(self):
        r1 = OrchestrationRequest(question="a")
        r2 = OrchestrationRequest(question="b")
        assert r1.request_id != r2.request_id

    # -- from_dict: minimal --------------------------------------------------

    def test_from_dict_minimal(self):
        req = OrchestrationRequest.from_dict({"question": "hello?"})
        assert req.question == "hello?"
        assert req.knowledge_sources == []
        assert req.output_channels == []
        assert req.team_strategy is TeamStrategy.BEST_FOR_DOMAIN
        assert req.output_format is OutputFormat.STANDARD

    def test_from_dict_empty_dict(self):
        req = OrchestrationRequest.from_dict({})
        assert req.question == ""

    # -- from_dict: knowledge_sources (string list) --------------------------

    def test_from_dict_knowledge_sources_strings(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_sources": ["slack:C123", "my-doc"],
        })
        assert len(req.knowledge_sources) == 2
        assert req.knowledge_sources[0].source_type == "slack"
        assert req.knowledge_sources[0].source_id == "C123"
        assert req.knowledge_sources[1].source_type == "document"
        assert req.knowledge_sources[1].source_id == "my-doc"

    # -- from_dict: knowledge_sources (dict list) ----------------------------

    def test_from_dict_knowledge_sources_dicts(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_sources": [
                {"type": "confluence", "id": "page-1", "lookback_minutes": 30, "max_items": 5},
            ],
        })
        assert len(req.knowledge_sources) == 1
        src = req.knowledge_sources[0]
        assert src.source_type == "confluence"
        assert src.source_id == "page-1"
        assert src.lookback_minutes == 30
        assert src.max_items == 5

    def test_from_dict_knowledge_sources_dict_defaults(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_sources": [{"type": "github"}],
        })
        src = req.knowledge_sources[0]
        assert src.source_type == "github"
        assert src.source_id == ""
        assert src.lookback_minutes == 60
        assert src.max_items == 50

    def test_from_dict_knowledge_sources_dict_minimal(self):
        """Dict with no keys uses defaults."""
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_sources": [{}],
        })
        src = req.knowledge_sources[0]
        assert src.source_type == "document"
        assert src.source_id == ""

    # -- from_dict: knowledge_context nested format --------------------------

    def test_from_dict_knowledge_context_sources(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_context": {
                "sources": ["github:org/repo"],
            },
        })
        assert len(req.knowledge_sources) == 1
        assert req.knowledge_sources[0].source_type == "github"
        assert req.knowledge_sources[0].source_id == "org/repo"

    def test_from_dict_knowledge_context_workspaces(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_context": {
                "workspaces": ["ws-1", "ws-2"],
            },
        })
        assert req.workspaces == ["ws-1", "ws-2"]

    def test_from_dict_knowledge_context_combined_with_knowledge_sources(self):
        """Both knowledge_sources and knowledge_context.sources are merged."""
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_sources": ["slack:C111"],
            "knowledge_context": {
                "sources": ["github:org/repo"],
            },
        })
        assert len(req.knowledge_sources) == 2
        types = {s.source_type for s in req.knowledge_sources}
        assert "slack" in types
        assert "github" in types

    def test_from_dict_knowledge_context_empty(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_context": {},
        })
        assert req.knowledge_sources == []
        assert req.workspaces == []

    # -- from_dict: workspaces -----------------------------------------------

    def test_from_dict_workspaces_direct(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "workspaces": ["ws-a"],
        })
        assert req.workspaces == ["ws-a"]

    def test_from_dict_workspaces_direct_overrides_nested(self):
        """Top-level workspaces takes precedence over knowledge_context.workspaces."""
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "workspaces": ["ws-a"],
            "knowledge_context": {"workspaces": ["ws-b"]},
        })
        assert req.workspaces == ["ws-a"]

    # -- from_dict: output_channels (string list) ----------------------------

    def test_from_dict_output_channels_strings(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "output_channels": ["slack:C456", "https://hooks.example.com/abc"],
        })
        assert len(req.output_channels) == 2
        assert req.output_channels[0].channel_type == "slack"
        assert req.output_channels[1].channel_type == "webhook"

    # -- from_dict: output_channels (dict list) ------------------------------

    def test_from_dict_output_channels_dicts(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "output_channels": [
                {"type": "teams", "id": "T123", "thread_id": "th-1"},
            ],
        })
        assert len(req.output_channels) == 1
        ch = req.output_channels[0]
        assert ch.channel_type == "teams"
        assert ch.channel_id == "T123"
        assert ch.thread_id == "th-1"

    def test_from_dict_output_channels_dict_defaults(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "output_channels": [{}],
        })
        ch = req.output_channels[0]
        assert ch.channel_type == "webhook"
        assert ch.channel_id == ""
        assert ch.thread_id is None

    # -- from_dict: team_strategy --------------------------------------------

    def test_from_dict_team_strategy_valid(self):
        for strategy in TeamStrategy:
            req = OrchestrationRequest.from_dict({
                "question": "q",
                "team_strategy": strategy.value,
            })
            assert req.team_strategy is strategy

    def test_from_dict_team_strategy_invalid_falls_back(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "team_strategy": "nonexistent_strategy",
        })
        assert req.team_strategy is TeamStrategy.BEST_FOR_DOMAIN

    def test_from_dict_team_strategy_absent_defaults(self):
        req = OrchestrationRequest.from_dict({"question": "q"})
        assert req.team_strategy is TeamStrategy.BEST_FOR_DOMAIN

    # -- from_dict: output_format --------------------------------------------

    def test_from_dict_output_format_valid(self):
        for fmt in OutputFormat:
            req = OrchestrationRequest.from_dict({
                "question": "q",
                "output_format": fmt.value,
            })
            assert req.output_format is fmt

    def test_from_dict_output_format_invalid_falls_back(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "output_format": "pdf",
        })
        assert req.output_format is OutputFormat.STANDARD

    def test_from_dict_output_format_absent_defaults(self):
        req = OrchestrationRequest.from_dict({"question": "q"})
        assert req.output_format is OutputFormat.STANDARD

    # -- from_dict: scalar fields --------------------------------------------

    def test_from_dict_agents(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "agents": ["claude", "gpt-4"],
        })
        assert req.agents == ["claude", "gpt-4"]

    def test_from_dict_require_consensus(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "require_consensus": False,
        })
        assert req.require_consensus is False

    def test_from_dict_priority(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "priority": "high",
        })
        assert req.priority == "high"

    def test_from_dict_max_rounds(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "max_rounds": 5,
        })
        assert req.max_rounds == 5

    def test_from_dict_max_rounds_default(self):
        req = OrchestrationRequest.from_dict({"question": "q"})
        assert req.max_rounds == MAX_ROUNDS

    def test_from_dict_timeout_seconds(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "timeout_seconds": 60.0,
        })
        assert req.timeout_seconds == 60.0

    def test_from_dict_template(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "template": "code_review",
        })
        assert req.template == "code_review"

    def test_from_dict_template_none(self):
        req = OrchestrationRequest.from_dict({"question": "q"})
        assert req.template is None

    def test_from_dict_notify(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "notify": False,
        })
        assert req.notify is False

    def test_from_dict_dry_run(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "dry_run": True,
        })
        assert req.dry_run is True

    def test_from_dict_metadata(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "metadata": {"source": "api", "user": "u1"},
        })
        assert req.metadata == {"source": "api", "user": "u1"}

    # -- from_dict: full payload ---------------------------------------------

    def test_from_dict_full_payload(self):
        data = {
            "question": "Should we migrate to K8s?",
            "knowledge_sources": ["slack:C001"],
            "knowledge_context": {"sources": ["github:org/repo"], "workspaces": ["ws-nested"]},
            "workspaces": ["ws-top"],
            "team_strategy": "diverse",
            "agents": ["claude", "gpt-4"],
            "output_channels": [
                "slack:C002",
                {"type": "email", "id": "admin@co.com"},
            ],
            "output_format": "decision_receipt",
            "require_consensus": False,
            "priority": "critical",
            "max_rounds": 3,
            "timeout_seconds": 120.0,
            "template": "architecture_review",
            "notify": False,
            "dry_run": True,
            "metadata": {"project": "infra"},
        }
        req = OrchestrationRequest.from_dict(data)

        assert req.question == "Should we migrate to K8s?"
        assert len(req.knowledge_sources) == 2  # slack + github
        assert req.workspaces == ["ws-top"]  # direct overrides nested
        assert req.team_strategy is TeamStrategy.DIVERSE
        assert req.agents == ["claude", "gpt-4"]
        assert len(req.output_channels) == 2
        assert req.output_format is OutputFormat.DECISION_RECEIPT
        assert req.require_consensus is False
        assert req.priority == "critical"
        assert req.max_rounds == 3
        assert req.timeout_seconds == 120.0
        assert req.template == "architecture_review"
        assert req.notify is False
        assert req.dry_run is True
        assert req.metadata == {"project": "infra"}

    # -- from_dict: mixed knowledge source types -----------------------------

    def test_from_dict_mixed_string_and_dict_sources(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_sources": [
                "slack:C111",
                {"type": "confluence", "id": "pg-1"},
            ],
        })
        assert len(req.knowledge_sources) == 2
        assert req.knowledge_sources[0].source_type == "slack"
        assert req.knowledge_sources[1].source_type == "confluence"

    # -- from_dict: mixed output channel types -------------------------------

    def test_from_dict_mixed_string_and_dict_channels(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "output_channels": [
                "slack:C789:ts1",
                {"type": "discord", "id": "D123"},
            ],
        })
        assert len(req.output_channels) == 2
        assert req.output_channels[0].channel_type == "slack"
        assert req.output_channels[0].thread_id == "ts1"
        assert req.output_channels[1].channel_type == "discord"

    # -- from_dict: ignoring unknown fields ----------------------------------

    def test_from_dict_ignores_unknown_fields(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "unknown_field": "ignored",
            "another": 42,
        })
        assert req.question == "q"


# ============================================================================
# F. OrchestrationResult
# ============================================================================


class TestOrchestrationResult:
    """Test OrchestrationResult construction and serialization."""

    # -- Direct construction defaults ----------------------------------------

    def test_defaults(self):
        result = OrchestrationResult(request_id="req-1", success=True)
        assert result.request_id == "req-1"
        assert result.success is True
        assert result.consensus_reached is False
        assert result.final_answer is None
        assert result.confidence is None
        assert result.agents_participated == []
        assert result.rounds_completed == 0
        assert result.duration_seconds == 0.0
        assert result.knowledge_context_used == []
        assert result.channels_notified == []
        assert result.receipt_id is None
        assert result.error is None
        # created_at is a valid ISO timestamp
        datetime.fromisoformat(result.created_at)

    def test_failure_result(self):
        result = OrchestrationResult(
            request_id="req-2",
            success=False,
            error="Timeout exceeded",
        )
        assert result.success is False
        assert result.error == "Timeout exceeded"

    def test_full_result(self):
        result = OrchestrationResult(
            request_id="req-3",
            success=True,
            consensus_reached=True,
            final_answer="Yes, adopt microservices.",
            confidence=0.87,
            agents_participated=["claude", "gpt-4", "gemini"],
            rounds_completed=5,
            duration_seconds=42.5,
            knowledge_context_used=["slack:C123", "confluence:pg-1"],
            channels_notified=["slack:C456"],
            receipt_id="rcpt-abc-123",
        )
        assert result.confidence == 0.87
        assert len(result.agents_participated) == 3
        assert result.rounds_completed == 5
        assert result.receipt_id == "rcpt-abc-123"

    # -- to_dict -------------------------------------------------------------

    def test_to_dict_keys(self):
        result = OrchestrationResult(request_id="r1", success=True)
        d = result.to_dict()
        expected_keys = {
            "request_id",
            "success",
            "consensus_reached",
            "final_answer",
            "confidence",
            "agents_participated",
            "rounds_completed",
            "duration_seconds",
            "knowledge_context_used",
            "channels_notified",
            "receipt_id",
            "error",
            "created_at",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        result = OrchestrationResult(
            request_id="r2",
            success=True,
            consensus_reached=True,
            final_answer="Go ahead.",
            confidence=0.95,
            agents_participated=["claude"],
            rounds_completed=3,
            duration_seconds=12.3,
            knowledge_context_used=["doc-1"],
            channels_notified=["slack:C1"],
            receipt_id="rcpt-1",
        )
        d = result.to_dict()
        assert d["request_id"] == "r2"
        assert d["success"] is True
        assert d["consensus_reached"] is True
        assert d["final_answer"] == "Go ahead."
        assert d["confidence"] == 0.95
        assert d["agents_participated"] == ["claude"]
        assert d["rounds_completed"] == 3
        assert d["duration_seconds"] == 12.3
        assert d["knowledge_context_used"] == ["doc-1"]
        assert d["channels_notified"] == ["slack:C1"]
        assert d["receipt_id"] == "rcpt-1"
        assert d["error"] is None

    def test_to_dict_failure(self):
        result = OrchestrationResult(
            request_id="r3",
            success=False,
            error="Agent unavailable",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "Agent unavailable"
        assert d["final_answer"] is None

    def test_to_dict_created_at_is_iso(self):
        result = OrchestrationResult(request_id="r4", success=True)
        d = result.to_dict()
        # Should be parseable as ISO datetime
        datetime.fromisoformat(d["created_at"])

    def test_to_dict_returns_plain_dict(self):
        """Verify the return type is a plain dict, not a dataclass."""
        result = OrchestrationResult(request_id="r5", success=True)
        d = result.to_dict()
        assert type(d) is dict

    def test_to_dict_lists_share_references(self):
        """to_dict returns the same list objects (shallow copy)."""
        result = OrchestrationResult(
            request_id="r6",
            success=True,
            agents_participated=["claude"],
        )
        d = result.to_dict()
        # to_dict does not deep-copy; lists are the same objects
        assert d["agents_participated"] is result.agents_participated

    def test_to_dict_roundtrip_preserves_data(self):
        """Converting to dict and checking all fields match."""
        result = OrchestrationResult(
            request_id="r7",
            success=True,
            consensus_reached=True,
            final_answer="approved",
            confidence=0.99,
            agents_participated=["a", "b"],
            rounds_completed=10,
            duration_seconds=99.9,
            knowledge_context_used=["k1", "k2"],
            channels_notified=["ch1"],
            receipt_id="rcpt-7",
            error=None,
        )
        d = result.to_dict()
        assert d["request_id"] == result.request_id
        assert d["success"] == result.success
        assert d["consensus_reached"] == result.consensus_reached
        assert d["final_answer"] == result.final_answer
        assert d["confidence"] == result.confidence
        assert d["agents_participated"] == result.agents_participated
        assert d["rounds_completed"] == result.rounds_completed
        assert d["duration_seconds"] == result.duration_seconds
        assert d["knowledge_context_used"] == result.knowledge_context_used
        assert d["channels_notified"] == result.channels_notified
        assert d["receipt_id"] == result.receipt_id
        assert d["error"] == result.error
        assert d["created_at"] == result.created_at


# ============================================================================
# G. Edge Cases / Cross-Cutting
# ============================================================================


class TestEdgeCases:
    """Cross-cutting edge cases for the models module."""

    def test_knowledge_source_equality(self):
        a = KnowledgeContextSource(source_type="slack", source_id="C1")
        b = KnowledgeContextSource(source_type="slack", source_id="C1")
        assert a == b

    def test_knowledge_source_inequality(self):
        a = KnowledgeContextSource(source_type="slack", source_id="C1")
        b = KnowledgeContextSource(source_type="github", source_id="C1")
        assert a != b

    def test_output_channel_equality(self):
        a = OutputChannel(channel_type="slack", channel_id="C1", thread_id="t1")
        b = OutputChannel(channel_type="slack", channel_id="C1", thread_id="t1")
        assert a == b

    def test_output_channel_inequality_thread(self):
        a = OutputChannel(channel_type="slack", channel_id="C1", thread_id="t1")
        b = OutputChannel(channel_type="slack", channel_id="C1", thread_id="t2")
        assert a != b

    def test_orchestration_result_with_zero_confidence(self):
        result = OrchestrationResult(request_id="r", success=True, confidence=0.0)
        d = result.to_dict()
        assert d["confidence"] == 0.0

    def test_orchestration_result_with_negative_duration(self):
        """No validation prevents negative durations - this is just data."""
        result = OrchestrationResult(request_id="r", success=True, duration_seconds=-1.0)
        assert result.duration_seconds == -1.0

    def test_from_dict_knowledge_context_without_sources_key(self):
        """knowledge_context dict without 'sources' key should not add sources."""
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "knowledge_context": {"workspaces": ["w1"]},
        })
        assert req.knowledge_sources == []

    def test_output_channel_from_string_webhook_non_http_id(self):
        """webhook:some-plain-id should keep channel_type=webhook."""
        ch = OutputChannel.from_string("webhook:my-hook-id")
        assert ch.channel_type == "webhook"
        assert ch.channel_id == "my-hook-id"

    def test_team_strategy_is_enum_member(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "team_strategy": "fast",
        })
        assert isinstance(req.team_strategy, TeamStrategy)

    def test_output_format_is_enum_member(self):
        req = OrchestrationRequest.from_dict({
            "question": "q",
            "output_format": "summary",
        })
        assert isinstance(req.output_format, OutputFormat)

    def test_request_metadata_default_is_independent(self):
        """Each request should get its own metadata dict."""
        r1 = OrchestrationRequest(question="a")
        r2 = OrchestrationRequest(question="b")
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata

    def test_request_agents_default_is_independent(self):
        """Each request should get its own agents list."""
        r1 = OrchestrationRequest(question="a")
        r2 = OrchestrationRequest(question="b")
        r1.agents.append("claude")
        assert r2.agents == []

    def test_result_agents_participated_default_is_independent(self):
        """Each result should get its own agents_participated list."""
        r1 = OrchestrationResult(request_id="1", success=True)
        r2 = OrchestrationResult(request_id="2", success=True)
        r1.agents_participated.append("claude")
        assert r2.agents_participated == []
