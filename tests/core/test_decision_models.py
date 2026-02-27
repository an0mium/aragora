"""
Tests for aragora.core.decision_models module.

Covers the factory methods, utilities, and validation logic
that are not exercised by the existing test_decision.py suite:

- normalize_document_ids() - pure utility with multiple branches
- DecisionRequest.__post_init__ validation
- DecisionRequest.from_chat_message() - chat platform factory
- DecisionRequest.from_http() - HTTP request factory (unified + legacy)
- DecisionRequest.from_voice() - voice transcription factory
- DecisionRequest.from_document() - document source factory
- DecisionRequest._detect_decision_type() - auto-detection heuristic
"""

import uuid
from datetime import datetime

import pytest

from aragora.core.decision_models import (
    DecisionConfig,
    DecisionRequest,
    DecisionResult,
    RequestContext,
    ResponseChannel,
    normalize_document_ids,
)
from aragora.core.decision_types import (
    DecisionType,
    InputSource,
    Priority,
)


# ---------------------------------------------------------------------------
# normalize_document_ids
# ---------------------------------------------------------------------------


class TestNormalizeDocumentIds:
    """Tests for the normalize_document_ids utility function."""

    def test_none_returns_empty_list(self):
        """None input returns an empty list."""
        assert normalize_document_ids(None) == []

    def test_empty_string_returns_empty_list(self):
        """Empty string is falsy, returns empty list."""
        assert normalize_document_ids("") == []

    def test_empty_list_returns_empty_list(self):
        """Empty list is falsy, returns empty list."""
        assert normalize_document_ids([]) == []

    def test_zero_returns_empty_list(self):
        """Zero (falsy non-None/non-str/non-list) returns empty list."""
        assert normalize_document_ids(0) == []

    def test_single_string_wrapped_in_list(self):
        """A single string value is normalized to a one-element list."""
        assert normalize_document_ids("doc-42") == ["doc-42"]

    def test_string_is_stripped(self):
        """Leading/trailing whitespace is stripped from a string input."""
        assert normalize_document_ids("  doc-1  ") == ["doc-1"]

    def test_list_of_strings_passed_through(self):
        """A list of distinct strings is returned as-is."""
        result = normalize_document_ids(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_list_items_are_stripped(self):
        """Whitespace is stripped from each list item."""
        result = normalize_document_ids(["  x ", " y  "])
        assert result == ["x", "y"]

    def test_deduplication_preserves_order(self):
        """Duplicate IDs are removed; first occurrence wins."""
        result = normalize_document_ids(["a", "b", "a", "c", "b"])
        assert result == ["a", "b", "c"]

    def test_whitespace_only_items_skipped(self):
        """Items that are whitespace-only after stripping are excluded."""
        result = normalize_document_ids(["a", "  ", "", "b"])
        assert result == ["a", "b"]

    def test_non_string_items_in_list_skipped(self):
        """Non-string items inside the list are silently ignored."""
        result = normalize_document_ids(["a", 123, None, "b"])
        assert result == ["a", "b"]

    def test_max_items_truncation(self):
        """Output is capped at max_items."""
        ids = [f"doc-{i}" for i in range(100)]
        result = normalize_document_ids(ids, max_items=5)
        assert len(result) == 5
        assert result == ["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"]

    def test_max_items_default_is_50(self):
        """Default max_items is 50."""
        ids = [f"doc-{i}" for i in range(60)]
        result = normalize_document_ids(ids)
        assert len(result) == 50

    def test_unsupported_type_returns_empty_list(self):
        """A dict or other unsupported type returns empty list."""
        assert normalize_document_ids({"id": "abc"}) == []
        assert normalize_document_ids(42) == []
        assert normalize_document_ids(True) == []

    def test_dedup_after_stripping(self):
        """Two items that are identical after stripping count as duplicates."""
        result = normalize_document_ids(["  x", "x  ", "x"])
        assert result == ["x"]


# ---------------------------------------------------------------------------
# DecisionRequest.__post_init__ validation
# ---------------------------------------------------------------------------


class TestDecisionRequestPostInit:
    """Tests for __post_init__ validation on DecisionRequest."""

    def test_empty_content_raises_value_error(self):
        """Empty string content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            DecisionRequest(content="")

    def test_whitespace_only_content_raises_value_error(self):
        """Whitespace-only content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            DecisionRequest(content="   \t\n  ")

    def test_valid_content_accepted(self):
        """Non-empty content is accepted without error."""
        req = DecisionRequest(content="Should we invest in AI?")
        assert req.content == "Should we invest in AI?"

    def test_auto_generates_response_channel_when_empty(self):
        """If no response_channels are given, one is auto-generated from source."""
        req = DecisionRequest(content="test", source=InputSource.SLACK)
        assert len(req.response_channels) == 1
        assert req.response_channels[0].platform == "slack"

    def test_existing_response_channels_preserved(self):
        """Explicitly provided response_channels are not overwritten."""
        channel = ResponseChannel(platform="discord", channel_id="ch-1")
        req = DecisionRequest(content="test", response_channels=[channel])
        assert len(req.response_channels) == 1
        assert req.response_channels[0].channel_id == "ch-1"

    def test_auto_detects_decision_type(self):
        """When decision_type is AUTO, it is replaced by auto-detection."""
        req = DecisionRequest(content="What is the best database?")
        # AUTO should have been replaced
        assert req.decision_type != DecisionType.AUTO

    def test_explicit_decision_type_preserved(self):
        """An explicitly set decision_type is not overridden."""
        req = DecisionRequest(content="test", decision_type=DecisionType.WORKFLOW)
        assert req.decision_type == DecisionType.WORKFLOW

    def test_documents_are_normalized(self):
        """Documents list is normalized via normalize_document_ids."""
        req = DecisionRequest(content="test", documents=["a", "  b ", "a"])
        assert req.documents == ["a", "b"]

    def test_request_id_auto_generated(self):
        """request_id is a valid UUID by default."""
        req = DecisionRequest(content="test")
        # Should not raise
        uuid.UUID(req.request_id)


# ---------------------------------------------------------------------------
# DecisionRequest._detect_decision_type
# ---------------------------------------------------------------------------


class TestDetectDecisionType:
    """Tests for the _detect_decision_type auto-detection heuristic."""

    def test_workflow_id_triggers_workflow_type(self):
        """If config has a workflow_id, the type is WORKFLOW."""
        config = DecisionConfig(workflow_id="wf-123")
        req = DecisionRequest(
            content="run the pipeline",
            decision_type=DecisionType.AUTO,
            config=config,
        )
        assert req.decision_type == DecisionType.WORKFLOW

    def test_gauntlet_keywords_trigger_gauntlet(self):
        """Gauntlet keywords in content trigger GAUNTLET type."""
        for keyword in ["validate", "stress test", "probe", "attack", "security"]:
            req = DecisionRequest(
                content=f"Please {keyword} this deployment",
                decision_type=DecisionType.AUTO,
            )
            assert req.decision_type == DecisionType.GAUNTLET, (
                f"keyword '{keyword}' should trigger GAUNTLET"
            )

    def test_quick_keywords_trigger_quick(self):
        """Quick keywords in content trigger QUICK type."""
        for keyword in ["quick", "fast", "simple", "brief"]:
            req = DecisionRequest(
                content=f"Give me a {keyword} answer",
                decision_type=DecisionType.AUTO,
            )
            assert req.decision_type == DecisionType.QUICK, (
                f"keyword '{keyword}' should trigger QUICK"
            )

    def test_default_is_debate(self):
        """Content without special keywords defaults to DEBATE."""
        req = DecisionRequest(
            content="What architecture should we use for the new service?",
            decision_type=DecisionType.AUTO,
        )
        assert req.decision_type == DecisionType.DEBATE

    def test_gauntlet_takes_priority_over_quick(self):
        """Gauntlet keywords are checked before quick keywords."""
        req = DecisionRequest(
            content="Quick security validate check",
            decision_type=DecisionType.AUTO,
        )
        # "validate" and "security" are gauntlet keywords;
        # gauntlet branch is checked first in the source code
        assert req.decision_type == DecisionType.GAUNTLET

    def test_workflow_takes_priority_over_gauntlet(self):
        """workflow_id takes priority over gauntlet keywords."""
        config = DecisionConfig(workflow_id="wf-1")
        req = DecisionRequest(
            content="validate the security pipeline",
            decision_type=DecisionType.AUTO,
            config=config,
        )
        assert req.decision_type == DecisionType.WORKFLOW

    def test_case_insensitive_keyword_matching(self):
        """Keyword detection is case-insensitive."""
        req = DecisionRequest(
            content="VALIDATE this input FAST",
            decision_type=DecisionType.AUTO,
        )
        # "validate" is gauntlet and checked first
        assert req.decision_type == DecisionType.GAUNTLET


# ---------------------------------------------------------------------------
# DecisionRequest.from_chat_message
# ---------------------------------------------------------------------------


class TestFromChatMessage:
    """Tests for DecisionRequest.from_chat_message factory."""

    def test_slack_platform_maps_to_slack_source(self):
        """Platform 'slack' maps to InputSource.SLACK."""
        req = DecisionRequest.from_chat_message(
            message="Should we rewrite in Rust?",
            platform="slack",
            channel_id="C12345",
        )
        assert req.source == InputSource.SLACK

    def test_discord_platform(self):
        """Platform 'discord' maps to InputSource.DISCORD."""
        req = DecisionRequest.from_chat_message(
            message="test", platform="discord", channel_id="ch-1"
        )
        assert req.source == InputSource.DISCORD

    def test_telegram_platform(self):
        """Platform 'telegram' maps to InputSource.TELEGRAM."""
        req = DecisionRequest.from_chat_message(
            message="test", platform="telegram", channel_id="ch-1"
        )
        assert req.source == InputSource.TELEGRAM

    def test_teams_platform(self):
        """Platform 'teams' maps to InputSource.TEAMS."""
        req = DecisionRequest.from_chat_message(message="test", platform="teams", channel_id="ch-1")
        assert req.source == InputSource.TEAMS

    def test_response_channel_set_from_params(self):
        """Response channel contains the provided channel_id, user_id, and thread_id."""
        req = DecisionRequest.from_chat_message(
            message="analyze this",
            platform="slack",
            channel_id="C99",
            user_id="U42",
            thread_id="t-abc",
        )
        assert len(req.response_channels) == 1
        rc = req.response_channels[0]
        assert rc.platform == "slack"
        assert rc.channel_id == "C99"
        assert rc.user_id == "U42"
        assert rc.thread_id == "t-abc"

    def test_context_includes_user_id(self):
        """Request context includes the user_id."""
        req = DecisionRequest.from_chat_message(
            message="test", platform="slack", channel_id="C1", user_id="U1"
        )
        assert req.context.user_id == "U1"

    def test_kwargs_stored_in_context_metadata(self):
        """Extra keyword arguments are stored in context.metadata."""
        req = DecisionRequest.from_chat_message(
            message="test",
            platform="slack",
            channel_id="C1",
            custom_field="hello",
        )
        assert req.context.metadata.get("custom_field") == "hello"

    def test_documents_kwarg_normalized(self):
        """documents kwarg is passed through normalize_document_ids."""
        req = DecisionRequest.from_chat_message(
            message="review docs",
            platform="slack",
            channel_id="C1",
            documents=["doc-1", "doc-1", "  doc-2  "],
        )
        assert req.documents == ["doc-1", "doc-2"]

    def test_document_ids_kwarg_also_accepted(self):
        """document_ids kwarg is an alternative to documents."""
        req = DecisionRequest.from_chat_message(
            message="review docs",
            platform="slack",
            channel_id="C1",
            document_ids=["d-a"],
        )
        assert req.documents == ["d-a"]

    def test_attachments_and_evidence_passed_through(self):
        """attachments and evidence kwargs are stored on the request."""
        req = DecisionRequest.from_chat_message(
            message="test",
            platform="slack",
            channel_id="C1",
            attachments=[{"url": "https://example.com/file.pdf"}],
            evidence=[{"type": "link", "url": "https://example.com"}],
        )
        assert len(req.attachments) == 1
        assert len(req.evidence) == 1

    def test_config_kwarg_used_when_provided(self):
        """An explicit config kwarg overrides the default DecisionConfig."""
        custom_config = DecisionConfig(timeout_seconds=60, rounds=5)
        req = DecisionRequest.from_chat_message(
            message="test",
            platform="slack",
            channel_id="C1",
            config=custom_config,
        )
        assert req.config.timeout_seconds == 60
        assert req.config.rounds == 5

    def test_decision_config_alias(self):
        """decision_config kwarg is also accepted (alias for config)."""
        custom_config = DecisionConfig(rounds=7)
        req = DecisionRequest.from_chat_message(
            message="test",
            platform="slack",
            channel_id="C1",
            decision_config=custom_config,
        )
        assert req.config.rounds == 7

    def test_decision_integrity_creates_config(self):
        """A decision_integrity dict creates a DecisionConfig with that field."""
        req = DecisionRequest.from_chat_message(
            message="test",
            platform="slack",
            channel_id="C1",
            decision_integrity={"include_receipt": True},
        )
        assert req.config.decision_integrity == {"include_receipt": True}

    def test_decision_integrity_bool_normalized(self):
        """A boolean decision_integrity is normalized to an empty dict."""
        req = DecisionRequest.from_chat_message(
            message="test",
            platform="slack",
            channel_id="C1",
            decision_integrity=True,
        )
        assert isinstance(req.config.decision_integrity, dict)

    def test_platform_is_case_insensitive(self):
        """Platform string is lowercased before mapping to InputSource."""
        req = DecisionRequest.from_chat_message(message="test", platform="SLACK", channel_id="C1")
        assert req.source == InputSource.SLACK


# ---------------------------------------------------------------------------
# DecisionRequest.from_http
# ---------------------------------------------------------------------------


class TestFromHttp:
    """Tests for DecisionRequest.from_http factory."""

    def test_unified_format_with_content_key(self):
        """Body with 'content' key uses from_dict (unified format)."""
        body = {
            "content": "Evaluate our CI pipeline",
            "decision_type": "debate",
            "source": "http_api",
        }
        req = DecisionRequest.from_http(body)
        assert req.content == "Evaluate our CI pipeline"
        assert req.decision_type == DecisionType.DEBATE

    def test_legacy_format_uses_question_key(self):
        """Legacy body without 'content' reads from 'question'."""
        body = {"question": "Is our API secure?"}
        req = DecisionRequest.from_http(body)
        assert req.content == "Is our API secure?"
        assert req.source == InputSource.HTTP_API

    def test_legacy_format_uses_task_key(self):
        """Legacy body falls back to 'task' if 'question' is absent."""
        body = {"task": "Review database schema"}
        req = DecisionRequest.from_http(body)
        assert req.content == "Review database schema"

    def test_legacy_format_uses_input_text_key(self):
        """Legacy body falls back to 'input_text' if 'question' and 'task' are absent."""
        body = {"input_text": "Analyze logs"}
        req = DecisionRequest.from_http(body)
        assert req.content == "Analyze logs"

    def test_legacy_format_maps_agents_and_rounds(self):
        """Legacy format extracts agents, rounds, consensus into DecisionConfig."""
        body = {
            "question": "test",
            "agents": ["agent-a", "agent-b"],
            "rounds": 5,
            "consensus": "unanimous",
            "timeout": 120,
        }
        req = DecisionRequest.from_http(body)
        assert req.config.agents == ["agent-a", "agent-b"]
        assert req.config.rounds == 5
        assert req.config.consensus == "unanimous"
        assert req.config.timeout_seconds == 120

    def test_legacy_format_default_decision_type_is_debate(self):
        """Legacy format defaults decision_type to DEBATE (not AUTO)."""
        body = {"question": "test"}
        req = DecisionRequest.from_http(body)
        assert req.decision_type == DecisionType.DEBATE

    def test_correlation_id_from_header(self):
        """X-Correlation-ID header sets context.correlation_id."""
        body = {"content": "test"}
        headers = {"X-Correlation-ID": "corr-abc-123"}
        req = DecisionRequest.from_http(body, headers=headers)
        assert req.context.correlation_id == "corr-abc-123"

    def test_request_id_header_fallback(self):
        """X-Request-ID header is used when X-Correlation-ID is absent."""
        body = {"content": "test"}
        headers = {"X-Request-ID": "req-xyz-789"}
        req = DecisionRequest.from_http(body, headers=headers)
        assert req.context.correlation_id == "req-xyz-789"

    def test_no_headers_uses_auto_generated_correlation_id(self):
        """Without headers, the correlation_id is auto-generated (a UUID)."""
        body = {"content": "test"}
        req = DecisionRequest.from_http(body)
        # Should be a valid UUID
        uuid.UUID(req.context.correlation_id)

    def test_legacy_format_with_documents(self):
        """Legacy format picks up documents from 'documents' key."""
        body = {"question": "test", "documents": ["doc-1", "doc-2"]}
        req = DecisionRequest.from_http(body)
        assert req.documents == ["doc-1", "doc-2"]

    def test_legacy_format_with_document_ids_alias(self):
        """Legacy format picks up documents from 'document_ids' key."""
        body = {"question": "test", "document_ids": ["d-a"]}
        req = DecisionRequest.from_http(body)
        assert req.documents == ["d-a"]

    def test_legacy_format_with_decision_integrity(self):
        """Legacy format passes decision_integrity into DecisionConfig."""
        body = {
            "question": "test",
            "decision_integrity": {"include_receipt": True, "include_plan": True},
        }
        req = DecisionRequest.from_http(body)
        assert req.config.decision_integrity["include_receipt"] is True
        assert req.config.decision_integrity["include_plan"] is True

    def test_legacy_format_with_attachments_and_evidence(self):
        """Legacy format passes attachments and evidence through."""
        body = {
            "question": "test",
            "attachments": [{"name": "file.pdf"}],
            "evidence": [{"source": "web"}],
        }
        req = DecisionRequest.from_http(body)
        assert len(req.attachments) == 1
        assert len(req.evidence) == 1

    def test_unified_format_with_full_body(self):
        """Unified format with all fields round-trips correctly."""
        body = {
            "content": "Complex analysis request",
            "decision_type": "gauntlet",
            "source": "slack",
            "priority": "high",
            "documents": ["d-1"],
            "config": {"rounds": 3, "consensus": "majority"},
            "context": {"user_id": "u-1", "tags": ["important"]},
        }
        req = DecisionRequest.from_http(body)
        assert req.content == "Complex analysis request"
        assert req.decision_type == DecisionType.GAUNTLET
        assert req.source == InputSource.SLACK
        assert req.priority == Priority.HIGH
        assert req.documents == ["d-1"]
        assert req.config.rounds == 3
        assert req.context.user_id == "u-1"


# ---------------------------------------------------------------------------
# DecisionRequest.from_voice
# ---------------------------------------------------------------------------


class TestFromVoice:
    """Tests for DecisionRequest.from_voice factory."""

    def test_slack_platform_maps_to_voice_slack(self):
        """Platform 'slack' maps to InputSource.VOICE_SLACK."""
        req = DecisionRequest.from_voice(
            transcription="What should we do?",
            platform="slack",
            channel_id="C1",
        )
        assert req.source == InputSource.VOICE_SLACK

    def test_telegram_platform_maps_to_voice_telegram(self):
        """Platform 'telegram' maps to InputSource.VOICE_TELEGRAM."""
        req = DecisionRequest.from_voice(transcription="test", platform="telegram", channel_id="C1")
        assert req.source == InputSource.VOICE_TELEGRAM

    def test_whatsapp_platform_maps_to_voice_whatsapp(self):
        """Platform 'whatsapp' maps to InputSource.VOICE_WHATSAPP."""
        req = DecisionRequest.from_voice(transcription="test", platform="whatsapp", channel_id="C1")
        assert req.source == InputSource.VOICE_WHATSAPP

    def test_unknown_platform_defaults_to_voice(self):
        """An unrecognized platform falls back to InputSource.VOICE."""
        req = DecisionRequest.from_voice(
            transcription="test", platform="custom_voip", channel_id="C1"
        )
        assert req.source == InputSource.VOICE

    def test_voice_response_enabled_by_default(self):
        """Voice response is enabled by default."""
        req = DecisionRequest.from_voice(transcription="test", platform="slack", channel_id="C1")
        rc = req.response_channels[0]
        assert rc.voice_enabled is True
        assert rc.response_format == "voice_with_text"

    def test_voice_response_disabled(self):
        """When voice_response=False, voice is disabled and format is 'full'."""
        req = DecisionRequest.from_voice(
            transcription="test",
            platform="slack",
            channel_id="C1",
            voice_response=False,
        )
        rc = req.response_channels[0]
        assert rc.voice_enabled is False
        assert rc.response_format == "full"

    def test_custom_voice_id(self):
        """Custom voice_id is set on the response channel."""
        req = DecisionRequest.from_voice(
            transcription="test",
            platform="slack",
            channel_id="C1",
            voice_id="custom-voice",
        )
        assert req.response_channels[0].voice_id == "custom-voice"

    def test_default_voice_id_is_narrator(self):
        """Default voice_id is 'narrator'."""
        req = DecisionRequest.from_voice(transcription="test", platform="slack", channel_id="C1")
        assert req.response_channels[0].voice_id == "narrator"

    def test_audio_duration_in_context_metadata(self):
        """audio_duration is stored in context metadata."""
        req = DecisionRequest.from_voice(
            transcription="test",
            platform="slack",
            channel_id="C1",
            audio_duration=12.5,
        )
        assert req.context.metadata["audio_duration"] == 12.5

    def test_transcription_source_in_metadata(self):
        """Context metadata always includes transcription_source: whisper."""
        req = DecisionRequest.from_voice(transcription="test", platform="slack", channel_id="C1")
        assert req.context.metadata["transcription_source"] == "whisper"

    def test_extra_kwargs_in_metadata(self):
        """Extra kwargs are merged into context metadata."""
        req = DecisionRequest.from_voice(
            transcription="test",
            platform="slack",
            channel_id="C1",
            language="en",
            model="whisper-large-v3",
        )
        assert req.context.metadata["language"] == "en"
        assert req.context.metadata["model"] == "whisper-large-v3"

    def test_platform_case_insensitive(self):
        """Platform matching is case-insensitive."""
        req = DecisionRequest.from_voice(transcription="test", platform="SLACK", channel_id="C1")
        assert req.source == InputSource.VOICE_SLACK

    def test_content_is_transcription(self):
        """The transcription text becomes the request content."""
        req = DecisionRequest.from_voice(
            transcription="Should we migrate to kubernetes?",
            platform="slack",
            channel_id="C1",
        )
        assert req.content == "Should we migrate to kubernetes?"


# ---------------------------------------------------------------------------
# DecisionRequest.from_document
# ---------------------------------------------------------------------------


class TestFromDocument:
    """Tests for DecisionRequest.from_document factory."""

    def test_google_drive_source(self):
        """'google_drive' platform maps to InputSource.GOOGLE_DRIVE."""
        req = DecisionRequest.from_document(
            content="Review this proposal",
            source_platform="google_drive",
            document_id="gdoc-123",
        )
        assert req.source == InputSource.GOOGLE_DRIVE

    def test_gdrive_alias(self):
        """'gdrive' alias also maps to InputSource.GOOGLE_DRIVE."""
        req = DecisionRequest.from_document(
            content="test", source_platform="gdrive", document_id="d-1"
        )
        assert req.source == InputSource.GOOGLE_DRIVE

    def test_onedrive_source(self):
        """'onedrive' maps to InputSource.ONEDRIVE."""
        req = DecisionRequest.from_document(
            content="test", source_platform="onedrive", document_id="d-1"
        )
        assert req.source == InputSource.ONEDRIVE

    def test_sharepoint_source(self):
        """'sharepoint' maps to InputSource.SHAREPOINT."""
        req = DecisionRequest.from_document(
            content="test", source_platform="sharepoint", document_id="d-1"
        )
        assert req.source == InputSource.SHAREPOINT

    def test_dropbox_source(self):
        """'dropbox' maps to InputSource.DROPBOX."""
        req = DecisionRequest.from_document(
            content="test", source_platform="dropbox", document_id="d-1"
        )
        assert req.source == InputSource.DROPBOX

    def test_s3_source(self):
        """'s3' maps to InputSource.S3."""
        req = DecisionRequest.from_document(content="test", source_platform="s3", document_id="d-1")
        assert req.source == InputSource.S3

    def test_confluence_source(self):
        """'confluence' maps to InputSource.CONFLUENCE."""
        req = DecisionRequest.from_document(
            content="test", source_platform="confluence", document_id="d-1"
        )
        assert req.source == InputSource.CONFLUENCE

    def test_notion_source(self):
        """'notion' maps to InputSource.NOTION."""
        req = DecisionRequest.from_document(
            content="test", source_platform="notion", document_id="d-1"
        )
        assert req.source == InputSource.NOTION

    def test_unknown_platform_defaults_to_internal(self):
        """An unrecognized platform defaults to InputSource.INTERNAL."""
        req = DecisionRequest.from_document(
            content="test", source_platform="unknown_store", document_id="d-1"
        )
        assert req.source == InputSource.INTERNAL

    def test_document_metadata_in_context(self):
        """Document ID, title, URL, and platform are stored in context metadata."""
        req = DecisionRequest.from_document(
            content="test",
            source_platform="google_drive",
            document_id="gdoc-456",
            document_title="Q4 Report",
            document_url="https://docs.google.com/doc/gdoc-456",
        )
        meta = req.context.metadata
        assert meta["document_id"] == "gdoc-456"
        assert meta["document_title"] == "Q4 Report"
        assert meta["document_url"] == "https://docs.google.com/doc/gdoc-456"
        assert meta["source_platform"] == "google_drive"

    def test_user_id_in_context(self):
        """user_id parameter is set on the request context."""
        req = DecisionRequest.from_document(
            content="test",
            source_platform="s3",
            document_id="d-1",
            user_id="u-42",
        )
        assert req.context.user_id == "u-42"

    def test_response_channel_platform_from_source(self):
        """Response channel uses the source_platform."""
        req = DecisionRequest.from_document(
            content="test",
            source_platform="confluence",
            document_id="d-1",
        )
        assert req.response_channels[0].platform == "confluence"

    def test_webhook_url_in_response_channel(self):
        """webhook_url kwarg is set on the response channel."""
        req = DecisionRequest.from_document(
            content="test",
            source_platform="s3",
            document_id="d-1",
            webhook_url="https://example.com/callback",
        )
        assert req.response_channels[0].webhook_url == "https://example.com/callback"

    def test_extra_kwargs_in_context_metadata(self):
        """Extra kwargs are merged into context metadata."""
        req = DecisionRequest.from_document(
            content="test",
            source_platform="s3",
            document_id="d-1",
            mime_type="application/pdf",
            file_size=1024,
        )
        assert req.context.metadata["mime_type"] == "application/pdf"
        assert req.context.metadata["file_size"] == 1024

    def test_platform_case_insensitive(self):
        """Platform matching is case-insensitive."""
        req = DecisionRequest.from_document(
            content="test", source_platform="NOTION", document_id="d-1"
        )
        assert req.source == InputSource.NOTION

    def test_response_format_is_full(self):
        """Response channel uses 'full' format for document sources."""
        req = DecisionRequest.from_document(content="test", source_platform="s3", document_id="d-1")
        assert req.response_channels[0].response_format == "full"


# ---------------------------------------------------------------------------
# ResponseChannel round-trip
# ---------------------------------------------------------------------------


class TestResponseChannelRoundTrip:
    """Tests for ResponseChannel to_dict / from_dict serialization."""

    def test_round_trip(self):
        """to_dict followed by from_dict produces an equivalent object."""
        original = ResponseChannel(
            platform="slack",
            channel_id="C1",
            user_id="U1",
            thread_id="t-1",
            webhook_url="https://hook.example.com",
            email_address="a@b.com",
            response_format="summary",
            include_reasoning=False,
            voice_enabled=True,
            voice_id="alloy",
            voice_only=True,
        )
        restored = ResponseChannel.from_dict(original.to_dict())
        assert restored.platform == original.platform
        assert restored.channel_id == original.channel_id
        assert restored.user_id == original.user_id
        assert restored.thread_id == original.thread_id
        assert restored.webhook_url == original.webhook_url
        assert restored.email_address == original.email_address
        assert restored.response_format == original.response_format
        assert restored.include_reasoning == original.include_reasoning
        assert restored.voice_enabled == original.voice_enabled
        assert restored.voice_id == original.voice_id
        assert restored.voice_only == original.voice_only

    def test_from_dict_defaults(self):
        """from_dict with empty dict uses sensible defaults."""
        rc = ResponseChannel.from_dict({})
        assert rc.platform == "http"
        assert rc.response_format == "full"
        assert rc.include_reasoning is True
        assert rc.voice_enabled is False
        assert rc.voice_id == "narrator"
        assert rc.voice_only is False


# ---------------------------------------------------------------------------
# RequestContext round-trip
# ---------------------------------------------------------------------------


class TestRequestContextRoundTrip:
    """Tests for RequestContext to_dict / from_dict serialization."""

    def test_round_trip(self):
        """to_dict followed by from_dict preserves key fields."""
        original = RequestContext(
            user_id="u-1",
            user_name="Alice",
            tenant_id="t-1",
            tags=["urgent", "security"],
            metadata={"source": "test"},
        )
        data = original.to_dict()
        restored = RequestContext.from_dict(data)
        assert restored.user_id == "u-1"
        assert restored.user_name == "Alice"
        assert restored.tenant_id == "t-1"
        assert restored.tags == ["urgent", "security"]
        assert restored.metadata == {"source": "test"}

    def test_from_dict_parses_iso_dates(self):
        """ISO date strings in created_at and deadline are parsed."""
        data = {
            "created_at": "2026-01-15T10:30:00+00:00",
            "deadline": "2026-01-16T10:30:00+00:00",
        }
        ctx = RequestContext.from_dict(data)
        assert isinstance(ctx.created_at, datetime)
        assert isinstance(ctx.deadline, datetime)


# ---------------------------------------------------------------------------
# DecisionConfig round-trip and from_dict edge cases
# ---------------------------------------------------------------------------


class TestDecisionConfigFromDict:
    """Tests for DecisionConfig.from_dict edge cases."""

    def test_decision_integrity_dict_passthrough(self):
        """A dict decision_integrity is passed through."""
        cfg = DecisionConfig.from_dict({"decision_integrity": {"include_receipt": True}})
        assert cfg.decision_integrity == {"include_receipt": True}

    def test_decision_integrity_bool_normalized(self):
        """A bool decision_integrity is normalized to empty dict."""
        cfg = DecisionConfig.from_dict({"decision_integrity": True})
        assert cfg.decision_integrity == {}

    def test_decision_integrity_none_normalized(self):
        """None decision_integrity is normalized to empty dict."""
        cfg = DecisionConfig.from_dict({"decision_integrity": None})
        assert cfg.decision_integrity == {}

    def test_decision_integrity_non_dict_non_bool(self):
        """A non-dict, non-bool value is normalized to empty dict."""
        cfg = DecisionConfig.from_dict({"decision_integrity": "yes"})
        assert cfg.decision_integrity == {}


# ---------------------------------------------------------------------------
# DecisionRequest.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestDecisionRequestRoundTrip:
    """Tests for DecisionRequest serialization round-trip."""

    def test_to_dict_contains_required_keys(self):
        """to_dict output contains all expected top-level keys."""
        req = DecisionRequest(content="test question")
        data = req.to_dict()
        expected_keys = {
            "request_id",
            "content",
            "decision_type",
            "source",
            "response_channels",
            "context",
            "config",
            "priority",
            "attachments",
            "evidence",
            "documents",
        }
        assert expected_keys.issubset(set(data.keys()))

    def test_from_dict_with_document_ids_alias(self):
        """from_dict accepts document_ids as an alias for documents."""
        data = {
            "content": "test",
            "document_ids": ["id-a", "id-b"],
        }
        req = DecisionRequest.from_dict(data)
        assert req.documents == ["id-a", "id-b"]

    def test_round_trip_preserves_content(self):
        """Content survives a to_dict / from_dict round-trip."""
        original = DecisionRequest(
            content="What architecture should we use?",
            decision_type=DecisionType.DEBATE,
            priority=Priority.HIGH,
        )
        data = original.to_dict()
        restored = DecisionRequest.from_dict(data)
        assert restored.content == original.content
        assert restored.decision_type == original.decision_type
        assert restored.priority == original.priority
        assert restored.request_id == original.request_id
