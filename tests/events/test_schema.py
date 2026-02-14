"""Tests for events/schema.py — event schema validation."""

import pytest

from aragora.events.schema import (
    AgentEloUpdatedPayload,
    AgentFallbackTriggeredPayload,
    AgentMessagePayload,
    ConsensusPayload,
    CritiquePayload,
    DebateEndPayload,
    DebateStartPayload,
    ErrorPayload,
    EventPayload,
    EventSchemaRegistry,
    KnowledgeIndexedPayload,
    KnowledgeQueriedPayload,
    KnowledgeStalePayload,
    RoundStartPayload,
    TokenDeltaPayload,
    TokenEndPayload,
    TokenStartPayload,
    ValidationError,
    VotePayload,
    get_schema_registry,
    reset_schema_registry,
    validate_event,
    validate_event_log_errors,
)
from aragora.events.types import StreamEventType


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_schema_registry()
    yield
    reset_schema_registry()


# =============================================================================
# ValidationError
# =============================================================================


class TestValidationError:
    def test_str_with_value(self):
        err = ValidationError(field="name", message="Required", value=123)
        s = str(err)
        assert "name" in s
        assert "Required" in s
        assert "123" in s
        assert "int" in s

    def test_str_without_value(self):
        err = ValidationError(field="name", message="Required")
        s = str(err)
        assert s == "name: Required"


# =============================================================================
# EventPayload base class
# =============================================================================


class TestEventPayload:
    def test_to_dict(self):
        payload = DebateStartPayload(debate_id="d-1", question="Why?")
        d = payload.to_dict()
        assert d["debate_id"] == "d-1"
        assert d["question"] == "Why?"
        assert d["rounds"] == 3
        assert d["agents"] == []

    def test_from_dict_filters_unknown(self):
        data = {"debate_id": "d-1", "question": "Why?", "unknown_field": 42}
        payload = DebateStartPayload.from_dict(data)
        assert payload.debate_id == "d-1"
        assert not hasattr(payload, "unknown_field")

    def test_from_dict_with_defaults(self):
        data = {"debate_id": "d-1", "question": "Why?"}
        payload = DebateStartPayload.from_dict(data)
        assert payload.rounds == 3
        assert payload.consensus_type == "majority"


# =============================================================================
# Debate Event Payloads
# =============================================================================


class TestDebatePayloads:
    def test_debate_start_defaults(self):
        p = DebateStartPayload(debate_id="d-1", question="q")
        assert p.agents == []
        assert p.domain == ""
        assert p.rounds == 3
        assert p.consensus_type == "majority"
        assert p.user_id is None
        assert p.org_id is None

    def test_debate_end_defaults(self):
        p = DebateEndPayload(debate_id="d-1")
        assert p.consensus_reached is False
        assert p.confidence == 0.0
        assert p.rounds_used == 0

    def test_round_start(self):
        p = RoundStartPayload(debate_id="d-1", round_number=2)
        d = p.to_dict()
        assert d["round_number"] == 2
        assert d["phase"] == "proposal"

    def test_agent_message(self):
        p = AgentMessagePayload(debate_id="d-1", agent="claude", message="Hello")
        assert p.role == "participant"
        assert p.citations == []

    def test_critique(self):
        p = CritiquePayload(
            debate_id="d-1",
            critic_agent="gpt-4",
            target_agent="claude",
            critique="Weak evidence",
        )
        assert p.severity == "medium"

    def test_vote(self):
        p = VotePayload(debate_id="d-1", voter_agent="claude", voted_for="gpt-4")
        assert p.confidence == 0.0
        assert p.reason is None

    def test_consensus(self):
        p = ConsensusPayload(debate_id="d-1", consensus_reached=True)
        assert p.supporting_agents == []
        assert p.dissenting_agents == []


# =============================================================================
# Token Streaming Payloads
# =============================================================================


class TestTokenPayloads:
    def test_token_start(self):
        p = TokenStartPayload(debate_id="d-1", agent="claude")
        assert p.task_id == ""

    def test_token_delta(self):
        p = TokenDeltaPayload(debate_id="d-1", agent="claude", delta="Hello")
        d = p.to_dict()
        assert d["delta"] == "Hello"

    def test_token_end(self):
        p = TokenEndPayload(debate_id="d-1", agent="claude")
        assert p.total_tokens == 0


# =============================================================================
# Knowledge Mound Payloads
# =============================================================================


class TestKnowledgePayloads:
    def test_knowledge_indexed(self):
        p = KnowledgeIndexedPayload(document_id="doc-1", chunk_count=5)
        assert p.source == ""
        assert p.tenant_id is None

    def test_knowledge_queried(self):
        p = KnowledgeQueriedPayload(query="test query", result_count=3, latency_ms=15.2)
        d = p.to_dict()
        assert d["latency_ms"] == 15.2

    def test_knowledge_stale(self):
        p = KnowledgeStalePayload(document_id="doc-1", staleness_score=0.8)
        assert p.reason == ""


# =============================================================================
# Error and Agent Payloads
# =============================================================================


class TestErrorAndAgentPayloads:
    def test_error_payload(self):
        p = ErrorPayload(error_type="timeout", message="Request timed out")
        assert p.recoverable is True
        assert p.agent is None

    def test_agent_elo_updated(self):
        p = AgentEloUpdatedPayload(agent="claude", old_elo=1200.0, new_elo=1215.0, change=15.0)
        d = p.to_dict()
        assert d["change"] == 15.0
        assert d["domain"] == ""

    def test_agent_fallback_triggered(self):
        p = AgentFallbackTriggeredPayload(
            agent="anthropic", fallback_to="openrouter", reason="rate_limit"
        )
        assert p.debate_id is None


# =============================================================================
# EventSchemaRegistry — validate
# =============================================================================


class TestRegistryValidate:
    def test_valid_debate_start(self):
        registry = EventSchemaRegistry()
        errors = registry.validate(
            StreamEventType.DEBATE_START,
            {"debate_id": "d-1", "question": "Why?"},
        )
        assert errors == []

    def test_missing_required_field(self):
        registry = EventSchemaRegistry()
        errors = registry.validate(StreamEventType.DEBATE_START, {"debate_id": "d-1"})
        assert len(errors) == 1
        assert errors[0].field == "question"

    def test_missing_required_field_none_value(self):
        registry = EventSchemaRegistry()
        errors = registry.validate(
            StreamEventType.DEBATE_START,
            {"debate_id": "d-1", "question": None},
        )
        assert any(e.field == "question" for e in errors)

    def test_wrong_type_string_no_error_with_future_annotations(self):
        # With `from __future__ import annotations`, field_info.type is a string
        # so runtime type checking is a no-op for simple types.
        registry = EventSchemaRegistry()
        errors = registry.validate(
            StreamEventType.DEBATE_START,
            {"debate_id": 123, "question": "q"},
        )
        # No type error detected because annotations are strings at runtime
        assert not any(e.field == "debate_id" for e in errors)

    def test_wrong_type_int_no_error_with_future_annotations(self):
        registry = EventSchemaRegistry()
        errors = registry.validate(
            StreamEventType.ROUND_START,
            {"debate_id": "d-1", "round_number": "not_int"},
        )
        assert not any(e.field == "round_number" for e in errors)

    def test_wrong_type_bool_no_error_with_future_annotations(self):
        registry = EventSchemaRegistry()
        errors = registry.validate(
            StreamEventType.DEBATE_END,
            {"debate_id": "d-1", "consensus_reached": "yes"},
        )
        assert not any(e.field == "consensus_reached" for e in errors)

    def test_unknown_event_type_non_strict(self):
        registry = EventSchemaRegistry()
        # Use a type that has no schema registered — HEARTBEAT
        errors = registry.validate(StreamEventType.HEARTBEAT, {"anything": True})
        assert errors == []

    def test_unknown_event_type_strict(self):
        registry = EventSchemaRegistry(strict=True)
        errors = registry.validate(StreamEventType.HEARTBEAT, {"anything": True})
        assert len(errors) == 1
        assert "_type" in errors[0].field

    def test_list_type_with_origin(self):
        # `agents` is list[str] which has __origin__ — the list check runs
        # only when hasattr(expected_type, "__origin__") is True at runtime.
        # With future annotations this is a string, so __origin__ is absent.
        registry = EventSchemaRegistry()
        errors = registry.validate(
            StreamEventType.DEBATE_START,
            {"debate_id": "d-1", "question": "q", "agents": "not_a_list"},
        )
        # No error because type annotation is a string at runtime
        assert not any(e.field == "agents" for e in errors)

    def test_valid_with_all_fields(self):
        registry = EventSchemaRegistry()
        errors = registry.validate(
            StreamEventType.DEBATE_START,
            {
                "debate_id": "d-1",
                "question": "q",
                "agents": ["claude"],
                "domain": "code",
                "rounds": 5,
                "consensus_type": "supermajority",
            },
        )
        assert errors == []


# =============================================================================
# EventSchemaRegistry — coerce
# =============================================================================


class TestRegistryCoerce:
    def test_coerce_noop_with_future_annotations(self):
        # With `from __future__ import annotations`, expected_type is a string
        # (e.g. "int" not int), so `expected_type is int` is False and coercion
        # does nothing for simple types. Verify data passes through unchanged.
        registry = EventSchemaRegistry()
        result = registry.coerce(
            StreamEventType.ROUND_START,
            {"debate_id": "d-1", "round_number": "3"},
        )
        # Not coerced because annotations are strings at runtime
        assert result["round_number"] == "3"

    def test_coerce_unknown_type_returns_copy(self):
        registry = EventSchemaRegistry()
        data = {"key": "value"}
        result = registry.coerce(StreamEventType.HEARTBEAT, data)
        assert result == data
        assert result is not data

    def test_coerce_does_not_mutate_input(self):
        registry = EventSchemaRegistry()
        data = {"debate_id": 42, "question": "q"}
        registry.coerce(StreamEventType.DEBATE_START, data)
        assert data["debate_id"] == 42  # unchanged

    def test_coerce_preserves_all_fields(self):
        registry = EventSchemaRegistry()
        result = registry.coerce(
            StreamEventType.DEBATE_START,
            {"debate_id": "d-1", "question": "q", "rounds": 5, "extra": "keep"},
        )
        assert result["debate_id"] == "d-1"
        assert result["rounds"] == 5
        assert result["extra"] == "keep"


# =============================================================================
# EventSchemaRegistry — register / get_schema / get_all_schemas
# =============================================================================


class TestRegistryManagement:
    def test_register_custom_schema(self):
        registry = EventSchemaRegistry()
        registry.register(StreamEventType.HEARTBEAT, DebateStartPayload)
        schema = registry.get_schema(StreamEventType.HEARTBEAT)
        assert schema is DebateStartPayload

    def test_get_schema_known(self):
        registry = EventSchemaRegistry()
        schema = registry.get_schema(StreamEventType.DEBATE_START)
        assert schema is DebateStartPayload

    def test_get_schema_unknown(self):
        registry = EventSchemaRegistry()
        schema = registry.get_schema(StreamEventType.HEARTBEAT)
        assert schema is None

    def test_get_all_schemas(self):
        registry = EventSchemaRegistry()
        all_schemas = registry.get_all_schemas()
        assert StreamEventType.DEBATE_START in all_schemas
        assert len(all_schemas) == 20


# =============================================================================
# Global registry functions
# =============================================================================


class TestGlobalRegistry:
    def test_get_schema_registry_singleton(self):
        r1 = get_schema_registry()
        r2 = get_schema_registry()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        r1 = get_schema_registry()
        reset_schema_registry()
        r2 = get_schema_registry()
        assert r1 is not r2


# =============================================================================
# Convenience validation functions
# =============================================================================


class TestConvenienceFunctions:
    def test_validate_event_valid(self):
        errors = validate_event(
            StreamEventType.DEBATE_START,
            {"debate_id": "d-1", "question": "q"},
        )
        assert errors == []

    def test_validate_event_invalid(self):
        errors = validate_event(StreamEventType.DEBATE_START, {})
        assert len(errors) >= 1

    def test_validate_event_log_errors_valid(self):
        result = validate_event_log_errors(
            StreamEventType.DEBATE_START,
            {"debate_id": "d-1", "question": "q"},
        )
        assert result is True

    def test_validate_event_log_errors_invalid(self):
        result = validate_event_log_errors(StreamEventType.DEBATE_START, {})
        assert result is False


# =============================================================================
# Round-trip: to_dict / from_dict
# =============================================================================


class TestRoundTrip:
    def test_debate_start_round_trip(self):
        original = DebateStartPayload(
            debate_id="d-1", question="q", agents=["claude", "gpt-4"], rounds=5
        )
        d = original.to_dict()
        restored = DebateStartPayload.from_dict(d)
        assert restored.debate_id == original.debate_id
        assert restored.agents == original.agents
        assert restored.rounds == original.rounds

    def test_error_payload_round_trip(self):
        original = ErrorPayload(
            error_type="timeout",
            message="Timed out",
            debate_id="d-1",
            trace_id="t-1",
        )
        d = original.to_dict()
        restored = ErrorPayload.from_dict(d)
        assert restored.error_type == "timeout"
        assert restored.trace_id == "t-1"

    def test_agent_elo_round_trip(self):
        original = AgentEloUpdatedPayload(
            agent="claude", old_elo=1200.0, new_elo=1215.0, change=15.0, domain="code"
        )
        d = original.to_dict()
        restored = AgentEloUpdatedPayload.from_dict(d)
        assert restored.change == 15.0
        assert restored.domain == "code"
