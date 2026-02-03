"""Tests for integrations/langchain/callbacks.py — LangChain callbacks."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aragora.integrations.langchain.callbacks import (
    AragoraCallbackHandler,
    _StubAgentAction,
    _StubAgentFinish,
    _StubLLMResult,
)


@pytest.fixture()
def handler():
    return AragoraCallbackHandler(
        aragora_url="http://localhost:8080",
        api_token="tok-123",
        debate_id="d-1",
        session_id="s-1",
    )


# =============================================================================
# Initialization
# =============================================================================


class TestInit:
    def test_defaults(self):
        h = AragoraCallbackHandler()
        assert h.aragora_url == "http://localhost:8080"
        assert h.api_token is None
        assert h.debate_id is None
        assert h.session_id is None
        assert h._events == []

    def test_custom_config(self, handler):
        assert handler.aragora_url == "http://localhost:8080"
        assert handler.api_token == "tok-123"
        assert handler.debate_id == "d-1"
        assert handler.session_id == "s-1"


# =============================================================================
# LLM callbacks
# =============================================================================


class TestLLMCallbacks:
    def test_on_llm_start(self, handler):
        handler.on_llm_start(
            serialized={"name": "gpt-4"},
            prompts=["Hello", "World"],
            run_id=uuid4(),
        )
        assert len(handler._events) == 1
        event = handler._events[0]
        assert event["type"] == "llm_start"
        assert event["data"]["model"] == "gpt-4"
        assert event["data"]["prompt_count"] == 2

    def test_on_llm_end_with_usage(self, handler):
        result = _StubLLMResult()
        result.llm_output = {"token_usage": {"total_tokens": 100}}
        handler.on_llm_end(response=result, run_id=uuid4())
        event = handler._events[0]
        assert event["type"] == "llm_end"
        assert event["data"]["token_usage"]["total_tokens"] == 100

    def test_on_llm_end_no_usage(self, handler):
        result = _StubLLMResult()
        result.llm_output = None
        handler.on_llm_end(response=result, run_id=uuid4())
        event = handler._events[0]
        assert event["data"]["token_usage"] == {}

    def test_on_llm_error(self, handler):
        handler.on_llm_error(error=ValueError("bad input"), run_id=uuid4())
        event = handler._events[0]
        assert event["type"] == "llm_error"
        assert "bad input" in event["data"]["error"]


# =============================================================================
# Chain callbacks
# =============================================================================


class TestChainCallbacks:
    def test_on_chain_start(self, handler):
        handler.on_chain_start(
            serialized={"name": "DebateChain"},
            inputs={"question": "test"},
            run_id=uuid4(),
        )
        event = handler._events[0]
        assert event["type"] == "chain_start"
        assert event["data"]["chain_type"] == "DebateChain"

    def test_on_chain_end(self, handler):
        handler.on_chain_end(
            outputs={"answer": "yes", "confidence": 0.9},
            run_id=uuid4(),
        )
        event = handler._events[0]
        assert event["type"] == "chain_end"
        assert "answer" in event["data"]["output_keys"]

    def test_on_chain_end_empty(self, handler):
        handler.on_chain_end(outputs={}, run_id=uuid4())
        event = handler._events[0]
        assert event["data"]["output_keys"] == []


# =============================================================================
# Tool callbacks
# =============================================================================


class TestToolCallbacks:
    def test_on_tool_start(self, handler):
        handler.on_tool_start(
            serialized={"name": "aragora_debate"},
            input_str="What is best?",
            run_id=uuid4(),
        )
        event = handler._events[0]
        assert event["type"] == "tool_start"
        assert event["data"]["tool"] == "aragora_debate"

    def test_on_tool_end(self, handler):
        handler.on_tool_end(output="Result here", run_id=uuid4())
        event = handler._events[0]
        assert event["type"] == "tool_end"
        assert event["data"]["output_length"] == 11

    def test_on_tool_end_empty(self, handler):
        handler.on_tool_end(output="", run_id=uuid4())
        event = handler._events[0]
        assert event["data"]["output_length"] == 0


# =============================================================================
# Agent callbacks
# =============================================================================


class TestAgentCallbacks:
    def test_on_agent_action(self, handler):
        action = _StubAgentAction()
        action.tool = "aragora_knowledge"
        handler.on_agent_action(action=action, run_id=uuid4())
        event = handler._events[0]
        assert event["type"] == "agent_action"
        assert event["data"]["tool"] == "aragora_knowledge"

    def test_on_agent_finish(self, handler):
        finish = _StubAgentFinish()
        handler.on_agent_finish(finish=finish, run_id=uuid4())
        event = handler._events[0]
        assert event["type"] == "agent_finish"


# =============================================================================
# Event management
# =============================================================================


class TestEventManagement:
    def test_get_events_returns_copy(self, handler):
        handler.on_llm_start(serialized={"name": "m"}, prompts=["p"], run_id=uuid4())
        events = handler.get_events()
        assert len(events) == 1
        events.clear()
        assert len(handler._events) == 1  # original unchanged

    def test_clear_events(self, handler):
        handler.on_llm_start(serialized={"name": "m"}, prompts=["p"], run_id=uuid4())
        handler.clear_events()
        assert handler._events == []

    def test_event_includes_debate_and_session(self, handler):
        handler.on_llm_start(serialized={"name": "m"}, prompts=["p"], run_id=uuid4())
        event = handler._events[0]
        assert event["debate_id"] == "d-1"
        assert event["session_id"] == "s-1"

    def test_multiple_events_accumulated(self, handler):
        handler.on_llm_start(serialized={}, prompts=["a"], run_id=uuid4())
        handler.on_llm_end(response=_StubLLMResult(), run_id=uuid4())
        handler.on_chain_start(serialized={}, inputs={}, run_id=uuid4())
        assert len(handler._events) == 3
