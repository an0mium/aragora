"""Protocol conformance and Result[T] tests.

Tests that:
1. Result[T] -- ok(), fail(), value access, error access, boolean truthiness
2. For each @runtime_checkable protocol:
   - A minimal conforming class passes isinstance()
   - A minimal non-conforming class (missing a method) fails isinstance()
3. Protocol method signatures match expected patterns

These are pure stdlib tests -- zero mocking, zero external dependencies.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest

from aragora.protocols.callback_types import Result
from aragora.protocols.agent_protocols import (
    AgentProtocol,
    StreamingAgentProtocol,
    ToolUsingAgentProtocol,
)
from aragora.protocols.debate_protocols import (
    ConsensusDetectorProtocol,
    ConsensusMemoryProtocol,
    DebateEmbeddingsProtocol,
    DebateResultProtocol,
    FlipDetectorProtocol,
    RankingSystemProtocol,
)
from aragora.protocols.event_protocols import (
    AsyncEventEmitterProtocol,
    BaseHandlerProtocol,
    EventEmitterProtocol,
    HandlerProtocol,
)
from aragora.protocols.feature_protocols import (
    BroadcastPipelineProtocol,
    EvidenceCollectorProtocol,
    EvidenceProtocol,
    InsightStoreProtocol,
    PopulationManagerProtocol,
    PromptEvolverProtocol,
    PulseManagerProtocol,
    StreamEventProtocol,
    VerificationBackendProtocol,
    WebhookConfigProtocol,
)
from aragora.protocols.memory_protocols import (
    ContinuumMemoryProtocol,
    CritiqueStoreProtocol,
    MemoryProtocol,
    TieredMemoryProtocol,
)
from aragora.protocols.storage_protocols import (
    DebateStorageProtocol,
    RedisClientProtocol,
    UserStoreProtocol,
)
from aragora.protocols.tracker_protocols import (
    CalibrationTrackerProtocol,
    DissentRetrieverProtocol,
    EloSystemProtocol,
    MomentDetectorProtocol,
    PersonaManagerProtocol,
    PositionLedgerProtocol,
    PositionTrackerProtocol,
    RelationshipTrackerProtocol,
)


# ============================================================================
# Result[T] tests
# ============================================================================


class TestResult:
    """Tests for the generic Result[T] dataclass."""

    def test_ok_creates_success_result(self) -> None:
        r = Result.ok(42)
        assert r.success is True
        assert r.value == 42
        assert r.error is None

    def test_ok_with_string(self) -> None:
        r = Result.ok("hello")
        assert r.success is True
        assert r.value == "hello"

    def test_ok_with_none_value(self) -> None:
        r = Result.ok(None)
        assert r.success is True
        assert r.value is None
        assert r.error is None

    def test_ok_with_complex_type(self) -> None:
        data = {"key": [1, 2, 3]}
        r = Result.ok(data)
        assert r.success is True
        assert r.value is data

    def test_fail_creates_failure_result(self) -> None:
        r = Result.fail("something went wrong")
        assert r.success is False
        assert r.value is None
        assert r.error == "something went wrong"

    def test_fail_with_empty_string(self) -> None:
        r = Result.fail("")
        assert r.success is False
        assert r.error == ""

    def test_truthiness_ok_result(self) -> None:
        """Result is a dataclass; bool(result) is always True for dataclass instances."""
        r = Result.ok(42)
        # Dataclass instances are always truthy -- but success field
        # is what callers should check.
        assert r.success is True

    def test_truthiness_fail_result(self) -> None:
        r = Result.fail("error")
        assert r.success is False

    def test_direct_construction_success(self) -> None:
        r = Result(success=True, value="data")
        assert r.success is True
        assert r.value == "data"
        assert r.error is None

    def test_direct_construction_failure(self) -> None:
        r = Result(success=False, error="oops")
        assert r.success is False
        assert r.value is None
        assert r.error == "oops"

    def test_direct_construction_defaults(self) -> None:
        r = Result(success=True)
        assert r.value is None
        assert r.error is None

    def test_ok_and_fail_are_classmethods(self) -> None:
        assert isinstance(inspect.getattr_static(Result, "ok"), classmethod)
        assert isinstance(inspect.getattr_static(Result, "fail"), classmethod)

    def test_result_is_dataclass(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(Result)

    def test_result_equality(self) -> None:
        """Dataclasses support equality comparison by default."""
        assert Result.ok(42) == Result.ok(42)
        assert Result.fail("x") == Result.fail("x")
        assert Result.ok(42) != Result.fail("x")
        assert Result.ok(1) != Result.ok(2)


# ============================================================================
# Helper: non-conforming empty class
# ============================================================================


class Empty:
    """A class that conforms to no protocol."""

    pass


# ============================================================================
# AgentProtocol tests
# ============================================================================


class ConformingAgent:
    name = "test-agent"

    async def respond(self, prompt: str, context: str | None = None) -> str:
        return "response"


class AgentMissingName:
    async def respond(self, prompt: str, context: str | None = None) -> str:
        return "response"


class AgentMissingRespond:
    name = "test-agent"


class TestAgentProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingAgent(), AgentProtocol)

    def test_missing_name_not_instance(self) -> None:
        assert not isinstance(AgentMissingName(), AgentProtocol)

    def test_missing_respond_not_instance(self) -> None:
        assert not isinstance(AgentMissingRespond(), AgentProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), AgentProtocol)

    def test_respond_is_coroutine_function(self) -> None:
        agent = ConformingAgent()
        assert asyncio.iscoroutinefunction(agent.respond)

    def test_protocol_is_runtime_checkable(self) -> None:
        assert getattr(AgentProtocol, "__protocol_attrs__", None) is not None or hasattr(
            AgentProtocol, "_is_runtime_protocol"
        )


# ============================================================================
# StreamingAgentProtocol tests
# ============================================================================


class ConformingStreamingAgent:
    name = "stream-agent"

    async def respond(self, prompt: str, context: str | None = None) -> str:
        return "response"

    async def stream(self, prompt: str, context: str | None = None) -> AsyncIterator[str]:
        yield "token"


class StreamingAgentMissingStream:
    name = "stream-agent"

    async def respond(self, prompt: str, context: str | None = None) -> str:
        return "response"


class TestStreamingAgentProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingStreamingAgent(), StreamingAgentProtocol)

    def test_missing_stream_not_instance(self) -> None:
        assert not isinstance(StreamingAgentMissingStream(), StreamingAgentProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), StreamingAgentProtocol)


# ============================================================================
# ToolUsingAgentProtocol tests
# ============================================================================


class ConformingToolAgent:
    name = "tool-agent"
    available_tools = ["search", "calculate"]

    async def respond(self, prompt: str, context: str | None = None) -> str:
        return "response"

    async def respond_with_tools(
        self, prompt: str, tools: list[dict[str, Any]], context: str | None = None
    ) -> str:
        return "tool-response"


class ToolAgentMissingTools:
    name = "tool-agent"

    async def respond(self, prompt: str, context: str | None = None) -> str:
        return "response"

    async def respond_with_tools(
        self, prompt: str, tools: list[dict[str, Any]], context: str | None = None
    ) -> str:
        return "tool-response"


class ToolAgentMissingRespondWithTools:
    name = "tool-agent"
    available_tools = ["search"]

    async def respond(self, prompt: str, context: str | None = None) -> str:
        return "response"


class TestToolUsingAgentProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingToolAgent(), ToolUsingAgentProtocol)

    def test_missing_available_tools_not_instance(self) -> None:
        assert not isinstance(ToolAgentMissingTools(), ToolUsingAgentProtocol)

    def test_missing_respond_with_tools_not_instance(self) -> None:
        assert not isinstance(ToolAgentMissingRespondWithTools(), ToolUsingAgentProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), ToolUsingAgentProtocol)


# ============================================================================
# DebateResultProtocol tests
# ============================================================================


class ConformingDebateResult:
    rounds = 3
    consensus_reached = True
    final_answer = "result"
    messages: list[Any] = []


class DebateResultMissingRounds:
    consensus_reached = True
    final_answer = "result"
    messages: list[Any] = []


class TestDebateResultProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingDebateResult(), DebateResultProtocol)

    def test_missing_rounds_not_instance(self) -> None:
        assert not isinstance(DebateResultMissingRounds(), DebateResultProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), DebateResultProtocol)


# ============================================================================
# ConsensusDetectorProtocol tests
# ============================================================================


class ConformingConsensusDetector:
    def check_consensus(self, votes: list[Any], threshold: float = 0.5) -> bool:
        return True

    def get_winner(self, votes: list[Any]) -> str | None:
        return "winner"


class ConsensusDetectorMissingGetWinner:
    def check_consensus(self, votes: list[Any], threshold: float = 0.5) -> bool:
        return True


class TestConsensusDetectorProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingConsensusDetector(), ConsensusDetectorProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(ConsensusDetectorMissingGetWinner(), ConsensusDetectorProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), ConsensusDetectorProtocol)


# ============================================================================
# RankingSystemProtocol tests
# ============================================================================


class ConformingRankingSystem:
    def get_rating(self, agent: str) -> float:
        return 1500.0

    def record_match(
        self, agent_a: str, agent_b: str, scores: dict[str, float], context: str
    ) -> None:
        pass

    def get_leaderboard(self, limit: int = 10) -> list[Any]:
        return []


class RankingSystemMissingRecordMatch:
    def get_rating(self, agent: str) -> float:
        return 1500.0

    def get_leaderboard(self, limit: int = 10) -> list[Any]:
        return []


class TestRankingSystemProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingRankingSystem(), RankingSystemProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(RankingSystemMissingRecordMatch(), RankingSystemProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), RankingSystemProtocol)


# ============================================================================
# ConsensusMemoryProtocol tests
# ============================================================================


class ConformingConsensusMemory:
    def store_outcome(
        self,
        topic: str,
        position: str,
        confidence: float,
        supporting_agents: list[str],
        debate_id: str,
        domain: str | None = None,
    ) -> str:
        return "outcome-id"

    def get_consensus(self, topic: str, domain: str | None = None) -> dict[str, Any] | None:
        return None

    def search_similar_topics(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return []

    def store_consensus(self, topic: str = "", conclusion: str = "", **kwargs: Any) -> Any:
        return "id"

    def update_cruxes(self, consensus_id: Any, cruxes: list[dict[str, Any]], **kwargs: Any) -> None:
        pass

    def store_vote(
        self, debate_id: str = "", vote_data: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        pass

    def store_dissent(
        self, debate_id: str = "", agent_id: str = "", dissent_type: Any = None, **kwargs: Any
    ) -> None:
        pass


class ConsensusMemoryMissingStoreOutcome:
    def get_consensus(self, topic: str, domain: str | None = None) -> dict[str, Any] | None:
        return None

    def search_similar_topics(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return []

    def store_consensus(self, **kwargs: Any) -> Any:
        return "id"

    def update_cruxes(self, consensus_id: Any, cruxes: list[dict[str, Any]], **kwargs: Any) -> None:
        pass

    def store_vote(self, **kwargs: Any) -> None:
        pass

    def store_dissent(self, **kwargs: Any) -> None:
        pass


class TestConsensusMemoryProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingConsensusMemory(), ConsensusMemoryProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(ConsensusMemoryMissingStoreOutcome(), ConsensusMemoryProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), ConsensusMemoryProtocol)


# ============================================================================
# DebateEmbeddingsProtocol tests
# ============================================================================


class ConformingDebateEmbeddings:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2]

    def index_debate(
        self, debate_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        pass

    def search_similar(
        self, query: str, limit: int = 10, threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        return []


class DebateEmbeddingsMissingEmbed:
    def index_debate(
        self, debate_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        pass

    def search_similar(
        self, query: str, limit: int = 10, threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        return []


class TestDebateEmbeddingsProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingDebateEmbeddings(), DebateEmbeddingsProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(DebateEmbeddingsMissingEmbed(), DebateEmbeddingsProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), DebateEmbeddingsProtocol)


# ============================================================================
# FlipDetectorProtocol tests
# ============================================================================


class ConformingFlipDetector:
    def detect_flip(
        self, agent: str, old_position: str, new_position: str, threshold: float = 0.3
    ) -> dict[str, Any] | None:
        return None

    def get_flip_history(self, agent: str, limit: int = 20) -> list[dict[str, Any]]:
        return []

    def get_consistency_score(self, agent: str) -> float:
        return 1.0

    def detect_flips_for_agent(self, agent: str, **kwargs: Any) -> list[dict[str, Any]]:
        return []


class FlipDetectorMissingDetect:
    def get_flip_history(self, agent: str, limit: int = 20) -> list[dict[str, Any]]:
        return []

    def get_consistency_score(self, agent: str) -> float:
        return 1.0

    def detect_flips_for_agent(self, agent: str, **kwargs: Any) -> list[dict[str, Any]]:
        return []


class TestFlipDetectorProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingFlipDetector(), FlipDetectorProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(FlipDetectorMissingDetect(), FlipDetectorProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), FlipDetectorProtocol)


# ============================================================================
# EventEmitterProtocol tests
# ============================================================================


class ConformingEventEmitter:
    def emit(self, event: Any, data: dict[str, Any] | None = None) -> None:
        pass

    def on(self, event_type: str, callback: Any) -> None:
        pass


class EventEmitterMissingOn:
    def emit(self, event: Any, data: dict[str, Any] | None = None) -> None:
        pass


class TestEventEmitterProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingEventEmitter(), EventEmitterProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(EventEmitterMissingOn(), EventEmitterProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), EventEmitterProtocol)


# ============================================================================
# AsyncEventEmitterProtocol tests
# ============================================================================


class ConformingAsyncEventEmitter:
    async def emit_async(self, event_type: str, data: dict[str, Any]) -> None:
        pass


class AsyncEventEmitterMissingMethod:
    pass


class TestAsyncEventEmitterProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingAsyncEventEmitter(), AsyncEventEmitterProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(AsyncEventEmitterMissingMethod(), AsyncEventEmitterProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), AsyncEventEmitterProtocol)

    def test_emit_async_is_coroutine(self) -> None:
        emitter = ConformingAsyncEventEmitter()
        assert asyncio.iscoroutinefunction(emitter.emit_async)


# ============================================================================
# HandlerProtocol tests
# ============================================================================


class ConformingHandler:
    def can_handle(self, path: str) -> bool:
        return True

    def handle(self, path: str, query: dict[str, Any], request_handler: Any) -> Any | None:
        return None


class HandlerMissingCanHandle:
    def handle(self, path: str, query: dict[str, Any], request_handler: Any) -> Any | None:
        return None


class TestHandlerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingHandler(), HandlerProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(HandlerMissingCanHandle(), HandlerProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), HandlerProtocol)


# ============================================================================
# BaseHandlerProtocol tests
# ============================================================================


class ConformingBaseHandler:
    ROUTES = ["/api/v1/test"]
    ctx: dict[str, Any] = {}

    def can_handle(self, path: str) -> bool:
        return True

    def handle(self, path: str, query: dict[str, Any], request_handler: Any) -> Any | None:
        return None

    def read_json_body(self, handler: Any) -> dict[str, Any] | None:
        return None


class BaseHandlerMissingRoutes:
    ctx: dict[str, Any] = {}

    def can_handle(self, path: str) -> bool:
        return True

    def handle(self, path: str, query: dict[str, Any], request_handler: Any) -> Any | None:
        return None

    def read_json_body(self, handler: Any) -> dict[str, Any] | None:
        return None


class TestBaseHandlerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingBaseHandler(), BaseHandlerProtocol)

    def test_missing_routes_not_instance(self) -> None:
        assert not isinstance(BaseHandlerMissingRoutes(), BaseHandlerProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), BaseHandlerProtocol)


# ============================================================================
# MemoryProtocol tests
# ============================================================================


class ConformingMemory:
    def store(self, content: str, **kwargs: Any) -> str:
        return "id"

    def query(self, **kwargs: Any) -> list[Any]:
        return []


class MemoryMissingStore:
    def query(self, **kwargs: Any) -> list[Any]:
        return []


class TestMemoryProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingMemory(), MemoryProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(MemoryMissingStore(), MemoryProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), MemoryProtocol)


# ============================================================================
# TieredMemoryProtocol tests
# ============================================================================


class ConformingTieredMemory:
    def store(self, content: str, tier: Any = None, importance: float = 0.5, **kwargs: Any) -> str:
        return "id"

    def query(
        self, tier: Any = None, limit: int = 10, min_importance: float = 0.0, **kwargs: Any
    ) -> list[Any]:
        return []

    def promote(self, entry_id: str, target_tier: Any) -> bool:
        return True

    def demote(self, entry_id: str, target_tier: Any) -> bool:
        return True

    def cleanup_expired_memories(self) -> int:
        return 0

    def enforce_tier_limits(self) -> None:
        pass


class TieredMemoryMissingPromote:
    def store(self, content: str, tier: Any = None, importance: float = 0.5, **kwargs: Any) -> str:
        return "id"

    def query(
        self, tier: Any = None, limit: int = 10, min_importance: float = 0.0, **kwargs: Any
    ) -> list[Any]:
        return []

    def demote(self, entry_id: str, target_tier: Any) -> bool:
        return True

    def cleanup_expired_memories(self) -> int:
        return 0

    def enforce_tier_limits(self) -> None:
        pass


class TestTieredMemoryProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingTieredMemory(), TieredMemoryProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(TieredMemoryMissingPromote(), TieredMemoryProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), TieredMemoryProtocol)


# ============================================================================
# CritiqueStoreProtocol tests
# ============================================================================


class ConformingCritiqueStore:
    def store_pattern(self, critique: Any, resolution: str) -> str:
        return "id"

    def retrieve_patterns(self, issue_type: str | None = None, limit: int = 10) -> list[Any]:
        return []

    def get_reputation(self, agent: str) -> dict[str, Any]:
        return {}


class CritiqueStoreMissingStorePattern:
    def retrieve_patterns(self, issue_type: str | None = None, limit: int = 10) -> list[Any]:
        return []

    def get_reputation(self, agent: str) -> dict[str, Any]:
        return {}


class TestCritiqueStoreProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingCritiqueStore(), CritiqueStoreProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(CritiqueStoreMissingStorePattern(), CritiqueStoreProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), CritiqueStoreProtocol)


# ============================================================================
# ContinuumMemoryProtocol tests
# ============================================================================


class ConformingContinuumMemory:
    def store(
        self, key: str, value: Any, tier: str = "medium", metadata: dict[str, Any] | None = None
    ) -> None:
        pass

    def retrieve(self, key: str, tier: str | None = None) -> Any | None:
        return None

    def search(self, query: str, limit: int = 10, tier: str | None = None) -> list[dict[str, Any]]:
        return []

    def get_context(self, task: str, limit: int = 5) -> str:
        return ""


class ContinuumMemoryMissingSearch:
    def store(
        self, key: str, value: Any, tier: str = "medium", metadata: dict[str, Any] | None = None
    ) -> None:
        pass

    def retrieve(self, key: str, tier: str | None = None) -> Any | None:
        return None

    def get_context(self, task: str, limit: int = 5) -> str:
        return ""


class TestContinuumMemoryProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingContinuumMemory(), ContinuumMemoryProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(ContinuumMemoryMissingSearch(), ContinuumMemoryProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), ContinuumMemoryProtocol)


# ============================================================================
# RedisClientProtocol tests
# ============================================================================


class ConformingRedisClient:
    def close(self) -> None:
        pass

    def ping(self) -> bool:
        return True

    def info(self, section: str | None = None) -> dict[str, Any]:
        return {}

    def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def get(self, key: str) -> Any:
        return None

    def set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool | None:
        return True

    def delete(self, *keys: str) -> int:
        return 0

    def exists(self, *keys: str) -> int:
        return 0

    def expire(self, key: str, seconds: int) -> bool:
        return True

    def ttl(self, key: str) -> int:
        return -1

    def incr(self, key: str) -> int:
        return 1

    def decr(self, key: str) -> int:
        return 0

    def hget(self, name: str, key: str) -> Any:
        return None

    def hset(self, name: str, key: str, value: Any) -> int:
        return 1

    def hgetall(self, name: str) -> dict[str, Any]:
        return {}

    def hdel(self, name: str, *keys: str) -> int:
        return 0

    def zadd(self, name: str, mapping: dict[str, float]) -> int:
        return 0

    def zrem(self, name: str, *members: str) -> int:
        return 0

    def zcard(self, name: str) -> int:
        return 0

    def zrangebyscore(self, name: str, min: Any, max: Any, withscores: bool = False) -> list[Any]:
        return []

    def zremrangebyscore(self, name: str, min: Any, max: Any) -> int:
        return 0

    def pipeline(self, transaction: bool = True) -> Any:
        return None


class RedisClientMissingPing:
    def close(self) -> None:
        pass

    def get(self, key: str) -> Any:
        return None


class TestRedisClientProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingRedisClient(), RedisClientProtocol)

    def test_missing_methods_not_instance(self) -> None:
        assert not isinstance(RedisClientMissingPing(), RedisClientProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), RedisClientProtocol)


# ============================================================================
# DebateStorageProtocol tests
# ============================================================================


class ConformingDebateStorage:
    def save_debate(self, debate_id: str, data: dict[str, Any]) -> None:
        pass

    def load_debate(self, debate_id: str) -> dict[str, Any] | None:
        return None

    def list_debates(self, limit: int = 100, org_id: str | None = None) -> list[Any]:
        return []

    def delete_debate(self, debate_id: str) -> bool:
        return True

    def get_debate(self, debate_id: str) -> dict[str, Any] | None:
        return None

    def get_debate_by_slug(self, slug: str) -> dict[str, Any] | None:
        return None

    def get_by_id(self, debate_id: str) -> dict[str, Any] | None:
        return None

    def get_by_slug(self, slug: str) -> dict[str, Any] | None:
        return None

    def list_recent(self, limit: int = 20, org_id: str | None = None) -> list[Any]:
        return []

    def search(
        self,
        query: str | None = None,
        agent: str | None = None,
        min_confidence: float | None = None,
        limit: int = 20,
        org_id: str | None = None,
    ) -> list[Any]:
        return []


class DebateStorageMissingSave:
    def load_debate(self, debate_id: str) -> dict[str, Any] | None:
        return None


class TestDebateStorageProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingDebateStorage(), DebateStorageProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(DebateStorageMissingSave(), DebateStorageProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), DebateStorageProtocol)


# ============================================================================
# UserStoreProtocol tests
# ============================================================================


class ConformingUserStore:
    def get_user_by_id(self, user_id: str) -> Any | None:
        return None

    def get_user_by_email(self, email: str) -> Any | None:
        return None

    def create_user(self, email: str, password_hash: str, password_salt: str, **kwargs: Any) -> Any:
        return {"id": "1"}

    def update_user(self, user_id: str, **kwargs: Any) -> bool:
        return True


class UserStoreMissingCreateUser:
    def get_user_by_id(self, user_id: str) -> Any | None:
        return None

    def get_user_by_email(self, email: str) -> Any | None:
        return None

    def update_user(self, user_id: str, **kwargs: Any) -> bool:
        return True


class TestUserStoreProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingUserStore(), UserStoreProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(UserStoreMissingCreateUser(), UserStoreProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), UserStoreProtocol)


# ============================================================================
# VerificationBackendProtocol tests
# ============================================================================


class ConformingVerificationBackend:
    @property
    def is_available(self) -> bool:
        return True

    def can_verify(self, claim: str, claim_type: str | None = None) -> bool:
        return True

    async def translate(self, claim: str) -> str:
        return "formal"

    async def prove(self, formal_statement: str) -> Any:
        return True


class VerificationBackendMissingProve:
    @property
    def is_available(self) -> bool:
        return True

    def can_verify(self, claim: str, claim_type: str | None = None) -> bool:
        return True

    async def translate(self, claim: str) -> str:
        return "formal"


class TestVerificationBackendProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingVerificationBackend(), VerificationBackendProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(VerificationBackendMissingProve(), VerificationBackendProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), VerificationBackendProtocol)


# ============================================================================
# PopulationManagerProtocol tests
# ============================================================================


class ConformingPopulationManager:
    def get_population(self, limit: int = 100) -> list[dict[str, Any]]:
        return []

    def update_fitness(
        self, genome_id: str, fitness_delta: float = 0.0, context: str | None = None, **kwargs: Any
    ) -> None:
        pass

    def breed(self, parent_a: str, parent_b: str, mutation_rate: float = 0.1) -> str:
        return "child-id"

    def select_for_breeding(self, count: int = 2, threshold: float = 0.8) -> list[str]:
        return []

    def get_or_create_population(self, agent_names: list[str], **kwargs: Any) -> Any:
        return None

    def evolve_population(self, population: Any, **kwargs: Any) -> Any:
        return None


class PopulationManagerMissingBreed:
    def get_population(self, limit: int = 100) -> list[dict[str, Any]]:
        return []

    def update_fitness(self, genome_id: str, **kwargs: Any) -> None:
        pass

    def select_for_breeding(self, count: int = 2, threshold: float = 0.8) -> list[str]:
        return []

    def get_or_create_population(self, agent_names: list[str], **kwargs: Any) -> Any:
        return None

    def evolve_population(self, population: Any, **kwargs: Any) -> Any:
        return None


class TestPopulationManagerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingPopulationManager(), PopulationManagerProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(PopulationManagerMissingBreed(), PopulationManagerProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), PopulationManagerProtocol)


# ============================================================================
# PulseManagerProtocol tests
# ============================================================================


class ConformingPulseManager:
    def get_trending(
        self, sources: list[str] | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        return []

    def record_debate_outcome(self, topic: str = "", platform: str = "", **kwargs: Any) -> None:
        pass

    def get_topic_analytics(self, days: int = 30) -> dict[str, Any]:
        return {}


class PulseManagerMissingGetTrending:
    def record_debate_outcome(self, **kwargs: Any) -> None:
        pass

    def get_topic_analytics(self, days: int = 30) -> dict[str, Any]:
        return {}


class TestPulseManagerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingPulseManager(), PulseManagerProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(PulseManagerMissingGetTrending(), PulseManagerProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), PulseManagerProtocol)


# ============================================================================
# PromptEvolverProtocol tests
# ============================================================================


class ConformingPromptEvolver:
    def get_current_prompt(self, agent: str) -> str:
        return "prompt"

    def record_outcome(
        self,
        agent: str,
        prompt_variant: str,
        success: bool,
        score: float,
        context: str | None = None,
    ) -> None:
        pass

    def evolve(self, agent: str) -> str | None:
        return None

    def get_evolution_history(self, agent: str, limit: int = 20) -> list[dict[str, Any]]:
        return []

    def extract_winning_patterns(
        self, debate_results: list[Any], **kwargs: Any
    ) -> list[dict[str, Any]]:
        return []

    def store_patterns(self, patterns: list[dict[str, Any]], **kwargs: Any) -> None:
        pass

    def update_performance(
        self, agent_name: str = "", version: Any | None = None, **kwargs: Any
    ) -> None:
        pass


class PromptEvolverMissingEvolve:
    def get_current_prompt(self, agent: str) -> str:
        return "prompt"

    def record_outcome(
        self, agent: str, prompt_variant: str, success: bool, score: float, **kwargs: Any
    ) -> None:
        pass

    def get_evolution_history(self, agent: str, limit: int = 20) -> list[dict[str, Any]]:
        return []

    def extract_winning_patterns(
        self, debate_results: list[Any], **kwargs: Any
    ) -> list[dict[str, Any]]:
        return []

    def store_patterns(self, patterns: list[dict[str, Any]], **kwargs: Any) -> None:
        pass

    def update_performance(self, **kwargs: Any) -> None:
        pass


class TestPromptEvolverProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingPromptEvolver(), PromptEvolverProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(PromptEvolverMissingEvolve(), PromptEvolverProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), PromptEvolverProtocol)


# ============================================================================
# InsightStoreProtocol tests
# ============================================================================


class ConformingInsightStore:
    def store_insight(
        self,
        insight_type: str,
        content: str,
        source_debate_id: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return "insight-id"

    def mark_applied(
        self, insight_id: str, target_debate_id: str, success: bool | None = None
    ) -> None:
        pass

    def get_recent_insights(
        self, insight_type: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        return []

    def get_effectiveness(self, insight_type: str | None = None) -> dict[str, Any]:
        return {}

    async def record_insight_usage(
        self, insight_id: str = "", debate_id: str = "", was_successful: bool = False, **kwargs: Any
    ) -> None:
        pass


class InsightStoreMissingStoreInsight:
    def mark_applied(self, insight_id: str, target_debate_id: str, **kwargs: Any) -> None:
        pass

    def get_recent_insights(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def get_effectiveness(self, **kwargs: Any) -> dict[str, Any]:
        return {}

    async def record_insight_usage(self, **kwargs: Any) -> None:
        pass


class TestInsightStoreProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingInsightStore(), InsightStoreProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(InsightStoreMissingStoreInsight(), InsightStoreProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), InsightStoreProtocol)


# ============================================================================
# BroadcastPipelineProtocol tests
# ============================================================================


class ConformingBroadcastPipeline:
    def should_broadcast(self, debate_result: Any, min_confidence: float = 0.8) -> bool:
        return True

    def queue_broadcast(
        self,
        debate_id: str,
        platforms: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        return "job-id"

    def get_broadcast_status(self, job_id: str) -> dict[str, Any]:
        return {"status": "pending"}

    def get_supported_platforms(self) -> list[str]:
        return ["slack"]

    async def run(self, debate_id: str, options: Any = None) -> Any:
        return None


class BroadcastPipelineMissingShouldBroadcast:
    def queue_broadcast(
        self, debate_id: str, platforms: list[str] | None = None, **kwargs: Any
    ) -> str:
        return "job-id"

    def get_broadcast_status(self, job_id: str) -> dict[str, Any]:
        return {}

    def get_supported_platforms(self) -> list[str]:
        return []

    async def run(self, debate_id: str, options: Any = None) -> Any:
        return None


class TestBroadcastPipelineProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingBroadcastPipeline(), BroadcastPipelineProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(BroadcastPipelineMissingShouldBroadcast(), BroadcastPipelineProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), BroadcastPipelineProtocol)


# ============================================================================
# EvidenceCollectorProtocol tests
# ============================================================================


class ConformingEvidenceCollector:
    def collect(
        self, query: str, sources: list[str] | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        return []

    def verify(self, claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        return {"verified": True}

    def get_sources(self) -> list[str]:
        return ["web"]


class EvidenceCollectorMissingCollect:
    def verify(self, claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    def get_sources(self) -> list[str]:
        return []


class TestEvidenceCollectorProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingEvidenceCollector(), EvidenceCollectorProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(EvidenceCollectorMissingCollect(), EvidenceCollectorProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), EvidenceCollectorProtocol)


# ============================================================================
# EvidenceProtocol tests (non-runtime_checkable, attribute-only)
# ============================================================================


class ConformingEvidence:
    id = "ev-1"
    source_type = "web"
    source_id = "src-1"
    content = "Evidence content"
    title = "Title"
    confidence = 0.9
    freshness = 0.8
    authority = 0.7
    metadata: dict[str, Any] = {}


class EvidenceMissingContent:
    id = "ev-1"
    source_type = "web"
    source_id = "src-1"
    title = "Title"
    confidence = 0.9
    freshness = 0.8
    authority = 0.7
    metadata: dict[str, Any] = {}


class TestEvidenceProtocol:
    """EvidenceProtocol is not @runtime_checkable, so isinstance checks
    will not work. We verify it is a Protocol and test structural typing
    through attribute presence."""

    def test_conforming_has_all_attributes(self) -> None:
        ev = ConformingEvidence()
        expected_attrs = [
            "id",
            "source_type",
            "source_id",
            "content",
            "title",
            "confidence",
            "freshness",
            "authority",
            "metadata",
        ]
        for attr in expected_attrs:
            assert hasattr(ev, attr), f"Missing attribute: {attr}"

    def test_non_conforming_missing_attribute(self) -> None:
        ev = EvidenceMissingContent()
        assert not hasattr(ev, "content")


# ============================================================================
# StreamEventProtocol tests (non-runtime_checkable, attribute-only)
# ============================================================================


class ConformingStreamEvent:
    type = "debate_start"
    data: dict[str, Any] = {}
    timestamp = 1234567890.0
    round = 1
    agent = "claude"


class StreamEventMissingType:
    data: dict[str, Any] = {}
    timestamp = 1234567890.0
    round = 1
    agent = "claude"


class TestStreamEventProtocol:
    def test_conforming_has_all_attributes(self) -> None:
        ev = ConformingStreamEvent()
        for attr in ["type", "data", "timestamp", "round", "agent"]:
            assert hasattr(ev, attr), f"Missing attribute: {attr}"

    def test_non_conforming_missing_attribute(self) -> None:
        ev = StreamEventMissingType()
        assert not hasattr(ev, "type")


# ============================================================================
# WebhookConfigProtocol tests (non-runtime_checkable, attribute-only)
# ============================================================================


class ConformingWebhookConfig:
    name = "test-hook"
    url = "https://example.com/hook"
    secret = "s3cret"
    event_types: set[str] = {"debate_end"}
    timeout_s = 30.0
    max_retries = 3


class WebhookConfigMissingUrl:
    name = "test-hook"
    secret = "s3cret"
    event_types: set[str] = {"debate_end"}
    timeout_s = 30.0
    max_retries = 3


class TestWebhookConfigProtocol:
    def test_conforming_has_all_attributes(self) -> None:
        wh = ConformingWebhookConfig()
        for attr in ["name", "url", "secret", "event_types", "timeout_s", "max_retries"]:
            assert hasattr(wh, attr), f"Missing attribute: {attr}"

    def test_non_conforming_missing_attribute(self) -> None:
        wh = WebhookConfigMissingUrl()
        assert not hasattr(wh, "url")


# ============================================================================
# EloSystemProtocol tests
# ============================================================================


class ConformingEloSystem:
    def get_rating(self, agent: str, domain: str = "") -> float:
        return 1500.0

    def record_match(
        self,
        debate_id: str,
        participants: list[str],
        scores: dict[str, float],
        domain: str = "",
        confidence_weight: float = 1.0,
        winner: str | None = None,
        loser: str | None = None,
        margin: float = 1.0,
    ) -> None:
        pass

    def get_leaderboard(self, limit: int = 10, domain: str = "") -> list[Any]:
        return []

    def get_match_history(self, agent: str, limit: int = 20) -> list[Any]:
        return []

    def get_ratings_batch(self, agents: list[str]) -> dict[str, Any]:
        return {}

    def update_voting_accuracy(
        self,
        agent_name: str,
        voted_for_consensus: bool,
        domain: str = "general",
        debate_id: str | None = None,
        apply_elo_bonus: bool = True,
        bonus_k_factor: float = 4.0,
    ) -> float:
        return 1500.0

    def apply_learning_bonus(
        self,
        agent_name: str,
        domain: str = "general",
        debate_id: str | None = None,
        bonus_factor: float = 0.5,
    ) -> float:
        return 1500.0


class EloSystemMissingRecordMatch:
    def get_rating(self, agent: str, domain: str = "") -> float:
        return 1500.0

    def get_leaderboard(self, limit: int = 10, domain: str = "") -> list[Any]:
        return []


class TestEloSystemProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingEloSystem(), EloSystemProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(EloSystemMissingRecordMatch(), EloSystemProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), EloSystemProtocol)


# ============================================================================
# CalibrationTrackerProtocol tests
# ============================================================================


class ConformingCalibrationTracker:
    def get_calibration(self, agent: str) -> dict[str, Any] | None:
        return None

    def record_prediction(
        self,
        agent: str,
        confidence: float,
        correct: bool,
        domain: str = "",
        debate_id: str | None = None,
        prediction_type: str | None = None,
    ) -> None:
        pass

    def get_calibration_score(self, agent: str) -> float:
        return 0.0


class CalibrationTrackerMissingRecordPrediction:
    def get_calibration(self, agent: str) -> dict[str, Any] | None:
        return None

    def get_calibration_score(self, agent: str) -> float:
        return 0.0


class TestCalibrationTrackerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingCalibrationTracker(), CalibrationTrackerProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(
            CalibrationTrackerMissingRecordPrediction(), CalibrationTrackerProtocol
        )

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), CalibrationTrackerProtocol)


# ============================================================================
# PositionLedgerProtocol tests
# ============================================================================


class ConformingPositionLedger:
    def record_position(
        self,
        agent_name: str,
        claim: str,
        stance: str,
        confidence: float,
        debate_id: str,
        round_num: int,
        domain: str | None = None,
    ) -> None:
        pass

    def get_positions(
        self, agent_name: str, limit: int = 10, claim_filter: str | None = None
    ) -> list[Any]:
        return []

    def get_agent_positions(
        self, agent_name: str, limit: int = 100, outcome_filter: str | None = None
    ) -> list[Any]:
        return []

    def get_consistency_score(self, agent_name: str) -> float:
        return 1.0

    def resolve_position(
        self,
        position_id: str | None = None,
        outcome: str | None = None,
        resolution_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        pass


class PositionLedgerMissingRecordPosition:
    def get_positions(self, agent_name: str, **kwargs: Any) -> list[Any]:
        return []

    def get_agent_positions(self, agent_name: str, **kwargs: Any) -> list[Any]:
        return []

    def get_consistency_score(self, agent_name: str) -> float:
        return 1.0

    def resolve_position(self, **kwargs: Any) -> None:
        pass


class TestPositionLedgerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingPositionLedger(), PositionLedgerProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(PositionLedgerMissingRecordPosition(), PositionLedgerProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), PositionLedgerProtocol)


# ============================================================================
# RelationshipTrackerProtocol tests
# ============================================================================


class ConformingRelationshipTracker:
    def get_relationship(self, agent_a: str, agent_b: str) -> dict[str, Any] | None:
        return None

    def update_relationship(
        self, agent_a: str, agent_b: str, outcome: str, debate_id: str = ""
    ) -> None:
        pass

    def get_allies(self, agent: str, threshold: float = 0.6) -> list[str]:
        return []

    def get_adversaries(self, agent: str, threshold: float = 0.6) -> list[str]:
        return []

    def update_from_debate(
        self,
        debate_id: str = "",
        participants: list[str] | None = None,
        winner: str | None = None,
        votes: dict[str, Any] | None = None,
        critiques: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        pass


class RelationshipTrackerMissingGetAllies:
    def get_relationship(self, agent_a: str, agent_b: str) -> dict[str, Any] | None:
        return None

    def update_relationship(
        self, agent_a: str, agent_b: str, outcome: str, debate_id: str = ""
    ) -> None:
        pass

    def get_adversaries(self, agent: str, threshold: float = 0.6) -> list[str]:
        return []

    def update_from_debate(self, **kwargs: Any) -> None:
        pass


class TestRelationshipTrackerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingRelationshipTracker(), RelationshipTrackerProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(RelationshipTrackerMissingGetAllies(), RelationshipTrackerProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), RelationshipTrackerProtocol)


# ============================================================================
# MomentDetectorProtocol tests
# ============================================================================


class ConformingMomentDetector:
    def detect_moment(
        self, content: str, context: dict[str, Any], threshold: float = 0.7
    ) -> dict[str, Any] | None:
        return None

    def get_moment_types(self) -> list[str]:
        return ["breakthrough"]

    def detect_upset_victory(
        self, winner: str = "", loser: str = "", debate_id: str = "", **kwargs: Any
    ) -> dict[str, Any] | None:
        return None

    def detect_calibration_vindication(
        self,
        agent_name: str = "",
        prediction_confidence: float = 0.0,
        was_correct: bool = False,
        domain: str = "",
        debate_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        return None

    def record_moment(self, moment: dict[str, Any] | None = None, **kwargs: Any) -> str | None:
        return "moment-id"


class MomentDetectorMissingDetect:
    def get_moment_types(self) -> list[str]:
        return []

    def detect_upset_victory(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def detect_calibration_vindication(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def record_moment(self, **kwargs: Any) -> str | None:
        return None


class TestMomentDetectorProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingMomentDetector(), MomentDetectorProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(MomentDetectorMissingDetect(), MomentDetectorProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), MomentDetectorProtocol)


# ============================================================================
# PersonaManagerProtocol tests
# ============================================================================


class ConformingPersonaManager:
    def get_persona(self, agent_name: str) -> dict[str, Any] | None:
        return None

    def update_persona(self, agent_name: str, updates: dict[str, Any]) -> None:
        pass

    def get_context_for_prompt(self, agent_name: str) -> str:
        return ""

    def record_performance(
        self,
        agent_name: str,
        domain: str,
        success: bool,
        action: str = "critique",
        debate_id: str | None = None,
    ) -> None:
        pass


class PersonaManagerMissingGetPersona:
    def update_persona(self, agent_name: str, updates: dict[str, Any]) -> None:
        pass

    def get_context_for_prompt(self, agent_name: str) -> str:
        return ""

    def record_performance(
        self, agent_name: str, domain: str, success: bool, **kwargs: Any
    ) -> None:
        pass


class TestPersonaManagerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingPersonaManager(), PersonaManagerProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(PersonaManagerMissingGetPersona(), PersonaManagerProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), PersonaManagerProtocol)


# ============================================================================
# DissentRetrieverProtocol tests
# ============================================================================


class ConformingDissentRetriever:
    def retrieve_dissent(self, topic: str, limit: int = 5, min_relevance: float = 0.5) -> list[Any]:
        return []

    def store_dissent(self, agent: str, position: str, debate_id: str, context: str = "") -> str:
        return "dissent-id"


class DissentRetrieverMissingRetrieve:
    def store_dissent(self, agent: str, position: str, debate_id: str, context: str = "") -> str:
        return "dissent-id"


class TestDissentRetrieverProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingDissentRetriever(), DissentRetrieverProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(DissentRetrieverMissingRetrieve(), DissentRetrieverProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), DissentRetrieverProtocol)


# ============================================================================
# PositionTrackerProtocol tests
# ============================================================================


class ConformingPositionTracker:
    def record_position(
        self,
        agent_name: str,
        position: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def get_position(self, agent_name: str) -> dict[str, Any] | None:
        return None

    def get_position_history(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def has_changed(self, agent_name: str, threshold: float = 0.3) -> bool:
        return False


class PositionTrackerMissingHasChanged:
    def record_position(
        self, agent_name: str, position: str, confidence: float = 1.0, **kwargs: Any
    ) -> None:
        pass

    def get_position(self, agent_name: str) -> dict[str, Any] | None:
        return None

    def get_position_history(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]:
        return []


class TestPositionTrackerProtocol:
    def test_conforming_isinstance(self) -> None:
        assert isinstance(ConformingPositionTracker(), PositionTrackerProtocol)

    def test_missing_method_not_instance(self) -> None:
        assert not isinstance(PositionTrackerMissingHasChanged(), PositionTrackerProtocol)

    def test_empty_not_instance(self) -> None:
        assert not isinstance(Empty(), PositionTrackerProtocol)


# ============================================================================
# Method signature tests
# ============================================================================


class TestMethodSignatures:
    """Verify that protocol methods have the expected parameter names and kinds."""

    def _get_params(self, cls: type, method_name: str) -> list[str]:
        """Get parameter names for a method, excluding 'self'."""
        method = getattr(cls, method_name)
        sig = inspect.signature(method)
        return [p for p in sig.parameters if p != "self"]

    def test_agent_respond_signature(self) -> None:
        params = self._get_params(AgentProtocol, "respond")
        assert "prompt" in params
        assert "context" in params

    def test_streaming_agent_stream_signature(self) -> None:
        params = self._get_params(StreamingAgentProtocol, "stream")
        assert "prompt" in params
        assert "context" in params

    def test_tool_agent_respond_with_tools_signature(self) -> None:
        params = self._get_params(ToolUsingAgentProtocol, "respond_with_tools")
        assert "prompt" in params
        assert "tools" in params
        assert "context" in params

    def test_consensus_detector_check_consensus_signature(self) -> None:
        params = self._get_params(ConsensusDetectorProtocol, "check_consensus")
        assert "votes" in params
        assert "threshold" in params

    def test_memory_store_signature(self) -> None:
        params = self._get_params(MemoryProtocol, "store")
        assert "content" in params

    def test_debate_storage_save_signature(self) -> None:
        params = self._get_params(DebateStorageProtocol, "save_debate")
        assert "debate_id" in params
        assert "data" in params

    def test_elo_system_record_match_signature(self) -> None:
        params = self._get_params(EloSystemProtocol, "record_match")
        assert "debate_id" in params
        assert "participants" in params
        assert "scores" in params

    def test_event_emitter_emit_signature(self) -> None:
        params = self._get_params(EventEmitterProtocol, "emit")
        assert "event" in params
        assert "data" in params

    def test_handler_handle_signature(self) -> None:
        params = self._get_params(HandlerProtocol, "handle")
        assert "path" in params
        assert "query" in params
        assert "request_handler" in params

    def test_redis_client_set_signature(self) -> None:
        params = self._get_params(RedisClientProtocol, "set")
        assert "key" in params
        assert "value" in params
        assert "ex" in params
        assert "nx" in params

    def test_flip_detector_detect_flip_signature(self) -> None:
        params = self._get_params(FlipDetectorProtocol, "detect_flip")
        assert "agent" in params
        assert "old_position" in params
        assert "new_position" in params
        assert "threshold" in params

    def test_position_tracker_has_changed_signature(self) -> None:
        params = self._get_params(PositionTrackerProtocol, "has_changed")
        assert "agent_name" in params
        assert "threshold" in params

    def test_calibration_tracker_record_prediction_signature(self) -> None:
        params = self._get_params(CalibrationTrackerProtocol, "record_prediction")
        assert "agent" in params
        assert "confidence" in params
        assert "correct" in params

    def test_evidence_collector_collect_signature(self) -> None:
        params = self._get_params(EvidenceCollectorProtocol, "collect")
        assert "query" in params
        assert "sources" in params
        assert "limit" in params


# ============================================================================
# Cross-protocol non-overlap tests
# ============================================================================


class TestProtocolIsolation:
    """Ensure that conforming to one protocol does not accidentally
    satisfy an unrelated protocol."""

    def test_agent_is_not_memory(self) -> None:
        assert not isinstance(ConformingAgent(), MemoryProtocol)

    def test_memory_is_not_agent(self) -> None:
        assert not isinstance(ConformingMemory(), AgentProtocol)

    def test_event_emitter_is_not_handler(self) -> None:
        assert not isinstance(ConformingEventEmitter(), HandlerProtocol)

    def test_handler_is_not_event_emitter(self) -> None:
        assert not isinstance(ConformingHandler(), EventEmitterProtocol)

    def test_debate_result_is_not_consensus_detector(self) -> None:
        assert not isinstance(ConformingDebateResult(), ConsensusDetectorProtocol)

    def test_redis_client_is_not_user_store(self) -> None:
        assert not isinstance(ConformingRedisClient(), UserStoreProtocol)

    def test_elo_system_is_not_calibration_tracker(self) -> None:
        assert not isinstance(ConformingEloSystem(), CalibrationTrackerProtocol)

    def test_position_tracker_is_not_position_ledger(self) -> None:
        """PositionTracker and PositionLedger have different record_position signatures."""
        assert not isinstance(ConformingPositionTracker(), PositionLedgerProtocol)
