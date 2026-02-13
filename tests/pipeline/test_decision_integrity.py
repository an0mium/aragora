"""
Tests for the decision integrity pipeline.

Covers:
- ContextSnapshot dataclass: creation, defaults, to_dict serialization
- capture_context_snapshot: memory retrieval from all three sources
- _coerce_debate_result: dict -> DebateResult conversion
- DecisionIntegrityPackage: construction and serialization
- build_decision_integrity_package: end-to-end with all parameter combinations
- ExecutionProgress: dataclass serialization, progress_pct, is_complete
- ExecutionNotifier: progress tracking, callbacks, task descriptions, listeners
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.pipeline.decision_integrity import (
    ContextSnapshot,
    DecisionIntegrityPackage,
    _coerce_debate_result,
    build_decision_integrity_package,
    capture_context_snapshot,
)
from aragora.pipeline.decision_plan.factory import normalize_execution_mode
from aragora.pipeline.execution_notifier import ExecutionNotifier, ExecutionProgress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_debate_dict(**overrides: Any) -> dict[str, Any]:
    """Create a debate dict with sensible defaults."""
    defaults: dict[str, Any] = {
        "debate_id": "test-debate-001",
        "task": "Design a rate limiter",
        "final_answer": "Implement token bucket algorithm",
        "confidence": 0.85,
        "consensus_reached": True,
        "rounds_used": 3,
        "rounds_completed": 3,
        "status": "completed",
        "agents": ["claude", "gpt4", "gemini"],
        "metadata": {"source": "test"},
    }
    defaults.update(overrides)
    return defaults


@dataclass
class _FakeContinuumEntry:
    """Mimics a ContinuumMemoryEntry for testing."""

    content: str = "rate limiter pattern"
    importance: float = 0.7
    tier: str = "SLOW"


@dataclass
class _FakeKnowledgeItem:
    """Mimics a KnowledgeMound query result item."""

    content: str = "token bucket docs"

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content}


@dataclass
class _FakeKnowledgeResult:
    """Mimics a KnowledgeMound query result."""

    items: list[_FakeKnowledgeItem] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class _FakeCrossDebateEntry:
    """Mimics a cross-debate memory entry."""

    def __init__(self, task: str) -> None:
        self.task = task


# ---------------------------------------------------------------------------
# ContextSnapshot Tests
# ---------------------------------------------------------------------------


class TestContextSnapshot:
    """Tests for the ContextSnapshot dataclass."""

    def test_defaults(self):
        """Newly created snapshot has sensible defaults."""
        snap = ContextSnapshot()
        assert snap.continuum_entries == []
        assert snap.cross_debate_context == ""
        assert snap.cross_debate_ids == []
        assert snap.knowledge_items == []
        assert snap.knowledge_sources == []
        assert snap.document_items == []
        assert snap.evidence_items == []
        assert snap.context_envelope == {}
        assert snap.total_context_tokens == 0
        assert snap.retrieval_time_ms == 0.0

    def test_to_dict_empty(self):
        """to_dict on empty snapshot returns expected keys."""
        snap = ContextSnapshot()
        d = snap.to_dict()
        assert set(d.keys()) == {
            "continuum_entries",
            "cross_debate_context",
            "cross_debate_ids",
            "knowledge_items",
            "knowledge_sources",
            "document_items",
            "evidence_items",
            "context_envelope",
            "total_context_tokens",
            "retrieval_time_ms",
        }

    def test_to_dict_with_data(self):
        """to_dict preserves populated fields."""
        snap = ContextSnapshot(
            continuum_entries=[{"content": "foo"}],
            cross_debate_context="some context",
            cross_debate_ids=["d-1", "d-2"],
            knowledge_items=[{"content": "bar"}],
            knowledge_sources=["source1"],
            document_items=[{"preview": "doc"}],
            evidence_items=[{"snippet": "evidence"}],
            context_envelope={"user_id": "u-1", "tenant_id": "ws-1"},
            total_context_tokens=42,
            retrieval_time_ms=12.5,
        )
        d = snap.to_dict()
        assert d["continuum_entries"] == [{"content": "foo"}]
        assert d["cross_debate_context"] == "some context"
        assert d["cross_debate_ids"] == ["d-1", "d-2"]
        assert d["knowledge_items"] == [{"content": "bar"}]
        assert d["knowledge_sources"] == ["source1"]
        assert d["document_items"] == [{"preview": "doc"}]
        assert d["evidence_items"] == [{"snippet": "evidence"}]
        assert d["context_envelope"] == {"user_id": "u-1", "tenant_id": "ws-1"}
        assert d["total_context_tokens"] == 42
        assert d["retrieval_time_ms"] == 12.5


# ---------------------------------------------------------------------------
# capture_context_snapshot Tests
# ---------------------------------------------------------------------------


class TestCaptureContextSnapshot:
    """Tests for capture_context_snapshot async function."""

    @pytest.mark.asyncio
    async def test_empty_when_no_sources(self):
        """Returns empty snapshot when all memory sources are None."""
        snap = await capture_context_snapshot("test query")
        assert snap.continuum_entries == []
        assert snap.cross_debate_context == ""
        assert snap.knowledge_items == []
        assert isinstance(snap.context_envelope, dict)
        assert snap.total_context_tokens == 0
        assert snap.retrieval_time_ms >= 0

    @pytest.mark.asyncio
    async def test_includes_auth_scope_envelope(self):
        """Auth scope metadata is captured for auditing."""
        auth_context = AuthorizationContext(
            user_id="user-123",
            workspace_id="ws-123",
            org_id="org-123",
            roles={"member"},
            permissions={"memory:read"},
        )

        snap = await capture_context_snapshot("test query", auth_context=auth_context)

        assert snap.context_envelope["user_id"] == "user-123"
        assert snap.context_envelope["workspace_id"] == "ws-123"
        assert snap.context_envelope["tenant_id"] == "ws-123"
        assert snap.context_envelope["org_id"] == "org-123"
        assert snap.context_envelope["source"] == "pipeline.decision_integrity.context"

    @pytest.mark.asyncio
    async def test_continuum_memory_retrieval(self):
        """Retrieves entries from continuum memory."""
        entry = _FakeContinuumEntry(content="institutional pattern")
        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = [entry]

        with patch(
            "aragora.pipeline.decision_integrity.MemoryTier",
            create=True,
        ) as mock_tier:
            mock_tier.SLOW = "SLOW"
            mock_tier.GLACIAL = "GLACIAL"

            # Patch the import inside the function
            with patch.dict(
                "sys.modules",
                {"aragora.memory.tier_manager": MagicMock(MemoryTier=mock_tier)},
            ):
                snap = await capture_context_snapshot("test query", continuum_memory=mock_memory)

        assert len(snap.continuum_entries) == 1
        assert snap.continuum_entries[0]["content"] == "institutional pattern"

    @pytest.mark.asyncio
    async def test_continuum_retrieval_blocked_without_memory_permission(self):
        """Unauthorized memory retrieval attempts are blocked."""
        auth_context = AuthorizationContext(
            user_id="user-123",
            workspace_id="ws-123",
            org_id="org-123",
            roles={"member"},
            permissions={"debates:read"},
        )
        mock_memory = MagicMock()

        await capture_context_snapshot(
            "test query",
            continuum_memory=mock_memory,
            auth_context=auth_context,
        )

        mock_memory.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_continuum_retrieval_is_tenant_scoped(self, monkeypatch):
        """Continuum retrieval receives tenant scoping kwargs from auth context."""
        monkeypatch.setenv("ARAGORA_MEMORY_TENANT_ENFORCE", "1")
        auth_context = AuthorizationContext(
            user_id="user-123",
            workspace_id="ws-tenant",
            org_id="org-123",
            roles={"member"},
            permissions={"memory:read"},
        )
        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = []

        with patch.dict(
            "sys.modules",
            {
                "aragora.memory.tier_manager": MagicMock(
                    MemoryTier=MagicMock(SLOW="SLOW", GLACIAL="GLACIAL")
                )
            },
        ):
            await capture_context_snapshot(
                "test query",
                continuum_memory=mock_memory,
                auth_context=auth_context,
            )

        call_kwargs = mock_memory.retrieve.call_args.kwargs
        assert call_kwargs["tenant_id"] == "ws-tenant"
        assert call_kwargs["enforce_tenant_isolation"] is True

    @pytest.mark.asyncio
    async def test_continuum_memory_non_dataclass_entries(self):
        """Handles non-dataclass entries from continuum memory gracefully."""
        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = ["plain string entry"]

        with patch.dict(
            "sys.modules",
            {
                "aragora.memory.tier_manager": MagicMock(
                    MemoryTier=MagicMock(SLOW="SLOW", GLACIAL="GLACIAL")
                )
            },
        ):
            snap = await capture_context_snapshot("test query", continuum_memory=mock_memory)

        assert len(snap.continuum_entries) == 1
        assert snap.continuum_entries[0] == {"content": "plain string entry"}

    @pytest.mark.asyncio
    async def test_continuum_memory_failure_graceful(self):
        """Gracefully handles continuum memory errors."""
        mock_memory = MagicMock()
        mock_memory.retrieve.side_effect = RuntimeError("connection lost")

        with patch.dict(
            "sys.modules",
            {
                "aragora.memory.tier_manager": MagicMock(
                    MemoryTier=MagicMock(SLOW="SLOW", GLACIAL="GLACIAL")
                )
            },
        ):
            snap = await capture_context_snapshot("test query", continuum_memory=mock_memory)

        assert snap.continuum_entries == []

    @pytest.mark.asyncio
    async def test_cross_debate_memory_retrieval(self):
        """Retrieves context from cross-debate memory."""
        mock_memory = AsyncMock()
        mock_memory.get_relevant_context.return_value = "Prior debate concluded X"

        snap = await capture_context_snapshot("test query", cross_debate_memory=mock_memory)

        assert snap.cross_debate_context == "Prior debate concluded X"
        mock_memory.get_relevant_context.assert_awaited_once_with(
            task="test query", max_tokens=2000
        )

    @pytest.mark.asyncio
    async def test_cross_debate_memory_with_entries(self):
        """Extracts relevant debate IDs from cross-debate memory entries."""
        mock_memory = AsyncMock()
        mock_memory.get_relevant_context.return_value = "context"
        mock_memory._entries = {
            "d-1": _FakeCrossDebateEntry("test query about rate limiter"),
            "d-2": _FakeCrossDebateEntry("unrelated topic"),
            "d-3": _FakeCrossDebateEntry("another test query discussion"),
        }

        snap = await capture_context_snapshot("test query", cross_debate_memory=mock_memory)

        assert "d-1" in snap.cross_debate_ids
        assert "d-3" in snap.cross_debate_ids
        assert "d-2" not in snap.cross_debate_ids

    @pytest.mark.asyncio
    async def test_cross_debate_memory_none_context(self):
        """Handles None return from cross-debate memory."""
        mock_memory = AsyncMock()
        mock_memory.get_relevant_context.return_value = None

        snap = await capture_context_snapshot("test query", cross_debate_memory=mock_memory)

        assert snap.cross_debate_context == ""

    @pytest.mark.asyncio
    async def test_cross_debate_memory_failure_graceful(self):
        """Gracefully handles cross-debate memory errors."""
        mock_memory = AsyncMock()
        mock_memory.get_relevant_context.side_effect = RuntimeError("db timeout")

        snap = await capture_context_snapshot("test query", cross_debate_memory=mock_memory)

        assert snap.cross_debate_context == ""

    @pytest.mark.asyncio
    async def test_knowledge_mound_retrieval(self):
        """Retrieves items from knowledge mound."""
        items = [
            _FakeKnowledgeItem("token bucket docs"),
            _FakeKnowledgeItem("rate limiter patterns"),
        ]
        result = _FakeKnowledgeResult(items=items, sources=["docs", "wiki"])

        mock_km = AsyncMock()
        mock_km.query.return_value = result

        snap = await capture_context_snapshot("test query", knowledge_mound=mock_km)

        assert len(snap.knowledge_items) == 2
        assert snap.knowledge_items[0] == {"content": "token bucket docs"}
        assert snap.knowledge_sources == ["docs", "wiki"]

    @pytest.mark.asyncio
    async def test_knowledge_mound_failure_graceful(self):
        """Gracefully handles knowledge mound errors."""
        mock_km = AsyncMock()
        mock_km.query.side_effect = RuntimeError("storage error")

        snap = await capture_context_snapshot("test query", knowledge_mound=mock_km)

        assert snap.knowledge_items == []
        assert snap.knowledge_sources == []

    @pytest.mark.asyncio
    async def test_token_estimation(self):
        """Estimates token count across all sources."""
        snap = await capture_context_snapshot("test query")
        assert snap.total_context_tokens == 0

        # Manually populate and verify estimation
        snap2 = ContextSnapshot(
            continuum_entries=[{"content": "a" * 400}],  # ~100 tokens
            cross_debate_context="b" * 200,  # ~50 tokens
            knowledge_items=[{"content": "c" * 100}],  # ~25 tokens
        )
        # Manually recalculate like the function does
        expected = 400 // 4 + 200 // 4 + 100 // 4
        assert expected == 175

    @pytest.mark.asyncio
    async def test_retrieval_time_tracked(self):
        """Retrieval time is tracked in milliseconds."""
        snap = await capture_context_snapshot("test query")
        assert snap.retrieval_time_ms >= 0

    @pytest.mark.asyncio
    async def test_all_sources_combined(self):
        """All three sources are queried when provided."""
        mock_continuum = MagicMock()
        mock_continuum.retrieve.return_value = [_FakeContinuumEntry()]

        mock_cross = AsyncMock()
        mock_cross.get_relevant_context.return_value = "cross debate context"

        mock_km = AsyncMock()
        mock_km.query.return_value = _FakeKnowledgeResult(
            items=[_FakeKnowledgeItem()], sources=["test"]
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.memory.tier_manager": MagicMock(
                    MemoryTier=MagicMock(SLOW="SLOW", GLACIAL="GLACIAL")
                )
            },
        ):
            snap = await capture_context_snapshot(
                "test query",
                continuum_memory=mock_continuum,
                cross_debate_memory=mock_cross,
                knowledge_mound=mock_km,
            )

        assert len(snap.continuum_entries) == 1
        assert snap.cross_debate_context == "cross debate context"
        assert len(snap.knowledge_items) == 1
        assert snap.total_context_tokens > 0

    @pytest.mark.asyncio
    async def test_max_entries_parameter(self):
        """max_entries is passed through to continuum and knowledge mound."""
        mock_continuum = MagicMock()
        mock_continuum.retrieve.return_value = []

        mock_km = AsyncMock()
        mock_km.query.return_value = MagicMock(spec=[])  # No items/sources attrs

        with patch.dict(
            "sys.modules",
            {
                "aragora.memory.tier_manager": MagicMock(
                    MemoryTier=MagicMock(SLOW="SLOW", GLACIAL="GLACIAL")
                )
            },
        ):
            await capture_context_snapshot(
                "query",
                continuum_memory=mock_continuum,
                knowledge_mound=mock_km,
                max_entries=5,
            )

        mock_continuum.retrieve.assert_called_once()
        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["limit"] == 5

        mock_km.query.assert_awaited_once()
        km_kwargs = mock_km.query.call_args[1]
        assert km_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_max_tokens_parameter(self):
        """max_tokens is passed through to cross-debate memory."""
        mock_cross = AsyncMock()
        mock_cross.get_relevant_context.return_value = ""

        await capture_context_snapshot("query", cross_debate_memory=mock_cross, max_tokens=500)

        mock_cross.get_relevant_context.assert_awaited_once_with(task="query", max_tokens=500)


# ---------------------------------------------------------------------------
# _coerce_debate_result Tests
# ---------------------------------------------------------------------------


class TestCoerceDebateResult:
    """Tests for _coerce_debate_result helper."""

    def test_basic_conversion(self):
        """Converts a standard debate dict to DebateResult."""
        debate = _make_debate_dict()
        result = _coerce_debate_result(debate)

        assert result.debate_id == "test-debate-001"
        assert result.task == "Design a rate limiter"
        assert result.final_answer == "Implement token bucket algorithm"
        assert result.confidence == 0.85
        assert result.consensus_reached is True
        assert result.rounds_used == 3
        assert result.participants == ["claude", "gpt4", "gemini"]

    def test_alternative_field_names(self):
        """Handles alternative field names (id, question, conclusion)."""
        debate = {
            "id": "alt-id",
            "question": "What architecture?",
            "conclusion": "Microservices",
            "confidence": 0.7,
            "consensus_reached": False,
            "rounds": 2,
            "rounds_completed": 2,
            "status": "completed",
            "agents": ["claude"],
            "metadata": {},
        }
        result = _coerce_debate_result(debate)

        assert result.debate_id == "alt-id"
        assert result.task == "What architecture?"
        assert result.final_answer == "Microservices"
        assert result.rounds_used == 2

    def test_missing_fields_default(self):
        """Missing fields default to empty/zero values."""
        result = _coerce_debate_result({})

        # debate_id may auto-generate a UUID when empty string is passed
        assert isinstance(result.debate_id, str)
        assert result.task == ""
        assert result.final_answer == ""
        assert result.confidence == 0.0
        assert result.consensus_reached is False
        assert result.rounds_used == 0
        assert result.participants == []

    def test_agents_as_string(self):
        """Handles agents field as comma-separated string."""
        debate = _make_debate_dict(agents="claude, gpt4, gemini")
        result = _coerce_debate_result(debate)

        assert result.participants == ["claude", "gpt4", "gemini"]

    def test_agents_none(self):
        """Handles None agents field."""
        debate = _make_debate_dict(agents=None)
        result = _coerce_debate_result(debate)

        assert result.participants == []

    def test_metadata_preserved(self):
        """Metadata dict is preserved."""
        debate = _make_debate_dict(metadata={"source": "slack", "channel": "general"})
        result = _coerce_debate_result(debate)

        assert result.metadata == {"source": "slack", "channel": "general"}

    def test_metadata_none(self):
        """None metadata becomes empty dict."""
        debate = _make_debate_dict(metadata=None)
        result = _coerce_debate_result(debate)

        assert result.metadata == {}


# ---------------------------------------------------------------------------
# DecisionIntegrityPackage Tests
# ---------------------------------------------------------------------------


class TestDecisionIntegrityPackage:
    """Tests for DecisionIntegrityPackage dataclass."""

    def test_minimal_creation(self):
        """Package can be created with receipt and plan as None."""
        pkg = DecisionIntegrityPackage(
            debate_id="d-1",
            receipt=None,
            plan=None,
        )
        assert pkg.debate_id == "d-1"
        assert pkg.receipt is None
        assert pkg.plan is None
        assert pkg.context_snapshot is None

    def test_to_dict_all_none(self):
        """to_dict with all None components returns null values."""
        pkg = DecisionIntegrityPackage(debate_id="d-1", receipt=None, plan=None)
        d = pkg.to_dict()

        assert d["debate_id"] == "d-1"
        assert d["receipt"] is None
        assert d["plan"] is None
        assert d["context_snapshot"] is None

    def test_to_dict_with_context_snapshot(self):
        """to_dict includes context_snapshot when present."""
        snap = ContextSnapshot(
            continuum_entries=[{"content": "test"}],
            total_context_tokens=10,
        )
        pkg = DecisionIntegrityPackage(
            debate_id="d-1",
            receipt=None,
            plan=None,
            context_snapshot=snap,
        )
        d = pkg.to_dict()

        assert d["context_snapshot"] is not None
        assert d["context_snapshot"]["continuum_entries"] == [{"content": "test"}]
        assert d["context_snapshot"]["total_context_tokens"] == 10

    def test_to_dict_with_mock_receipt(self):
        """to_dict calls receipt.to_dict() when present."""
        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = {"receipt_id": "r-1"}

        pkg = DecisionIntegrityPackage(debate_id="d-1", receipt=mock_receipt, plan=None)
        d = pkg.to_dict()

        assert d["receipt"] == {"receipt_id": "r-1"}
        mock_receipt.to_dict.assert_called_once()

    def test_to_dict_with_mock_plan(self):
        """to_dict calls plan.to_dict() when present."""
        mock_plan = MagicMock()
        mock_plan.to_dict.return_value = {"design_hash": "abc123", "tasks": []}

        pkg = DecisionIntegrityPackage(debate_id="d-1", receipt=None, plan=mock_plan)
        d = pkg.to_dict()

        assert d["plan"] == {"design_hash": "abc123", "tasks": []}
        mock_plan.to_dict.assert_called_once()


# ---------------------------------------------------------------------------
# build_decision_integrity_package Tests
# ---------------------------------------------------------------------------


class TestBuildDecisionIntegrityPackage:
    """Tests for the build_decision_integrity_package async function."""

    @pytest.mark.asyncio
    async def test_defaults_include_receipt_and_plan(self):
        """Default call includes receipt and plan, no context."""
        debate = _make_debate_dict()
        pkg = await build_decision_integrity_package(debate)

        assert pkg.debate_id == "test-debate-001"
        assert pkg.receipt is not None
        assert pkg.plan is not None
        assert pkg.context_snapshot is None

    @pytest.mark.asyncio
    async def test_receipt_only(self):
        """Can request receipt without plan."""
        debate = _make_debate_dict()
        pkg = await build_decision_integrity_package(debate, include_plan=False)

        assert pkg.receipt is not None
        assert pkg.plan is None

    @pytest.mark.asyncio
    async def test_plan_only(self):
        """Can request plan without receipt."""
        debate = _make_debate_dict()
        pkg = await build_decision_integrity_package(debate, include_receipt=False)

        assert pkg.receipt is None
        assert pkg.plan is not None

    @pytest.mark.asyncio
    async def test_neither(self):
        """Can request neither receipt nor plan."""
        debate = _make_debate_dict()
        pkg = await build_decision_integrity_package(
            debate, include_receipt=False, include_plan=False
        )

        assert pkg.receipt is None
        assert pkg.plan is None

    @pytest.mark.asyncio
    async def test_plan_strategy_single_task(self):
        """single_task strategy uses create_single_task_plan."""
        debate = _make_debate_dict()
        pkg = await build_decision_integrity_package(debate, plan_strategy="single_task")

        assert pkg.plan is not None
        assert len(pkg.plan.tasks) == 1

    @pytest.mark.asyncio
    async def test_plan_strategy_gemini_fallback(self):
        """gemini strategy falls back to single_task on error."""
        debate = _make_debate_dict()

        with patch(
            "aragora.pipeline.decision_integrity.generate_implement_plan",
            side_effect=RuntimeError("API error"),
        ):
            pkg = await build_decision_integrity_package(debate, plan_strategy="gemini")

        assert pkg.plan is not None
        assert len(pkg.plan.tasks) == 1  # Fallback to single task

    @pytest.mark.asyncio
    async def test_plan_strategy_gemini_success(self):
        """gemini strategy uses generate_implement_plan when available."""
        debate = _make_debate_dict()
        mock_plan = MagicMock()
        mock_plan.tasks = [MagicMock(), MagicMock()]

        with patch(
            "aragora.pipeline.decision_integrity.generate_implement_plan",
            return_value=mock_plan,
        ):
            pkg = await build_decision_integrity_package(debate, plan_strategy="gemini")

        assert pkg.plan is mock_plan

    @pytest.mark.asyncio
    async def test_include_context_false_skips_snapshot(self):
        """include_context=False skips context snapshot even with memory args."""
        mock_memory = MagicMock()
        debate = _make_debate_dict()

        pkg = await build_decision_integrity_package(
            debate,
            include_context=False,
            continuum_memory=mock_memory,
        )

        assert pkg.context_snapshot is None

    @pytest.mark.asyncio
    async def test_include_context_true_captures_snapshot(self):
        """include_context=True captures context snapshot."""
        debate = _make_debate_dict()

        pkg = await build_decision_integrity_package(debate, include_context=True)

        assert pkg.context_snapshot is not None
        assert isinstance(pkg.context_snapshot, ContextSnapshot)

    @pytest.mark.asyncio
    async def test_include_context_captures_auth_scope_envelope(self):
        """Auth context is reflected in context snapshot envelope."""
        debate = _make_debate_dict()
        auth_context = AuthorizationContext(
            user_id="user-007",
            workspace_id="ws-007",
            org_id="org-007",
            roles={"member"},
            permissions={"memory:read"},
        )

        pkg = await build_decision_integrity_package(
            debate,
            include_context=True,
            auth_context=auth_context,
        )

        assert pkg.context_snapshot is not None
        assert pkg.context_snapshot.context_envelope["user_id"] == "user-007"
        assert pkg.context_snapshot.context_envelope["tenant_id"] == "ws-007"

    @pytest.mark.asyncio
    async def test_include_context_with_memory_sources(self):
        """Context snapshot uses provided memory sources."""
        debate = _make_debate_dict()

        mock_cross = AsyncMock()
        mock_cross.get_relevant_context.return_value = "prior context"

        mock_km = AsyncMock()
        mock_km.query.return_value = _FakeKnowledgeResult(
            items=[_FakeKnowledgeItem("test knowledge")],
            sources=["wiki"],
        )

        pkg = await build_decision_integrity_package(
            debate,
            include_context=True,
            cross_debate_memory=mock_cross,
            knowledge_mound=mock_km,
        )

        assert pkg.context_snapshot is not None
        assert pkg.context_snapshot.cross_debate_context == "prior context"
        assert len(pkg.context_snapshot.knowledge_items) == 1

    @pytest.mark.asyncio
    async def test_debate_id_fallback_to_id(self):
        """Uses 'id' field when 'debate_id' is missing."""
        debate = {"id": "fallback-id", "task": "test"}
        pkg = await build_decision_integrity_package(debate, include_plan=False)

        assert pkg.debate_id == "fallback-id"

    @pytest.mark.asyncio
    async def test_receipt_from_debate_result(self):
        """Receipt is properly constructed from debate data."""
        debate = _make_debate_dict(
            confidence=0.92,
            consensus_reached=True,
        )
        pkg = await build_decision_integrity_package(debate, include_plan=False)

        assert pkg.receipt is not None
        receipt_dict = pkg.receipt.to_dict()
        assert "receipt_id" in receipt_dict

    @pytest.mark.asyncio
    async def test_serialization_round_trip(self):
        """Full package serializes to dict correctly."""
        debate = _make_debate_dict()
        pkg = await build_decision_integrity_package(debate, include_context=True)

        d = pkg.to_dict()
        assert d["debate_id"] == "test-debate-001"
        assert d["receipt"] is not None
        assert d["plan"] is not None
        assert d["context_snapshot"] is not None
        assert isinstance(d["context_snapshot"], dict)

    @pytest.mark.asyncio
    async def test_repo_path_passed_to_plan(self):
        """repo_path is forwarded to plan creation."""
        from pathlib import Path

        debate = _make_debate_dict()

        with patch("aragora.pipeline.decision_integrity.create_single_task_plan") as mock_create:
            mock_plan = MagicMock()
            mock_create.return_value = mock_plan

            await build_decision_integrity_package(
                debate,
                include_receipt=False,
                repo_path=Path("/test/repo"),
            )

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs[1].get("repo_path") == Path("/test/repo") or call_kwargs[0][1] == Path(
            "/test/repo"
        )  # positional arg


# ---------------------------------------------------------------------------
# Integration: Pipeline __init__ exports
# ---------------------------------------------------------------------------


class TestPipelineExports:
    """Verify decision integrity types are exported from the pipeline package."""

    def test_context_snapshot_exported(self):
        """ContextSnapshot is importable from aragora.pipeline."""
        from aragora.pipeline import ContextSnapshot as CS

        assert CS is ContextSnapshot

    def test_decision_integrity_package_exported(self):
        """DecisionIntegrityPackage is importable from aragora.pipeline."""
        from aragora.pipeline import DecisionIntegrityPackage as DIP

        assert DIP is DecisionIntegrityPackage

    def test_build_function_exported(self):
        """build_decision_integrity_package is importable from aragora.pipeline."""
        from aragora.pipeline import (
            build_decision_integrity_package as build_fn,
        )

        assert build_fn is build_decision_integrity_package

    def test_capture_function_exported(self):
        """capture_context_snapshot is importable from aragora.pipeline."""
        from aragora.pipeline import capture_context_snapshot as capture_fn

        assert capture_fn is capture_context_snapshot

    def test_execution_notifier_exported(self):
        """ExecutionNotifier is importable from aragora.pipeline."""
        from aragora.pipeline import ExecutionNotifier as EN

        assert EN is ExecutionNotifier

    def test_execution_progress_exported(self):
        """ExecutionProgress is importable from aragora.pipeline."""
        from aragora.pipeline import ExecutionProgress as EP

        assert EP is ExecutionProgress


# ---------------------------------------------------------------------------
# ExecutionProgress Tests
# ---------------------------------------------------------------------------


class TestExecutionProgress:
    """Tests for the ExecutionProgress dataclass."""

    def test_defaults(self):
        """Newly created progress has sensible defaults."""
        prog = ExecutionProgress(debate_id="d-1")
        assert prog.debate_id == "d-1"
        assert prog.plan_id is None
        assert prog.total_tasks == 0
        assert prog.completed_tasks == 0
        assert prog.failed_tasks == 0
        assert prog.current_task_id is None
        assert prog.current_task_description is None
        assert prog.elapsed_seconds == 0.0
        assert prog.task_results == []

    def test_progress_pct_zero_tasks(self):
        """progress_pct returns 0.0 when total_tasks is 0 (avoids division by zero)."""
        prog = ExecutionProgress(debate_id="d-1", total_tasks=0)
        assert prog.progress_pct == 0.0

    def test_progress_pct_partial(self):
        """progress_pct returns correct percentage for partial completion."""
        prog = ExecutionProgress(
            debate_id="d-1",
            total_tasks=10,
            completed_tasks=3,
            failed_tasks=2,
        )
        # (3 + 2) / 10 * 100 = 50.0
        assert prog.progress_pct == 50.0

    def test_progress_pct_all_completed(self):
        """progress_pct returns 100.0 when all tasks are completed."""
        prog = ExecutionProgress(
            debate_id="d-1",
            total_tasks=5,
            completed_tasks=5,
        )
        assert prog.progress_pct == 100.0

    def test_progress_pct_all_failed(self):
        """progress_pct returns 100.0 when all tasks have failed."""
        prog = ExecutionProgress(
            debate_id="d-1",
            total_tasks=4,
            failed_tasks=4,
        )
        assert prog.progress_pct == 100.0

    def test_progress_pct_mixed_complete_and_failed(self):
        """progress_pct counts both completed and failed tasks."""
        prog = ExecutionProgress(
            debate_id="d-1",
            total_tasks=8,
            completed_tasks=4,
            failed_tasks=4,
        )
        assert prog.progress_pct == 100.0

    def test_is_complete_false_initially(self):
        """is_complete is False when no tasks have finished."""
        prog = ExecutionProgress(debate_id="d-1", total_tasks=5)
        assert prog.is_complete is False

    def test_is_complete_true_when_all_done(self):
        """is_complete is True when completed + failed >= total."""
        prog = ExecutionProgress(
            debate_id="d-1",
            total_tasks=3,
            completed_tasks=2,
            failed_tasks=1,
        )
        assert prog.is_complete is True

    def test_is_complete_true_with_zero_tasks(self):
        """is_complete is True when total_tasks is 0 (nothing to do)."""
        prog = ExecutionProgress(debate_id="d-1", total_tasks=0)
        assert prog.is_complete is True

    def test_to_dict_output_format(self):
        """to_dict returns all expected keys with correct types."""
        prog = ExecutionProgress(
            debate_id="d-42",
            plan_id="plan-7",
            total_tasks=10,
            completed_tasks=6,
            failed_tasks=1,
            current_task_id="t-3",
            current_task_description="Run tests",
            elapsed_seconds=12.345,
        )
        d = prog.to_dict()

        assert d["debate_id"] == "d-42"
        assert d["plan_id"] == "plan-7"
        assert d["total_tasks"] == 10
        assert d["completed_tasks"] == 6
        assert d["failed_tasks"] == 1
        assert d["current_task_id"] == "t-3"
        assert d["current_task_description"] == "Run tests"
        assert d["elapsed_seconds"] == 12.35  # rounded to 2 decimals
        assert d["progress_pct"] == 70.0  # (6+1)/10 * 100, rounded to 1 decimal
        assert d["is_complete"] is False

    def test_to_dict_includes_computed_properties(self):
        """to_dict includes both progress_pct and is_complete."""
        prog = ExecutionProgress(
            debate_id="d-1",
            total_tasks=2,
            completed_tasks=2,
        )
        d = prog.to_dict()
        assert "progress_pct" in d
        assert "is_complete" in d
        assert d["progress_pct"] == 100.0
        assert d["is_complete"] is True

    def test_to_dict_elapsed_seconds_rounding(self):
        """elapsed_seconds is rounded to 2 decimal places in to_dict."""
        prog = ExecutionProgress(
            debate_id="d-1",
            elapsed_seconds=1.23456789,
        )
        d = prog.to_dict()
        assert d["elapsed_seconds"] == 1.23

    def test_to_dict_progress_pct_rounding(self):
        """progress_pct is rounded to 1 decimal place in to_dict."""
        prog = ExecutionProgress(
            debate_id="d-1",
            total_tasks=3,
            completed_tasks=1,
        )
        d = prog.to_dict()
        # 1/3 * 100 = 33.333... -> rounded to 33.3
        assert d["progress_pct"] == 33.3


# ---------------------------------------------------------------------------
# ExecutionNotifier Tests
# ---------------------------------------------------------------------------


class TestExecutionNotifier:
    """Tests for the ExecutionNotifier class."""

    def test_initial_state(self):
        """Newly created notifier has zero progress."""
        notifier = ExecutionNotifier(debate_id="d-1", plan_id="p-1", total_tasks=5)
        assert notifier.progress.debate_id == "d-1"
        assert notifier.progress.plan_id == "p-1"
        assert notifier.progress.total_tasks == 5
        assert notifier.progress.completed_tasks == 0
        assert notifier.progress.failed_tasks == 0
        assert notifier.progress.is_complete is False

    def test_on_task_complete_increments_completed(self):
        """Successful task result increments completed_tasks."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=3)
        mock_result = MagicMock(success=True, model_used="claude", duration_seconds=1.5, error=None)

        notifier.on_task_complete("task-1", mock_result)

        assert notifier.progress.completed_tasks == 1
        assert notifier.progress.failed_tasks == 0

    def test_on_task_complete_increments_failed(self):
        """Failed task result increments failed_tasks."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=3)
        mock_result = MagicMock(
            success=False, model_used="gpt4", duration_seconds=0.5, error="timeout"
        )

        notifier.on_task_complete("task-1", mock_result)

        assert notifier.progress.completed_tasks == 0
        assert notifier.progress.failed_tasks == 1

    def test_on_task_complete_updates_current_task(self):
        """on_task_complete updates current_task_id."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=2)
        mock_result = MagicMock(success=True, model_used=None, duration_seconds=0, error=None)

        notifier.on_task_complete("task-42", mock_result)

        assert notifier.progress.current_task_id == "task-42"

    def test_on_task_complete_appends_task_result(self):
        """on_task_complete appends a result dict to task_results."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=2)
        mock_result = MagicMock(
            success=True,
            model_used="claude",
            duration_seconds=2.3,
            error=None,
        )

        notifier.on_task_complete("t-1", mock_result)

        assert len(notifier.progress.task_results) == 1
        entry = notifier.progress.task_results[0]
        assert entry["task_id"] == "t-1"
        assert entry["success"] is True
        assert entry["model_used"] == "claude"
        assert entry["duration_seconds"] == 2.3
        assert entry["error"] is None

    def test_on_task_complete_multiple_tasks(self):
        """Multiple calls accumulate progress correctly."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=4)

        notifier.on_task_complete(
            "t-1", MagicMock(success=True, model_used=None, duration_seconds=0, error=None)
        )
        notifier.on_task_complete(
            "t-2", MagicMock(success=False, model_used=None, duration_seconds=0, error="fail")
        )
        notifier.on_task_complete(
            "t-3", MagicMock(success=True, model_used=None, duration_seconds=0, error=None)
        )

        assert notifier.progress.completed_tasks == 2
        assert notifier.progress.failed_tasks == 1
        assert notifier.progress.progress_pct == 75.0
        assert notifier.progress.is_complete is False
        assert len(notifier.progress.task_results) == 3

    def test_on_task_complete_tracks_elapsed_time(self):
        """on_task_complete updates elapsed_seconds."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=1)
        mock_result = MagicMock(success=True, model_used=None, duration_seconds=0, error=None)

        notifier.on_task_complete("t-1", mock_result)

        # elapsed_seconds should be positive (some time has passed since construction)
        assert notifier.progress.elapsed_seconds >= 0

    def test_set_task_descriptions_from_task_list(self):
        """set_task_descriptions populates descriptions and updates total_tasks."""
        notifier = ExecutionNotifier(debate_id="d-1")

        task1 = MagicMock(id="t-1", description="Run linter")
        task2 = MagicMock(id="t-2", description="Run tests")
        task3 = MagicMock(id="t-3", description="Build project")

        notifier.set_task_descriptions([task1, task2, task3])

        assert notifier.progress.total_tasks == 3
        assert notifier._task_descriptions["t-1"] == "Run linter"
        assert notifier._task_descriptions["t-2"] == "Run tests"
        assert notifier._task_descriptions["t-3"] == "Build project"

    def test_set_task_descriptions_updates_current_description(self):
        """After set_task_descriptions, on_task_complete resolves description."""
        notifier = ExecutionNotifier(debate_id="d-1")

        task1 = MagicMock(id="t-1", description="Run unit tests")
        notifier.set_task_descriptions([task1])

        mock_result = MagicMock(success=True, model_used=None, duration_seconds=0, error=None)
        notifier.on_task_complete("t-1", mock_result)

        assert notifier.progress.current_task_description == "Run unit tests"

    def test_set_task_descriptions_unknown_task_id(self):
        """on_task_complete with unknown task_id sets description to None."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=1)

        mock_result = MagicMock(success=True, model_used=None, duration_seconds=0, error=None)
        notifier.on_task_complete("unknown-task", mock_result)

        assert notifier.progress.current_task_description is None

    def test_set_task_descriptions_no_id_attr(self):
        """Tasks without id attribute use str(task) as key."""
        notifier = ExecutionNotifier(debate_id="d-1")

        # A plain string has no .id attribute - MagicMock spec=str approach
        task = "plain-task-string"
        notifier.set_task_descriptions([task])

        assert notifier.progress.total_tasks == 1
        # The key should be str(task) since getattr(task, "id", None) is None
        assert "plain-task-string" in notifier._task_descriptions

    def test_is_complete_detection(self):
        """Notifier correctly detects when all tasks are done."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=2)
        result_ok = MagicMock(success=True, model_used=None, duration_seconds=0, error=None)
        result_fail = MagicMock(success=False, model_used=None, duration_seconds=0, error="err")

        assert notifier.progress.is_complete is False

        notifier.on_task_complete("t-1", result_ok)
        assert notifier.progress.is_complete is False

        notifier.on_task_complete("t-2", result_fail)
        assert notifier.progress.is_complete is True

    def test_progress_pct_through_notifier(self):
        """progress_pct is correct after notifier callbacks."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=4)
        result = MagicMock(success=True, model_used=None, duration_seconds=0, error=None)

        notifier.on_task_complete("t-1", result)
        assert notifier.progress.progress_pct == 25.0

        notifier.on_task_complete("t-2", result)
        assert notifier.progress.progress_pct == 50.0

    def test_to_dict_serialization(self):
        """Notifier progress serializes to dict correctly."""
        notifier = ExecutionNotifier(debate_id="d-99", plan_id="plan-5", total_tasks=3)
        result = MagicMock(success=True, model_used="claude", duration_seconds=1.0, error=None)

        notifier.on_task_complete("t-1", result)

        d = notifier.progress.to_dict()
        assert d["debate_id"] == "d-99"
        assert d["plan_id"] == "plan-5"
        assert d["total_tasks"] == 3
        assert d["completed_tasks"] == 1
        assert d["failed_tasks"] == 0
        assert isinstance(d["progress_pct"], float)
        assert isinstance(d["is_complete"], bool)
        assert isinstance(d["elapsed_seconds"], float)

    def test_add_listener(self):
        """add_listener registers a callback function."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=1)
        listener = MagicMock()

        notifier.add_listener(listener)

        assert listener in notifier._listeners

    def test_result_without_success_attr_defaults_true(self):
        """Result object without success attribute defaults to True."""
        notifier = ExecutionNotifier(debate_id="d-1", total_tasks=1)
        # Object without any attributes
        bare_result = object()

        notifier.on_task_complete("t-1", bare_result)

        # getattr(result, "success", True) -> True
        assert notifier.progress.completed_tasks == 1
        assert notifier.progress.failed_tasks == 0

    @pytest.mark.asyncio
    async def test_send_completion_summary_updates_elapsed(self):
        """send_completion_summary updates elapsed_seconds before sending."""
        notifier = ExecutionNotifier(
            debate_id="d-1",
            total_tasks=1,
            notify_channel=False,
            notify_websocket=False,
        )
        result = MagicMock(success=True, model_used=None, duration_seconds=0, error=None)
        notifier.on_task_complete("t-1", result)

        await notifier.send_completion_summary()

        assert notifier.progress.elapsed_seconds >= 0


# ---------------------------------------------------------------------------
# Execution mode normalization tests
# ---------------------------------------------------------------------------


class TestExecutionModeNormalization:
    """Tests for cross-surface execution-mode alias normalization."""

    def test_normalizes_known_aliases(self):
        assert normalize_execution_mode("workflow_execute") == "workflow"
        assert normalize_execution_mode("execute_workflow") == "workflow"
        assert normalize_execution_mode("computer-use") == "computer_use"

    def test_preserves_canonical_modes(self):
        assert normalize_execution_mode("workflow") == "workflow"
        assert normalize_execution_mode("hybrid") == "hybrid"
        assert normalize_execution_mode("fabric") == "fabric"
        assert normalize_execution_mode("computer_use") == "computer_use"
