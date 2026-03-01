"""Tests for aragora.debate.knowledge_mound_ops module.

Covers KnowledgeMoundOperations:
- Initialization (defaults, custom values)
- fetch_knowledge_context (retrieval, formatting, auth, retry, metrics, errors)
- ingest_debate_outcome (ingestion, thresholds, metadata, notify, metrics, errors)
"""

from __future__ import annotations

import enum
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mound(**overrides) -> AsyncMock:
    """Create a mock KnowledgeMound with sensible defaults."""
    mound = AsyncMock()
    mound.workspace_id = "ws-1"
    mound.query_semantic = AsyncMock(return_value=[])
    mound.query_with_visibility = AsyncMock(return_value=[])
    mound.store = AsyncMock(return_value=None)
    mound.initialize = AsyncMock()
    for k, v in overrides.items():
        setattr(mound, k, v)
    return mound


def _make_item(
    *,
    source: str = "debate",
    confidence: float | str = 0.9,
    content: str = "Some knowledge content",
    item_id: str = "item-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        source=source,
        confidence=confidence,
        content=content,
        id=item_id,
    )


def _make_result(**overrides) -> SimpleNamespace:
    defaults = dict(
        id="debate-123",
        final_answer="The answer is X",
        confidence=0.92,
        consensus_reached=True,
        rounds_used=3,
        participants=["agent-a", "agent-b"],
        winner="agent-a",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_env(**overrides) -> SimpleNamespace:
    defaults = dict(task="Design a rate limiter")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        ops = KnowledgeMoundOperations()
        assert ops.knowledge_mound is None
        assert ops.enable_retrieval is True
        assert ops.enable_ingestion is True
        assert ops._notify_callback is None
        assert ops._metrics is None
        assert ops._last_km_item_ids == []

    def test_custom_values(self):
        mound = _make_mound()
        cb = MagicMock()
        metrics = MagicMock()
        ops = KnowledgeMoundOperations(
            knowledge_mound=mound,
            enable_retrieval=False,
            enable_ingestion=False,
            notify_callback=cb,
            metrics=metrics,
        )
        assert ops.knowledge_mound is mound
        assert ops.enable_retrieval is False
        assert ops.enable_ingestion is False
        assert ops._notify_callback is cb
        assert ops._metrics is metrics


# ---------------------------------------------------------------------------
# 2. fetch_knowledge_context
# ---------------------------------------------------------------------------


class TestFetchKnowledgeContext:
    @pytest.mark.asyncio
    async def test_no_mound_returns_none(self):
        ops = KnowledgeMoundOperations(knowledge_mound=None)
        result = await ops.fetch_knowledge_context("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieval_disabled_returns_none(self):
        ops = KnowledgeMoundOperations(
            knowledge_mound=_make_mound(),
            enable_retrieval=False,
        )
        result = await ops.fetch_knowledge_context("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_query_interface_returns_none(self):
        ops = KnowledgeMoundOperations(
            knowledge_mound=SimpleNamespace(store=AsyncMock()),
            enable_retrieval=True,
        )
        result = await ops.fetch_knowledge_context("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_semantic_success_with_list(self):
        items = [_make_item(), _make_item(item_id="item-2", source="fact")]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task", limit=5)

        assert result is not None
        assert "KNOWLEDGE MOUND CONTEXT" in result
        mound.query_semantic.assert_awaited_once_with(
            text="task",
            limit=5,
            min_confidence=0.5,
        )

    @pytest.mark.asyncio
    async def test_results_with_items_attribute(self):
        items = [_make_item()]
        results_obj = SimpleNamespace(items=items)
        mound = _make_mound()
        mound.query_semantic.return_value = results_obj

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert result is not None
        assert "KNOWLEDGE MOUND CONTEXT" in result

    @pytest.mark.asyncio
    async def test_empty_results_returns_none(self):
        mound = _make_mound()
        mound.query_semantic.return_value = []

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert result is None
        assert ops._last_km_item_ids == []

    @pytest.mark.asyncio
    async def test_none_results_returns_none(self):
        """Results object that is neither list nor has .items returns None."""
        mound = _make_mound()
        mound.query_semantic.return_value = None

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_formats_output_correctly(self):
        items = [
            _make_item(source="debate", confidence=0.9, content="Answer about X"),
            _make_item(source="fact", confidence=0.75, content="Fact about Y", item_id="item-2"),
        ]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")

        assert result.startswith("## KNOWLEDGE MOUND CONTEXT")
        assert "Relevant knowledge from organizational memory:" in result
        assert "**[debate]** (confidence: 90%)" in result
        assert "Answer about X" in result
        assert "**[fact]** (confidence: 75%)" in result
        assert "Fact about Y" in result

    @pytest.mark.asyncio
    async def test_confidence_to_float_numeric(self):
        items = [_make_item(confidence=0.72)]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert "(confidence: 72%)" in result

    @pytest.mark.asyncio
    async def test_confidence_to_float_int(self):
        items = [_make_item(confidence=1)]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert "(confidence: 100%)" in result

    @pytest.mark.asyncio
    async def test_confidence_to_float_string_labels(self):
        """String labels map to predefined confidence values."""
        test_cases = [
            ("verified", "95%"),
            ("high", "80%"),
            ("medium", "60%"),
            ("low", "30%"),
            ("unverified", "20%"),
            ("unknown_label", "50%"),  # default fallback
        ]
        for label, expected in test_cases:
            items = [_make_item(confidence=label)]
            mound = _make_mound()
            mound.query_semantic.return_value = items

            ops = KnowledgeMoundOperations(knowledge_mound=mound)
            result = await ops.fetch_knowledge_context("task")
            assert f"(confidence: {expected})" in result, (
                f"Label '{label}' should map to {expected}"
            )

    @pytest.mark.asyncio
    async def test_confidence_to_float_enum_with_value(self):
        """Enum-like objects with .value attribute are handled."""

        class ConfLevel(enum.Enum):
            HIGH = "high"

        items = [_make_item(confidence=ConfLevel.HIGH)]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert "(confidence: 80%)" in result

    @pytest.mark.asyncio
    async def test_confidence_to_float_non_numeric_non_string_returns_default(self):
        """Non-numeric, non-string, non-enum value returns 0.5 default."""
        items = [_make_item(confidence=object())]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert "(confidence: 50%)" in result

    @pytest.mark.asyncio
    async def test_tracks_last_km_item_ids(self):
        items = [
            _make_item(item_id="id-1"),
            _make_item(item_id="id-2"),
        ]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.fetch_knowledge_context("task")
        assert ops._last_km_item_ids == ["id-1", "id-2"]

    @pytest.mark.asyncio
    async def test_tracks_item_ids_via_item_id_attr(self):
        """Falls back to item_id attr when id is None."""
        item = SimpleNamespace(
            source="test",
            confidence=0.8,
            content="content",
            item_id="fallback-id",
        )
        # No .id attribute
        assert not hasattr(item, "id") or getattr(item, "id", None) is not None
        # Let's create one with id=None
        item2 = SimpleNamespace(
            source="test",
            confidence=0.8,
            content="content",
            id=None,
            item_id="fallback-id-2",
        )
        mound = _make_mound()
        mound.query_semantic.return_value = [item2]

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.fetch_knowledge_context("task")
        assert ops._last_km_item_ids == ["fallback-id-2"]

    @pytest.mark.asyncio
    async def test_auth_context_with_visibility(self):
        auth = SimpleNamespace(user_id="user-1", workspace_id="ws-1", org_id="org-1")
        items = [_make_item()]
        mound = _make_mound()
        mound.query_with_visibility.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task", limit=5, auth_context=auth)

        assert result is not None
        mound.query_with_visibility.assert_awaited_once_with(
            "task",
            actor_id="user-1",
            actor_workspace_id="ws-1",
            actor_org_id="org-1",
            limit=5,
        )
        mound.query_semantic.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auth_context_without_workspace_falls_to_semantic(self):
        auth = SimpleNamespace(user_id="user-1", workspace_id="", org_id=None)
        items = [_make_item()]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task", limit=5, auth_context=auth)

        assert result is not None
        mound.query_semantic.assert_awaited_once_with(
            text="task",
            limit=5,
            min_confidence=0.5,
        )
        mound.query_with_visibility.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auth_context_without_user_id_falls_to_semantic(self):
        auth = SimpleNamespace(user_id="", workspace_id="ws-1", org_id=None)
        items = [_make_item()]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task", limit=5, auth_context=auth)

        assert result is not None
        mound.query_semantic.assert_awaited_once()
        mound.query_with_visibility.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auth_context_mound_without_visibility_method(self):
        """When mound lacks query_with_visibility, falls to query_semantic."""
        auth = SimpleNamespace(user_id="user-1", workspace_id="ws-1", org_id="org-1")
        items = [_make_item()]
        mound = _make_mound()
        del mound.query_with_visibility  # Remove the method
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task", auth_context=auth)

        assert result is not None
        mound.query_semantic.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_runtime_error_not_initialized_retries(self):
        items = [_make_item()]
        mound = _make_mound()
        mound.query_semantic.side_effect = [
            RuntimeError("Knowledge Mound not initialized"),
            items,
        ]

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")

        assert result is not None
        mound.initialize.assert_awaited_once()
        assert mound.query_semantic.await_count == 2

    @pytest.mark.asyncio
    async def test_runtime_error_not_initialized_with_auth_retries(self):
        """Retry path also uses visibility query when auth_context present."""
        auth = SimpleNamespace(user_id="user-1", workspace_id="ws-1", org_id="org-1")
        items = [_make_item()]
        mound = _make_mound()
        mound.query_with_visibility.side_effect = [
            RuntimeError("not initialized yet"),
            items,
        ]

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task", auth_context=auth)

        assert result is not None
        mound.initialize.assert_awaited_once()
        assert mound.query_with_visibility.await_count == 2

    @pytest.mark.asyncio
    async def test_runtime_error_other_message_is_caught_by_outer(self):
        """RuntimeError with a message that is NOT 'not initialized' re-raises
        into the outer except clause which catches RuntimeError and returns None."""
        mound = _make_mound()
        mound.query_semantic.side_effect = RuntimeError("some other error")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")

        # The outer handler catches RuntimeError and returns None
        assert result is None
        mound.initialize.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_success(self):
        items = [_make_item()]
        mound = _make_mound()
        mound.query_semantic.return_value = items
        metrics = MagicMock()

        ops = KnowledgeMoundOperations(knowledge_mound=mound, metrics=metrics)
        await ops.fetch_knowledge_context("task")

        metrics.record.assert_called_once()
        call_args = metrics.record.call_args
        from aragora.knowledge.mound.metrics import OperationType

        assert call_args[0][0] == OperationType.QUERY
        assert call_args[1]["success"] is True
        assert call_args[1]["error"] is None
        assert call_args[1]["metadata"] == {"operation": "fetch_knowledge_context"}

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_failure(self):
        mound = _make_mound()
        mound.query_semantic.side_effect = ValueError("bad query")
        metrics = MagicMock()

        ops = KnowledgeMoundOperations(knowledge_mound=mound, metrics=metrics)
        result = await ops.fetch_knowledge_context("task")

        assert result is None
        metrics.record.assert_called_once()
        call_args = metrics.record.call_args
        assert call_args[1]["success"] is False
        assert "ValueError" in call_args[1]["error"]

    @pytest.mark.asyncio
    async def test_error_handling_type_error(self):
        mound = _make_mound()
        mound.query_semantic.side_effect = TypeError("type issue")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_error_handling_attribute_error(self):
        mound = _make_mound()
        mound.query_semantic.side_effect = AttributeError("no attr")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_error_handling_connection_error(self):
        mound = _make_mound()
        mound.query_semantic.side_effect = ConnectionError("connection lost")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert result is None

    @pytest.mark.asyncio
    async def test_content_truncated_to_300_chars(self):
        long_content = "A" * 500
        items = [_make_item(content=long_content)]
        mound = _make_mound()
        mound.query_semantic.return_value = items

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        # The formatted content should have the truncated version
        assert "A" * 300 in result
        assert "A" * 301 not in result

    @pytest.mark.asyncio
    async def test_item_without_source_defaults_to_unknown(self):
        item = SimpleNamespace(confidence=0.8, content="test", id="id-1")
        mound = _make_mound()
        mound.query_semantic.return_value = [item]

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        result = await ops.fetch_knowledge_context("task")
        assert "**[unknown]**" in result


# ---------------------------------------------------------------------------
# 3. ingest_debate_outcome
# ---------------------------------------------------------------------------


class TestIngestDebateOutcome:
    @pytest.mark.asyncio
    async def test_no_mound_returns(self):
        ops = KnowledgeMoundOperations(knowledge_mound=None)
        # Should not raise
        await ops.ingest_debate_outcome(_make_result())

    @pytest.mark.asyncio
    async def test_ingestion_disabled_returns(self):
        ops = KnowledgeMoundOperations(
            knowledge_mound=_make_mound(),
            enable_ingestion=False,
        )
        await ops.ingest_debate_outcome(_make_result())
        # mound.store should not be called
        ops.knowledge_mound.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_low_confidence_skips(self):
        mound = _make_mound()
        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(confidence=0.5))
        mound.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_confidence_at_threshold_skips(self):
        """Confidence exactly at 0.85 boundary -- result.confidence < 0.85 is False, so it proceeds."""
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="n-1")
        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(confidence=0.85), _make_env())
        mound.store.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_confidence_below_threshold_skips(self):
        mound = _make_mound()
        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(confidence=0.84))
        mound.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_final_answer_skips(self):
        mound = _make_mound()
        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(final_answer=""))
        mound.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_final_answer_skips(self):
        mound = _make_mound()
        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(final_answer=None))
        mound.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_successful_ingestion_with_callback(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-42")
        callback = MagicMock()

        ops = KnowledgeMoundOperations(
            knowledge_mound=mound,
            notify_callback=callback,
        )
        await ops.ingest_debate_outcome(_make_result(), _make_env())

        mound.store.assert_awaited_once()
        call_args = mound.store.call_args[0][0]
        assert "Debate Conclusion:" in call_args.content
        assert call_args.metadata["debate_id"] == "debate-123"
        assert call_args.metadata["task"] == "Design a rate limiter"
        assert call_args.metadata["confidence"] == 0.92

        callback.assert_called_once_with(
            "knowledge_ingested",
            {
                "details": "Stored debate conclusion in Knowledge Mound",
                "metric": 0.92,
            },
        )

    @pytest.mark.asyncio
    async def test_successful_ingestion_without_callback(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(), _make_env())

        mound.store.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_metadata_includes_crux_claims(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        result = _make_result()
        result.debate_cruxes = [
            {"claim": "Rate limiting should be token-based"},
            {"claim": "Sliding window is better than fixed window"},
        ]

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(result, _make_env())

        call_args = mound.store.call_args[0][0]
        assert "crux_claims" in call_args.metadata
        assert len(call_args.metadata["crux_claims"]) == 2
        assert "Rate limiting should be token-based" in call_args.metadata["crux_claims"][0]

    @pytest.mark.asyncio
    async def test_crux_claims_limited_to_5(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        result = _make_result()
        result.debate_cruxes = [{"claim": f"Claim {i}"} for i in range(10)]

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(result, _make_env())

        call_args = mound.store.call_args[0][0]
        assert len(call_args.metadata["crux_claims"]) == 5

    @pytest.mark.asyncio
    async def test_crux_claims_non_dict_items_raises(self):
        """Non-dict crux items cause AttributeError (str has no .get).

        The outer except only catches RuntimeError/ValueError/OSError/KeyError,
        so AttributeError propagates. This documents current behavior.
        """
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        result = _make_result()
        result.debate_cruxes = ["plain string crux"]

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        with pytest.raises(AttributeError, match="get"):
            await ops.ingest_debate_outcome(result, _make_env())

    @pytest.mark.asyncio
    async def test_ingestion_result_without_node_id_not_success(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id=None)
        callback = MagicMock()

        ops = KnowledgeMoundOperations(
            knowledge_mound=mound,
            notify_callback=callback,
        )
        await ops.ingest_debate_outcome(_make_result(), _make_env())

        # node_id is None, so success is False and callback not invoked
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingestion_result_none_not_success(self):
        mound = _make_mound()
        mound.store.return_value = None
        callback = MagicMock()

        ops = KnowledgeMoundOperations(
            knowledge_mound=mound,
            notify_callback=callback,
        )
        await ops.ingest_debate_outcome(_make_result(), _make_env())
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_handling_runtime_error(self):
        mound = _make_mound()
        mound.store.side_effect = RuntimeError("storage failure")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        # Should not raise
        await ops.ingest_debate_outcome(_make_result(), _make_env())

    @pytest.mark.asyncio
    async def test_error_handling_value_error(self):
        mound = _make_mound()
        mound.store.side_effect = ValueError("bad value")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(), _make_env())

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_success(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")
        metrics = MagicMock()

        ops = KnowledgeMoundOperations(knowledge_mound=mound, metrics=metrics)
        await ops.ingest_debate_outcome(_make_result(), _make_env())

        metrics.record.assert_called_once()
        call_args = metrics.record.call_args
        from aragora.knowledge.mound.metrics import OperationType

        assert call_args[0][0] == OperationType.STORE
        assert call_args[1]["success"] is True
        assert call_args[1]["error"] is None
        assert call_args[1]["metadata"] == {"operation": "ingest_debate_outcome"}

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_failure(self):
        mound = _make_mound()
        mound.store.side_effect = RuntimeError("fail")
        metrics = MagicMock()

        ops = KnowledgeMoundOperations(knowledge_mound=mound, metrics=metrics)
        await ops.ingest_debate_outcome(_make_result(), _make_env())

        metrics.record.assert_called_once()
        call_args = metrics.record.call_args
        assert call_args[1]["success"] is False
        assert "RuntimeError" in call_args[1]["error"]

    @pytest.mark.asyncio
    async def test_metadata_participants_limited_to_10(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        result = _make_result(participants=[f"agent-{i}" for i in range(20)])
        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(result, _make_env())

        call_args = mound.store.call_args[0][0]
        assert len(call_args.metadata["participants"]) == 10

    @pytest.mark.asyncio
    async def test_final_answer_truncated_to_2000(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        long_answer = "X" * 3000
        result = _make_result(final_answer=long_answer)
        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(result, _make_env())

        call_args = mound.store.call_args[0][0]
        # "Debate Conclusion: " prefix + 2000 chars of the answer
        assert len(call_args.content) == len("Debate Conclusion: ") + 2000

    @pytest.mark.asyncio
    async def test_env_task_truncated_to_500(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        env = _make_env(task="T" * 1000)
        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(), env)

        call_args = mound.store.call_args[0][0]
        assert len(call_args.metadata["task"]) == 500

    @pytest.mark.asyncio
    async def test_no_env_uses_empty_task(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(), env=None)

        call_args = mound.store.call_args[0][0]
        assert call_args.metadata["task"] == ""

    @pytest.mark.asyncio
    async def test_workspace_id_from_mound(self):
        mound = _make_mound()
        mound.workspace_id = "ws-custom"
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(_make_result(), _make_env())

        call_args = mound.store.call_args[0][0]
        assert call_args.workspace_id == "ws-custom"

    @pytest.mark.asyncio
    async def test_no_debate_cruxes_attr(self):
        """When result has no debate_cruxes attribute, metadata has no crux_claims."""
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        result = _make_result()
        # SimpleNamespace won't have debate_cruxes unless we add it
        assert not hasattr(result, "debate_cruxes")

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(result, _make_env())

        call_args = mound.store.call_args[0][0]
        assert "crux_claims" not in call_args.metadata

    @pytest.mark.asyncio
    async def test_empty_debate_cruxes_no_crux_claims(self):
        mound = _make_mound()
        mound.store.return_value = SimpleNamespace(node_id="node-1")

        result = _make_result()
        result.debate_cruxes = []

        ops = KnowledgeMoundOperations(knowledge_mound=mound)
        await ops.ingest_debate_outcome(result, _make_env())

        call_args = mound.store.call_args[0][0]
        assert "crux_claims" not in call_args.metadata
