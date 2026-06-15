"""Tests for the Golden 5 simplified API surface (aragora.golden)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.golden import (
    WorkflowHandle,
    debate,
    recall,
    receipt,
    remember,
    review,
    workflow,
)


# ---------------------------------------------------------------------------
# debate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debate_with_int_agents():
    """debate(task, agents=3) auto-creates DemoAgents and returns DebateResult."""
    result = await debate("Should we adopt microservices?", agents=3, rounds=1)

    from aragora.core_types import DebateResult

    assert isinstance(result, DebateResult)
    assert result.task == "Should we adopt microservices?"
    assert result.status == "completed"
    assert len(result.participants) == 3


@pytest.mark.asyncio
async def test_debate_with_agent_list():
    """debate() accepts an explicit list of agent instances."""
    from aragora.agents.demo_agent import DemoAgent

    agents = [
        DemoAgent(name="alpha", role="proposer"),
        DemoAgent(name="beta", role="critic"),
    ]
    result = await debate("Plan a release", agents=agents, rounds=1)

    assert result.participants == ["alpha", "beta"]
    assert result.final_answer  # non-empty


@pytest.mark.asyncio
async def test_debate_custom_rounds():
    """The rounds parameter flows through to the protocol."""
    result = await debate("Quick test", agents=2, rounds=2)

    assert result.rounds_used == 2


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remember_stores_result(tmp_path):
    """remember() stores a debate result in continuum memory."""
    from aragora.core_types import DebateResult

    fake_result = DebateResult(
        debate_id="test-debate-001",
        task="testing",
        final_answer="42",
        confidence=0.9,
        consensus_reached=True,
        rounds_used=1,
        status="completed",
        participants=["a"],
        proposals={"a": "42"},
        messages=[],
        critiques=[],
        votes=[],
    )

    with patch("aragora.memory.continuum.core.ContinuumMemory") as MockCMS:
        mock_entry = MagicMock()
        mock_instance = MockCMS.return_value
        mock_instance.store = AsyncMock(return_value=mock_entry)

        entry = await remember(fake_result, tier="fast", importance=0.9)

        MockCMS.assert_called_once()
        mock_instance.store.assert_awaited_once()
        call_kwargs = mock_instance.store.call_args
        assert call_kwargs.kwargs["key"] == "test-debate-001"
        assert call_kwargs.kwargs["content"] == "42"
        assert call_kwargs.kwargs["importance"] == 0.9
        assert entry is mock_entry


@pytest.mark.asyncio
async def test_remember_stores_string(tmp_path):
    """remember() can store a plain string."""
    with patch("aragora.memory.continuum.core.ContinuumMemory") as MockCMS:
        mock_entry = MagicMock()
        mock_instance = MockCMS.return_value
        mock_instance.store = AsyncMock(return_value=mock_entry)

        entry = await remember("important fact", tier="slow", importance=0.5)

        call_kwargs = mock_instance.store.call_args
        assert call_kwargs.kwargs["content"] == "important fact"
        assert call_kwargs.kwargs["key"].startswith("golden-")


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_retrieves():
    """recall() delegates to ContinuumMemory.retrieve()."""
    with patch("aragora.memory.continuum.core.ContinuumMemory") as MockCMS:
        mock_entries = [MagicMock(), MagicMock()]
        mock_instance = MockCMS.return_value
        mock_instance.retrieve = MagicMock(return_value=mock_entries)

        results = await recall("microservices tradeoffs", limit=5)

        mock_instance.retrieve.assert_called_once_with(query="microservices tradeoffs", limit=5)
        assert results == mock_entries


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_string_content():
    """review() runs gauntlet on string content."""
    with patch("aragora.gauntlet.runner.GauntletRunner") as MockRunner:
        mock_result = MagicMock()
        mock_instance = MockRunner.return_value
        mock_instance.run = AsyncMock(return_value=mock_result)

        result = await review("some policy text", context="compliance audit")

        mock_instance.run.assert_awaited_once_with("some policy text", context="compliance audit")
        assert result is mock_result


@pytest.mark.asyncio
async def test_review_file_path(tmp_path):
    """review() reads the file if the content looks like an existing path."""
    test_file = tmp_path / "spec.md"
    test_file.write_text("# Architecture Spec\nDetails here.", encoding="utf-8")

    with patch("aragora.gauntlet.runner.GauntletRunner") as MockRunner:
        mock_result = MagicMock()
        mock_instance = MockRunner.return_value
        mock_instance.run = AsyncMock(return_value=mock_result)

        result = await review(str(test_file))

        # The runner should receive the file *contents*, not the path
        call_args = mock_instance.run.call_args
        assert "Architecture Spec" in call_args.args[0]
        assert result is mock_result


# ---------------------------------------------------------------------------
# workflow
# ---------------------------------------------------------------------------


def test_workflow_builder_chaining():
    """workflow().step().step() returns a WorkflowHandle with steps."""
    wf = workflow("deploy")
    returned = wf.step("build").step("test").step("ship")

    assert returned is wf  # chaining returns same handle
    assert isinstance(wf, WorkflowHandle)
    assert wf.name == "deploy"
    assert wf.steps == ["build", "test", "ship"]


@pytest.mark.asyncio
async def test_workflow_run_executes_steps():
    """WorkflowHandle.run() calls debate() for each step."""
    wf = workflow("ci").step("lint").step("test")

    with patch("aragora.golden.debate", new_callable=AsyncMock) as mock_debate:
        mock_debate.return_value = MagicMock()
        results = await wf.run()

    assert mock_debate.await_count == 2
    assert "lint" in results
    assert "test" in results


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------


def test_receipt_from_debate_result():
    """receipt() creates a DecisionReceipt from a DebateResult."""
    from aragora.core_types import DebateResult

    fake_result = DebateResult(
        debate_id="receipt-test-001",
        task="testing receipts",
        final_answer="approved",
        confidence=0.95,
        consensus_reached=True,
        rounds_used=3,
        status="completed",
        participants=["a", "b"],
        proposals={"a": "yes", "b": "yes"},
        messages=[],
        critiques=[],
        votes=[],
    )

    r = receipt(fake_result)

    from aragora.gauntlet.receipt_models import DecisionReceipt

    assert isinstance(r, DecisionReceipt)
    assert r.confidence == 0.95


def test_receipt_from_gauntlet_result():
    """receipt() creates a DecisionReceipt from a GauntletResult."""
    from aragora.gauntlet.result import GauntletResult

    fake_result = GauntletResult(
        gauntlet_id="gauntlet-test-001",
        input_hash="abc123",
        input_summary="test input",
        started_at="2026-01-01T00:00:00",
        completed_at="2026-01-01T00:01:00",
    )

    r = receipt(fake_result)

    from aragora.gauntlet.receipt_models import DecisionReceipt

    assert isinstance(r, DecisionReceipt)


def test_receipt_rejects_unknown_type():
    """receipt() raises TypeError for unrecognised types."""
    with pytest.raises(TypeError, match="Cannot create receipt"):
        receipt("not a result object")


# ---------------------------------------------------------------------------
# package-level imports
# ---------------------------------------------------------------------------


def test_golden_imports_from_package():
    """Golden API functions are accessible from the aragora package.

    Note: ``aragora.debate`` may resolve to the ``aragora.debate`` subpackage
    module if it was imported before the lazy ``_EXPORT_MAP`` lookup fires.
    We verify the *other* five names that have no subpackage collision, plus
    verify ``debate`` is directly importable from ``aragora.golden``.
    """
    import aragora
    from aragora.golden import debate as golden_debate
    from aragora.golden import recall as golden_recall
    from aragora.golden import receipt as golden_receipt
    from aragora.golden import remember as golden_remember
    from aragora.golden import review as golden_review
    from aragora.golden import workflow as golden_workflow

    # These names don't collide with subpackage names
    assert aragora.remember is golden_remember
    assert aragora.recall is golden_recall
    assert aragora.review is golden_review
    assert aragora.workflow is golden_workflow
    assert aragora.receipt is golden_receipt

    # debate() is directly usable from aragora.golden even if
    # aragora.debate resolves to the subpackage in some import orders
    assert callable(golden_debate)
