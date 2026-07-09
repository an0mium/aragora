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

    from aragora.core_types import DebateResult, DebateStatus, normalize_debate_status

    assert isinstance(result, DebateResult)
    assert result.task == "Should we adopt microservices?"
    # Legacy status is "consensus_reached" or "completed" depending on whether
    # the demo agents converge; both project to the canonical COMPLETED state.
    assert normalize_debate_status(result.status) == DebateStatus.COMPLETED
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
    """All six golden API names are usable from the aragora package."""
    import asyncio

    import aragora
    from aragora.golden import WorkflowHandle
    from aragora.golden import recall as golden_recall
    from aragora.golden import receipt as golden_receipt
    from aragora.golden import remember as golden_remember

    # These names don't collide with subpackage names.
    assert aragora.remember is golden_remember
    assert aragora.recall is golden_recall
    assert aragora.receipt is golden_receipt

    # ``debate``, ``review``, and ``workflow`` collide with same-named subpackages.
    # Whichever object is bound (golden callable, or the callable subpackage
    # module once it has been imported), calling it must delegate to the
    # golden implementation (#8780).
    assert callable(aragora.debate)
    assert callable(aragora.review)
    assert callable(aragora.workflow)

    wf = aragora.workflow("golden-package-check")
    assert isinstance(wf, WorkflowHandle)

    coro = aragora.debate("golden-package-check")
    assert asyncio.iscoroutine(coro)
    coro.close()

    review_coro = aragora.review("golden-package-check")
    assert asyncio.iscoroutine(review_coro)
    review_coro.close()


# ---------------------------------------------------------------------------
# import-order determinism (#8780)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_IMPORT_ORDER_CHECK = """
import asyncio

{imports}

from aragora.golden import WorkflowHandle

wf = aragora.workflow("golden-order-check")
assert isinstance(wf, WorkflowHandle), type(wf)

coro = aragora.debate("golden-order-check")
assert asyncio.iscoroutine(coro), type(coro)
coro.close()

review_coro = aragora.review("golden-order-check")
assert asyncio.iscoroutine(review_coro), type(review_coro)
review_coro.close()

# Submodule resolution via sys.modules must be unaffected by the guard.
import aragora.workflow.engine
import aragora.debate.orchestrator
import aragora.review.protocol

print("GOLDEN_ORDER_OK")
"""


@pytest.mark.parametrize(
    "imports",
    [
        pytest.param(
            "import aragora.workflow\nimport aragora.debate\nimport aragora.review\nimport aragora",
            id="submodules-first",
        ),
        pytest.param(
            "import aragora\nimport aragora.workflow\nimport aragora.debate\nimport aragora.review",
            id="package-first",
        ),
        pytest.param(
            # Touch the lazy golden exports before the submodules load, so the
            # import system rebinds the package attributes to module objects.
            "import aragora\n"
            "assert callable(aragora.workflow) and callable(aragora.debate) and callable(aragora.review)\n"
            "import aragora.workflow\nimport aragora.debate\nimport aragora.review",
            id="package-first-lazy-attr-touched",
        ),
    ],
)
def test_golden_callables_survive_import_order(imports: str) -> None:
    """Colliding Golden API names are callable in every import order (#8780)."""
    import subprocess
    import sys

    code = _IMPORT_ORDER_CHECK.format(imports=imports)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "GOLDEN_ORDER_OK" in proc.stdout
