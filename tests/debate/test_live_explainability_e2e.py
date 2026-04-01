"""End-to-end coverage for live explainability flowing into receipts."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.core import DebateResult, Environment, TaskComplexity
from aragora.debate.context import DebateContext
from aragora.debate.event_bus import EventBus
from aragora.debate.orchestrator_runner import (
    _DebateExecutionState,
    handle_debate_completion,
    setup_debate_infrastructure,
)
from aragora.gauntlet.receipt_models import DecisionReceipt


class _FakeArena:
    """Lightweight Arena double with the attributes wiring code expects."""

    def __init__(self) -> None:
        self.env = MagicMock(spec=Environment)
        self.env.task = "Should we enable live explainability receipts?"
        self.env.context = {}

        agents = []
        for name in ("claude", "gpt4", "gemini"):
            agent = MagicMock()
            agent.name = name
            agent.model = f"{name}-model"
            agents.append(agent)
        self.agents = agents

        self.protocol = MagicMock()
        self.protocol.enable_km_belief_sync = False
        self.protocol.enable_hook_tracking = False
        self.protocol.rounds = 3
        self.protocol.checkpoint_cleanup_on_success = True
        self.protocol.enable_translation = False

        self._budget_coordinator = MagicMock()
        self._budget_coordinator.check_budget_before_debate = MagicMock()
        self._budget_coordinator.autotuner = None

        self._trackers = MagicMock()
        self._trackers.on_debate_start = MagicMock()
        self._trackers.on_debate_complete = MagicMock()

        self.extensions = MagicMock()
        self.extensions.on_debate_complete = MagicMock()
        self.extensions.setup_debate_budget = MagicMock()

        self.event_bus = EventBus()
        self._event_emitter = MagicMock()

        self._emit_agent_preview = MagicMock()
        self._create_pending_debate_bead = AsyncMock(return_value=None)
        self._init_hook_tracking = AsyncMock(return_value={})
        self._ingest_debate_outcome = AsyncMock()
        self._update_debate_bead = AsyncMock()
        self._complete_hook_tracking = AsyncMock()
        self._create_debate_bead = AsyncMock(return_value=None)
        self._queue_for_supabase_sync = MagicMock()
        self.cleanup_checkpoints = AsyncMock(return_value=0)
        self._cleanup_convergence_cache = MagicMock()
        self._teardown_agent_channels = AsyncMock()
        self._translate_conclusions = AsyncMock()

        self.enable_introspection = False
        self.active_introspection_tracker = None
        self.enable_live_explainability = True
        self.live_explainability_stream = None
        self.enable_post_debate_workflow = False
        self.disable_post_debate_pipeline = True
        self.enable_auto_execution = False
        self.post_debate_config = None
        self.compliance_monitor = None


@pytest.fixture
def fake_arena():
    return _FakeArena()


@pytest.fixture
def execution_state():
    ctx = MagicMock(spec=DebateContext)
    ctx.env = MagicMock()
    ctx.env.task = "Should we enable live explainability receipts?"
    ctx.result = DebateResult(
        debate_id="debate-live-explainability-e2e",
        task="Should we enable live explainability receipts?",
        consensus_reached=True,
        confidence=0.91,
        messages=[],
        critiques=[],
        votes=[],
        rounds_used=2,
        participants=["claude", "gpt4", "gemini"],
        final_answer="Yes, keep the live factor snapshot in the final receipt.",
        duration_seconds=4.2,
        metadata={},
    )
    ctx.domain = "architecture"
    ctx.post_debate_workflow_triggered = False
    return _DebateExecutionState(
        debate_id="debate-live-explainability-e2e",
        correlation_id="corr-live-explainability-e2e",
        domain="architecture",
        task_complexity=TaskComplexity.MODERATE,
        ctx=ctx,
        debate_status="completed",
        debate_start_time=time.perf_counter() - 5.0,
    )


@pytest.mark.asyncio
async def test_eventbus_snapshot_survives_receipt_roundtrip(fake_arena, execution_state):
    """EventBus-driven explainability factors should survive final receipt serialization."""
    await setup_debate_infrastructure(fake_arena, execution_state)

    bus = fake_arena.event_bus
    bus.emit_sync(
        "agent_message",
        debate_id="debate-live-explainability-e2e",
        agent="claude",
        content="Keep a live factor snapshot so reviewers can inspect the reasoning drift.",
        role="proposer",
        round_num=1,
    )
    bus.emit_sync(
        "agent_message",
        debate_id="debate-live-explainability-e2e",
        agent="gpt4",
        content="The receipt should preserve the evidence that shifted the debate.",
        role="proposer",
        round_num=1,
    )
    bus.emit_sync(
        "agent_message",
        debate_id="debate-live-explainability-e2e",
        agent="gemini",
        content="Only keep the condensed factors to avoid receipt bloat.",
        role="critic",
        round_num=1,
    )
    bus.emit_sync(
        "vote",
        debate_id="debate-live-explainability-e2e",
        agent="claude",
        choice="preserve_live_explainability",
        confidence=0.92,
        round_num=2,
    )
    bus.emit_sync(
        "vote",
        debate_id="debate-live-explainability-e2e",
        agent="gpt4",
        choice="preserve_live_explainability",
        confidence=0.85,
        round_num=2,
    )
    bus.emit_sync(
        "vote",
        debate_id="debate-live-explainability-e2e",
        agent="gemini",
        choice="preserve_live_explainability",
        confidence=0.73,
        round_num=2,
    )

    await handle_debate_completion(fake_arena, execution_state)

    result = execution_state.ctx.result
    live_explainability = result.metadata["live_explainability"]
    live_explainability["unexpected"] = "<script>alert(1)</script>"
    live_explainability["factors"].append(
        {
            "name": "Injected factor",
            "contribution": 0.42,
            "explanation": "  Extra factor with bounded text.  ",
            "trend": "up",
            "ignored": "<svg onload=alert(1)>",
        }
    )
    assert live_explainability["factors"]
    assert live_explainability["vote_count"] == 3
    assert live_explainability["evidence_count"] == 3

    receipt = DecisionReceipt.from_debate_result(result)
    assert receipt.explainability is not None
    stored_explainability = receipt.explainability["live_explainability"]
    assert stored_explainability["factors"]
    assert stored_explainability["leading_position"] == "preserve_live_explainability"
    assert "unexpected" not in stored_explainability
    assert "ignored" not in stored_explainability["factors"][-1]

    receipt.sign()
    assert receipt.verify_signature()

    restored = DecisionReceipt.from_dict(receipt.to_dict())
    assert restored.explainability is not None
    assert (
        restored.explainability["live_explainability"]["narrative"]
        == live_explainability["narrative"]
    )
    assert (
        restored.explainability["live_explainability"]["factors"]
        == stored_explainability["factors"]
    )
    assert restored.verify_signature()

    restored.explainability["live_explainability"]["narrative"] = "tampered"
    assert not restored.verify_signature()
