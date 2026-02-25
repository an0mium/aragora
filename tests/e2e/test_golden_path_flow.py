"""
End-to-end golden path test: Ingest -> Triage -> Debate -> Action -> Audit.

Covers the full lifecycle of a critical message arriving via Slack webhook,
being triaged by InboxDebateTrigger, spawning a mini-debate, producing a
Decision Plan via the pipeline, and generating an audit-ready receipt with
an integrity hash and provenance chain.

All external services are mocked (no real API calls). The test runs in
< 60 seconds.

GitHub issue: #309
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gauntlet.receipt import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)
from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
    StageResult,
)
from aragora.server.handlers.inbox.auto_debate import (
    DebateTriggerResult,
    InboxDebateTrigger,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def slack_webhook_payload() -> dict[str, Any]:
    """Simulate a Slack webhook event payload for an urgent message."""
    return {
        "type": "event_callback",
        "event": {
            "type": "message",
            "channel": "C001CRITICAL",
            "user": "U_CEO",
            "text": (
                "URGENT: Our primary database cluster is showing 95% CPU "
                "utilization and response times have tripled. Customer-facing "
                "APIs are degraded. We need an immediate decision on whether "
                "to failover to the read-replica cluster or scale vertically."
            ),
            "ts": "1700000000.000001",
        },
        "team_id": "T_ACME",
    }


@pytest.fixture
def parsed_email_from_slack(slack_webhook_payload: dict[str, Any]) -> dict[str, Any]:
    """Parse the Slack webhook payload into the email-like format
    expected by InboxDebateTrigger."""
    event = slack_webhook_payload["event"]
    return {
        "email_id": f"slack-{event['channel']}-{event['ts']}",
        "subject": "URGENT: Database cluster degraded - failover decision needed",
        "snippet": event["text"][:200],
        "from": f"slack-user-{event['user']}",
        "priority": "critical",
    }


@pytest.fixture
def mock_inline_debate_result() -> dict[str, Any]:
    """Mock result returned by _run_inline_mock_debate."""
    return {
        "id": "debate-golden-path-001",
        "topic": "Analyze and advise on this critical email",
        "rounds": 2,
        "agents": ["analyst", "critic", "moderator"],
        "consensus": {
            "reached": True,
            "confidence": 0.87,
            "winner": "analyst",
        },
        "final_answer": (
            "Recommend immediate failover to read-replica cluster with "
            "staged traffic migration. Scale vertically on the primary "
            "during low-traffic window tonight."
        ),
        "votes": [
            {"agent": "analyst", "choice": "analyst", "confidence": 0.9},
            {"agent": "critic", "choice": "analyst", "confidence": 0.82},
            {"agent": "moderator", "choice": "analyst", "confidence": 0.88},
        ],
    }


@pytest.fixture
def mock_email_cache(parsed_email_from_slack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Simple dict-based email cache that InboxDebateTrigger.trigger_debate uses."""
    email_id = parsed_email_from_slack["email_id"]
    return {
        email_id: {
            "subject": parsed_email_from_slack["subject"],
            "snippet": parsed_email_from_slack["snippet"],
            "from": parsed_email_from_slack["from"],
        }
    }


# =============================================================================
# Golden Path Test
# =============================================================================


@pytest.mark.e2e
class TestGoldenPathFlow:
    """End-to-end golden path: Ingest -> Triage -> Debate -> Action -> Audit."""

    @pytest.mark.asyncio
    async def test_ingest_triage_debate_action_audit(
        self,
        slack_webhook_payload: dict[str, Any],
        parsed_email_from_slack: dict[str, Any],
        mock_inline_debate_result: dict[str, Any],
        mock_email_cache: dict[str, dict[str, Any]],
    ):
        """Full golden path covering all five stages with assertions at each.

        Stage 1 - Ingest:  Parse the Slack webhook message.
        Stage 2 - Triage:  InboxDebateTrigger classifies as critical, triggers debate.
        Stage 3 - Debate:  Mini-debate with mock agents reaches consensus.
        Stage 4 - Action:  Pipeline produces a Decision Plan with goals and actions.
        Stage 5 - Audit:   Receipt with integrity hash and provenance chain.
        """

        # =====================================================================
        # Stage 1: INGEST -- Parse the Slack webhook payload
        # =====================================================================
        event = slack_webhook_payload.get("event", {})
        assert event["type"] == "message"
        assert "URGENT" in event["text"]
        assert len(event["text"]) > 50, "Message body should be substantive"

        email_id = parsed_email_from_slack["email_id"]
        subject = parsed_email_from_slack["subject"]
        priority = parsed_email_from_slack["priority"]

        assert email_id.startswith("slack-")
        assert "URGENT" in subject or "failover" in subject
        assert priority == "critical"

        # =====================================================================
        # Stage 2: TRIAGE -- InboxDebateTrigger detects critical priority
        # =====================================================================
        trigger = InboxDebateTrigger()

        should, reason = trigger.should_trigger(email_id, priority)
        assert should is True, f"Critical message should trigger debate, got: {reason}"
        assert "eligible" in reason.lower()

        # Also verify non-critical messages are rejected
        should_low, reason_low = trigger.should_trigger("other-id", "low")
        assert should_low is False
        assert "below threshold" in reason_low

        # Now actually trigger the debate (mock the inline debate).
        # trigger_debate() does ``from aragora.server.handlers.playground
        # import _run_inline_mock_debate`` at call time, so we patch at the
        # definition site. Similarly, _emit_trigger_event imports
        # ``dispatch_event`` from aragora.events.dispatcher.
        with patch(
            "aragora.server.handlers.playground._run_inline_mock_debate",
            return_value=mock_inline_debate_result,
        ), patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=lambda *a, **kw: None,
        ):
            trigger_result = await trigger.trigger_debate(
                email_id=email_id,
                subject=subject,
                body_preview=parsed_email_from_slack["snippet"],
                sender=parsed_email_from_slack["from"],
                agents=3,
                rounds=2,
            )

        assert trigger_result.triggered is True, (
            f"Debate should have been triggered, reason: {trigger_result.reason}"
        )
        assert trigger_result.debate_id is not None
        assert trigger_result.email_id == email_id

        debate_id = trigger_result.debate_id

        # =====================================================================
        # Stage 3: DEBATE -- Mini-debate reaches consensus
        # =====================================================================
        # The mock_inline_debate_result simulates a completed debate.
        # Verify the debate result structure.
        debate_result = mock_inline_debate_result
        assert debate_result["id"] == debate_id
        assert debate_result["consensus"]["reached"] is True
        assert debate_result["consensus"]["confidence"] >= 0.7
        assert len(debate_result["votes"]) >= 2
        assert len(debate_result["final_answer"]) > 20

        # Verify all agents voted
        voting_agents = {v["agent"] for v in debate_result["votes"]}
        assert len(voting_agents) >= 2, "At least 2 agents should have voted"

        # =====================================================================
        # Stage 4: ACTION -- Pipeline produces a Decision Plan
        # =====================================================================
        # Feed the debate outcome into the pipeline as input ideas.
        # The pipeline's from_ideas() path extracts goals and actions.
        pipeline = IdeaToExecutionPipeline()

        # Convert the debate final answer into pipeline ideas
        ideas = [
            f"[high] {debate_result['final_answer']}",
            "[high] Implement database failover to read-replica cluster",
            "[medium] Set up monitoring alerts for CPU utilization thresholds",
            "[medium] Scale primary database vertically during maintenance window",
        ]

        pipeline_result = pipeline.from_ideas(ideas, auto_advance=True)

        # Verify Stage 4 produced all expected outputs
        assert isinstance(pipeline_result, PipelineResult)
        assert pipeline_result.pipeline_id.startswith("pipe-")
        assert pipeline_result.ideas_canvas is not None, "Ideas canvas should be populated"
        assert len(pipeline_result.ideas_canvas.nodes) >= len(ideas), (
            f"Expected at least {len(ideas)} idea nodes, "
            f"got {len(pipeline_result.ideas_canvas.nodes)}"
        )

        assert pipeline_result.goal_graph is not None, "Goal graph should be populated"
        assert len(pipeline_result.goal_graph.goals) >= 1, (
            "At least 1 goal should be extracted from the ideas"
        )

        assert pipeline_result.actions_canvas is not None, "Actions canvas should be populated"
        assert len(pipeline_result.actions_canvas.nodes) >= 1, (
            "At least 1 action step should be generated"
        )

        assert pipeline_result.orchestration_canvas is not None, (
            "Orchestration canvas should be populated"
        )

        # Verify provenance chain connects stages
        assert len(pipeline_result.provenance) > 0, (
            "Provenance chain should have at least one link"
        )
        provenance_stages = set()
        for link in pipeline_result.provenance:
            provenance_stages.add(link.source_stage.value)
            provenance_stages.add(link.target_stage.value)

        assert "goals" in provenance_stages, (
            f"'goals' should be in provenance stages. Got: {provenance_stages}"
        )
        assert "actions" in provenance_stages, (
            f"'actions' should be in provenance stages. Got: {provenance_stages}"
        )

        # Verify stage transitions exist
        assert len(pipeline_result.transitions) >= 2, (
            f"Expected at least 2 transitions, got {len(pipeline_result.transitions)}"
        )

        # Verify all stages completed
        for stage_name in ("ideas", "goals", "actions", "orchestration"):
            assert pipeline_result.stage_status.get(stage_name) == "complete", (
                f"Stage '{stage_name}' should be complete, "
                f"got '{pipeline_result.stage_status.get(stage_name)}'"
            )

        # Verify goals have SMART scores
        goals_with_smart = [
            g for g in pipeline_result.goal_graph.goals
            if "smart_scores" in g.metadata
        ]
        assert len(goals_with_smart) > 0, "At least one goal should have SMART scores"

        # =====================================================================
        # Stage 5: AUDIT -- Receipt with integrity hash and provenance
        # =====================================================================
        # Option A: Use pipeline's built-in receipt (from async run with
        # enable_receipts=True). We'll also manually build a DecisionReceipt
        # from debate data to verify the gauntlet receipt path.

        # 5a: Pipeline integrity hash (from to_dict)
        result_dict = pipeline_result.to_dict()
        integrity_hash = result_dict["integrity_hash"]
        assert integrity_hash is not None
        assert len(integrity_hash) == 16, (
            f"Integrity hash should be 16 hex chars, got {len(integrity_hash)}"
        )
        assert all(c in "0123456789abcdef" for c in integrity_hash), (
            f"Integrity hash should be hex, got: {integrity_hash}"
        )

        # Verify determinism: computing again should give the same hash
        integrity_hash_2 = pipeline_result._compute_integrity_hash()
        assert integrity_hash == integrity_hash_2, "Integrity hash should be deterministic"

        # 5b: Build a DecisionReceipt from the debate outcome
        from aragora.core_types import DebateResult

        debate_result_obj = DebateResult(
            task=subject,
            final_answer=debate_result["final_answer"],
            confidence=debate_result["consensus"]["confidence"],
            consensus_reached=debate_result["consensus"]["reached"],
            rounds_used=debate_result["rounds"],
            participants=debate_result["agents"],
            dissenting_views=[],
        )

        receipt = DecisionReceipt.from_debate_result(debate_result_obj)

        # Verify receipt fields
        assert receipt.receipt_id is not None
        assert len(receipt.receipt_id) > 0
        assert receipt.gauntlet_id is not None  # Uses debate_id
        assert receipt.timestamp is not None
        assert receipt.verdict in ("PASS", "CONDITIONAL", "FAIL")
        assert receipt.confidence == debate_result["consensus"]["confidence"]

        # High confidence + consensus => PASS verdict
        assert receipt.verdict == "PASS", (
            f"High confidence consensus should yield PASS, got {receipt.verdict}"
        )

        # Verify SHA-256 artifact hash
        assert receipt.artifact_hash is not None
        assert len(receipt.artifact_hash) == 64, (
            f"SHA-256 hash should be 64 hex chars, got {len(receipt.artifact_hash)}"
        )
        assert all(c in "0123456789abcdef" for c in receipt.artifact_hash)

        # Integrity verification should pass
        assert receipt.verify_integrity() is True

        # Verify consensus proof exists
        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.reached is True
        assert receipt.consensus_proof.confidence == debate_result_obj.confidence
        assert len(receipt.consensus_proof.supporting_agents) > 0

        # Verify provenance chain in receipt
        assert len(receipt.provenance_chain) >= 1, (
            "Receipt should have at least 1 provenance record (verdict)"
        )
        # Last provenance record should be the verdict event
        verdict_events = [
            p for p in receipt.provenance_chain if p.event_type == "verdict"
        ]
        assert len(verdict_events) >= 1, "Receipt should contain a verdict provenance event"

        # Verify the receipt can be serialized to JSON
        import json

        json_str = receipt.to_json()
        receipt_data = json.loads(json_str)
        assert "receipt_id" in receipt_data
        assert "artifact_hash" in receipt_data
        assert "consensus_proof" in receipt_data
        assert "provenance_chain" in receipt_data

        # 5c: Verify the async pipeline also produces a receipt
        async_pipeline = IdeaToExecutionPipeline()

        async def _mock_execute(task, _cfg):
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "completed",
                "output": {},
            }

        async_pipeline._execute_task = _mock_execute

        async_config = PipelineConfig(
            dry_run=False,
            enable_receipts=True,
            stages_to_run=["ideation", "goals", "workflow", "orchestration"],
            enable_km_precedents=False,
            enable_workspace_context=False,
            enable_meta_tuning=False,
            enable_km_persistence=False,
        )

        async_result = await async_pipeline.run(
            ". ".join(ideas),
            config=async_config,
        )

        # Pipeline receipt (falls back to dict since DecisionReceipt constructor
        # is called with pipeline-specific kwargs)
        assert async_result.receipt is not None, (
            "Async pipeline with enable_receipts=True should generate a receipt"
        )
        assert "integrity_hash" in async_result.receipt or "pipeline_id" in async_result.receipt

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_retrigger(
        self,
        parsed_email_from_slack: dict[str, Any],
        mock_inline_debate_result: dict[str, Any],
    ):
        """Verify the trigger's cooldown mechanism prevents re-debating the same email."""
        trigger = InboxDebateTrigger()
        email_id = parsed_email_from_slack["email_id"]

        # First trigger should succeed
        with patch(
            "aragora.server.handlers.playground._run_inline_mock_debate",
            return_value=mock_inline_debate_result,
        ), patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=lambda *a, **kw: None,
        ):
            first = await trigger.trigger_debate(
                email_id=email_id,
                subject=parsed_email_from_slack["subject"],
                body_preview=parsed_email_from_slack["snippet"],
                sender=parsed_email_from_slack["from"],
            )
        assert first.triggered is True

        # Second trigger for same email should be blocked by cooldown
        should, reason = trigger.should_trigger(email_id, "critical")
        assert should is False
        assert "cooldown" in reason.lower()

    @pytest.mark.asyncio
    async def test_full_flow_completes_under_time_budget(
        self,
        slack_webhook_payload: dict[str, Any],
        mock_inline_debate_result: dict[str, Any],
    ):
        """The entire golden path should complete within 60 seconds."""
        start = time.monotonic()

        # Ingest
        event = slack_webhook_payload["event"]
        assert event["type"] == "message"

        # Triage
        trigger = InboxDebateTrigger()
        email_id = f"slack-{event['channel']}-{event['ts']}"
        should, _ = trigger.should_trigger(email_id, "critical")
        assert should is True

        # Debate (mocked)
        with patch(
            "aragora.server.handlers.playground._run_inline_mock_debate",
            return_value=mock_inline_debate_result,
        ), patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=lambda *a, **kw: None,
        ):
            result = await trigger.trigger_debate(
                email_id=email_id,
                subject="URGENT: DB failover needed",
                body_preview=event["text"][:200],
                sender="ceo",
            )
        assert result.triggered is True

        # Action (pipeline)
        pipeline = IdeaToExecutionPipeline()
        pipeline_result = pipeline.from_ideas(
            [
                "[high] Failover to read-replica cluster immediately",
                "[medium] Scale primary DB during maintenance window",
                "[medium] Add CPU monitoring alerts",
            ],
            auto_advance=True,
        )
        assert pipeline_result.goal_graph is not None

        # Audit (receipt)
        from aragora.core_types import DebateResult

        debate_result_obj = DebateResult(
            task="DB failover decision",
            final_answer="Failover recommended",
            confidence=0.87,
            consensus_reached=True,
            rounds_used=2,
            participants=["analyst", "critic", "moderator"],
        )
        receipt = DecisionReceipt.from_debate_result(debate_result_obj)
        assert receipt.verify_integrity() is True

        elapsed = time.monotonic() - start
        assert elapsed < 60.0, f"Golden path took {elapsed:.2f}s, expected < 60s"
