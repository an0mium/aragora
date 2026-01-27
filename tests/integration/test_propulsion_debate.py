"""
End-to-End Integration Tests: Propulsion Engine ↔ Debate Phases.

Tests the integration between the Propulsion Engine and debate flow:
1. Propulsion event firing during debates
2. Handler registration and invocation
3. Chained stage execution
4. Event propagation through debate phases
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.propulsion import (
    PropulsionEngine,
    PropulsionPayload,
    PropulsionPriority,
    PropulsionResult,
    reset_propulsion_engine,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def propulsion_engine():
    """Create a fresh propulsion engine for each test."""
    reset_propulsion_engine()
    engine = PropulsionEngine(max_concurrent=5)
    yield engine
    # Cleanup
    engine.clear_results()


@pytest.fixture
def mock_debate_context():
    """Create a mock debate context."""
    return {
        "debate_id": "debate-prop-test",
        "topic": "Should AI be regulated?",
        "round": 1,
        "agents": ["claude", "gpt-4", "gemini"],
    }


@pytest.fixture
def proposal_payload(mock_debate_context):
    """Create a proposal payload."""
    return PropulsionPayload(
        data={
            "proposals": [
                {"agent": "claude", "content": "AI needs regulation for safety"},
                {"agent": "gpt-4", "content": "Self-regulation is better"},
                {"agent": "gemini", "content": "Balanced approach needed"},
            ],
            "debate_id": mock_debate_context["debate_id"],
        },
        source_molecule_id=mock_debate_context["debate_id"],
        priority=PropulsionPriority.NORMAL,
    )


@pytest.fixture
def critique_payload(mock_debate_context):
    """Create a critique payload."""
    return PropulsionPayload(
        data={
            "critiques": [
                {"agent": "claude", "target": "gpt-4", "content": "Lacks enforcement"},
                {"agent": "gpt-4", "target": "claude", "content": "Too restrictive"},
            ],
            "debate_id": mock_debate_context["debate_id"],
        },
        source_molecule_id=mock_debate_context["debate_id"],
        priority=PropulsionPriority.NORMAL,
    )


# ============================================================================
# Integration Tests
# ============================================================================


class TestPropulsionDebateFlow:
    """Tests for Propulsion ↔ Debate phase integration."""

    @pytest.mark.asyncio
    async def test_proposals_ready_event_fires(self, propulsion_engine, proposal_payload):
        """Test that proposals_ready event fires correctly."""
        received_payloads = []

        async def proposals_handler(payload: PropulsionPayload):
            received_payloads.append(payload)
            return {"status": "critiques_started"}

        propulsion_engine.register_handler(
            "proposals_ready",
            proposals_handler,
            name="critique_starter",
        )

        results = await propulsion_engine.propel("proposals_ready", proposal_payload)

        assert len(results) == 1
        assert results[0].success
        assert results[0].handler_name == "critique_starter"
        assert len(received_payloads) == 1
        assert "proposals" in received_payloads[0].data

    @pytest.mark.asyncio
    async def test_critiques_ready_event_handling(self, propulsion_engine, critique_payload):
        """Test that critiques_ready event triggers revision phase."""
        revision_started = False

        async def critiques_handler(payload: PropulsionPayload):
            nonlocal revision_started
            revision_started = True
            critiques = payload.data.get("critiques", [])
            return {"critiques_count": len(critiques)}

        propulsion_engine.register_handler(
            "critiques_ready",
            critiques_handler,
            name="revision_starter",
        )

        results = await propulsion_engine.propel("critiques_ready", critique_payload)

        assert revision_started
        assert results[0].success
        assert results[0].result["critiques_count"] == 2

    @pytest.mark.asyncio
    async def test_full_debate_phase_chain(
        self, propulsion_engine, proposal_payload, mock_debate_context
    ):
        """Test chaining through all debate phases."""
        phase_order = []

        async def on_proposals(payload):
            phase_order.append("proposals_ready")
            return {"next": "critiques"}

        async def on_critiques(payload):
            phase_order.append("critiques_ready")
            return {"next": "revisions"}

        async def on_revisions(payload):
            phase_order.append("revisions_done")
            return {"next": "consensus"}

        async def on_consensus(payload):
            phase_order.append("consensus_check")
            return {"reached": True}

        propulsion_engine.register_handler("proposals_ready", on_proposals)
        propulsion_engine.register_handler("critiques_ready", on_critiques)
        propulsion_engine.register_handler("revisions_done", on_revisions)
        propulsion_engine.register_handler("consensus_check", on_consensus)

        # Chain all phases
        debate_id = mock_debate_context["debate_id"]
        results = await propulsion_engine.chain(
            [
                (
                    "proposals_ready",
                    PropulsionPayload(
                        data={"proposals": []},
                        source_molecule_id=debate_id,
                    ),
                ),
                (
                    "critiques_ready",
                    PropulsionPayload(
                        data={"critiques": []},
                        source_molecule_id=debate_id,
                    ),
                ),
                (
                    "revisions_done",
                    PropulsionPayload(
                        data={"revisions": []},
                        source_molecule_id=debate_id,
                    ),
                ),
                (
                    "consensus_check",
                    PropulsionPayload(
                        data={"final_positions": []},
                        source_molecule_id=debate_id,
                    ),
                ),
            ]
        )

        assert len(results) == 4
        assert all(len(stage_results) == 1 for stage_results in results)
        assert phase_order == [
            "proposals_ready",
            "critiques_ready",
            "revisions_done",
            "consensus_check",
        ]

    @pytest.mark.asyncio
    async def test_priority_based_handler_execution(self, propulsion_engine):
        """Test that handlers execute in priority order."""
        execution_order = []

        async def low_priority_handler(payload):
            execution_order.append("low")
            return {"priority": "low"}

        async def high_priority_handler(payload):
            execution_order.append("high")
            return {"priority": "high"}

        async def critical_handler(payload):
            execution_order.append("critical")
            return {"priority": "critical"}

        propulsion_engine.register_handler(
            "priority_test",
            low_priority_handler,
            name="low",
            priority=PropulsionPriority.LOW,
        )
        propulsion_engine.register_handler(
            "priority_test",
            high_priority_handler,
            name="high",
            priority=PropulsionPriority.HIGH,
        )
        propulsion_engine.register_handler(
            "priority_test",
            critical_handler,
            name="critical",
            priority=PropulsionPriority.CRITICAL,
        )

        payload = PropulsionPayload(data={"test": True})
        await propulsion_engine.propel("priority_test", payload)

        # Critical -> High -> Low (lower enum value = higher priority)
        assert execution_order == ["critical", "high", "low"]

    @pytest.mark.asyncio
    async def test_deadline_expired_payload(self, propulsion_engine):
        """Test that expired payloads are not processed."""
        handler_called = False

        async def handler(payload):
            nonlocal handler_called
            handler_called = True
            return {"processed": True}

        propulsion_engine.register_handler("deadline_test", handler)

        # Create expired payload
        expired_payload = PropulsionPayload(
            data={"test": True},
            deadline=datetime.now(timezone.utc) - timedelta(minutes=5),
        )

        results = await propulsion_engine.propel("deadline_test", expired_payload)

        assert not handler_called
        assert len(results) == 1
        assert not results[0].success
        assert "expired" in results[0].error_message.lower()

    @pytest.mark.asyncio
    async def test_handler_filtering(self, propulsion_engine):
        """Test that handlers can filter payloads."""
        claude_calls = []
        gpt_calls = []

        async def claude_handler(payload):
            claude_calls.append(payload)
            return {"handler": "claude"}

        async def gpt_handler(payload):
            gpt_calls.append(payload)
            return {"handler": "gpt"}

        # Filter by agent_affinity
        propulsion_engine.register_handler(
            "agent_specific",
            claude_handler,
            name="claude_handler",
            filter_fn=lambda p: p.agent_affinity == "claude",
        )
        propulsion_engine.register_handler(
            "agent_specific",
            gpt_handler,
            name="gpt_handler",
            filter_fn=lambda p: p.agent_affinity == "gpt-4",
        )

        # Send Claude-specific payload
        claude_payload = PropulsionPayload(
            data={"response": "from claude"},
            agent_affinity="claude",
        )
        await propulsion_engine.propel("agent_specific", claude_payload)

        # Send GPT-specific payload
        gpt_payload = PropulsionPayload(
            data={"response": "from gpt"},
            agent_affinity="gpt-4",
        )
        await propulsion_engine.propel("agent_specific", gpt_payload)

        assert len(claude_calls) == 1
        assert len(gpt_calls) == 1
        assert claude_calls[0].agent_affinity == "claude"
        assert gpt_calls[0].agent_affinity == "gpt-4"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, propulsion_engine):
        """Test retry behavior on handler failure."""
        attempt_count = 0

        async def flaky_handler(payload):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Simulated failure")
            return {"success": True}

        propulsion_engine.register_handler("retry_test", flaky_handler)

        payload = PropulsionPayload(data={"test": True}, max_attempts=5)
        results = await propulsion_engine.propel_with_retry(
            "retry_test",
            payload,
            max_retries=5,
            backoff_base=0.01,  # Fast backoff for testing
        )

        assert attempt_count == 3
        assert results[0].success

    @pytest.mark.asyncio
    async def test_chain_stops_on_failure(self, propulsion_engine):
        """Test that chain stops when a stage fails."""
        stage_calls = []

        async def stage1_handler(payload):
            stage_calls.append("stage1")
            return {"stage": 1}

        async def stage2_handler(payload):
            stage_calls.append("stage2")
            raise ValueError("Stage 2 failed")

        async def stage3_handler(payload):
            stage_calls.append("stage3")
            return {"stage": 3}

        propulsion_engine.register_handler("stage1", stage1_handler)
        propulsion_engine.register_handler("stage2", stage2_handler)
        propulsion_engine.register_handler("stage3", stage3_handler)

        results = await propulsion_engine.chain(
            [
                ("stage1", PropulsionPayload(data={})),
                ("stage2", PropulsionPayload(data={})),
                ("stage3", PropulsionPayload(data={})),
            ],
            stop_on_failure=True,
        )

        # Should stop at stage2
        assert len(results) == 2
        assert stage_calls == ["stage1", "stage2"]
        assert "stage3" not in stage_calls

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_handlers(self, propulsion_engine):
        """Test broadcasting payload to multiple event types."""
        received_events = []

        async def logging_handler(payload):
            received_events.append("logging")
            return {"logged": True}

        async def metrics_handler(payload):
            received_events.append("metrics")
            return {"recorded": True}

        async def audit_handler(payload):
            received_events.append("audit")
            return {"audited": True}

        propulsion_engine.register_handler("logging", logging_handler)
        propulsion_engine.register_handler("metrics", metrics_handler)
        propulsion_engine.register_handler("audit", audit_handler)

        payload = PropulsionPayload(data={"debate_completed": True})
        results = await propulsion_engine.broadcast(
            ["logging", "metrics", "audit"],
            payload,
        )

        assert len(results) == 3
        assert set(received_events) == {"logging", "metrics", "audit"}
        assert all(len(results[event]) == 1 for event in results)


class TestPropulsionDebateRoundFlow:
    """Tests for propulsion in multi-round debates."""

    @pytest.mark.asyncio
    async def test_round_transition_events(self, propulsion_engine, mock_debate_context):
        """Test events fired during round transitions."""
        round_events = []

        async def round_complete_handler(payload):
            round_num = payload.data.get("round")
            round_events.append(f"round_{round_num}_complete")
            return {"next_round": round_num + 1}

        propulsion_engine.register_handler("round_complete", round_complete_handler)

        # Simulate 3 rounds completing
        for round_num in range(1, 4):
            payload = PropulsionPayload(
                data={
                    "round": round_num,
                    "debate_id": mock_debate_context["debate_id"],
                },
                source_molecule_id=mock_debate_context["debate_id"],
            )
            await propulsion_engine.propel("round_complete", payload)

        assert round_events == [
            "round_1_complete",
            "round_2_complete",
            "round_3_complete",
        ]

    @pytest.mark.asyncio
    async def test_consensus_reached_event(self, propulsion_engine, mock_debate_context):
        """Test consensus_reached event terminates debate flow."""
        debate_state = {"ongoing": True, "consensus": None}

        async def consensus_handler(payload):
            debate_state["ongoing"] = False
            debate_state["consensus"] = payload.data.get("consensus_value")
            return {"debate_finished": True}

        propulsion_engine.register_handler("consensus_reached", consensus_handler)

        payload = PropulsionPayload(
            data={
                "consensus_value": 0.85,
                "winning_position": "Balanced regulation is needed",
                "debate_id": mock_debate_context["debate_id"],
            },
            source_molecule_id=mock_debate_context["debate_id"],
            priority=PropulsionPriority.HIGH,
        )

        await propulsion_engine.propel("consensus_reached", payload)

        assert not debate_state["ongoing"]
        assert debate_state["consensus"] == 0.85

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, propulsion_engine):
        """Test that propulsion statistics are tracked correctly."""

        async def success_handler(payload):
            return {"success": True}

        async def fail_handler(payload):
            raise ValueError("Intentional failure")

        propulsion_engine.register_handler("success_event", success_handler)
        propulsion_engine.register_handler("fail_event", fail_handler)

        # Successful propulsions
        for _ in range(3):
            await propulsion_engine.propel(
                "success_event",
                PropulsionPayload(data={}),
            )

        # Failed propulsions
        for _ in range(2):
            await propulsion_engine.propel(
                "fail_event",
                PropulsionPayload(data={}),
            )

        stats = propulsion_engine.get_stats()

        assert stats["total_propelled"] == 5
        assert stats["successful"] == 3
        assert stats["failed"] == 2
        assert stats["registered_handlers"]["success_event"] == 1
        assert stats["registered_handlers"]["fail_event"] == 1
