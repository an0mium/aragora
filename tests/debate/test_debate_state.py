"""Tests for DebateContext and AgentWorkspace state containers.

Covers:
- AgentWorkspace: creation, clear, to_dict
- DebateContext: defaults, helper methods, state mutation, finalization
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.debate.debate_state import AgentWorkspace, DebateContext


# ---------------------------------------------------------------------------
# AgentWorkspace
# ---------------------------------------------------------------------------


class TestAgentWorkspace:
    def test_creation(self):
        ws = AgentWorkspace(agent_id="claude")
        assert ws.agent_id == "claude"
        assert ws.memory == {}
        assert ws.tool_results == {}
        assert ws.state == {}
        assert ws.created_at > 0

    def test_memory_and_state(self):
        ws = AgentWorkspace(agent_id="gpt")
        ws.memory["notes"] = "important"
        ws.tool_results["search"] = ["result1"]
        ws.state["iteration"] = 3
        assert ws.memory["notes"] == "important"
        assert ws.state["iteration"] == 3

    def test_clear(self):
        ws = AgentWorkspace(agent_id="gpt")
        ws.memory["k"] = "v"
        ws.tool_results["k"] = "v"
        ws.state["k"] = "v"
        ws.clear()
        assert ws.memory == {}
        assert ws.tool_results == {}
        assert ws.state == {}

    def test_to_dict(self):
        ws = AgentWorkspace(agent_id="gemini")
        ws.memory["x"] = 1
        ws.state["y"] = 2
        d = ws.to_dict()
        assert d["agent_id"] == "gemini"
        assert d["memory"] == {"x": 1}
        assert d["state"] == {"y": 2}
        assert "created_at" in d


# ---------------------------------------------------------------------------
# DebateContext - Defaults
# ---------------------------------------------------------------------------


class TestDebateContextDefaults:
    def test_default_creation(self):
        ctx = DebateContext()
        assert ctx.env is not None
        assert ctx.agents == []
        assert ctx.domain == "general"
        assert ctx.debate_id == ""
        assert ctx.current_round == 0
        assert ctx.proposals == {}
        assert ctx.avg_novelty == 1.0

    def test_custom_fields(self):
        ctx = DebateContext(
            debate_id="test-123",
            domain="economics",
            session_id="sess-1",
            org_id="org-1",
        )
        assert ctx.debate_id == "test-123"
        assert ctx.domain == "economics"
        assert ctx.session_id == "sess-1"
        assert ctx.org_id == "org-1"


# ---------------------------------------------------------------------------
# DebateContext - Agent lookup
# ---------------------------------------------------------------------------


class TestGetAgentByName:
    def test_found(self):
        agent = MagicMock()
        agent.name = "claude"
        ctx = DebateContext(agents=[agent])
        assert ctx.get_agent_by_name("claude") is agent

    def test_not_found(self):
        ctx = DebateContext(agents=[])
        assert ctx.get_agent_by_name("missing") is None

    def test_multiple_agents(self):
        a1 = MagicMock()
        a1.name = "claude"
        a2 = MagicMock()
        a2.name = "gpt"
        ctx = DebateContext(agents=[a1, a2])
        assert ctx.get_agent_by_name("gpt") is a2


# ---------------------------------------------------------------------------
# DebateContext - Workspaces
# ---------------------------------------------------------------------------


class TestWorkspaces:
    def test_get_workspace_creates(self):
        ctx = DebateContext()
        ws = ctx.get_workspace("claude")
        assert isinstance(ws, AgentWorkspace)
        assert ws.agent_id == "claude"

    def test_get_workspace_returns_same(self):
        ctx = DebateContext()
        ws1 = ctx.get_workspace("claude")
        ws1.memory["key"] = "value"
        ws2 = ctx.get_workspace("claude")
        assert ws1 is ws2
        assert ws2.memory["key"] == "value"

    def test_get_workspace_isolation(self):
        ctx = DebateContext()
        ws_a = ctx.get_workspace("agent_a")
        ws_b = ctx.get_workspace("agent_b")
        ws_a.memory["shared"] = "a_value"
        assert "shared" not in ws_b.memory

    def test_clear_workspaces(self):
        ctx = DebateContext()
        ctx.get_workspace("a").memory["x"] = 1
        ctx.get_workspace("b").memory["y"] = 2
        ctx.clear_workspaces()
        assert ctx.agent_workspaces == {}


# ---------------------------------------------------------------------------
# DebateContext - Proposals
# ---------------------------------------------------------------------------


class TestProposals:
    def test_get_proposal_empty(self):
        ctx = DebateContext()
        assert ctx.get_proposal("claude") == ""

    def test_get_proposal_set(self):
        ctx = DebateContext()
        ctx.proposals["claude"] = "My proposal is..."
        assert ctx.get_proposal("claude") == "My proposal is..."


# ---------------------------------------------------------------------------
# DebateContext - Messages
# ---------------------------------------------------------------------------


class TestAddMessage:
    def test_adds_to_context_and_partial(self):
        ctx = DebateContext()
        msg = MagicMock()
        ctx.add_message(msg)
        assert msg in ctx.context_messages
        assert msg in ctx.partial_messages

    def test_adds_to_result_if_present(self):
        ctx = DebateContext()
        ctx.result = MagicMock()
        ctx.result.messages = []
        msg = MagicMock()
        ctx.add_message(msg)
        assert msg in ctx.result.messages

    def test_no_result_ok(self):
        ctx = DebateContext()
        ctx.result = None
        msg = MagicMock()
        ctx.add_message(msg)  # Should not raise
        assert msg in ctx.context_messages


# ---------------------------------------------------------------------------
# DebateContext - Agent failures
# ---------------------------------------------------------------------------


class TestRecordAgentFailure:
    def test_records_failure(self):
        ctx = DebateContext()
        ctx.record_agent_failure("claude", "proposal", "timeout", "timed out after 30s")
        assert "claude" in ctx.agent_failures
        assert len(ctx.agent_failures["claude"]) == 1
        record = ctx.agent_failures["claude"][0]
        assert record["phase"] == "proposal"
        assert record["error_type"] == "timeout"
        assert record["message"] == "timed out after 30s"
        assert "timestamp" in record

    def test_records_multiple_failures(self):
        ctx = DebateContext()
        ctx.record_agent_failure("claude", "proposal", "timeout", "msg1")
        ctx.record_agent_failure("claude", "critique", "api_error", "msg2")
        assert len(ctx.agent_failures["claude"]) == 2

    def test_empty_agent_name_defaults(self):
        ctx = DebateContext()
        ctx.record_agent_failure("", "proposal", "error", "msg")
        assert "unknown" in ctx.agent_failures

    def test_provider_field(self):
        ctx = DebateContext()
        ctx.record_agent_failure("claude", "proposal", "error", "msg", provider="anthropic")
        assert ctx.agent_failures["claude"][0]["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# DebateContext - Critiques
# ---------------------------------------------------------------------------


class TestAddCritique:
    def test_adds_to_round_and_partial(self):
        ctx = DebateContext()
        critique = MagicMock()
        ctx.add_critique(critique)
        assert critique in ctx.round_critiques
        assert critique in ctx.partial_critiques

    def test_adds_to_result_if_present(self):
        ctx = DebateContext()
        ctx.result = MagicMock()
        ctx.result.critiques = []
        critique = MagicMock()
        ctx.add_critique(critique)
        assert critique in ctx.result.critiques

    def test_critiques_property_alias(self):
        ctx = DebateContext()
        critique = MagicMock()
        ctx.add_critique(critique)
        assert ctx.critiques is ctx.round_critiques
        assert critique in ctx.critiques


# ---------------------------------------------------------------------------
# DebateContext - Finalize result
# ---------------------------------------------------------------------------


class TestFinalizeResult:
    def test_finalize_sets_duration(self):
        import time

        ctx = DebateContext()
        ctx.start_time = time.time() - 5.0
        ctx.result = MagicMock()
        ctx.result.rounds_used = 0
        ctx.result.status = ""
        ctx.result.consensus_reached = False
        ctx.current_round = 3
        ctx.agents = []

        ctx.finalize_result()
        assert ctx.result.duration_seconds >= 4.0

    def test_finalize_sets_rounds(self):
        ctx = DebateContext()
        ctx.start_time = 0
        ctx.current_round = 4
        ctx.result = MagicMock()
        ctx.result.rounds_used = 0
        ctx.result.status = ""
        ctx.result.consensus_reached = True
        ctx.agents = []

        ctx.finalize_result()
        assert ctx.result.rounds_used == 4
        assert ctx.result.rounds_completed == 4

    def test_finalize_sets_winner(self):
        ctx = DebateContext()
        ctx.start_time = 0
        ctx.winner_agent = "claude"
        ctx.result = MagicMock()
        ctx.result.rounds_used = 1
        ctx.result.status = ""
        ctx.result.consensus_reached = True
        ctx.agents = []
        ctx.current_round = 1

        ctx.finalize_result()
        assert ctx.result.winner == "claude"

    def test_finalize_copies_novelty(self):
        ctx = DebateContext()
        ctx.start_time = 0
        ctx.per_agent_novelty = {"claude": [0.8, 0.6]}
        ctx.avg_novelty = 0.7
        ctx.result = MagicMock()
        ctx.result.rounds_used = 1
        ctx.result.status = ""
        ctx.result.consensus_reached = False
        ctx.agents = []
        ctx.current_round = 1

        ctx.finalize_result()
        assert ctx.result.per_agent_novelty == {"claude": [0.8, 0.6]}
        assert ctx.result.avg_novelty == 0.7

    def test_finalize_status_consensus(self):
        ctx = DebateContext()
        ctx.start_time = 0
        ctx.result = MagicMock()
        ctx.result.rounds_used = 1
        ctx.result.status = ""
        ctx.result.consensus_reached = True
        ctx.agents = []
        ctx.current_round = 1

        ctx.finalize_result()
        assert ctx.result.status == "consensus_reached"

    def test_finalize_status_completed(self):
        ctx = DebateContext()
        ctx.start_time = 0
        ctx.result = MagicMock()
        ctx.result.rounds_used = 1
        ctx.result.status = ""
        ctx.result.consensus_reached = False
        ctx.agents = []
        ctx.current_round = 1

        ctx.finalize_result()
        assert ctx.result.status == "completed"

    def test_finalize_preserves_existing_status(self):
        ctx = DebateContext()
        ctx.start_time = 0
        ctx.result = MagicMock()
        ctx.result.rounds_used = 1
        ctx.result.status = "cancelled"
        ctx.result.consensus_reached = False
        ctx.agents = []
        ctx.current_round = 1

        ctx.finalize_result()
        assert ctx.result.status == "cancelled"

    def test_finalize_sets_debate_id(self):
        ctx = DebateContext()
        ctx.start_time = 0
        ctx.debate_id = "debate-xyz"
        ctx.result = MagicMock()
        ctx.result.rounds_used = 1
        ctx.result.status = ""
        ctx.result.consensus_reached = False
        ctx.agents = []
        ctx.current_round = 1

        ctx.finalize_result()
        assert ctx.result.debate_id == "debate-xyz"
        assert ctx.result.id == "debate-xyz"

    def test_finalize_none_result(self):
        ctx = DebateContext()
        ctx.result = None
        r = ctx.finalize_result()
        assert r is None


# ---------------------------------------------------------------------------
# DebateContext - Summary dict
# ---------------------------------------------------------------------------


class TestToSummaryDict:
    def test_summary_dict(self):
        agent = MagicMock()
        agent.name = "claude"
        ctx = DebateContext(
            debate_id="d1",
            domain="tech",
            agents=[agent],
            proposers=[agent],
        )
        ctx.proposals["claude"] = "text"
        ctx.winner_agent = "claude"

        d = ctx.to_summary_dict()
        assert d["debate_id"] == "d1"
        assert d["domain"] == "tech"
        assert d["agents"] == ["claude"]
        assert d["proposers"] == ["claude"]
        assert d["num_proposals"] == 1
        assert d["winner"] == "claude"


# ---------------------------------------------------------------------------
# DebateContext - Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_is_cancelled_no_token(self):
        ctx = DebateContext()
        assert ctx.is_cancelled is False

    def test_is_cancelled_with_token(self):
        token = MagicMock()
        token.is_cancelled = True
        ctx = DebateContext(cancellation_token=token)
        assert ctx.is_cancelled is True

    def test_check_cancellation_no_token(self):
        ctx = DebateContext()
        ctx.check_cancellation()  # Should not raise

    def test_check_cancellation_with_token(self):
        token = MagicMock()
        ctx = DebateContext(cancellation_token=token)
        ctx.check_cancellation()
        token.check.assert_called_once()
