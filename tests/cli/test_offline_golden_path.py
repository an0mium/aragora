"""Offline/demo golden-path behavior checks for CLI debate flows."""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_run_debate_offline_is_network_free(monkeypatch):
    """Offline mode should not attempt audience networking and should disable network-backed subsystems."""
    from aragora.cli.commands import debate as debate_cmd

    monkeypatch.setenv("ARAGORA_OFFLINE", "1")

    with (
        patch.object(debate_cmd, "create_agent", return_value=MagicMock(name="demo-agent")),
        patch.object(debate_cmd, "Arena") as mock_arena,
        patch.object(
            debate_cmd,
            "get_event_emitter_if_available",
            side_effect=AssertionError("should not probe network in offline mode"),
        ),
        patch.object(debate_cmd, "CritiqueStore") as mock_store,
    ):
        mock_result = MagicMock()
        mock_arena.return_value.run = AsyncMock(return_value=mock_result)

        await debate_cmd.run_debate(
            task="test offline",
            agents_str="demo",
            rounds=1,
            learn=True,
            enable_audience=True,
        )

        mock_store.assert_not_called()
        _, kwargs = mock_arena.call_args
        assert kwargs["knowledge_mound"] is None
        assert kwargs["auto_create_knowledge_mound"] is False
        assert kwargs["enable_knowledge_retrieval"] is False
        assert kwargs["enable_knowledge_ingestion"] is False
        assert kwargs["enable_cross_debate_memory"] is False
        assert kwargs["use_rlm_limiter"] is False
        assert kwargs["enable_ml_delegation"] is False
        assert kwargs["enable_quality_gates"] is False
        assert kwargs["enable_consensus_estimation"] is False


def test_cmd_ask_demo_forces_local_offline(monkeypatch):
    """Demo mode should always execute locally with offline-safe settings."""
    from aragora.cli.commands import debate as debate_cmd

    monkeypatch.delenv("ARAGORA_OFFLINE", raising=False)

    args = argparse.Namespace(
        task="smoke demo",
        agents="claude,openai",
        rounds=5,
        consensus="judge",
        context="",
        learn=True,
        db=":memory:",
        demo=True,
        api=False,
        local=False,
        graph=False,
        matrix=False,
        decision_integrity=False,
        auto_select=False,
        auto_select_config=None,
        enable_verticals=False,
        vertical=None,
        calibration=True,
        evidence_weighting=True,
        trending=True,
        mode=None,
        api_url="http://localhost:8080",
        api_key=None,
        verbose=False,
        graph_rounds=3,
        branch_threshold=0.7,
        max_branches=3,
        scenario=None,
        matrix_rounds=3,
        di_include_context=False,
        di_plan_strategy="single_task",
        di_execution_mode=None,
    )

    with patch.object(debate_cmd, "run_debate", new_callable=AsyncMock) as mock_run_debate:
        mock_result = MagicMock()
        mock_result.final_answer = "demo answer"
        mock_result.dissenting_views = []
        mock_run_debate.return_value = mock_result

        debate_cmd.cmd_ask(args)

        call_kwargs = mock_run_debate.call_args.kwargs
        assert call_kwargs["agents_str"] == "demo,demo,demo"
        assert call_kwargs["rounds"] == 2
        assert call_kwargs["learn"] is False
        assert call_kwargs["enable_audience"] is False
        assert call_kwargs["offline"] is True
        assert call_kwargs["protocol_overrides"]["enable_research"] is False
        assert call_kwargs["protocol_overrides"]["enable_llm_synthesis"] is False

    import os

    assert os.getenv("ARAGORA_OFFLINE") == "1"
