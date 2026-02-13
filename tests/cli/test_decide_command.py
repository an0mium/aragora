"""Tests for decision pipeline CLI command behavior."""

from __future__ import annotations

import argparse
from unittest.mock import patch


def _make_args(**overrides):
    base = {
        "task": "Test task",
        "agents": "codex,claude",
        "rounds": 3,
        "execution_mode": None,
        "computer_use": False,
        "hybrid": False,
        "fabric": False,
        "implementation_profile": None,
        "fabric_models": None,
        "channel_targets": None,
        "thread_id": None,
        "thread_id_by_platform": None,
        "auto_select": False,
        "auto_select_config": None,
        "context": None,
        "context_file": None,
        "document": None,
        "documents": None,
        "no_knowledge": False,
        "no_cross_memory": False,
        "enable_supermemory": False,
        "supermemory_container": None,
        "supermemory_max_items": None,
        "enable_belief_guidance": False,
        "auto_approve": False,
        "dry_run": True,
        "budget_limit": None,
        "verbose": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_decide_promotes_execution_mode_into_profile_when_missing():
    """--execution-mode should persist into implementation_profile for later execution."""
    from aragora.cli.commands import decide as decide_cmd

    args = _make_args(execution_mode="hybrid")

    with (
        patch.object(decide_cmd, "run_decide", return_value="coro") as mock_run_decide,
        patch.object(decide_cmd.asyncio, "run", return_value={}),
    ):
        decide_cmd.cmd_decide(args)

    kwargs = mock_run_decide.call_args.kwargs
    assert kwargs["execution_mode"] == "hybrid"
    assert kwargs["implementation_profile"] == {"execution_mode": "hybrid"}


def test_cmd_decide_does_not_override_profile_execution_mode():
    """Existing implementation_profile execution_mode should remain authoritative."""
    from aragora.cli.commands import decide as decide_cmd

    args = _make_args(
        execution_mode="hybrid",
        implementation_profile='{"execution_mode":"fabric","max_parallel":2}',
    )

    with (
        patch.object(decide_cmd, "run_decide", return_value="coro") as mock_run_decide,
        patch.object(decide_cmd.asyncio, "run", return_value={}),
    ):
        decide_cmd.cmd_decide(args)

    kwargs = mock_run_decide.call_args.kwargs
    assert kwargs["execution_mode"] == "hybrid"
    assert kwargs["implementation_profile"]["execution_mode"] == "fabric"
    assert kwargs["implementation_profile"]["max_parallel"] == 2


def test_cmd_decide_normalizes_execution_mode_alias() -> None:
    """Known execution-mode aliases should be normalized before run_decide."""
    from aragora.cli.commands import decide as decide_cmd

    args = _make_args(execution_mode="workflow_execute")

    with (
        patch.object(decide_cmd, "run_decide", return_value="coro") as mock_run_decide,
        patch.object(decide_cmd.asyncio, "run", return_value={}),
    ):
        decide_cmd.cmd_decide(args)

    kwargs = mock_run_decide.call_args.kwargs
    assert kwargs["execution_mode"] == "workflow"
    assert kwargs["implementation_profile"]["execution_mode"] == "workflow"
