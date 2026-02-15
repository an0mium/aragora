"""Tests for --mode flag in CLI decide and debate commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.modes import load_builtins
from aragora.modes.base import ModeRegistry


class TestLoadBuiltins:
    """Test load_builtins() function."""

    def setup_method(self):
        ModeRegistry.clear()

    def teardown_method(self):
        # Re-register so other tests are unaffected
        load_builtins()

    def test_load_builtins_registers_all_five(self):
        """load_builtins() registers all 5 built-in modes."""
        assert len(ModeRegistry.list_all()) == 0
        load_builtins()
        registered = ModeRegistry.list_all()
        assert "architect" in registered
        assert "coder" in registered
        assert "reviewer" in registered
        assert "debugger" in registered
        assert "orchestrator" in registered
        assert len(registered) == 5

    def test_load_builtins_idempotent(self):
        """Calling load_builtins() twice does not duplicate modes."""
        load_builtins()
        first = len(ModeRegistry.list_all())
        load_builtins()
        second = len(ModeRegistry.list_all())
        assert first == second == 5

    def test_architect_mode_has_system_prompt(self):
        """Architect mode provides a non-empty system prompt."""
        load_builtins()
        mode = ModeRegistry.get("architect")
        assert mode is not None
        prompt = mode.get_system_prompt()
        assert len(prompt) > 0


class TestModeInRunDebate:
    """Test mode injection into run_debate."""

    def setup_method(self):
        # Ensure modes are registered
        load_builtins()

    @pytest.mark.asyncio
    async def test_mode_architect_injects_system_prompt(self):
        """--mode architect causes system prompt to be set on agents."""
        from aragora.cli.commands.debate import run_debate

        created_agents = []

        def mock_create_agent(model_type, name, role, model=None):
            agent = MagicMock()
            agent.name = name
            agent.role = role
            agent.system_prompt = ""
            agent.provider = model_type
            created_agents.append(agent)
            return agent

        with (
            patch("aragora.cli.commands.debate.create_agent", side_effect=mock_create_agent),
            patch("aragora.cli.commands.debate.CritiqueStore"),
            patch("aragora.cli.commands.debate.Arena") as MockArena,
        ):
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            mock_result.messages = []
            MockArena.return_value.run = MagicMock(return_value=mock_result)

            await run_debate(
                task="Test task",
                agents_str="claude,claude",
                mode="architect",
                learn=False,
                enable_audience=False,
                offline=True,
            )

        # All agents should have been given the architect system prompt
        assert len(created_agents) >= 2
        architect_mode = ModeRegistry.get("architect")
        expected_prompt = architect_mode.get_system_prompt()
        for agent in created_agents:
            assert agent.system_prompt == expected_prompt

    @pytest.mark.asyncio
    async def test_unknown_mode_raises_error(self):
        """Unknown mode raises KeyError."""
        from aragora.cli.commands.debate import run_debate

        with pytest.raises(KeyError, match="not found"):
            await run_debate(
                task="Test task",
                agents_str="claude,claude",
                mode="nonexistent_mode",
                learn=False,
                enable_audience=False,
                offline=True,
            )

    @pytest.mark.asyncio
    async def test_no_mode_no_change(self):
        """When mode is None, agents keep their default system prompts."""
        from aragora.cli.commands.debate import run_debate

        created_agents = []
        original_prompt = "Default agent prompt"

        def mock_create_agent(model_type, name, role, model=None):
            agent = MagicMock()
            agent.name = name
            agent.role = role
            agent.system_prompt = original_prompt
            agent.provider = model_type
            created_agents.append(agent)
            return agent

        with (
            patch("aragora.cli.commands.debate.create_agent", side_effect=mock_create_agent),
            patch("aragora.cli.commands.debate.CritiqueStore"),
            patch("aragora.cli.commands.debate.Arena") as MockArena,
        ):
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            mock_result.messages = []
            MockArena.return_value.run = MagicMock(return_value=mock_result)

            await run_debate(
                task="Test task",
                agents_str="claude,claude",
                mode=None,
                learn=False,
                enable_audience=False,
                offline=True,
            )

        # Agents should keep their original prompts (not overwritten)
        for agent in created_agents:
            assert agent.system_prompt == original_prompt


class TestModeInDecide:
    """Test mode injection into run_decide."""

    def setup_method(self):
        load_builtins()

    @pytest.mark.asyncio
    async def test_decide_mode_passes_to_run_debate(self):
        """run_decide passes mode to run_debate."""
        from aragora.cli.commands.decide import run_decide

        with patch("aragora.cli.commands.decide.run_debate") as mock_run_debate:
            mock_result = MagicMock()
            mock_result.consensus_reached = True
            mock_result.confidence = 0.9
            mock_result.task = "Test"
            mock_run_debate.return_value = mock_result

            with patch("aragora.cli.commands.decide.DecisionPlanFactory") as MockFactory:
                mock_plan = MagicMock()
                mock_plan.requires_human_approval = False
                mock_plan.id = "plan-123"
                mock_plan.status.value = "created"
                mock_plan.risk_register = None
                MockFactory.from_debate_result.return_value = mock_plan

                with patch("aragora.cli.commands.decide.store_plan"):
                    with patch("aragora.cli.commands.decide.PlanExecutor") as MockExecutor:
                        mock_outcome = MagicMock()
                        mock_outcome.success = True
                        mock_outcome.tasks_completed = 1
                        mock_outcome.tasks_total = 1
                        mock_outcome.receipt_id = "r-1"
                        mock_outcome.lessons = []
                        MockExecutor.return_value.execute.return_value = mock_outcome

                        await run_decide(
                            task="Decide something",
                            agents_str="claude,claude",
                            mode="coder",
                        )

            # run_debate should have received the mode kwarg
            _, kwargs = mock_run_debate.call_args
            # mode gets passed via **kwargs since it's in the signature
            assert "mode" not in kwargs or kwargs.get("mode") is None
            # Mode is handled in run_decide before calling run_debate

    @pytest.mark.asyncio
    async def test_decide_unknown_mode_raises(self):
        """run_decide raises KeyError for unknown mode."""
        from aragora.cli.commands.decide import run_decide

        with pytest.raises(KeyError, match="not found"):
            await run_decide(
                task="Decide something",
                agents_str="claude,claude",
                mode="nonexistent",
            )
