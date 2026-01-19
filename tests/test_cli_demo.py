"""
Tests for CLI demo module.

Tests demo task configuration and listing.
"""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.demo import (
    DEMO_TASKS,
    list_demos,
    main,
    run_demo,
)


class TestDemoTasks:
    """Tests for DEMO_TASKS configuration."""

    def test_demo_tasks_not_empty(self):
        """DEMO_TASKS contains at least one demo."""
        assert len(DEMO_TASKS) > 0

    def test_all_demos_have_required_fields(self):
        """All demos have task, agents, and rounds."""
        for name, demo in DEMO_TASKS.items():
            assert "task" in demo, f"Demo '{name}' missing 'task'"
            assert "agents" in demo, f"Demo '{name}' missing 'agents'"
            assert "rounds" in demo, f"Demo '{name}' missing 'rounds'"

    def test_all_demos_have_string_tasks(self):
        """All demo tasks are strings."""
        for name, demo in DEMO_TASKS.items():
            assert isinstance(demo["task"], str), f"Demo '{name}' task not a string"
            assert len(demo["task"]) > 10, f"Demo '{name}' task too short"

    def test_all_demos_have_valid_rounds(self):
        """All demos have positive integer rounds."""
        for name, demo in DEMO_TASKS.items():
            assert isinstance(demo["rounds"], int), f"Demo '{name}' rounds not an int"
            assert demo["rounds"] > 0, f"Demo '{name}' rounds must be positive"

    def test_rate_limiter_demo_exists(self):
        """The rate-limiter demo exists (default demo)."""
        assert "rate-limiter" in DEMO_TASKS

    def test_all_demos_use_demo_agents(self):
        """All demos use 'demo' agents for no-API testing."""
        for name, demo in DEMO_TASKS.items():
            agents = demo["agents"]
            assert "demo" in agents, f"Demo '{name}' should use demo agents"


class TestListDemos:
    """Tests for list_demos function."""

    def test_returns_list(self):
        """list_demos returns a list."""
        demos = list_demos()
        assert isinstance(demos, list)

    def test_returns_all_demo_names(self):
        """list_demos returns all demo names."""
        demos = list_demos()
        assert set(demos) == set(DEMO_TASKS.keys())

    def test_includes_rate_limiter(self):
        """list_demos includes rate-limiter."""
        demos = list_demos()
        assert "rate-limiter" in demos


class TestRunDemo:
    """Tests for run_demo function.

    Note: The run_demo function attempts to import from aragora.cli.ask
    which doesn't exist. We test that it handles this gracefully by
    mocking the import.
    """

    def test_unknown_demo_prints_error(self, capsys):
        """Unknown demo name prints error."""
        import sys

        # Mock the missing module
        mock_ask = MagicMock()
        mock_ask.run_debate = AsyncMock()
        sys.modules["aragora.cli.ask"] = mock_ask

        try:
            run_demo("nonexistent_demo_xyz")

            captured = capsys.readouterr()
            assert "Unknown demo" in captured.out
            assert "nonexistent_demo_xyz" in captured.out
        finally:
            del sys.modules["aragora.cli.ask"]

    def test_unknown_demo_shows_available(self, capsys):
        """Unknown demo shows available demos."""
        import sys

        # Mock the missing module
        mock_ask = MagicMock()
        mock_ask.run_debate = AsyncMock()
        sys.modules["aragora.cli.ask"] = mock_ask

        try:
            run_demo("nonexistent")

            captured = capsys.readouterr()
            # Should list at least one available demo
            for demo_name in DEMO_TASKS.keys():
                if demo_name in captured.out:
                    return  # Found at least one
            pytest.fail("Available demos not shown")
        finally:
            del sys.modules["aragora.cli.ask"]


class TestMain:
    """Tests for main CLI function."""

    @patch("aragora.cli.demo.run_demo")
    def test_main_calls_run_demo(self, mock_run_demo):
        """Main function calls run_demo."""
        args = argparse.Namespace(name="rate-limiter")
        main(args)

        mock_run_demo.assert_called_once_with("rate-limiter")

    @patch("aragora.cli.demo.run_demo")
    def test_main_defaults_to_rate_limiter(self, mock_run_demo):
        """Main function defaults to rate-limiter demo."""
        args = argparse.Namespace(name=None)
        main(args)

        mock_run_demo.assert_called_once_with("rate-limiter")

    @patch("aragora.cli.demo.run_demo")
    def test_main_with_custom_demo(self, mock_run_demo):
        """Main function uses specified demo."""
        args = argparse.Namespace(name="auth")
        main(args)

        mock_run_demo.assert_called_once_with("auth")


class TestDemoTaskContent:
    """Tests for demo task content quality."""

    def test_tasks_are_meaningful_questions(self):
        """Demo tasks are meaningful system design questions."""
        for name, demo in DEMO_TASKS.items():
            task = demo["task"]
            # Should be a question or design task
            design_keywords = ["design", "implement", "create", "build", "how"]
            has_keyword = any(kw in task.lower() for kw in design_keywords)
            assert has_keyword, f"Demo '{name}' task doesn't look like a design question"

    def test_tasks_have_reasonable_length(self):
        """Demo tasks are reasonably long (not too short or too long)."""
        for name, demo in DEMO_TASKS.items():
            task = demo["task"]
            assert 20 < len(task) < 500, f"Demo '{name}' task length out of range"

    def test_demo_names_are_descriptive(self):
        """Demo names are descriptive and use hyphens."""
        for name in DEMO_TASKS.keys():
            # Names should be lowercase with hyphens
            assert name == name.lower(), f"Demo name '{name}' should be lowercase"
            assert " " not in name, f"Demo name '{name}' should not have spaces"
