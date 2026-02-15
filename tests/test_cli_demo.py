"""
Tests for CLI demo module.

Tests demo task configuration and listing.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

pytest.importorskip(
    "aragora_debate", reason="aragora_debate package is required for CLI demo tests"
)

from aragora.cli.demo import (
    DEMO_TASKS,
    _DEFAULT_DEMO,
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
        """All demos have topic and description."""
        for name, demo in DEMO_TASKS.items():
            assert "topic" in demo, f"Demo '{name}' missing 'topic'"
            assert "description" in demo, f"Demo '{name}' missing 'description'"

    def test_all_demos_have_string_topics(self):
        """All demo topics are strings."""
        for name, demo in DEMO_TASKS.items():
            assert isinstance(demo["topic"], str), f"Demo '{name}' topic not a string"
            assert len(demo["topic"]) > 10, f"Demo '{name}' topic too short"

    def test_rate_limiter_demo_exists(self):
        """The rate-limiter demo exists."""
        assert "rate-limiter" in DEMO_TASKS

    def test_default_demo_exists(self):
        """The default demo exists."""
        assert _DEFAULT_DEMO in DEMO_TASKS


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
    """Tests for run_demo function."""

    def test_unknown_demo_prints_error(self, capsys):
        """Unknown demo name prints error."""
        run_demo("nonexistent_demo_xyz")

        captured = capsys.readouterr()
        assert "Unknown demo" in captured.out
        assert "nonexistent_demo_xyz" in captured.out

    def test_unknown_demo_shows_available(self, capsys):
        """Unknown demo shows available demos."""
        run_demo("nonexistent")

        captured = capsys.readouterr()
        # Should list at least one available demo
        for demo_name in DEMO_TASKS.keys():
            if demo_name in captured.out:
                return
        pytest.fail("Available demos not shown")


class TestMain:
    """Tests for main CLI function."""

    @patch("aragora.cli.demo.run_demo")
    def test_main_calls_run_demo(self, mock_run_demo):
        """Main function calls run_demo."""
        args = argparse.Namespace(
            name="rate-limiter",
            list_demos=False,
            server=False,
            topic=None,
        )
        main(args)
        mock_run_demo.assert_called_once_with("rate-limiter", receipt_path=None)

    @patch("aragora.cli.demo.run_demo")
    def test_main_defaults_to_default_demo(self, mock_run_demo):
        """Main function defaults to the default demo."""
        args = argparse.Namespace(
            name=None,
            list_demos=False,
            server=False,
            topic=None,
        )
        main(args)
        mock_run_demo.assert_called_once_with(_DEFAULT_DEMO, receipt_path=None)

    @patch("aragora.cli.demo.run_demo")
    def test_main_with_custom_demo(self, mock_run_demo):
        """Main function uses specified demo."""
        args = argparse.Namespace(
            name="auth",
            list_demos=False,
            server=False,
            topic=None,
        )
        main(args)
        mock_run_demo.assert_called_once_with("auth", receipt_path=None)

    def test_main_list_flag(self, capsys):
        """Main function lists demos with --list flag."""
        args = argparse.Namespace(
            name=None,
            list_demos=True,
            server=False,
            topic=None,
        )
        main(args)
        captured = capsys.readouterr()
        assert "Available demos:" in captured.out


class TestDemoTaskContent:
    """Tests for demo task content quality."""

    def test_topics_are_meaningful_questions(self):
        """Demo topics are meaningful questions or design tasks."""
        for name, demo in DEMO_TASKS.items():
            topic = demo["topic"]
            keywords = ["design", "should", "implement", "create", "build", "how", "migrate"]
            has_keyword = any(kw in topic.lower() for kw in keywords)
            assert has_keyword, f"Demo '{name}' topic doesn't look like a question"

    def test_topics_have_reasonable_length(self):
        """Demo topics are reasonably long."""
        for name, demo in DEMO_TASKS.items():
            topic = demo["topic"]
            assert 20 < len(topic) < 500, f"Demo '{name}' topic length out of range"

    def test_demo_names_are_descriptive(self):
        """Demo names are lowercase with hyphens."""
        for name in DEMO_TASKS.keys():
            assert name == name.lower(), f"Demo name '{name}' should be lowercase"
            assert " " not in name, f"Demo name '{name}' should not have spaces"
