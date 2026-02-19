"""Tests for scripts/demo_pipeline.py — the Idea-to-Execution Pipeline demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure scripts/ is importable
_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import demo_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_run(
    ideas: list[str] | None = None,
    *,
    demo: bool = False,
    dry_run: bool = False,
    json_output: bool = False,
) -> tuple[int, str]:
    """Run main() with the given flags and capture stdout.

    Returns (exit_code, captured_stdout).
    """
    argv: list[str] = []
    if demo:
        argv.append("--demo")
    if dry_run:
        argv.append("--dry-run")
    if json_output:
        argv.append("--json")
    if ideas:
        argv.extend(ideas)

    with mock.patch("sys.stdout") as mock_stdout:
        lines: list[str] = []
        mock_stdout.write = lambda s: lines.append(s)
        mock_stdout.flush = lambda: None
        exit_code = demo_pipeline.main(argv)

    return exit_code, "".join(lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDemoPipelineWithDemoIdeas:
    """Test that --demo produces meaningful output."""

    def test_demo_ideas_produce_output(self) -> None:
        exit_code, output = _capture_run(demo=True, dry_run=True)
        assert exit_code == 0
        assert "Aragora Idea-to-Execution Pipeline" in output
        assert "Stage 1" in output
        assert "Stage 2" in output
        assert "Stage 3" in output
        assert "Provenance Chain" in output

    def test_demo_ideas_generate_nodes(self) -> None:
        exit_code, output = _capture_run(demo=True, dry_run=True)
        assert exit_code == 0
        # Should mention generated nodes at each stage
        assert "Generated" in output
        assert "nodes" in output


class TestDemoPipelineWithCustomIdeas:
    """Test that custom ideas work correctly."""

    def test_custom_ideas_work(self) -> None:
        exit_code, output = _capture_run(
            ideas=["Build a rate limiter", "Add caching"],
            dry_run=True,
        )
        assert exit_code == 0
        assert "Processing 2 ideas" in output
        assert "Stage 1" in output

    def test_single_idea_works(self) -> None:
        exit_code, output = _capture_run(
            ideas=["Build a rate limiter"],
            dry_run=True,
        )
        assert exit_code == 0
        assert "Processing 1 ideas" in output


class TestDemoPipelineDryRun:
    """Test that --dry-run skips execution."""

    def test_dry_run_skips_execution(self) -> None:
        exit_code, output = _capture_run(demo=True, dry_run=True)
        assert exit_code == 0
        assert "DRY RUN" in output
        assert "Skipping execution" in output

    def test_without_dry_run_mentions_server(self) -> None:
        exit_code, output = _capture_run(demo=True)
        assert exit_code == 0
        assert "Execution requires a running Aragora server" in output


class TestDemoPipelineJsonOutput:
    """Test that --json produces valid JSON."""

    def test_json_produces_valid_json(self) -> None:
        exit_code, output = _capture_run(
            ideas=["Build a rate limiter", "Add caching"],
            json_output=True,
            dry_run=True,
        )
        assert exit_code == 0

        # The output includes the header lines before JSON — extract the JSON block
        # Find the first '{' which starts the JSON object
        json_start = output.index("{")
        json_str = output[json_start:]
        parsed = json.loads(json_str)

        assert "input_ideas" in parsed
        assert "stages" in parsed
        assert "summary" in parsed
        assert parsed["summary"]["ideas"] == 2
        assert parsed["dry_run"] is True

    def test_json_has_all_stages(self) -> None:
        exit_code, output = _capture_run(
            ideas=["Build a rate limiter"],
            json_output=True,
        )
        assert exit_code == 0

        json_start = output.index("{")
        parsed = json.loads(output[json_start:])

        stages = parsed["stages"]
        assert "ideas_to_goals" in stages
        assert "goals_to_tasks" in stages
        assert "tasks_to_workflow" in stages

        # Each stage should have nodes, edges, provenance
        for stage_name, stage_data in stages.items():
            assert "nodes" in stage_data, f"{stage_name} missing nodes"
            assert "edges" in stage_data, f"{stage_name} missing edges"
            assert "provenance" in stage_data, f"{stage_name} missing provenance"


class TestDemoPipelineEmptyIdeas:
    """Test that empty ideas shows help."""

    def test_no_args_shows_help(self) -> None:
        """No arguments should print help and return exit code 1."""
        exit_code, output = _capture_run()
        assert exit_code == 1

    def test_empty_list_shows_help(self) -> None:
        """Explicit empty list should print help and return exit code 1."""
        exit_code, output = _capture_run(ideas=[])
        assert exit_code == 1
