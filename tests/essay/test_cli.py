"""Tests for the 'aragora essay' CLI command."""

from __future__ import annotations

import argparse
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal Namespace for essay_command tests."""
    defaults = {
        "essay_subcommand": None,
        "input": None,
        "output": None,
        "rounds": 3,
        "models": None,
        "target_words": 1200,
        "voice_notes": None,
        "rubric": None,
        "dry_run": False,
        "resume": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Test 1: refine --dry-run
# ---------------------------------------------------------------------------


def test_essay_command_dry_run(tmp_path: Path) -> None:
    """essay refine --dry-run should call pipeline.run with dry_run=True."""
    ideas_file = tmp_path / "ideas.md"
    ideas_file.write_text("Some raw brainstorm notes about education and AI.", encoding="utf-8")

    extraction_result = {
        "thesis": "AI will reshape education.",
        "outline": "1. Introduction\n2. Evidence\n3. Conclusion",
        "raw_extraction": "full LLM output",
    }

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run = AsyncMock(return_value=extraction_result)

    mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)

    args = _make_args(
        essay_subcommand="refine",
        input=str(ideas_file),
        dry_run=True,
    )

    with patch("aragora.cli.commands.essay.EssayRefinementPipeline", mock_pipeline_cls):
        from aragora.cli.commands.essay import essay_command

        essay_command(args)

    # pipeline.run must have been awaited with dry_run=True
    mock_pipeline_instance.run.assert_awaited_once()
    call_kwargs = mock_pipeline_instance.run.call_args
    assert call_kwargs.kwargs.get("dry_run") is True


# ---------------------------------------------------------------------------
# Test 2: score subcommand
# ---------------------------------------------------------------------------


def test_essay_score_subcommand(tmp_path: Path) -> None:
    """essay score should call evaluate_essay with the draft contents."""
    draft_file = tmp_path / "draft.md"
    draft_content = "This is a well-structured essay about technology and society."
    draft_file.write_text(draft_content, encoding="utf-8")

    mock_score = MagicMock()
    mock_score.overall = 0.82
    mock_score.thesis_clarity = 0.85
    mock_score.argument_coherence = 0.80
    mock_score.evidence_grounding = 0.75
    mock_score.rhetorical_force = 0.80
    mock_score.concision = 0.85
    mock_score.factual_accuracy = 0.90
    mock_score.originality = 0.75
    mock_score.severity_notes = []
    mock_score.suggestions = ["Add more concrete examples."]
    mock_score.weakest_paragraph = None

    mock_evaluate = AsyncMock(return_value=mock_score)
    mock_load_rubric = MagicMock(return_value={"weights": {}})
    mock_agent = MagicMock()
    mock_create_agent = MagicMock(return_value=mock_agent)

    args = _make_args(
        essay_subcommand="score",
        input=str(draft_file),
    )

    with (
        patch("aragora.cli.commands.essay.evaluate_essay", mock_evaluate),
        patch("aragora.cli.commands.essay.load_rubric", mock_load_rubric),
        patch("aragora.cli.commands.essay.create_agent", mock_create_agent),
    ):
        from aragora.cli.commands.essay import essay_command

        essay_command(args)

    mock_evaluate.assert_awaited_once()
    call_args = mock_evaluate.call_args
    # First positional arg must be the draft content
    assert call_args.args[0] == draft_content
