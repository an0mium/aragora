"""Tests for scripts/run_dogfood_ab_pairs.py benchmark input validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import run_dogfood_ab_pairs  # noqa: E402


def test_parse_args_accepts_positive_pair_count() -> None:
    args = run_dogfood_ab_pairs.parse_args(["--pairs", "1"])

    assert args.pairs == 1


def test_main_rejects_zero_pairs_before_writing_output(tmp_path: Path) -> None:
    output_root = tmp_path / "dogfood-ab-output"

    with pytest.raises(SystemExit):
        run_dogfood_ab_pairs.main(["--pairs", "0", "--output-root", str(output_root)])

    assert not output_root.exists()


def test_main_rejects_negative_pairs_before_writing_output(tmp_path: Path) -> None:
    output_root = tmp_path / "dogfood-ab-output"

    with pytest.raises(SystemExit):
        run_dogfood_ab_pairs.main(["--pairs", "-2", "--output-root", str(output_root)])

    assert not output_root.exists()
