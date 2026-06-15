"""Tests for scripts/benchmark_code_intelligence.py benchmark input guards."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import benchmark_code_intelligence  # noqa: E402

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_code_intelligence.py"


def test_cli_rejects_zero_iterations_before_emitting_results(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(tmp_path), "--iterations", "0"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be a positive integer" in result.stderr
    assert "CODE INTELLIGENCE BENCHMARK" not in result.stdout


def test_run_benchmarks_rejects_zero_iterations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        asyncio.run(
            benchmark_code_intelligence.run_benchmarks(
                str(tmp_path),
                iterations=0,
                benchmarks=["indexing"],
            )
        )


def test_run_benchmarks_rejects_bool_iterations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        asyncio.run(
            benchmark_code_intelligence.run_benchmarks(
                str(tmp_path),
                iterations=True,
                benchmarks=["indexing"],
            )
        )
