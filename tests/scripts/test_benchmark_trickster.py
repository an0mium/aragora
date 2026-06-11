import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_trickster.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_trickster", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_benchmark_rejects_empty_case_collection() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="requires at least one test case"):
        asyncio.run(module.run_benchmark([], rounds=2, seed=42))


def test_run_benchmark_rejects_non_positive_round_count() -> None:
    module = _load_module()
    test_case = module.TestCase(
        question="Should we use a simple benchmark guard?",
        category=module.CATEGORY_CLEAR,
    )

    with pytest.raises(ValueError, match="rounds must be a positive integer"):
        asyncio.run(module.run_benchmark([test_case], rounds=0, seed=42))


def test_run_benchmark_rejects_boolean_round_count() -> None:
    module = _load_module()
    test_case = module.TestCase(
        question="Should we use a simple benchmark guard?",
        category=module.CATEGORY_CLEAR,
    )

    with pytest.raises(ValueError, match="rounds must be a positive integer"):
        asyncio.run(module.run_benchmark([test_case], rounds=True, seed=42))


def test_cli_rejects_non_positive_round_count_before_running() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--rounds", "0", "--quick"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be a positive integer" in result.stderr
    assert "Trickster A/B Benchmark" not in result.stdout


def test_generate_report_rejects_empty_results() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="requires at least one result"):
        module.generate_report([], duration=0.0)


def test_generate_report_accepts_single_result() -> None:
    module = _load_module()
    test_case = module.TestCase(
        question="Should we use a simple benchmark guard?",
        category=module.CATEGORY_CLEAR,
    )
    result = module.ABResult(
        test_case=test_case,
        with_trickster=module.RunMetrics(
            confidence=0.8,
            consensus_reached=True,
            trickster_interventions=1,
            final_similarity=0.7,
            evidence_quality_avg=0.6,
        ),
        without_trickster=module.RunMetrics(
            confidence=0.6,
            consensus_reached=False,
            final_similarity=0.5,
            evidence_quality_avg=0.4,
        ),
    )

    report = module.generate_report([result], duration=1.25)

    assert "**Test cases:** 1" in report
    assert "| Consensus rate | 1/1 (100%) | 0/1 (0%) | +1.000 |" in report
