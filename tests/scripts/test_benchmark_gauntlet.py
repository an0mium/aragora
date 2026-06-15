"""Tests for scripts/benchmark_gauntlet.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_gauntlet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_gauntlet", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _decision(module, *, name: str, category: str, actual_verdict: str):
    return module.DecisionResult(
        name=name,
        category=category,
        expected_verdict=actual_verdict,
        actual_verdict=actual_verdict,
        verdict_correct=True,
        findings_count=0,
        critical_count=0,
        high_count=0,
        medium_count=0,
        low_count=0,
        robustness_score=1.0,
        confidence=0.95,
        verdict_reasoning="fixture verdict",
    )


def test_generate_report_rejects_empty_decision_corpus() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="requires at least one decision"):
        module.generate_report(module.BenchmarkResults())


def test_generate_report_requires_strong_and_weak_categories() -> None:
    module = _load_module()
    results = module.BenchmarkResults(
        decisions=[
            _decision(
                module,
                name="Only strong",
                category="strong",
                actual_verdict="pass",
            )
        ]
    )

    with pytest.raises(ValueError, match="requires both strong and weak"):
        module.generate_report(results)


def test_generate_report_rejects_unsupported_decision_categories() -> None:
    module = _load_module()
    results = module.BenchmarkResults(
        decisions=[
            _decision(
                module,
                name="Strong",
                category="strong",
                actual_verdict="pass",
            ),
            _decision(
                module,
                name="Weak",
                category="weak",
                actual_verdict="fail",
            ),
            _decision(
                module,
                name="Unreported",
                category="neutral",
                actual_verdict="conditional",
            ),
        ]
    )

    with pytest.raises(ValueError, match="unsupported: neutral"):
        module.generate_report(results)


def test_generate_report_accepts_minimal_balanced_corpus() -> None:
    module = _load_module()
    results = module.BenchmarkResults(
        decisions=[
            _decision(
                module,
                name="Strong",
                category="strong",
                actual_verdict="pass",
            ),
            _decision(
                module,
                name="Weak",
                category="weak",
                actual_verdict="fail",
            ),
        ]
    )

    report = module.generate_report(results)

    assert "Verdict accuracy" in report
    assert "Strong decisions averaged" in report
