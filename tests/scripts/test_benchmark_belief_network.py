import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_belief_network.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_belief_network", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_report_rejects_default_zero_results() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="requires at least one claim"):
        module.generate_report(module.BenchmarkResults())


def test_generate_report_rejects_crux_count_without_details() -> None:
    module = _load_module()
    results = module.BenchmarkResults(
        total_claims=2,
        total_relations=1,
        cruxes_found=1,
    )

    with pytest.raises(ValueError, match="crux count must match crux details"):
        module.generate_report(results)


def test_generate_report_accepts_minimal_valid_results() -> None:
    module = _load_module()
    results = module.BenchmarkResults(
        total_claims=2,
        total_relations=1,
        graph_density=0.5,
        propagation_converged=True,
        propagation_iterations=3,
        cruxes_found=1,
        crux_details=[
            {
                "statement": "Cost uncertainty drives the decision.",
                "author": "cfo",
                "crux_score": 0.9,
                "influence_score": 0.8,
                "disagreement_score": 0.7,
                "uncertainty_score": 0.6,
                "centrality_score": 0.5,
                "affected_claims": 2,
            }
        ],
        consensus_probability=0.7,
        contested_claims=1,
    )

    report = module.generate_report(results)

    assert "**Claims analyzed:** 2" in report
    assert "**Crux claims identified:** 1" in report
    assert "Cost uncertainty drives the decision." in report
