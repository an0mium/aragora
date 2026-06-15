import importlib.util
import math
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


def _minimal_valid_results(module):
    return module.BenchmarkResults(
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


def test_generate_report_rejects_impossible_probability_fields() -> None:
    module = _load_module()
    results = _minimal_valid_results(module)
    results.consensus_probability = 1.5

    with pytest.raises(ValueError, match="consensus_probability must be between 0 and 1"):
        module.generate_report(results)


def test_generate_report_rejects_non_finite_rate_fields() -> None:
    module = _load_module()
    results = _minimal_valid_results(module)
    results.graph_density = float("nan")

    with pytest.raises(ValueError, match="graph_density must be a finite non-negative value"):
        module.generate_report(results)


def test_generate_report_accepts_raw_maximum_average_uncertainty() -> None:
    module = _load_module()
    results = _minimal_valid_results(module)
    results.average_uncertainty = math.log2(3)

    report = module.generate_report(results)

    assert "| Average uncertainty | 1.585 |" in report


@pytest.mark.parametrize("invalid_uncertainty", [float("nan"), float("inf"), -0.01])
def test_generate_report_rejects_invalid_average_uncertainty(
    invalid_uncertainty: float,
) -> None:
    module = _load_module()
    results = _minimal_valid_results(module)
    results.average_uncertainty = invalid_uncertainty

    with pytest.raises(
        ValueError,
        match="average_uncertainty must be a finite non-negative value",
    ):
        module.generate_report(results)


def test_generate_report_rejects_contested_count_above_total_claims() -> None:
    module = _load_module()
    results = _minimal_valid_results(module)
    results.contested_claims = 3

    with pytest.raises(ValueError, match="contested_claims cannot exceed total_claims"):
        module.generate_report(results)


def test_generate_report_accepts_minimal_valid_results() -> None:
    module = _load_module()
    results = _minimal_valid_results(module)

    report = module.generate_report(results)

    assert "**Claims analyzed:** 2" in report
    assert "**Crux claims identified:** 1" in report
    assert "Cost uncertainty drives the decision." in report
