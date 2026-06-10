"""Tests for the DIC-19 proof-unit corpus (docs/status/proof_units/).

Validates that every committed manifest loads through the schema and that the
constraint graph correctly resolves cross-unit claim relationships.  No network
access, no subprocess execution, no queue mutation.

Advances issue #6030 (DIC-19 — proof-carrying code unit constraint graph).
Flag: ARAGORA_PROOF_UNIT_SCAN_ENABLED (default OFF).  Tests that exercise the
directory scanner call enable_proof_unit_scan() directly so they never mutate
os.environ and always clean up via the autouse fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.epistemic.constraint_graph import ProofUnitConstraintGraph
from aragora.epistemic.proof_unit import (
    enable_proof_unit_scan,
    load_proof_unit_from_yaml,
    load_proof_units_from_dir,
    reset_proof_unit_scan,
)

_PROOF_UNITS_DIR = Path(__file__).parents[2] / "docs" / "status" / "proof_units"
_BENCHMARK_YAML = _PROOF_UNITS_DIR / "benchmark_truth_publication.yaml"
_SHIFT_YAML = _PROOF_UNITS_DIR / "proof_first_shift.yaml"
_SHARED_CLAIM = "b0.benchmark_truth.complete_current_corpus"


@pytest.fixture(autouse=True)
def _reset_scan() -> pytest.IterableFixture:  # type: ignore[type-arg]
    reset_proof_unit_scan()
    yield
    reset_proof_unit_scan()


class TestBenchmarkTruthPublicationManifest:
    """benchmark_truth_publication.yaml schema and field assertions."""

    def test_manifest_file_exists(self) -> None:
        assert _BENCHMARK_YAML.exists(), f"fixture missing: {_BENCHMARK_YAML}"

    def test_loads_without_error(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit is not None

    def test_code_unit_id(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit.code_unit_id == "benchmark_truth.publication.render"

    def test_source_path_points_to_real_script(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        repo_root = Path(__file__).parents[2]
        assert (repo_root / unit.source_path).exists(), (
            f"source_path {unit.source_path!r} does not exist in the repo"
        )

    def test_shared_claim_present(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert _SHARED_CLAIM in unit.claims

    def test_decay_policy_repair_required(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit.decay_policy.failed_claim == "repair_required"

    def test_fallback_policy_fail_closed(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit.fallback_policy.default == "fail_closed"

    def test_passes_schema_validation(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit.validate() == []

    def test_freshness_sla_at_least_one_hour(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit.freshness_sla_hours >= 1

    def test_owner_is_set(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit.owner

    def test_has_at_least_one_verifier(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit.verifiers

    def test_verifier_is_command_kind(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        kinds = [v.get("kind") for v in unit.verifiers]
        assert "command" in kinds

    def test_has_at_least_one_decision_receipt(self) -> None:
        unit = load_proof_unit_from_yaml(_BENCHMARK_YAML)
        assert unit.decision_receipts


class TestConstraintGraphCorpus:
    """Cross-unit impact analysis over the full proof_units corpus."""

    def test_corpus_has_two_units(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        assert len(units) == 2

    def test_corpus_contains_both_known_ids(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        ids = {u.code_unit_id for u in units}
        assert "proof_first.shift.green_criteria" in ids
        assert "benchmark_truth.publication.render" in ids

    def test_graph_builds_from_full_corpus(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        graph = ProofUnitConstraintGraph(units)
        assert graph.unit_count == 2

    def test_shared_claim_impacts_both_units(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        graph = ProofUnitConstraintGraph(units)
        impact = graph.impact_set([_SHARED_CLAIM])
        assert "proof_first.shift.green_criteria" in impact
        assert "benchmark_truth.publication.render" in impact

    def test_unrelated_claim_impacts_no_units(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        graph = ProofUnitConstraintGraph(units)
        assert graph.impact_set(["nonexistent.claim.xyz"]) == set()

    def test_impact_set_without_edges_equals_multi_hop(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        graph = ProofUnitConstraintGraph(units)
        direct = graph.impact_set([_SHARED_CLAIM])
        multi = graph.multi_hop_impact_set([_SHARED_CLAIM])
        assert direct == multi

    def test_to_dict_snapshot_includes_both_units(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        graph = ProofUnitConstraintGraph(units)
        snap = graph.to_dict()
        assert snap["unit_count"] == 2
        assert "benchmark_truth.publication.render" in snap["units"]
        assert "proof_first.shift.green_criteria" in snap["units"]

    def test_claim_index_contains_shared_claim(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        graph = ProofUnitConstraintGraph(units)
        snap = graph.to_dict()
        assert _SHARED_CLAIM in snap["claim_index"]
        assert len(snap["claim_index"][_SHARED_CLAIM]) == 2

    def test_no_dependency_edges_by_default(self) -> None:
        enable_proof_unit_scan()
        units = load_proof_units_from_dir(_PROOF_UNITS_DIR)
        graph = ProofUnitConstraintGraph(units)
        assert graph.edge_count == 0
