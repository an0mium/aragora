"""Focused guards for the Contract Drift measurement-authority Tier-4 constants."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aragora.cli.commands import review_queue
from scripts import tier4_merge_train


EXPECTED_AUTHORITY_PREFIXES = (
    "scripts/check_contract_drift_ratchet.py",
    "scripts/generate_contract_drift_inventory.py",
    "scripts/baselines/contract_drift_inventory.json",
    "scripts/sdk_path_normalize.py",
    "scripts/baselines/internal_route_prefixes.json",
    "scripts/baselines/contract_drift_program.json",
    "scripts/check_sdk_parity.py",
    "scripts/validate_openapi_routes.py",
)

REPORTING_ONLY_PATHS = (
    "scripts/contract_drift_report.py",
    "scripts/generate_contract_drift_backlog.py",
    "scripts/generate_contract_drift_issue_plan.py",
)

UNRELATED_SIBLING_PATHS = (
    "scripts/check_cross_sdk_parity.py",
    "scripts/sdk_parity_audit.py",
    "scripts/baselines/validate_openapi_routes.json",
    "scripts/baselines/check_sdk_parity.json",
)


def _assigned_string_literals(source_path: Path, assignment_name: str) -> tuple[str, ...]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != assignment_name:
            continue
        if not isinstance(node.value, ast.Tuple):
            raise AssertionError(f"{assignment_name} must be a literal tuple")
        values = tuple(
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
        if len(values) != len(node.value.elts):
            raise AssertionError(f"{assignment_name} must contain only string literals")
        return values
    raise AssertionError(f"{assignment_name} assignment not found in {source_path}")


def test_authority_roots_are_tier4() -> None:
    assert review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES == EXPECTED_AUTHORITY_PREFIXES
    assert review_queue.CONTRACT_DRIFT_AUTHORITY_TIER == 4
    for path in EXPECTED_AUTHORITY_PREFIXES:
        tier, name, _reason = review_queue._classify_model_review_tier([path])
        assert tier == 4, path
        assert name == "tier_4_preapproval_required", path


def test_classifier_and_merge_train_constants_match() -> None:
    assert tier4_merge_train.CONTRACT_DRIFT_AUTHORITY_PREFIXES == EXPECTED_AUTHORITY_PREFIXES
    assert (
        tier4_merge_train.CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION
        == review_queue.CONTRACT_DRIFT_AUTHORITY_POLICY_VERSION
    )
    assert (
        tier4_merge_train.CONTRACT_DRIFT_AUTHORITY_TIER
        == review_queue.CONTRACT_DRIFT_AUTHORITY_TIER
        == 4
    )
    assert (
        tier4_merge_train.CONTRACT_DRIFT_AUTHORITY_CANONICAL_SOURCE
        == "aragora.cli.commands.review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES"
    )
    assert set(tier4_merge_train.SERIALIZED_TIER4_PREFIXES) == set(review_queue.TIER_4_PREFIXES)


@pytest.mark.parametrize("path", EXPECTED_AUTHORITY_PREFIXES)
def test_merge_train_matches_each_authority_root(path: str) -> None:
    assert tier4_merge_train.matches_serialized_path(path) == path


@pytest.mark.parametrize("path", REPORTING_ONLY_PATHS + UNRELATED_SIBLING_PATHS)
def test_reporting_and_unrelated_siblings_remain_below_tier4(path: str) -> None:
    tier, _name, _reason = review_queue._classify_model_review_tier([path])
    assert tier == 2, path
    assert tier4_merge_train.matches_serialized_path(path) is None
    assert path not in review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES


def test_review_queue_changes_are_constants_only_for_cdg_classifier() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    review_queue_path = repo_root / "aragora/cli/commands/review_queue.py"
    merge_train_path = repo_root / "scripts/tier4_merge_train.py"

    assert (
        _assigned_string_literals(review_queue_path, "CONTRACT_DRIFT_AUTHORITY_PREFIXES")
        == EXPECTED_AUTHORITY_PREFIXES
    )
    assert (
        _assigned_string_literals(merge_train_path, "CONTRACT_DRIFT_AUTHORITY_PREFIXES")
        == EXPECTED_AUTHORITY_PREFIXES
    )

    parser_source = (repo_root / "aragora/cli/parser.py").read_text(encoding="utf-8")
    for command_name in ("classify-tier", "authority-manifest", "boundary-validator"):
        assert command_name not in parser_source
