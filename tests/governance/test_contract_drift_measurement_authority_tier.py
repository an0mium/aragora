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

FILE_ROOT_SUFFIXES = (".bak", ".old", ".pyx", "/child", "x")


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


def _assignment_line_span(source_path: Path, assignment_name: str) -> int:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id == assignment_name:
            return node.end_lineno - node.lineno
    raise AssertionError(f"{assignment_name} assignment not found in {source_path}")


def _canonical_matched_rule(path: str) -> str | None:
    for prefix in review_queue.TIER_4_PREFIXES:
        if review_queue._matches_prefix(path, (prefix,)):
            return prefix
    return None


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
    assert tier4_merge_train.SERIALIZED_TIER4_PREFIXES == review_queue.TIER_4_PREFIXES


@pytest.mark.parametrize("path", EXPECTED_AUTHORITY_PREFIXES)
def test_merge_train_matches_each_authority_root(path: str) -> None:
    assert tier4_merge_train.matches_serialized_path(path) == path


@pytest.mark.parametrize(
    ("path", "expected_rule"),
    [
        *((root, root) for root in review_queue.TIER_4_PREFIXES),
        *(
            (f"{root}{suffix}", None)
            for root in review_queue.TIER_4_PREFIXES
            if not root.endswith("/")
            for suffix in FILE_ROOT_SUFFIXES
        ),
        *(
            (f"{root}nested/authority.txt", root)
            for root in review_queue.TIER_4_PREFIXES
            if root.endswith("/")
        ),
    ],
)
def test_canonical_and_merge_train_boundary_matrix_match(
    path: str, expected_rule: str | None
) -> None:
    assert _canonical_matched_rule(path) == expected_rule
    assert tier4_merge_train.matches_serialized_path(path) == expected_rule
    tier, _name, _reason = review_queue._classify_model_review_tier([path])
    assert (tier == 4) == (expected_rule is not None)


@pytest.mark.parametrize(
    "path",
    [root.rstrip("/") for root in review_queue.TIER_4_PREFIXES if root.endswith("/")],
)
def test_directory_authority_requires_slash_boundary(path: str) -> None:
    assert _canonical_matched_rule(path) is None
    assert tier4_merge_train.matches_serialized_path(path) is None


@pytest.mark.parametrize(
    "path",
    (
        "aragora/knowledge/mound/metrics.py",
        "aragora/knowledge/mound/metrics_health_bridge.py",
    ),
)
def test_existing_tier2_metric_stem_paths_remain_tier2(path: str) -> None:
    tier, name, _reason = review_queue._classify_model_review_tier([path])
    assert tier == 2
    assert name == "tier_2_live_automation"


def test_tier2_metric_stem_matches_when_checked_as_single_rule() -> None:
    prefix = "aragora/knowledge/mound/metrics"
    assert review_queue._matches_prefix(f"{prefix}_health_bridge.py", (prefix,))


@pytest.mark.parametrize("path", REPORTING_ONLY_PATHS + UNRELATED_SIBLING_PATHS)
def test_reporting_and_unrelated_siblings_remain_below_tier4(path: str) -> None:
    tier, _name, _reason = review_queue._classify_model_review_tier([path])
    assert tier == 2, path
    assert tier4_merge_train.matches_serialized_path(path) is None
    assert path not in review_queue.CONTRACT_DRIFT_AUTHORITY_PREFIXES


def test_authority_constants_and_tuple_shape_remain_exact() -> None:
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
    assert _assignment_line_span(review_queue_path, "CONTRACT_DRIFT_AUTHORITY_PREFIXES") == 0


@pytest.mark.parametrize(
    "command_name",
    ("classify-tier", "authority-manifest", "boundary-validator"),
)
def test_review_queue_parser_rejects_unapproved_authority_commands(command_name: str) -> None:
    from aragora.cli.parser import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["review-queue", command_name])
    assert exc_info.value.code == 2
