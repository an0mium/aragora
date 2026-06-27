"""Fixture-driven productization tests for the ``blocked_auth_failure``
rescue class scenario corpus (lane P76).

This module loads ``docs/benchmarks/auth_failure_scenarios.json`` and asserts
that every canonical auth-failure shape — 401 mid-tool-call, 403 quota
exceeded, missing-env-var preflight, expired-token-refresh-failed, and a
vendor-explicit-block — folds into ``TerminalClass.BLOCKED_AUTH_FAILURE`` via
the existing terminal-truth classifier in
``aragora.swarm.terminal_truth.classify_from_metrics``.

The tests are pure fixture exercises: no worker subprocesses, no network
calls, no AI key consumption. Together they lock the productization
contract so future B0 ticks of these shapes get classified deterministically
instead of needing per-tick human review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm.terminal_truth import TerminalClass, classify_from_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = REPO_ROOT / "docs" / "benchmarks" / "auth_failure_scenarios.json"

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "class",
    "description",
    "linkage",
    "scenarios",
}
REQUIRED_SCENARIO_KEYS = {
    "id",
    "description",
    "shape",
    "metrics_row",
    "expected_classification",
    "expected_terminal_truth_class",
}
REQUIRED_SHAPE_KEYS = {
    "trigger",
    "tool_name_pattern",
    "error_pattern",
    "agent_response_class",
}
REQUIRED_METRICS_ROW_KEYS = {
    "worker_status",
    "worker_outcome",
    "elapsed_seconds",
    "files_changed",
    "has_deliverable",
    "publish_action",
}


def _load_corpus() -> dict[str, Any]:
    payload = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "auth_failure_scenarios.json must be a JSON object"
    return payload


def _load_scenarios() -> list[dict[str, Any]]:
    scenarios = _load_corpus()["scenarios"]
    assert isinstance(scenarios, list)
    return scenarios


def _scenario_ids() -> list[str]:
    return [str(scenario["id"]) for scenario in _load_scenarios()]


def test_corpus_has_required_top_level_schema() -> None:
    """The scenario corpus must declare a stable top-level schema so
    downstream consumers (the rescue productization report, B0 status
    renderer, and operator dashboards) can parse it without a per-file
    schema sniff."""
    payload = _load_corpus()
    missing = REQUIRED_TOP_LEVEL_KEYS - set(payload.keys())
    assert not missing, f"auth_failure_scenarios.json missing top-level keys: {missing}"
    assert payload["class"] == "blocked_auth_failure"
    assert payload["schema_version"] == 1
    scenarios = payload["scenarios"]
    assert isinstance(scenarios, list)
    assert 3 <= len(scenarios) <= 10, (
        "Corpus must hold between 3 and 10 canonical shapes; outside that range "
        "indicates either an incomplete corpus or scope creep."
    )


@pytest.mark.parametrize("scenario_id", _scenario_ids())
def test_every_scenario_classifies_to_blocked_auth_failure(scenario_id: str) -> None:
    """Per-scenario invariant: the synthesizable boss_metrics row attached to
    each canonical auth-failure shape folds into ``BLOCKED_AUTH_FAILURE``.

    If this test fails for a given scenario id, the productization contract
    for that auth-failure shape has regressed and the classifier change that
    caused it should be reverted, or the scenario corpus updated alongside
    a deliberate taxonomy change."""
    scenario = next(item for item in _load_scenarios() if item["id"] == scenario_id)
    row = dict(scenario["metrics_row"])
    result = classify_from_metrics(row)
    assert result is TerminalClass.BLOCKED_AUTH_FAILURE, (
        f"scenario {scenario_id!r} (outcome={row.get('worker_outcome')!r}) "
        f"classified as {result.value!r}; expected blocked_auth_failure"
    )
    assert scenario["expected_classification"] == "blocked_auth_failure"
    assert scenario["expected_terminal_truth_class"] == "blocked_auth_failure"


def test_scenario_ids_are_unique() -> None:
    """Each scenario must carry a stable, unique id so downstream consumers
    can reference a specific shape (for example from a B0 tick receipt)
    without ambiguity."""
    ids = _scenario_ids()
    assert len(ids) == len(set(ids)), f"duplicate scenario ids: {ids}"


def test_every_scenario_satisfies_schema_integrity() -> None:
    """Locks the per-scenario schema: id/description/shape/metrics_row/
    expected_classification/expected_terminal_truth_class plus the nested
    shape and metrics_row keys. Catches accidental field drops during edits.
    """
    scenarios = _load_scenarios()
    for idx, scenario in enumerate(scenarios):
        missing_top = REQUIRED_SCENARIO_KEYS - set(scenario.keys())
        assert not missing_top, f"scenario[{idx}] missing scenario keys: {missing_top}"

        shape = scenario["shape"]
        assert isinstance(shape, dict), f"scenario[{idx}].shape must be a dict"
        missing_shape = REQUIRED_SHAPE_KEYS - set(shape.keys())
        assert not missing_shape, f"scenario[{idx}].shape missing keys: {missing_shape}"

        row = scenario["metrics_row"]
        assert isinstance(row, dict), f"scenario[{idx}].metrics_row must be a dict"
        missing_row = REQUIRED_METRICS_ROW_KEYS - set(row.keys())
        assert not missing_row, f"scenario[{idx}].metrics_row missing keys: {missing_row}"

        assert scenario["expected_classification"] == "blocked_auth_failure"
        assert scenario["expected_terminal_truth_class"] == "blocked_auth_failure"
        assert shape["agent_response_class"] == "halt_with_blocked_auth_failure"


def test_corpus_covers_required_canonical_triggers() -> None:
    """The corpus must collectively cover the canonical auth-failure
    triggers called out in the productization brief: 401 mid-tool-call,
    403/quota, missing env var, token refresh failed, and a vendor block.

    This guarantees the next B0 tick of any of these shapes has a regression
    anchor in the corpus already."""
    triggers = {scenario["shape"]["trigger"] for scenario in _load_scenarios()}
    required_triggers = {
        "tool_call_returned_401",
        "tool_call_returned_403",
        "missing_required_credential_env",
        "token_refresh_returned_auth_error",
        "vendor_explicit_credential_block",
    }
    missing = required_triggers - triggers
    assert not missing, f"corpus missing required canonical triggers: {missing}"


def test_linkage_points_to_canonical_modules() -> None:
    """The linkage block must reference the canonical classifier and the
    existing productization artefacts so this corpus stays discoverable
    from the rescue productization report and B0 status docs."""
    linkage = _load_corpus()["linkage"]
    assert linkage["classifier_module"] == "aragora.swarm.terminal_truth"
    assert linkage["classifier_function"] == "classify_from_metrics"
    assert linkage["terminal_class_enum"] == "TerminalClass.BLOCKED_AUTH_FAILURE"
    for key in (
        "existing_terminal_truth_fixture",
        "existing_productization_test",
        "rescue_productization_ledger",
        "b0_status_doc",
        "rescue_productization_pipeline",
    ):
        assert isinstance(linkage[key], str) and linkage[key], (
            f"linkage.{key} must be a non-empty string"
        )
