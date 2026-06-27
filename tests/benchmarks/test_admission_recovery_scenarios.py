"""Validate the A2 admission-recovery productization fixture for #7209.

This test suite locks the productization story rendered by:
- ``docs/benchmarks/admission_recovery_scenarios.json`` (PR-3 fixture)
- the matching ``admission_class_credential_envelope_synthesis_v1`` entry in
  ``docs/benchmarks/rescue_productization.json``

The fixture is intentionally additive over the live #7225 / #7228 / #7248
productization landings: it records the canonical admission-recovery
scenarios so the rescue-map's ``target_kind=fixture`` linkage has a real
artefact to point at, and so future regressions can be caught by holding
the scenarios' shape stable.

No code paths are exercised here; the fixture is data + docs only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "docs" / "benchmarks" / "admission_recovery_scenarios.json"
RESCUE_MAP_PATH = REPO_ROOT / "docs" / "benchmarks" / "rescue_productization.json"
CORPUS_PATH = REPO_ROOT / "docs" / "benchmarks" / "corpus.json"

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(scope="module")
def fixture_payload() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rescue_map_payload() -> dict[str, Any]:
    return json.loads(RESCUE_MAP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus_payload() -> dict[str, Any]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def test_fixture_has_expected_top_level_shape(fixture_payload: dict[str, Any]) -> None:
    assert fixture_payload["schema_version"] == 1
    assert fixture_payload["tracking_issue"] == "#7209"
    assert isinstance(fixture_payload.get("story"), str) and fixture_payload["story"]
    assert isinstance(fixture_payload.get("scenarios"), list)
    assert len(fixture_payload["scenarios"]) >= 1
    assert isinstance(fixture_payload.get("non_goals"), list)
    assert isinstance(fixture_payload.get("verification_paths"), list)


def test_fixture_scenarios_have_required_fields(fixture_payload: dict[str, Any]) -> None:
    required = {
        "id",
        "summary",
        "admission_path",
        "failure_class_before",
        "feature_flag",
        "feature_flag_default",
        "feature_pr",
        "rescue_map_class",
        "reference_issues",
        "expected_post_admission_signal",
    }
    scenario_ids: set[str] = set()
    for scenario in fixture_payload["scenarios"]:
        missing = required - scenario.keys()
        assert not missing, f"scenario {scenario.get('id')} missing fields: {sorted(missing)}"
        scenario_ids.add(scenario["id"])
        assert scenario["feature_flag_default"] == "off", (
            f"productized feature flag must default OFF: {scenario['id']}"
        )
        assert isinstance(scenario["reference_issues"], list)
        assert scenario["reference_issues"], (
            f"scenario {scenario['id']} must list at least one reference_issue"
        )
    assert len(scenario_ids) == len(fixture_payload["scenarios"]), (
        "scenario ids must be unique within the fixture"
    )


def test_fixture_documents_pr1_corpus_aware_dispatch_admission(
    fixture_payload: dict[str, Any],
) -> None:
    scenario = next(
        item
        for item in fixture_payload["scenarios"]
        if item["id"] == "corpus_aware_dispatch_admission"
    )
    assert scenario["failure_class_before"] == "blocked_not_dispatch_bounded"
    assert scenario["admission_path"] == "corpus_aware_dispatch"
    assert scenario["feature_flag"] == "ARAGORA_CORPUS_AWARE_DISPATCH"
    assert scenario["feature_pr"] == "#7225"
    assert scenario.get("repair_pr") == "#7228"
    assert scenario["rescue_map_class"] == "admission_class_corpus_synthesis_v1"


def test_fixture_documents_pr2_credential_envelope_synthesis_admission(
    fixture_payload: dict[str, Any],
) -> None:
    scenario = next(
        item
        for item in fixture_payload["scenarios"]
        if item["id"] == "credential_envelope_corpus_synthesis_admission"
    )
    assert scenario["failure_class_before"] == "blocked_auth_failure"
    assert scenario["admission_path"] == "credential_envelope_corpus_synthesis"
    assert scenario["feature_flag"] == "ARAGORA_CREDENTIAL_ENVELOPE_PROBE"
    assert scenario["feature_pr"] == "#7248"
    assert scenario["rescue_map_class"] == "admission_class_credential_envelope_synthesis_v1"
    assert scenario.get("preconditions"), "PR-2 scenario must enumerate preconditions"
    assert "default OFF" in scenario.get("behavior_when_flag_off", "") or (
        "no-op" in scenario.get("behavior_when_flag_off", "")
    )


def test_reference_issues_match_rev4_corpus_membership(
    fixture_payload: dict[str, Any], corpus_payload: dict[str, Any]
) -> None:
    corpus_by_id: dict[int, dict[str, Any]] = {
        int(entry["issue_id"]): entry for entry in corpus_payload.get("issues", [])
    }
    for scenario in fixture_payload["scenarios"]:
        for ref in scenario["reference_issues"]:
            issue_id = int(ref["issue_id"])
            assert issue_id in corpus_by_id, (
                f"reference issue #{issue_id} in scenario {scenario['id']} is not in "
                "docs/benchmarks/corpus.json"
            )
            corpus_entry = corpus_by_id[issue_id]
            assert corpus_entry["execution_class"] == ref["execution_class"], (
                f"reference issue #{issue_id} execution_class drifted: "
                f"fixture={ref['execution_class']} corpus={corpus_entry['execution_class']}"
            )
            assert corpus_entry["added_in_revision"] == ref["added_in_revision"], (
                f"reference issue #{issue_id} added_in_revision drifted"
            )


def test_rescue_map_has_credential_envelope_synthesis_v1_entry(
    rescue_map_payload: dict[str, Any],
) -> None:
    entries = rescue_map_payload["entries"]
    entry = next(
        item
        for item in entries
        if item["class"] == "admission_class_credential_envelope_synthesis_v1"
    )
    assert entry["target_kind"] == "fixture"
    assert entry["target"] == "docs/benchmarks/admission_recovery_scenarios.json"
    assert "#7248" in entry["notes"]
    assert "#7209" in entry["notes"]
    assert "ARAGORA_CREDENTIAL_ENVELOPE_PROBE" in entry["notes"]


def test_rescue_map_entry_resolves_to_an_existing_fixture(
    rescue_map_payload: dict[str, Any],
) -> None:
    for entry in rescue_map_payload["entries"]:
        if entry.get("target_kind") != "fixture":
            continue
        target = entry.get("target", "")
        if not target:
            continue
        candidate = REPO_ROOT / target
        assert candidate.is_file(), (
            f"rescue_productization.json entry class={entry['class']!r} points at "
            f"fixture {target!r} which is missing from the repo"
        )


def test_rescue_map_class_names_remain_unique(rescue_map_payload: dict[str, Any]) -> None:
    classes = [entry["class"] for entry in rescue_map_payload["entries"]]
    assert len(classes) == len(set(classes)), (
        "rescue_productization.json entries must have unique class names"
    )


def test_rescue_map_entry_is_linked_fixture_via_harvest_loader(
    rescue_map_payload: dict[str, Any],
) -> None:
    """The harvester's productization-status classifier sees the new entry as
    a linked fixture (not unlinked / linked_other)."""
    from harvest_rescue_classes import _productization_fields, load_productization_map

    loaded = load_productization_map(RESCUE_MAP_PATH)
    assert "admission_class_credential_envelope_synthesis_v1" in loaded
    fields = _productization_fields(
        "admission_class_credential_envelope_synthesis_v1",
        productization_map=loaded,
    )
    assert fields["productization_status"] == "linked_fixture"
    assert fields["productization_target_kind"] == "fixture"
    assert fields["productization_target"].endswith("admission_recovery_scenarios.json")


def test_fixture_non_goals_remain_in_scope_for_pr3(
    fixture_payload: dict[str, Any],
) -> None:
    non_goals = fixture_payload.get("non_goals", [])
    must_have_substrings = (
        "live benchmark",
        "CredentialEnvelope substrate",
        "rev-4 corpus membership criteria",
        "dispatch_followups.py",
    )
    joined = " | ".join(non_goals).lower()
    for needle in must_have_substrings:
        assert needle.lower() in joined, (
            f"fixture non_goals should call out {needle!r}: got {non_goals!r}"
        )


def test_verification_paths_reference_both_rescue_map_entries(
    fixture_payload: dict[str, Any],
) -> None:
    verifications = fixture_payload["verification_paths"]
    classes_referenced = {v["class"] for v in verifications if v.get("kind") == "rescue_map_entry"}
    assert "admission_class_corpus_synthesis_v1" in classes_referenced
    assert "admission_class_credential_envelope_synthesis_v1" in classes_referenced
