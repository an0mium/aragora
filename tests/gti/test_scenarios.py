from pathlib import Path
import json

import pytest

from aragora.gti.scenarios import Scenario, load_scenarios

CORPUS = Path("docs/status/generated/gti/scenarios.json")


def test_corpus_loads_and_has_12_to_15_scenarios():
    scenarios = load_scenarios(CORPUS)
    assert 12 <= len(scenarios) <= 15
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_scenario_ids_unique_and_failure_modes_valid():
    scenarios = load_scenarios(CORPUS)
    ids = [s.id for s in scenarios]
    assert len(ids) == len(set(ids))
    valid = {
        "stale_source",
        "stale_memory",
        "false_green",
        "wrong_taxonomy",
        "historical_as_current",
        "self_aware_stale",
    }
    assert {s.failure_mode for s in scenarios} <= valid


def test_corpus_includes_control_scenarios_that_are_fresh_and_true():
    # Controls guard against a gate that flags everything.
    scenarios = load_scenarios(CORPUS)
    controls = [s for s in scenarios if s.belief_matches_truth]
    assert len(controls) >= 2


def _scenario(**kw):
    base = dict(
        id="GTI-TEST-001",
        failure_mode="stale_source",
        belief_presented="outdated claim",
        ground_truth="current claim",
        canonical_source="docs/status/source.md",
        belief_matches_truth=False,
        belief_age_days=9.0,
        freshness_ttl_days=7.0,
        quorum_would_flag=True,
        expected="detect",
        consequential_action_if_wrong="ship stale proof",
    )
    base.update(kw)
    return Scenario(**base)


def test_scenario_rejects_negative_belief_age():
    with pytest.raises(ValueError, match="belief_age_days"):
        _scenario(belief_age_days=-1.0)


def test_scenario_rejects_non_finite_belief_age():
    with pytest.raises(ValueError, match="belief_age_days"):
        _scenario(belief_age_days=float("nan"))


def test_scenario_rejects_non_bool_belief_matches_truth():
    with pytest.raises(ValueError, match="belief_matches_truth"):
        _scenario(belief_matches_truth="false")


def test_scenario_rejects_non_bool_quorum_would_flag():
    with pytest.raises(ValueError, match="quorum_would_flag"):
        _scenario(quorum_would_flag="false")


def test_scenario_rejects_non_positive_freshness_ttl():
    with pytest.raises(ValueError, match="freshness_ttl_days"):
        _scenario(freshness_ttl_days=0.0)


def test_scenario_rejects_non_finite_freshness_ttl():
    with pytest.raises(ValueError, match="freshness_ttl_days"):
        _scenario(freshness_ttl_days=float("inf"))


def test_load_scenarios_rejects_missing_scenarios_list(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"items": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scenarios list"):
        load_scenarios(path)


def test_load_scenarios_rejects_non_object_entries(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": ["not-an-object"]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="entry 0"):
        load_scenarios(path)


def test_load_scenarios_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "scenarios.json"
    entry = _scenario().__dict__
    path.write_text(json.dumps({"scenarios": [entry, entry]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate scenario id"):
        load_scenarios(path)
