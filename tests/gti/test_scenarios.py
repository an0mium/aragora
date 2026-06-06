from pathlib import Path

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
