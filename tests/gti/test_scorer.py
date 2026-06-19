from pathlib import Path

from aragora.gti.scenarios import load_scenarios
from aragora.gti.scorer import Metrics, score_corpus

CORPUS = Path("docs/status/generated/gti/scenarios.json")


def test_score_corpus_returns_both_arms_and_delta():
    scenarios = load_scenarios(CORPUS)
    result = score_corpus(scenarios)
    assert isinstance(result.naive, Metrics)
    assert isinstance(result.gated, Metrics)
    # Gated must reduce the headline delusion rate vs naive on this corpus.
    assert result.gated.stale_belief_action_rate < result.naive.stale_belief_action_rate
    assert result.delta.stale_belief_action_rate > 0
    assert result.gated.false_green_rate <= result.naive.false_green_rate


def test_rates_are_fractions():
    scenarios = load_scenarios(CORPUS)
    result = score_corpus(scenarios)
    for m in (result.naive, result.gated):
        for value in (
            m.stale_belief_action_rate,
            m.detection_rate,
            m.correction_rate,
            m.false_green_rate,
        ):
            assert 0.0 <= value <= 1.0
