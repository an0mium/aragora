from pathlib import Path

from aragora.gti.scenarios import load_scenarios
from aragora.gti.scorer import score_corpus


def test_gated_beats_naive_end_to_end_on_real_corpus():
    scenarios = load_scenarios(Path("docs/status/generated/gti/scenarios.json"))
    result = score_corpus(scenarios)
    # Headline claim: the gated path materially reduces stale-belief action.
    assert result.delta.stale_belief_action_rate >= 0.3
    # And never increases false greens.
    assert result.delta.false_green_rate >= 0.0
    # Honesty guard: the gate is NOT omniscient (>=1 undetectable honest miss).
    assert result.gated.stale_belief_action_rate > 0.0
