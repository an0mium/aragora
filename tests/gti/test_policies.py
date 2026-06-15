from aragora.gti.policies import gated_policy, naive_policy
from aragora.gti.scenarios import Scenario


def _scn(**kw):
    base = dict(
        id="X",
        failure_mode="stale_source",
        belief_presented="b",
        ground_truth="g",
        canonical_source="c",
        belief_matches_truth=False,
        belief_age_days=9.0,
        freshness_ttl_days=7.0,
        quorum_would_flag=False,
        expected="correct",
        consequential_action_if_wrong="bad",
    )
    base.update(kw)
    return Scenario(**base)


def test_naive_acts_on_stale_belief_and_reports_green():
    out = naive_policy(_scn(belief_matches_truth=False))
    assert out.acted_on_stale_belief is True
    assert out.reported_green_but_wrong is True
    assert out.detected_stale is False


def test_naive_on_true_belief_is_fine():
    out = naive_policy(_scn(belief_matches_truth=True))
    assert out.acted_on_stale_belief is False
    assert out.reported_green_but_wrong is False


def test_gated_catches_stale_by_age():
    out = gated_policy(_scn(belief_age_days=9.0, freshness_ttl_days=7.0, quorum_would_flag=False))
    assert out.detected_stale is True
    assert out.corrected is True
    assert out.acted_on_stale_belief is False


def test_gated_treats_exact_ttl_boundary_as_fresh():
    # "Past TTL" means strictly older than the TTL; equality is still fresh.
    out = gated_policy(_scn(belief_age_days=7.0, freshness_ttl_days=7.0, quorum_would_flag=False))
    assert out.detected_stale is False
    assert out.acted_on_stale_belief is True


def test_gated_catches_via_quorum_when_age_ok():
    out = gated_policy(_scn(belief_age_days=0.0, freshness_ttl_days=7.0, quorum_would_flag=True))
    assert out.detected_stale is True
    assert out.acted_on_stale_belief is False


def test_gated_misses_undetectable_stale_belief_honestly():
    # Wrong belief, fresh by age, quorum would not flag => the gate cannot catch it.
    out = gated_policy(
        _scn(
            belief_matches_truth=False,
            belief_age_days=0.0,
            freshness_ttl_days=7.0,
            quorum_would_flag=False,
        )
    )
    assert out.detected_stale is False
    assert out.acted_on_stale_belief is True


def test_gated_does_not_false_flag_fresh_true_control():
    out = gated_policy(
        _scn(
            belief_matches_truth=True,
            belief_age_days=0.0,
            freshness_ttl_days=7.0,
            quorum_would_flag=False,
        )
    )
    assert out.detected_stale is False
    assert out.acted_on_stale_belief is False
