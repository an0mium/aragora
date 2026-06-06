from aragora.gti.receipt import BeliefProvenance, validate_belief_provenance

NOW = "2026-06-06T12:00:00+00:00"


def _belief(**kw):
    base = dict(
        belief_id="b1",
        source="git rev-parse origin/main",
        as_of="2026-06-06T11:59:00+00:00",
        verification_method="git",
        freshness_ttl_seconds=300.0,
        was_revalidated_at_decision=False,
    )
    base.update(kw)
    return BeliefProvenance(**base)


def test_fresh_belief_is_valid():
    assert validate_belief_provenance([_belief()], NOW) == []


def test_missing_source_is_invalid():
    problems = validate_belief_provenance([_belief(source="")], NOW)
    assert any("missing" in p for p in problems)


def test_missing_as_of_is_invalid():
    problems = validate_belief_provenance([_belief(as_of="")], NOW)
    assert any("missing" in p for p in problems)


def test_past_ttl_without_revalidation_is_invalid():
    problems = validate_belief_provenance(
        [_belief(as_of="2026-06-06T11:00:00+00:00", freshness_ttl_seconds=300.0)], NOW
    )
    assert any("ttl" in p.lower() for p in problems)


def test_past_ttl_but_revalidated_is_valid():
    problems = validate_belief_provenance(
        [
            _belief(
                as_of="2026-06-06T11:00:00+00:00",
                freshness_ttl_seconds=300.0,
                was_revalidated_at_decision=True,
            )
        ],
        NOW,
    )
    assert problems == []
