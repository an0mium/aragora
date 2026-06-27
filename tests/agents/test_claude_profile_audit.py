"""Tests for Claude profile-pool subscription-seat collision analysis."""

from __future__ import annotations

from aragora.agents.claude_profile_audit import ProfileIdentity, analyze_profiles


def _real_pool() -> list[ProfileIdentity]:
    # The live topology observed 2026-06-04. Exercises every collision type:
    #   - max-03/max-06: same email AND same org (121e7534) -> org_seat
    #   - max-09/max-12: DIFFERENT emails, same Team org (d16e9a80) -> org_seat
    #   - max-08/max-09: same email, different orgs -> shared_credential
    #   - max-12/max-13: same email, different orgs -> shared_credential
    return [
        ProfileIdentity("max-01", "anomium@gmail.com", "4f4be2d3", "anomium org", "max"),
        ProfileIdentity("max-02", "scarmani@gmail.com", "f3dc8855", "scarmani org", "max"),
        ProfileIdentity("max-03", "ap@synaptent.com", "121e7534", "ap org", "max"),
        ProfileIdentity("max-04", "liftmode@liftmode.com", "26bee17b", "liftmode org", "max"),
        ProfileIdentity("max-05", "root@liftmode.com", "640528f9", "root org", "max"),
        ProfileIdentity("max-06", "ap@synaptent.com", "121e7534", "ap org", "max"),
        ProfileIdentity("max-07", "radnoem@gmail.com", "1e896949", "radnoem org", "max"),
        ProfileIdentity("max-08", "synaptent@synaptent.com", "0ff1715d", "synaptent org", "max"),
        ProfileIdentity("max-09", "synaptent@synaptent.com", "d16e9a80", "Synaptent", "team"),
        ProfileIdentity("max-10", "armand.tuzel@gmail.com", "2c14611b", "tuzel org", "max"),
        ProfileIdentity("max-11", "verborgen.doel@gmail.com", "d70bbcb2", "verborgen org", "max"),
        ProfileIdentity("max-12", "armand@synaptent.com", "d16e9a80", "Synaptent", "team"),
        ProfileIdentity("max-13", "armand@synaptent.com", "dd44fcd1", "armand org", "max"),
    ]


def test_org_seat_collisions_catch_same_org_even_with_different_emails():
    result = analyze_profiles(_real_pool())
    seats = {g.key: g.profiles for g in result.org_seat_collisions}
    assert seats == {
        "121e7534": ("max-03", "max-06"),  # same email + same org
        "d16e9a80": ("max-09", "max-12"),  # different emails, same Team org (the hidden one)
    }
    assert all(g.severity == "high" for g in result.org_seat_collisions)


def test_shared_credential_flags_same_email_across_distinct_orgs():
    result = analyze_profiles(_real_pool())
    creds = {g.key: g.profiles for g in result.shared_credential_collisions}
    assert creds == {
        "synaptent@synaptent.com": ("max-08", "max-09"),
        "armand@synaptent.com": ("max-12", "max-13"),
    }
    assert all(g.severity == "warn" for g in result.shared_credential_collisions)


def test_same_email_same_org_is_only_a_seat_collision_not_credential():
    # max-03/max-06 share email AND org; must NOT also appear as shared_credential.
    result = analyze_profiles(_real_pool())
    cred_keys = {g.key for g in result.shared_credential_collisions}
    assert "ap@synaptent.com" not in cred_keys


def test_distinct_pool_has_no_collisions():
    pool = [
        ProfileIdentity("max-01", "a@x.com", "org-a", "A", "max"),
        ProfileIdentity("max-02", "b@x.com", "org-b", "B", "max"),
        ProfileIdentity("max-03", "c@x.com", "org-c", "C", "team"),
    ]
    result = analyze_profiles(pool)
    assert not result.has_collisions
    assert result.distinct_org_count == 3
    assert result.recommendations == []


def test_missing_org_id_does_not_create_false_collisions():
    # Two profiles with blank org_id must not be grouped together.
    pool = [
        ProfileIdentity("max-01", "a@x.com", "", "", ""),
        ProfileIdentity("max-02", "b@x.com", "", "", ""),
    ]
    result = analyze_profiles(pool)
    assert result.org_seat_collisions == []


def test_recommendations_name_keep_and_free_profiles():
    result = analyze_profiles(_real_pool())
    joined = "\n".join(result.recommendations)
    # Each high-severity seat collision yields a keep/free recommendation.
    assert "keep max-03" in joined and "free max-06" in joined
    assert "keep max-09" in joined and "free max-12" in joined
