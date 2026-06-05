"""Tests for the Claude pool verify-snapshot helpers."""

from __future__ import annotations

from aragora.agents.claude_pool_health import build_snapshot, classify_probe, is_healthy


def test_classify_real_output_is_ok():
    assert classify_probe("OK", returncode=0) == "ok"
    assert classify_probe("Here is your answer.\nLine two", returncode=0) == "ok"


def test_classify_401_is_expired():
    assert (
        classify_probe("Failed to authenticate. API Error: 401 Invalid authentication credentials")
        == "expired"
    )
    assert classify_probe("OAuth token has expired") == "expired"


def test_classify_missing_credentials_is_not_configured():
    assert classify_probe("No such file or directory: .credentials.json") == "not_configured"
    assert classify_probe("not logged in") == "not_configured"


def test_classify_timeout_and_empty_are_unauthenticated():
    assert classify_probe("", timed_out=True) == "unauthenticated"
    assert classify_probe("") == "unauthenticated"


def test_classify_nonzero_returncode_without_text_marker_is_expired():
    assert classify_probe("weird failure", returncode=2) == "expired"


def test_is_healthy_matches_routing_unhealthy_set():
    assert is_healthy("ok")
    for bad in ("expired", "not_configured", "unauthenticated", "logged_out"):
        assert not is_healthy(bad)


def test_build_snapshot_shape_and_counts():
    records = [
        {"name": "max-01", "email": "a@x", "state": "ok"},
        {"name": "max-02", "email": "b@x", "state": "ok"},
        {"name": "max-03", "email": "c@x", "state": "expired"},
    ]
    snap = build_snapshot(records, generated_at="2026-06-05T02:25:00Z")
    assert snap["generated_at"] == "2026-06-05T02:25:00Z"
    assert snap["total"] == 3
    assert snap["healthy"] == 2
    assert snap["profiles"][0] == {"name": "max-01", "email": "a@x", "state": "ok"}
    # The snapshot is consumable by review_routing._load_pool_health (name+state).
    states = {p["name"]: p["state"] for p in snap["profiles"]}
    assert states["max-03"] == "expired"


def test_build_snapshot_defaults_missing_state():
    snap = build_snapshot([{"name": "max-09"}], generated_at="x")
    assert snap["profiles"][0]["state"] == "unauthenticated"
    assert snap["healthy"] == 0
