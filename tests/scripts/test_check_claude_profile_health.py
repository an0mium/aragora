"""Tests for the Claude profile health monitor."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import scripts.check_claude_profile_health as mod
from aragora.agents.claude_profile_audit import ProfileIdentity


def _identity(
    profile: str,
    *,
    email: str | None = None,
    org_id: str | None = None,
    org_name: str | None = None,
    token_live: bool | None = True,
) -> ProfileIdentity:
    return ProfileIdentity(
        profile=profile,
        email=email or f"{profile}@example.com",
        org_id=org_id or f"org-{profile}",
        org_name=org_name or f"{profile} org",
        plan="max",
        token_live=token_live,
    )


def test_health_blocks_org_seat_collision_and_low_live_token_floor() -> None:
    health = mod.evaluate_profile_health(
        [
            _identity("max-01", email="one@example.com", org_id="shared-org", token_live=True),
            _identity("max-02", email="two@example.com", org_id="shared-org", token_live=False),
            _identity("max-13", token_live=False),
        ],
        min_live_tokens=2,
        required_profiles=("max-13",),
    )

    assert health["ok"] is False
    blocker_kinds = {blocker["kind"] for blocker in health["blockers"]}
    assert blocker_kinds == {"org_seat_collision", "live_token_floor"}
    assert health["live_profiles"] == ["max-01"]
    assert health["expired_profiles"] == ["max-02", "max-13"]
    assert health["org_seat_collisions"][0]["profiles"] == ["max-01", "max-02"]


def test_health_blocks_missing_required_profile() -> None:
    health = mod.evaluate_profile_health(
        [_identity("max-01"), _identity("max-02")],
        min_live_tokens=1,
        required_profiles=("max-13",),
    )

    assert health["ok"] is False
    assert health["missing_required_profiles"] == ["max-13"]
    assert [blocker["kind"] for blocker in health["blockers"]] == ["missing_required_profile"]


def test_health_passes_distinct_pool_with_enough_live_tokens() -> None:
    health = mod.evaluate_profile_health(
        [_identity("max-01"), _identity("max-02"), _identity("max-13")],
        min_live_tokens=3,
        required_profiles=("max-13",),
    )

    assert health["ok"] is True
    assert health["blockers"] == []
    assert health["live_token_count"] == 3


def test_parse_required_profiles_keeps_custom_profiles_when_defaults_disabled() -> None:
    profiles = mod._parse_required_profiles(
        ["max-01,max-02", "max-01"],
        disable_defaults=True,
    )

    assert profiles == ["max-01", "max-02"]


def test_write_operator_handoff_is_idempotent_and_actionable(tmp_path) -> None:
    observed_at = datetime(2026, 6, 5, 2, 30, tzinfo=UTC)
    health = mod.evaluate_profile_health(
        [
            _identity("max-03", email="ap@example.com", org_id="shared-org", token_live=False),
            _identity("max-06", email="ap@example.com", org_id="shared-org", token_live=False),
            _identity("max-13", token_live=True),
        ],
        min_live_tokens=2,
        required_profiles=("max-13",),
    )

    first = mod.write_operator_handoff(health, handoff_dir=tmp_path, observed_at=observed_at)
    second = mod.write_operator_handoff(health, handoff_dir=tmp_path, observed_at=observed_at)

    assert first == second
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["idempotency_key"] == "claude-profile-health"
    assert payload["priority"] == "HIGH"
    assert payload["requires_human_account_action"] is True
    assert payload["requested_action"]["profiles_to_refresh"] == ["max-03", "max-06"]
    assert payload["requested_action"]["org_seat_collisions"][0]["profiles"] == [
        "max-03",
        "max-06",
    ]


def test_main_json_writes_handoff_and_returns_failure(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "_collect",
        lambda profiles: [
            _identity("max-01", token_live=False),
            _identity("max-13", token_live=True),
        ],
    )

    rc = mod.main(
        [
            "--json",
            "--write-handoff",
            "--handoff-dir",
            str(tmp_path),
            "--min-live-tokens",
            "2",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["operator_handoff_path"] == str(tmp_path / mod.DEFAULT_HANDOFF_FILENAME)
    assert payload["blockers"][0]["kind"] == "live_token_floor"
