#!/usr/bin/env python3
"""Check Claude profile-pool health and optionally write an operator handoff.

This is a read-only monitor by default. It reuses ``audit_claude_profiles`` for
profile identity collection and fails when the pool has subscription-seat
collisions, too few live tokens, or a required profile is missing from the
configured pool. Pass ``--write-handoff`` to persist one idempotent operator
handoff JSON for account-side repair work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.agents.claude_profile_audit import (  # noqa: E402
    CollisionGroup,
    ProfileIdentity,
    analyze_profiles,
)
from scripts.audit_claude_profiles import _collect, _default_profiles  # noqa: E402

DEFAULT_MIN_LIVE_TOKENS = 4
DEFAULT_REQUIRED_PROFILES = ("max-13",)
DEFAULT_HANDOFF_DIR = Path(".aragora") / "operator-handoffs"
DEFAULT_HANDOFF_FILENAME = "claude-profile-health.json"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_handoff_dir() -> Path:
    raw = os.environ.get("ARAGORA_OPERATOR_HANDOFF_DIR", "").strip()
    return Path(raw) if raw else DEFAULT_HANDOFF_DIR


def _collision_dict(group: CollisionGroup) -> dict[str, Any]:
    return {
        "kind": group.kind,
        "severity": group.severity,
        "key": group.key,
        "label": group.label,
        "profiles": list(group.profiles),
        "detail": group.detail,
    }


def _identity_dict(identity: ProfileIdentity) -> dict[str, Any]:
    return {
        "profile": identity.profile,
        "email": identity.email,
        "org_id": identity.org_id,
        "org_name": identity.org_name,
        "plan": identity.plan,
        "token_live": identity.token_live,
    }


def _profile_names(identities: Sequence[ProfileIdentity]) -> list[str]:
    return [identity.profile for identity in identities]


def evaluate_profile_health(
    identities: Sequence[ProfileIdentity],
    *,
    min_live_tokens: int = DEFAULT_MIN_LIVE_TOKENS,
    required_profiles: Sequence[str] = DEFAULT_REQUIRED_PROFILES,
) -> dict[str, Any]:
    """Return a structured health verdict for collected Claude profiles."""

    identity_list = list(identities)
    audit = analyze_profiles(identity_list)
    profile_names = _profile_names(identity_list)
    profile_set = set(profile_names)
    live_profiles = sorted(
        identity.profile for identity in identity_list if identity.token_live is True
    )
    expired_profiles = sorted(
        identity.profile for identity in identity_list if identity.token_live is False
    )
    unknown_token_profiles = sorted(
        identity.profile for identity in identity_list if identity.token_live is None
    )
    missing_required_profiles = [
        profile for profile in required_profiles if profile and profile not in profile_set
    ]

    blockers: list[dict[str, Any]] = []
    if audit.org_seat_collisions:
        blockers.append(
            {
                "kind": "org_seat_collision",
                "severity": "high",
                "message": "Claude profiles share one subscription orgId.",
                "collisions": [_collision_dict(group) for group in audit.org_seat_collisions],
            }
        )
    if len(live_profiles) < min_live_tokens:
        blockers.append(
            {
                "kind": "live_token_floor",
                "severity": "high",
                "message": (
                    f"Only {len(live_profiles)} Claude profiles have live tokens; "
                    f"minimum is {min_live_tokens}."
                ),
                "live_profiles": live_profiles,
                "expired_profiles": expired_profiles,
                "unknown_token_profiles": unknown_token_profiles,
            }
        )
    if missing_required_profiles:
        blockers.append(
            {
                "kind": "missing_required_profile",
                "severity": "high",
                "message": "Required Claude review profile(s) are missing from the configured pool.",
                "missing_profiles": missing_required_profiles,
                "configured_profiles": profile_names,
            }
        )

    warnings: list[dict[str, Any]] = []
    if audit.shared_credential_collisions:
        warnings.append(
            {
                "kind": "shared_credential_collision",
                "severity": "warn",
                "message": "Claude profiles share a login across distinct orgs.",
                "collisions": [
                    _collision_dict(group) for group in audit.shared_credential_collisions
                ],
            }
        )
    if unknown_token_profiles:
        warnings.append(
            {
                "kind": "unknown_token_state",
                "severity": "warn",
                "message": "Some Claude profiles have no readable token-expiry state.",
                "profiles": unknown_token_profiles,
            }
        )

    recommendations = list(audit.recommendations)
    if expired_profiles:
        recommendations.append(
            "Refresh or re-login expired Claude profiles: " + ", ".join(expired_profiles) + "."
        )
    if missing_required_profiles:
        recommendations.append(
            "Add required Claude profile(s) to ARAGORA_CLAUDE_REVIEW_PROFILES: "
            + ", ".join(missing_required_profiles)
            + "."
        )

    return {
        "ok": not blockers,
        "profile_count": len(identity_list),
        "configured_profiles": profile_names,
        "required_profiles": [profile for profile in required_profiles if profile],
        "missing_required_profiles": missing_required_profiles,
        "min_live_tokens": min_live_tokens,
        "live_token_count": len(live_profiles),
        "live_profiles": live_profiles,
        "expired_profiles": expired_profiles,
        "unknown_token_profiles": unknown_token_profiles,
        "distinct_org_count": audit.distinct_org_count,
        "distinct_email_count": audit.distinct_email_count,
        "org_seat_collisions": [_collision_dict(group) for group in audit.org_seat_collisions],
        "shared_credential_collisions": [
            _collision_dict(group) for group in audit.shared_credential_collisions
        ],
        "blockers": blockers,
        "warnings": warnings,
        "recommendations": recommendations,
        "profiles": [_identity_dict(identity) for identity in identity_list],
    }


def build_operator_handoff(
    health: dict[str, Any],
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    now = observed_at or datetime.now(UTC)
    timestamp = now.isoformat().replace("+00:00", "Z")
    return {
        "idempotency_key": "claude-profile-health",
        "kind": "claude_profile_health",
        "created_at": timestamp,
        "updated_at": timestamp,
        "priority": "HIGH" if not health.get("ok") else "LOW",
        "status": "blocked" if not health.get("ok") else "healthy",
        "requires_human_account_action": not health.get("ok"),
        "task": "Repair Claude profile pool health for heterogeneous review quorum.",
        "requested_action": {
            "type": "operator_account_action",
            "org_seat_collisions": health.get("org_seat_collisions", []),
            "profiles_to_refresh": health.get("expired_profiles", []),
            "missing_required_profiles": health.get("missing_required_profiles", []),
            "recommendations": health.get("recommendations", []),
        },
        "profile_health": health,
    }


def write_operator_handoff(
    health: dict[str, Any],
    *,
    handoff_dir: Path = DEFAULT_HANDOFF_DIR,
    filename: str = DEFAULT_HANDOFF_FILENAME,
    observed_at: datetime | None = None,
) -> Path:
    handoff_dir.mkdir(parents=True, exist_ok=True)
    path = handoff_dir / filename
    payload = build_operator_handoff(health, observed_at=observed_at)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def _render_text(health: dict[str, Any], *, handoff_path: Path | None = None) -> None:
    status = "ok" if health["ok"] else "blocked"
    print(f"Claude profile health: {status}")
    print(
        f"profiles={health['profile_count']} live={health['live_token_count']}/"
        f"{health['min_live_tokens']} distinct_orgs={health['distinct_org_count']} "
        f"distinct_logins={health['distinct_email_count']}"
    )
    if health["blockers"]:
        print("\nBlockers:")
        for blocker in health["blockers"]:
            print(f"  - {blocker['kind']}: {blocker['message']}")
    if health["warnings"]:
        print("\nWarnings:")
        for warning in health["warnings"]:
            print(f"  - {warning['kind']}: {warning['message']}")
    if health["recommendations"]:
        print("\nRecommendations:")
        for recommendation in health["recommendations"]:
            print(f"  * {recommendation}")
    if handoff_path is not None:
        print(f"\noperator_handoff={handoff_path}")


def _parse_required_profiles(values: Sequence[str], *, disable_defaults: bool) -> list[str]:
    profiles = [] if disable_defaults else list(DEFAULT_REQUIRED_PROFILES)
    for value in values:
        for item in value.split(","):
            profile = item.strip()
            if profile and profile not in profiles:
                profiles.append(profile)
    return profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*", help="Profiles to check (default: audit default)")
    parser.add_argument(
        "--min-live-tokens",
        type=int,
        default=_env_int("ARAGORA_CLAUDE_PROFILE_MIN_LIVE_TOKENS", DEFAULT_MIN_LIVE_TOKENS),
        help="Minimum profiles with unexpired Claude tokens before the monitor fails.",
    )
    parser.add_argument(
        "--require-profile",
        action="append",
        default=[],
        help="Profile that must appear in the configured pool. Repeatable or comma-separated.",
    )
    parser.add_argument(
        "--no-default-required-profiles",
        action="store_true",
        help="Do not require the default max-13 review profile.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--write-handoff", action="store_true", help="Write operator handoff JSON")
    parser.add_argument(
        "--handoff-dir",
        type=Path,
        default=_default_handoff_dir(),
        help="Directory for --write-handoff output.",
    )
    parser.add_argument(
        "--handoff-filename",
        default=DEFAULT_HANDOFF_FILENAME,
        help="Filename for --write-handoff output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = args.profiles or _default_profiles()
    required_profiles = _parse_required_profiles(
        args.require_profile,
        disable_defaults=args.no_default_required_profiles,
    )
    health = evaluate_profile_health(
        _collect(profiles),
        min_live_tokens=max(args.min_live_tokens, 0),
        required_profiles=required_profiles,
    )
    handoff_path = None
    if args.write_handoff:
        handoff_path = write_operator_handoff(
            health,
            handoff_dir=args.handoff_dir,
            filename=args.handoff_filename,
        )
        health["operator_handoff_path"] = str(handoff_path)

    if args.json:
        print(json.dumps(health, indent=2, sort_keys=True))
    else:
        _render_text(health, handoff_path=handoff_path)
    return 0 if health["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
