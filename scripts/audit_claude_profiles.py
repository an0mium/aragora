#!/usr/bin/env python3
"""Audit the Claude subscription-profile pool for subscription-seat collisions.

Collects each profile's live claude.ai identity (``claude_profile.sh status``)
plus its access-token expiry, then reports the topology and any collisions via
:mod:`aragora.agents.claude_profile_audit`.

Usage:
    python3 scripts/audit_claude_profiles.py [--json] [profile ...]

Without explicit profiles it audits max-01..max-13 (override with
``ARAGORA_CLAUDE_REVIEW_PROFILES``). Read-only: it runs ``status`` (no login,
no live inference probe) and reads local credential files only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.agents.claude_profile_audit import (  # noqa: E402
    ProfileIdentity,
    analyze_profiles,
)

DEFAULT_PROFILES = tuple(f"max-{i:02d}" for i in range(1, 14))


def _default_profiles() -> list[str]:
    raw = os.environ.get("ARAGORA_CLAUDE_REVIEW_PROFILES", "").strip()
    if not raw:
        return list(DEFAULT_PROFILES)
    out: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name not in out:
            out.append(name)
    return out or list(DEFAULT_PROFILES)


def _profile_root() -> Path:
    override = os.environ.get("CLAUDE_PROFILE_ROOT", "").strip()
    return Path(override) if override else Path.home() / ".aragora-claude"


def _status_json(profile_tool: Path, profile: str) -> dict:
    try:
        proc = subprocess.run(
            [str(profile_tool), "status", profile],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    out = proc.stdout.strip()
    if not out:
        return {}
    # status may prepend wrapper lines; find the JSON object.
    start = out.find("{")
    if start < 0:
        return {}
    try:
        return json.loads(out[start:])
    except json.JSONDecodeError:
        return {}


def _token_live(profile_root: Path, profile: str) -> bool | None:
    cred = profile_root / profile / ".claude" / ".credentials.json"
    try:
        data = json.loads(cred.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    oauth = data.get("claudeAiOauth") or data.get("oauth") or data
    exp = oauth.get("expiresAt") or oauth.get("expires_at")
    if not isinstance(exp, (int, float)):
        return None
    seconds = exp / 1000 if exp > 1e12 else exp
    return datetime.fromtimestamp(seconds, tz=timezone.utc) > datetime.now(timezone.utc)


def _collect(profiles: list[str]) -> list[ProfileIdentity]:
    profile_tool = REPO_ROOT / "scripts" / "claude_profile.sh"
    profile_root = _profile_root()
    identities: list[ProfileIdentity] = []
    for profile in profiles:
        status = _status_json(profile_tool, profile)
        identities.append(
            ProfileIdentity(
                profile=profile,
                email=str(status.get("email", "") or ""),
                org_id=str(status.get("orgId", "") or ""),
                org_name=str(status.get("orgName", "") or ""),
                plan=str(status.get("subscriptionType", "") or ""),
                token_live=_token_live(profile_root, profile),
            )
        )
    return identities


def _render(identities: list[ProfileIdentity], result) -> None:
    def live(t: bool | None) -> str:
        return "live" if t else ("EXPIRED" if t is False else "?")

    print("Claude profile pool topology")
    print("=" * 78)
    print(f"{'profile':9} {'token':8} {'plan':5} {'org':30} {'email'}")
    for i in identities:
        print(
            f"{i.profile:9} {live(i.token_live):8} {i.plan or '-':5} "
            f"{(i.org_name or '-')[:30]:30} {i.email or '-'}"
        )
    print(
        f"\n{result.profile_count} profiles | {result.distinct_org_count} distinct orgs "
        f"| {result.distinct_email_count} distinct logins"
    )
    if result.org_seat_collisions:
        print("\n[HIGH] Same-subscription-seat collisions (consolidate):")
        for g in result.org_seat_collisions:
            print(f"  - {g.detail}")
    if result.shared_credential_collisions:
        print("\n[warn] Shared-login (different orgs — distinct subs, may contend):")
        for g in result.shared_credential_collisions:
            print(f"  - {g.detail}")
    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  * {rec}")
    if not result.has_collisions:
        print("\nNo subscription-seat collisions detected.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*", help="Profiles to audit (default: max-01..max-13)")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    profiles = args.profiles or _default_profiles()
    identities = _collect(profiles)
    result = analyze_profiles(identities)

    if args.json:
        payload = result.to_dict()
        payload["profiles"] = [
            {
                "profile": i.profile,
                "email": i.email,
                "org_id": i.org_id,
                "org_name": i.org_name,
                "plan": i.plan,
                "token_live": i.token_live,
            }
            for i in identities
        ]
        print(json.dumps(payload, indent=2))
    else:
        _render(identities, result)
    # Exit non-zero on a high-severity (org-seat) collision so CI/cron can alert.
    return 1 if result.org_seat_collisions else 0


if __name__ == "__main__":
    raise SystemExit(main())
