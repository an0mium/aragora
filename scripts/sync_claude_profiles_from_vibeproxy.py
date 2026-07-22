#!/usr/bin/env python3
"""Keep aragora Claude profiles alive by consuming VibeProxy's fresh tokens.

Root cause this solves
----------------------
Anthropic OAuth **single-use-rotates** the refresh token on every refresh: once
a refresh token is used it is replaced and the old one is revoked. When the same
Claude account is refreshed by more than one independent process, whichever
refreshes first wins and every other holder is left with a revoked token -> dead.

Aragora's hourly ``claude_pool_verify.py`` keeps *valid-refresh* profiles alive
by probing them (the probe triggers the CLI's own refresh). That is correct in
isolation, but it loses a race it cannot see: VibeProxy's ``cli-proxy-api`` also
refreshes the very same accounts on its own schedule, and two aragora profiles
sometimes share one login. The result observed in the field is 0/13 healthy.

The fix (the "VibeProxy approach", done right)
----------------------------------------------
VibeProxy stays alive because it is the **single owner** of each account's
refresh token. So aragora stops competing to refresh and instead **consumes**
VibeProxy's already-fresh access token: this script copies the freshly-refreshed
access token from ``~/.cli-proxy-api/claude-<email>.json`` into the matching
aragora profile credential, translating the on-disk format.

To guarantee aragora can never rotate VibeProxy's token out from under it, the
synced aragora credential is written **without a usable refresh token**
(``--blank-refresh``, the default): aragora becomes a pure token consumer. If a
sync cycle ever lags past the access-token expiry the profile simply reports
expired until the next cycle heals it -- it never revokes VibeProxy's live token.
Run this on a short interval (well under the ~8h access-token TTL) via launchd.

One profile per VibeProxy account
---------------------------------
VibeProxy authenticates by email and holds exactly one org per email, so two
profiles that are different orgs under one shared login (a personal Max org and
a Synaptent team org, both under ``synaptent@synaptent.com``) cannot both be
sourced from VibeProxy -- syncing both would collapse one subscription onto the
other's org. ``VIBEPROXY_SYNC_TARGET`` is therefore a strict 1:1 email->profile
map, and same-email siblings / duplicate seats are listed in
``NATIVE_ONLY_REASON`` and never synced (they stay on native login).

This script only reads VibeProxy auth files and writes aragora profile
credentials on the local machine. It never prints token material and never makes
a network call.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VIBEPROXY_AUTH_DIR = Path.home() / ".cli-proxy-api"
ARAGORA_PROFILE_ROOT = Path.home() / ".aragora-claude"
PROFILE_TOOL = Path(__file__).resolve().parent / "claude_profile.sh"

# The single profile each VibeProxy account (keyed by EMAIL) may sync into.
#
# INVARIANT: no VibeProxy email is the source for more than one profile. This is
# load-bearing. VibeProxy authenticates by email and holds exactly ONE org per
# email, so two profiles that are *different orgs under one shared login* (e.g. a
# personal Max org and a Synaptent team org, both under synaptent@synaptent.com)
# cannot both be sourced from VibeProxy -- syncing both would silently collapse
# one subscription onto the other's org. The non-VibeProxy org stays on native
# `scripts/claude_profiles_bootstrap.sh login`.
VIBEPROXY_SYNC_TARGET: dict[str, str] = {
    "anomium@gmail.com": "max-01",
    "scarmani@gmail.com": "max-02",
    "liftmode@liftmode.com": "max-04",
    "root@liftmode.com": "max-05",
    "ap@synaptent.com": "max-06",
    "radnoem@gmail.com": "max-07",
    "synaptent@synaptent.com": "max-09",  # Synaptent team org (VibeProxy's org for this email)
    "verborgen.doel@gmail.com": "max-11",
    "armand@synaptent.com": "max-12",  # Synaptent team org
    "ringrift.ai@gmail.com": "max-13",  # repointed from the max-12 duplicate to this distinct account
}

# Profiles deliberately NOT VibeProxy-synced, with why. A shared login's distinct
# org, or a duplicate of another profile's exact account. These stay native.
NATIVE_ONLY_REASON: dict[str, str] = {
    "max-03": "shares ap@synaptent.com with max-06 (its own org); native login only",
    "max-08": (
        "personal Max org under synaptent@synaptent.com; VibeProxy holds the "
        "Synaptent team org for that email (synced to max-09). Native login only."
    ),
    "max-10": "no VibeProxy account for armand.tuzel@gmail.com; native login only",
}

# Reverse index: profile -> the VibeProxy email that sources it (if any).
PROFILE_TO_EMAIL: dict[str, str] = {
    profile: email for email, profile in VIBEPROXY_SYNC_TARGET.items()
}

# Scope/plan fields VibeProxy does not carry; preserved from an existing aragora
# credential when present, else these Max-plan defaults.
_DEFAULT_SCOPES = [
    "user:file_upload",
    "user:inference",
    "user:mcp_servers",
    "user:profile",
    "user:sessions:claude_code",
]
_DEFAULT_SUBSCRIPTION = "max"


@dataclass
class SyncResult:
    profile: str
    email: str
    action: str  # synced | skipped_no_source | skipped_fresh | skipped_no_email | error
    detail: str = ""


def _vibeproxy_path(email: str) -> Path:
    return VIBEPROXY_AUTH_DIR / f"claude-{email}.json"


def _profile_cred_path(profile: str) -> Path:
    return ARAGORA_PROFILE_ROOT / profile / ".claude" / ".credentials.json"


def _iso_to_epoch_ms(value: str) -> int:
    return int(_dt.datetime.fromisoformat(value).timestamp() * 1000)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def translate_credential(
    vibeproxy: dict,
    existing_oauth: dict | None,
    *,
    blank_refresh: bool,
) -> dict:
    """Build an aragora ``.credentials.json`` payload from a VibeProxy account.

    Preserves plan/scope metadata from any existing aragora credential; those
    fields are absent from VibeProxy's format. When ``blank_refresh`` is set the
    refresh token is emptied so aragora can never rotate VibeProxy's live token.
    """
    oauth = dict(existing_oauth or {})
    oauth["accessToken"] = vibeproxy["access_token"]
    oauth["refreshToken"] = "" if blank_refresh else vibeproxy.get("refresh_token", "")
    oauth["expiresAt"] = _iso_to_epoch_ms(vibeproxy["expired"])
    oauth.setdefault("scopes", list(_DEFAULT_SCOPES))
    oauth.setdefault("subscriptionType", _DEFAULT_SUBSCRIPTION)
    return {"claudeAiOauth": oauth}


def _current_access_token(cred: dict | None) -> str | None:
    if not cred:
        return None
    return (cred.get("claudeAiOauth") or {}).get("accessToken")


def sync_profile(
    profile: str,
    *,
    blank_refresh: bool,
    apply: bool,
) -> SyncResult:
    if profile in NATIVE_ONLY_REASON:
        return SyncResult(profile, "", "skipped_native_only", NATIVE_ONLY_REASON[profile])
    email = PROFILE_TO_EMAIL.get(profile, "")
    if not email:
        return SyncResult(profile, email, "skipped_no_email", "no VibeProxy source assigned")
    vp = _load_json(_vibeproxy_path(email))
    if vp is None or vp.get("disabled") or not vp.get("access_token"):
        return SyncResult(profile, email, "skipped_no_source", "no live VibeProxy account")

    cred_path = _profile_cred_path(profile)
    existing = _load_json(cred_path)
    existing_oauth = (existing or {}).get("claudeAiOauth")

    # Idempotent: if the profile already holds this exact access token, do nothing.
    if _current_access_token(existing) == vp["access_token"]:
        return SyncResult(profile, email, "skipped_fresh", "already current")

    payload = translate_credential(vp, existing_oauth, blank_refresh=blank_refresh)
    if not apply:
        return SyncResult(profile, email, "synced", "dry-run")

    cred_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace with owner-only permissions on the credential.
    tmp = cred_path.with_suffix(".credentials.json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, cred_path)
    return SyncResult(profile, email, "synced", "applied")


def _probe(profile: str, timeout: int = 60) -> bool:
    """Live-probe a profile the same way scripts/claude_pool_verify.py does."""
    try:
        proc = subprocess.run(
            [str(PROFILE_TOOL), "exec", profile, "--", "claude", "--print", "-p", "-"],
            input="hi",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*", help="Profiles to sync (default: all mapped)")
    parser.add_argument("--apply", action="store_true", help="Write files (default: dry-run)")
    parser.add_argument(
        "--keep-refresh",
        action="store_true",
        help="Keep VibeProxy's refresh token in the synced credential (default: blank it)",
    )
    parser.add_argument(
        "--probe-after",
        action="store_true",
        help="After --apply, live-probe each synced profile and report health",
    )
    parser.add_argument("--json", dest="json_output", action="store_true")
    args = parser.parse_args(argv)

    # Default set = the 1:1 sync targets; native-only profiles are shown only
    # when named explicitly (so their skip reason is visible on request).
    profiles = args.profiles or list(VIBEPROXY_SYNC_TARGET.values())
    results: list[SyncResult] = []
    for profile in profiles:
        results.append(
            sync_profile(
                profile,
                blank_refresh=not args.keep_refresh,
                apply=args.apply,
            )
        )

    probed: dict[str, bool] = {}
    if args.probe_after and args.apply:
        for r in results:
            if r.action == "synced":
                probed[r.profile] = _probe(r.profile)

    if args.json_output:
        print(
            json.dumps(
                {
                    "applied": args.apply,
                    "blank_refresh": not args.keep_refresh,
                    "results": [r.__dict__ for r in results],
                    "probed": probed,
                }
            )
        )
    else:
        for r in results:
            probe_note = ""
            if r.profile in probed:
                probe_note = "  probe=" + ("LIVE" if probed[r.profile] else "DEAD")
            print(f"  {r.profile:8} {r.email:28} {r.action:18} {r.detail}{probe_note}")
        synced = sum(1 for r in results if r.action == "synced")
        live = sum(1 for v in probed.values() if v)
        tail = f"; {live}/{len(probed)} probed live" if probed else ""
        print(f"Sync: {synced}/{len(results)} profiles updated{tail}")

    # Non-zero when a probe ran and any synced profile came back dead.
    if probed and not all(probed.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
