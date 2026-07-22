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
Run this more often than VibeProxy's refresh lead so a synced token never lapses
between cycles (VibeProxy refreshes ~10 min before the ~8h access-token expiry).

Mapping lives in a local, untracked config
-------------------------------------------
The email->profile mapping is operator PII and account inventory, so it is NOT
in tracked source. It is read from ``~/.aragora/claude_profile_sync.json``
(override with ``ARAGORA_PROFILE_SYNC_CONFIG``). See the ``.example`` beside this
script for the format. VibeProxy authenticates by email and holds exactly one
org per email, so ``sync_target`` is a strict 1:1 email->profile map; profiles
that are a *different org under a shared login* (a personal Max org and a team
org under one email) or a duplicate seat go in ``native_only`` and stay on native
``scripts/claude_profiles_bootstrap.sh login``.

Safety
------
This script only reads VibeProxy auth files and writes aragora profile
credentials on the local machine. It never prints token material and never makes
a network call. It refuses to clobber a profile that holds a fresh *native* login
(live access token + real refresh token) unless ``--force``, and always writes a
``0600`` ``.bak`` before replacing a credential.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VIBEPROXY_AUTH_DIR = Path.home() / ".cli-proxy-api"
ARAGORA_PROFILE_ROOT = Path.home() / ".aragora-claude"
DEFAULT_CONFIG_PATH = Path.home() / ".aragora" / "claude_profile_sync.json"
# Skip a VibeProxy source whose own token expires within this margin, so a
# stopped/crashed proxy's stale token never overwrites a profile.
_STALE_SOURCE_MARGIN_SECONDS = 300

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


class ConfigError(RuntimeError):
    """The sync mapping config is missing or malformed."""


@dataclass(frozen=True)
class SyncConfig:
    sync_target: dict[str, str]  # email -> profile (strict 1:1)
    native_only: dict[str, str]  # profile -> reason

    @property
    def profile_to_email(self) -> dict[str, str]:
        return {profile: email for email, profile in self.sync_target.items()}


@dataclass
class SyncResult:
    profile: str
    email: str
    # synced | skipped_no_source | skipped_stale_source | skipped_fresh
    # | skipped_native_only | skipped_native_login | skipped_no_email | error
    action: str
    detail: str = ""


def _config_path() -> Path:
    override = os.environ.get("ARAGORA_PROFILE_SYNC_CONFIG", "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None) -> SyncConfig:
    path = path or _config_path()
    raw = _load_json(path)
    if raw is None:
        raise ConfigError(
            f"sync mapping config not found or unreadable at {path}. "
            f"Copy {Path(__file__).with_name('claude_profile_sync.json.example')} "
            f"there and fill in your email->profile map."
        )
    sync_target = dict(raw.get("sync_target") or {})
    native_only = dict(raw.get("native_only") or {})
    if not sync_target:
        raise ConfigError(f"config {path} has an empty 'sync_target' map")
    # 1:1 invariant: an email maps to one profile (dict guarantees), and no
    # profile is both a sync target and native-only.
    overlap = set(sync_target.values()) & set(native_only)
    if overlap:
        raise ConfigError(
            f"config {path}: profiles are both synced and native-only: {sorted(overlap)}"
        )
    dupes = [p for p in set(sync_target.values()) if list(sync_target.values()).count(p) > 1]
    if dupes:
        raise ConfigError(f"config {path}: profile(s) mapped from >1 email: {sorted(set(dupes))}")
    return SyncConfig(sync_target=sync_target, native_only=native_only)


def _vibeproxy_path(email: str) -> Path:
    return VIBEPROXY_AUTH_DIR / f"claude-{email}.json"


def _profile_cred_path(profile: str) -> Path:
    return ARAGORA_PROFILE_ROOT / profile / ".claude" / ".credentials.json"


def _iso_to_epoch_ms(value: str) -> int:
    # VibeProxy stamps ISO 8601; tolerate a trailing Z (Py<3.11) and treat a
    # naive stamp as UTC rather than silently as local time.
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = _dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return int(parsed.timestamp() * 1000)


def _now_ms() -> int:
    return int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)


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


def _oauth(cred: dict | None) -> dict:
    return (cred or {}).get("claudeAiOauth") or {}


def _write_owner_only(path: Path, data: str) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)


def sync_profile(
    profile: str,
    config: SyncConfig,
    *,
    blank_refresh: bool,
    apply: bool,
    force: bool = False,
) -> SyncResult:
    if profile in config.native_only:
        return SyncResult(profile, "", "skipped_native_only", config.native_only[profile])
    email = config.profile_to_email.get(profile, "")
    if not email:
        return SyncResult(profile, email, "skipped_no_email", "no VibeProxy source assigned")
    try:
        vp = _load_json(_vibeproxy_path(email))
        if vp is None or vp.get("disabled") or not vp.get("access_token"):
            return SyncResult(profile, email, "skipped_no_source", "no live VibeProxy account")

        # A stopped/crashed proxy leaves a stale (expired) token with disabled
        # still false; syncing it would blank the profile's refresh token AND
        # install a dead access token, i.e. the exact failure this prevents.
        expired_raw = vp.get("expired")
        if not expired_raw:
            return SyncResult(profile, email, "skipped_no_source", "source missing 'expired' field")
        vp_expiry_ms = _iso_to_epoch_ms(expired_raw)
        if vp_expiry_ms <= _now_ms() + _STALE_SOURCE_MARGIN_SECONDS * 1000:
            return SyncResult(
                profile, email, "skipped_stale_source", "VibeProxy token expired/expiring"
            )

        cred_path = _profile_cred_path(profile)
        existing = _load_json(cred_path)
        existing_oauth = _oauth(existing)

        # Idempotent: profile already holds this exact access token.
        if existing_oauth.get("accessToken") == vp["access_token"]:
            return SyncResult(profile, email, "skipped_fresh", "already current")

        # Never clobber a fresh NATIVE login (live access token + real refresh
        # token) without --force: that would destroy a working refresh token.
        existing_refresh = existing_oauth.get("refreshToken") or ""
        existing_exp = existing_oauth.get("expiresAt")
        existing_live = isinstance(existing_exp, int) and existing_exp > _now_ms()
        if existing_refresh and existing_live and not force:
            return SyncResult(
                profile, email, "skipped_native_login", "live native login present (use --force)"
            )

        payload = translate_credential(vp, existing_oauth, blank_refresh=blank_refresh)
        if not apply:
            return SyncResult(profile, email, "synced", "dry-run")

        cred_path.parent.mkdir(parents=True, exist_ok=True)
        # Back up the existing credential (0600) before replacing it.
        if cred_path.exists():
            _write_owner_only(
                cred_path.with_name(".credentials.json.bak"),
                cred_path.read_text(encoding="utf-8"),
            )
        # Atomic, never-world-readable write.
        tmp = cred_path.with_name(".credentials.json.tmp")
        _write_owner_only(tmp, json.dumps(payload))
        os.replace(tmp, cred_path)
        return SyncResult(profile, email, "synced", "applied")
    except (OSError, ValueError, KeyError) as exc:
        # Isolate per-profile failures so one malformed source does not abort the
        # whole batch. Never include token material in the message.
        return SyncResult(profile, email, "error", f"{type(exc).__name__}: {exc}")


def _resolve_profile_tool() -> Path | None:
    sibling = Path(__file__).resolve().parent / "claude_profile.sh"
    if sibling.exists():
        return sibling
    found = shutil.which("claude_profile.sh")
    return Path(found) if found else None


def _probe(profile: str, timeout: int = 60) -> bool | None:
    """Live-probe a profile like claude_pool_verify.py. Returns None if the
    probe tool is unavailable (e.g. the deployed standalone copy)."""
    tool = _resolve_profile_tool()
    if tool is None:
        return None
    try:
        proc = subprocess.run(
            [str(tool), "exec", profile, "--", "claude", "--print", "-p", "-"],
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
        "--force",
        action="store_true",
        help="Overwrite even a profile holding a fresh native login",
    )
    parser.add_argument(
        "--probe-after",
        action="store_true",
        help="After --apply, live-probe each synced profile and report health",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to the sync mapping config")
    parser.add_argument("--json", dest="json_output", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Default set = the 1:1 sync targets; native-only profiles are shown only
    # when named explicitly (so their skip reason is visible on request).
    profiles = args.profiles or list(config.sync_target.values())
    results: list[SyncResult] = [
        sync_profile(
            profile,
            config,
            blank_refresh=not args.keep_refresh,
            apply=args.apply,
            force=args.force,
        )
        for profile in profiles
    ]

    probed: dict[str, bool | None] = {}
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
                state = probed[r.profile]
                probe_note = "  probe=" + (
                    "SKIP" if state is None else ("LIVE" if state else "DEAD")
                )
            print(f"  {r.profile:8} {r.email:28} {r.action:20} {r.detail}{probe_note}")
        synced = sum(1 for r in results if r.action == "synced")
        live = sum(1 for v in probed.values() if v)
        tail = (
            f"; {live}/{sum(1 for v in probed.values() if v is not None)} probed live"
            if probed
            else ""
        )
        print(f"Sync: {synced}/{len(results)} profiles updated{tail}")

    # Surface hard problems via exit code so a launchd log flags them:
    # any per-profile error, a probed-dead profile, or a total source loss
    # (every mapped profile lost its VibeProxy source).
    if any(r.action == "error" for r in results):
        return 1
    if probed and any(v is False for v in probed.values()):
        return 1
    only_targets = [r for r in results if r.action != "skipped_native_only"]
    if only_targets and all(r.action == "skipped_no_source" for r in only_targets):
        print("error: no live VibeProxy sources for any mapped profile", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
