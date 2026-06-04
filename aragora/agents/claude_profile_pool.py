"""Route the debate Claude CLI agent through the authenticated subscription pool.

The debate :class:`~aragora.agents.cli_agents.ClaudeAgent` historically shelled
out to a bare ``claude --print -p -`` against the *default* ``$HOME/.claude``.
On hosts where that default profile is not logged in (the common case for
headless / multi-account fleets) the call hangs or 401s, so decision-integrity
debates silently degraded to a single agent.

The review path (:mod:`aragora.swarm.review_routing`) already solves this by
running ``scripts/claude_profile.sh exec <profile> -- claude ...`` against the
pool of authenticated Claude Max/Team subscriptions under
``~/.aragora-claude/max-*``. This module exposes the same routing for the agents
layer as a small, dependency-light helper so the debate path can reuse the pool
without importing the swarm review stack.

Design notes:
- **Non-breaking.** When ``scripts/claude_profile.sh`` is absent (e.g. an
  installed package rather than a repo checkout) or no usable profile is found,
  the caller is handed the original ``base_cmd`` unchanged.
- **Health-aware.** Profiles flagged unhealthy in the verify-backed snapshot
  (``.aragora/claude_pool_health.json``) are dropped. ``claude auth status`` is
  deliberately *not* consulted: it reports ``loggedIn: true`` even for expired
  tokens (see ``scripts/claude_profiles_bootstrap.sh``), so only the live-probe
  snapshot is trusted. A missing/malformed snapshot leaves the order untouched
  and relies on the caller's existing runtime fallback.
- **Opt-out.** ``ARAGORA_CLAUDE_DISABLE_PROFILE_POOL`` forces the bare command.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Mirror of the review-side constants (aragora/swarm/review_routing.py). Kept
# local so the agents layer does not import the swarm review stack.
_DEFAULT_PROFILES: tuple[str, ...] = tuple(f"max-{index:02d}" for index in range(1, 14))
_POOL_HEALTH_RELATIVE_PATH = ".aragora/claude_pool_health.json"
_UNHEALTHY_PROFILE_STATES = {"expired", "not_configured", "unauthenticated", "logged_out"}
_TRUTHY = {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    # aragora/agents/claude_profile_pool.py -> parents[2] == repo root.
    return Path(__file__).resolve().parents[2]


def _profile_script(repo_root: Path) -> Path | None:
    script = repo_root / "scripts" / "claude_profile.sh"
    return script if script.exists() else None


def _configured_profiles() -> list[str]:
    raw = str(os.environ.get("ARAGORA_CLAUDE_REVIEW_PROFILES", "")).strip()
    if not raw:
        return list(_DEFAULT_PROFILES)
    result: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if name and name not in result:
            result.append(name)
    return result or list(_DEFAULT_PROFILES)


def _pool_health_path(repo_root: Path) -> Path:
    override = str(os.environ.get("ARAGORA_CLAUDE_POOL_HEALTH_FILE", "")).strip()
    if override:
        return Path(override).expanduser()
    return repo_root / _POOL_HEALTH_RELATIVE_PATH


def _pool_health_states(repo_root: Path) -> dict[str, str]:
    """Return ``{profile: state}`` from the verify-backed snapshot, or ``{}``.

    Accepts both shapes seen in the wild: a ``{"profiles": [{"profile", "state"}]}``
    list and a flat ``{profile: state}`` mapping. Any error yields ``{}`` so the
    caller leaves the profile order untouched.
    """
    try:
        payload = json.loads(_pool_health_path(repo_root).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    states: dict[str, str] = {}
    items = payload.get("profiles")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("profile") or item.get("name") or "").strip()
            state = str(item.get("state") or item.get("status") or "").strip().lower()
            if name:
                states[name] = state
    else:
        for key, value in payload.items():
            if key == "generated_at" or not isinstance(value, str):
                continue
            states[str(key)] = value.strip().lower()
    return states


def _healthy_profiles(repo_root: Path, profiles: list[str]) -> list[str]:
    states = _pool_health_states(repo_root)
    if not states:
        # No verify-backed snapshot: we cannot assess health, so keep the full
        # list and let the caller's runtime fallback handle any dead profile.
        return profiles
    # A snapshot exists, so trust it even when it filters everything out: if every
    # configured profile is known-bad, returning an empty list (=> select_profile
    # yields None => bare claude => OpenRouter fallback) is correct. Falling back
    # to the full known-bad list would defeat the safety gate this module exists
    # for. Profiles absent from the snapshot (state is None) are treated as usable.
    return [p for p in profiles if states.get(p) not in _UNHEALTHY_PROFILE_STATES]


def select_profile(*, repo_root: Path | None = None, index: int | None = None) -> str | None:
    """Pick a usable Claude subscription profile, or ``None`` to use bare claude.

    ``ARAGORA_CLAUDE_PROFILE`` pins a specific profile. Otherwise a healthy
    profile is chosen from the pool, rotated by ``index`` (defaulting to the
    current pid so concurrent processes spread across subscriptions).
    """
    if str(os.environ.get("ARAGORA_CLAUDE_DISABLE_PROFILE_POOL", "")).strip().lower() in _TRUTHY:
        return None
    override = str(os.environ.get("ARAGORA_CLAUDE_PROFILE", "")).strip()
    if override:
        return override
    root = repo_root or _repo_root()
    if _profile_script(root) is None:
        return None
    profiles = _healthy_profiles(root, _configured_profiles())
    if not profiles:
        return None
    cursor = os.getpid() if index is None else index
    return profiles[cursor % len(profiles)]


def build_claude_command(
    base_cmd: list[str],
    *,
    repo_root: Path | None = None,
    index: int | None = None,
) -> tuple[list[str], bool]:
    """Return ``(command, used_profile)``.

    When a usable profile is found, wraps ``base_cmd`` in
    ``claude_profile.sh exec <profile> -- <base_cmd>``; otherwise returns
    ``base_cmd`` unchanged so behaviour is identical to the bare-CLI path.
    """
    root = repo_root or _repo_root()
    script = _profile_script(root)
    profile = select_profile(repo_root=root, index=index)
    if not profile or script is None:
        return list(base_cmd), False
    return [str(script), "exec", profile, "--", *base_cmd], True


def strip_profile_preamble(text: str) -> str:
    """Drop the ``claude_profile.sh`` preamble from wrapped CLI output.

    The wrapper emits exactly two preamble lines, in order, *before* the model
    output: ``Using profile home: ...`` then ``Command: ...``. Only that leading,
    in-order block is removed, so a legitimate model answer line that happens to
    start with ``Command:`` (or sits anywhere past the preamble) is preserved.
    """
    lines = text.splitlines()
    index = 0
    if index < len(lines) and lines[index].startswith("Using profile home:"):
        index += 1
        if index < len(lines) and lines[index].startswith("Command:"):
            index += 1
    return "\n".join(lines[index:]).strip()
