"""Pure helpers for the Claude profile-pool verify snapshot.

``scripts/claude_pool_verify.py`` live-probes each profile and writes
``.aragora/claude_pool_health.json``. That snapshot is the *verify-backed* health
source consumed by:
  - ``aragora/swarm/review_routing.py`` (skip unhealthy review profiles), and
  - ``aragora/agents/claude_profile_pool.py`` (skip unhealthy debate profiles).

``claude auth status`` is deliberately not used for liveness: it reports
``loggedIn: true`` even for expired/revoked tokens. Only a real completion probe
distinguishes a usable profile, so this module classifies *probe output*.

Snapshot shape (matches the format both consumers parse)::

    {"generated_at": "...Z", "healthy": 2, "total": 13,
     "profiles": [{"name": "max-01", "email": "a@x", "state": "ok"}, ...]}
"""

from __future__ import annotations

# States consumers treat as unusable (mirror review_routing._UNHEALTHY_PROFILE_STATES).
UNHEALTHY_STATES = {"expired", "not_configured", "unauthenticated", "logged_out"}

_AUTH_FAILURE_MARKERS = (
    "401",
    "invalid authentication",
    "failed to authenticate",
    "unauthorized",
    "oauth token has expired",
)
_NOT_CONFIGURED_MARKERS = (
    "no such file",
    "not logged in",
    "no credentials",
    "claude_profile.sh not found",
)


def classify_probe(stdout: str, *, returncode: int | None = None, timed_out: bool = False) -> str:
    """Map a completion-probe result to a health state.

    - ``ok``: the probe produced real model output.
    - ``expired``: an auth failure (401 / invalid / expired token) — the common
      revoked/expired case.
    - ``not_configured``: the profile has no usable credentials at all.
    - ``unauthenticated``: timed out or produced no output (treated as unusable).
    """
    text = (stdout or "").strip()
    lowered = text.lower()
    if timed_out:
        return "unauthenticated"
    if any(marker in lowered for marker in _NOT_CONFIGURED_MARKERS):
        return "not_configured"
    if any(marker in lowered for marker in _AUTH_FAILURE_MARKERS):
        return "expired"
    if not text:
        # No output and not a recognized error: cannot confirm liveness.
        return "unauthenticated"
    if returncode not in (None, 0):
        return "expired"
    return "ok"


def is_healthy(state: str) -> bool:
    return state not in UNHEALTHY_STATES


def build_snapshot(records: list[dict], *, generated_at: str) -> dict:
    """Build the snapshot dict from per-profile ``{name,email,state}`` records."""
    profiles = [
        {
            "name": str(r.get("name", "")),
            "email": str(r.get("email", "") or ""),
            "state": str(r.get("state", "unauthenticated")),
        }
        for r in records
    ]
    healthy = sum(1 for p in profiles if is_healthy(p["state"]))
    return {
        "generated_at": generated_at,
        "healthy": healthy,
        "total": len(profiles),
        "profiles": profiles,
    }
