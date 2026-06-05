"""Audit the Claude subscription-profile pool for subscription-seat collisions.

The pool (``~/.aragora-claude/max-*``, driven by ``scripts/claude_profile.sh``)
backs the debate/review Claude agents. Each profile is one authenticated
claude.ai *session*, but a session is keyed by the *organization* it bills
inference to — not by the login email. A single login (email) can belong to
several organizations (e.g. a personal Max org and a shared Team org), and two
profiles that select **different** orgs are genuinely distinct subscription
seats even when their email matches.

The failure that actually hurts the pool is two profiles sharing the **same
org** (same ``orgId``): they draw on one subscription seat / rate-limit pool and
a fresh login to that org can rotate the other's session. That collision can
occur with the *same* email (a true duplicate) or — easy to miss — with two
*different* emails that both belong to the same Team org.

This module is the pure, dependency-free analysis. ``scripts/audit_claude_profiles.py``
collects the live identities (via ``claude auth status``) and renders the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfileIdentity:
    """One profile's claude.ai identity, parsed from ``claude auth status``."""

    profile: str
    email: str = ""
    org_id: str = ""
    org_name: str = ""
    plan: str = ""
    # Optional live-probe / expiry result: True=usable, False=expired/401,
    # None=unknown. Audit logic never *depends* on this; it is passed through
    # for the rendered report.
    token_live: bool | None = None


@dataclass(frozen=True)
class CollisionGroup:
    kind: str  # "org_seat" | "shared_credential"
    severity: str  # "high" | "warn"
    key: str  # the shared org_id or email
    label: str  # human label (org name or email)
    profiles: tuple[str, ...]
    detail: str


@dataclass
class AuditResult:
    profile_count: int
    distinct_org_count: int
    distinct_email_count: int
    org_seat_collisions: list[CollisionGroup] = field(default_factory=list)
    shared_credential_collisions: list[CollisionGroup] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def has_collisions(self) -> bool:
        return bool(self.org_seat_collisions or self.shared_credential_collisions)

    def to_dict(self) -> dict:
        def _grp(g: CollisionGroup) -> dict:
            return {
                "kind": g.kind,
                "severity": g.severity,
                "key": g.key,
                "label": g.label,
                "profiles": list(g.profiles),
                "detail": g.detail,
            }

        return {
            "profile_count": self.profile_count,
            "distinct_org_count": self.distinct_org_count,
            "distinct_email_count": self.distinct_email_count,
            "org_seat_collisions": [_grp(g) for g in self.org_seat_collisions],
            "shared_credential_collisions": [_grp(g) for g in self.shared_credential_collisions],
            "recommendations": list(self.recommendations),
        }


def _group(identities, key):
    out: dict[str, list[ProfileIdentity]] = {}
    for ident in identities:
        value = (key(ident) or "").strip()
        if value:
            out.setdefault(value, []).append(ident)
    return out


def analyze_profiles(identities: list[ProfileIdentity]) -> AuditResult:
    """Compute subscription-seat and shared-credential collisions.

    - **org_seat** (high): >=2 profiles share a non-empty ``org_id`` — same
      subscription seat. The dangerous one; covers same-email duplicates *and*
      different-email-same-Team-org collisions.
    - **shared_credential** (warn): >=2 profiles share an ``email`` but span
      >=2 distinct ``org_id`` values — distinct subscriptions reached through one
      login. Usually fine, but the user-level session may still contend.
    """
    by_org = _group(identities, lambda i: i.org_id)
    by_email = _group(identities, lambda i: i.email)

    org_seat: list[CollisionGroup] = []
    for org_id, members in sorted(by_org.items()):
        if len(members) < 2:
            continue
        names = sorted(m.profile for m in members)
        label = next((m.org_name for m in members if m.org_name), org_id)
        org_seat.append(
            CollisionGroup(
                kind="org_seat",
                severity="high",
                key=org_id,
                label=label,
                profiles=tuple(names),
                detail=(
                    f"{', '.join(names)} all bill inference to org '{label}' "
                    f"({org_id[:8]}) — one subscription seat shared across "
                    f"{len(names)} profiles."
                ),
            )
        )

    shared_cred: list[CollisionGroup] = []
    for email, members in sorted(by_email.items()):
        if len(members) < 2:
            continue
        distinct_orgs = {m.org_id for m in members if m.org_id}
        if len(distinct_orgs) < 2:
            # Same email + (same or unknown org) is already captured by org_seat
            # when the org matches; nothing distinct to flag here.
            continue
        names = sorted(m.profile for m in members)
        shared_cred.append(
            CollisionGroup(
                kind="shared_credential",
                severity="warn",
                key=email,
                label=email,
                profiles=tuple(names),
                detail=(
                    f"{', '.join(names)} share login '{email}' across "
                    f"{len(distinct_orgs)} different orgs — distinct subscriptions, "
                    "but one user-level session token may still rotate the others."
                ),
            )
        )

    recommendations: list[str] = []
    for g in org_seat:
        keep, *drop = g.profiles
        recommendations.append(
            f"Consolidate org-seat collision on '{g.label}': keep {keep}, free "
            f"{', '.join(drop)}, and point the freed slot(s) at a distinct account."
        )
    for g in shared_cred:
        recommendations.append(
            f"Shared login '{g.label}' spans {', '.join(g.profiles)}: prefer "
            "separate logins per profile to avoid session-level contention."
        )

    return AuditResult(
        profile_count=len(identities),
        distinct_org_count=len({i.org_id for i in identities if i.org_id}),
        distinct_email_count=len({i.email for i in identities if i.email}),
        org_seat_collisions=org_seat,
        shared_credential_collisions=shared_cred,
        recommendations=recommendations,
    )
