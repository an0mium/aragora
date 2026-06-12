"""
EU AI Act Art. 14 / NIST AI 600-1 human-oversight evidence pack (#8230, ODR-6).

Assembles, for a time window, the settled decisions of a repository operating
the credential-separated human gate (this repo's own loop): each decision's
:class:`~aragora.compliance.oversight_attestation.OversightAttestationRecord`,
tamper-evident-trail anchor references when present, and the **canonical**
EU AI Act Article 14 / NIST AI 600-1 crosswalk from the ODR content profile
(``docs/specs/OPEN_DECISION_RECEIPT.md`` section 7 — copied verbatim, never
reinterpreted here).

Sources, in honesty order:

- local settlement receipts under ``.aragora/review-queue/receipts/``
  (trusted operator-controlled store, exact-head bound);
- optionally, merged PRs fetched from GitHub (injected fetcher), classified
  via the gate-trusted ``aragora/human-settlement`` status / Tier-4
  preapproval comment, with everything else recorded as the explicit
  ``autonomous`` disposition;
- the local intent chain ``.aragora/trail/intent-chain.jsonl`` (TET T1) for
  trail anchor references — gracefully recorded absent when the chain does
  not exist yet.

Like the attestation extractor, the pack never fabricates: every gap is an
explicit absence with a reason, because a visibly weak evidence pack is worth
more to an auditor than a strong-looking fabricated one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from aragora.compliance.oversight_attestation import (
    attestation_from_local_settlement_receipt,
    classify_settled_pr,
)
from aragora.gauntlet.odr_export import jcs_canonicalize

logger = logging.getLogger(__name__)

__all__ = [
    "ART14_NIST_CROSSWALK",
    "OversightEvidencePack",
    "build_oversight_pack",
    "collect_github_settlements",
    "collect_local_settlements",
    "collect_trail_anchors",
]

PACK_SCHEMA_VERSION = "1.0"

DEFAULT_RECEIPTS_DIR = Path(".aragora") / "review-queue" / "receipts"
DEFAULT_TRAIL_CHAIN = Path(".aragora") / "trail" / "intent-chain.jsonl"

# Canonical regulatory crosswalk — copied VERBATIM from
# docs/specs/OPEN_DECISION_RECEIPT.md section 7 ("Compliance mapping —
# EU AI Act Art. 14 / NIST AI 600-1"). That spec is the single source of
# truth for clause interpretation; this module must not add or alter clause
# readings. The table maps *evidence availability*, not legal conformity.
ART14_NIST_CROSSWALK: list[dict[str, str]] = [
    {
        "odr_field": "subject (binding + digest)",
        "eu_ai_act_art14": (
            '14(4)(a) — enables the overseer to "duly monitor" exactly which '
            "input the decision concerns"
        ),
        "nist_ai_600_1": "GV-1.2 / MP-2: documented system context and provenance of inputs",
    },
    {
        "odr_field": "claim.verdict",
        "eu_ai_act_art14": "14(4)(c) — output the human must be able to correctly interpret",
        "nist_ai_600_1": "MS-2.5: traceable system outputs",
    },
    {
        "odr_field": "reasoning.summary",
        "eu_ai_act_art14": (
            '14(4)(c)/(d) — interpretation aids; basis for deciding "not to use" the output'
        ),
        "nist_ai_600_1": "MS-2.8: documented rationale supporting explanation",
    },
    {
        "odr_field": "quorum.participants + independence",
        "eu_ai_act_art14": (
            "14(4)(b) — awareness of automation bias is operationalized by "
            "disclosing model-family homogeneity"
        ),
        "nist_ai_600_1": ("GV-6.1 / MP-5.1: third-party/model diversity and provenance disclosure"),
    },
    {
        "odr_field": "quorum.dissent",
        "eu_ai_act_art14": (
            "14(4)(d) — preserved dissent gives the overseer concrete grounds "
            "to disregard the output"
        ),
        "nist_ai_600_1": "MS-3.3: capture of disagreement/uncertainty in evaluation",
    },
    {
        "odr_field": "confidence + calibration",
        "eu_ai_act_art14": (
            "14(4)(b)/(c) — calibrated (or honestly uncalibrated) confidence counters over-reliance"
        ),
        "nist_ai_600_1": "MS-2.3 / MS-4: measured, documented confidence with provenance",
    },
    {
        "odr_field": "cruxes",
        "eu_ai_act_art14": (
            "14(4)(d) — identifies the load-bearing points a human should "
            "probe before overriding or accepting"
        ),
        "nist_ai_600_1": "MP-2.3: identification of decision-critical assumptions",
    },
    {
        "odr_field": "attestation",
        "eu_ai_act_art14": (
            "14(4)(e) — records whether a human exercised the ability to "
            "intervene; `autonomous` makes non-intervention auditable"
        ),
        "nist_ai_600_1": (
            "GV-3.2: human oversight roles and responsibilities are recorded per decision"
        ),
    },
    {
        "odr_field": "signatures / JCS digest (section 5-6)",
        "eu_ai_act_art14": (
            "14(1) — effective oversight presupposes the record itself is trustworthy"
        ),
        "nist_ai_600_1": "MS-2.7: integrity/verifiability of AI system records",
    },
    {
        "odr_field": "source",
        "eu_ai_act_art14": (
            "14(4)(a) — path back to full-fidelity native record for deeper monitoring"
        ),
        "nist_ai_600_1": "GV-1.5: auditability via linked provenance",
    },
]


def _portable_path(path: Path) -> str:
    """Render a path with the user's home directory abbreviated to ``~``.

    Evidence packs are committed/shared artifacts; absolute home-rooted paths
    are both non-portable and leak local layout.
    """
    try:
        return "~/" + path.resolve().relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def _parse_iso(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class OversightEvidencePack:
    """Assembled Art. 14 / NIST AI 600-1 human-oversight evidence bundle."""

    generated_at: str
    window_days: int
    window_start: str
    repo: str | None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    trail: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def human_attested_count(self) -> int:
        return sum(1 for d in self.decisions if d["attestation"]["disposition"] == "human_attested")

    @property
    def autonomous_count(self) -> int:
        return sum(1 for d in self.decisions if d["attestation"]["disposition"] == "autonomous")

    @property
    def absence_count(self) -> int:
        return sum(len(d["attestation"].get("absences") or []) for d in self.decisions)

    def summary(self) -> dict[str, Any]:
        return {
            "decisions": len(self.decisions),
            "human_attested": self.human_attested_count,
            "autonomous": self.autonomous_count,
            "recorded_absences": self.absence_count,
            "trail_anchors_present": bool(self.trail.get("records")),
        }

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": PACK_SCHEMA_VERSION,
            "kind": "aragora.oversight_evidence_pack",
            "generated_at": self.generated_at,
            "window_days": self.window_days,
            "window_start": self.window_start,
            "repo": self.repo,
            "summary": self.summary(),
            "decisions": self.decisions,
            "trail": self.trail,
            "regulatory_crosswalk": {
                "source": (
                    "docs/specs/OPEN_DECISION_RECEIPT.md section 7 (canonical; "
                    "copied verbatim, maps evidence availability, not legal "
                    "conformity — conformity assessment remains the deployer's "
                    "process)"
                ),
                "eu_ai_act_article": "Article 14 (Human oversight)",
                "nist_profile": "NIST AI 600-1 (Generative AI Profile)",
                "rows": ART14_NIST_CROSSWALK,
            },
            "notes": self.notes,
        }
        body["integrity_hash"] = hashlib.sha256(jcs_canonicalize(body)).hexdigest()
        return body

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        data = self.to_dict()
        lines: list[str] = []
        lines.append("# Human-Oversight Evidence Pack — EU AI Act Art. 14 / NIST AI 600-1")
        lines.append("")
        lines.append(f"- **Generated:** {self.generated_at}")
        lines.append(f"- **Window:** last {self.window_days} days (since {self.window_start})")
        if self.repo:
            lines.append(f"- **Repository:** {self.repo}")
        lines.append(f"- **Integrity hash (JCS/SHA-256):** `{data['integrity_hash']}`")
        lines.append("")

        summary = data["summary"]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Settled decisions in window: **{summary['decisions']}**")
        lines.append(
            f"- Human-attested (oversight identity recorded): **{summary['human_attested']}**"
        )
        lines.append(
            f"- Autonomous (explicit non-intervention disposition): **{summary['autonomous']}**"
        )
        lines.append(
            f"- Recorded absences (honesty markers, not gaps papered over): "
            f"**{summary['recorded_absences']}**"
        )
        lines.append("")

        lines.append("## Settled decisions and attestations")
        lines.append("")
        lines.append(
            "| Decision | Disposition | Oversight identity | Attested at | Mechanism | Head SHA | Evidence digest |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for decision in self.decisions:
            att = decision["attestation"]
            subject = att.get("subject") or {}
            label = decision.get("label") or (
                f"PR #{subject['pr_number']}" if subject.get("pr_number") else "(unidentified)"
            )
            observed = att.get("observed") or {}
            head = observed.get("head_sha") or subject.get("head_sha") or "—"
            head_short = head[:12] if head != "—" else head
            digest = observed.get("evidence_digest") or "—"
            digest_short = digest if digest == "—" else f"`{str(digest)[:20]}…`"
            lines.append(
                "| {label} | {disp} | {who} | {when} | {mech} | `{head}` | {digest} |".format(
                    label=label,
                    disp=att["disposition"],
                    who=att.get("attestor_id") or "—",
                    when=att.get("attested_at") or "—",
                    mech=att.get("mechanism") or "—",
                    head=head_short,
                    digest=digest_short,
                )
            )
        if not self.decisions:
            lines.append("| (none in window) | — | — | — | — | — | — |")
        lines.append("")

        absences = [
            (decision.get("label") or "", absence)
            for decision in self.decisions
            for absence in (decision["attestation"].get("absences") or [])
        ]
        lines.append("## Recorded absences")
        lines.append("")
        if absences:
            lines.append(
                "The following fields are genuinely missing from the source "
                "settlement artifacts. They are recorded, not implied or "
                "fabricated:"
            )
            lines.append("")
            for label, absence in absences:
                lines.append(f"- **{label}** — `{absence['field']}`: {absence['reason']}")
        else:
            lines.append("None. Every attestation field was extractable from a real artifact.")
        lines.append("")

        lines.append("## Tamper-evident trail anchors")
        lines.append("")
        trail = data["trail"]
        if trail.get("records"):
            lines.append(f"- Chain path: `{trail.get('path')}`")
            lines.append(f"- Records in window: {len(trail['records'])}")
            lines.append(f"- Chain head hash: `{trail.get('head_hash')}`")
            lines.append(f"- Chain verified: {trail.get('verified')}")
        else:
            lines.append(f"- {trail.get('note') or 'No trail records available for this window.'}")
        lines.append("")

        lines.append("## EU AI Act Article 14 / NIST AI 600-1 crosswalk")
        lines.append("")
        lines.append(
            "Canonical mapping from the Open Decision Receipt content profile "
            "(`docs/specs/OPEN_DECISION_RECEIPT.md` section 7). It maps *evidence "
            "availability*, not legal conformity; conformity assessment remains "
            "the deployer's process."
        )
        lines.append("")
        lines.append(
            "| ODR field | EU AI Act Art. 14 (Human oversight) | NIST AI 600-1 (GenAI profile) |"
        )
        lines.append("|---|---|---|")
        for row in ART14_NIST_CROSSWALK:
            lines.append(
                f"| `{row['odr_field']}` | {row['eu_ai_act_art14']} | {row['nist_ai_600_1']} |"
            )
        lines.append("")

        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")
        return "\n".join(lines)


def collect_local_settlements(
    receipts_dir: str | Path,
    *,
    window_start: datetime,
) -> list[dict[str, Any]]:
    """Collect attestations from local settlement receipts inside the window.

    Reads ``pr-*.json`` receipts from the review-queue store; receipts that
    fail to parse are skipped with a log line (never invented). Only receipts
    whose ``reviewed_at`` falls inside the window are included; receipts
    without a parseable ``reviewed_at`` are excluded from a *windowed* pack
    because their recency cannot be established.
    """
    directory = Path(receipts_dir)
    decisions: list[dict[str, Any]] = []
    if not directory.is_dir():
        return decisions
    for path in sorted(directory.glob("pr-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable settlement receipt %s: %s", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        reviewed_at = _parse_iso(str(payload.get("reviewed_at") or ""))
        if reviewed_at is None or reviewed_at < window_start:
            continue
        record = attestation_from_local_settlement_receipt(
            payload, receipt_path=_portable_path(path)
        )
        pr_number = payload.get("pr_number")
        decisions.append(
            {
                "label": f"PR #{pr_number}" if pr_number else path.name,
                "source": "local_settlement_receipt",
                "attestation": record.to_dict(),
                "odr_attestation": record.to_odr_attestation(),
            }
        )
    return decisions


def collect_github_settlements(
    *,
    repo: str,
    window_start: datetime,
    gh_json: Callable[[list[str]], Any],
    limit: int = 100,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Classify merged PRs from GitHub into oversight attestations.

    ``gh_json`` is an injected callable executing a ``gh`` CLI invocation and
    returning parsed JSON (tests stub it; the CLI provides a subprocess-backed
    one). Each merged PR in the window is classified via the gate-trusted
    ``aragora/human-settlement`` commit status (creator login = oversight
    identity) with Tier-4 preapproval-comment fallback; everything else gets
    the explicit ``autonomous`` disposition. Fetch failures degrade to notes,
    never to invented data.
    """
    decisions: list[dict[str, Any]] = []
    notes: list[str] = []
    since = window_start.date().isoformat()
    try:
        merged = gh_json(
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "merged",
                "--search",
                f"merged:>={since}",
                "--limit",
                str(limit),
                "--json",
                "number,title,mergedAt,headRefOid,url",
            ]
        )
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning("GitHub merged-PR listing failed: %s", exc)
        notes.append(
            f"GitHub merged-PR listing for {repo} failed; pack contains local "
            "settlement receipts only"
        )
        return decisions, notes

    for pr in merged or []:
        if not isinstance(pr, dict):
            continue
        merged_at = _parse_iso(str(pr.get("mergedAt") or ""))
        if merged_at is None or merged_at < window_start:
            continue
        number = pr.get("number")
        head_sha = str(pr.get("headRefOid") or "").strip() or None

        statuses: list[dict[str, Any]] = []
        if head_sha:
            try:
                raw_statuses = gh_json(["api", f"repos/{repo}/commits/{head_sha}/statuses"])
                if isinstance(raw_statuses, list):
                    statuses = [s for s in raw_statuses if isinstance(s, dict)]
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("Status fetch failed for PR #%s: %s", number, exc)
                notes.append(
                    f"PR #{number}: commit-status fetch failed; oversight "
                    "classification fell back to the autonomous disposition "
                    "check without status evidence"
                )

        comments: list[dict[str, Any]] = []
        has_settlement_status = any(
            str(s.get("context") or "") == "aragora/human-settlement" for s in statuses
        )
        if has_settlement_status and number is not None:
            try:
                raw_comments = gh_json(["api", f"repos/{repo}/issues/{number}/comments"])
                if isinstance(raw_comments, list):
                    comments = [c for c in raw_comments if isinstance(c, dict)]
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("Comment fetch failed for PR #%s: %s", number, exc)

        record = classify_settled_pr(
            repo=repo,
            pr_number=int(number) if number is not None else None,
            head_sha=head_sha,
            statuses=statuses,
            comments=comments,
        )
        decision: dict[str, Any] = {
            "label": f"PR #{number}" if number is not None else "(unnumbered PR)",
            "source": "github_merged_pr",
            "title": str(pr.get("title") or ""),
            "url": str(pr.get("url") or ""),
            "merged_at": str(pr.get("mergedAt") or ""),
            "attestation": record.to_dict(),
            "odr_attestation": record.to_odr_attestation(),
        }
        decisions.append(decision)
    return decisions, notes


def collect_trail_anchors(
    chain_path: str | Path,
    *,
    window_start: datetime,
) -> dict[str, Any]:
    """Collect intent-chain anchor references for the window, gracefully.

    Reads ``.aragora/trail/intent-chain.jsonl`` (TET T1) when present. A
    missing or empty chain is reported as an explicit note — expected while
    the trail is young — never as fabricated anchors.
    """
    path = Path(chain_path)
    if not path.is_file():
        return {
            "path": _portable_path(path),
            "records": [],
            "note": (
                "intent chain not present at this path yet (TET T1 chain file "
                "absent); no trail anchors to reference"
            ),
        }
    try:
        from aragora.trail.intent_chain import chain_head_hash, read_records, verify_chain

        records = read_records(path)
        verified, bad_index = verify_chain(path)
        head_hash = chain_head_hash(path)
    except (OSError, ValueError) as exc:
        logger.warning("Intent chain read failed for %s: %s", path, exc)
        return {
            "path": _portable_path(path),
            "records": [],
            "note": "intent chain exists but could not be read/verified; see logs",
        }

    windowed = []
    for rec in records:
        ts = _parse_iso(str(rec.get("ts") or rec.get("timestamp") or ""))
        if ts is not None and ts < window_start:
            continue
        windowed.append(
            {
                "seq": rec.get("seq"),
                "record_hash": rec.get("record_hash"),
                "intent_type": rec.get("intent_type"),
                "ts": rec.get("ts") or rec.get("timestamp"),
            }
        )
    result: dict[str, Any] = {
        "path": _portable_path(path),
        "records": windowed,
        "head_hash": head_hash,
        "verified": bool(verified),
    }
    if not verified:
        result["note"] = f"chain verification FAILED at record index {bad_index}"
    elif not windowed:
        result["note"] = "intent chain present but has no records in this window"
    return result


def build_oversight_pack(
    *,
    window_days: int = 30,
    repo: str | None = None,
    receipts_dir: str | Path | None = None,
    trail_chain_path: str | Path | None = None,
    gh_json: Callable[[list[str]], Any] | None = None,
    github_limit: int = 100,
    now: datetime | None = None,
) -> OversightEvidencePack:
    """Assemble the human-oversight evidence pack for the window.

    Pure assembly over the injected sources; the CLI wires real paths and a
    ``gh``-backed fetcher. ``repo`` + ``gh_json`` enable the GitHub layer
    (merged-PR classification incl. explicit autonomous dispositions); without
    them the pack contains the local settlement-receipt layer only, with a
    note saying so.
    """
    current = now or datetime.now(timezone.utc)
    window_start = current - timedelta(days=window_days)

    pack = OversightEvidencePack(
        generated_at=current.isoformat(),
        window_days=window_days,
        window_start=window_start.isoformat(),
        repo=repo,
    )

    local = collect_local_settlements(
        receipts_dir if receipts_dir is not None else DEFAULT_RECEIPTS_DIR,
        window_start=window_start,
    )

    github_decisions: list[dict[str, Any]] = []
    if repo and gh_json is not None:
        github_decisions, gh_notes = collect_github_settlements(
            repo=repo,
            window_start=window_start,
            gh_json=gh_json,
            limit=github_limit,
        )
        pack.notes.extend(gh_notes)
    else:
        pack.notes.append(
            "GitHub settlement layer not queried (no repo/fetcher supplied); "
            "pack reflects the local settlement-receipt store only and "
            "therefore cannot enumerate autonomous merges"
        )

    # GitHub decisions take precedence for the same PR (richer classification);
    # local receipts cover everything else, including PRs settled while
    # offline.
    github_pr_numbers = {
        d["attestation"]["subject"].get("pr_number")
        for d in github_decisions
        if d["attestation"]["subject"].get("pr_number") is not None
    }
    for decision in local:
        pr_number = decision["attestation"]["subject"].get("pr_number")
        if pr_number is not None and pr_number in github_pr_numbers:
            for gh_decision in github_decisions:
                if gh_decision["attestation"]["subject"].get("pr_number") == pr_number:
                    gh_decision.setdefault("corroborating_sources", []).append(
                        {
                            "type": "local_settlement_receipt",
                            "attestation": decision["attestation"],
                        }
                    )
            continue
        github_decisions.append(decision)

    def _sort_key(decision: dict[str, Any]) -> tuple[int, str]:
        pr_number = decision["attestation"]["subject"].get("pr_number") or 0
        return (int(pr_number), str(decision.get("label") or ""))

    pack.decisions = sorted(github_decisions, key=_sort_key)
    pack.trail = collect_trail_anchors(
        trail_chain_path if trail_chain_path is not None else DEFAULT_TRAIL_CHAIN,
        window_start=window_start,
    )
    return pack
