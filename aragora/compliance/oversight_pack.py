"""EU AI Act Article 14 / NIST oversight evidence-pack generator (#8230).

Assembles decision receipts and their human-oversight attestations over a
time window into a single audit-ready bundle: per-receipt oversight
dispositions, attestor identities, mechanisms, an Article 14 clause mapping,
and a NIST AI RMF cross-reference — with an integrity digest over the
canonicalized payload.

Honesty rules (matching the ODR export's):

- Dispositions are **recorded, never implied**: a receipt with no attestation
  is counted as ``autonomous``, not skipped.
- Receipts that cannot be windowed (no parseable timestamp) are excluded and
  *counted as excluded*, not silently dropped.
- Clause statuses are computed from the evidence actually present in the
  window (``satisfied`` / ``partial``), never asserted unconditionally.

CLI: ``aragora compliance oversight-pack --window 30d`` (see
``aragora/cli/commands/compliance.py``). Clause mapping documentation:
``docs/compliance/ART14_OVERSIGHT_PACK.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aragora.gauntlet.attestation import (
    AUTONOMOUS_DISPOSITION,
    HUMAN_ATTESTED_DISPOSITION,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARTICLE_14_CLAUSES",
    "NIST_CROSS_REFERENCES",
    "build_oversight_pack",
    "load_receipts_from_dirs",
    "render_oversight_pack_markdown",
]

PACK_ARTIFACT = "aragora.oversight_pack"
PACK_VERSION = "0.1"

# EU AI Act (Regulation (EU) 2024/1689) Article 14 clause map. Each entry
# names the requirement and the receipt-level evidence the pack derives its
# status from. Documented for auditors in docs/compliance/ART14_OVERSIGHT_PACK.md.
ARTICLE_14_CLAUSES: list[dict[str, str]] = [
    {
        "clause": "14(1)",
        "requirement": (
            "High-risk AI systems shall be designed and developed such that "
            "they can be effectively overseen by natural persons during use."
        ),
        "evidence_basis": (
            "A dedicated oversight identity layer, credential-separated from "
            "execution identities; receipts carry per-decision oversight "
            "dispositions (human_attested / autonomous)."
        ),
    },
    {
        "clause": "14(2)",
        "requirement": (
            "Oversight shall aim at preventing or minimising risks to health, "
            "safety or fundamental rights."
        ),
        "evidence_basis": (
            "Adversarial multi-model review precedes decisions; receipts "
            "record verdicts, confidence, and preserved dissent so overseers "
            "act on risk signals rather than summaries."
        ),
    },
    {
        "clause": "14(3)",
        "requirement": (
            "Oversight measures shall be built into the system or identified "
            "for implementation by the deployer."
        ),
        "evidence_basis": (
            "The settlement mechanism (head-bound human-settlement status, "
            "preapproval comments) is machine-checkable and cited per "
            "attestation in this pack."
        ),
    },
    {
        "clause": "14(4)(a)",
        "requirement": (
            "Overseers can understand the capacities and limitations of the "
            "system and monitor its operation."
        ),
        "evidence_basis": (
            "Receipts expose agent responses, confidence, dissenting views, "
            "and provenance chains for every windowed decision."
        ),
    },
    {
        "clause": "14(4)(b)",
        "requirement": "Overseers remain aware of automation bias.",
        "evidence_basis": (
            "Heterogeneous-model dissent is preserved verbatim in receipts "
            "rather than averaged away; hollow-consensus detection is part of "
            "the debate protocol."
        ),
    },
    {
        "clause": "14(4)(c)",
        "requirement": "Overseers can correctly interpret the system's output.",
        "evidence_basis": (
            "Receipts carry verdict reasoning and explainability blocks; "
            "attestations record the exact evidence digest the overseer saw."
        ),
    },
    {
        "clause": "14(4)(d)",
        "requirement": (
            "Overseers can decide not to use the system or otherwise "
            "disregard, override or reverse its output."
        ),
        "evidence_basis": (
            "Human settlement is the merge/act gate for oversight-tier "
            "decisions: absence of a human attestation withholds settlement; "
            "dispositions record which decisions a human actually accepted."
        ),
    },
    {
        "clause": "14(4)(e)",
        "requirement": (
            "Overseers can intervene in the operation or interrupt the system "
            "(stop button or similar)."
        ),
        "evidence_basis": (
            "Operational kill-switches and halt files stop autonomous lanes; "
            "interruption capability is procedural and referenced, not "
            "evidenced per-receipt (status is at most 'partial' from receipts "
            "alone)."
        ),
    },
]

# NIST AI RMF 1.0 cross-references for the same oversight evidence.
NIST_CROSS_REFERENCES: list[dict[str, str]] = [
    {
        "function": "GOVERN 3.2",
        "requirement": (
            "Policies define human oversight roles and responsibilities for AI systems."
        ),
        "evidence_basis": "Oversight identity layer and per-decision attestor records.",
    },
    {
        "function": "MANAGE 2.4",
        "requirement": (
            "Mechanisms are in place to supersede, disengage, or deactivate "
            "AI systems that demonstrate performance or outcomes inconsistent "
            "with intended use."
        ),
        "evidence_basis": "Human settlement gate; withheld attestation withholds action.",
    },
    {
        "function": "MEASURE 2.10",
        "requirement": "Human-AI configurations are evaluated (automation bias).",
        "evidence_basis": "Preserved dissent and confidence in every windowed receipt.",
    },
]


def _parse_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


INVALID_ATTESTATION_DISPOSITION = "invalid_attestation"


def _attestation_invalid_reason(block: Mapping[str, Any]) -> str | None:
    """Why a ``human_attested`` block does NOT count as oversight evidence.

    The pack never trusts a bare ``{"disposition": "human_attested"}``: to be
    counted, a block must name who attested, when, and via what mechanism —
    and must satisfy the same identity-separation rule the attestation
    builder enforces (attestor != execution identity). Returns ``None`` when
    the block is countable.
    """
    attestor = block.get("attestor")
    attestor_id = str(attestor.get("id", "") or "").strip() if isinstance(attestor, Mapping) else ""
    if not attestor_id:
        return "missing attestor.id"
    if not str(block.get("attested_at", "") or "").strip():
        return "missing attested_at"
    mech = block.get("mechanism")
    mech_type = str(mech.get("type", "") or "").strip() if isinstance(mech, Mapping) else ""
    if not mech_type:
        return "missing mechanism.type"
    execution = block.get("execution_identity")
    execution_id = (
        str(execution.get("id", "") or "").strip() if isinstance(execution, Mapping) else ""
    )
    if execution_id and execution_id.lower() == attestor_id.lower():
        return "attestor equals execution identity"
    return None


def _receipt_entry(
    receipt: Mapping[str, Any],
    attestations: Mapping[str, Mapping[str, Any]],
    source: str | None,
) -> dict[str, Any]:
    """Normalize a native DecisionReceipt dict or an ODR document."""
    is_odr = "odr_version" in receipt
    receipt_id = str(receipt.get("receipt_id", "") or "")

    attestation: Mapping[str, Any] | None = None
    if is_odr and isinstance(receipt.get("attestation"), Mapping):
        attestation = receipt["attestation"]
    if receipt_id and receipt_id in attestations:
        attestation = attestations[receipt_id]

    disposition = AUTONOMOUS_DISPOSITION
    attestor_id: str | None = None
    mechanism: str | None = None
    observed: dict[str, Any] = {}
    attested_at: str | None = None
    invalid_reason: str | None = None
    if attestation:
        disposition = str(attestation.get("disposition", HUMAN_ATTESTED_DISPOSITION))
        if disposition == HUMAN_ATTESTED_DISPOSITION:
            invalid_reason = _attestation_invalid_reason(attestation)
        if invalid_reason:
            # Malformed oversight claims are recorded visibly, never counted
            # as human oversight and never silently dropped.
            disposition = INVALID_ATTESTATION_DISPOSITION
        else:
            attestor = attestation.get("attestor") or {}
            attestor_id = str(attestor.get("id", "") or "") or None
            mech = attestation.get("mechanism") or {}
            mechanism = str(mech.get("type", "") or "") or None
            if isinstance(attestation.get("observed"), Mapping):
                observed = dict(attestation["observed"])
            attested_at = str(attestation.get("attested_at", "") or "") or None

    if is_odr:
        timestamp = receipt.get("issued_at")
        claim = receipt.get("claim") or {}
        verdict = str(claim.get("decision", "") or "") if isinstance(claim, Mapping) else ""
        confidence_block = receipt.get("confidence") or {}
        confidence = (
            confidence_block.get("value") if isinstance(confidence_block, Mapping) else None
        )
    else:
        timestamp = receipt.get("timestamp") or receipt.get("produced_at")
        verdict = str(receipt.get("verdict", "") or "")
        confidence = receipt.get("confidence")

    entry: dict[str, Any] = {
        "receipt_id": receipt_id,
        "format": "odr" if is_odr else "native",
        "timestamp": timestamp,
        "verdict": verdict,
        "confidence": confidence,
        "disposition": disposition,
    }
    if attestor_id:
        entry["attestor"] = attestor_id
    if mechanism:
        entry["mechanism"] = mechanism
    if attested_at:
        entry["attested_at"] = attested_at
    if observed:
        entry["observed"] = observed
    if invalid_reason:
        entry["attestation_invalid_reason"] = invalid_reason
    if source:
        entry["source"] = source
    return entry


def _clause_status(
    clause: str, human_attested: int, total: int, settlements: int = 0
) -> tuple[str, str]:
    """Honest per-clause status from windowed evidence.

    ``settlements`` counts human-settlement attestations fetched from the
    repository trail (see :mod:`aragora.compliance.oversight_fetch`) — real
    oversight events with resolvable refs, counted for the identity-dependent
    clauses alongside receipt-level attestations.
    """
    human_evidence = human_attested + settlements
    if clause == "14(4)(e)":
        return (
            "partial",
            "interruption capability is procedural (kill-switches, halt files); "
            "not evidenced per-receipt by this pack",
        )
    if clause in ("14(1)", "14(3)", "14(4)(d)"):
        # Identity-dependent clauses need at least one real human attestation.
        if human_evidence > 0:
            basis = f"{human_attested}/{total} windowed decisions human-attested"
            if settlements:
                basis += f"; {settlements} human-settlement attestation(s) from the trail"
            return "satisfied", basis
        return (
            "partial",
            "oversight mechanism exists but no windowed decision carries a human attestation",
        )
    if total == 0:
        return "partial", "no receipts in window"
    return "satisfied", f"evidence present on all {total} windowed receipts"


def load_receipts_from_dirs(
    directories: list[str | Path],
) -> list[tuple[dict[str, Any], str]]:
    """Load ``(receipt_dict, source_path)`` pairs from ``*.json`` files.

    Files that are not JSON objects or carry no ``receipt_id`` are skipped
    with a debug log — directories may hold unrelated JSON.
    """
    loaded: list[tuple[dict[str, Any], str]] = []
    for directory in directories:
        root = Path(directory).expanduser()
        if not root.is_dir():
            logger.debug("oversight_pack: skipping missing directory %s", root)
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.debug("oversight_pack: unreadable %s: %s", path, exc)
                continue
            if isinstance(data, dict) and data.get("receipt_id"):
                loaded.append((data, str(path)))
    return loaded


def build_oversight_pack(
    receipts: Sequence[tuple[Mapping[str, Any], str | None] | Mapping[str, Any]],
    *,
    window_days: int = 30,
    now: datetime | None = None,
    attestations: Mapping[str, Mapping[str, Any]] | None = None,
    settlement_attestations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the Article 14 / NIST oversight evidence pack.

    Args:
        receipts: Receipt dicts (native ``DecisionReceipt.to_dict()`` or ODR
            documents), optionally paired with their source path.
        window_days: Rolling window; receipts older than this are excluded.
        now: Window anchor (defaults to current UTC; injectable for tests).
        attestations: Optional ``receipt_id -> attestation block`` mapping,
            overriding/supplying attestations for native receipts (e.g. built
            with :mod:`aragora.gauntlet.attestation`).
        settlement_attestations: Optional human-settlement attestation entries
            from the repository trail (the ``attestations`` list returned by
            :func:`aragora.compliance.oversight_fetch.fetch_settlement_attestations`)
            — real oversight events counted alongside receipt-level
            attestations for the identity-dependent Article 14 clauses.

    Returns:
        JSON-serializable pack dict with an integrity digest.
    """
    anchor = now or datetime.now(timezone.utc)
    cutoff = anchor - timedelta(days=window_days)
    attestation_map = dict(attestations or {})

    entries: list[dict[str, Any]] = []
    excluded_no_timestamp = 0
    excluded_out_of_window = 0
    for item in receipts:
        if isinstance(item, tuple):
            receipt, source = item
        else:
            receipt, source = item, None
        entry = _receipt_entry(receipt, attestation_map, source)
        parsed = _parse_timestamp(entry.get("timestamp"))
        if parsed is None:
            excluded_no_timestamp += 1
            continue
        if parsed < cutoff or parsed > anchor:
            excluded_out_of_window += 1
            continue
        entries.append(entry)

    entries.sort(key=lambda e: str(e.get("timestamp", "")))

    dispositions = Counter(e["disposition"] for e in entries)
    human_attested = dispositions.get(HUMAN_ATTESTED_DISPOSITION, 0)
    attestors = sorted({e["attestor"] for e in entries if e.get("attestor")})
    mechanisms = Counter(e["mechanism"] for e in entries if e.get("mechanism"))

    settlements: list[dict[str, Any]] = []
    settlements_invalid: list[dict[str, Any]] = []
    for raw_settlement in settlement_attestations or []:
        settlement = dict(raw_settlement)
        block = settlement.get("attestation") or {}
        # Same trust boundary as receipt-level blocks: a settlement entry is
        # counted only when its attestation names who/when/how and satisfies
        # identity separation; malformed entries are reported, not counted.
        reason = _attestation_invalid_reason(block) if isinstance(block, Mapping) else "no block"
        if reason:
            settlements_invalid.append({**settlement, "invalid_reason": reason})
            continue
        settlements.append(settlement)
        attestor = (block.get("attestor") or {}).get("id")
        if attestor and attestor not in attestors:
            attestors.append(attestor)
        mech = (block.get("mechanism") or {}).get("type")
        if mech:
            mechanisms[mech] += 1
    attestors.sort()

    article_14 = []
    for clause_def in ARTICLE_14_CLAUSES:
        status, basis = _clause_status(
            clause_def["clause"], human_attested, len(entries), settlements=len(settlements)
        )
        article_14.append({**clause_def, "status": status, "status_basis": basis})

    pack: dict[str, Any] = {
        "artifact": PACK_ARTIFACT,
        "version": PACK_VERSION,
        "framework": {
            "eu_ai_act": "Regulation (EU) 2024/1689, Article 14 (Human Oversight)",
            "nist": "NIST AI RMF 1.0 (cross-referenced functions)",
        },
        "generated_at": anchor.isoformat(),
        "window": {
            "days": window_days,
            "start": cutoff.isoformat(),
            "end": anchor.isoformat(),
        },
        "summary": {
            "receipts_in_window": len(entries),
            "human_attested": human_attested,
            "autonomous": dispositions.get(AUTONOMOUS_DISPOSITION, 0),
            "other_dispositions": {
                k: v
                for k, v in dispositions.items()
                if k not in (HUMAN_ATTESTED_DISPOSITION, AUTONOMOUS_DISPOSITION)
            },
            "attestor_identities": attestors,
            "mechanisms": dict(mechanisms),
            "settlement_attestations": len(settlements),
            "settlement_attestations_invalid": len(settlements_invalid),
            "excluded_no_timestamp": excluded_no_timestamp,
            "excluded_out_of_window": excluded_out_of_window,
        },
        "receipts": entries,
        "settlement_attestations": settlements,
        "settlement_attestations_invalid": settlements_invalid,
        "article_14_mapping": article_14,
        "nist_cross_references": list(NIST_CROSS_REFERENCES),
    }

    # Integrity digest over the canonical payload (RFC 8785 JCS, same basis
    # as ODR content digests) so the bundle itself is tamper-evident.
    from aragora.gauntlet.odr_export import jcs_canonicalize

    pack["integrity"] = {
        "algorithm": "sha256/jcs",
        "content_digest": hashlib.sha256(jcs_canonicalize(pack)).hexdigest(),
    }
    return pack


def render_oversight_pack_markdown(pack: Mapping[str, Any]) -> str:
    """Render the pack as an auditor-readable Markdown report."""
    summary = pack.get("summary", {})
    window = pack.get("window", {})
    lines = [
        "# Human Oversight Evidence Pack (EU AI Act Article 14)",
        "",
        f"Generated: {pack.get('generated_at', '')}",
        f"Window: {window.get('days', '?')} days "
        f"({window.get('start', '')} → {window.get('end', '')})",
        f"Integrity: `{pack.get('integrity', {}).get('content_digest', '')}` (sha256/jcs)",
        "",
        "## Summary",
        "",
        f"- Receipts in window: **{summary.get('receipts_in_window', 0)}**",
        f"- Human-attested: **{summary.get('human_attested', 0)}**",
        f"- Autonomous (explicitly recorded): **{summary.get('autonomous', 0)}**",
        f"- Attestor identities: {', '.join(summary.get('attestor_identities', [])) or '(none)'}",
        f"- Mechanisms: {json.dumps(summary.get('mechanisms', {}))}",
        f"- Excluded (no timestamp): {summary.get('excluded_no_timestamp', 0)}; "
        f"out of window: {summary.get('excluded_out_of_window', 0)}",
        "",
        "## Article 14 clause mapping",
        "",
        "| Clause | Status | Basis |",
        "|---|---|---|",
    ]
    for clause in pack.get("article_14_mapping", []):
        lines.append(
            f"| {clause.get('clause', '')} | {clause.get('status', '')} | "
            f"{clause.get('status_basis', '')} |"
        )
    lines += [
        "",
        "## NIST AI RMF cross-references",
        "",
        "| Function | Evidence basis |",
        "|---|---|",
    ]
    for ref in pack.get("nist_cross_references", []):
        lines.append(f"| {ref.get('function', '')} | {ref.get('evidence_basis', '')} |")
    settlements = pack.get("settlement_attestations", [])
    if settlements:
        lines += [
            "",
            "## Human-settlement attestations (repository trail)",
            "",
            "| PR | Attestor | Attested at | Head SHA | Ref |",
            "|---|---|---|---|---|",
        ]
        for settlement in settlements:
            block = settlement.get("attestation", {})
            attestor = (block.get("attestor") or {}).get("id", "")
            observed = block.get("observed") or {}
            mech = block.get("mechanism") or {}
            lines.append(
                f"| #{settlement.get('pr', '')} | {attestor} | "
                f"{block.get('attested_at', '')} | "
                f"`{str(observed.get('head_sha', ''))[:12]}` | "
                f"{mech.get('ref', '')} |"
            )
    lines += [
        "",
        "## Receipts",
        "",
        "| Receipt | Timestamp | Verdict | Disposition | Attestor | Mechanism |",
        "|---|---|---|---|---|---|",
    ]
    for entry in pack.get("receipts", []):
        lines.append(
            f"| `{str(entry.get('receipt_id', ''))[:16]}` | {entry.get('timestamp', '')} | "
            f"{entry.get('verdict', '')} | {entry.get('disposition', '')} | "
            f"{entry.get('attestor', '')} | {entry.get('mechanism', '')} |"
        )
    lines.append("")
    lines.append(
        "Clause requirement texts and evidence definitions: "
        "`docs/compliance/ART14_OVERSIGHT_PACK.md`."
    )
    lines.append("")
    return "\n".join(lines)
