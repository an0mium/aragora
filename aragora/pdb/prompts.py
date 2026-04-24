"""Prompt templates for PDB Mode 3 Protocol B.

Three prompt families, one per phase:

- :func:`findings_prompt` — first-round per-slot structured findings
- :func:`critique_prompt` — second-round peer critique with peer
  findings as context
- :func:`synthesis_prompt` — single synthesis pass binding panel votes
  into a final brief

Every template binds the four PR anchors (``repo``, ``pr_number``,
``base_sha``, ``head_sha``) so provider output can be cross-checked
against the binding, and preserves ``core`` / ``heterodox`` /
``regulatory`` lens identity so dissent does not collapse during
critique or synthesis.

This module is **prompt-only**. It does not invoke providers, does not
stream, and does not parse responses. Tests in
``tests/pdb/test_prompts.py`` assert the rendered text contains the
binding fields and the lens identifiers.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from aragora.pdb.panel_config import PDBPanelSlot
from aragora.review.builder import PanelVote
from aragora.swarm.pr_review_protocol import PRReviewBinding

__all__ = [
    "LENS_INSTRUCTIONS",
    "critique_prompt",
    "findings_prompt",
    "render_binding_header",
    "synthesis_prompt",
]


LENS_INSTRUCTIONS: Mapping[str, str] = {
    "core": (
        "You hold the CORE lens. Read the diff with the assumption the change "
        "compiles and ships; your job is to catch correctness, logic, and "
        "security defects before merge. Stay close to the code."
    ),
    "heterodox": (
        "You hold the HETERODOX lens. Deliberately argue the weakest case for "
        "this PR: maintainability debt, hidden coupling, mis-scoped abstractions, "
        "and long-tail risks the core reviewers will miss. Disagreement is the "
        "point; do not defer to the majority."
    ),
    "regulatory": (
        "You hold the REGULATORY lens. Read the diff from a European / regulated-"
        "market perspective. Call out failures of duty-of-care, data-handling, "
        "disclosure, or cross-border obligations WITHOUT claiming the code "
        "itself is unlawful — you are a perspective, not a compliance oracle."
    ),
}


# ---------------------------------------------------------------------------
# Binding header
# ---------------------------------------------------------------------------


def render_binding_header(
    *,
    binding: PRReviewBinding,
    pr_title: str,
    pr_body: str,
    labels: Sequence[str],
    changed_files: Sequence[str],
) -> str:
    """Render the shared PR-binding preamble used by all three phases.

    Keeping this in one function ensures every prompt emits the exact
    same binding text, which makes response cross-validation trivial:
    any prompt that lacks ``repo`` / ``pr_number`` / ``base_sha`` /
    ``head_sha`` is a bug.
    """
    labels_line = ", ".join(labels) if labels else "(none)"
    changed_preview = "\n".join(f"- {path}" for path in changed_files[:40])
    if len(changed_files) > 40:
        changed_preview += f"\n- ... and {len(changed_files) - 40} more"
    body_trimmed = (pr_body or "").strip()
    if len(body_trimmed) > 2000:
        body_trimmed = body_trimmed[:2000] + "\n... [truncated]"
    return (
        "## PR binding\n"
        f"repo: {binding.repo}\n"
        f"pr_number: {binding.pr_number}\n"
        f"base_sha: {binding.base_sha}\n"
        f"head_sha: {binding.head_sha}\n"
        f"title: {pr_title}\n"
        f"labels: {labels_line}\n\n"
        "## PR description\n"
        f"{body_trimmed or '(empty)'}\n\n"
        "## Changed files\n"
        f"{changed_preview or '(none)'}"
    )


def _lens_block(slot: PDBPanelSlot) -> str:
    instructions = LENS_INSTRUCTIONS.get(
        slot.lens,
        f"You hold the {slot.lens.upper()} lens. Write findings grounded in the "
        "diff; do not flatten your stance to match other reviewers.",
    )
    return (
        "## Your assignment\n"
        f"slot_id: {slot.slot_id}\n"
        f"review_role: {slot.review_role}\n"
        f"lens: {slot.lens}\n"
        f"family: {slot.family}\n\n"
        f"{instructions}"
    )


# ---------------------------------------------------------------------------
# Findings round
# ---------------------------------------------------------------------------


def findings_prompt(
    *,
    slot: PDBPanelSlot,
    binding: PRReviewBinding,
    pr_title: str,
    pr_body: str,
    labels: Sequence[str],
    changed_files: Sequence[str],
    diff_excerpt: str,
    validation_summary: Mapping[str, object] | None = None,
) -> str:
    """Render the first-round findings prompt for a single slot.

    The template asks for STRUCTURED output, not essays. Response
    parsing is PR3 territory; PR2 tests only assert the rendered text
    carries the binding and lens identifiers.
    """
    header = render_binding_header(
        binding=binding,
        pr_title=pr_title,
        pr_body=pr_body,
        labels=labels,
        changed_files=changed_files,
    )
    lens = _lens_block(slot)
    validation_block = _format_validation_summary(validation_summary)
    diff_block = (diff_excerpt or "(no diff provided)").rstrip()

    return (
        "# PR Decision Brief — findings round (Protocol B)\n\n"
        f"{header}\n\n"
        f"{lens}\n\n"
        "## Validation signals\n"
        f"{validation_block}\n\n"
        "## Diff excerpt\n"
        "```diff\n"
        f"{diff_block}\n"
        "```\n\n"
        "## Output contract\n"
        "Return a compact JSON object with these keys (no prose outside the JSON):\n"
        '  - "recommendation": one of "approve", "request_changes", "defer".\n'
        '  - "confidence": float in [0.0, 1.0].\n'
        '  - "top_findings": array (up to 5) of objects with '
        '{"finding_id", "category", "severity" ('
        '"low"|"medium"|"high"), "summary", "evidence" (array of short strings)}.\n'
        '  - "contested_finding_ids": array of finding_ids you expect other lenses\n'
        "    may disagree on (may be empty).\n"
        '  - "reason": one-to-two-sentence justification of your recommendation\n'
        "    grounded in your lens; do not flatten your stance to match the\n"
        "    majority.\n\n"
        "Rules:\n"
        "- Cite specific files or hunks in evidence when possible.\n"
        "- Preserve your lens identity; disagreement is a feature, not a bug.\n"
        "- Do not invent findings absent from the diff; prefer an empty list to\n"
        "  fabricated risk.\n"
    )


def _format_validation_summary(summary: Mapping[str, object] | None) -> str:
    if not summary:
        return "(no validation signals provided)"
    rows = []
    for key, value in summary.items():
        rows.append(f"- {key}: {value}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Critique round
# ---------------------------------------------------------------------------


def critique_prompt(
    *,
    slot: PDBPanelSlot,
    binding: PRReviewBinding,
    pr_title: str,
    pr_body: str,
    labels: Sequence[str],
    changed_files: Sequence[str],
    peer_findings: Mapping[str, str],
) -> str:
    """Render the second-round critique prompt for a single slot.

    ``peer_findings`` maps ``slot_id`` → a pre-serialized summary of
    that peer's findings (JSON text is fine). The critique phase is
    where dissent should crystallize, so the template explicitly
    invites disagreement with other lenses.
    """
    header = render_binding_header(
        binding=binding,
        pr_title=pr_title,
        pr_body=pr_body,
        labels=labels,
        changed_files=changed_files,
    )
    lens = _lens_block(slot)
    peer_block_rows = []
    for peer_slot_id, summary in peer_findings.items():
        if peer_slot_id == slot.slot_id:
            continue
        peer_block_rows.append(
            f"### peer slot: {peer_slot_id}\n{summary.strip() or '(empty summary)'}"
        )
    peer_block = "\n\n".join(peer_block_rows) if peer_block_rows else "(no peer findings)"

    return (
        "# PR Decision Brief — critique round (Protocol B)\n\n"
        f"{header}\n\n"
        f"{lens}\n\n"
        "## Peer findings for critique\n"
        f"{peer_block}\n\n"
        "## Output contract\n"
        "Return a compact JSON object with these keys (no prose outside the JSON):\n"
        '  - "recommendation": one of "approve", "request_changes", "defer"\n'
        "    (your updated position after reading peers).\n"
        '  - "confidence": float in [0.0, 1.0].\n'
        '  - "agrees_with": array of peer slot_ids you broadly agree with\n'
        "    (may be empty).\n"
        '  - "disagrees_with": array of peer slot_ids you push back on\n'
        "    (may be empty).\n"
        '  - "contested_finding_ids": array of finding_ids still in dispute.\n'
        '  - "reason": one-to-two-sentence critique summary grounded in your\n'
        "    lens.\n\n"
        "Rules:\n"
        "- Do not pretend to agree where you do not; record dissent explicitly.\n"
        "- Keep peer-citation grounded: name the slot_id you are critiquing.\n"
    )


# ---------------------------------------------------------------------------
# Synthesis pass
# ---------------------------------------------------------------------------


def synthesis_prompt(
    *,
    synthesizer_slot: PDBPanelSlot,
    binding: PRReviewBinding,
    pr_title: str,
    pr_body: str,
    labels: Sequence[str],
    changed_files: Sequence[str],
    votes: Sequence[PanelVote],
) -> str:
    """Render the single synthesis prompt for the panel.

    The synthesizer is not a majority-averager. This template is
    explicit: the synthesizer should surface disagreement rather than
    flatten it, because the landed :func:`aragora.review.builder.build_brief`
    already computes a deterministic recommendation from the votes.
    The synthesizer's job in Protocol B is to produce the top-line
    summary and validation-paragraph text surrounding those votes.
    """
    header = render_binding_header(
        binding=binding,
        pr_title=pr_title,
        pr_body=pr_body,
        labels=labels,
        changed_files=changed_files,
    )
    lens = _lens_block(synthesizer_slot)
    vote_rows = []
    for vote in votes:
        vote_rows.append(
            "- {slot}: {role}/{lens_label} → {position} (confidence {conf:.2f}) — {reason}".format(
                slot=vote.finding.agent,
                role=vote.finding.role.value,
                lens_label=_infer_lens_label(vote),
                position=vote.position.value,
                conf=vote.finding.confidence,
                reason=(vote.reason or "").strip() or "(no reason given)",
            )
        )
    vote_block = "\n".join(vote_rows) if vote_rows else "(no panel votes)"

    return (
        "# PR Decision Brief — synthesis (Protocol B)\n\n"
        f"{header}\n\n"
        f"{lens}\n\n"
        "## Panel votes to synthesize\n"
        f"{vote_block}\n\n"
        "## Output contract\n"
        "Return a compact JSON object with these keys (no prose outside the JSON):\n"
        '  - "top_line": 1-3 sentence executive summary of the panel\'s verdict.\n'
        '  - "validation_summary": one paragraph summarising validation signals,\n'
        "    CI state, and evidence strength.\n"
        '  - "preserved_dissent": array of {"slot_id", "lens", "position",\n'
        '    "reason"} objects, one for every dissenting view, verbatim — do\n'
        "    not collapse into one summary line.\n\n"
        "Rules:\n"
        "- Preserve disagreement explicitly in preserved_dissent; this is the\n"
        "  layer that dissenters survive into the final brief.\n"
        "- The brief's recommendation class is set by the builder from the\n"
        "  votes; do NOT override it here.\n"
    )


def _infer_lens_label(vote: PanelVote) -> str:
    """Best-effort lens label for the synthesis template.

    :class:`PanelVote` does not carry a lens field (it comes from the
    landed builder, which is lens-agnostic). We fall back to the role
    suffix to give the synthesizer something stable to cite.
    """
    role = vote.finding.role.value
    return role.split("_", 1)[0]
