"""Tests for :mod:`aragora.pdb.prompts`.

Covers the prompt-shape invariants required by the spec:

- every phase binds ``repo``, ``pr_number``, ``base_sha``, ``head_sha``
- findings/critique prompts preserve ``core`` / ``heterodox`` /
  ``regulatory`` lens identity
- findings prompt requests structured JSON (not essays) per the
  "request structured findings, not essays" rule
- critique prompt includes peer findings from OTHER slots only
- synthesis prompt invites disagreement preservation and lists every
  panel vote verbatim
- binding header is shared across phases (one source of truth)
"""

from __future__ import annotations

from aragora.pdb.panel_config import PDBPanelSlot
from aragora.pdb.prompts import (
    critique_prompt,
    findings_prompt,
    render_binding_header,
    synthesis_prompt,
)
from aragora.review.builder import PanelVote
from aragora.review.protocol import DissentPosition, ReviewRole, RoleFinding
from aragora.swarm.pr_review_protocol import PRReviewBinding


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _binding() -> PRReviewBinding:
    return PRReviewBinding(
        repo="synaptent/aragora",
        pr_number=4242,
        base_sha="deadbeef1111",
        head_sha="cafef00d2222",
    )


def _slot(slot_id: str, lens: str, review_role: str = "logic_reviewer") -> PDBPanelSlot:
    return PDBPanelSlot(
        slot_id=slot_id,
        review_role=review_role,
        lens=lens,
        family=slot_id.split("_", 1)[0],
        candidates=(f"{slot_id}-cli",),
        required=(lens == "core"),
    )


def _vote(slot_id: str, role: ReviewRole, position: DissentPosition, reason: str) -> PanelVote:
    return PanelVote(
        finding=RoleFinding(
            role=role,
            agent=f"{slot_id}:mock",
            model="mock-1",
            confidence=0.8,
            finding_text=f"{slot_id} says {position.value}",
        ),
        position=position,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Binding header
# ---------------------------------------------------------------------------


def test_binding_header_contains_all_four_anchors() -> None:
    text = render_binding_header(
        binding=_binding(),
        pr_title="Refactor budget accounting",
        pr_body="Description here",
        labels=("backend", "priority:high"),
        changed_files=("aragora/pdb/budget.py",),
    )
    for needle in ("synaptent/aragora", "4242", "deadbeef1111", "cafef00d2222"):
        assert needle in text
    assert "backend" in text
    assert "priority:high" in text


def test_binding_header_truncates_very_long_body() -> None:
    text = render_binding_header(
        binding=_binding(),
        pr_title="t",
        pr_body="x" * 5000,
        labels=(),
        changed_files=(),
    )
    assert "[truncated]" in text


def test_binding_header_changed_files_truncation() -> None:
    many = tuple(f"src/file_{i}.py" for i in range(60))
    text = render_binding_header(
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=many,
    )
    assert "and 20 more" in text


# ---------------------------------------------------------------------------
# Findings prompt
# ---------------------------------------------------------------------------


def _require_binding_anchors(text: str) -> None:
    for needle in ("synaptent/aragora", "4242", "deadbeef1111", "cafef00d2222"):
        assert needle in text, f"prompt missing anchor {needle!r}"


def test_findings_prompt_binds_anchors_and_lens_core() -> None:
    slot = _slot("claude_core", "core", "logic_reviewer")
    text = findings_prompt(
        slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="body",
        labels=(),
        changed_files=("a.py",),
        diff_excerpt="diff --git a/a.py b/a.py\n+pass\n",
        validation_summary={"checks": "green"},
    )
    _require_binding_anchors(text)
    assert "CORE lens" in text
    assert "slot_id: claude_core" in text
    assert "review_role: logic_reviewer" in text
    assert "family: claude" in text
    # structured-JSON contract
    assert '"recommendation"' in text
    assert '"confidence"' in text
    assert '"top_findings"' in text
    assert '"contested_finding_ids"' in text
    assert '"reason"' in text


def test_findings_prompt_preserves_heterodox_lens_identity() -> None:
    slot = _slot("grok_h", "heterodox", "skeptic")
    text = findings_prompt(
        slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=(),
        diff_excerpt="",
    )
    _require_binding_anchors(text)
    assert "HETERODOX lens" in text
    assert "Disagreement is the point" in text


def test_findings_prompt_preserves_regulatory_lens_identity() -> None:
    slot = _slot("mistral_r", "regulatory", "skeptic")
    text = findings_prompt(
        slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=(),
        diff_excerpt="",
    )
    _require_binding_anchors(text)
    assert "REGULATORY lens" in text
    # Must be framed as a perspective, not a compliance oracle
    assert "perspective" in text
    assert "compliance oracle" in text


def test_findings_prompt_preserves_advocate_lens_identity() -> None:
    # Advocate lens (#6505 fix #3) is the counterweight to the skeptic
    # lenses. Prompt must ask for the STRONGEST case FOR the PR, not
    # flatten into approval-cheerleading.
    slot = _slot("claude_advocate", "advocate", "skeptic")
    text = findings_prompt(
        slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=(),
        diff_excerpt="",
    )
    _require_binding_anchors(text)
    assert "ADVOCATE lens" in text
    # Framed as a position, not a rubber stamp
    assert "STRONGEST case FOR" in text
    # The anti-invention guard is load-bearing: prevents the advocate
    # from manufacturing benefits absent from the diff.
    assert "Prefer an empty findings list to manufactured virtues" in text


def test_findings_prompt_handles_missing_validation_summary() -> None:
    slot = _slot("claude_core", "core", "logic_reviewer")
    text = findings_prompt(
        slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=(),
        diff_excerpt="",
        validation_summary=None,
    )
    assert "no validation signals provided" in text


# ---------------------------------------------------------------------------
# Critique prompt
# ---------------------------------------------------------------------------


def test_critique_prompt_binds_anchors_and_peer_findings() -> None:
    slot = _slot("gpt_core", "core", "security_reviewer")
    text = critique_prompt(
        slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=(),
        peer_findings={
            "claude_core": "recommendation: approve\nsummary: looks good",
            "grok_h": "recommendation: defer\nsummary: concerns with scope",
        },
    )
    _require_binding_anchors(text)
    assert "claude_core" in text
    assert "grok_h" in text
    # Must not include itself among peers
    assert "### peer slot: gpt_core" not in text
    # Structured-JSON contract for critique
    assert '"agrees_with"' in text
    assert '"disagrees_with"' in text
    assert '"contested_finding_ids"' in text
    # Heterogeneity instruction preserved
    assert "record dissent explicitly" in text


def test_critique_prompt_handles_empty_peers() -> None:
    slot = _slot("claude_core", "core", "logic_reviewer")
    text = critique_prompt(
        slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=(),
        peer_findings={},
    )
    _require_binding_anchors(text)
    assert "no peer findings" in text


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------


def test_synthesis_prompt_lists_panel_votes_and_preserves_dissent() -> None:
    slot = _slot("claude_core", "core", "logic_reviewer")
    votes = (
        _vote("claude_core", ReviewRole.LOGIC, DissentPosition.APPROVE, "Clean."),
        _vote(
            "gpt_core",
            ReviewRole.SECURITY,
            DissentPosition.REQUEST_CHANGES,
            "Missing input validation.",
        ),
        _vote("grok_h", ReviewRole.SKEPTIC, DissentPosition.DEFER, "Unclear rollout plan."),
    )
    text = synthesis_prompt(
        synthesizer_slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=(),
        votes=votes,
    )
    _require_binding_anchors(text)
    # The synthesizer's own lens must still be visible
    assert "CORE lens" in text
    # Every vote must appear (slot position + reason)
    assert "approve" in text.lower()
    assert "request_changes" in text
    assert "defer" in text
    assert "Missing input validation" in text
    # JSON contract
    assert '"top_line"' in text
    assert '"validation_summary"' in text
    assert '"preserved_dissent"' in text
    # Preserves dissent rather than flattening
    assert "Preserve disagreement" in text
    # Builder-controls-recommendation rule is explicit
    assert "do NOT override it" in text


def test_synthesis_prompt_handles_empty_votes() -> None:
    slot = _slot("claude_core", "core", "logic_reviewer")
    text = synthesis_prompt(
        synthesizer_slot=slot,
        binding=_binding(),
        pr_title="t",
        pr_body="",
        labels=(),
        changed_files=(),
        votes=(),
    )
    _require_binding_anchors(text)
    assert "no panel votes" in text
