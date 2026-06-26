"""Tests for the [P0]/[P1]/[P2] non-finding false-positive in the verdict scanner.

`has_blocking_or_negative_verdict` is the blocking-language scan used at BOTH
collect-time and gate-time to decide whether a reviewer's evidence counts. It
treated ANY line starting with `[P0]`/`[P1]`/`[P2]` as a blocking finding — including
`"[P1] None:"` / `"[P1] N/A"`, which low-cost models (and claude/codex) emit to
say "no P0/P1/P2 issues". That silently de-counted a PASS vote.

The fix must be conservative: a real finding that merely *starts with* "none"/"no"
(e.g. `[P1] None of the inputs are validated`, `[P1] no auth check`) MUST still
block. Only an explicit no-finding *head* (the text before any colon being exactly
"none", "n/a", "no issues", ...) is treated as non-blocking.
"""

from __future__ import annotations

import pytest

from aragora.cli.commands.review_queue_comment_verdicts import (
    has_blocking_finding_or_label,
    has_blocking_model_dissent,
    highest_blocking_severity,
    highest_finding_priority,
    has_blocking_or_negative_verdict,
)


# --- the bug: explicit no-finding heads must NOT block --------------------


def test_p1_none_colon_is_not_blocking():
    # The exact qwen/claude pattern that was de-counting PASS votes.
    assert not has_blocking_or_negative_verdict("- [P1] None: defense-in-depth is solid")


def test_p1_bare_none_is_not_blocking():
    assert not has_blocking_or_negative_verdict("- [P1] none")


def test_p0_no_issues_colon_is_not_blocking():
    assert not has_blocking_or_negative_verdict("- [P0] No issues: clean implementation")


def test_p1_na_is_not_blocking():
    assert not has_blocking_or_negative_verdict("- [P1] N/A")


def test_p1_none_found_is_not_blocking():
    assert not has_blocking_or_negative_verdict("- [P1] none found")


def test_p1_no_blocking_findings_is_not_blocking():
    assert not has_blocking_or_negative_verdict("- [P1] no blocking findings")


def test_full_pass_body_with_p1_none_lines_counts():
    body = (
        "## Qwen independent model review\n\n"
        "Model family: qwen\n\n"
        "Verdict: PASS\n"
        "- [P1] None: comprehensive guard conditions\n"
        "- [P2] None: tests cover the mismatch path\n\n"
        "dogfood: yes\n"
    )
    assert not has_blocking_or_negative_verdict(body)


# --- adversarial integrity: real findings MUST still block ----------------


def test_real_p1_finding_starting_with_none_still_blocks():
    # "None of the inputs..." is a REAL finding that happens to start with "none".
    assert has_blocking_or_negative_verdict("- [P1] None of the inputs are validated")


def test_real_p1_finding_starting_with_no_still_blocks():
    assert has_blocking_or_negative_verdict("- [P1] no authentication on the admin endpoint")


def test_real_p1_finding_still_blocks():
    assert has_blocking_or_negative_verdict("- [P1] settlement gate can be bypassed")


def test_bare_p1_tag_blocks_conservatively():
    # A bare "[P1]" with no text is ambiguous; default to blocking.
    assert has_blocking_or_negative_verdict("[P1]")


def test_p0_real_finding_still_blocks():
    assert has_blocking_or_negative_verdict("[P0] settlement gate bypass")


def test_p2_real_finding_still_blocks():
    assert has_blocking_or_negative_verdict("[P2] prepared apply bypasses the freeze proof")


def test_p3_finding_remains_non_blocking():
    assert not has_blocking_or_negative_verdict("[P3] clarify operator warning text")


def test_negative_verdict_line_still_blocks():
    assert has_blocking_or_negative_verdict("Verdict: CHANGES-REQUESTED")


def test_fenced_parser_examples_do_not_block():
    body = (
        "Verdict: PASS\n\n"
        "The parser needs to understand examples like:\n"
        "```markdown\n"
        "Verdict: CHANGES-REQUESTED\n"
        "[P2] This is quoted gate syntax, not a live finding.\n"
        "Blockers: no authentication on admin endpoint\n"
        "```\n"
        "The implementation is otherwise safe."
    )
    assert has_blocking_or_negative_verdict(body) is False
    assert has_blocking_finding_or_label(body) is False
    assert highest_blocking_severity(body) is None


def test_blockquoted_parser_examples_do_not_block():
    body = (
        "Verdict: PASS\n\n"
        "> Verdict: CHANGES-REQUESTED\n"
        "> [P1] Quoted example text, not a live finding.\n"
        "> Blockers: no validation\n"
        "\n"
        "No live blockers."
    )
    assert has_blocking_or_negative_verdict(body) is False
    assert has_blocking_finding_or_label(body) is False
    assert highest_blocking_severity(body) is None


def test_live_blockquoted_finding_still_blocks():
    body = "Verdict: PASS\n\n> [P1] settlement gate bypass remains live reviewer output\n"
    assert has_blocking_or_negative_verdict(body) is True
    assert has_blocking_finding_or_label(body) is True
    assert highest_blocking_severity(body) == "P1"


def test_prose_starting_with_backticks_does_not_open_fence():
    body = (
        "Verdict: PASS\n"
        "``` is used for code examples in markdown prose.\n"
        "- [P1] live settlement gate bypass remains\n"
    )
    assert has_blocking_or_negative_verdict(body) is True
    assert has_blocking_finding_or_label(body) is True
    assert highest_blocking_severity(body) == "P1"


def test_single_line_code_span_does_not_open_fence():
    body = (
        "Verdict: PASS\n"
        "```[P1] quoted inline example```\n"
        "- [P1] live settlement gate bypass remains\n"
    )
    assert has_blocking_or_negative_verdict(body) is True
    assert has_blocking_finding_or_label(body) is True
    assert highest_blocking_severity(body) == "P1"


def test_unclosed_fence_fails_closed_for_priority_finding():
    body = (
        "Verdict: PASS\n```markdown\n- [P1] unclosed quoted finding cannot be silently discarded\n"
    )
    assert has_blocking_or_negative_verdict(body) is True
    assert has_blocking_finding_or_label(body) is True
    assert highest_blocking_severity(body) == "P1"


def test_indented_parser_example_does_not_block():
    body = (
        "Verdict: PASS\n\n"
        "    Verdict: CHANGES-REQUESTED\n"
        "    [P1] indented parser example, not a live finding\n"
        "No live blockers.\n"
    )
    assert has_blocking_or_negative_verdict(body) is False
    assert has_blocking_finding_or_label(body) is False
    assert highest_blocking_severity(body) is None


def test_real_finding_after_fenced_example_still_blocks():
    body = (
        "Verdict: PASS\n"
        "```\n"
        "[P1] quoted example ignored\n"
        "```\n"
        "- [P1] live settlement gate bypass remains"
    )
    assert has_blocking_or_negative_verdict(body) is True
    assert has_blocking_finding_or_label(body) is True
    assert highest_blocking_severity(body) == "P1"


# --- Blocker-label path: bare "no" finding bypass (openai #8574 P1) --------
#
# `_NON_BLOCKING_PREFIXES` used to include a bare "no", so a populated Blocker
# label whose finding text started with "no" ("no authentication", "no
# validation", ...) was wrongly demoted to advisory under the severity gate — a
# merge-gate bypass for common security phrasing. The invariant: a populated
# Blocker label ALWAYS blocks.


@pytest.mark.parametrize(
    "body",
    [
        "Blockers: no authentication on admin endpoint",
        "Blockers: no validation",
        "Blockers: no authorization",
        "Blockers: no rate limiting on the login route",
        "Blocking finding: no input sanitization",
    ],
)
def test_blocker_label_security_phrasing_starting_with_no_blocks(body):
    assert has_blocking_finding_or_label(body) is True
    assert has_blocking_or_negative_verdict(body) is True


@pytest.mark.parametrize(
    "body",
    [
        "Blockers: none",
        "Blockers: no issues",
        "Blockers: no issue",
        "Blockers: no findings",
        "Blockers: no blockers",
        "Blocking findings: no blocking findings",
        "Blockers: no changes needed",
    ],
)
def test_blocker_label_genuine_no_finding_stays_non_blocking(body):
    assert has_blocking_finding_or_label(body) is False
    assert has_blocking_or_negative_verdict(body) is False


def test_p1_none_head_unchanged_no_finding():
    # The [P0]/[P1] no-finding head behavior is preserved by the fix.
    assert has_blocking_finding_or_label("[P1] None: clean") is False
    assert has_blocking_finding_or_label("[P1] N/A") is False
    assert has_blocking_finding_or_label("[P1] no issues") is False


# --- Regression: the no-finding "no <noun>" match must NOT leak into the
# negative-verdict (Verdict/Decision/Recommendation) check. A *positive* verdict
# phrased "no concerns" / "no changes needed" / "no blockers" is an APPROVAL and
# must stay non-blocking on the default flag-OFF merge-gate path. (Caught by both
# claude and openai on #8574; the shared helper had short-circuited unconditionally.)
@pytest.mark.parametrize(
    "body",
    [
        "Verdict: no concerns",
        "Verdict: no blocking issues found",
        "Decision: no blockers",
        "Recommendation: no changes needed",
        "Verdict: no issues",
    ],
)
def test_positive_verdict_with_no_noun_is_not_blocking(body):
    assert has_blocking_or_negative_verdict(body) is False


def test_approving_model_review_with_no_concerns_not_blocking():
    body = (
        "## Independent model review\n"
        "Reviewer: openai\n\n"
        "Verdict: no concerns\n\n"
        "- The change is well-scoped and tested.\n"
    )
    assert has_blocking_or_negative_verdict(body) is False


# --- Blocker-label no-finding regex precision (claude #8574 collect-3 P2s) ---
@pytest.mark.parametrize(
    "value",
    [
        "no significant issues",
        "no remaining blockers",
        "no other findings",
        "no minor issues",
        "no blocking findings",  # legacy form, via optional "blocking " prefix
    ],
)
def test_blocker_label_hedged_no_finding_stays_non_blocking(value):
    # Adjective-hedged no-finding declarations must NOT over-block (#1).
    assert has_blocking_or_negative_verdict(f"Blockers: {value}") is False


@pytest.mark.parametrize(
    "value",
    [
        "no blocking on the auth path but SQLi on line 40",  # the fail-open edge (#2)
        "no authentication on admin endpoint",
        "no validation of user input",
        "no rate limiting and an open redirect",
    ],
)
def test_blocker_label_real_finding_with_no_prefix_still_blocks(value):
    # Standalone "blocking" token dropped: a real finding phrased "no blocking … but X"
    # (or "no authentication …") must still block — no merge-gate bypass.
    assert has_blocking_or_negative_verdict(f"Blockers: {value}") is True


@pytest.mark.parametrize(
    "value",
    [
        "no concerns",
        "no major concerns",
        "no problems",
        "no serious problems",
    ],
)
def test_blocker_label_vague_no_finding_fails_closed_without_explicit_disposition(value):
    assert has_blocking_or_negative_verdict(f"Blockers: {value}") is False
    body = f"Verdict: CHANGES-REQUESTED\nBlockers: {value}"
    assert has_blocking_model_dissent(body, severity_gated=True) is True


@pytest.mark.parametrize("value", ["no concerns", "no major concerns", "no problems"])
def test_vague_no_finding_label_can_pair_with_followup_only_disposition(value):
    body = f"Verdict: CHANGES-REQUESTED\nBlockers: {value}\nFollow-up-only: yes"
    assert has_blocking_model_dissent(body, severity_gated=True) is False


# --- Reconciliation with #8555 (main makes [P2] block the DEFAULT gate) vs the
# severity-gate flag (which makes [P2] advisory). Pins the divergence: a [P2] marker
# blocks has_blocking_or_negative_verdict (flag-OFF default) but is NOT a blocking
# severity for the flag-ON helpers, so it becomes advisory under the flag.
def test_p2_blocks_default_path_but_is_advisory_under_severity_gate():
    body = "[P2] prepared apply bypasses the freeze proof"
    # flag-OFF default scanner: [P2] blocks (matches main #8555)
    assert has_blocking_or_negative_verdict(body) is True
    # flag-ON severity helpers: [P2] is NOT a blocking finding/label -> advisory
    assert has_blocking_finding_or_label(body) is False
    assert highest_blocking_severity(body) is None
    assert highest_finding_priority(body) == "P2"
    assert has_blocking_model_dissent(body, severity_gated=True) is False


def test_severity_gated_bare_changes_requested_still_blocks():
    body = "Verdict: CHANGES-REQUESTED\nNeeds another look before merge."
    assert has_blocking_model_dissent(body, severity_gated=True) is True


@pytest.mark.parametrize(
    "verdict_line",
    [
        "Verdict: CHANGES-REQUESTED (P2 only)",
        "Verdict: CHANGES-REQUESTED - P2 follow-ups only",
        "Decision: CHANGES-REQUESTED - P2/P3 follow-ups only",
        "Recommendation: CHANGES-REQUESTED - P3 advisory follow-ups only",
    ],
)
def test_severity_gated_qualified_low_severity_verdict_is_metadata(verdict_line):
    body = "\n".join(
        [
            verdict_line,
            "[P2] Add a regression test for the advisory path.",
            "No P0/P1 findings.",
            "LGTM otherwise.",
            "Focused adversarial dogfood passed.",
        ]
    )
    assert has_blocking_model_dissent(body, severity_gated=True) is False


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: CHANGES-REQUESTED (auth broken)\n[P2] Add docs.",
        "Verdict: CHANGES-REQUESTED - P2 follow-ups only\n[P2] Add docs.\n"
        "LGTM otherwise, but auth remains broken.",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\nNo P0/P1 findings, but auth remains broken.",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Focused adversarial dogfood passed, but null leak remains.",
    ],
)
def test_severity_gated_qualified_verdict_or_benign_prose_with_real_tail_blocks(body):
    assert has_blocking_model_dissent(body, severity_gated=True) is True


def test_severity_gated_explicit_no_blockers_can_be_advisory():
    body = "Verdict: CHANGES-REQUESTED\nBlockers: none\nFollow-up-only: yes"
    assert has_blocking_model_dissent(body, severity_gated=True) is False


def test_severity_gated_explicit_multiline_no_blockers_can_be_advisory():
    body = "Verdict: CHANGES-REQUESTED\nBlockers:\n- none\nFollow-up-only: yes"
    assert has_blocking_model_dissent(body, severity_gated=True) is False


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: CHANGES-REQUESTED\nThis is not follow-up-only; auth remains broken.",
        "Verdict: CHANGES-REQUESTED\nDo not treat the SQLi as non-blocking follow-ups.",
        "Verdict: CHANGES-REQUESTED\nNot non-blocking follow-ups: release is unsafe.",
    ],
)
def test_severity_gated_negated_followup_only_still_blocks(body):
    assert has_blocking_model_dissent(body, severity_gated=True) is True


def test_severity_gated_anchored_followup_only_can_be_advisory():
    body = "Verdict: CHANGES-REQUESTED\nFollow-up-only: yes"
    assert has_blocking_model_dissent(body, severity_gated=True) is False


@pytest.mark.parametrize(
    "followup_value",
    ["none", "n/a", "not applicable"],
)
def test_severity_gated_negating_followup_only_value_still_blocks(followup_value):
    body = f"Verdict: CHANGES-REQUESTED\nFollow-up-only: {followup_value}"
    assert has_blocking_model_dissent(body, severity_gated=True) is True


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\nAuth bypass remains untriaged.",
        "Verdict: CHANGES-REQUESTED because auth remains broken.\n[P2] Add docs.",
        "Verdict: CHANGES-REQUESTED\n[P3] Add docs.\nRecommendation: do not merge until auth is fixed.",
    ],
)
def test_severity_gated_p2_p3_plus_unstructured_prose_still_blocks(body):
    assert has_blocking_model_dissent(body, severity_gated=True) is True


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: CHANGES-REQUESTED\nBlockers: none\nAuth bypass remains untriaged.",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\nBlockers: none\nAuth bypass remains.",
        "Verdict: CHANGES-REQUESTED\nBlockers: none\nDogfood: failed on the admin path.",
        "Verdict: CHANGES-REQUESTED\nFollow-up-only: SQLi on admin path remains exploitable.",
    ],
)
def test_severity_gated_declarations_with_substantive_dissent_still_block(body):
    assert has_blocking_model_dissent(body, severity_gated=True) is True


def test_severity_gated_standard_metadata_with_low_severity_finding_is_advisory():
    body = "\n".join(
        [
            "## Claude independent model review",
            "Reviewer: claude (anthropic) - independent adversarial model review "
            "via the Aragora Claude reviewer, grounded on the exact PR head.",
            "Head: a6ffdd9 (a6ffdd9aa4cb6a2838c8647a2f436d5b9135e5a6), "
            "committed 2026-06-24T13:56:06Z.",
            "PR: #8590.",
            "Model family: claude",
            "Verdict: CHANGES-REQUESTED",
            "- [P2] Add a follow-up regression test.",
            "dogfood: yes",
        ]
    )

    assert has_blocking_model_dissent(body, severity_gated=True) is False


@pytest.mark.parametrize(
    "head_line",
    [
        "Head: a6ffdd9",
        "Head SHA: a6ffdd9aa4cb6a2838c8647a2f436d5b9135e5a6",
        "Current head: a6ffdd9aa4cb6a2838c8647a2f436d5b9135e5a6",
        "Head: a6ffdd9 (a6ffdd9aa4cb6a2838c8647a2f436d5b9135e5a6)",
        "Head: a6ffdd9 (a6ffdd9aa4cb6a2838c8647a2f436d5b9135e5a6), committed 2026-06-24T13:56:06Z.",
    ],
)
def test_severity_gated_safe_head_metadata_with_low_severity_finding_is_advisory(head_line):
    body = "\n".join(
        [
            "## Claude independent model review",
            "Reviewer: claude (anthropic) - independent adversarial model review",
            head_line,
            "PR: #8590.",
            "Model family: claude",
            "Verdict: CHANGES-REQUESTED",
            "- [P2] Add a follow-up regression test.",
            "dogfood: yes",
        ]
    )

    assert has_blocking_model_dissent(body, severity_gated=True) is False


@pytest.mark.parametrize("family", ["yi", "glm", "minimax", "hermes", "sonar", "perplexity"])
def test_severity_gated_known_family_metadata_with_low_severity_finding_is_advisory(family):
    body = "\n".join(
        [
            f"## {family.title()} independent model review",
            f"Reviewer: {family} (reviewer)",
            "Head: a6ffdd9 (a6ffdd9aa4cb6a2838c8647a2f436d5b9135e5a6)",
            "PR: #8590.",
            f"Model family: {family}",
            "Verdict: CHANGES-REQUESTED",
            "- [P2] Add a follow-up regression test.",
            "dogfood: yes",
        ]
    )

    assert has_blocking_model_dissent(body, severity_gated=True) is False


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: CHANGES-REQUESTED\nReviewer: auth bypass on admin endpoint\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P3] nit\nReviewer: this is actually critical, do not merge.",
        "Verdict: CHANGES-REQUESTED\nReviewer: claude noted problems\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\nReviewer: openai found issues\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\nReviewer: grok has concerns\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Reviewer: claude security hole\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Reviewer: claude incorrect logic\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Model family: grok validate inputs carefully\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Reviewer: claude missing auth\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Model family: openai null leak\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Reviewer: grok risky endpoint\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\nHead: deadbeef auth bypass remains\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Head: a6ffdd9 (deadbeef) do not merge\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] Add docs.\n"
        "Current head: a6ffdd9aa4cb6a2838c8647a2f436d5b9135e5a6 do not merge\n"
        "Follow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\nBlockers: none\nPR: merging this will break production.",
        "Verdict: CHANGES-REQUESTED\n[P3] nit\nHead-of-line blocking is the real "
        "defect; do not ship.",
        "## Claude independent model review found auth bypass\n"
        "Verdict: CHANGES-REQUESTED\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED from grok\nFollow-up-only: yes",
        "Verdict: CHANGES-REQUESTED\n[P2] docs\nReviewer: claude rce vulnerability",
        "Verdict: CHANGES-REQUESTED\n[P2] docs\nModel family: openai rce vulnerability",
    ],
)
def test_severity_gated_metadata_hiding_dissent_still_blocks(body):
    assert has_blocking_model_dissent(body, severity_gated=True) is True


def test_p0_p1_block_both_paths():
    for sev in ("P0", "P1"):
        body = f"[{sev}] settlement gate can be bypassed"
        assert has_blocking_or_negative_verdict(body) is True
        assert has_blocking_finding_or_label(body) is True
        assert highest_blocking_severity(body) == sev


@pytest.mark.parametrize(
    "value",
    [
        "no major concerns but SQLi on line 40",
        "no issues except an open redirect",
        "no blockers however auth is missing",
        "no significant problems aside from the race condition",
    ],
)
def test_blocker_label_hedged_no_finding_with_real_tail_blocks(value):
    # Fail-closed: a no-finding prefix followed by a contrastive real finding blocks.
    assert has_blocking_or_negative_verdict(f"Blockers: {value}") is True


@pytest.mark.parametrize(
    "value",
    [
        "no issues found",
        "no concerns identified",
        "no blockers noted",
        "no major issues.",
        "no findings whatsoever",
    ],
)
def test_blocker_label_pure_no_finding_with_benign_tail_stays_non_blocking(value):
    assert has_blocking_or_negative_verdict(f"Blockers: {value}") is False


@pytest.mark.parametrize(
    "value",
    [
        "none but SQLi on line 40",
        "n/a - auth bypass remains",
        "zero, but the open redirect is unfixed",
        "not applicable however the race condition stands",
    ],
)
def test_blocker_label_legacy_prefix_with_real_tail_blocks(value):
    # openai #8574 P2: legacy _NON_BLOCKING_PREFIXES (none/n-a/zero) must also be
    # fail-closed — a substantive contrastive tail still blocks.
    assert has_blocking_or_negative_verdict(f"Blockers: {value}") is True


@pytest.mark.parametrize("value", ["none", "none found", "n/a", "not applicable", "zero", "[]"])
def test_blocker_label_legacy_prefix_pure_stays_non_blocking(value):
    assert has_blocking_or_negative_verdict(f"Blockers: {value}") is False
