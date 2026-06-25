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
    highest_blocking_severity,
    has_blocking_or_negative_verdict,
    has_unlabeled_soft_dissent_phrase,
    _is_markdown_indented_code_line as verdicts_is_markdown_indented_code_line,
)
from aragora.cli.commands.review_queue import (
    _is_markdown_indented_code_line as review_queue_is_markdown_indented_code_line,
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


@pytest.mark.parametrize(
    "body",
    [
        "Terrible security hole must not merge.",
        "Do not merge until the auth bypass is fixed.",
        "This is not safe to ship.",
    ],
)
def test_prose_only_merge_blocking_dissent_still_blocks(body):
    assert has_blocking_or_negative_verdict(body)


@pytest.mark.parametrize(
    "body",
    [
        "No sql injection found.",
        "Verdict: PASS\nNo sql injection found.",
        "No SQL injection or auth bypass found.",
        "Verdict: PASS\nNo authentication bypass found.",
        "Verdict: PASS\nNo security hole found.",
        "I reviewed sql injection tests.",
        "Reviewed auth bypass tests.",
        "Added sql injection regression tests.",
        "Discussed sql injection risks; none found.",
        "SQL injection coverage is adequate.",
        "Authentication bypass tests passed.",
        "No authentication bypass",
        "No security hole",
        "This closes the auth bypass; no findings.",
        "This PR fixes an auth bypass.",
        "fixes an auth bypass.",
        "fixes auth bypass.",
        "The SQL injection fix is correct.",
    ],
)
def test_benign_security_phrase_no_finding_prose_does_not_block(body):
    assert not has_blocking_or_negative_verdict(body)


@pytest.mark.parametrize(
    "body",
    [
        "Residual auth bypass risk remains",
        "This PR still has an auth bypass.",
        "Security hole remains in admin routing",
        "SQL injection coverage is inadequate",
    ],
)
def test_substantive_security_phrase_still_blocks(body):
    assert has_blocking_or_negative_verdict(body)


@pytest.mark.parametrize(
    "body",
    [
        "```python\nassert 'do not merge' in fixture_text\n```",
        "> do not merge until the fixture says so",
        "    do not merge until the fixture says so",
    ],
)
def test_prose_blocking_phrases_inside_examples_do_not_block(body):
    assert not has_blocking_or_negative_verdict(body)


@pytest.mark.parametrize(
    "body",
    [
        "```\nDo not merge until auth is fixed.\n```",
        "> Do not merge until auth is fixed.",
        "    Do not merge until auth is fixed.",
        "```\nVerdict: CHANGES-REQUESTED\n```",
        "```\n[P1] hidden blocker survives fenced formatting.\n```",
        "> example: do not merge until auth is fixed.",
        "> sample: auth bypass remains",
        "    example: do not merge until auth is fixed.",
        "    sample: auth bypass remains",
    ],
)
def test_untrusted_formatting_cannot_hide_secondary_dissent(body):
    assert has_blocking_or_negative_verdict(f"Verdict: PASS\n{body}")


@pytest.mark.parametrize(
    "line",
    [
        "Verdict: PASS",
        "  Verdict: PASS",
        "    Verdict: PASS",
        "\tVerdict: PASS",
        "    ",
    ],
)
def test_indented_code_helper_matches_review_queue(line):
    assert verdicts_is_markdown_indented_code_line(
        line
    ) == review_queue_is_markdown_indented_code_line(line)


def test_free_form_needs_revision_before_merge_blocks():
    assert has_blocking_or_negative_verdict("Needs revision before merge.")


def test_benign_security_phrase_does_not_mask_later_merge_blocker():
    body = "No sql injection found. Do not merge until the auth bypass is fixed."
    assert has_blocking_or_negative_verdict(body)


def test_same_line_pass_with_priority_finding_still_blocks():
    assert has_blocking_or_negative_verdict("Verdict: PASS; [P1] real blocker")


def test_post_verdict_ordinary_priority_marker_finding_blocks():
    body = "Verdict: PASS\nNote: still see [P1] missing rate limit"
    assert has_blocking_or_negative_verdict(body)


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: PASS; no [P1] findings",
        "Verdict: PASS; no [P2] findings.",
        "No [P1] findings.",
        "No remaining [P2] blockers found.",
    ],
)
def test_inline_priority_marker_no_finding_prose_stays_non_blocking(body):
    assert not has_blocking_or_negative_verdict(body)


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: PASS; no [P1] auth bypass",
        "No [P2] auth bypass remains.",
    ],
)
def test_inline_priority_marker_with_concrete_finding_still_blocks(body):
    assert has_blocking_or_negative_verdict(body)


def test_severity_gate_finding_or_label_catches_same_line_p1_finding():
    assert has_blocking_finding_or_label("Verdict: PASS; [P1] auth bypass")
    assert highest_blocking_severity("Verdict: PASS; [P1] auth bypass") == "P1"


def test_severity_gate_finding_or_label_keeps_same_line_p2_advisory():
    assert not has_blocking_finding_or_label("Verdict: PASS; [P2] follow-up docs")
    assert highest_blocking_severity("Verdict: PASS; [P2] follow-up docs") is None


def test_reasoning_tags_do_not_hide_same_line_priority_finding():
    body = "<thinking>Verdict: PASS; [P1] auth bypass</thinking>\nVerdict: PASS\nNo findings."
    assert has_blocking_or_negative_verdict(body)
    assert has_blocking_finding_or_label(body)


def test_reasoning_tags_do_not_fragment_blocking_phrase():
    assert has_blocking_or_negative_verdict("Verdict: PASS\nDo <thinking>not</thinking> merge.")


def test_newline_split_blocking_phrase_still_blocks():
    assert has_blocking_or_negative_verdict("Verdict: PASS\nDo not\nmerge.")


def test_three_line_split_blocking_phrase_still_blocks():
    assert has_blocking_or_negative_verdict("Verdict: PASS\nDo\nnot\nmerge.")


@pytest.mark.parametrize(
    "body",
    [
        "The parser should not merge dissent fragments incorrectly.",
        "For example, comments saying do not merge should be rejected.",
    ],
)
def test_meta_review_merge_phrase_does_not_block(body):
    assert not has_blocking_or_negative_verdict(body)


def test_real_merge_blocker_with_subject_still_blocks():
    assert has_blocking_or_negative_verdict("This PR should not merge until auth is fixed.")


@pytest.mark.parametrize(
    "body",
    [
        "<thinking>Supportive with fixes.</thinking>",
        "<analysis>\nPASS with conditions.\n</analysis>",
        "LGTM but the fallback can still count conditional evidence.",
        "Approved but the unclosed reasoning tail is still stripped.",
        "Looks good but this needs a parser repair first.",
    ],
)
def test_reasoning_tags_do_not_hide_unlabeled_soft_dissent_phrase(body):
    assert has_unlabeled_soft_dissent_phrase(body)


def test_reasoning_tags_do_not_fragment_unlabeled_soft_dissent_phrase():
    assert has_unlabeled_soft_dissent_phrase("Looks <thinking>good</thinking> but needs repair.")


def test_newline_split_unlabeled_soft_dissent_phrase_blocks():
    assert has_unlabeled_soft_dissent_phrase("Pass\nwith notes.")


def test_three_line_split_unlabeled_soft_dissent_phrase_blocks():
    assert has_unlabeled_soft_dissent_phrase("Pass\nwith\nnotes.")


@pytest.mark.parametrize(
    "body",
    [
        "LGTM but verify CI.",
        "Looks good but confirm checks pass.",
        "Approved but ensure required checks are green.",
    ],
)
def test_operational_check_caveats_are_not_soft_dissent(body):
    assert not has_unlabeled_soft_dissent_phrase(body)


@pytest.mark.parametrize(
    "body",
    [
        "LGTM but needs repair.",
        "Approved but with fixes.",
        "Looks good but [P1] missing rate limit.",
    ],
)
def test_substantive_caveats_remain_soft_dissent(body):
    assert has_unlabeled_soft_dissent_phrase(body)


def test_unlabeled_soft_dissent_phrase_accepts_plain_no_finding_prose():
    assert not has_unlabeled_soft_dissent_phrase("Verdict: PASS\nNo findings.")
    assert not has_unlabeled_soft_dissent_phrase("No SQL injection or auth bypass found.")


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: PASS; `[P1] auth bypass`",
        'Verdict: PASS; "[P1] auth bypass"',
        "Verdict: PASS; '[P2] settlement proof missing'",
    ],
)
def test_same_line_pass_with_wrapped_priority_finding_still_blocks(body):
    assert has_blocking_or_negative_verdict(body)


def test_parenthetical_priority_backlog_reference_is_not_blocking():
    assert not has_blocking_or_negative_verdict("Verdict: PASS (tracked as [P2] in backlog)")


@pytest.mark.parametrize(
    "body",
    [
        "Verdict: PASS ([P1] issue: auth bypass)",
        "Verdict: PASS (see issue [P1] auth bypass)",
        "Verdict: PASS (ticket [P1] SQLi)",
        "Verdict: PASS (tracked as [P2] in backlog: auth bypass)",
    ],
)
def test_parenthetical_priority_with_concrete_finding_blocks(body):
    assert has_blocking_or_negative_verdict(body)


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
        "Blockers: no concerns",
        "Blockers: no problems",
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
        "no major concerns",
        "no significant issues",
        "no serious problems",
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
