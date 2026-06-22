"""Tests for the [P0]/[P1] non-finding false-positive in the verdict scanner.

`has_blocking_or_negative_verdict` is the blocking-language scan used at BOTH
collect-time and gate-time to decide whether a reviewer's evidence counts. It
treated ANY line starting with `[P0]`/`[P1]` as a blocking finding — including
`"[P1] None:"` / `"[P1] N/A"`, which low-cost models (and claude/codex) emit to
say "no P0/P1 issues". That silently de-counted a PASS vote.

The fix must be conservative: a real finding that merely *starts with* "none"/"no"
(e.g. `[P1] None of the inputs are validated`, `[P1] no auth check`) MUST still
block. Only an explicit no-finding *head* (the text before any colon being exactly
"none", "n/a", "no issues", ...) is treated as non-blocking.
"""

from __future__ import annotations

from aragora.cli.commands.review_queue_comment_verdicts import (
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


def test_negative_verdict_line_still_blocks():
    assert has_blocking_or_negative_verdict("Verdict: CHANGES-REQUESTED")
