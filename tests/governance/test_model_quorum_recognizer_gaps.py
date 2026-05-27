"""Governance tests pinning the current state of `_infer_model_reviewer_from_text`.

These tests are the current-state characterization target for the Tier 4
model-quorum-family expansion patch designed in
`docs/specs/MODEL_QUORUM_FAMILY_EXPANSION.md`. They exist for two reasons:

1. They *pin* the current state of the recognizer so that the Tier 4 patch
   has an explicit, machine-checkable regression floor: the patch must
   keep the existing claude/codex/gemini/grok/tesla/harvey/factory markers
   working while *adding* recognition for the gap families.

2. They *demonstrate* the gap: already-routed families such as OpenAI,
   Mistral/Codestral, DeepSeek, Qwen, Kimi/Moonshot and new proposed
   families such as GLM, MiniMax, Yi, and Hermes currently return
   `unknown_model_reviewer`, which means a reviewer signal posted by that
   family — even if grounded on the current head SHA, even if posted by a
   non-author account — would not be counted toward the quorum.

After the Tier 4 patch lands, the gap-demonstration tests will need to be
inverted (assert that the family IS recognized). At that point this file
also becomes the regression floor for the new state.

Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`:

  > "A change that adds a new family marker ... is a Tier 4
  >  merge-authority self-modification. ... The pre-approval artifact ...
  >  is a design document in docs/specs/ ... and governance tests
  >  in tests/governance/ that characterize the current gate behavior
  >  and pin the gap to be inverted by the implementation."

This file IS that pre-approval test surface. The implementation that
inverts the gap-demonstration assertions waits for operator preapproval.
"""

from __future__ import annotations

import pytest

from aragora.cli.commands.review_queue import _infer_model_reviewer_from_text

# ----- Markers expected to RESOLVE today (regression floor) -----
#
# These currently work; the Tier 4 patch must keep them working. Each
# marker string follows the structured "<provider> independent semantic
# review" / "<provider> review" patterns the recognizer already accepts.


_EXISTING_MARKERS_HEAD_SHA = "abc1234"


@pytest.mark.parametrize(
    "comment_body, expected_family",
    [
        ("Claude independent semantic review on head abc1234", "claude"),
        ("Codex review on head abc1234", "codex"),
        ("Gemini independent semantic review on head abc1234", "gemini"),
        ("Grok independent review on head abc1234", "grok"),
        # tesla/harvey/factory are vendor markers in the existing recognizer; pinned for completeness
        ("Tesla independent semantic review on head abc1234", "tesla"),
        ("Harvey independent semantic review on head abc1234", "harvey"),
        ("Factory independent semantic review on head abc1234", "factory"),
    ],
)
def test_existing_recognizers_still_resolve(comment_body: str, expected_family: str) -> None:
    """REGRESSION FLOOR: families recognized today must stay recognized.

    The Tier 4 patch must NOT break any of these. If any of these fail
    after the patch lands, the patch has regressed and must be reverted
    or fixed before merge.

    The list reflects the actual current state of
    `_infer_model_reviewer_from_text` (it scans the first markdown heading
    or first 200 chars and matches against a narrow tuple of 7 markers).
    OpenAI/Mistral/DeepSeek/Qwen/Kimi are NOT in this list — those are in
    the GAP test below.
    """
    assert _infer_model_reviewer_from_text(comment_body) == expected_family


# ----- Markers expected to NOT resolve today (gap demonstration) -----
#
# Each of these families is either (a) already routable via OpenRouter
# but not recognized as a reviewer source, or (b) a new family to be
# added per the design spec. After the Tier 4 patch these assertions
# should INVERT — but until then, this file documents the gap.


_GAP_MARKERS_HEAD_SHA = "abc1234"


@pytest.mark.parametrize(
    "comment_body, family_name",
    [
        # Already-routed-by-aragora families that the recognizer DOES NOT count today.
        # This is the surprise gap: aragora pays for OpenRouter access to these models
        # via api_agents/openrouter.py, but their PR-comment signals are silently dropped.
        ("OpenAI independent model review on head abc1234", "openai"),
        ("Anthropic independent semantic review on head abc1234", "anthropic"),
        ("Mistral independent model review on head abc1234", "mistral"),
        ("Codestral independent semantic review on head abc1234", "mistral (via codestral marker)"),
        ("DeepSeek independent semantic review on head abc1234", "deepseek"),
        ("Qwen independent semantic review on head abc1234", "qwen"),
        ("Kimi independent semantic review on head abc1234", "kimi"),
        ("Moonshot independent semantic review on head abc1234", "kimi (via moonshot marker)"),
        # New families to add per the design spec + operator answer #2.
        ("GLM independent semantic review on head abc1234", "glm"),
        ("Zhipu independent semantic review on head abc1234", "glm (via zhipu marker)"),
        ("Z-AI independent semantic review on head abc1234", "glm (via z-ai marker)"),
        ("MiniMax independent semantic review on head abc1234", "minimax"),
        ("Yi-Large independent semantic review on head abc1234", "yi"),
        ("Nous Hermes independent semantic review on head abc1234", "hermes"),
        ("Hermes independent semantic review on head abc1234", "hermes"),
    ],
)
def test_proposed_family_markers_currently_unrecognized(
    comment_body: str,
    family_name: str,
) -> None:
    """GAP DEMONSTRATION: families that should resolve after the patch.

    Each input here would post a valid reviewer comment that the
    recognizer SHOULD count under the design in
    `docs/specs/MODEL_QUORUM_FAMILY_EXPANSION.md`, but today it returns
    `unknown_model_reviewer` because the marker is missing from
    `_infer_model_reviewer_from_text`'s current seven-marker tuple.

    After the Tier 4 patch lands this test will be inverted (or moved
    to a positive-recognition test parameterized by the new families).
    The current assertion is the gap-of-record.
    """
    assert _infer_model_reviewer_from_text(comment_body) == "unknown_model_reviewer", (
        f"Family {family_name!r} now appears to be recognized. If the Tier 4 patch "
        "has landed, invert this assertion (or move to the positive-recognition "
        f"suite) — the comment {comment_body!r} should now resolve to {family_name!r}."
    )


def test_unknown_garbage_stays_unknown() -> None:
    """SAFETY FLOOR: arbitrary prose without a known marker stays unknown.

    Guards against the failure mode of an over-eager recognizer matching
    arbitrary substrings (the same class of bug as PR #7438's parser
    fallback). The Tier 4 patch must preserve this.
    """
    inputs = [
        "Looking at this PR, my analysis says it should land.",
        "Some unidentified reviewer agent looked at this on head abc1234",
        "The change touches a security path so I'd defer.",
        "",
    ]
    for body in inputs:
        assert _infer_model_reviewer_from_text(body) == "unknown_model_reviewer", (
            f"prose {body!r} should not produce a counted reviewer family"
        )


def test_recognizer_is_case_insensitive() -> None:
    """The recognizer already lowercases input; pin that behavior."""
    assert _infer_model_reviewer_from_text("CLAUDE REVIEW on head abc1234") == "claude"
    assert _infer_model_reviewer_from_text("gemini Review on head abc1234") == "gemini"
