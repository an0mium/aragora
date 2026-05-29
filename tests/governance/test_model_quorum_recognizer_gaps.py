"""Governance tests for direct model-family recognition.

These tests were the current-state characterization target for the Tier 4
model-quorum-family expansion patch designed in
`docs/specs/MODEL_QUORUM_FAMILY_EXPANSION.md`. After the lineage-bound
quorum implementation they pin the new positive behavior:

1. They *pin* the current state of the recognizer so that the Tier 4 patch
   has an explicit, machine-checkable regression floor: the patch must
   keep the existing claude/codex/gemini/grok/tesla/harvey/factory markers
   working while *adding* recognition for the gap families.

2. They assert that canonical model-family headings such as OpenAI,
   Mistral/Codestral, DeepSeek, Qwen, Kimi/Moonshot, GLM, MiniMax, Yi,
   and Hermes now resolve to canonical family IDs for lineage-bound
   quorum evidence.

Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change governance`:

  > "A change that adds a new family marker ... is a Tier 4
  >  merge-authority self-modification. ... The pre-approval artifact ...
  >  is a design document in docs/specs/ ... and governance tests
  >  in tests/governance/ that characterize the current gate behavior
  >  and pin the gap to be inverted by the implementation."

This file now records that the pre-approved implementation intentionally
inverted the former gap-demonstration assertions.
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


# ----- Markers expected to resolve after lineage-bound implementation -----
#
# Each of these families is canonical for lineage-bound quorum counting.


_GAP_MARKERS_HEAD_SHA = "abc1234"


@pytest.mark.parametrize(
    "comment_body, expected_family",
    [
        ("OpenAI independent model review on head abc1234", "openai"),
        ("Anthropic independent semantic review on head abc1234", "claude"),
        ("Mistral independent model review on head abc1234", "mistral"),
        ("Codestral independent semantic review on head abc1234", "mistral"),
        ("DeepSeek independent semantic review on head abc1234", "deepseek"),
        ("Qwen independent semantic review on head abc1234", "qwen"),
        ("Kimi independent semantic review on head abc1234", "kimi"),
        ("Moonshot independent semantic review on head abc1234", "kimi"),
        ("GLM independent semantic review on head abc1234", "glm"),
        ("Zhipu independent semantic review on head abc1234", "glm"),
        ("Z-AI independent semantic review on head abc1234", "glm"),
        ("MiniMax independent semantic review on head abc1234", "minimax"),
        ("Yi-Large independent semantic review on head abc1234", "yi"),
        ("Nous Hermes independent semantic review on head abc1234", "hermes"),
        ("Hermes independent semantic review on head abc1234", "hermes"),
    ],
)
def test_canonical_model_family_markers_resolve(
    comment_body: str,
    expected_family: str,
) -> None:
    """Canonical families resolve to the lineage ID used by quorum counting."""
    assert _infer_model_reviewer_from_text(comment_body) == expected_family


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
