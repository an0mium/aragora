"""Governance tests for the model-lineage-disclosure recognizer extension.

These tests are the Tier 4 pre-approval regression target for the
change designed in
`docs/specs/MODEL_LINEAGE_DISCLOSURE.md`. They exist for two reasons:

1. They *pin* the current recognizer state — bare-`str` return shape,
   collapsed harness-vs-lineage behavior, heading-only scanning —
   so the implementation PR has a machine-checkable regression
   floor. If a future change quietly extends the recognizer
   without going through the pre-approval discipline, these tests
   fail loudly.

2. They *codify* the proposed contract — the `LINEAGE_REGEX`,
   the model-ID prefix normalization table, the body-scan
   precedence, the backwards-compatibility floor — as a passing
   executable spec. The implementation PR's job is to make the
   recognizer satisfy these same predicates, not to invent new
   ones.

Per `docs/REVIEW_AUTHORITY_PRINCIPLES.md::Family-additive change
governance`:

  > "A change that adds a new family marker to the recognizer in
  >  aragora/cli/commands/review_queue.py::_infer_model_reviewer_from_text,
  >  or changes which family counts at which Tier, is a Tier 4
  >  merge-authority self-modification per the Tier table above.
  >  It requires human preapproval before implementation and
  >  before merge."

This file IS that pre-approval test surface. The implementation
that satisfies the proposed contract waits for explicit operator
Tier 4 preapproval at the implementation step.
"""

from __future__ import annotations

import re

import pytest

from aragora.cli.commands.review_queue import _infer_model_reviewer_from_text

# ---------------------------------------------------------------------------
# Current-state regression floor: the recognizer returns a bare str
# and collapses harness identity over model lineage. The implementation
# PR will change both behaviors; until then, these tests pin the floor.
# ---------------------------------------------------------------------------


def test_recognizer_currently_returns_bare_str() -> None:
    """REGRESSION FLOOR: today the recognizer returns a plain str.

    The implementation PR will change this to a NamedTuple/dataclass
    with `family_marker` and `model_lineage` fields. Until then,
    callers consume a plain str. If this test fails, the recognizer
    has been changed without pre-approval discipline.
    """
    result = _infer_model_reviewer_from_text("## Factory focused dogfood\n")
    assert isinstance(result, str), (
        "Recognizer return type changed from str to a structured "
        "type without the lineage-disclosure pre-approval. If the "
        "implementation PR has landed, invert this test (or move it "
        "to a positive-present suite) — it is the regression floor "
        "documenting that the change went through Tier 4 discipline."
    )


@pytest.mark.parametrize(
    "body, expected_family",
    [
        # All five of these have IDENTICAL recognizer output today
        # despite naming wildly different underlying models in the
        # body. That uniform collapse IS the gap this design closes.
        (
            "## Factory focused dogfood\n\n**Reviewer:** Factory Droid (gpt-5.5)\n",
            "factory",
        ),
        (
            "## Factory focused dogfood\n\n**Reviewer:** Factory Droid (claude-opus-4-7)\n",
            "factory",
        ),
        (
            "## Factory focused dogfood\n\n**Reviewer:** Factory Droid (gemini-3.5-flash)\n",
            "factory",
        ),
        (
            "## Factory focused dogfood\n\n**Reviewer:** Factory Droid (deepseek-v3)\n",
            "factory",
        ),
        (
            "## Factory focused dogfood\n\n(no Reviewer field; pre-lineage-disclosure era)\n",
            "factory",
        ),
    ],
)
def test_recognizer_currently_collapses_harness_over_lineage(
    body: str, expected_family: str
) -> None:
    """REGRESSION FLOOR: today's recognizer cannot see model lineage.

    All five inputs above return `"factory"` today regardless of which
    underlying model the comment body declares. The implementation PR
    will distinguish them via the `model_lineage` field. Until then,
    the uniform collapse is pinned as a regression floor — proves the
    gap is real and forces the implementation to break this test
    intentionally.
    """
    assert _infer_model_reviewer_from_text(body) == expected_family, (
        "Recognizer's harness-level collapse changed without the "
        "lineage-disclosure pre-approval. If the implementation PR "
        "has landed, invert these assertions (or move them to a "
        "positive-present suite asserting the new disambiguated "
        "behavior)."
    )


# ---------------------------------------------------------------------------
# Proposed contract: LINEAGE_REGEX parses well-formed Reviewer fields.
# The implementation's body parser MUST behave equivalently. Tests
# below pin both sides as a passing executable spec.
# ---------------------------------------------------------------------------


# Reference implementation of the lineage-disclosure regex. The
# implementation PR MUST use this exact pattern (or a strict
# refinement, in which case the refinement itself is a Tier 4 pre-
# approval).
LINEAGE_REGEX = re.compile(
    r"^\s*\*{0,2}Reviewer(?:\s+lineage)?:\*{0,2}\s*"
    r"(?P<harness>[A-Za-z][A-Za-z0-9 .\-_/]*?)"
    r"\s*\((?P<model_id>[a-z][a-z0-9 .\-_/]*)\)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _strip_fenced_code_blocks(markdown: str) -> str:
    """Reference contract: lineage parsing ignores fenced code blocks."""
    prose_lines: list[str] = []
    in_fence = False
    fence_marker: str | None = None

    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
            continue
        if not in_fence:
            prose_lines.append(line)

    return "\n".join(prose_lines)


def _search_lineage_body_prose(markdown: str) -> re.Match[str] | None:
    """Reference parser shape: apply LINEAGE_REGEX only to body prose."""
    return LINEAGE_REGEX.search(_strip_fenced_code_blocks(markdown))


@pytest.mark.parametrize(
    "body, expected_harness, expected_model_id",
    [
        # Bold-prefixed (canonical operator-attestation form).
        (
            "**Reviewer:** Factory Droid (gpt-5.5)",
            "Factory Droid",
            "gpt-5.5",
        ),
        (
            "**Reviewer:** Factory Droid (claude-opus-4-7)",
            "Factory Droid",
            "claude-opus-4-7",
        ),
        (
            "**Reviewer:** Claude Code (claude-sonnet-4-5)",
            "Claude Code",
            "claude-sonnet-4-5",
        ),
        (
            "**Reviewer:** Codex CLI (gpt-5-codex)",
            "Codex CLI",
            "gpt-5-codex",
        ),
        (
            "**Reviewer:** Aragora harness (grok-4-3-mini)",
            "Aragora harness",
            "grok-4-3-mini",
        ),
        (
            "**Reviewer:** Factory Droid (gemini-3.5-flash)",
            "Factory Droid",
            "gemini-3.5-flash",
        ),
        # Reviewer lineage variant.
        (
            "**Reviewer lineage:** Factory Droid (deepseek-v3)",
            "Factory Droid",
            "deepseek-v3",
        ),
        # Non-bold variant (still valid).
        (
            "Reviewer: Factory Droid (qwen-2.5-72b)",
            "Factory Droid",
            "qwen-2.5-72b",
        ),
        # Single-asterisk italic variant.
        (
            "*Reviewer:* Codex CLI (gpt-5-codex)",
            "Codex CLI",
            "gpt-5-codex",
        ),
        # Mistral with dotted version.
        (
            "**Reviewer:** Aragora harness (mistral-large-2.1)",
            "Aragora harness",
            "mistral-large-2.1",
        ),
    ],
)
def test_lineage_regex_accepts_well_formed_disclosures(
    body: str, expected_harness: str, expected_model_id: str
) -> None:
    """LINEAGE_REGEX positive cases — well-formed disclosures match."""
    match = LINEAGE_REGEX.search(body)
    assert match is not None, (
        f"body {body!r} should match LINEAGE_REGEX but did not. The "
        "regex contract has drifted to under-accept."
    )
    assert match.group("harness").strip() == expected_harness, (
        f"body {body!r} captured wrong harness "
        f"{match.group('harness')!r} (expected {expected_harness!r})"
    )
    assert match.group("model_id").strip().lower() == expected_model_id.lower(), (
        f"body {body!r} captured wrong model_id "
        f"{match.group('model_id')!r} (expected {expected_model_id!r})"
    )


@pytest.mark.parametrize(
    "body",
    [
        # Empty / arbitrary.
        "",
        "Some random review body without a Reviewer field.",
        # Missing the keyword.
        "**Model:** Factory Droid (gpt-5.5)",
        "**Author:** Factory Droid (gpt-5.5)",
        # Missing parens.
        "**Reviewer:** Factory Droid gpt-5.5",
        # Missing model_id inside parens.
        "**Reviewer:** Factory Droid ()",
        # Missing harness.
        "**Reviewer:** (gpt-5.5)",
        # Mis-spelled keyword.
        "**Reviwer:** Factory Droid (gpt-5.5)",
        "**Reveiwer:** Factory Droid (gpt-5.5)",
        # Note: uppercase model_id like "GPT-5.5" is technically
        # accepted by the current regex (IGNORECASE flag makes
        # `[a-z]` match `[a-zA-Z]`). Convention is lowercase
        # (matching the implementation PR's lineage prefix table,
        # which must stay aligned with AgentSpec/ALLOWED_AGENT_TYPES)
        # but the regex itself doesn't enforce case. Operators
        # treating uppercase model_id as a different lineage than
        # lowercase would need to add a separate normalization
        # check — out of scope of this design.
    ],
)
def test_lineage_regex_rejects_ill_formed_disclosures(body: str) -> None:
    """LINEAGE_REGEX negative cases — ill-formed bodies do NOT match."""
    assert LINEAGE_REGEX.search(body) is None, (
        f"body {body!r} should NOT match LINEAGE_REGEX but did. The "
        "regex contract has drifted to over-accept."
    )


# ---------------------------------------------------------------------------
# Proposed contract: model-ID prefix → normalized lineage family.
# ---------------------------------------------------------------------------


MODEL_LINEAGE_PREFIX_TABLE = {
    # OpenAI lineage.
    "gpt-": "openai",
    "openai-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    # Anthropic lineage.
    "claude-": "anthropic",
    "anthropic-": "anthropic",
    # Google lineage.
    "gemini-": "google",
    "palm-": "google",
    # xAI lineage.
    "grok-": "xai",
    "xai-": "xai",
    # Mistral lineage.
    "mistral-": "mistral",
    "codestral-": "mistral",
    # DeepSeek lineage.
    "deepseek-": "deepseek",
    # Qwen lineage.
    "qwen-": "qwen",
    # Kimi / Moonshot lineage.
    "kimi-": "kimi",
    "moonshot-": "kimi",
    # Meta / Llama lineage.
    "llama-": "meta",
}


def _normalize_model_lineage(model_id: str) -> str:
    """Reference implementation of the model-lineage normalization.

    The implementation MUST use this exact prefix table (or a strict
    superset, in which case the additions are themselves a Tier 4
    pre-approval).
    """
    lower = model_id.lower().strip()
    for prefix, lineage in MODEL_LINEAGE_PREFIX_TABLE.items():
        if lower.startswith(prefix):
            return lineage
    return "unknown_model_lineage"


@pytest.mark.parametrize(
    "model_id, expected_lineage",
    [
        # OpenAI variants.
        ("gpt-5.5", "openai"),
        ("gpt-5-codex", "openai"),
        ("gpt-4o", "openai"),
        ("o1-preview", "openai"),
        ("o3-mini", "openai"),
        # Anthropic variants.
        ("claude-opus-4-7", "anthropic"),
        ("claude-sonnet-4-5", "anthropic"),
        ("claude-haiku-4", "anthropic"),
        # Google variants.
        ("gemini-3.5-flash", "google"),
        ("gemini-2.0-pro", "google"),
        # xAI variants.
        ("grok-4-3", "xai"),
        ("grok-4-3-mini", "xai"),
        # Mistral variants.
        ("mistral-large-2.1", "mistral"),
        ("codestral-2501", "mistral"),
        # DeepSeek / Qwen / Kimi / Llama.
        ("deepseek-v3", "deepseek"),
        ("qwen-2.5-72b", "qwen"),
        ("kimi-k2", "kimi"),
        ("moonshot-v1", "kimi"),
        ("llama-3.1-70b", "meta"),
        # Unknown prefix.
        ("unknown-model-x", "unknown_model_lineage"),
        ("custom-finetune-v1", "unknown_model_lineage"),
    ],
)
def test_model_lineage_prefix_normalization(model_id: str, expected_lineage: str) -> None:
    """Model-ID prefix table positive cases — each prefix normalizes."""
    assert _normalize_model_lineage(model_id) == expected_lineage, (
        f"model_id {model_id!r} normalized to wrong lineage. The "
        "prefix table has drifted from the spec."
    )


# ---------------------------------------------------------------------------
# Backwards-compatibility floor: comments without the Reviewer field
# MUST still resolve to a recognizable family marker today. The
# implementation PR's Variant A (recommended) preserves this; Variant
# B (future) un-counts them for Tier 2+.
# ---------------------------------------------------------------------------


def test_pre_lineage_era_comments_still_count_under_variant_a() -> None:
    """Backwards compat: pre-lineage-era comments still produce a family.

    The implementation PR will return
    `RecognizedReviewer(family_marker="factory", model_lineage=None)`
    for comments lacking the Reviewer field. Under Variant A
    (recommended for first implementation), the merge-quorum evaluator
    still counts the signal; the `lineage_undeclared: true` flag is
    recorded for observability.

    Today (pre-implementation), the recognizer returns just "factory"
    for such comments. This test pins the family-marker continuity
    floor: the implementation PR cannot quietly drop the family marker
    even when the body has no Reviewer field.
    """
    pre_lineage_bodies = [
        "## Factory focused dogfood\n\nLooks good.\n",
        "## Codex review\n\nVerdict: approve.\n",
        "## Claude independent semantic review on head abc1234\n\nNo findings.\n",
        "## Grok independent model review\n\nReceipt verified.\n",
    ]
    for body in pre_lineage_bodies:
        family = _infer_model_reviewer_from_text(body)
        assert family != "unknown_model_reviewer", (
            f"pre-lineage-era body {body!r} should still produce a "
            "recognized family marker. The implementation PR cannot "
            "quietly demote pre-lineage comments to unknown — that "
            "would invalidate existing in-flight evidence on open PRs."
        )


# ---------------------------------------------------------------------------
# Safety floor: the LINEAGE_REGEX must NOT match heading-only or
# code-block content, only structured Reviewer fields in body prose.
# ---------------------------------------------------------------------------


def test_lineage_regex_does_not_match_heading() -> None:
    """Heading-only content with a paren expression is NOT a Reviewer field."""
    body = "## Reviewer notes (gpt-5.5)\n\nSome review.\n"
    match = LINEAGE_REGEX.search(body)
    # The heading line "## Reviewer notes (gpt-5.5)" starts with "## "
    # not the "Reviewer:" prefix. Must not match.
    assert match is None, (
        "LINEAGE_REGEX matched a markdown heading; the implementation "
        "must scan body prose for the structured Reviewer field, not "
        "any line containing 'Reviewer' and parens."
    )


def test_lineage_parser_ignores_reviewer_shaped_code_block_content() -> None:
    """Reviewer-shaped lines in code blocks are not attestations.

    Defensive concern: a reviewer pasting code with a literal
    `Reviewer: foo (bar)` line (no comment-marker prefix) must not
    populate model_lineage. The implementation PR must use this
    regex-plus-context contract, not the raw regex alone.

    Note: bodies with `# Reviewer: ...` (hash-prefixed Python comments)
    do NOT match because the regex starts with `^\\s*\\*{0,2}Reviewer`
    and `#` is neither whitespace nor an asterisk. Only un-prefixed
    Reviewer-shaped lines inside code blocks need explicit fence
    filtering.
    """
    body = "Sample code:\n\n```\nReviewer: SomeAgent (some-model-v1)\n```\n"
    raw_match = LINEAGE_REGEX.search(body)
    assert raw_match is not None, (
        "This fixture must remain capable of demonstrating why raw "
        "LINEAGE_REGEX is insufficient by itself."
    )

    prose_match = _search_lineage_body_prose(body)
    assert prose_match is None, (
        "The lineage parser contract must ignore Reviewer-shaped "
        "lines inside fenced code blocks so example text cannot be "
        "laundered into model-lineage evidence."
    )
