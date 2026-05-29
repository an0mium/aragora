"""Governance tests for the lineage-bound model-review quorum gap.

These tests are the Tier 4 pre-approval regression target for the design
in ``docs/specs/ADVISORY_REVIEW_RECOGNIZABLE_HEADER.md``.

They intentionally characterize today's unsafe behavior. The future
implementation PR should invert the gap assertions so counted quorum
signals are keyed by disclosed underlying model family, not by
router/product surface markers such as ``factory`` or ``codex``.
"""

from __future__ import annotations

from typing import Any

from aragora.cli.commands.review_queue import (
    _counted_model_reviewer_ids,
    _infer_model_reviewer_from_text,
    _model_review_signals_from_comments,
)


HEAD_SHA = "113a706c92831c0fb889d6e3da35ee454ceb6a94"


def _comment(body: str, *, author: str = "operator") -> dict[str, Any]:
    return {
        "author": {"login": author},
        "body": body,
        "createdAt": "2026-05-27T16:25:18Z",
    }


def _review_body(
    heading_family: str,
    *,
    model_family: str | None = None,
    model_id: str = "gpt-5.5",
) -> str:
    body = f"## {heading_family} independent semantic review on head {HEAD_SHA}\n\n"
    if model_family is not None:
        body += (
            f"**Reviewer harness:** {heading_family.lower()}\n"
            f"**Model family:** {model_family}\n"
            f"**Model id:** {model_id}\n"
            "**Receipt artifact:** /tmp/review-receipt.md\n\n"
        )
    body += "No blocking findings. This is an independent semantic review.\n"
    return body


def _counted_from_bodies(*bodies: str) -> list[str]:
    signals = _model_review_signals_from_comments(
        [_comment(body) for body in bodies],
        head_sha=HEAD_SHA,
    )
    return _counted_model_reviewer_ids(signals, [])


def test_current_factory_without_model_family_counts_as_factory_gap() -> None:
    """Current gap: a router marker counts without lineage disclosure.

    Desired implementation behavior: advisory-only, with
    ``missing_model_family_disclosure`` in the signal's identity
    problems.
    """
    body = _review_body("Factory")

    assert _infer_model_reviewer_from_text(body) == "factory"
    assert _counted_from_bodies(body) == ["factory"]


def test_current_codex_without_model_family_counts_as_codex_gap() -> None:
    """Current gap: Codex counts as a family without model disclosure.

    The selected policy requires Codex to disclose underlying model
    lineage before it contributes to heterogeneity.
    """
    body = _review_body("Codex")

    assert _infer_model_reviewer_from_text(body) == "codex"
    assert _counted_from_bodies(body) == ["codex"]


def test_current_factory_and_codex_openai_disclosures_still_count_as_two_surface_ids() -> None:
    """Current gap: lineage metadata is ignored during counting.

    Both comments disclose OpenAI lineage, but today's packet would
    count the surface markers as two signals: ``codex`` and ``factory``.
    The implementation must count this as one model family: ``openai``.
    """
    factory = _review_body("Factory", model_family="openai", model_id="gpt-5.5")
    codex = _review_body("Codex", model_family="openai", model_id="gpt-5.5-codex")

    assert _counted_from_bodies(factory, codex) == ["codex", "factory"]


def test_current_factory_openai_and_claude_count_for_wrong_reason() -> None:
    """Current behavior satisfies two-signal quorum for the wrong reason.

    Desired implementation behavior is still two counted families, but
    they should be ``claude`` and ``openai`` rather than ``claude`` and
    ``factory``.
    """
    factory = _review_body("Factory", model_family="openai", model_id="gpt-5.5")
    claude = _review_body("Claude", model_family="claude", model_id="claude-opus-4-7")

    assert _counted_from_bodies(factory, claude) == ["claude", "factory"]


def test_current_heading_model_family_conflict_counts_heading_gap() -> None:
    """Current gap: explicit conflicting lineage does not reject a signal.

    Desired implementation behavior: ``## Claude ...`` plus
    ``Model family: openai`` is identity-conflicted and does not count.
    """
    body = _review_body("Claude", model_family="openai", model_id="gpt-5.5")

    assert _infer_model_reviewer_from_text(body) == "claude"
    assert _counted_from_bodies(body) == ["claude"]


def test_current_advisory_aragora_header_stays_unknown() -> None:
    """The old aggregated advisory-review header remains uncounted."""
    body = "## Aragora Code Review\n\nAdvisory-only review. No issues found.\n"

    assert _infer_model_reviewer_from_text(body) == "unknown_model_reviewer"
    assert _counted_from_bodies(body) == []


def test_body_only_family_names_do_not_override_first_heading() -> None:
    """Body prose cannot turn an unknown first heading into a signal."""
    body = (
        "## Aragora Code Review\n\n"
        "**Reviewer harness:** factory\n"
        "**Model family:** openai\n"
        "**Model id:** gpt-5.5\n\n"
        "The body also mentions Claude, Gemini, Grok, Codex, and Factory.\n"
    )

    assert _infer_model_reviewer_from_text(body) == "unknown_model_reviewer"
    assert _counted_from_bodies(body) == []


def test_diff_quoted_family_names_do_not_override_first_heading() -> None:
    """Quoted review input must not be parsed as review identity."""
    body = (
        "## Aragora Code Review\n\n"
        "```diff\n"
        "+from aragora.agents.api_agents.anthropic import claude_client\n"
        "+claude_client.invoke(prompt='gemini-style review by grok')\n"
        "```\n"
    )

    assert _infer_model_reviewer_from_text(body) == "unknown_model_reviewer"
    assert _counted_from_bodies(body) == []
