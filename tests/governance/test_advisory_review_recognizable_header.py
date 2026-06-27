"""Governance tests for the lineage-bound model-review quorum gap.

These tests are the Tier 4 pre-approval regression target for the design
in ``docs/specs/ADVISORY_REVIEW_RECOGNIZABLE_HEADER.md``.

They pin the strict lineage-bound implementation: counted quorum
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


def _signals_from_bodies(*bodies: str) -> list[dict[str, Any]]:
    return _model_review_signals_from_comments(
        [_comment(body) for body in bodies],
        head_sha=HEAD_SHA,
    )


def test_factory_without_model_family_is_advisory_only() -> None:
    """Router markers require explicit lineage disclosure to count."""
    body = _review_body("Factory")

    assert _infer_model_reviewer_from_text(body) == "factory"
    assert _counted_from_bodies(body) == []
    signal = _signals_from_bodies(body)[0]
    assert signal["surface_reviewer_id"] == "factory"
    assert "missing_model_family_disclosure" in signal["identity_problems"]


def test_codex_without_model_family_is_advisory_only() -> None:
    """Codex is a product/harness marker and must disclose model lineage."""
    body = _review_body("Codex")

    assert _infer_model_reviewer_from_text(body) == "codex"
    assert _counted_from_bodies(body) == []
    signal = _signals_from_bodies(body)[0]
    assert signal["surface_reviewer_id"] == "codex"
    assert "missing_model_family_disclosure" in signal["identity_problems"]


def test_factory_and_codex_openai_disclosures_count_as_one_model_family() -> None:
    """Two router comments disclosing the same family count once."""
    factory = _review_body("Factory", model_family="openai", model_id="gpt-5.5")
    codex = _review_body("Codex", model_family="openai", model_id="gpt-5.5-codex")

    assert _counted_from_bodies(factory, codex) == ["openai"]


def test_factory_openai_and_claude_count_as_two_model_families() -> None:
    """Mixed router and direct-family signals count by lineage."""
    factory = _review_body("Factory", model_family="openai", model_id="gpt-5.5")
    claude = _review_body("Claude", model_family="claude", model_id="claude-opus-4-7")

    assert _counted_from_bodies(factory, claude) == ["claude", "openai"]


def test_heading_model_family_conflict_is_rejected() -> None:
    """``## Claude ...`` plus ``Model family: openai`` does not count."""
    body = _review_body("Claude", model_family="openai", model_id="gpt-5.5")

    assert _infer_model_reviewer_from_text(body) == "claude"
    assert _counted_from_bodies(body) == []
    signal = _signals_from_bodies(body)[0]
    assert "heading_model_family_conflict" in signal["identity_problems"]


def test_unknown_model_family_disclosure_is_visible_but_uncounted() -> None:
    body = _review_body("Factory", model_family="unknown-provider", model_id="mystery-v1")

    assert _counted_from_bodies(body) == []
    signal = _signals_from_bodies(body)[0]
    assert signal["surface_reviewer_id"] == "factory"
    assert "unknown_model_family" in signal["identity_problems"]


def test_missing_receipt_artifact_is_diagnostic_metadata() -> None:
    body = (
        f"## Factory independent semantic review on head {HEAD_SHA}\n\n"
        "**Reviewer harness:** factory\n"
        "**Model family:** openai\n"
        "**Model id:** gpt-5.5\n\n"
        "No blocking findings. This is an independent semantic review.\n"
    )

    assert _counted_from_bodies(body) == ["openai"]
    signal = _signals_from_bodies(body)[0]
    assert signal["model_family"] == "openai"
    assert "missing_receipt_artifact" in signal["identity_problems"]


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
