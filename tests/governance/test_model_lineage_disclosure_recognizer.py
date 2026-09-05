"""Governance tests for strict lineage-bound reviewer identity parsing."""

from __future__ import annotations

import pytest

from aragora.cli.commands.review_queue import (
    _normalize_model_reviewer_id,
    _resolve_model_review_identity,
)
from aragora.swarm.quorum_evidence import (
    _CODEX_DEFAULT_MODELS,
    _OPENROUTER_REVIEWER_MODELS,
    canonical_family,
)


def _body(
    heading: str,
    *,
    model_family: str | None = None,
    model_id: str = "gpt-6-astra",
    receipt: str | None = "/tmp/review.md",
) -> str:
    text = f"## {heading}\n\n"
    text += "**Reviewer harness:** factory\n"
    if model_family is not None:
        text += f"**Model family:** {model_family}\n"
    text += f"**Model id:** {model_id}\n"
    if receipt is not None:
        text += f"**Receipt artifact:** {receipt}\n"
    text += "\nNo blocking findings.\n"
    return text


def test_router_heading_requires_model_family_disclosure() -> None:
    identity = _resolve_model_review_identity(_body("Factory focused dogfood"))

    assert identity.surface_reviewer_id == "factory"
    assert identity.model_family == ""
    assert "missing_model_family_disclosure" in identity.identity_problems


def test_router_heading_with_canonical_family_counts_by_model_family() -> None:
    identity = _resolve_model_review_identity(
        _body("Factory focused dogfood", model_family="openai", model_id="gpt-6-astra")
    )

    assert identity.surface_reviewer_id == "factory"
    assert identity.model_family == "openai"
    assert identity.model_id == "gpt-6-astra"
    assert identity.identity_source == "model_family_metadata"


def test_direct_family_heading_self_maps_without_model_family_metadata() -> None:
    identity = _resolve_model_review_identity(
        "## Claude independent semantic review on head abc1234\n\nNo findings.\n"
    )

    assert identity.surface_reviewer_id == "claude"
    assert identity.model_family == "claude"
    assert identity.identity_source == "direct_heading"


def test_direct_heading_conflicting_model_family_is_rejected() -> None:
    identity = _resolve_model_review_identity(
        _body("Claude independent semantic review", model_family="openai", model_id="gpt-6-astra")
    )

    assert identity.surface_reviewer_id == "claude"
    assert identity.model_family == "openai"
    assert "heading_model_family_conflict" in identity.identity_problems


def test_unknown_model_family_is_reported() -> None:
    identity = _resolve_model_review_identity(
        _body("Factory independent semantic review", model_family="not-a-family")
    )

    assert identity.surface_reviewer_id == "factory"
    assert identity.model_family == ""
    assert "unknown_model_family" in identity.identity_problems


def test_body_only_metadata_does_not_override_unknown_heading() -> None:
    identity = _resolve_model_review_identity(
        "## Aragora Code Review\n\n"
        "**Reviewer harness:** factory\n"
        "**Model family:** openai\n"
        "**Model id:** gpt-6-astra\n"
        "**Receipt artifact:** /tmp/review.md\n"
    )

    assert identity.surface_reviewer_id == "unknown_model_reviewer"
    assert identity.model_family == "openai"
    assert "unknown_surface_reviewer" in identity.identity_problems


def test_fenced_metadata_does_not_override_nearby_block() -> None:
    identity = _resolve_model_review_identity(
        "## Factory independent semantic review\n\n"
        "```md\n"
        "**Model family:** openai\n"
        "```\n"
        "No structured metadata outside the example block.\n"
    )

    assert identity.surface_reviewer_id == "factory"
    assert identity.model_family == ""
    assert "missing_model_family_disclosure" in identity.identity_problems


def test_later_heading_metadata_does_not_override_first_heading() -> None:
    identity = _resolve_model_review_identity(
        "## Factory independent semantic review\n\n"
        "No metadata near the first heading.\n\n"
        "## Claude follow-up\n\n"
        "**Model family:** claude\n"
        "**Model id:** claude-fable-5-1\n"
        "**Receipt artifact:** /tmp/review.md\n"
    )

    assert identity.surface_reviewer_id == "factory"
    assert identity.model_family == ""
    assert "missing_model_family_disclosure" in identity.identity_problems


# --- Frontier refresh (2026-09-04): Claude Fable 5.1 / GPT-6 Astra ----------


@pytest.mark.parametrize(
    "text,family",
    [
        ("Model family: openai\nModel: gpt-6-astra", "openai"),
        ("Reviewer: claude (claude-fable-5-1)", "claude"),
        ("model=x-ai/grok-4.6", "grok"),
        ("model=meta/muse-spark-1.3", "meta"),
    ],
)
def test_identity_resolver_recognises_frontier_ids(text: str, family: str) -> None:
    assert _resolve_model_review_identity(text).model_family == family


def test_reviewer_map_is_frontier() -> None:
    assert _OPENROUTER_REVIEWER_MODELS["claude"] == "anthropic/claude-fable-5.1"
    assert _OPENROUTER_REVIEWER_MODELS["openai"] == "openai/gpt-6-astra"
    assert _OPENROUTER_REVIEWER_MODELS["grok"] == "x-ai/grok-4.6"
    assert _OPENROUTER_REVIEWER_MODELS["deepseek"] == "deepseek/deepseek-v4-pro-0813"
    assert _OPENROUTER_REVIEWER_MODELS["kimi"] == "moonshotai/kimi-k3"
    assert _OPENROUTER_REVIEWER_MODELS["meta"] == "meta/muse-spark-1.3"
    assert _CODEX_DEFAULT_MODELS[0] == "gpt-6-astra"
    for fam in ("claude", "openai", "grok", "gemini", "deepseek", "qwen", "kimi", "meta"):
        assert canonical_family(_OPENROUTER_REVIEWER_MODELS[fam]) == fam


@pytest.mark.parametrize("family", ["tencent", "bytedance"])
def test_dropped_reviewer_families_have_no_map_entry(family: str) -> None:
    # tencent/bytedance have no priced, active catalog row (test_reachable_
    # defaults.py requires every reachable default to be one), so they were
    # dropped from the live reviewer map. They remain recognized families
    # (FAMILY_PROVIDERS/FAMILY_DISPLAY/_FAMILY_ALIASES) so historical evidence
    # comments still parse.
    assert family not in _OPENROUTER_REVIEWER_MODELS


# --- Bounded "meta" marker (fix round 1, #9069): "metadata" is not "meta" --


def test_metadata_heading_is_not_recognised_as_meta() -> None:
    identity = _resolve_model_review_identity("## Round 3 metadata summary\n\nNo findings.\n")

    assert identity.surface_reviewer_id == "unknown_model_reviewer"
    assert identity.model_family == ""


def test_model_metadata_check_heading_is_not_recognised_as_meta() -> None:
    identity = _resolve_model_review_identity("## Model metadata check\n\nNo findings.\n")

    assert identity.surface_reviewer_id == "unknown_model_reviewer"
    assert identity.model_family == ""


def test_review_metadata_heading_with_deepseek_family_has_no_conflict() -> None:
    # "Review metadata" must not resolve the heading itself to the "meta"
    # family via a bare "meta" substring match on "metadata" -- that would
    # spuriously conflict with the genuinely-disclosed deepseek family below.
    identity = _resolve_model_review_identity(
        _body("Review metadata", model_family="deepseek", model_id="deepseek-v4-pro-0813")
    )

    assert identity.model_family == "deepseek"
    assert "heading_model_family_conflict" not in identity.identity_problems


def test_meta_independent_review_heading_still_resolves_to_meta() -> None:
    # Positive case: the collector's own meta reviewer heading must still work
    # once the marker is bounded.
    identity = _resolve_model_review_identity("## Meta independent model review\n\nNo findings.\n")

    assert identity.surface_reviewer_id == "meta"
    assert identity.model_family == "meta"


def test_normalize_model_reviewer_id_metadata_is_not_meta() -> None:
    assert _normalize_model_reviewer_id("metadata") != "meta"
    assert _normalize_model_reviewer_id("metadata") == ""
