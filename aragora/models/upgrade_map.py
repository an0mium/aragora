"""Single old→current model-ID map.

Runtime: ``resolve_model_id`` normalises any legacy or superseded spelling
before catalog or pricing lookups. Build time:
``scripts/refresh_model_literals.py`` rewrites literals with the same table,
so the repo never disagrees with the runtime.

Contract (frontier-model-refresh, Task 2, 2026-09-04 controller rulings):

* ``UPGRADES`` keys are spellings of RETIRED or ABSENT models ONLY. A
  spelling that Task 1 attached as a catalog *alias* to an ACTIVE row (e.g.
  ``mistral-medium-latest``, ``qwen/qwen3.8-max``) must NOT be a key here —
  it already resolves via ``spec_or_none``/``by_any_id`` alias lookup, and
  duplicating it as an UPGRADES key would risk the two paths drifting.
  Two bare spellings are the deliberate exception: ``mistral-large`` and
  ``gemini-3.1-pro`` are kept as UPGRADES keys because Task 1 kept them OFF
  their rows' ``aliases`` tuples (they collide with static routing
  hand-rows in ``aragora/routing/provider_config.py``), so this map is
  their only path back to the current id.
* ``resolve_model_id`` checks, in order: (1) ``UPGRADES`` exact-key hit;
  (2) ``spec_or_none`` resolution to an ACTIVE (non-retired) catalog row,
  returning its ``canonical_id``; (3) ``spec_or_none`` resolution to a
  RETIRED row not covered by (1) or (2), returning that row's family
  frontier (``frontier_for(spec.family).canonical_id``); (4) the input
  unchanged. ``None`` in, ``None`` out.
* ``RETIRED_PATTERN`` is built from ``UPGRADES`` keys only, with boundary
  guards so a retired key that happens to be a literal *prefix* of a
  longer ACTIVE spelling (``"claude-fable-5"`` vs. active
  ``"claude-fable-5-1"``; ``"kimi-k2"`` vs. active ``"kimi-k2.7-code"``;
  ``"deepseek-v4-pro"`` vs. active ``"deepseek-v4-pro-0813"``) never
  matches that active spelling. ``tests/models/test_upgrade_map.py``
  asserts this against every active row's ``all_ids()``.
"""

from __future__ import annotations

import re

from aragora.models.catalog import frontier_for, spec_or_none

_ANTHROPIC = "claude-fable-5-1"
_OPENAI = "gpt-6-astra"
_OPENAI_VALUE = "gpt-5.6-terra"
_GOOGLE_PRO = "gemini-3.1-pro-preview"
_GOOGLE_FLASH = "gemini-3.8-flash"
_XAI = "grok-4.6"
_MISTRAL_LARGE = "mistral-large-2512"
_MISTRAL_MEDIUM = "mistral-medium-2604"
_DEEPSEEK = "deepseek-v4-pro-0813"
_QWEN = "qwen3.8-2.4t-a95b"
_KIMI = "kimi-k3"
_META = "muse-spark-1.3"

UPGRADES: dict[str, str] = {
    # Anthropic — everything Claude that is not the current Fable goes to
    # Fable 5.1. NOTE: "claude-fable-5.1" and "anthropic/claude-fable-5.1"
    # are deliberately absent — Task 1 made both catalog aliases of the
    # ACTIVE claude-fable-5-1 row (controller ruling 1).
    **{
        k: _ANTHROPIC
        for k in (
            "claude-fable-5",
            "anthropic/claude-fable-5",
            "claude-3-opus-20240229",
            "claude-3-opus",
            "claude-3-5-sonnet-20241022",
            "claude-3.5-sonnet",
            "claude-3-5-sonnet-20240620",
            "claude-3-7-sonnet-20250219",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-sonnet-4.6",
            "claude-opus-4-20250514",
            "claude-opus-4",
            "claude-opus-4-1-20250805",
            "claude-opus-4-5-20251101",
            "claude-opus-4-6",
            "claude-opus-4.6",
            "claude-opus-4-7",
            "claude-opus-4.7",
            "claude-opus-4.1",
            "anthropic/claude-opus-4.1",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-sonnet-4",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-opus-4",
        )
    },
    # OpenAI — flagship spellings → Astra; small/cheap spellings → Terra
    **{
        k: _OPENAI
        for k in (
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.5",
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.3",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5.6-sol",
            "openai/gpt-4o",
            "openai/gpt-5.3",
            "openai/gpt-5.4",
            "openai/gpt-5.5",
            "openai/gpt-5.6-sol",
            "o1",
            "o3",
            "o3-pro",
            "o4-mini",
        )
    },
    **{
        k: _OPENAI_VALUE
        for k in (
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-5-mini",
            "gpt-5.4-mini",
            "gpt-5.6-luna",
            "openai/gpt-4o-mini",
            "openai/gpt-5.4-mini",
            "openai/gpt-5.6-luna",
            "o1-mini",
            "o3-mini",
        )
    },
    # Google. NOTE: "google/gemini-3.1-pro" is deliberately absent — Task 1
    # made it a catalog alias of the ACTIVE gemini-3.1-pro-preview row
    # (controller ruling 1). The bare "gemini-3.1-pro" spelling stays: it
    # collides with a routing hand-row, so it is NOT a catalog alias, and
    # this map is its only path back to the current id.
    **{
        k: _GOOGLE_PRO
        for k in (
            "gemini-3-pro",
            "gemini-3.1-pro",
            "google/gemini-3-pro",
            "gemini-2.5-pro",
            "gemini-1.5-pro",
            "gemini-pro",
        )
    },
    **{
        k: _GOOGLE_FLASH
        for k in (
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-3-flash",
            "gemini-3.5-flash",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "google/gemini-2.0-flash",
            "google/gemini-3-flash-preview",
        )
    },
    # xAI
    **{
        k: _XAI
        for k in (
            "grok-2",
            "grok-3",
            "grok-3-mini",
            "grok-4",
            "grok-4-latest",
            "grok-4.3",
            "grok-4.5",
            "x-ai/grok-4",
            "x-ai/grok-4.3",
            "x-ai/grok-4.5",
        )
    },
    # Mistral. NOTE: "mistral-large-latest" and "mistral-medium-latest"
    # (plus "mistral-medium-3.5") are deliberately absent — Task 1 made
    # them catalog aliases of their ACTIVE rows (controller ruling 1). The
    # bare "mistral-large" spelling stays for the same reason as
    # "gemini-3.1-pro" above: it collides with a routing hand-row, so it
    # is not a catalog alias, and this map is its only path back.
    **{
        k: _MISTRAL_LARGE
        for k in ("mistral-large", "mistral-large-2411", "mistralai/mistral-large")
    },
    **{
        k: _MISTRAL_MEDIUM
        for k in ("mistral-medium", "mistral-medium-3.1", "mistralai/mistral-medium-3.1")
    },
    # OpenRouter-routed families
    **{
        k: _DEEPSEEK
        for k in (
            "deepseek-r1",
            "deepseek/deepseek-r1",
            "deepseek-v3",
            "deepseek/deepseek-v3",
            "deepseek-v4-pro",
            "deepseek/deepseek-v4-pro",
            "deepseek-chat",
            "deepseek/deepseek-chat",
        )
    },
    # NOTE: "qwen3.8-max" and "qwen/qwen3.8-max" are deliberately absent —
    # Task 1 made both catalog aliases of the ACTIVE qwen3.8-2.4t-a95b row
    # (controller ruling 1).
    **{
        k: _QWEN
        for k in (
            "qwen3-max",
            "qwen/qwen3-max",
            "qwen3.5-plus-02-15",
            "qwen/qwen3.5-plus-02-15",
            "qwen3.7-max",
            "qwen/qwen3.7-max",
            "qwen3-coder",
            "qwen/qwen3-coder",
        )
    },
    **{
        k: _KIMI
        for k in (
            "kimi-k2",
            "moonshotai/kimi-k2",
            "kimi-k2.5",
            "moonshotai/kimi-k2.5",
            "kimi-k2.6",
            "moonshotai/kimi-k2.6",
            "kimi-k2-thinking",
            "moonshotai/kimi-k2-thinking",
            "moonshot-v1-8k",
        )
    },
    **{
        k: _META
        for k in (
            "llama-3.3-70b",
            "meta-llama/llama-3.3-70b-instruct",
            "llama-4-maverick",
            "meta-llama/llama-4-maverick",
            "llama-4-scout",
            "meta-llama/llama-4-scout",
            "meta/muse-spark-1.1",
            "meta/muse-spark-1.2",
        )
    },
}

# Characters that make up a single model-id "token". A match must not be
# immediately preceded or followed by one of these — otherwise a retired
# key that happens to be a literal prefix of a longer id (active or not)
# would falsely match as a substring (e.g. retired "claude-fable-5" is a
# prefix of active "claude-fable-5-1"; retired-adjacent "kimi-k2" is a
# prefix of active "kimi-k2.7-code"). This is the exact collision class
# controller ruling 3 guards against.
_TOKEN_CHAR = r"[A-Za-z0-9_.\-/]"

RETIRED_PATTERN: re.Pattern[str] = re.compile(
    rf"(?<!{_TOKEN_CHAR})"
    rf"(?:{'|'.join(re.escape(k) for k in sorted(UPGRADES, key=len, reverse=True))})"
    rf"(?!{_TOKEN_CHAR})"
)


def resolve_model_id(model_id: str | None) -> str | None:
    """Map a legacy or superseded model spelling to the current catalog id.

    Order: an exact ``UPGRADES`` key hit wins; otherwise a spelling that
    resolves (via any catalog spelling: canonical/direct/openrouter/alias)
    to an ACTIVE row returns that row's ``canonical_id`` unchanged; a
    spelling that resolves to a RETIRED row not covered by ``UPGRADES``
    returns that row's family frontier; anything else passes through
    unchanged. ``None`` in, ``None`` out.
    """
    if model_id is None:
        return None
    upgraded = UPGRADES.get(model_id)
    if upgraded is not None:
        return upgraded
    spec = spec_or_none(model_id)
    if spec is None:
        return model_id
    if not spec.retired:
        return spec.canonical_id
    return frontier_for(spec.family).canonical_id
