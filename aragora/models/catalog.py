"""Canonical model catalog — the single typed source of model identity.

Phase 1 of the catalog design adjudicated on #9073/#9075 (2026-07-16): those
PRs empirically demonstrated that model identity and pricing were duplicated
across at least ELEVEN runtime tables, each independently discovered drifted
by adversarial review (three live provider reprices were caught mid-review).

Contract:

* ``CATALOG`` is the canonical record: ids, aliases, pricing (USD per 1M
  tokens), context/output limits, release + soak dates.
* ``catalog_snapshot.json`` (same directory) is the committed capture of the
  live OpenRouter catalog used for OFFLINE validation: required CI never
  performs network calls. ``scripts/model_catalog_drift.py`` is the advisory
  live-vs-snapshot differ.
* Existing runtime tables (billing/metering/routing/pdb/cost-estimation) are
  NOT rewired in phase 1. Instead, ``tests/models/test_catalog.py`` enforces
  that every table row for an ENFORCED model matches this catalog, so the
  mirrors can no longer drift silently. Phase 2 migrates consumers to
  projections generated from here.
* Quorum-family ELIGIBILITY deliberately stays in governance policy
  (``aragora/swarm/quorum_evidence.py``), never in this catalog: which model
  may produce merge-authority evidence is a Tier-4 governance decision, not a
  runtime lookup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

__all__ = [
    "CATALOG",
    "ENFORCED_MODELS",
    "FRONTIER",
    "ModelSpec",
    "by_any_id",
    "frontier_for",
    "load_snapshot",
    "snapshot_path",
    "spec_or_none",
    "utc_today",
]


def utc_today() -> date:
    """Canonical 'today' for soak governance, anchored to UTC.

    Soak windows are a governance boundary (must-not-adopt-before dates),
    so they must not flip earlier or later depending on the host's local
    timezone."""
    return datetime.now(timezone.utc).date()


@dataclass(frozen=True)
class ModelSpec:
    """Canonical identity + economics for one model."""

    canonical_id: str
    provider: str
    direct_id: str
    openrouter_id: str
    input_per_mtok: float
    output_per_mtok: float
    context_window: int
    max_output_tokens: int
    release_date: date
    # Merge-authority surfaces must not adopt a model before this date
    # (the 14-day availability rule); None = long-established.
    soak_until: date | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    # Documented long-context tier: requests whose PROMPT reaches
    # ``long_context_threshold`` tokens are billed at the ``*_long`` rates for
    # ALL tokens in the request (xAI's documented model; source recorded per
    # entry). None = flat pricing. The OpenRouter snapshot mirrors only the
    # flat fields, so tier rates are verified against provider pricing pages.
    long_context_threshold: int | None = None
    input_per_mtok_long: float | None = None
    output_per_mtok_long: float | None = None
    # Pretraining lineage, used to group rows for frontier_for()/FRONTIER
    # lookup: anthropic, openai, google, xai, mistral, deepseek, qwen,
    # moonshot, meta, zai, minimax.
    family: str = ""
    # Product tier within a family: "flagship" (the family's headline
    # reasoning model), "value" (a faster/cheaper sibling released
    # alongside or after the flagship, e.g. a "flash"/"terra"-style SKU),
    # "fallback" (an older or heavier line kept resolvable/priced but not
    # promoted as the family's current default, e.g. Opus alongside
    # Fable), or "code" (a specialized coding variant). frontier_for()
    # only considers "flagship" rows, so a same-family non-flagship row
    # released later than the flagship never displaces it.
    tier: str = "flagship"
    supports_sampling_params: bool = True
    thinking_default_on: bool = False
    forced_tool_choice_allowed: bool = True
    max_tokens_param: str = "max_tokens"  # or "max_completion_tokens"
    reasoning_effort_default: str | None = None
    cache_read_per_mtok: float | None = None
    retired: bool = False

    def all_ids(self) -> tuple[str, ...]:
        return (self.canonical_id, self.direct_id, self.openrouter_id, *self.aliases)

    def rates_for(self, prompt_tokens: int) -> tuple[float, float]:
        """Applicable (input, output) USD-per-MTok pair for a request whose
        prompt is ``prompt_tokens`` long. Falls back to flat rates when the
        model has no documented tier."""
        if (
            self.long_context_threshold is not None
            and prompt_tokens >= self.long_context_threshold
            and self.input_per_mtok_long is not None
            and self.output_per_mtok_long is not None
        ):
            return (self.input_per_mtok_long, self.output_per_mtok_long)
        return (self.input_per_mtok, self.output_per_mtok)

    def is_under_soak(self, today: date | None = None) -> bool:
        """True while the model is inside its post-release soak window
        (before ``soak_until``; the 14-day availability rule).

        Adoption surfaces — merge-authority evidence, routing candidate
        enumeration — must not offer the model while under soak. Id
        resolution (``by_any_id``) and cost lookup for its ids remain
        valid throughout the window. Defaults to the UTC calendar date
        (``utc_today``): soak is a governance boundary and must not shift
        with the host timezone.
        """
        if self.soak_until is None:
            return False
        return (today if today is not None else utc_today()) < self.soak_until


# Prices are USD per 1M tokens, captured from the live OpenRouter catalog
# (most recently refreshed 2026-08-16, plus a 2026-09-04 incremental capture
# for the frontier-model-refresh rows added that day; see
# catalog_snapshot.json for the raw capture). Direct-provider ids are
# verified against provider model lists.
CATALOG: dict[str, ModelSpec] = {
    spec.canonical_id: spec
    for spec in (
        ModelSpec(
            canonical_id="claude-fable-5",
            provider="anthropic",
            family="anthropic",
            direct_id="claude-fable-5",
            openrouter_id="anthropic/claude-fable-5",
            input_per_mtok=10.00,
            output_per_mtok=50.00,
            context_window=1_000_000,
            max_output_tokens=128_000,
            release_date=date(2026, 6, 20),
            supports_sampling_params=False,
            thinking_default_on=True,
            # Superseded by claude-fable-5-1 (frontier-model-refresh,
            # 2026-09-04); retained resolvable/priced, not the frontier pick.
            retired=True,
        ),
        ModelSpec(
            canonical_id="claude-opus-5",
            provider="anthropic",
            family="anthropic",
            # tier="fallback": Opus 5 released 2026-07-24, LATER than Fable
            # 5.1 (2026-06-24), so without this frontier_for() would pick
            # Opus over Fable by date — wrong per the frontier-model-refresh
            # spec, which names Fable 5.1 as the anthropic frontier.
            # frontier_for() only considers tier="flagship" rows.
            tier="fallback",
            direct_id="claude-opus-5",
            openrouter_id="anthropic/claude-opus-5",
            # Same economics as Opus 4.8 ($5/$25) per the provider model page.
            input_per_mtok=5.00,
            output_per_mtok=25.00,
            context_window=1_000_000,
            max_output_tokens=128_000,
            release_date=date(2026, 7, 24),
            # SOAK WAIVED BY OPERATOR (2026-07-24). Opus 5 is a day-0 model and
            # would normally carry soak_until=2026-08-07 under the 14-day
            # availability rule, which would bar it from merge-authority
            # evidence and routing candidate enumeration. The operator
            # explicitly directed an immediate repo-wide bump, so it is
            # adoptable from release. Reinstating the window is a one-line
            # change: soak_until=date(2026, 8, 7).
            soak_until=None,
            supports_sampling_params=False,
            thinking_default_on=True,
        ),
        ModelSpec(
            # Retained deliberately: still Active upstream (retires no sooner
            # than 2027-05-28) AND it is Opus 5's documented fallback target for
            # cyber-classifier refusals, so it must stay resolvable and priced.
            canonical_id="claude-opus-4-8",
            provider="anthropic",
            family="anthropic",
            tier="fallback",  # see claude-opus-5 for why
            direct_id="claude-opus-4-8",
            openrouter_id="anthropic/claude-opus-4.8",
            input_per_mtok=5.00,
            output_per_mtok=25.00,
            context_window=1_000_000,
            max_output_tokens=128_000,
            release_date=date(2026, 2, 10),
            aliases=("claude-opus-4.8", "anthropic/claude-opus-4-8"),
            supports_sampling_params=False,
            thinking_default_on=True,
        ),
        ModelSpec(
            canonical_id="gpt-5.6-sol",
            provider="openai",
            family="openai",
            direct_id="gpt-5.6-sol",
            openrouter_id="openai/gpt-5.6-sol",
            input_per_mtok=5.00,
            output_per_mtok=30.00,
            context_window=1_050_000,
            max_output_tokens=128_000,
            release_date=date(2026, 7, 9),
            soak_until=date(2026, 7, 23),
            supports_sampling_params=False,
            max_tokens_param="max_completion_tokens",
            # Superseded by gpt-6-astra (frontier-model-refresh, 2026-09-04).
            retired=True,
        ),
        ModelSpec(
            canonical_id="gpt-5.5",
            provider="openai",
            family="openai",
            direct_id="gpt-5.5",
            openrouter_id="openai/gpt-5.5",
            # Provider repriced from 2.50/10.00 on ~2026-07-14 (caught live
            # during the #9073/#9075 reviews).
            input_per_mtok=5.00,
            output_per_mtok=30.00,
            context_window=1_050_000,
            max_output_tokens=128_000,
            release_date=date(2025, 11, 1),
            supports_sampling_params=False,
            max_tokens_param="max_completion_tokens",
            # Superseded by gpt-6-astra (frontier-model-refresh, 2026-09-04).
            retired=True,
        ),
        ModelSpec(
            canonical_id="grok-4.5",
            provider="xai",
            family="xai",
            direct_id="grok-4.5",
            openrouter_id="x-ai/grok-4.5",
            input_per_mtok=2.00,
            output_per_mtok=6.00,
            context_window=500_000,
            max_output_tokens=64_000,
            release_date=date(2026, 7, 8),
            soak_until=date(2026, 7, 22),
            # docs.x.ai/developers/pricing (verified 2026-07-27): prompts
            # >= 200k tokens bill 2x for the whole request.
            long_context_threshold=200_000,
            input_per_mtok_long=4.00,
            output_per_mtok_long=12.00,
            # Superseded by grok-4.6 (frontier-model-refresh, 2026-09-04).
            retired=True,
        ),
        ModelSpec(
            canonical_id="grok-4.3",
            provider="xai",
            family="xai",
            direct_id="grok-4.3",
            openrouter_id="x-ai/grok-4.3",
            input_per_mtok=1.25,
            output_per_mtok=2.50,
            context_window=1_000_000,
            max_output_tokens=64_000,
            release_date=date(2026, 4, 1),
            # docs.x.ai/developers/pricing (verified 2026-07-27).
            long_context_threshold=200_000,
            input_per_mtok_long=2.50,
            output_per_mtok_long=5.00,
            # Superseded by grok-4.6 (frontier-model-refresh, 2026-09-04).
            retired=True,
        ),
        ModelSpec(
            canonical_id="sonar-reasoning-pro",
            provider="perplexity",
            family="perplexity",
            direct_id="sonar-reasoning-pro",
            openrouter_id="perplexity/sonar-reasoning-pro",
            input_per_mtok=2.00,
            output_per_mtok=8.00,
            context_window=128_000,
            max_output_tokens=128_000,
            release_date=date(2025, 3, 7),
        ),
        ModelSpec(
            canonical_id="command-a",
            provider="cohere",
            family="cohere",
            direct_id="command-a-03-2025",
            openrouter_id="cohere/command-a",
            input_per_mtok=2.50,
            output_per_mtok=10.00,
            context_window=256_000,
            max_output_tokens=8_192,
            release_date=date(2025, 3, 13),
        ),
        ModelSpec(
            canonical_id="jamba-large-1.7",
            provider="ai21",
            family="ai21",
            direct_id="jamba-large",
            openrouter_id="ai21/jamba-large-1.7",
            input_per_mtok=2.00,
            output_per_mtok=8.00,
            context_window=256_000,
            max_output_tokens=4_096,
            release_date=date(2025, 8, 8),
        ),
        ModelSpec(
            # Supersedes qwen3.8-max (frontier-model-refresh, 2026-09-04);
            # the old canonical/openrouter spellings are kept as aliases so
            # existing lookups by "qwen3.8-max" / "qwen/qwen3.8-max" keep
            # resolving (controller ruling 2).
            canonical_id="qwen3.8-2.4t-a95b",
            provider="openrouter",
            family="qwen",
            direct_id="qwen3.8-2.4t-a95b",
            openrouter_id="qwen/qwen3.8-2.4t-a95b",
            input_per_mtok=2.00,
            output_per_mtok=6.00,
            context_window=1_048_576,
            max_output_tokens=131_072,
            release_date=date(2026, 8, 12),
            aliases=("qwen3.8-max", "qwen/qwen3.8-max"),
        ),
        ModelSpec(
            canonical_id="qwen3.7-max",
            provider="alibaba",
            family="qwen",
            direct_id="qwen3.7-max",
            openrouter_id="qwen/qwen3.7-max",
            # Repriced from 1.25/3.75 between 2026-07-10 and 2026-07-16
            # (caught live during the #9073 drain).
            input_per_mtok=1.475,
            output_per_mtok=4.425,
            context_window=1_000_000,
            max_output_tokens=32_768,
            release_date=date(2026, 6, 1),
            # Superseded by qwen3.8-2.4t-a95b (frontier-model-refresh,
            # 2026-09-04).
            retired=True,
        ),
        ModelSpec(
            canonical_id="kimi-k3",
            provider="moonshot",
            family="moonshot",
            direct_id="kimi-k3",
            openrouter_id="moonshotai/kimi-k3",
            input_per_mtok=3.00,
            output_per_mtok=15.00,
            context_window=1_048_576,
            # OpenRouter does not currently publish a completion-token cap;
            # retain the existing conservative application cap.
            max_output_tokens=32_768,
            release_date=date(2026, 7, 16),
            soak_until=date(2026, 7, 30),
        ),
        ModelSpec(
            canonical_id="kimi-k2.7-code",
            provider="moonshot",
            family="moonshot",
            tier="code",
            direct_id="kimi-k2.7-code",
            openrouter_id="moonshotai/kimi-k2.7-code",
            # Live OpenRouter reprice captured by the 2026-08-16 snapshot
            # refresh; the reviewer/runtime pin itself remains unchanged.
            input_per_mtok=0.71,
            output_per_mtok=3.50,
            context_window=262_144,
            max_output_tokens=32_768,
            release_date=date(2026, 6, 15),
            # NOTE: "moonshotai/kimi-k2.6" is deliberately NOT an alias here.
            # It is a distinct, live, separately-priced OpenRouter model
            # ($0.95/$4.00 per MTok on the live catalog, 2026-07-17), so
            # aliasing it onto k2.7-code would force k2.6 mirror rows to the
            # wrong prices. Catalog k2.6 as its own ModelSpec if it needs
            # enforcement.
        ),
        # ---------------------------------------------------------------
        # Frontier model refresh (2026-09-04): current per-provider frontier
        # rows. See FRONTIER / frontier_for() below for the canonical
        # per-family "current default" lookup these feed.
        # ---------------------------------------------------------------
        ModelSpec(
            canonical_id="claude-fable-5-1",
            provider="anthropic",
            family="anthropic",
            direct_id="claude-fable-5-1",
            openrouter_id="anthropic/claude-fable-5.1",
            input_per_mtok=10.0,
            output_per_mtok=50.0,
            cache_read_per_mtok=0.25,
            context_window=1_000_000,
            max_output_tokens=128_000,
            release_date=date(2026, 6, 24),
            supports_sampling_params=False,
            thinking_default_on=True,
            forced_tool_choice_allowed=False,
            aliases=("claude-fable-5.1", "anthropic/claude-fable-5-1"),
        ),
        ModelSpec(
            canonical_id="gpt-6-astra",
            provider="openai",
            family="openai",
            direct_id="gpt-6-astra",
            openrouter_id="openai/gpt-6-astra",
            input_per_mtok=10.0,
            output_per_mtok=50.0,
            cache_read_per_mtok=1.0,
            context_window=1_050_000,
            max_output_tokens=128_000,
            release_date=date(2026, 9, 3),
            soak_until=date(2026, 9, 17),
            supports_sampling_params=False,
            max_tokens_param="max_completion_tokens",
            reasoning_effort_default="high",
            long_context_threshold=272_000,
            input_per_mtok_long=20.0,
            output_per_mtok_long=75.0,
        ),
        ModelSpec(
            canonical_id="gpt-5.6-terra",
            provider="openai",
            family="openai",
            tier="value",
            direct_id="gpt-5.6-terra",
            openrouter_id="openai/gpt-5.6-terra",
            input_per_mtok=2.0,
            output_per_mtok=12.0,
            context_window=1_050_000,
            max_output_tokens=128_000,
            release_date=date(2026, 7, 9),
            supports_sampling_params=False,
            max_tokens_param="max_completion_tokens",
            reasoning_effort_default="medium",
        ),
        ModelSpec(
            canonical_id="gemini-3.1-pro-preview",
            provider="google",
            family="google",
            direct_id="gemini-3.1-pro-preview",
            openrouter_id="google/gemini-3.1-pro-preview",
            input_per_mtok=2.0,
            output_per_mtok=12.0,
            context_window=1_048_576,
            max_output_tokens=65_536,
            release_date=date(2026, 2, 19),
            aliases=("gemini-3.1-pro", "google/gemini-3.1-pro"),
        ),
        ModelSpec(
            # tier="value": Flash released 2026-09-02, LATER than
            # Pro-preview's 2026-02-19, so without this frontier_for()
            # would pick Flash over Pro by date — wrong per the
            # frontier-model-refresh spec, which names Gemini 3.1 Pro as
            # the google frontier. frontier_for() only considers
            # tier="flagship" rows.
            canonical_id="gemini-3.8-flash",
            provider="google",
            family="google",
            tier="value",
            direct_id="gemini-3.8-flash",
            openrouter_id="google/gemini-3.8-flash",
            input_per_mtok=0.75,
            output_per_mtok=3.75,
            context_window=1_048_576,
            max_output_tokens=65_536,
            release_date=date(2026, 9, 2),
            soak_until=date(2026, 9, 16),
        ),
        ModelSpec(
            canonical_id="grok-4.6",
            provider="xai",
            family="xai",
            direct_id="grok-4.6",
            openrouter_id="x-ai/grok-4.6",
            input_per_mtok=2.0,
            output_per_mtok=6.0,
            context_window=500_000,
            max_output_tokens=128_000,
            release_date=date(2026, 8, 12),
            long_context_threshold=200_000,
            input_per_mtok_long=4.0,
            output_per_mtok_long=12.0,
        ),
        ModelSpec(
            canonical_id="mistral-medium-2604",
            provider="mistral",
            family="mistral",
            direct_id="mistral-medium-2604",
            openrouter_id="mistralai/mistral-medium-3-5",
            input_per_mtok=1.5,
            output_per_mtok=7.5,
            context_window=262_144,
            max_output_tokens=262_144,
            release_date=date(2026, 4, 30),
            aliases=("mistral-medium-3.5", "mistral-medium-latest"),
        ),
        ModelSpec(
            canonical_id="mistral-large-2512",
            provider="mistral",
            family="mistral",
            direct_id="mistral-large-2512",
            openrouter_id="mistralai/mistral-large-2512",
            input_per_mtok=0.5,
            output_per_mtok=1.5,
            context_window=262_144,
            max_output_tokens=131_072,
            release_date=date(2025, 12, 1),
            aliases=("mistral-large-latest", "mistral-large"),
        ),
        ModelSpec(
            canonical_id="deepseek-v4-pro-0813",
            provider="openrouter",
            family="deepseek",
            direct_id="deepseek-v4-pro-0813",
            openrouter_id="deepseek/deepseek-v4-pro-0813",
            # Live OpenRouter capture (2026-09-04): $0.00000112068 /
            # $0.00000336204 per token, i.e. 1.1207 / 3.362 per MTok (not the
            # cleaner 1.12 / 3.36 in the original spec) — snapshot refresh
            # would otherwise flag drift against the committed capture.
            input_per_mtok=1.1207,
            output_per_mtok=3.362,
            context_window=1_048_576,
            max_output_tokens=131_072,
            release_date=date(2026, 8, 12),
        ),
        ModelSpec(
            canonical_id="muse-spark-1.3",
            provider="openrouter",
            family="meta",
            direct_id="muse-spark-1.3",
            openrouter_id="meta/muse-spark-1.3",
            input_per_mtok=1.25,
            output_per_mtok=4.25,
            context_window=1_048_576,
            max_output_tokens=131_072,
            release_date=date(2026, 9, 2),
            soak_until=date(2026, 9, 16),
        ),
        ModelSpec(
            canonical_id="glm-5.2",
            provider="openrouter",
            family="zai",
            direct_id="glm-5.2",
            openrouter_id="z-ai/glm-5.2",
            # Live OpenRouter capture (2026-09-04): $0.000000966 /
            # $0.000003036 per token, i.e. 0.966 / 3.036 per MTok.
            input_per_mtok=0.966,
            output_per_mtok=3.036,
            context_window=1_048_576,
            max_output_tokens=131_072,
            release_date=date(2026, 5, 1),
        ),
        ModelSpec(
            canonical_id="minimax-m3",
            provider="openrouter",
            family="minimax",
            direct_id="minimax-m3",
            openrouter_id="minimax/minimax-m3",
            input_per_mtok=0.30,
            output_per_mtok=1.20,
            context_window=1_048_576,
            max_output_tokens=131_072,
            release_date=date(2026, 5, 31),
        ),
    )
}


# Models whose runtime-table rows are ENFORCED against this catalog by
# tests/models/test_catalog.py. Grows as legacy rows are adjudicated (several
# older mirror rows are known-stale vs the live snapshot — e.g. the deepseek
# and qwen3-max rows — and enter enforcement only once their discrepancies
# are resolved, not silently overwritten).
#
# Deliberately NOT ``tuple(CATALOG)``: the frontier-model-refresh (2026-09-04)
# added thirteen new rows whose runtime-table mirrors (pdb/billing/metering/
# provider_config/model_selector/openrouter fallback) have not been migrated
# yet — that migration is later-task scope. Auto-enforcing them here would
# fail the mirror-consistency tests below for rows that were never meant to
# be wired yet. ``qwen3.8-max`` is dropped (not replaced by
# ``qwen3.8-2.4t-a95b``) for the same reason: its canonical/direct id changed,
# and the mirror tables still key on the old spelling. It is resolvable via
# the ``aliases`` tuple on the new ``qwen3.8-2.4t-a95b`` row (``by_any_id``),
# but not yet enforced under the new id; the two runtime modules that used
# to subscript ``CATALOG["qwen3.8-max"]`` directly were updated to the new
# canonical id instead (frontier-model-refresh controller ruling 2 — no
# duplicate ``CATALOG`` keys).
ENFORCED_MODELS: tuple[str, ...] = (
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "gpt-5.6-sol",
    "gpt-5.5",
    "grok-4.5",
    "grok-4.3",
    "sonar-reasoning-pro",
    "command-a",
    "jamba-large-1.7",
    "qwen3.7-max",
    "kimi-k3",
    "kimi-k2.7-code",
)

_ID_INDEX: dict[str, ModelSpec] = {}
for _spec in CATALOG.values():
    for _mid in _spec.all_ids():
        _ID_INDEX[_mid] = _spec


def by_any_id(model_id: str) -> ModelSpec | None:
    """Resolve any known spelling (canonical/direct/openrouter/alias)."""
    return _ID_INDEX.get(str(model_id).strip())


def spec_or_none(model_id: str | None) -> ModelSpec | None:
    """Like ``by_any_id`` but tolerates ``None``/falsy input (convenience for
    optional model-id fields threaded through call sites)."""
    if not model_id:
        return None
    return by_any_id(str(model_id))


def _is_frontier_candidate(spec: ModelSpec) -> bool:
    return bool(spec.family) and spec.tier == "flagship" and not spec.retired


def frontier_for(family: str) -> ModelSpec:
    """Newest non-retired, flagship-tier catalog row for ``family`` (a
    pretraining lineage, see ``ModelSpec.family``). Only ``tier="flagship"``
    rows are considered, so a same-family "value"/"fallback"/"code" row
    released later than the flagship never displaces it. Raises if the
    family has no active flagship row."""
    rows = [s for s in CATALOG.values() if s.family == family and _is_frontier_candidate(s)]
    if not rows:
        raise KeyError(f"no active flagship catalog row for family {family!r}")
    return max(rows, key=lambda s: (s.release_date, s.canonical_id))


FRONTIER: dict[str, str] = {
    fam: frontier_for(fam).canonical_id
    for fam in sorted({s.family for s in CATALOG.values() if _is_frontier_candidate(s)})
}


def snapshot_path() -> Path:
    return Path(__file__).resolve().parent / "catalog_snapshot.json"


def load_snapshot() -> dict[str, dict[str, float | int | str]]:
    """Load the committed live-catalog capture (offline; no network)."""
    return json.loads(snapshot_path().read_text(encoding="utf-8"))["models"]
