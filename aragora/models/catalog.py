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
from datetime import date
from pathlib import Path

__all__ = [
    "CATALOG",
    "ENFORCED_MODELS",
    "ModelSpec",
    "by_any_id",
    "load_snapshot",
    "snapshot_path",
]


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

    def all_ids(self) -> tuple[str, ...]:
        return (self.canonical_id, self.direct_id, self.openrouter_id, *self.aliases)


# Prices are USD per 1M tokens, captured from the live OpenRouter catalog on
# 2026-07-16 (see catalog_snapshot.json for the raw capture). Direct-provider
# ids verified against provider model lists where noted on the source PRs.
CATALOG: dict[str, ModelSpec] = {
    spec.canonical_id: spec
    for spec in (
        ModelSpec(
            canonical_id="claude-fable-5",
            provider="anthropic",
            direct_id="claude-fable-5",
            openrouter_id="anthropic/claude-fable-5",
            input_per_mtok=10.00,
            output_per_mtok=50.00,
            context_window=1_000_000,
            max_output_tokens=128_000,
            release_date=date(2026, 6, 20),
        ),
        ModelSpec(
            canonical_id="claude-opus-4-8",
            provider="anthropic",
            direct_id="claude-opus-4-8",
            openrouter_id="anthropic/claude-opus-4.8",
            input_per_mtok=5.00,
            output_per_mtok=25.00,
            context_window=1_000_000,
            max_output_tokens=128_000,
            release_date=date(2026, 2, 10),
            aliases=("claude-opus-4.8", "anthropic/claude-opus-4-8"),
        ),
        ModelSpec(
            canonical_id="gpt-5.6-sol",
            provider="openai",
            direct_id="gpt-5.6-sol",
            openrouter_id="openai/gpt-5.6-sol",
            input_per_mtok=5.00,
            output_per_mtok=30.00,
            context_window=1_050_000,
            max_output_tokens=128_000,
            release_date=date(2026, 7, 9),
            soak_until=date(2026, 7, 23),
        ),
        ModelSpec(
            canonical_id="gpt-5.5",
            provider="openai",
            direct_id="gpt-5.5",
            openrouter_id="openai/gpt-5.5",
            # Provider repriced from 2.50/10.00 on ~2026-07-14 (caught live
            # during the #9073/#9075 reviews).
            input_per_mtok=5.00,
            output_per_mtok=30.00,
            context_window=1_050_000,
            max_output_tokens=128_000,
            release_date=date(2025, 11, 1),
        ),
        ModelSpec(
            canonical_id="grok-4.5",
            provider="xai",
            direct_id="grok-4.5",
            openrouter_id="x-ai/grok-4.5",
            input_per_mtok=2.00,
            output_per_mtok=6.00,
            context_window=500_000,
            max_output_tokens=64_000,
            release_date=date(2026, 7, 8),
            soak_until=date(2026, 7, 22),
        ),
        ModelSpec(
            canonical_id="grok-4.3",
            provider="xai",
            direct_id="grok-4.3",
            openrouter_id="x-ai/grok-4.3",
            input_per_mtok=1.25,
            output_per_mtok=2.50,
            context_window=1_000_000,
            max_output_tokens=64_000,
            release_date=date(2026, 4, 1),
        ),
        ModelSpec(
            canonical_id="qwen3.7-max",
            provider="alibaba",
            direct_id="qwen3.7-max",
            openrouter_id="qwen/qwen3.7-max",
            # Repriced from 1.25/3.75 between 2026-07-10 and 2026-07-16
            # (caught live during the #9073 drain).
            input_per_mtok=1.475,
            output_per_mtok=4.425,
            context_window=1_000_000,
            max_output_tokens=32_768,
            release_date=date(2026, 6, 1),
        ),
        ModelSpec(
            canonical_id="kimi-k2.7-code",
            provider="moonshot",
            direct_id="kimi-k2.7-code",
            openrouter_id="moonshotai/kimi-k2.7-code",
            # Prompt rate corrected 0.72 -> 0.75 per live catalog (#9073).
            input_per_mtok=0.75,
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
    )
}

# Models whose runtime-table rows are ENFORCED against this catalog by
# tests/models/test_catalog.py. Grows as legacy rows are adjudicated (several
# older mirror rows are known-stale vs the live snapshot — e.g. the deepseek
# and qwen3-max rows — and enter enforcement only once their discrepancies
# are resolved, not silently overwritten).
ENFORCED_MODELS: tuple[str, ...] = tuple(CATALOG)

_ID_INDEX: dict[str, ModelSpec] = {}
for _spec in CATALOG.values():
    for _mid in _spec.all_ids():
        _ID_INDEX[_mid] = _spec


def by_any_id(model_id: str) -> ModelSpec | None:
    """Resolve any known spelling (canonical/direct/openrouter/alias)."""
    return _ID_INDEX.get(str(model_id).strip())


def snapshot_path() -> Path:
    return Path(__file__).resolve().parent / "catalog_snapshot.json"


def load_snapshot() -> dict[str, dict[str, float | int | str]]:
    """Load the committed live-catalog capture (offline; no network)."""
    return json.loads(snapshot_path().read_text(encoding="utf-8"))["models"]
