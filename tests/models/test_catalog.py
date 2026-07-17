"""Catalog invariants + mirror-consistency enforcement (phase 1).

The teeth of the canonical catalog: every runtime table that carries a row
for an ENFORCED model must agree with ``aragora.models.CATALOG``. These
tests are what previously took adversarial review rounds to discover by
hand — #9073/#9075 found eleven drifting mirrors and three live provider
reprices across their review cycles.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from aragora.models import CATALOG, ENFORCED_MODELS, by_any_id, load_snapshot

# ---------------------------------------------------------------------------
# Catalog internal invariants
# ---------------------------------------------------------------------------


def test_catalog_ids_are_unique_across_all_spellings() -> None:
    seen: dict[str, str] = {}
    for spec in CATALOG.values():
        for mid in spec.all_ids():
            assert mid not in seen or seen[mid] == spec.canonical_id, (
                f"id {mid!r} claimed by both {seen[mid]} and {spec.canonical_id}"
            )
            seen[mid] = spec.canonical_id


def test_catalog_prices_positive_and_output_gte_input() -> None:
    for spec in CATALOG.values():
        assert spec.input_per_mtok > 0
        assert spec.output_per_mtok >= spec.input_per_mtok


def test_soak_dates_only_on_recent_releases() -> None:
    for spec in CATALOG.values():
        if spec.soak_until is not None:
            assert spec.soak_until > spec.release_date
            assert (spec.soak_until - spec.release_date).days == 14


def test_by_any_id_resolves_every_spelling() -> None:
    for spec in CATALOG.values():
        for mid in spec.all_ids():
            assert by_any_id(mid) is spec
    assert by_any_id("no-such-model") is None


# ---------------------------------------------------------------------------
# Snapshot consistency (offline; required CI never touches the network)
# ---------------------------------------------------------------------------


def test_catalog_matches_committed_snapshot() -> None:
    snapshot = load_snapshot()
    for spec in CATALOG.values():
        row = snapshot.get(spec.openrouter_id)
        assert row is not None, f"snapshot missing {spec.openrouter_id}"
        assert float(row["input_per_mtok"]) == pytest.approx(spec.input_per_mtok), (
            f"{spec.canonical_id}: catalog input {spec.input_per_mtok} != "
            f"snapshot {row['input_per_mtok']} — refresh the snapshot or fix the catalog"
        )
        assert float(row["output_per_mtok"]) == pytest.approx(spec.output_per_mtok)


# ---------------------------------------------------------------------------
# Mirror-consistency enforcement
# ---------------------------------------------------------------------------


def _approx_pair(actual_in: float, actual_out: float, spec) -> None:
    assert float(actual_in) == pytest.approx(spec.input_per_mtok), (
        f"{spec.canonical_id}: input {actual_in} != catalog {spec.input_per_mtok}"
    )
    assert float(actual_out) == pytest.approx(spec.output_per_mtok), (
        f"{spec.canonical_id}: output {actual_out} != catalog {spec.output_per_mtok}"
    )


@pytest.mark.parametrize("canonical_id", ENFORCED_MODELS)
def test_pdb_price_table_matches_catalog(canonical_id: str) -> None:
    from aragora.pdb.real_invoker import _PRICE_PER_MTOK

    spec = CATALOG[canonical_id]
    for mid in (spec.direct_id, spec.canonical_id):
        if mid in _PRICE_PER_MTOK:
            _approx_pair(*_PRICE_PER_MTOK[mid], spec)
            break
    else:
        pytest.fail(f"pdb _PRICE_PER_MTOK has no row for {canonical_id}")


@pytest.mark.parametrize("canonical_id", ENFORCED_MODELS)
def test_billing_usage_matches_catalog(canonical_id: str) -> None:
    from aragora.billing.usage import PROVIDER_PRICING

    spec = CATALOG[canonical_id]
    checked = False
    for provider_key, table in PROVIDER_PRICING.items():
        for mid in spec.all_ids():
            if mid in table and f"{mid}-output" in table:
                _approx_pair(table[mid], table[f"{mid}-output"], spec)
                checked = True
    assert checked, f"billing usage.PROVIDER_PRICING has no row for {canonical_id}"


@pytest.mark.parametrize("canonical_id", ENFORCED_MODELS)
def test_metering_models_matches_catalog(canonical_id: str) -> None:
    from aragora.services.metering_models import MODEL_PRICING

    spec = CATALOG[canonical_id]
    checked = False
    for table in MODEL_PRICING.values():
        for mid in spec.all_ids():
            if mid in table and f"{mid}-output" in table:
                _approx_pair(table[mid], table[f"{mid}-output"], spec)
                checked = True
    assert checked, (
        f"metering MODEL_PRICING has no row for {canonical_id} — "
        "UsageMeter._calculate_token_cost falls back to provider/default rates "
        "for missing rows, silently mis-billing live usage; add the row to "
        "aragora/services/metering_models.py from the catalog"
    )


@pytest.mark.parametrize("canonical_id", ENFORCED_MODELS)
def test_debate_costs_matches_catalog(canonical_id: str) -> None:
    from aragora.billing.debate_costs import DEFAULT_PROVIDER_RATES

    spec = CATALOG[canonical_id]
    checked = False
    for table in DEFAULT_PROVIDER_RATES.values():
        for mid in spec.all_ids():
            pair = table.get(mid)
            if isinstance(pair, tuple) and len(pair) == 2:
                _approx_pair(pair[0], pair[1], spec)
                checked = True
    if not checked:
        pytest.skip(f"debate_costs has no row for {canonical_id} (presence not yet required)")


@pytest.mark.parametrize("canonical_id", ENFORCED_MODELS)
def test_provider_config_matches_catalog(canonical_id: str) -> None:
    from aragora.routing.provider_config import PROVIDER_PRICING as ROUTING_PRICING

    spec = CATALOG[canonical_id]
    checked = False
    for mid in spec.all_ids():
        row = ROUTING_PRICING.get(mid)
        if row is not None:
            _approx_pair(row.input_cost_per_1k * 1000, row.output_cost_per_1k * 1000, spec)
            checked = True
    if not checked:
        pytest.skip(f"provider_config has no row for {canonical_id} (presence not yet required)")


def test_model_selector_profiles_match_catalog() -> None:
    """Every model_selector profile whose model_id resolves to an ENFORCED
    catalog spec must carry the catalog's pricing (per-1k mirror of the
    per-MTok catalog values)."""
    from aragora.agents.model_selector import MODEL_PROFILES

    checked: set[str] = set()
    for profile in MODEL_PROFILES.values():
        spec = by_any_id(profile.model_id)
        if spec is None or spec.canonical_id not in ENFORCED_MODELS:
            continue
        _approx_pair(profile.cost_input_per_1k * 1000, profile.cost_output_per_1k * 1000, spec)
        checked.add(spec.canonical_id)
    # Anchors: these enforced models are known to have selector profiles
    # (gpt4 -> gpt-5.5, claude-opus -> claude-opus-4-8). Deleting or renaming
    # those rows off-catalog fails here instead of passing vacuously.
    assert {"gpt-5.5", "claude-opus-4-8"} <= checked, (
        f"model_selector enforcement anchors missing: checked only {sorted(checked)}"
    )


# Fallback targets that intentionally have no catalog spec yet. Each entry
# needs adjudication before cataloging (deepseek-v4-pro's mirror rows are
# known-stale vs the live catalog — see MODEL_CATALOG.md "Known-stale legacy
# rows"). Remove an entry once its model is cataloged; a stale entry fails
# the hygiene assertion below.
_UNCATALOGED_FALLBACK_TARGETS = {"deepseek/deepseek-v4-pro"}


def test_fallback_targets_resolve_to_catalog_specs() -> None:
    """Every OpenRouter fallback TARGET must resolve to a catalog spec (a
    dead/renamed slug fails — the dead-slug class of #9073), except for the
    explicit allowlist above; and a target resolving to an ENFORCED spec must
    be exactly that spec's ``openrouter_id``."""
    from aragora.agents.api_agents.openrouter import OPENROUTER_FALLBACK_MODELS

    targets = set(OPENROUTER_FALLBACK_MODELS.values())
    # Allowlist hygiene: every allowlisted slug must still be a live target.
    assert _UNCATALOGED_FALLBACK_TARGETS <= targets, (
        "stale _UNCATALOGED_FALLBACK_TARGETS entries: "
        f"{sorted(_UNCATALOGED_FALLBACK_TARGETS - targets)}"
    )

    checked_enforced: set[str] = set()
    for target in sorted(targets):
        spec = by_any_id(target)
        if target in _UNCATALOGED_FALLBACK_TARGETS:
            assert spec is None, (
                f"{target!r} is now cataloged — remove it from _UNCATALOGED_FALLBACK_TARGETS"
            )
            continue
        assert spec is not None, (
            f"fallback target {target!r} resolves to no catalog spec — dead or "
            "uncataloged slug; catalog it (with live verification) or add it to "
            "_UNCATALOGED_FALLBACK_TARGETS with adjudication"
        )
        if spec.canonical_id in ENFORCED_MODELS:
            assert target == spec.openrouter_id, (
                f"fallback target {target!r} resolves to enforced "
                f"{spec.canonical_id} but is not its OpenRouter slug "
                f"{spec.openrouter_id!r}"
            )
            checked_enforced.add(spec.canonical_id)
    # Falsifiability anchor: the map is known to target gpt-5.5 and opus-4.8.
    assert checked_enforced, "no enforced fallback targets were checked"
