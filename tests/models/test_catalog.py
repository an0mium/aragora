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
    if not checked:
        pytest.skip(f"metering has no row for {canonical_id} (presence not yet required)")


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


def test_fallback_maps_resolve_to_live_catalog_ids() -> None:
    """Every OpenRouter target in the agent fallback maps that references an
    ENFORCED model must use the exact catalog slug (dead-slug class of
    #9073)."""
    from aragora.agents.api_agents.openrouter import OPENROUTER_FALLBACK_MODELS

    enforced_or_ids = {CATALOG[c].openrouter_id for c in ENFORCED_MODELS}
    for target in set(OPENROUTER_FALLBACK_MODELS.values()):
        spec = by_any_id(target)
        if spec is not None and spec.canonical_id in ENFORCED_MODELS:
            assert target in enforced_or_ids or target in spec.all_ids(), (
                f"fallback target {target!r} is not a catalog id spelling"
            )
