"""Tests for :mod:`aragora.models.pricing_mirror`.

Verifies the generated table shapes cover every catalog spelling with the
exact catalog rate, and that the legacy tables each module publishes
(``PROVIDER_PRICING`` / ``_PRICE_PER_MTOK`` / ``DEFAULT_PROVIDER_RATES`` /
``MODEL_PRICING`` / routing ``PROVIDER_PRICING``) contain the mirrored
rows -- the catalog is the single source of truth and the generated row
wins on a key collision with the hand-written legacy dict.
"""

from __future__ import annotations

from decimal import Decimal

from aragora.models import pricing_mirror as pm
from aragora.models.catalog import CATALOG


def test_usage_rows_cover_every_active_row_with_exact_prices() -> None:
    rows = pm.usage_rows()
    spec = CATALOG["gpt-6-astra"]
    assert rows["openai"]["gpt-6-astra"] == Decimal("10.00")
    assert rows["openai"]["gpt-6-astra-output"] == Decimal("50.00")
    for s in CATALOG.values():
        if s.retired:
            continue
        assert rows[s.provider][s.canonical_id] == Decimal(str(s.input_per_mtok))


def test_usage_rows_include_retired_rows_too() -> None:
    """Old receipts referencing a retired model id must still resolve."""
    rows = pm.usage_rows()
    for s in CATALOG.values():
        if not s.retired:
            continue
        assert rows[s.provider][s.canonical_id] == Decimal(str(s.input_per_mtok))
        assert rows[s.provider][f"{s.canonical_id}-output"] == Decimal(str(s.output_per_mtok))


def test_usage_rows_cover_every_spelling_including_aliases() -> None:
    rows = pm.usage_rows()
    for s in CATALOG.values():
        for spelling in s.all_ids():
            assert rows[s.provider][spelling] == Decimal(str(s.input_per_mtok))
            assert rows[s.provider][f"{spelling}-output"] == Decimal(str(s.output_per_mtok))


def test_pdb_rows_cover_every_spelling() -> None:
    rows = pm.pdb_rows()
    for s in CATALOG.values():
        for sp in s.all_ids():
            assert rows[sp] == (s.input_per_mtok, s.output_per_mtok)


def test_debate_cost_rows_cover_every_spelling() -> None:
    rows = pm.debate_cost_rows()
    for s in CATALOG.values():
        for sp in s.all_ids():
            assert rows[s.provider][sp] == (
                Decimal(str(s.input_per_mtok)),
                Decimal(str(s.output_per_mtok)),
            )


def test_metering_rows_matches_usage_rows_shape() -> None:
    """MODEL_PRICING's hand-written value shape (provider -> {id: Decimal,
    f"{id}-output": Decimal}) matches PROVIDER_PRICING's exactly, so the
    generator output must too."""
    assert pm.metering_rows() == pm.usage_rows()


def test_provider_config_rows_keys_only_canonical_ids() -> None:
    """Enumeration consumers must never see an alias/direct/openrouter
    spelling occupy its own candidate slot."""
    rows = pm.provider_config_rows()
    for s in CATALOG.values():
        if s.is_under_soak():
            continue
        assert s.canonical_id in rows
    for key in rows:
        assert key in CATALOG, f"provider_config_rows() key {key!r} is not a canonical catalog id"


def test_provider_config_rows_excludes_under_soak_rows() -> None:
    from datetime import date

    for s in CATALOG.values():
        if s.soak_until is None:
            continue
        rows = pm.provider_config_rows(today=s.soak_until - (s.soak_until - s.release_date) // 2)
        assert s.canonical_id not in rows, (
            f"{s.canonical_id} is under soak and must not be projected"
        )


def test_legacy_tables_contain_mirror_rows() -> None:
    from aragora.billing.usage import PROVIDER_PRICING
    from aragora.pdb.real_invoker import _PRICE_PER_MTOK
    from aragora.billing.debate_costs import DEFAULT_PROVIDER_RATES
    from aragora.routing.provider_config import PROVIDER_PRICING as ROUTING

    assert PROVIDER_PRICING["anthropic"]["claude-fable-5-1"] == Decimal("10.00")
    assert _PRICE_PER_MTOK["claude-fable-5-1"] == (10.00, 50.00)
    assert DEFAULT_PROVIDER_RATES["openai"]["gpt-6-astra"] == (Decimal("10.00"), Decimal("50.00"))
    assert (
        ROUTING["grok-4.6"].input_cost_per_1k == 0.002
        and ROUTING["grok-4.6"].output_cost_per_1k == 0.006
    )


def test_legacy_tables_still_resolve_pre_existing_hand_rows() -> None:
    """The mirror must not delete hand rows for models the catalog doesn't
    know about (receipts pinned to those spellings must keep resolving)."""
    from aragora.billing.usage import PROVIDER_PRICING
    from aragora.pdb.real_invoker import _PRICE_PER_MTOK

    assert PROVIDER_PRICING["openai"]["gpt-4o"] == Decimal("2.50")
    assert _PRICE_PER_MTOK["gpt-4o"] == (2.50, 10.00)
