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
        if s.retired:
            continue
        assert s.canonical_id in rows
    for key in rows:
        assert key in CATALOG, f"provider_config_rows() key {key!r} is not a canonical catalog id"


def test_provider_config_rows_excludes_retired_rows() -> None:
    """The routing roster is a candidate set: a retired row is dead on the
    wire, so offering it is worse than useless."""
    rows = pm.provider_config_rows()
    retired = [s.canonical_id for s in CATALOG.values() if s.retired]
    assert retired, "fixture assumption: the catalog carries at least one retired row"
    for canonical_id in retired:
        assert canonical_id not in rows, f"{canonical_id} is retired and must not be projected"


def test_provider_config_rows_are_not_calendar_dependent() -> None:
    """Soak is NOT a projection filter (final review #7).

    Filtering the enumerated table by soak inverted the candidate set (it
    withheld the current defaults while offering retired ids) and made a
    module-level constant change contents on a wall-clock date, which is a
    latent flake for any membership or count assertion. Soak gating lives on
    the SELECTION path instead (provider_router._is_under_soak).
    """
    from datetime import date

    soaking = [s for s in CATALOG.values() if s.soak_until is not None and not s.retired]
    assert soaking, "fixture assumption: the catalog carries at least one soaking row"

    baseline = set(pm.provider_config_rows())
    for s in soaking:
        assert s.canonical_id in baseline, (
            f"{s.canonical_id} is soaking but active; it must still be a routing candidate"
        )
        mid_soak = s.release_date + (s.soak_until - s.release_date) // 2
        assert set(pm.provider_config_rows(today=mid_soak)) == baseline
    assert set(pm.provider_config_rows(today=date(2099, 1, 1))) == baseline


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
