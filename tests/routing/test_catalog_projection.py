"""Phase-2 catalog projection into the routing pricing table.

Regression for the founder-identified routing bug: the Pareto /
decision-stakes router saw $0 estimated cost for every frontier pin because
the hand-maintained PROVIDER_PRICING rows had gone stale. Cataloged models
now project from aragora.models.CATALOG (single source).

Only CANONICAL ids are projected into the enumerated table: enumeration
consumers (get_available_models, get_models_within_budget,
ProviderRouter._details_from_pricing) treat table keys as distinct candidate
models, so alias spellings must not occupy extra candidate slots. Aliases
price through the by_any_id fallback in get_estimated_cost instead.

Models still inside their catalog soak window (ModelSpec.is_under_soak) are
not projected at all — the enumerated table is an adoption surface — but
their ids keep pricing through the same fallback.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from aragora.models import CATALOG, ModelSpec, by_any_id
from aragora.routing import provider_config
from aragora.routing.provider_config import (
    PROVIDER_PRICING,
    ProviderPricing,
    _apply_catalog_projection,
    get_available_models,
    get_estimated_cost,
    get_models_within_budget,
)
from aragora.routing.provider_router import ProviderRouter


class TestCatalogProjection:
    def test_adoptable_canonical_ids_have_a_pricing_row(self) -> None:
        for spec in CATALOG.values():
            if spec.is_under_soak():
                assert spec.canonical_id not in PROVIDER_PRICING, (
                    f"{spec.canonical_id} is under soak and must not be enumerated"
                )
            else:
                assert spec.canonical_id in PROVIDER_PRICING, (
                    f"no projected row for {spec.canonical_id}"
                )

    def test_projected_rows_match_catalog_rates(self) -> None:
        for spec in CATALOG.values():
            if spec.is_under_soak():
                continue
            row = PROVIDER_PRICING[spec.canonical_id]
            assert row.input_cost_per_1k * 1000 == pytest.approx(spec.input_per_mtok)
            assert row.output_cost_per_1k * 1000 == pytest.approx(spec.output_per_mtok)
            assert row.context_window == spec.context_window

    def test_no_frontier_pin_estimates_zero(self) -> None:
        """THE bug: every platform default routed at $0 before the projection.

        Covers every spelling — canonical via the table, alias/openrouter
        spellings AND every id of under-soak models via the by_any_id
        fallback in get_estimated_cost.
        """
        for spec in CATALOG.values():
            for model_id in spec.all_ids():
                cost = get_estimated_cost(model_id, 1_000_000, 1_000_000)
                assert cost > 0.0, f"{model_id} still estimates $0"

    def test_estimated_cost_math_matches_catalog(self) -> None:
        spec = CATALOG["claude-fable-5"]
        cost = get_estimated_cost("claude-fable-5", 1_000_000, 1_000_000)
        assert cost == pytest.approx(spec.input_per_mtok + spec.output_per_mtok)

    def test_unknown_model_still_returns_zero(self) -> None:
        assert get_estimated_cost("no-such-model", 1_000_000, 1_000_000) == 0.0

    def test_legacy_hand_rows_survive_projection(self) -> None:
        """Non-catalog legacy models keep their hand-maintained rows."""
        assert "claude-opus-4" in PROVIDER_PRICING
        assert get_estimated_cost("claude-opus-4", 1_000_000, 1_000_000) > 0.0


class TestAliasesDoNotInflateEnumeration:
    """One catalog model = at most ONE candidate slot, regardless of aliases.

    Regression for the #9364 quorum P2: projecting every all_ids() spelling
    into PROVIDER_PRICING made enumeration consumers count one model 2-4
    times under different spellings, silently defeating team heterogeneity.
    """

    def test_catalog_model_occupies_at_most_one_available_slot(self) -> None:
        available = set(get_available_models())
        for spec in CATALOG.values():
            spellings_in_table = set(spec.all_ids()) & available
            expected = set() if spec.is_under_soak() else {spec.canonical_id}
            assert spellings_in_table == expected, (
                f"{spec.canonical_id} occupies {sorted(spellings_in_table)} — "
                "aliases must not be enumerated as distinct candidates"
            )

    def test_budget_enumeration_has_no_alias_duplicates(self) -> None:
        # kimi-k2.7-code has 3 spellings; a generous budget admits them all
        # if they are (wrongly) projected as separate rows.
        affordable = get_models_within_budget(budget_per_debate=1_000.0)
        for spec in CATALOG.values():
            spellings = [m for m in affordable if m in set(spec.all_ids())]
            expected = [] if spec.is_under_soak() else [spec.canonical_id]
            assert spellings == expected

    def test_details_from_pricing_returns_distinct_models(self) -> None:
        """The reachable no-metrics fallback must not fill multiple agent
        slots with the same catalog model under different spellings."""
        router = ProviderRouter()  # empty metrics store -> pricing fallback
        details = router.select_providers_with_details(num_agents=len(PROVIDER_PRICING))
        selected = [d["provider"] for d in details]
        assert len(selected) == len(set(selected))
        for spec in CATALOG.values():
            occupied = [m for m in selected if m in set(spec.all_ids())]
            assert len(occupied) <= 1, (
                f"{spec.canonical_id} fills {len(occupied)} candidate slots: {occupied}"
            )

    def test_alias_spellings_still_price_via_fallback(self) -> None:
        """Aliases are not enumerated, but every spelling must still cost
        exactly what the canonical id costs (by_any_id fallback path)."""
        for spec in CATALOG.values():
            canonical_cost = get_estimated_cost(spec.canonical_id, 2000, 1000)
            for model_id in spec.all_ids():
                assert get_estimated_cost(model_id, 2000, 1000) == pytest.approx(canonical_cost)

    def test_no_table_key_is_an_alias_of_a_catalog_model(self) -> None:
        """Round-2 residual (#9364, openai): a pre-existing hand row keyed by
        an alias spelling of a catalog model would survive a plain update()
        and enumerate that model in a second slot. The applied table must
        contain no key that by_any_id resolves to a DIFFERENT canonical id.

        (Verified while fixing: the review's `deepseek-r1` example does not
        actually resolve — deepseek is not cataloged — so no live row is
        affected today; this pins the invariant against catalog growth.)
        """
        for key in PROVIDER_PRICING:
            spec = by_any_id(key)
            assert spec is None or spec.canonical_id == key, (
                f"table key {key!r} is an alias of catalog model "
                f"{spec.canonical_id!r} and would occupy a duplicate slot"
            )

    def test_legacy_alias_keyed_hand_row_is_filtered_but_still_prices(self) -> None:
        """Simulate the residual directly: a legacy hand row keyed by a real
        alias spelling must be dropped by _apply_catalog_projection, while
        genuinely non-catalog hand rows (deepseek-r1) are kept, and cost
        lookup on the dropped legacy key resolves to the canonical price."""
        kimi = CATALOG["kimi-k2.7-code"]
        legacy_alias_key = kimi.openrouter_id  # "moonshotai/kimi-k2.7-code"
        table = {
            legacy_alias_key: ProviderPricing(
                provider_name="moonshot",
                model_name="kimi-k2.7-code",
                input_cost_per_1k=0.00042,  # stale hand price
                output_cost_per_1k=0.00099,
                context_window=128_000,
            ),
            "deepseek-r1": PROVIDER_PRICING["deepseek-r1"],
        }
        _apply_catalog_projection(table)

        # The alias-keyed hand row no longer occupies a second slot...
        assert legacy_alias_key not in table
        assert kimi.canonical_id in table
        # ...non-catalog hand rows survive untouched...
        assert "deepseek-r1" in table
        # ...and the legacy spelling still prices at the canonical catalog
        # rate through the by_any_id fallback in get_estimated_cost.
        canonical_cost = get_estimated_cost(kimi.canonical_id, 2000, 1000)
        assert get_estimated_cost(legacy_alias_key, 2000, 1000) == pytest.approx(canonical_cost)
        assert canonical_cost > 0.0


class TestUnderSoakModelsNotEnumerated:
    """Round-3 residual (#9364, openai): the enumerated table is an adoption
    surface, so catalog models still inside their soak window (must-not-adopt
    before ModelSpec.soak_until) must not appear in get_available_models(),
    get_models_within_budget(), or the no-metrics pricing fallback — while
    cost lookup for their ids keeps working via the by_any_id fallback,
    which has no soak gating.
    """

    def test_no_enumerated_key_is_under_soak_today(self) -> None:
        """Invariant on the real applied table."""
        for key in get_available_models():
            spec = by_any_id(key)
            assert spec is None or not spec.is_under_soak(), (
                f"{key!r} is enumerated but {spec.canonical_id} is under soak "
                f"until {spec.soak_until}"
            )

    def test_details_from_pricing_never_offers_under_soak_models(self) -> None:
        router = ProviderRouter()  # empty metrics store -> pricing fallback
        details = router.select_providers_with_details(num_agents=len(PROVIDER_PRICING) + 5)
        for entry in details:
            spec = by_any_id(entry["provider"])
            assert spec is None or not spec.is_under_soak()

    def test_under_soak_model_excluded_but_all_ids_still_price(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Synthetic spec with a stable future soak_until (no wall-clock
        dependence on the real catalog's dates): excluded from the applied
        table, yet every spelling prices at the catalog rate."""
        soaking = ModelSpec(
            canonical_id="soak-model-x",
            provider="testprov",
            direct_id="soak-model-x",
            openrouter_id="testprov/soak-model-x",
            input_per_mtok=4.00,
            output_per_mtok=20.00,
            context_window=100_000,
            max_output_tokens=8_192,
            release_date=date.today() - timedelta(days=1),
            soak_until=date.today() + timedelta(days=13),
            aliases=("soak-alias-x",),
        )
        adoptable = ModelSpec(
            canonical_id="ready-model-y",
            provider="testprov",
            direct_id="ready-model-y",
            openrouter_id="testprov/ready-model-y",
            input_per_mtok=1.00,
            output_per_mtok=2.00,
            context_window=100_000,
            max_output_tokens=8_192,
            release_date=date.today() - timedelta(days=60),
        )
        fake_catalog = {s.canonical_id: s for s in (soaking, adoptable)}

        def fake_by_any_id(model_id: str) -> ModelSpec | None:
            for s in fake_catalog.values():
                if str(model_id).strip() in s.all_ids():
                    return s
            return None

        monkeypatch.setattr(provider_config, "CATALOG", fake_catalog)
        monkeypatch.setattr(provider_config, "by_any_id", fake_by_any_id)

        table: dict[str, ProviderPricing] = {}
        _apply_catalog_projection(table)

        # Under-soak model has NO enumerated row; adoptable one does.
        assert "ready-model-y" in table
        assert not any(key in table for key in soaking.all_ids())

        # Cost lookup still resolves every under-soak spelling at the
        # catalog rate through the by_any_id fallback.
        monkeypatch.setattr(provider_config, "PROVIDER_PRICING", table)
        for model_id in soaking.all_ids():
            cost = provider_config.get_estimated_cost(model_id, 1_000_000, 1_000_000)
            assert cost == pytest.approx(24.00), f"{model_id} lost cost lookup under soak"

    def test_soak_expiry_admits_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once soak_until passes, the same spec projects normally."""
        expired = ModelSpec(
            canonical_id="soaked-model-z",
            provider="testprov",
            direct_id="soaked-model-z",
            openrouter_id="testprov/soaked-model-z",
            input_per_mtok=3.00,
            output_per_mtok=9.00,
            context_window=100_000,
            max_output_tokens=8_192,
            release_date=date.today() - timedelta(days=15),
            soak_until=date.today(),  # must-not-adopt BEFORE today: now OK
        )
        monkeypatch.setattr(provider_config, "CATALOG", {expired.canonical_id: expired})
        monkeypatch.setattr(
            provider_config,
            "by_any_id",
            lambda mid: expired if str(mid).strip() in expired.all_ids() else None,
        )
        table: dict[str, ProviderPricing] = {}
        _apply_catalog_projection(table)
        assert "soaked-model-z" in table
