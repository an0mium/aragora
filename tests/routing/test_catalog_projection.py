"""Phase-2 catalog projection into the routing pricing table.

Regression for the founder-identified routing bug: the Pareto /
decision-stakes router saw $0 estimated cost for every frontier pin because
the hand-maintained PROVIDER_PRICING rows had gone stale. Cataloged models
now project from aragora.models.CATALOG (single source).
"""

from __future__ import annotations

import pytest

from aragora.models import CATALOG
from aragora.routing.provider_config import PROVIDER_PRICING, get_estimated_cost


class TestCatalogProjection:
    def test_every_catalog_spelling_has_a_pricing_row(self) -> None:
        for spec in CATALOG.values():
            for model_id in spec.all_ids():
                assert model_id in PROVIDER_PRICING, f"no projected row for {model_id}"

    def test_projected_rows_match_catalog_rates(self) -> None:
        for spec in CATALOG.values():
            row = PROVIDER_PRICING[spec.canonical_id]
            assert row.input_cost_per_1k * 1000 == pytest.approx(spec.input_per_mtok)
            assert row.output_cost_per_1k * 1000 == pytest.approx(spec.output_per_mtok)
            assert row.context_window == spec.context_window

    def test_no_frontier_pin_estimates_zero(self) -> None:
        """THE bug: every platform default routed at $0 before the projection."""
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
