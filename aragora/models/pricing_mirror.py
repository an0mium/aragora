"""Generate the legacy pricing-table shapes from the catalog so a price
lives in one place.

Phase 1 (``aragora/models/catalog.py``) made ``CATALOG`` the canonical
economics record but deliberately left the five runtime pricing tables
(``aragora.billing.usage.PROVIDER_PRICING``, ``aragora.pdb.real_invoker
._PRICE_PER_MTOK``, ``aragora.billing.debate_costs.DEFAULT_PROVIDER_RATES``,
``aragora.services.metering_models.MODEL_PRICING`` and
``aragora.routing.provider_config.PROVIDER_PRICING``) hand-maintained, each
enforced only for the ``ENFORCED_MODELS`` subset. This module is phase 2:
it derives every one of those table shapes from ``CATALOG`` so a price
change only ever needs to happen in one place.

Each consumer module keeps its hand-written dict under a ``_LEGACY_*`` name
(receipts and env overrides pinned to an old spelling must keep resolving)
and publishes ``{**_LEGACY_*, **generated}`` — the generated row wins on a
key collision, since the catalog is the more recently verified source.

Retired catalog rows are still emitted (see ``_active`` below): an old
receipt referencing a retired model id must still price, so retirement
only affects *adoption* surfaces (routing enumeration, frontier picks),
never pricing lookup.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from aragora.models.catalog import CATALOG, ModelSpec

if TYPE_CHECKING:
    from datetime import date

    from aragora.routing.provider_config import ProviderPricing

__all__ = [
    "debate_cost_rows",
    "metering_rows",
    "pdb_rows",
    "provider_config_rows",
    "usage_rows",
]


def _dec(x: float) -> Decimal:
    """Render a catalog float as the ``Decimal`` shape the legacy tables
    use: two decimal places for round dollar-and-cents rates (``10.0`` ->
    ``Decimal("10.00")``), four for rates with finer-grained cents (e.g.
    the qwen3.7-max reprice ``1.475``, the deepseek-v4-pro-0813 live
    capture ``1.1207``)."""
    return Decimal(f"{x:.4f}").normalize() if x != round(x, 2) else Decimal(f"{x:.2f}")


def _active() -> list[ModelSpec]:
    """Every catalog row, retired included.

    Retired rows stay priced: old receipts must still resolve to a real
    rate, even though retired models are no longer adopted by routing/
    frontier-selection surfaces (those checks live on ``ModelSpec.retired``
    itself, not here)."""
    return [s for s in CATALOG.values()]


def usage_rows() -> dict[str, dict[str, Decimal]]:
    """Shape of ``aragora.billing.usage.PROVIDER_PRICING``: provider ->
    {id: input_rate, f"{id}-output": output_rate}, one entry per spelling
    in ``ModelSpec.all_ids()`` so canonical/direct/openrouter/alias ids all
    resolve."""
    out: dict[str, dict[str, Decimal]] = {}
    for s in _active():
        prov = out.setdefault(s.provider, {})
        for spelling in s.all_ids():
            prov[spelling] = _dec(s.input_per_mtok)
            prov[f"{spelling}-output"] = _dec(s.output_per_mtok)
    return out


def pdb_rows() -> dict[str, tuple[float, float]]:
    """Shape of ``aragora.pdb.real_invoker._PRICE_PER_MTOK``: spelling ->
    (input, output) rate, flat across providers."""
    return {sp: (s.input_per_mtok, s.output_per_mtok) for s in _active() for sp in s.all_ids()}


def debate_cost_rows() -> dict[str, dict[str, tuple[Decimal, Decimal]]]:
    """Shape of ``aragora.billing.debate_costs.DEFAULT_PROVIDER_RATES``:
    provider -> {spelling: (input, output)}."""
    out: dict[str, dict[str, tuple[Decimal, Decimal]]] = {}
    for s in _active():
        prov = out.setdefault(s.provider, {})
        for sp in s.all_ids():
            prov[sp] = (_dec(s.input_per_mtok), _dec(s.output_per_mtok))
    return out


def metering_rows() -> dict[str, dict[str, Decimal]]:
    """Shape of ``aragora.services.metering_models.MODEL_PRICING``: same
    provider -> {id: input, f"{id}-output": output} shape as
    ``usage_rows()`` (verified against the hand-written dict before writing
    this — the two tables are documented as aligned)."""
    return usage_rows()


def provider_config_rows(
    today: "date | None" = None,
    *,
    catalog: "dict[str, ModelSpec] | None" = None,
) -> "dict[str, ProviderPricing]":
    """Shape of ``aragora.routing.provider_config.PROVIDER_PRICING``'s
    catalog projection: CANONICAL id only (never alias/direct/openrouter
    spellings) so enumeration consumers never see one model occupy several
    candidate slots, and never a RETIRED row. This is the single generator
    ``aragora.routing.provider_config._catalog_projection`` calls; alias
    spellings still price via the ``by_any_id`` fallback in
    ``get_estimated_cost``, unaffected by this function.

    The filter is retirement, not soak (frontier-model-refresh final review
    #7). It used to be the other way round, which inverted the intended
    candidate set: the routing roster offered `gpt-5.5` and `grok-4.5`
    (retired, dead on the wire) while withholding `gpt-6-astra`,
    `gemini-3.8-flash` and `muse-spark-1.3` (the current defaults, merely
    soaking). Soak gating still applies to routing SELECTION, where it
    belongs, via ``provider_router._is_under_soak``; keeping it here also
    made a module-level constant calendar-dependent -- the table would
    silently gain rows on 2026-09-16/17 -- which is a latent flake for any
    membership or count assertion.

    ``today`` is retained for call-site compatibility and is unused.

    ``catalog`` defaults to ``aragora.models.catalog.CATALOG`` but accepts
    an override so ``provider_config._catalog_projection`` can pass its own
    module-level ``CATALOG`` name through unchanged -- tests monkeypatch
    that module attribute directly to inject synthetic rows, and this
    function must observe the patched value rather than re-importing the
    real catalog itself.

    ``ProviderPricing`` is imported inside the function (not at module
    scope) because ``provider_config.py`` imports this module: a top-level
    import here would cycle.
    """
    from aragora.routing.provider_config import ProviderPricing

    rows = CATALOG if catalog is None else catalog
    return {
        s.canonical_id: ProviderPricing(
            provider_name=s.provider,
            model_name=s.direct_id,
            input_cost_per_1k=s.input_per_mtok / 1000.0,
            output_cost_per_1k=s.output_per_mtok / 1000.0,
            context_window=s.context_window,
        )
        for s in rows.values()
        if not s.retired
    }
