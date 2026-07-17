"""Provider pricing configuration and cost estimation.

Contains static pricing data for supported AI model providers
and utility functions for estimating debate costs.

Pricing is per 1M tokens (consistent with aragora.billing.usage).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ProviderPricing:
    """Pricing and capabilities for a single provider/model combination."""

    provider_name: str
    model_name: str
    input_cost_per_1k: float  # USD per 1K input tokens
    output_cost_per_1k: float  # USD per 1K output tokens
    context_window: int  # Maximum context window in tokens
    supports_streaming: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            "context_window": self.context_window,
            "supports_streaming": self.supports_streaming,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderPricing:
        """Deserialize from dictionary."""
        return cls(
            provider_name=data["provider_name"],
            model_name=data["model_name"],
            input_cost_per_1k=data["input_cost_per_1k"],
            output_cost_per_1k=data["output_cost_per_1k"],
            context_window=data["context_window"],
            supports_streaming=data.get("supports_streaming", True),
        )


from aragora.models import CATALOG, by_any_id

# Legacy hand-maintained rows (models not yet in the canonical catalog).
# Cataloged models are PROJECTED from aragora.models.CATALOG below — do not
# hand-edit rows for cataloged ids here; edit the catalog instead
# (docs/architecture/MODEL_CATALOG.md).
#
# NOTE for direct importers of this dict: it is refreshed IN PLACE (soak
# windows expire by calendar date) whenever any accessor in this module
# runs on a new date. Enumerate via get_available_models() /
# _current_pricing_table() for guaranteed date-fresh soak gating; a bare
# from-import read sees the last-refreshed snapshot.
PROVIDER_PRICING: dict[str, ProviderPricing] = {
    "claude-opus-4": ProviderPricing(
        provider_name="anthropic",
        model_name="claude-opus-4",
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.025,
        context_window=200_000,
    ),
    "claude-sonnet-4": ProviderPricing(
        provider_name="anthropic",
        model_name="claude-sonnet-4",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        context_window=200_000,
    ),
    "gpt-4o": ProviderPricing(
        provider_name="openai",
        model_name="gpt-4o",
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.010,
        context_window=128_000,
    ),
    "gpt-4o-mini": ProviderPricing(
        provider_name="openai",
        model_name="gpt-4o-mini",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        context_window=128_000,
    ),
    "deepseek-v4-pro": ProviderPricing(
        provider_name="deepseek",
        model_name="deepseek-v4-pro",
        input_cost_per_1k=0.00174,
        output_cost_per_1k=0.00348,
        context_window=1_048_576,
    ),
    "deepseek-r1": ProviderPricing(
        provider_name="deepseek",
        model_name="deepseek-v4-pro",
        input_cost_per_1k=0.00174,
        output_cost_per_1k=0.00348,
        context_window=1_048_576,
    ),
    "deepseek-chat": ProviderPricing(
        provider_name="deepseek",
        model_name="deepseek-chat",
        input_cost_per_1k=0.00028,
        output_cost_per_1k=0.00042,
        context_window=64_000,
    ),
    "mistral-large": ProviderPricing(
        provider_name="mistral",
        model_name="mistral-large",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.006,
        context_window=128_000,
    ),
    "gemini-2.0-flash": ProviderPricing(
        provider_name="google",
        model_name="gemini-2.0-flash",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.003,
        context_window=1_000_000,
        supports_streaming=True,
    ),
}


def _catalog_projection() -> dict[str, ProviderPricing]:
    """Phase-2 catalog consumer (first projection, founder-directed):
    every catalog model prices identically here and in aragora.models.

    Before this projection, the Pareto/decision-stakes router saw $0 for
    every frontier pin because this table's hand-maintained rows had gone
    stale — a real routing bug, not hygiene (#9355 follow-up).

    Only the CANONICAL id of each catalog model is projected. Enumeration
    consumers (get_available_models, get_models_within_budget,
    ProviderRouter._details_from_pricing) treat table keys as distinct
    candidate models, so projecting alias/openrouter spellings would let
    one model occupy several candidate slots and crowd out genuinely
    distinct providers. Alias spellings still price correctly through the
    ``by_any_id`` fallback in :func:`get_estimated_cost`.

    Models still inside their catalog soak window are NOT projected
    (#9364 round-3): the enumerated table is an adoption surface, and the
    catalog contract says a model must not be adopted before
    ``soak_until``. Cost lookup for under-soak ids keeps working through
    the same ``by_any_id`` fallback, which has no soak gating."""
    rows: dict[str, ProviderPricing] = {}
    for spec in CATALOG.values():
        if spec.is_under_soak():
            continue
        rows[spec.canonical_id] = ProviderPricing(
            provider_name=spec.provider,
            model_name=spec.direct_id,
            input_cost_per_1k=spec.input_per_mtok / 1000.0,
            output_cost_per_1k=spec.output_per_mtok / 1000.0,
            context_window=spec.context_window,
        )
    return rows


def _apply_catalog_projection(table: dict[str, ProviderPricing]) -> None:
    """Apply the canonical-only projection to a pricing table, in place.

    Enforced POST-CONDITION (the single invariant behind the #9364
    round 1-4 findings, instead of pruning discovered cases one by one):

        Every key of the resulting table that ``by_any_id`` resolves to a
        catalog model (a) IS that model's canonical_id and (b) that model
        is NOT under soak.

    The projection is applied first (it overrides any hand row keyed by an
    adoptable model's canonical id: single source), then one sweep deletes
    every violating key — alias-keyed hand rows, stale canonical rows of
    under-soak models, or any future spelling the catalog learns about.
    Hand rows for models unknown to the catalog are preserved. Cost lookup
    for every dropped spelling (aliases and under-soak ids alike) still
    works via the ``by_any_id`` fallback in ``get_estimated_cost``, which
    has no soak gating.
    """
    table.update(_catalog_projection())
    for key in list(table):
        spec = by_any_id(key)
        if spec is not None and (key != spec.canonical_id or spec.is_under_soak()):
            del table[key]


# Soak gating is a function of the calendar date, so the applied projection
# is memoized per day rather than frozen at import (#9364 round-5): a
# long-running server imported before a model's soak_until must start
# enumerating it once the date passes.
_projection_refreshed_on: date | None = None


def _refresh_projection_if_stale() -> None:
    """Re-apply the catalog projection if the date rolled since last time.

    Mutates PROVIDER_PRICING IN PLACE, so from-importers of the dict also
    observe refreshed contents — but only after any accessor in this module
    has run on the new date. Direct dict readers that never go through an
    accessor see the last-refreshed snapshot (import-time, at worst).

    Concurrency: plain check-and-swap, matching this module's otherwise
    lock-free conventions; a concurrent duplicate refresh is idempotent
    (the sweep is a fixed point) so the race is harmless.
    """
    global _projection_refreshed_on
    today = date.today()
    if _projection_refreshed_on == today:
        return
    _apply_catalog_projection(PROVIDER_PRICING)
    _projection_refreshed_on = today


def _current_pricing_table() -> dict[str, ProviderPricing]:
    """Date-fresh view of PROVIDER_PRICING for enumeration consumers."""
    _refresh_projection_if_stale()
    return PROVIDER_PRICING


# Initial application (also stamps the memo date).
_refresh_projection_if_stale()


def get_estimated_cost(
    provider: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for a given provider and token usage.

    Args:
        provider: Model key in PROVIDER_PRICING (e.g. "claude-opus-4"),
            or any catalog spelling (canonical/direct/openrouter/alias)
            resolvable by ``aragora.models.by_any_id``.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD. Keys missing from PROVIDER_PRICING fall
        back to catalog ``by_any_id`` resolution — this is the load-bearing
        path for alias spellings of cataloged models, which are deliberately
        NOT projected into the enumerated table. Returns 0.0 only when the
        model is unknown to both the table and the catalog.
    """
    pricing = _current_pricing_table().get(provider)
    if pricing is None:
        spec = by_any_id(provider)
        if spec is None:
            return 0.0
        return (input_tokens / 1_000_000.0) * spec.input_per_mtok + (
            output_tokens / 1_000_000.0
        ) * spec.output_per_mtok

    input_cost = (input_tokens / 1000.0) * pricing.input_cost_per_1k
    output_cost = (output_tokens / 1000.0) * pricing.output_cost_per_1k
    return input_cost + output_cost


def get_available_models() -> list[str]:
    """Return list of all model keys with known pricing."""
    return list(_current_pricing_table().keys())


def get_cheapest_model() -> str:
    """Return the model key with the lowest combined cost per 1K tokens."""
    table = _current_pricing_table()
    return min(
        table,
        key=lambda k: table[k].input_cost_per_1k + table[k].output_cost_per_1k,
    )


def get_models_within_budget(
    budget_per_debate: float,
    estimated_input_tokens: int = 2000,
    estimated_output_tokens: int = 1000,
) -> list[str]:
    """Return model keys whose estimated cost fits within a per-debate budget.

    Args:
        budget_per_debate: Maximum cost per debate in USD.
        estimated_input_tokens: Expected input tokens per debate.
        estimated_output_tokens: Expected output tokens per debate.

    Returns:
        List of model keys sorted by cost (cheapest first).
    """
    affordable: list[tuple[float, str]] = []
    for model_key, pricing in _current_pricing_table().items():
        cost = get_estimated_cost(model_key, estimated_input_tokens, estimated_output_tokens)
        if cost <= budget_per_debate:
            affordable.append((cost, model_key))
    affordable.sort()
    return [model_key for _, model_key in affordable]


__all__ = [
    "ProviderPricing",
    "PROVIDER_PRICING",
    "get_estimated_cost",
    "get_available_models",
    "get_cheapest_model",
    "get_models_within_budget",
]
