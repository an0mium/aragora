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


from aragora.models import CATALOG, by_any_id, utc_today

# Legacy hand-maintained rows (models not yet in the canonical catalog).
# Cataloged models are PROJECTED from aragora.models.CATALOG below — do not
# hand-edit rows for cataloged ids here; edit the catalog instead
# (docs/architecture/MODEL_CATALOG.md). This base dict is never mutated;
# each snapshot rebuild starts from a copy of it.
_HAND_ROWS: dict[str, ProviderPricing] = {
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

# Compat export for from-importers. Rebound (never mutated) to a fresh
# immutable snapshot whenever an accessor below runs on a new UTC date, so:
# - module-attribute readers (aragora.routing.provider_config.PROVIDER_PRICING)
#   see date-fresh soak gating after any accessor has run;
# - from-importers hold whichever snapshot object was current when they
#   imported — possibly stale, but NEVER mutated under them, so iteration
#   is always safe (stale-but-never-corrupt);
# - code that needs guaranteed freshness must use get_available_models() /
#   current_pricing_table().
# Initialized by the _refresh_projection_if_stale() call further down.
PROVIDER_PRICING: dict[str, ProviderPricing] = {}


def _catalog_projection(today: date | None = None) -> dict[str, ProviderPricing]:
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
    the same ``by_any_id`` fallback, which has no soak gating.

    ``today`` anchors soak evaluation; pass one value through a whole
    rebuild so gating and the refresh stamp share a coherent date."""
    rows: dict[str, ProviderPricing] = {}
    for spec in CATALOG.values():
        if spec.is_under_soak(today):
            continue
        rows[spec.canonical_id] = ProviderPricing(
            provider_name=spec.provider,
            model_name=spec.direct_id,
            input_cost_per_1k=spec.input_per_mtok / 1000.0,
            output_cost_per_1k=spec.output_per_mtok / 1000.0,
            context_window=spec.context_window,
        )
    return rows


def _apply_catalog_projection(table: dict[str, ProviderPricing], today: date | None = None) -> None:
    """Apply the canonical-only projection to a pricing table, in place.

    Only ever called on tables that are NOT yet published (snapshot builds
    and tests); published snapshots are immutable — see
    ``_refresh_projection_if_stale``.

    Enforced POST-CONDITION (the single invariant behind the #9364
    round 1-4 findings, instead of pruning discovered cases one by one):

        Every key of the resulting table that ``by_any_id`` resolves to a
        catalog model (a) IS that model's canonical_id and (b) that model
        is NOT under soak (as of ``today``).

    The projection is applied first (it overrides any hand row keyed by an
    adoptable model's canonical id: single source), then one sweep deletes
    every violating key — alias-keyed hand rows, stale canonical rows of
    under-soak models, or any future spelling the catalog learns about.
    Hand rows for models unknown to the catalog are preserved. Cost lookup
    for every dropped spelling (aliases and under-soak ids alike) still
    works via the ``by_any_id`` fallback in ``get_estimated_cost``, which
    has no soak gating.
    """
    table.update(_catalog_projection(today))
    for key in list(table):
        spec = by_any_id(key)
        if spec is not None and (key != spec.canonical_id or spec.is_under_soak(today)):
            del table[key]


def _build_pricing_snapshot(today: date) -> dict[str, ProviderPricing]:
    """Build a fresh pricing table (hand rows + projection) for ``today``."""
    table = dict(_HAND_ROWS)
    _apply_catalog_projection(table, today=today)
    return table


# Soak gating is a function of the calendar date, so the published snapshot
# is memoized per UTC day rather than frozen at import (#9364 round-5): a
# long-running server imported before a model's soak_until must start
# enumerating it once the date passes.
_projection_refreshed_on: date | None = None


def _refresh_projection_if_stale() -> None:
    """Publish a fresh snapshot if the UTC date rolled since last time.

    Snapshot semantics (#9364 round-6): a NEW dict is built off-line and
    then ATOMICALLY rebound to ``PROVIDER_PRICING``; a published snapshot
    is never mutated. Re-entrant callers (get_models_within_budget
    iterating while get_estimated_cost refreshes mid-loop) and concurrent
    threads therefore always iterate a stable object by construction —
    they may briefly see the previous day's snapshot, never a dict that
    changes size under them.

    One ``today`` value (UTC — soak is a governance boundary) is threaded
    through the whole rebuild so soak gating and the memo stamp agree.
    Plain check-and-swap, matching this module's lock-free conventions:
    a concurrent duplicate rebuild publishes an identical snapshot.
    """
    global _projection_refreshed_on, PROVIDER_PRICING
    today = utc_today()
    if _projection_refreshed_on == today:
        return
    snapshot = _build_pricing_snapshot(today)
    # Stamp before publish: a concurrent reader between the two statements
    # gets the previous day's snapshot without triggering a duplicate
    # rebuild (either ordering is benign; this one avoids wasted work).
    _projection_refreshed_on = today
    PROVIDER_PRICING = snapshot


def current_pricing_table() -> dict[str, ProviderPricing]:
    """Date-fresh pricing snapshot for enumeration consumers.

    The returned dict is immutable-by-convention: safe to iterate across
    re-entrant refreshes and from other threads."""
    _refresh_projection_if_stale()
    return PROVIDER_PRICING


# Back-compat private alias (pre-round-7 name).
_current_pricing_table = current_pricing_table


# Initial publication (also stamps the memo date).
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
    pricing = current_pricing_table().get(provider)
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
    return list(current_pricing_table().keys())


def get_cheapest_model() -> str:
    """Return the model key with the lowest combined cost per 1K tokens."""
    table = current_pricing_table()
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
    for model_key, pricing in current_pricing_table().items():
        cost = get_estimated_cost(model_key, estimated_input_tokens, estimated_output_tokens)
        if cost <= budget_per_debate:
            affordable.append((cost, model_key))
    affordable.sort()
    return [model_key for _, model_key in affordable]


__all__ = [
    "ProviderPricing",
    "PROVIDER_PRICING",
    "current_pricing_table",
    "get_estimated_cost",
    "get_available_models",
    "get_cheapest_model",
    "get_models_within_budget",
]
