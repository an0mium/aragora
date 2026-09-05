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

    soaking = [
        (s, s.soak_until) for s in CATALOG.values() if s.soak_until is not None and not s.retired
    ]
    assert soaking, "fixture assumption: the catalog carries at least one soaking row"

    baseline = set(pm.provider_config_rows())
    for spec, soak_until in soaking:
        assert soak_until is not None
        assert spec.canonical_id in baseline, (
            f"{spec.canonical_id} is soaking but active; it must still be a routing candidate"
        )
        mid_soak = spec.release_date + (soak_until - spec.release_date) // 2
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


# ---------------------------------------------------------------------------
# Bucket emission (2026-09-05 merge-gate fix wave: C-P1 / O-P2c on #9989)
#
# Every row must price under EVERY provider label a caller legitimately
# passes for it, not only under ``ModelSpec.provider``:
#   * ``openrouter`` -- OpenRouterAgent.agent_type is "openrouter", and that
#     is what the monthly budget guard and orchestrator_runner hand
#     ``calculate_token_cost``;
#   * the FAMILY bucket for an openrouter-provider row -- what
#     ``cost_estimation._pricing_provider`` derives for it.
# ---------------------------------------------------------------------------


def _default_cost(tokens_in: int, tokens_out: int) -> Decimal:
    """Cost of the openrouter bucket's documented $2/$8 default rate."""
    return (Decimal(tokens_in) / Decimal(10**6)) * Decimal("2.00") + (
        Decimal(tokens_out) / Decimal(10**6)
    ) * Decimal("8.00")


def _expected_cost(spec, tokens_in: int, tokens_out: int) -> Decimal:
    """Tier-aware, like ``calculate_token_cost`` itself: a prompt at or above
    a row's documented ``long_context_threshold`` bills every token in the
    request at the higher rate (finding O-P2b on #9989). Using the flat
    fields here would make this helper disagree with the code it checks for
    exactly the rows the tier exists for (``gpt-6-astra``, ``grok-4.6``)."""
    input_rate, output_rate = spec.rates_for(tokens_in)
    return (Decimal(tokens_in) / Decimal(10**6)) * Decimal(str(input_rate)) + (
        Decimal(tokens_out) / Decimal(10**6)
    ) * Decimal(str(output_rate))


def test_openrouter_bucket_prices_every_openrouter_slug() -> None:
    """Every catalog row's OpenRouter slug prices at the catalog rate under
    ``provider="openrouter"`` -- the exact regression the reviewer measured
    (``anthropic/claude-fable-5.1`` billed $10 instead of $60 per MTok
    pair, a ~6x under-count of Fable-via-OpenRouter spend)."""
    from aragora.billing.usage import calculate_token_cost

    assert calculate_token_cost(
        "openrouter", "anthropic/claude-fable-5.1", 1_000_000, 1_000_000
    ) == Decimal("60.00")

    for s in CATALOG.values():
        for slug in (i for i in s.all_ids() if "/" in i):
            assert calculate_token_cost("openrouter", slug, 1_000, 500) == _expected_cost(
                s, 1_000, 500
            ), f"{slug} does not price under the openrouter bucket"


def test_every_via_openrouter_pin_prices_non_default() -> None:
    """Every ``*_VIA_OPENROUTER`` model pin -- the fallback target of every
    native agent after the frontier refresh -- must price at its catalog
    rate, never at the bucket default."""
    from aragora.billing.usage import calculate_token_cost
    from aragora.config import model_pins
    from aragora.models.catalog import by_any_id

    pins = {n: getattr(model_pins, n) for n in dir(model_pins) if n.endswith("_VIA_OPENROUTER")}
    assert len(pins) >= 10, f"pin discovery found only {sorted(pins)}"
    for name, slug in sorted(pins.items()):
        spec = by_any_id(slug)
        assert spec is not None, f"{name} pins uncataloged slug {slug!r}"
        cost = calculate_token_cost("openrouter", slug, 1_000_000, 1_000_000)
        assert cost == _expected_cost(spec, 1_000_000, 1_000_000), name
        if _expected_cost(spec, 1_000_000, 1_000_000) != _default_cost(1_000_000, 1_000_000):
            assert cost != _default_cost(1_000_000, 1_000_000), (
                f"{name} ({slug}) silently billed at the openrouter default rate"
            )


def test_openrouter_bucket_does_not_gain_bare_spellings() -> None:
    """Only slash-bearing (OpenRouter-shaped) spellings enter the
    ``openrouter`` bucket from a NATIVE row: a bare id under that bucket
    would claim a rate for a spelling no OpenRouter call ever sends."""
    rows = pm.usage_rows()
    native_bare = {
        s.canonical_id
        for s in CATALOG.values()
        if s.provider != "openrouter"
        for spelling in s.all_ids()
        if "/" not in spelling
    }
    leaked = sorted(m for m in native_bare if m in rows["openrouter"])
    assert not leaked, f"bare native spellings leaked into the openrouter bucket: {leaked}"


def test_every_row_also_prices_under_its_family_bucket() -> None:
    """``cost_estimation._pricing_provider`` asks for the FAMILY bucket
    whenever that family names a live pricing bucket (``deepseek`` for
    ``deepseek-v4-pro-0813``), so every row of a family must be emitted
    there -- including a same-family row reached through a DIFFERENT
    provider (``qwen3.7-max`` is provider ``alibaba``, family ``qwen``),
    which would otherwise be shadowed down to the default rate by its
    openrouter-provider sibling creating the bucket."""
    rows = pm.usage_rows()
    for s in CATALOG.values():
        if not s.family:
            continue
        for spelling in s.all_ids():
            assert rows[s.family][spelling] == Decimal(str(s.input_per_mtok)), (
                f"{spelling} missing from the {s.family!r} bucket"
            )
    assert rows["qwen"]["qwen3.7-max"] == Decimal(str(CATALOG["qwen3.7-max"].input_per_mtok))


def test_cost_estimation_pairs_all_price_non_default() -> None:
    """Every (provider, model_key) pair ``cost_estimation`` can emit prices
    at a real catalog rate. This is the reviewer's O-P2c finding generalized:
    the DeepSeek pair was silently falling back to $2/$8."""
    from aragora.billing.usage import calculate_token_cost
    from aragora.models.catalog import by_any_id
    from aragora.server.handlers.debates.cost_estimation import MODEL_PROVIDER_MAP

    for spelling, (provider, model_key) in sorted(MODEL_PROVIDER_MAP.items()):
        spec = by_any_id(model_key) or by_any_id(spelling)
        if spec is None:
            continue  # legacy hand row for an uncataloged spelling
        assert calculate_token_cost(provider, model_key, 1_000, 500) == _expected_cost(
            spec, 1_000, 500
        ), f"{spelling!r} -> ({provider}, {model_key}) does not price at its catalog rate"


def test_deepseek_cost_estimation_pair_is_non_default() -> None:
    """Falsifiability anchor for the pair above (the reviewer's exact case)."""
    from aragora.billing.usage import calculate_token_cost
    from aragora.server.handlers.debates.cost_estimation import MODEL_PROVIDER_MAP

    provider, model_key = MODEL_PROVIDER_MAP["deepseek-v4-pro-0813"]
    assert (provider, model_key) == ("deepseek", "deepseek-v4-pro-0813")
    cost = calculate_token_cost(provider, model_key, 1_000_000, 1_000_000)
    assert cost == Decimal("1.1207") + Decimal("3.362")
    assert cost != _default_cost(1_000_000, 1_000_000)


def test_pre_pr_explicit_openrouter_rows_price_as_before() -> None:
    """The three OpenRouter spellings that had explicit hand rows before the
    frontier refresh must keep their exact pre-PR prices."""
    from aragora.billing.usage import calculate_token_cost

    assert calculate_token_cost(
        "openrouter", "anthropic/claude-opus-5", 1_000_000, 1_000_000
    ) == Decimal("30.00")
    assert calculate_token_cost(
        "openrouter", "anthropic/claude-fable-5", 1_000_000, 1_000_000
    ) == Decimal("60.00")
    assert calculate_token_cost("openrouter", "openai/gpt-5.5", 1_000_000, 1_000_000) == Decimal(
        "35.00"
    )
