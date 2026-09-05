"""Tests for the single old→current model upgrade map.

See ``aragora/models/upgrade_map.py`` for the contract. Controller rulings
(frontier-model-refresh, Task 2, 2026-09-04) refine the original brief:

1. ``UPGRADES`` keys must be spellings of RETIRED or ABSENT models only —
   several brief-listed spellings turned out to be aliases Task 1 attached
   to ACTIVE rows (a collision class Task 1 itself hit once already, for
   ``mistral-large`` / ``gemini-3.1-pro``), so those spellings were dropped
   from the key lists in ``upgrade_map.py``. ``mistral-large`` and
   ``gemini-3.1-pro`` themselves stay as UPGRADES keys: Task 1 deliberately
   kept those two bare spellings OFF their rows' aliases (they collide with
   static routing hand-rows), so the upgrade map is their only path back to
   the current id.
2. ``resolve_model_id`` also handles retired-but-catalogued spellings that
   are *not* UPGRADES keys: it falls back to the retired row's family
   frontier via ``frontier_for``.
3. ``RETIRED_PATTERN`` must never match a spelling belonging to an ACTIVE
   catalog row — including as a *prefix* of a longer active spelling (e.g.
   retired key ``"claude-fable-5"`` must not match inside active canonical
   id ``"claude-fable-5-1"``; retired-adjacent key ``"kimi-k2"`` must not
   match inside active canonical id ``"kimi-k2.7-code"``).
"""

from __future__ import annotations

import pytest

from aragora.models.catalog import CATALOG
from aragora.models.upgrade_map import RETIRED_PATTERN, UPGRADES, resolve_model_id


@pytest.mark.parametrize(
    "old,new",
    [
        ("claude-fable-5", "claude-fable-5-1"),
        ("anthropic/claude-fable-5", "claude-fable-5-1"),
        ("claude-3-opus-20240229", "claude-fable-5-1"),
        ("claude-sonnet-4-6", "claude-fable-5-1"),
        ("gpt-4", "gpt-6-astra"),
        ("gpt-4o", "gpt-6-astra"),
        ("gpt-4o-mini", "gpt-5.6-terra"),
        ("gpt-5.5", "gpt-6-astra"),
        ("gpt-5.6-sol", "gpt-6-astra"),
        ("openai/gpt-5.3", "gpt-6-astra"),
        # Controller ruling 4: new case, not in the original brief list.
        ("openai/gpt-5.5", "gpt-6-astra"),
        ("gemini-2.0-flash", "gemini-3.8-flash"),
        ("gemini-1.5-flash", "gemini-3.8-flash"),
        ("gemini-3-pro", "gemini-3.1-pro-preview"),
        # Kept per controller ruling 1: "gemini-3.1-pro" is not a catalog
        # alias (Task 1 kept it off the row to avoid a routing hand-row
        # collision), so this resolves via UPGRADES, not alias lookup.
        ("gemini-3.1-pro", "gemini-3.1-pro-preview"),
        ("grok-2", "grok-4.6"),
        ("grok-4-latest", "grok-4.6"),
        ("x-ai/grok-4.5", "grok-4.6"),
        # Kept per controller ruling 1: same rationale as gemini-3.1-pro.
        ("mistral-large", "mistral-large-2512"),
        # Resolves via alias lookup now (mistral-medium-2604 carries this
        # spelling as a catalog alias), NOT via UPGRADES — controller
        # ruling 1 removed it as an UPGRADES key since it collides with an
        # active row's alias.
        ("mistral-medium-latest", "mistral-medium-2604"),
        ("deepseek-r1", "deepseek-v4-pro-0813"),
        ("deepseek/deepseek-v4-pro", "deepseek-v4-pro-0813"),
        ("qwen3-max", "qwen3.8-2.4t-a95b"),
        ("qwen/qwen3.7-max", "qwen3.8-2.4t-a95b"),
        # Controller ruling 4: new case. Resolves via alias lookup (the
        # active qwen3.8-2.4t-a95b row carries this spelling as a catalog
        # alias), NOT via UPGRADES — ruling 1 removed it as a key.
        ("qwen/qwen3.8-max", "qwen3.8-2.4t-a95b"),
        ("kimi-k2", "kimi-k3"),
        ("moonshotai/kimi-k2-thinking", "kimi-k3"),
        ("llama-3.3-70b", "muse-spark-1.3"),
        ("meta-llama/llama-4-maverick", "muse-spark-1.3"),
    ],
)
def test_known_upgrades(old: str, new: str) -> None:
    assert resolve_model_id(old) == new


def test_every_target_is_an_active_catalog_row() -> None:
    for old, new in UPGRADES.items():
        assert new in CATALOG, (old, new)
        assert not CATALOG[new].retired, (old, new)


def test_current_ids_pass_through_and_none_is_none() -> None:
    assert resolve_model_id("claude-fable-5-1") == "claude-fable-5-1"
    assert resolve_model_id("some-unknown-model") == "some-unknown-model"
    assert resolve_model_id(None) is None


def test_retired_pattern_matches_keys_only() -> None:
    for old in UPGRADES:
        assert RETIRED_PATTERN.search(old), old
    assert not RETIRED_PATTERN.search("claude-fable-5-1")
    assert not RETIRED_PATTERN.search("gpt-6-astra")


def test_retired_pattern_only_contains_retired_or_absent_spellings() -> None:
    """Controller ruling 1: every UPGRADES key must belong to a RETIRED or
    ABSENT catalog row — never an ACTIVE row's canonical/direct/openrouter
    id or alias. This is the direct guard for the collision class the
    ruling identifies (spellings Task 1 attached as aliases to active rows,
    e.g. ``mistral-medium-latest``, ``qwen/qwen3.8-max``)."""
    active_spellings: set[str] = set()
    for spec in CATALOG.values():
        if not spec.retired:
            active_spellings.update(spec.all_ids())
    collisions = sorted(set(UPGRADES) & active_spellings)
    assert not collisions, collisions


def test_retired_pattern_never_matches_any_active_catalog_spelling() -> None:
    """Controller ruling 3: guards the collision class Task 1 hit, where a
    retired-row spelling (e.g. ``"claude-fable-5"``, ``"kimi-k2"``,
    ``"deepseek-v4-pro"``) is a literal *prefix* of a longer ACTIVE row's
    id (``"claude-fable-5-1"``, ``"kimi-k2.7-code"``,
    ``"deepseek-v4-pro-0813"``). A naive substring pattern would falsely
    flag/rewrite the active id; RETIRED_PATTERN must not match it."""
    for spec in CATALOG.values():
        if spec.retired:
            continue
        for spelling in spec.all_ids():
            assert not RETIRED_PATTERN.search(spelling), (spec.canonical_id, spelling)


def test_retired_row_spellings_without_upgrades_entry_use_family_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_model_id ruling 2's third branch: a spelling that belongs to
    a RETIRED catalog row but has no UPGRADES entry still resolves to that
    row's family frontier via spec_or_none + frontier_for. Every current
    retired row's spellings already have UPGRADES entries (belt-and-braces
    from the brief's exhaustive listing), so this branch is exercised here
    by removing one to prove the fallback works independently of the map."""
    assert "qwen3.7-max" in UPGRADES  # sanity: brief already covers it directly
    monkeypatch.delitem(UPGRADES, "qwen3.7-max")
    assert resolve_model_id("qwen3.7-max") == "qwen3.8-2.4t-a95b"


def test_removed_active_alias_spellings_are_not_upgrades_keys() -> None:
    """Controller ruling 1's exact removal list: these spellings collide
    with aliases Task 1 attached to ACTIVE rows, so they must not be
    UPGRADES keys — even though resolve_model_id still maps them correctly
    via catalog alias resolution (see test_known_upgrades)."""
    removed = {
        "claude-fable-5.1",
        "anthropic/claude-fable-5.1",
        "anthropic/claude-fable-5-1",
        "google/gemini-3.1-pro",
        "mistral-large-latest",
        "mistral-medium-3.5",
        "mistral-medium-latest",
        "qwen3.8-max",
        "qwen/qwen3.8-max",
    }
    assert not (removed & set(UPGRADES))
    for spelling in removed:
        spec = CATALOG.get(
            next(
                (s.canonical_id for s in CATALOG.values() if spelling in s.all_ids()),
                "",
            )
        )
        if spec is not None:
            assert not spec.retired, spelling
