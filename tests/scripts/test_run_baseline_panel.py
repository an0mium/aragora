"""Tests for ``scripts/run_baseline_panel.py``'s model pinning.

The single-family baseline panel is only a valid measurement if the JUDGE
comes from a different model family than the panel it scores. PR 3's trial
sweep collapsed this script's two retired pins onto one current id and
erased the distinction silently -- nothing referenced the script, so no test
failed. The 2026-09-04 wave-3 controller ruling requires both ids to come
from ``aragora.config.model_pins`` and to stay in different families; these
tests pin that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from aragora.config.model_pins import FABLE_51_DIRECT, GPT6_ASTRA_DIRECT
from aragora.models.catalog import spec_or_none

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_baseline_panel.py"


def _load(monkeypatch: pytest.MonkeyPatch, provider: str) -> Any:
    monkeypatch.setenv("BASELINE_PROVIDER", provider)
    spec = importlib.util.spec_from_file_location(f"baseline_panel_{provider}", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_panel_and_judge_are_different_families(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    module = _load(monkeypatch, provider)
    panel = spec_or_none(module.PANEL_MODEL)
    judge = spec_or_none(module.JUDGE_MODEL)
    assert panel is not None and judge is not None
    assert panel.family != judge.family, (
        f"BASELINE_PROVIDER={provider}: judge {module.JUDGE_MODEL} shares the "
        f"panel's family {panel.family}, so it cannot score the panel independently"
    )


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_both_models_come_from_pins_and_are_active(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    module = _load(monkeypatch, provider)
    assert {module.PANEL_MODEL, module.JUDGE_MODEL} == {FABLE_51_DIRECT, GPT6_ASTRA_DIRECT}
    for model_id in (module.PANEL_MODEL, module.JUDGE_MODEL):
        spec = spec_or_none(model_id)
        assert spec is not None and not spec.retired, model_id


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_each_role_routes_to_its_own_catalog_provider(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """Panel and judge are different providers now, so one process-wide
    provider string can no longer route both calls."""
    module = _load(monkeypatch, provider)
    assert module.PANEL_PROVIDER == spec_or_none(module.PANEL_MODEL).provider
    assert module.JUDGE_PROVIDER == spec_or_none(module.JUDGE_MODEL).provider
    assert module.PANEL_PROVIDER != module.JUDGE_PROVIDER


def test_budget_estimate_comes_from_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rail used to read a four-row hand table keyed on retired ids."""
    module = _load(monkeypatch, "anthropic")
    spec = spec_or_none(FABLE_51_DIRECT)
    assert spec is not None
    expected = (1_000 / 1_000_000) * spec.input_per_mtok + (800 / 1_000_000) * spec.output_per_mtok
    assert module._estimate_usd(FABLE_51_DIRECT, 1_000, 800) == pytest.approx(expected)
