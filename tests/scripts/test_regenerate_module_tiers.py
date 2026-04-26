"""External invariants for scripts/regenerate_module_tiers.py.

The classifier's --check mode compares live evidence against the
committed aragora/module_tiers.yaml. This test suite adds a second
line of defence: catastrophic regressions (counter returns 0, regex
typo matches nothing, wrong threshold inverted) surface independent
of whether the committed yaml happens to agree with the bug.

Invariants are deliberately loose — normal codebase evolution must
not break them; only a counting bug should.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "regenerate_module_tiers.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("regenerate_module_tiers", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["regenerate_module_tiers"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    mod = _load_module()
    return mod.gather_tiers()


def test_classifier_finds_most_top_level_modules(report):
    """Evidence must cover >= 100 top-level aragora/ modules.

    Catches: regex typo that skips modules, path typo, empty git
    ls-files result, etc.
    """
    assert len(report.modules) >= 100, (
        f"only {len(report.modules)} modules classified — a counter "
        f"or path regex may have broken; expected at least 100"
    )


def test_core_tier_non_empty_with_known_members(report):
    """core tier must always include the load-bearing modules.

    Catches: manual-override map accidentally emptied, debate/agents
    not registering as core because test counter returns 0, etc.
    """
    by_name = {m.name: m for m in report.modules}
    must_be_core = {"debate", "agents", "core", "memory", "cli", "config"}
    for name in must_be_core:
        assert name in by_name, f"{name!r} missing from classified modules"
        assert by_name[name].tier == "core", (
            f"module {name!r} should be tier=core but is {by_name[name].tier!r}"
        )


def test_shipped_non_python_surfaces_are_not_deprecated(report):
    """Non-Python product surfaces can be live without Python import references."""
    by_name = {m.name: m for m in report.modules}
    live = by_name["live"]
    assert live.tier == "integrated"
    assert live.override_reason


def test_no_module_has_both_zero_importers_and_zero_tests_in_core(report):
    """A module with no importers AND no tests cannot be tier=core.

    Catches: override map pushing a truly-dead module to core.
    """
    core = [m for m in report.modules if m.tier == "core"]
    for m in core:
        if m.importer_count == 0 and m.test_file_count == 0:
            # Allowed only if explicitly manually promoted via override.
            assert m.override_reason, (
                f"module {m.name!r} is tier=core but has 0 importers and "
                f"0 tests, and no override_reason. Remove the override or "
                f"demote the tier."
            )


def test_every_module_has_positive_py_file_count(report):
    """Every classified module must have >= 1 .py file.

    Catches: top-level-modules glob picking up empty dirs or
    __pycache__ leaks.
    """
    for m in report.modules:
        assert m.py_file_count >= 1, f"module {m.name!r} has 0 py_files — glob/filter bug?"


def test_tier_distribution_is_sane(report):
    """Sanity: core should be a minority; not all modules lump into one tier."""
    counts: dict[str, int] = {}
    for m in report.modules:
        counts[m.tier] = counts.get(m.tier, 0) + 1

    total = sum(counts.values())
    # No single tier should hold more than 90% — that would mean
    # thresholds are degenerate.
    for tier, count in counts.items():
        assert count / total < 0.9, (
            f"tier {tier!r} holds {count}/{total} modules (>90%) — thresholds may be miscalibrated"
        )
    # Core should exist.
    assert counts.get("core", 0) > 0, "no module classified as core"


def test_yaml_round_trip_preserves_tiers(tmp_path, monkeypatch):
    """Writing yaml and re-parsing must yield the same tier per module.

    Catches: parser/renderer disagreement (extra quoting, missing
    keys, etc.).
    """
    mod = _load_module()
    report = mod.gather_tiers()
    yaml_path = tmp_path / "tiers.yaml"
    yaml_path.write_text(mod.render_yaml(report))

    parsed = mod.parse_current(yaml_path)
    for m in report.modules:
        assert m.name in parsed, f"module {m.name!r} dropped during round-trip"
        assert parsed[m.name]["tier"] == m.tier, (
            f"round-trip tier mismatch for {m.name!r}: {parsed[m.name]['tier']!r} != {m.tier!r}"
        )


def test_check_mode_is_idempotent(tmp_path, monkeypatch):
    """Running regenerate twice must report no drift."""
    mod = _load_module()
    report = mod.gather_tiers()
    monkeypatch.setattr(mod, "TIERS_YAML", tmp_path / "tiers.yaml")
    (tmp_path / "tiers.yaml").write_text(mod.render_yaml(report))

    report_b = mod.gather_tiers()
    drifted, drifts = mod.check_drift(report_b)
    assert not drifted, f"drift detected between two back-to-back classifications: {drifts}"
