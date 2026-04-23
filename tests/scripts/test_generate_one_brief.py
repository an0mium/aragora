"""Tests for :mod:`scripts.generate_one_brief` helpers.

Scoped to the pure formatting / summarization helpers — the CLI's
network + filesystem side effects are covered by
:mod:`tests.brief_engine.test_storage_helpers` (for persistence) and
:mod:`tests.pdb.test_invoker_factory` (for invoker wiring).

Mission context: PR #6441 2026-04-22 Mode 3 dogfood findings —
Fix 3 (confidence bucketing) turned the raw float ``0.82/5`` display
into a 1..5 bucket with the raw float preserved in parentheses
(``4/5 (raw=0.82)``). These tests lock in that contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_one_brief.py"


def _load_cli_module():
    """Import ``scripts/generate_one_brief.py`` as a module.

    The script is not installed under any package, so we load it via
    importlib. We set a unique module name to avoid collisions with
    other tests that may import it.
    """
    spec = importlib.util.spec_from_file_location(
        "aragora_tests._generate_one_brief_cli",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def cli():
    return _load_cli_module()


class TestFormatConfidence:
    """The n/5 bucket must never display a raw float as a 5-point score."""

    def test_buckets_zero_to_one_into_one_to_five_inclusive(self, cli) -> None:
        # 0.82 → round(4.1) = 4 → "4/5 (raw=0.82)"
        assert cli._format_confidence(0.82) == "4/5 (raw=0.82)"

    def test_midpoint(self, cli) -> None:
        # 0.5 → round(2.5) = 2 (banker's rounding in Python 3) → "2/5".
        # Assert on the observed Python-3 behavior so the test stays
        # deterministic across platforms.
        bucket = cli._format_confidence(0.5)
        assert bucket in {"2/5 (raw=0.50)", "3/5 (raw=0.50)"}

    def test_upper_bound_clamps_to_five(self, cli) -> None:
        assert cli._format_confidence(1.0) == "5/5 (raw=1.00)"

    def test_lower_bound_clamps_to_one(self, cli) -> None:
        # round(0.0) = 0, but we clamp floor to 1 so the display never
        # shows "0/5" (which would imply "no confidence at all" rather
        # than "low confidence").
        assert cli._format_confidence(0.0) == "1/5 (raw=0.00)"

    def test_tiny_positive_value_is_bucket_one(self, cli) -> None:
        assert cli._format_confidence(0.01) == "1/5 (raw=0.01)"

    def test_high_value_is_bucket_five(self, cli) -> None:
        # 0.95 * 5 = 4.75 → round to 5
        assert cli._format_confidence(0.95) == "5/5 (raw=0.95)"

    def test_raw_preserved_precision(self, cli) -> None:
        result = cli._format_confidence(0.834)
        assert "raw=0.83" in result
        assert result.endswith(")")

    def test_include_raw_false_drops_suffix(self, cli) -> None:
        assert cli._format_confidence(0.82, include_raw=False) == "4/5"
        assert cli._format_confidence(1.0, include_raw=False) == "5/5"

    def test_out_of_range_value_falls_back_to_raw(self, cli) -> None:
        # Values outside [0, 1] surface as the original input so we
        # catch unexpected callers rather than silently clipping.
        assert cli._format_confidence(4) == "4/5"
        assert cli._format_confidence(3.7) == "3.7/5"

    def test_non_numeric_value_falls_back_to_raw(self, cli) -> None:
        assert cli._format_confidence("oops") == "oops/5"

    def test_does_not_emit_raw_float_as_scale(self, cli) -> None:
        """Regression for the dogfood finding: never print ``0.82/5``.

        The pre-fix CLI printed ``confidence:    0.82/5`` which looked
        like the confidence was a 5-point number. The bucketed output
        must use an integer numerator for 0..1 floats.
        """
        result = cli._format_confidence(0.82)
        # The numerator before '/' must be a digit.
        numerator = result.split("/")[0]
        assert numerator.isdigit(), f"bucket numerator must be an integer, got {numerator!r}"
        assert numerator != "0.82"
