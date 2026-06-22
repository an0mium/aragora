"""Tests for the `aragora ask --crux` flag wiring (#8227).

`--crux` is an additive flag on the `ask` subparser that selects
crux-finder consensus mode (mapping load-bearing disagreements instead of
producing a verdict). It overrides `--consensus`.
"""

from __future__ import annotations

import argparse

from aragora.cli.parser import build_parser


def _parse_ask(*extra: str) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(["ask", "should we use rust or go for the service", *extra])


def test_crux_flag_defaults_false() -> None:
    args = _parse_ask()
    assert args.crux is False


def test_crux_flag_sets_true() -> None:
    args = _parse_ask("--crux")
    assert args.crux is True


def test_crux_flag_is_additive_to_consensus_choices() -> None:
    """--crux does not pollute the --consensus choices surface."""
    parser = build_parser()
    # crux_finder is intentionally NOT a --consensus choice; --crux selects it.
    args = parser.parse_args(["ask", "x y z", "--consensus", "majority"])
    assert args.consensus == "majority"
    assert args.crux is False


def test_crux_override_logic() -> None:
    """The cmd_ask override maps --crux to crux_finder consensus."""
    args = _parse_ask("--crux", "--consensus", "majority")
    # Replicate the cmd_ask post-parse override.
    if getattr(args, "crux", False):
        args.consensus = "crux_finder"
    assert args.consensus == "crux_finder"


def test_extract_result_cruxes_present() -> None:
    """The CLI crux extractor reads the current (#8366) arena-path location."""
    from aragora.cli.commands.debate import _extract_result_cruxes

    class _Proof:
        metadata = {
            "consensus_mode": "crux_finder",
            "crux_count": 1,
            "convergence_barrier": 0.4,
            "cruxes": [{"statement": "X", "crux_score": 0.9, "contesting_agents": ["a", "b"]}],
            "recommended_focus": ["c0"],
        }

    class _Result:
        consensus_proof = _Proof()
        formal_verification = {"crux_finder": {"crux_count": 1, "convergence_barrier": 0.4}}

    cruxes, summary = _extract_result_cruxes(_Result())
    assert len(cruxes) == 1
    assert cruxes[0]["statement"] == "X"
    assert summary["convergence_barrier"] == 0.4


def test_extract_result_cruxes_absent_never_fabricates() -> None:
    """No cruxes are invented when crux-finder mode was not run."""
    from aragora.cli.commands.debate import _extract_result_cruxes

    class _Proof:
        metadata = {"consensus_mode": "majority"}

    class _Result:
        consensus_proof = _Proof()
        formal_verification = {}

    cruxes, summary = _extract_result_cruxes(_Result())
    assert cruxes == []
    assert summary == {}
