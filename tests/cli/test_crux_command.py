"""Tests for the `aragora crux` CLI verb (Crux A3 / #6039)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aragora.cli.commands import crux as crux_cmd
from aragora.cli.parser import build_parser


def _fake_proof(*, debate_id: str = "debate-xyz") -> SimpleNamespace:
    """Minimal stand-in for an `aragora.debate.consensus.ConsensusProof`.

    Only the attributes the CLI reads are populated.
    """
    return SimpleNamespace(
        debate_id=debate_id,
        task="Should we ship?",
        final_claim="__CRUX_MAP__: no verdict by design; see CruxReceipt.cruxes",
        metadata={
            "consensus_mode": "crux_finder",
            "approach": "A",
            "cruxes": [
                {
                    "claim_id": "c1",
                    "statement": "Is the core assumption sound?",
                    "author": "agent-alpha",
                    "crux_score": 0.82,
                    "influence_score": 0.7,
                    "disagreement_score": 0.6,
                    "uncertainty_score": 0.5,
                    "centrality_score": 0.8,
                    "affected_claims": ["c2"],
                    "contesting_agents": ["agent-alpha", "agent-beta"],
                    "resolution_impact": 0.4,
                }
            ],
            "counterfactuals": [
                {
                    "claim_id": "c1",
                    "condition": "Resolve c1 to high confidence",
                    "outcome_change": "Reduces total network uncertainty by 0.400",
                    "likelihood": 0.5,
                    "affected_claims": ["c2"],
                }
            ],
            "recommended_focus": ["c1"],
            "convergence_barrier": 0.62,
            "crux_count": 1,
        },
    )


def _fake_debate_result(*, debate_id: str = "debate-xyz") -> SimpleNamespace:
    return SimpleNamespace(
        debate_id=debate_id,
        consensus_proof=_fake_proof(debate_id=debate_id),
        proposals={"agent-alpha": "proposal-alpha", "agent-beta": "proposal-beta"},
    )


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def test_crux_parser_registered_and_dispatches_to_cmd_crux() -> None:
    parser = build_parser()
    args = parser.parse_args(["crux", "Should we ship?", "--rounds", "2", "--top-k", "3"])
    assert args.command == "crux"
    assert args.question == "Should we ship?"
    assert args.rounds == 2
    assert args.top_k == 3
    # Lazy loader resolves to cmd_crux.
    assert callable(args.func)


def test_crux_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["crux", "Q"])
    assert args.rounds == 3
    assert args.top_k == 5
    assert args.min_score == pytest.approx(0.3)
    assert args.format == "markdown"
    assert args.no_counterfactuals is False
    assert args.dry_run is False


# ---------------------------------------------------------------------------
# Dry-run fast path — no debate executed
# ---------------------------------------------------------------------------


def test_cmd_crux_dry_run_does_not_invoke_debate(capsys) -> None:
    args = argparse.Namespace(
        question="Should we ship?",
        agents=None,
        rounds=3,
        top_k=5,
        min_score=0.3,
        no_counterfactuals=False,
        format="markdown",
        receipt=None,
        output=None,
        dry_run=True,
    )

    with patch.object(crux_cmd, "_run_crux_debate") as mock_run:
        crux_cmd.cmd_crux(args)

    mock_run.assert_not_called()
    captured = capsys.readouterr()
    assert "dry-run" in captured.out
    assert "crux-finder" in captured.out


def test_cmd_crux_empty_question_exits() -> None:
    args = argparse.Namespace(
        question="   ",
        agents=None,
        rounds=3,
        top_k=5,
        min_score=0.3,
        no_counterfactuals=False,
        format="markdown",
        receipt=None,
        output=None,
        dry_run=False,
    )
    with pytest.raises(SystemExit) as exc_info:
        crux_cmd.cmd_crux(args)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# End-to-end with a mocked debate
# ---------------------------------------------------------------------------


def test_cmd_crux_full_path_prints_markdown(capsys) -> None:
    args = argparse.Namespace(
        question="Should we ship?",
        agents="alpha,beta",
        rounds=2,
        top_k=5,
        min_score=0.3,
        no_counterfactuals=False,
        format="markdown",
        receipt=None,
        output=None,
        dry_run=False,
    )

    async def _fake_run(*_a, **_k):
        return _fake_debate_result()

    with patch.object(crux_cmd, "_run_crux_debate", side_effect=_fake_run):
        crux_cmd.cmd_crux(args)

    captured = capsys.readouterr()
    # Markdown headline is the crux map, not a decision.
    assert captured.out.startswith("# Crux Map — Should we ship?")
    assert "Is the core assumption sound?" in captured.out
    # Receipt id + checksum emitted for audit trail.
    assert "crux-" in captured.out


def test_cmd_crux_writes_receipt_and_output_files(tmp_path: Path, capsys) -> None:
    receipt_path = tmp_path / "crux.json"
    output_path = tmp_path / "crux.md"
    args = argparse.Namespace(
        question="Should we ship?",
        agents="alpha,beta",
        rounds=2,
        top_k=5,
        min_score=0.3,
        no_counterfactuals=False,
        format="markdown",
        receipt=str(receipt_path),
        output=str(output_path),
        dry_run=False,
    )

    async def _fake_run(*_a, **_k):
        return _fake_debate_result()

    with patch.object(crux_cmd, "_run_crux_debate", side_effect=_fake_run):
        crux_cmd.cmd_crux(args)

    # Receipt file: JSON with stable fields.
    payload = json.loads(receipt_path.read_text())
    assert payload["question"] == "Should we ship?"
    assert payload["checksum"]
    assert len(payload["checksum"]) == 16
    assert payload["cruxes"], "cruxes should round-trip via the receipt file"

    # Rendered output file matches stdout.
    rendered = output_path.read_text()
    assert rendered.startswith("# Crux Map — Should we ship?")

    err = capsys.readouterr().err
    assert "Receipt saved to" in err
    assert "Rendered output saved to" in err


def test_cmd_crux_json_output_serializes_receipt(capsys) -> None:
    args = argparse.Namespace(
        question="Q",
        agents="a,b",
        rounds=1,
        top_k=5,
        min_score=0.3,
        no_counterfactuals=False,
        format="json",
        receipt=None,
        output=None,
        dry_run=False,
    )

    async def _fake_run(*_a, **_k):
        return _fake_debate_result()

    with patch.object(crux_cmd, "_run_crux_debate", side_effect=_fake_run):
        crux_cmd.cmd_crux(args)

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["question"] == "Q"
    assert payload["cruxes"][0]["claim_id"] == "c1"
    assert payload["checksum"] and len(payload["checksum"]) == 16


def test_cmd_crux_rejects_result_without_crux_proof(capsys) -> None:
    args = argparse.Namespace(
        question="Q",
        agents="a",
        rounds=1,
        top_k=5,
        min_score=0.3,
        no_counterfactuals=False,
        format="markdown",
        receipt=None,
        output=None,
        dry_run=False,
    )

    async def _fake_run(*_a, **_k):
        # No consensus_proof at all — simulates a mode fall-back or early failure.
        return SimpleNamespace(consensus_proof=None, proposals={})

    with patch.object(crux_cmd, "_run_crux_debate", side_effect=_fake_run):
        with pytest.raises(SystemExit) as exc_info:
            crux_cmd.cmd_crux(args)
    assert exc_info.value.code == 1
    assert "could not build receipt" in capsys.readouterr().err


def test_cmd_crux_rejects_non_crux_proof(capsys) -> None:
    """Rejects a ConsensusProof that wasn't produced by crux-finder mode."""
    wrong_proof = SimpleNamespace(
        debate_id="d",
        task="q",
        final_claim="some verdict",
        metadata={"consensus_mode": "majority"},
    )
    args = argparse.Namespace(
        question="Q",
        agents="a",
        rounds=1,
        top_k=5,
        min_score=0.3,
        no_counterfactuals=False,
        format="markdown",
        receipt=None,
        output=None,
        dry_run=False,
    )

    async def _fake_run(*_a, **_k):
        return SimpleNamespace(consensus_proof=wrong_proof, proposals={})

    with patch.object(crux_cmd, "_run_crux_debate", side_effect=_fake_run):
        with pytest.raises(SystemExit) as exc_info:
            crux_cmd.cmd_crux(args)
    assert exc_info.value.code == 1
    assert "crux-finder" in capsys.readouterr().err
