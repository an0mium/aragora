from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.cli.commands.debate import _persist_debate_receipt
from aragora.cli.commands.receipt import cmd_receipt_verify
from aragora.gauntlet.receipt_models import DecisionReceipt


def test_persisted_debate_receipt_verifies_with_receipt_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = SimpleNamespace(
        debate_id="debate-smoke",
        task="Verify provider-bootstrap dogfood receipts.",
        consensus_reached=True,
        confidence=0.87,
        final_answer="Provider bootstrap receipt is verifiable.",
        rounds_used=1,
        dissenting_views=[],
        metadata={
            "agent_models": {
                "grok_proposer": {
                    "provider": "xai",
                    "provider_display": "xAI",
                    "model": "grok-4-latest",
                    "llm_label": "grok-4-latest via xAI",
                }
            }
        },
        messages=[
            SimpleNamespace(
                agent="grok_proposer",
                role="proposer",
                round=0,
                content="Provider bootstrap receipt is verifiable.",
            )
        ],
    )

    receipt_path = _persist_debate_receipt(result)

    assert receipt_path is not None
    data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    assert data["receipt_id"] == "debate-debate-smoke"
    assert data["verdict"] == "PASS"
    assert data["timestamp"].endswith("Z")
    assert len(data["artifact_hash"]) == 64
    assert DecisionReceipt.from_dict(data).verify_integrity() is True

    with pytest.raises(SystemExit) as excinfo:
        cmd_receipt_verify(argparse.Namespace(receipt=receipt_path, verbose=False))

    assert excinfo.value.code == 0
    assert "Result: VALID" in capsys.readouterr().out


def _mixed_family_result() -> SimpleNamespace:
    """A mixed-family debate result where mistral participated (critique)
    but produced no Message — the #8101 silent-drop scenario."""
    return SimpleNamespace(
        debate_id="debate-mixed",
        task="Mixed family debate",
        consensus_reached=True,
        confidence=0.8,
        final_answer="Answer",
        rounds_used=1,
        dissenting_views=[],
        participants=["grok", "mistral-api"],
        proposals={"grok": "Proposal text"},
        critiques=[SimpleNamespace(agent="mistral-api", target_agent="grok")],
        votes=[],
        metadata={
            "agent_models": {
                "grok": {
                    "provider": "xai",
                    "provider_display": "xAI",
                    "model": "grok-4-latest",
                    "llm_label": "grok-4-latest via xAI",
                },
                "mistral-api": {
                    "provider": "mistral",
                    "provider_display": "Mistral",
                    "model": "mistral-large-latest",
                    "llm_label": "mistral-large-latest via Mistral",
                },
            },
            "agent_roster": {
                "requested": ["grok", "mistral-api"],
                "created": ["grok", "mistral-api"],
                "failed": [],
            },
        },
        messages=[SimpleNamespace(agent="grok", role="proposer", round=0, content="Proposal text")],
    )


def test_receipt_records_full_roster_in_mixed_family_debate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #8101: mixed grok,mistral-api debates must not silently drop
    mistral from the receipt agents list just because it authored no Message."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    receipt_path = _persist_debate_receipt(_mixed_family_result())

    assert receipt_path is not None
    data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    assert data["agents"] == ["grok", "mistral-api"]
    assert data["agents_requested"] == ["grok", "mistral-api"]
    assert data["agents_failed"] == []
    assert DecisionReceipt.from_dict(data).verify_integrity() is True


def test_receipt_agent_contributions_are_artifact_backed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-agent contribution counts must reflect actual artifacts so the
    roster entry cannot be mistaken for fabricated participation."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    receipt_path = _persist_debate_receipt(_mixed_family_result())

    assert receipt_path is not None
    data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    contributions = data["agent_contributions"]
    assert contributions["grok"]["messages"] == 1
    assert contributions["grok"]["proposals"] == 1
    assert contributions["mistral-api"]["critiques"] == 1
    assert contributions["mistral-api"]["messages"] == 0
    # Consensus support must stay artifact-backed (both contributed here).
    assert set(data["consensus_proof"]["supporting_agents"]) == {"grok", "mistral-api"}


def test_receipt_does_not_count_silent_agents_as_supporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent in the roster with zero artifacts is listed (requested/created)
    but must NOT appear among consensus supporting_agents."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = _mixed_family_result()
    result.critiques = []  # mistral now produced nothing at all

    receipt_path = _persist_debate_receipt(result)

    assert receipt_path is not None
    data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    assert data["agents"] == ["grok", "mistral-api"]
    assert data["agent_contributions"]["mistral-api"] == {
        "messages": 0,
        "proposals": 0,
        "critiques": 0,
        "votes": 0,
    }
    assert data["consensus_proof"]["supporting_agents"] == ["grok"]


def test_receipt_agents_fall_back_to_artifact_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a participants roster, agents are unioned from all artifact
    sources (messages, proposals, critiques, votes) — not just messages."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = _mixed_family_result()
    del result.participants
    result.metadata.pop("agent_roster")

    receipt_path = _persist_debate_receipt(result)

    assert receipt_path is not None
    data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    assert set(data["agents"]) == {"grok", "mistral-api"}
