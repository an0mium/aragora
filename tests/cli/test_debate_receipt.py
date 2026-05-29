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
