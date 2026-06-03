"""Regression test for issue #7393.

`aragora demo --receipt <path>` must produce receipt files that pass
`aragora receipt verify <path>` with ``Result: VALID (3/3 checks passed)``.

The original bug was a producer/consumer canonicalization mismatch: the demo
writer stored the receipt *signature* in the ``artifact_hash`` field (and
omitted the ``timestamp`` field), while ``aragora receipt verify`` recomputes a
content-addressable SHA-256 over ``{receipt_id, gauntlet_id, input_hash,
risk_summary, verdict, confidence}`` and also requires ``timestamp``. The two
hashes never agreed, so every demo receipt failed verification (1/3 checks).

This test exercises the *full* CLI repro end-to-end (``demo.main`` with
``--receipt`` followed by ``cmd_receipt_verify``) so the round-trip invariant is
guarded against regression, not just the helper functions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aragora.cli.commands.receipt import cmd_receipt_verify
from aragora.cli.demo import (
    _build_live_receipt_data,
    main as demo_main,
)
from aragora.gauntlet.receipt_models import DecisionReceipt


_TOPIC = "Should Aragora prioritize the EU AI Act compliance pipeline?"


def _verify_receipt_file(receipt_path: Path, capsys) -> str:
    """Run ``aragora receipt verify`` on a file and return its stdout."""
    with pytest.raises(SystemExit) as exc:
        cmd_receipt_verify(argparse.Namespace(receipt=str(receipt_path), verbose=True))
    out = capsys.readouterr().out
    assert exc.value.code == 0, f"verify exited {exc.value.code}; output:\n{out}"
    return out


def test_demo_receipt_verifies_end_to_end(tmp_path, capsys):
    """The exact issue #7393 repro: demo --receipt then receipt verify == VALID."""
    receipt_path = tmp_path / "receipt.json"

    args = argparse.Namespace(
        name=None,
        topic=_TOPIC,
        list_demos=False,
        server=False,
        receipt=str(receipt_path),
        offline=True,
    )
    demo_main(args)

    assert receipt_path.exists(), "demo --receipt did not write the receipt file"

    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    # The two fields whose absence/mismatch caused the original INVALID result.
    assert saved.get("timestamp"), "receipt is missing the required 'timestamp' field"
    assert saved.get("artifact_hash"), "receipt is missing 'artifact_hash'"

    # Stored hash must equal the recomputed content hash (no signature-vs-hash mixup).
    receipt = DecisionReceipt.from_dict(saved)
    assert receipt.verify_integrity() is True

    out = _verify_receipt_file(receipt_path, capsys)
    assert "Result: VALID (3/3 checks passed)" in out


def test_demo_receipt_survives_json_persistence(tmp_path, capsys):
    """Receipt integrity holds across the file write/read JSON round-trip."""
    receipt_path = tmp_path / "receipt.json"
    args = argparse.Namespace(
        name=None,
        topic=_TOPIC,
        list_demos=False,
        server=False,
        receipt=str(receipt_path),
        offline=True,
    )
    demo_main(args)

    reloaded = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert DecisionReceipt.from_dict(reloaded).verify_integrity() is True


def test_live_demo_receipt_builder_is_verifiable(tmp_path, capsys):
    """The live-demo receipt *builder* produces a verifiable receipt.

    This covers ``_build_live_receipt_data`` (the helper that maps a playground
    debate response into receipt fields) and asserts the resulting receipt
    satisfies the same round-trip invariant. It does not drive the full live
    ``_run_real_demo`` → ``_save_live_demo_receipt`` path (which requires API
    access); it guards the builder against the same hash/timestamp regression.
    """
    live_result = {
        "receipt_id": "DR-LIVE-7393",
        "consensus_reached": True,
        "participants": ["claude", "gpt", "gemini"],
        "verdict": "consensus",
        "confidence": 0.71,
        "rounds_used": 3,
        "final_answer": "Prioritize the EU AI Act compliance pipeline.",
        "dissenting_views": [],
        "proposals": {"claude": "yes", "gpt": "yes"},
    }

    receipt_data = _build_live_receipt_data(live_result, _TOPIC, elapsed=2.5)
    assert receipt_data.get("timestamp")
    assert receipt_data.get("artifact_hash")
    assert receipt_data["question"] == _TOPIC

    receipt_path = tmp_path / "live-receipt.json"
    receipt_path.write_text(json.dumps(receipt_data, indent=2, default=str), encoding="utf-8")

    out = _verify_receipt_file(receipt_path, capsys)
    assert "Result: VALID (3/3 checks passed)" in out
