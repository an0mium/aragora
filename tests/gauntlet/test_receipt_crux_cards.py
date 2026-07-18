"""DecisionReceipt carries crux cards additively (#8227 / #9046 phase 1).

Acceptance (work order on #9046):
- flag OFF → today's byte-identical receipts (no `cruxes` key at all);
- flag ON → receipts carry detected cruxes with per-crux dissent attribution;
- additive-only: `aragora-verify` / ODR consumers of pre-crux receipts are
  unaffected (the ODR `cruxes` block already validated present/absent).
"""

from __future__ import annotations

import hashlib
import json

from aragora.core_types import DebateResult, Message
from aragora.gauntlet.odr_export import decision_receipt_to_odr
from aragora.gauntlet.receipt_models import DecisionReceipt


def _sample_cards() -> dict:
    return {
        "items": [
            {
                "claim_id": "c1",
                "statement": "The latency budget holds under burst load",
                "author": "agent-alpha",
                "crux_score": 0.82,
                "contesting_agents": ["agent-beta"],
                "affected_claims": ["c2", "c3"],
            }
        ],
        "total_claims": 5,
        "total_disagreements": 1,
        "convergence_barrier": 0.41,
        "detector": "belief_network",
    }


def _debate_result(metadata: dict | None = None) -> DebateResult:
    return DebateResult(
        debate_id="d-crux",
        task="Should we ship feature X?",
        final_answer="Yes, ship behind a canary with latency alerts.",
        confidence=0.85,
        consensus_reached=True,
        rounds_used=2,
        participants=["agent-alpha", "agent-beta"],
        messages=[
            Message(
                role="proposer",
                agent="agent-alpha",
                content="Ship it: the latency budget holds under burst load.",
            ),
            Message(
                role="critic",
                agent="agent-beta",
                content="The latency budget does not hold under burst load.",
            ),
        ],
        winner="agent-alpha",
        metadata=metadata or {},
    )


class TestReceiptCruxesField:
    def test_flag_off_receipt_has_no_cruxes_key(self) -> None:
        """No crux metadata → field is None and serialization omits the key,
        keeping flag-off receipts byte-identical to pre-flag receipts."""
        receipt = DecisionReceipt.from_debate_result(_debate_result())
        assert receipt.cruxes is None
        assert "cruxes" not in receipt.to_dict()

    def test_crux_cards_metadata_flows_to_receipt(self) -> None:
        cards = _sample_cards()
        receipt = DecisionReceipt.from_debate_result(_debate_result(metadata={"crux_cards": cards}))
        assert receipt.cruxes == cards
        data = receipt.to_dict()
        assert data["cruxes"]["items"][0]["contesting_agents"] == ["agent-beta"]

    def test_round_trip_from_dict(self) -> None:
        cards = _sample_cards()
        receipt = DecisionReceipt.from_debate_result(_debate_result(metadata={"crux_cards": cards}))
        restored = DecisionReceipt.from_dict(receipt.to_dict())
        assert restored.cruxes == cards

    def test_round_trip_without_cruxes(self) -> None:
        receipt = DecisionReceipt.from_debate_result(_debate_result())
        restored = DecisionReceipt.from_dict(receipt.to_dict())
        assert restored.cruxes is None

    def test_empty_items_not_attached(self) -> None:
        receipt = DecisionReceipt.from_debate_result(
            _debate_result(metadata={"crux_cards": {"items": []}})
        )
        assert receipt.cruxes is None

    def test_malformed_metadata_ignored(self) -> None:
        receipt = DecisionReceipt.from_debate_result(
            _debate_result(metadata={"crux_cards": "not-a-dict"})
        )
        assert receipt.cruxes is None

    def test_integrity_hash_binds_cruxes(self) -> None:
        """Crux cards are audit content: the artifact hash binds them when
        present (tampering breaks verify_integrity), while pre-crux receipts
        hash exactly as before so stored hashes keep verifying."""
        without = DecisionReceipt.from_debate_result(_debate_result())
        legacy_material = json.dumps(
            {
                "receipt_id": without.receipt_id,
                "gauntlet_id": without.gauntlet_id,
                "input_hash": without.input_hash,
                "risk_summary": without.risk_summary,
                "verdict": without.verdict,
                "confidence": without.confidence,
            },
            sort_keys=True,
        )
        assert without.artifact_hash == hashlib.sha256(legacy_material.encode()).hexdigest()

        with_cards = DecisionReceipt.from_debate_result(
            _debate_result(metadata={"crux_cards": _sample_cards()})
        )
        assert with_cards.verify_integrity()

    def test_tampered_cruxes_fail_integrity(self) -> None:
        receipt = DecisionReceipt.from_debate_result(
            _debate_result(metadata={"crux_cards": _sample_cards()})
        )
        restored = DecisionReceipt.from_dict(receipt.to_dict())
        assert restored.verify_integrity()
        restored.cruxes["items"][0]["statement"] = "tampered claim"
        assert not restored.verify_integrity()
        restored2 = DecisionReceipt.from_dict(receipt.to_dict())
        restored2.cruxes["items"][0]["contesting_agents"] = []
        assert not restored2.verify_integrity()


class TestOdrExportCruxes:
    def test_receipt_cruxes_populate_odr_block(self) -> None:
        receipt = DecisionReceipt.from_debate_result(
            _debate_result(metadata={"crux_cards": _sample_cards()})
        )
        odr = decision_receipt_to_odr(receipt)
        assert odr["cruxes"]["status"] == "present"
        assert odr["cruxes"]["items"][0]["contesting_agents"] == ["agent-beta"]

    def test_without_cruxes_block_stays_absent(self) -> None:
        receipt = DecisionReceipt.from_debate_result(_debate_result())
        odr = decision_receipt_to_odr(receipt)
        assert odr["cruxes"]["status"] == "absent"

    def test_explicit_crux_set_still_wins(self) -> None:
        receipt = DecisionReceipt.from_debate_result(
            _debate_result(metadata={"crux_cards": _sample_cards()})
        )
        odr = decision_receipt_to_odr(receipt, crux_set=[{"claim": "explicit"}])
        assert odr["cruxes"]["items"] == [{"claim": "explicit"}]

    def test_odr_schema_validates_crux_receipt(self) -> None:
        import json
        from pathlib import Path

        import jsonschema

        schema = json.loads((Path("aragora/gauntlet/odr_schema.json")).read_text(encoding="utf-8"))
        receipt = DecisionReceipt.from_debate_result(
            _debate_result(metadata={"crux_cards": _sample_cards()})
        )
        odr = decision_receipt_to_odr(receipt)
        jsonschema.validate(odr, schema)
