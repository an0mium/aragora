"""Tests for aragora_debate.receipt."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aragora_debate.receipt import ReceiptBuilder
from aragora_debate.types import (
    Consensus,
    ConsensusMethod,
    DebateResult,
    DecisionReceipt,
    DissentRecord,
    Verdict,
)


def _make_result(**kwargs):
    defaults = dict(
        task="Should we deploy to prod?",
        status="consensus_reached",
        rounds_used=3,
        confidence=0.85,
        consensus_reached=True,
        participants=["claude", "gpt4"],
        proposals={"claude": "Yes deploy", "gpt4": "Yes with caveats"},
        consensus=Consensus(
            reached=True,
            method=ConsensusMethod.MAJORITY,
            confidence=0.85,
            supporting_agents=["claude", "gpt4"],
            dissenting_agents=[],
        ),
    )
    defaults.update(kwargs)
    return DebateResult(**defaults)


class TestReceiptBuilder:
    def test_from_result_basic(self):
        result = _make_result()
        receipt = ReceiptBuilder.from_result(result)

        assert receipt.receipt_id.startswith("DR-")
        assert receipt.question == "Should we deploy to prod?"
        assert receipt.verdict == Verdict.APPROVED
        assert receipt.confidence == 0.85
        assert "claude" in receipt.agents

    def test_from_result_no_consensus(self):
        result = _make_result(
            consensus_reached=False,
            confidence=0.3,
            consensus=Consensus(
                reached=False,
                method=ConsensusMethod.MAJORITY,
                confidence=0.3,
                supporting_agents=[],
                dissenting_agents=["claude", "gpt4"],
            ),
        )
        receipt = ReceiptBuilder.from_result(result)
        assert receipt.verdict == Verdict.REJECTED

    def test_from_result_needs_review(self):
        result = _make_result(
            consensus_reached=False,
            confidence=0.5,
            consensus=Consensus(
                reached=False,
                method=ConsensusMethod.MAJORITY,
                confidence=0.5,
                supporting_agents=["claude"],
                dissenting_agents=["gpt4"],
            ),
        )
        receipt = ReceiptBuilder.from_result(result)
        assert receipt.verdict == Verdict.NEEDS_REVIEW

    def test_from_result_with_dissent(self):
        result = _make_result(
            consensus=Consensus(
                reached=True,
                method=ConsensusMethod.MAJORITY,
                confidence=0.7,
                supporting_agents=["claude"],
                dissenting_agents=["gpt4"],
                dissents=[DissentRecord(agent="gpt4", reasons=["risk too high"])],
            ),
        )
        receipt = ReceiptBuilder.from_result(result)
        assert receipt.verdict == Verdict.APPROVED_WITH_CONDITIONS

    def test_signature_present(self):
        result = _make_result()
        receipt = ReceiptBuilder.from_result(result)
        assert receipt.signature is not None
        assert receipt.signature_algorithm == "SHA-256-content-hash"
        assert len(receipt.signature) == 64  # hex SHA-256


class TestHMACSigning:
    def test_sign_and_verify(self):
        result = _make_result()
        receipt = ReceiptBuilder.from_result(result)

        ReceiptBuilder.sign_hmac(receipt, "test-secret-key")
        assert receipt.signature_algorithm == "HMAC-SHA256"
        assert receipt.signature is not None

        assert ReceiptBuilder.verify_hmac(receipt, "test-secret-key") is True

    def test_wrong_key_fails(self):
        result = _make_result()
        receipt = ReceiptBuilder.from_result(result)

        ReceiptBuilder.sign_hmac(receipt, "correct-key")
        assert ReceiptBuilder.verify_hmac(receipt, "wrong-key") is False

    def test_tampered_receipt_fails(self):
        result = _make_result()
        receipt = ReceiptBuilder.from_result(result)

        ReceiptBuilder.sign_hmac(receipt, "key")
        receipt.confidence = 0.99  # tamper
        assert ReceiptBuilder.verify_hmac(receipt, "key") is False

    def test_verify_unsigned_receipt(self):
        result = _make_result()
        receipt = ReceiptBuilder.from_result(result)
        assert ReceiptBuilder.verify_hmac(receipt, "key") is False


class TestExport:
    def test_to_json(self):
        result = _make_result()
        receipt = ReceiptBuilder.from_result(result)
        json_str = ReceiptBuilder.to_json(receipt)
        assert '"receipt_id"' in json_str
        assert '"verdict"' in json_str

    def test_to_html(self):
        result = _make_result()
        receipt = ReceiptBuilder.from_result(result)
        html = ReceiptBuilder.to_html(receipt)
        assert "<!DOCTYPE html>" in html
        assert "Decision Receipt" in html
        assert receipt.receipt_id in html

    def test_to_html_with_dissent(self):
        result = _make_result(
            consensus=Consensus(
                reached=True,
                method=ConsensusMethod.MAJORITY,
                confidence=0.7,
                supporting_agents=["claude"],
                dissenting_agents=["gpt4"],
                dissents=[DissentRecord(agent="gpt4", reasons=["too risky"])],
            ),
        )
        receipt = ReceiptBuilder.from_result(result)
        html = ReceiptBuilder.to_html(receipt)
        assert "Dissenting" in html
        assert "gpt4" in html
