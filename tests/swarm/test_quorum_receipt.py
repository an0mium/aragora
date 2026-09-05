"""A merge-quorum PR-review outcome must become a portable, verifiable
DecisionReceipt. This bridge is the M2 core: it turns the result of a
heterogeneous-model PR review (CollectOutcome) into a DecisionReceipt that
exports to a schema-conformant ODR with an internally consistent quorum block
(supporting/dissenting agents are a subset of the participants), so
aragora-verify can verify it independently."""

from __future__ import annotations

import jsonschema

from aragora.gauntlet.odr_export import decision_receipt_to_odr, load_odr_schema
from aragora.gauntlet.receipt_models import DecisionReceipt
from aragora.swarm.quorum_evidence import CollectOutcome, EvidenceItem
from aragora.swarm.quorum_receipt import collect_outcome_to_decision_receipt


def _supportive_outcome(
    *,
    action: str = "prepare",
    action_reason: str = "dry-run; re-run with --apply to post",
    posted: list[str] | None = None,
) -> CollectOutcome:
    return CollectOutcome(
        repo="synaptent/aragora",
        pr=8667,
        head_sha="b" * 40,
        head_committed_at="2026-06-27T09:00:00+00:00",
        tier=2,
        action=action,
        action_reason=action_reason,
        posted=list(posted or []),
        tiered_gate=False,
        items=[
            EvidenceItem(
                family="claude", body="PASS: looks correct", would_count=True, verdict="pass"
            ),
            EvidenceItem(family="openai", body="PASS: agreed", would_count=True, verdict="pass"),
        ],
    )


def _outcome(*, tier: int = 1) -> CollectOutcome:
    return CollectOutcome(
        repo="synaptent/aragora",
        pr=8667,
        head_sha="a" * 40,
        head_committed_at="2026-06-27T08:00:00+00:00",
        tier=tier,
        action="prepare",
        action_reason="reviewer dissent present (grok); prepared evidence only",
        tiered_gate=False,
        items=[
            EvidenceItem(
                family="claude", body="PASS: looks correct", would_count=True, verdict="pass"
            ),
            EvidenceItem(family="openai", body="PASS: agreed", would_count=True, verdict="pass"),
            EvidenceItem(
                family="grok",
                body="[P1] correctness: off-by-one in the loop bound",
                would_count=True,
                verdict="changes_requested",
            ),
        ],
    )


def test_bridge_returns_a_decision_receipt():
    receipt = collect_outcome_to_decision_receipt(_outcome())
    assert isinstance(receipt, DecisionReceipt)
    assert "8667" in receipt.receipt_id


def test_bridge_preserves_quorum_families():
    receipt = collect_outcome_to_decision_receipt(_outcome())
    proof = receipt.consensus_proof
    assert proof is not None
    assert proof.supporting_agents == []
    assert receipt.risk_summary["supportive"] == 2
    assert "grok" in proof.dissenting_agents


def test_bridge_fails_closed_when_supportive_quorum_has_dissent():
    receipt = collect_outcome_to_decision_receipt(_outcome())

    assert receipt.verdict == "CHANGES_REQUESTED"
    assert receipt.consensus_proof is not None
    assert receipt.consensus_proof.reached is False


def test_bridge_fails_closed_when_supportive_quorum_is_prepare_only():
    receipt = collect_outcome_to_decision_receipt(_supportive_outcome())

    assert receipt.verdict == "CHANGES_REQUESTED"
    assert receipt.consensus_proof is not None
    assert receipt.consensus_proof.reached is False
    assert receipt.consensus_proof.supporting_agents == []


def test_bridge_passes_when_supportive_quorum_was_posted():
    receipt = collect_outcome_to_decision_receipt(
        _supportive_outcome(
            action="post",
            action_reason="posted exact-head evidence",
            posted=["claude", "openai"],
        )
    )

    assert receipt.verdict == "PASS"
    assert receipt.consensus_proof is not None
    assert receipt.consensus_proof.reached is True
    assert receipt.consensus_proof.supporting_agents == ["claude", "openai"]


def test_bridge_fails_closed_when_posted_families_do_not_satisfy_quorum():
    receipt = collect_outcome_to_decision_receipt(
        _supportive_outcome(
            action="post",
            action_reason="posting attempted",
            posted=["claude"],
        )
    )

    assert receipt.verdict == "CHANGES_REQUESTED"
    assert receipt.consensus_proof is not None
    assert receipt.consensus_proof.reached is False
    assert receipt.consensus_proof.supporting_agents == ["claude"]


def test_bridge_excludes_unposted_supportive_families_from_consensus_proof():
    receipt = collect_outcome_to_decision_receipt(
        _supportive_outcome(
            action="post",
            action_reason="partial posting attempted",
            posted=["openai"],
        )
    )

    assert receipt.risk_summary["supportive"] == 2
    assert receipt.consensus_proof is not None
    assert receipt.consensus_proof.supporting_agents == ["openai"]


def test_one_posted_family_failure_is_stable_when_tiered_gate_env_is_on(monkeypatch):
    monkeypatch.setenv("ARAGORA_ENABLE_TIERED_MERGE_GATE", "1")

    receipt = collect_outcome_to_decision_receipt(
        _supportive_outcome(
            action="post",
            action_reason="posting attempted",
            posted=["claude"],
        )
    )

    assert receipt.settlement_metadata["tiered_gate"] is False
    assert receipt.verdict == "CHANGES_REQUESTED"
    assert receipt.consensus_proof is not None
    assert receipt.consensus_proof.reached is False


def test_bridged_odr_fails_closed_when_supportive_quorum_has_dissent():
    odr = decision_receipt_to_odr(collect_outcome_to_decision_receipt(_outcome()))

    assert odr["claim"]["verdict"] == "CHANGES_REQUESTED"
    assert odr["quorum"]["reached"] is False
    assert odr["quorum"]["supporting_agents"] == []
    assert odr["quorum"]["dissent"]["present"] is True


def test_bridged_odr_discloses_provider_lineage_for_participants():
    odr = decision_receipt_to_odr(collect_outcome_to_decision_receipt(_outcome()))
    participants = {row["agent"]: row["model_family"] for row in odr["quorum"]["participants"]}

    assert participants == {
        "claude": "anthropic",
        "grok": "xai",
        "openai": "openai",
    }


def test_bridged_receipt_exports_to_conformant_odr():
    receipt = collect_outcome_to_decision_receipt(_outcome())
    odr = decision_receipt_to_odr(receipt)
    jsonschema.validate(odr, load_odr_schema())


def test_bridged_odr_quorum_block_is_internally_consistent():
    # aragora-verify FAILs a receipt whose supporting/dissenting agents are not
    # among the participants. The bridge must keep them aligned.
    odr = decision_receipt_to_odr(collect_outcome_to_decision_receipt(_outcome()))
    quorum = odr["quorum"]
    participants = {p["agent"] for p in quorum.get("participants", [])}
    for agent in quorum.get("supporting_agents", []):
        assert agent in participants, f"supporting agent {agent} not a participant"
    dissent = quorum.get("dissent") or {}
    for agent in dissent.get("dissenting_agents", []):
        assert agent in participants, f"dissenting agent {agent} not a participant"


def test_bridge_metadata_records_pr_provenance():
    receipt = collect_outcome_to_decision_receipt(_outcome(tier=2))
    meta = receipt.settlement_metadata
    assert meta["repo"] == "synaptent/aragora"
    assert meta["pr"] == 8667
    assert meta["head_sha"] == "a" * 40
    assert meta["tier"] == 2


def test_example_merge_quorum_receipt_matches_emitter():
    # The committed example is the emitter<->verifier contract for PR-review
    # receipts (verified independently in aragora-verify). It must equal exactly
    # what the bridge + odr_export emit today.
    import json
    from pathlib import Path

    example = Path("docs/specs/examples/example-merge-quorum-receipt.odr.json")
    expected = decision_receipt_to_odr(collect_outcome_to_decision_receipt(_outcome()))
    actual = json.loads(example.read_text(encoding="utf-8"))
    assert actual == expected, "example merge-quorum receipt is stale; regenerate it"
