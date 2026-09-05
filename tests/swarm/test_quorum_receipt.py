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


def test_v02_bridge_retains_pass_findings_and_observations():
    from aragora.gauntlet.odr_verify import verify_odr_document
    from aragora.swarm.quorum_evidence import ReviewerResult

    outcome = _outcome()
    outcome.items[0].body = "- [P3] advisory from a PASS reviewer"
    outcome.failures = [ReviewerResult(family="grok", ok=False, text="", error="transport boom")]
    outcome.timed_out_families = ["openai"]
    doc = decision_receipt_to_odr(collect_outcome_to_decision_receipt(outcome))
    assert len(doc["quorum"]["verdicts"]) == 3
    assert doc["subject"]["pr_number"] == outcome.pr
    assert doc["quorum"]["rule"]["required_signals"] == 2
    dissent = doc["quorum"]["dissent"]
    assert [f["severity"] for f in dissent["findings"]] == ["P3", "P1"]
    assert dissent["blocking"] is True and dissent["severity_max"] == "P1"
    assert {o["kind"] for o in doc["reasoning"]["observations"]} == {"failure", "timeout"}
    assert verify_odr_document(doc).ok
    import json
    from pathlib import Path

    legacy = json.loads(
        Path("docs/specs/examples/example-merge-quorum-receipt.odr.json").read_text()
    )
    assert legacy["odr_version"] == "0.1" and verify_odr_document(legacy).ok


def test_v02_bridge_dict_preserves_timeout_and_adjudication():
    from aragora.swarm.review_adjudicator import AdjudicationResult, AdjudicationVerdict

    outcome = _outcome().to_dict()
    outcome["timed_out_families"] = ["openai"]
    outcome["adjudication"] = AdjudicationResult(
        verdict=AdjudicationVerdict.SETTLE, reason="resolved"
    ).to_receipt_dict()
    doc = decision_receipt_to_odr(collect_outcome_to_decision_receipt(outcome))
    assert doc["reasoning"]["observations"][0]["kind"] == "timeout"
    assert doc["adjudication"]["verdict"] == "settle"
    jsonschema.validate(doc, load_odr_schema())
    legacy = decision_receipt_to_odr(
        DecisionReceipt.from_dict({"receipt_id": "legacy", "verdict": "PASS"})
    )
    assert "adjudication" not in legacy
    legacy.update(odr_version="0.1", profile="https://aragora.ai/specs/open-decision-receipt/v0.1")
    jsonschema.validate(legacy, load_odr_schema())


def test_v02_observations_without_summary_preserve_legacy_absence():
    from aragora.swarm.quorum_evidence import ReviewerResult

    legacy = decision_receipt_to_odr(DecisionReceipt.from_dict({"receipt_id": "empty"}))
    assert legacy["reasoning"]["status"] == "absent"
    legacy.update(odr_version="0.1", profile="https://aragora.ai/specs/open-decision-receipt/v0.1")
    jsonschema.validate(legacy, load_odr_schema())
    outcome = _outcome()
    outcome.action_reason = ""
    outcome.failures = [ReviewerResult(family="grok", ok=False, text="", error="transport boom")]
    doc = decision_receipt_to_odr(collect_outcome_to_decision_receipt(outcome))
    assert doc["reasoning"]["observations"][0]["detail"] == "transport boom"
    jsonschema.validate(doc, load_odr_schema())


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
    # Freeze the published v0.1 content; v0.2 adds audit detail without rewriting it.
    import json
    from pathlib import Path

    example = Path("docs/specs/examples/example-merge-quorum-receipt.odr.json")
    expected = decision_receipt_to_odr(collect_outcome_to_decision_receipt(_outcome()))
    actual = json.loads(example.read_text(encoding="utf-8"))
    from aragora.gauntlet.odr_verify import verify_odr_document

    assert verify_odr_document(actual).ok and verify_odr_document(expected).ok
    assert actual["odr_version"] == "0.1" and expected["odr_version"] == "0.2"
    expected.update(odr_version=actual["odr_version"], profile=actual["profile"])
    for key in ("repository", "pr_number", "head_sha"):
        expected["subject"].pop(key)
    for key in ("verdicts", "rule"):
        expected["quorum"].pop(key)
    for key in ("findings", "severity_max", "blocking"):
        expected["quorum"]["dissent"].pop(key)
    expected["attestation"].pop("mechanism")
    assert actual == expected
