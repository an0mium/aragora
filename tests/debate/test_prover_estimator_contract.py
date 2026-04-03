"""Contract test for prover-estimator consensus mode."""

from __future__ import annotations

from typing import Any

import pytest

from aragora.debate.consensus import build_proof_from_prover_estimator
from aragora.debate.prover_estimator import ProverEstimatorEngine


class ScriptedAgent:
    """Deterministic mock agent that records which stage was invoked."""

    def __init__(self, responses: dict[str, str], stage_order: list[str]):
        self._responses = responses
        self._stage_order = stage_order

    async def generate(self, prompt: str) -> str:
        if prompt.startswith("You are a Prover in a truth-seeking debate protocol."):
            stage = "decompose"
        elif prompt.startswith("You are the Prover responding to probability estimates."):
            stage = "challenge"
        elif prompt.startswith("You are an Estimator in a truth-seeking debate protocol."):
            stage = "estimate"
        elif prompt.startswith(
            "You are the Estimator re-evaluating after receiving evidence-based challenges."
        ):
            stage = "reestimate"
        else:
            raise AssertionError(f"Unexpected prompt:\n{prompt}")

        self._stage_order.append(stage)
        return self._responses[stage]


@pytest.mark.asyncio
async def test_prover_estimator_contract_executes_five_stages_in_order():
    stage_order: list[str] = []

    prover = ScriptedAgent(
        responses={
            "decompose": (
                "SUBCLAIM [A]: The migration requires online schema compatibility\n"
                "IMPORTANCE: 0.9\n"
                "EVIDENCE: Rolling migrations depend on backward-compatible reads\n"
                "DEPENDS_ON: none\n"
                "\n"
                "SUBCLAIM [B]: The rollout needs an observability gate\n"
                "IMPORTANCE: 0.7\n"
                "EVIDENCE: Canary metrics catch regressions before global impact\n"
                "DEPENDS_ON: A\n"
            ),
            "challenge": (
                "CHALLENGE [A]:\n"
                "TYPE: evidence\n"
                "EVIDENCE: The runbook requires dual-read support during deploy\n"
                "REVISED_PROBABILITY: 0.82\n"
            ),
        },
        stage_order=stage_order,
    )
    estimator = ScriptedAgent(
        responses={
            "estimate": (
                "ESTIMATE [A]:\n"
                "PROBABILITY: 0.55\n"
                "REASONING: Compatibility is plausible but not yet demonstrated\n"
                "CONFIDENCE: 0.61\n"
                "OBFUSCATION: NO\n"
                "\n"
                "ESTIMATE [B]:\n"
                "PROBABILITY: 0.74\n"
                "REASONING: Observability gating is a common production control\n"
                "CONFIDENCE: 0.66\n"
                "OBFUSCATION: NO\n"
            ),
            "reestimate": (
                "REESTIMATE [A]:\n"
                "PROBABILITY: 0.79\n"
                "REASONING: The dual-read requirement materially strengthens compatibility\n"
                "CONFIDENCE: 0.81\n"
                "OBFUSCATION: NO\n"
            ),
        },
        stage_order=stage_order,
    )

    engine = ProverEstimatorEngine(
        prover=prover,
        estimator=estimator,
        max_challenge_rounds=1,
    )

    original_aggregate = engine._aggregate_confidence

    def track_aggregate(subclaims: list[Any], estimates: list[Any]) -> float:
        stage_order.append("aggregate")
        return original_aggregate(subclaims, estimates)

    engine._aggregate_confidence = track_aggregate

    result = await engine.run("Can we safely roll out the migration without downtime?")
    result.debate_id = "pe-contract-2029"
    proof = build_proof_from_prover_estimator(result)

    assert stage_order == [
        "decompose",
        "estimate",
        "challenge",
        "reestimate",
        "aggregate",
    ]
    assert [subclaim.id for subclaim in result.subclaims] == ["A", "B"]
    assert [challenge.subclaim_id for challenge in result.challenges] == ["A"]

    assert proof.metadata["consensus_mode"] == "prover_estimator"
    assert proof.metadata["subclaim_count"] == 2
    assert proof.metadata["challenge_count"] == 1
    assert proof.metadata["estimator_confidence_scores"] == {
        "A": pytest.approx(0.81),
        "B": pytest.approx(0.66),
    }

    estimator_evidence = {
        evidence.metadata["subclaim_id"]: evidence
        for evidence in proof.evidence_chain
        if evidence.source == "estimator"
    }
    assert estimator_evidence["A"].metadata["estimator_confidence"] == pytest.approx(0.81)
    assert estimator_evidence["B"].metadata["estimator_confidence"] == pytest.approx(0.66)
