from __future__ import annotations

from aragora.advocates import (
    AdvocateInput,
    LocalMockUserInterestAdvocate,
    RulesUserInterestAdvocate,
)


def test_rules_advocate_blocks_human_risk_pr() -> None:
    advocate = RulesUserInterestAdvocate()
    result = advocate.evaluate(
        AdvocateInput(
            task_type="pr_triage",
            artifact_summary="Tier 4 workflow policy change",
            proposed_action="merge",
            context_features={"tier": 4, "requires_human_risk_settlement": True},
        )
    )

    assert result.decision == "block"
    assert result.confidence >= 0.9
    assert "tier" in result.cited_features


def test_rules_advocate_accepts_clean_low_risk_merge() -> None:
    advocate = RulesUserInterestAdvocate()
    result = advocate.evaluate(
        AdvocateInput(
            task_type="pr_triage",
            artifact_summary="Docs-only status update",
            proposed_action="merge",
            context_features={"tier": 0, "failing_checks": 0, "pending_checks": 0},
        )
    )

    assert result.decision == "accept"


def test_mock_local_advocate_challenges_dependabot_when_rules_are_unsure() -> None:
    advocate = LocalMockUserInterestAdvocate()
    result = advocate.evaluate(
        AdvocateInput(
            task_type="pr_triage",
            artifact_summary="Dependabot dependency update",
            proposed_action="merge",
            context_features={},
        )
    )

    assert result.decision == "challenge"
    assert result.cited_features == ("artifact_summary",)
