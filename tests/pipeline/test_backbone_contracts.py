from __future__ import annotations

from aragora.interrogation.crystallizer import CrystallizedSpec, MoSCoWItem
from aragora.interrogation.engine import InterrogationResult, PrioritizedQuestion
from aragora.pipeline.backbone_contracts import (
    IntakeBundle,
    OutcomeFeedbackRecord,
    ReceiptEnvelope,
    SpecBundle,
)
from aragora.pipeline.outcome_feedback import PipelineOutcome
from aragora.prompt_engine.spec_validator import ValidationResult, ValidatorRole
from aragora.prompt_engine.types import PromptIntent, RiskItem, Specification, SpecFile


def test_intake_bundle_from_prompt_intent_preserves_core_fields() -> None:
    intent = PromptIntent(
        raw_prompt="Improve onboarding flow",
        intent_type="improvement",
        related_knowledge=[{"source": "km", "id": "doc-1"}],
    )

    bundle = IntakeBundle.from_prompt_intent(
        intent,
        trust_tiers=["operator-authored", "internal-retrieved"],
        taint_flags=["external_note"],
        origin_metadata={"entrypoint": "prompt_engine"},
    )

    assert bundle.raw_intent == "Improve onboarding flow"
    assert bundle.context_refs == [{"source": "km", "id": "doc-1"}]
    assert bundle.trust_tiers == ["operator-authored", "internal-retrieved"]
    assert bundle.taint_flags == ["external_note"]
    assert bundle.origin_metadata["entrypoint"] == "prompt_engine"


def test_spec_bundle_from_prompt_spec_surfaces_missing_execution_fields() -> None:
    spec = Specification(
        title="Onboarding improvements",
        problem_statement="Users drop off too early.",
        proposed_solution="Tighten the onboarding flow and clarify next steps.",
        success_criteria=["Increase activation conversion", "Reduce first-session confusion"],
        file_changes=[
            SpecFile(
                path="aragora/live/src/app/(app)/onboarding/page.tsx",
                action="modify",
                description="Improve onboarding copy",
            )
        ],
        risks=[
            RiskItem(
                description="UX regression",
                likelihood="medium",
                impact="medium",
                mitigation="Keep old copy behind a rollout guard.",
            )
        ],
        confidence=0.72,
    )
    validation = ValidationResult(
        role_results={ValidatorRole.UX_ADVOCATE: {"passed": True, "confidence": 0.9}},
        overall_confidence=0.91,
        passed=True,
    )

    bundle = SpecBundle.from_prompt_spec(spec, validation=validation)

    assert bundle.title == "Onboarding improvements"
    assert bundle.objectives == ["Tighten the onboarding flow and clarify next steps."]
    assert bundle.acceptance_criteria == [
        "Increase activation conversion",
        "Reduce first-session confusion",
    ]
    assert bundle.verification_plan == bundle.acceptance_criteria
    assert bundle.rollback_plan == ["Keep old copy behind a rollout guard."]
    assert bundle.owner_file_scopes == ["aragora/live/src/app/(app)/onboarding/page.tsx"]
    assert bundle.confidence == 0.91
    assert bundle.source_kind == "prompt_engine_spec"
    assert bundle.missing_required_fields == ["constraints"]
    assert bundle.is_execution_grade is False


def test_spec_bundle_from_interrogation_result_preserves_constraints_and_open_questions() -> None:
    crystallized = CrystallizedSpec(
        title="Execution-grade spec",
        problem_statement="Turn a vague request into a concrete implementation plan.",
        requirements=[
            MoSCoWItem(description="Capture owner files", priority="must"),
            MoSCoWItem(description="Define rollback path", priority="must"),
        ],
        success_criteria=["Every generated task has acceptance criteria"],
        risks=[
            {"risk": "Spec drift", "mitigation": "Block execution when required fields are absent"}
        ],
        constraints=["No direct prompt-to-execute path"],
    )
    result = InterrogationResult(
        original_prompt="Make the pipeline safer",
        dimensions=["safety", "execution"],
        research_summary="Current path allows soft degradation.",
        prioritized_questions=[
            PrioritizedQuestion(
                question="Should automated runs fail closed on missing rollback plans?",
                why_it_matters="Execution safety depends on it.",
                priority_score=0.95,
            )
        ],
        crystallized_spec=crystallized,
    )

    bundle = SpecBundle.from_interrogation_result(result)

    assert bundle.title == "Execution-grade spec"
    assert bundle.constraints == ["No direct prompt-to-execute path"]
    assert bundle.acceptance_criteria == ["Every generated task has acceptance criteria"]
    assert bundle.rollback_plan == ["Block execution when required fields are absent"]
    assert bundle.open_questions == ["Should automated runs fail closed on missing rollback plans?"]
    assert bundle.owner_file_scopes == []
    assert "owner_file_scopes" in bundle.missing_required_fields


def test_receipt_envelope_from_pipeline_receipt_flattens_provenance() -> None:
    receipt = {
        "receipt_id": "receipt-123",
        "pipeline_id": "pipe-001",
        "generated_at": "2026-03-06T12:00:00Z",
        "content_hash": "abc123",
        "provenance": {
            "ideas": [{"id": "i1", "label": "Idea"}],
            "goals": [{"id": "g1", "label": "Goal"}],
        },
        "execution": {"status": "completed"},
    }

    envelope = ReceiptEnvelope.from_pipeline_receipt(
        receipt,
        policy_gate_result={"allowed": True},
        taint_summary={"tainted": False},
    )

    assert envelope.receipt_id == "receipt-123"
    assert envelope.artifact_hash == "abc123"
    assert envelope.verdict == "pass"
    assert envelope.policy_gate_result == {"allowed": True}
    assert envelope.taint_summary == {"tainted": False}
    assert envelope.provenance_chain == [
        {"stage": "ideas", "id": "i1", "label": "Idea"},
        {"stage": "goals", "id": "g1", "label": "Goal"},
    ]


def test_outcome_feedback_record_from_pipeline_outcome_derives_next_action() -> None:
    outcome = PipelineOutcome(
        pipeline_id="pipe-001",
        run_type="user_project",
        domain="product",
        spec_completeness=0.8,
        execution_succeeded=False,
        tests_passed=3,
        tests_failed=2,
        files_changed=1,
        total_duration_s=42.0,
    )

    record = OutcomeFeedbackRecord.from_pipeline_outcome(
        outcome,
        receipt_ref="receipt-123",
    )

    assert record.receipt_ref == "receipt-123"
    assert record.pipeline_id == "pipe-001"
    assert record.objective_fidelity == outcome.overall_quality_score
    assert record.execution_outcome["tests_failed"] == 2
    assert record.next_action_recommendation == "run_bug_fix_loop"
