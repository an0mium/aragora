"""Tests for vertical-specific evaluation weight profiles and rubrics.

Validates that healthcare, financial, legal, and compliance weight profiles
are correctly configured with proper weights and domain-specific rubrics.
"""

import pytest

from aragora.evaluation.llm_judge import (
    EvaluationDimension,
    EvaluationRubric,
    WEIGHT_PROFILES,
    VERTICAL_RUBRICS,
)


class TestVerticalWeightProfiles:
    """Tests for domain-specific weight profiles."""

    VERTICAL_PROFILES = [
        "healthcare_hipaa",
        "healthcare_clinical",
        "financial_audit",
        "financial_risk",
        "legal_contract",
        "legal_due_diligence",
        "compliance_sox",
    ]

    @pytest.mark.parametrize("profile_name", VERTICAL_PROFILES)
    def test_profile_exists(self, profile_name):
        """Each vertical profile should be registered."""
        assert profile_name in WEIGHT_PROFILES

    @pytest.mark.parametrize("profile_name", VERTICAL_PROFILES)
    def test_weights_sum_to_one(self, profile_name):
        """Weights in each profile should sum to approximately 1.0."""
        total = sum(WEIGHT_PROFILES[profile_name].values())
        assert abs(total - 1.0) < 0.01, f"{profile_name} weights sum to {total}"

    @pytest.mark.parametrize("profile_name", VERTICAL_PROFILES)
    def test_all_dimensions_present(self, profile_name):
        """Each profile should have weights for all 8 dimensions."""
        for dim in EvaluationDimension:
            assert dim in WEIGHT_PROFILES[profile_name], (
                f"{profile_name} missing dimension {dim.value}"
            )

    @pytest.mark.parametrize("profile_name", VERTICAL_PROFILES)
    def test_no_negative_weights(self, profile_name):
        """No weight should be negative."""
        for dim, weight in WEIGHT_PROFILES[profile_name].items():
            assert weight >= 0, f"{profile_name}.{dim.value} = {weight}"

    def test_healthcare_hipaa_prioritizes_safety(self):
        """HIPAA profile should heavily weight safety and accuracy."""
        profile = WEIGHT_PROFILES["healthcare_hipaa"]
        assert profile[EvaluationDimension.SAFETY] >= 0.20
        assert profile[EvaluationDimension.ACCURACY] >= 0.20
        assert profile[EvaluationDimension.CREATIVITY] == 0.0

    def test_healthcare_clinical_prioritizes_evidence(self):
        """Clinical review profile should weight evidence highly."""
        profile = WEIGHT_PROFILES["healthcare_clinical"]
        assert profile[EvaluationDimension.EVIDENCE] >= 0.15
        assert profile[EvaluationDimension.ACCURACY] >= 0.20

    def test_financial_audit_prioritizes_accuracy(self):
        """Financial audit profile should heavily weight accuracy."""
        profile = WEIGHT_PROFILES["financial_audit"]
        assert profile[EvaluationDimension.ACCURACY] >= 0.25
        assert profile[EvaluationDimension.COMPLETENESS] >= 0.15
        assert profile[EvaluationDimension.CREATIVITY] == 0.0

    def test_legal_contract_prioritizes_completeness(self):
        """Legal contract profile should weight completeness for clause coverage."""
        profile = WEIGHT_PROFILES["legal_contract"]
        assert profile[EvaluationDimension.COMPLETENESS] >= 0.20
        assert profile[EvaluationDimension.ACCURACY] >= 0.20

    def test_compliance_sox_prioritizes_completeness_and_accuracy(self):
        """SOX compliance profile should prioritize completeness and accuracy."""
        profile = WEIGHT_PROFILES["compliance_sox"]
        assert profile[EvaluationDimension.COMPLETENESS] >= 0.20
        assert profile[EvaluationDimension.ACCURACY] >= 0.20
        assert profile[EvaluationDimension.EVIDENCE] >= 0.10

    def test_safety_critical_domains_disable_creativity(self):
        """Healthcare, financial audit, legal, and SOX profiles should disable creativity."""
        safety_critical = [
            "healthcare_hipaa",
            "financial_audit",
            "legal_contract",
            "legal_due_diligence",
            "compliance_sox",
        ]
        for name in safety_critical:
            assert WEIGHT_PROFILES[name][EvaluationDimension.CREATIVITY] == 0.0, (
                f"{name} should have 0.0 creativity weight"
            )


class TestVerticalRubrics:
    """Tests for domain-specific evaluation rubrics."""

    def test_healthcare_rubrics_exist(self):
        """Healthcare vertical should have domain-specific rubrics."""
        assert "healthcare" in VERTICAL_RUBRICS
        rubrics = VERTICAL_RUBRICS["healthcare"]
        assert EvaluationDimension.ACCURACY in rubrics
        assert EvaluationDimension.SAFETY in rubrics
        assert EvaluationDimension.COMPLETENESS in rubrics

    def test_financial_rubrics_exist(self):
        """Financial vertical should have domain-specific rubrics."""
        assert "financial" in VERTICAL_RUBRICS
        rubrics = VERTICAL_RUBRICS["financial"]
        assert EvaluationDimension.ACCURACY in rubrics
        assert EvaluationDimension.COMPLETENESS in rubrics
        assert EvaluationDimension.EVIDENCE in rubrics

    def test_legal_rubrics_exist(self):
        """Legal vertical should have domain-specific rubrics."""
        assert "legal" in VERTICAL_RUBRICS
        rubrics = VERTICAL_RUBRICS["legal"]
        assert EvaluationDimension.ACCURACY in rubrics
        assert EvaluationDimension.COMPLETENESS in rubrics
        assert EvaluationDimension.REASONING in rubrics

    def test_rubrics_are_valid_evaluation_rubric_instances(self):
        """All vertical rubrics should be EvaluationRubric instances."""
        for vertical, rubrics in VERTICAL_RUBRICS.items():
            for dim, rubric in rubrics.items():
                assert isinstance(rubric, EvaluationRubric), (
                    f"{vertical}.{dim.value} is not an EvaluationRubric"
                )
                assert rubric.dimension == dim

    def test_rubrics_have_all_score_levels(self):
        """Each rubric should have descriptions for all 5 score levels."""
        for vertical, rubrics in VERTICAL_RUBRICS.items():
            for dim, rubric in rubrics.items():
                assert rubric.score_1, f"{vertical}.{dim.value} missing score_1"
                assert rubric.score_2, f"{vertical}.{dim.value} missing score_2"
                assert rubric.score_3, f"{vertical}.{dim.value} missing score_3"
                assert rubric.score_4, f"{vertical}.{dim.value} missing score_4"
                assert rubric.score_5, f"{vertical}.{dim.value} missing score_5"

    def test_healthcare_safety_rubric_mentions_phi(self):
        """Healthcare safety rubric should reference PHI protection."""
        rubric = VERTICAL_RUBRICS["healthcare"][EvaluationDimension.SAFETY]
        prompt = rubric.to_prompt().lower()
        assert "phi" in prompt

    def test_financial_accuracy_rubric_mentions_gaap(self):
        """Financial accuracy rubric should reference GAAP/IFRS."""
        rubric = VERTICAL_RUBRICS["financial"][EvaluationDimension.ACCURACY]
        prompt = rubric.to_prompt().lower()
        assert "gaap" in prompt or "ifrs" in prompt

    def test_legal_accuracy_rubric_mentions_citations(self):
        """Legal accuracy rubric should reference legal citations."""
        rubric = VERTICAL_RUBRICS["legal"][EvaluationDimension.ACCURACY]
        prompt = rubric.to_prompt().lower()
        assert "citation" in prompt or "statute" in prompt

    def test_rubric_to_prompt_format(self):
        """Rubric to_prompt() should produce formatted scoring guide."""
        rubric = VERTICAL_RUBRICS["healthcare"][EvaluationDimension.ACCURACY]
        prompt = rubric.to_prompt()
        assert "Score 1" in prompt
        assert "Score 5" in prompt
        assert "ACCURACY" in prompt


class TestVerticalProfileIntegration:
    """Integration tests for vertical profiles with deliberation templates."""

    def test_hipaa_template_has_matching_profile(self):
        """HIPAA deliberation template should have a matching weight profile."""
        from aragora.deliberation.templates.builtins import HIPAA_COMPLIANCE

        assert HIPAA_COMPLIANCE.consensus_threshold >= 0.8
        assert "healthcare_hipaa" in WEIGHT_PROFILES

    def test_financial_audit_template_has_matching_profile(self):
        """Financial audit template should have a matching weight profile."""
        from aragora.deliberation.templates.builtins import FINANCIAL_AUDIT

        assert FINANCIAL_AUDIT.consensus_threshold >= 0.8
        assert "financial_audit" in WEIGHT_PROFILES

    def test_contract_review_template_has_matching_profile(self):
        """Contract review template should have a matching weight profile."""
        from aragora.deliberation.templates.builtins import CONTRACT_REVIEW

        assert CONTRACT_REVIEW.consensus_threshold >= 0.8
        assert "legal_contract" in WEIGHT_PROFILES

    def test_due_diligence_template_has_matching_profile(self):
        """Due diligence template should have a matching weight profile."""
        from aragora.deliberation.templates.builtins import DUE_DILIGENCE

        assert DUE_DILIGENCE.consensus_threshold >= 0.7
        assert "legal_due_diligence" in WEIGHT_PROFILES

    def test_soc2_template_has_matching_profile(self):
        """SOC 2 audit template should have a matching compliance profile."""
        from aragora.deliberation.templates.builtins import SOC2_AUDIT

        assert SOC2_AUDIT.consensus_threshold >= 0.8
        assert "compliance_sox" in WEIGHT_PROFILES

    def test_all_vertical_templates_have_high_consensus(self):
        """All vertical-specific templates should require high consensus."""
        from aragora.deliberation.templates.builtins import (
            HIPAA_COMPLIANCE,
            CLINICAL_REVIEW,
            FINANCIAL_AUDIT,
            CONTRACT_REVIEW,
        )

        for template in [HIPAA_COMPLIANCE, CLINICAL_REVIEW, FINANCIAL_AUDIT, CONTRACT_REVIEW]:
            assert template.consensus_threshold >= 0.7, (
                f"{template.name} should have consensus >= 0.7"
            )
