"""
EU AI Act Compliance Handler.

Provides REST API endpoints for EU AI Act compliance:
- Risk classification of AI use cases
- Conformity report generation from decision receipts
- Full compliance artifact bundle generation (Articles 12, 13, 14)

Endpoints:
    POST /api/v2/compliance/eu-ai-act/classify        - Classify use case risk level
    POST /api/v2/compliance/eu-ai-act/audit            - Generate conformity report
    POST /api/v2/compliance/eu-ai-act/generate-bundle  - Generate full artifact bundle
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.handlers.base import HandlerResult, error_response, json_response
from aragora.rbac.decorators import require_permission
from aragora.observability.metrics import track_handler

logger = logging.getLogger(__name__)

# Lazy-load EU AI Act module to avoid import-time overhead
_classifier = None
_report_generator = None
_artifact_generator = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        from aragora.compliance.eu_ai_act import RiskClassifier

        _classifier = RiskClassifier()
    return _classifier


def _get_report_generator():
    global _report_generator
    if _report_generator is None:
        from aragora.compliance.eu_ai_act import ConformityReportGenerator

        _report_generator = ConformityReportGenerator()
    return _report_generator


def _get_artifact_generator(**kwargs):
    global _artifact_generator
    if _artifact_generator is None or kwargs:
        from aragora.compliance.eu_ai_act import ComplianceArtifactGenerator

        if kwargs:
            return ComplianceArtifactGenerator(**kwargs)
        _artifact_generator = ComplianceArtifactGenerator()
    return _artifact_generator


class EUAIActMixin:
    """Mixin providing EU AI Act compliance handler methods.

    Endpoints:
        POST /api/v2/compliance/eu-ai-act/classify
        POST /api/v2/compliance/eu-ai-act/audit
        POST /api/v2/compliance/eu-ai-act/generate-bundle
    """

    @track_handler("compliance/eu-ai-act/classify", method="POST")
    @require_permission("compliance:read")
    async def _eu_ai_act_classify(self, body: dict[str, Any]) -> HandlerResult:
        """Classify an AI use case by EU AI Act risk level.

        Body:
            description (str): Free-text description of the AI use case.

        Returns:
            RiskClassification with risk level, rationale, obligations.
        """
        description = body.get("description", "").strip()
        if not description:
            return error_response("Missing required field: 'description'", 400)

        classifier = _get_classifier()
        classification = classifier.classify(description)

        return json_response(
            {
                "classification": classification.to_dict(),
            }
        )

    @track_handler("compliance/eu-ai-act/audit", method="POST")
    @require_permission("compliance:read")
    async def _eu_ai_act_audit(self, body: dict[str, Any]) -> HandlerResult:
        """Generate a conformity report from a decision receipt.

        Body:
            receipt (dict): Decision receipt data (receipt_id, input_summary,
                verdict, verdict_reasoning, agents, etc.)

        Returns:
            ConformityReport with article-by-article assessment.
        """
        receipt = body.get("receipt")
        if not receipt or not isinstance(receipt, dict):
            return error_response("Missing required field: 'receipt' (must be a dict)", 400)

        generator = _get_report_generator()
        report = generator.generate(receipt)

        return json_response(
            {
                "conformity_report": report.to_dict(),
            }
        )

    @track_handler("compliance/eu-ai-act/generate-bundle", method="POST")
    @require_permission("compliance:write")
    async def _eu_ai_act_generate_bundle(self, body: dict[str, Any]) -> HandlerResult:
        """Generate a full EU AI Act compliance artifact bundle.

        Produces Articles 12 (Record-Keeping), 13 (Transparency), and
        14 (Human Oversight) artifacts bundled with a conformity report
        and SHA-256 integrity hash.

        Body:
            receipt (dict): Decision receipt data.
            provider_name (str, optional): Provider organization name.
            provider_contact (str, optional): Provider contact email.
            eu_representative (str, optional): EU representative name.
            system_name (str, optional): AI system name.
            system_version (str, optional): AI system version.

        Returns:
            ComplianceArtifactBundle with all articles and integrity hash.
        """
        receipt = body.get("receipt")
        if not receipt or not isinstance(receipt, dict):
            return error_response("Missing required field: 'receipt' (must be a dict)", 400)

        # Extract optional provider customization
        gen_kwargs: dict[str, str] = {}
        for key in (
            "provider_name",
            "provider_contact",
            "eu_representative",
            "system_name",
            "system_version",
        ):
            val = body.get(key)
            if val and isinstance(val, str):
                gen_kwargs[key] = val

        generator = _get_artifact_generator(**gen_kwargs)
        bundle = generator.generate(receipt)

        return json_response(
            {
                "bundle": bundle.to_dict(),
            }
        )
