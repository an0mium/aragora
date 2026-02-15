"""
Compliance Artifact Hook - Auto-generate compliance artifacts after debates.

Generates EU AI Act (Art 12/13/14) compliance artifact bundles for
decisions that meet a risk threshold, following the ReceiptDeliveryHook pattern.

Usage:
    from aragora.debate.hooks.compliance_artifact_hook import create_compliance_artifact_hook
    from aragora.debate.hooks import HookManager

    hook_manager = HookManager()
    hook = create_compliance_artifact_hook(frameworks=["eu_ai_act"])
    hook_manager.register("post_debate", hook.on_post_debate)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.debate.context import DebateContext
    from aragora.core_types import DebateResult

logger = logging.getLogger(__name__)


class ComplianceArtifactHook:
    """Hook for automatic compliance artifact generation after debates.

    Subscribes to POST_DEBATE events and generates compliance artifacts
    when the debate risk level exceeds the configured threshold.
    """

    def __init__(
        self,
        frameworks: list[str] | None = None,
        min_risk_level: str = "HIGH",
        enabled: bool = True,
    ):
        """Initialize the compliance artifact hook.

        Args:
            frameworks: Compliance frameworks to generate for (default: ["eu_ai_act"])
            min_risk_level: Minimum risk level to trigger generation (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)
            enabled: Whether the hook is active
        """
        self.frameworks = frameworks or ["eu_ai_act"]
        self.min_risk_level = min_risk_level
        self.enabled = enabled
        self._risk_order = ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def _meets_risk_threshold(self, risk_level: str) -> bool:
        """Check if risk level meets the minimum threshold."""
        try:
            level_idx = self._risk_order.index(risk_level.upper())
            min_idx = self._risk_order.index(self.min_risk_level.upper())
            return level_idx >= min_idx
        except ValueError:
            return False

    def on_post_debate(self, ctx: DebateContext, result: DebateResult) -> None:
        """Generate compliance artifacts after debate completion.

        Args:
            ctx: Debate context
            result: Debate result
        """
        if not self.enabled:
            return

        try:
            # Build a receipt dict from the result for risk classification
            receipt_dict = self._build_receipt_dict(ctx, result)

            # Classify risk level
            risk_level = self._classify_risk(receipt_dict)

            if not self._meets_risk_threshold(risk_level):
                logger.debug(
                    "Skipping compliance artifacts: risk=%s below threshold=%s",
                    risk_level,
                    self.min_risk_level,
                )
                return

            # Generate compliance artifacts
            self._generate_artifacts(receipt_dict, risk_level, ctx)

        except ImportError:
            logger.debug("Compliance modules not available for artifact generation")
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning("Compliance artifact generation failed: %s", e)

    def _build_receipt_dict(self, ctx: DebateContext, result: DebateResult) -> dict[str, Any]:
        """Build a receipt-like dict from debate context and result."""
        return {
            "debate_id": getattr(ctx, "debate_id", ""),
            "task": getattr(result, "task", ""),
            "verdict": "PASS" if getattr(result, "consensus_reached", False) else "CONDITIONAL",
            "confidence": getattr(result, "confidence", 0.0),
            "agents_used": [getattr(a, "name", str(a)) for a in getattr(result, "agents", [])],
            "rounds": getattr(result, "total_rounds", 0),
            "consensus_reached": getattr(result, "consensus_reached", False),
            "final_answer": getattr(result, "final_answer", ""),
        }

    def _classify_risk(self, receipt_dict: dict[str, Any]) -> str:
        """Classify risk level of the decision."""
        try:
            from aragora.compliance.risk_classifier import RiskClassifier

            classifier = RiskClassifier()
            return classifier.classify_receipt(receipt_dict)
        except (ImportError, AttributeError):
            # Default to HIGH if classifier unavailable (conservative)
            confidence = receipt_dict.get("confidence", 0.0)
            if confidence < 0.5:
                return "HIGH"
            return "MEDIUM"

    def _generate_artifacts(
        self,
        receipt_dict: dict[str, Any],
        risk_level: str,
        ctx: DebateContext,
    ) -> None:
        """Generate and store compliance artifacts."""
        from aragora.compliance.artifact_generator import ComplianceArtifactGenerator

        generator = ComplianceArtifactGenerator()
        bundle = generator.generate(receipt_dict)

        logger.info(
            "Generated compliance artifacts for debate %s: risk=%s, hash=%s",
            receipt_dict.get("debate_id", "unknown"),
            risk_level,
            getattr(bundle, "integrity_hash", "N/A")[:16],
        )

        # Emit event if emitter available
        try:
            from aragora.events.types import StreamEvent, StreamEventType

            if hasattr(ctx, "event_emitter") and ctx.event_emitter:
                event = StreamEvent(
                    type=StreamEventType.COMPLIANCE_ARTIFACT_GENERATED,
                    data={
                        "debate_id": receipt_dict.get("debate_id", ""),
                        "risk_level": risk_level,
                        "frameworks": self.frameworks,
                        "integrity_hash": getattr(bundle, "integrity_hash", ""),
                    },
                )
                ctx.event_emitter.emit(event)
        except (ImportError, AttributeError):
            pass


def create_compliance_artifact_hook(
    frameworks: list[str] | None = None,
    min_risk_level: str = "HIGH",
    enabled: bool = True,
) -> ComplianceArtifactHook:
    """Create a compliance artifact hook.

    Args:
        frameworks: Compliance frameworks (default: ["eu_ai_act"])
        min_risk_level: Minimum risk level to trigger (default: HIGH)
        enabled: Whether the hook is active

    Returns:
        Configured ComplianceArtifactHook
    """
    return ComplianceArtifactHook(
        frameworks=frameworks,
        min_risk_level=min_risk_level,
        enabled=enabled,
    )
