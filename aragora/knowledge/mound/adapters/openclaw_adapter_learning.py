"""
OpenClaw adapter learning and batch sync operations.

Bidirectional learning methods that extract patterns from OpenClaw action
outcomes and batch sync operations between KM and OpenClaw.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from aragora.knowledge.mound.adapters.openclaw_adapter_models import (
    ActionPattern,
    ActionStatus,
    KMValidationResult,
    OpenClawKMSyncResult,
    PatternType,
)

logger = logging.getLogger(__name__)


class OpenClawLearningMixin:
    """Mixin providing bidirectional learning and batch sync for OpenClawAdapter.

    Requires the host class to provide:
    - self._actions: dict[str, dict[str, Any]]
    - self._patterns: dict[str, dict[str, Any]]
    - self._km_validations: dict[str, KMValidationResult]
    - self.ID_PREFIX: str
    - self._emit_event(event_type, data): method
    """

    if TYPE_CHECKING:
        _actions: Any
        ID_PREFIX: Any
        _patterns: Any
        _emit_event: Any
        _km_validations: Any

    # =========================================================================
    # Bidirectional Learning Methods
    # =========================================================================

    async def extract_action_patterns(
        self,
        workspace_id: str,
        min_observations: int = 3,
    ) -> list[ActionPattern]:
        """
        Extract patterns from OpenClaw action outcomes.

        Analyzes action history to identify success and failure patterns
        that can be used to improve future action planning.

        Args:
            workspace_id: Workspace to analyze.
            min_observations: Minimum observations to form a pattern.

        Returns:
            List of extracted patterns.
        """
        start = time.time()
        patterns = []

        # Group actions by capability
        capability_outcomes: dict[str, list[dict[str, Any]]] = {}
        for action in self._actions.values():
            if action.get("workspace_id") != workspace_id:
                continue

            for capability in action.get("capabilities_used", []):
                if capability not in capability_outcomes:
                    capability_outcomes[capability] = []
                capability_outcomes[capability].append(action)

        # Analyze each capability
        for capability, actions in capability_outcomes.items():
            if len(actions) < min_observations:
                continue

            # Calculate success rate
            success_count = sum(1 for a in actions if a.get("result") == ActionStatus.SUCCESS.value)
            success_rate = success_count / len(actions)

            # Determine pattern type
            if success_rate >= 0.8:
                pattern_type = PatternType.SUCCESS_PATTERN
                recommendation = f"Use {capability} - high success rate ({success_rate:.1%})"
            elif success_rate <= 0.3:
                pattern_type = PatternType.FAILURE_PATTERN
                recommendation = f"Avoid {capability} alone - low success rate ({success_rate:.1%})"
            else:
                pattern_type = PatternType.CAPABILITY_PATTERN
                recommendation = (
                    f"Use {capability} with caution - moderate success ({success_rate:.1%})"
                )

            pattern = ActionPattern(
                pattern_id=f"{self.ID_PREFIX}pattern_{capability}_{workspace_id}",
                pattern_type=pattern_type,
                description=f"Pattern for capability: {capability}",
                success_rate=success_rate,
                observation_count=len(actions),
                capabilities_involved=[capability],
                recommendation=recommendation,
                confidence=min(0.9, 0.5 + len(actions) * 0.02),
                contributing_actions=[a.get("id", "") for a in actions[:10]],
            )

            patterns.append(pattern)

            # Store pattern
            self._patterns[pattern.pattern_id] = pattern.to_dict()

        self._emit_event(
            "km_openclaw_patterns_extracted",
            {
                "workspace_id": workspace_id,
                "patterns_count": len(patterns),
            },
        )

        logger.info(
            f"Extracted {len(patterns)} patterns for workspace {workspace_id} "
            f"in {(time.time() - start) * 1000:.1f}ms"
        )

        return patterns

    async def get_failure_patterns(
        self,
        workspace_id: str,
        limit: int = 10,
    ) -> list[ActionPattern]:
        """
        Get failure patterns to avoid in future actions.

        Args:
            workspace_id: Workspace to query.
            limit: Maximum patterns to return.

        Returns:
            List of failure patterns.
        """
        patterns = []

        for pattern_data in self._patterns.values():
            if pattern_data.get("pattern_type") == PatternType.FAILURE_PATTERN.value:
                pattern = ActionPattern.from_dict(pattern_data)
                patterns.append(pattern)

        # Sort by observation count (most observed first)
        patterns.sort(key=lambda p: p.observation_count, reverse=True)

        return patterns[:limit]

    async def get_success_patterns(
        self,
        workspace_id: str,
        limit: int = 10,
    ) -> list[ActionPattern]:
        """
        Get success patterns to replicate in future actions.

        Args:
            workspace_id: Workspace to query.
            limit: Maximum patterns to return.

        Returns:
            List of success patterns.
        """
        patterns = []

        for pattern_data in self._patterns.values():
            if pattern_data.get("pattern_type") == PatternType.SUCCESS_PATTERN.value:
                pattern = ActionPattern.from_dict(pattern_data)
                patterns.append(pattern)

        # Sort by success rate and confidence
        patterns.sort(
            key=lambda p: (p.success_rate * p.confidence),
            reverse=True,
        )

        return patterns[:limit]

    async def cross_debate_learning(
        self,
        debate_id: str,
        action_outcomes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Learn from action outcomes across debates.

        Analyzes how actions performed in the context of debates
        to improve future debate-driven task execution.

        Args:
            debate_id: The debate ID.
            action_outcomes: List of action outcome dicts.

        Returns:
            Learning results with patterns and recommendations.
        """
        results: dict[str, Any] = {
            "debate_id": debate_id,
            "actions_analyzed": len(action_outcomes),
            "patterns_updated": 0,
            "recommendations": [],
        }

        for outcome in action_outcomes:
            action_id = outcome.get("action_id")
            if not action_id:
                continue

            # Create/update validation
            validation = KMValidationResult(
                action_id=action_id,
                km_confidence=outcome.get("confidence", 0.5),
                cross_debate_utility=outcome.get("utility", 0.0),
                validation_count=self._km_validations.get(
                    action_id, KMValidationResult(action_id=action_id)
                ).validation_count
                + 1,
                was_supported=outcome.get("was_supported", False),
                was_contradicted=outcome.get("was_contradicted", False),
            )

            self._km_validations[action_id] = validation

            # Update action in storage
            if action_id in self._actions:
                self._actions[action_id]["km_validated"] = True
                self._actions[action_id]["km_confidence"] = validation.km_confidence
                self._actions[action_id]["cross_debate_utility"] = validation.cross_debate_utility

        # Generate recommendations based on validations
        high_utility_actions = [
            v for v in self._km_validations.values() if v.cross_debate_utility >= 0.7
        ]

        if high_utility_actions:
            results["recommendations"].append(
                f"Found {len(high_utility_actions)} high-utility actions for cross-debate reuse"
            )

        self._emit_event(
            "km_openclaw_cross_debate_learning",
            {
                "debate_id": debate_id,
                "actions_analyzed": len(action_outcomes),
                "high_utility_count": len(high_utility_actions),
            },
        )

        return results

    # =========================================================================
    # Batch Sync Operations
    # =========================================================================

    async def sync_validations_from_km(
        self,
        km_items: list[dict[str, Any]],
        min_confidence: float = 0.7,
    ) -> OpenClawKMSyncResult:
        """
        Batch sync KM validations back to OpenClaw.

        Processes KM items to:
        - Validate action outcomes
        - Extract new patterns
        - Push context updates to OpenClaw

        Args:
            km_items: KM items with validation data.
            min_confidence: Minimum confidence for applying changes.

        Returns:
            OpenClawKMSyncResult with sync results.
        """
        start_time = time.time()
        result = OpenClawKMSyncResult()

        for item in km_items:
            try:
                metadata = item.get("metadata", {})
                action_id = metadata.get("openclaw_action_id")

                if not action_id:
                    continue

                result.actions_analyzed += 1

                confidence = item.get("confidence", 0.5)
                if confidence < min_confidence:
                    continue

                # Create validation
                validation = KMValidationResult(
                    action_id=action_id,
                    km_confidence=confidence,
                    was_supported=metadata.get("was_supported", False),
                    was_contradicted=metadata.get("was_contradicted", False),
                )

                self._km_validations[action_id] = validation

                # Update action if exists
                action_key = None
                for key, action in self._actions.items():
                    if action.get("action_id") == action_id:
                        action_key = key
                        break

                if action_key:
                    self._actions[action_key]["km_validated"] = True
                    self._actions[action_key]["km_confidence"] = confidence
                    result.actions_updated += 1

            except Exception as e:
                error_msg = f"Error processing KM item: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)

        result.duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"KM to OpenClaw sync complete: "
            f"analyzed={result.actions_analyzed}, updated={result.actions_updated}, "
            f"errors={len(result.errors)}, duration={result.duration_ms:.1f}ms"
        )

        return result


__all__ = [
    "OpenClawLearningMixin",
]
