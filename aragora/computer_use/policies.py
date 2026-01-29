"""
Computer-Use Policy Definitions.

Extends sandbox policies for UI-specific controls:
- Action type allowlists/denylists
- Domain/URL restrictions
- Element type controls (inputs, buttons, etc.)
- Text pattern validation
- Coordinate boundary enforcement

Safety: All actions are validated against policies before execution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aragora.computer_use.actions import Action, ActionType, ClickAction, TypeAction

logger = logging.getLogger(__name__)


class PolicyDecision(str, Enum):
    """Decision made by policy check."""

    ALLOW = "allow"
    DENY = "deny"
    AUDIT = "audit"  # Allow but log
    REQUIRE_APPROVAL = "require_approval"


@dataclass
class ActionRule:
    """Rule for action type control."""

    action_type: ActionType
    decision: PolicyDecision = PolicyDecision.ALLOW
    reason: str = ""
    max_per_session: int = 0  # 0 = unlimited
    cooldown_seconds: float = 0.0  # Minimum time between actions


@dataclass
class DomainRule:
    """Rule for URL/domain restrictions."""

    pattern: str  # Regex pattern for URL
    decision: PolicyDecision = PolicyDecision.ALLOW
    reason: str = ""
    allow_subpaths: bool = True

    def matches(self, url: str) -> bool:
        """Check if rule matches URL."""
        return bool(re.match(self.pattern, url, re.IGNORECASE))


@dataclass
class ElementRule:
    """Rule for UI element interaction."""

    element_type: str  # input, button, link, textarea, select, etc.
    decision: PolicyDecision = PolicyDecision.ALLOW
    reason: str = ""
    allow_with_patterns: list[str] = field(default_factory=list)  # Allowed text patterns
    deny_with_patterns: list[str] = field(default_factory=list)  # Denied text patterns


@dataclass
class TextPatternRule:
    """Rule for text input validation."""

    pattern: str  # Regex pattern
    decision: PolicyDecision = PolicyDecision.DENY
    reason: str = ""
    applies_to: list[ActionType] = field(default_factory=lambda: [ActionType.TYPE])


@dataclass
class BoundaryRule:
    """Rule for coordinate boundaries."""

    min_x: int = 0
    max_x: int = 3840  # 4K width
    min_y: int = 0
    max_y: int = 2160  # 4K height
    reason: str = "Screen boundary enforcement"


@dataclass
class RateLimits:
    """Rate limits for computer-use actions."""

    max_actions_per_minute: int = 60
    max_clicks_per_minute: int = 30
    max_keystrokes_per_minute: int = 300
    max_screenshots_per_minute: int = 20
    cooldown_after_error_seconds: float = 2.0


@dataclass
class ComputerPolicy:
    """
    Complete policy for computer-use actions.

    Combines action rules, domain rules, element rules, and rate limits
    to define what a computer-use session can do.
    """

    name: str
    description: str = ""

    # Default decision
    default_decision: PolicyDecision = PolicyDecision.DENY

    # Rules
    action_rules: list[ActionRule] = field(default_factory=list)
    domain_rules: list[DomainRule] = field(default_factory=list)
    element_rules: list[ElementRule] = field(default_factory=list)
    text_pattern_rules: list[TextPatternRule] = field(default_factory=list)
    boundary_rule: BoundaryRule = field(default_factory=BoundaryRule)

    # Rate limits
    rate_limits: RateLimits = field(default_factory=RateLimits)

    # Safety settings
    require_screenshot_before_action: bool = True
    require_human_approval_for_sensitive: bool = True
    block_credential_fields: bool = True
    audit_all_actions: bool = True

    # Session limits
    max_actions_per_task: int = 100
    max_consecutive_errors: int = 3
    timeout_per_action_seconds: float = 10.0
    total_timeout_seconds: float = 300.0

    def add_action_allowlist(
        self,
        action_types: list[ActionType],
        reason: str = "",
    ) -> None:
        """Add action types to allowlist."""
        for action_type in action_types:
            self.action_rules.append(
                ActionRule(
                    action_type=action_type,
                    decision=PolicyDecision.ALLOW,
                    reason=reason,
                )
            )

    def add_action_denylist(
        self,
        action_types: list[ActionType],
        reason: str = "",
    ) -> None:
        """Add action types to denylist."""
        for action_type in action_types:
            self.action_rules.append(
                ActionRule(
                    action_type=action_type,
                    decision=PolicyDecision.DENY,
                    reason=reason,
                )
            )

    def add_domain_allowlist(
        self,
        patterns: list[str],
        reason: str = "",
    ) -> None:
        """Add URL patterns to allowlist."""
        for pattern in patterns:
            self.domain_rules.append(
                DomainRule(
                    pattern=pattern,
                    decision=PolicyDecision.ALLOW,
                    reason=reason,
                )
            )

    def add_domain_denylist(
        self,
        patterns: list[str],
        reason: str = "",
    ) -> None:
        """Add URL patterns to denylist."""
        for pattern in patterns:
            self.domain_rules.append(
                DomainRule(
                    pattern=pattern,
                    decision=PolicyDecision.DENY,
                    reason=reason,
                )
            )

    def add_sensitive_text_patterns(
        self,
        patterns: list[str],
        reason: str = "",
    ) -> None:
        """Add patterns that should never be typed."""
        for pattern in patterns:
            self.text_pattern_rules.append(
                TextPatternRule(
                    pattern=pattern,
                    decision=PolicyDecision.DENY,
                    reason=reason,
                )
            )


class ComputerPolicyChecker:
    """
    Validates actions against a computer-use policy.

    Provides consistent policy enforcement with audit logging.
    """

    def __init__(self, policy: ComputerPolicy):
        self.policy = policy
        self._audit_log: list[dict[str, Any]] = []
        self._action_counts: dict[ActionType, int] = {}
        self._last_action_time: dict[ActionType, float] = {}
        self._total_actions = 0
        self._consecutive_errors = 0

    def check_action(
        self,
        action: Action,
        current_url: str | None = None,
    ) -> tuple[bool, str]:
        """
        Check if an action is allowed.

        Args:
            action: The action to validate
            current_url: Current browser URL (if applicable)

        Returns:
            (allowed, reason) tuple
        """
        # Check session limits
        if self._total_actions >= self.policy.max_actions_per_task:
            return False, f"Exceeded max actions per task ({self.policy.max_actions_per_task})"

        if self._consecutive_errors >= self.policy.max_consecutive_errors:
            return False, f"Too many consecutive errors ({self.policy.max_consecutive_errors})"

        # Check action type rules
        allowed, reason = self._check_action_type(action.action_type)
        if not allowed:
            self._log_check(action, allowed, reason)
            return allowed, reason

        # Check domain rules if URL provided
        if current_url:
            allowed, reason = self._check_domain(current_url)
            if not allowed:
                self._log_check(action, allowed, reason)
                return allowed, reason

        # Check coordinate boundaries for click/move actions
        if isinstance(action, ClickAction):
            allowed, reason = self._check_coordinates(action.x, action.y)
            if not allowed:
                self._log_check(action, allowed, reason)
                return allowed, reason

        # Check text patterns for type actions
        if isinstance(action, TypeAction):
            allowed, reason = self._check_text_patterns(action.text)
            if not allowed:
                self._log_check(action, allowed, reason)
                return allowed, reason

        self._log_check(action, True, "Policy passed")
        self._total_actions += 1
        return True, "allowed"

    def _check_action_type(self, action_type: ActionType) -> tuple[bool, str]:
        """Check if action type is allowed."""
        for rule in self.policy.action_rules:
            if rule.action_type == action_type:
                allowed = rule.decision in (PolicyDecision.ALLOW, PolicyDecision.AUDIT)
                return allowed, rule.reason or f"Action {action_type.value} policy"

        # Default decision
        allowed = self.policy.default_decision in (PolicyDecision.ALLOW, PolicyDecision.AUDIT)
        return allowed, "default policy"

    def _check_domain(self, url: str) -> tuple[bool, str]:
        """Check if URL is allowed."""
        for rule in self.policy.domain_rules:
            if rule.matches(url):
                allowed = rule.decision in (PolicyDecision.ALLOW, PolicyDecision.AUDIT)
                return allowed, rule.reason or f"Domain policy for {url}"

        # Default to deny for domains
        return False, "URL not in allowlist"

    def _check_coordinates(self, x: int, y: int) -> tuple[bool, str]:
        """Check if coordinates are within bounds."""
        rule = self.policy.boundary_rule
        if x < rule.min_x or x > rule.max_x:
            return False, f"X coordinate {x} outside bounds [{rule.min_x}, {rule.max_x}]"
        if y < rule.min_y or y > rule.max_y:
            return False, f"Y coordinate {y} outside bounds [{rule.min_y}, {rule.max_y}]"
        return True, "coordinates within bounds"

    def _check_text_patterns(self, text: str) -> tuple[bool, str]:
        """Check text against sensitive patterns."""
        for rule in self.policy.text_pattern_rules:
            if ActionType.TYPE in rule.applies_to:
                if re.search(rule.pattern, text, re.IGNORECASE):
                    if rule.decision == PolicyDecision.DENY:
                        return False, rule.reason or "Text matches sensitive pattern"

        return True, "text allowed"

    def record_success(self) -> None:
        """Record a successful action."""
        self._consecutive_errors = 0

    def record_error(self) -> None:
        """Record an action error."""
        self._consecutive_errors += 1

    def reset(self) -> None:
        """Reset session state."""
        self._action_counts.clear()
        self._last_action_time.clear()
        self._total_actions = 0
        self._consecutive_errors = 0

    def _log_check(
        self,
        action: Action,
        allowed: bool,
        reason: str,
    ) -> None:
        """Log a policy check."""
        entry = {
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "allowed": allowed,
            "reason": reason,
            "timestamp": action.created_at,
        }

        if self.policy.audit_all_actions or not allowed:
            logger.info(f"Computer-use policy check: {entry}")

        self._audit_log.append(entry)

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the audit log."""
        return self._audit_log.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        return {
            "total_actions": self._total_actions,
            "consecutive_errors": self._consecutive_errors,
            "max_actions": self.policy.max_actions_per_task,
            "audit_entries": len(self._audit_log),
        }


def create_default_computer_policy() -> ComputerPolicy:
    """Create a balanced policy for typical computer-use."""
    policy = ComputerPolicy(
        name="default",
        description="Default policy for safe computer-use automation",
        default_decision=PolicyDecision.DENY,
    )

    # Allow all standard action types
    policy.add_action_allowlist(
        [
            ActionType.SCREENSHOT,
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.TYPE,
            ActionType.KEY,
            ActionType.SCROLL,
            ActionType.MOVE,
            ActionType.WAIT,
        ],
        reason="Standard UI actions",
    )

    # Allow common safe domains
    policy.add_domain_allowlist(
        [
            r"^https?://localhost(:\d+)?(/.*)?$",
            r"^https?://127\.0\.0\.1(:\d+)?(/.*)?$",
            r"^file://.*$",
        ],
        reason="Local development",
    )

    # Block sensitive text patterns
    policy.add_sensitive_text_patterns(
        [
            r"password\s*[:=]\s*\S+",  # password = xxx
            r"api[_-]?key\s*[:=]\s*\S+",  # api_key = xxx
            r"secret\s*[:=]\s*\S+",  # secret = xxx
            r"bearer\s+\S{20,}",  # Bearer tokens
            r"sk-[a-zA-Z0-9]{32,}",  # OpenAI API keys
            r"ghp_[a-zA-Z0-9]{36,}",  # GitHub tokens
        ],
        reason="Credential exfiltration prevention",
    )

    return policy


def create_strict_computer_policy() -> ComputerPolicy:
    """Create a strict policy for high-security environments."""
    policy = ComputerPolicy(
        name="strict",
        description="Strict policy requiring approval for most actions",
        default_decision=PolicyDecision.DENY,
        require_human_approval_for_sensitive=True,
        block_credential_fields=True,
        max_actions_per_task=50,
        max_consecutive_errors=2,
    )

    # Only allow read-only actions by default
    policy.add_action_allowlist(
        [ActionType.SCREENSHOT, ActionType.SCROLL, ActionType.WAIT],
        reason="Read-only observation",
    )

    # Clicks and typing require audit
    policy.action_rules.append(
        ActionRule(
            action_type=ActionType.CLICK,
            decision=PolicyDecision.AUDIT,
            reason="Click actions audited",
        )
    )
    policy.action_rules.append(
        ActionRule(
            action_type=ActionType.TYPE,
            decision=PolicyDecision.AUDIT,
            reason="Type actions audited",
        )
    )

    # Only localhost
    policy.add_domain_allowlist(
        [r"^https?://localhost(:\d+)?(/.*)?$"],
        reason="Localhost only",
    )

    # Strict rate limits
    policy.rate_limits = RateLimits(
        max_actions_per_minute=30,
        max_clicks_per_minute=15,
        max_keystrokes_per_minute=100,
        max_screenshots_per_minute=10,
        cooldown_after_error_seconds=5.0,
    )

    return policy


def create_readonly_computer_policy() -> ComputerPolicy:
    """Create a read-only policy for observation-only tasks."""
    policy = ComputerPolicy(
        name="readonly",
        description="Read-only policy - no clicks or typing allowed",
        default_decision=PolicyDecision.DENY,
    )

    # Only allow observation actions
    policy.add_action_allowlist(
        [ActionType.SCREENSHOT, ActionType.SCROLL, ActionType.WAIT],
        reason="Read-only observation",
    )

    # Block all interactive actions
    policy.add_action_denylist(
        [
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
            ActionType.TYPE,
            ActionType.KEY,
            ActionType.DRAG,
        ],
        reason="Interactive actions blocked in read-only mode",
    )

    return policy


__all__ = [
    "ActionRule",
    "BoundaryRule",
    "ComputerPolicy",
    "ComputerPolicyChecker",
    "DomainRule",
    "ElementRule",
    "PolicyDecision",
    "RateLimits",
    "TextPatternRule",
    "create_default_computer_policy",
    "create_readonly_computer_policy",
    "create_strict_computer_policy",
]
