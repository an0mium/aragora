"""
Email triage rule loader.

Loads priority rules from a YAML config file and applies them as overrides
to the EmailPriorityAnalyzer scoring. This allows deployment-specific
email categorization without code changes.

Usage:
    from aragora.analysis.email_triage import TriageRuleEngine

    engine = TriageRuleEngine.from_yaml("deploy/liftmode/agents/email_triage.yaml")
    score = engine.apply_rules(subject, from_address, snippet, labels)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TriageRule:
    """A single triage rule with keywords and priority level."""

    label: str
    keywords: list[str]
    priority: str  # "high", "medium", "low"


@dataclass
class TriageScore:
    """Result of triage rule evaluation."""

    priority: str  # "high", "medium", "low", "none"
    matched_rule: str  # label of matched rule
    score_boost: float  # adjustment to apply to base score
    should_escalate: bool  # flagged for human review


@dataclass
class TriageConfig:
    """Parsed triage configuration from YAML."""

    rules: list[TriageRule] = field(default_factory=list)
    escalation_keywords: list[str] = field(default_factory=list)
    auto_handle_threshold: float = 0.85
    gmail_labels: dict[str, str] = field(default_factory=dict)
    sync_interval_minutes: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriageConfig:
        """Parse config from YAML dict."""
        rules: list[TriageRule] = []

        for priority_level, rule_list in data.get("priority_rules", {}).items():
            for rule_data in rule_list:
                rules.append(
                    TriageRule(
                        label=rule_data.get("label", ""),
                        keywords=rule_data.get("keywords", []),
                        priority=priority_level,
                    )
                )

        escalation = data.get("escalation", {})
        return cls(
            rules=rules,
            escalation_keywords=[kw.lower() for kw in escalation.get("always_flag", [])],
            auto_handle_threshold=escalation.get("auto_handle_threshold", 0.85),
            gmail_labels=data.get("gmail_labels", {}),
            sync_interval_minutes=data.get("sync", {}).get("interval_minutes", 5),
        )


class TriageRuleEngine:
    """
    Applies deployment-specific triage rules to emails.

    Loads rules from a YAML config and evaluates them against email fields.
    Returns a TriageScore with priority level, matched rule, and score boost.
    """

    PRIORITY_BOOSTS = {
        "high": 0.35,
        "medium": 0.15,
        "low": -0.25,
    }

    def __init__(self, config: TriageConfig) -> None:
        self.config = config
        # Pre-lowercase all keywords for fast matching
        self._rules_by_priority: dict[str, list[TriageRule]] = {}
        for rule in config.rules:
            self._rules_by_priority.setdefault(rule.priority, []).append(rule)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TriageRuleEngine:
        """Load triage rules from a YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            logger.warning("Triage config not found at %s — using defaults", path)
            return cls(TriageConfig())

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = TriageConfig.from_dict(data)
        logger.info(
            "Loaded %d triage rules from %s (%d escalation keywords)",
            len(config.rules),
            path.name,
            len(config.escalation_keywords),
        )
        return cls(config)

    def apply_rules(
        self,
        subject: str,
        from_address: str,
        snippet: str,
        labels: list[str] | None = None,
    ) -> TriageScore:
        """
        Evaluate triage rules against an email.

        Checks high-priority rules first (short-circuit on match),
        then medium, then low. Returns the first match.
        """
        text = f"{subject} {snippet} {from_address}".lower()

        # Check escalation first
        should_escalate = any(kw in text for kw in self.config.escalation_keywords)

        # Check rules in priority order: high → medium → low
        for priority in ("high", "medium", "low"):
            for rule in self._rules_by_priority.get(priority, []):
                if any(kw.lower() in text for kw in rule.keywords):
                    return TriageScore(
                        priority=priority,
                        matched_rule=rule.label,
                        score_boost=self.PRIORITY_BOOSTS.get(priority, 0.0),
                        should_escalate=should_escalate,
                    )

        return TriageScore(
            priority="none",
            matched_rule="",
            score_boost=0.0,
            should_escalate=should_escalate,
        )

    def get_gmail_label(self, matched_rule: str) -> str | None:
        """Get the Gmail label name for a matched rule."""
        return self.config.gmail_labels.get(matched_rule)
