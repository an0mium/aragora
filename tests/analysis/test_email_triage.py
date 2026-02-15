"""Tests for email triage rule engine."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aragora.analysis.email_triage import TriageConfig, TriageRuleEngine


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Create a sample triage YAML config."""
    config = tmp_path / "email_triage.yaml"
    config.write_text(
        textwrap.dedent("""\
        priority_rules:
          high:
            - label: "orders"
              keywords: ["order", "purchase", "refund", "chargeback"]
            - label: "shipping"
              keywords: ["tracking", "delivery", "shipment", "lost package"]
            - label: "regulatory"
              keywords: ["FDA", "cGMP", "COA", "certificate of analysis"]
          medium:
            - label: "vendors"
              keywords: ["supplier", "vendor", "wholesale", "bulk order"]
            - label: "inventory"
              keywords: ["stock", "restock", "out of stock", "backorder"]
          low:
            - label: "newsletters"
              keywords: ["unsubscribe", "newsletter", "weekly digest"]
            - label: "marketing"
              keywords: ["promotion", "campaign", "SEO"]

        escalation:
          always_flag:
            - "legal"
            - "lawsuit"
            - "FDA warning"
            - "adverse event"
            - "recall"
          auto_handle_threshold: 0.85

        gmail_labels:
          orders: "LiftMode/Orders"
          shipping: "LiftMode/Shipping"
          vendors: "LiftMode/Vendors"
          regulatory: "LiftMode/Regulatory"

        sync:
          interval_minutes: 5
    """)
    )
    return config


class TestTriageConfig:
    """Test TriageConfig parsing."""

    def test_from_dict_parses_rules(self) -> None:
        data = {
            "priority_rules": {
                "high": [{"label": "urgent", "keywords": ["asap", "deadline"]}],
                "low": [{"label": "spam", "keywords": ["unsubscribe"]}],
            },
        }
        config = TriageConfig.from_dict(data)
        assert len(config.rules) == 2
        assert config.rules[0].priority == "high"
        assert config.rules[1].priority == "low"

    def test_from_dict_parses_escalation(self) -> None:
        data = {
            "priority_rules": {},
            "escalation": {
                "always_flag": ["Legal", "FDA Warning"],
                "auto_handle_threshold": 0.9,
            },
        }
        config = TriageConfig.from_dict(data)
        assert "legal" in config.escalation_keywords
        assert "fda warning" in config.escalation_keywords
        assert config.auto_handle_threshold == 0.9

    def test_from_dict_defaults(self) -> None:
        config = TriageConfig.from_dict({})
        assert config.rules == []
        assert config.escalation_keywords == []
        assert config.auto_handle_threshold == 0.85
        assert config.sync_interval_minutes == 5


class TestTriageRuleEngine:
    """Test rule matching and scoring."""

    def test_load_from_yaml(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        assert len(engine.config.rules) == 7
        assert len(engine.config.escalation_keywords) == 5

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(tmp_path / "nonexistent.yaml")
        assert engine.config.rules == []

    def test_high_priority_order_email(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="Re: My order #12345",
            from_address="customer@gmail.com",
            snippet="Where is my order?",
        )
        assert result.priority == "high"
        assert result.matched_rule == "orders"
        assert result.score_boost > 0

    def test_high_priority_regulatory(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="COA Request for Phenibut batch 2024-Q1",
            from_address="supplier@vendor.com",
            snippet="Please provide certificate of analysis",
        )
        assert result.priority == "high"
        assert result.matched_rule == "regulatory"

    def test_medium_priority_vendor(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="Wholesale pricing update",
            from_address="sales@supplier.com",
            snippet="New pricing for Q2 raw materials",
        )
        assert result.priority == "medium"
        assert result.matched_rule == "vendors"

    def test_low_priority_newsletter(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="Weekly digest from Industry News",
            from_address="news@industry.com",
            snippet="Click to unsubscribe",
        )
        assert result.priority == "low"
        assert result.score_boost < 0

    def test_no_match_returns_none_priority(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="Hey, coffee tomorrow?",
            from_address="friend@personal.com",
            snippet="Want to grab lunch?",
        )
        assert result.priority == "none"
        assert result.matched_rule == ""
        assert result.score_boost == 0.0

    def test_escalation_flagged(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="FDA Warning Letter - Immediate Action Required",
            from_address="compliance@fda.gov",
            snippet="This is regarding a recall",
        )
        assert result.should_escalate is True

    def test_escalation_not_flagged_normal_email(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="New order #54321",
            from_address="customer@gmail.com",
            snippet="I placed an order",
        )
        assert result.should_escalate is False

    def test_high_priority_checked_first(self, sample_config_path: Path) -> None:
        """Email matching both high and low rules returns high."""
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="Order tracking newsletter",
            from_address="shop@store.com",
            snippet="Your order tracking update",
        )
        # "order" and "tracking" match high, "newsletter" matches low
        # High is checked first
        assert result.priority == "high"

    def test_gmail_label_lookup(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        assert engine.get_gmail_label("orders") == "LiftMode/Orders"
        assert engine.get_gmail_label("shipping") == "LiftMode/Shipping"
        assert engine.get_gmail_label("nonexistent") is None

    def test_case_insensitive_matching(self, sample_config_path: Path) -> None:
        engine = TriageRuleEngine.from_yaml(sample_config_path)
        result = engine.apply_rules(
            subject="URGENT: FDA cGMP Inspection",
            from_address="inspector@fda.gov",
            snippet="Upcoming inspection",
        )
        assert result.priority == "high"
        assert result.matched_rule == "regulatory"
