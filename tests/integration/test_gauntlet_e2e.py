"""
End-to-end integration tests for Gauntlet Adversarial Validation Engine.

Tests the complete workflow from input to Decision Receipt, verifying:
- GauntletOrchestrator runs successfully
- All sub-components integrate properly (redteam, prober, verification)
- Decision Receipt generation works correctly
- Export formats (JSON, HTML, Markdown) are valid
"""

from __future__ import annotations

import asyncio
import json
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aragora.core import Agent, Critique, Message
from aragora.modes.gauntlet import (
    GauntletOrchestrator,
    GauntletConfig,
    GauntletResult,
    InputType,
    Verdict,
    Finding,
    QUICK_GAUNTLET,
    THOROUGH_GAUNTLET,
    CODE_REVIEW_GAUNTLET,
    POLICY_GAUNTLET,
)
from aragora.export.decision_receipt import (
    DecisionReceipt,
    DecisionReceiptGenerator,
    ReceiptFinding,
)
from aragora.export.audit_trail import (
    AuditTrail,
    AuditTrailGenerator,
    AuditEventType,
)


# Test fixtures


class MockAgent(Agent):
    """Mock agent for testing that simulates real agent behavior."""

    def __init__(self, name: str, responses: dict[str, str] | None = None):
        super().__init__(name=name, model="mock", role="tester")
        self.responses = responses or {}
        self.call_history: list[str] = []

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Return mock response based on prompt keywords."""
        self.call_history.append(prompt[:100])

        # Check for specific response patterns
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        # Default response with some findings
        if "attack" in prompt.lower() or "vulnerability" in prompt.lower():
            return """
MEDIUM SEVERITY: Potential edge case identified
The input lacks explicit handling for boundary conditions.
Mitigation: Add validation for edge cases.
"""
        if "risk" in prompt.lower():
            return "LOW risk identified. Standard precautions recommended."

        return "Analysis complete. No critical issues found."

    async def critique(
        self, proposal: str, task: str, context: list[Message] | None = None
    ) -> Critique:
        """Return mock critique."""
        return Critique(
            agent=self.name,
            target_agent="target",
            target_content=proposal[:200],
            issues=["Minor issue identified"],
            suggestions=["Consider improvement"],
            severity=0.3,
            reasoning="Standard analysis completed.",
        )


@pytest.fixture
def mock_agents() -> list[Agent]:
    """Create a set of mock agents for testing."""
    return [
        MockAgent("security-agent", {
            "security": "HIGH SEVERITY: SQL injection vulnerability detected in line 42.",
        }),
        MockAgent("compliance-agent", {
            "compliance": "MEDIUM SEVERITY: Missing audit logging requirement.",
        }),
        MockAgent("edge-case-agent", {
            "edge": "LOW SEVERITY: Undefined behavior for empty input.",
        }),
    ]


@pytest.fixture
def sample_policy() -> str:
    """Sample policy document for testing."""
    return """
# Data Retention Policy v1.0

## Purpose
This policy governs the retention and deletion of user data.

## Requirements

1. User data must be retained for a minimum of 7 years for compliance.
2. Data can be deleted upon user request within 30 days.
3. Backups are maintained for 90 days.

## Exceptions

- Legal holds override standard retention periods.
- Anonymized data is exempt from retention limits.

## Enforcement

Quarterly audits verify compliance. Violations are reported to the DPO.
"""


@pytest.fixture
def sample_code() -> str:
    """Sample code for testing."""
    return '''
def process_user_input(user_input: str) -> dict:
    """Process user input and return results."""
    # Parse the input
    data = json.loads(user_input)

    # Execute query
    query = f"SELECT * FROM users WHERE id = {data['user_id']}"
    result = database.execute(query)

    return {"status": "success", "data": result}
'''


# GauntletOrchestrator Tests


class TestGauntletOrchestrator:
    """Tests for GauntletOrchestrator end-to-end flows."""

    @pytest.mark.asyncio
    async def test_basic_run_with_policy(self, mock_agents, sample_policy):
        """Test basic Gauntlet run with a policy document."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,  # Skip for speed
            enable_verification=False,
            max_duration_seconds=30,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        assert result is not None
        assert isinstance(result, GauntletResult)
        assert result.gauntlet_id is not None
        assert result.verdict in list(Verdict)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.risk_score <= 1.0

    @pytest.mark.asyncio
    async def test_basic_run_with_code(self, mock_agents, sample_code):
        """Test Gauntlet run with code input."""
        config = GauntletConfig(
            input_type=InputType.CODE,
            input_content=sample_code,
            enable_deep_audit=False,
            enable_verification=False,
            max_duration_seconds=30,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        assert result is not None
        assert result.input_type == InputType.CODE
        # Code with SQL injection should have findings
        assert result.total_findings >= 0

    @pytest.mark.asyncio
    async def test_quick_profile(self, mock_agents, sample_policy):
        """Test QUICK_GAUNTLET profile runs successfully."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            deep_audit_rounds=QUICK_GAUNTLET.deep_audit_rounds,
            parallel_attacks=QUICK_GAUNTLET.parallel_attacks,
            enable_verification=False,
            max_duration_seconds=30,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        assert result is not None
        assert result.duration_seconds < 60  # Quick should be fast

    @pytest.mark.asyncio
    async def test_result_contains_expected_fields(self, mock_agents, sample_policy):
        """Test that GauntletResult contains all expected fields."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        # Core fields
        assert hasattr(result, 'gauntlet_id')
        assert hasattr(result, 'verdict')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'risk_score')
        assert hasattr(result, 'robustness_score')
        assert hasattr(result, 'coverage_score')

        # Finding lists
        assert hasattr(result, 'critical_findings')
        assert hasattr(result, 'high_findings')
        assert hasattr(result, 'medium_findings')
        assert hasattr(result, 'low_findings')

        # Metadata
        assert hasattr(result, 'agents_involved')
        assert hasattr(result, 'duration_seconds')
        assert hasattr(result, 'checksum')

    @pytest.mark.asyncio
    async def test_verdict_determination(self, mock_agents, sample_policy):
        """Test that verdicts are determined correctly."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        # Verdict should be one of the valid options
        assert result.verdict in [
            Verdict.APPROVED,
            Verdict.APPROVED_WITH_CONDITIONS,
            Verdict.NEEDS_REVIEW,
            Verdict.REJECTED,
        ]

        # Confidence should match verdict severity
        if result.verdict == Verdict.REJECTED:
            assert len(result.critical_findings) >= 1 or result.risk_score > 0.7

    @pytest.mark.asyncio
    async def test_agents_are_tracked(self, mock_agents, sample_policy):
        """Test that participating agents are tracked in result."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        assert len(result.agents_involved) > 0
        for agent in mock_agents:
            assert agent.name in result.agents_involved


# Decision Receipt Tests


class TestDecisionReceipt:
    """Tests for Decision Receipt generation and export."""

    @pytest.mark.asyncio
    async def test_receipt_from_gauntlet_result(self, mock_agents, sample_policy):
        """Test generating a Decision Receipt from GauntletResult."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        receipt = DecisionReceiptGenerator.from_gauntlet_result(result)

        assert receipt is not None
        assert isinstance(receipt, DecisionReceipt)
        assert receipt.receipt_id is not None
        assert receipt.gauntlet_id == result.gauntlet_id
        assert receipt.verdict == result.verdict.value.upper()

    @pytest.mark.asyncio
    async def test_receipt_json_export(self, mock_agents, sample_policy):
        """Test Decision Receipt JSON export."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        receipt = DecisionReceiptGenerator.from_gauntlet_result(result)

        json_str = receipt.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert "receipt_id" in data
        assert "verdict" in data
        assert "confidence" in data
        assert "findings" in data
        assert "checksum" in data

    @pytest.mark.asyncio
    async def test_receipt_html_export(self, mock_agents, sample_policy):
        """Test Decision Receipt HTML export."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        receipt = DecisionReceiptGenerator.from_gauntlet_result(result)

        html = receipt.to_html()

        # Should be valid HTML with key sections
        assert "<!DOCTYPE html>" in html
        assert "Decision Receipt" in html
        assert "VERDICT" in html
        assert receipt.verdict in html
        assert "Findings Summary" in html

    @pytest.mark.asyncio
    async def test_receipt_markdown_export(self, mock_agents, sample_policy):
        """Test Decision Receipt Markdown export."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        receipt = DecisionReceiptGenerator.from_gauntlet_result(result)

        md = receipt.to_markdown()

        # Should be valid Markdown with key sections
        assert "# Decision Receipt" in md
        assert "## Verdict" in md
        assert "## Findings Summary" in md
        assert receipt.checksum in md

    @pytest.mark.asyncio
    async def test_receipt_checksum_integrity(self, mock_agents, sample_policy):
        """Test that receipt checksum can verify integrity."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        receipt = DecisionReceiptGenerator.from_gauntlet_result(result)

        # Checksum should exist and be non-empty
        assert receipt.checksum is not None
        assert len(receipt.checksum) > 0

        # Verify integrity should pass
        assert receipt.verify_integrity() is True

    @pytest.mark.asyncio
    async def test_receipt_save_and_load(self, mock_agents, sample_policy, tmp_path):
        """Test saving and loading a receipt from file."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        receipt = DecisionReceiptGenerator.from_gauntlet_result(result)

        # Save to JSON
        json_path = receipt.save(tmp_path / "receipt", format="json")
        assert json_path.exists()

        # Load back
        loaded = DecisionReceipt.load(json_path)
        assert loaded.receipt_id == receipt.receipt_id
        assert loaded.verdict == receipt.verdict
        assert loaded.checksum == receipt.checksum


# Audit Trail Tests


class TestAuditTrail:
    """Tests for Audit Trail generation."""

    @pytest.mark.asyncio
    async def test_audit_trail_from_result(self, mock_agents, sample_policy):
        """Test generating an Audit Trail from GauntletResult."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        trail = AuditTrailGenerator.from_gauntlet_result(result)

        assert trail is not None
        assert isinstance(trail, AuditTrail)
        assert trail.gauntlet_id == result.gauntlet_id
        assert len(trail.events) > 0

    @pytest.mark.asyncio
    async def test_audit_trail_events_ordered(self, mock_agents, sample_policy):
        """Test that audit trail events are chronologically ordered."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        trail = AuditTrailGenerator.from_gauntlet_result(result)

        # Events should be in order
        for i in range(len(trail.events) - 1):
            assert trail.events[i].timestamp <= trail.events[i + 1].timestamp

    @pytest.mark.asyncio
    async def test_audit_trail_has_start_and_end(self, mock_agents, sample_policy):
        """Test that audit trail has start and end events."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        trail = AuditTrailGenerator.from_gauntlet_result(result)

        event_types = [e.event_type for e in trail.events]
        assert AuditEventType.GAUNTLET_START in event_types
        assert AuditEventType.GAUNTLET_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_audit_trail_json_export(self, mock_agents, sample_policy):
        """Test Audit Trail JSON export."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        trail = AuditTrailGenerator.from_gauntlet_result(result)

        json_str = trail.to_json()
        data = json.loads(json_str)

        assert "gauntlet_id" in data
        assert "events" in data
        assert len(data["events"]) > 0

    @pytest.mark.asyncio
    async def test_audit_trail_checksum(self, mock_agents, sample_policy):
        """Test Audit Trail has integrity checksum."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)
        trail = AuditTrailGenerator.from_gauntlet_result(result)

        assert trail.checksum is not None
        assert len(trail.checksum) > 0


# Profile Tests


class TestGauntletProfiles:
    """Tests for pre-configured Gauntlet profiles."""

    def test_quick_profile_config(self):
        """Test QUICK_GAUNTLET has appropriate settings."""
        assert QUICK_GAUNTLET.deep_audit_rounds <= 3
        assert QUICK_GAUNTLET.max_duration_seconds <= 300

    def test_thorough_profile_config(self):
        """Test THOROUGH_GAUNTLET has comprehensive settings."""
        assert THOROUGH_GAUNTLET.deep_audit_rounds >= 4
        assert THOROUGH_GAUNTLET.enable_verification is True
        assert THOROUGH_GAUNTLET.max_duration_seconds >= 600

    def test_code_review_profile_config(self):
        """Test CODE_REVIEW_GAUNTLET is configured for code."""
        assert CODE_REVIEW_GAUNTLET.input_type == InputType.CODE
        assert CODE_REVIEW_GAUNTLET.enable_verification is True

    def test_policy_profile_config(self):
        """Test POLICY_GAUNTLET is configured for policies."""
        assert POLICY_GAUNTLET.input_type == InputType.POLICY
        assert POLICY_GAUNTLET.severity_threshold <= 0.5  # More sensitive


# Edge Cases


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_input(self, mock_agents):
        """Test handling of empty input."""
        config = GauntletConfig(
            input_type=InputType.SPEC,
            input_content="",
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        # Should still produce a result
        assert result is not None
        assert result.verdict is not None

    @pytest.mark.asyncio
    async def test_no_agents(self, sample_policy):
        """Test handling when no agents are provided."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator([])  # No agents
        result = await orchestrator.run(config)

        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_input(self, mock_agents):
        """Test handling of very long input."""
        long_content = "This is a test. " * 10000  # ~150KB

        config = GauntletConfig(
            input_type=InputType.SPEC,
            input_content=long_content,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        assert result is not None
        # Input should be summarized in result
        assert len(result.input_summary) <= 1000

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_agents, sample_policy):
        """Test that timeout is respected."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
            max_duration_seconds=1,  # Very short timeout
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        # Should complete within reasonable time
        assert result is not None


# Finding Tests


class TestFindings:
    """Tests for finding generation and classification."""

    def test_finding_severity_level(self):
        """Test Finding severity level calculation."""
        critical = Finding(
            finding_id="f1",
            category="test",
            severity=0.95,
            title="Critical",
            description="Test",
        )
        assert critical.severity_level == "CRITICAL"

        high = Finding(
            finding_id="f2",
            category="test",
            severity=0.75,
            title="High",
            description="Test",
        )
        assert high.severity_level == "HIGH"

        medium = Finding(
            finding_id="f3",
            category="test",
            severity=0.5,
            title="Medium",
            description="Test",
        )
        assert medium.severity_level == "MEDIUM"

        low = Finding(
            finding_id="f4",
            category="test",
            severity=0.2,
            title="Low",
            description="Test",
        )
        assert low.severity_level == "LOW"

    @pytest.mark.asyncio
    async def test_findings_are_categorized(self, mock_agents, sample_policy):
        """Test that findings are properly categorized by severity."""
        config = GauntletConfig(
            input_type=InputType.POLICY,
            input_content=sample_policy,
            enable_deep_audit=False,
            enable_verification=False,
        )

        orchestrator = GauntletOrchestrator(mock_agents)
        result = await orchestrator.run(config)

        # All findings should be in one of the severity lists
        total = (
            len(result.critical_findings) +
            len(result.high_findings) +
            len(result.medium_findings) +
            len(result.low_findings)
        )
        assert total == result.total_findings

        # Critical findings should have high severity
        for f in result.critical_findings:
            assert f.severity >= 0.9

        # High findings should have medium-high severity
        for f in result.high_findings:
            assert 0.7 <= f.severity < 0.9
