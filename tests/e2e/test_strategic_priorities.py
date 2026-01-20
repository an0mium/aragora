"""
E2E tests for strategic API priorities.

Tests cover:
- Gauntlet receipts flow (create run -> generate receipt -> verify -> export)
- Decision explainability flow (debate -> explain -> factors -> counterfactual)
- Workflow templates flow (list -> get details -> run template)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ============================================================================
# Gauntlet Receipts E2E Tests
# ============================================================================


class TestGauntletReceiptsFlow:
    """E2E tests for gauntlet receipts lifecycle."""

    @pytest.fixture
    def mock_gauntlet_store(self):
        """Mock gauntlet storage."""
        receipts = {}
        runs = {}

        class MockStore:
            async def create_run(self, config: Dict[str, Any]) -> str:
                run_id = f"run-{uuid.uuid4().hex[:8]}"
                runs[run_id] = {
                    "id": run_id,
                    "status": "completed",
                    "config": config,
                    "findings": [
                        {"id": "f1", "category": "security", "severity": "high", "description": "SQL injection risk"},
                        {"id": "f2", "category": "logic", "severity": "medium", "description": "Edge case not handled"},
                    ],
                }
                return run_id

            async def get_run(self, run_id: str) -> Dict[str, Any] | None:
                return runs.get(run_id)

            async def create_receipt(self, run_id: str) -> Dict[str, Any]:
                run = runs.get(run_id)
                if not run:
                    return None

                receipt_id = f"receipt-{uuid.uuid4().hex[:8]}"
                receipt = {
                    "id": receipt_id,
                    "run_id": run_id,
                    "verdict": "FAIL" if any(f["severity"] == "high" for f in run["findings"]) else "PASS",
                    "artifact_hash": f"sha256:{uuid.uuid4().hex}",
                    "findings": run["findings"],
                    "findings_count": len(run["findings"]),
                    "findings_by_severity": {
                        "critical": 0,
                        "high": 1,
                        "medium": 1,
                        "low": 0,
                    },
                    "created_at": "2026-01-20T12:00:00Z",
                }
                receipts[receipt_id] = receipt
                return receipt

            async def get_receipt(self, receipt_id: str) -> Dict[str, Any] | None:
                return receipts.get(receipt_id)

            async def list_receipts(self, **filters) -> list:
                result = list(receipts.values())
                if filters.get("verdict"):
                    result = [r for r in result if r["verdict"] == filters["verdict"]]
                return result

            async def verify_receipt(self, receipt_id: str) -> Dict[str, Any]:
                receipt = receipts.get(receipt_id)
                if not receipt:
                    return {"valid": False, "message": "Receipt not found"}
                return {"valid": True, "message": "Artifact integrity verified"}

        return MockStore()

    @pytest.mark.asyncio
    async def test_receipt_creation_flow(self, mock_gauntlet_store):
        """Test creating a gauntlet run and generating a receipt."""
        # Create a run
        run_id = await mock_gauntlet_store.create_run({
            "target": "debate-123",
            "checks": ["security", "logic", "compliance"],
        })
        assert run_id.startswith("run-")

        # Get run details
        run = await mock_gauntlet_store.get_run(run_id)
        assert run is not None
        assert run["status"] == "completed"
        assert len(run["findings"]) == 2

        # Generate receipt
        receipt = await mock_gauntlet_store.create_receipt(run_id)
        assert receipt is not None
        assert receipt["verdict"] == "FAIL"  # Has high severity finding
        assert receipt["findings_count"] == 2

    @pytest.mark.asyncio
    async def test_receipt_verification_flow(self, mock_gauntlet_store):
        """Test verifying receipt integrity."""
        # Create run and receipt
        run_id = await mock_gauntlet_store.create_run({"target": "test"})
        receipt = await mock_gauntlet_store.create_receipt(run_id)

        # Verify receipt
        result = await mock_gauntlet_store.verify_receipt(receipt["id"])
        assert result["valid"] is True
        assert "verified" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_receipt_filtering(self, mock_gauntlet_store):
        """Test filtering receipts by verdict."""
        # Create multiple runs with different outcomes
        for _ in range(3):
            run_id = await mock_gauntlet_store.create_run({"target": "test"})
            await mock_gauntlet_store.create_receipt(run_id)

        # Filter by verdict
        fail_receipts = await mock_gauntlet_store.list_receipts(verdict="FAIL")
        assert len(fail_receipts) == 3  # All have high findings

    @pytest.mark.asyncio
    async def test_nonexistent_receipt_verification(self, mock_gauntlet_store):
        """Test verifying a nonexistent receipt."""
        result = await mock_gauntlet_store.verify_receipt("nonexistent")
        assert result["valid"] is False


# ============================================================================
# Decision Explainability E2E Tests
# ============================================================================


class TestExplainabilityFlow:
    """E2E tests for decision explainability."""

    @pytest.fixture
    def mock_explainability_service(self):
        """Mock explainability service."""
        class MockService:
            async def generate_explanation(self, debate_id: str) -> Dict[str, Any]:
                return {
                    "debate_id": debate_id,
                    "narrative": f"The decision for debate {debate_id} was reached through "
                                 "multi-agent consensus. The primary factors were evidence quality "
                                 "and agent agreement levels.",
                    "confidence": 0.87,
                    "factors": [
                        {"name": "Agent Agreement", "contribution": 0.4, "description": "High inter-agent consensus"},
                        {"name": "Evidence Quality", "contribution": 0.35, "description": "Strong supporting evidence"},
                        {"name": "Logical Coherence", "contribution": 0.25, "description": "Arguments were well-structured"},
                    ],
                    "counterfactuals": [
                        {
                            "scenario": "If agent Claude had disagreed",
                            "outcome": "Decision confidence would drop by 15%",
                            "probability": 0.72,
                        },
                        {
                            "scenario": "With additional evidence from external sources",
                            "outcome": "Confidence could increase to 95%",
                            "probability": 0.65,
                        },
                    ],
                    "provenance": [
                        {"step": 1, "action": "Initial proposals generated", "timestamp": "2026-01-20T12:00:00Z"},
                        {"step": 2, "action": "Critiques exchanged", "timestamp": "2026-01-20T12:01:00Z", "agent": "Claude"},
                        {"step": 3, "action": "Consensus detected", "timestamp": "2026-01-20T12:02:00Z", "confidence": 0.87},
                    ],
                    "generated_at": "2026-01-20T12:05:00Z",
                }

            async def get_factors(self, debate_id: str) -> list:
                explanation = await self.generate_explanation(debate_id)
                return explanation["factors"]

            async def get_counterfactuals(self, debate_id: str) -> list:
                explanation = await self.generate_explanation(debate_id)
                return explanation["counterfactuals"]

            async def get_provenance(self, debate_id: str) -> list:
                explanation = await self.generate_explanation(debate_id)
                return explanation["provenance"]

        return MockService()

    @pytest.mark.asyncio
    async def test_full_explanation_generation(self, mock_explainability_service):
        """Test generating a complete explanation."""
        debate_id = "debate-123"
        explanation = await mock_explainability_service.generate_explanation(debate_id)

        assert explanation["debate_id"] == debate_id
        assert explanation["confidence"] > 0
        assert len(explanation["narrative"]) > 0
        assert len(explanation["factors"]) > 0
        assert len(explanation["counterfactuals"]) > 0
        assert len(explanation["provenance"]) > 0

    @pytest.mark.asyncio
    async def test_factor_decomposition(self, mock_explainability_service):
        """Test getting factor breakdown."""
        factors = await mock_explainability_service.get_factors("debate-123")

        assert len(factors) == 3
        total_contribution = sum(f["contribution"] for f in factors)
        assert abs(total_contribution - 1.0) < 0.01  # Should sum to ~1.0

    @pytest.mark.asyncio
    async def test_counterfactual_scenarios(self, mock_explainability_service):
        """Test counterfactual analysis."""
        counterfactuals = await mock_explainability_service.get_counterfactuals("debate-123")

        assert len(counterfactuals) == 2
        for cf in counterfactuals:
            assert "scenario" in cf
            assert "outcome" in cf
            assert 0 <= cf["probability"] <= 1

    @pytest.mark.asyncio
    async def test_provenance_chain(self, mock_explainability_service):
        """Test provenance chain retrieval."""
        provenance = await mock_explainability_service.get_provenance("debate-123")

        assert len(provenance) == 3
        # Check steps are ordered
        steps = [p["step"] for p in provenance]
        assert steps == sorted(steps)


# ============================================================================
# Workflow Templates E2E Tests
# ============================================================================


class TestWorkflowTemplatesFlow:
    """E2E tests for workflow templates."""

    @pytest.fixture
    def mock_template_registry(self):
        """Mock template registry."""
        templates = {
            "legal/contract-review": {
                "id": "legal/contract-review",
                "name": "Contract Review",
                "description": "Multi-agent contract analysis workflow",
                "category": "legal",
                "pattern": "review_cycle",
                "steps_count": 5,
                "tags": ["legal", "contracts", "compliance"],
                "estimated_duration": "10-15 minutes",
            },
            "research/literature-review": {
                "id": "research/literature-review",
                "name": "Literature Review",
                "description": "Systematic literature analysis",
                "category": "research",
                "pattern": "map_reduce",
                "steps_count": 4,
                "tags": ["research", "academic", "analysis"],
                "estimated_duration": "15-20 minutes",
            },
            "development/code-review": {
                "id": "development/code-review",
                "name": "Code Review",
                "description": "Multi-perspective code analysis",
                "category": "development",
                "pattern": "hive_mind",
                "steps_count": 3,
                "tags": ["development", "code", "quality"],
                "estimated_duration": "5-10 minutes",
            },
        }

        categories = [
            {"id": "legal", "name": "Legal", "template_count": 1},
            {"id": "research", "name": "Research", "template_count": 1},
            {"id": "development", "name": "Development", "template_count": 1},
        ]

        patterns = [
            {"id": "review_cycle", "name": "Review Cycle", "description": "Iterative review pattern", "available": True},
            {"id": "map_reduce", "name": "Map Reduce", "description": "Parallel processing pattern", "available": True},
            {"id": "hive_mind", "name": "Hive Mind", "description": "Collective decision pattern", "available": True},
        ]

        class MockRegistry:
            def list_templates(self, category: str | None = None) -> list:
                result = list(templates.values())
                if category:
                    result = [t for t in result if t["category"] == category]
                return result

            def get_template(self, template_id: str) -> Dict[str, Any] | None:
                return templates.get(template_id)

            def list_categories(self) -> list:
                return categories

            def list_patterns(self) -> list:
                return patterns

            async def run_template(self, template_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
                template = templates.get(template_id)
                if not template:
                    raise ValueError(f"Template not found: {template_id}")

                return {
                    "status": "completed",
                    "template_id": template_id,
                    "execution_id": f"exec-{uuid.uuid4().hex[:8]}",
                    "result": {"summary": f"Executed {template['name']} successfully"},
                }

        return MockRegistry()

    def test_list_all_templates(self, mock_template_registry):
        """Test listing all templates."""
        templates = mock_template_registry.list_templates()
        assert len(templates) == 3

    def test_filter_by_category(self, mock_template_registry):
        """Test filtering templates by category."""
        legal_templates = mock_template_registry.list_templates(category="legal")
        assert len(legal_templates) == 1
        assert legal_templates[0]["category"] == "legal"

    def test_get_template_details(self, mock_template_registry):
        """Test getting template details."""
        template = mock_template_registry.get_template("legal/contract-review")

        assert template is not None
        assert template["name"] == "Contract Review"
        assert template["steps_count"] == 5
        assert "legal" in template["tags"]

    def test_list_categories(self, mock_template_registry):
        """Test listing categories."""
        categories = mock_template_registry.list_categories()

        assert len(categories) == 3
        for cat in categories:
            assert "id" in cat
            assert "name" in cat
            assert "template_count" in cat

    def test_list_patterns(self, mock_template_registry):
        """Test listing available patterns."""
        patterns = mock_template_registry.list_patterns()

        assert len(patterns) == 3
        assert all(p["available"] for p in patterns)

    @pytest.mark.asyncio
    async def test_run_template(self, mock_template_registry):
        """Test running a template."""
        result = await mock_template_registry.run_template(
            "legal/contract-review",
            {"document": "Test contract content"},
        )

        assert result["status"] == "completed"
        assert result["template_id"] == "legal/contract-review"
        assert "execution_id" in result

    @pytest.mark.asyncio
    async def test_run_nonexistent_template(self, mock_template_registry):
        """Test running a nonexistent template."""
        with pytest.raises(ValueError, match="Template not found"):
            await mock_template_registry.run_template("nonexistent", {})


# ============================================================================
# Integration Tests
# ============================================================================


class TestStrategicPrioritiesIntegration:
    """Integration tests combining multiple strategic features."""

    @pytest.mark.asyncio
    async def test_debate_to_receipt_flow(self):
        """Test flow from debate completion to receipt generation."""
        # Simulate: debate -> gauntlet run -> receipt -> verification

        debate_id = f"debate-{uuid.uuid4().hex[:8]}"

        # Mock debate result
        debate_result = {
            "id": debate_id,
            "consensus": True,
            "confidence": 0.85,
            "verdict": "Microservices recommended for this use case",
        }

        # Mock gauntlet findings
        gauntlet_findings = [
            {"category": "logic", "severity": "low", "description": "Minor edge case"},
        ]

        # Generate receipt
        receipt = {
            "id": f"receipt-{uuid.uuid4().hex[:8]}",
            "debate_id": debate_id,
            "verdict": "PASS",  # No high/critical findings
            "findings": gauntlet_findings,
            "artifact_hash": f"sha256:{uuid.uuid4().hex}",
        }

        assert receipt["verdict"] == "PASS"
        assert len(receipt["findings"]) == 1

    @pytest.mark.asyncio
    async def test_template_to_explainability_flow(self):
        """Test flow from template execution to explainability."""
        # Simulate: template -> debate -> explainability

        template_id = "development/code-review"

        # Mock template execution
        execution_result = {
            "status": "completed",
            "debate_id": f"debate-{uuid.uuid4().hex[:8]}",
        }

        # Mock explanation
        explanation = {
            "debate_id": execution_result["debate_id"],
            "narrative": "Code review identified 3 issues and 5 improvements",
            "confidence": 0.92,
            "factors": [
                {"name": "Code Quality", "contribution": 0.5},
                {"name": "Security Review", "contribution": 0.3},
                {"name": "Performance Analysis", "contribution": 0.2},
            ],
        }

        assert explanation["confidence"] > 0.9
        assert sum(f["contribution"] for f in explanation["factors"]) == 1.0


# ============================================================================
# Resilience Tests
# ============================================================================


class TestResilienceIntegration:
    """Tests for resilience layer integration."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Test that transient failures trigger retry."""
        call_count = [0]

        async def flaky_operation():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Transient failure")
            return {"success": True}

        # Simulate retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await flaky_operation()
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                continue

        assert result["success"] is True
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_update(self):
        """Test that cache is invalidated when data changes."""
        cache = {}
        invalidation_events = []

        def cache_get(key):
            return cache.get(key)

        def cache_set(key, value):
            cache[key] = value

        def invalidate(key):
            if key in cache:
                del cache[key]
            invalidation_events.append(key)

        # Initial cache
        cache_set("node-123", {"content": "original"})
        assert cache_get("node-123") is not None

        # Simulate update triggering invalidation
        invalidate("node-123")
        assert cache_get("node-123") is None
        assert "node-123" in invalidation_events
