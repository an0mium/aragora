"""
Tests for aragora.pipeline.test_plan module.

Tests the TestCase, TestPlan, and TestPlanGenerator classes for:
- Dataclass creation and serialization
- Test filtering by type and priority
- Markdown generation
- Test case generation from debate artifacts
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from aragora.pipeline.test_plan import (
    TestCase,
    TestPlan,
    TestPlanGenerator,
    TestType,
    TestPriority,
)


class TestTestType:
    """Tests for TestType enum."""

    def test_all_types_exist(self):
        """Test all expected test types exist."""
        expected_types = {"unit", "integration", "e2e", "performance", "security", "regression"}
        actual_types = {t.value for t in TestType}
        assert actual_types == expected_types

    def test_type_values(self):
        """Test specific enum values."""
        assert TestType.UNIT.value == "unit"
        assert TestType.INTEGRATION.value == "integration"
        assert TestType.E2E.value == "e2e"
        assert TestType.PERFORMANCE.value == "performance"
        assert TestType.SECURITY.value == "security"
        assert TestType.REGRESSION.value == "regression"


class TestTestPriority:
    """Tests for TestPriority enum."""

    def test_all_priorities_exist(self):
        """Test all expected priorities exist."""
        expected = {"p0", "p1", "p2", "p3"}
        actual = {p.value for p in TestPriority}
        assert actual == expected

    def test_priority_values(self):
        """Test specific priority values."""
        assert TestPriority.P0.value == "p0"
        assert TestPriority.P1.value == "p1"
        assert TestPriority.P2.value == "p2"
        assert TestPriority.P3.value == "p3"


class TestTestCase:
    """Tests for the TestCase dataclass."""

    def test_test_case_creation_minimal(self):
        """Test creating TestCase with minimal fields."""
        test = TestCase(
            id="tc-001",
            title="Test Login",
            description="Verify login functionality",
            test_type=TestType.UNIT,
            priority=TestPriority.P1,
        )

        assert test.id == "tc-001"
        assert test.title == "Test Login"
        assert test.test_type == TestType.UNIT
        assert test.priority == TestPriority.P1
        assert test.preconditions == []
        assert test.steps == []
        assert test.expected_result == ""
        assert test.automated is False
        assert test.implemented is False

    def test_test_case_creation_full(self):
        """Test creating TestCase with all fields."""
        test = TestCase(
            id="tc-002",
            title="Integration Test",
            description="Test API integration",
            test_type=TestType.INTEGRATION,
            priority=TestPriority.P0,
            preconditions=["Database running", "API key configured"],
            steps=["Send request", "Verify response", "Check logs"],
            expected_result="200 OK with valid JSON",
            related_claim_ids=["claim-1", "claim-2"],
            related_critique_ids=["crit-1"],
            automated=True,
            implemented=True,
        )

        assert len(test.preconditions) == 2
        assert len(test.steps) == 3
        assert test.expected_result == "200 OK with valid JSON"
        assert len(test.related_claim_ids) == 2
        assert test.automated is True
        assert test.implemented is True

    def test_test_case_to_dict(self):
        """Test TestCase serialization."""
        test = TestCase(
            id="tc-dict",
            title="Dict Test",
            description="Test serialization",
            test_type=TestType.E2E,
            priority=TestPriority.P2,
            preconditions=["Pre1"],
            steps=["Step1"],
            expected_result="Result",
            automated=True,
        )

        d = test.to_dict()

        assert d["id"] == "tc-dict"
        assert d["title"] == "Dict Test"
        assert d["test_type"] == "e2e"  # Enum value
        assert d["priority"] == "p2"  # Enum value
        assert d["preconditions"] == ["Pre1"]
        assert d["steps"] == ["Step1"]
        assert d["automated"] is True

    def test_all_test_types_serialization(self):
        """Test that all test types serialize correctly."""
        for test_type in TestType:
            test = TestCase(
                id=f"tc-{test_type.value}",
                title=f"Test {test_type.value}",
                description="Test",
                test_type=test_type,
                priority=TestPriority.P1,
            )
            d = test.to_dict()
            assert d["test_type"] == test_type.value


class TestTestPlan:
    """Tests for the TestPlan dataclass."""

    @pytest.fixture
    def sample_tests(self):
        """Create sample test cases."""
        return [
            TestCase(
                id="t1", title="Unit 1", description="D1",
                test_type=TestType.UNIT, priority=TestPriority.P0,
                implemented=True, automated=True
            ),
            TestCase(
                id="t2", title="Unit 2", description="D2",
                test_type=TestType.UNIT, priority=TestPriority.P1,
                implemented=True
            ),
            TestCase(
                id="t3", title="Integration", description="D3",
                test_type=TestType.INTEGRATION, priority=TestPriority.P1,
                implemented=False
            ),
            TestCase(
                id="t4", title="E2E", description="D4",
                test_type=TestType.E2E, priority=TestPriority.P2,
                automated=True
            ),
            TestCase(
                id="t5", title="Security", description="D5",
                test_type=TestType.SECURITY, priority=TestPriority.P0
            ),
        ]

    @pytest.fixture
    def plan(self, sample_tests):
        """Create a populated TestPlan."""
        p = TestPlan(
            debate_id="plan-001",
            title="Test Plan",
            description="Description",
            critical_paths=["auth", "payment"],
        )
        for test in sample_tests:
            p.add_test(test)
        return p

    def test_empty_plan(self):
        """Test empty TestPlan."""
        plan = TestPlan(
            debate_id="empty",
            title="Empty Plan",
            description="No tests",
        )

        assert plan.debate_id == "empty"
        assert plan.test_cases == []
        assert plan.target_coverage == 0.8
        assert plan.summary["total_tests"] == 0

    def test_add_test(self, sample_tests):
        """Test adding tests to plan."""
        plan = TestPlan(
            debate_id="add-test",
            title="Title",
            description="Desc",
        )

        for test in sample_tests:
            plan.add_test(test)

        assert len(plan.test_cases) == 5

    def test_get_by_type(self, plan):
        """Test filtering by test type."""
        unit_tests = plan.get_by_type(TestType.UNIT)
        integration_tests = plan.get_by_type(TestType.INTEGRATION)
        e2e_tests = plan.get_by_type(TestType.E2E)
        security_tests = plan.get_by_type(TestType.SECURITY)

        assert len(unit_tests) == 2
        assert len(integration_tests) == 1
        assert len(e2e_tests) == 1
        assert len(security_tests) == 1

    def test_get_by_priority(self, plan):
        """Test filtering by priority."""
        p0_tests = plan.get_by_priority(TestPriority.P0)
        p1_tests = plan.get_by_priority(TestPriority.P1)
        p2_tests = plan.get_by_priority(TestPriority.P2)
        p3_tests = plan.get_by_priority(TestPriority.P3)

        assert len(p0_tests) == 2
        assert len(p1_tests) == 2
        assert len(p2_tests) == 1
        assert len(p3_tests) == 0

    def test_get_unimplemented(self, plan):
        """Test getting unimplemented tests."""
        unimpl = plan.get_unimplemented()

        assert len(unimpl) == 3  # t3, t4, t5
        assert all(not t.implemented for t in unimpl)

    def test_summary(self, plan):
        """Test summary statistics."""
        summary = plan.summary

        assert summary["total_tests"] == 5
        assert summary["by_type"]["unit"] == 2
        assert summary["by_type"]["integration"] == 1
        assert summary["by_priority"]["p0"] == 2
        assert summary["by_priority"]["p1"] == 2
        assert summary["automated"] == 2
        assert summary["implemented"] == 2

    def test_to_markdown(self, plan):
        """Test markdown generation."""
        md = plan.to_markdown()

        assert "# Test Plan:" in md
        assert "plan-001" in md
        assert "80%" in md  # target_coverage
        assert "P0" in md
        assert "P1" in md
        assert "Unit 1" in md
        assert "auth" in md  # critical path
        assert "Generated by aragora" in md

    def test_to_markdown_implemented_markers(self, plan):
        """Test that implementation status shows in markdown."""
        md = plan.to_markdown()

        # Implemented tests should have [x]
        assert "[x]" in md
        # Unimplemented tests should have [ ]
        assert "[ ]" in md
        # Automated tests should show [AUTO]
        assert "[AUTO]" in md

    def test_to_dict(self, plan):
        """Test dictionary serialization."""
        d = plan.to_dict()

        assert d["debate_id"] == "plan-001"
        assert d["title"] == "Test Plan"
        assert len(d["test_cases"]) == 5
        assert "summary" in d
        assert d["target_coverage"] == 0.8
        assert d["critical_paths"] == ["auth", "payment"]

    def test_custom_target_coverage(self):
        """Test custom target coverage."""
        plan = TestPlan(
            debate_id="coverage",
            title="High Coverage",
            description="Desc",
            target_coverage=0.95,
        )

        assert plan.target_coverage == 0.95
        md = plan.to_markdown()
        assert "95%" in md


class TestTestPlanGenerator:
    """Tests for the TestPlanGenerator class."""

    @pytest.fixture
    def mock_artifact(self):
        """Create a mock DebateArtifact."""
        artifact = MagicMock()
        artifact.debate_id = "gen-plan-001"
        artifact.task = "Implement user authentication system."

        # Mock consensus
        consensus = MagicMock()
        consensus.final_answer = """
1. Implement JWT-based authentication
2. Use bcrypt for password hashing
3. Add rate limiting to login endpoint
4. Create user session management
5. Ensure secure cookie handling
"""
        artifact.consensus_proof = consensus

        # Mock trace data with critiques
        artifact.trace_data = {
            "events": [
                {
                    "event_type": "agent_critique",
                    "event_id": "crit-001",
                    "content": {
                        "issues": [
                            "What about session timeout?",
                            "Consider brute force protection",
                        ]
                    }
                },
                {
                    "event_type": "agent_critique",
                    "event_id": "crit-002",
                    "content": {
                        "issues": [
                            "Edge case: concurrent logins",
                        ]
                    }
                },
            ]
        }

        return artifact

    def test_generator_creation(self, mock_artifact):
        """Test TestPlanGenerator initialization."""
        gen = TestPlanGenerator(mock_artifact)
        assert gen.artifact == mock_artifact

    def test_generate_returns_plan(self, mock_artifact):
        """Test that generate returns a TestPlan."""
        gen = TestPlanGenerator(mock_artifact)
        plan = gen.generate()

        assert isinstance(plan, TestPlan)
        assert plan.debate_id == "gen-plan-001"
        assert "authentication" in plan.title.lower()

    def test_generate_includes_consensus_tests(self, mock_artifact):
        """Test that consensus-based tests are generated."""
        gen = TestPlanGenerator(mock_artifact)
        plan = gen.generate()

        # Should have tests from consensus (implement, use, add, etc.)
        integration_tests = plan.get_by_type(TestType.INTEGRATION)
        assert len(integration_tests) > 0

        # Consensus tests should be P1 priority
        p1_tests = plan.get_by_priority(TestPriority.P1)
        assert len(p1_tests) > 0

    def test_generate_includes_critique_tests(self, mock_artifact):
        """Test that critique-based tests are generated."""
        gen = TestPlanGenerator(mock_artifact)
        plan = gen.generate()

        # Should have tests from critiques (edge cases)
        unit_tests = plan.get_by_type(TestType.UNIT)
        assert any("edge" in t.title.lower() for t in unit_tests)

    def test_generate_with_no_consensus(self, mock_artifact):
        """Test generation when no consensus reached."""
        mock_artifact.consensus_proof = None

        gen = TestPlanGenerator(mock_artifact)
        plan = gen.generate()

        # Should still generate some tests (standard ones)
        assert isinstance(plan, TestPlan)

    def test_generate_with_no_trace_data(self, mock_artifact):
        """Test generation with empty trace data."""
        mock_artifact.trace_data = None

        gen = TestPlanGenerator(mock_artifact)
        plan = gen.generate()

        # Should still work, just fewer tests from critiques
        assert isinstance(plan, TestPlan)

    def test_extract_title_with_period(self, mock_artifact):
        """Test title extraction with sentence ending."""
        gen = TestPlanGenerator(mock_artifact)
        title = gen._extract_title()

        assert "authentication" in title.lower()
        assert title.endswith(".")

    def test_extract_title_long(self, mock_artifact):
        """Test title extraction from long task."""
        mock_artifact.task = "A" * 100 + " very long task description"

        gen = TestPlanGenerator(mock_artifact)
        title = gen._extract_title()

        assert len(title) <= 63  # 60 + "..."


class TestTestPlanStandardTests:
    """Tests for standard test case generation."""

    @pytest.fixture
    def mock_artifact(self):
        """Create minimal mock artifact."""
        artifact = MagicMock()
        artifact.debate_id = "std-test"
        artifact.task = "Task description."
        artifact.consensus_proof = None
        artifact.trace_data = None
        return artifact

    def test_standard_tests_added(self, mock_artifact):
        """Test that standard tests are always added."""
        gen = TestPlanGenerator(mock_artifact)
        plan = gen.generate()

        # Even with no consensus/critiques, should have some tests
        assert len(plan.test_cases) > 0


class TestIntegration:
    """Integration tests for test_plan module."""

    def test_full_workflow(self):
        """Test complete test plan workflow."""
        # Create artifact
        artifact = MagicMock()
        artifact.debate_id = "integration-test-plan"
        artifact.task = "Add caching layer for database queries."

        consensus = MagicMock()
        consensus.final_answer = """
Implementation plan:
1. Add Redis caching layer
2. Implement cache invalidation on writes
3. Add cache warming on startup
4. Create monitoring for cache hit rates
"""
        artifact.consensus_proof = consensus

        artifact.trace_data = {
            "events": [
                {
                    "event_type": "agent_critique",
                    "event_id": "c1",
                    "content": {
                        "issues": ["Memory limits?", "TTL configuration?"]
                    }
                }
            ]
        }

        # Generate plan
        gen = TestPlanGenerator(artifact)
        plan = gen.generate()

        # Verify plan structure
        assert plan.debate_id == "integration-test-plan"
        assert len(plan.test_cases) > 0

        # Verify filtering
        summary = plan.summary
        assert summary["total_tests"] > 0

        # Verify serialization
        d = plan.to_dict()
        assert "test_cases" in d
        assert "summary" in d

        # Verify markdown
        md = plan.to_markdown()
        assert "integration-test-plan" in md
        assert "Test Plan" in md

    def test_test_case_traceability(self):
        """Test that test cases have proper traceability."""
        test = TestCase(
            id="trace-test",
            title="Traceable Test",
            description="Test with traceability",
            test_type=TestType.INTEGRATION,
            priority=TestPriority.P1,
            related_claim_ids=["claim-1", "claim-2"],
            related_critique_ids=["critique-1"],
        )

        d = test.to_dict()
        assert d["related_claim_ids"] == ["claim-1", "claim-2"]
        assert d["related_critique_ids"] == ["critique-1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
