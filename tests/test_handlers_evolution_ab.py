"""
Tests for Evolution A/B Testing Handler.

Tests cover:
- A/B test listing with filters
- Getting specific tests
- Creating new A/B tests
- Recording debate results
- Concluding tests
- Cancelling tests
- Error handling
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Mock A/B Testing Classes
# =============================================================================


class MockABTestStatus(Enum):
    """Mock status enum for A/B tests."""
    ACTIVE = "active"
    CONCLUDED = "concluded"
    CANCELLED = "cancelled"


class MockABTest:
    """Mock A/B test for testing."""

    def __init__(
        self,
        test_id: str = "test-123",
        agent: str = "claude",
        baseline_version: int = 1,
        evolved_version: int = 2,
        status: MockABTestStatus = MockABTestStatus.ACTIVE,
        baseline_wins: int = 0,
        evolved_wins: int = 0,
        metadata: Optional[dict] = None,
    ):
        self.id = test_id
        self.agent = agent
        self.baseline_version = baseline_version
        self.evolved_version = evolved_version
        self.status = status
        self.baseline_wins = baseline_wins
        self.evolved_wins = evolved_wins
        self.metadata = metadata or {}
        self.started_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent": self.agent,
            "baseline_version": self.baseline_version,
            "evolved_version": self.evolved_version,
            "status": self.status.value,
            "baseline_wins": self.baseline_wins,
            "evolved_wins": self.evolved_wins,
            "metadata": self.metadata,
            "started_at": self.started_at.isoformat(),
        }


class MockABTestResult:
    """Mock A/B test conclusion result."""

    def __init__(
        self,
        test_id: str = "test-123",
        winner: str = "evolved",
        confidence: float = 0.95,
        recommendation: str = "Promote evolved version",
        stats: Optional[dict] = None,
    ):
        self.test_id = test_id
        self.winner = winner
        self.confidence = confidence
        self.recommendation = recommendation
        self.stats = stats or {}


class MockABTestManager:
    """Mock A/B test manager for testing."""

    def __init__(self, db_path: str = "test.db"):
        self.db_path = db_path
        self.tests: dict[str, MockABTest] = {}
        self._test_counter = 0

    def get_test(self, test_id: str) -> Optional[MockABTest]:
        """Get a specific test."""
        return self.tests.get(test_id)

    def get_active_test(self, agent: str) -> Optional[MockABTest]:
        """Get active test for an agent."""
        for test in self.tests.values():
            if test.agent == agent and test.status == MockABTestStatus.ACTIVE:
                return test
        return None

    def get_agent_tests(self, agent: str, limit: int = 50) -> list[MockABTest]:
        """Get tests for an agent."""
        agent_tests = [t for t in self.tests.values() if t.agent == agent]
        return sorted(agent_tests, key=lambda t: t.started_at, reverse=True)[:limit]

    def start_test(
        self,
        agent: str,
        baseline_version: int,
        evolved_version: int,
        metadata: Optional[dict] = None,
    ) -> MockABTest:
        """Start a new A/B test."""
        # Check for existing active test
        if self.get_active_test(agent):
            raise ValueError(f"Active test already exists for agent {agent}")

        self._test_counter += 1
        test_id = f"test-{self._test_counter}"
        test = MockABTest(
            test_id=test_id,
            agent=agent,
            baseline_version=baseline_version,
            evolved_version=evolved_version,
            metadata=metadata or {},
        )
        self.tests[test_id] = test
        return test

    def record_result(
        self,
        agent: str,
        debate_id: str,
        variant: str,
        won: bool,
    ) -> Optional[MockABTest]:
        """Record a debate result."""
        test = self.get_active_test(agent)
        if not test:
            return None

        if won:
            if variant == "baseline":
                test.baseline_wins += 1
            else:
                test.evolved_wins += 1
        return test

    def conclude_test(self, test_id: str, force: bool = False) -> MockABTestResult:
        """Conclude an A/B test."""
        test = self.tests.get(test_id)
        if not test:
            raise ValueError("Test not found")
        if test.status != MockABTestStatus.ACTIVE:
            raise ValueError("Test is not active")

        total = test.baseline_wins + test.evolved_wins
        if total < 10 and not force:
            raise ValueError("Insufficient data - need at least 10 results")

        test.status = MockABTestStatus.CONCLUDED
        winner = "evolved" if test.evolved_wins > test.baseline_wins else "baseline"

        return MockABTestResult(
            test_id=test_id,
            winner=winner,
            confidence=0.95 if total >= 10 else 0.5,
            recommendation=f"Promote {winner} version",
            stats={
                "baseline_wins": test.baseline_wins,
                "evolved_wins": test.evolved_wins,
            },
        )

    def cancel_test(self, test_id: str) -> bool:
        """Cancel an A/B test."""
        test = self.tests.get(test_id)
        if not test or test.status != MockABTestStatus.ACTIVE:
            return False
        test.status = MockABTestStatus.CANCELLED
        return True


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ab_module():
    """Mock the A/B testing module availability."""
    with patch.dict("sys.modules", {"aragora.evolution.ab_testing": MagicMock()}):
        yield


@pytest.fixture
def ab_handler():
    """Create EvolutionABTestingHandler with mock context."""
    import aragora.server.handlers.evolution_ab_testing as ab_module

    # Save original value
    original_value = ab_module.AB_TESTING_AVAILABLE

    # Patch module-level variable
    ab_module.AB_TESTING_AVAILABLE = True

    try:
        handler = ab_module.EvolutionABTestingHandler({"ab_tests_db": "test.db"})
        handler._manager = MockABTestManager()
        yield handler
    finally:
        # Restore original value
        ab_module.AB_TESTING_AVAILABLE = original_value


@pytest.fixture
def ab_handler_unavailable():
    """Create handler when A/B testing is not available."""
    import aragora.server.handlers.evolution_ab_testing as ab_module

    # Save original value
    original_value = ab_module.AB_TESTING_AVAILABLE

    # Patch module-level variable
    ab_module.AB_TESTING_AVAILABLE = False

    try:
        handler = ab_module.EvolutionABTestingHandler({})
        yield handler
    finally:
        # Restore original value
        ab_module.AB_TESTING_AVAILABLE = original_value


# =============================================================================
# Route Matching Tests
# =============================================================================


class TestRouteMatching:
    """Tests for route matching."""

    def test_can_handle_ab_tests_list(self, ab_handler):
        """Test handler matches A/B tests list route."""
        assert ab_handler.can_handle("/api/evolution/ab-tests") is True
        assert ab_handler.can_handle("/api/evolution/ab-tests/") is True

    def test_can_handle_ab_test_detail(self, ab_handler):
        """Test handler matches A/B test detail route."""
        assert ab_handler.can_handle("/api/evolution/ab-tests/test-123") is True

    def test_can_handle_active_test(self, ab_handler):
        """Test handler matches active test route."""
        assert ab_handler.can_handle("/api/evolution/ab-tests/claude/active") is True

    def test_can_handle_record(self, ab_handler):
        """Test handler matches record route."""
        assert ab_handler.can_handle("/api/evolution/ab-tests/test-123/record") is True

    def test_can_handle_conclude(self, ab_handler):
        """Test handler matches conclude route."""
        assert ab_handler.can_handle("/api/evolution/ab-tests/test-123/conclude") is True

    def test_cannot_handle_unrelated(self, ab_handler):
        """Test handler rejects unrelated routes."""
        assert ab_handler.can_handle("/api/debates") is False
        assert ab_handler.can_handle("/api/evolution/mutations") is False


# =============================================================================
# Module Availability Tests
# =============================================================================


class TestModuleAvailability:
    """Tests for A/B testing module availability."""

    def test_unavailable_get(self, ab_handler_unavailable):
        """Test GET returns 503 when module unavailable."""
        result = ab_handler_unavailable.handle(
            "/api/evolution/ab-tests",
            {},
        )
        assert result.status_code == 503

    def test_unavailable_post(self, ab_handler_unavailable):
        """Test POST returns 503 when module unavailable."""
        result = ab_handler_unavailable.handle_post(
            "/api/evolution/ab-tests",
            {},
        )
        assert result.status_code == 503

    def test_unavailable_delete(self, ab_handler_unavailable):
        """Test DELETE returns 503 when module unavailable."""
        result = ab_handler_unavailable.handle_delete(
            "/api/evolution/ab-tests/test-123",
        )
        assert result.status_code == 503


# =============================================================================
# List Tests
# =============================================================================


class TestListTests:
    """Tests for listing A/B tests."""

    def test_list_empty(self, ab_handler):
        """Test listing when no tests exist."""
        with patch.object(ab_handler, "_get_all_tests", return_value=[]):
            result = ab_handler.handle(
                "/api/evolution/ab-tests",
                {},
            )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["tests"] == []
        assert body["count"] == 0

    def test_list_with_tests(self, ab_handler):
        """Test listing existing tests."""
        # Add tests
        ab_handler.manager.start_test("claude", 1, 2)
        ab_handler.manager.start_test("gpt4", 1, 2)

        with patch.object(
            ab_handler, "_get_all_tests",
            return_value=list(ab_handler.manager.tests.values())
        ):
            result = ab_handler.handle(
                "/api/evolution/ab-tests",
                {},
            )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["count"] == 2

    def test_list_by_agent(self, ab_handler):
        """Test listing tests for specific agent."""
        # Add tests
        ab_handler.manager.start_test("claude", 1, 2)
        ab_handler.manager.start_test("gpt4", 1, 2)

        result = ab_handler.handle(
            "/api/evolution/ab-tests",
            {"agent": "claude"},
        )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["tests"][0]["agent"] == "claude"


# =============================================================================
# Get Test Tests
# =============================================================================


class TestGetTest:
    """Tests for getting specific A/B test."""

    def test_get_test_success(self, ab_handler):
        """Test getting existing test."""
        test = ab_handler.manager.start_test("claude", 1, 2)

        result = ab_handler.handle(
            f"/api/evolution/ab-tests/{test.id}",
            {},
        )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["id"] == test.id
        assert body["agent"] == "claude"

    def test_get_test_not_found(self, ab_handler):
        """Test getting non-existent test."""
        result = ab_handler.handle(
            "/api/evolution/ab-tests/nonexistent",
            {},
        )

        assert result.status_code == 404


# =============================================================================
# Get Active Test Tests
# =============================================================================


class TestGetActiveTest:
    """Tests for getting active test for agent."""

    def test_get_active_test_exists(self, ab_handler):
        """Test getting active test when one exists."""
        ab_handler.manager.start_test("claude", 1, 2)

        result = ab_handler.handle(
            "/api/evolution/ab-tests/claude/active",
            {},
        )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert body["has_active_test"] is True
        assert body["test"] is not None

    def test_get_active_test_not_exists(self, ab_handler):
        """Test getting active test when none exists."""
        result = ab_handler.handle(
            "/api/evolution/ab-tests/gpt4/active",
            {},
        )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["agent"] == "gpt4"
        assert body["has_active_test"] is False
        assert body["test"] is None


# =============================================================================
# Create Test Tests
# =============================================================================


class TestCreateTest:
    """Tests for creating A/B tests."""

    def test_create_test_success(self, ab_handler):
        """Test successful test creation."""
        result = ab_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
                "baseline_version": 1,
                "evolved_version": 2,
            },
        )

        assert result.status_code == 201
        import json
        body = json.loads(result.body)
        assert "test" in body
        assert body["test"]["agent"] == "claude"

    def test_create_test_missing_agent(self, ab_handler):
        """Test creation fails without agent."""
        result = ab_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "baseline_version": 1,
                "evolved_version": 2,
            },
        )

        assert result.status_code == 400

    def test_create_test_missing_versions(self, ab_handler):
        """Test creation fails without versions."""
        result = ab_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
            },
        )

        assert result.status_code == 400

    def test_create_test_conflict(self, ab_handler):
        """Test creation fails when active test exists."""
        # Create first test
        ab_handler.manager.start_test("claude", 1, 2)

        # Try to create another
        result = ab_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
                "baseline_version": 2,
                "evolved_version": 3,
            },
        )

        assert result.status_code == 409

    def test_create_test_with_metadata(self, ab_handler):
        """Test creation with metadata."""
        result = ab_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
                "baseline_version": 1,
                "evolved_version": 2,
                "metadata": {"description": "Test improvement"},
            },
        )

        assert result.status_code == 201
        import json
        body = json.loads(result.body)
        assert body["test"]["metadata"]["description"] == "Test improvement"


# =============================================================================
# Record Result Tests
# =============================================================================


class TestRecordResult:
    """Tests for recording debate results."""

    def test_record_result_success(self, ab_handler):
        """Test successful result recording."""
        test = ab_handler.manager.start_test("claude", 1, 2)

        result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test.id}/record",
            {
                "debate_id": "debate-123",
                "variant": "evolved",
                "won": True,
            },
        )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["test"]["evolved_wins"] == 1

    def test_record_baseline_win(self, ab_handler):
        """Test recording baseline win."""
        test = ab_handler.manager.start_test("claude", 1, 2)

        result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test.id}/record",
            {
                "debate_id": "debate-123",
                "variant": "baseline",
                "won": True,
            },
        )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["test"]["baseline_wins"] == 1

    def test_record_result_test_not_found(self, ab_handler):
        """Test recording for non-existent test."""
        result = ab_handler.handle_post(
            "/api/evolution/ab-tests/nonexistent/record",
            {
                "debate_id": "debate-123",
                "variant": "evolved",
                "won": True,
            },
        )

        assert result.status_code == 404

    def test_record_result_missing_debate_id(self, ab_handler):
        """Test recording without debate_id."""
        test = ab_handler.manager.start_test("claude", 1, 2)

        result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test.id}/record",
            {
                "variant": "evolved",
                "won": True,
            },
        )

        assert result.status_code == 400

    def test_record_result_invalid_variant(self, ab_handler):
        """Test recording with invalid variant."""
        test = ab_handler.manager.start_test("claude", 1, 2)

        result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test.id}/record",
            {
                "debate_id": "debate-123",
                "variant": "invalid",
                "won": True,
            },
        )

        assert result.status_code == 400

    def test_record_result_concluded_test(self, ab_handler):
        """Test recording for concluded test."""
        test = ab_handler.manager.start_test("claude", 1, 2)
        test.status = MockABTestStatus.CONCLUDED

        result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test.id}/record",
            {
                "debate_id": "debate-123",
                "variant": "evolved",
                "won": True,
            },
        )

        assert result.status_code == 400


# =============================================================================
# Conclude Test Tests
# =============================================================================


class TestConcludeTest:
    """Tests for concluding A/B tests."""

    def test_conclude_test_success(self, ab_handler):
        """Test successful test conclusion."""
        test = ab_handler.manager.start_test("claude", 1, 2)
        # Add enough results
        for _ in range(10):
            ab_handler.manager.record_result("claude", f"debate-{_}", "evolved", True)

        result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test.id}/conclude",
            {},
        )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["result"]["winner"] == "evolved"
        assert body["result"]["confidence"] == 0.95

    def test_conclude_test_insufficient_data(self, ab_handler):
        """Test conclusion fails with insufficient data."""
        test = ab_handler.manager.start_test("claude", 1, 2)
        # Only 3 results
        for i in range(3):
            ab_handler.manager.record_result("claude", f"debate-{i}", "evolved", True)

        result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test.id}/conclude",
            {},
        )

        assert result.status_code == 400

    def test_conclude_test_force(self, ab_handler):
        """Test force conclusion with insufficient data."""
        test = ab_handler.manager.start_test("claude", 1, 2)
        # Only 3 results
        for i in range(3):
            ab_handler.manager.record_result("claude", f"debate-{i}", "evolved", True)

        result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test.id}/conclude",
            {"force": True},
        )

        assert result.status_code == 200

    def test_conclude_test_not_found(self, ab_handler):
        """Test conclusion for non-existent test."""
        result = ab_handler.handle_post(
            "/api/evolution/ab-tests/nonexistent/conclude",
            {},
        )

        assert result.status_code == 400  # Manager raises ValueError


# =============================================================================
# Cancel Test Tests
# =============================================================================


class TestCancelTest:
    """Tests for cancelling A/B tests."""

    def test_cancel_test_success(self, ab_handler):
        """Test successful test cancellation."""
        test = ab_handler.manager.start_test("claude", 1, 2)

        result = ab_handler.handle_delete(
            f"/api/evolution/ab-tests/{test.id}",
        )

        assert result.status_code == 200
        import json
        body = json.loads(result.body)
        assert body["test_id"] == test.id

    def test_cancel_test_not_found(self, ab_handler):
        """Test cancelling non-existent test."""
        result = ab_handler.handle_delete(
            "/api/evolution/ab-tests/nonexistent",
        )

        assert result.status_code == 404

    def test_cancel_test_already_concluded(self, ab_handler):
        """Test cancelling already concluded test."""
        test = ab_handler.manager.start_test("claude", 1, 2)
        test.status = MockABTestStatus.CONCLUDED

        result = ab_handler.handle_delete(
            f"/api/evolution/ab-tests/{test.id}",
        )

        assert result.status_code == 404


# =============================================================================
# Manager Not Configured Tests
# =============================================================================


class TestManagerNotConfigured:
    """Tests for when manager is not configured."""

    def test_list_without_manager(self, ab_handler):
        """Test listing without manager returns 503."""
        ab_handler._manager = None

        # Mock the manager property to return None
        with patch.object(
            type(ab_handler), "manager",
            new_callable=lambda: property(lambda self: None)
        ):
            result = ab_handler._list_tests({})

        assert result.status_code == 503

    def test_get_without_manager(self, ab_handler):
        """Test getting test without manager returns 503."""
        ab_handler._manager = None

        with patch.object(
            type(ab_handler), "manager",
            new_callable=lambda: property(lambda self: None)
        ):
            result = ab_handler._get_test("test-123")

        assert result.status_code == 503

    def test_create_without_manager(self, ab_handler):
        """Test creating test without manager returns 503."""
        ab_handler._manager = None

        with patch.object(
            type(ab_handler), "manager",
            new_callable=lambda: property(lambda self: None)
        ):
            result = ab_handler._create_test({"agent": "claude"})

        assert result.status_code == 503


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for A/B testing flow."""

    def test_full_ab_test_lifecycle(self, ab_handler):
        """Test complete A/B test lifecycle."""
        # 1. Create test
        create_result = ab_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
                "baseline_version": 1,
                "evolved_version": 2,
            },
        )
        assert create_result.status_code == 201

        import json
        test_id = json.loads(create_result.body)["test"]["id"]

        # 2. Check it's the active test
        active_result = ab_handler.handle(
            "/api/evolution/ab-tests/claude/active",
            {},
        )
        assert active_result.status_code == 200
        active_body = json.loads(active_result.body)
        assert active_body["has_active_test"] is True

        # 3. Record results (always record as wins to count as results)
        for i in range(10):
            variant = "evolved" if i % 2 == 0 else "baseline"
            record_result = ab_handler.handle_post(
                f"/api/evolution/ab-tests/{test_id}/record",
                {
                    "debate_id": f"debate-{i}",
                    "variant": variant,
                    "won": True,  # Always record as win so it counts
                },
            )
            assert record_result.status_code == 200

        # 4. Conclude test
        conclude_result = ab_handler.handle_post(
            f"/api/evolution/ab-tests/{test_id}/conclude",
            {},
        )
        assert conclude_result.status_code == 200
        conclude_body = json.loads(conclude_result.body)
        assert "winner" in conclude_body["result"]

        # 5. Verify no active test after conclusion
        active_result2 = ab_handler.handle(
            "/api/evolution/ab-tests/claude/active",
            {},
        )
        active_body2 = json.loads(active_result2.body)
        assert active_body2["has_active_test"] is False

    def test_cancel_and_create_new(self, ab_handler):
        """Test cancelling and creating a new test."""
        # Create first test
        ab_handler.handle_post(
            "/api/evolution/ab-tests",
            {"agent": "claude", "baseline_version": 1, "evolved_version": 2},
        )

        import json

        # Get test ID
        list_result = ab_handler.handle(
            "/api/evolution/ab-tests",
            {"agent": "claude"},
        )
        test_id = json.loads(list_result.body)["tests"][0]["id"]

        # Cancel it
        cancel_result = ab_handler.handle_delete(
            f"/api/evolution/ab-tests/{test_id}",
        )
        assert cancel_result.status_code == 200

        # Create new test should succeed
        create_result = ab_handler.handle_post(
            "/api/evolution/ab-tests",
            {"agent": "claude", "baseline_version": 2, "evolved_version": 3},
        )
        assert create_result.status_code == 201

