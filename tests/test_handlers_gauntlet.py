"""
Tests for GauntletHandler endpoints.

Endpoints tested:
- GET /api/gauntlet/templates - List available Gauntlet templates
- GET /api/gauntlet/templates/{id} - Get template details
- POST /api/gauntlet/run - Run a Gauntlet validation
- GET /api/gauntlet/results/{id} - Get Gauntlet result
- GET /api/gauntlet/results/{id}/receipt - Get Decision Receipt
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass, field

from aragora.server.handlers import GauntletHandler, HandlerResult
from aragora.server.handlers.base import clear_cache
import aragora.server.handlers.gauntlet as gauntlet_module


# ============================================================================
# Mock Classes for Gauntlet Types
# ============================================================================

@dataclass
class MockGauntletFinding:
    """Mock finding for tests."""
    id: str = "finding-001"
    title: str = "Test Finding"
    description: str = "A test finding description"
    severity: str = "medium"


@dataclass
class MockGauntletResult:
    """Mock GauntletResult for tests."""
    id: str = "result-001"
    passed: bool = True
    confidence: float = 0.85
    verdict_summary: str = "PASS: No critical issues"
    robustness_score: float = 0.9
    risk_score: float = 0.1
    total_duration_ms: int = 1500
    findings: list = field(default_factory=list)
    severity_counts: dict = field(default_factory=lambda: {"critical": 0, "high": 0, "medium": 1, "low": 0})
    critical_findings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "passed": self.passed,
            "confidence": self.confidence,
            "verdict_summary": self.verdict_summary,
            "robustness_score": self.robustness_score,
            "risk_score": self.risk_score,
            "total_duration_ms": self.total_duration_ms,
            "findings_count": len(self.findings),
            "severity_counts": self.severity_counts,
        }


@dataclass
class MockDecisionReceipt:
    """Mock DecisionReceipt for tests."""
    receipt_id: str = "receipt-001"
    gauntlet_id: str = "result-001"
    verdict: str = "PASS"
    confidence: float = 0.85

    def to_dict(self):
        return {
            "receipt_id": self.receipt_id,
            "gauntlet_id": self.gauntlet_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
        }

    def to_markdown(self):
        return f"# Decision Receipt\n\n**Verdict:** {self.verdict}\n**Confidence:** {self.confidence:.1%}"

    def to_html(self):
        return f"<h1>Decision Receipt</h1><p>Verdict: {self.verdict}</p>"

    @classmethod
    def from_gauntlet_result(cls, result):
        return cls(gauntlet_id=result.id, confidence=result.confidence)


@dataclass
class MockGauntletConfig:
    """Mock GauntletConfig for tests."""
    attack_rounds: int = 2
    probe_categories: list = field(default_factory=list)

    def to_dict(self):
        return {
            "attack_rounds": self.attack_rounds,
            "probe_categories": self.probe_categories,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def mock_list_templates():
    """Mock list_templates function."""
    return [
        {"id": "quick", "name": "Quick Validation", "description": "Fast stress-test"},
        {"id": "thorough", "name": "Thorough Validation", "description": "Comprehensive analysis"},
        {"id": "security", "name": "Security Assessment", "description": "Security-focused"},
    ]


def mock_get_template(template_id: str):
    """Mock get_template function."""
    templates = {"quick": MockGauntletConfig(), "thorough": MockGauntletConfig(), "security": MockGauntletConfig()}
    if template_id not in templates:
        raise ValueError(f"Unknown template: {template_id}")
    return templates[template_id]


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_gauntlet_module():
    """Create mock gauntlet module components."""
    return {
        "templates": [
            {"id": "quick", "name": "Quick Validation", "description": "Fast stress-test"},
            {"id": "thorough", "name": "Thorough Validation", "description": "Comprehensive analysis"},
            {"id": "security", "name": "Security Assessment", "description": "Security-focused"},
        ],
        "config": MockGauntletConfig(),
        "result": MockGauntletResult(),
        "receipt": MockDecisionReceipt(),
    }


@pytest.fixture
def gauntlet_handler(mock_gauntlet_module):
    """Create a GauntletHandler with mocked dependencies."""
    ctx = {"nomic_dir": "/tmp/nomic"}
    handler = GauntletHandler(ctx)

    # Cache a mock result for retrieval tests
    handler._results_cache["result-001"] = MockGauntletResult()

    return handler


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestGauntletHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_templates_list(self, gauntlet_handler):
        """Should handle /api/gauntlet/templates."""
        assert gauntlet_handler.can_handle("/api/gauntlet/templates") is True

    def test_can_handle_template_by_id(self, gauntlet_handler):
        """Should handle /api/gauntlet/templates/{id}."""
        assert gauntlet_handler.can_handle("/api/gauntlet/templates/quick") is True

    def test_can_handle_run(self, gauntlet_handler):
        """Should handle /api/gauntlet/run."""
        assert gauntlet_handler.can_handle("/api/gauntlet/run") is True

    def test_can_handle_results(self, gauntlet_handler):
        """Should handle /api/gauntlet/results/{id}."""
        assert gauntlet_handler.can_handle("/api/gauntlet/results/result-001") is True

    def test_can_handle_receipt(self, gauntlet_handler):
        """Should handle /api/gauntlet/results/{id}/receipt."""
        assert gauntlet_handler.can_handle("/api/gauntlet/results/result-001/receipt") is True

    def test_cannot_handle_unknown_route(self, gauntlet_handler):
        """Should not handle unknown routes."""
        assert gauntlet_handler.can_handle("/api/debates") is False
        assert gauntlet_handler.can_handle("/api/unknown") is False


# ============================================================================
# Template List Tests
# ============================================================================

class TestGauntletTemplates:
    """Tests for /api/gauntlet/templates endpoint."""

    def test_list_templates_returns_all(self, gauntlet_handler):
        """Should return list of templates."""
        # The handler uses _init_gauntlet() which sets up list_templates
        # If gauntlet module is available, it will return real templates
        result = gauntlet_handler.handle("/api/gauntlet/templates", {}, None)

        # Either 200 with templates or 503 if module unavailable
        if result.status_code == 200:
            data = json.loads(result.body)
            assert "templates" in data
            assert "count" in data
            assert isinstance(data["templates"], list)
        else:
            assert result.status_code == 503

    def test_list_templates_with_mocked_init(self, gauntlet_handler):
        """Should return templates when gauntlet is available."""
        # Inject mock functions
        gauntlet_module.list_templates = mock_list_templates
        gauntlet_module.get_template = mock_get_template
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            result = gauntlet_handler.handle("/api/gauntlet/templates", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "templates" in data
            assert len(data["templates"]) == 3
        finally:
            # Reset
            gauntlet_module._gauntlet_initialized = False

    def test_list_templates_unavailable(self):
        """Should return 503 when gauntlet not available."""
        # Reset initialization to force unavailable state
        original_init = gauntlet_module._gauntlet_initialized
        original_available = gauntlet_module.GAUNTLET_AVAILABLE
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.GAUNTLET_AVAILABLE = False

        try:
            handler = GauntletHandler({})
            result = handler.handle("/api/gauntlet/templates", {}, None)

            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"]
        finally:
            gauntlet_module._gauntlet_initialized = original_init
            gauntlet_module.GAUNTLET_AVAILABLE = original_available

    def test_get_template_success(self, gauntlet_handler):
        """Should return template details."""
        # Inject mock
        gauntlet_module.get_template = mock_get_template
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            result = gauntlet_handler.handle("/api/gauntlet/templates/quick", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["id"] == "quick"
            assert "config" in data
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_get_template_not_found(self, gauntlet_handler):
        """Should return 404 for unknown template."""
        gauntlet_module.get_template = mock_get_template
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            result = gauntlet_handler.handle("/api/gauntlet/templates/nonexistent", {}, None)
            assert result.status_code == 404
        finally:
            gauntlet_module._gauntlet_initialized = False


# ============================================================================
# Run Gauntlet Tests
# ============================================================================

class TestGauntletRun:
    """Tests for POST /api/gauntlet/run endpoint."""

    def test_run_requires_input_text(self):
        """Should require input_text in body."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            handler = GauntletHandler({})
            # Pre-set orchestrator to avoid lazy loading issues
            mock_orch = MagicMock()
            handler._orchestrator = mock_orch

            result = handler.handle_post("/api/gauntlet/run", {}, None)

            assert result.status_code == 400
            data = json.loads(result.body)
            assert "input_text" in data["error"].lower()
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_run_input_too_large(self):
        """Should reject input over 50KB."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            handler = GauntletHandler({})
            mock_orch = MagicMock()
            handler._orchestrator = mock_orch

            large_input = "x" * 60000  # 60KB

            result = handler.handle_post(
                "/api/gauntlet/run",
                {"input_text": large_input},
                None
            )

            assert result.status_code == 400
            data = json.loads(result.body)
            assert "too large" in data["error"].lower()
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_run_with_invalid_template(self):
        """Should return 400 for invalid template."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.get_template = lambda x: (_ for _ in ()).throw(ValueError("Unknown template"))

        try:
            handler = GauntletHandler({})
            mock_orch = MagicMock()
            handler._orchestrator = mock_orch

            result = handler.handle_post(
                "/api/gauntlet/run",
                {"input_text": "test input", "template": "nonexistent"},
                None
            )

            assert result.status_code == 400
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_run_unavailable(self):
        """Should return 503 when gauntlet not available."""
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.GAUNTLET_AVAILABLE = False

        try:
            handler = GauntletHandler({})
            result = handler.handle_post(
                "/api/gauntlet/run",
                {"input_text": "test"},
                None
            )

            assert result.status_code == 503
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_run_with_template_success(self):
        """Should run gauntlet with template and return result."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.get_template = mock_get_template

        try:
            mock_result = MockGauntletResult()
            mock_orch = MagicMock()

            # Mock the async run to return result directly
            async def mock_run(**kwargs):
                return mock_result
            mock_orch.run = mock_run

            handler = GauntletHandler({})
            handler._orchestrator = mock_orch

            result = handler.handle_post(
                "/api/gauntlet/run",
                {"input_text": "Test code to validate", "template": "quick"},
                None
            )

            assert result.status_code == 201
            data = json.loads(result.body)
            assert "id" in data
            assert "passed" in data
            assert "confidence" in data
        finally:
            gauntlet_module._gauntlet_initialized = False


# ============================================================================
# Results Tests
# ============================================================================

class TestGauntletResults:
    """Tests for /api/gauntlet/results/{id} endpoint."""

    def test_get_result_success(self, gauntlet_handler):
        """Should return cached result."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            result = gauntlet_handler.handle("/api/gauntlet/results/result-001", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["id"] == "result-001"
            assert data["passed"] is True
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_get_result_not_found(self, gauntlet_handler):
        """Should return 404 for unknown result."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            result = gauntlet_handler.handle("/api/gauntlet/results/nonexistent", {}, None)

            assert result.status_code == 404
            data = json.loads(result.body)
            assert "not found" in data["error"].lower()
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_get_result_validates_id(self, gauntlet_handler):
        """Should validate result ID format."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            result = gauntlet_handler.handle("/api/gauntlet/results/<script>", {}, None)
            assert result.status_code == 400
        finally:
            gauntlet_module._gauntlet_initialized = False


# ============================================================================
# Receipt Tests
# ============================================================================

class TestGauntletReceipt:
    """Tests for /api/gauntlet/results/{id}/receipt endpoint."""

    def test_get_receipt_json(self, gauntlet_handler):
        """Should return receipt as JSON."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.DecisionReceipt = MockDecisionReceipt

        try:
            result = gauntlet_handler.handle(
                "/api/gauntlet/results/result-001/receipt",
                {"format": "json"},
                None
            )

            assert result.status_code == 200
            assert result.content_type == "application/json"
            data = json.loads(result.body)
            assert "receipt_id" in data
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_get_receipt_markdown(self, gauntlet_handler):
        """Should return receipt as markdown."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.DecisionReceipt = MockDecisionReceipt

        try:
            result = gauntlet_handler.handle(
                "/api/gauntlet/results/result-001/receipt",
                {"format": "markdown"},
                None
            )

            assert result.status_code == 200
            assert result.content_type == "text/markdown"
            assert b"Decision Receipt" in result.body
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_get_receipt_html(self, gauntlet_handler):
        """Should return receipt as HTML."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.DecisionReceipt = MockDecisionReceipt

        try:
            result = gauntlet_handler.handle(
                "/api/gauntlet/results/result-001/receipt",
                {"format": "html"},
                None
            )

            assert result.status_code == 200
            assert result.content_type == "text/html"
            assert b"<h1>" in result.body
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_get_receipt_result_not_found(self, gauntlet_handler):
        """Should return 404 for unknown result."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            result = gauntlet_handler.handle(
                "/api/gauntlet/results/nonexistent/receipt",
                {},
                None
            )
            assert result.status_code == 404
        finally:
            gauntlet_module._gauntlet_initialized = False


# ============================================================================
# Edge Cases and Security Tests
# ============================================================================

class TestGauntletSecurity:
    """Security and edge case tests."""

    def test_path_traversal_in_template_id(self, gauntlet_handler):
        """Should reject path traversal attempts."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.get_template = mock_get_template

        try:
            # This tests validate_path_segment
            result = gauntlet_handler.handle("/api/gauntlet/templates/../../../etc/passwd", {}, None)

            # Should return 400 for path traversal or None if not matching
            assert result is None or result.status_code in [400, 404]
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_path_traversal_in_result_id(self, gauntlet_handler):
        """Should reject path traversal in result ID."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True

        try:
            result = gauntlet_handler.handle("/api/gauntlet/results/../sensitive", {}, None)

            # Should return 400 for path traversal or None if not matching
            assert result is None or result.status_code in [400, 404]
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_xss_in_format_param(self, gauntlet_handler):
        """Should sanitize format parameter."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.DecisionReceipt = MockDecisionReceipt

        try:
            result = gauntlet_handler.handle(
                "/api/gauntlet/results/result-001/receipt",
                {"format": "<script>alert(1)</script>"},
                None
            )

            # Should default to JSON for invalid format
            assert result.status_code == 200
            assert result.content_type == "application/json"
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_long_format_param_truncated(self, gauntlet_handler):
        """Should truncate overly long format parameter."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.DecisionReceipt = MockDecisionReceipt

        try:
            result = gauntlet_handler.handle(
                "/api/gauntlet/results/result-001/receipt",
                {"format": "x" * 100},
                None
            )

            # Should default to JSON
            assert result.status_code == 200
        finally:
            gauntlet_module._gauntlet_initialized = False


# ============================================================================
# Integration Tests
# ============================================================================

class TestGauntletIntegration:
    """Integration tests for full workflows."""

    def test_full_template_workflow(self, gauntlet_handler):
        """Test listing then fetching template."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.list_templates = mock_list_templates
        gauntlet_module.get_template = mock_get_template

        try:
            # List templates
            list_result = gauntlet_handler.handle("/api/gauntlet/templates", {}, None)
            assert list_result.status_code == 200

            # Get specific template
            get_result = gauntlet_handler.handle("/api/gauntlet/templates/quick", {}, None)
            assert get_result.status_code == 200
        finally:
            gauntlet_module._gauntlet_initialized = False

    def test_result_to_receipt_workflow(self, gauntlet_handler):
        """Test getting result then its receipt."""
        gauntlet_module.GAUNTLET_AVAILABLE = True
        gauntlet_module._gauntlet_initialized = True
        gauntlet_module.DecisionReceipt = MockDecisionReceipt

        try:
            # Get result
            result_resp = gauntlet_handler.handle("/api/gauntlet/results/result-001", {}, None)
            assert result_resp.status_code == 200

            # Get receipt
            receipt_resp = gauntlet_handler.handle(
                "/api/gauntlet/results/result-001/receipt",
                {},
                None
            )
            assert receipt_resp.status_code == 200
        finally:
            gauntlet_module._gauntlet_initialized = False
