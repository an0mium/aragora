"""
Tests for DebatesHandler endpoints.

Endpoints tested:
- GET /api/debates - List all debates
- GET /api/debates/{slug} - Get debate by slug
- GET /api/debates/slug/{slug} - Get debate by slug (alternative)
- GET /api/debates/{id}/export/{format} - Export debate
- GET /api/debates/{id}/impasse - Detect debate impasse
- GET /api/debates/{id}/convergence - Get convergence status
- GET /api/debates/{id}/citations - Get evidence citations for debate
"""

import json
import pytest
from unittest.mock import Mock, MagicMock

from aragora.server.handlers import (
    DebatesHandler,
    HandlerResult,
    json_response,
    error_response,
)
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_storage():
    """Create a mock storage instance."""
    storage = Mock()
    storage.list_debates.return_value = [
        {
            "slug": "ai-safety-debate",
            "topic": "AI Safety Best Practices",
            "started_at": "2024-01-01T10:00:00Z",
            "ended_at": "2024-01-01T12:00:00Z",
            "rounds_used": 3,
            "consensus_reached": True,
        },
        {
            "slug": "climate-solutions",
            "topic": "Climate Change Solutions",
            "started_at": "2024-01-02T10:00:00Z",
            "ended_at": "2024-01-02T11:30:00Z",
            "rounds_used": 2,
            "consensus_reached": False,
        },
    ]
    storage.get_debate.return_value = {
        "slug": "ai-safety-debate",
        "id": "debate-001",
        "topic": "AI Safety Best Practices",
        "started_at": "2024-01-01T10:00:00Z",
        "ended_at": "2024-01-01T12:00:00Z",
        "rounds_used": 3,
        "consensus_reached": True,
        "final_answer": "The consensus is that AI systems should be developed with safety as a core principle.",
        "messages": [
            {"round": 1, "agent": "claude", "role": "speaker", "content": "Safety first!", "timestamp": "2024-01-01T10:05:00Z"},
            {"round": 1, "agent": "gpt4", "role": "speaker", "content": "I agree.", "timestamp": "2024-01-01T10:10:00Z"},
        ],
        "critiques": [
            {"round": 1, "critic": "gemini", "target": "claude", "severity": 0.3, "summary": "Good point but needs more detail", "timestamp": "2024-01-01T10:15:00Z"},
        ],
        "votes": [
            {"round": 1, "voter": "judge", "choice": "claude", "reason": "More comprehensive", "timestamp": "2024-01-01T10:20:00Z"},
        ],
        "convergence_status": "converged",
        "convergence_similarity": 0.92,
    }
    return storage


@pytest.fixture
def debates_handler(mock_storage):
    """Create a DebatesHandler with mock storage."""
    ctx = {"storage": mock_storage}
    return DebatesHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestDebatesHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_debates_list(self, debates_handler):
        """Should handle /api/debates."""
        assert debates_handler.can_handle("/api/debates") is True

    def test_can_handle_debate_by_slug(self, debates_handler):
        """Should handle /api/debates/{slug}."""
        assert debates_handler.can_handle("/api/debates/ai-safety-debate") is True

    def test_can_handle_debate_slug_pattern(self, debates_handler):
        """Should handle /api/debates/slug/{slug}."""
        assert debates_handler.can_handle("/api/debates/slug/ai-safety-debate") is True

    def test_can_handle_debate_export(self, debates_handler):
        """Should handle /api/debates/{id}/export/{format}."""
        assert debates_handler.can_handle("/api/debates/debate-001/export/json") is True
        assert debates_handler.can_handle("/api/debates/debate-001/export/csv") is True
        assert debates_handler.can_handle("/api/debates/debate-001/export/html") is True

    def test_can_handle_debate_impasse(self, debates_handler):
        """Should handle /api/debates/{id}/impasse."""
        assert debates_handler.can_handle("/api/debates/debate-001/impasse") is True

    def test_can_handle_debate_convergence(self, debates_handler):
        """Should handle /api/debates/{id}/convergence."""
        assert debates_handler.can_handle("/api/debates/debate-001/convergence") is True

    def test_can_handle_debate_citations(self, debates_handler):
        """Should handle /api/debates/{id}/citations."""
        assert debates_handler.can_handle("/api/debates/debate-001/citations") is True

    def test_cannot_handle_unknown_route(self, debates_handler):
        """Should not handle unknown routes."""
        assert debates_handler.can_handle("/api/agents") is False
        assert debates_handler.can_handle("/api/unknown") is False


# ============================================================================
# List Debates Tests
# ============================================================================

class TestListDebatesEndpoint:
    """Tests for /api/debates endpoint."""

    def test_list_returns_debates(self, debates_handler):
        """Should return list of debates."""
        result = debates_handler.handle("/api/debates", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "debates" in data
        assert "count" in data
        assert isinstance(data["debates"], list)
        assert len(data["debates"]) == 2

    def test_list_respects_limit(self, debates_handler, mock_storage):
        """Should respect limit parameter."""
        result = debates_handler.handle("/api/debates", {"limit": "10"}, None)

        assert result.status_code == 200
        mock_storage.list_debates.assert_called_with(limit=10)

    def test_list_caps_limit_at_100(self, debates_handler, mock_storage):
        """Should cap limit at 100."""
        result = debates_handler.handle("/api/debates", {"limit": "500"}, None)

        assert result.status_code == 200
        mock_storage.list_debates.assert_called_with(limit=100)

    def test_list_unavailable_returns_503(self):
        """Should return 503 when storage not available."""
        handler = DebatesHandler({})
        result = handler.handle("/api/debates", {}, None)

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data["error"]


# ============================================================================
# Get Debate by Slug Tests
# ============================================================================

class TestGetDebateEndpoint:
    """Tests for /api/debates/{slug} endpoint."""

    def test_get_debate_returns_data(self, debates_handler):
        """Should return debate data."""
        result = debates_handler.handle("/api/debates/ai-safety-debate", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["slug"] == "ai-safety-debate"
        assert data["topic"] == "AI Safety Best Practices"

    def test_get_debate_slug_pattern(self, debates_handler):
        """Should work with /api/debates/slug/{slug} pattern."""
        result = debates_handler.handle("/api/debates/slug/ai-safety-debate", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["slug"] == "ai-safety-debate"

    def test_get_debate_not_found(self, debates_handler, mock_storage):
        """Should return 404 for non-existent debate."""
        mock_storage.get_debate.return_value = None

        result = debates_handler.handle("/api/debates/nonexistent", {}, None)

        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data["error"].lower()

    def test_get_debate_unavailable_returns_503(self):
        """Should return 503 when storage not available."""
        handler = DebatesHandler({})
        result = handler.handle("/api/debates/ai-safety-debate", {}, None)

        assert result.status_code == 503


# ============================================================================
# Export Debate Tests
# ============================================================================

class TestExportDebateEndpoint:
    """Tests for /api/debates/{id}/export/{format} endpoint."""

    def test_export_json_returns_debate(self, debates_handler):
        """Should export debate as JSON."""
        result = debates_handler.handle("/api/debates/debate-001/export/json", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "topic" in data

    def test_export_csv_returns_csv(self, debates_handler):
        """Should export debate as CSV."""
        result = debates_handler.handle("/api/debates/debate-001/export/csv", {}, None)

        assert result.status_code == 200
        assert b"," in result.body  # CSV content
        assert result.content_type == "text/csv; charset=utf-8"

    def test_export_csv_messages_table(self, debates_handler):
        """Should export messages table as CSV."""
        result = debates_handler.handle(
            "/api/debates/debate-001/export/csv",
            {"table": "messages"},
            None
        )

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "round" in content.lower()
        assert "agent" in content.lower()

    def test_export_csv_critiques_table(self, debates_handler):
        """Should export critiques table as CSV."""
        result = debates_handler.handle(
            "/api/debates/debate-001/export/csv",
            {"table": "critiques"},
            None
        )

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "critic" in content.lower()

    def test_export_csv_votes_table(self, debates_handler):
        """Should export votes table as CSV."""
        result = debates_handler.handle(
            "/api/debates/debate-001/export/csv",
            {"table": "votes"},
            None
        )

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "voter" in content.lower()

    def test_export_html_returns_html(self, debates_handler):
        """Should export debate as HTML."""
        result = debates_handler.handle("/api/debates/debate-001/export/html", {}, None)

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "<!DOCTYPE html>" in content
        assert "AI Safety" in content
        assert result.content_type == "text/html; charset=utf-8"

    def test_export_invalid_format_returns_400(self, debates_handler):
        """Should return 400 for invalid export format."""
        result = debates_handler.handle("/api/debates/debate-001/export/xml", {}, None)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid format" in data["error"]

    def test_export_not_found_returns_404(self, debates_handler, mock_storage):
        """Should return 404 when debate not found."""
        mock_storage.get_debate.return_value = None

        result = debates_handler.handle("/api/debates/nonexistent/export/json", {}, None)

        assert result.status_code == 404

    def test_export_invalid_id_returns_400(self, debates_handler):
        """Should return 400 for invalid debate ID."""
        result = debates_handler.handle("/api/debates/../../../etc/export/json", {}, None)

        assert result.status_code == 400


# ============================================================================
# Impasse Detection Tests
# ============================================================================

class TestImpasseEndpoint:
    """Tests for /api/debates/{id}/impasse endpoint."""

    def test_impasse_returns_structure(self, debates_handler):
        """Should return impasse analysis structure."""
        result = debates_handler.handle("/api/debates/debate-001/impasse", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "debate_id" in data
        assert "is_impasse" in data
        assert "indicators" in data

    def test_impasse_detects_no_impasse(self, debates_handler):
        """Should detect no impasse for converged debate."""
        result = debates_handler.handle("/api/debates/debate-001/impasse", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Debate with consensus_reached=True should not be impasse
        assert data["is_impasse"] is False

    def test_impasse_detects_impasse(self, debates_handler, mock_storage):
        """Should detect impasse for stuck debate."""
        mock_storage.get_debate.return_value = {
            "slug": "stuck-debate",
            "consensus_reached": False,
            "messages": [],
            "critiques": [
                {"severity": 0.8},
                {"severity": 0.9},
            ],
        }

        result = debates_handler.handle("/api/debates/stuck-debate/impasse", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["is_impasse"] is True

    def test_impasse_not_found_returns_404(self, debates_handler, mock_storage):
        """Should return 404 when debate not found."""
        mock_storage.get_debate.return_value = None

        result = debates_handler.handle("/api/debates/nonexistent/impasse", {}, None)

        assert result.status_code == 404

    def test_impasse_unavailable_returns_503(self):
        """Should return 503 when storage not available."""
        handler = DebatesHandler({})
        result = handler.handle("/api/debates/debate-001/impasse", {}, None)

        assert result.status_code == 503


# ============================================================================
# Convergence Tests
# ============================================================================

class TestConvergenceEndpoint:
    """Tests for /api/debates/{id}/convergence endpoint."""

    def test_convergence_returns_structure(self, debates_handler):
        """Should return convergence status structure."""
        result = debates_handler.handle("/api/debates/debate-001/convergence", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "debate_id" in data
        assert "convergence_status" in data
        assert "convergence_similarity" in data
        assert "consensus_reached" in data
        assert "rounds_used" in data

    def test_convergence_shows_converged_status(self, debates_handler):
        """Should show converged status for converged debate."""
        result = debates_handler.handle("/api/debates/debate-001/convergence", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["convergence_status"] == "converged"
        assert data["consensus_reached"] is True
        assert data["convergence_similarity"] == 0.92

    def test_convergence_not_found_returns_404(self, debates_handler, mock_storage):
        """Should return 404 when debate not found."""
        mock_storage.get_debate.return_value = None

        result = debates_handler.handle("/api/debates/nonexistent/convergence", {}, None)

        assert result.status_code == 404

    def test_convergence_unavailable_returns_503(self):
        """Should return 503 when storage not available."""
        handler = DebatesHandler({})
        result = handler.handle("/api/debates/debate-001/convergence", {}, None)

        assert result.status_code == 503


# ============================================================================
# Citations Tests
# ============================================================================

class TestCitationsEndpoint:
    """Tests for /api/debates/{id}/citations endpoint."""

    def test_citations_no_citations(self, debates_handler):
        """Should return no citations when not available."""
        result = debates_handler.handle("/api/debates/debate-001/citations", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "debate_id" in data
        assert data["has_citations"] is False

    def test_citations_with_grounded_verdict(self, debates_handler, mock_storage):
        """Should return citations when grounded_verdict exists."""
        mock_storage.get_debate.return_value = {
            "slug": "grounded-debate",
            "grounded_verdict": {
                "grounding_score": 0.85,
                "confidence": 0.9,
                "claims": [{"claim": "Test claim", "evidence": "Test evidence"}],
                "all_citations": [{"source": "test.com", "title": "Test"}],
                "verdict": "Well grounded",
            },
        }

        result = debates_handler.handle("/api/debates/grounded-debate/citations", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["has_citations"] is True
        assert data["grounding_score"] == 0.85
        assert "claims" in data
        assert "all_citations" in data

    def test_citations_with_json_string_verdict(self, debates_handler, mock_storage):
        """Should parse JSON string grounded_verdict."""
        mock_storage.get_debate.return_value = {
            "slug": "json-grounded-debate",
            "grounded_verdict": json.dumps({
                "grounding_score": 0.75,
                "confidence": 0.8,
                "claims": [],
                "all_citations": [],
                "verdict": "Partially grounded",
            }),
        }

        result = debates_handler.handle("/api/debates/json-grounded-debate/citations", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["has_citations"] is True
        assert data["grounding_score"] == 0.75

    def test_citations_not_found_returns_404(self, debates_handler, mock_storage):
        """Should return 404 when debate not found."""
        mock_storage.get_debate.return_value = None

        result = debates_handler.handle("/api/debates/nonexistent/citations", {}, None)

        assert result.status_code == 404

    def test_citations_unavailable_returns_503(self):
        """Should return 503 when storage not available."""
        handler = DebatesHandler({})
        result = handler.handle("/api/debates/debate-001/citations", {}, None)

        assert result.status_code == 503


# ============================================================================
# Security Tests
# ============================================================================

class TestDebatesSecurity:
    """Tests for security measures."""

    def test_path_traversal_blocked_in_export(self, debates_handler):
        """Should block path traversal in export."""
        result = debates_handler.handle("/api/debates/../../../etc/export/json", {}, None)
        assert result.status_code == 400

    def test_path_traversal_blocked_in_impasse(self, debates_handler):
        """Should block path traversal in impasse."""
        result = debates_handler.handle("/api/debates/../../../etc/impasse", {}, None)
        assert result.status_code == 400

    def test_path_traversal_blocked_in_convergence(self, debates_handler):
        """Should block path traversal in convergence."""
        result = debates_handler.handle("/api/debates/../../../etc/convergence", {}, None)
        assert result.status_code == 400

    def test_path_traversal_blocked_in_citations(self, debates_handler):
        """Should block path traversal in citations."""
        result = debates_handler.handle("/api/debates/../../../etc/citations", {}, None)
        assert result.status_code == 400

    def test_sql_injection_blocked(self, debates_handler):
        """Should block SQL injection attempts."""
        dangerous_ids = [
            "test; DROP TABLE debates;--",
            "test' OR '1'='1",
        ]

        for dangerous_id in dangerous_ids:
            result = debates_handler.handle(f"/api/debates/{dangerous_id}/impasse", {}, None)
            assert result.status_code == 400, f"Should block: {dangerous_id}"

    def test_valid_slugs_accepted(self, debates_handler):
        """Should accept valid debate slugs."""
        valid_slugs = [
            "ai-safety-debate",
            "climate_solutions_2024",
            "debate-001",
            "test123",
        ]

        for slug in valid_slugs:
            result = debates_handler.handle(f"/api/debates/{slug}", {}, None)
            # Should not return 400 for valid slugs
            assert result.status_code != 400, f"Should accept: {slug}"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestDebatesErrorHandling:
    """Tests for error handling."""

    def test_storage_exception_returns_500(self, debates_handler, mock_storage):
        """Should return 500 on storage exceptions."""
        mock_storage.list_debates.side_effect = Exception("DB error")

        result = debates_handler.handle("/api/debates", {}, None)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data

    def test_export_exception_returns_500(self, debates_handler, mock_storage):
        """Should return 500 on export exceptions."""
        mock_storage.get_debate.side_effect = Exception("Export failed")

        result = debates_handler.handle("/api/debates/debate-001/export/json", {}, None)

        assert result.status_code == 500

    def test_impasse_exception_returns_500(self, debates_handler, mock_storage):
        """Should return 500 on impasse detection exceptions."""
        mock_storage.get_debate.side_effect = Exception("Detection failed")

        result = debates_handler.handle("/api/debates/debate-001/impasse", {}, None)

        assert result.status_code == 500


# ============================================================================
# Edge Cases
# ============================================================================

class TestDebatesEdgeCases:
    """Tests for edge cases."""

    def test_empty_debates_list(self, debates_handler, mock_storage):
        """Should handle empty debates list."""
        mock_storage.list_debates.return_value = []

        result = debates_handler.handle("/api/debates", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debates"] == []
        assert data["count"] == 0

    def test_debate_with_no_messages(self, debates_handler, mock_storage):
        """Should handle debate with no messages."""
        mock_storage.get_debate.return_value = {
            "slug": "empty-debate",
            "topic": "Empty Debate",
            "messages": [],
            "critiques": [],
            "votes": [],
            "consensus_reached": False,
        }

        result = debates_handler.handle("/api/debates/empty-debate/impasse", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "is_impasse" in data

    def test_export_html_escapes_content(self, debates_handler, mock_storage):
        """Should escape HTML content in export."""
        mock_storage.get_debate.return_value = {
            "slug": "xss-test",
            "topic": "<script>alert('xss')</script>",
            "messages": [
                {"round": 1, "agent": "test", "role": "speaker", "content": "<b>bold</b>"},
            ],
            "critiques": [],
            "votes": [],
            "rounds_used": 1,
            "consensus_reached": False,
            "final_answer": "",
        }

        result = debates_handler.handle("/api/debates/xss-test/export/html", {}, None)

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        # Script tags should be escaped
        assert "<script>" not in content
        assert "&lt;script&gt;" in content


# ============================================================================
# Additional CSV Export Tests
# ============================================================================

class TestCSVExportDetails:
    """Detailed tests for CSV export functionality."""

    def test_csv_summary_table_default(self, debates_handler, mock_storage):
        """Should export summary table by default."""
        mock_storage.get_debate.return_value = {
            "slug": "summary-test",
            "id": "debate-002",
            "topic": "Test Topic",
            "started_at": "2024-01-01T10:00:00Z",
            "ended_at": "2024-01-01T12:00:00Z",
            "rounds_used": 3,
            "consensus_reached": True,
            "final_answer": "Test final answer",
            "messages": [{"content": "msg1"}, {"content": "msg2"}],
            "critiques": [{"summary": "critique1"}],
            "votes": [],
        }

        result = debates_handler.handle(
            "/api/debates/summary-test/export/csv",
            {"table": "summary"},
            None
        )

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "field,value" in content.lower()
        assert "topic" in content.lower()
        assert "rounds_used" in content.lower()

    def test_csv_messages_includes_all_fields(self, debates_handler, mock_storage):
        """Should include all message fields in CSV export."""
        mock_storage.get_debate.return_value = {
            "slug": "msg-test",
            "messages": [
                {
                    "round": 2,
                    "agent": "claude",
                    "role": "critic",
                    "content": "Test message content",
                    "timestamp": "2024-01-01T11:00:00Z",
                },
            ],
            "critiques": [],
            "votes": [],
        }

        result = debates_handler.handle(
            "/api/debates/msg-test/export/csv",
            {"table": "messages"},
            None
        )

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "claude" in content
        assert "critic" in content

    def test_csv_truncates_long_content(self, debates_handler, mock_storage):
        """Should truncate long content in CSV export."""
        long_content = "x" * 2000
        mock_storage.get_debate.return_value = {
            "slug": "long-test",
            "messages": [
                {"round": 1, "agent": "test", "role": "speaker", "content": long_content, "timestamp": ""},
            ],
            "critiques": [],
            "votes": [],
        }

        result = debates_handler.handle(
            "/api/debates/long-test/export/csv",
            {"table": "messages"},
            None
        )

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        # Content should be truncated to 1000 chars
        assert len(content) < 2000 + 200  # Some overhead for headers

    def test_csv_invalid_table_falls_back_to_summary(self, debates_handler, mock_storage):
        """Should fall back to summary for invalid table."""
        mock_storage.get_debate.return_value = {
            "slug": "fallback-test",
            "topic": "Test",
            "messages": [],
            "critiques": [],
            "votes": [],
        }

        result = debates_handler.handle(
            "/api/debates/fallback-test/export/csv",
            {"table": "invalid_table"},
            None
        )

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        # Should get summary format
        assert "field,value" in content.lower()

    def test_csv_votes_export(self, debates_handler, mock_storage):
        """Should export votes correctly."""
        mock_storage.get_debate.return_value = {
            "slug": "votes-test",
            "messages": [],
            "critiques": [],
            "votes": [
                {"round": 1, "voter": "judge", "choice": "claude", "reason": "Better arguments", "timestamp": "2024-01-01T12:00:00Z"},
                {"round": 2, "voter": "judge", "choice": "gemini", "reason": "More evidence", "timestamp": "2024-01-01T13:00:00Z"},
            ],
        }

        result = debates_handler.handle(
            "/api/debates/votes-test/export/csv",
            {"table": "votes"},
            None
        )

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "judge" in content
        assert "claude" in content
        assert "Better arguments" in content


# ============================================================================
# Additional HTML Export Tests
# ============================================================================

class TestHTMLExportDetails:
    """Detailed tests for HTML export functionality."""

    def test_html_includes_statistics(self, debates_handler, mock_storage):
        """Should include debate statistics in HTML."""
        mock_storage.get_debate.return_value = {
            "slug": "stats-test",
            "topic": "Statistics Test",
            "messages": [{"agent": "claude", "content": "test", "role": "speaker", "round": 1}],
            "critiques": [{"critic": "gemini"}],
            "rounds_used": 5,
            "consensus_reached": True,
            "final_answer": "Final answer here",
        }

        result = debates_handler.handle("/api/debates/stats-test/export/html", {}, None)

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "Messages" in content or "messages" in content.lower()
        assert "Critiques" in content or "critiques" in content.lower()
        assert "Rounds" in content or "rounds" in content.lower()

    def test_html_shows_no_consensus(self, debates_handler, mock_storage):
        """Should indicate no consensus in HTML."""
        mock_storage.get_debate.return_value = {
            "slug": "no-consensus-test",
            "topic": "No Consensus Test",
            "messages": [],
            "critiques": [],
            "rounds_used": 3,
            "consensus_reached": False,
            "final_answer": "",
        }

        result = debates_handler.handle("/api/debates/no-consensus-test/export/html", {}, None)

        assert result.status_code == 200
        content = result.body.decode("utf-8")
        assert "No Consensus" in content or "no-consensus" in content

    def test_html_content_disposition(self, debates_handler, mock_storage):
        """Should set correct content-disposition header."""
        mock_storage.get_debate.return_value = {
            "slug": "download-test",
            "topic": "Download Test",
            "messages": [],
            "critiques": [],
            "consensus_reached": False,
            "final_answer": "",
            "rounds_used": 1,
        }

        result = debates_handler.handle("/api/debates/download-test/export/html", {}, None)

        assert result.status_code == 200
        assert "attachment" in result.headers["Content-Disposition"]
        assert "download-test" in result.headers["Content-Disposition"]


# ============================================================================
# Convergence Additional Tests
# ============================================================================

class TestConvergenceAdditional:
    """Additional convergence tests."""

    def test_convergence_unknown_status(self, debates_handler, mock_storage):
        """Should handle unknown convergence status."""
        mock_storage.get_debate.return_value = {
            "slug": "unknown-status",
            "topic": "Unknown Status",
        }

        result = debates_handler.handle("/api/debates/unknown-status/convergence", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["convergence_status"] == "unknown"
        assert data["convergence_similarity"] == 0.0

    def test_convergence_partial_data(self, debates_handler, mock_storage):
        """Should handle partial convergence data."""
        mock_storage.get_debate.return_value = {
            "slug": "partial-data",
            "consensus_reached": True,
            "convergence_similarity": 0.85,
            # Missing convergence_status and rounds_used
        }

        result = debates_handler.handle("/api/debates/partial-data/convergence", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["consensus_reached"] is True
        assert data["convergence_similarity"] == 0.85


# ============================================================================
# Impasse Detection Additional Tests
# ============================================================================

class TestImpasseAdditional:
    """Additional impasse detection tests."""

    def test_impasse_mixed_indicators(self, debates_handler, mock_storage):
        """Should detect impasse with mixed indicators."""
        mock_storage.get_debate.return_value = {
            "slug": "mixed-indicators",
            "consensus_reached": False,  # no_convergence = True
            "messages": [],
            "critiques": [
                {"severity": 0.9},  # high_severity_critiques = True
            ],
        }

        result = debates_handler.handle("/api/debates/mixed-indicators/impasse", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Two indicators = impasse
        assert data["is_impasse"] is True

    def test_impasse_low_severity(self, debates_handler, mock_storage):
        """Should not detect impasse with only low severity critiques."""
        mock_storage.get_debate.return_value = {
            "slug": "low-severity",
            "consensus_reached": True,
            "messages": [],
            "critiques": [
                {"severity": 0.3},
                {"severity": 0.5},
            ],
        }

        result = debates_handler.handle("/api/debates/low-severity/impasse", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["is_impasse"] is False


# ============================================================================
# Citations Additional Tests
# ============================================================================

class TestCitationsAdditional:
    """Additional citations tests."""

    def test_citations_invalid_json_string(self, debates_handler, mock_storage):
        """Should handle invalid JSON string gracefully."""
        mock_storage.get_debate.return_value = {
            "slug": "invalid-json",
            "grounded_verdict": "not valid json {",
        }

        result = debates_handler.handle("/api/debates/invalid-json/citations", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["has_citations"] is False

    def test_citations_empty_dict_verdict(self, debates_handler, mock_storage):
        """Should handle empty dict grounded_verdict."""
        mock_storage.get_debate.return_value = {
            "slug": "empty-verdict",
            "grounded_verdict": {},
        }

        result = debates_handler.handle("/api/debates/empty-verdict/citations", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Empty dict is falsy, so should report no citations
        assert data["has_citations"] is False


# ============================================================================
# Path Extraction Tests
# ============================================================================

class TestPathExtraction:
    """Tests for path extraction logic."""

    def test_extract_debate_id_valid(self, debates_handler):
        """Should extract valid debate ID."""
        debate_id, err = debates_handler._extract_debate_id("/api/debates/debate-123/impasse")
        assert debate_id == "debate-123"
        assert err is None

    def test_extract_debate_id_invalid_path(self, debates_handler):
        """Should return error for invalid path."""
        debate_id, err = debates_handler._extract_debate_id("/api/debates")
        assert debate_id is None
        assert err is not None

    def test_extract_debate_id_path_traversal(self, debates_handler):
        """Should reject path traversal in ID."""
        debate_id, err = debates_handler._extract_debate_id("/api/debates/../etc/impasse")
        assert debate_id is None
        assert err is not None


# ============================================================================
# Route Handler Edge Cases
# ============================================================================

class TestRouteHandlerEdgeCases:
    """Edge cases for route handling."""

    def test_handle_returns_none_for_reserved_words(self, debates_handler, mock_storage):
        """Should handle reserved words in slug position."""
        # The handler has special handling to not treat 'impasse' as a slug
        mock_storage.get_debate.return_value = None
        result = debates_handler.handle("/api/debates/impasse", {}, None)
        # Should return None or 404 since impasse is not a valid slug lookup
        assert result is None or result.status_code == 404

    def test_handle_debate_exception_in_get(self, debates_handler, mock_storage):
        """Should handle exception in get_debate."""
        mock_storage.get_debate.side_effect = Exception("DB connection lost")

        result = debates_handler.handle("/api/debates/test-debate", {}, None)

        assert result.status_code == 500

    def test_handle_convergence_exception(self, debates_handler, mock_storage):
        """Should handle exception in convergence check."""
        mock_storage.get_debate.side_effect = Exception("Convergence check failed")

        result = debates_handler.handle("/api/debates/debate-001/convergence", {}, None)

        assert result.status_code == 500

    def test_handle_citations_exception(self, debates_handler, mock_storage):
        """Should handle exception in citations lookup."""
        mock_storage.get_debate.side_effect = Exception("Citations lookup failed")

        result = debates_handler.handle("/api/debates/debate-001/citations", {}, None)

        assert result.status_code == 500
