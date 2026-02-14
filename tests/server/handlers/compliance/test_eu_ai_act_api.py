"""
Tests for EU AI Act Compliance Handler API endpoints.

Tests cover:
- Risk classification endpoint (POST /api/v2/compliance/eu-ai-act/classify)
- Conformity report endpoint (POST /api/v2/compliance/eu-ai-act/audit)
- Artifact bundle generation (POST /api/v2/compliance/eu-ai-act/generate-bundle)
- Input validation for all endpoints
- Route matching in ComplianceHandler.handle()
"""

from __future__ import annotations

import json
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.compliance.handler import ComplianceHandler


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def handler():
    """Create a compliance handler instance."""
    return ComplianceHandler(server_context={})


@pytest.fixture
def sample_receipt():
    """A realistic decision receipt for testing."""
    return {
        "receipt_id": "rcpt-test-001",
        "input_summary": "Evaluate candidate resumes for hiring decision",
        "verdict": "APPROVE",
        "verdict_reasoning": "Candidate meets requirements for employment screening role",
        "confidence": 0.85,
        "agents": [
            {"name": "analyst-1", "model": "claude-3", "role": "proposer"},
            {"name": "analyst-2", "model": "gpt-4", "role": "critic"},
        ],
        "dissenting_agents": [],
        "rounds": 3,
        "evidence_sources": ["internal_policy.pdf", "hr_guidelines.md"],
    }


@pytest.fixture
def minimal_receipt():
    """A minimal receipt with only required fields."""
    return {
        "receipt_id": "rcpt-minimal",
        "verdict": "REJECT",
    }


# ============================================================================
# Classify Endpoint Tests
# ============================================================================


class TestEUAIActClassify:
    """Tests for POST /api/v2/compliance/eu-ai-act/classify."""

    @pytest.mark.asyncio
    async def test_classify_high_risk_employment(self, handler):
        """Employment screening is classified as HIGH risk (Annex III)."""
        result = await handler._eu_ai_act_classify(
            {"description": "AI system for employment screening and hiring decisions"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        classification = body["classification"]
        assert classification["risk_level"] == "high"
        assert classification["annex_iii_category"] is not None
        assert len(classification["obligations"]) > 0

    @pytest.mark.asyncio
    async def test_classify_unacceptable_risk(self, handler):
        """Social scoring is classified as UNACCEPTABLE risk."""
        result = await handler._eu_ai_act_classify(
            {"description": "AI-based social scoring system for citizen behavior"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["classification"]["risk_level"] == "unacceptable"

    @pytest.mark.asyncio
    async def test_classify_limited_risk(self, handler):
        """Chatbot use case is LIMITED risk."""
        result = await handler._eu_ai_act_classify(
            {"description": "Customer service chatbot for answering FAQ questions"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["classification"]["risk_level"] == "limited"

    @pytest.mark.asyncio
    async def test_classify_minimal_risk(self, handler):
        """Generic analytics is MINIMAL risk."""
        result = await handler._eu_ai_act_classify(
            {"description": "Internal analytics dashboard for sales trends"}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["classification"]["risk_level"] == "minimal"

    @pytest.mark.asyncio
    async def test_classify_returns_rationale(self, handler):
        """Classification includes rationale."""
        result = await handler._eu_ai_act_classify(
            {"description": "AI for employment decisions"}
        )
        body = json.loads(result.body)
        assert body["classification"]["rationale"]

    @pytest.mark.asyncio
    async def test_classify_returns_applicable_articles(self, handler):
        """High-risk classification lists applicable articles."""
        result = await handler._eu_ai_act_classify(
            {"description": "AI system for employment screening and hiring decisions"}
        )
        body = json.loads(result.body)
        assert body["classification"]["risk_level"] == "high"
        assert len(body["classification"]["applicable_articles"]) > 0

    @pytest.mark.asyncio
    async def test_classify_missing_description(self, handler):
        """Missing description returns 400."""
        result = await handler._eu_ai_act_classify({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_classify_empty_description(self, handler):
        """Empty description returns 400."""
        result = await handler._eu_ai_act_classify({"description": "  "})
        assert result.status_code == 400


# ============================================================================
# Audit (Conformity Report) Endpoint Tests
# ============================================================================


class TestEUAIActAudit:
    """Tests for POST /api/v2/compliance/eu-ai-act/audit."""

    @pytest.mark.asyncio
    async def test_audit_generates_report(self, handler, sample_receipt):
        """Audit endpoint generates a conformity report."""
        result = await handler._eu_ai_act_audit({"receipt": sample_receipt})
        assert result.status_code == 200
        body = json.loads(result.body)
        report = body["conformity_report"]
        assert "overall_status" in report
        assert "article_mappings" in report

    @pytest.mark.asyncio
    async def test_audit_report_has_articles(self, handler, sample_receipt):
        """Conformity report includes article-by-article mappings."""
        result = await handler._eu_ai_act_audit({"receipt": sample_receipt})
        body = json.loads(result.body)
        report = body["conformity_report"]
        # Should have multiple article mappings (Articles 9-15)
        assert len(report.get("article_mappings", [])) >= 1

    @pytest.mark.asyncio
    async def test_audit_minimal_receipt(self, handler, minimal_receipt):
        """Audit works with minimal receipt data."""
        result = await handler._eu_ai_act_audit({"receipt": minimal_receipt})
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_audit_missing_receipt(self, handler):
        """Missing receipt returns 400."""
        result = await handler._eu_ai_act_audit({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_audit_invalid_receipt_type(self, handler):
        """Non-dict receipt returns 400."""
        result = await handler._eu_ai_act_audit({"receipt": "not-a-dict"})
        assert result.status_code == 400


# ============================================================================
# Generate Bundle Endpoint Tests
# ============================================================================


class TestEUAIActGenerateBundle:
    """Tests for POST /api/v2/compliance/eu-ai-act/generate-bundle."""

    @pytest.mark.asyncio
    async def test_generate_bundle_success(self, handler, sample_receipt):
        """Bundle generation returns a complete artifact bundle."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": sample_receipt}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        bundle = body["bundle"]
        assert bundle["bundle_id"].startswith("EUAIA-")
        assert bundle["receipt_id"] == "rcpt-test-001"
        assert bundle["regulation"] == "EU AI Act (Regulation 2024/1689)"

    @pytest.mark.asyncio
    async def test_bundle_contains_all_articles(self, handler, sample_receipt):
        """Bundle includes Articles 12, 13, and 14."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": sample_receipt}
        )
        body = json.loads(result.body)
        bundle = body["bundle"]
        assert "article_12_record_keeping" in bundle
        assert "article_13_transparency" in bundle
        assert "article_14_human_oversight" in bundle

    @pytest.mark.asyncio
    async def test_bundle_article_12_structure(self, handler, sample_receipt):
        """Article 12 artifact has expected structure."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": sample_receipt}
        )
        body = json.loads(result.body)
        art12 = body["bundle"]["article_12_record_keeping"]
        assert art12["article"] == "Article 12"
        assert art12["title"] == "Record-Keeping"
        assert "event_log" in art12
        assert "retention_policy" in art12

    @pytest.mark.asyncio
    async def test_bundle_article_13_structure(self, handler, sample_receipt):
        """Article 13 artifact has expected structure."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": sample_receipt}
        )
        body = json.loads(result.body)
        art13 = body["bundle"]["article_13_transparency"]
        assert art13["article"] == "Article 13"
        assert "provider_identity" in art13
        assert "known_risks" in art13

    @pytest.mark.asyncio
    async def test_bundle_article_14_structure(self, handler, sample_receipt):
        """Article 14 artifact has expected structure."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": sample_receipt}
        )
        body = json.loads(result.body)
        art14 = body["bundle"]["article_14_human_oversight"]
        assert art14["article"] == "Article 14"
        assert "oversight_model" in art14
        assert "override_capability" in art14

    @pytest.mark.asyncio
    async def test_bundle_has_integrity_hash(self, handler, sample_receipt):
        """Bundle includes SHA-256 integrity hash."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": sample_receipt}
        )
        body = json.loads(result.body)
        integrity_hash = body["bundle"]["integrity_hash"]
        assert len(integrity_hash) == 64  # SHA-256 hex digest

    @pytest.mark.asyncio
    async def test_bundle_has_risk_classification(self, handler, sample_receipt):
        """Bundle includes risk classification."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": sample_receipt}
        )
        body = json.loads(result.body)
        classification = body["bundle"]["risk_classification"]
        assert "risk_level" in classification
        assert "rationale" in classification

    @pytest.mark.asyncio
    async def test_bundle_has_conformity_report(self, handler, sample_receipt):
        """Bundle includes conformity report."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": sample_receipt}
        )
        body = json.loads(result.body)
        report = body["bundle"]["conformity_report"]
        assert "overall_status" in report

    @pytest.mark.asyncio
    async def test_bundle_with_custom_provider(self, handler, sample_receipt):
        """Bundle generation accepts custom provider details."""
        result = await handler._eu_ai_act_generate_bundle({
            "receipt": sample_receipt,
            "provider_name": "Acme Corp",
            "provider_contact": "legal@acme.com",
            "system_name": "Acme Decision Engine",
            "system_version": "1.0.0",
        })
        assert result.status_code == 200
        body = json.loads(result.body)
        provider = body["bundle"]["article_13_transparency"]["provider_identity"]
        assert provider["name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_bundle_minimal_receipt(self, handler, minimal_receipt):
        """Bundle generation works with minimal receipt data."""
        result = await handler._eu_ai_act_generate_bundle(
            {"receipt": minimal_receipt}
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["bundle"]["bundle_id"].startswith("EUAIA-")

    @pytest.mark.asyncio
    async def test_bundle_missing_receipt(self, handler):
        """Missing receipt returns 400."""
        result = await handler._eu_ai_act_generate_bundle({})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_bundle_invalid_receipt(self, handler):
        """Non-dict receipt returns 400."""
        result = await handler._eu_ai_act_generate_bundle({"receipt": 42})
        assert result.status_code == 400


# ============================================================================
# Route Integration Tests
# ============================================================================


class TestEUAIActRouting:
    """Tests for EU AI Act routes through ComplianceHandler.handle()."""

    @staticmethod
    def _make_mock_handler(body_dict: dict[str, Any], method: str = "POST"):
        """Create a mock HTTP handler with proper body and headers."""
        mock_handler = MagicMock()
        mock_handler.command = method
        body_bytes = json.dumps(body_dict).encode()
        mock_handler.rfile = MagicMock()
        mock_handler.rfile.read.return_value = body_bytes
        headers = MagicMock()
        headers.get = lambda k, d=None: (
            str(len(body_bytes)) if k == "Content-Length" else d
        )
        # Make headers iterable for dict(handler.headers)
        headers.__iter__ = lambda self: iter([])
        mock_handler.headers = headers
        return mock_handler

    @pytest.mark.asyncio
    async def test_classify_route(self, handler):
        """Classify route dispatches correctly."""
        mock_handler = self._make_mock_handler(
            {"description": "AI for employment decisions"}
        )
        result = await handler.handle(
            "/api/v2/compliance/eu-ai-act/classify", {}, mock_handler
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "classification" in body

    @pytest.mark.asyncio
    async def test_audit_route(self, handler, sample_receipt):
        """Audit route dispatches correctly."""
        mock_handler = self._make_mock_handler({"receipt": sample_receipt})
        result = await handler.handle(
            "/api/v2/compliance/eu-ai-act/audit", {}, mock_handler
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "conformity_report" in body

    @pytest.mark.asyncio
    async def test_generate_bundle_route(self, handler, sample_receipt):
        """Generate bundle route dispatches correctly."""
        mock_handler = self._make_mock_handler({"receipt": sample_receipt})
        result = await handler.handle(
            "/api/v2/compliance/eu-ai-act/generate-bundle", {}, mock_handler
        )
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "bundle" in body

    @pytest.mark.asyncio
    async def test_unknown_eu_ai_act_path_returns_404(self, handler):
        """Unknown sub-path returns 404."""
        mock_handler = self._make_mock_handler({}, method="GET")
        result = await handler.handle(
            "/api/v2/compliance/eu-ai-act/unknown", {}, mock_handler
        )
        assert result.status_code == 404
