"""Comprehensive tests for the EU AI Act compliance handler.

Tests the EUAIActMixin and its three endpoints:
    POST /api/v2/compliance/eu-ai-act/classify
    POST /api/v2/compliance/eu-ai-act/audit
    POST /api/v2/compliance/eu-ai-act/generate-bundle

Covers:
- Normal operation for all three endpoints
- Missing/invalid body parameters (error_response paths)
- Lazy-loading of classifier, report generator, and artifact generator
- Provider kwargs forwarding for generate-bundle
- Edge cases: empty strings, whitespace-only, non-dict receipt, etc.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.compliance.eu_ai_act import (
    EUAIActMixin,
    _get_artifact_generator,
    _get_classifier,
    _get_report_generator,
)
from aragora.server.handlers.base import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Concrete class from the mixin (needs a real instance to call methods)
# ---------------------------------------------------------------------------


class _Handler(EUAIActMixin):
    """Minimal concrete class that incorporates the mixin."""

    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_lazy_singletons():
    """Reset the module-level lazy-loaded singletons between tests."""
    import aragora.server.handlers.compliance.eu_ai_act as mod

    mod._classifier = None
    mod._report_generator = None
    mod._artifact_generator = None
    yield
    mod._classifier = None
    mod._report_generator = None
    mod._artifact_generator = None


@pytest.fixture
def handler():
    return _Handler()


@pytest.fixture
def mock_classification():
    """A mock RiskClassification with a to_dict method."""
    mock = MagicMock()
    mock.to_dict.return_value = {
        "risk_level": "high",
        "annex_iii_category": "Employment and worker management",
        "annex_iii_number": 4,
        "rationale": "Matched recruitment keywords",
        "matched_keywords": ["recruitment"],
        "applicable_articles": ["Article 6", "Article 9"],
        "obligations": ["Risk management system required."],
    }
    return mock


@pytest.fixture
def mock_report():
    """A mock ConformityReport with a to_dict method."""
    mock = MagicMock()
    mock.to_dict.return_value = {
        "report_id": "EUAIA-abc12345",
        "receipt_id": "rcpt-001",
        "overall_status": "partial",
        "article_mappings": [],
        "summary": "Assessment complete",
    }
    return mock


@pytest.fixture
def mock_bundle():
    """A mock ComplianceArtifactBundle with a to_dict method."""
    mock = MagicMock()
    mock.to_dict.return_value = {
        "bundle_id": "EUAIA-bndl0001",
        "receipt_id": "rcpt-002",
        "regulation": "EU AI Act (Regulation 2024/1689)",
        "integrity_hash": "abcdef1234567890",
    }
    return mock


@pytest.fixture
def sample_receipt() -> dict[str, Any]:
    return {
        "receipt_id": "rcpt-001",
        "input_summary": "Evaluate recruitment candidates",
        "verdict": "approved",
        "verdict_reasoning": "Consensus reached on top candidate",
        "confidence": 0.85,
        "robustness_score": 0.72,
        "risk_summary": {"total": 3, "critical": 0},
        "provenance_chain": [
            {"event_type": "debate_start", "timestamp": "2026-01-01T00:00:00Z"},
            {"event_type": "consensus", "timestamp": "2026-01-01T00:01:00Z"},
        ],
        "consensus_proof": {
            "supporting_agents": ["agent-a", "agent-b"],
            "dissenting_agents": ["agent-c"],
            "method": "weighted_majority",
            "agreement_ratio": 0.67,
        },
        "dissenting_views": [{"agent": "agent-c", "reason": "Insufficient data"}],
        "config_used": {"rounds": 3, "protocol": "adversarial"},
        "artifact_hash": "sha256:abc",
        "signature": "sig:xyz",
    }


# =========================================================================
# _eu_ai_act_classify tests
# =========================================================================


class TestClassifyEndpoint:
    """Tests for POST /api/v2/compliance/eu-ai-act/classify."""

    @pytest.mark.asyncio
    async def test_classify_success(self, handler, mock_classification):
        """Happy path: valid description returns classification."""
        with patch("aragora.server.handlers.compliance.eu_ai_act._get_classifier") as mock_get:
            classifier = MagicMock()
            classifier.classify.return_value = mock_classification
            mock_get.return_value = classifier

            result = await handler._eu_ai_act_classify(
                {"description": "recruitment AI for screening CVs"}
            )

        assert _status(result) == 200
        body = _body(result)
        assert "classification" in body
        assert body["classification"]["risk_level"] == "high"

    @pytest.mark.asyncio
    async def test_classify_missing_description(self, handler):
        """Missing 'description' field returns 400."""
        result = await handler._eu_ai_act_classify({})
        assert _status(result) == 400
        assert "description" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_classify_empty_description(self, handler):
        """Empty string description returns 400."""
        result = await handler._eu_ai_act_classify({"description": ""})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_classify_whitespace_only_description(self, handler):
        """Whitespace-only description returns 400 after strip."""
        result = await handler._eu_ai_act_classify({"description": "   \n\t  "})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_classify_strips_whitespace(self, handler, mock_classification):
        """Leading/trailing whitespace is stripped before classification."""
        with patch("aragora.server.handlers.compliance.eu_ai_act._get_classifier") as mock_get:
            classifier = MagicMock()
            classifier.classify.return_value = mock_classification
            mock_get.return_value = classifier

            result = await handler._eu_ai_act_classify({"description": "  recruitment  "})

        assert _status(result) == 200
        classifier.classify.assert_called_once_with("recruitment")

    @pytest.mark.asyncio
    async def test_classify_calls_to_dict(self, handler):
        """Verify to_dict is called on the classification result."""
        mock_cls = MagicMock()
        mock_cls.to_dict.return_value = {"risk_level": "minimal"}

        with patch("aragora.server.handlers.compliance.eu_ai_act._get_classifier") as mock_get:
            classifier = MagicMock()
            classifier.classify.return_value = mock_cls
            mock_get.return_value = classifier

            result = await handler._eu_ai_act_classify({"description": "simple analytics tool"})

        mock_cls.to_dict.assert_called_once()
        assert _body(result)["classification"]["risk_level"] == "minimal"

    @pytest.mark.asyncio
    async def test_classify_description_none_raises(self, handler):
        """None description raises AttributeError (None has no .strip())."""
        with pytest.raises(AttributeError):
            await handler._eu_ai_act_classify({"description": None})

    @pytest.mark.asyncio
    async def test_classify_extra_fields_ignored(self, handler, mock_classification):
        """Extra fields in the body are ignored."""
        with patch("aragora.server.handlers.compliance.eu_ai_act._get_classifier") as mock_get:
            classifier = MagicMock()
            classifier.classify.return_value = mock_classification
            mock_get.return_value = classifier

            result = await handler._eu_ai_act_classify(
                {"description": "recruitment system", "extra_field": "ignored"}
            )

        assert _status(result) == 200


# =========================================================================
# _eu_ai_act_audit tests
# =========================================================================


class TestAuditEndpoint:
    """Tests for POST /api/v2/compliance/eu-ai-act/audit."""

    @pytest.mark.asyncio
    async def test_audit_success(self, handler, sample_receipt, mock_report):
        """Happy path: valid receipt returns conformity report."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_report_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_report
            mock_get.return_value = gen

            result = await handler._eu_ai_act_audit({"receipt": sample_receipt})

        assert _status(result) == 200
        body = _body(result)
        assert "conformity_report" in body
        assert body["conformity_report"]["report_id"] == "EUAIA-abc12345"

    @pytest.mark.asyncio
    async def test_audit_missing_receipt(self, handler):
        """Missing 'receipt' field returns 400."""
        result = await handler._eu_ai_act_audit({})
        assert _status(result) == 400
        assert "receipt" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_audit_receipt_none(self, handler):
        """receipt=None returns 400."""
        result = await handler._eu_ai_act_audit({"receipt": None})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_audit_receipt_not_dict(self, handler):
        """receipt as a list returns 400."""
        result = await handler._eu_ai_act_audit({"receipt": ["not", "a", "dict"]})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_audit_receipt_string(self, handler):
        """receipt as string returns 400."""
        result = await handler._eu_ai_act_audit({"receipt": "not_a_dict"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_audit_receipt_integer(self, handler):
        """receipt as integer returns 400."""
        result = await handler._eu_ai_act_audit({"receipt": 42})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_audit_receipt_empty_dict(self, handler):
        """Empty dict receipt is falsy, so it returns 400."""
        result = await handler._eu_ai_act_audit({"receipt": {}})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_audit_passes_receipt_to_generator(self, handler, sample_receipt, mock_report):
        """Verify the receipt dict is passed directly to the generator."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_report_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_report
            mock_get.return_value = gen

            await handler._eu_ai_act_audit({"receipt": sample_receipt})

        gen.generate.assert_called_once_with(sample_receipt)

    @pytest.mark.asyncio
    async def test_audit_calls_to_dict(self, handler, sample_receipt):
        """Verify to_dict is called on the report."""
        mock_rpt = MagicMock()
        mock_rpt.to_dict.return_value = {"overall_status": "conformant"}

        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_report_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_rpt
            mock_get.return_value = gen

            result = await handler._eu_ai_act_audit({"receipt": sample_receipt})

        mock_rpt.to_dict.assert_called_once()
        assert _body(result)["conformity_report"]["overall_status"] == "conformant"


# =========================================================================
# _eu_ai_act_generate_bundle tests
# =========================================================================


class TestGenerateBundleEndpoint:
    """Tests for POST /api/v2/compliance/eu-ai-act/generate-bundle."""

    @pytest.mark.asyncio
    async def test_generate_bundle_success(self, handler, sample_receipt, mock_bundle):
        """Happy path: valid receipt returns artifact bundle."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            result = await handler._eu_ai_act_generate_bundle({"receipt": sample_receipt})

        assert _status(result) == 200
        body = _body(result)
        assert "bundle" in body
        assert body["bundle"]["bundle_id"] == "EUAIA-bndl0001"

    @pytest.mark.asyncio
    async def test_generate_bundle_missing_receipt(self, handler):
        """Missing 'receipt' field returns 400."""
        result = await handler._eu_ai_act_generate_bundle({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_generate_bundle_receipt_none(self, handler):
        """receipt=None returns 400."""
        result = await handler._eu_ai_act_generate_bundle({"receipt": None})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_generate_bundle_receipt_not_dict(self, handler):
        """receipt as string returns 400."""
        result = await handler._eu_ai_act_generate_bundle({"receipt": "bad"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_generate_bundle_receipt_list(self, handler):
        """receipt as list returns 400."""
        result = await handler._eu_ai_act_generate_bundle({"receipt": [1, 2, 3]})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_generate_bundle_receipt_bool(self, handler):
        """receipt as boolean returns 400 -- bool is not dict."""
        result = await handler._eu_ai_act_generate_bundle({"receipt": True})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_generate_bundle_with_provider_name(self, handler, sample_receipt, mock_bundle):
        """Optional provider_name is forwarded to the artifact generator."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            await handler._eu_ai_act_generate_bundle(
                {"receipt": sample_receipt, "provider_name": "Acme Corp"}
            )

        mock_get.assert_called_once_with(provider_name="Acme Corp")

    @pytest.mark.asyncio
    async def test_generate_bundle_with_all_optional_fields(
        self, handler, sample_receipt, mock_bundle
    ):
        """All five optional kwargs are forwarded."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            await handler._eu_ai_act_generate_bundle(
                {
                    "receipt": sample_receipt,
                    "provider_name": "AcmeCorp",
                    "provider_contact": "compliance@acme.com",
                    "eu_representative": "EU Rep Ltd",
                    "system_name": "AcmeAI",
                    "system_version": "1.0.0",
                }
            )

        mock_get.assert_called_once_with(
            provider_name="AcmeCorp",
            provider_contact="compliance@acme.com",
            eu_representative="EU Rep Ltd",
            system_name="AcmeAI",
            system_version="1.0.0",
        )

    @pytest.mark.asyncio
    async def test_generate_bundle_non_string_optional_ignored(
        self, handler, sample_receipt, mock_bundle
    ):
        """Non-string optional values (int, list, None) are not forwarded."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            await handler._eu_ai_act_generate_bundle(
                {
                    "receipt": sample_receipt,
                    "provider_name": 123,
                    "provider_contact": None,
                    "eu_representative": ["not", "a", "string"],
                    "system_name": True,
                    "system_version": {"nested": "dict"},
                }
            )

        # None/non-string values should be filtered out, so kwargs should be empty
        mock_get.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_generate_bundle_empty_string_optional_ignored(
        self, handler, sample_receipt, mock_bundle
    ):
        """Empty string optional values are not forwarded (falsy check)."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            await handler._eu_ai_act_generate_bundle(
                {
                    "receipt": sample_receipt,
                    "provider_name": "",
                    "system_version": "",
                }
            )

        # Empty strings are falsy, so they should be filtered out
        mock_get.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_generate_bundle_partial_optional_fields(
        self, handler, sample_receipt, mock_bundle
    ):
        """Only provided optional string fields are forwarded."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            await handler._eu_ai_act_generate_bundle(
                {
                    "receipt": sample_receipt,
                    "system_name": "MySystem",
                }
            )

        mock_get.assert_called_once_with(system_name="MySystem")

    @pytest.mark.asyncio
    async def test_generate_bundle_unknown_optional_fields_ignored(
        self, handler, sample_receipt, mock_bundle
    ):
        """Fields not in the allowed list are not forwarded."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            await handler._eu_ai_act_generate_bundle(
                {
                    "receipt": sample_receipt,
                    "unknown_field": "should_be_ignored",
                    "another_field": "also_ignored",
                }
            )

        mock_get.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_generate_bundle_passes_receipt_to_generator(
        self, handler, sample_receipt, mock_bundle
    ):
        """Verify the receipt dict is passed to the generator's generate method."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            await handler._eu_ai_act_generate_bundle({"receipt": sample_receipt})

        gen.generate.assert_called_once_with(sample_receipt)

    @pytest.mark.asyncio
    async def test_generate_bundle_calls_to_dict(self, handler, sample_receipt):
        """Verify to_dict is called on the bundle result."""
        mock_bndl = MagicMock()
        mock_bndl.to_dict.return_value = {"bundle_id": "test"}

        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bndl
            mock_get.return_value = gen

            result = await handler._eu_ai_act_generate_bundle({"receipt": sample_receipt})

        mock_bndl.to_dict.assert_called_once()
        assert _body(result)["bundle"]["bundle_id"] == "test"

    @pytest.mark.asyncio
    async def test_generate_bundle_empty_receipt_rejected(self, handler):
        """Empty dict receipt is falsy, so it returns 400."""
        result = await handler._eu_ai_act_generate_bundle({"receipt": {}})
        assert _status(result) == 400


# =========================================================================
# Lazy-loading / singleton tests
# =========================================================================


class TestLazyLoading:
    """Tests for the module-level lazy-loading functions."""

    def test_get_classifier_creates_singleton(self):
        """_get_classifier creates a RiskClassifier on first call."""
        with patch("aragora.compliance.eu_ai_act.RiskClassifier") as MockCls:
            instance = MagicMock()
            MockCls.return_value = instance

            result = _get_classifier()

        assert result is instance
        MockCls.assert_called_once()

    def test_get_classifier_reuses_singleton(self):
        """_get_classifier returns the same instance on subsequent calls."""
        import aragora.server.handlers.compliance.eu_ai_act as mod

        sentinel = MagicMock()
        mod._classifier = sentinel

        result = _get_classifier()
        assert result is sentinel

    def test_get_report_generator_creates_singleton(self):
        """_get_report_generator creates a ConformityReportGenerator on first call."""
        with patch("aragora.compliance.eu_ai_act.ConformityReportGenerator") as MockCls:
            instance = MagicMock()
            MockCls.return_value = instance

            result = _get_report_generator()

        assert result is instance
        MockCls.assert_called_once()

    def test_get_report_generator_reuses_singleton(self):
        """_get_report_generator returns the same instance on subsequent calls."""
        import aragora.server.handlers.compliance.eu_ai_act as mod

        sentinel = MagicMock()
        mod._report_generator = sentinel

        result = _get_report_generator()
        assert result is sentinel

    def test_get_artifact_generator_creates_singleton_no_kwargs(self):
        """_get_artifact_generator creates a ComplianceArtifactGenerator when no kwargs."""
        with patch("aragora.compliance.eu_ai_act.ComplianceArtifactGenerator") as MockCls:
            instance = MagicMock()
            MockCls.return_value = instance

            result = _get_artifact_generator()

        assert result is instance
        MockCls.assert_called_once_with()

    def test_get_artifact_generator_reuses_singleton_no_kwargs(self):
        """_get_artifact_generator returns cached instance when no kwargs."""
        import aragora.server.handlers.compliance.eu_ai_act as mod

        sentinel = MagicMock()
        mod._artifact_generator = sentinel

        result = _get_artifact_generator()
        assert result is sentinel

    def test_get_artifact_generator_with_kwargs_creates_new(self):
        """_get_artifact_generator with kwargs creates a fresh instance (not cached)."""
        with patch("aragora.compliance.eu_ai_act.ComplianceArtifactGenerator") as MockCls:
            instance = MagicMock()
            MockCls.return_value = instance

            result = _get_artifact_generator(provider_name="Test Corp")

        MockCls.assert_called_once_with(provider_name="Test Corp")
        assert result is instance

    def test_get_artifact_generator_with_kwargs_bypasses_cache(self):
        """Even if a cached instance exists, kwargs produce a new one."""
        import aragora.server.handlers.compliance.eu_ai_act as mod

        cached = MagicMock()
        mod._artifact_generator = cached

        with patch("aragora.compliance.eu_ai_act.ComplianceArtifactGenerator") as MockCls:
            fresh = MagicMock()
            MockCls.return_value = fresh

            result = _get_artifact_generator(system_name="NewSystem")

        assert result is fresh
        assert result is not cached


# =========================================================================
# Response format tests
# =========================================================================


class TestResponseFormat:
    """Verify the shape and status codes of responses."""

    @pytest.mark.asyncio
    async def test_classify_200_response_shape(self, handler, mock_classification):
        """Classify 200 response wraps classification in expected key."""
        with patch("aragora.server.handlers.compliance.eu_ai_act._get_classifier") as mock_get:
            classifier = MagicMock()
            classifier.classify.return_value = mock_classification
            mock_get.return_value = classifier

            result = await handler._eu_ai_act_classify({"description": "chatbot system"})

        assert _status(result) == 200
        body = _body(result)
        assert set(body.keys()) == {"classification"}

    @pytest.mark.asyncio
    async def test_audit_200_response_shape(self, handler, mock_report):
        """Audit 200 response wraps report in expected key."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_report_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_report
            mock_get.return_value = gen

            result = await handler._eu_ai_act_audit({"receipt": {"receipt_id": "test"}})

        assert _status(result) == 200
        body = _body(result)
        assert set(body.keys()) == {"conformity_report"}

    @pytest.mark.asyncio
    async def test_bundle_200_response_shape(self, handler, mock_bundle):
        """Bundle 200 response wraps bundle in expected key."""
        with patch(
            "aragora.server.handlers.compliance.eu_ai_act._get_artifact_generator"
        ) as mock_get:
            gen = MagicMock()
            gen.generate.return_value = mock_bundle
            mock_get.return_value = gen

            result = await handler._eu_ai_act_generate_bundle({"receipt": {"receipt_id": "test"}})

        assert _status(result) == 200
        body = _body(result)
        assert set(body.keys()) == {"bundle"}

    @pytest.mark.asyncio
    async def test_error_responses_have_error_key(self, handler):
        """All error responses include an 'error' key."""
        r1 = await handler._eu_ai_act_classify({})
        r2 = await handler._eu_ai_act_audit({})
        r3 = await handler._eu_ai_act_generate_bundle({})

        for r in (r1, r2, r3):
            assert _status(r) == 400
            assert "error" in _body(r)

    @pytest.mark.asyncio
    async def test_classify_error_message_mentions_description(self, handler):
        """Classify error specifically mentions 'description'."""
        result = await handler._eu_ai_act_classify({})
        assert "description" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_audit_error_message_mentions_receipt(self, handler):
        """Audit error specifically mentions 'receipt'."""
        result = await handler._eu_ai_act_audit({})
        assert "receipt" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_bundle_error_message_mentions_receipt(self, handler):
        """Bundle error specifically mentions 'receipt'."""
        result = await handler._eu_ai_act_generate_bundle({})
        assert "receipt" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_result_is_handler_result(self, handler, mock_classification):
        """All successful responses are HandlerResult instances."""
        with patch("aragora.server.handlers.compliance.eu_ai_act._get_classifier") as mock_get:
            classifier = MagicMock()
            classifier.classify.return_value = mock_classification
            mock_get.return_value = classifier

            result = await handler._eu_ai_act_classify({"description": "test system"})

        assert isinstance(result, HandlerResult)

    @pytest.mark.asyncio
    async def test_error_result_is_handler_result(self, handler):
        """Error responses are also HandlerResult instances."""
        result = await handler._eu_ai_act_classify({})
        assert isinstance(result, HandlerResult)
