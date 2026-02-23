"""
Tests for the EU AI Act Compliance Artifact Generation Handler.

Covers:
- POST /api/v1/compliance/eu-ai-act/bundles  (generate bundle)
- GET  /api/v1/compliance/eu-ai-act/bundles/{bundle_id}  (retrieve bundle)
- Individual article generation (12, 13, 14)
- Missing debate_id handling (falls back to synthetic receipt)
- Permission checks
- Validation of articles parameter
- Bundle storage and retrieval round-trip
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.compliance_eu_ai_act import (
    EUAIActComplianceHandler,
    _synthetic_receipt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _parse_data(result) -> dict:
    """Extract the 'data' envelope from a HandlerResult."""
    body = _body(result)
    return body.get("data", body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to handle/handle_post."""

    def __init__(self, method: str = "GET", body: dict[str, Any] | None = None):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _tmp_db(tmp_path):
    """Provide a temporary SQLite DB path for bundle storage."""
    return str(tmp_path / "test_bundles.db")


@pytest.fixture
def handler(_tmp_db):
    """Create a handler instance with a fresh temp DB."""
    h = EUAIActComplianceHandler(ctx={})
    # Patch the global bundle generator to use our temp DB
    from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

    gen = EUAIActBundleGenerator(db_path=_tmp_db)
    with patch(
        "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
        return_value=gen,
    ):
        h._test_generator = gen
        yield h


@pytest.fixture
def bundle_generator(_tmp_db):
    """Direct access to a temp EUAIActBundleGenerator."""
    from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

    return EUAIActBundleGenerator(db_path=_tmp_db)


# ---------------------------------------------------------------------------
# Test: can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_matches_bundles_path(self):
        h = EUAIActComplianceHandler(ctx={})
        assert h.can_handle("/api/v1/compliance/eu-ai-act/bundles") is True

    def test_matches_bundle_id_path(self):
        h = EUAIActComplianceHandler(ctx={})
        assert h.can_handle("/api/v1/compliance/eu-ai-act/bundles/EUAIA-abc123") is True

    def test_rejects_unrelated_path(self):
        h = EUAIActComplianceHandler(ctx={})
        assert h.can_handle("/api/v1/compliance/reports") is False


# ---------------------------------------------------------------------------
# Test: POST /api/v1/compliance/eu-ai-act/bundles
# ---------------------------------------------------------------------------


class TestPostBundle:
    def test_generate_bundle_no_debate_id(self, handler, _tmp_db):
        """POST without debate_id uses synthetic receipt."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler("POST", body={})
            result = handler.handle_post(
                "/api/v1/compliance/eu-ai-act/bundles", {}, mock_http
            )

        assert _status(result) == 201
        data = _parse_data(result)
        assert data["bundle_id"].startswith("EUAIA-")
        assert "articles" in data
        assert "generated_at" in data

    def test_generate_bundle_with_debate_id(self, handler, _tmp_db):
        """POST with debate_id loads from storage or falls back to synthetic."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler(
                "POST", body={"debate_id": "debate-123"}
            )
            result = handler.handle_post(
                "/api/v1/compliance/eu-ai-act/bundles", {}, mock_http
            )

        assert _status(result) == 201
        data = _parse_data(result)
        assert data["bundle_id"].startswith("EUAIA-")

    def test_generate_bundle_with_scope(self, handler, _tmp_db):
        """POST with scope includes scope in the bundle."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler(
                "POST", body={"scope": "Q1 2026 hiring decisions"}
            )
            result = handler.handle_post(
                "/api/v1/compliance/eu-ai-act/bundles", {}, mock_http
            )

        assert _status(result) == 201
        data = _parse_data(result)
        assert data["articles"].get("scope") == "Q1 2026 hiring decisions"

    def test_generate_bundle_with_specific_articles(self, handler, _tmp_db):
        """POST with articles=[12] generates only Article 12."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler("POST", body={"articles": [12]})
            result = handler.handle_post(
                "/api/v1/compliance/eu-ai-act/bundles", {}, mock_http
            )

        assert _status(result) == 201
        data = _parse_data(result)
        articles = data["articles"]
        assert "article_12_record_keeping" in articles
        assert "article_13_transparency" not in articles
        assert "article_14_human_oversight" not in articles

    def test_generate_bundle_with_articles_13_14(self, handler, _tmp_db):
        """POST with articles=[13, 14] generates Articles 13 and 14."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler("POST", body={"articles": [13, 14]})
            result = handler.handle_post(
                "/api/v1/compliance/eu-ai-act/bundles", {}, mock_http
            )

        assert _status(result) == 201
        data = _parse_data(result)
        articles = data["articles"]
        assert "article_13_transparency" in articles
        assert "article_14_human_oversight" in articles
        assert "article_12_record_keeping" not in articles

    def test_invalid_articles_param_not_list(self, handler, _tmp_db):
        """POST with articles as string returns 400."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler("POST", body={"articles": "12"})
            result = handler.handle_post(
                "/api/v1/compliance/eu-ai-act/bundles", {}, mock_http
            )

        assert _status(result) == 400
        body = _body(result)
        assert "list" in body.get("error", "").lower()

    def test_invalid_article_numbers(self, handler, _tmp_db):
        """POST with invalid article numbers returns 400."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler("POST", body={"articles": [12, 99]})
            result = handler.handle_post(
                "/api/v1/compliance/eu-ai-act/bundles", {}, mock_http
            )

        assert _status(result) == 400
        body = _body(result)
        assert "99" in body.get("error", "")

    def test_wrong_path_returns_none(self, handler, _tmp_db):
        """POST to wrong path returns None."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler("POST", body={})
            result = handler.handle_post(
                "/api/v1/compliance/reports/generate", {}, mock_http
            )

        assert result is None


# ---------------------------------------------------------------------------
# Test: GET /api/v1/compliance/eu-ai-act/bundles/{bundle_id}
# ---------------------------------------------------------------------------


class TestGetBundle:
    def test_retrieve_existing_bundle(self, _tmp_db):
        """GET for existing bundle returns data envelope."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)

        # Generate a bundle first
        receipt = _synthetic_receipt()
        stored = gen.generate(receipt)
        bundle_id = stored["bundle_id"]

        h = EUAIActComplianceHandler(ctx={})
        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler("GET")
            result = h.handle(
                f"/api/v1/compliance/eu-ai-act/bundles/{bundle_id}", {}, mock_http
            )

        assert _status(result) == 200
        data = _parse_data(result)
        assert data["bundle_id"] == bundle_id
        assert data["status"] == "complete"
        assert "articles" in data

    def test_retrieve_nonexistent_bundle(self, _tmp_db):
        """GET for missing bundle returns 404."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        h = EUAIActComplianceHandler(ctx={})

        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            mock_http = _MockHTTPHandler("GET")
            result = h.handle(
                "/api/v1/compliance/eu-ai-act/bundles/EUAIA-nonexistent",
                {},
                mock_http,
            )

        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Test: Bundle round-trip (POST then GET)
# ---------------------------------------------------------------------------


class TestBundleRoundTrip:
    def test_post_then_get(self, _tmp_db):
        """A generated bundle can be retrieved by its ID."""
        from aragora.compliance.eu_ai_act import EUAIActBundleGenerator

        gen = EUAIActBundleGenerator(db_path=_tmp_db)
        h = EUAIActComplianceHandler(ctx={})

        with patch(
            "aragora.server.handlers.compliance_eu_ai_act._get_bundle_generator",
            return_value=gen,
        ):
            # POST
            mock_post = _MockHTTPHandler("POST", body={"debate_id": "d-001"})
            post_result = h.handle_post(
                "/api/v1/compliance/eu-ai-act/bundles", {}, mock_post
            )
            assert _status(post_result) == 201
            bundle_id = _parse_data(post_result)["bundle_id"]

            # GET
            mock_get = _MockHTTPHandler("GET")
            get_result = h.handle(
                f"/api/v1/compliance/eu-ai-act/bundles/{bundle_id}",
                {},
                mock_get,
            )
            assert _status(get_result) == 200
            retrieved = _parse_data(get_result)
            assert retrieved["bundle_id"] == bundle_id
            assert retrieved["status"] == "complete"


# ---------------------------------------------------------------------------
# Test: EUAIActBundleGenerator directly
# ---------------------------------------------------------------------------


class TestEUAIActBundleGenerator:
    def test_generate_all_articles(self, bundle_generator):
        """Generate a bundle with all three articles."""
        receipt = _synthetic_receipt()
        result = bundle_generator.generate(receipt)
        assert result["bundle_id"].startswith("EUAIA-")
        assert "article_12_record_keeping" in result["articles"]
        assert "article_13_transparency" in result["articles"]
        assert "article_14_human_oversight" in result["articles"]

    def test_generate_single_article(self, bundle_generator):
        """Generate a bundle with only Article 13."""
        receipt = _synthetic_receipt()
        result = bundle_generator.generate(receipt, articles=[13])
        assert "article_13_transparency" in result["articles"]
        assert "article_12_record_keeping" not in result["articles"]

    def test_conformity_report_included(self, bundle_generator):
        """Bundle always includes conformity report and risk classification."""
        receipt = _synthetic_receipt()
        result = bundle_generator.generate(receipt, articles=[12])
        assert "conformity_report" in result["articles"]
        assert "risk_classification" in result["articles"]

    def test_integrity_hash_present(self, bundle_generator):
        """Bundle includes an integrity hash."""
        receipt = _synthetic_receipt()
        result = bundle_generator.generate(receipt)
        assert "integrity_hash" in result["articles"]
        assert len(result["articles"]["integrity_hash"]) == 64

    def test_store_and_retrieve(self, bundle_generator):
        """Store a bundle and retrieve it."""
        receipt = _synthetic_receipt()
        result = bundle_generator.generate(receipt)
        retrieved = bundle_generator.get(result["bundle_id"])
        assert retrieved is not None
        assert retrieved["bundle_id"] == result["bundle_id"]

    def test_get_nonexistent_returns_none(self, bundle_generator):
        """Retrieving a missing bundle returns None."""
        assert bundle_generator.get("EUAIA-does-not-exist") is None


# ---------------------------------------------------------------------------
# Test: Permission checks (using no_auto_auth marker)
# ---------------------------------------------------------------------------


class TestPermissions:
    @pytest.mark.no_auto_auth
    def test_generate_requires_compliance_generate(self):
        """POST without permission should be rejected by the decorator."""
        # When no_auto_auth is set, the RBAC decorator is active.
        # The decorator will look for an auth context and reject if missing.
        h = EUAIActComplianceHandler(ctx={})
        mock_http = _MockHTTPHandler("POST", body={})
        result = h.handle_post(
            "/api/v1/compliance/eu-ai-act/bundles", {}, mock_http
        )
        # The decorator may return 401/403 or None depending on RBAC setup.
        # We just verify it does not return 201 (success).
        if result is not None:
            assert _status(result) != 201


# ---------------------------------------------------------------------------
# Test: Synthetic receipt
# ---------------------------------------------------------------------------


class TestSyntheticReceipt:
    def test_has_required_fields(self):
        """Synthetic receipt has all fields needed for bundle generation."""
        receipt = _synthetic_receipt()
        assert "receipt_id" in receipt
        assert "input_summary" in receipt
        assert "confidence" in receipt
        assert "robustness_score" in receipt
        assert "provenance_chain" in receipt
        assert "config_used" in receipt
        assert receipt["config_used"]["require_approval"] is True

    def test_debate_id_overrides_receipt_id(self):
        """Passing debate_id sets the receipt_id."""
        receipt = _synthetic_receipt("my-debate-42")
        assert receipt["receipt_id"] == "my-debate-42"


__all__ = [
    "TestCanHandle",
    "TestPostBundle",
    "TestGetBundle",
    "TestBundleRoundTrip",
    "TestEUAIActBundleGenerator",
    "TestPermissions",
    "TestSyntheticReceipt",
]
