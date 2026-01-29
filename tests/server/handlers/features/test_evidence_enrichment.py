"""Tests for Evidence Enrichment Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.evidence_enrichment import (
    EvidenceEnrichmentHandler,
    _get_evidence_enrichment,
    _set_evidence_enrichment,
)


@pytest.fixture
def handler():
    """Create handler instance."""
    return EvidenceEnrichmentHandler(ctx={})


class TestEvidenceEnrichmentHandler:
    """Tests for EvidenceEnrichmentHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(EvidenceEnrichmentHandler, "ROUTES")
        routes = EvidenceEnrichmentHandler.ROUTES
        assert "/api/v1/findings/batch-evidence" in routes

    def test_can_handle_batch_evidence(self, handler):
        """Test can_handle for batch evidence route."""
        assert handler.can_handle("/api/v1/findings/batch-evidence") is True

    def test_can_handle_finding_evidence_routes(self, handler):
        """Test can_handle for finding evidence routes."""
        assert handler.can_handle("/api/v1/findings/finding123/evidence") is True
        assert handler.can_handle("/api/v1/findings/abc/evidence") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/evidence/") is False
        assert handler.can_handle("/api/v1/findings/batch") is False


class TestEvidenceEnrichmentUtilities:
    """Tests for evidence enrichment utilities."""

    def test_get_evidence_enrichment(self):
        """Test getting evidence enrichment from finding."""
        mock_finding = MagicMock()
        mock_finding._evidence_enrichment = {"test": "data"}

        result = _get_evidence_enrichment(mock_finding)
        assert result == {"test": "data"}

    def test_get_evidence_enrichment_none(self):
        """Test getting evidence enrichment when not set."""
        mock_finding = MagicMock(spec=[])  # No _evidence_enrichment attribute

        result = _get_evidence_enrichment(mock_finding)
        assert result is None

    def test_set_evidence_enrichment(self):
        """Test setting evidence enrichment on finding."""
        mock_finding = MagicMock()
        enrichment = {"test": "data"}

        _set_evidence_enrichment(mock_finding, enrichment)
        assert mock_finding._evidence_enrichment == enrichment


class TestGetFindingEvidence:
    """Tests for getting finding evidence."""

    def test_get_evidence_finding_not_found(self, handler):
        """Test getting evidence for non-existent finding."""
        mock_handler = MagicMock()

        with (
            patch(
                "aragora.server.handlers.features.evidence_enrichment.get_document_auditor"
            ) as mock_auditor,
            patch(
                "aragora.server.handlers.features.evidence_enrichment.require_user_auth",
                lambda f: f,
            ),
        ):
            mock_auditor.return_value = MagicMock(_sessions={})

            result = handler._get_finding_evidence("invalid-finding")
            assert result.status == 404

    def test_get_evidence_not_enriched(self, handler):
        """Test getting evidence when not yet enriched."""
        mock_handler = MagicMock()

        mock_finding = MagicMock()
        mock_finding.id = "finding123"
        mock_session = MagicMock()
        mock_session.findings = [mock_finding]

        with (
            patch(
                "aragora.server.handlers.features.evidence_enrichment.get_document_auditor"
            ) as mock_auditor,
            patch(
                "aragora.server.handlers.features.evidence_enrichment.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.evidence_enrichment._get_evidence_enrichment",
                return_value=None,
            ),
        ):
            mock_auditor.return_value = MagicMock(_sessions={"session1": mock_session})

            result = handler._get_finding_evidence("finding123")
            assert result.status == 200

            import json

            body = json.loads(result.body)
            assert body["finding_id"] == "finding123"
            assert body["evidence"] is None

    def test_get_evidence_enriched(self, handler):
        """Test getting evidence when enriched."""
        mock_finding = MagicMock()
        mock_finding.id = "finding123"
        mock_finding._evidence_enrichment = MagicMock()
        mock_finding._evidence_enrichment.to_dict.return_value = {"sources": []}
        mock_session = MagicMock()
        mock_session.findings = [mock_finding]

        with (
            patch(
                "aragora.server.handlers.features.evidence_enrichment.get_document_auditor"
            ) as mock_auditor,
            patch(
                "aragora.server.handlers.features.evidence_enrichment.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.evidence_enrichment._get_evidence_enrichment",
                return_value=mock_finding._evidence_enrichment,
            ),
        ):
            mock_auditor.return_value = MagicMock(_sessions={"session1": mock_session})

            result = handler._get_finding_evidence("finding123")
            assert result.status == 200

            import json

            body = json.loads(result.body)
            assert body["finding_id"] == "finding123"
            assert body["evidence"] == {"sources": []}


class TestEnrichFinding:
    """Tests for enriching a single finding."""

    def test_enrich_finding_not_found(self, handler):
        """Test enriching non-existent finding."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "read_json_body", return_value={}),
            patch(
                "aragora.server.handlers.features.evidence_enrichment.require_user_auth",
                lambda f: f,
            ),
            patch(
                "aragora.server.handlers.features.evidence_enrichment._run_async",
                side_effect=ValueError("Finding not found: invalid"),
            ),
        ):
            result = handler._enrich_finding(mock_handler, "invalid-finding")
            assert result.status == 404


class TestBatchEnrich:
    """Tests for batch evidence enrichment."""

    def test_batch_enrich_missing_body(self, handler):
        """Test batch enrich requires body."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "read_json_body", return_value=None),
            patch(
                "aragora.server.handlers.features.evidence_enrichment.require_user_auth",
                lambda f: f,
            ),
        ):
            result = handler._batch_enrich(mock_handler)
            assert result.status == 400

    def test_batch_enrich_missing_finding_ids(self, handler):
        """Test batch enrich requires finding_ids."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "read_json_body", return_value={"config": {}}),
            patch(
                "aragora.server.handlers.features.evidence_enrichment.require_user_auth",
                lambda f: f,
            ),
        ):
            result = handler._batch_enrich(mock_handler)
            assert result.status == 400

    def test_batch_enrich_empty_finding_ids(self, handler):
        """Test batch enrich rejects empty finding_ids."""
        mock_handler = MagicMock()

        with (
            patch.object(handler, "read_json_body", return_value={"finding_ids": []}),
            patch(
                "aragora.server.handlers.features.evidence_enrichment.require_user_auth",
                lambda f: f,
            ),
        ):
            result = handler._batch_enrich(mock_handler)
            assert result.status == 400


class TestAsyncEnrichment:
    """Tests for async enrichment methods."""

    @pytest.mark.asyncio
    async def test_run_enrichment_finding_not_found(self, handler):
        """Test async enrichment with non-existent finding."""
        with patch(
            "aragora.server.handlers.features.evidence_enrichment.get_document_auditor"
        ) as mock_auditor:
            mock_auditor.return_value = MagicMock(_sessions={})

            with pytest.raises(ValueError, match="Finding not found"):
                await handler._run_enrichment(
                    finding_id="invalid",
                    document_content=None,
                    related_documents={},
                    config_dict={},
                    document_store=None,
                )

    @pytest.mark.asyncio
    async def test_run_enrichment_success(self, handler):
        """Test successful async enrichment."""
        mock_finding = MagicMock()
        mock_finding.id = "finding123"
        mock_finding.document_id = "doc123"
        mock_session = MagicMock()
        mock_session.findings = [mock_finding]

        mock_enrichment = MagicMock()
        mock_enrichment.to_dict.return_value = {"sources": [], "adjusted_confidence": 0.85}

        with (
            patch(
                "aragora.server.handlers.features.evidence_enrichment.get_document_auditor"
            ) as mock_auditor,
            patch(
                "aragora.server.handlers.features.evidence_enrichment.FindingEvidenceCollector",
                create=True,
            ) as MockCollector,
        ):
            mock_auditor.return_value = MagicMock(_sessions={"session1": mock_session})
            mock_collector_instance = MagicMock()
            mock_collector_instance.enrich_finding = AsyncMock(return_value=mock_enrichment)
            MockCollector.return_value = mock_collector_instance

            result = await handler._run_enrichment(
                finding_id="finding123",
                document_content="Test content",
                related_documents={},
                config_dict={},
                document_store=None,
            )

            assert result["finding_id"] == "finding123"
            assert "enrichment" in result

    @pytest.mark.asyncio
    async def test_run_batch_enrichment(self, handler):
        """Test async batch enrichment."""
        mock_finding = MagicMock()
        mock_finding.id = "finding123"
        mock_finding.document_id = "doc123"
        mock_session = MagicMock()
        mock_session.findings = [mock_finding]

        mock_enrichment = MagicMock()
        mock_enrichment.to_dict.return_value = {"sources": []}

        with (
            patch(
                "aragora.server.handlers.features.evidence_enrichment.get_document_auditor"
            ) as mock_auditor,
            patch(
                "aragora.server.handlers.features.evidence_enrichment.FindingEvidenceCollector",
                create=True,
            ) as MockCollector,
        ):
            mock_auditor.return_value = MagicMock(_sessions={"session1": mock_session})
            mock_collector_instance = MagicMock()
            mock_collector_instance.enrich_findings_batch = AsyncMock(
                return_value={"finding123": mock_enrichment}
            )
            MockCollector.return_value = mock_collector_instance

            result = await handler._run_batch_enrichment(
                finding_ids=["finding123", "invalid"],
                config_dict={},
                document_store=None,
            )

            assert "enrichments" in result
            assert "errors" in result
            assert result["processed"] == 1
            assert result["failed"] == 1
