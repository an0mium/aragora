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
    return EvidenceEnrichmentHandler({})


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
            assert result.status_code == 404

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
            assert result.status_code == 200

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
            assert result.status_code == 200

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
            assert result.status_code == 404


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
            assert result.status_code == 400

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
            assert result.status_code == 400

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
            assert result.status_code == 400


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


# =============================================================================
# Test Circuit Breaker
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_get_evidence_circuit_breaker(self):
        """Test getting circuit breaker instance."""
        from aragora.server.handlers.features.evidence_enrichment import (
            get_evidence_circuit_breaker,
        )

        cb = get_evidence_circuit_breaker()
        assert cb is not None
        assert cb.name == "evidence_enrichment_handler"

    def test_get_evidence_circuit_breaker_status(self):
        """Test getting circuit breaker status."""
        from aragora.server.handlers.features.evidence_enrichment import (
            get_evidence_circuit_breaker_status,
        )

        status = get_evidence_circuit_breaker_status()
        assert isinstance(status, dict)
        # Check that status contains expected keys from to_dict()
        assert "config" in status
        assert "single_mode" in status
        assert status["config"]["failure_threshold"] == 5

    def test_circuit_breaker_is_singleton(self):
        """Test circuit breaker returns same instance."""
        from aragora.server.handlers.features.evidence_enrichment import (
            get_evidence_circuit_breaker,
        )

        cb1 = get_evidence_circuit_breaker()
        cb2 = get_evidence_circuit_breaker()
        assert cb1 is cb2


# =============================================================================
# Test Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting on handler methods."""

    def test_handle_has_rate_limit_decorator(self, handler):
        """Test that handle method has rate limit decorator."""
        # Check if the method has the _rate_limited attribute set by the decorator
        assert hasattr(handler.handle, "_rate_limited") or hasattr(
            EvidenceEnrichmentHandler.handle, "__wrapped__"
        )

    def test_handle_post_has_rate_limit_decorator(self, handler):
        """Test that handle_post method has rate limit decorator."""
        assert hasattr(handler.handle_post, "_rate_limited") or hasattr(
            EvidenceEnrichmentHandler.handle_post, "__wrapped__"
        )


# =============================================================================
# Test Handler Initialization
# =============================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_handler_with_none_context(self):
        """Test handler initialization with None context."""
        handler = EvidenceEnrichmentHandler(None)
        assert handler.ctx == {}

    def test_handler_with_server_context(self):
        """Test handler initialization with server_context."""
        ctx = {"document_store": MagicMock()}
        handler = EvidenceEnrichmentHandler(server_context=ctx)
        assert handler.ctx == ctx

    def test_handler_with_ctx_kwarg(self):
        """Test handler initialization with ctx kwarg."""
        ctx = {"key": "value"}
        handler = EvidenceEnrichmentHandler(ctx=ctx)
        assert handler.ctx == ctx

    def test_server_context_takes_precedence(self):
        """Test that server_context takes precedence over ctx."""
        ctx = {"key": "ctx_value"}
        server_ctx = {"key": "server_value"}
        handler = EvidenceEnrichmentHandler(ctx=ctx, server_context=server_ctx)
        assert handler.ctx == server_ctx


# =============================================================================
# Test Document Store Integration
# =============================================================================


class TestDocumentStore:
    """Tests for document store integration."""

    def test_get_document_store_when_present(self):
        """Test getting document store from context."""
        mock_store = MagicMock()
        handler = EvidenceEnrichmentHandler({"document_store": mock_store})
        assert handler._get_document_store() is mock_store

    def test_get_document_store_when_absent(self):
        """Test getting document store when not in context."""
        handler = EvidenceEnrichmentHandler({})
        assert handler._get_document_store() is None


# =============================================================================
# Test Config Parsing
# =============================================================================


class TestConfigParsing:
    """Tests for configuration parsing in enrichment."""

    @pytest.mark.asyncio
    async def test_run_enrichment_with_custom_config(self, handler):
        """Test enrichment with custom config values."""
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
            patch(
                "aragora.server.handlers.features.evidence_enrichment.EvidenceConfig",
                create=True,
            ) as MockConfig,
        ):
            mock_auditor.return_value = MagicMock(_sessions={"session1": mock_session})
            mock_collector_instance = MagicMock()
            mock_collector_instance.enrich_finding = AsyncMock(return_value=mock_enrichment)
            MockCollector.return_value = mock_collector_instance

            config_dict = {
                "max_sources_per_finding": 10,
                "enable_external_sources": False,
                "enable_cross_reference": False,
            }

            result = await handler._run_enrichment(
                finding_id="finding123",
                document_content="Test content",
                related_documents={},
                config_dict=config_dict,
                document_store=None,
            )

            assert result["finding_id"] == "finding123"
            # Verify config was created with custom values
            MockConfig.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_batch_enrichment_with_max_concurrent(self, handler):
        """Test batch enrichment respects max_concurrent config."""
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

            config_dict = {"max_concurrent": 3}

            result = await handler._run_batch_enrichment(
                finding_ids=["finding123"],
                config_dict=config_dict,
                document_store=None,
            )

            # Verify enrich_findings_batch was called with max_concurrent
            mock_collector_instance.enrich_findings_batch.assert_called_once()
            call_kwargs = mock_collector_instance.enrich_findings_batch.call_args[1]
            assert call_kwargs["max_concurrent"] == 3


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_can_handle_with_deep_nested_path(self, handler):
        """Test can_handle with deeply nested path."""
        assert handler.can_handle("/api/v1/findings/a/b/c/evidence") is False

    def test_can_handle_with_empty_finding_id(self, handler):
        """Test can_handle with empty finding id segment."""
        # Path /api/v1/findings//evidence should still match pattern
        assert handler.can_handle("/api/v1/findings//evidence") is True

    def test_handle_returns_none_for_unmatched_path(self, handler):
        """Test handle returns None for paths that don't match."""
        mock_handler = MagicMock()
        result = handler.handle("/api/v1/other/path", {}, mock_handler)
        assert result is None

    def test_handle_post_returns_none_for_unmatched_path(self, handler):
        """Test handle_post returns None for paths that don't match."""
        mock_handler = MagicMock()
        result = handler.handle_post("/api/v1/other/path", {}, mock_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_run_enrichment_gets_doc_from_store(self, handler):
        """Test enrichment retrieves document from store when content not provided."""
        mock_finding = MagicMock()
        mock_finding.id = "finding123"
        mock_finding.document_id = "doc123"
        mock_session = MagicMock()
        mock_session.findings = [mock_finding]

        mock_doc = MagicMock()
        mock_doc.content = "Document content from store"
        mock_store = MagicMock()
        mock_store.get.return_value = mock_doc

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
            mock_collector_instance.enrich_finding = AsyncMock(return_value=mock_enrichment)
            MockCollector.return_value = mock_collector_instance

            result = await handler._run_enrichment(
                finding_id="finding123",
                document_content=None,  # Not provided
                related_documents={},
                config_dict={},
                document_store=mock_store,
            )

            # Verify document store was called
            mock_store.get.assert_called_once_with("doc123")
            assert result["finding_id"] == "finding123"

    @pytest.mark.asyncio
    async def test_run_batch_enrichment_all_findings_missing(self, handler):
        """Test batch enrichment when all findings are missing."""
        with patch(
            "aragora.server.handlers.features.evidence_enrichment.get_document_auditor"
        ) as mock_auditor:
            mock_auditor.return_value = MagicMock(_sessions={})

            result = await handler._run_batch_enrichment(
                finding_ids=["missing1", "missing2", "missing3"],
                config_dict={},
                document_store=None,
            )

            assert result["processed"] == 0
            assert result["failed"] == 3
            assert len(result["errors"]) == 3

    def test_can_handle_with_special_characters_in_finding_id(self, handler):
        """Test can_handle with special characters in finding id."""
        # UUID-style finding IDs
        assert (
            handler.can_handle("/api/v1/findings/a1b2c3d4-e5f6-7890-abcd-ef1234567890/evidence")
            is True
        )
        # Numeric IDs
        assert handler.can_handle("/api/v1/findings/12345/evidence") is True

    @pytest.mark.asyncio
    async def test_run_enrichment_with_related_documents(self, handler):
        """Test enrichment with related documents provided."""
        mock_finding = MagicMock()
        mock_finding.id = "finding123"
        mock_finding.document_id = "doc123"
        mock_session = MagicMock()
        mock_session.findings = [mock_finding]

        mock_enrichment = MagicMock()
        mock_enrichment.to_dict.return_value = {"sources": [], "cross_references": 2}

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

            related_docs = {
                "doc2": "Related document content 1",
                "doc3": "Related document content 2",
            }

            result = await handler._run_enrichment(
                finding_id="finding123",
                document_content="Main document content",
                related_documents=related_docs,
                config_dict={},
                document_store=None,
            )

            # Verify enrich_finding was called with related_documents
            mock_collector_instance.enrich_finding.assert_called_once()
            call_kwargs = mock_collector_instance.enrich_finding.call_args[1]
            assert call_kwargs["related_documents"] == related_docs
