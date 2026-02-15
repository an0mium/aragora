"""Tests for formal verification handler.

Covers:
- Route matching (can_handle)
- Request routing (handle_async dispatch)
- Verify claim endpoint (POST /api/v1/verify/claim)
- Batch verification (POST /api/v1/verify/batch)
- Status endpoint (GET /api/v1/verify/status)
- Translation endpoint (POST /api/v1/verify/translate)
- History listing with pagination/filtering
- History entry retrieval and proof tree
- RBAC permission checks
- Input validation (missing body, invalid JSON, empty claim)
- Proof tree building logic
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    return json.loads(result.body)


from aragora.server.handlers.verification.formal_verification import (
    FormalVerificationHandler,
    VerificationHistoryEntry,
    _build_proof_tree,
    _generate_verification_id,
    _verification_history,
)


# ============================================================================
# Route Matching
# ============================================================================


class TestCanHandle:
    """Test route matching logic."""

    def test_verify_claim_route(self):
        handler = FormalVerificationHandler()
        assert handler.can_handle("/api/v1/verify/claim") is True

    def test_verify_batch_route(self):
        handler = FormalVerificationHandler()
        assert handler.can_handle("/api/v1/verify/batch") is True

    def test_verify_status_route(self):
        handler = FormalVerificationHandler()
        assert handler.can_handle("/api/v1/verify/status") is True

    def test_verify_translate_route(self):
        handler = FormalVerificationHandler()
        assert handler.can_handle("/api/v1/verify/translate") is True

    def test_verify_history_route(self):
        handler = FormalVerificationHandler()
        assert handler.can_handle("/api/v1/verify/history") is True

    def test_verify_history_entry_route(self):
        handler = FormalVerificationHandler()
        assert handler.can_handle("/api/v1/verify/history/abc123") is True

    def test_verify_history_tree_route(self):
        handler = FormalVerificationHandler()
        assert handler.can_handle("/api/v1/verify/history/abc123/tree") is True

    def test_unknown_route(self):
        handler = FormalVerificationHandler()
        assert handler.can_handle("/api/v1/other") is False


# ============================================================================
# Proof Tree Building
# ============================================================================


class TestBuildProofTree:
    """Test proof tree construction from verification results."""

    def test_unverified_returns_none(self):
        result = {"is_verified": False}
        assert _build_proof_tree(result) is None

    def test_no_formal_statement_returns_none(self):
        result = {"is_verified": True, "formal_statement": ""}
        assert _build_proof_tree(result) is None

    def test_verified_basic_tree(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t1 : True := trivial",
            "claim": "Test claim",
            "language": "lean4",
            "status": "proof_found",
        }
        tree = _build_proof_tree(result)
        assert tree is not None
        assert len(tree) >= 3
        assert tree[0]["id"] == "root"
        assert tree[0]["type"] == "claim"
        assert tree[1]["id"] == "translation"
        assert tree[2]["id"] == "verification"

    def test_verified_with_proof_steps(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t1 : True := by simp",
            "claim": "Test",
            "language": "lean4",
            "status": "proof_found",
            "proof_steps": ["step A", "step B"],
        }
        tree = _build_proof_tree(result)
        assert len(tree) == 5  # root + translation + verification + 2 steps
        assert tree[3]["type"] == "proof_step"
        assert tree[3]["step_number"] == 1


# ============================================================================
# Verification ID Generation
# ============================================================================


class TestGenerateVerificationId:
    """Test verification ID generation."""

    def test_deterministic(self):
        id1 = _generate_verification_id("claim", 1234.0)
        id2 = _generate_verification_id("claim", 1234.0)
        assert id1 == id2

    def test_different_claims(self):
        id1 = _generate_verification_id("claim1", 1234.0)
        id2 = _generate_verification_id("claim2", 1234.0)
        assert id1 != id2

    def test_length(self):
        vid = _generate_verification_id("test", 1234.0)
        assert len(vid) == 16


# ============================================================================
# Verify Claim (async)
# ============================================================================


class TestVerifyClaim:
    """Test POST /api/v1/verify/claim."""

    @pytest.mark.asyncio
    async def test_missing_body(self):
        handler = FormalVerificationHandler()
        result = await handler._handle_verify_claim(MagicMock(), None)
        assert result.status_code == 400
        assert "body" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        handler = FormalVerificationHandler()
        result = await handler._handle_verify_claim(MagicMock(), b"not json")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_claim(self):
        handler = FormalVerificationHandler()
        body = json.dumps({"claim": ""}).encode()
        result = await handler._handle_verify_claim(MagicMock(), body)
        assert result.status_code == 400
        assert "claim" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_verification(self):
        handler = FormalVerificationHandler()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "status": "proof_found",
            "is_verified": True,
            "formal_statement": "theorem t1 : True := trivial",
            "language": "lean4",
        }

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_result
            )
            body = json.dumps({"claim": "1 + 1 = 2"}).encode()
            result = await handler._handle_verify_claim(MagicMock(), body)
            assert result.status_code == 200
            data = _body(result)
            assert data["status"] == "proof_found"
            assert "history_id" in data


# ============================================================================
# Batch Verification (async)
# ============================================================================


class TestVerifyBatch:
    """Test POST /api/v1/verify/batch."""

    @pytest.mark.asyncio
    async def test_missing_body(self):
        handler = FormalVerificationHandler()
        result = await handler._handle_verify_batch(MagicMock(), None)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_claims(self):
        handler = FormalVerificationHandler()
        body = json.dumps({"claims": []}).encode()
        result = await handler._handle_verify_batch(MagicMock(), body)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_too_many_claims(self):
        handler = FormalVerificationHandler()
        claims = [{"claim": f"claim_{i}"} for i in range(21)]
        body = json.dumps({"claims": claims}).encode()
        result = await handler._handle_verify_batch(MagicMock(), body)
        assert result.status_code == 400
        assert "20" in _body(result)["error"]


# ============================================================================
# Verify Status
# ============================================================================


class TestVerifyStatus:
    """Test GET /api/v1/verify/status."""

    def test_status_report(self):
        handler = FormalVerificationHandler()
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {
            "backends": [
                {"language": "z3_smt", "available": True},
            ],
            "any_available": True,
        }
        handler._manager = mock_manager

        with patch.dict("sys.modules", {
            "aragora.verification.deepseek_prover": MagicMock(
                DeepSeekProverTranslator=MagicMock(
                    return_value=MagicMock(is_available=False),
                ),
            ),
        }):
            result = handler._handle_verify_status(MagicMock())
            assert result.status_code == 200
            data = _body(result)
            assert "backends" in data
            assert "deepseek_prover_available" in data


# ============================================================================
# Translate
# ============================================================================


class TestTranslate:
    """Test POST /api/v1/verify/translate."""

    @pytest.mark.asyncio
    async def test_missing_body(self):
        handler = FormalVerificationHandler()
        result = await handler._handle_translate(MagicMock(), None)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_claim(self):
        handler = FormalVerificationHandler()
        body = json.dumps({"claim": ""}).encode()
        result = await handler._handle_translate(MagicMock(), body)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_target_language(self):
        handler = FormalVerificationHandler()
        body = json.dumps({"claim": "test", "target_language": "coq"}).encode()

        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            result = await handler._handle_translate(MagicMock(), body)
            assert result.status_code == 400
            assert "coq" in _body(result)["error"].lower()


# ============================================================================
# History
# ============================================================================


class TestHistory:
    """Test GET /api/v1/verify/history."""

    def test_empty_history(self):
        handler = FormalVerificationHandler()
        # Clear global history
        _verification_history.clear()

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_store_factory:
            mock_store_factory.get.return_value = None
            result = handler._handle_get_history({})
            assert result.status_code == 200
            data = _body(result)
            assert data["entries"] == []
            assert data["total"] == 0
            assert data["source"] == "in_memory"

    def test_history_with_entries(self):
        handler = FormalVerificationHandler()
        _verification_history.clear()

        # Add test entries
        entry = VerificationHistoryEntry(
            id="test_001",
            claim="1+1=2",
            claim_type="MATHEMATICAL",
            context="",
            result={"status": "proof_found"},
            timestamp=time.time(),
        )
        _verification_history["test_001"] = entry

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_store_factory:
            mock_store_factory.get.return_value = None
            result = handler._handle_get_history({})
            assert result.status_code == 200
            data = _body(result)
            assert data["total"] == 1
            assert data["entries"][0]["id"] == "test_001"

        # Cleanup
        _verification_history.clear()


# ============================================================================
# History Entry Retrieval
# ============================================================================


class TestHistoryEntry:
    """Test GET /api/v1/verify/history/{id}."""

    def test_entry_not_found(self):
        handler = FormalVerificationHandler()
        _verification_history.clear()

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_store_factory:
            mock_store_factory.get.return_value = None
            result = handler._handle_get_history_entry(
                "/api/v1/verify/history/nonexistent"
            )
            assert result.status_code == 404

    def test_entry_found(self):
        handler = FormalVerificationHandler()
        _verification_history.clear()

        entry = VerificationHistoryEntry(
            id="abc123",
            claim="test claim",
            claim_type=None,
            context="",
            result={"status": "proof_found", "is_verified": True},
            timestamp=time.time(),
        )
        _verification_history["abc123"] = entry

        result = handler._handle_get_history_entry("/api/v1/verify/history/abc123")
        assert result.status_code == 200
        data = _body(result)
        assert data["id"] == "abc123"
        assert data["claim"] == "test claim"

        _verification_history.clear()

    def test_tree_request_no_tree(self):
        handler = FormalVerificationHandler()
        _verification_history.clear()

        entry = VerificationHistoryEntry(
            id="abc123",
            claim="test",
            claim_type=None,
            context="",
            result={"status": "failed", "is_verified": False},
            timestamp=time.time(),
        )
        _verification_history["abc123"] = entry

        result = handler._handle_get_history_entry(
            "/api/v1/verify/history/abc123/tree"
        )
        assert result.status_code == 200
        data = _body(result)
        assert data["nodes"] == []

        _verification_history.clear()

    def test_tree_request_with_tree(self):
        handler = FormalVerificationHandler()
        _verification_history.clear()

        proof_tree = [{"id": "root", "type": "claim", "content": "test"}]
        entry = VerificationHistoryEntry(
            id="abc123",
            claim="test",
            claim_type=None,
            context="",
            result={"status": "proof_found"},
            timestamp=time.time(),
            proof_tree=proof_tree,
        )
        _verification_history["abc123"] = entry

        result = handler._handle_get_history_entry(
            "/api/v1/verify/history/abc123/tree"
        )
        assert result.status_code == 200
        data = _body(result)
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["id"] == "root"

        _verification_history.clear()


# ============================================================================
# Handle Async Routing
# ============================================================================


class TestHandleAsyncRouting:
    """Test handle_async dispatches to correct sub-handler."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self):
        handler = FormalVerificationHandler()
        with patch.object(handler, "_check_permission", return_value=None):
            result = await handler.handle_async(
                MagicMock(), "GET", "/api/v1/verify/unknown"
            )
            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        handler = FormalVerificationHandler()
        from aragora.server.handlers.base import error_response

        with patch.object(
            handler, "_check_permission",
            return_value=error_response("Permission denied", 403),
        ):
            result = await handler.handle_async(
                MagicMock(), "POST", "/api/v1/verify/claim"
            )
            assert result.status_code == 403


# ============================================================================
# VerificationHistoryEntry
# ============================================================================


class TestVerificationHistoryEntry:
    """Test VerificationHistoryEntry data model."""

    def test_to_dict(self):
        ts = time.time()
        entry = VerificationHistoryEntry(
            id="e1",
            claim="test claim",
            claim_type="MATHEMATICAL",
            context="ctx",
            result={"verified": True},
            timestamp=ts,
            proof_tree=[{"id": "root"}],
        )
        d = entry.to_dict()
        assert d["id"] == "e1"
        assert d["claim"] == "test claim"
        assert d["claim_type"] == "MATHEMATICAL"
        assert d["has_proof_tree"] is True
        assert "timestamp_iso" in d

    def test_to_dict_no_proof_tree(self):
        entry = VerificationHistoryEntry(
            id="e2",
            claim="test",
            claim_type=None,
            context="",
            result={},
            timestamp=time.time(),
        )
        d = entry.to_dict()
        assert d["has_proof_tree"] is False
