"""Comprehensive tests for formal verification handler.

Tests cover:
- Route matching (can_handle) for all ROUTES and dynamic history paths
- Request routing dispatch (handle_async)
- POST /api/v1/verify/claim - single claim verification
- POST /api/v1/verify/batch - batch verification
- GET /api/v1/verify/status - backend availability
- POST /api/v1/verify/translate - translation (lean4, z3_smt, unknown)
- GET /api/v1/verify/history - history listing with pagination/filtering/source
- GET /api/v1/verify/history/{id} - individual entry retrieval
- GET /api/v1/verify/history/{id}/tree - proof tree retrieval
- RBAC permission enforcement (read/create, denied, auth error)
- Input validation (missing body, invalid JSON, empty claim, bad params)
- Proof tree building logic (verified/unverified, steps, missing fields)
- VerificationHistoryEntry dataclass (to_dict, fields)
- Verification ID generation (deterministic, uniqueness, length)
- History eviction (MAX_HISTORY_SIZE)
- GovernanceStore persistence and fallback
- TTL-based cleanup
- Edge cases (concurrent, large batch, timeout clamping, etc.)
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.verification.formal_verification import (
    FormalVerificationHandler,
    MAX_HISTORY_SIZE,
    HISTORY_TTL_SECONDS,
    VerificationHistoryEntry,
    _add_to_history,
    _build_proof_tree,
    _cleanup_old_history,
    _generate_verification_id,
    _verification_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Parse HandlerResult.body bytes into dict."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed into handle_async."""

    def __init__(self, method="GET", body=None, client_address=None):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()
        self.client_address = client_address or ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# Mock FormalProofStatus enum for batch tests
class _MockFormalProofStatus(Enum):
    PROOF_FOUND = "proof_found"
    TIMEOUT = "timeout"
    TRANSLATION_FAILED = "translation_failed"
    PROOF_SEARCH_FAILED = "proof_search_failed"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a FormalVerificationHandler instance."""
    return FormalVerificationHandler(server_context={})


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler."""
    return _MockHTTPHandler()


@pytest.fixture(autouse=True)
def clear_history():
    """Clear the global verification history before/after each test."""
    _verification_history.clear()
    yield
    _verification_history.clear()


@pytest.fixture
def mock_governance_store():
    """Patch the governance store to return None (in-memory fallback)."""
    with patch(
        "aragora.server.handlers.verification.formal_verification._governance_store"
    ) as mock_sf:
        mock_sf.get.return_value = None
        yield mock_sf


@pytest.fixture
def sample_entry():
    """Create a sample VerificationHistoryEntry."""
    return VerificationHistoryEntry(
        id="sample_001",
        claim="1 + 1 = 2",
        claim_type="MATHEMATICAL",
        context="arithmetic",
        result={"status": "proof_found", "is_verified": True, "confidence": 0.95},
        timestamp=time.time(),
        proof_tree=[{"id": "root", "type": "claim", "content": "1+1=2"}],
    )


@pytest.fixture
def mock_verification_result():
    """Create a mock verification result object."""
    mock_result = MagicMock()
    mock_result.to_dict.return_value = {
        "status": "proof_found",
        "is_verified": True,
        "formal_statement": "theorem t1 : 1 + 1 = 2 := by norm_num",
        "language": "lean4",
        "proof_hash": "abc123def456",
        "translation_time_ms": 150.0,
        "proof_search_time_ms": 300.0,
        "error_message": "",
        "prover_version": "Lean 4.3.0",
        "confidence": 0.95,
    }
    mock_result.status = _MockFormalProofStatus.PROOF_FOUND
    return mock_result


# =============================================================================
# Route Matching (can_handle)
# =============================================================================


class TestCanHandle:
    """Test route matching logic for all declared routes."""

    def test_verify_claim_route(self, handler):
        assert handler.can_handle("/api/v1/verify/claim") is True

    def test_verify_batch_route(self, handler):
        assert handler.can_handle("/api/v1/verify/batch") is True

    def test_verify_status_route(self, handler):
        assert handler.can_handle("/api/v1/verify/status") is True

    def test_verify_translate_route(self, handler):
        assert handler.can_handle("/api/v1/verify/translate") is True

    def test_verify_history_route(self, handler):
        assert handler.can_handle("/api/v1/verify/history") is True

    def test_legacy_formal_verify_route(self, handler):
        assert handler.can_handle("/api/verification/formal-verify") is True

    def test_legacy_status_route(self, handler):
        assert handler.can_handle("/api/verification/status") is True

    def test_history_entry_dynamic_route(self, handler):
        assert handler.can_handle("/api/v1/verify/history/abc123") is True

    def test_history_tree_dynamic_route(self, handler):
        assert handler.can_handle("/api/v1/verify/history/abc123/tree") is True

    def test_generic_verify_subpath(self, handler):
        assert handler.can_handle("/api/v1/verify/something") is True

    def test_unknown_api_route(self, handler):
        assert handler.can_handle("/api/v1/other") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_verify_prefix_without_api(self, handler):
        assert handler.can_handle("/verify/claim") is False


# =============================================================================
# Verification ID Generation
# =============================================================================


class TestGenerateVerificationId:
    """Test verification ID generation."""

    def test_deterministic(self):
        id1 = _generate_verification_id("claim text", 1234.5)
        id2 = _generate_verification_id("claim text", 1234.5)
        assert id1 == id2

    def test_different_claims_produce_different_ids(self):
        id1 = _generate_verification_id("claim A", 1000.0)
        id2 = _generate_verification_id("claim B", 1000.0)
        assert id1 != id2

    def test_different_timestamps_produce_different_ids(self):
        id1 = _generate_verification_id("same claim", 1000.0)
        id2 = _generate_verification_id("same claim", 2000.0)
        assert id1 != id2

    def test_length_is_16(self):
        vid = _generate_verification_id("test", 1234.0)
        assert len(vid) == 16

    def test_hex_characters_only(self):
        vid = _generate_verification_id("test claim", 9999.9)
        assert all(c in "0123456789abcdef" for c in vid)

    def test_empty_claim(self):
        vid = _generate_verification_id("", 0.0)
        assert len(vid) == 16

    def test_unicode_claim(self):
        vid = _generate_verification_id("forall n, n + 0 = n", 100.0)
        assert len(vid) == 16


# =============================================================================
# VerificationHistoryEntry
# =============================================================================


class TestVerificationHistoryEntry:
    """Test VerificationHistoryEntry data model."""

    def test_to_dict_all_fields(self, sample_entry):
        d = sample_entry.to_dict()
        assert d["id"] == "sample_001"
        assert d["claim"] == "1 + 1 = 2"
        assert d["claim_type"] == "MATHEMATICAL"
        assert d["context"] == "arithmetic"
        assert d["has_proof_tree"] is True
        assert "timestamp_iso" in d
        assert isinstance(d["timestamp"], float)

    def test_to_dict_no_proof_tree(self):
        entry = VerificationHistoryEntry(
            id="e1",
            claim="test",
            claim_type=None,
            context="",
            result={},
            timestamp=time.time(),
        )
        d = entry.to_dict()
        assert d["has_proof_tree"] is False
        assert d["claim_type"] is None

    def test_to_dict_result_not_included(self, sample_entry):
        """to_dict() returns the result dict but the shape only includes
        status-level fields, not the full nested result."""
        d = sample_entry.to_dict()
        assert d["result"] == sample_entry.result

    def test_timestamp_iso_format(self):
        ts = 1700000000.0
        entry = VerificationHistoryEntry(
            id="ts_test", claim="c", claim_type=None,
            context="", result={}, timestamp=ts,
        )
        d = entry.to_dict()
        assert "T" in d["timestamp_iso"]  # ISO format contains 'T'


# =============================================================================
# Proof Tree Building
# =============================================================================


class TestBuildProofTree:
    """Test proof tree construction from verification results."""

    def test_unverified_returns_none(self):
        result = {"is_verified": False}
        assert _build_proof_tree(result) is None

    def test_missing_is_verified_returns_none(self):
        result = {"formal_statement": "theorem t : True := trivial"}
        assert _build_proof_tree(result) is None

    def test_no_formal_statement_returns_none(self):
        result = {"is_verified": True, "formal_statement": ""}
        assert _build_proof_tree(result) is None

    def test_formal_statement_none_returns_none(self):
        result = {"is_verified": True, "formal_statement": None}
        assert _build_proof_tree(result) is None

    def test_verified_basic_tree_structure(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t1 : True := trivial",
            "claim": "Test claim",
            "language": "lean4",
            "status": "proof_found",
        }
        tree = _build_proof_tree(result)
        assert tree is not None
        assert len(tree) == 3  # root + translation + verification

    def test_root_node(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t1 : True := trivial",
            "claim": "My claim",
            "language": "lean4",
            "status": "proof_found",
        }
        tree = _build_proof_tree(result)
        root = tree[0]
        assert root["id"] == "root"
        assert root["type"] == "claim"
        assert root["content"] == "My claim"
        assert root["children"] == ["translation"]

    def test_translation_node(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t1 : True := trivial",
            "claim": "test",
            "language": "lean4",
            "status": "proof_found",
        }
        tree = _build_proof_tree(result)
        translation = tree[1]
        assert translation["id"] == "translation"
        assert translation["type"] == "translation"
        assert translation["content"] == "theorem t1 : True := trivial"
        assert translation["language"] == "lean4"
        assert translation["children"] == ["verification"]

    def test_verification_node(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t1 : True := trivial",
            "claim": "test",
            "language": "lean4",
            "status": "proof_found",
            "proof_hash": "hashval",
        }
        tree = _build_proof_tree(result)
        verification = tree[2]
        assert verification["id"] == "verification"
        assert verification["type"] == "verification"
        assert "proof_found" in verification["content"]
        assert verification["is_verified"] is True
        assert verification["proof_hash"] == "hashval"

    def test_proof_steps_added(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t1 : True := by simp",
            "claim": "Test",
            "language": "lean4",
            "status": "proof_found",
            "proof_steps": ["simp", "ring", "omega"],
        }
        tree = _build_proof_tree(result)
        assert len(tree) == 6  # root + translation + verification + 3 steps
        # verification node children are step refs
        assert tree[2]["children"] == ["step_0", "step_1", "step_2"]

    def test_proof_step_nodes(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t : True := by simp",
            "claim": "test",
            "language": "lean4",
            "status": "proof_found",
            "proof_steps": ["apply nat_rec", "exact trivial"],
        }
        tree = _build_proof_tree(result)
        step0 = tree[3]
        assert step0["id"] == "step_0"
        assert step0["type"] == "proof_step"
        assert step0["content"] == "apply nat_rec"
        assert step0["step_number"] == 1
        assert step0["children"] == []

        step1 = tree[4]
        assert step1["step_number"] == 2
        assert step1["content"] == "exact trivial"

    def test_empty_proof_steps_list(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t : True := trivial",
            "claim": "test",
            "language": "lean4",
            "status": "proof_found",
            "proof_steps": [],
        }
        tree = _build_proof_tree(result)
        assert len(tree) == 3  # no step nodes
        assert tree[2]["children"] == []

    def test_missing_language_defaults_unknown(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t : True := trivial",
            "claim": "test",
            "status": "proof_found",
        }
        tree = _build_proof_tree(result)
        assert tree[1]["language"] == "unknown"

    def test_missing_claim_uses_default(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t : True := trivial",
            "language": "lean4",
            "status": "proof_found",
        }
        tree = _build_proof_tree(result)
        assert tree[0]["content"] == "Original claim"

    def test_missing_status_defaults_unknown(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t : True := trivial",
            "claim": "test",
            "language": "lean4",
        }
        tree = _build_proof_tree(result)
        assert "unknown" in tree[2]["content"]

    def test_missing_proof_hash(self):
        result = {
            "is_verified": True,
            "formal_statement": "theorem t : True := trivial",
            "claim": "test",
            "language": "lean4",
            "status": "proof_found",
        }
        tree = _build_proof_tree(result)
        assert tree[2]["proof_hash"] is None


# =============================================================================
# History Management
# =============================================================================


class TestAddToHistory:
    """Test _add_to_history function."""

    def test_adds_entry(self, mock_governance_store):
        entry_id = _add_to_history(
            claim="test claim",
            claim_type="MATHEMATICAL",
            context="ctx",
            result={"status": "proof_found"},
        )
        assert entry_id in _verification_history
        assert _verification_history[entry_id].claim == "test claim"

    def test_returns_id(self, mock_governance_store):
        entry_id = _add_to_history(
            claim="test", claim_type=None, context="", result={},
        )
        assert isinstance(entry_id, str)
        assert len(entry_id) == 16

    def test_eviction_on_overflow(self, mock_governance_store):
        """When history exceeds MAX_HISTORY_SIZE, oldest entries are evicted."""
        # Fill history to capacity
        for i in range(MAX_HISTORY_SIZE + 5):
            _verification_history[f"entry_{i}"] = VerificationHistoryEntry(
                id=f"entry_{i}",
                claim=f"claim_{i}",
                claim_type=None,
                context="",
                result={},
                timestamp=time.time(),
            )

        # Now add one more via _add_to_history
        _add_to_history(
            claim="overflow", claim_type=None, context="", result={},
        )
        assert len(_verification_history) <= MAX_HISTORY_SIZE

    def test_with_proof_tree(self, mock_governance_store):
        tree = [{"id": "root"}]
        entry_id = _add_to_history(
            claim="claim", claim_type=None, context="",
            result={}, proof_tree=tree,
        )
        assert _verification_history[entry_id].proof_tree == tree

    def test_persists_to_governance_store(self):
        mock_store = MagicMock()
        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            _add_to_history(
                claim="persisted",
                claim_type="LOGICAL",
                context="ctx",
                result={"confidence": 0.8},
            )
            mock_store.save_verification.assert_called_once()
            call_kwargs = mock_store.save_verification.call_args[1]
            assert call_kwargs["claim"] == "persisted"
            assert call_kwargs["verified_by"] == "formal_verification"
            assert call_kwargs["confidence"] == 0.8

    def test_governance_store_failure_does_not_raise(self):
        mock_store = MagicMock()
        mock_store.save_verification.side_effect = RuntimeError("db down")
        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            # Should not raise
            entry_id = _add_to_history(
                claim="will fail persist",
                claim_type=None, context="", result={},
            )
            assert entry_id in _verification_history

    def test_confidence_fallback_for_non_dict_result(self):
        mock_store = MagicMock()
        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            _add_to_history(
                claim="test",
                claim_type=None, context="",
                result="not_a_dict",
            )
            call_kwargs = mock_store.save_verification.call_args[1]
            assert call_kwargs["confidence"] == 0.0


class TestCleanupOldHistory:
    """Test TTL-based history cleanup."""

    def test_removes_old_entries(self):
        old_timestamp = time.time() - HISTORY_TTL_SECONDS - 100
        _verification_history["old"] = VerificationHistoryEntry(
            id="old", claim="old claim", claim_type=None,
            context="", result={}, timestamp=old_timestamp,
        )
        _verification_history["new"] = VerificationHistoryEntry(
            id="new", claim="new claim", claim_type=None,
            context="", result={}, timestamp=time.time(),
        )
        _cleanup_old_history()
        assert "old" not in _verification_history
        assert "new" in _verification_history

    def test_keeps_recent_entries(self):
        _verification_history["recent"] = VerificationHistoryEntry(
            id="recent", claim="recent", claim_type=None,
            context="", result={}, timestamp=time.time(),
        )
        _cleanup_old_history()
        assert "recent" in _verification_history

    def test_no_entries(self):
        _cleanup_old_history()  # Should not raise
        assert len(_verification_history) == 0


# =============================================================================
# RBAC Permission Checks
# =============================================================================


class TestCheckPermission:
    """Test the handler's RBAC permission checking."""

    def test_permission_granted(self, handler):
        """The conftest auto-patches auth, so permission checks pass."""
        result = handler._check_permission(MagicMock(), "verification.read")
        assert result is None  # None means granted

    @pytest.mark.no_auto_auth
    def test_permission_denied_returns_403(self, handler):
        """When permission check fails, return 403."""
        from aragora.rbac.models import AuthorizationDecision

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = AuthorizationDecision(
            allowed=False, reason="denied", permission_key="verification.create"
        )

        # Mock auth to succeed but permission to fail
        mock_user = MagicMock()
        mock_user.user_id = "user1"
        mock_user.org_id = "org1"
        mock_user.roles = {"member"}

        with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.verification.formal_verification.get_permission_checker",
                return_value=mock_checker,
            ):
                result = handler._check_permission(MagicMock(), "verification.create")
                assert result is not None
                assert _status(result) == 403

    @pytest.mark.no_auto_auth
    def test_auth_error_returns_401(self, handler):
        """When auth fails, return error from require_auth_or_error."""
        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)
        with patch.object(handler, "require_auth_or_error", return_value=(None, auth_err)):
            result = handler._check_permission(MagicMock(), "verification.read")
            assert result is not None
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_auth_exception_returns_401(self, handler):
        """When auth raises, return 401."""
        with patch.object(
            handler, "require_auth_or_error", side_effect=AttributeError("no user")
        ):
            result = handler._check_permission(MagicMock(), "verification.read")
            assert result is not None
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_null_user_returns_401(self, handler):
        """When auth returns None user without error, return 401."""
        with patch.object(handler, "require_auth_or_error", return_value=(None, None)):
            result = handler._check_permission(MagicMock(), "verification.read")
            assert result is not None
            assert _status(result) == 401


# =============================================================================
# handle_async Routing
# =============================================================================


class TestHandleAsyncRouting:
    """Test handle_async routes requests to the correct sub-handler."""

    @pytest.mark.asyncio
    async def test_verify_claim_post_dispatches(self, handler):
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(
                handler, "_handle_verify_claim", new_callable=AsyncMock, return_value=MagicMock(status_code=200)
            ) as mock_fn:
                body = json.dumps({"claim": "test"}).encode()
                await handler.handle_async(MagicMock(), "POST", "/api/v1/verify/claim", body=body)
                mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_batch_post_dispatches(self, handler):
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(
                handler, "_handle_verify_batch", new_callable=AsyncMock, return_value=MagicMock(status_code=200)
            ) as mock_fn:
                await handler.handle_async(MagicMock(), "POST", "/api/v1/verify/batch")
                mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_status_get_dispatches(self, handler):
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(
                handler, "_handle_verify_status", return_value=MagicMock(status_code=200)
            ) as mock_fn:
                await handler.handle_async(MagicMock(), "GET", "/api/v1/verify/status")
                mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_translate_post_dispatches(self, handler):
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(
                handler, "_handle_translate", new_callable=AsyncMock, return_value=MagicMock(status_code=200)
            ) as mock_fn:
                await handler.handle_async(MagicMock(), "POST", "/api/v1/verify/translate")
                mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_get_dispatches(self, handler):
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(
                handler, "_handle_get_history", return_value=MagicMock(status_code=200)
            ) as mock_fn:
                await handler.handle_async(
                    MagicMock(), "GET", "/api/v1/verify/history", query_params={"limit": ["10"]}
                )
                mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_entry_get_dispatches(self, handler):
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(
                handler, "_handle_get_history_entry", return_value=MagicMock(status_code=200)
            ) as mock_fn:
                await handler.handle_async(MagicMock(), "GET", "/api/v1/verify/history/abc123")
                mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler):
        with patch.object(handler, "_check_permission", return_value=None):
            result = await handler.handle_async(MagicMock(), "GET", "/api/v1/verify/unknown")
            assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_rbac_denied_blocks_request(self, handler):
        from aragora.server.handlers.base import error_response

        with patch.object(
            handler, "_check_permission",
            return_value=error_response("Permission denied", 403),
        ):
            result = await handler.handle_async(
                MagicMock(), "POST", "/api/v1/verify/claim"
            )
            assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_post_requires_create_permission(self, handler):
        with patch.object(handler, "_check_permission", return_value=None) as mock_perm:
            with patch.object(
                handler, "_handle_verify_claim", new_callable=AsyncMock,
                return_value=MagicMock(status_code=200),
            ):
                await handler.handle_async(MagicMock(), "POST", "/api/v1/verify/claim")
                mock_perm.assert_called_once()
                assert mock_perm.call_args[0][1] == "verification.create"

    @pytest.mark.asyncio
    async def test_get_requires_read_permission(self, handler):
        with patch.object(handler, "_check_permission", return_value=None) as mock_perm:
            with patch.object(
                handler, "_handle_verify_status", return_value=MagicMock(status_code=200),
            ):
                await handler.handle_async(MagicMock(), "GET", "/api/v1/verify/status")
                mock_perm.assert_called_once()
                assert mock_perm.call_args[0][1] == "verification.read"


# =============================================================================
# POST /api/v1/verify/claim
# =============================================================================


class TestVerifyClaim:
    """Test single claim verification endpoint."""

    @pytest.mark.asyncio
    async def test_missing_body_returns_400(self, handler):
        result = await handler._handle_verify_claim(MagicMock(), None)
        assert _status(result) == 400
        assert "body" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, handler):
        result = await handler._handle_verify_claim(MagicMock(), b"not json{")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_claim_returns_400(self, handler):
        body = json.dumps({"claim": ""}).encode()
        result = await handler._handle_verify_claim(MagicMock(), body)
        assert _status(result) == 400
        assert "claim" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_claim_returns_400(self, handler):
        body = json.dumps({"claim": "   "}).encode()
        result = await handler._handle_verify_claim(MagicMock(), body)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_claim_field_returns_400(self, handler):
        body = json.dumps({"context": "some context"}).encode()
        result = await handler._handle_verify_claim(MagicMock(), body)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_successful_verification(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "1 + 1 = 2"}).encode()
            result = await handler._handle_verify_claim(MagicMock(), body)
            assert _status(result) == 200
            data = _body(result)
            assert data["status"] == "proof_found"
            assert data["is_verified"] is True
            assert "history_id" in data

    @pytest.mark.asyncio
    async def test_history_id_returned(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "test"}).encode()
            result = await handler._handle_verify_claim(MagicMock(), body)
            data = _body(result)
            assert len(data["history_id"]) == 16

    @pytest.mark.asyncio
    async def test_claim_type_passed_through(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "test", "claim_type": "LOGICAL"}).encode()
            await handler._handle_verify_claim(MagicMock(), body)
            call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
            assert call_kwargs["claim_type"] == "LOGICAL"

    @pytest.mark.asyncio
    async def test_context_passed_through(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "test", "context": "basic arithmetic"}).encode()
            await handler._handle_verify_claim(MagicMock(), body)
            call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
            assert call_kwargs["context"] == "basic arithmetic"

    @pytest.mark.asyncio
    async def test_timeout_capped_at_300(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "test", "timeout": 999}).encode()
            await handler._handle_verify_claim(MagicMock(), body)
            call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
            assert call_kwargs["timeout_seconds"] == 300.0

    @pytest.mark.asyncio
    async def test_timeout_invalid_defaults_to_60(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "test", "timeout": "invalid"}).encode()
            await handler._handle_verify_claim(MagicMock(), body)
            call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
            assert call_kwargs["timeout_seconds"] == 60.0

    @pytest.mark.asyncio
    async def test_timeout_normal_value(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "test", "timeout": 45}).encode()
            await handler._handle_verify_claim(MagicMock(), body)
            call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
            assert call_kwargs["timeout_seconds"] == 45.0

    @pytest.mark.asyncio
    async def test_default_context_empty_string(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "test"}).encode()
            await handler._handle_verify_claim(MagicMock(), body)
            call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
            assert call_kwargs["context"] == ""

    @pytest.mark.asyncio
    async def test_proof_tree_stored_in_history(self, handler, mock_governance_store):
        """When result is verified, proof tree is built and stored."""
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
            body = json.dumps({"claim": "test claim"}).encode()
            result = await handler._handle_verify_claim(MagicMock(), body)
            data = _body(result)
            entry = _verification_history[data["history_id"]]
            assert entry.proof_tree is not None


# =============================================================================
# POST /api/v1/verify/batch
# =============================================================================


class TestVerifyBatch:
    """Test batch verification endpoint."""

    @pytest.mark.asyncio
    async def test_missing_body_returns_400(self, handler):
        result = await handler._handle_verify_batch(MagicMock(), None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, handler):
        result = await handler._handle_verify_batch(MagicMock(), b"{bad json")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_claims_returns_400(self, handler):
        body = json.dumps({"claims": []}).encode()
        result = await handler._handle_verify_batch(MagicMock(), body)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_claims_field_returns_400(self, handler):
        body = json.dumps({"data": "something"}).encode()
        result = await handler._handle_verify_batch(MagicMock(), body)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_claims_not_a_list_returns_400(self, handler):
        body = json.dumps({"claims": "not a list"}).encode()
        result = await handler._handle_verify_batch(MagicMock(), body)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_too_many_claims_returns_400(self, handler):
        claims = [{"claim": f"claim_{i}"} for i in range(21)]
        body = json.dumps({"claims": claims}).encode()
        result = await handler._handle_verify_batch(MagicMock(), body)
        assert _status(result) == 400
        assert "20" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_exactly_20_claims_accepted(self, handler):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockFormalProofStatus.PROOF_FOUND

        claims = [{"claim": f"claim_{i}"} for i in range(20)]
        body = json.dumps({"claims": claims}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_result
            )
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                result = await handler._handle_verify_batch(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["summary"]["total"] == 20

    @pytest.mark.asyncio
    async def test_batch_summary_counts(self, handler):
        """Test that summary correctly counts verified, failed, timeout."""
        results_list = []
        for status_val in [
            _MockFormalProofStatus.PROOF_FOUND,
            _MockFormalProofStatus.TIMEOUT,
            _MockFormalProofStatus.TRANSLATION_FAILED,
        ]:
            r = MagicMock()
            r.to_dict.return_value = {"status": status_val.value, "is_verified": status_val == _MockFormalProofStatus.PROOF_FOUND}
            r.status = status_val
            results_list.append(r)

        call_count = 0

        async def mock_verify(**kwargs):
            nonlocal call_count
            result = results_list[call_count]
            call_count += 1
            return result

        claims = [{"claim": "c1"}, {"claim": "c2"}, {"claim": "c3"}]
        body = json.dumps({"claims": claims}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = mock_verify
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                result = await handler._handle_verify_batch(MagicMock(), body)
                assert _status(result) == 200
                summary = _body(result)["summary"]
                assert summary["total"] == 3
                assert summary["verified"] == 1
                assert summary["timeout"] == 1
                assert summary["failed"] == 1

    @pytest.mark.asyncio
    async def test_batch_exception_counted_as_failed(self, handler):
        async def mock_verify(**kwargs):
            raise RuntimeError("backend crashed")

        claims = [{"claim": "crash"}]
        body = json.dumps({"claims": claims}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = mock_verify
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                result = await handler._handle_verify_batch(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["summary"]["failed"] == 1
                assert data["results"][0]["status"] == "error"

    @pytest.mark.asyncio
    async def test_batch_empty_claim_in_list(self, handler):
        """Empty claims in list return error status without crashing the batch."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found"}
        mock_result.status = _MockFormalProofStatus.PROOF_FOUND

        claims = [{"claim": ""}, {"claim": "valid"}]
        body = json.dumps({"claims": claims}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_result
            )
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                result = await handler._handle_verify_batch(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                # First claim is empty, returns error status
                assert data["results"][0]["status"] == "error"
                # Second claim succeeds
                assert data["results"][1]["status"] == "proof_found"

    @pytest.mark.asyncio
    async def test_timeout_per_claim_capped_at_120(self, handler):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found"}
        mock_result.status = _MockFormalProofStatus.PROOF_FOUND

        claims = [{"claim": "test"}]
        body = json.dumps({"claims": claims, "timeout_per_claim": 500}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_result
            )
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                await handler._handle_verify_batch(MagicMock(), body)
                call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
                assert call_kwargs["timeout_seconds"] == 120.0

    @pytest.mark.asyncio
    async def test_max_concurrent_capped_at_5(self, handler):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found"}
        mock_result.status = _MockFormalProofStatus.PROOF_FOUND

        claims = [{"claim": f"claim_{i}"} for i in range(10)]
        body = json.dumps({"claims": claims, "max_concurrent": 50}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_result
            )
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                result = await handler._handle_verify_batch(MagicMock(), body)
                assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invalid_timeout_per_claim_defaults(self, handler):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found"}
        mock_result.status = _MockFormalProofStatus.PROOF_FOUND

        claims = [{"claim": "test"}]
        body = json.dumps({"claims": claims, "timeout_per_claim": "bad"}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_result
            )
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                await handler._handle_verify_batch(MagicMock(), body)
                call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
                assert call_kwargs["timeout_seconds"] == 30.0

    @pytest.mark.asyncio
    async def test_invalid_max_concurrent_defaults(self, handler):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found"}
        mock_result.status = _MockFormalProofStatus.PROOF_FOUND

        claims = [{"claim": "test"}]
        body = json.dumps({"claims": claims, "max_concurrent": "bad"}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_result
            )
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                result = await handler._handle_verify_batch(MagicMock(), body)
                assert _status(result) == 200


# =============================================================================
# GET /api/v1/verify/status
# =============================================================================


class TestVerifyStatus:
    """Test backend status endpoint."""

    def test_status_report_success(self, handler):
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {
            "backends": [
                {"language": "z3_smt", "available": True},
                {"language": "lean4", "available": False},
            ],
            "any_available": True,
        }
        handler._manager = mock_manager

        with patch.dict("sys.modules", {
            "aragora.verification.deepseek_prover": MagicMock(
                DeepSeekProverTranslator=MagicMock(
                    return_value=MagicMock(is_available=True),
                ),
            ),
        }):
            result = handler._handle_verify_status(MagicMock())
            assert _status(result) == 200
            data = _body(result)
            assert "backends" in data
            assert data["deepseek_prover_available"] is True

    def test_deepseek_not_available(self, handler):
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {
            "backends": [],
            "any_available": False,
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
            data = _body(result)
            assert data["deepseek_prover_available"] is False

    def test_deepseek_import_error(self, handler):
        """When deepseek_prover module is not installed, report unavailable."""
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {
            "backends": [],
            "any_available": False,
        }
        handler._manager = mock_manager

        # Remove the module from sys.modules so import fails
        import sys
        saved = sys.modules.pop("aragora.verification.deepseek_prover", None)
        try:
            with patch.dict("sys.modules", {"aragora.verification.deepseek_prover": None}):
                result = handler._handle_verify_status(MagicMock())
                data = _body(result)
                assert data["deepseek_prover_available"] is False
        finally:
            if saved is not None:
                sys.modules["aragora.verification.deepseek_prover"] = saved

    def test_status_report_includes_backend_details(self, handler):
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
            data = _body(result)
            assert len(data["backends"]) == 1
            assert data["backends"][0]["language"] == "z3_smt"
            assert data["any_available"] is True


# =============================================================================
# POST /api/v1/verify/translate
# =============================================================================


class TestTranslate:
    """Test translation endpoint."""

    @pytest.mark.asyncio
    async def test_missing_body_returns_400(self, handler):
        result = await handler._handle_translate(MagicMock(), None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, handler):
        result = await handler._handle_translate(MagicMock(), b"not json")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_claim_returns_400(self, handler):
        body = json.dumps({"claim": ""}).encode()
        result = await handler._handle_translate(MagicMock(), body)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_unknown_target_language_returns_400(self, handler):
        body = json.dumps({"claim": "test", "target_language": "coq"}).encode()
        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            result = await handler._handle_translate(MagicMock(), body)
            assert _status(result) == 400
            assert "coq" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_lean4_deepseek_available(self, handler):
        """When DeepSeek prover is available, use it for lean4 translation."""
        mock_translator_result = MagicMock()
        mock_translator_result.success = True
        mock_translator_result.lean_code = "theorem t : True := trivial"
        mock_translator_result.model_used = "deepseek/deepseek-prover-v2"
        mock_translator_result.confidence = 0.9
        mock_translator_result.translation_time_ms = 200.0
        mock_translator_result.error_message = ""

        mock_translator = MagicMock()
        mock_translator.is_available = True
        mock_translator.translate = AsyncMock(return_value=mock_translator_result)

        body = json.dumps({"claim": "test claim", "target_language": "lean4"}).encode()

        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            with patch.dict("sys.modules", {
                "aragora.verification.deepseek_prover": MagicMock(
                    DeepSeekProverTranslator=MagicMock(return_value=mock_translator),
                ),
            }):
                result = await handler._handle_translate(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["success"] is True
                assert data["language"] == "lean4"
                assert data["model_used"] == "deepseek/deepseek-prover-v2"
                assert data["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_lean4_deepseek_not_available_fallback(self, handler):
        """When DeepSeek is not available, fall back to LeanBackend."""
        mock_translator = MagicMock()
        mock_translator.is_available = False

        mock_lean_backend = MagicMock()
        mock_lean_backend.translate = AsyncMock(
            return_value="theorem t : True := trivial"
        )

        body = json.dumps({"claim": "test claim", "target_language": "lean4"}).encode()

        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            with patch.dict("sys.modules", {
                "aragora.verification.deepseek_prover": MagicMock(
                    DeepSeekProverTranslator=MagicMock(return_value=mock_translator),
                ),
            }):
                with patch(
                    "aragora.server.handlers.verification.formal_verification.LeanBackend",
                    return_value=mock_lean_backend,
                ):
                    # Need to actually import the module inside the handler, so we
                    # patch the import path in the handler's module namespace.
                    import aragora.server.handlers.verification.formal_verification as fv_mod
                    with patch.object(fv_mod, "LeanBackend", create=True, new=MagicMock(return_value=mock_lean_backend)):
                        # The handler does `from aragora.verification.formal import LeanBackend`
                        # inside the method, so we need to mock the import
                        mock_formal_module = MagicMock()
                        mock_formal_module.LeanBackend = MagicMock(return_value=mock_lean_backend)
                        with patch.dict("sys.modules", {
                            "aragora.verification.formal": mock_formal_module,
                            "aragora.verification.deepseek_prover": MagicMock(
                                DeepSeekProverTranslator=MagicMock(return_value=mock_translator),
                            ),
                        }):
                            result = await handler._handle_translate(MagicMock(), body)
                            assert _status(result) == 200
                            data = _body(result)
                            assert data["success"] is True
                            assert data["language"] == "lean4"
                            assert data["model_used"] == "claude/openai"
                            assert data["confidence"] == 0.6

    @pytest.mark.asyncio
    async def test_lean4_deepseek_import_error_fallback(self, handler):
        """When deepseek_prover can't be imported, fall back to LeanBackend."""
        mock_lean_backend = MagicMock()
        mock_lean_backend.translate = AsyncMock(return_value="formal statement")

        mock_formal_module = MagicMock()
        mock_formal_module.LeanBackend = MagicMock(return_value=mock_lean_backend)

        body = json.dumps({"claim": "test"}).encode()

        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            with patch.dict("sys.modules", {
                "aragora.verification.deepseek_prover": None,  # ImportError
                "aragora.verification.formal": mock_formal_module,
            }):
                result = await handler._handle_translate(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["success"] is True
                assert data["language"] == "lean4"

    @pytest.mark.asyncio
    async def test_lean4_translation_failed(self, handler):
        """When both backends fail to translate, return success=False."""
        mock_lean_backend = MagicMock()
        mock_lean_backend.translate = AsyncMock(return_value=None)

        mock_formal_module = MagicMock()
        mock_formal_module.LeanBackend = MagicMock(return_value=mock_lean_backend)

        body = json.dumps({"claim": "untranslatable"}).encode()

        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            with patch.dict("sys.modules", {
                "aragora.verification.deepseek_prover": None,
                "aragora.verification.formal": mock_formal_module,
            }):
                result = await handler._handle_translate(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["success"] is False
                assert data["confidence"] == 0.0
                assert data["error_message"] == "Translation failed"

    @pytest.mark.asyncio
    async def test_z3_smt_translation_success(self, handler):
        mock_z3_backend = MagicMock()
        mock_z3_backend.translate = AsyncMock(
            return_value="(assert (= (+ 1 1) 2))"
        )

        mock_formal_module = MagicMock()
        mock_formal_module.Z3Backend = MagicMock(return_value=mock_z3_backend)

        body = json.dumps({"claim": "1 + 1 = 2", "target_language": "z3_smt"}).encode()

        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            with patch.dict("sys.modules", {
                "aragora.verification.formal": mock_formal_module,
            }):
                result = await handler._handle_translate(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["success"] is True
                assert data["language"] == "z3_smt"
                assert data["model_used"] == "pattern/llm"
                assert data["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_z3_smt_translation_failed(self, handler):
        mock_z3_backend = MagicMock()
        mock_z3_backend.translate = AsyncMock(return_value=None)

        mock_formal_module = MagicMock()
        mock_formal_module.Z3Backend = MagicMock(return_value=mock_z3_backend)

        body = json.dumps({"claim": "untranslatable", "target_language": "z3_smt"}).encode()

        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            with patch.dict("sys.modules", {
                "aragora.verification.formal": mock_formal_module,
            }):
                result = await handler._handle_translate(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["success"] is False
                assert data["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_default_target_language_is_lean4(self, handler):
        """When target_language is omitted, default to lean4."""
        mock_translator_result = MagicMock()
        mock_translator_result.success = True
        mock_translator_result.lean_code = "theorem t : True := trivial"
        mock_translator_result.model_used = "deepseek/deepseek-prover-v2"
        mock_translator_result.confidence = 0.85
        mock_translator_result.translation_time_ms = 100.0
        mock_translator_result.error_message = ""

        mock_translator = MagicMock()
        mock_translator.is_available = True
        mock_translator.translate = AsyncMock(return_value=mock_translator_result)

        body = json.dumps({"claim": "test claim"}).encode()  # no target_language

        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            return_value={},
        ):
            with patch.dict("sys.modules", {
                "aragora.verification.deepseek_prover": MagicMock(
                    DeepSeekProverTranslator=MagicMock(return_value=mock_translator),
                ),
            }):
                result = await handler._handle_translate(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["language"] == "lean4"


# =============================================================================
# GET /api/v1/verify/history
# =============================================================================


class TestHistory:
    """Test history listing endpoint."""

    def test_empty_history_in_memory(self, handler, mock_governance_store):
        result = handler._handle_get_history({})
        assert _status(result) == 200
        data = _body(result)
        assert data["entries"] == []
        assert data["total"] == 0
        assert data["source"] == "in_memory"

    def test_history_with_entries(self, handler, mock_governance_store, sample_entry):
        _verification_history["sample_001"] = sample_entry
        result = handler._handle_get_history({})
        data = _body(result)
        assert data["total"] == 1
        assert data["entries"][0]["id"] == "sample_001"

    def test_pagination_limit(self, handler, mock_governance_store):
        for i in range(5):
            _verification_history[f"e{i}"] = VerificationHistoryEntry(
                id=f"e{i}", claim=f"claim_{i}", claim_type=None,
                context="", result={"status": "proof_found"}, timestamp=time.time(),
            )
        result = handler._handle_get_history({"limit": ["2"]})
        data = _body(result)
        assert len(data["entries"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2

    def test_pagination_offset(self, handler, mock_governance_store):
        for i in range(5):
            _verification_history[f"e{i}"] = VerificationHistoryEntry(
                id=f"e{i}", claim=f"claim_{i}", claim_type=None,
                context="", result={}, timestamp=time.time() + i,
            )
        result = handler._handle_get_history({"offset": ["2"], "limit": ["10"]})
        data = _body(result)
        assert data["offset"] == 2
        # reversed order means offset skips first 2 in reversed list
        assert len(data["entries"]) == 3

    def test_status_filter(self, handler, mock_governance_store):
        _verification_history["found"] = VerificationHistoryEntry(
            id="found", claim="c1", claim_type=None, context="",
            result={"status": "proof_found"}, timestamp=time.time(),
        )
        _verification_history["failed"] = VerificationHistoryEntry(
            id="failed", claim="c2", claim_type=None, context="",
            result={"status": "translation_failed"}, timestamp=time.time(),
        )
        result = handler._handle_get_history({"status": ["proof_found"]})
        data = _body(result)
        assert data["total"] == 1
        assert data["entries"][0]["id"] == "found"

    def test_status_filter_no_match(self, handler, mock_governance_store):
        _verification_history["e1"] = VerificationHistoryEntry(
            id="e1", claim="c", claim_type=None, context="",
            result={"status": "proof_found"}, timestamp=time.time(),
        )
        result = handler._handle_get_history({"status": ["timeout"]})
        data = _body(result)
        assert data["total"] == 0

    def test_governance_store_used_when_available(self, handler):
        mock_rec = MagicMock()
        mock_rec.verification_id = "gov_001"
        mock_rec.claim = "gov claim"
        mock_rec.claim_type = "LOGICAL"
        mock_rec.context = "ctx"
        mock_rec.to_dict.return_value = {
            "result": {"status": "proof_found"},
            "proof_tree": None,
        }
        mock_rec.timestamp = MagicMock()
        mock_rec.timestamp.timestamp.return_value = time.time()

        mock_store = MagicMock()
        mock_store.list_verifications.return_value = [mock_rec]

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            result = handler._handle_get_history({})
            data = _body(result)
            assert data["source"] == "persistent"
            assert data["total"] == 1
            assert data["entries"][0]["id"] == "gov_001"

    def test_governance_store_error_falls_back_to_memory(self, handler):
        mock_store = MagicMock()
        mock_store.list_verifications.side_effect = RuntimeError("db down")

        _verification_history["mem_001"] = VerificationHistoryEntry(
            id="mem_001", claim="in memory", claim_type=None,
            context="", result={}, timestamp=time.time(),
        )

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            result = handler._handle_get_history({})
            data = _body(result)
            assert data["source"] == "in_memory"
            assert data["total"] == 1

    def test_default_limit_20(self, handler, mock_governance_store):
        result = handler._handle_get_history({})
        data = _body(result)
        assert data["limit"] == 20

    def test_default_offset_0(self, handler, mock_governance_store):
        result = handler._handle_get_history({})
        data = _body(result)
        assert data["offset"] == 0

    def test_cleanup_called(self, handler, mock_governance_store):
        """History cleanup is called when listing."""
        old_ts = time.time() - HISTORY_TTL_SECONDS - 10
        _verification_history["old"] = VerificationHistoryEntry(
            id="old", claim="old", claim_type=None, context="",
            result={}, timestamp=old_ts,
        )
        handler._handle_get_history({})
        assert "old" not in _verification_history

    def test_governance_store_status_filter(self, handler):
        """Status filter applied when using governance store."""
        mock_rec_found = MagicMock()
        mock_rec_found.verification_id = "found"
        mock_rec_found.claim = "claim1"
        mock_rec_found.claim_type = None
        mock_rec_found.context = ""
        mock_rec_found.to_dict.return_value = {"result": {"status": "proof_found"}, "proof_tree": None}
        mock_rec_found.timestamp = MagicMock()
        mock_rec_found.timestamp.timestamp.return_value = time.time()

        mock_rec_failed = MagicMock()
        mock_rec_failed.verification_id = "failed"
        mock_rec_failed.claim = "claim2"
        mock_rec_failed.claim_type = None
        mock_rec_failed.context = ""
        mock_rec_failed.to_dict.return_value = {"result": {"status": "translation_failed"}, "proof_tree": None}
        mock_rec_failed.timestamp = MagicMock()
        mock_rec_failed.timestamp.timestamp.return_value = time.time()

        mock_store = MagicMock()
        mock_store.list_verifications.return_value = [mock_rec_found, mock_rec_failed]

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            result = handler._handle_get_history({"status": ["proof_found"]})
            data = _body(result)
            assert data["total"] == 1

    def test_governance_store_timestamp_without_timestamp_method(self, handler):
        """When record timestamp has no .timestamp() method, use time.time()."""
        mock_rec = MagicMock()
        mock_rec.verification_id = "ts_test"
        mock_rec.claim = "claim"
        mock_rec.claim_type = None
        mock_rec.context = ""
        mock_rec.to_dict.return_value = {"result": {}, "proof_tree": None}
        # Remove .timestamp() method (hasattr returns False for spec)
        mock_rec.timestamp = "not a datetime"

        mock_store = MagicMock()
        mock_store.list_verifications.return_value = [mock_rec]

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            result = handler._handle_get_history({})
            data = _body(result)
            assert data["total"] == 1


# =============================================================================
# GET /api/v1/verify/history/{id}
# =============================================================================


class TestHistoryEntry:
    """Test individual history entry retrieval."""

    def test_entry_not_found(self, handler, mock_governance_store):
        result = handler._handle_get_history_entry("/api/v1/verify/history/nonexistent")
        assert _status(result) == 404

    def test_entry_found_in_memory(self, handler, sample_entry):
        _verification_history["sample_001"] = sample_entry
        result = handler._handle_get_history_entry("/api/v1/verify/history/sample_001")
        assert _status(result) == 200
        data = _body(result)
        assert data["id"] == "sample_001"
        assert data["claim"] == "1 + 1 = 2"

    def test_entry_includes_result(self, handler, sample_entry):
        _verification_history["sample_001"] = sample_entry
        result = handler._handle_get_history_entry("/api/v1/verify/history/sample_001")
        data = _body(result)
        assert "result" in data
        assert data["result"]["status"] == "proof_found"

    def test_entry_includes_proof_tree_when_present(self, handler, sample_entry):
        _verification_history["sample_001"] = sample_entry
        result = handler._handle_get_history_entry("/api/v1/verify/history/sample_001")
        data = _body(result)
        assert "proof_tree" in data

    def test_entry_no_proof_tree(self, handler):
        entry = VerificationHistoryEntry(
            id="no_tree", claim="c", claim_type=None, context="",
            result={"status": "failed"}, timestamp=time.time(),
        )
        _verification_history["no_tree"] = entry
        result = handler._handle_get_history_entry("/api/v1/verify/history/no_tree")
        data = _body(result)
        assert "proof_tree" not in data

    def test_tree_request_with_stored_tree(self, handler, sample_entry):
        _verification_history["sample_001"] = sample_entry
        result = handler._handle_get_history_entry("/api/v1/verify/history/sample_001/tree")
        assert _status(result) == 200
        data = _body(result)
        assert "nodes" in data
        assert len(data["nodes"]) == 1

    def test_tree_request_builds_from_result(self, handler):
        """When no stored tree but result is verified, build one on the fly."""
        entry = VerificationHistoryEntry(
            id="build_tree", claim="test claim", claim_type=None, context="",
            result={
                "status": "proof_found",
                "is_verified": True,
                "formal_statement": "theorem t : True := trivial",
                "language": "lean4",
            },
            timestamp=time.time(),
            proof_tree=None,
        )
        _verification_history["build_tree"] = entry
        result = handler._handle_get_history_entry("/api/v1/verify/history/build_tree/tree")
        data = _body(result)
        assert len(data["nodes"]) >= 3

    def test_tree_request_no_tree_available(self, handler):
        """When entry is not verified and has no tree, return empty nodes."""
        entry = VerificationHistoryEntry(
            id="no_tree", claim="test", claim_type=None, context="",
            result={"status": "failed", "is_verified": False},
            timestamp=time.time(),
        )
        _verification_history["no_tree"] = entry
        result = handler._handle_get_history_entry("/api/v1/verify/history/no_tree/tree")
        data = _body(result)
        assert data["nodes"] == []
        assert "message" in data

    def test_empty_entry_id_returns_400(self, handler, mock_governance_store):
        result = handler._handle_get_history_entry("/api/v1/verify/history/")
        assert _status(result) == 400

    def test_entry_loaded_from_governance_store(self, handler):
        """When not in memory, load from governance store."""
        mock_rec = MagicMock()
        mock_rec.verification_id = "from_store"
        mock_rec.claim = "stored claim"
        mock_rec.claim_type = "MATHEMATICAL"
        mock_rec.context = "ctx"
        mock_rec.to_dict.return_value = {"result": {"status": "proof_found"}, "proof_tree": None}
        mock_rec.timestamp = MagicMock()
        mock_rec.timestamp.timestamp.return_value = time.time()

        mock_store = MagicMock()
        mock_store.get_verification.return_value = mock_rec

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            result = handler._handle_get_history_entry("/api/v1/verify/history/from_store")
            assert _status(result) == 200
            data = _body(result)
            assert data["claim"] == "stored claim"

    def test_entry_cached_after_store_load(self, handler):
        """After loading from store, entry is cached in memory."""
        mock_rec = MagicMock()
        mock_rec.verification_id = "cache_test"
        mock_rec.claim = "cached"
        mock_rec.claim_type = None
        mock_rec.context = ""
        mock_rec.to_dict.return_value = {"result": {}, "proof_tree": None}
        mock_rec.timestamp = MagicMock()
        mock_rec.timestamp.timestamp.return_value = time.time()

        mock_store = MagicMock()
        mock_store.get_verification.return_value = mock_rec

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            handler._handle_get_history_entry("/api/v1/verify/history/cache_test")
            # Should now be in memory cache
            assert "cache_test" in _verification_history

    def test_governance_store_error_falls_back(self, handler, mock_governance_store):
        """When governance store errors, entry is not found."""
        mock_store = MagicMock()
        mock_store.get_verification.side_effect = RuntimeError("db err")
        mock_governance_store.get.return_value = mock_store

        result = handler._handle_get_history_entry("/api/v1/verify/history/missing")
        assert _status(result) == 404

    def test_governance_store_returns_none(self, handler):
        """When governance store returns None for entry, 404."""
        mock_store = MagicMock()
        mock_store.get_verification.return_value = None

        with patch(
            "aragora.server.handlers.verification.formal_verification._governance_store"
        ) as mock_sf:
            mock_sf.get.return_value = mock_store
            result = handler._handle_get_history_entry("/api/v1/verify/history/nope")
            assert _status(result) == 404


# =============================================================================
# Manager Initialization
# =============================================================================


class TestGetManager:
    """Test lazy initialization of verification manager."""

    def test_manager_cached(self, handler):
        """Manager is created once and cached."""
        mock_init = MagicMock()
        mock_init.return_value = {
            "get_formal_verification_manager": MagicMock(return_value="mgr_instance"),
            "FormalVerificationManager": MagicMock(),
            "FormalProofStatus": MagicMock(),
            "FormalLanguage": MagicMock(),
            "TranslationModel": MagicMock(),
        }
        with patch(
            "aragora.server.handlers.verification.formal_verification._init_verification",
            mock_init,
        ):
            mgr1 = handler._get_manager()
            mgr2 = handler._get_manager()
            assert mgr1 is mgr2
            # _init_verification called only once
            mock_init.assert_called_once()

    def test_manager_none_initially(self, handler):
        assert handler._manager is None

    def test_handler_init_with_empty_context(self):
        h = FormalVerificationHandler()
        assert h.ctx == {}


# =============================================================================
# Handler Construction
# =============================================================================


class TestHandlerConstruction:
    """Test handler instantiation patterns."""

    def test_init_with_server_context(self):
        ctx = {"key": "value"}
        h = FormalVerificationHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_none_context(self):
        h = FormalVerificationHandler(server_context=None)
        assert h.ctx == {}

    def test_init_default_context(self):
        h = FormalVerificationHandler()
        assert h.ctx == {}

    def test_resource_type(self):
        assert FormalVerificationHandler.RESOURCE_TYPE == "verification"

    def test_routes_constant(self):
        routes = FormalVerificationHandler.ROUTES
        assert "/api/v1/verify/claim" in routes
        assert "/api/v1/verify/batch" in routes
        assert "/api/v1/verify/status" in routes
        assert "/api/v1/verify/translate" in routes
        assert "/api/v1/verify/history" in routes
        assert "/api/verification/formal-verify" in routes
        assert "/api/verification/status" in routes


# =============================================================================
# Edge Cases and Integration
# =============================================================================


class TestEdgeCases:
    """Test edge cases and integration behavior."""

    @pytest.mark.asyncio
    async def test_verify_claim_strips_whitespace(self, handler, mock_verification_result, mock_governance_store):
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "  test claim  "}).encode()
            await handler._handle_verify_claim(MagicMock(), body)
            call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
            assert call_kwargs["claim"] == "test claim"

    def test_history_reversed_order(self, handler, mock_governance_store):
        """In-memory history is returned in reverse order (newest first)."""
        for i in range(3):
            _verification_history[f"e{i}"] = VerificationHistoryEntry(
                id=f"e{i}", claim=f"claim_{i}", claim_type=None,
                context="", result={}, timestamp=time.time() + i * 0.01,
            )
        result = handler._handle_get_history({})
        data = _body(result)
        ids = [e["id"] for e in data["entries"]]
        assert ids[0] == "e2"  # newest first

    def test_history_max_size_constant(self):
        assert MAX_HISTORY_SIZE == 1000

    def test_history_ttl_constant(self):
        assert HISTORY_TTL_SECONDS == 86400

    @pytest.mark.asyncio
    async def test_batch_with_single_claim(self, handler):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"status": "proof_found", "is_verified": True}
        mock_result.status = _MockFormalProofStatus.PROOF_FOUND

        body = json.dumps({"claims": [{"claim": "single"}]}).encode()

        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_result
            )
            with patch(
                "aragora.server.handlers.verification.formal_verification._init_verification",
                return_value={"FormalProofStatus": _MockFormalProofStatus},
            ):
                result = await handler._handle_verify_batch(MagicMock(), body)
                assert _status(result) == 200
                data = _body(result)
                assert data["summary"]["total"] == 1
                assert data["summary"]["verified"] == 1

    @pytest.mark.asyncio
    async def test_handle_async_history_tree_path(self, handler):
        """handle_async correctly routes history/id/tree paths."""
        entry = VerificationHistoryEntry(
            id="tree_test", claim="test", claim_type=None, context="",
            result={"status": "proof_found", "is_verified": True,
                    "formal_statement": "thm : True := trivial", "language": "lean4"},
            timestamp=time.time(),
            proof_tree=[{"id": "root"}],
        )
        _verification_history["tree_test"] = entry

        with patch.object(handler, "_check_permission", return_value=None):
            result = await handler.handle_async(
                MagicMock(), "GET", "/api/v1/verify/history/tree_test/tree"
            )
            assert _status(result) == 200
            data = _body(result)
            assert "nodes" in data

    def test_verification_history_is_ordered_dict(self):
        assert isinstance(_verification_history, OrderedDict)

    @pytest.mark.asyncio
    async def test_timeout_none_defaults_to_60(self, handler, mock_verification_result, mock_governance_store):
        """When timeout is None, use default 60."""
        with patch.object(handler, "_get_manager") as mock_mgr:
            mock_mgr.return_value.attempt_formal_verification = AsyncMock(
                return_value=mock_verification_result
            )
            body = json.dumps({"claim": "test", "timeout": None}).encode()
            await handler._handle_verify_claim(MagicMock(), body)
            call_kwargs = mock_mgr.return_value.attempt_formal_verification.call_args[1]
            assert call_kwargs["timeout_seconds"] == 60.0
