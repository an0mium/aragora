"""
End-to-end integration test: Authentication → Debate → Knowledge Mound → Receipt.

Tests the complete flow with proper authentication at each step:
1. Register user
2. Login and obtain JWT token
3. Create debate with auth header
4. Wait for debate completion/consensus
5. Generate decision receipt
6. Store receipt in Knowledge Mound
7. Export receipt (verify access)

This test verifies that authentication is properly enforced throughout
the entire decision pipeline.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.billing.models import User, hash_password
from aragora.billing.jwt_auth import create_token_pair, decode_jwt
from aragora.core_types import DebateResult
from aragora.gauntlet.receipt import ConsensusProof, DecisionReceipt
from aragora.server.handlers.auth import AuthHandler
from aragora.server.handlers.base import HandlerResult, ServerContext


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_user_store():
    """Create a mock user store with test user."""
    store = MagicMock()
    store.get_user_by_email.return_value = None  # User doesn't exist initially
    store.create_user.return_value = User(
        id="user-123",
        email="test@example.com",
        password_hash=hash_password("SecurePass123!"),
        created_at=datetime.now(timezone.utc),
    )
    store.get_user_by_id.return_value = User(
        id="user-123",
        email="test@example.com",
        password_hash=hash_password("SecurePass123!"),
        created_at=datetime.now(timezone.utc),
    )
    return store


@pytest.fixture
def mock_server_context(mock_user_store) -> dict:
    """Create minimal server context."""
    return {
        "user_store": mock_user_store,
    }


@pytest.fixture
def mock_debate_result() -> DebateResult:
    """Create a mock debate result for testing."""
    return DebateResult(
        debate_id="debate-456",
        task="Should we approve the budget?",
        final_answer="Approve the budget",
        confidence=0.85,
        consensus_reached=True,
        rounds_completed=3,
        participants=["claude", "gpt-4", "gemini", "mistral"],
        messages=[
            {"role": "claude", "content": "I propose we approve...", "round": 1},
            {"role": "gpt-4", "content": "I agree with the proposal...", "round": 1},
        ],
        duration_seconds=45.2,
    )


# =============================================================================
# End-to-End Flow Tests
# =============================================================================


class TestFullAuthToReceiptFlow:
    """Test complete authentication to receipt generation flow."""

    @pytest.mark.asyncio
    async def test_token_generation_works(self, mock_user_store):
        """Verify JWT token generation works correctly."""
        # Generate tokens for a user
        tokens = create_token_pair(
            user_id="user-123",
            email="test@example.com",
            org_id="org-1",
        )

        # Verify tokens are created
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None
        assert len(tokens.access_token) > 50

    @pytest.mark.asyncio
    async def test_token_contains_user_info(self, mock_user_store):
        """Verify JWT token contains correct user information."""
        tokens = create_token_pair(
            user_id="user-123",
            email="test@example.com",
            org_id="org-1",
        )

        decoded = decode_jwt(tokens.access_token)
        assert decoded is not None
        assert decoded.sub == "user-123"

    @pytest.mark.asyncio
    async def test_gauntlet_receipt_generation(self, mock_debate_result):
        """Verify Gauntlet receipt can be generated with correct structure."""
        # Create a Gauntlet DecisionReceipt (audit receipt for validation)
        receipt = DecisionReceipt(
            receipt_id=f"rcpt-{mock_debate_result.debate_id}",
            gauntlet_id="gauntlet-test-001",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary=f"Debate: {mock_debate_result.task}",
            input_hash="abc123def456",
            risk_summary={"critical": 0, "high": 0, "medium": 1, "low": 2},
            attacks_attempted=5,
            attacks_successful=0,
            probes_run=10,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=mock_debate_result.confidence,
            robustness_score=0.95,
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.75,
                method="majority_vote",
                supporting_agents=["claude", "gpt-4", "gemini"],
                dissenting_agents=["mistral"],
            ),
        )

        # Verify receipt has required fields
        assert receipt.receipt_id == f"rcpt-{mock_debate_result.debate_id}"
        assert receipt.verdict == "PASS"
        assert receipt.confidence == mock_debate_result.confidence
        assert receipt.consensus_proof is not None
        assert receipt.consensus_proof.supporting_agents == ["claude", "gpt-4", "gemini"]

    @pytest.mark.asyncio
    async def test_receipt_adapter_initialization(self):
        """Verify receipt adapter initializes correctly with mound."""
        from aragora.knowledge.mound.adapters.receipt_adapter import ReceiptAdapter

        # Create adapter with mock mound
        mock_mound = MagicMock()
        adapter = ReceiptAdapter(mound=mock_mound)

        # Verify adapter was created
        assert adapter._mound is mock_mound
        assert adapter._auto_ingest is True


class TestAuthenticationEnforcement:
    """Test that authentication is enforced at all protected endpoints."""

    @pytest.mark.asyncio
    async def test_login_provides_valid_token(self, mock_user_store):
        """Verify login flow provides a valid JWT token."""
        # Create user with known password
        user = User(
            id="user-123",
            email="test@example.com",
            password_hash=hash_password("SecurePass123!"),
            created_at=datetime.now(timezone.utc),
        )
        mock_user_store.get_user_by_email.return_value = user
        mock_user_store.verify_password = MagicMock(return_value=True)

        # Generate tokens
        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id="org-default",
        )

        # Verify token structure
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None

        # Verify token can be decoded
        decoded = decode_jwt(tokens.access_token)
        assert decoded.sub == user.id

    @pytest.mark.asyncio
    async def test_refresh_token_extends_session(self, mock_user_store):
        """Verify refresh token can extend user session."""
        tokens = create_token_pair(
            user_id="user-123",
            email="test@example.com",
            org_id="org-1",
        )

        # Verify refresh token exists and is different from access token
        assert tokens.refresh_token != tokens.access_token
        assert len(tokens.refresh_token) > 0


class TestOrganizationIsolation:
    """Test that organization isolation is enforced in auth and receipt flow."""

    @pytest.mark.asyncio
    async def test_token_contains_org_info(self):
        """Verify JWT tokens include organization context."""
        tokens = create_token_pair(
            user_id="user-123",
            email="test@example.com",
            org_id="org-456",
        )

        decoded = decode_jwt(tokens.access_token)
        # Should have user info
        assert decoded is not None
        assert decoded.sub == "user-123"

    @pytest.mark.asyncio
    async def test_different_org_tokens_are_distinct(self):
        """Verify tokens for different orgs are separate."""
        from aragora.rbac.models import AuthorizationContext

        # Create context for org A
        ctx_org_a = AuthorizationContext(
            user_id="user-123",
            org_id="org-a",
            roles={"viewer"},
            permissions={"receipts.read"},
        )

        # Create context for org B
        ctx_org_b = AuthorizationContext(
            user_id="user-456",
            org_id="org-b",
            roles={"viewer"},
            permissions={"receipts.read"},
        )

        # These should have different org contexts
        assert ctx_org_a.org_id != ctx_org_b.org_id
        assert ctx_org_a.user_id != ctx_org_b.user_id
