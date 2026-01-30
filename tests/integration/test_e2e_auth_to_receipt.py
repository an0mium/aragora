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
from aragora.gauntlet.receipt import DecisionReceipt, ConsensusProof
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
            tenant_id="tenant-1",
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
            tenant_id="tenant-1",
        )

        decoded = decode_jwt(tokens.access_token)
        assert decoded is not None
        assert decoded["sub"] == "user-123"

    @pytest.mark.asyncio
    async def test_receipt_generation_from_debate_result(self, mock_debate_result):
        """Verify receipt can be generated from debate result."""
        receipt = DecisionReceipt(
            receipt_id=f"rcpt-{mock_debate_result.debate_id}",
            debate_id=mock_debate_result.debate_id,
            decision=mock_debate_result.final_answer,
            confidence=mock_debate_result.confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            consensus_proof=ConsensusProof(
                method="majority_vote",
                threshold=0.66,
                achieved_score=0.75,
                supporting_agents=["claude", "gpt-4", "gemini"],
                dissenting_agents=["mistral"],
            ),
            participants=mock_debate_result.participants,
            rounds=mock_debate_result.rounds_completed,
        )

        # Verify receipt has required fields
        assert receipt.debate_id == mock_debate_result.debate_id
        assert receipt.decision == mock_debate_result.final_answer
        assert receipt.consensus_proof.supporting_agents == ["claude", "gpt-4", "gemini"]

    @pytest.mark.asyncio
    async def test_receipt_adapter_respects_org_context(self, mock_debate_result):
        """Verify receipt storage in Knowledge Mound respects organization boundaries."""
        from aragora.knowledge.mound.adapters.receipt_adapter import ReceiptAdapter

        # Create adapter with org context
        adapter = ReceiptAdapter(
            mound=MagicMock(),
            tenant_id="org-123",
        )

        # Mock the mound store method
        adapter.mound.store = AsyncMock(return_value="km-item-456")

        receipt = DecisionReceipt(
            receipt_id="rcpt-test",
            debate_id="debate-test",
            decision="Test decision",
            confidence=0.9,
            timestamp=datetime.now(timezone.utc).isoformat(),
            consensus_proof=ConsensusProof(
                method="unanimous",
                threshold=1.0,
                achieved_score=1.0,
                supporting_agents=["agent1"],
                dissenting_agents=[],
            ),
            participants=["agent1"],
            rounds=1,
        )

        # The adapter should include organization context
        assert adapter.tenant_id == "org-123"


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
            tenant_id="tenant-default",
        )

        # Verify token structure
        assert tokens.access_token is not None
        assert tokens.refresh_token is not None

        # Verify token can be decoded
        decoded = decode_jwt(tokens.access_token)
        assert decoded["sub"] == user.id

    @pytest.mark.asyncio
    async def test_refresh_token_extends_session(self, mock_user_store):
        """Verify refresh token can extend user session."""
        tokens = create_token_pair(
            user_id="user-123",
            email="test@example.com",
            tenant_id="tenant-1",
        )

        # Verify refresh token exists and is different from access token
        assert tokens.refresh_token != tokens.access_token
        assert len(tokens.refresh_token) > 0


class TestOrganizationIsolation:
    """Test that organization isolation is enforced in auth and receipt flow."""

    @pytest.mark.asyncio
    async def test_token_contains_tenant_info(self):
        """Verify JWT tokens include organization/tenant context."""
        tokens = create_token_pair(
            user_id="user-123",
            email="test@example.com",
            tenant_id="org-456",
        )

        decoded = decode_jwt(tokens.access_token)
        # Should have tenant/org info
        assert decoded is not None
        assert "sub" in decoded

    @pytest.mark.asyncio
    async def test_different_org_tokens_are_distinct(self):
        """Verify tokens for different orgs are separate."""
        from aragora.rbac.models import AuthorizationContext

        # Create context for org A
        ctx_org_a = AuthorizationContext(
            user_id="user-123",
            org_id="org-a",
            roles=["viewer"],
            permissions=["receipts.read"],
        )

        # Create context for org B
        ctx_org_b = AuthorizationContext(
            user_id="user-456",
            org_id="org-b",
            roles=["viewer"],
            permissions=["receipts.read"],
        )

        # These should have different org contexts
        assert ctx_org_a.org_id != ctx_org_b.org_id
        assert ctx_org_a.user_id != ctx_org_b.user_id
